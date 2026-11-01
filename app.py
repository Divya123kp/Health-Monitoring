"""
app.py — Flask + Socket.IO dashboard for the sensor prediction pipeline.

Fixes applied:
  * NO eventlet: monkey_patch turned the serial thread into a green thread,
    and pyserial's blocking readline() froze the whole process after
    "[serial] opened COMx". Real OS threads (async_mode="threading") fix that.
  * serial reader auto-starts on boot
  * 3-in-a-row abnormal alert based on LSTM_GAN prediction
"""
import os
import re
import secrets
import sqlite3
import time
import threading
from collections import deque
from functools import wraps
from pathlib import Path
from queue import Queue, Empty

from flask import (Flask, render_template, jsonify, request, session,
                   redirect, url_for, flash)
from flask_socketio import SocketIO
from werkzeug.security import generate_password_hash, check_password_hash

import database as db
import config  # noqa: F401  (loads .env secrets)
import alerts
from i18n import tr, js_bundle, normalize_lang, LANGS
from predictor import (SensorPredictor, TARGET_COLS, band_label, should_predict,
                       NORMAL_LO, NORMAL_HI)

# --- optional serial input ---
try:
    from serial_reader import SerialReader
    SERIAL_AVAILABLE = True
except Exception as e:
    print(f"[app] serial_reader import failed: {e}")
    SERIAL_AVAILABLE = False


# ---------------------------------------------------------------------------
# Flask + Socket.IO
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent


def _load_secret_key():
    """Persistent random key so logins survive restarts (stored in .secret_key)."""
    env = os.getenv("SECRET_KEY")
    if env:
        return env
    f = BASE_DIR / ".secret_key"
    if f.exists():
        return f.read_text().strip()
    key = secrets.token_hex(32)
    f.write_text(key)
    return key


app = Flask(__name__)
app.config["SECRET_KEY"] = _load_secret_key()
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["PERMANENT_SESSION_LIFETIME"] = 60 * 60 * 24 * 7   # 7 days (remember me)

# Real threads so the blocking serial read can't starve the web server.
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading",
                    logger=False, engineio_logger=False)

db.init_db()
predictor = SensorPredictor(models_dir="sensor_prediction_models")


# ---------------------------------------------------------------------------
# Serial ingestion
# ---------------------------------------------------------------------------
serial_queue = Queue()
serial_reader = None

SERIAL_PORT = "COM8"
SERIAL_BAUD = 9600


# ---------------------------------------------------------------------------
# 3-in-a-row abnormal tracker (based on LSTM_GAN prediction)
# ---------------------------------------------------------------------------
ALERT_WINDOW   = 3          # need 3 consecutive abnormal readings
ALERT_COOLDOWN = 60         # min seconds between two Telegram fires

consecutive_abnormal = deque(maxlen=ALERT_WINDOW)
alert_sent_for_current_streak = False
_last_alert_ts = 0.0
MANUAL_COOLDOWN = 5         # seconds between manual-test Telegram alerts (anti double-click)
_last_manual_alert_ts = 0.0


# Last payload that actually contained model predictions. The dashboard uses it to
# fill the "All models" table on page load and to keep showing the newest
# prediction while the sensor has no finger on it.
last_prediction_payload = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _gan_row_or_fallback(results):
    """LSTM_GAN if present, else MEDIAN, else the last row."""
    if not results:
        return None
    for name in ("LSTM_GAN", "MEDIAN"):
        r = next((row for row in results if row["Model"] == name), None)
        if r:
            return r
    return results[-1]


# Alerts are raised ONLY for heart beat (HR) and oxygen level (SpO2) anomalies.
ALERT_COLS = ("HR", "SpO2")


def _is_prediction_abnormal(row):
    if row is None:
        return False
    return any(band_label(c, row[c]) in ("LOW", "HIGH") for c in ALERT_COLS)


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------
def process_reading(row: dict, manual: bool = False, user_id=None):
    """manual=True: reading came from the dashboard test form (not the sensor)."""
    global alert_sent_for_current_streak, _last_alert_ts, _last_manual_alert_ts
    global last_prediction_payload

    # coerce all values to floats
    latest = {}
    for c in TARGET_COLS:
        v = row.get(c, 0)
        try:
            latest[c] = float(v) if v is not None and v != "" else 0.0
        except (TypeError, ValueError):
            latest[c] = 0.0

    # raw abnormal check (any channel - display only)
    raw_abnormal = any(band_label(c, latest[c]) in ("LOW", "HIGH") for c in TARGET_COLS)
    # raw abnormal check restricted to HR / SpO2 (used for alerting when models are skipped)
    raw_vital_abnormal = any(band_label(c, latest[c]) in ("LOW", "HIGH") for c in ALERT_COLS)

    status_in = "/".join(band_label(c, latest[c]) for c in TARGET_COLS)

    # store raw reading
    reading_id = db.insert_reading(
        latest["IR"], latest["RED"], latest["HR"], latest["SpO2"],
        status_in, raw_abnormal,
    )

    # --- predict (skipped when input is tiny/zero) ---
    ok, why = should_predict(latest)
    results = []
    if ok:
        results, ok, why = predictor.predict(latest, guard=True, stabilize=not manual)
        if ok and results:
            db.insert_predictions(reading_id, results)

    # --- which row do we judge? ---
    chosen = _gan_row_or_fallback(results)
    if chosen is not None:
        is_abn = _is_prediction_abnormal(chosen)
        judged_on = f"GAN prediction ({chosen['Model']})"
    else:
        is_abn = raw_vital_abnormal
        judged_on = "raw input (models skipped)"

    fire_alert = False
    telegram_sent = None

    if manual:
        # Manual test: do NOT touch the sensor streak; alert right away if abnormal.
        streak = sum(consecutive_abnormal)
        now = time.time()
        if is_abn and ok and (now - _last_manual_alert_ts) > MANUAL_COOLDOWN:
            fire_alert = True
            _last_manual_alert_ts = now
    else:
        # --- 3-in-a-row streak logic ---
        consecutive_abnormal.append(is_abn)
        streak = sum(consecutive_abnormal)

        if len(consecutive_abnormal) == ALERT_WINDOW and streak == ALERT_WINDOW:
            now = time.time()
            if (not alert_sent_for_current_streak) and (now - _last_alert_ts) > ALERT_COOLDOWN:
                fire_alert = True
                alert_sent_for_current_streak = True
                _last_alert_ts = now

        if not is_abn:
            alert_sent_for_current_streak = False

    print(
        f"[app] {'MANUAL' if manual else 'in'} IR={latest['IR']:.0f} RED={latest['RED']:.0f} "
        f"HR={latest['HR']:.0f} SpO2={latest['SpO2']:.1f} | "
        f"predicted={ok} | {judged_on} | is_abn={is_abn} | "
        f"streak={streak}/{ALERT_WINDOW} | fire={fire_alert}"
    )

    email_queued = None
    if fire_alert:
        # who gets it: the logged-in tester for manual input, otherwise the most
        # recently logged-in user. E-mail = registered address, language = user's choice.
        # A problem while sending alerts must NEVER stop the reading reaching the dashboard.
        try:
            recipient = (db.get_user(user_id) if (manual and user_id) else None) or db.latest_active_user()
            # report the values the alert was judged on (HR + SpO2 only)
            src = chosen if chosen is not None else latest
            msg, telegram_sent, email_queued = alerts.dispatch(
                recipient, src["HR"], src["SpO2"], manual=manual)
            if msg is None:                   # nothing abnormal in HR/SpO2 -> nothing to send
                fire_alert = False
            else:
                db.insert_alert(reading_id, msg, int(bool(telegram_sent) or bool(email_queued)))
                print("[app] ✅ ALERT FIRED" + (" (manual)" if manual else ""))
        except Exception as e:
            import traceback
            print(f"[app] alert sending failed (dashboard still updated): {e}")
            traceback.print_exc()
            telegram_sent = False

    # --- broadcast to frontend (every reading, predicted or not) ---
    payload = {
        "reading_id": reading_id,
        "ts": time.strftime("%H:%M:%S"),
        "input": latest,
        "input_status": status_in,
        "abnormal": raw_abnormal,
        "predicted": ok,
        "reason": why if not ok else "ok",
        "reason_code": ("ok" if ok else ("missing" if str(why).startswith("Missing") else "tiny")),
        "results": results,
        "streak": streak,
        "alert_fired": fire_alert,
        "judged_on": judged_on,
        "manual": manual,
        "telegram_sent": telegram_sent,
        "email_queued": email_queued,
    }
    if ok and results:
        last_prediction_payload = payload
    socketio.emit("new_reading", payload)
    return payload


def serial_worker():
    """Consume the serial queue forever (runs in a background thread)."""
    print("[serial_worker] started")
    while True:
        try:
            row = serial_queue.get(timeout=1)
            process_reading(row)
        except Empty:
            continue
        except Exception as e:
            import traceback
            print(f"[serial_worker] error: {e}")
            traceback.print_exc()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------
def current_user():
    uid = session.get("uid")
    return db.get_user(uid) if uid else None


def current_lang():
    u = current_user()
    if u and u.get("lang"):
        return normalize_lang(u["lang"])
    return normalize_lang(session.get("lang"))


@app.context_processor
def inject_user():
    lang = current_lang()
    return {
        "user": current_user(),
        "lang": lang,
        "LANGS": LANGS,
        "_": lambda key, **kw: tr(key, lang, **kw),
        "i18n_js": js_bundle(lang),
    }


@app.route("/set-language", methods=["POST"])
def set_language():
    lang = normalize_lang(request.form.get("lang"))
    session["lang"] = lang
    uid = session.get("uid")
    if uid:
        db.set_user_lang(uid, lang)
    target = request.form.get("next")
    if target and target.startswith("/") and not target.startswith("//"):
        return redirect(target)
    return redirect(url_for("home"))


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not current_user():
            # no session, or a stale session whose user no longer exists in the DB
            session.pop("uid", None)
            if request.path.startswith("/api/"):
                return jsonify({"error": "login required"}), 401
            return redirect(url_for("login", next=request.path))
        return view(*args, **kwargs)
    return wrapped


def _safe_next(target):
    """Only allow same-site relative redirects."""
    if target and target.startswith("/") and not target.startswith("//"):
        return target
    return url_for("monitor")


EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
USER_RE = re.compile(r"^[A-Za-z0-9_.]{3,24}$")


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
@app.route("/")
def home():
    return render_template("home.html", stats=db.stats_summary())


@app.route("/api/slides")
def api_slides():
    """Images dropped into static/slides/ are used for the full-screen slideshow."""
    folder = BASE_DIR / "static" / "slides"
    exts = {".jpg", ".jpeg", ".png", ".webp", ".avif"}
    files = []
    if folder.exists():
        files = sorted(p.name for p in folder.iterdir() if p.suffix.lower() in exts)
    return jsonify([f"/static/slides/{f}" for f in files])


@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user() and request.method == "GET":
        return redirect(url_for("monitor"))
    error = None
    if request.method == "POST":
        ident = request.form.get("identifier", "").strip()
        pw = request.form.get("password", "")
        u = db.get_user_by_login(ident)
        if u and check_password_hash(u["password_hash"], pw):
            session.clear()
            session["uid"] = u["id"]
            session["lang"] = normalize_lang(u.get("lang"))
            session.permanent = bool(request.form.get("remember"))
            db.touch_login(u["id"])
            return redirect(_safe_next(request.args.get("next")))
        error = tr("err_login", normalize_lang(session.get("lang")))
    return render_template("login.html", error=error)


@app.route("/register", methods=["GET", "POST"])
def register():
    if current_user() and request.method == "GET":
        return redirect(url_for("monitor"))
    error, form = None, {}
    lang = normalize_lang(session.get("lang"))
    if request.method == "POST":
        form = {k: request.form.get(k, "").strip()
                for k in ("full_name", "username", "email")}
        pw = request.form.get("password", "")
        pw2 = request.form.get("confirm", "")
        if not form["full_name"]:
            error = tr("err_name", lang)
        elif not USER_RE.match(form["username"]):
            error = tr("err_username", lang)
        elif not EMAIL_RE.match(form["email"]):
            error = tr("err_email", lang)
        elif len(pw) < 6:
            error = tr("err_pw_len", lang)
        elif pw != pw2:
            error = tr("err_pw_match", lang)
        else:
            try:
                uid = db.create_user(form["full_name"], form["username"],
                                     form["email"], generate_password_hash(pw), lang)
                session.clear()
                session["uid"] = uid
                session["lang"] = lang
                db.touch_login(uid)
                return redirect(url_for("monitor"))
            except sqlite3.IntegrityError:
                error = tr("err_dup", lang)
    return render_template("register.html", error=error, form=form)


@app.route("/logout")
def logout():
    lang = session.get("lang")
    session.clear()
    if lang:
        session["lang"] = lang
    return redirect(url_for("home"))


@app.route("/monitor")
@login_required
def monitor():
    return render_template("monitor.html")


@app.route("/dashboard")
@login_required
def dashboard():
    return render_template("dashboard.html")


@app.route("/api/predict", methods=["POST"])
@login_required
def api_predict():
    data = request.get_json(force=True)
    row = {c: data.get(c, 0) for c in TARGET_COLS}
    return jsonify(process_reading(row, manual=True, user_id=session.get("uid")))


@app.route("/api/readings")
@login_required
def api_readings():
    limit = int(request.args.get("limit", 200))
    return jsonify(db.recent_readings(limit))


def _latest_prediction_from_db():
    """Rebuild the newest full prediction set from SQLite (used after a restart)."""
    with db.db() as con:
        row = con.execute("SELECT reading_id FROM predictions ORDER BY id DESC LIMIT 1").fetchone()
        if not row:
            return None
        rid = row["reading_id"]
        preds = [dict(r) for r in con.execute(
            "SELECT model,IR,RED,HR,SpO2,status FROM predictions WHERE reading_id=? ORDER BY id", (rid,))]
        rd = con.execute("SELECT * FROM readings WHERE id=?", (rid,)).fetchone()
    if not preds or not rd:
        return None
    rd = dict(rd)
    results = [{"Model": p["model"], "IR": p["IR"], "RED": p["RED"], "HR": p["HR"],
                "SpO2": p["SpO2"], "Status": p["status"]} for p in preds]
    inp = {c: rd[c] for c in TARGET_COLS}
    return {
        "reading_id": rid,
        "ts": str(rd.get("ts", ""))[11:19],
        "input": inp,
        "input_status": rd.get("status", ""),
        "abnormal": bool(rd.get("is_abnormal")),
        "predicted": True, "reason": "ok", "reason_code": "ok",
        "results": results, "streak": 0, "alert_fired": False,
        "judged_on": "GAN prediction (restored)", "manual": False,
    }


@app.route("/api/latest_prediction")
@login_required
def api_latest_prediction():
    """Newest set of model predictions (memory first, then database)."""
    p = last_prediction_payload or _latest_prediction_from_db()
    return jsonify(p or {})


@app.route("/api/predictions")
@login_required
def api_predictions():
    limit = int(request.args.get("limit", 200))
    return jsonify(db.recent_predictions(limit))


@app.route("/api/alerts")
@login_required
def api_alerts():
    return jsonify(db.recent_alerts(50))


@app.route("/api/performance")
@login_required
def api_performance():
    rows = db.model_performance()
    out = {}
    for r in rows:
        out[r["model"]] = {
            "mae_ir":   r["mae_ir"]   or 0,
            "mae_red":  r["mae_red"]  or 0,
            "mae_hr":   r["mae_hr"]   or 0,
            "mae_spo2": r["mae_spo2"] or 0,
            "n":        r["n"] or 0,
        }
    return jsonify(out)


@app.route("/api/models")
@login_required
def api_models():
    return jsonify({
        "loaded": list(predictor.models.keys()),
        "expected": [
            "RandomForest_NoGAN", "RandomForest_GAN",
            "ExtraTrees_NoGAN", "ExtraTrees_GAN",
            "HistGradientBoosting_NoGAN", "HistGradientBoosting_GAN",
            "LSTM_NoGAN", "LSTM_GAN",
        ],
        "models_dir": str(predictor.md.resolve()),
        "files_present": sorted(p.name for p in predictor.md.glob("*")),
    })


@app.route("/api/config")
@login_required
def api_config():
    return jsonify({"lo": NORMAL_LO, "hi": NORMAL_HI, "stats": db.stats_summary()})


@socketio.on("connect")
def _ws_connect():
    # only logged-in users may receive live data
    if not current_user():
        print("[socket] connection refused - not logged in (log in again)")
        return False
    print("[socket] dashboard connected")


@app.route("/api/serial/status")
@login_required
def serial_status():
    running = bool(serial_reader and serial_reader.is_alive())
    return jsonify({
        "running": running,
        "port": SERIAL_PORT,
        "baud": SERIAL_BAUD,
        "queue_size": serial_queue.qsize(),
    })


@app.route("/api/serial/start", methods=["POST"])
@login_required
def serial_start():
    global serial_reader, SERIAL_PORT, SERIAL_BAUD
    if not SERIAL_AVAILABLE:
        return jsonify({"ok": False, "msg": "pyserial not installed"})
    if serial_reader and serial_reader.is_alive():
        return jsonify({"ok": True, "msg": "already running"})

    body = request.get_json(force=True) or {}
    SERIAL_PORT = body.get("port", SERIAL_PORT)
    SERIAL_BAUD = int(body.get("baud", SERIAL_BAUD))

    serial_reader = SerialReader(port=SERIAL_PORT, baud=SERIAL_BAUD,
                                 out_queue=serial_queue)
    serial_reader.start()
    return jsonify({"ok": True, "msg": f"started on {SERIAL_PORT} @ {SERIAL_BAUD}"})


@app.route("/api/serial/stop", methods=["POST"])
@login_required
def serial_stop():
    global serial_reader
    if serial_reader:
        serial_reader.stop()
        serial_reader = None
        return jsonify({"ok": True, "msg": "stopped"})
    return jsonify({"ok": True, "msg": "not running"})


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------
def _start_background():
    """Start the queue consumer + serial reader (called once at boot)."""
    global serial_reader
    threading.Thread(target=serial_worker, daemon=True).start()

    if SERIAL_AVAILABLE:
        try:
            serial_reader = SerialReader(port=SERIAL_PORT, baud=SERIAL_BAUD,
                                         out_queue=serial_queue)
            serial_reader.start()
            print(f"[app] Serial reader started on {SERIAL_PORT} @ {SERIAL_BAUD}")
        except Exception as e:
            print(f"[app] Could not start serial reader: {e}")
    else:
        print("[app] pyserial not available — use /api/predict via HTTP instead.")


if __name__ == "__main__":
    _start_background()
    print("[app] Server ready → http://localhost:5000")
    socketio.run(app, host="0.0.0.0", port=5000, debug=False,
                 use_reloader=False, allow_unsafe_werkzeug=True)