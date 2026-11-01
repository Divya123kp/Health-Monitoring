"""
alerts.py — builds and sends the Telegram + Gmail alert.

Rules
  * Only heart beat (HR) and oxygen level (SpO2) can trigger / appear in an alert.
  * Plain text only, no attachments: user name, heart beat, oxygen level, status, reason.
  * Written in the language the user chose; e-mail goes to the address they registered with.
"""
import threading

import gmail
import predictor as P
from i18n import tr, normalize_lang
from telegram_alert import send_alert

VITALS = ("HR", "SpO2")


def build_alert(lang, name, hr, spo2, manual=False):
    """Return (subject, body) in `lang`, or None if HR and SpO2 are both normal."""
    lang = normalize_lang(lang)
    vals = {"HR": float(hr), "SpO2": float(spo2)}
    band = {c: P.band_label(c, vals[c]) for c in VITALS}
    if all(b == "ok" for b in band.values()):
        return None

    def word(b):
        return tr({"HIGH": "al_hi", "LOW": "al_lo"}.get(b, "al_ok"), lang)

    fmt = {"HR": f"{vals['HR']:.0f}", "SpO2": f"{vals['SpO2']:.0f}"}
    reasons = []
    for c, key in (("HR", "al_hr"), ("SpO2", "al_sp")):
        if band[c] in ("HIGH", "LOW"):
            reasons.append(tr(f"{key}_{'hi' if band[c] == 'HIGH' else 'lo'}", lang,
                              v=fmt[c], lo=P.NORMAL_LO[c], hi=P.NORMAL_HI[c]))

    hr_abn, sp_abn = band["HR"] != "ok", band["SpO2"] != "ok"
    what = tr("al_what_both" if hr_abn and sp_abn else "al_what_hr" if hr_abn else "al_what_sp", lang)
    subject = tr("al_subject_test" if manual else "al_subject", lang, what=what)

    body = "\n".join([
        tr("al_title_test" if manual else "al_title", lang),
        "",
        f"{tr('al_name', lang)}: {name}",
        f"{tr('heart_beat', lang)}: {fmt['HR']} {tr('al_unit_bpm', lang)} ({word(band['HR'])})",
        f"{tr('oxygen_level', lang)}: {fmt['SpO2']} % ({word(band['SpO2'])})",
        f"{tr('al_status', lang)}: {tr('al_abnormal', lang)}",
        f"{tr('al_reason', lang)}: " + " ".join(reasons),
    ])
    return subject, body


def _email_worker(to_addr, subject, body):
    try:
        gmail.send_from_env(to_addr, subject, body)
        print(f"[gmail] sent to {to_addr}")
    except Exception as e:
        print(f"[gmail] FAILED: {e}")


def dispatch(user, hr, spo2, manual=False):
    """
    Send the alert for `user` (a users-table dict: full_name, email, lang).
    Telegram goes out immediately; the e-mail is sent in a background thread so the
    sensor loop is never blocked by SMTP.
    Returns (text_sent, telegram_ok, email_queued) — text_sent is None if nothing to alert.
    """
    name = (user or {}).get("full_name") or "-"
    lang = (user or {}).get("lang") or "en"
    built = build_alert(lang, name, hr, spo2, manual=manual)
    if built is None:
        return None, None, False
    subject, body = built

    telegram_ok = bool(send_alert(body))

    to_addr = (user or {}).get("email")
    email_queued = False
    if to_addr:
        threading.Thread(target=_email_worker, args=(to_addr, subject, body),
                         daemon=True).start()
        email_queued = True
    else:
        print("[gmail] no registered user e-mail - e-mail alert skipped")
    return body, telegram_ok, email_queued
