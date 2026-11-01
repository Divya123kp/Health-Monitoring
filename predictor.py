"""
predictor.py
------------
Loads trained models + metadata and exposes a `SensorPredictor.predict()`
that returns per-model results. LSTM models use a rolling buffer of
seq_len readings.
"""
import json
import warnings
from collections import deque
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

TARGET_COLS   = ["IR", "RED", "HR", "SpO2"]
RELATIVE_COLS = ["IR", "RED"]

CLASSICAL = [f"{m}_{v}" for m in ("RandomForest", "ExtraTrees", "HistGradientBoosting")
             for v in ("NoGAN", "GAN")]
LSTMS = ["LSTM_NoGAN", "LSTM_GAN"]

# --- fallback defaults (overwritten by training_metadata.json) ---
DATA_LO   = {"IR": 0, "RED": 0, "HR": 0, "SpO2": 0}
DATA_HI   = {"IR": 1_124_224, "RED": 89_898, "HR": 89, "SpO2": 120}
NORMAL_LO = {"IR": 46_575, "RED": 67_676, "HR": 40, "SpO2": 90}
NORMAL_HI = {"IR": 1_124_224, "RED": 89_898, "HR": 89, "SpO2": 100}

# strict minimum for "is this even worth predicting?"
MIN_PREDICT_VALUE = {"IR": 1000, "RED": 1000, "HR": 5, "SpO2": 5}


# ---------------------------------------------------------------------------
# Feature engineering (must match the training script exactly)
# ---------------------------------------------------------------------------
def extract_features(X):
    X = pd.DataFrame(X, columns=TARGET_COLS).astype(float)
    F, N = pd.DataFrame(index=X.index), pd.DataFrame(index=X.index)

    for c in TARGET_COLS:
        v = X[c]
        lo, hi = NORMAL_LO[c], NORMAL_HI[c]
        F[c] = np.log1p(v.clip(lower=0)) if c in RELATIVE_COLS else v
        N[c] = (v - lo) / (hi - lo)
        F[f"{c}_below"] = (v < lo).astype(float).where(v.notna())
        F[f"{c}_above"] = (v > hi).astype(float).where(v.notna())
        F[f"{c}_dev"] = N[c].clip(upper=0) + (N[c] - 1).clip(lower=0)

    for i in range(len(TARGET_COLS)):
        for j in range(i + 1, len(TARGET_COLS)):
            a, b = TARGET_COLS[i], TARGET_COLS[j]
            F[f"diff_{a}_{b}"] = N[a] - N[b]

    F["n_below"]  = sum(F[f"{c}_below"] for c in TARGET_COLS)
    F["n_above"]  = sum(F[f"{c}_above"] for c in TARGET_COLS)
    F["row_mean"] = N.mean(axis=1)
    F["row_std"]  = N.std(axis=1)
    F["row_min"]  = N.min(axis=1)
    F["row_max"]  = N.max(axis=1)
    return F.replace([np.inf, -np.inf], np.nan)


def from_model_space(A):
    A = pd.DataFrame(np.asarray(A, dtype=float), columns=TARGET_COLS)
    for c in RELATIVE_COLS:
        A[c] = np.expm1(A[c].clip(lower=0, upper=20))
    return A.values


def clip_to_range(pred):
    lo = np.array([DATA_LO[c] for c in TARGET_COLS], float)
    hi = np.array([DATA_HI[c] for c in TARGET_COLS], float)
    return np.clip(pred, lo, hi)


def band_label(c, v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "?"
    return "LOW" if v < NORMAL_LO[c] else ("HIGH" if v > NORMAL_HI[c] else "ok")


def is_clearly_abnormal(latest: dict) -> bool:
    for c in TARGET_COLS:
        v = latest.get(c)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return True
        if v < MIN_PREDICT_VALUE[c]:
            return True
        if v < 0.5 * NORMAL_LO[c] or v > 1.5 * NORMAL_HI[c]:
            return True
    return False


def should_predict(latest: dict):
    """Return (ok, reason). Only predict when all channels are in a sane range."""
    missing = [c for c in TARGET_COLS
               if latest.get(c) is None or np.isnan(latest.get(c, np.nan))]
    if missing:
        return False, f"Missing values: {missing}"

    too_small = [c for c in TARGET_COLS if latest[c] < MIN_PREDICT_VALUE[c]]
    if too_small:
        return False, f"Values too tiny/zero on: {too_small}"

    return True, "ok"


# ---------------------------------------------------------------------------
# Heart-rate stabiliser  (fluctuation correction)
# ---------------------------------------------------------------------------
HR_MAX_JUMP   = 12     # bpm: a change bigger than this vs the last good HR is a "sudden fluctuation"
HR_REACQUIRE  = 5      # consecutive readings that must agree before a new level is trusted
HR_AGREE_TOL  = 8      # bpm: readings within this of each other count as "agreeing"
HR_SMOOTH     = 0.5    # 0..1  weight of a new accepted reading (1 = no smoothing)


class HRStabilizer:
    """
    Keeps the last legitimate heart-rate and ignores sudden jumps.

      * HR below the normal minimum (NORMAL_LO['HR'], 40) is NOT touched -
        it passes straight through exactly like before.
      * A reading close to the last good value is accepted (lightly smoothed).
      * A sudden jump is ignored: the OLD value is shown instead.
      * If the new level stays steady for HR_REACQUIRE readings in a row it is
        real (not noise) and becomes the new reference.
    """

    def __init__(self):
        self.last = None        # last legitimate (smoothed) HR
        self.pending = []       # recent out-of-band readings

    def reset(self):
        self.last, self.pending = None, []

    def update(self, hr):
        """Returns (stable_hr, was_held).  stable_hr is None => caller keeps raw/model value."""
        if hr is None or np.isnan(hr) or hr < NORMAL_LO["HR"]:
            return None, False                       # <40 : leave behaviour unchanged

        if self.last is None:                        # first legitimate reading
            self.last = float(hr)
            return self.last, False

        if abs(hr - self.last) <= HR_MAX_JUMP:       # normal beat-to-beat drift
            self.pending.clear()
            self.last = HR_SMOOTH * hr + (1 - HR_SMOOTH) * self.last
            return self.last, False

        # --- sudden fluctuation: hold the old value ---
        self.pending.append(float(hr))
        if len(self.pending) > HR_REACQUIRE:
            self.pending.pop(0)
        if (len(self.pending) == HR_REACQUIRE and
                max(self.pending) - min(self.pending) <= HR_AGREE_TOL):
            self.last = float(np.mean(self.pending))  # steady -> genuine change
            self.pending.clear()
            return self.last, False
        return self.last, True


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------
class SensorPredictor:
    def __init__(self, models_dir="sensor_prediction_models", seq_len=10):
        global DATA_LO, DATA_HI, NORMAL_LO, NORMAL_HI

        self.md = Path(models_dir)
        if not self.md.exists():
            raise FileNotFoundError(f"Models folder not found: {self.md.resolve()}")

        meta_path = self.md / "training_metadata.json"
        if meta_path.exists():
            meta = json.load(open(meta_path, encoding="utf-8"))
            DATA_LO   = meta.get("data_lo",   DATA_LO)
            DATA_HI   = meta.get("data_hi",   DATA_HI)
            NORMAL_LO = meta.get("normal_lo", NORMAL_LO)
            NORMAL_HI = meta.get("normal_hi", NORMAL_HI)

        self.seq_len = seq_len
        self.buffer = deque(maxlen=seq_len)
        self.models = self._load_models()
        self.hr_stab = HRStabilizer()

    def _load_models(self):
        models = {}

        # classical
        for n in CLASSICAL:
            f = self.md / f"{n}.joblib"
            if f.exists():
                try:
                    models[n] = joblib.load(f)
                except Exception as e:
                    print(f"[predictor] skip {n}: {e}")

        # LSTMs
        for n in LSTMS:
            f = self.md / f"{n}.keras"
            prep_f = self.md / f"{n}_preprocessor.joblib"
            if not f.exists():
                print(f"[predictor] {n}: missing {f.name}")
                continue
            if not prep_f.exists():
                print(f"[predictor] {n}: missing {prep_f.name}")
                continue
            try:
                from tensorflow.keras.models import load_model
                models[n] = {
                    "model": load_model(f, compile=False),
                    "prep":  joblib.load(prep_f),
                }
            except Exception as e:
                print(f"[predictor] skip {n}: {e}")

        print(f"[predictor] loaded models: {list(models.keys())}")
        return models

    def push_reading(self, row: dict):
        self.buffer.append(row)

    # --- per-family prediction ---
    def _predict_classical(self, name, X):
        a = self.models[name]
        Xs = a["scaler"].transform(a["imputer"].transform(extract_features(X.tail(1))))
        out = a["model"].predict(Xs)
        out = from_model_space(a["y_scaler"].inverse_transform(out))
        return clip_to_range(out)[-1]

    def _predict_lstm(self, name, X):
        a = self.models[name]
        prep = a["prep"]
        seq_len = prep["seq_len"]

        X = X.tail(seq_len).copy()
        if len(X) < seq_len:
            pad = pd.concat([X.iloc[[0]]] * (seq_len - len(X)), ignore_index=True)
            X = pd.concat([pad, X], ignore_index=True)

        feats = prep["scaler"].transform(prep["imputer"].transform(extract_features(X)))
        out = a["model"].predict(feats[None, ...].astype(np.float32), verbose=0)
        out = from_model_space(prep["y_scaler"].inverse_transform(out))
        return clip_to_range(out)[-1]

    # --- public API ---
    def predict(self, latest: dict, guard: bool = True, stabilize: bool = True):
        """stabilize=False (manual test input) skips the HR fluctuation hold and leaves its state alone."""
        ok, reason = should_predict(latest)
        if not ok:
            return [], False, reason

        self.push_reading(latest)
        X = pd.DataFrame(list(self.buffer), columns=TARGET_COLS)

        raw_preds = []                       # [(name, np.array[4])]
        for name in self.models:
            try:
                if name.startswith("LSTM"):
                    p = self._predict_lstm(name, X)
                else:
                    p = self._predict_classical(name, X)

                if guard:
                    p = self._apply_guard(p, latest)
                raw_preds.append((name, np.array(p, float)))
            except Exception as e:
                print(f"[predictor] {name} failed: {e}")

        # ---- heart-rate fluctuation correction -------------------------
        hr_idx = TARGET_COLS.index("HR")
        stable, held = self.hr_stab.update(latest.get("HR")) if stabilize else (None, False)
        if stable is not None and raw_preds:
            # keep each model's *relative* difference (small), centre it on the stable HR
            centre = float(np.median([p[hr_idx] for _, p in raw_preds]))
            for _, p in raw_preds:
                # (no upper clip: a steady genuine HR above 89 must still show as HIGH)
                p[hr_idx] = max(0.0, stable + float(np.clip(p[hr_idx] - centre, -3, 3)))
            if held:
                print(f"[predictor] HR {latest['HR']:.0f} = sudden jump -> holding {stable:.0f}")

        rows = []
        for name, p in raw_preds:
            status = "/".join(band_label(c, p[i]) for i, c in enumerate(TARGET_COLS))
            rows.append({
                "Model": name,
                **{c: float(p[i]) for i, c in enumerate(TARGET_COLS)},
                "Status": status,
            })

        if len(rows) > 1:
            med = [float(np.median([r[c] for r in rows])) for c in TARGET_COLS]
            status = "/".join(band_label(c, med[i]) for i, c in enumerate(TARGET_COLS))
            rows.append({
                "Model": "MEDIAN",
                **{c: med[i] for i, c in enumerate(TARGET_COLS)},
                "Status": status,
            })

        return rows, True, "ok"

    @staticmethod
    def _apply_guard(pred, latest):
        pred = np.array(pred, float)
        for j, c in enumerate(TARGET_COLS):
            v = latest.get(c)
            if v is None or np.isnan(v):
                continue
            clearly_low  = v < 0.90 * NORMAL_LO[c]
            clearly_high = v > NORMAL_HI[c]
            in_band = NORMAL_LO[c] <= pred[j] <= NORMAL_HI[c]
            if (clearly_low or clearly_high) and in_band:
                pred[j] = v
        return pred