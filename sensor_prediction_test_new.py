"""
sensor_prediction_test_new.py
-----------------------------
Loads the models saved by train_sensor_models.py and predicts the corrected IR, RED, HR, SpO2.

Abnormal inputs (0, 10, SpO2 > 100, ...) are meant to stay ABNORMAL in the output. Every prediction is
therefore labelled NORMAL / LOW / HIGH per channel.

Usage
    python sensor_prediction_test_new.py                    # type one reading (blank = missing)
    python sensor_prediction_test_new.py --demo             # run a set of normal + abnormal test readings
    python sensor_prediction_test_new.py --csv recent.csv   # last rows = a real sequence for the LSTMs
    python sensor_prediction_test_new.py --model LSTM_GAN   # one model only
    python sensor_prediction_test_new.py --guard            # safety net, see below

--guard : if an input channel is clearly outside its normal band but a model still answers inside the band,
          the input value is kept instead (a model can then never "normalise" a very abnormal input).
"""
import argparse, json, sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")

TARGET_COLS = ["IR", "RED", "HR", "SpO2"]          # must match training order
RELATIVE_COLS = ["IR", "RED"]
CLASSICAL = [f"{m}_{v}" for m in ("RandomForest", "ExtraTrees", "HistGradientBoosting")
             for v in ("NoGAN", "GAN")]
LSTMS = ["LSTM_NoGAN", "LSTM_GAN"]

# defaults - overwritten by training_metadata.json when it is present
DATA_LO = {"IR": 0, "RED": 0, "HR": 0, "SpO2": 0}
DATA_HI = {"IR": 1_124_224, "RED": 89_898, "HR": 89, "SpO2": 120}
NORMAL_LO = {"IR": 46_575, "RED": 67_676, "HR": 40, "SpO2": 90}
NORMAL_HI = {"IR": 1_124_224, "RED": 89_898, "HR": 89, "SpO2": 100}


# ---- identical to the training script ---------------------------------------
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
    F["n_below"] = sum(F[f"{c}_below"] for c in TARGET_COLS)
    F["n_above"] = sum(F[f"{c}_above"] for c in TARGET_COLS)
    F["row_mean"], F["row_std"] = N.mean(axis=1), N.std(axis=1)
    F["row_min"], F["row_max"] = N.min(axis=1), N.max(axis=1)
    return F.replace([np.inf, -np.inf], np.nan)


def from_model_space(A):
    A = pd.DataFrame(np.asarray(A, dtype=float), columns=TARGET_COLS)
    for c in RELATIVE_COLS:
        A[c] = np.expm1(A[c].clip(lower=0, upper=20))
    return A.values
# -----------------------------------------------------------------------------


def clip_to_range(pred):
    lo = np.array([DATA_LO[c] for c in TARGET_COLS], float)
    hi = np.array([DATA_HI[c] for c in TARGET_COLS], float)
    return np.clip(pred, lo, hi)


def band_label(c, v):
    if pd.isna(v):
        return "?"
    return "LOW" if v < NORMAL_LO[c] else ("HIGH" if v > NORMAL_HI[c] else "ok")


def predict_classical(models_dir, name, X):
    a = joblib.load(models_dir / f"{name}.joblib")
    Xs = a["scaler"].transform(a["imputer"].transform(extract_features(X)))
    out = a["model"].predict(Xs)
    out = from_model_space(a["y_scaler"].inverse_transform(out))      # targets were log/standardised in training
    return clip_to_range(out)[-1]


def predict_lstm(models_dir, name, X):
    from tensorflow.keras.models import load_model          # imported lazily
    prep = joblib.load(models_dir / f"{name}_preprocessor.joblib")
    seq_len = prep["seq_len"]
    if len(X) < seq_len:                                     # pad by repeating the oldest row
        X = pd.concat([X.iloc[[0]]] * (seq_len - len(X)) + [X], ignore_index=True)
    X = X.tail(seq_len)
    feats = prep["scaler"].transform(prep["imputer"].transform(extract_features(X)))
    model = load_model(models_dir / f"{name}.keras", compile=False)
    out = model.predict(feats[None, ...].astype(np.float32), verbose=0)
    out = from_model_space(prep["y_scaler"].inverse_transform(out))
    return clip_to_range(out)[-1]


def apply_guard(pred, latest):
    """Keep clearly abnormal input values (below 90 % of the lower band edge / above the upper edge)."""
    pred = np.array(pred, float)
    for j, c in enumerate(TARGET_COLS):
        v = latest[c]
        if pd.isna(v):
            continue
        clearly_low = v < 0.90 * NORMAL_LO[c]
        clearly_high = v > NORMAL_HI[c] * (1.0 if c == "SpO2" else 1.0)
        in_band = NORMAL_LO[c] <= pred[j] <= NORMAL_HI[c]
        if (clearly_low or clearly_high) and in_band:
            pred[j] = v
    return pred


def ask_reading():
    print("Enter the sensor values (blank = missing, 0 is treated as a real reading of 0):")
    vals = {}
    for c in TARGET_COLS:
        while True:
            raw = input(f"  {c}: ").strip()
            if raw == "":
                vals[c] = np.nan
                break
            try:
                vals[c] = float(raw)
                break
            except ValueError:
                print("   please enter a number")
    return pd.DataFrame([vals], columns=TARGET_COLS)


DEMO_CASES = {
    "normal":                    dict(IR=380000, RED=80000, HR=72, SpO2=97),
    "normal (noisy)":            dict(IR=95000, RED=71000, HR=58, SpO2=95),
    "ALL ZERO (sensor off)":     dict(IR=0, RED=0, HR=0, SpO2=0),
    "ALL TINY (10,10,10,10)":    dict(IR=10, RED=10, HR=10, SpO2=10),
    "IR dropout (IR=0)":         dict(IR=0, RED=80000, HR=72, SpO2=97),
    "HR=0 only":                 dict(IR=380000, RED=80000, HR=0, SpO2=97),
    "SpO2=10 only":              dict(IR=380000, RED=80000, HR=72, SpO2=10),
    "RED=10, HR=0":              dict(IR=380000, RED=10, HR=0, SpO2=97),
    "SpO2 artefact 115":         dict(IR=380000, RED=80000, HR=72, SpO2=115),
}


def run_models(md, X, names, args):
    rows = []
    for n in names:
        try:
            if n.startswith("LSTM"):
                p = predict_lstm(md, n, X)
            else:
                p = predict_classical(md, n, X.tail(1))
            if args.guard:
                p = apply_guard(p, X.iloc[-1])
            rows.append([n] + [float(v) for v in p])
        except ImportError:
            print(f"Skipping {n}: TensorFlow is not installed.")
        except FileNotFoundError as e:
            print(f"Skipping {n}: file not found ({e.filename})")
    if not rows:
        sys.exit("No predictions produced.")
    res = pd.DataFrame(rows, columns=["Model"] + TARGET_COLS)
    if len(res) > 1:
        res.loc[len(res)] = ["MEDIAN of models"] + res[TARGET_COLS].median().tolist()
    return res


def show(res, latest):
    print("Input :", {c: (None if pd.isna(latest[c]) else float(latest[c])) for c in TARGET_COLS},
          "| input status:", {c: band_label(c, latest[c]) for c in TARGET_COLS})
    out = res.copy()
    for c in TARGET_COLS:
        out[c] = out[c].round(1)
    out["Status (IR/RED/HR/SpO2)"] = [" / ".join(band_label(c, r[c]) for c in TARGET_COLS) for _, r in res.iterrows()]
    pd.set_option("display.width", 220)
    print(out.to_string(index=False))


def main():
    global DATA_LO, DATA_HI, NORMAL_LO, NORMAL_HI
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", default="sensor_prediction_models")
    ap.add_argument("--model", default="all", help="all | " + " | ".join(CLASSICAL + LSTMS))
    ap.add_argument("--csv", help="CSV with IR,RED,HR,SpO2 (oldest -> newest) for a real LSTM sequence")
    ap.add_argument("--demo", action="store_true", help="run built-in normal and abnormal test readings")
    ap.add_argument("--guard", action="store_true", help="never let a clearly abnormal input become a normal output")
    args = ap.parse_args()

    md = Path(args.models_dir)
    if not md.exists():
        sys.exit(f"Models folder not found: {md.resolve()}  (run train_sensor_models.py first)")
    meta_path = md / "training_metadata.json"
    if meta_path.exists():
        meta = json.load(open(meta_path, encoding="utf-8"))
        DATA_LO, DATA_HI = meta.get("data_lo", DATA_LO), meta.get("data_hi", DATA_HI)
        NORMAL_LO, NORMAL_HI = meta.get("normal_lo", NORMAL_LO), meta.get("normal_hi", NORMAL_HI)
    else:
        print("Note: training_metadata.json missing - using built-in ranges.")

    names = CLASSICAL + LSTMS if args.model == "all" else [args.model]

    if args.demo:
        for title, vals in DEMO_CASES.items():
            X = pd.DataFrame([vals], columns=TARGET_COLS)
            print("\n" + "=" * 100 + f"\nCASE: {title}")
            show(run_models(md, X, names, args), X.iloc[-1])
        return

    if args.csv:
        X = pd.read_csv(args.csv)
        X.columns = [c.strip() for c in X.columns]
        X = X.rename(columns={c: t for t in TARGET_COLS for c in X.columns if c.lower() == t.lower()})
        X = X[TARGET_COLS].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
        print(f"Loaded {len(X)} readings from {args.csv}; classical models use the last one.")
    else:
        X = ask_reading()
        print("\n(Single reading: the LSTMs need a sequence, so this reading is repeated. "
              "Use --csv with recent readings for a proper LSTM prediction.)")

    print()
    show(run_models(md, X, names, args), X.iloc[-1])


if __name__ == "__main__":
    main()
