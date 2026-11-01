# %% [markdown]
# # Sensor Value Prediction & Correction — ML + LSTM, with and without GAN augmentation
#
# 4 algorithms x 2 variants = **8 models**
#
# | Algorithm | No GAN | GAN-augmented |
# |---|---|---|
# | Random Forest | yes | yes |
# | Extra Trees | yes | yes |
# | HistGradientBoosting | yes | yes |
# | LSTM | yes | yes |
#
# **Goal.** Given a (noisy) reading of `IR, RED, HR, SpO2`, predict the clean value.
# **Abnormal readings must stay abnormal**: an input such as `0` or `10` must NOT be "corrected" to a normal value.
#
# Dataset: `merged_numerics_extended.csv` (simulated, 0 .. upper limits, ~24 % abnormal rows).
# The CSV holds clean reference values; sensor noise is added here during training.
#
# The notebook runs the whole experiment for several random seeds (different train/test recording splits)
# and reports mean +- std, so the GAN-vs-No-GAN comparison is not decided by one lucky run.

# %%
import os, sys, json, random, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

if "ipykernel" not in sys.modules:           # headless run -> no display needed
    import matplotlib
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GroupShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# %% [markdown]
# ## 1. Configuration

# %%
DATA_PATH = os.environ.get("DATA_PATH", "merged_numerics_extended.csv")
OUT_DIR = Path(os.environ.get("OUT_DIR", "sensor_prediction_models"))
PLOT_DIR = OUT_DIR / "plots"
OUT_DIR.mkdir(exist_ok=True)
PLOT_DIR.mkdir(exist_ok=True)

N_SEEDS = int(os.environ.get("N_SEEDS", 3))          # independent repetitions of the whole experiment
QUICK = os.environ.get("QUICK", "0") == "1"          # tiny settings, only to smoke-test the code

TARGET_COLS = ["IR", "RED", "HR", "SpO2"]            # order matters for saved artifacts
RELATIVE_COLS = ["IR", "RED"]                        # huge-scale optical channels -> log space
GROUP_COL, TIME_COL = "Source_File", "Time [s]"

# Hard limits of the data (0 .. upper value) and the NORMAL operating band of each channel.
# Anything outside the normal band is ABNORMAL (0, 10, SpO2 > 100, ...).
DATA_LO = {"IR": 0, "RED": 0, "HR": 0, "SpO2": 0}
DATA_HI = {"IR": 1_124_224, "RED": 89_898, "HR": 89, "SpO2": 120}
NORMAL_LO = {"IR": 46_575, "RED": 67_676, "HR": 40, "SpO2": 90}
NORMAL_HI = {"IR": 1_124_224, "RED": 89_898, "HR": 89, "SpO2": 100}

# Sensor-noise model used to build the (noisy input -> clean target) pairs.
NOISE_REL = {"IR": 0.03, "RED": 0.03, "HR": 0.025, "SpO2": 0.01}      # relative noise
NOISE_FLOOR = {"IR": 150.0, "RED": 60.0, "HR": 0.5, "SpO2": 0.2}      # absolute noise (so 0 stays ~0)
CLEAN_ROW_FRAC = 0.15        # share of rows left un-noised (users sometimes type exact values)
SPIKE_PROB, SPIKE_REL = 0.03, 0.15
GAIN_PROB, GAIN_RANGE = 0.03, (0.85, 1.15)

# Accuracy tolerances: relative (IR, RED) with an absolute floor, absolute (HR, SpO2)
TOL_REL = {"IR": 0.05, "RED": 0.05}
TOL_FLOOR = {"IR": 2500.0, "RED": 800.0, "HR": 3.0, "SpO2": 2.0}

TEST_SIZE = 0.20
SEQ_LEN = 10
LSTM_EPOCHS = 4 if QUICK else 25
LSTM_BATCH = 128
N_TREES = 20 if QUICK else 100

# GAN (conditional WGAN-GP)
LATENT_DIM = 32
GAN_STEPS = 60 if QUICK else 1500
GAN_BATCH = 256
GAN_N_CRITIC = 3
GAN_SYNTH_RATIO = 0.50        # synthetic rows added = 50 % of the real training rows
GAN_ABNORMAL_SHARE = 0.50     # share of synthetic rows requested with >= 1 abnormal channel

SEEDS = [42 + 11 * i for i in range(N_SEEDS)]
print("TensorFlow", tf.__version__, "| seeds:", SEEDS, "| quick:", QUICK)

# %% [markdown]
# ## 2. Load data

# %%
df = pd.read_csv(DATA_PATH)
df.columns = [str(c).strip() for c in df.columns]
rename = {c: w for w in TARGET_COLS + [GROUP_COL, TIME_COL] for c in df.columns if c.lower() == w.lower() and c != w}
df = df.rename(columns=rename)
missing = [c for c in TARGET_COLS if c not in df.columns]
if missing:
    raise ValueError(f"Dataset must contain {TARGET_COLS}. Missing: {missing}")
if GROUP_COL not in df.columns:
    df[GROUP_COL] = "single_recording"
if TIME_COL not in df.columns:
    df[TIME_COL] = df.groupby(GROUP_COL).cumcount()
df = df.sort_values([GROUP_COL, TIME_COL], kind="stable").reset_index(drop=True)
for c in TARGET_COLS:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=TARGET_COLS).reset_index(drop=True)

clean = df[TARGET_COLS].astype(float).reset_index(drop=True)
groups = df[GROUP_COL].to_numpy()
times = df[TIME_COL].to_numpy()

print("Shape:", df.shape, "| recordings:", df[GROUP_COL].nunique())
print(clean.describe().T[["min", "50%", "max"]])


def status_matrix(values):
    """True where the value lies OUTSIDE the channel's normal band (= abnormal)."""
    v = np.asarray(values, dtype=float)
    lo = np.array([NORMAL_LO[c] for c in TARGET_COLS])
    hi = np.array([NORMAL_HI[c] for c in TARGET_COLS])
    return (v < lo) | (v > hi)


st_all = status_matrix(clean.values)
print("\nAbnormal share per channel (%):", dict(zip(TARGET_COLS, (100 * st_all.mean(0)).round(1))))
print(f"Rows with >=1 abnormal channel: {100 * st_all.any(1).mean():.1f} %")

# %% [markdown]
# ## 3. Features (one function shared by training and testing)
# Everything is expressed relative to the NORMAL band, so the models can see "how far outside normal" a value is.

# %%
def extract_features(X):
    X = pd.DataFrame(X, columns=TARGET_COLS).astype(float)
    F, N = pd.DataFrame(index=X.index), pd.DataFrame(index=X.index)
    for c in TARGET_COLS:
        v = X[c]
        lo, hi = NORMAL_LO[c], NORMAL_HI[c]
        F[c] = np.log1p(v.clip(lower=0)) if c in RELATIVE_COLS else v
        N[c] = (v - lo) / (hi - lo)                                   # 0 = lower normal edge, 1 = upper edge
        F[f"{c}_below"] = (v < lo).astype(float).where(v.notna())
        F[f"{c}_above"] = (v > hi).astype(float).where(v.notna())
        F[f"{c}_dev"] = N[c].clip(upper=0) + (N[c] - 1).clip(lower=0)  # 0 inside band, <0 below, >0 above
    for i in range(len(TARGET_COLS)):
        for j in range(i + 1, len(TARGET_COLS)):
            a, b = TARGET_COLS[i], TARGET_COLS[j]
            F[f"diff_{a}_{b}"] = N[a] - N[b]
    F["n_below"] = sum(F[f"{c}_below"] for c in TARGET_COLS)
    F["n_above"] = sum(F[f"{c}_above"] for c in TARGET_COLS)
    F["row_mean"], F["row_std"] = N.mean(axis=1), N.std(axis=1)
    F["row_min"], F["row_max"] = N.min(axis=1), N.max(axis=1)
    return F.replace([np.inf, -np.inf], np.nan)


def to_model_space(X):
    X = pd.DataFrame(X, columns=TARGET_COLS).astype(float).copy()
    for c in RELATIVE_COLS:
        X[c] = np.log1p(X[c].clip(lower=0))
    return X


def from_model_space(A):
    A = pd.DataFrame(np.asarray(A, dtype=float), columns=TARGET_COLS)
    for c in RELATIVE_COLS:
        A[c] = np.expm1(A[c].clip(lower=0, upper=20))
    return A.values


def clip_to_range(pred):
    lo = np.array([DATA_LO[c] for c in TARGET_COLS], dtype=float)
    hi = np.array([DATA_HI[c] for c in TARGET_COLS], dtype=float)
    return np.clip(np.asarray(pred, dtype=float), lo, hi)


print("Engineered features:", extract_features(clean.head(5)).shape[1])

# %% [markdown]
# ## 4. Sensor-noise simulation
# Noise is relative (plus a tiny absolute floor), so a true `0` stays ~`0` and abnormal readings are never turned into normal ones.

# %%
def make_corrupted(Y, seed):
    rng = np.random.default_rng(seed)
    Y = np.asarray(Y, dtype=float)
    X = Y.copy()
    rel = np.array([NOISE_REL[c] for c in TARGET_COLS])
    floor = np.array([NOISE_FLOOR[c] for c in TARGET_COLS])
    scale = rel * np.abs(Y) + floor
    X += rng.normal(0, 1, X.shape) * scale
    spike = rng.random(X.shape) < SPIKE_PROB
    X += spike * rng.normal(0, 1, X.shape) * (SPIKE_REL * np.abs(Y) + 3 * floor)
    gain = rng.random(X.shape) < GAIN_PROB
    X[gain] *= rng.uniform(*GAIN_RANGE, gain.sum())
    keep = rng.random(len(X)) < CLEAN_ROW_FRAC                    # some rows are exact
    X[keep] = Y[keep]
    return clip_to_range(X)


# %% [markdown]
# ## 5. Metrics
# * **Tolerance accuracy** — prediction within tolerance of the true value (main score).
# * **Status accuracy** — predicted value is in the right band (normal vs abnormal).
# * **Abnormal kept %** — of the truly abnormal values, how many stay abnormal (this is the "do not predict normal for abnormal input" requirement).
# * **Normal kept %** — of the truly normal values, how many stay normal.

# %%
def tolerance_accuracy(yt, yp):
    per = {}
    for i, c in enumerate(TARGET_COLS):
        err = np.abs(yt[:, i] - yp[:, i])
        tol = np.maximum(TOL_REL[c] * np.abs(yt[:, i]), TOL_FLOOR[c]) if c in TOL_REL else TOL_FLOOR[c]
        per[c] = 100.0 * float(np.mean(err <= tol))
    return float(np.mean(list(per.values()))), per


def evaluate(y_true, y_pred, name):
    yt, yp = np.asarray(y_true, float), np.asarray(y_pred, float)
    mae = mean_absolute_error(yt, yp, multioutput="raw_values")
    rmse = np.sqrt(mean_squared_error(yt, yp, multioutput="raw_values"))
    r2 = r2_score(yt, yp, multioutput="raw_values")
    span = np.array([DATA_HI[c] - DATA_LO[c] for c in TARGET_COLS], float)
    acc, per = tolerance_accuracy(yt, yp)
    st_t, st_p = status_matrix(yt), status_matrix(yp)
    ab_kept = 100.0 * float((st_p & st_t).sum() / max(st_t.sum(), 1))
    nm_kept = 100.0 * float((~st_p & ~st_t).sum() / max((~st_t).sum(), 1))
    row = {"Model": name, "Tolerance_Accuracy_%": acc, "Status_Accuracy_%": 100.0 * float((st_t == st_p).mean()),
           "Abnormal_Kept_%": ab_kept, "Normal_Kept_%": nm_kept,
           "NMAE_%": 100.0 * float((mae / span).mean()), "R2": float(r2.mean()),
           "MAE": float(mae.mean()), "RMSE": float(rmse.mean())}
    for i, c in enumerate(TARGET_COLS):
        row[f"{c}_MAE"], row[f"{c}_R2"], row[f"{c}_Accuracy_%"] = float(mae[i]), float(r2[i]), per[c]
    return row


# %% [markdown]
# ## 6. Classical models

# %%
def make_preprocessor():
    return {"imputer": SimpleImputer(strategy="median"), "scaler": RobustScaler()}


def fit_prep(prep, X):
    return prep["scaler"].fit_transform(prep["imputer"].fit_transform(extract_features(X)))


def apply_prep(prep, X):
    return prep["scaler"].transform(prep["imputer"].transform(extract_features(X)))


def classic_factories(seed):
    return {
        "RandomForest": lambda: RandomForestRegressor(n_estimators=N_TREES, min_samples_leaf=8, random_state=seed, n_jobs=-1),
        "ExtraTrees": lambda: ExtraTreesRegressor(n_estimators=N_TREES, min_samples_leaf=8, random_state=seed, n_jobs=-1),
        "HistGradientBoosting": lambda: MultiOutputRegressor(HistGradientBoostingRegressor(
            max_iter=60 if QUICK else 250, learning_rate=0.05, l2_regularization=0.1,
            early_stopping=False, random_state=seed)),
    }


def train_classic(name, factory, X_fit, y_fit):
    """Targets are moved to model space (log1p for IR/RED) and standardised, so that all 4 outputs carry equal
    weight. Without this, multi-output trees only look at IR (its variance is ~1e8 x larger than HR's)."""
    prep = make_preprocessor()
    Xs = fit_prep(prep, X_fit)
    y_scaler = StandardScaler()
    ys = y_scaler.fit_transform(to_model_space(y_fit))
    model = factory()
    model.fit(Xs, ys)
    return {"type": "sklearn", "model": model, "imputer": prep["imputer"], "scaler": prep["scaler"],
            "y_scaler": y_scaler, "target_cols": TARGET_COLS}


def predict_classic(art, X):
    Xs = art["scaler"].transform(art["imputer"].transform(extract_features(X)))
    out = from_model_space(art["y_scaler"].inverse_transform(art["model"].predict(Xs)))
    return clip_to_range(out)


# %% [markdown]
# ## 7. Conditional GAN (WGAN-GP) for synthetic sensor records
# The generator is conditioned on the **normal/abnormal pattern** of the 4 channels, so we can ask for e.g.
# "IR abnormal + SpO2 abnormal, others normal". Generated records whose actual status does not match the
# requested pattern are rejected. Real clean records are used as training data; synthetic records are then corrupted
# with the same noise model and added to the training set.

# %%
def sample_conditions(cond_real, n, rng, abnormal_share):
    ab_rows = cond_real[cond_real.sum(1) > 0]
    n_ab = int(round(n * abnormal_share)) if len(ab_rows) else 0
    c_ab = ab_rows[rng.integers(0, len(ab_rows), n_ab)] if n_ab else np.zeros((0, len(TARGET_COLS)), np.float32)
    c = np.vstack([c_ab, np.zeros((n - n_ab, len(TARGET_COLS)), np.float32)])
    return c[rng.permutation(len(c))].astype(np.float32)


class ConditionalGAN:
    def __init__(self, seed):
        self.seed = seed

    def fit(self, Y_clean):
        tf.random.set_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        D, C = len(TARGET_COLS), len(TARGET_COLS)
        Ym = to_model_space(Y_clean)
        self.scaler = StandardScaler().fit(Ym.values)
        real = self.scaler.transform(Ym.values).astype(np.float32)
        cond = status_matrix(Y_clean.values).astype(np.float32)
        self.cond_real = cond
        ab_idx, nm_idx = np.where(cond.sum(1) > 0)[0], np.where(cond.sum(1) == 0)[0]

        def mlp(inp, out, name):
            h = layers.Concatenate()(inp)
            for _ in range(3):
                h = layers.LeakyReLU(0.2)(layers.Dense(128)(h))
            return models.Model(inp, layers.Dense(out)(h), name=name)

        self.G = mlp([layers.Input((LATENT_DIM,)), layers.Input((C,))], D, "generator")
        critic = mlp([layers.Input((D,)), layers.Input((C,))], 1, "critic")
        optG = tf.keras.optimizers.Adam(1e-4, 0.5, 0.9)
        optC = tf.keras.optimizers.Adam(1e-4, 0.5, 0.9)
        G = self.G

        @tf.function
        def critic_step(xr, c):
            B = tf.shape(xr)[0]
            z = tf.random.normal((B, LATENT_DIM))
            with tf.GradientTape() as tape:
                xf = G([z, c], training=True)
                eps = tf.random.uniform((B, 1))
                xh = eps * xr + (1 - eps) * xf
                with tf.GradientTape() as gp_tape:
                    gp_tape.watch(xh)
                    d_hat = critic([xh, c], training=True)
                g = gp_tape.gradient(d_hat, xh)
                gp = tf.reduce_mean((tf.sqrt(tf.reduce_sum(tf.square(g), 1) + 1e-12) - 1.0) ** 2)
                loss = tf.reduce_mean(critic([xf, c], training=True)) - tf.reduce_mean(critic([xr, c], training=True)) + 10.0 * gp
            critic_opt_grads = tape.gradient(loss, critic.trainable_variables)
            optC.apply_gradients(zip(critic_opt_grads, critic.trainable_variables))
            return loss

        @tf.function
        def gen_step(c):
            z = tf.random.normal((tf.shape(c)[0], LATENT_DIM))
            with tf.GradientTape() as tape:
                loss = -tf.reduce_mean(critic([G([z, c], training=True), c], training=True))
            optG.apply_gradients(zip(tape.gradient(loss, G.trainable_variables), G.trainable_variables))
            return loss

        def batch():
            half = GAN_BATCH // 2        # half abnormal, half normal rows -> critic sees rare patterns often
            idx = np.concatenate([rng.choice(ab_idx, half) if len(ab_idx) else rng.choice(nm_idx, half),
                                  rng.choice(nm_idx, GAN_BATCH - half)])
            return tf.constant(real[idx]), tf.constant(cond[idx])

        self.history = {"critic": [], "gen": []}
        for step in range(GAN_STEPS):
            for _ in range(GAN_N_CRITIC):
                xr, c = batch()
                dl = critic_step(xr, c)
            _, c = batch()
            gl = gen_step(c)
            self.history["critic"].append(float(dl))
            self.history["gen"].append(float(gl))
        return self

    def generate(self, cond, rng):
        z = rng.normal(0, 1, (len(cond), LATENT_DIM)).astype(np.float32)
        out = self.G.predict([z, cond.astype(np.float32)], verbose=0)
        raw = from_model_space(self.scaler.inverse_transform(out))
        return np.rint(clip_to_range(raw))

    def sample(self, n, rng, abnormal_share=GAN_ABNORMAL_SHARE):
        """n accepted synthetic CLEAN records (+ requested conditions). Quotas are filled separately for abnormal
        and normal patterns, so rejection of mismatching samples cannot skew the abnormal/normal mix."""
        n_ab = int(round(n * abnormal_share))
        out_r, out_c = [], []
        for quota, share in ((n_ab, 1.0), (n - n_ab, 0.0)):
            got_r, got_c, need, tries = [], [], quota, 0
            while need > 0 and tries < 12:
                cond = sample_conditions(self.cond_real, int(need * 1.6) + 16, rng, share)
                rec = self.generate(cond, rng)
                ok = (status_matrix(rec) == cond.astype(bool)).all(1)       # reject pattern mismatches
                got_r.append(rec[ok]); got_c.append(cond[ok])
                need -= int(ok.sum()); tries += 1
            if got_r:
                out_r.append(np.vstack(got_r)[:quota]); out_c.append(np.vstack(got_c)[:quota])
        rec, cond = np.vstack(out_r), np.vstack(out_c)
        perm = rng.permutation(len(rec))
        return rec[perm], cond[perm]


def synthetic_recordings(gan, n_rows, rng):
    """Turn GAN records into short contiguous 'episodes' (8-40 s) so the LSTM gets realistic sequences."""
    n_seg = int(n_rows / 20) + 20
    base, cond = gan.sample(n_seg, rng)
    rows = []
    for b, c in zip(base, cond):
        L = int(rng.integers(8, 41))
        seg = np.tile(b, (L, 1)).astype(float)
        for j, ch in enumerate(TARGET_COLS):
            if b[j] == 0:                                       # exact zero stays exact zero
                continue
            e = rng.normal(0, 0.01 if c[j] == 0 else 0.02, L)
            ar = np.zeros(L); ar[0] = e[0]
            for t in range(1, L):
                ar[t] = 0.9 * ar[t - 1] + e[t] * 0.45
            seg[:, j] = b[j] * (1 + ar)
        rows.append(seg)
        if sum(len(r) for r in rows) >= n_rows:
            break
    out = clip_to_range(np.vstack(rows))
    return pd.DataFrame(out[:n_rows], columns=TARGET_COLS)


# %% [markdown]
# ## 8. LSTM helpers

# %%
def to_sequences(X, y, seq_len, grp, tim):
    """Sliding windows that never cross a recording boundary or a time gap. Returns windows, targets, end-row indices."""
    X, y = np.asarray(X), np.asarray(y, dtype=np.float32)
    grp, tim = np.asarray(grp), np.asarray(tim, dtype=float)
    xs, ys, ends = [], [], []
    for i in range(seq_len - 1, len(X)):
        s = i - seq_len + 1
        if grp[s] != grp[i] or (tim[i] - tim[s]) != seq_len - 1:
            continue
        xs.append(X[s:i + 1]); ys.append(y[i]); ends.append(i)
    return np.asarray(xs, np.float32), np.asarray(ys, np.float32), np.asarray(ends)


def build_lstm(n_features, seed):
    tf.random.set_seed(seed)
    m = models.Sequential([
        layers.Input(shape=(SEQ_LEN, n_features)),
        layers.LSTM(64, return_sequences=True), layers.Dropout(0.2),
        layers.LSTM(32), layers.Dense(32, activation="relu"), layers.Dense(len(TARGET_COLS))])
    m.compile(optimizer="adam", loss="mse")
    return m


def train_lstm(X_raw, Y_clean, grp, tim, seed):
    """X_raw: noisy inputs, Y_clean: clean targets (raw units). Returns model + preprocessing bundle + history."""
    imp, sc, ysc = SimpleImputer(strategy="median"), RobustScaler(), StandardScaler()
    Xs = sc.fit_transform(imp.fit_transform(extract_features(X_raw)))
    ys = ysc.fit_transform(to_model_space(Y_clean))
    Xw, yw, _ = to_sequences(Xs, ys, SEQ_LEN, grp, tim)
    model = build_lstm(Xw.shape[-1], seed)
    hist = model.fit(Xw, yw, validation_split=0.15, epochs=LSTM_EPOCHS, batch_size=LSTM_BATCH, verbose=0,
                     callbacks=[callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)])
    prep = {"imputer": imp, "scaler": sc, "y_scaler": ysc, "relative_cols": RELATIVE_COLS, "log_space": True,
            "target_cols": TARGET_COLS, "seq_len": SEQ_LEN}
    return model, prep, hist.history


def predict_lstm_windows(model, prep, X_raw, grp, tim, y_dummy):
    Xs = prep["scaler"].transform(prep["imputer"].transform(extract_features(X_raw)))
    Xw, _, ends = to_sequences(Xs, y_dummy, SEQ_LEN, grp, tim)
    out = model.predict(Xw, verbose=0, batch_size=1024)
    return clip_to_range(from_model_space(prep["y_scaler"].inverse_transform(out))), ends


# %% [markdown]
# ## 9. One complete experiment (split -> 8 models -> evaluation)

# %%
def run_experiment(seed, save_artifacts=False):
    t0 = time.time()
    np.random.seed(seed); random.seed(seed); tf.random.set_seed(seed)
    rng = np.random.default_rng(seed)

    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=seed)
    tr, te = next(gss.split(clean, groups=groups))
    tr, te = np.sort(tr), np.sort(te)
    y_tr, y_te = clean.iloc[tr].reset_index(drop=True), clean.iloc[te].reset_index(drop=True)
    g_tr, t_tr, g_te, t_te = groups[tr], times[tr], groups[te], times[te]
    X_tr = pd.DataFrame(make_corrupted(y_tr.values, seed + 1), columns=TARGET_COLS)
    X_te = pd.DataFrame(make_corrupted(y_te.values, seed + 2), columns=TARGET_COLS)
    print(f"[seed {seed}] train rows {len(tr)} ({len(set(g_tr))} recordings) | test rows {len(te)} ({len(set(g_te))} recordings)")

    # Every model is scored on the SAME rows: the last row of each valid LSTM window of the test recordings.
    _, _, ends = to_sequences(np.zeros((len(te), 1)), np.zeros((len(te), 4)), SEQ_LEN, g_te, t_te)
    y_eval = y_te.iloc[ends].values
    results, preds, artifacts = [], {}, {}

    def record(name, pred):
        preds[name] = np.asarray(pred)
        results.append(evaluate(y_eval, pred, name))

    # ---- classical, No GAN
    for name, fac in classic_factories(seed).items():
        art = train_classic(name, fac, X_tr, y_tr)
        artifacts[f"{name}_NoGAN"] = art
        record(f"{name} (No GAN)", predict_classic(art, X_te.iloc[ends]))
    print(f"[seed {seed}] classical No-GAN done  ({time.time() - t0:.0f}s)")

    # ---- GAN
    gan = ConditionalGAN(seed).fit(y_tr)
    n_syn = int(GAN_SYNTH_RATIO * len(y_tr))
    syn_clean, _ = gan.sample(n_syn, rng)
    syn_clean = pd.DataFrame(syn_clean, columns=TARGET_COLS)
    syn_noisy = pd.DataFrame(make_corrupted(syn_clean.values, seed + 3), columns=TARGET_COLS)
    X_aug = pd.concat([X_tr, syn_noisy], ignore_index=True)
    y_aug = pd.concat([y_tr, syn_clean], ignore_index=True)
    print(f"[seed {seed}] GAN trained: {len(syn_clean)} synthetic rows "
          f"({100 * status_matrix(syn_clean.values).any(1).mean():.0f}% abnormal)  ({time.time() - t0:.0f}s)")

    # ---- classical, GAN
    for name, fac in classic_factories(seed).items():
        art = train_classic(name, fac, X_aug, y_aug)
        artifacts[f"{name}_GAN"] = art
        record(f"{name} (GAN)", predict_classic(art, X_te.iloc[ends]))
    print(f"[seed {seed}] classical GAN done  ({time.time() - t0:.0f}s)")

    # ---- LSTM, No GAN
    m_no, prep_no, h_no = train_lstm(X_tr, y_tr, g_tr, t_tr, seed)
    p, e = predict_lstm_windows(m_no, prep_no, X_te, g_te, t_te, y_te.values)
    assert (e == ends).all()
    record("LSTM (No GAN)", p)

    # ---- LSTM, GAN  (real recordings + synthetic episodes as one extra 'recording')
    syn_seq_clean = synthetic_recordings(gan, int(GAN_SYNTH_RATIO * len(y_tr)), rng)
    syn_seq_noisy = pd.DataFrame(make_corrupted(syn_seq_clean.values, seed + 4), columns=TARGET_COLS)
    X_l = pd.concat([X_tr, syn_seq_noisy], ignore_index=True)
    y_l = pd.concat([y_tr, syn_seq_clean], ignore_index=True)
    g_l = np.concatenate([g_tr, np.full(len(syn_seq_clean), "__synthetic__")])
    t_l = np.concatenate([t_tr, np.arange(len(syn_seq_clean))])
    m_gan, prep_gan, h_gan = train_lstm(X_l, y_l, g_l, t_l, seed)
    p, e = predict_lstm_windows(m_gan, prep_gan, X_te, g_te, t_te, y_te.values)
    record("LSTM (GAN)", p)
    print(f"[seed {seed}] LSTM done  ({time.time() - t0:.0f}s)")

    out = {"results": pd.DataFrame(results).assign(Seed=seed), "y_eval": y_eval, "preds": preds,
           "gan_hist": gan.history, "lstm_hist": {"No GAN": h_no, "GAN": h_gan},
           "syn_clean": syn_clean, "real_clean": y_tr}
    if save_artifacts:
        for k, a in artifacts.items():
            joblib.dump(a, OUT_DIR / f"{k}.joblib", compress=3)
        m_no.save(OUT_DIR / "LSTM_NoGAN.keras"); joblib.dump(prep_no, OUT_DIR / "LSTM_NoGAN_preprocessor.joblib")
        m_gan.save(OUT_DIR / "LSTM_GAN.keras"); joblib.dump(prep_gan, OUT_DIR / "LSTM_GAN_preprocessor.joblib")
        gan.G.save(OUT_DIR / "sensor_cgan_generator.keras")
        joblib.dump(gan.scaler, OUT_DIR / "sensor_cgan_scaler.joblib")
    return out


# %% [markdown]
# ## 10. Run all seeds

# %%
runs = []
for k, s in enumerate(SEEDS):
    runs.append(run_experiment(s, save_artifacts=(k == 0)))      # models of the FIRST seed are the ones saved for testing
all_res = pd.concat([r["results"] for r in runs], ignore_index=True)
all_res.to_csv(OUT_DIR / "model_comparison_all_seeds.csv", index=False)

# %% [markdown]
# ## 11. Summary over seeds

# %%
METRICS = ["Tolerance_Accuracy_%", "Status_Accuracy_%", "Abnormal_Kept_%", "Normal_Kept_%", "NMAE_%", "R2"]
g = all_res.groupby("Model")[METRICS]
summary = g.mean().join(g.std().add_suffix("_std")).sort_values("Tolerance_Accuracy_%", ascending=False)
summary.to_csv(OUT_DIR / "model_comparison_summary.csv")
pd.set_option("display.width", 220, "display.max_columns", 30)
print("\n=== Mean over", len(SEEDS), "seeds (sorted by tolerance accuracy) ===")
print(summary[METRICS].round(2).to_string())

print("\n=== GAN vs No-GAN, paired per seed (positive = GAN better) ===")
rows = []
for alg in ["RandomForest", "ExtraTrees", "HistGradientBoosting", "LSTM"]:
    a = all_res[all_res.Model == f"{alg} (GAN)"].sort_values("Seed")
    b = all_res[all_res.Model == f"{alg} (No GAN)"].sort_values("Seed")
    d = a["Tolerance_Accuracy_%"].to_numpy() - b["Tolerance_Accuracy_%"].to_numpy()
    d_ab = a["Abnormal_Kept_%"].to_numpy() - b["Abnormal_Kept_%"].to_numpy()
    rows.append({"Algorithm": alg, "dAccuracy_%": d.mean(), "GAN wins (seeds)": f"{(d > 0).sum()}/{len(d)}",
                 "dAbnormal_Kept_%": d_ab.mean()})
gan_vs = pd.DataFrame(rows)
gan_vs.to_csv(OUT_DIR / "gan_vs_nogan.csv", index=False)
print(gan_vs.round(3).to_string(index=False))

best = summary.index[0]
n_gan_better = int((gan_vs["dAccuracy_%"] > 0).sum())
print(f"\nBest model by mean tolerance accuracy : {best}")
print(f"GAN variant better than its No-GAN twin: {n_gan_better} of {len(gan_vs)} algorithms")
if "(GAN)" in best:
    print("=> The best overall model is a GAN-augmented model.")
else:
    print("=> The best overall model is NOT a GAN-augmented model in this experiment.")

# %% [markdown]
# ## 12. Plots

# %%
order = list(summary.index)
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for ax, m in zip(axes, ["Tolerance_Accuracy_%", "Abnormal_Kept_%", "NMAE_%"]):
    sns.barplot(data=all_res, y="Model", x=m, order=order, errorbar="sd", ax=ax, capsize=0.2)
    ax.set_title(m + " (mean +- sd over seeds)"); ax.set_ylabel("")
plt.tight_layout(); plt.savefig(PLOT_DIR / "01_model_comparison.png", dpi=170); plt.close()

plt.figure(figsize=(11, 5))
tmp = all_res.assign(Algorithm=all_res.Model.str.replace(r" \(.*\)", "", regex=True),
                     Variant=np.where(all_res.Model.str.contains(r"\(GAN\)"), "GAN", "No GAN"))
sns.barplot(data=tmp, x="Algorithm", y="Tolerance_Accuracy_%", hue="Variant", errorbar="sd", capsize=0.1)
plt.title("GAN vs No-GAN per algorithm (mean +- sd over seeds)"); plt.tight_layout()
plt.savefig(PLOT_DIR / "02_gan_vs_nogan.png", dpi=170); plt.close()

r0 = runs[0]
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(r0["gan_hist"]["critic"], label="critic"); axes[0].plot(r0["gan_hist"]["gen"], label="generator")
axes[0].set_title("cGAN (WGAN-GP) training loss"); axes[0].legend()
for k, h in r0["lstm_hist"].items():
    axes[1].plot(h["val_loss"], label=f"LSTM {k} val")
axes[1].set_title("LSTM validation loss"); axes[1].legend()
plt.tight_layout(); plt.savefig(PLOT_DIR / "03_training_curves.png", dpi=170); plt.close()

fig, axes = plt.subplots(1, 4, figsize=(18, 4))
for i, c in enumerate(TARGET_COLS):
    real, syn = r0["real_clean"][c], r0["syn_clean"][c]
    bins = np.histogram_bin_edges(np.concatenate([real, syn]), bins=40)
    axes[i].hist(real, bins, alpha=0.5, density=True, label="real"); axes[i].hist(syn, bins, alpha=0.5, density=True, label="GAN")
    axes[i].set_title(c); axes[i].legend()
plt.suptitle("Real vs GAN-generated clean records"); plt.tight_layout()
plt.savefig(PLOT_DIR / "04_real_vs_gan_distribution.png", dpi=170); plt.close()

best_pred, y_ev = r0["preds"][best], r0["y_eval"]
fig, axes = plt.subplots(2, 2, figsize=(13, 10))
for i, c in enumerate(TARGET_COLS):
    ax = axes.ravel()[i]
    ax.scatter(y_ev[:, i], best_pred[:, i], s=6, alpha=0.3)
    lim = [0, max(y_ev[:, i].max(), best_pred[:, i].max())]
    ax.plot(lim, lim, "r--"); ax.set_title(f"{c}: actual vs predicted ({best})")
plt.tight_layout(); plt.savefig(PLOT_DIR / "05_actual_vs_predicted_best.png", dpi=170); plt.close()
print("Plots saved to", PLOT_DIR)

# %% [markdown]
# ## 13. Save metadata (read by the test script)

# %%
metadata = {
    "target_columns": TARGET_COLS, "relative_columns": RELATIVE_COLS, "sequence_length": SEQ_LEN,
    "data_lo": DATA_LO, "data_hi": DATA_HI, "normal_lo": NORMAL_LO, "normal_hi": NORMAL_HI,
    "target_min": DATA_LO, "target_max": DATA_HI,
    "noise_rel": NOISE_REL, "noise_floor": NOISE_FLOOR, "tolerance_rel": TOL_REL, "tolerance_floor": TOL_FLOOR,
    "seeds": SEEDS, "saved_models_from_seed": SEEDS[0],
    "algorithms": ["RandomForest", "ExtraTrees", "HistGradientBoosting", "LSTM"], "variants": ["No GAN", "GAN"],
    "best_model_mean_over_seeds": best,
    "summary": summary[METRICS].round(3).to_dict(orient="index"),
}
with open(OUT_DIR / "training_metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)
print("Saved:", sorted(p.name for p in OUT_DIR.iterdir()))
print("Training complete. Model directory:", OUT_DIR.resolve())
