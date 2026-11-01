"""
generate_dataset.py
-------------------
Builds a SIMULATED sensor dataset in exactly the same format as merged_numerics.csv:

    Source_File, Time [s], HR, IR, RED, SpO2        (1 row per second, 481 s per recording)

The data now spans the full range from 0 up to the upper limits you specified:

    IR   : 0 .. 1,124,224      (normal operating band 46,575 .. 1,124,224)
    RED  : 0 ..    89,898      (normal operating band 67,676 ..    89,898)
    HR   : 0 ..        89      (normal operating band     40 ..        89)
    SpO2 : 0 ..       120      (normal operating band     90 ..       100)

Anything outside a "normal band" is ABNORMAL (sensor off, dropout to 0, tiny values like 10,
very low HR / SpO2, SpO2 > 100 artefact ...). Abnormal readings appear as contiguous
EPISODES inside each recording, so sequence models (LSTM) see realistic onset/offset.

The CSV holds the CLEAN reference values. The training notebook adds sensor noise on top of them.

Run:  python generate_dataset.py            -> merged_numerics_extended.csv
"""
import numpy as np
import pandas as pd

SEED = 2024
N_RECORDINGS = 60
N_SECONDS = 481
P_EPISODE_START = 0.014          # per-second chance that an abnormal episode starts (~25 % abnormal rows)
OUT_FILE = "merged_numerics_extended.csv"

# ---- ranges given by the user -------------------------------------------------------------
DATA_HI = {"IR": 1_124_224, "RED": 89_898, "HR": 89, "SpO2": 120}
NORMAL_LO = {"IR": 46_575, "RED": 67_676, "HR": 40, "SpO2": 90}
NORMAL_HI = {"IR": 1_124_224, "RED": 89_898, "HR": 89, "SpO2": 100}
COLS = ["IR", "RED", "HR", "SpO2"]

rng = np.random.default_rng(SEED)


def ar1(n, phi, sigma):
    """Stationary AR(1) series with marginal std = sigma."""
    e = rng.normal(0, sigma * np.sqrt(1 - phi ** 2), n)
    x = np.zeros(n)
    x[0] = rng.normal(0, sigma)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    return x


def normal_recording(n):
    ir_base = np.exp(rng.uniform(np.log(80_000), np.log(1_000_000)))   # recordings differ a lot in IR level
    ir_rel = ar1(n, 0.985, 0.06)
    ir = ir_base * (1 + ir_rel)
    red_base = rng.uniform(70_000, 86_000)
    red = red_base * (1 + 0.20 * ir_rel + ar1(n, 0.95, 0.01))
    hr = rng.uniform(52, 84) + ar1(n, 0.97, 3.0)
    spo2 = rng.uniform(94, 99) + ar1(n, 0.97, 0.8)
    out = {
        "IR": np.clip(ir, NORMAL_LO["IR"] + 3_000, NORMAL_HI["IR"] - 3_000),
        "RED": np.clip(red, NORMAL_LO["RED"] + 500, NORMAL_HI["RED"] - 500),
        "HR": np.clip(hr, NORMAL_LO["HR"] + 2, NORMAL_HI["HR"] - 1),
        "SpO2": np.clip(spo2, NORMAL_LO["SpO2"] + 1, NORMAL_HI["SpO2"]),
    }
    return out


def abnormal_level(c):
    """One abnormal level for channel c. Mix of exact 0, tiny values (1..~50), and 'low' values."""
    kind = rng.choice(["zero", "tiny", "low", "high"], p=[0.30, 0.20, 0.40, 0.10])
    if kind == "high" and c != "SpO2":
        kind = "low"
    if kind == "zero":
        return 0.0, True
    if kind == "tiny":
        hi = {"IR": 50, "RED": 50, "HR": 15, "SpO2": 20}[c]
        return float(rng.integers(1, hi + 1)), False
    if kind == "low":
        if c == "SpO2":
            return float(rng.uniform(0, 88)), False
        if c == "HR":
            return float(rng.uniform(0, 37)), False
        return float(rng.uniform(0, 0.93 * NORMAL_LO[c])), False
    return float(rng.uniform(102, 120)), False              # SpO2 > 100 (impossible / artefact)


def add_episodes(data, n):
    """Overwrite contiguous stretches with abnormal values (1-4 channels at a time)."""
    t = 0
    while t < n:
        if rng.random() < P_EPISODE_START:
            dur = int(rng.integers(6, 41))
            k = int(rng.choice([1, 2, 3, 4], p=[0.50, 0.25, 0.10, 0.15]))   # 4 = whole sensor off
            chans = list(rng.choice(COLS, size=k, replace=False))
            end = min(n, t + dur)
            for c in chans:
                level, exact_zero = abnormal_level(c)
                seg = np.full(end - t, level)
                if not exact_zero and level > 0:
                    seg = seg * (1 + rng.normal(0, 0.02, end - t))
                seg = np.clip(seg, 0, DATA_HI[c])
                data[c][t:end] = seg
            t = end + 1
        else:
            t += 1


def status(df):
    return pd.DataFrame({c: (df[c] < NORMAL_LO[c]) | (df[c] > NORMAL_HI[c]) for c in COLS})


def main():
    frames = []
    for r in range(1, N_RECORDINGS + 1):
        data = normal_recording(N_SECONDS)
        add_episodes(data, N_SECONDS)
        df = pd.DataFrame({c: np.rint(data[c]).astype(np.int64) for c in COLS})
        for c in COLS:                                     # hard dataset limits
            df[c] = df[c].clip(0, DATA_HI[c])
        df.insert(0, "Time [s]", np.arange(N_SECONDS))
        df.insert(0, "Source_File", f"sim_{r:02d}_Numerics.csv")
        frames.append(df)

    full = pd.concat(frames, ignore_index=True)
    # same column order / header style as merged_numerics.csv  (Source_File, Time [s], HR, IR, RED, SpO2)
    full = full[["Source_File", "Time [s]", "HR", "IR", "RED", "SpO2"]]
    full.columns = ["Source_File", "Time [s]", " HR", "IR", "RED", " SpO2"]
    full.to_csv(OUT_FILE, index=False)

    chk = full.rename(columns=lambda c: c.strip())
    st = status(chk)
    print(f"Wrote {OUT_FILE}: {full.shape[0]} rows, {chk.Source_File.nunique()} recordings")
    print("\nValue ranges:")
    print(chk[COLS].agg(["min", "max"]).T.to_string())
    print("\nAbnormal share per channel (%):")
    print((100 * st.mean()).round(1).to_string())
    print(f"Rows with at least one abnormal channel: {100 * st.any(axis=1).mean():.1f} %")
    print(f"Rows where channel value is exactly 0   : "
          + ", ".join(f"{c}={100 * (chk[c] == 0).mean():.1f}%" for c in COLS))


if __name__ == "__main__":
    main()
