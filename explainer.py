"""
explainer.py
------------
Feature attribution for sklearn + Keras models, plus a rule engine.

Attribution strategy per model type:
  * sklearn tree models -> shap.TreeExplainer (fast, exact)
  * sklearn other       -> shap.PermutationExplainer (robust, no flatten)
  * Keras LSTM          -> shap.GradientExplainer with 3-D background
                           fallback: tf.GradientTape saliency
                           fallback: permutation importance on inputs

If SHAP itself is missing, we still produce permutation importance and
the rule-based causes/remedies so the XAI panel is never empty.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

from predictor import (
    TARGET_COLS, NORMAL_LO, NORMAL_HI,
    extract_features, band_label,
)


# ===========================================================================
# RULE ENGINE — always runs, never fails
# ===========================================================================
def _severity_from_status(status: str) -> str:
    if "SpO2 HIGH" in status or "SpO2 LOW" in status or "HR HIGH" in status:
        return "critical"
    if "LOW" in status or "HIGH" in status:
        return "warning"
    return "info"


def _ratio_reason(IR, RED):
    if RED == 0 or IR == 0:
        return None
    r = IR / RED
    if r < 0.8:
        return f"IR/RED ratio is unusually low ({r:.2f}) — weak perfusion or sensor not seated"
    if r > 3.0:
        return f"IR/RED ratio is unusually high ({r:.2f}) — possible motion artefact or ambient light"
    return None


def rule_engine(latest: dict) -> dict:
    causes, remedies = [], []
    status = "/".join(band_label(c, latest[c]) for c in TARGET_COLS)
    severity = _severity_from_status(status)

    IR, RED, HR, SpO2 = (latest.get(c, 0) for c in TARGET_COLS)

    if max(IR, RED, HR, SpO2) < 50:
        causes.append("Sensor appears off — all channels near zero")
        remedies.append("Check the sensor is powered and the finger is inserted")
        remedies.append("Verify cable/connector between sensor and MCU")
        return {"causes": causes, "remedies": remedies, "severity": "critical"}

    if SpO2 < NORMAL_LO["SpO2"]:
        causes.append(f"SpO2 {SpO2:.0f}% is below the 90% safe threshold")
        remedies.append("Reposition the finger and confirm the sensor is well-seated")
        remedies.append("Warm cold extremities — poor perfusion lowers SpO2")
        remedies.append("If confirmed, apply supplemental oxygen and escalate")
    elif SpO2 > NORMAL_HI["SpO2"]:
        causes.append(f"SpO2 {SpO2:.0f}% exceeds 100% — sensor artefact")
        remedies.append("Check for ambient light entering the sensor")
        remedies.append("Inspect ADC saturation — reduce LED current")

    if HR < NORMAL_LO["HR"]:
        causes.append(f"Heart rate {HR:.0f} bpm is below the 40 bpm floor")
        remedies.append("Look for motion artefacts or poor contact")
        remedies.append("If sustained, consider clinical bradycardia review")
    elif HR > NORMAL_HI["HR"]:
        causes.append(f"Heart rate {HR:.0f} bpm exceeds the 89 bpm ceiling")
        remedies.append("Rule out motion — ask patient to stay still")
        remedies.append("If sustained, consider clinical tachycardia review")

    ratio = _ratio_reason(IR, RED)
    if ratio:
        causes.append(ratio)
        remedies.append("Re-seat the sensor; ensure no ambient light enters")
        remedies.append("Check for finger movement and cable strain")

    if not causes:
        causes.append("No abnormal pattern detected by the rule engine")
        remedies.append("Continue monitoring")

    return {"causes": causes, "remedies": remedies, "severity": severity}


# ===========================================================================
# ATTRIBUTION — three strategies, tried in order
# ===========================================================================

def _shap_tree_explainer(model, background_features: np.ndarray):
    """Fast & exact for RandomForest / ExtraTrees / HistGradientBoosting."""
    return shap.TreeExplainer(model)


def _shap_permutation_explainer(predict_fn, background_features: np.ndarray):
    """Model-agnostic, uses arbitrary callable, no internal shape tricks."""
    return shap.PermutationExplainer(predict_fn, background_features)


def _shap_gradient_explainer_lstm(keras_model, background_3d: np.ndarray):
    """Correct SHAP explainer for Keras sequences. background_3d shape: (N, T, F)."""
    return shap.GradientExplainer(keras_model, background_3d)


# -- fallback 1: gradient saliency via tf.GradientTape ----------------------
def _gradient_saliency(keras_model, x_3d: np.ndarray) -> np.ndarray:
    """
    Returns per-feature importance (averaged over time and outputs).
    x_3d shape: (1, T, F).
    """
    try:
        import tensorflow as tf
    except Exception:
        return None

    x = tf.convert_to_tensor(x_3d, dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(x)
        y = keras_model(x, training=False)   # (1, n_outputs)
    grads = tape.gradient(y, x)              # (1, T, F)
    if grads is None:
        return None
    g = np.abs(grads.numpy())                # (1, T, F)
    return g.mean(axis=(0, 1))               # (F,)


# -- fallback 2: permutation importance on the raw input -------------------
def _permutation_importance(predict_fn, x_row: dict, n_perturb: int = 8) -> dict:
    """
    For each feature, replace it with a perturbation (median of the normal band)
    and measure the L2 change in prediction. Returns {feature: score}.
    """
    base = predict_fn(pd.DataFrame([x_row]))
    base = np.asarray(base, dtype=float).flatten()

    scores = {}
    for c in TARGET_COLS:
        deltas = []
        for _ in range(n_perturb):
            lo, hi = NORMAL_LO[c], NORMAL_HI[c]
            perturbed = dict(x_row)
            perturbed[c] = float(np.random.uniform(lo, hi))
            y = predict_fn(pd.DataFrame([perturbed]))
            y = np.asarray(y, dtype=float).flatten()
            deltas.append(float(np.linalg.norm(y - base)))
        scores[c] = float(np.mean(deltas))

    # normalise to a comparable scale
    total = sum(scores.values()) or 1.0
    return {k: v / total for k, v in scores.items()}


# ===========================================================================
# UNIFIED EXPLAINER
# ===========================================================================
class ModelExplainer:
    def __init__(self, predictor, background_size: int = 20):
        self.predictor = predictor
        self.background_size = background_size
        self._cache = {}   # name -> "explainer info"

    # ---- background builders --------------------------------------------
    def _background_raw(self) -> pd.DataFrame:
        rng = np.random.default_rng(42)
        cols = {c: rng.uniform(NORMAL_LO[c], NORMAL_HI[c], self.background_size)
                for c in TARGET_COLS}
        return pd.DataFrame(cols, columns=TARGET_COLS)

    # ---- public API ------------------------------------------------------
    def explain(self, name: str, latest: dict, top_k: int = 8) -> dict | None:
        """
        Returns:
          {
            "method": "shap_tree" | "shap_permutation" | "shap_gradient" |
                      "saliency" | "permutation" | "unavailable",
            "contributions": [{"feature","value","direction"} ...],
            "base_value": float,
          }
        or None if nothing could be computed.
        """
        if name not in self.predictor.models:
            return None

        bundle = self.predictor.models[name]

        # ---- LSTM --------------------------------------------------------
        if name.startswith("LSTM"):
            return self._explain_lstm(name, bundle, latest, top_k)

        # ---- classical ---------------------------------------------------
        return self._explain_classical(name, bundle, latest, top_k)

    # ------------------------------------------------------------------
    # Classical sklearn models
    # ------------------------------------------------------------------
    def _explain_classical(self, name, bundle, latest, top_k):
        imputer, scaler = bundle["imputer"], bundle["scaler"]
        model = bundle["model"]

        feats = extract_features(pd.DataFrame([latest]))
        x = scaler.transform(imputer.transform(feats))

        contributions = None
        method = "unavailable"
        base_value = 0.0

        # 1) TreeExplainer — fastest & exact for forests / boosting
        if SHAP_AVAILABLE:
            try:
                expl = self._cache.get(name)
                if expl is None:
                    try:
                        expl = shap.TreeExplainer(model)
                    except Exception:
                        expl = None
                    self._cache[name] = expl

                if expl is not None:
                    sv = expl.shap_values(x)
                    sv = self._reduce_shap_output(sv, x.shape[1])
                    contributions = sv
                    method = "shap_tree"
                    base_value = float(np.mean(expl.expected_value)) if np.ndim(expl.expected_value) == 0 \
                        else float(np.mean(expl.expected_value))
            except Exception as e:
                print(f"[explainer] TreeExplainer failed for {name}: {e}")

        # 2) PermutationExplainer as SHAP fallback
        if contributions is None and SHAP_AVAILABLE:
            try:
                bg = scaler.transform(imputer.transform(self._background_raw()))
                fn = lambda arr: model.predict(arr)
                expl = shap.PermutationExplainer(fn, bg, seed=0)
                sv = expl(x, max_evals=64, silent=True).values
                sv = self._reduce_shap_output(sv, x.shape[1])
                contributions = sv
                method = "shap_permutation"
            except Exception as e:
                print(f"[explainer] PermutationExplainer failed for {name}: {e}")

        # 3) Pure-python permutation importance
        if contributions is None:
            def raw_predict(df: pd.DataFrame):
                f = extract_features(df)
                xs = scaler.transform(imputer.transform(f))
                return model.predict(xs)
            scores = _permutation_importance(raw_predict, latest)
            contributions = np.array([scores[c] for c in TARGET_COLS])
            method = "permutation"

        return self._format(
            contributions, method, base_value,
            feature_names=list(feats.columns), top_k=top_k,
        )

    # ------------------------------------------------------------------
    # Keras LSTM
    # ------------------------------------------------------------------
    def _explain_lstm(self, name, bundle, latest, top_k):
        prep = bundle["prep"]
        keras_model = bundle["model"]
        seq_len = prep["seq_len"]
        imputer, scaler = prep["imputer"], prep["scaler"]

        # build a proper 3-D background: (N, T, F)
        bg_raw = self._background_raw()
        bg_feats = extract_features(bg_raw)
        bg_scaled = scaler.transform(imputer.transform(bg_feats))     # (N, F)
        bg_3d = np.repeat(bg_scaled[:, None, :], seq_len, axis=1)     # (N, T, F)

        # single input as 3-D
        cur_feats = extract_features(pd.DataFrame([latest]))
        cur_scaled = scaler.transform(imputer.transform(cur_feats))   # (1, F)
        cur_3d = np.repeat(cur_scaled[:, None, :], seq_len, axis=1)   # (1, T, F)

        contributions = None
        method = "unavailable"

        # 1) SHAP GradientExplainer — correct for Keras, no flatten
        if SHAP_AVAILABLE:
            try:
                expl = self._cache.get(name)
                if expl is None:
                    expl = _shap_gradient_explainer_lstm(
                        keras_model, bg_3d.astype(np.float32)
                    )
                    self._cache[name] = expl
                sv = expl.shap_values(cur_3d.astype(np.float32))    # list or array
                sv = self._reduce_shap_output(sv, cur_scaled.shape[1])
                contributions = sv
                method = "shap_gradient"
            except Exception as e:
                print(f"[explainer] GradientExplainer failed for {name}: {e}")

        # 2) tf.GradientTape saliency fallback
        if contributions is None:
            try:
                g = _gradient_saliency(keras_model, cur_3d.astype(np.float32))
                if g is not None and g.size == cur_scaled.shape[1]:
                    contributions = g
                    method = "saliency"
            except Exception as e:
                print(f"[explainer] saliency failed for {name}: {e}")

        # 3) Permutation importance on the raw sensor values
        if contributions is None:
            def raw_predict(df: pd.DataFrame):
                f = extract_features(df)
                s = scaler.transform(imputer.transform(f))          # (n, F)
                x3 = np.repeat(s[:, None, :], seq_len, axis=1)
                out = keras_model.predict(x3.astype(np.float32), verbose=0)
                return out
            scores = _permutation_importance(raw_predict, latest)
            contributions = np.array([scores[c] for c in TARGET_COLS])
            method = "permutation"

        return self._format(
            contributions, method, 0.0,
            feature_names=list(cur_feats.columns), top_k=top_k,
        )

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _reduce_shap_output(sv, n_features: int) -> np.ndarray:
        """Normalize SHAP outputs into a 1-D array of length n_features."""
        a = np.asarray(sv, dtype=object) if isinstance(sv, list) else np.asarray(sv)
        if isinstance(sv, list):
            # multi-output: list of arrays, each (n_samples, n_features)
            stacked = np.stack([np.asarray(x) for x in sv], axis=0)
            # average signed over outputs, then take first (only) sample
            v = stacked.mean(axis=0)
            v = v[0] if v.ndim == 2 else v
        else:
            a = a.astype(float)
            if a.ndim == 3:                      # (n_samples, n_features, n_outputs)
                a = a.mean(axis=-1)
            v = a[0] if a.ndim == 2 else a
        v = np.asarray(v).flatten()
        # LSTM contributions may come back at length n_features * something;
        # truncate to the features we actually care about
        if v.size > n_features:
            v = v[:n_features]
        if v.size < n_features:
            v = np.pad(v, (0, n_features - v.size))
        return v

    @staticmethod
    def _format(contributions, method, base_value, feature_names, top_k):
        contributions = np.asarray(contributions, dtype=float).flatten()
        n = min(len(feature_names), contributions.size)
        pairs = list(zip(feature_names[:n], contributions[:n].tolist()))
        pairs.sort(key=lambda kv: abs(kv[1]), reverse=True)
        return {
            "method": method,
            "base_value": base_value,
            "contributions": [
                {"feature": f, "value": v,
                 "direction": "up" if v >= 0 else "down"}
                for f, v in pairs[:top_k]
            ],
        }


# ===========================================================================
# FACADE USED BY app.py
# ===========================================================================
_global_explainer: ModelExplainer | None = None


def get_explainer(predictor) -> ModelExplainer:
    global _global_explainer
    if _global_explainer is None:
        _global_explainer = ModelExplainer(predictor, background_size=20)
    return _global_explainer


def build_narrative(latest: dict, contributions: list[dict], method: str) -> str:
    if not contributions:
        return ("No feature attribution available. "
                "Rule-based analysis shown below.")
    top = contributions[0]
    direction = "increased" if top["value"] > 0 else "decreased"
    feature = top["feature"].replace("_", " ")
    return (f"Top driver: **{feature}** {direction} the prediction "
            f"by {abs(top['value']):.3f} (method: {method}). "
            f"Input: IR={latest['IR']:.0f}, RED={latest['RED']:.0f}, "
            f"HR={latest['HR']:.0f}, SpO2={latest['SpO2']:.1f}.")


def explain_prediction(predictor, latest: dict, results: list[dict]) -> dict:
    """
    ALWAYS returns a non-empty dict so the XAI panel is never blank.
    """
    # pick preferred model
    model_used = None
    for name in ("LSTM_GAN", "LSTM_NoGAN",
                 "RandomForest_GAN", "RandomForest_NoGAN",
                 "ExtraTrees_GAN", "ExtraTrees_NoGAN",
                 "HistGradientBoosting_GAN", "HistGradientBoosting_NoGAN"):
        if any(r["Model"] == name for r in results) or name in predictor.models:
            model_used = name
            break

    rules = rule_engine(latest)

    contributions = []
    method = "unavailable"
    if model_used:
        try:
            out = get_explainer(predictor).explain(model_used, latest)
            if out:
                contributions = out["contributions"]
                method = out["method"]
        except Exception as e:
            print(f"[explainer] failed for {model_used}: {e}")

    narrative = build_narrative(latest, contributions, method)

    return {
        "model_used": model_used or "(none)",
        "narrative": narrative,
        "contributions": contributions,
        "causes": rules["causes"],
        "remedies": rules["remedies"],
        "severity": rules["severity"],
        "shap_available": SHAP_AVAILABLE,
        "method": method,
    }