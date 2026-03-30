#!/usr/bin/env python3
#***************************************************************#
#** This script is part of Thireus' GGUF Tool Suite.          **#
#** group0_enricher.py is a tool that fills the gaps of the   **#
#** group0/kld_results_partial.csv degradation data.          **#
#**                                                           **#
#** ********************************************************* **#
#** --------------- Updated: Mar-30-2026 -------------------- **#
#** ********************************************************* **#
#**                                                           **#
#** Author: Thireus <gguf@thireus.com>                        **#
#**                                                           **#
#** https://gguf.thireus.com/                                 **#
#** Thireus' GGUF Tool Suite - Quantize LLMs Like a Chef       **#
#**                                  ·     ·       ·~°          **#
#**     Λ,,Λ             ₚₚₗ  ·° ᵍᵍᵐˡ   · ɪᴋ_ʟʟᴀᴍᴀ.ᴄᴘᴘ°   ᴮᶠ¹⁶ ·  **#
#**    (:·ω·)       。··°      ·   ɢɢᴜғ   ·°·  ₕᵤ𝓰𝓰ᵢₙ𝓰𝒻ₐ𝒸ₑ   ·°   **#
#**    /    o―ヽニニフ))             · · ɪǫ3_xxs      ~·°        **#
#**    し―-J                                                   **#
#**                                                           **#
#** Copyright © 2026 - Thireus.  ₛₚₑₙ𝒹ᵢₙ𝓰 ₕₐₗ𝒻 ₘᵧ ₛₐₗₐᵣᵧ ₒₙ ₜₒₖₑₙₛ **#
#***************************************************************#
#**PLEASE REFER TO THE README FILE FOR ADDITIONAL INFORMATION!**#
#***************************************************************#

import argparse
import csv
import json
import math
import re
import sys
from bisect import bisect_left
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt


# -----------------------------
# Core helpers
# -----------------------------

MISSING_STRINGS = {"", "404", "404.0", "404%", "404.0%"}


def norm_qtype(q: str) -> str:
    return str(q).strip().upper()


def is_missing_value(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and math.isnan(v):
        return True
    return str(v).strip().lower() in MISSING_STRINGS


def to_float(v: Any) -> Optional[float]:
    if is_missing_value(v):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def format_float(x: float, precision: int = 6) -> str:
    s = f"{x:.{precision}f}"
    return s.rstrip("0").rstrip(".") if "." in s else s


def interp_1d(xs: Sequence[float], ys: Sequence[float], x: float) -> Optional[float]:
    if not xs:
        return None
    if len(xs) == 1:
        return ys[0]
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]

    i = bisect_left(xs, x)
    x0, x1 = xs[i - 1], xs[i]
    y0, y1 = ys[i - 1], ys[i]
    if x1 == x0:
        return (y0 + y1) / 2.0
    t = (x - x0) / (x1 - x0)
    return y0 + t * (y1 - y0)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def parse_csv_numeric_value(raw_value: Any, style: Optional[str]) -> Optional[float]:
    """Parse CSV numeric values according to the declared style.

    style:
      - "percent": values are written like "2.12%"
      - "float": values are written like "0.0212"
      - None: best effort parsing
    """
    if is_missing_value(raw_value):
        return None

    s = str(raw_value).strip()
    if s.startswith("+"):
        s = s[1:].strip()

    if style == "percent":
        if not s.endswith("%"):
            raise ValueError(f"Expected percent value, got: {raw_value!r}")
        s = s[:-1].strip()
        return float(s) / 100.0

    if style == "float":
        if s.endswith("%"):
            raise ValueError(f"Expected absolute float value, got percent string: {raw_value!r}")
        return float(s)

    # Best effort fallback.
    if s.endswith("%"):
        s = s[:-1].strip()
        return float(s) / 100.0
    return float(s)


def infer_degradation_style(path: str) -> Optional[str]:
    """Infer whether the CSV uses percentage strings or absolute floats.

    Returns:
        "percent" if all non-missing values are percent strings,
        "float" if all non-missing values are absolute values,
        None if no usable values exist.

    Raises:
        ValueError if the file mixes percent and float styles.
    """
    fields, rows = read_csv_rows(path)
    qtype_col = next((c for c in fields if c.upper() == "QTYPE"), None)
    if qtype_col is None:
        raise ValueError(f"{path} must contain a QTYPE column")

    value_cols = [c for c in fields if c.upper() != "QTYPE"]
    if not value_cols:
        raise ValueError(f"{path} must contain at least one value column besides QTYPE")
    value_col = value_cols[0]

    saw_percent = False
    saw_float = False

    for row in rows:
        val = row.get(value_col)
        if is_missing_value(val):
            continue

        s = str(val).strip()
        if s.endswith("%"):
            saw_percent = True
            s = s[:-1].strip()
        else:
            saw_float = True

        if s.startswith("+"):
            s = s[1:].strip()

        try:
            float(s)
        except ValueError:
            continue

        if saw_percent and saw_float:
            raise ValueError(
                f"{path} mixes percentage and absolute values; please keep a single style in the file."
            )

    if saw_percent:
        return "percent"
    if saw_float:
        return "float"
    return None


def lookup_qtype_with_r_suffix(values: Mapping[str, Any], qtype: str) -> Optional[float]:
    """Lookup a qtype value directly, then try the base or _r4/_r8 sibling.

    The degradation data for iq1_s != iq1_s_r4, this is the only exception.
    """
    q = norm_qtype(qtype)
    direct_val = to_float(values.get(q))
    if direct_val is not None:
        return direct_val

    if q in {"IQ1_S", "IQ1_S_R4"}:
        return None

    base_qtype = re.sub(r"_R[48]$", "", q, flags=re.IGNORECASE)
    if base_qtype != q:
        base_val = to_float(values.get(base_qtype))
        if base_val is not None:
            return base_val

    for suffix in ("_R4", "_R8"):
        candidate = base_qtype + suffix
        candidate_val = to_float(values.get(candidate))
        if candidate_val is not None:
            return candidate_val

    return None


def format_output_value(value: float, style: str, precision: int) -> str:
    """Format output according to the target CSV style."""
    if style == "percent":
        return f"{format_float(value * 100.0, precision)}%"
    return format_float(value, precision)


def weighted_median(values: Sequence[float], weights: Sequence[float]) -> Optional[float]:
    """Compute a weighted median for positive weights."""
    pairs = [
        (float(v), float(w))
        for v, w in zip(values, weights)
        if w > 0 and math.isfinite(v) and math.isfinite(w)
    ]
    if not pairs:
        return None

    pairs.sort(key=lambda p: p[0])
    total = sum(w for _, w in pairs)
    if total <= 0:
        return None

    cutoff = total / 2.0
    accum = 0.0
    for value, weight in pairs:
        accum += weight
        if accum >= cutoff:
            return value

    return pairs[-1][0]


# -----------------------------
# BPW table
# -----------------------------

# --- BPW lookup table for GGUF quant dtypes ---
BPW_TABLE = {
    'F32': 32,
    'F16': 16,
    'BF16': 16,
    'Q8_0_R8': 8.5,
    'Q8_0': 8.5,
    'Q8_K_R8': 8.0625,
    'Q8_KV': 8,
    'F8': 8,
    'IQ6_K': 6.625,
    'Q6_K_R4': 6.5625,
    'Q6_K': 6.5625,
    'Q6_0_R4': 6.5,
    'Q6_0': 6.5,
    'Q5_1': 6,
    'Q5_K_R4': 5.5,
    'Q5_K': 5.5,
    'Q5_0_R4': 5.5,
    'Q5_0': 5.5,
    'IQ5_K_R4': 5.5,
    'IQ5_K': 5.5,
    'IQ5_KS_R4': 5.25,
    'IQ5_KS': 5.25,
    'Q4_1': 5,
    'Q4_K_R4': 4.5,
    'Q4_K': 4.5,
    'Q4_0_R8': 4.5,
    'Q4_0': 4.5,
    'IQ4_NL_R4': 4.5,
    'IQ4_NL': 4.5,
    'IQ4_K_R4': 4.5,
    'IQ4_K': 4.5,
    'IQ4_XS_R8': 4.25,
    'IQ4_XS': 4.25,
    'IQ4_KS_R4': 4.25,
    'IQ4_KS': 4.25,
    'IQ4_KT': 4,
    'IQ4_KSS': 4,
    'IQ3_KL': 4,
    'IQ3_M': 3.66,
    'Q3_K_R4': 3.4375,
    'Q3_K': 3.4375,
    'IQ3_S_R4': 3.4375,
    'IQ3_S': 3.4375,
    'IQ3_K_R4': 3.4375,
    'IQ3_K': 3.4375,
    'IQ3_XS': 3.3,
    'IQ3_KS': 3.1875,
    'IQ3_KT': 3.125,
    'IQ3_XXS_R4': 3.0625,
    'IQ3_XXS': 3.0625,
    'IQ2_M_R4': 2.7,
    'IQ2_M': 2.7,
    'IQ2_KL': 2.6875,
    'Q2_K_R4': 2.625,
    'Q2_K': 2.625,
    'IQ2_S': 2.5625,
    'IQ2_K_R4': 2.375,
    'IQ2_K': 2.375,
    'IQ2_XS_R4': 2.3125,
    'IQ2_XS': 2.3125,
    'IQ2_KS': 2.1875,
    'IQ2_KT': 2.125,
    'IQ2_XXS_R4': 2.0625,
    'IQ2_XXS': 2.0625,
    'IQ2_BN_R4': 2,
    'IQ2_BN': 2,
    'IQ1_M_R4': 1.75,
    'IQ1_M': 1.75,
    'IQ1_KT': 1.75,
    'IQ1_BN': 1.625,
    'IQ1_S': 1.5625,
    'IQ1_S_R4': 1.5
}
BPW_TABLE = {k.upper(): float(v) for k, v in BPW_TABLE.items()}


# -----------------------------
# Default values and equations
# -----------------------------

# Qwen3-4B-Thinking-2507 group0.csv values
DEFAULT_REFERENCE_VALUES = {'bf16': 0.0, 'iq1_bn': 14.758228, 'iq1_kt': 2.692801, 'iq1_m': 4.684445, 'iq1_m_r4': 4.617296, 'iq1_s': 4.562480, 'iq1_s_r4': 5.124850, 'iq2_bn': 15.467749, 'iq2_bn_r4': 15.445743, 'iq2_k': 0.883945, 'iq2_k_r4': 0.883945, 'iq2_kl': 0.584754, 'iq2_ks': 1.347207, 'iq2_kt': 1.214565, 'iq2_s': 0.465971, 'iq2_xs': 0.633596, 'iq2_xs_r4': 0.636844, 'iq2_xxs': 1.202639, 'iq2_xxs_r4': 1.208328, 'iq3_k': 0.164337, 'iq3_k_r4': 0.164337, 'iq3_ks': 0.211158, 'iq3_kt': 0.214378, 'iq3_s': 0.210123, 'iq3_s_r4': 0.213001, 'iq3_xxs': 0.348842, 'iq3_xxs_r4': 0.351102, 'iq4_k': 0.034494, 'iq4_k_r4': 0.034494, 'iq4_ks': 0.047722, 'iq4_ks_r4': 0.047722, 'iq4_kss': 0.073993, 'iq4_kt': 0.071823, 'iq4_nl': 0.052065, 'iq4_nl_r4': 0.051893, 'iq4_xs': 0.052575, 'iq4_xs_r8': 0.055797, 'iq5_k': 0.009814, 'iq5_k_r4': 0.009814, 'iq5_ks': 0.012268, 'iq5_ks_r4': 0.012268, 'iq6_k': 0.003411, 'q2_K': 0.895361, 'q2_k_r4': 0.896636, 'q3_K': 0.226457, 'q3_k_r4': 0.228156, 'q4_0': 0.070737, 'q4_0_r8': 0.070601, 'q4_1': 0.050200, 'q4_K': 0.046677, 'q4_k_r4': 0.046609, 'q5_0': 0.018810, 'q5_0_r4': 0.018876, 'q5_1': 0.014465, 'q5_K': 0.015590, 'q5_k_r4': 0.015766, 'q6_0': 0.005317, 'q6_0_r4': 0.005244, 'q6_K': 0.004040, 'q6_k_r4': 0.006687, 'q8_0': 0.001449, 'q8_0_r8': 0.001515, 'q8_k_r8': 0.004102, 'q8_KV': 0.038383}

# Default quant degradation EQUATION (used when user does not supply --reference-csv).
# This default equation was obtained by running:
# cd models/Qwen3-4B-Thinking-2507/group0 && ../../../model_tensor_bpw_metric.py --results-csv kld_results.csv --c-free --exclude-qtypes '.*_bn.*$' --transforms "identity" --ignore-outliers 50 --allow-impure-map --plot --p-grid-max 15 --p-grid-steps 100 --d-from-lowest 1 --penalize-above 15 --resemblance-metric r2
# against the Qwen3-4B-Thinking-2507's kld_results.csv found in the models/Qwen3-4B-Thinking-2507/group0 directory.
DEFAULT_REFERENCE_MEAN = "y = 0 + 3.09340197611e+17 * ( x + 6.95719165416 )^(-18.0541201489)"


def read_csv_rows(path: str) -> Tuple[List[str], List[Dict[str, str]]]:
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"{path} has no header row")
        return list(reader.fieldnames), list(reader)


def parse_equation_to_callable(eq_str: str) -> Optional[Callable[[float], float]]:
    if not eq_str or not isinstance(eq_str, str):
        return None

    s = eq_str.strip()
    if s.lower().startswith("y"):
        parts = s.split("=", 1)
        if len(parts) == 2:
            s = parts[1].strip()
    s = s.replace("^", "**")

    try:
        code = compile(s, "<quant-equation>", "eval")
    except Exception as e:
        print(f"[Error] Failed to parse quant degradation equation: {e}", file=sys.stderr)
        return None

    safe_globals = {"__builtins__": None}
    for name in dir(math):
        if not name.startswith("_"):
            safe_globals[name] = getattr(math, name)

    def f(x_val: float):
        try:
            locals_map = {"x": float(x_val)}
            return float(eval(code, safe_globals, locals_map))
        except Exception as e:
            raise RuntimeError(f"Error evaluating quant degradation equation at x={x_val}: {e}")

    return f


def build_csv_qtype_data(
    path: str,
    style: Optional[str] = None,
    apply_sibling_fallback: bool = False,
) -> Tuple[
    List[str],
    List[Dict[str, str]],
    str,
    str,
    Dict[str, float],
    Dict[str, float],
    List[str],
    Dict[str, str],
    Optional[str],
]:
    """Load qtype/value data from a CSV and preserve original qtype casing/order."""
    fields, rows = read_csv_rows(path)
    qtype_col = next((c for c in fields if c.upper() == "QTYPE"), None)
    if qtype_col is None:
        raise ValueError(f"{path} must contain a QTYPE column")

    value_cols = [c for c in fields if c.upper() != "QTYPE"]
    if not value_cols:
        raise ValueError(f"{path} must contain at least one value column besides QTYPE")
    value_col = value_cols[0]

    if style is None:
        style = infer_degradation_style(path)

    raw_values: Dict[str, float] = {}
    norm_values: Dict[str, float] = {}
    qtype_order: List[str] = []
    qtype_case_by_norm: Dict[str, str] = {}

    for row in rows:
        raw_qtype = str(row.get(qtype_col, "")).strip()
        if not raw_qtype:
            continue
        qnorm = norm_qtype(raw_qtype)

        parsed = parse_csv_numeric_value(row.get(value_col), style)
        if parsed is None:
            continue

        if qnorm not in qtype_case_by_norm:
            qtype_order.append(raw_qtype)
        qtype_case_by_norm[qnorm] = raw_qtype
        raw_values[raw_qtype] = parsed
        norm_values[qnorm] = parsed

    # Apply sibling fallback as early as possible for the target CSV.
    # The degradation data for iq1_s != iq1_s_r4, this is the only exception.
    if apply_sibling_fallback:
        additions_raw: Dict[str, float] = {}
        additions_norm: Dict[str, float] = {}

        for qnorm in list(qtype_case_by_norm.keys()):
            if qnorm in {"IQ1_S", "IQ1_S_R4"}:
                continue

            if qnorm in norm_values:
                continue

            base_qtype = re.sub(r"_R[48]$", "", qnorm, flags=re.IGNORECASE)
            candidate_values: List[float] = []

            if base_qtype != qnorm and base_qtype in norm_values:
                candidate_values.append(norm_values[base_qtype])

            for suffix in ("_R4", "_R8"):
                candidate = base_qtype + suffix
                if candidate in norm_values:
                    candidate_values.append(norm_values[candidate])

            if candidate_values:
                val = candidate_values[0]
                additions_norm[qnorm] = val
                additions_raw[qtype_case_by_norm[qnorm]] = val

        for qnorm, val in additions_norm.items():
            norm_values[qnorm] = val
            raw_values[qtype_case_by_norm[qnorm]] = val

    return (
        fields,
        rows,
        qtype_col,
        value_col,
        raw_values,
        norm_values,
        qtype_order,
        qtype_case_by_norm,
        style,
    )


def load_reference_data(args: argparse.Namespace) -> Tuple[Dict[str, float], Dict[str, float], List[str], Dict[str, str], Callable[[float], float], str]:
    """Return raw and normalized reference data plus original order/casing and mean function."""
    if args.reference_csv:
        if not args.reference_mean_equation:
            raise SystemExit("[Error] --reference-mean-equation is required when using --reference-csv")

        (
            _fields,
            _rows,
            _qtype_col,
            _value_col,
            ref_raw_values,
            ref_norm_values,
            ref_order,
            ref_case_by_norm,
            ref_style,
        ) = build_csv_qtype_data(args.reference_csv, apply_sibling_fallback=False)

        ref_mean_fn = parse_equation_to_callable(args.reference_mean_equation)
        if ref_mean_fn is None:
            raise SystemExit("[Error] Invalid --reference-mean-equation provided; aborting.")
        return ref_raw_values, ref_norm_values, ref_order, ref_case_by_norm, ref_mean_fn, (ref_style or "float")

    ref_raw_values = dict(DEFAULT_REFERENCE_VALUES)
    ref_norm_values = {norm_qtype(k): v for k, v in ref_raw_values.items()}
    ref_order = list(ref_raw_values.keys())
    ref_case_by_norm = {norm_qtype(k): k for k in ref_raw_values.keys()}
    ref_mean_fn = parse_equation_to_callable(DEFAULT_REFERENCE_MEAN)
    if ref_mean_fn is None:
        raise SystemExit("[Error] Could not parse built-in reference mean equation; aborting.")
    return ref_raw_values, ref_norm_values, ref_order, ref_case_by_norm, ref_mean_fn, "float"


# -----------------------------
# Filling logic
# -----------------------------


def transpose_reference_value_to_target_curve(
    ref_val: float,
    ref_mean: float,
    target_mean: float,
) -> float:
    """Normalize a reference value into the target curve's amplitude space."""
    eps = 1e-12
    if abs(ref_mean) < eps:
        return float(target_mean + (ref_val - ref_mean))
    amplitude_factor = target_mean / ref_mean
    return float(target_mean + (ref_val - ref_mean) * amplitude_factor)


CORRECTION_FACTOR_MIN = 0.5
CORRECTION_FACTOR_MAX = 2.0
CORRECTION_EDGE_BLEND_START = 0.35
CORRECTION_RESIDUAL_FLOOR_RATIO = 0.05


def compute_local_correction_factor(
    x: float,
    shared_samples: List[Dict[str, float]],
    global_factor: float,
) -> float:
    """Estimate a local correction factor from nearby known qtypes.

    Known qtypes closer to the mean receive more weight, and qtypes farther away in
    BPW from the assessed x receive less weight.
    """
    if not shared_samples:
        return global_factor

    samples = sorted(shared_samples, key=lambda s: abs(s["x"] - x))
    neighbors = samples[: min(9, len(samples))]

    if not neighbors:
        return global_factor

    eps = 1e-12
    span = max(max(abs(s["x"] - x) for s in neighbors), 0.5)

    weights: List[float] = []
    factors: List[float] = []

    for s in neighbors:
        d = abs(s["x"] - x)
        bpw_weight = 1.0 / ((d / span) + 1e-6) ** 2
        weight = bpw_weight * s["mean_weight"]
        weights.append(weight)
        factors.append(s["factor"])

    local_factor = weighted_median(factors, weights)
    if local_factor is None:
        local_factor = global_factor

    local_factor = clamp(local_factor, CORRECTION_FACTOR_MIN, CORRECTION_FACTOR_MAX)

    domain_min = min(s["x"] for s in shared_samples)
    domain_max = max(s["x"] for s in shared_samples)
    domain_span = max(domain_max - domain_min, 1e-12)

    distance_to_edge = min(x - domain_min, domain_max - x)
    if distance_to_edge <= 0:
        edge_confidence = 0.0
    else:
        edge_confidence = clamp(distance_to_edge / max(domain_span * CORRECTION_EDGE_BLEND_START, 1e-12), 0.0, 1.0)

    # Near the edges, reduce the correction so the transposed value remains stable.
    blended = 1.0 + edge_confidence * (local_factor - 1.0)
    blended = 0.75 * blended + 0.25 * global_factor
    return clamp(blended, CORRECTION_FACTOR_MIN, CORRECTION_FACTOR_MAX)


def fill_missing_degradation_values(
    reference_values: Mapping[str, Any],
    target_values: Mapping[str, Any],
    reference_mean_fn: Callable[[float], float],
    target_mean_fn: Callable[[float], float],
    bpw_table: Mapping[str, float],
    keep_existing_target_values: bool = True,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """Fill target gaps by transposing the reference values onto the target mean.

    What this does:
      1) For each reference qtype, normalize the reference point into target-curve space
         by matching its amplitude against the target mean curve.

      2) Learn a local correction factor from nearby known qtypes, favoring points that
         are close to their mean and close in BPW to the assessed qtype.

      3) For each missing target qtype, apply the local correction factor to its
         transposed reference value.

    The sibling fallback for base/_r4/_r8 remains intact and is applied before transfer.
    """
    ref = {norm_qtype(k): v for k, v in reference_values.items()}
    tgt = {norm_qtype(k): v for k, v in target_values.items()}

    # First normalize every reference value into target-space so the amplitude is aligned with the target curve.
    transposed_ref_values: Dict[str, float] = {}
    shared_samples: List[Dict[str, float]] = []

    for qtype, ref_raw_val in ref.items():
        x = bpw_table.get(qtype)
        if x is None:
            continue

        y_ref = to_float(ref_raw_val)
        if y_ref is None:
            continue

        ref_mean = reference_mean_fn(x)
        tgt_mean = target_mean_fn(x)

        transposed_ref = transpose_reference_value_to_target_curve(y_ref, ref_mean, tgt_mean)
        transposed_ref_values[qtype] = float(transposed_ref)

    eps = 1e-12
    for qtype in sorted(set(ref) & set(tgt)):
        x = bpw_table.get(qtype)
        if x is None:
            continue

        y_ref = to_float(ref.get(qtype))
        y_tgt = to_float(tgt.get(qtype))
        transposed_ref = transposed_ref_values.get(qtype)

        if y_ref is None or y_tgt is None or transposed_ref is None:
            continue

        tgt_mean = target_mean_fn(x)
        target_disp = y_tgt - tgt_mean
        transposed_disp = transposed_ref - tgt_mean

        if abs(transposed_disp) < eps:
            continue

        ref_mean = reference_mean_fn(x)
        ref_disp = y_ref - ref_mean

        # Regularize the ratio so tiny residuals near the mean do not explode.
        residual_floor = max(abs(ref_mean), abs(tgt_mean), 1e-12) * CORRECTION_RESIDUAL_FLOOR_RATIO
        factor = (abs(target_disp) + residual_floor) / (abs(transposed_disp) + residual_floor)

        ref_mean_distance = abs(ref_disp) / max(abs(ref_mean), eps)
        tgt_mean_distance = abs(target_disp) / max(abs(tgt_mean), eps)
        transposed_mean_distance = abs(transposed_disp) / max(abs(tgt_mean), eps)

        # Favor samples that sit closer to the mean in both spaces.
        mean_distance = (ref_mean_distance + tgt_mean_distance + transposed_mean_distance) / 3.0
        mean_weight = 1.0 / ((1.0 + mean_distance) ** 2)

        # If the sample falls on the opposite side of the mean after transposition,
        # it should influence the correction much less.
        if (target_disp > 0) != (transposed_disp > 0):
            mean_weight *= 0.1

        shared_samples.append(
            {
                "x": float(x),
                "factor": float(factor),
                "mean_weight": float(mean_weight),
            }
        )

    if shared_samples:
        global_factor = weighted_median(
            [s["factor"] for s in shared_samples],
            [s["mean_weight"] for s in shared_samples],
        )
        if global_factor is None:
            global_factor = 1.0
    else:
        global_factor = 1.0

    factor_by_bpw = defaultdict(list)
    for s in shared_samples:
        factor_by_bpw[s["x"]].append(s["factor"])

    correction_anchors = {str(x): median(factor_by_bpw[x]) for x in sorted(factor_by_bpw)}

    output_keys = list(tgt.keys())
    for qtype in ref.keys():
        if qtype not in tgt:
            output_keys.append(qtype)

    enriched: Dict[str, float] = {}
    filled: Dict[str, float] = {}

    for qtype in output_keys:
        x = bpw_table.get(qtype)
        tgt_val = to_float(tgt.get(qtype))

        if keep_existing_target_values and tgt_val is not None:
            enriched[qtype] = tgt_val
            continue

        # Sibling fallback: base <-> _r4/_r8, except iq1_s / iq1_s_r4.
        sibling_val = lookup_qtype_with_r_suffix(tgt, qtype)
        if sibling_val is not None:
            enriched[qtype] = sibling_val
            filled[qtype] = sibling_val
            continue

        # Use the transposed reference value, then locally correct it using nearby known qtypes.
        transposed_ref_val = transposed_ref_values.get(qtype)
        if x is not None and transposed_ref_val is not None:
            local_factor = compute_local_correction_factor(float(x), shared_samples, global_factor)
            tgt_mean = target_mean_fn(x)
            corrected = tgt_mean + ((transposed_ref_val - tgt_mean) * local_factor)
            enriched[qtype] = float(corrected)
            filled[qtype] = float(corrected)
            continue

        if x is not None:
            pred = float(target_mean_fn(x))
            enriched[qtype] = pred
            filled[qtype] = pred
            continue

        enriched[qtype] = 0.0
        filled[qtype] = 0.0

    return enriched, filled, correction_anchors


# -----------------------------
# Output helpers
# -----------------------------


def build_output_rows_preserving_reference_order(
    target_fields: List[str],
    qtype_col: str,
    value_col: str,
    target_rows: List[Dict[str, str]],
    reference_order: List[str],
    enriched: Mapping[str, float],
    target_style: str,
    precision: int,
) -> List[Dict[str, Any]]:
    """Write rows in reference order/casing when the qtype exists in the reference.

    Any qtype present in the reference uses the exact reference casing.
    Target-only qtypes are appended at the end in their original target order/casing.
    """
    target_rows_by_norm: Dict[str, Dict[str, str]] = {}
    target_order: List[str] = []

    for row in target_rows:
        qnorm = norm_qtype(row.get(qtype_col, ""))
        if qnorm not in target_rows_by_norm:
            target_order.append(qnorm)
        target_rows_by_norm[qnorm] = dict(row)

    output_rows: List[Dict[str, Any]] = []
    emitted: set[str] = set()

    # Emit every qtype that exists in the reference CSV in the exact order provided there.
    # This guarantees the enriched CSV follows the reference qtype ordering, regardless
    # of how the target CSV was ordered.
    for ref_qtype in reference_order:
        qnorm = norm_qtype(ref_qtype)
        emitted.add(qnorm)

        if qnorm in target_rows_by_norm:
            row = dict(target_rows_by_norm[qnorm])
        else:
            row = {c: "" for c in target_fields}

        row[qtype_col] = ref_qtype

        if qnorm in enriched:
            row[value_col] = format_output_value(enriched[qnorm], target_style, precision)

        output_rows.append(row)

    for qnorm in target_order:
        if qnorm in emitted:
            continue

        row = dict(target_rows_by_norm[qnorm])
        if qnorm in enriched:
            row[value_col] = format_output_value(enriched[qnorm], target_style, precision)
        output_rows.append(row)

    return output_rows


# -----------------------------
# Metrics and plot
# -----------------------------


def compute_error_metrics(
    predicted: Mapping[str, float],
    truth: Mapping[str, Any],
    only_keys: Optional[Iterable[str]] = None,
) -> Dict[str, float]:
    keys = list(only_keys) if only_keys is not None else sorted(set(predicted) & set(truth))

    abs_errors: List[float] = []
    signed_errors: List[float] = []
    squared_errors: List[float] = []
    relative_errors: List[float] = []

    used = 0
    for k in keys:
        p = to_float(predicted.get(k))
        t = to_float(truth.get(k))
        if p is None or t is None:
            continue
        err = p - t
        abs_err = abs(err)
        abs_errors.append(abs_err)
        signed_errors.append(err)
        squared_errors.append(err * err)
        if abs(t) > 1e-12:
            relative_errors.append(abs_err / abs(t))
        used += 1

    if not used:
        return {
            "count": 0.0,
            "mae": float("nan"),
            "rmse": float("nan"),
            "max_abs_error": float("nan"),
            "mean_signed_error": float("nan"),
            "mape_pct": float("nan"),
        }

    mae = sum(abs_errors) / used
    rmse = math.sqrt(sum(squared_errors) / used)
    max_abs_error = max(abs_errors)
    mean_signed_error = sum(signed_errors) / used
    mape = (sum(relative_errors) / len(relative_errors) * 100.0) if relative_errors else float("nan")

    return {
        "count": float(used),
        "mae": mae,
        "rmse": rmse,
        "max_abs_error": max_abs_error,
        "mean_signed_error": mean_signed_error,
        "mape_pct": mape,
    }


def make_comparison_plot(
    predicted: Mapping[str, float],
    filled_keys: Mapping[str, float],
    target_values: Mapping[str, Any],
    reference_values: Mapping[str, Any],
    reference_mean_fn: Callable[[float], float],
    target_mean_fn: Callable[[float], float],
    bpw_table: Mapping[str, float],
    compare_truth: Optional[Mapping[str, Any]] = None,
) -> None:
    plt.figure(figsize=(12, 7))

    ref_x: List[float] = []
    ref_y: List[float] = []
    ref_labels: List[str] = []

    tgt_x: List[float] = []
    tgt_y: List[float] = []
    tgt_labels: List[str] = []

    guessed_x: List[float] = []
    guessed_y: List[float] = []
    guessed_labels: List[str] = []

    gt_x: List[float] = []
    gt_y: List[float] = []
    gt_labels: List[str] = []

    for qtype, val in reference_values.items():
        x = bpw_table.get(norm_qtype(qtype))
        y = to_float(val)
        if x is not None and y is not None:
            ref_x.append(x)
            ref_y.append(y)
            ref_labels.append(norm_qtype(qtype))

    for qtype, val in target_values.items():
        x = bpw_table.get(norm_qtype(qtype))
        y = to_float(val)
        if x is not None and y is not None:
            tgt_x.append(x)
            tgt_y.append(y)
            tgt_labels.append(norm_qtype(qtype))

    guessed_qtypes = set(filled_keys.keys())
    for qtype, pred_val in predicted.items():
        if qtype in guessed_qtypes:
            x = bpw_table.get(norm_qtype(qtype))
            y = to_float(pred_val)
            if x is not None and y is not None:
                guessed_x.append(x)
                guessed_y.append(y)
                guessed_labels.append(norm_qtype(qtype))

    if compare_truth is not None:
        for qtype in guessed_qtypes:
            x = bpw_table.get(norm_qtype(qtype))
            y = to_float(compare_truth.get(qtype))
            if x is not None and y is not None:
                gt_x.append(x)
                gt_y.append(y)
                gt_labels.append(norm_qtype(qtype))

    if ref_x:
        plt.scatter(ref_x, ref_y, marker="+", s=90, color="black", label="Reference values", alpha=0.9)
    if tgt_x:
        plt.scatter(tgt_x, tgt_y, marker="+", s=90, color="blue", label="Target values", alpha=0.8)
    if guessed_x:
        plt.scatter(guessed_x, guessed_y, marker="+", s=110, color="red", label="Filled target values", alpha=0.95)
    if gt_x:
        plt.scatter(gt_x, gt_y, marker="+", s=110, color="green", label="Ground truth for filled values", alpha=0.95)

    all_curve_x = [x for x in BPW_TABLE.values() if isinstance(x, (int, float))]
    if all_curve_x:
        x_min = min(all_curve_x)
        x_max = max(all_curve_x)
        xs = [x_min + (x_max - x_min) * i / 300.0 for i in range(301)]
        ref_curve_y = [reference_mean_fn(x) for x in xs]
        tgt_curve_y = [target_mean_fn(x) for x in xs]
        plt.plot(xs, ref_curve_y, color="black", linestyle="--", linewidth=1.5, label="Reference mean curve")
        plt.plot(xs, tgt_curve_y, color="blue", linestyle="--", linewidth=1.5, label="Target mean curve")

    all_x = ref_x + tgt_x + guessed_x + gt_x
    all_labels = ref_labels + tgt_labels + guessed_labels + gt_labels
    if all_x:
        uniq: Dict[float, List[str]] = {}
        for x, lab in zip(all_x, all_labels):
            bucket = uniq.setdefault(x, [])
            if lab not in bucket:
                bucket.append(lab)
        xticks = sorted(uniq.keys())
        xticklabels = ["\n".join(uniq[x][:4]) for x in xticks]
        plt.xticks(xticks, xticklabels, rotation=45, ha="right")

    plt.xlabel("BPW / qtype")
    plt.ylabel("Value")
    plt.title("Quant degradation comparison")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.show()


# -----------------------------
# CLI
# -----------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fill missing quant degradation values using a reference CSV and mean equations."
    )
    parser.add_argument(
        "--target-csv",
        required=True,
        type=str,
        help="Path to the target CSV to enrich (must contain QTYPE column and one value column).",
    )
    parser.add_argument(
        "--target-mean-equation",
        required=True,
        type=str,
        help=(
            'Equation for the target mean curve, e.g. "y = 0 + 5.735e22 * ( x + 10.2181080113 )^(-20.8421052632)".'
        ),
    )
    parser.add_argument(
        "--output-csv",
        required=True,
        type=str,
        help="Path where the enriched CSV will be written.",
    )
    parser.add_argument(
        "--reference-csv",
        type=str,
        default=None,
        help="Optional reference CSV containing known quant degradation values.",
    )
    parser.add_argument(
        "--reference-mean-equation",
        type=str,
        default=None,
        help="Optional mean equation for the reference dataset. Required if --reference-csv is provided.",
    )
    parser.add_argument(
        "--compare-with-csv",
        type=str,
        default=None,
        help="Optional fully-known CSV to compare predictions against and report error metrics.",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=6,
        help="Decimal precision for output values.",
    )
    parser.add_argument(
        "--report-filled-only",
        action="store_true",
        help="Print only the entries that were filled or changed.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    (
        target_fields,
        target_rows,
        target_qtype_col,
        target_value_col,
        _target_raw_values,
        target_values,
        target_order,
        _target_case_by_norm,
        target_style,
    ) = build_csv_qtype_data(args.target_csv, apply_sibling_fallback=True)

    target_mean_fn = parse_equation_to_callable(args.target_mean_equation)
    if target_mean_fn is None:
        raise SystemExit("[Error] Invalid --target-mean-equation provided; aborting.")

    (
        reference_raw_values,
        reference_values,
        reference_order,
        _reference_case_by_norm,
        reference_mean_fn,
        reference_style,
    ) = load_reference_data(args)

    # Style check: percent must match percent, float must match float.
    if target_style is None:
        target_style = reference_style
    if reference_style is None:
        reference_style = target_style

    if reference_style != target_style:
        raise SystemExit(
            f"[Error] Value format mismatch: reference data is '{reference_style}' "
            f"but target data is '{target_style}'. Both CSVs must use the same format "
            f"(percent vs absolute float)."
        )

    enriched, filled, correction_anchors = fill_missing_degradation_values(
        reference_values=reference_values,
        target_values=target_values,
        reference_mean_fn=reference_mean_fn,
        target_mean_fn=target_mean_fn,
        bpw_table=BPW_TABLE,
        keep_existing_target_values=True,
    )

    output_rows = build_output_rows_preserving_reference_order(
        target_fields=target_fields,
        qtype_col=target_qtype_col,
        value_col=target_value_col,
        target_rows=target_rows,
        reference_order=reference_order,
        enriched=enriched,
        target_style=target_style or "float",
        precision=args.precision,
    )

    with Path(args.output_csv).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=target_fields)
        writer.writeheader()
        writer.writerows(output_rows)

    compare_truth = None
    if args.compare_with_csv:
        (
            _cmp_fields,
            _cmp_rows,
            cmp_qtype_col,
            cmp_value_col,
            _cmp_raw_values,
            compare_truth,
            _cmp_order,
            _cmp_case_by_norm,
            _cmp_style,
        ) = build_csv_qtype_data(args.compare_with_csv, apply_sibling_fallback=False)

        metrics = compute_error_metrics(enriched, compare_truth)
        make_comparison_plot(
            enriched,
            filled,
            target_values,
            reference_raw_values,
            reference_mean_fn,
            target_mean_fn,
            BPW_TABLE,
            compare_truth=compare_truth,
        )

        print("[Compare] Error metrics vs known-all-values CSV:")
        print(f"[Compare] count={int(metrics['count'])}")
        print(f"[Compare] MAE={metrics['mae']:.10f}")
        print(f"[Compare] RMSE={metrics['rmse']:.10f}")
        print(f"[Compare] MaxAbsError={metrics['max_abs_error']:.10f}")
        print(f"[Compare] MeanSignedError={metrics['mean_signed_error']:.10f}")
        if not math.isnan(metrics['mape_pct']):
            print(f"[Compare] MAPE%={metrics['mape_pct']:.6f}")

    print(f"[Info] Wrote enriched CSV to: {args.output_csv}")
    print(f"[Info] Filled {len(filled)} values")
    if correction_anchors:
        print(f"[Info] Learned local correction anchors: {json.dumps(correction_anchors, sort_keys=True)}")

    if args.report_filled_only:
        print("[Info] Filled entries:")
        for k in sorted(filled):
            print(f"{k},{format_float(filled[k], args.precision)}")


if __name__ == "__main__":
    main()
