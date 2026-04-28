#!/usr/bin/env python3
#***************************************************************#
#** This script is part of Thireus' GGUF Tool Suite.          **#
#** group0_enricher.py is a tool that fills the gaps of the   **#
#** group0/kld_results_partial.csv degradation data.          **#
#**                                                           **#
#** ********************************************************* **#
#** --------------- Updated: Apr-28-2026 -------------------- **#
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


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def parse_csv_numeric_value(raw_value: Any, style: Optional[str]) -> Optional[float]:
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

    if s.endswith("%"):
        s = s[:-1].strip()
        return float(s) / 100.0
    return float(s)


def infer_degradation_style(path: str) -> Optional[str]:
    fields, rows = read_csv_rows(path)
    qtype_col = next((c for c in fields if c.upper() == "QTYPE"), None)
    if qtype_col is None:
        raise ValueError(f"{path} must contain a QTYPE column")

    value_cols = [c for c in fields if c.upper() != "QTYPE"]
    if not value_cols:
        raise ValueError(f"{path} must contain at least one value column beside QTYPE")
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
    if style == "percent":
        return f"{format_float(value * 100.0, precision)}%"
    return format_float(value, precision)


def weighted_median(values: Sequence[float], weights: Sequence[float]) -> Optional[float]:
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


def should_unify_r_variants(qtype_a: str, qtype_b: str) -> bool:
    """Return False if the pair is the known exception (iq1_s / iq1_s_r4)."""
    a, b = norm_qtype(qtype_a), norm_qtype(qtype_b)
    if {a, b} == {"IQ1_S", "IQ1_S_R4"}:
        return False
    return True


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


# ---------------------------------------
# Default reference values and equations
# ---------------------------------------

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
            raise RuntimeError(f"Error evaluating equation at x={x_val}: {e}")

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

    if apply_sibling_fallback:
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
                raw_values[qtype_case_by_norm[qnorm]] = val

        for qnorm, val in additions_norm.items():
            norm_values[qnorm] = val

    # DEBUG: print loaded target data (first few)
    if apply_sibling_fallback:
        print("\n[DEBUG] Loaded target qtypes (first 20):")
        for q in sorted(norm_values)[:20]:
            print(f"  {q}: {norm_values[q]}")

    return (
        fields, rows, qtype_col, value_col,
        raw_values, norm_values, qtype_order, qtype_case_by_norm, style,
    )


def load_reference_data(args: argparse.Namespace) -> Tuple[
    Dict[str, float], Dict[str, float], List[str], Dict[str, str],
    Callable[[float], float], str
]:
    if args.reference_csv:
        if not args.reference_mean_equation:
            raise SystemExit("[Error] --reference-mean-equation is required when using --reference-csv")
        (
            _fields, _rows, _qtype_col, _value_col,
            ref_raw_values, ref_norm_values, ref_order, ref_case_by_norm, ref_style,
        ) = build_csv_qtype_data(args.reference_csv, apply_sibling_fallback=False)

        ref_mean_fn = parse_equation_to_callable(args.reference_mean_equation)
        if ref_mean_fn is None:
            raise SystemExit("[Error] Invalid --reference-mean-equation provided; aborting.")

        # DEBUG: print reference key values
        print("\n[DEBUG] Loaded reference qtypes (iq1_s, iq1_s_r4):")
        for key in ['IQ1_S', 'IQ1_S_R4']:
            val = ref_norm_values.get(key, "MISSING")
            print(f"  {key}: {val}")
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
# New transformation logic with global scaling & sibling unification
# -----------------------------

def get_highest_bpw_qtypes(
    norm_values: Mapping[str, Any],
    bpw_table: Mapping[str, float],
) -> Tuple[float, set]:
    """Return (max_bpw, set of qtypes that achieve that BPW) among valid entries."""
    max_bpw = 0.0
    for q, v in norm_values.items():
        x = bpw_table.get(q)
        if x is not None and not is_missing_value(v):
            if x > max_bpw:
                max_bpw = x
    qtypes = set()
    for q, v in norm_values.items():
        x = bpw_table.get(q)
        if x is not None and abs(x - max_bpw) < 1e-9 and not is_missing_value(v):
            qtypes.add(norm_qtype(q))
    return max_bpw, qtypes


def extract_bpw_range(
    ref_norm: Mapping[str, Any],
    tgt_norm: Mapping[str, Any],
    bpw_table: Mapping[str, float],
) -> Tuple[float, float]:
    ref_max_x, _ = get_highest_bpw_qtypes(ref_norm, bpw_table)
    tgt_max_x, _ = get_highest_bpw_qtypes(tgt_norm, bpw_table)
    print(f"[DEBUG] extract_bpw_range: ref_max_x = {ref_max_x}, tgt_max_x = {tgt_max_x}")
    return ref_max_x, tgt_max_x


def compute_global_envelope_ratio(
    ref_norm: Mapping[str, Any],
    tgt_norm: Mapping[str, Any],
    ref_mean_fn: Callable[[float], float],
    tgt_mean_adj_fn: Callable[[float], float],
    bpw_table: Mapping[str, float],
) -> float:
    eps = 1e-12
    ratios = []
    ratios_info = []
    skip_qtypes = {'IQ1_S', 'IQ1_S_R4'}
    for qtype in sorted(set(ref_norm.keys()) & set(tgt_norm.keys())):
        if qtype in skip_qtypes:
            continue
        x = bpw_table.get(qtype)
        if x is None:
            continue
        ref_val = to_float(ref_norm.get(qtype))
        tgt_val = to_float(tgt_norm.get(qtype))
        if ref_val is None or tgt_val is None:
            continue

        dev_ref = ref_val - ref_mean_fn(x)
        if abs(dev_ref) < eps:
            continue

        dev_tgt = tgt_val - tgt_mean_adj_fn(x)
        ratio = dev_tgt / dev_ref
        if ratio <= 0:
            ratios_info.append((qtype, x, ref_val, ref_mean_fn(x), tgt_val, tgt_mean_adj_fn(x), ratio, "discarded(<=0)"))
            continue
        ratios_info.append((qtype, x, ref_val, ref_mean_fn(x), tgt_val, tgt_mean_adj_fn(x), ratio, "kept"))
        ratios.append(clamp(ratio, 0.05, 20.0))

    print("\n[DEBUG] Candidate envelope ratios:")
    for info in ratios_info:
        print(f"  {info[0]}: x={info[1]:.3f} ref={info[2]:.6f} ref_mean={info[3]:.6f} tgt={info[4]:.6f} tgt_mean_adj={info[5]:.6f} ratio={info[6]:.6f} {info[7]}")
    print(f"  positive ratios (clamped): {ratios}")

    if not ratios:
        print("[DEBUG] No positive ratios, using default 1.0")
        return 1.0

    alpha = median(ratios)
    alpha = clamp(alpha, 0.1, 10.0)
    print(f"[DEBUG] Computed global alpha = {alpha:.6f}")
    return alpha


def preserve_sibling_ratios_from_target(
    enriched: Dict[str, float],
    tgt_norm: Dict[str, Any],
) -> None:
    for qtype in list(enriched.keys()):
        base = re.sub(r"_R[48]$", "", qtype, flags=re.IGNORECASE)
        if base == qtype:
            continue
        if base not in tgt_norm or qtype not in tgt_norm:
            continue
        base_tgt = to_float(tgt_norm[base])
        variant_tgt = to_float(tgt_norm[qtype])
        if base_tgt is None or variant_tgt is None or base_tgt == 0:
            continue
        ratio = variant_tgt / base_tgt
        old_val = enriched.get(qtype, 0.0)
        new_val = enriched.get(base, 0.0) * ratio
        enriched[qtype] = new_val
        print(f"[DEBUG ratio preserve] {qtype} vs {base}: target ratio {ratio:.6f}, setting {qtype} from {old_val:.6f} to {new_val:.6f}")


def apply_sibling_unification(
    enriched: Dict[str, float],
    tgt_norm: Dict[str, Any],
) -> None:
    all_qtypes = set(enriched.keys())

    for qtype in sorted(all_qtypes):
        tgt_original = to_float(tgt_norm.get(qtype))
        if tgt_original is not None:
            print(f"[DEBUG sibling] {qtype} was originally present ({tgt_original}), skip unification")
            continue

        base = re.sub(r"_R[48]$", "", qtype, flags=re.IGNORECASE)
        if base != qtype and base in all_qtypes:
            if not should_unify_r_variants(qtype, base):
                print(f"[DEBUG sibling] {qtype} vs {base} is the iq1_s exception, skip")
                continue
            base_tgt = to_float(tgt_norm.get(base))
            if base_tgt is not None:
                old_val = enriched[qtype]
                new_val = enriched[base]
                enriched[qtype] = new_val
                print(f"[DEBUG sibling] {qtype} missing in target; base {base} was present. "
                      f"Changing {qtype} from {old_val:.6f} to {new_val:.6f} (copied from base)")
                continue

        for suffix in ("_R4", "_R8"):
            variant = qtype + suffix
            if variant in all_qtypes:
                if not should_unify_r_variants(qtype, variant):
                    print(f"[DEBUG sibling] {qtype} vs {variant} is iq1_s exception, skip")
                    continue
                variant_tgt = to_float(tgt_norm.get(variant))
                if variant_tgt is not None:
                    old_val = enriched[qtype]
                    new_val = enriched[variant]
                    enriched[qtype] = new_val
                    print(f"[DEBUG sibling] {qtype} missing in target; variant {variant} was present. "
                          f"Changing {qtype} from {old_val:.6f} to {new_val:.6f} (copied from variant)")
                    break


def fill_all_qtypes_by_global_transposition(
    reference_values: Mapping[str, Any],
    target_values: Mapping[str, Any],
    reference_mean_fn: Callable[[float], float],
    target_mean_fn: Callable[[float], float],
    bpw_table: Mapping[str, float],
) -> Tuple[Dict[str, float], Dict[str, float], float]:
    ref_norm = {norm_qtype(k): v for k, v in reference_values.items()}
    tgt_norm = {norm_qtype(k): v for k, v in target_values.items()}

    ref_max_x, ref_highest_qtypes = get_highest_bpw_qtypes(ref_norm, bpw_table)
    tgt_max_x, tgt_highest_qtypes = get_highest_bpw_qtypes(tgt_norm, bpw_table)

    # Determine if the two datasets share the same "baseline" (highest valid quant type)
    same_highest_qtype = bool(ref_highest_qtypes & tgt_highest_qtypes)
    same_bpw = (abs(ref_max_x - tgt_max_x) < 1e-9)

    if same_bpw or same_highest_qtype:
        # Same baseline: no offset, preserve existing target values
        def tgt_mean_adj(x: float) -> float:
            return target_mean_fn(x)
        full_overwrite = False
        print(f"[Info] Baseline same (BPW match: {same_bpw}, same highest qtype: {same_highest_qtype}) "
              f"→ no baseline offset; existing target values preserved.")
    elif tgt_max_x < ref_max_x:
        # Baseline mismatch: offset target mean and overwrite all
        y_at_ref_max = target_mean_fn(ref_max_x)
        def tgt_mean_adj(x: float) -> float:
            return target_mean_fn(x) - y_at_ref_max
        full_overwrite = True
        print(f"[Info] Baseline adjustment: target max BPW = {tgt_max_x}, ref max BPW = {ref_max_x}. "
              f"Subtracted {y_at_ref_max:.6g} from target mean to align at BPW={ref_max_x}.")
        print(f"[DEBUG] y_at_ref_max = {y_at_ref_max:.10f}")
    else:
        # tgt_max_x > ref_max_x (shouldn't happen, but treat as same baseline)
        def tgt_mean_adj(x: float) -> float:
            return target_mean_fn(x)
        full_overwrite = False
        print(f"[Info] Target max BPW ({tgt_max_x}) >= ref max BPW ({ref_max_x}) → no offset; preserving values.")

    alpha = compute_global_envelope_ratio(
        ref_norm, tgt_norm,
        reference_mean_fn, tgt_mean_adj,
        bpw_table,
    )
    print(f"[Info] Global envelope ratio α = {alpha:.6f}")

    enriched = {}
    filled = {}

    for qtype in sorted(ref_norm.keys()):
        x = bpw_table.get(qtype)
        ref_val = to_float(ref_norm[qtype])

        if ref_val is None and x is not None:
            ref_val = lookup_qtype_with_r_suffix(ref_norm, qtype)

        if x is not None and ref_val is not None:
            ref_mean = reference_mean_fn(x)
            tgt_m = tgt_mean_adj(x)
            dev_ref = ref_val - ref_mean
            pred = tgt_m + alpha * dev_ref
            pred = max(0.0, pred)

            if not full_overwrite and qtype in tgt_norm and not is_missing_value(tgt_norm[qtype]):
                enriched[qtype] = to_float(tgt_norm[qtype])
                if qtype in ('IQ1_S', 'IQ1_S_R4'):
                    print(f"[DEBUG transpose] {qtype}: preserved original target value {enriched[qtype]:.6f}")
            else:
                enriched[qtype] = float(pred)
                if qtype in ('IQ1_S', 'IQ1_S_R4'):
                    print(f"\n[DEBUG transpose] {qtype}: x={x:.3f} ref_val={ref_val:.6f} ref_mean={ref_mean:.6f} dev_ref={dev_ref:.6f} "
                          f"tgt_mean_adj={tgt_m:.6f} alpha={alpha:.6f} pred={pred:.6f}")
        elif x is not None:
            pred = max(0.0, tgt_mean_adj(x))
            if not full_overwrite and qtype in tgt_norm and not is_missing_value(tgt_norm[qtype]):
                enriched[qtype] = to_float(tgt_norm[qtype])
            else:
                enriched[qtype] = pred
        else:
            enriched[qtype] = 0.0

        tgt_original = to_float(tgt_norm.get(qtype))
        if tgt_original is None:
            filled[qtype] = enriched[qtype]

    if full_overwrite:
        print("\n[DEBUG] Applying target sibling ratio preservation...")
        preserve_sibling_ratios_from_target(enriched, tgt_norm)
    else:
        print("\n[DEBUG] Skipping sibling ratio preservation (existing target values are kept).")

    print("\n[DEBUG] Starting sibling unification...")
    apply_sibling_unification(enriched, tgt_norm)

    print(f"\n[DEBUG] Final enriched values for iq1_s: {enriched.get('IQ1_S')}")
    print(f"[DEBUG] Final enriched values for iq1_s_r4: {enriched.get('IQ1_S_R4')}")

    return enriched, filled, alpha


# -----------------------------
# Reference ranking adjustment
# -----------------------------

def adjust_reference_for_ordering(
    ref_norm: Dict[str, float],
    tgt_norm: Dict[str, Any],
    ref_mean_fn: Callable[[float], float],
    tgt_mean_adj_fn: Callable[[float], float],
    bpw_table: Dict[str, float],
    aggressiveness: float = 0.8,
    min_discrepancy: float = 0.1,
    max_discrepancy: float = 2.0,
) -> Dict[str, float]:
    """
    Adjust reference values to better respect the ordering / ratios observed
    in the known target values, while staying anchored to the reference mean curve.

    Returns a new dictionary with adjusted normalized reference values.
    """
    factor_map: Dict[str, Optional[float]] = {}  # qtype -> deviation multiplier
    new_ref: Dict[str, float] = {}

    # First pass: adjust known qtypes
    for qtype, ref_val in ref_norm.items():
        x = bpw_table.get(qtype)
        if x is None:
            new_ref[qtype] = ref_val
            continue

        tgt_val = to_float(tgt_norm.get(qtype))
        if tgt_val is None:
            # No target info – keep original for now
            new_ref[qtype] = ref_val
            continue

        ref_val_float = float(ref_val)
        ref_mean = ref_mean_fn(x)
        tgt_adj_mean = tgt_mean_adj_fn(x)

        ref_orig_dev = ref_val_float - ref_mean
        tgt_dev_adj = tgt_val - tgt_adj_mean

        if abs(ref_orig_dev) < 1e-12:
            # Reference point lies exactly on its mean – directly adopt target deviation
            new_ref_dev = tgt_dev_adj
            factor_map[qtype] = None
            new_val = max(0.0, ref_mean + new_ref_dev)
            new_ref[qtype] = new_val
        else:
            ratio = tgt_dev_adj / ref_orig_dev
            disc = abs(ratio - 1.0)

            # Map discrepancy to weight w ∈ [0, aggressiveness]
            if disc <= min_discrepancy:
                w = 0.0
            elif disc >= max_discrepancy:
                w = aggressiveness
            else:
                w = aggressiveness * (disc - min_discrepancy) / (max_discrepancy - min_discrepancy)

            new_dev = ref_orig_dev + w * (tgt_dev_adj - ref_orig_dev)
            new_val = max(0.0, ref_mean + new_dev)
            new_ref[qtype] = new_val
            factor = new_dev / ref_orig_dev
            factor_map[qtype] = factor

    # Second pass: propagate factors to unknown siblings (only when base/variant is known)
    for qtype in sorted(new_ref.keys()):
        if qtype in factor_map:
            continue  # already adjusted from target data
        x = bpw_table.get(qtype)
        if x is None:
            continue

        # Try to find a known counterpart
        base = re.sub(r"_R[48]$", "", qtype, flags=re.IGNORECASE)
        known_q = None
        if base != qtype and base in factor_map and factor_map[base] is not None:
            known_q = base
        else:
            for suffix in ("_R4", "_R8"):
                variant = qtype + suffix
                if variant in factor_map and factor_map[variant] is not None:
                    known_q = variant
                    break

        if known_q is not None:
            factor = factor_map[known_q]
            # compute original reference deviation for this missing qtype
            orig_ref_val = float(new_ref.get(qtype, ref_norm.get(qtype, 0.0)))
            ref_mean = ref_mean_fn(x)
            orig_dev = orig_ref_val - ref_mean
            new_dev = orig_dev * factor
            new_val = max(0.0, ref_mean + new_dev)
            new_ref[qtype] = new_val

    return new_ref


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
    target_rows_by_norm: Dict[str, Dict[str, str]] = {}
    target_order: List[str] = []

    for row in target_rows:
        qnorm = norm_qtype(row.get(qtype_col, ""))
        if qnorm not in target_rows_by_norm:
            target_order.append(qnorm)
        target_rows_by_norm[qnorm] = dict(row)

    output_rows: List[Dict[str, Any]] = []
    emitted: set = set()

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
            "count": 0.0, "mae": float("nan"), "rmse": float("nan"),
            "max_abs_error": float("nan"), "mean_signed_error": float("nan"),
            "mape_pct": float("nan"),
        }

    mae = sum(abs_errors) / used
    rmse = math.sqrt(sum(squared_errors) / used)
    max_abs_error = max(abs_errors)
    mean_signed_error = sum(signed_errors) / used
    mape = (sum(relative_errors) / len(relative_errors) * 100.0) if relative_errors else float("nan")

    return {
        "count": float(used), "mae": mae, "rmse": rmse,
        "max_abs_error": max_abs_error, "mean_signed_error": mean_signed_error,
        "mape_pct": mape,
    }


def make_comparison_plot(
    predicted: Mapping[str, float],
    filled_keys: Mapping[str, float],
    target_values: Mapping[str, Any],
    reference_values: Mapping[str, Any],
    reference_mean_fn: Callable[[float], float],
    target_mean_fn: Callable[[float], float],
    target_mean_adj_fn: Optional[Callable] = None,
    alpha: float = 1.0,
    bpw_table: Mapping[str, float] = BPW_TABLE,
    compare_truth: Optional[Mapping[str, Any]] = None,
) -> None:
    plt.figure(figsize=(12, 7))

    ref_x, ref_y, ref_lbl = [], [], []
    tgt_x, tgt_y, tgt_lbl = [], [], []
    guessed_x, guessed_y, guessed_lbl = [], [], []
    gt_x, gt_y, gt_lbl = [], [], []

    for qtype, val in reference_values.items():
        x = bpw_table.get(norm_qtype(qtype))
        y = to_float(val)
        if x is not None and y is not None:
            ref_x.append(x); ref_y.append(y); ref_lbl.append(norm_qtype(qtype))

    for qtype, val in target_values.items():
        x = bpw_table.get(qtype)
        y = to_float(val)
        if x is not None and y is not None:
            tgt_x.append(x); tgt_y.append(y); tgt_lbl.append(qtype)

    guessed_qtypes = set(filled_keys.keys())
    for qtype, pred_val in predicted.items():
        if qtype in guessed_qtypes:
            x = bpw_table.get(qtype)
            y = to_float(pred_val)
            if x is not None and y is not None:
                guessed_x.append(x); guessed_y.append(y); guessed_lbl.append(qtype)

    if compare_truth is not None:
        for qtype in guessed_qtypes:
            x = bpw_table.get(qtype)
            y = to_float(compare_truth.get(qtype))
            if x is not None and y is not None:
                gt_x.append(x); gt_y.append(y); gt_lbl.append(qtype)

    if ref_x:
        plt.scatter(ref_x, ref_y, marker="+", s=90, color="black", label="Reference values", alpha=0.9)
    if tgt_x:
        plt.scatter(tgt_x, tgt_y, marker="+", s=90, color="blue", label="Target original", alpha=0.8)
    if guessed_x:
        plt.scatter(guessed_x, guessed_y, marker="+", s=110, color="red", label="Filled/predicted values", alpha=0.95)
    if gt_x:
        plt.scatter(gt_x, gt_y, marker="+", s=110, color="green", label="Ground truth", alpha=0.95)

    all_curve_x = sorted(set(BPW_TABLE.values()))
    if all_curve_x:
        x_min = min(all_curve_x); x_max = max(all_curve_x)
        xs = [x_min + (x_max - x_min) * i / 300.0 for i in range(301)]
        ref_curve = [reference_mean_fn(x) for x in xs]
        tgt_curve = [target_mean_fn(x) for x in xs]
        tgt_adj_curve = [target_mean_adj_fn(x) if target_mean_adj_fn else target_mean_fn(x) for x in xs] if target_mean_adj_fn else None
        plt.plot(xs, ref_curve, color="black", linestyle="--", linewidth=1.5, label="Reference mean")
        plt.plot(xs, tgt_curve, color="blue", linestyle="--", linewidth=1.5, label="Target mean (original)")
        if tgt_adj_curve:
            plt.plot(xs, tgt_adj_curve, color="cyan", linestyle="--", linewidth=1.5,
                     label=f"Target mean adjusted (α={alpha:.3f})")

    all_x = ref_x + tgt_x + guessed_x + gt_x
    all_labels = ref_lbl + tgt_lbl + guessed_lbl + gt_lbl
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
    plt.title("Enriched quant degradation")
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
    parser.add_argument("--target-csv", required=True, help="Path to the target CSV to enrich.")
    parser.add_argument("--target-mean-equation", required=True, help="Mean equation for the target curve.")
    parser.add_argument("--output-csv", required=True, help="Path for the enriched CSV.")
    parser.add_argument("--reference-csv", default=None, help="Optional reference CSV.")
    parser.add_argument("--reference-mean-equation", default=None, help="Mean equation for the reference if --reference-csv given.")
    parser.add_argument("--compare-with-csv", default=None, help="Fully‑known CSV for error metrics.")
    parser.add_argument("--precision", type=int, default=6, help="Decimal precision for output values.")
    parser.add_argument("--report-filled-only", action="store_true", help="Print only the filled entries.")
    parser.add_argument("--ref-ranking-aggressiveness", type=float, default=None,
                       help="Aggressiveness for reference ranking adjustment (0-1). If provided, only this single "
                            "pass is used (no averaging).")
    parser.add_argument("--no-mean-ranking", action="store_true",
                       help="Disable the default averaging of ranked and non‑ranked passes. "
                            "When set, only the ranking pass is used (with default aggressiveness 0.8, "
                            "or the value supplied by --ref-ranking-aggressiveness).")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    (
        target_fields, target_rows, target_qtype_col, target_value_col,
        _target_raw_values, target_values, target_order, _target_case_by_norm, target_style,
    ) = build_csv_qtype_data(args.target_csv, apply_sibling_fallback=True)

    target_mean_fn = parse_equation_to_callable(args.target_mean_equation)
    if target_mean_fn is None:
        raise SystemExit("[Error] Invalid --target-mean-equation provided; aborting.")

    (
        reference_raw_values, reference_values, reference_order,
        reference_case_by_norm, reference_mean_fn, reference_style,
    ) = load_reference_data(args)

    if target_style is None:
        target_style = reference_style
    if reference_style is None:
        reference_style = target_style

    if reference_style != target_style:
        raise SystemExit(
            f"[Error] Value format mismatch: reference data is '{reference_style}' "
            f"but target data is '{target_style}'. Both CSVs must use the same format."
        )

    # Determine baseline offset information (unchanged logic)
    ref_norm_for_baseline = {norm_qtype(k): v for k, v in reference_values.items()}
    tgt_norm_for_baseline = {norm_qtype(k): v for k, v in target_values.items()}
    ref_max_x, _ = get_highest_bpw_qtypes(ref_norm_for_baseline, BPW_TABLE)
    tgt_max_x, _ = get_highest_bpw_qtypes(tgt_norm_for_baseline, BPW_TABLE)
    if tgt_max_x < ref_max_x:
        y_off = target_mean_fn(ref_max_x)
        tgt_mean_adj_fn = lambda x: target_mean_fn(x) - y_off
    else:
        tgt_mean_adj_fn = target_mean_fn

    # Default ranking aggressiveness
    DEFAULT_AGGRESSIVENESS = 0.8

    # Determine which passes to run and how to combine them
    do_ranking = True   # we always need the ranking pass in some form
    do_no_ranking = False
    averaging_mode = False

    if args.ref_ranking_aggressiveness is not None:
        # User explicitly set aggressiveness -> single pass with that value
        rank_agg = args.ref_ranking_aggressiveness
        do_no_ranking = False
        averaging_mode = False
        print(f"[Info] Using single ranking pass with aggressiveness = {rank_agg}")
    elif args.no_mean_ranking:
        # User disables averaging -> single ranking pass with default aggressiveness
        rank_agg = DEFAULT_AGGRESSIVENESS
        do_no_ranking = False
        averaging_mode = False
        print("[Info] --no-mean-ranking active: using single ranking pass with default aggressiveness.")
    else:
        # Default: average the ranking and non‑ranking passes
        rank_agg = DEFAULT_AGGRESSIVENESS
        do_no_ranking = True
        averaging_mode = True
        print("[Info] Default mode: averaging ranking and non‑ranking passes.")

    # ---------- Compute enriched values ----------
    enriched_final: Dict[str, float] = {}
    filled_final: Dict[str, float] = {}
    alphas = []

    # Helper that runs a single enrichment pass and returns the enriched dict
    def run_enrichment_pass(ref_for_pass: Dict[str, float], label: str) -> Dict[str, float]:
        print(f"\n[Step] Enrichment pass: {label}")
        enriched_pass, filled_pass, alpha_pass = fill_all_qtypes_by_global_transposition(
            reference_values=ref_for_pass,
            target_values=target_values,
            reference_mean_fn=reference_mean_fn,
            target_mean_fn=target_mean_fn,
            bpw_table=BPW_TABLE,
        )
        alphas.append(alpha_pass)
        return enriched_pass

    # Pass 1: no ranking (original reference)
    if do_no_ranking:
        enriched_no_rank = run_enrichment_pass(reference_values, "no ranking (original reference)")

    # Pass 2: with ranking
    # Adjust reference
    adjusted_ref_norm = adjust_reference_for_ordering(
        ref_norm=reference_values,
        tgt_norm=target_values,
        ref_mean_fn=reference_mean_fn,
        tgt_mean_adj_fn=tgt_mean_adj_fn,
        bpw_table=BPW_TABLE,
        aggressiveness=rank_agg,
    )
    enriched_rank = run_enrichment_pass(adjusted_ref_norm, f"ranking (aggressiveness={rank_agg})")

    # Combine results
    if averaging_mode and do_no_ranking:
        print("\n[Info] Averaging the two enrichment passes:")
        all_qtypes = set(enriched_no_rank.keys()) | set(enriched_rank.keys())
        for q in sorted(all_qtypes):
            v1 = enriched_no_rank.get(q, 0.0)
            v2 = enriched_rank.get(q, 0.0)
            enriched_final[q] = (v1 + v2) / 2.0
        # Also compute a unified filled dict (keys missing in target)
        for q in all_qtypes:
            if q not in target_values or is_missing_value(target_values[q]):
                filled_final[q] = enriched_final[q]
    else:
        enriched_final = enriched_rank
        # filled dict from the ranking pass (the fill_all... returns filled, we need to capture it)
        # We'll recompute filled from enriched_final for consistency
        for qtype in enriched_final:
            if qtype not in target_values or is_missing_value(target_values[qtype]):
                filled_final[qtype] = enriched_final[qtype]

    # For plotting / output we need a reference dict that matches the final enriched.
    # For the plot we can use the adjusted reference from ranking pass, or the original.
    # We'll use the adjusted ref for consistency (since final mostly comes from ranking pass).
    # Build adjusted reference raw values for plotting
    adjusted_reference_raw = {}
    for norm_q, raw_q in reference_case_by_norm.items():
        if norm_q in adjusted_ref_norm:
            adjusted_reference_raw[raw_q] = adjusted_ref_norm[norm_q]

    # Output rows
    output_rows = build_output_rows_preserving_reference_order(
        target_fields=target_fields,
        qtype_col=target_qtype_col,
        value_col=target_value_col,
        target_rows=target_rows,
        reference_order=reference_order,
        enriched=enriched_final,
        target_style=target_style or "float",
        precision=args.precision,
    )

    with Path(args.output_csv).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=target_fields)
        writer.writeheader()
        writer.writerows(output_rows)

    # Comparison and plot
    compare_truth = None
    if args.compare_with_csv:
        (
            _cmp_fields, _cmp_rows, cmp_qtype_col, cmp_value_col,
            _cmp_raw_values, compare_truth, _cmp_order, _cmp_case_by_norm, _cmp_style,
        ) = build_csv_qtype_data(args.compare_with_csv, apply_sibling_fallback=False)

        metrics = compute_error_metrics(enriched_final, compare_truth)
        make_comparison_plot(
            enriched_final, filled_final,
            target_values, adjusted_reference_raw,   # show adjusted ref on plot
            reference_mean_fn, target_mean_fn, tgt_mean_adj_fn,
            alpha=alphas[-1] if alphas else 1.0,    # use last alpha for display
            bpw_table=BPW_TABLE,
            compare_truth=compare_truth,
        )

        print("\n[Compare] Error metrics vs known-all-values CSV:")
        print(f"[Compare] count={int(metrics['count'])}")
        print(f"[Compare] MAE={metrics['mae']:.10f}")
        print(f"[Compare] RMSE={metrics['rmse']:.10f}")
        print(f"[Compare] MaxAbsError={metrics['max_abs_error']:.10f}")
        print(f"[Compare] MeanSignedError={metrics['mean_signed_error']:.10f}")
        if not math.isnan(metrics['mape_pct']):
            print(f"[Compare] MAPE%={metrics['mape_pct']:.6f}")

    # Final report
    print(f"\n[Info] Wrote enriched CSV to: {args.output_csv}")
    if len(alphas) == 1:
        print(f"[Info] Global envelope ratio α = {alphas[0]:.6f}")
    else:
        print(f"[Info] Global envelope ratio α (ranking) = {alphas[1]:.6f}, α (no ranking) = {alphas[0]:.6f}")
    print(f"[Info] Predicted values for {len(enriched_final)} qtypes")
    if args.report_filled_only:
        print("[Info] Entries that were missing in target (now filled):")
        for k in sorted(filled_final):
            print(f"{k},{format_float(filled_final[k], args.precision)}")


if __name__ == "__main__":
    main()