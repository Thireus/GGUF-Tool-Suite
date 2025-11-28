#!/usr/bin/env python3
#***************************************************************#
#** This script is part of Thireus' GGUF Tool Suite.          **#
#** model_tensor_bpw_metric.py is a tool that compares pure   **#
#** quantized model bpw versus reported metrics such as KLD.  **#
#**                                                           **#
#** ********************************************************* **#
#** --------------- Updated: Nov-27-2025 -------------------- **#
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
#** Copyright © 2025 - Thireus.          ᵥᵢᵦₑ 𝒸ₒ𝒹ᵢₙ𝑔 ₑₙ𝑔ᵢₙₑₑᵣ **#
#***************************************************************#
#**PLEASE REFER TO THE README FILE FOR ADDITIONAL INFORMATION!**#
#***************************************************************#

from __future__ import annotations
import argparse
import csv
import math
import os
import re
import sys
import json
from typing import List, Optional, Dict, Any, Tuple

import numpy as np
import matplotlib.pyplot as plt

# concurrency imports
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

RE_DTYPE = re.compile(r"\bdtype=([^:]+)", flags=re.IGNORECASE)
RE_ELEMENTS = re.compile(r"\belements=(\d+)", flags=re.IGNORECASE)
RE_BYTES = re.compile(r"\bbytes=(\d+)", flags=re.IGNORECASE)

ACCEPT_F32 = {"f32"}  # only accept 'f32' (not 'fp32' or 'float32')


# --------------------------
# utilities & map parsing
# --------------------------

def parse_map_line(line: str) -> Optional[Tuple[Optional[str], Optional[int], Optional[int], str]]:
    if not line or not line.strip():
        return None
    parts = line.split(":")
    name = parts[2] if len(parts) >= 3 else parts[0]

    dtype_m = RE_DTYPE.search(line)
    elems_m = RE_ELEMENTS.search(line)
    bytes_m = RE_BYTES.search(line)

    dtype = dtype_m.group(1).lower() if dtype_m else None
    elements = int(elems_m.group(1)) if elems_m else None
    bytes_ = int(bytes_m.group(1)) if bytes_m else None

    return (dtype, elements, bytes_, name)


def infer_qtype_from_filename(path: str) -> Optional[str]:
    base = os.path.basename(path)
    m = re.search(r"tensors\.([^.]+)\.map$", base, flags=re.IGNORECASE)
    if m:
        return m.group(1).lower()
    m2 = re.search(r"([iqIQ]\d[_A-Za-z0-9\-]*)", base)
    if m2:
        return m2.group(1).lower()
    return None


def family_of(qtoken: Optional[str]) -> Optional[str]:
    if not qtoken:
        return None
    s = qtoken.lower()
    s = re.sub(r'(_r\d+.*)$', '', s)
    return s


def human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024.0:
            return f"{n:.2f} {unit}"
        n /= 1024.0
    return f"{n:.2f} PB"


def load_parsed_entries_from_mapfile(path: str) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as fh:
        lines = [ln.rstrip("\n") for ln in fh if ln.strip()]
    qtype = infer_qtype_from_filename(path)
    parsed = []
    for ln in lines:
        p = parse_map_line(ln)
        if p is None:
            continue
        dtype, elements, bytes_, name = p
        parsed.append({"dtype": dtype, "elements": elements, "bytes": bytes_, "name": name, "line": ln})
    return qtype, parsed


def compute_bpw_for_qtype(parsed_entries: List[Dict[str, Any]],
                          declared_qtype: Optional[str],
                          allow_impure_map: bool,
                          fail_on_missing_bytes: bool) -> Tuple[Optional[float], Dict[str, Any]]:
    declared_family = family_of(declared_qtype)
    hard_impure = []
    soft_impure = []
    accepted_mask = []

    for e in parsed_entries:
        dt = e["dtype"]
        if dt is None:
            hard_impure.append((e["name"], "<none>"))
            accepted_mask.append(False)
            continue

        if any(dt == x or dt.startswith(x + "_") or dt.startswith(x + "-") for x in ACCEPT_F32):
            accepted_mask.append(True)
            continue

        if declared_qtype and declared_qtype in dt:
            accepted_mask.append(True)
            continue

        dt_family = family_of(dt)
        if declared_family is not None and dt_family == declared_family:
            soft_impure.append((e["name"], dt))
            accepted_mask.append(True)
            continue

        hard_impure.append((e["name"], dt))
        accepted_mask.append(False)

    if hard_impure and not allow_impure_map:
        return None, {
            "reason": "hard_impure",
            "hard_impure": hard_impure,
            "soft_impure": soft_impure,
            "total_elements_with_bytes": None,
            "total_bytes": None,
            "missing_bytes_count": None,
        }

    total_elements_with_bytes = 0
    total_bytes = 0
    missing_bytes_count = 0

    for e, accepted in zip(parsed_entries, accepted_mask):
        if not accepted:
            continue
        elems = e["elements"]
        b = e["bytes"]
        if elems is None:
            continue
        if b is None:
            missing_bytes_count += 1
            continue
        total_elements_with_bytes += elems
        total_bytes += b

    if missing_bytes_count and fail_on_missing_bytes:
        return None, {
            "reason": "missing_bytes",
            "hard_impure": hard_impure,
            "soft_impure": soft_impure,
            "total_elements_with_bytes": total_elements_with_bytes,
            "total_bytes": total_bytes,
            "missing_bytes_count": missing_bytes_count,
        }

    if total_elements_with_bytes == 0 or total_bytes == 0:
        return None, {
            "reason": "no_bytes_available",
            "hard_impure": hard_impure,
            "soft_impure": soft_impure,
            "total_elements_with_bytes": total_elements_with_bytes,
            "total_bytes": total_bytes,
            "missing_bytes_count": missing_bytes_count,
        }

    bpw = (total_bytes * 8) / total_elements_with_bytes
    return bpw, {
        "reason": "ok",
        "hard_impure": hard_impure,
        "soft_impure": soft_impure,
        "total_elements_with_bytes": total_elements_with_bytes,
        "total_bytes": total_bytes,
        "missing_bytes_count": missing_bytes_count,
    }


# --------------------------
# transforms & fitter (generalized)
# --------------------------

def apply_transform(vals: np.ndarray, kind: str, log_base: Optional[float] = None) -> np.ndarray:
    """Apply transform to vals. kind in {'ln','log10','log2','logn','identity'}.
    For 'logn', provide log_base (N > 0, N != 1)."""
    if kind == "ln":
        return np.log(vals)
    if kind == "log10":
        return np.log10(vals)
    if kind == "log2":
        return np.log2(vals)
    if kind == "logn":
        if log_base is None or log_base <= 0 or log_base == 1:
            raise ValueError("logn requires a valid base != 1")
        # compute natural log divided by ln(N)
        return np.log(vals) / math.log(log_base)
    if kind == "identity":
        return vals
    raise ValueError(f"Unknown transform kind: {kind}")


def _compute_resemblance_score(y_true: np.ndarray,
                             y_pred: np.ndarray,
                             metric: str = "asym_abs",
                             penalize_above: float = 2.0,
                             penalize_below: float = 1.0) -> float:
    """
    Compute a scalar score (lower is better) based on the chosen metric.
    Supported metrics:
      - 'sse'         : sum squared error (lower better)
      - 'mae'/'abs_mean': mean absolute error
      - 'median_abs'  : median absolute error
      - 'r2'          : 1 - R^2 (lower better; R^2 higher is better)
      - 'asym_abs'    : asymmetric mean absolute error, penalize_above applied to over-predictions (pred>true),
                        penalize_below to under-predictions.
      - 'penalize_above' : same as asym_abs but penalize_above fixed to 2.0 (convenience)
      - 'penalize_below' : same as asym_abs but penalize_below fixed to 1.0 (convenience)
    """
    # ensure numpy arrays
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)

    # mask finite
    mask = np.isfinite(y_t) & np.isfinite(y_p)
    if not np.any(mask):
        return float("inf")

    yt = y_t[mask]
    yp = y_p[mask]
    resid = yp - yt
    abs_resid = np.abs(resid)

    metric = (metric or "asym_abs").lower()
    if metric == "sse":
        return float(np.sum(resid ** 2))
    if metric in {"mae", "abs_mean"}:
        return float(np.mean(abs_resid))
    if metric == "median_abs":
        return float(np.median(abs_resid))
    if metric == "r2":
        # compute r2; return 1 - r2 (so lower is better)
        ss_res = np.sum((yt - yp) ** 2)
        ss_tot = np.sum((yt - np.mean(yt)) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot != 0 else float("nan")
        # If r2 is nan, return large number
        if not np.isfinite(r2):
            return float("inf")
        return float(1.0 - r2)
    # asymmetric absolute
    if metric in {"asym_abs", "penalize_above", "penalize_below"}:
        if metric == "penalize_above":
            pa = max(1.0, float(penalize_above))
            pb = 1.0
        elif metric == "penalize_below":
            pa = 1.0
            pb = max(1.0, float(penalize_below))
        else:
            pa = max(0.0, float(penalize_above))
            pb = max(0.0, float(penalize_below))
        weights = np.where(resid > 0, pa, pb)  # over-predictions get pa
        weighted = abs_resid * weights
        return float(np.mean(weighted))

    # fallback to sse
    return float(np.sum(resid ** 2))


# mapping for pair -> result
_pair_map = {
    ('+', '-'): '-',
    ('-', '+'): '-',
    ('+', '+'): '+',
    ('-', '-'): '+',
}

_pair_re = re.compile(r'([+\-])\s+([+\-])')  # at least one space between the signs

def collapse_sign_pairs(s: str) -> str:
    """
    Collapse sign pairs separated by one or more spaces using simple algebraic rules.
    Only matches sign1 <spaces> sign2 so it won't change signs that are part of numbers (e.g. "1e-5").
    Repeats until no change (handles chains like "+ - - +").
    """
    prev = None
    out = s
    while prev != out:
        prev = out
        # Replace each matched pair with a single sign surrounded by single spaces.
        # We use a function so we can look up the algebraic collapse result.
        def _repl(m):
            a, b = m.group(1), m.group(2)
            new = _pair_map.get((a, b), '+' if a == b else '-')  # fallback (shouldn't be needed)
            # keep single spaces on both sides of the resulting operator
            return f' {new} '
        out = _pair_re.sub(_repl, out)
    # tidy up multiple spaces that may have been created, but preserve spacing around other tokens
    out = re.sub(r'\s{2,}', ' ', out)
    return out.strip()


def fit_model_general(xarr: np.ndarray,
                      yarr: np.ndarray,
                      d_fixed: Optional[float] = None,
                      c_fixed: Optional[float] = None,
                      transforms: Optional[List[str]] = None,
                      # tunable grids (exposed via CLI)
                      p_grid_min: float = 0.2,
                      p_grid_max: float = 3.0,
                      p_grid_steps: int = 15,
                      logn_base_min: float = 2.0,
                      logn_base_max: float = 100.0,
                      logn_base_steps: int = 8,
                      c_mult_min: float = -0.9,
                      c_mult_max: float = 10.0,
                      c_mult_steps: int = 40,
                      b_grid_steps: int = 60,
                      # refinement densities
                      b_refine_steps: int = 60,
                      c_refine_steps: int = 60,
                      p_refine_steps: int = 20,
                      N_refine_steps: int = 12,
                      # identity-specific s_grid params (existing flags)
                      identity_s_min: float = -1.0,
                      identity_s_max: float = 1.0,
                      identity_s_steps: int = 9,
                      # resemblance metric options
                      resemblance_metric: str = "asym_abs",
                      penalize_above: float = 2.0,
                      penalize_below: float = 1.0
                      ) -> Tuple[Optional[Dict[str, float]], Dict[str, Any]]:
    """
    Fit model y = d + a * (T)^{-p} where T = transform(b * (x - c)),
    trying multiple transforms and exponent p values.

    xarr is bpw (independent variable), yarr is the metric (dependent variable).
    If d_fixed is provided, d is anchored and we solve for a (and possibly other params).
    If c_fixed is provided, c is anchored (and b is still searched).
    Otherwise, c and b are both searched in the grid.
    Returns params dict containing a,b,c,d,p,transform,r2,sse or None if no fit.

    Notes / Changes:
      - 'logn' transform is supported: the base N is searched over a small grid to pick the best.
      - we only require vals > 0 for transforms that need positive inputs (natural/log bases);
        identity transform accepts negative vals.
      - warnings from invalid operations (like power of negative non-integer) are suppressed
        where we explicitly check isfinite after computation.
      - selection of the 'best' candidate now uses a configurable metric (resemblance_metric) rather
        than always using SSE. Default selection prefers matches to lower actual values by using
        an asymmetric absolute error (over-predictions penalized more).
    """
    # default transforms/p grid (added 'logn' to allow variable log base)
    if transforms is None:
        transforms = ["ln", "log2", "log10", "logn", "identity"]
    # build p_grid from tunables
    try:
        p_grid = np.linspace(float(p_grid_min), float(p_grid_max), int(p_grid_steps))
    except Exception:
        p_grid = np.linspace(0.2, 3.0, 15)

    # prepare a grid for log-base N when transform == 'logn'
    try:
        N_grid = np.unique(np.concatenate([
            np.array([2.0, 10.0]),
            np.logspace(np.log10(float(logn_base_min)), np.log10(float(logn_base_max)), num=int(logn_base_steps))
        ]))
    except Exception:
        N_grid = np.unique(np.concatenate([
            np.array([2.0, 10.0]),
            np.logspace(np.log10(2.0), np.log10(100.0), num=8)
        ]))

    # shift xarr if includes nonpositive values so that logs won't blow up for log transforms; track shift to apply consistently
    # Note: shifting is still applied to allow some transforms to work; identity transform can accept negatives without shift,
    # but we keep behaviour consistent with earlier code by shifting entire xarr when min <= 0.
    if np.min(xarr) <= 0:
        shift = abs(np.min(xarr)) + 1e-12
        xfit = xarr + shift
    else:
        shift = 0.0
        xfit = xarr.copy()

    xmin = max(np.min(xfit), 1e-12)
    xmax = np.max(xfit)
    x_median = np.median(xfit)

    # b grid (logspace): we search multiplicative scale of b
    b_min = 0.1 / max(xmax, 1e-12)
    b_max = 10.0 / max(xmin, 1e-12)
    b_min = max(b_min, 1e-12)
    b_max = min(b_max, 1e12)
    if b_max <= b_min:
        b_min, b_max = 1e-6, 1e6
    # Respect user-provided b_grid_steps
    try:
        b_grid = np.logspace(np.log10(b_min), np.log10(b_max), num=int(b_grid_steps))
    except Exception:
        b_grid = np.logspace(np.log10(b_min), np.log10(b_max), num=60)

    # c multipliers relative to median(x) used only when c is not fixed
    try:
        c_multipliers = np.linspace(float(c_mult_min), float(c_mult_max), int(c_mult_steps))
    except Exception:
        c_multipliers = np.linspace(-0.9, 10.0, 40)

    # Prepare identity s_grid from user-provided parameters, with safe defaults/guards.
    if identity_s_steps is None or not isinstance(identity_s_steps, int) or identity_s_steps <= 0:
        identity_s_steps = 9
    try:
        s_grid_default = np.logspace(float(identity_s_min), float(identity_s_max), num=int(identity_s_steps))
    except Exception:
        s_grid_default = np.logspace(-1.0, 1.0, num=9)

    best_score = float("inf")
    best_sse_for_best = float("inf")
    best = None

    # main grid search: when c_fixed is provided, do not iterate over c_grid; use provided c_fixed.
    for transform in transforms:
        # For identity transform we do NOT search over b (merge b into a), use b=1.0 fixed.
        # But we still want to emulate the effect b had on c placement, so create a small s_grid
        # (multiplicative scales applied to c candidates) for identity.
        local_b_grid = [1.0] if transform == "identity" else b_grid
        s_grid = s_grid_default if transform == "identity" else [1.0]

        for b in local_b_grid:
            # compute c grid relative to b * median(x) only if c is not fixed
            for s in s_grid:
                if c_fixed is None:
                    if transform == "identity":
                        # emulate varying b's effect on c placement by scaling the c_multipliers with s
                        c_grid = x_median * c_multipliers * float(s)
                    else:
                        bx_med = b * x_median
                        c_grid = bx_med * c_multipliers
                else:
                    c_grid = [float(c_fixed)]

                for c in c_grid:
                    # compute 'vals' for transform. IMPORTANT: use b * (xfit - c) per requirement.
                    vals = b * (xfit - c)

                    # For log-like transforms, we still require vals > 0. For identity we accept any vals.
                    # if transform in {"ln", "log10", "log2", "logn"}:
                    #     # skip combos where any vals are nonpositive (cannot take log)
                    #     if np.any(vals <= 0):
                    #         continue

                    # Now handle transform-specific processing.
                    if transform == "logn":
                        # iterate over possible log bases
                        for Nbase in N_grid:
                            # apply transform safely (suppress warnings during log)
                            with np.errstate(divide="ignore", invalid="ignore"):
                                Tvals = apply_transform(vals, "logn", log_base=float(Nbase))
                            # skip cases where Tvals contains zeros (would blow up when raising to -p)
                            if np.any(Tvals == 0):
                                continue
                            for p in p_grid:
                                # compute S = Tvals ** (-p) while suppressing invalid warnings (e.g. negative bases)
                                with np.errstate(invalid="ignore", divide="ignore"):
                                    S = Tvals ** (-p)
                                if d_fixed is None:
                                    # solve for d and a via linear least squares: y = d + a * S
                                    G = np.vstack([np.ones_like(S), S]).T
                                    try:
                                        coeffs, *_ = np.linalg.lstsq(G, yarr, rcond=None)
                                    except Exception:
                                        continue
                                    d_est, a_est = float(coeffs[0]), float(coeffs[1])
                                    pred = G.dot(coeffs)
                                else:
                                    # solve for a in closed form: minimize ||a*S - (y - d)||^2
                                    numer = np.sum((yarr - d_fixed) * S)
                                    denom = np.sum(S * S)
                                    if denom == 0:
                                        continue
                                    a_est = numer / denom
                                    d_est = float(d_fixed)
                                    pred = d_est + a_est * S

                                # compute resemblance score (lower is better)
                                score = _compute_resemblance_score(yarr, pred, metric=resemblance_metric,
                                                                 penalize_above=penalize_above,
                                                                 penalize_below=penalize_below)
                                sse = float(np.sum((yarr - pred) ** 2))
                                if score < best_score:
                                    best_score = score
                                    best_sse_for_best = sse
                                    best = {"a": float(a_est), "b": float(b), "c": float(c), "d": float(d_est),
                                            "p": float(p), "transform": "logn", "log_base": float(Nbase),
                                            "sse": sse, "score": score, "shift": shift}

                    else:
                        # other transforms: ln, log10, log2, identity
                        with np.errstate(divide="ignore", invalid="ignore"):
                            Tvals = apply_transform(vals, transform)
                        if transform in {"identity"}:
                            if not np.all(np.isfinite(Tvals)):
                                continue
                        if np.any(Tvals == 0):
                            continue

                        for p in p_grid:
                            # compute S while suppressing invalid-value warnings (user noted harmless warning)
                            with np.errstate(invalid="ignore", divide="ignore"):
                                S = Tvals ** (-p)
                            if transform in {"identity"}:
                                if not np.all(np.isfinite(S)):
                                    continue
                            if d_fixed is None:
                                # solve for d and a via linear least squares: y = d + a * S
                                G = np.vstack([np.ones_like(S), S]).T
                                try:
                                    coeffs, *_ = np.linalg.lstsq(G, yarr, rcond=None)
                                except Exception:
                                    continue
                                d_est, a_est = float(coeffs[0]), float(coeffs[1])
                                pred = G.dot(coeffs)
                            else:
                                # solve for a in closed form: minimize ||a*S - (y - d)||^2
                                numer = np.sum((yarr - d_fixed) * S)
                                denom = np.sum(S * S)
                                if denom == 0:
                                    continue
                                a_est = numer / denom
                                d_est = float(d_fixed)
                                pred = d_est + a_est * S

                            # compute resemblance score (lower is better)
                            score = _compute_resemblance_score(yarr, pred, metric=resemblance_metric,
                                                             penalize_above=penalize_above,
                                                             penalize_below=penalize_below)
                            sse = float(np.sum((yarr - pred) ** 2))
                            if score < best_score:
                                best_score = score
                                best_sse_for_best = sse
                                best = {"a": float(a_est), "b": float(b), "c": float(c), "d": float(d_est),
                                        "p": float(p), "transform": transform, "sse": sse, "score": score, "shift": shift}

    if best is None:
        return None, {"reason": "no_valid_fit"}

    # refinement around best parameters (refine b, c, p around best found)
    t_best = best["transform"]
    b_center = best["b"]
    c_center = best["c"]
    p_center = best["p"]
    log_base_best = best.get("log_base", None)

    # refine b near center - for identity, keep b fixed to 1.0 (no refinement)
    if t_best == "identity":
        b_refined = np.array([1.0])
    else:
        b_low = max(b_center * 0.6, 1e-12)
        b_high = b_center * 1.6
        try:
            b_refined = np.logspace(math.log10(b_low), math.log10(b_high), num=int(b_refine_steps))
        except Exception:
            b_refined = np.logspace(math.log10(b_low), math.log10(b_high), num=60)

    # refine c near center - when c_fixed was provided originally, we still refine around it
    if abs(c_center) > 1e-12:
        # for identity widen the refinement range a bit to capture placements that earlier b grid emulated
        if t_best == "identity":
            c_low = c_center * 0.5
            c_high = c_center * 1.5
        else:
            c_low = c_center * 0.7
            c_high = c_center * 1.3
    else:
        # if c_center == 0 use a range relative to b_center * x_median
        if t_best == "identity":
            c_low = -0.6 * x_median
            c_high = 2.0 * x_median
        else:
            c_low = -0.3 * b_center * x_median
            c_high = 1.0 * b_center * x_median
    try:
        c_refined = np.linspace(c_low, c_high, num=int(c_refine_steps))
    except Exception:
        c_refined = np.linspace(c_low, c_high, num=60)

    # refine p near center
    p_low = max(0.05, p_center * 0.6)
    p_high = p_center * 1.6
    try:
        p_refined = np.linspace(p_low, p_high, num=int(p_refine_steps))
    except Exception:
        p_refined = np.linspace(p_low, p_high, num=20)

    # If the chosen transform was logn, also prepare a refined N grid centered on log_base_best
    if t_best == "logn" and log_base_best is not None:
        # small multiplicative sweep around chosen base
        N_low = max(1.1, log_base_best * 0.6)
        N_high = log_base_best * 1.6
        try:
            N_refined = np.unique(np.logspace(math.log10(N_low), math.log10(max(N_high, N_low * 1.001)), num=int(N_refine_steps)))
        except Exception:
            N_refined = np.unique(np.logspace(math.log10(N_low), math.log10(max(N_high, N_low * 1.001)), num=12))
    else:
        N_refined = None

    best_score2 = float("inf")
    best2 = None
    best_sse_for_best2 = float("inf")

    for b in b_refined:
        for c in c_refined:
            vals = b * (xfit - c)
            # same positivity requirement for log-like transforms
            if t_best in {"ln", "log10", "log2", "logn"}:
                if np.any(vals <= 0):
                    continue

            if t_best == "logn":
                # iterate over refined N if available, else try original small grid
                bases_to_try = N_refined if N_refined is not None else N_grid
                for Nbase in bases_to_try:
                    with np.errstate(divide="ignore", invalid="ignore"):
                        Tvals = apply_transform(vals, "logn", log_base=float(Nbase))
                    if not np.all(np.isfinite(Tvals)):
                        continue
                    if np.any(Tvals == 0):
                        continue
                    for p in p_refined:
                        with np.errstate(invalid="ignore", divide="ignore"):
                            S = Tvals ** (-p)
                        if not np.all(np.isfinite(S)):
                            continue
                        if d_fixed is None:
                            G = np.vstack([np.ones_like(S), S]).T
                            try:
                                coeffs, *_ = np.linalg.lstsq(G, yarr, rcond=None)
                            except Exception:
                                continue
                            d_est, a_est = float(coeffs[0]), float(coeffs[1])
                            pred = G.dot(coeffs)
                        else:
                            numer = np.sum((yarr - d_fixed) * S)
                            denom = np.sum(S * S)
                            if denom == 0:
                                continue
                            a_est = numer / denom
                            d_est = float(d_fixed)
                            pred = d_est + a_est * S
                        score = _compute_resemblance_score(yarr, pred, metric=resemblance_metric,
                                                         penalize_above=penalize_above, penalize_below=penalize_below)
                        sse = float(np.sum((yarr - pred) ** 2))
                        if score < best_score2:
                            best_score2 = score
                            best_sse_for_best2 = sse
                            best2 = {"a": float(a_est), "b": float(b), "c": float(c), "d": float(d_est),
                                     "p": float(p), "transform": "logn", "log_base": float(Nbase),
                                     "sse": sse, "score": score, "shift": shift}

            else:
                with np.errstate(divide="ignore", invalid="ignore"):
                    Tvals = apply_transform(vals, t_best)
                if not np.all(np.isfinite(Tvals)):
                    continue
                if np.any(Tvals == 0):
                    continue
                for p in p_refined:
                    with np.errstate(invalid="ignore", divide="ignore"):
                        S = Tvals ** (-p)
                    if not np.all(np.isfinite(S)):
                        continue
                    if d_fixed is None:
                        G = np.vstack([np.ones_like(S), S]).T
                        try:
                            coeffs, *_ = np.linalg.lstsq(G, yarr, rcond=None)
                        except Exception:
                            continue
                        d_est, a_est = float(coeffs[0]), float(coeffs[1])
                        pred = G.dot(coeffs)
                    else:
                        numer = np.sum((yarr - d_fixed) * S)
                        denom = np.sum(S * S)
                        if denom == 0:
                            continue
                        a_est = numer / denom
                        d_est = float(d_fixed)
                        pred = d_est + a_est * S
                    score = _compute_resemblance_score(yarr, pred, metric=resemblance_metric,
                                                     penalize_above=penalize_above, penalize_below=penalize_below)
                    sse = float(np.sum((yarr - pred) ** 2))
                    if score < best_score2:
                        best_score2 = score
                        best_sse_for_best2 = sse
                        best2 = {"a": float(a_est), "b": float(b), "c": float(c), "d": float(d_est),
                                 "p": float(p), "transform": t_best, "sse": sse, "score": score, "shift": shift}

    final = best2 if best2 is not None else best

    # compute final R^2
    shift = final.get("shift", 0.0)
    if shift != 0.0:
        xfit_final = xarr + shift
    else:
        xfit_final = xarr
    vals = final["b"] * (xfit_final - final["c"])

    # apply final transform and compute S_final safely, suppressing invalid-value warnings as harmless.
    if final["transform"] == "logn":
        log_base = final.get("log_base", 10.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            Tvals_final = apply_transform(vals, "logn", log_base=float(log_base))
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            Tvals_final = apply_transform(vals, final["transform"])

    with np.errstate(invalid="ignore", divide="ignore"):
        S_final = Tvals_final ** (-final["p"])
    pred = final["d"] + final["a"] * S_final
    # remove any non-finite predictions from residual calc by masking
    finite_mask = np.isfinite(pred) & np.isfinite(yarr)
    if not np.any(finite_mask):
        return None, {"reason": "no_finite_predictions"}
    ss_res = np.sum((yarr[finite_mask] - pred[finite_mask]) ** 2)
    ss_tot = np.sum((yarr[finite_mask] - np.mean(yarr[finite_mask])) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot != 0 else float("nan")
    final["r2"] = float(r2)
    final["sse"] = float(np.sum((yarr[finite_mask] - pred[finite_mask]) ** 2))
    # ensure 'score' is present in final for downstream selection across processes
    if "score" not in final:
        final["score"] = _compute_resemblance_score(yarr[finite_mask], pred[finite_mask],
                                                  metric=resemblance_metric,
                                                  penalize_above=penalize_above,
                                                  penalize_below=penalize_below)
    return final, {"reason": "ok", "r2": final["r2"], "sse": final["sse"], "score": float(final["score"])}


# --------------------------
# outlier utilities
# --------------------------

def modified_z_score(values: np.ndarray) -> np.ndarray:
    """
    Compute the modified z-score (based on median absolute deviation).
    Returns absolute modified z-scores for each value.
    """
    # Use float conversion
    vals = np.asarray(values, dtype=float)
    med = np.median(vals)
    mad = np.median(np.abs(vals - med))
    if mad == 0:
        # fallback to standard z-score if MAD is zero
        mean = np.mean(vals)
        std = np.std(vals)
        if std == 0:
            return np.zeros_like(vals)
        return np.abs((vals - mean) / std)
    # constant 0.6745 makes modified z comparable to Z for normal dist.
    mod_z = 0.6745 * (vals - med) / mad
    return np.abs(mod_z)


# --------------------------
# Helper: predict & r2 computation from params (used by reconsideration loop)
# --------------------------

def _predict_from_params(params: Dict[str, Any], xarr: np.ndarray) -> np.ndarray:
    """
    Given params dict (from fit_model_general) and xarr (bpw values),
    compute predicted y values according to the same formula used in fitter.
    Returns an array of same length as xarr with finite or non-finite entries where appropriate.
    """
    if params is None:
        return np.full_like(xarr, np.nan, dtype=float)
    # copy to float arrays
    x = np.asarray(xarr, dtype=float)
    shift = float(params.get("shift", 0.0))
    if shift != 0.0:
        x = x + shift
    b = float(params.get("b", 1.0))
    c = float(params.get("c", 0.0))
    p = float(params.get("p", 1.0))
    a = float(params.get("a", 0.0))
    d = float(params.get("d", 0.0))
    transform = params.get("transform", "identity")
    log_base = params.get("log_base", None)

    vals = b * (x - c)
    try:
        if transform == "logn":
            base = float(log_base) if log_base is not None else 10.0
            with np.errstate(divide="ignore", invalid="ignore"):
                T = apply_transform(vals, "logn", log_base=base)
        else:
            with np.errstate(divide="ignore", invalid="ignore"):
                T = apply_transform(vals, transform)
    except Exception:
        # return all NaNs on failure
        return np.full_like(x, np.nan, dtype=float)

    with np.errstate(invalid="ignore", divide="ignore"):
        S = T ** (-p)
    ypred = np.full_like(x, np.nan, dtype=float)
    mask = np.isfinite(T) & (T != 0.0) & np.isfinite(S)
    ypred[mask] = d + a * S[mask]
    return ypred


def _compute_r2_from_pred_and_true(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Compute R^2 between y_pred and y_true. Returns NaN when not defined.
    This uses the same R^2 formula used elsewhere in the script.
    """
    yp = np.asarray(y_pred, dtype=float)
    yt = np.asarray(y_true, dtype=float)
    mask = np.isfinite(yp) & np.isfinite(yt)
    if not np.any(mask):
        return float("nan")
    ss_res = np.sum((yt[mask] - yp[mask]) ** 2)
    ss_tot = np.sum((yt[mask] - np.mean(yt[mask])) ** 2)
    if ss_tot == 0:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


# --------------------------
# plotting & CSV integration (bpw is X, user column is Y)
# --------------------------

# Top-level worker for process pool. It must be picklable (i.e., module-level) for ProcessPoolExecutor.
def _fit_pair_process_worker(xarr_shared, yarr_shared,
                             d_cand_local: Optional[float],
                             c_cand_local: Optional[float],
                             transforms_local: Optional[List[str]] = None,
                             identity_s_min_local: float = -1.0,
                             identity_s_max_local: float = 1.0,
                             identity_s_steps_local: int = 9,
                             p_grid_min_local: float = 0.2,
                             p_grid_max_local: float = 3.0,
                             p_grid_steps_local: int = 15,
                             logn_base_min_local: float = 2.0,
                             logn_base_max_local: float = 100.0,
                             logn_base_steps_local: int = 8,
                             c_mult_min_local: float = -0.9,
                             c_mult_max_local: float = 10.0,
                             c_mult_steps_local: int = 40,
                             b_grid_steps_local: int = 60,
                             b_refine_steps_local: int = 60,
                             c_refine_steps_local: int = 60,
                             p_refine_steps_local: int = 20,
                             N_refine_steps_local: int = 12,
                             resemblance_metric_local: str = "asym_abs",
                             penalize_above_local: float = 2.0,
                             penalize_below_local: float = 1.0):
    """
    Worker run in separate process. xarr_shared, yarr_shared are numpy arrays (pickled to worker).
    Limit BLAS threads inside worker to avoid oversubscription.
    """
    # Limit BLAS / native threaded libs inside each worker to 1 to avoid contention.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    try:
        # Convert to numpy arrays in worker context (they may already be np arrays after unpickling)
        x_local = np.asarray(xarr_shared, dtype=float)
        y_local = np.asarray(yarr_shared, dtype=float)
        params_local, details_local = fit_model_general(
            x_local, y_local,
            d_fixed=d_cand_local,
            c_fixed=c_cand_local,
            transforms=transforms_local,
            p_grid_min=p_grid_min_local,
            p_grid_max=p_grid_max_local,
            p_grid_steps=p_grid_steps_local,
            logn_base_min=logn_base_min_local,
            logn_base_max=logn_base_max_local,
            logn_base_steps=logn_base_steps_local,
            c_mult_min=c_mult_min_local,
            c_mult_max=c_mult_max_local,
            c_mult_steps=c_mult_steps_local,
            b_grid_steps=b_grid_steps_local,
            b_refine_steps=b_refine_steps_local,
            c_refine_steps=c_refine_steps_local,
            p_refine_steps=p_refine_steps_local,
            N_refine_steps=N_refine_steps_local,
            identity_s_min=identity_s_min_local,
            identity_s_max=identity_s_max_local,
            identity_s_steps=identity_s_steps_local,
            resemblance_metric=resemblance_metric_local,
            penalize_above=penalize_above_local,
            penalize_below=penalize_below_local
        )
        return (d_cand_local, c_cand_local, params_local, details_local, None)
    except Exception as e:
        return (d_cand_local, c_cand_local, None, None, str(e))


def compute_and_plot_from_csv(csv_path: str,
                              bpw_column: str = "bpw",
                              ycol_identifier: Optional[str] = None,
                              hide_empty: bool = False,
                              plot_output: Optional[str] = None,
                              fit_equation: bool = True,
                              d_from_lowest_k: Optional[int] = None,
                              d_free: bool = False,
                              c_from_lowest_k: Optional[int] = None,
                              c_free: bool = False,
                              ignore_outliers_threshold: float = 30.0,
                              threads: Optional[int] = None,
                              metric_name: Optional[str] = "metric",
                              predict_bpw_values: Optional[List[float]] = None,
                              transforms: Optional[List[str]] = None,
                              suppress_plot: bool = False,
                              equation_only: bool = False,
                              identity_s_min: float = -1.0,
                              identity_s_max: float = 1.0,
                              identity_s_steps: int = 9,
                              p_grid_min: float = 0.2,
                              p_grid_max: float = 3.0,
                              p_grid_steps: int = 15,
                              logn_base_min: float = 2.0,
                              logn_base_max: float = 100.0,
                              logn_base_steps: int = 8,
                              c_mult_min: float = -0.9,
                              c_mult_max: float = 10.0,
                              c_mult_steps: int = 40,
                              b_grid_steps: int = 60,
                              b_refine_steps: int = 60,
                              c_refine_steps: int = 60,
                              p_refine_steps: int = 20,
                              N_refine_steps: int = 12,
                              resemblance_metric: str = "asym_abs",
                              penalize_above: float = 2.0,
                              penalize_below: float = 1.0,
                              drift_below: float = 0.0,
                              drift_above: float = 0.0) -> Optional[List[float]]:
    """
    Read the produced bpw CSV (bpw_<input>.csv), plot metric (y) vs bpw (x).
    ycol_identifier can be a column name or integer index (defaults to index 2).
    """
    # Flag indicating machine-friendly output mode (predict-only)
    machine_mode = bool(predict_bpw_values and len(predict_bpw_values) > 0)

    with open(csv_path, newline="", encoding="utf-8") as inf:
        reader = csv.DictReader(inf)
        rows = list(reader)
        headers = list(reader.fieldnames or [])

    # find qtype column to determine default positions but we use bpw directly
    qcol = None
    qcol_index = None
    for i, h in enumerate(headers):
        if h.lower() == "qtype":
            qcol = h
            qcol_index = i
            break
    if qcol is None:
        raise ValueError("CSV must have a 'qtype' column to determine default y column if not provided.")

    # determine y column name (user metric)
    if ycol_identifier is None:
        # default: column index 2 (per your request)
        yidx = 2
        if yidx < 0 or yidx >= len(headers):
            raise ValueError("default ycol index 2 is out of range; please specify --ycol.")
        ycol = headers[yidx]
    else:
        try:
            yidx = int(ycol_identifier)
            if yidx < 0 or yidx >= len(headers):
                raise ValueError("ycol index out of range.")
            ycol = headers[yidx]
        except ValueError:
            if ycol_identifier in headers:
                ycol = ycol_identifier
            else:
                matches = [h for h in headers if h.lower() == ycol_identifier.lower()]
                if matches:
                    ycol = matches[0]
                else:
                    raise ValueError(f"ycol '{ycol_identifier}' not found in CSV headers.")

    # extract arrays: bpw as xarr, chosen column as yarr
    xs = []
    ys = []
    valid_rows = []
    for r in rows:
        bpb = r.get(bpw_column)
        bmetric = r.get(ycol)
        if bpb is None or bmetric is None:
            continue
        if bpb == "" or bmetric == "":
            continue
        if "404" in str(bpb) or "404" in str(bmetric):
            continue
        try:
            xv = float(str(bpb).strip().replace(",", ""))
            yv = float(str(bmetric).strip().replace(",", ""))
        except Exception:
            continue
        xs.append(xv)
        ys.append(yv)
        valid_rows.append(r)

    if len(xs) < 4:
        raise ValueError("Not enough valid rows to plot/fit. Need at least 4 valid numeric rows.")

    # Keep original arrays for plotting; we'll create filtered versions for fitting if requested.
    xarr_full = np.array(xs, dtype=float)  # bpw -> X axis (all points)
    yarr_full = np.array(ys, dtype=float)  # metric -> Y axis (all points)

    # Determine outliers (if requested) using modified z-score on both x and y.
    # A point is considered an outlier if it's extreme in x OR y.
    filtered_mask = np.ones_like(xarr_full, dtype=bool)
    if ignore_outliers_threshold and ignore_outliers_threshold > 0.0:
        mz_x = modified_z_score(xarr_full)
        mz_y = modified_z_score(yarr_full)
        outlier_mask = (mz_x > ignore_outliers_threshold) | (mz_y > ignore_outliers_threshold)
        num_outliers = int(np.sum(outlier_mask))
        if num_outliers > 0:
            print(f"[info] Ignoring {num_outliers} outlier(s) (threshold={ignore_outliers_threshold}) for fitting.", file=sys.stderr)
            filtered_mask = ~outlier_mask
        else:
            print(f"[info] No outliers detected at threshold {ignore_outliers_threshold}.", file=sys.stderr)

    # prepare arrays used for fitting (after outlier exclusion)
    xarr = xarr_full[filtered_mask]
    yarr = yarr_full[filtered_mask]
    if len(xarr) < 4:
        raise ValueError("Not enough valid rows remain after outlier removal to plot/fit. Need at least 4 valid numeric rows.")

    # The scatter plot should show all points (including outliers) so user can inspect.
    # Only create plotting objects if plotting is not suppressed.
    if not suppress_plot:
        fig, ax = plt.subplots()

        ax.scatter(xarr_full, yarr_full, label="data")

        # construct metric display label: e.g. "accuracy (column_name)"
        metric_display = f"{metric_name} ({ycol})"
        ax.set_xlabel("bpw")
        ax.set_ylabel(f"{metric_display}")
        ax.set_title(f"{metric_display} vs bpw")
    else:
        # If plotting is suppressed, still build metric_display for later labels/printing.
        metric_display = f"{metric_name} ({ycol})"

    if not fit_equation:
        if not suppress_plot:
            fig.tight_layout()
            if plot_output:
                plt.savefig(plot_output, dpi=300)
                print(f"Wrote plot to {plot_output}", file=sys.stderr)
            else:
                try:
                    if not machine_mode:
                        plt.show()
                    else:
                        fallback = "bpw_plot.png"
                        plt.savefig(fallback, dpi=300)
                        print(f"Interactive display not available; saved plot to {fallback}", file=sys.stderr)
                except Exception:
                    fallback = "bpw_plot.png"
                    plt.savefig(fallback, dpi=300)
                    print(f"Interactive display not available; saved plot to {fallback}", file=sys.stderr)
        else:
            # plotting suppressed: do not create/save/show any figure; in machine_mode return empty list or predictions later
            if machine_mode:
                print("[]")
                return []
            return None

    # Determine d candidates (anchoring/inference) BEFORE fitting other params.
    # Use filtered (outlier-excluded) arrays for candidate generation, as requested.
    d_candidates: List[Optional[float]] = []
    if d_free:
        # d is free: indicate by using None (fit will treat d_fixed=None)
        d_candidates = [None]
    else:
        # user wants to anchor d to mean of K lowest y values OR try all K from 1..N
        if d_from_lowest_k is not None:
            K = max(1, int(d_from_lowest_k))
            order = np.argsort(yarr)  # sort by Y (metric), take lowest K
            k_used = min(K, len(order))
            indices = order[:k_used]
            d_val = float(np.mean(yarr[indices]))
            d_candidates = [d_val]
            print(f"[info] Anchoring d to mean of {k_used} lowest y values (after outlier removal) -> d = {d_val:.12g}", file=sys.stderr)
        else:
            # neither d_free nor d_from_lowest provided -> generate candidate list using K=1..N
            order = np.argsort(yarr)
            d_candidates = []
            for K in range(1, len(yarr) + 1):
                indices = order[:K]
                d_candidates.append(float(np.mean(yarr[indices])))
            # keep unique candidates while preserving order
            seen = set()
            d_candidates = [x for x in d_candidates if not (x in seen or seen.add(x))]

    # Determine c candidates (anchoring/inference) BEFORE fitting other params.
    # Use filtered arrays for computing c candidates as well.
    c_candidates: List[Optional[float]] = []
    if c_free:
        c_candidates = [None]
    else:
        if c_from_lowest_k is not None:
            K = max(1, int(c_from_lowest_k))
            orderx = np.argsort(xarr)  # sort by X (bpw), take lowest K
            k_used = min(K, len(orderx))
            indices = orderx[:k_used]
            c_val = float(np.mean(xarr[indices]))
            c_candidates = [c_val]
            print(f"[info] Anchoring c to mean of {k_used} lowest x values (after outlier removal) -> c = {c_val:.12g}", file=sys.stderr)
        else:
            # neither c_free nor c_from_lowest provided -> try all K=1..N
            orderx = np.argsort(xarr)
            c_candidates = []
            for K in range(1, len(xarr) + 1):
                indices = orderx[:K]
                c_candidates.append(float(np.mean(xarr[indices])))
            seen = set()
            c_candidates = [x for x in c_candidates if not (x in seen or seen.add(x))]

    # iterate over d/c candidate pairs and pick best fit (on outlier-excluded data)
    best_overall = None
    best_details = None
    best_params = None

    # --- Begin parallel region: evaluate fit_model_general for each (d_cand,c_cand) pair in parallel ---
    pairs = [(d_cand, c_cand) for d_cand in d_candidates for c_cand in c_candidates]

    if len(pairs) == 0:
        best_params = None
        best_details = None
    else:
        # Determine number of worker processes to use
        if threads is None or (isinstance(threads, int) and threads <= 0):
            max_workers = multiprocessing.cpu_count()
        else:
            max_workers = int(threads)

        # cap to number of tasks
        max_workers = max(1, min(max_workers, len(pairs)))

        # To avoid BLAS-based contention, we also set these env vars in the main process.
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

        # submit tasks to process pool. Each worker gets (xarr, yarr, d_cand, c_cand).
        # Note: passing numpy arrays to processes implies pickling; it's acceptable for moderate-sized arrays.
        tasks = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_pair = {}
            for d_c, c_c in pairs:
                # pass transforms along to worker so users can override which transforms are tried
                fut = executor.submit(
                    _fit_pair_process_worker,
                    xarr, yarr, d_c, c_c,
                    transforms,
                    float(identity_s_min), float(identity_s_max), int(identity_s_steps),
                    float(p_grid_min), float(p_grid_max), int(p_grid_steps),
                    float(logn_base_min), float(logn_base_max), int(logn_base_steps),
                    float(c_mult_min), float(c_mult_max), int(c_mult_steps),
                    int(b_grid_steps),
                    int(b_refine_steps), int(c_refine_steps), int(p_refine_steps), int(N_refine_steps),
                    str(resemblance_metric), float(penalize_above), float(penalize_below)
                )
                future_to_pair[fut] = (d_c, c_c)
            results = []
            for fut in as_completed(future_to_pair):
                d_cand_local, c_cand_local = future_to_pair[fut]
                try:
                    res = fut.result()
                except Exception as exc:
                    # In case the executor raised (should be rare because worker returns exceptions as strings)
                    results.append((d_cand_local, c_cand_local, None, None, f"executor-exception: {exc}"))
                else:
                    results.append(res)

        # Evaluate results to pick best by the resemblance 'score' (lower is better)
        for (d_cand_local, c_cand_local, params_local, details_local, exc_str) in results:
            if exc_str:
                # Report worker exception to stderr but continue scanning other results.
                print(f"[worker-error] d={d_cand_local} c={c_cand_local} -> {exc_str}", file=sys.stderr)
                continue
            if params_local is None or details_local is None:
                continue
            # details_local should include 'score' per fitter return
            score = details_local.get("score", params_local.get("score", float("inf")))
            if best_overall is None or score < best_overall:
                best_overall = score
                best_params = params_local
                best_details = details_local
                selected_d = d_cand_local
                selected_c = c_cand_local
    # --- End parallel region ---

    if best_params is None:
        # If machine_mode and predictions were requested, return empty array to stdout for easy piping.
        if machine_mode:
            # If equation_only is requested but no fit, still print empty JSON to stdout (consistent with previous machine behavior)
            print("[]")
            return []
        print("Warning: no valid fit found among transforms; plotting scatter only.", file=sys.stderr)
        if not suppress_plot:
            fig.tight_layout()
            if plot_output:
                plt.savefig(plot_output, dpi=300)
                print(f"Wrote plot to {plot_output}", file=sys.stderr)
            else:
                try:
                    if not machine_mode:
                        plt.show()
                    else:
                        fallback = "bpw_plot.png"
                        plt.savefig(fallback, dpi=300)
                        print(f"Interactive display not available; saved plot to {fallback}", file=sys.stderr)
                except Exception:
                    fallback = "bpw_plot.png"
                    plt.savefig(fallback, dpi=300)
                    print(f"Interactive display not available; saved plot to {fallback}", file=sys.stderr)
        return None

    # Selected parameters
    params = best_params
    details = best_details

    a = params["a"]
    b = params["b"]
    c = params["c"]
    d = params["d"]
    p = params["p"]
    transform = params["transform"]
    r2 = params.get("r2", float("nan"))
    #print("r2_baseline:", r2)
    log_base = params.get("log_base", None)
    selected_score = params.get("score", details.get("score", float("nan")))

    # --------------------------
    # NEW: Reconsideration loop based on explicit user-provided --drift-below/--drift-above
    # --------------------------
    # This loop triggers only when the user explicitly passed the flags on the command line,
    # not when default values are used. We detect presence via sys.argv.
    user_passed_drift_below = any(arg.startswith("--drift-below") for arg in sys.argv[1:])
    user_passed_drift_above = any(arg.startswith("--drift-above") for arg in sys.argv[1:])
    if user_passed_drift_below or user_passed_drift_above:
        # Interpret drift flags as percentages of the baseline R^2 (drift threshold).
        # Example: baseline R^2 = 0.95 and --drift-below 5 means allowed lower bound is 0.95 * (1 - 0.05) = 0.9025.
        pa_percent = float(drift_below) if user_passed_drift_below else 0.0
        pb_percent = float(drift_above) if user_passed_drift_above else 0.0
        # Use the more permissive percentage (largest percent) when both provided
        allowed_percent = max(pa_percent, pb_percent)

        # Baseline: compute R^2 of the selected equation vs the **initial fitting set** (the points present before the loop started).
        # NOTE: per your request, all subsequent R^2 calculations are evaluated against this same initial set (xarr, yarr).
        baseline_pred = _predict_from_params(params, xarr)
        r2_baseline = _compute_r2_from_pred_and_true(baseline_pred, yarr)
        if not np.isfinite(r2_baseline):
            # fallback to stored r2 if available
            if np.isfinite(r2):
                r2_baseline = float(r2)
            else:
                r2_baseline = float("nan")

        if not np.isfinite(r2_baseline):
            print("[reconsider] Baseline R^2 is non-finite; skipping reconsideration loop.", file=sys.stderr)
        else:
            # allowed lower bound
            allowed_lower_r2 = r2_baseline * (1.0 - allowed_percent / 100.0)

            # Prepare working indices (indices into the initial fitting set) that were used for fitting.
            # We'll operate in terms of indices into the initial-fit arrays (xarr_full indices filtered_mask True).
            init_indices = np.nonzero(filtered_mask)[0].tolist()  # global indices in full arrays that were used initially
            working_indices = init_indices.copy()
            if len(working_indices) < 4:
                print("[reconsider] Too few points in the fitting set to perform reconsideration loop.", file=sys.stderr)
            else:
                last_valid_params = params
                last_valid_r2 = r2_baseline
                prev_params = params

                removed_global_indices: List[int] = []
                loop_iter = 0
                while True:
                    loop_iter += 1
                    # Build arrays for current working set (these are subsets of the initial-fit points)
                    x_work_global = np.array([xarr_full[i] for i in working_indices], dtype=float)
                    y_work_global = np.array([yarr_full[i] for i in working_indices], dtype=float)

                    # compute residuals on the current working set using prev_params
                    pred_on_work = _predict_from_params(prev_params, x_work_global)
                    resid_work = y_work_global - pred_on_work  # positive => actual above predicted

                    # choose candidate to remove based on user's chosen polarity
                    candidate_pos = None
                    if user_passed_drift_below and not user_passed_drift_above:
                        positive_mask = np.isfinite(resid_work) & (resid_work > 0)
                        if not np.any(positive_mask):
                            # nothing above; stop
                            break
                        # choose largest positive resid
                        idx_in_work = int(np.nanargmax(np.where(positive_mask, resid_work, -np.inf)))
                        candidate_pos = idx_in_work
                    elif user_passed_drift_above and not user_passed_drift_below:
                        negative_mask = np.isfinite(resid_work) & (resid_work < 0)
                        if not np.any(negative_mask):
                            # nothing below; stop
                            break
                        idx_in_work = int(np.nanargmin(np.where(negative_mask, resid_work, np.inf)))
                        candidate_pos = idx_in_work
                    else:
                        # both provided: remove the point with largest absolute residual
                        finite_mask = np.isfinite(resid_work)
                        if not np.any(finite_mask):
                            break
                        idx_in_work = int(np.nanargmax(np.abs(np.where(finite_mask, resid_work, 0.0))))
                        candidate_pos = idx_in_work

                    # global index to remove (index into xarr_full / yarr_full)
                    global_idx = working_indices.pop(candidate_pos)
                    removed_global_indices.append(global_idx)

                    # Build subset arrays for refitting (these are still subsets of initial-fit points)
                    x_subset = np.array([xarr_full[i] for i in working_indices], dtype=float)
                    y_subset = np.array([yarr_full[i] for i in working_indices], dtype=float)

                    # If too few remain, restore last removal and stop
                    if len(x_subset) < 4:
                        print("[reconsider] Stopping removal: too few points remain for refit.", file=sys.stderr)
                        # restore last removed point to working_indices
                        working_indices.insert(candidate_pos, global_idx)
                        removed_global_indices.pop()
                        break

                    # Refit on the reduced set. Keep same d/c anchoring that was selected earlier.
                    try:
                        refit_params, refit_details = fit_model_general(
                            x_subset, y_subset,
                            d_fixed=selected_d,
                            c_fixed=selected_c,
                            transforms=transforms,
                            p_grid_min=p_grid_min,
                            p_grid_max=p_grid_max,
                            p_grid_steps=p_grid_steps,
                            logn_base_min=logn_base_min,
                            logn_base_max=logn_base_max,
                            logn_base_steps=logn_base_steps,
                            c_mult_min=c_mult_min,
                            c_mult_max=c_mult_max,
                            c_mult_steps=c_mult_steps,
                            b_grid_steps=b_grid_steps,
                            b_refine_steps=b_refine_steps,
                            c_refine_steps=c_refine_steps,
                            p_refine_steps=p_refine_steps,
                            N_refine_steps=N_refine_steps,
                            identity_s_min=identity_s_min,
                            identity_s_max=identity_s_max,
                            identity_s_steps=identity_s_steps,
                            resemblance_metric=resemblance_metric,
                            penalize_above=penalize_above,
                            penalize_below=penalize_below
                        )
                    except Exception as ex:
                        # Refit failed: restore index and stop.
                        print(f"[reconsider] refit failed after removing index {global_idx}: {ex}", file=sys.stderr)
                        working_indices.insert(candidate_pos, global_idx)
                        removed_global_indices.pop()
                        break

                    if refit_params is None:
                        # restore and stop
                        working_indices.insert(candidate_pos, global_idx)
                        removed_global_indices.pop()
                        break

                    # Evaluate refit parameters against the **initial fitting set** (xarr, yarr) - per your requirement.
                    pred_refit_on_initial = _predict_from_params(refit_params, xarr)
                    r2_refit_on_initial = _compute_r2_from_pred_and_true(pred_refit_on_initial, yarr)

                    # If R^2 is non-finite, revert and stop
                    if not np.isfinite(r2_refit_on_initial):
                        print("[reconsider] Non-finite R^2 encountered for refit; stopping reconsideration loop.", file=sys.stderr)
                        working_indices.insert(candidate_pos, global_idx)
                        removed_global_indices.pop()
                        break

                    # Stop condition: if refit R^2 drops below allowed lower bound, revert last removal and stop.
                    if r2_refit_on_initial < allowed_lower_r2:
                        # revert last removal and stop
                        working_indices.insert(candidate_pos, global_idx)
                        removed_global_indices.pop()
                        print(f"[reconsider] Stopping: refit R^2 {r2_refit_on_initial:.6g} < allowed lower bound {allowed_lower_r2:.6g}", file=sys.stderr)
                        break
                    else:
                        # Accept refit and continue to next iteration
                        last_valid_params = refit_params
                        last_valid_r2 = r2_refit_on_initial
                        prev_params = refit_params
                        # continue loop

                # end while loop

                # If we accepted any removals, adopt last_valid_params
                if last_valid_params is not params:
                    print(f"[reconsider] Removed {len(removed_global_indices)} point(s) during reconsideration; using last valid refit.", file=sys.stderr)
                    params = last_valid_params
                    details = {"reason": "reconsidered", "r2_initial": last_valid_r2}
                    # update local variables for downstream plotting/text
                    a = params["a"]
                    b = params["b"]
                    c = params["c"]
                    d = params["d"]
                    p = params["p"]
                    transform = params["transform"]
                    r2 = params.get("r2", float("nan"))
                    log_base = params.get("log_base", None)
                else:
                    print("[reconsider] No removals performed or none accepted within threshold; keeping original fit.", file=sys.stderr)

    # --------------------------
    # End of reconsideration loop
    # --------------------------

    # If machine_mode (predict-only), compute predictions for requested bpw values and print JSON to stdout,
    # and ensure all other printing goes to stderr. However, if equation_only is requested, prefer to print
    # only the equation to stdout (not the JSON predictions).
    if machine_mode and not equation_only:
        preds = []
        for xv in predict_bpw_values:
            try:
                x_val = float(xv)
            except Exception:
                preds.append(None)
                continue
            # For identity transform, b was fixed to 1.0 and 'a' is the merged coefficient for (x-c)^(-p).
            vals_single = params["b"] * (x_val - params["c"])
            # handle transform and compute S
            try:
                with np.errstate(divide="ignore", invalid="ignore"):
                    if params["transform"] == "logn":
                        base_to_use = params.get("log_base", 10.0)
                        T_single = apply_transform(np.array([vals_single], dtype=float), "logn", log_base=float(base_to_use))[0]
                    else:
                        T_single = apply_transform(np.array([vals_single], dtype=float), params["transform"])[0]
                    # If T_single is not finite or zero, prediction is None
                    if not np.isfinite(T_single) or T_single == 0.0:
                        preds.append(None)
                        continue
                    S_single = float(T_single ** (-params["p"]))
                    y_pred = float(params["d"] + params["a"] * S_single)
                    preds.append(y_pred)
            except Exception:
                preds.append(None)
        # Convert None to null in JSON by leaving them as None in Python -> JSON null.
        print(json.dumps(preds))
        return preds

    # If machine_mode and equation_only: print only the equation to stdout (no JSON), then return.
    if machine_mode and equation_only:
        # Build grapher_eq exactly as in non-machine flow below.
        if params["transform"] == "ln":
            grapher_transform = "ln"
        elif params["transform"] == "log10":
            grapher_transform = "log10"
        elif params["transform"] == "log2":
            grapher_transform = "log2"
        elif params["transform"] == "logn":
            grapher_transform = f"log{params.get('log_base'):.6g}" if params.get("log_base") is not None else "logN"
        else:
            grapher_transform = "identity"

        # For 'identity' we emit the simplified merged-a equation: y = d + a * (x - c)^(-p)
        if grapher_transform == "identity":
            grapher_eq = f"y = {params['d']:.12g} + {params['a']:.12g} * ( x - {params['c']:.12g} )^(-{params['p']:.12g})"
        else:
            grapher_eq = f"y = {params['d']:.12g} + {params['a']:.12g} * {grapher_transform}( {params['b']:.12g} * (x - {params['c']:.12g}) )^(-{params['p']:.12g})"
        # normalize adjacent signs
        grapher_eq = collapse_sign_pairs(grapher_eq)
        # Print only the equation line to stdout.
        print(grapher_eq)
        return [grapher_eq]

    # construct fitted curve (x is bpw). Use same formula b * (x - c).
    x_plot = np.linspace(np.min(xarr_full), np.max(xarr_full), 600)
    vals_plot = params["b"] * (x_plot - params["c"])

    # apply transform to plotting vals (safely suppress warnings)
    if params["transform"] == "logn":
        base_to_use = params.get("log_base", 10.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            T_plot = apply_transform(vals_plot, "logn", log_base=float(base_to_use))
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            T_plot = apply_transform(vals_plot, params["transform"])

    with np.errstate(invalid="ignore", divide="ignore"):
        y_plot = np.full_like(x_plot, np.nan, dtype=float)
        S_plot = T_plot ** (-params["p"])
        mask = np.isfinite(T_plot) & (T_plot != 0.0) & np.isfinite(S_plot)
        y_plot[mask] = params["d"] + params["a"] * S_plot[mask]

    # Draw fitted curve (based on outlier-excluded fit) on top of full scatter, only if plotting enabled
    if not suppress_plot:
        ax.plot(x_plot, y_plot, label=f"fit ({params['transform']}, p={params['p']:.3g})")
        # safely fetch possible reconsideration variables (may be None if not set)
        r2_baseline_val = locals().get("r2_baseline", None)
        last_valid_r2_val = locals().get("last_valid_r2", None)

        # build R² display (prefer last_valid_r2 if available, show drift vs baseline when possible)
        if last_valid_r2_val is not None and np.isfinite(last_valid_r2_val) and r2_baseline_val is not None and np.isfinite(r2_baseline_val):
            drift_pct = abs((last_valid_r2_val - r2_baseline_val) / r2_baseline_val) * 100.0
            r2_display = f"R²={last_valid_r2_val:.4f} ({drift_pct:.2f}% drift of baseline {r2_baseline_val:.4f})"
        else:
            r2_val = params.get("r2", float("nan"))
            r2_display = f"R²={r2_val:.4f}"

        # build transform name and equation string for display
        if params["transform"] == "ln":
            grapher_transform = "ln"
        elif params["transform"] == "log10":
            grapher_transform = "log10"
        elif params["transform"] == "log2":
            grapher_transform = "log2"
        elif params["transform"] == "logn":
            grapher_transform = f"log{params.get('log_base', 10.0):.6g}"
        else:
            grapher_transform = "identity"

        if grapher_transform == "identity":
            eq_str = f"y = {params['d']:.12g} + {params['a']:.12g} * ( x - {params['c']:.12g} )^(-{params['p']:.12g})"
        else:
            eq_str = f"y = {params['d']:.12g} + {params['a']:.12g} * {grapher_transform}( {params['b']:.12g} * (x - {params['c']:.12g}) )^(-{params['p']:.12g})"# normalize adjacent signs
        # normalize adjacent signs
        eq_str = collapse_sign_pairs(eq_str)

        # place multi-line text on the plot (transform, p, R² with drift if available, and equation)
        ax.text(
            0.10, 0.98,
            f"{params['transform']}, p={params['p']:.4g}\n{r2_display}\n{eq_str}",
            transform=ax.transAxes, va="top", fontsize=8, wrap=True
        )
        ax.legend()
        fig.tight_layout()

    # stdout prints - param details and grapher-friendly equation. Use x for bpw in equation.
    # These must go to stderr so interactive/piping is easier.
    print("Selected transform:", params["transform"], file=sys.stderr)
    print(f"Selected p: {params['p']:.12g}", file=sys.stderr)
    print("Fitted parameters (fit was computed after excluding outliers if requested):", file=sys.stderr)
    print(f"  d = {params['d']:.12g}", file=sys.stderr)
    print(f"  a = {params['a']:.12g}", file=sys.stderr)
    if params["transform"] != "identity":
        print(f"  b = {params['b']:.12g}", file=sys.stderr)
    print(f"  c = {params['c']:.12g}", file=sys.stderr)
    if 'last_valid_r2' in locals() and np.isfinite(last_valid_r2) and 'r2_baseline' in locals() and np.isfinite(r2_baseline):
        drift_pct = abs((last_valid_r2 - r2_baseline) / r2_baseline) * 100.0
        print(f"  R^2 = {last_valid_r2:.6f} ({drift_pct:.2f}% drift of baseline: {r2_baseline:.6f})", file=sys.stderr)
    else:
        print(f"  R^2 = {params.get('r2', float('nan')):.6f}", file=sys.stderr)
    print(f"  resemblance_metric = {resemblance_metric}", file=sys.stderr)
    print(f"  resemblance_score = {selected_score:.6g}", file=sys.stderr)
    print("", file=sys.stderr)
    print("Equation (copy-paste-friendly, variable x = bpw):", file=sys.stderr)
    if params["transform"] == "ln":
        grapher_transform = "ln"
    elif params["transform"] == "log10":
        grapher_transform = "log10"
    elif params["transform"] == "log2":
        grapher_transform = "log2"
    elif params["transform"] == "logn":
        grapher_transform = f"log{params.get('log_base'):.6g}" if params.get("log_base") is not None else "logN"
    else:
        grapher_transform = "identity"

    # Note: for identity we now use the simplified merged-a equation form (no separate b).
    if grapher_transform == "identity":
        grapher_eq = f"y = {params['d']:.12g} + {params['a']:.12g} * ( x - {params['c']:.12g} )^(-{params['p']:.12g})"
    else:
        grapher_eq = f"y = {params['d']:.12g} + {params['a']:.12g} * {grapher_transform}( {params['b']:.12g} * (x - {params['c']:.12g}) )^(-{params['p']:.12g})"
    # normalize adjacent signs
    grapher_eq = collapse_sign_pairs(grapher_eq)

    # If the user requested equation_only in non-machine mode, we should print the equation to stdout
    # (and nothing else to stdout). The diagnostics above go to stderr.
    if equation_only:
        print(grapher_eq)
    else:
        print(grapher_eq, file=sys.stderr)

    if not suppress_plot:
        if plot_output:
            plt.savefig(plot_output, dpi=300)
            print(f"Wrote plot to {plot_output}", file=sys.stderr)
        else:
            try:
                plt.show()
            except Exception:
                fallback = "bpw_plot.png"
                plt.savefig(fallback, dpi=300)
                print(f"Interactive display not available; saved plot to {fallback}", file=sys.stderr)

    return None


# --------------------------
# CSV writing (unchanged) & CLI glue
# --------------------------

def write_bpw_results_csv_from_rows(input_csv_path: str,
                                    out_csv_path: str,
                                    parsed_mapfiles: Dict[str, List[Dict[str, Any]]],
                                    qtypes_to_consider: Optional[List[str]],
                                    allow_impure_map: bool,
                                    fail_on_missing_bytes: bool,
                                    hide_empty: bool) -> None:
    if not os.path.exists(input_csv_path):
        raise FileNotFoundError(input_csv_path)

    with open(input_csv_path, newline="", encoding="utf-8") as inf:
        reader = csv.DictReader(inf)
        rows = list(reader)
        input_fieldnames = list(reader.fieldnames or [])

    qcol = None
    qcol_index = None
    for idx, fn in enumerate(input_fieldnames):
        if fn.lower() == "qtype":
            qcol = fn
            qcol_index = idx
            break
    if qcol is None:
        raise ValueError("Input CSV must contain a 'qtype' column (case-insensitive).")

    out_fieldnames = list(input_fieldnames)
    if "bpw" not in [f.lower() for f in out_fieldnames]:
        insert_pos = (qcol_index + 1) if qcol_index is not None else len(out_fieldnames)
        out_fieldnames.insert(insert_pos, "bpw")
    else:
        cur_idx = next(i for i, fn in enumerate(out_fieldnames) if fn.lower() == "bpw")
        if qcol_index is not None and cur_idx != qcol_index + 1:
            bpw_val = out_fieldnames.pop(cur_idx)
            insert_pos = qcol_index + 1
            out_fieldnames.insert(insert_pos, bpw_val)

    output_rows: List[Dict[str, Any]] = []

    for row in rows:
        qval = (row.get(qcol) or "").strip()
        if not qval:
            if hide_empty:
                continue
            row["bpw"] = "404"
            output_rows.append(row)
            continue

        qval_l = qval.lower()
        if qtypes_to_consider is not None and qval_l not in [q.lower() for q in qtypes_to_consider]:
            if hide_empty:
                continue
            row["bpw"] = "404"
            output_rows.append(row)
            continue

        parsed_entries = parsed_mapfiles.get(qval_l)
        if parsed_entries is None:
            csv_dir = os.path.dirname(os.path.abspath(input_csv_path)) or "."
            candidate = os.path.join(csv_dir, f"tensors.{qval}.map")
            if os.path.exists(candidate):
                _, parsed_entries = load_parsed_entries_from_mapfile(candidate)
                parsed_mapfiles[qval_l] = parsed_entries
            else:
                if hide_empty:
                    continue
                row["bpw"] = "404"
                output_rows.append(row)
                print(f"[CSV] qtype={qval}: no map file found; wrote 404.", file=sys.stderr)
                continue

        bpw_val, details = compute_bpw_for_qtype(parsed_entries,
                                                 declared_qtype=qval_l,
                                                 allow_impure_map=allow_impure_map,
                                                 fail_on_missing_bytes=fail_on_missing_bytes)
        if bpw_val is None:
            if hide_empty:
                continue
            row["bpw"] = "404"
            output_rows.append(row)
            print(f"[CSV] qtype={qval}: bpw not computed (reason={details.get('reason')}).", file=sys.stderr)
        else:
            row["bpw"] = f"{bpw_val:.6f}"
            output_rows.append(row)
            if details.get("soft_impure"):
                print(f"[CSV] qtype={qval}: soft-impure tensors present (same q-family variants).", file=sys.stderr)
            if details.get("hard_impure") and allow_impure_map:
                print(f"[CSV] qtype={qval}: hard-impure tensors present but --allow-impure-map used; BPW computed.", file=sys.stderr)

    def bpw_key(r: Dict[str, Any]):
        v = r.get("bpw")
        if v is None:
            return float("-inf")
        try:
            fv = float(v)
            return fv
        except Exception:
            return float("-inf")

    output_rows.sort(key=bpw_key, reverse=True)

    with open(out_csv_path, "w", newline="", encoding="utf-8") as ouf:
        writer = csv.DictWriter(ouf, fieldnames=out_fieldnames)
        writer.writeheader()
        for r in output_rows:
            writer.writerow(r)


def write_bpw_results_csv_from_rows_and_maybe_plot(input_csv_path: str,
                                                   out_csv_path: str,
                                                   parsed_mapfiles: Dict[str, List[Dict[str, Any]]],
                                                   qtypes_to_consider: Optional[List[str]],
                                                   allow_impure_map: bool,
                                                   fail_on_missing_bytes: bool,
                                                   hide_empty: bool,
                                                   plot: bool = False,
                                                   ycol_identifier: Optional[str] = None,
                                                   plot_output: Optional[str] = None,
                                                   fit_equation: bool = True,
                                                   d_from_lowest_k: Optional[int] = None,
                                                   d_free: bool = False,
                                                   c_from_lowest_k: Optional[int] = None,
                                                   c_free: bool = False,
                                                   ignore_outliers_threshold: float = 30.0,
                                                   threads: Optional[int] = None,
                                                   metric_name: Optional[str] = "metric",
                                                   predict_bpw_values: Optional[List[float]] = None,
                                                   transforms: Optional[List[str]] = None,
                                                   suppress_plot: bool = False,
                                                   equation_only: bool = False,
                                                   identity_s_min: float = -1.0,
                                                   identity_s_max: float = 1.0,
                                                   identity_s_steps: int = 9,
                                                   p_grid_min: float = 0.2,
                                                   p_grid_max: float = 3.0,
                                                   p_grid_steps: int = 15,
                                                   logn_base_min: float = 2.0,
                                                   logn_base_max: float = 100.0,
                                                   logn_base_steps: int = 8,
                                                   c_mult_min: float = -0.9,
                                                   c_mult_max: float = 10.0,
                                                   c_mult_steps: int = 40,
                                                   b_grid_steps: int = 60,
                                                   b_refine_steps: int = 60,
                                                   c_refine_steps: int = 60,
                                                   p_refine_steps: int = 20,
                                                   N_refine_steps: int = 12,
                                                   resemblance_metric: str = "asym_abs",
                                                   penalize_above: float = 2.0,
                                                   penalize_below: float = 1.0,
                                                   drift_below: float = 0.0,
                                                   drift_above: float = 0.0) -> Optional[List[float]]:
    write_bpw_results_csv_from_rows(input_csv_path, out_csv_path, parsed_mapfiles,
                                    qtypes_to_consider, allow_impure_map, fail_on_missing_bytes, hide_empty)
    # If the caller explicitly requested plotting, do the previous behavior.
    if plot:
        return compute_and_plot_from_csv(out_csv_path,
                                         bpw_column="bpw",
                                         ycol_identifier=ycol_identifier,
                                         hide_empty=hide_empty,
                                         plot_output=plot_output,
                                         fit_equation=fit_equation,
                                         d_from_lowest_k=d_from_lowest_k,
                                         d_free=d_free,
                                         c_from_lowest_k=c_from_lowest_k,
                                         c_free=c_free,
                                         ignore_outliers_threshold=ignore_outliers_threshold,
                                         threads=threads,
                                         metric_name=metric_name,
                                         predict_bpw_values=predict_bpw_values,
                                         transforms=transforms,
                                         suppress_plot=suppress_plot,
                                         equation_only=equation_only,
                                         identity_s_min=identity_s_min,
                                         identity_s_max=identity_s_max,
                                         identity_s_steps=identity_s_steps,
                                         p_grid_min=p_grid_min,
                                         p_grid_max=p_grid_max,
                                         p_grid_steps=p_grid_steps,
                                         logn_base_min=logn_base_min,
                                         logn_base_max=logn_base_max,
                                         logn_base_steps=logn_base_steps,
                                         c_mult_min=c_mult_min,
                                         c_mult_max=c_mult_max,
                                         c_mult_steps=c_mult_steps,
                                         b_grid_steps=b_grid_steps,
                                         b_refine_steps=b_refine_steps,
                                         c_refine_steps=c_refine_steps,
                                         p_refine_steps=p_refine_steps,
                                         N_refine_steps=N_refine_steps,
                                         resemblance_metric=resemblance_metric,
                                         penalize_above=penalize_above,
                                         penalize_below=penalize_below,
                                         drift_below=drift_below,
                                         drift_above=drift_above)
    # If plotting was not requested but predictions were requested, still run the fitter (with plotting suppressed if requested)
    if predict_bpw_values or equation_only:
        return compute_and_plot_from_csv(out_csv_path,
                                         bpw_column="bpw",
                                         ycol_identifier=ycol_identifier,
                                         hide_empty=hide_empty,
                                         plot_output=plot_output,
                                         fit_equation=fit_equation,
                                         d_from_lowest_k=d_from_lowest_k,
                                         d_free=d_free,
                                         c_from_lowest_k=c_from_lowest_k,
                                         c_free=c_free,
                                         ignore_outliers_threshold=ignore_outliers_threshold,
                                         threads=threads,
                                         metric_name=metric_name,
                                         predict_bpw_values=predict_bpw_values,
                                         transforms=transforms,
                                         suppress_plot=suppress_plot,
                                         equation_only=equation_only,
                                         identity_s_min=identity_s_min,
                                         identity_s_max=identity_s_max,
                                         identity_s_steps=identity_s_steps,
                                         p_grid_min=p_grid_min,
                                         p_grid_max=p_grid_max,
                                         p_grid_steps=p_grid_steps,
                                         logn_base_min=logn_base_min,
                                         logn_base_max=logn_base_max,
                                         logn_base_steps=logn_base_steps,
                                         c_mult_min=c_mult_min,
                                         c_mult_max=c_mult_max,
                                         c_mult_steps=c_mult_steps,
                                         b_grid_steps=b_grid_steps,
                                         b_refine_steps=b_refine_steps,
                                         c_refine_steps=c_refine_steps,
                                         p_refine_steps=p_refine_steps,
                                         N_refine_steps=N_refine_steps,
                                         resemblance_metric=resemblance_metric,
                                         penalize_above=penalize_above,
                                         penalize_below=penalize_below,
                                         drift_below=drift_below,
                                         drift_above=drift_above)
    return None


# --------------------------
# CLI entrypoint
# --------------------------

def main():
    ap = argparse.ArgumentParser(description="Compute model BPW from tensors.qtype.map files.")
    ap.add_argument("map_file", nargs="?", default=None,
                    help="(Legacy) single map file to process (optional). Use --map-files for many.")
    ap.add_argument("--map-files", nargs="+", default=None,
                    help="One or more map files to process (optional).")
    ap.add_argument("--qtypes", nargs="+", default=None,
                    help="One or more qtypes to process (mutually exclusive with --map-files).")
    ap.add_argument("--results-csv", type=str, default=None,
                    help="Optional input CSV with a 'qtype' column. Produces bpw_<input_csv>.")
    ap.add_argument("--hide-empty", action="store_true",
                    help="When producing CSV, hide rows where bpw wasn't computed; otherwise fill '404'.")
    ap.add_argument("--allow-impure-map", action="store_true",
                    help="Allow hard-impure tensors (different families) and compute BPW with a warning.")
    ap.add_argument("--fail-on-missing-bytes", action="store_true",
                    help="Fail when tensors lack bytes= for a qtype (treated as missing bpw).")
    ap.add_argument("--plot", action="store_true", help="Produce a plot of chosen metric (y) vs bpw (x) from output CSV.")
    ap.add_argument("--ycol", type=str, default="2",
                    help="Y column name or index for plotting (defaults to '2' when not provided).")
    ap.add_argument("--plot-output", type=str, default=None, help="Path to save plot PNG (optional).")
    ap.add_argument("--no-equation", "--skip-fit", action="store_true",
                    help="Do not compute or plot the fitted equation (only scatter).")
    # Note: we changed defaults so that when user doesn't provide --d-from-lowest or --c-from-lowest
    # the code will try all K=1..N candidates. If a user wants a specific anchoring value, they can provide K.
    ap.add_argument("--d-from-lowest", type=int, default=None,
                    help="When not using --d-free, anchor d to mean of the K lowest-y rows' y values (provide K). If omitted, try all K=1..N and pick best.")
    ap.add_argument("--d-free", action="store_true", help="Allow fitting d as a free parameter (do not anchor to lowest values).")
    # New c parameters: parallel to d, but c is based on lowest X (bpw) values and used as b*(x - c)
    ap.add_argument("--c-from-lowest", type=int, default=None,
                    help="When not using --c-free, anchor c to mean of the K lowest-x rows' x values (provide K). If omitted, try all K=1..N and pick best.")
    ap.add_argument("--c-free", action="store_true", help="Allow fitting c as a free parameter (do not anchor to lowest-x values).")
    # New outlier handling parameter: modified z-score threshold. default 30 per user request.
    ap.add_argument("--ignore-outliers", type=float, default=30.0,
                    help="Ignore outliers (modified z-score) when fitting if >0. Default 30. Set to 0 to disable.")
    # Threading parameter: number of processes to use. Default 0/None => use machine CPU count.
    ap.add_argument("--threads", type=int, default=0,
                    help="Number of worker processes to use for fitting (default 0 => use machine CPU count).")
    ap.add_argument("--metric-name", type=str, default=None,
                    help="Optional metric name used for plot labeling. If omitted and --results-csv is provided, infer from results CSV filename like 'metricname_results.csv'; otherwise default 'metric'.")
    # New: exclude qtypes by regex
    ap.add_argument("--exclude-qtypes", nargs="+", default=None,
                    help="One or more regular expressions; qtypes matching any will be excluded (case-insensitive).")
    # New: predict bpw values - machine-friendly mode prints JSON array to stdout and everything else to stderr
    ap.add_argument("--predict-bpw-values", nargs="+", type=float, default=None,
                    help="Provide a list of bpw values; when an equation is produced, print JSON array of predicted metric values to stdout. Other textual output goes to stderr.")
    # New: allow user to specify the transforms to try (defaults to ln, log2, log10, logn, identity)
    ap.add_argument("--transforms", nargs="+", default=None,
                    help="Space-separated (or comma-separated) list of transforms to try. Choices: ln, log2, log10, logn, identity. Default: ln log2 log10 logn identity.")
    # New: print only the equation to stdout (one line). When used, the equation string is printed to stdout and
    # other normal stdout outputs (like JSON predictions) are suppressed. Diagnostic messages still go to stderr.
    ap.add_argument("--equation-only", action="store_true",
                    help="Print only the fitted equation to stdout (one line). Works with or without --plot. Overrides --predict-bpw-values stdout output.")
    # New identity s_grid configurables: exponent range and step count for the multiplicative scale grid used when identity transform is chosen.
    ap.add_argument("--identity-scale-min", type=float, default=-1.0,
                    help="Minimum exponent for identity c-scaling grid (used as 10**min to 10**max). Default -1.0 (affects identity transform only)")
    ap.add_argument("--identity-scale-max", type=float, default=1.0,
                    help="Maximum exponent for identity c-scaling grid (used as 10**min to 10**max). Default 1.0 (affects identity transform only)")
    ap.add_argument("--identity-scale-steps", type=int, default=9,
                    help="Number of steps to use for identity c-scaling grid. Default 9 (affects identity transform only)")
    # New: p-grid parameters (affects all transforms)
    ap.add_argument("--p-grid-min", type=float, default=0.2,
                    help="Minimum exponent p to search (applies to all transforms). Default 0.2")
    ap.add_argument("--p-grid-max", type=float, default=3.0,
                    help="Maximum exponent p to search (applies to all transforms). Default 3.0")
    ap.add_argument("--p-grid-steps", type=int, default=15,
                    help="Number of p grid steps (applies to all transforms). Default 15")
    # New: controls for logn base grid (affects logn transform)
    ap.add_argument("--logn-base-min", type=float, default=2.0,
                    help="Minimum base for logn grid (affects logn transform). Default 2.0")
    ap.add_argument("--logn-base-max", type=float, default=100.0,
                    help="Maximum base for logn grid (affects logn transform). Default 100.0")
    ap.add_argument("--logn-base-steps", type=int, default=8,
                    help="Number of logn base steps (affects logn transform). Default 8")
    # New: c_multipliers grid controls (affects how c candidates are generated for all transforms)
    ap.add_argument("--c-mult-min", type=float, default=-0.9,
                    help="Minimum multiplier for c candidate generation (applies to all transforms). Default -0.9")
    ap.add_argument("--c-mult-max", type=float, default=10.0,
                    help="Maximum multiplier for c candidate generation (applies to all transforms). Default 10.0")
    ap.add_argument("--c-mult-steps", type=int, default=40,
                    help="Number of steps for c multipliers grid (applies to all transforms). Default 40")
    # New: b grid density (affects non-identity transforms)
    ap.add_argument("--b-grid-steps", type=int, default=60,
                    help="Number of steps for b grid (affects non-identity transforms). Default 60")
    # New: refinement densities for final search refinement stage
    ap.add_argument("--b-refine-steps", type=int, default=60,
                    help="Number of b refinement steps in final refinement (affects non-identity transforms). Default 60")
    ap.add_argument("--c-refine-steps", type=int, default=60,
                    help="Number of c refinement steps in final refinement. Default 60")
    ap.add_argument("--p-refine-steps", type=int, default=20,
                    help="Number of p refinement steps in final refinement. Default 20")
    ap.add_argument("--N-refine-steps", type=int, default=12,
                    help="Number of base refinement steps for logn in final refinement. Default 12")
    # Resemblance metric options (new)
    ap.add_argument("--resemblance-metric", type=str, default="asym_abs",
                    help="Resemblance metric used to select the best candidate among searches. Choices: sse, mae, abs_mean, median_abs, r2, asym_abs, penalize_above, penalize_below. Default 'asym_abs' (favours matching lower actual values).")
    ap.add_argument("--penalize-above", type=float, default=2.0,
                    help="Penalty multiplier applied to over-predictions when using asymmetric resemblance metrics (default 2.0).")
    ap.add_argument("--penalize-below", type=float, default=1.0,
                    help="Penalty multiplier applied to under-predictions when using asymmetric resemblance metrics (default 1.0).")
    ap.add_argument("--drift-below", type=float, default=0.0,
                    help="R^2 drift threshold in percent - removes values above and furthest to the theoretical curve. Example: --drift-below 5 means an allowed R^2 drop of 5%% of the baseline R^2 (baseline 0.95 -> allowed lower bound 0.95*(1-0.05)=0.9025). When provided this triggers the reconsideration loop.")
    ap.add_argument("--drift-above", type=float, default=0.0,
                    help="R^2 drift threshold in percent - removes values below and furthest to the theoretical curve. Example: --drift-above 5 means an allowed R^2 drop of 5%% of the baseline R^2 (baseline 0.95 -> allowed lower bound 0.95*(1-0.05)=0.9025). When provided this triggers the reconsideration loop.")
    args = ap.parse_args()

    if args.map_files and args.qtypes:
        print("ERROR: --map-files and --qtypes cannot both be used.", file=sys.stderr)
        sys.exit(2)

    # Normalize transforms argument into a list (split commas if user passed a single comma-separated string)
    transforms_arg: Optional[List[str]] = None
    if args.transforms:
        tmp = []
        for t in args.transforms:
            # allow comma-separated single argument
            parts = [s.strip() for s in t.split(",") if s.strip()]
            tmp.extend(parts)
        # keep unique while preserving order
        seen_t = set()
        transforms_arg = [x for x in tmp if not (x in seen_t or seen_t.add(x))]

    parsed_mapfiles: Dict[str, List[Dict[str, Any]]] = {}
    if args.map_files:
        for mf in args.map_files:
            try:
                q, parsed = load_parsed_entries_from_mapfile(mf)
            except Exception as ex:
                print(f"ERROR: failed to read map file {mf}: {ex}", file=sys.stderr)
                sys.exit(3)
            if q is None:
                print(f"Warning: could not infer qtype from map filename '{mf}'; skipping.", file=sys.stderr)
                continue
            parsed_mapfiles[q.lower()] = parsed

    if args.map_file:
        try:
            q, parsed = load_parsed_entries_from_mapfile(args.map_file)
        except Exception as ex:
            print(f"ERROR: failed to read map file {args.map_file}: {ex}", file=sys.stderr)
            sys.exit(3)
        if q is None:
            print(f"Warning: could not infer qtype from map filename '{args.map_file}'.", file=sys.stderr)
        else:
            parsed_mapfiles.setdefault(q.lower(), parsed)

    qtypes_to_process: Optional[List[str]] = None

    # Helper to apply exclude regexes to a list of qtypes (lowercase strings).
    def apply_excludes(qtypes: List[str], exclude_patterns: Optional[List[str]]) -> List[str]:
        if not exclude_patterns:
            return qtypes
        compiled = []
        for pat in exclude_patterns:
            try:
                compiled.append(re.compile(pat, flags=re.IGNORECASE))
            except re.error:
                print(f"Warning: invalid exclude regex '{pat}' - ignoring.", file=sys.stderr)
        if not compiled:
            return qtypes
        filtered = []
        for q in qtypes:
            if any(rx.search(q) for rx in compiled):
                print(f"[exclude] qtype '{q}' excluded by regex.", file=sys.stderr)
                continue
            filtered.append(q)
        return filtered

    if args.results_csv:
        with open(args.results_csv, newline="", encoding="utf-8") as inf:
            reader = csv.DictReader(inf)
            if not reader.fieldnames:
                print(f"ERROR: results CSV {args.results_csv} has no header.", file=sys.stderr)
                sys.exit(4)
            qcol = None
            for fn in reader.fieldnames:
                if fn.lower() == "qtype":
                    qcol = fn
                    break
            if qcol is None:
                print(f"ERROR: results CSV {args.results_csv} must have a 'qtype' column.", file=sys.stderr)
                sys.exit(4)
            csv_qtypes = []
            for r in reader:
                val = (r.get(qcol) or "").strip()
                if val:
                    csv_qtypes.append(val.lower())
            csv_qtypes = list(dict.fromkeys(csv_qtypes))

        # apply exclude patterns to csv_qtypes
        csv_qtypes = apply_excludes(csv_qtypes, args.exclude_qtypes)

        if args.qtypes:
            qtypes_to_process = [q.lower() for q in args.qtypes]
            qtypes_to_process = apply_excludes(qtypes_to_process, args.exclude_qtypes)
        else:
            qtypes_to_process = csv_qtypes

        if not args.map_files and not args.map_file:
            csv_dir = os.path.dirname(os.path.abspath(args.results_csv)) or "."
            for q in qtypes_to_process:
                if q in parsed_mapfiles:
                    continue
                cand = os.path.join(csv_dir, f"tensors.{q}.map")
                if os.path.exists(cand):
                    try:
                        _, parsed = load_parsed_entries_from_mapfile(cand)
                        parsed_mapfiles[q] = parsed
                    except Exception as ex:
                        print(f"Warning: failed to read inferred map file {cand}: {ex}", file=sys.stderr)
    else:
        if args.qtypes:
            qtypes_to_process = [q.lower() for q in args.qtypes]
            qtypes_to_process = apply_excludes(qtypes_to_process, args.exclude_qtypes)
            for q in list(qtypes_to_process):
                if q in parsed_mapfiles:
                    continue
                cand = os.path.join(".", f"tensors.{q}.map")
                if os.path.exists(cand):
                    try:
                        _, parsed = load_parsed_entries_from_mapfile(cand)
                        parsed_mapfiles[q] = parsed
                    except Exception as ex:
                        print(f"Warning: failed to read inferred map file {cand}: {ex}", file=sys.stderr)
        elif parsed_mapfiles:
            qtypes_to_process = list(parsed_mapfiles.keys())
            qtypes_to_process = apply_excludes(qtypes_to_process, args.exclude_qtypes)
        elif args.map_file:
            qtypes_to_process = list(parsed_mapfiles.keys())
            qtypes_to_process = apply_excludes(qtypes_to_process, args.exclude_qtypes)
        else:
            print("ERROR: no input map file(s) provided. Use --map-files, positional map_file, or --results-csv (with map files next to CSV).", file=sys.stderr)
            sys.exit(5)

    if args.results_csv:
        in_csv = args.results_csv
        out_csv = os.path.join(os.path.dirname(os.path.abspath(in_csv)) or ".", "bpw_" + os.path.basename(in_csv))
        try:
            # infer metric_name from csv filename if not provided
            metric_name = args.metric_name
            if not args.metric_name:
                base = os.path.basename(in_csv)
                m = re.match(r"(?i)^(.+)_results\.csv$", base)
                if m:
                    metric_name = m.group(1)

            # If predict mode requested, we want to handle exceptions differently: print [] to stdout on error.
            if args.predict_bpw_values:
                try:
                    preds = write_bpw_results_csv_from_rows_and_maybe_plot(input_csv_path=in_csv,
                                                                           out_csv_path=out_csv,
                                                                           parsed_mapfiles=parsed_mapfiles,
                                                                           qtypes_to_consider=qtypes_to_process,
                                                                           allow_impure_map=args.allow_impure_map,
                                                                           fail_on_missing_bytes=args.fail_on_missing_bytes,
                                                                           hide_empty=args.hide_empty,
                                                                           plot=args.plot,
                                                                           ycol_identifier=args.ycol,
                                                                           plot_output=args.plot_output,
                                                                           fit_equation=(not args.no_equation),
                                                                           d_from_lowest_k=args.d_from_lowest,
                                                                           d_free=args.d_free,
                                                                           c_from_lowest_k=args.c_from_lowest,
                                                                           c_free=args.c_free,
                                                                           ignore_outliers_threshold=(args.ignore_outliers if args.ignore_outliers and float(args.ignore_outliers) > 0.0 else 0.0),
                                                                           threads=(args.threads if args.threads and int(args.threads) > 0 else None),
                                                                           metric_name=metric_name,
                                                                           predict_bpw_values=args.predict_bpw_values,
                                                                           transforms=transforms_arg,
                                                                           suppress_plot=(not args.plot),
                                                                           equation_only=args.equation_only,
                                                                           identity_s_min=args.identity_scale_min,
                                                                           identity_s_max=args.identity_scale_max,
                                                                           identity_s_steps=args.identity_scale_steps,
                                                                           p_grid_min=args.p_grid_min,
                                                                           p_grid_max=args.p_grid_max,
                                                                           p_grid_steps=args.p_grid_steps,
                                                                           logn_base_min=args.logn_base_min,
                                                                           logn_base_max=args.logn_base_max,
                                                                           logn_base_steps=args.logn_base_steps,
                                                                           c_mult_min=args.c_mult_min,
                                                                           c_mult_max=args.c_mult_max,
                                                                           c_mult_steps=args.c_mult_steps,
                                                                           b_grid_steps=args.b_grid_steps,
                                                                           b_refine_steps=args.b_refine_steps,
                                                                           c_refine_steps=args.c_refine_steps,
                                                                           p_refine_steps=args.p_refine_steps,
                                                                           N_refine_steps=args.N_refine_steps,
                                                                           resemblance_metric=args.resemblance_metric,
                                                                           penalize_above=args.penalize_above,
                                                                           penalize_below=args.penalize_below,
                                                                           drift_below=args.drift_below,
                                                                           drift_above=args.drift_above)
                    # If the compute function printed predictions to stdout, we're done. Still print CSV path to stderr for diagnostics.
                    print(f"Wrote BPW CSV: {out_csv}", file=sys.stderr)
                except Exception as ex:
                    # On error in predict mode: print empty JSON array to stdout and details to stderr.
                    print("[]")
                    print(f"ERROR (predict mode): failed to produce {out_csv}: {ex}", file=sys.stderr)
                    # exit quietly with success (machine-mode expects [] on failure)
                    return
            else:
                # non-machine mode: preserve previous behavior (raise/exit on error)
                write_bpw_results_csv_from_rows_and_maybe_plot(input_csv_path=in_csv,
                                                               out_csv_path=out_csv,
                                                               parsed_mapfiles=parsed_mapfiles,
                                                               qtypes_to_consider=qtypes_to_process,
                                                               allow_impure_map=args.allow_impure_map,
                                                               fail_on_missing_bytes=args.fail_on_missing_bytes,
                                                               hide_empty=args.hide_empty,
                                                               plot=args.plot,
                                                               ycol_identifier=args.ycol,
                                                               plot_output=args.plot_output,
                                                               fit_equation=(not args.no_equation),
                                                               d_from_lowest_k=args.d_from_lowest,
                                                               d_free=args.d_free,
                                                               c_from_lowest_k=args.c_from_lowest,
                                                               c_free=args.c_free,
                                                               ignore_outliers_threshold=(args.ignore_outliers if args.ignore_outliers and float(args.ignore_outliers) > 0.0 else 0.0),
                                                               threads=(args.threads if args.threads and int(args.threads) > 0 else None),
                                                               metric_name=metric_name,
                                                               predict_bpw_values=None,
                                                               transforms=transforms_arg,
                                                               suppress_plot=(not args.plot),
                                                               equation_only=args.equation_only,
                                                               identity_s_min=args.identity_scale_min,
                                                               identity_s_max=args.identity_scale_max,
                                                               identity_s_steps=args.identity_scale_steps,
                                                               p_grid_min=args.p_grid_min,
                                                               p_grid_max=args.p_grid_max,
                                                               p_grid_steps=args.p_grid_steps,
                                                               logn_base_min=args.logn_base_min,
                                                               logn_base_max=args.logn_base_max,
                                                               logn_base_steps=args.logn_base_steps,
                                                               c_mult_min=args.c_mult_min,
                                                               c_mult_max=args.c_mult_max,
                                                               c_mult_steps=args.c_mult_steps,
                                                               b_grid_steps=args.b_grid_steps,
                                                               b_refine_steps=args.b_refine_steps,
                                                               c_refine_steps=args.c_refine_steps,
                                                               p_refine_steps=args.p_refine_steps,
                                                               N_refine_steps=args.N_refine_steps,
                                                               resemblance_metric=args.resemblance_metric,
                                                               penalize_above=args.penalize_above,
                                                               penalize_below=args.penalize_below,
                                                               drift_below=args.drift_below,
                                                               drift_above=args.drift_above)
                print(f"Wrote BPW CSV: {out_csv}")
        except Exception as ex:
            print(f"ERROR: failed to produce {out_csv}: {ex}", file=sys.stderr)
            sys.exit(6)

    if not args.results_csv:
        for q in qtypes_to_process:
            parsed_entries = parsed_mapfiles.get(q)
            if parsed_entries is None:
                print("=" * 60)
                print(f"qtype: {q}")
                print("  No map file found for this qtype; BPW not computed.")
                print("=" * 60)
                continue
            bpw_val, details = compute_bpw_for_qtype(parsed_entries,
                                                     declared_qtype=q,
                                                     allow_impure_map=args.allow_impure_map,
                                                     fail_on_missing_bytes=args.fail_on_missing_bytes)
            print("=" * 60)
            print(f"qtype: {q}")
            if details.get("total_elements_with_bytes") is not None:
                print(f"  Total elements (used): {details.get('total_elements_with_bytes'):,}")
            if details.get("total_bytes") is not None:
                tb = details.get("total_bytes")
                print(f"  Total bytes (sum)    : {tb:,} ({human_bytes(tb)})")
            if bpw_val is None:
                print(f"  BPW: not computed (reason={details.get('reason')})")
            else:
                print(f"  BPW: {bpw_val:.6f} bits/weight")
            if details.get("soft_impure"):
                print("  NOTE: soft-impure tensors (same q-family variants) were present. See stderr for details.")
            if details.get("hard_impure") and args.allow_impure_map:
                print("  WARNING: hard-impure tensors present but --allow-impure-map used; BPW computed anyway.")
            print("=" * 60)

    return


if __name__ == "__main__":
    main()
