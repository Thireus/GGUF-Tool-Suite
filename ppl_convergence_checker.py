#!/usr/bin/env python3
#***************************************************************#
#** This script is part of Thireus' GGUF Tool Suite.          **#
#** ppl_convergence_checker.py helps identify the min chunks  **#
#** required to produce quality calibration data.             **#
#**                                                           **#
#** ********************************************************* **#
#** --------------- Updated: Nov-09-2025 -------------------- **#
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
#** Copyright © 2025 - Thireus. ₚₗₑₐₛₑ ₛᵢᵣ, 𝒸ₐₙ ᵢ ₕₐₛ ₛₒₘₑ ᵥᵣₐₘ? **#
#***************************************************************#
#**PLEASE REFER TO THE README FILE FOR ADDITIONAL INFORMATION!**#
#***************************************************************#

"""
convergence_checker.py

Parse bench_ppl_kld_result.*.QTYPE.CHUNKS.txt files and compute a *resemblance* curve
and interactive comparison for a chosen metric (default KLD).

Fixes included in this version:
 - Resemblance is 100% for the final (max) chunk and decreases for less similar chunks.
 - Resemblance is clamped to [0, 100].
 - CSV now contains a suggested chunk per tensor based on a per-tensor resemblance-to-final test.
 - Y axis labeling uses the full metric name (e.g. "KL Divergence") and does not append "(normalized)".
 - Default acceptance threshold now expresses the *required resemblance percent* (default 95.0).

Usage examples:
    pip install numpy pandas matplotlib plotly
    python convergence_checker.py --qtype iq1_kt --chunks 250

Outputs:
 - <out_prefix>_resemblance.png : resemblance % vs chunk (global)
 - <out_prefix>_summary.csv     : per-tensor suggested chunk + global suggestion
 - <interactive_out>            : interactive HTML (requires plotly)

"""
from __future__ import annotations
import argparse
import os
import re
import sys
from typing import Dict, List, Union, Tuple, Optional

try:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go
    import plotly.io as pio
    PLOTLY_OK = True
except Exception as e:
    print("Error: missing dependency. Please install numpy, pandas, matplotlib, plotly.\n", e)
    sys.exit(2)

# Regexes
FNAME_RE = re.compile(r'^bench_ppl_kld_result\.(?P<tensor>.+)\.(?P<qtype>[^.]+)\.(?P<chunks>\d+)\.txt$')
LINE_RE = re.compile(
    r"^\s*(?P<chunk>\d+)\s+"
    r"(?P<ppl_mean>-?\d*\.?\d+)\s*(?:±|\+/-|\+)\s*(?P<ppl_err>-?\d*\.?\d+)\s+"
    r"(?P<ln_mean>-?\d*\.?\d+)\s*(?:±|\+/-|\+)\s*(?P<ln_err>-?\d*\.?\d+)\s+"
    r"(?P<kld_mean>-?\d*\.?\d+)\s*(?:±|\+/-|\+)\s*(?P<kld_err>-?\d*\.?\d+)\s+"
    r"(?P<dp_mean>-?\d*\.?\d+)\s*(?:±|\+/-|\+)\s*(?P<dp_err>-?\d*\.?\d+)\s*%\s+"
    r"(?P<same_mean>-?\d*\.?\d+)\s*(?:±|\+/-|\+)\s*(?P<same_err>-?\d*\.?\d+)\s*%\s*$"
)

CATEGORY_KEYS = {
    'PPL': ('ppl_mean', 'ppl_err'),
    'LN': ('ln_mean', 'ln_err'),
    'KLD': ('kld_mean', 'kld_err'),
    'DPRMS': ('dp_mean', 'dp_err'),
    'SAME_TOP_P': ('same_mean', 'same_err'),
}

CATEGORY_FULLNAME = {
    'PPL': 'Perplexity',
    'LN': 'ln(PPL(Q)/PPL(base))',
    'KLD': 'KL Divergence',
    'DPRMS': 'Δp RMS',
    'SAME_TOP_P': 'Same top p (%)',
}

BLK_RE = re.compile(r'blk\.?([0-9]+)', flags=re.IGNORECASE)
GROUP_RE = re.compile(r'group\.?([0-9]+)', flags=re.IGNORECASE)
_SPLIT_DIGITS_RE = re.compile(r'(\d+)')


# ----------------- file discovery & parsing -----------------

def find_files(qtype: str, chunks: int) -> List[Tuple[str, str]]:
    res = []
    for fn in os.listdir('.'):
        m = FNAME_RE.match(fn)
        if not m:
            continue
        if m.group('qtype') != qtype:
            continue
        if int(m.group('chunks')) != chunks:
            continue
        res.append((fn, m.group('tensor')))
    return sorted(res)


def parse_file(path: str) -> pd.DataFrame:
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            m = LINE_RE.match(line)
            if m:
                d = {k: float(v) for k, v in m.groupdict().items()}
                rows.append(d)
    if not rows:
        raise ValueError(f"No parsable rows in {path}")
    df = pd.DataFrame(rows)
    df = df.set_index('chunk').sort_index()
    return df


def _nat_tokens(s: str):
    """
    Turn a string into a list of tokens where digit sequences become ints
    and non-digits become lowercase strings. Useful for natural/alphanumeric sort.
    Example: "ffn_up12.weight" -> ["ffn_up", 12, ".weight"]
    """
    parts = _SPLIT_DIGITS_RE.split(s)
    tokens = []
    for p in parts:
        if not p:
            continue
        if p.isdigit():
            tokens.append(int(p))
        else:
            tokens.append(p.lower())
    return tokens

def sort_tensors_by_id(tensors: List[str]) -> List[str]:
    def key_fn(name: str):
        # Try blk first
        m = BLK_RE.search(name)
        if m:
            blk_id = int(m.group(1))
            # remainder: everything after the matched "blk..."; use the portion following the numeric id
            # find index of where the numeric id ends to get remainder consistently
            endpos = m.end()
            remainder = name[endpos:]  # could be like ".ffn_up_exps.weight"
            return (0, blk_id, _nat_tokens(remainder), name.lower())
        # Try group
        m = GROUP_RE.search(name)
        if m:
            grp_id = int(m.group(1))
            endpos = m.end()
            remainder = name[endpos:]
            return (1, grp_id, _nat_tokens(remainder), name.lower())
        # fallback: leave others after blk/group, sorted case-insensitively
        return (2, name.lower())
    return sorted(tensors, key=key_fn)


# ----------------- robust stats & normalization -----------------
def robust_stats_ignore_outliers(vals: np.ndarray, mad_thresh: float = 3.5, eps: float = 1e-12):
    vals = np.asarray(vals, dtype=float)
    finite_mask = np.isfinite(vals)
    if finite_mask.sum() == 0:
        return 0.0, 0.0, 0.0, 0.0, finite_mask
    v = vals[finite_mask]
    median = float(np.median(v))
    deviations = np.abs(v - median)
    mad = float(np.median(deviations))
    if mad > eps:
        modified_z = 0.6745 * (v - median) / (mad + eps)
        use_mask = np.abs(modified_z) <= mad_thresh
    else:
        std = float(np.std(v))
        if std < eps:
            use_mask = np.ones_like(v, dtype=bool)
        else:
            z = (v - median) / (std + eps)
            use_mask = np.abs(z) <= mad_thresh
    if use_mask.sum() < 1:
        use_mask = np.ones_like(use_mask, dtype=bool)
    used = v[use_mask]
    mean = float(np.mean(used))
    diffs = used - mean
    ups = diffs[diffs > 0]
    lows = -diffs[diffs < 0]
    up_mean = float(np.mean(ups)) if ups.size > 0 else 0.0
    low_mean = float(np.mean(lows)) if lows.size > 0 else 0.0
    amplitude = (up_mean + low_mean) / 2.0
    full_mask = np.zeros_like(vals, dtype=bool)
    full_mask[np.where(finite_mask)[0]] = use_mask
    return mean, up_mean, low_mean, amplitude, full_mask


def normalize_matrix_with_offset_and_stretch(val_df: pd.DataFrame, mad_thresh: float = 3.5) -> pd.DataFrame:
    idx = val_df.index
    cols = val_df.columns
    final_vals = val_df.iloc[-1].astype(float).values
    mean_final, up_final, low_final, amp_final, mask_final = robust_stats_ignore_outliers(final_vals, mad_thresh)
    eps = 1e-12
    out = pd.DataFrame(index=idx, columns=cols, dtype=float)
    for c in idx:
        vals = val_df.loc[c].astype(float).values
        mean_c, up_c, low_c, amp_c, mask_c = robust_stats_ignore_outliers(vals, mad_thresh)
        stretch_up = up_final / (up_c + eps) if up_c > eps else 1.0
        stretch_down = low_final / (low_c + eps) if low_c > eps else 1.0
        normalized = np.empty_like(vals, dtype=float)
        for i, v in enumerate(vals):
            if not np.isfinite(v):
                normalized[i] = np.nan
                continue
            if v >= mean_c:
                normalized[i] = mean_final + (v - mean_c) * stretch_up
            else:
                normalized[i] = mean_final - (mean_c - v) * stretch_down
        out.loc[c] = normalized
    return out


# ----------------- resemblance metrics -----------------
def compute_global_resemblance(val_df_norm: pd.DataFrame, metric: str = 'mean_abs', eps: float = 1e-12) -> pd.Series:
    """Compute a global resemblance percent per chunk (100% = identical to final chunk).

    We'll use:
      m_c = metric distance between val_df_norm.loc[c] and val_df_norm.loc[final]
      baseline = mean absolute dev of final chunk (a measure of typical amplitude)
      resemblance = clamp(0,100, 100*(1 - m_c / (baseline + eps)))
    """
    idx = val_df_norm.index
    final_vals = val_df_norm.iloc[-1].astype(float).values
    mean_final = float(np.nanmean(final_vals))
    baseline = float(np.nanmean(np.abs(final_vals - mean_final)))
    if baseline <= eps:
        baseline = float(np.nanmean(np.abs(final_vals - np.median(final_vals)))) + eps
    res = []
    for c in idx:
        vals = val_df_norm.loc[c].astype(float).values
        if metric == 'mean_abs':
            m = float(np.nanmean(np.abs(vals - final_vals)))
        elif metric == 'rms':
            m = float(np.sqrt(np.nanmean((vals - final_vals) ** 2)))
        elif metric == 'median_abs':
            m = float(np.nanmedian(np.abs(vals - final_vals)))
        else:
            raise ValueError(f"Unknown metric: {metric}")
        resemblance = 100.0 * (1.0 - (m / (baseline + eps)))
        resemblance = max(0.0, min(100.0, resemblance))
        res.append(resemblance)
    return pd.Series(res, index=idx)


def compute_per_tensor_resemblance(val_df_norm: pd.DataFrame, metric: str = 'mean_abs', eps: float = 1e-12) -> Dict[str, pd.Series]:
    """Compute per-tensor resemblance series. Return dict tensor -> Series(index=chunks).

    For each tensor t we compute m_t(c)=distance between v_t(c) and v_t(final).
    We normalize by a per-tensor baseline designed to be robust and avoid tiny denominators:
      denom_t = max(|v_t(final) - mean_final_all_tensors|, global_baseline/10, eps)
    Then resemblance_t(c) = clamp(0,100, 100*(1 - m_t(c)/denom_t)).
    """
    idx = val_df_norm.index
    final_vals = val_df_norm.iloc[-1].astype(float).values
    mean_final_all = float(np.nanmean(final_vals))
    global_baseline = float(np.nanmean(np.abs(final_vals - mean_final_all)))
    if global_baseline <= eps:
        global_baseline = float(np.nanmean(np.abs(final_vals - np.median(final_vals)))) + eps
    out: Dict[str, pd.Series] = {}
    for j, col in enumerate(val_df_norm.columns):
        v_final = float(final_vals[j])
        denom = max(abs(v_final - mean_final_all), global_baseline / 10.0, eps)
        series_vals = []
        for c in idx:
            v = float(val_df_norm.at[c, col])
            m = abs(v - v_final)
            resemblance = 100.0 * (1.0 - (m / denom))
            resemblance = max(0.0, min(100.0, resemblance))
            series_vals.append(resemblance)
        out[col] = pd.Series(series_vals, index=idx)
    return out


# ----------------- interactive HTML -----------------
def create_interactive_tensor_comparison(
    val_df: pd.DataFrame,
    out_html: str,
    normalize: bool,
    resemblance_series: pd.Series,
    title_prefix: str = 'Tensor comparison',
    metric_name: Optional[str] = None
) -> None:
    """
    Interactive HTML with a mode toggle:
      - Per-Tensor view (default): x = tensor names, y = metric values (final vs selected chunk)
      - Correlation view: x = selected chunk metric values, y = final chunk metric values (shows identity line y = x)
    """
    chunks = list(val_df.index.astype(int))
    tensors_unsorted = val_df.columns.tolist()
    tensors = sort_tensors_by_id(tensors_unsorted)
    val_df = val_df[tensors]

    # final (max) chunk values (y in correlation mode)
    final_vals = val_df.iloc[-1].values.astype(float)

    # compute overall min/max for identity/reference line (for correlation view)
    all_vals = val_df.values.astype(float)
    all_vals_min = float(np.nanmin(all_vals))
    all_vals_max = float(np.nanmax(all_vals))
    padding = (all_vals_max - all_vals_min) * 0.02 if all_vals_max > all_vals_min else 0.01
    min_range = all_vals_min - padding
    max_range = all_vals_max + padding

    # Build frames: each frame contains all traces so animation swaps them cleanly.
    frames = []
    for c in chunks:
        selected_vals = val_df.loc[c].values.astype(float)
        r = float(resemblance_series.loc[c]) if c in resemblance_series.index else float('nan')
        frame_title = f"{title_prefix} - chunk {c} - {r:.3f}% resemblance"

        # final (per-tensor) trace: x=tensors, y=final_vals
        final_tensor_trace = go.Scatter(
            x=tensors,
            y=final_vals,
            mode='markers',
            name='final_chunk',
            marker=dict(size=8),
            hovertemplate='%{x}<br>final=%{y:.6f}<extra></extra>'
        )

        # selected (per-tensor) trace: x=tensors, y=selected_vals
        selected_tensor_trace = go.Scatter(
            x=tensors,
            y=selected_vals,
            mode='markers',
            name=('normalized selected_chunk' if normalize else 'selected_chunk'),
            marker=dict(size=8),
            hovertemplate='%{x}<br>selected=%{y:.6f}<extra></extra>'
        )

        # identity/reference line y = x (for correlation mode)
        identity_trace = go.Scatter(
            x=[min_range, max_range],
            y=[min_range, max_range],
            mode='lines',
            name='y = x',
            line=dict(dash='dash'),
            hoverinfo='skip'
        )

        # correlation points - x=selected_vals, y=final_vals
        corr_trace = go.Scatter(
            x=selected_vals,
            y=final_vals,
            mode='markers',
            name='selected_vs_final',
            customdata=tensors,
            hovertemplate='%{customdata}<br>selected=%{x:.6f}<br>final=%{y:.6f}<extra></extra>'
        )

        frames.append(go.Frame(
            data=[final_tensor_trace, selected_tensor_trace, identity_trace, corr_trace],
            name=str(c),
            layout=go.Layout(title=frame_title)
        ))

    # initial data = first chunk frame
    init_frame = frames[0].data
    fig = go.Figure(data=list(init_frame), frames=frames)

    # Slider steps (animate by chunk)
    steps = []
    for c in chunks:
        steps.append(dict(
            method='animate',
            label=str(c),
            args=[[str(c)], dict(mode='immediate', frame=dict(duration=0, redraw=True), transition=dict(duration=0))]
        ))

    slider = dict(
        active=0,
        pad={'t': 18},
        steps=steps,
        currentvalue={'prefix': 'Selected chunk: '},
        x=0.03, y=0.0, len=0.94, xanchor='left', yanchor='bottom'
    )

    # Play/Pause buttons (anchored to the right, same vertical band as mode buttons)
    play_pause_updatemenu = dict(
        type='buttons',
        showactive=False,
        direction='right',   # horizontal layout for Play/Pause
        y=0.92,              # horizontal band shared with mode buttons
        x=0.98,              # far right
        xanchor='right',
        yanchor='bottom',
        pad={'t': 0, 'r': 10},
        buttons=[
            dict(label='Play', method='animate', args=[None, dict(frame=dict(duration=200, redraw=True), transition=dict(duration=0), fromcurrent=True, mode='immediate')]),
            dict(label='Pause', method='animate', args=[[None], dict(frame=dict(duration=0, redraw=False), mode='immediate')])
        ]
    )

    # Mode toggle: Tensor vs Correlation
    # Trace indices in each frame: 0=final_tensor, 1=selected_tensor, 2=identity, 3=corr_points
    tensor_mode_visible = [True, True, False, False]
    corr_mode_visible = [False, False, True, True]

    # Decide tickangle in Per-Tensor mode for readability
    tensor_tickangle = 45 if len(tensors) > 20 else 0

    # metric (y-axis) title: prefer metric_name if provided, else a short fallback
    metric_y_title = metric_name if metric_name is not None else ''

    # Mode toggle buttons
    # IMPORTANT: do NOT mix parent 'xaxis' dict and dotted 'xaxis.tickangle' keys in the same args.
    mode_toggle = dict(
        type='buttons',
        showactive=True,
        direction='left',
        x=0.03,
        y=0.92,            # lowered slightly to avoid overlapping title
        xanchor='left',
        yanchor='bottom',
        pad={'r': 10, 't': 6},
        buttons=[
            dict(
                label='Per-Tensor view',
                method='update',
                args=[
                    {'visible': tensor_mode_visible},
                    {
                        # intentionally do not set 'title' here so the current frame title remains visible
                        'xaxis': {'title': '', 'tickangle': tensor_tickangle},
                        'yaxis': {'title': metric_y_title},
                        'legend': {'orientation': 'v', 'x': 1.01, 'xanchor': 'left', 'y': 0.5}
                    }
                ]
            ),
            dict(
                label='Correlation view',
                method='update',
                args=[
                    {'visible': corr_mode_visible},
                    {
                        # keep title untouched (so it shows current chunk/frame title)
                        'xaxis': {'title': 'Selected chunk metric value', 'tickangle': 0},
                        'yaxis': {'title': 'Final chunk metric value'},
                        'legend': {'orientation': 'v', 'x': 1.01, 'xanchor': 'left', 'y': 0.5}
                    }
                ]
            )
        ]
    )

    # Put the mode toggle and play/pause updatemenus on the same horizontal band
    updatemenus = [mode_toggle, play_pause_updatemenu]

    # Layout: start in Per-Tensor mode. Note xaxis title intentionally blank to save space.
    initial_title = frames[0].layout.title.text if hasattr(frames[0].layout, 'title') else f"{title_prefix} - chunk {chunks[0]}"
    fig.update_layout(
        title=initial_title,
        xaxis_title='',  # intentionally empty to save vertical space (Per-Tensor mode)
        yaxis_title=metric_y_title,
        sliders=[slider],
        updatemenus=updatemenus,
        legend=dict(orientation='v', x=1.01, xanchor='left', y=0.5),
        margin=dict(t=40, r=120, b=50)
    )

    # If many tensors, angle labels for readability (applies in Per-Tensor mode)
    if len(tensors) > 20:
        fig.update_xaxes(tickangle=45)

    # Make sure initial visibility corresponds to Per-Tensor mode
    for i, vis in enumerate(tensor_mode_visible):
        if i < len(fig.data):
            fig.data[i].visible = vis

    # Save HTML
    pio.write_html(fig, file=out_html, auto_open=False)

# ----------------- main -----------------
def main(argv=None):
    p = argparse.ArgumentParser(description="Detect chunk convergence and compute resemblance curve")
    p.add_argument('--qtype', required=True, help='TARGETQTYPE present in filenames (required)')
    p.add_argument('--chunks', type=int, default=250, help='CHUNKS value in filenames (default: 250)')
    p.add_argument('--category', default='KLD', choices=list(CATEGORY_KEYS.keys()), help='Which metric/category to analyze (default: KLD)')
    p.add_argument('--accept_percent', type=float, default=95.0, help='Acceptance resemblance percent (0-100). Default 95 = 95%% similar to final chunk')
    p.add_argument('--out_prefix', default='convergence', help='Prefix for output files')
    p.add_argument('--interactive_out', default='convergence_interactive.html', help='Path to save interactive HTML')
    p.add_argument('--no-normalisation', action='store_true', help='Disable normalization')
    p.add_argument('--mad-threshold', type=float, default=3.5, help='MAD threshold for outlier filtering during normalization')
    p.add_argument('--resemblance_metric', choices=['mean_abs', 'rms', 'median_abs'], default='mean_abs', help='Metric used to compute resemblance to final chunk')
    args = p.parse_args(argv)

    files = find_files(args.qtype, args.chunks)
    if not files:
        print(f"No files found for qtype={args.qtype} and chunks={args.chunks} in current directory.")
        sys.exit(1)

    mean_key, err_key = CATEGORY_KEYS[args.category]

    per_tensor_series: Dict[str, pd.Series] = {}
    missing: List[Tuple[str, str]] = []
    for path, tensor in files:
        try:
            df = parse_file(path)
        except Exception as e:
            missing.append((path, str(e)))
            continue
        if mean_key not in df.columns:
            missing.append((path, 'missing column'))
            continue
        s = df[mean_key].astype(float)
        s.index = s.index.astype(int)
        s = s.sort_index()
        per_tensor_series[tensor] = s

    if not per_tensor_series:
        print("No parseable tensors found. Exiting.")
        sys.exit(1)

    common_idx = np.arange(1, args.chunks + 1)
    val_df = pd.DataFrame({t: s.reindex(common_idx).values for t, s in per_tensor_series.items()}, index=common_idx)
    val_df = val_df.ffill().bfill()

    normalize = not args.no_normalisation
    if normalize:
        val_df_norm = normalize_matrix_with_offset_and_stretch(val_df, mad_thresh=args.mad_threshold)
    else:
        val_df_norm = val_df.astype(float)

    # compute resemblance series (global)
    resemblance_series = compute_global_resemblance(val_df_norm, metric=args.resemblance_metric)

    # compute per-tensor resemblance series and per-tensor suggested chunk
    per_tensor_resemblance = compute_per_tensor_resemblance(val_df_norm, metric=args.resemblance_metric)
    per_tensor_suggest: Dict[str, Optional[int]] = {}
    for t, series in per_tensor_resemblance.items():
        # find first chunk where resemblance >= accept_percent
        idxs = np.where(series.values >= args.accept_percent)[0]
        per_tensor_suggest[t] = int(series.index[idxs[0]]) if idxs.size > 0 else None

    # global suggested chunk (first chunk where global resemblance >= accept_percent)
    idxs = np.where(resemblance_series.values >= args.accept_percent)[0]
    global_suggest = int(resemblance_series.index[idxs[0]]) if idxs.size > 0 else None

    # Save resemblance PNG
    out_png = f"{args.out_prefix}_resemblance.png"
    plt.figure(figsize=(10, 6))
    plt.plot(resemblance_series.index, resemblance_series.values, linewidth=2.0, label='Resemblance (%)')
    plt.axhline(args.accept_percent, color='k', linestyle='--', linewidth=1, label=f'accept {args.accept_percent}%')
    if global_suggest is not None:
        plt.axvline(global_suggest, color='green', linestyle=':', linewidth=1.5, label=f'suggested chunk {global_suggest}')
    plt.xlabel('Chunk')
    plt.ylabel(f"{CATEGORY_FULLNAME.get(args.category, args.category)}")
    plt.title(f'Resemblance curve (normalized={"ON" if normalize else "OFF"}) for {CATEGORY_FULLNAME.get(args.category, args.category)} (qtype={args.qtype})')
    plt.legend(loc='best')
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_png)

    # Save CSV summary with per-tensor suggested chunk
    out_csv = f"{args.out_prefix}_summary.csv"
    rows = []
    for t in per_tensor_series:
        rows.append({'tensor': t, 'suggest_chunk': per_tensor_suggest.get(t, 'None') if per_tensor_suggest.get(t, None) is not None else 'None'})
    rows.append({'tensor': 'GLOBAL_RESEMBLANCE', 'suggest_chunk': global_suggest if global_suggest is not None else 'None'})
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    # Interactive HTML
    try:
        create_interactive_tensor_comparison(val_df_norm if normalize else val_df, args.interactive_out, normalize, resemblance_series,
                                             title_prefix=f'{CATEGORY_FULLNAME.get(args.category, args.category)} comparison')
        print(f"Interactive HTML saved to {args.interactive_out}")
    except Exception as e:
        print("Warning: failed to create interactive HTML:", e)

    # Print summary
    print('\nParsed tensors:')
    for t in per_tensor_series:
        print(' -', t)
    print('\nNormalization:', 'ON' if normalize else 'OFF')
    print('Global suggested minimal chunk (resemblance >= {0:.2f}%):'.format(args.accept_percent), global_suggest)
    print('Resemblance PNG saved to', out_png)
    print('CSV summary saved to', out_csv)

    if missing:
        print('\nWarnings parsing files:')
        for m in missing:
            print(' -', m)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())