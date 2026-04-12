#!/usr/bin/env python3
#***************************************************************#
#** This script is part of Thireus' GGUF Tool Suite.          **#
#** plot_recipes.py a tool that plots model recipe perplexity **#
#** results on a graph and saves them as svg and csv files.   **#
#**                                                           **#
#** ********************************************************* **#
#** --------------- Updated: Apr-13-2026 -------------------- **#
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
#** Copyright © 2026 - Thireus.         Fₐ𝒸ₜ₋𝒸ₕₑ𝒸ₖₛ? Wₕₐₜ’ₛ ₜₕₐₜ? **#
#***************************************************************#
#**PLEASE REFER TO THE README FILE FOR ADDITIONAL INFORMATION!**#
#***************************************************************#

# Requires: pip install matplotlib

import os
import re
import argparse
from pathlib import Path
import matplotlib
import csv

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

ADDITIONAL_SCALE_FACTOR_TABLE = {
    'IQ1_BN': 2,
    'IQ1_KT': 4,
    'IQ2_BN': 4,
    'IQ2_BN_R4': 4,
    'IQ2_KL': 2,
    'IQ2_KS': 2,
    'IQ2_KT': 4,
    'IQ3_KS': 2,
    'IQ3_KT': 4,
    'IQ4_KS': 4,
    'IQ4_KSS': 4,
    'IQ4_KS_R4': 4,
    'IQ4_KT': 4,
    'IQ5_KS': 4,
    'IQ5_KS_R4': 4,
    'Q8_KV': 8,
    'IQ1_S_R4': 2,
    'IQ1_M_R4': 2,
    'Q8_KV_R8': 4
}

def extract_model_name(base: str) -> str:
    """
    Extracts the model name from a filename of the form:
      <model>.<USERID>-<bpw>bpw-... 

    Examples:
      >>> extract_model_name("GLM-4.5-Air.ROOT-3.9070bpw-xyz")
      "GLM-4.5-Air"
      >>> extract_model_name("GLM-4.5.4.ROOT-1.2345bpw-...")
      "GLM-4.5.4"
    """
    # 1) chop off everything from 'bpw' onward
    head, *_ = base.partition('bpw')
    # 2) drop the trailing "-<number>" before the 'bpw'
    head = head.rsplit('-', 1)[0]
    # 3) drop the trailing ".<USERID>"
    model = head.rsplit('.', 1)[0]
    return model


def extract_recipe_author(base: str, recipe_path: str) -> str:
    """Extract the author field used by the DB export."""
    normalized = os.path.normpath(recipe_path).replace(os.sep, '/')
    if '/recipe_examples/' in f'/{normalized}':
        return 'GGUF-Tool-Suite'

    stem = re.sub(r'\.recipe(?:\.txt)?$', '', base, flags=re.IGNORECASE)
    m_bpw = re.search(r'([0-9]+\.?[0-9]*)bpw', stem, flags=re.IGNORECASE)
    if not m_bpw:
        return ''

    left = stem[:m_bpw.start()]
    if '.' not in left:
        return ''

    author_part = left.rsplit('.', 1)[1]
    author_part = re.sub(r'-\d+$', '', author_part)
    return author_part


def extract_recipe_quant(base: str) -> str:
    """Extract the quant tag used by the DB export."""
    stem = re.sub(r'\.recipe(?:\.txt)?$', '', base, flags=re.IGNORECASE)
    parts = stem.split('.')
    if parts:
        candidate = parts[-1]
        if re.fullmatch(r'[0-9a-fA-F]+_[0-9a-fA-F]+', candidate):
            return candidate
    m = re.search(r'([0-9a-fA-F]+_[0-9a-fA-F]+)(?=\.recipe(?:\.txt)?$)', base, re.IGNORECASE)
    return m.group(1) if m else ''


def build_recipe_url(recipe_path: str) -> str:
    """Build the URL field used by the DB export."""
    normalized = os.path.normpath(recipe_path).replace(os.sep, '/')
    marker = 'recipe_examples/'
    idx = normalized.find(marker)
    if idx != -1:
        rel = normalized[idx:]
        return f'https://github.com/Thireus/GGUF-Tool-Suite/blob/main/{rel}'
    try:
        return Path(recipe_path).resolve().as_uri()
    except Exception:
        return recipe_path


def parse_filename(filename):
    # Extract model name, bpw, ppl from recipe filename
    base = os.path.basename(filename)
    model_name = extract_model_name(base)
    m_bpw = re.search(r"([0-9]+\.?[0-9]*)bpw", base)
    m_ppl = re.search(r"([0-9]+\.?[0-9]*)ppl", base)
    bpw = float(m_bpw.group(1)) if m_bpw else None
    ppl = float(m_ppl.group(1)) if m_ppl else None
    return model_name, bpw, ppl


def collect_recipe_data(recipe_dir):
    """
    Collect recipe data from recipe_dir and its immediate subdirectories.

    Returns:
      data: dict mapping model -> dict mapping source_label -> list of (bpw, ppl)
        - source_label is '' (empty string) for recipes directly under recipe_dir (root series)
          or '<subdir>' for recipes inside an immediate subdirectory.
      rows: list of (MODEL_NAME, AUTHOR, QUANT, BPW, PPL, URL)
    """
    data = {}
    rows = []
    dir_name = os.path.basename(os.path.normpath(recipe_dir))

    # 1) Collect files directly in recipe_dir (root series labeled by empty string)
    try:
        with os.scandir(recipe_dir) as it:
            for entry in it:
                if entry.is_file() and (entry.name.endswith('.recipe') or entry.name.endswith('.recipe.txt')):
                    model, bpw, ppl = parse_filename(entry.name)
                    if None in (bpw, ppl):
                        continue
                    # root-level series: use empty string as source label
                    data.setdefault(model, {}).setdefault('', []).append((bpw, ppl))
                    rows.append((
                        model,
                        extract_recipe_author(entry.name, entry.path),
                        extract_recipe_quant(entry.name),
                        bpw,
                        ppl,
                        build_recipe_url(entry.path),
                    ))
    except FileNotFoundError:
        # If the provided recipe_dir does not exist, return empty data
        return data, rows

    # 2) Collect files in immediate subdirectories (each subdir becomes its own series labeled by subdir name)
    try:
        for entry in os.scandir(recipe_dir):
            if entry.is_dir():
                subdir = entry.name
                subpath = entry.path
                label = subdir  # only subdir name (no dir_name prefix)
                for fname in os.listdir(subpath):
                    if not fname.endswith('.recipe') and not fname.endswith('.recipe.txt'):
                        continue
                    model, bpw, ppl = parse_filename(fname)
                    if None in (bpw, ppl):
                        continue
                    data.setdefault(model, {}).setdefault(label, []).append((bpw, ppl))
                    full_path = os.path.join(subpath, fname)
                    rows.append((
                        model,
                        extract_recipe_author(fname, full_path),
                        extract_recipe_quant(fname),
                        bpw,
                        ppl,
                        build_recipe_url(full_path),
                    ))
                # If subdir had no recipes, nothing was added — that's handled by above logic
    except FileNotFoundError:
        # already handled; no further action
        pass

    return data, rows


def collect_imported_data(import_file):
    # Parse additional series from pipe-delimited DB
    # Format: model_name|author|recipe_name|bpw|ppl|model_link
    imported = {}
    rows = []
    with open(import_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split('|')
            if len(parts) < 5:
                continue
            model_name, author = parts[0], parts[1]
            recipe_name = parts[2] if len(parts) > 2 else ''
            bpw_str, ppl_str = parts[3], parts[4]
            model_link = parts[5] if len(parts) > 5 else ''
            try:
                bpw = float(bpw_str)
                ppl = float(ppl_str)
            except ValueError:
                continue
            # Group by model and author
            imported.setdefault(model_name, {}).setdefault(author, []).append((bpw, ppl))
            rows.append((model_name, author, recipe_name, bpw, ppl, model_link))
    return imported, rows

# Helper: map a numeric bpw to the nearest QTYPE from BPW_TABLE with filtering and tie-break rules.
def map_bpw_to_qtype(bpw):
    """
    Map a numeric bpw to the nearest QTYPE key from BPW_TABLE.

    Important rules implemented:
    - Filter out keys that begin with f (case-insensitive).
    - Filter out keys that end with _bn, _kv or _r<digit> (case-insensitive).
    - Prefer non-IQ variants over IQ when distances are equal.
    - When several candidates share the same base bpw, prefer:
      * no additional scale factor first when the target bpw is at or below that base bpw
      * the highest additional scale factor first when the target bpw is above that base bpw
    - Return the selected nearest QTYPE formatted in lower case, except:
      * if it begins with 'q' and ends with '_k' -> format as e.g. 'q4_K'
      * if it begins with 'q' and ends with '_kv' -> format as e.g. 'q8_KV'
    """
    if bpw is None:
        return ''  # fallback empty

    # prepare regex for filtering prefixes (f)
    prefix_filter_re = re.compile(r'^f', re.IGNORECASE)
    # prepare regex for filtering suffixes (_bn, _kv, _r[0-9]+)
    suffix_filter_re = re.compile(r'_(bn|kv|r\d+)$', re.IGNORECASE)

    def candidate_extra_scale_factor(qtype_key: str) -> float:
        return float(ADDITIONAL_SCALE_FACTOR_TABLE.get(qtype_key.upper(), 0.0))

    def candidate_has_extra_scale_factor(qtype_key: str) -> bool:
        return qtype_key.upper() in ADDITIONAL_SCALE_FACTOR_TABLE

    candidates = []
    for key, val in BPW_TABLE.items():
        # skip filtered keys
        if prefix_filter_re.search(key):
            continue
        if suffix_filter_re.search(key):
            continue
        diff = abs(bpw - float(val))
        candidates.append((diff, key, float(val)))

    if not candidates:
        # If nothing remains after filtering, fall back to all keys (no filter)
        for key, val in BPW_TABLE.items():
            diff = abs(bpw - float(val))
            candidates.append((diff, key, float(val)))

    if not candidates:
        return ''

    # Find the smallest distance first
    best_diff = min(diff for diff, _, _ in candidates)
    best_candidates = [(key, val) for diff, key, val in candidates if abs(diff - best_diff) <= 1e-9]

    # If all best candidates share the same base bpw, use the additional-scale tie-break logic.
    base_values = {val for _, val in best_candidates}
    if len(best_candidates) > 1 and len(base_values) == 1:
        base_val = next(iter(base_values))
        # For bpw above the base, prefer candidates with the highest additional scale factor.
        # For bpw below or equal to the base, prefer candidates with the lowest additional scale factor.
        if bpw > base_val:
            best_candidates = sorted(
                best_candidates,
                key=lambda kv: (
                    -candidate_extra_scale_factor(kv[0]),
                    0 if not candidate_has_extra_scale_factor(kv[0]) else 1,
                    kv[0]
                )
            )
        else:
            best_candidates = sorted(
                best_candidates,
                key=lambda kv: (
                    candidate_extra_scale_factor(kv[0]),
                    0 if not candidate_has_extra_scale_factor(kv[0]) else 1,
                    kv[0]
                )
            )
    else:
        # Keep prior preference: non-IQ over IQ when equal distance, then lexicographic.
        if len(best_candidates) > 1:
            non_iq = [k for k, _ in best_candidates if not k.upper().startswith('IQ')]
            if non_iq:
                chosen = sorted(non_iq)[0]
                chosen_val = next(val for k, val in best_candidates if k == chosen)
                best_candidates = [(chosen, chosen_val)]
            else:
                best_candidates = sorted(best_candidates, key=lambda kv: kv[0])

    chosen = best_candidates[0][0]

    # format chosen according to rules: lower case, but special _K/_KV uppercase if begins with q
    chosen_lower = chosen.lower()
    if chosen_lower.startswith('q') and chosen_lower.endswith('_k'):
        # e.g. chosen 'Q4_K' -> result 'q4_K'
        prefix = chosen_lower[:-2]  # drop '_k'
        return f"{prefix}_K"
    if chosen_lower.startswith('q') and chosen_lower.endswith('_kv'):
        prefix = chosen_lower[:-3]  # drop '_kv'
        return f"{prefix}_KV"
    # default: lower case
    return chosen_lower


def plot_data(recipe_data, recipe_rows, recipe_dir, imported_data=None, imported_rows=None, export=False, out_dir=None, export_csv=False, export_db=False, export_csv_exclude_others=False):
    # Use non-interactive backend if exporting
    if export:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from collections.abc import Iterable

    dir_name = os.path.basename(os.path.normpath(recipe_dir))
    # recipe markers now only use: '+', 'x', '*'
    recipe_markers = ['x', '+', '*', '.']
    # imported markers remain as requested
    imported_markers = ['o', 's', '^', 'D', 'v', 'p', 'h', '<', '>']

    if not export:
        plt.ion()

    # recipe_data is expected to be: { model: { source_label: [(bpw,ppl), ...], ... }, ... }
    db_rows = []
    if imported_rows:
        db_rows.extend(imported_rows)
    if recipe_rows:
        db_rows.extend(recipe_rows)

    for model, sources in recipe_data.items():
        # If a model has no sources (shouldn't happen), skip
        if not sources:
            continue

        fig = plt.figure()

        # prepare color cycle so recipe series can reserve the "first" colours
        prop_cycle = plt.rcParams.get('axes.prop_cycle')
        color_cycle = []

        # defensive extraction of colors from prop_cycle/by_key()
        try:
            if prop_cycle is not None:
                by_key = getattr(prop_cycle, 'by_key', None)
                if callable(by_key):
                    try:
                        by_key_res = by_key()
                    except Exception:
                        by_key_res = None
                else:
                    by_key_res = None

                # 1) if it's a dict-like, use .get safely
                if isinstance(by_key_res, dict):
                    try:
                        vals = by_key_res.get('color')
                    except Exception:
                        vals = None

                    if isinstance(vals, (list, tuple)):
                        color_cycle = list(vals)
                    elif isinstance(vals, str):
                        color_cycle = [vals]
                    elif isinstance(vals, Iterable):
                        try:
                            color_cycle = list(vals)
                        except Exception:
                            # coerce elements to string as a last resort
                            try:
                                color_cycle = [str(x) for x in vals]
                            except Exception:
                                color_cycle = []
                    else:
                        color_cycle = []

                # 2) if it exposes a .get method (mapping-like), try calling it guarded
                elif by_key_res is not None and hasattr(by_key_res, 'get'):
                    get_fn = getattr(by_key_res, 'get', None)
                    if callable(get_fn):
                        try:
                            vals = get_fn('color')
                        except Exception:
                            vals = None

                        if isinstance(vals, (list, tuple)):
                            color_cycle = list(vals)
                        elif isinstance(vals, str):
                            color_cycle = [vals]
                        elif isinstance(vals, Iterable):
                            try:
                                color_cycle = list(vals)
                            except Exception:
                                try:
                                    color_cycle = [str(x) for x in vals]
                                except Exception:
                                    color_cycle = []
                        else:
                            color_cycle = []

                # 3) fallback: try to treat by_key_res as an iterable of strings (guarded)
                elif by_key_res is not None and isinstance(by_key_res, Iterable):
                    try:
                        seq = []
                        for x in by_key_res:
                            seq.append(x)
                        if seq and all(isinstance(x, str) for x in seq):
                            color_cycle = seq
                        else:
                            # coerce to strings if necessary
                            color_cycle = [str(x) for x in seq] if seq else []
                    except Exception:
                        color_cycle = []
        except Exception:
            color_cycle = []

        # final fallback to Matplotlib color names if nothing found
        if not color_cycle:
            color_cycle = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']

        num_recipe_series = len(sources)
        recipe_handles = []
        recipe_labels = []
        imported_handles = []
        imported_labels = []

        # collect CSV rows for this model if requested
        csv_rows = []  # list of (QTYPE, bpw, ppl)
        csv_no_others_rows = []  # list of (QTYPE, bpw, ppl) excluding imported series, except when author is "pure"

        # Plot any imported series for this model FIRST so they appear behind recipe series
        if imported_data and model in imported_data:
            for idx, (author, series_vals) in enumerate(imported_data[model].items()):
                if not series_vals:
                    continue
                xs_imp, ys_imp = zip(*sorted(series_vals))
                marker = imported_markers[idx % len(imported_markers)]
                # pick imported colours *after* the reserved recipe colours so recipe keeps the first colours
                color = color_cycle[(num_recipe_series + idx) % len(color_cycle)]
                # zorder lower so imported points are drawn behind recipe points
                line = plt.plot(xs_imp, ys_imp, marker=marker, markersize=3, linestyle='', label=author, zorder=1, color=color, alpha=0.75)[0]
                imported_handles.append(line)
                imported_labels.append(author)
                # add to csv rows: map bpw to nearest QTYPE
                for x, y in zip(xs_imp, ys_imp):
                    qtype = map_bpw_to_qtype(x)
                    csv_rows.append((qtype, x, y))
                    if author == "pure":
                        csv_no_others_rows.append((qtype, x, y))

        # plot each source series (root and each subdir) for this model (drawn after imported -> on top)
        for idx, (source_label, vals) in enumerate(sorted(sources.items())):
            if not vals:
                continue
            xs, ys = zip(*sorted(vals))
            # series_label: omit dir_name; empty source_label -> no suffix, subdir -> use subdir only
            if source_label:
                series_label = f"Thireus' GGUF Tool Suite {source_label}"
            else:
                series_label = "Thireus' GGUF Tool Suite"
            # Use recipe-specific markers: '+', 'x', '*'
            marker = recipe_markers[idx % len(recipe_markers)]
            # recipe colours are the *first* colours in the cycle so they match original look
            color = color_cycle[idx % len(color_cycle)]
            # zorder higher so recipe points are on top
            line = plt.plot(xs, ys, marker=marker, linestyle='', label=series_label, zorder=2, color=color)[0]
            recipe_handles.append(line)
            recipe_labels.append(series_label)
            # add to csv rows: map bpw to nearest QTYPE
            for x, y in zip(xs, ys):
                qtype = map_bpw_to_qtype(x)
                csv_rows.append((qtype, x, y))
                csv_no_others_rows.append((qtype, x, y))

        plt.xlabel('Bits per weight (bpw)')
        plt.ylabel('Perplexity (ppl)')
        plt.title(f'{model}: ppl vs bpw')
        # Make legend/label text smaller — but show recipe entries first (on top)
        handles = recipe_handles + imported_handles
        labels = recipe_labels + imported_labels
        plt.legend(handles, labels, fontsize='small')

        # Ensure out_dir exists if we will export files
        if (export or export_csv or export_db or export_csv_exclude_others) and out_dir:
            try:
                os.makedirs(out_dir, exist_ok=True)
            except Exception:
                pass

        if export:
            filename = f"{model}.svg"
            out_path = os.path.join(out_dir or '.', filename)
            plt.savefig(out_path, format='svg')
            plt.close(fig)
        else:
            fig.show()

        # Write CSV for this model if requested
        if export_csv:
            csv_filename = f"{model}.csv"
            csv_path = os.path.join(out_dir or '.', csv_filename)
            try:
                with open(csv_path, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['QTYPE', 'bpw', 'ppl'])
                    for row in sorted(csv_rows, key=lambda r: r[1], reverse=True):
                        # row is (QTYPE, bpw, ppl)
                        writer.writerow([row[0], row[1], row[2]])
            except Exception as e:
                # If writing fails, print a warning but continue
                print(f"Warning: failed to write CSV {csv_path}: {e}")

        # Write CSV for this model excluding imported entries if requested
        if export_csv_exclude_others and csv_no_others_rows:
            csv_filename = f"{model}_no-others.csv"
            csv_path = os.path.join(out_dir or '.', csv_filename)
            try:
                with open(csv_path, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['QTYPE', 'bpw', 'ppl'])
                    for row in sorted(csv_no_others_rows, key=lambda r: r[1], reverse=True):
                        # row is (QTYPE, bpw, ppl)
                        writer.writerow([row[0], row[1], row[2]])
            except Exception as e:
                # If writing fails, print a warning but continue
                print(f"Warning: failed to write CSV {csv_path}: {e}")

    if export_db:
        db_filename = 'all_ppl.db'
        db_path = os.path.join(out_dir or '.', db_filename)
        try:
            with open(db_path, 'w', newline='') as dbfile:
                writer = csv.writer(dbfile, delimiter='|', lineterminator='\n')
                for row in db_rows:
                    writer.writerow([row[0], row[1], row[2], row[3], row[4], row[5]])
        except Exception as e:
            print(f"Warning: failed to write DB {db_path}: {e}")

    if not export:
        print("Plots are displayed in separate windows. Close them manually when done.")
        input("Press Enter to exit and close all plots...")
        plt.close('all')


def main():
    parser = argparse.ArgumentParser(
        description='Plot ppl vs bpw from recipe filenames, with optional imported series.')
    parser.add_argument('recipe_dir', help='Directory containing .recipe files')
    parser.add_argument('--import', dest='import_file', help='Import additional series from DB file')
    parser.add_argument('--export', action='store_true', help='Export plots as SVG without rendering')
    parser.add_argument('--export-csv', action='store_true', help='Export CSV files with plotted points for each model (same base name as SVG)')
    parser.add_argument('--export-csv-exclude-others', action='store_true', help='Export CSV files with plotted points for each model excluding imported entries (same base name as SVG but with _no-others.csv)')
    parser.add_argument('--export-db', action='store_true', help='Export a combined all_ppl.db file with all plotted entries')
    parser.add_argument('--out-dir', default='.', help='Output directory for exported SVG and CSV files')
    args = parser.parse_args()

    recipe_data, recipe_rows = collect_recipe_data(args.recipe_dir)
    imported_data = None
    imported_rows = None
    if args.import_file:
        imported_data, imported_rows = collect_imported_data(args.import_file)

    plot_data(
        recipe_data,
        recipe_rows,
        args.recipe_dir,
        imported_data=imported_data,
        imported_rows=imported_rows,
        export=args.export,
        out_dir=args.out_dir,
        export_csv=args.export_csv,
        export_db=args.export_db,
        export_csv_exclude_others=args.export_csv_exclude_others
    )

if __name__ == '__main__':
    main()
