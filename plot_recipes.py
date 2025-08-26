#!/usr/bin/env python3
#***************************************************************#
#** This script is part of Thireus' GGUF Tool Suite.          **#
#** plot_ppl.py a useful tensor ppl visualisation utility to  **#
#** identify tensor quantisation sensitiveness patterns.      **#
#**                                                           **#
#** ********************************************************* **#
#** --------------- Updated: Aug-26-2025 -------------------- **#
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
#** Copyright © 2025 - Thireus.         Fₐ𝒸ₜ₋𝒸ₕₑ𝒸ₖₛ? Wₕₐₜ’ₛ ₜₕₐₜ? **#
#***************************************************************#
#**PLEASE REFER TO THE README FILE FOR ADDITIONAL INFORMATION!**#
#***************************************************************#

# Requires: pip install matplotlib

import os
import re
import argparse
import matplotlib

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
    """
    data = {}
    dir_name = os.path.basename(os.path.normpath(recipe_dir))

    # 1) Collect files directly in recipe_dir (root series labeled by empty string)
    try:
        with os.scandir(recipe_dir) as it:
            for entry in it:
                if entry.is_file() and entry.name.endswith('.recipe'):
                    model, bpw, ppl = parse_filename(entry.name)
                    if None in (bpw, ppl):
                        continue
                    # root-level series: use empty string as source label
                    data.setdefault(model, {}).setdefault('', []).append((bpw, ppl))
    except FileNotFoundError:
        # If the provided recipe_dir does not exist, return empty data
        return data

    # 2) Collect files in immediate subdirectories (each subdir becomes its own series labeled by subdir name)
    try:
        for entry in os.scandir(recipe_dir):
            if entry.is_dir():
                subdir = entry.name
                subpath = entry.path
                label = subdir  # only subdir name (no dir_name prefix)
                for fname in os.listdir(subpath):
                    if not fname.endswith('.recipe'):
                        continue
                    model, bpw, ppl = parse_filename(fname)
                    if None in (bpw, ppl):
                        continue
                    data.setdefault(model, {}).setdefault(label, []).append((bpw, ppl))
                # If subdir had no recipes, nothing was added — that's handled by above logic
    except FileNotFoundError:
        # already handled; no further action
        pass

    return data

def collect_imported_data(import_file):
    # Parse additional series from pipe-delimited DB
    # Format: model_name|author|recipe_name|bpw|ppl|model_link
    imported = {}
    markers = []
    with open(import_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split('|')
            if len(parts) < 5:
                continue
            model_name, author = parts[0], parts[1]
            bpw_str, ppl_str = parts[3], parts[4]
            try:
                bpw = float(bpw_str)
                ppl = float(ppl_str)
            except ValueError:
                continue
            # Group by model and author
            imported.setdefault(model_name, {}).setdefault(author, []).append((bpw, ppl))
    return imported

def plot_data(recipe_data, recipe_dir, imported_data=None, export=False, out_dir=None):
    # Use non-interactive backend if exporting
    if export:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from collections.abc import Iterable

    dir_name = os.path.basename(os.path.normpath(recipe_dir))
    # recipe markers now only use: '+', 'x', '*'
    recipe_markers = ['+', 'x', '*', '.']
    # imported markers remain as requested
    imported_markers = ['o', 's', '^', 'D', 'v', 'p', 'h', '<', '>']

    if not export:
        plt.ion()

    # recipe_data is expected to be: { model: { source_label: [(bpw,ppl), ...], ... }, ... }
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

        plt.xlabel('Bits per weight (bpw)')
        plt.ylabel('Perplexity (ppl)')
        plt.title(f'{model}: ppl vs bpw')
        # Make legend/label text smaller — but show recipe entries first (on top)
        handles = recipe_handles + imported_handles
        labels = recipe_labels + imported_labels
        plt.legend(handles, labels, fontsize='small')

        if export:
            filename = f"{model}.svg"
            out_path = os.path.join(out_dir or '.', filename)
            plt.savefig(out_path, format='svg')
            plt.close(fig)
        else:
            fig.show()

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
    parser.add_argument('--out-dir', default='.', help='Output directory for exported SVG files')
    args = parser.parse_args()

    recipe_data = collect_recipe_data(args.recipe_dir)
    imported_data = None
    if args.import_file:
        imported_data = collect_imported_data(args.import_file)

    plot_data(
        recipe_data,
        args.recipe_dir,
        imported_data=imported_data,
        export=args.export,
        out_dir=args.out_dir
    )

if __name__ == '__main__':
    main()
