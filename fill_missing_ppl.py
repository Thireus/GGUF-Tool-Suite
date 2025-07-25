#!/usr/bin/env python3
#***************************************************************#
#** This script is part of Thireus' GGUF Tool Suite.          **#
#** fill_missing_ppl.py is a tool that interpolates partial   **#
#** tensor ppl benchmarks.                                    **#
#**                                                           **#
#** ********************************************************* **#
#** --------------- Updated: Jul-10-2025 -------------------- **#
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
#** Copyright © 2025 - Thireus.        ᵢₙ𝒻ₑᵣₑₙ𝒸ₑ ₜᵢₘₑ: 𝒻ₒᵣₑᵥₑᵣ **#
#***************************************************************#
#**PLEASE REFER TO THE README FILE FOR ADDITIONAL INFORMATION!**#
#***************************************************************#

# Requires: pip install pandas numpy scipy tqdm

import os
import pandas as pd
import numpy as np
from scipy import stats, interpolate
from tqdm import tqdm
import argparse
import re

# Global counters
NB_FILLED = 0
NB_EXISTS = 0

def parse_pct(val):
    """
    Parse a percentage string (e.g., '+0.03%', '404%') into a float.
    Returns np.nan if parsing fails or if the value is 404 (treated as missing).
    """
    try:
        s = str(val).strip()
        if s.replace('%','') == '404':
            return np.nan
        return float(s.replace('%', '').replace('+', ''))
    except:
        return np.nan


def detect_classes_and_layers(columns):
    """
    Scan column names to identify tensor classes and layers.
    Groups columns matching:
      blk.{layer}.ffn_{direction}_{metric}.weight
    into classes keyed by metric, each with:
      - 'dirs': {'up': [...], 'down': [...], 'gate': [...]}
      - 'layers': sorted list of layer indices seen for that metric

    Returns:
      classes: dict(metric -> {'dirs': ..., 'layers': [...]})
      all_layers: sorted list of all layer indices
    """
    global NB_EXISTS
    pattern = re.compile(
        r"^blk\.(?P<layer>\d+)\.ffn_(?P<dir>up|down|gate)(?:_(?P<metric>\w+))?\.weight$"
    )
    classes = {}
    all_layer_set = set()
    
    for col in columns:
        m = pattern.match(col)
        if not m:
            print(f"Unmatched column: {col}")
            NB_EXISTS += 1 # We'll assume these have already been computed (if not the script won't move pass the other layers because not computed either)
            continue

        layer = int(m.group('layer'))
        dir_ = m.group('dir')
        metric = m.group('metric')
        
        all_layer_set.add(layer)
        
        if metric not in classes:
            classes[metric] = {'dirs': {'up': [], 'down': [], 'gate': []}, 'layer_set': set()}
        
        classes[metric]['dirs'][dir_].append((layer, col))
        classes[metric]['layer_set'].add(layer)
    
    # Finalize structure: sort layers and columns
    for parts in classes.values():
        for d, col_tuples in parts['dirs'].items():
            parts['dirs'][d] = [c for _, c in sorted(col_tuples, key=lambda x: x[0])]
        parts['layers'] = sorted(parts.pop('layer_set'))

    all_layers = sorted(all_layer_set)
    return classes, all_layers


def fit_piecewise_linear(x, y):
    """
    Fit a two-segment piecewise linear model to (x, y).
    First segment: x <= peak_x (argmax of y), second: x >= peak_x.
    Fallbacks to global linear if segments insufficient.
    """
    x, y = np.array(x), np.array(y)
    if np.any(~np.isnan(y)):
        peak_x = x[np.nanargmax(y)]
    else:
        peak_x = x[len(x)//2]
    pred = np.empty_like(y)
    mask1 = x <= peak_x
    if mask1.sum() >= 2:
        s1, i1, *_ = stats.linregress(x[mask1], y[mask1])
        pred[mask1] = i1 + s1 * x[mask1]
    else:
        pred[mask1] = np.nan
    mask2 = x >= peak_x
    if mask2.sum() >= 2:
        s2, i2, *_ = stats.linregress(x[mask2], y[mask2])
        pred[mask2] = i2 + s2 * x[mask2]
    else:
        pred[mask2] = np.nan
    # Global fallback for NaN
    if np.isnan(pred).any():
        valid = ~np.isnan(y)
        if valid.sum() >= 2:
            s, i, *_ = stats.linregress(x[valid], y[valid])
            pred[np.isnan(pred)] = i + s * x[np.isnan(pred)]
        else:
            pred[np.isnan(pred)] = 0.0
    return pred


def fit_spline(x, y, k=3, s=None):
    """
    Fit a univariate smoothing spline of degree k to (x, y).
    Falls back to piecewise linear if not enough points or spline fails.
    """
    x, y = np.array(x), np.array(y)
    mask = ~np.isnan(y)
    if mask.sum() >= k + 1:
        try:
            spl = interpolate.UnivariateSpline(x[mask], y[mask], k=k, s=s)
            return spl(x)
        except:
            pass
    return fit_piecewise_linear(x, y)


def evaluate_methods(x, y, methods, min_points=3):
    """
    Compare interpolation methods by cross-validation:
    Mask ~20% of known points, compute MSE, select best.
    If too few known (<min_points), defaults to piecewise.

    Returns (best_name, full_prediction)
    """
    mask_known = ~np.isnan(y)
    if mask_known.sum() < min_points:
        return 'piecewise', methods['piecewise'](x, y)
    idx = np.where(mask_known)[0]
    np.random.seed(0)
    val_n = max(1, int(0.2 * idx.size))
    val_idx = np.random.choice(idx, val_n, replace=False)
    errors = {}
    for name, fn in methods.items():
        y_train = y.copy()
        y_train[val_idx] = np.nan
        pred = fn(x, y_train)
        errors[name] = np.nanmean((pred[val_idx] - y[val_idx])**2)
    # pick the key with minimal error
    best = min(errors.keys(), key=lambda k: errors[k])
    return best, methods[best](x, y)


def transform_to_pct_if_needed(df):
    ppl_columns = [col for col in df.columns if not col.startswith("QTYPE")]

    # Always filter out missing codes 404 (str or float) and '404%'
    df[ppl_columns] = df[ppl_columns].replace({"404": np.nan, "404%": np.nan, 404.0: np.nan})

    # Check for any existing percent strings
    contains_pct = df[ppl_columns].apply(
        lambda col: col.map(lambda v: isinstance(v, str) and '%' in v)
    ).values.any()
    if contains_pct:
        return df, True

    print("Transforming to percentage scale...")
    numeric_df = df[ppl_columns].copy()
    # After replacing '404', convert to float directly
    numeric_df = numeric_df.astype(float)
    baseline = numeric_df.min().min()
    print(f"Baseline PPL: {baseline}")

    transformed = ((numeric_df - baseline) / baseline) * 100

    # Prepare output with object dtype to allow strings
    transformed_output = pd.DataFrame(index=df.index, columns=ppl_columns, dtype=object)

    for col in ppl_columns:
        for i, val in enumerate(df[col]):
            if pd.isna(df.at[i, col]):
                # preserve missing
                transformed_output.at[i, col] = np.nan
            else:
                transformed_output.at[i, col] = f"{transformed.at[i, col]:.4f}%"

    # Ensure QTYPE columns are preserved and merge transformed results
    df[ppl_columns] = transformed_output
    return df, False


def process_dataframe(df):
    """
    For each row (qtype), group its PPL%% columns by metric and direction.
    Compute group mean curves (up+down, up+gate, down+gate, all three) and individual series fits.
    For each series, select the interpolation (group or individual) with minimal deviation.
    Ignore outliers >3x avg before fitting.
    Logs detailed diagnostics for each row/metric.
    """
    global NB_FILLED, NB_EXISTS

    ppl_columns = [col for col in df.columns if not col.startswith("QTYPE")]
    contains_pct = df[ppl_columns].apply(lambda col: col.map(lambda v: isinstance(v, str) and '%' in v)).values.any()
    
    classes, layers = detect_classes_and_layers(ppl_columns)
    # x = np.array(layers) # Not used anymore
    out = df.astype(object)
    methods = {'piecewise': fit_piecewise_linear, 'spline': fit_spline}

    df_iter = df.iterrows()

    for idx, row in tqdm(df_iter, total=len(df), desc="Filling rows"):
        # skip qtypes with no available values
        row_vals = np.array([parse_pct(row[c]) for c in ppl_columns])
        if np.all(np.isnan(row_vals)):
            print(f"Row {idx}: no data available, skipping")
            continue
        print(f"Row {idx}: processing classes...")
        for metric, parts in classes.items():
            dirs = parts['dirs']
            print(f"  Metric '{metric}': dirs present = {[d for d in dirs if dirs[d]]}")
            _x = parts['layers'] # Some classes can have less or more layers than others

            # parse and clean each direction
            arrs = {}
            for d, cols in dirs.items():
                if not cols:
                    continue
                arr = np.array([parse_pct(row[c]) for c in cols], dtype=float)
                avg_val = np.nanmean(arr)
                # remove extreme outliers
                outliers = arr > 3 * avg_val
                if np.any(outliers):
                    print(f"    Removing {outliers.sum()} outliers in {d}")
                    arr[outliers] = np.nan
                arrs[d] = arr
                print(f"    {d}: {np.count_nonzero(~np.isnan(arr))}/{len(arr)} known values")
            if not arrs:
                continue
            # generate mean curves for all combinations
            combos = []  # list of (name, dirs_list, mean_curve)
            all_dirs = list(arrs.keys())
            # Stack arrays into a 2D array: shape = (num_arrays, array_length)
            stacked = np.stack(list(arrs.values()))
            # Mean is used in case we face lone tensors
            mean_ppl = []
            for i in range(stacked.shape[1]):  # iterate over columns (positions)
                col = stacked[:, i]
                valid = col[~np.isnan(col)]
                if valid.size == 0:
                    mean_ppl.append(np.nan)
                else:
                    mean_ppl.append(valid.mean())
            # single individual series also considered
            for d in all_dirs:
                arr = np.asarray(arrs[d])
                if arr.size > 0 and np.all(np.isnan(arr)):
                    print(
                        f"Warning: No value for lone tensor '{dirs[d]}', attempting to use mean value of other dirs (which is the wrong approach, but we have no other option)."
                    )
                    # Replace arr values with corresponding values from mean_ppl
                    arr[:] = mean_ppl  # in-place assignment
                    arrs[d] = arr      # update the dictionary (optional, but safe)
                    best_pred = arrs[d]
                    combos.append((f"ind_{d}", [d], best_pred))
                    if arr.size > 0 and np.all(np.isnan(arr)):
                        print(
                            f"Error: Lone tensor '{dirs[d]}' cannot be interpolated, incomplete mean value of other dirs... Either exlude these tensors from the dataset or compute their ppl."
                        )
                        exit(1)
                    continue
                best_pred = methods['piecewise'](_x, arrs[d])
                combos.append((f"ind_{d}", [d], best_pred))
            # combinations of 2 and all 3
            import itertools
            for r in [2, len(all_dirs)]:
                for subset in itertools.combinations(all_dirs, r):
                    stack = np.vstack([arrs[d] for d in subset])
                    mean_curve = np.nanmean(stack, axis=0)
                    name = "mean_" + "_".join(sorted(subset))
                    combos.append((name, list(subset), mean_curve))

            # fit combos with chosen method
            fitted = {}
            for name, ds, curve in combos:
                # evaluate each curve: if individual, already pred; if mean, choose method
                if name.startswith('ind_'):
                    fitted[name] = curve
                else:
                    # select best interpolation on mean curve
                    _, pred = evaluate_methods(_x, curve, methods)
                    fitted[name] = pred
                print(f"    Combo {name}: used fitted series")

                        # for each direction pick best source
            for d, cols in dirs.items():
                arr = arrs.get(d, np.full_like(_x, np.nan))
                known = ~np.isnan(arr)
                best_name = None
                best_err = np.inf
                best_pred = None
                # compare deviations across combos that include d
                for name, ds, _ in combos:
                    if d not in ds:
                        continue
                    pred_curve = fitted[name]
                    # compute scaling factor
                    f = np.nanmedian(arr[known] / pred_curve[known]) if known.sum() else 1.0
                    err = np.nanmean((f * pred_curve[known] - arr[known])**2) if known.sum() else np.inf
                    if err < best_err:
                        best_err = err
                        best_name = name
                        best_pred = f * pred_curve
                if best_pred is None:
                    if not np.all(np.isnan(arr)):
                        best_pred = arr
                        print(f"      {d}: best source origin, mse={best_err:.4f}")
                    else:
                        # fallback to zeros if nothing selected
                        best_pred = np.zeros_like(_x)
                        print(f"      Warning: no prediction for {dirs[d]}, defaulting to zero")
                else:
                    print(f"      {d}: best source {best_name}, mse={best_err:.4f}")
                # fill
                for i, c in enumerate(cols):
                    orig = parse_pct(row[c])
                    if np.isnan(orig):
                        val = f"{best_pred[i]:.4f}%"
                        out.at[idx, c] = val
                        print(f"        filled {c} = {val}")
                        NB_FILLED += 1
                    else:
                        val = f"{orig:.4f}%"
                        out.at[idx, c] = val
                        print(f"        exists {c} = {val}")
                        NB_EXISTS += 1
    return out


def main(input_csv, output_csv=None):
    """
    Load the input CSV, process missing PPL% entries,
    and write filled output CSV. If output_csv is not provided,
    append the interpolation percentage to the input filename.
    """
    global NB_FILLED, NB_EXISTS

    print(f"Loading {input_csv}")
    df = pd.read_csv(input_csv)
    df, contains_pct = transform_to_pct_if_needed(df)

    print("Processing...")
    filled_df = process_dataframe(df)

    # Compute summary
    total = NB_FILLED + NB_EXISTS
    pct = (NB_FILLED / total * 100) if total > 0 else 0

    # Determine output filename
    if output_csv is None:
        base, ext = os.path.splitext(input_csv)
        output_csv = f"{base}_{int(round(pct)):02d}percent_interpolated{ext}"

    print(f"Writing to {output_csv}")
    filled_df.to_csv(output_csv, index=False)

    # Display summary
    print(f"Interpolated {NB_FILLED}/{total} ({pct:.1f}%)")
    print("Done.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Interpolate missing PPL% in CSV files.")
    parser.add_argument('input_csv', help='Path to the input CSV file')
    parser.add_argument('output_csv', nargs='?', default=None,
                        help=('Optional path for the output CSV. '  
                              'If omitted, will append interpolation percentage to input filename.'))
    args = parser.parse_args()
    main(args.input_csv, args.output_csv)
