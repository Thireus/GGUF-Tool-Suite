#!/usr/bin/env bash
#***************************************************************#
#** This script is part of Thireus' GGUF Tool Suite.          **#
#** ppl_list.sh is a helper tool that collects the predicted  **#
#** ppl equations of models using model_tensor_bpw_metric.py  **#
#**                                                           **#
#** ********************************************************* **#
#** --------------- Updated: Apr-14-2026 -------------------- **#
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
#** Copyright © 2026 - Thireus.          ᵣₑₚᵣₒₘₚₜ ᵤₙₜᵢₗ ₛₐₜᵢₛ𝒻ᵢₑ𝒹 **#
#***************************************************************#
#**PLEASE REFER TO THE README FILE FOR ADDITIONAL INFORMATION!**#
#***************************************************************#

# Finds all .csv in current dir, runs the model script on those with >5 data rows,
# extracts the "y = ..." equation from stdout and writes model_name:equation to ppl_list.txt

set -u

OUTFILE="ppl_list.txt"
CMD="../model_tensor_bpw_metric.py"

# sanity check: script exists
if [ ! -f "$CMD" ]; then
  echo "Error: command not found at $CMD" >&2
  exit 1
fi

# gather CSV files (handles spaces in names), separate _no-others.csv to prioritize
shopt -s nullglob
files_no_others=( *_no-others.csv )
files_all=( *.csv )
shopt -u nullglob

# combine, no_others first, avoiding duplicates
files=()
for f in "${files_no_others[@]}"; do
  files+=( "$f" )
done
for f in "${files_all[@]}"; do
  if [[ "$f" == *_no-others.csv ]]; then
    continue
  fi
  files+=( "$f" )
done

if [ ${#files[@]} -eq 0 ]; then
  echo "No CSV files found in current directory."
  exit 0
fi

# truncate/overwrite output file
: > "$OUTFILE"

# Keep track of models we have already successfully processed
declare -A processed_models

for f in "${files[@]}"; do
  # Determine model name and fallback file
  if [[ "$f" == *_no-others.csv ]]; then
    model_name="${f%_no-others.csv}"
    fallback_f="${model_name}.csv"
  else
    model_name="${f%.*}"
    fallback_f=""
  fi

  # If we already have an equation for this model, skip
  if [[ -n "${processed_models[$model_name]:-}" ]]; then
    continue
  fi

  cur_f="$f"

  while true; do
    # count data rows (exclude header)
    # read header line (safe for files with spaces in name)
    if ! IFS= read -r header_line < "$cur_f"; then
      rows=0
    else
      # count data rows (tail + wc; no awk)
      rows=$(tail -n +2 -- "$cur_f" | wc -l | tr -d '[:space:]')
      # fallback if wc output is weird
      if ! [[ $rows =~ ^[0-9]+$ ]]; then
        rows=0
      fi
    fi

    # ensure at least 6 data rows
    if [ "$rows" -le 5 ]; then
      echo "Skipping '$cur_f' (only $rows data rows)"
      if [ -n "$fallback_f" ] && [ -f "$fallback_f" ]; then
        echo "Falling back to '$fallback_f'"
        cur_f="$fallback_f"
        fallback_f=""
        continue
      fi
      break
    fi

    # find bpw column index (0-based) from header (no awk)
    IFS=',' read -r -a header_cols <<< "$header_line"
    bpw_idx=-1
    for i in "${!header_cols[@]}"; do
      col=$(printf '%s' "${header_cols[$i]}" | tr -d '[:space:]' | tr -d '"')
      if [ "$col" = "bpw" ]; then
        bpw_idx=$i
        break
      fi
    done

    if [ "$bpw_idx" -lt 0 ]; then
      echo "Skipping '$cur_f' (no bpw column in header)"
      if [ -n "$fallback_f" ] && [ -f "$fallback_f" ]; then
        echo "Falling back to '$fallback_f'"
        cur_f="$fallback_f"
        fallback_f=""
        continue
      fi
      break
    fi

    # check that at least one bpw value (in data rows) is > 4 (use bc for float compare)
    has_gt4=0
    # read data rows and test the bpw column
    while IFS= read -r line; do
      [ -z "$line" ] && continue
      IFS=',' read -r -a fields <<< "$line"
      val=${fields[$bpw_idx]:-}
      # strip quotes and whitespace
      val=$(printf '%s' "$val" | tr -d '"' | tr -d '[:space:]')
      [ -z "$val" ] && continue
      # use bc for floating-point comparison; invalid values yield 0
      cmp=$(echo "$val > 4" | bc -l 2>/dev/null || echo 0)
      if [ "$cmp" = "1" ]; then
        has_gt4=1
        break
      fi
    done < <(tail -n +2 -- "$cur_f")

    if [ "$has_gt4" -ne 1 ]; then
      echo "Skipping '$cur_f' (no bpw > 4 in data rows)"
      if [ -n "$fallback_f" ] && [ -f "$fallback_f" ]; then
        echo "Falling back to '$fallback_f'"
        cur_f="$fallback_f"
        fallback_f=""
        continue
      fi
      break
    fi

    echo "Processing '$cur_f' ($rows rows)..."

    # Try with progressively fewer filters if no equation is found
    # Attempt 1: original flags (--ignore-bpw-below 1.8 --ignore-ppl-above 20)
    # Attempt 2: without --ignore-ppl-above
    # Attempt 3: without --ignore-bpw-below and --ignore-ppl-above
    equation=""
    for attempt in 1 2 3; do
      case $attempt in
        1) extra_flags="--ignore-bpw-below 1.8 --ignore-ppl-above 20.5" ;;
        2) extra_flags="--ignore-bpw-below 1.8" ;;
        3) extra_flags="" ;;
      esac

      # run command, capture stdout (suppress stderr). continue on non-zero exit.
      output=$("$CMD" --recipe-results-csv "$cur_f" --metric-name "perplexity" --d-from-lowest 1 --c-free --transforms "identity" --ignore-outliers 35 --p-grid-max 15 --p-grid-steps 100 --penalize-above 15 --drift-below 15 --resemblance-metric "abs_mean" $extra_flags --equation-only 2>/dev/null) || true

      # extract first line that looks like "y = ..."
      equation=$(printf '%s\n' "$output" | grep -m1 -E '^\s*y\s*=' | sed 's/^[[:space:]]*//')

      # If attempt 1 or 2 yields an equation, check that the lowest bpw data point is not below the curve
      if [ -n "$equation" ] && [ "$attempt" -le 2 ]; then
        # find perplexity column index (0-based) from header
        ppl_idx=-1
        for i in "${!header_cols[@]}"; do
          col=$(printf '%s' "${header_cols[$i]}" | tr -d '[:space:]' | tr -d '"')
          if [ "$col" = "perplexity" ] || [ "$col" = "ppl" ]; then
            ppl_idx=$i
            break
          fi
        done

        if [ "$ppl_idx" -ge 0 ]; then
          attempt_failed=0
          rhs="${equation#*=}"
          # Adjust common math syntax for Python compatibility
          rhs=$(printf '%s' "$rhs" | sed 's/\^/**/g')
          
          min_x=""
          min_y=""
          while IFS= read -r line; do
            [ -z "$line" ] && continue
            IFS=',' read -r -a fields <<< "$line"
            x_val=${fields[$bpw_idx]:-}
            y_val=${fields[$ppl_idx]:-}
            # strip quotes and whitespace
            x_val=$(printf '%s' "$x_val" | tr -d '"' | tr -d '[:space:]')
            y_val=$(printf '%s' "$y_val" | tr -d '"' | tr -d '[:space:]')
            [ -z "$x_val" ] || [ -z "$y_val" ] && continue

            if [ -z "$min_x" ]; then
              min_x="$x_val"
              min_y="$y_val"
            else
              # compare x_val < min_x
              cmp=$(echo "$x_val < $min_x" | bc -l 2>/dev/null || echo 0)
              if [ "$cmp" = "1" ]; then
                min_x="$x_val"
                min_y="$y_val"
              fi
            fi
          done < <(tail -n +2 -- "$cur_f")

          if [ -n "$min_x" ] && [ -n "$min_y" ]; then
            # Evaluate equation at min_x
            y_eq=$(python3 -c "from math import *; x=${min_x}; print(${rhs})" 2>/dev/null)

            if [ -n "$y_eq" ]; then
              # Compare min_y < y_eq
              cmp=$(echo "$min_y < $y_eq" | bc -l 2>/dev/null || echo 0)
              if [ "$cmp" = "1" ]; then
                attempt_failed=1
                echo "Attempt $attempt failed: lowest bpw data point (bpw=$min_x, ppl=$min_y) is below equation (y=$y_eq)"
              fi
            fi
          fi

          if [ "$attempt_failed" -eq 1 ]; then
            equation=""
          fi
        fi
      fi

      if [ -n "$equation" ]; then
        break
      fi

      echo "No equation found for '$cur_f' (attempt $attempt/3)"
    done

    if [ -n "$equation" ]; then
      printf '%s:%s\n' "$model_name" "$equation" >> "$OUTFILE"
      echo "Saved: $model_name"
      processed_models[$model_name]=1
      break
    else
      echo "No equation found for '$cur_f'"
      if [ -n "$fallback_f" ] && [ -f "$fallback_f" ]; then
        echo "Falling back to '$fallback_f'"
        cur_f="$fallback_f"
        fallback_f=""
        continue
      fi
      echo "Skipping '$model_name' completely"
      break
    fi
  done
done

echo "Done. Results written to $OUTFILE"
