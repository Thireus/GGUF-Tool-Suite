#!/usr/bin/env bash
#***************************************************************#
#** This script is part of Thireus' GGUF Tool Suite.          **#
#** ppl_list.sh is a helper tool that collects the predicted  **#
#** ppl equations of models using model_tensor_bpw_metric.py  **#
#**                                                           **#
#** ********************************************************* **#
#** --------------- Updated: Jun-06-2026 -------------------- **#
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

    # If this is a _no-others.csv file, require more than 2 points with bpw < 4.
    # Otherwise, the base CSV takes over completely.
    if [[ "$cur_f" == *_no-others.csv ]]; then
      below4_count=0
      while IFS= read -r line; do
        [ -z "$line" ] && continue
        IFS=',' read -r -a fields <<< "$line"

        x_val=${fields[$bpw_idx]:-}
        x_val=$(printf '%s' "$x_val" | tr -d '"' | tr -d '[:space:]')
        [ -z "$x_val" ] && continue

        if [ "$(awk -v x="$x_val" 'BEGIN { print (x < 4) ? 1 : 0 }')" = "1" ]; then
          below4_count=$((below4_count + 1))
        fi
      done < <(tail -n +2 -- "$cur_f")

      if [ "$below4_count" -le 2 ] && [ -n "$fallback_f" ] && [ -f "$fallback_f" ]; then
        echo "Skipping '$cur_f' (only $below4_count points with bpw < 4, need >2); base CSV takes over"
        cur_f="$fallback_f"
        fallback_f=""
        continue
      fi
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

    # Find a column index by header name
    find_col_idx() {
      local wanted1=$1
      local wanted2=$2
      local idx=-1
      local i col

      for i in "${!header_cols[@]}"; do
        col=$(printf '%s' "${header_cols[$i]}" | tr -d '[:space:]"')
        if [ "$col" = "$wanted1" ] || [ "$col" = "$wanted2" ]; then
          idx=$i
          break
        fi
      done

      printf '%s' "$idx"
    }

    # Find perplexity column index (0-based) from header
    ppl_idx=$(find_col_idx perplexity ppl)

    # Try with progressively fewer filters if no equation is found
    # Attempt 1: original flags (--ignore-bpw-below 1.8 --ignore-ppl-above 20)
    # Attempt 2: without --ignore-ppl-above
    # Attempt 3: without --ignore-bpw-below and --ignore-ppl-above

    # Evaluate the RHS at a given x using awk.
    # Awk handles scientific notation and ^.
    eval_rhs() {
      local x_val=$1
      local rhs=$2

      echo "[DEBUG] x = $x_val" >&2
      echo "[DEBUG] rhs = $rhs" >&2

      # Build awk program separately for clarity
      local awk_prog="BEGIN { print ($rhs) }"
      echo "[DEBUG] awk program = $awk_prog" >&2

      # Capture result
      local y_val
      y_val=$(awk -v x="$x_val" "$awk_prog")

      echo "[DEBUG] y = $y_val" >&2

      # Return value
      printf '%s\n' "$y_val"
    }

    # The fitted curve represents the *theoretical optimum* (lower envelope) of ppl
    # vs bpw, so it must sit at or just below the data. We reject a candidate only
    # when its curve rises meaningfully ABOVE the data's lower envelope: for every
    # data point we compare the curve to the minimum ppl observed within a small bpw
    # window (so duplicate/noisy measurements at the same bpw don't force a rejection
    # when the curve already hugs the better recipe), allowing a small tolerance.
    # Drooping below the data is always allowed (that is what a lower bound does).
    validate_equation() {
      local equation=$1
      local rhs ret
      local win=0.05   # bpw window used to build the local lower envelope
      local tol=0.05   # curve may sit up to 5% above the local envelope minimum

      rhs="${equation#*=}"
      rhs=$(printf '%s' "$rhs" | sed 's/^[[:space:]]*//')

      # Single awk pass: load (bpw, ppl), then for each point compare the curve to
      # the minimum ppl within +/-win bpw. exit 0 = ok, 1 = overshoot, 2 = nan/inf.
      awk -F',' -v bcol=$((bpw_idx + 1)) -v pcol=$((ppl_idx + 1)) \
              -v win="$win" -v tol="$tol" '
        function curve(x) { return ('"$rhs"') }
        NR == 1 { next }
        {
          b = $bcol; p = $pcol
          gsub(/["[:space:]\r]/, "", b); gsub(/["[:space:]\r]/, "", p)
          if (b == "" || p == "") next
          n++; X[n] = b + 0; Y[n] = p + 0
        }
        END {
          for (i = 1; i <= n; i++) {
            lm = Y[i]
            for (j = 1; j <= n; j++) {
              d = X[j] - X[i]; if (d < 0) d = -d
              if (d <= win && Y[j] < lm) lm = Y[j]
            }
            cv = curve(X[i])
            # Reject non-finite results. NaN is matched as a string because some awk
            # builds (BWK/macOS) evaluate (cv != cv) as false for NaN; the finite-range
            # test additionally catches +/-inf (and NaN) on IEEE awks.
            if ((cv "") ~ /[nN][aA][nN]/ || !(cv > -1e300 && cv < 1e300)) { exit 2 }
            if (cv > lm * (1.0 + tol)) {
              printf "Attempt failed: curve (y=%g) above lower envelope (bpw=%g, envmin=%g) beyond %g tol\n", cv, X[i], lm, tol > "/dev/stderr"
              exit 1
            }
          }
          exit 0
        }
      ' "$cur_f"
      ret=$?

      [ "$ret" -eq 0 ]
    }

    # Detect a degenerate (flat / "wall at the edge") fit. The curve d + a*(x-c)^(-p)
    # is only a meaningful lower envelope when its pole x=c sits safely BELOW the
    # lowest fitted bpw; then it rises smoothly through the steep low-bpw region. When
    # the good-recipe data lacks low-bpw spread (e.g. all points clustered near the
    # plateau), the fitter parks the pole at/above the data, giving a flat plateau with
    # an unconstrained vertical wall that says nothing about the region it must predict.
    # We refuse to emit such curves (the caller skips the model rather than guessing).
    # Returns 0 (true) when degenerate. Equations are always identity-form here.
    is_degenerate() {
      local equation=$1 rhs
      rhs="${equation#*=}"
      awk -F',' -v bcol=$((bpw_idx + 1)) -v eq="$rhs" -v margin=0.10 '
        BEGIN {
          if (match(eq, /\([ \t]*x[ \t]*[+-][ \t]*[0-9.eE+-]+[ \t]*\)/)) {
            term = substr(eq, RSTART, RLENGTH)
            if (term ~ /x[ \t]*\+/) sgn = 1; else sgn = -1
            n = term; sub(/.*x[ \t]*[+-][ \t]*/, "", n); sub(/[ \t]*\).*/, "", n)
            cval = n + 0
            pole = (sgn == 1) ? -cval : cval   # pole location on the x (bpw) axis
            haspole = 1
          }
        }
        NR > 1 {
          b = $bcol; gsub(/["[:space:]\r]/, "", b)
          if (b != "") { v = b + 0; if (!seen || v < m) { m = v; seen = 1 } }
        }
        END {
          if (!haspole || !seen) exit 1          # cannot determine -> treat as ok
          if (pole >= m - margin) exit 0         # pole not below data -> degenerate
          exit 1
        }
      ' "$cur_f"
    }

    score_equation() {
      local equation=$1
      local rhs sum="0" count=0 x_val y_val y_eq diff
      local line

      rhs="${equation#*=}"
      rhs=$(printf '%s' "$rhs" | sed 's/^[[:space:]]*//')

      while IFS= read -r line; do
        [ -z "$line" ] && continue
        IFS=',' read -r -a fields <<< "$line"

        x_val=${fields[$bpw_idx]:-}
        y_val=${fields[$ppl_idx]:-}

        # strip quotes and whitespace
        x_val=$(printf '%s' "$x_val" | tr -d '"' | tr -d '[:space:]')
        y_val=$(printf '%s' "$y_val" | tr -d '"' | tr -d '[:space:]')
        [ -z "$x_val" ] || [ -z "$y_val" ] && continue

        # Only score points below 4 bpw
        if [ "$(awk -v x="$x_val" 'BEGIN { print (x < 4) ? 1 : 0 }')" != "1" ]; then
          continue
        fi

        y_eq=$(eval_rhs "$x_val" "$rhs") || continue
        [ -n "$y_eq" ] || continue

        case "$y_eq" in
          *nan*|*NaN*|*inf*|*Inf*|-inf*|-Inf*)
            continue
            ;;
        esac

        diff=$(awk -v a="$y_val" -v b="$y_eq" '
          BEGIN {
            d = a - b;
            if (d != d) exit 1;       # NaN
            if (d < 0) d = -d;
            if (d != d) exit 1;       # NaN after abs
            printf "%.12f", d
          }' 2>/dev/null) || continue

        case "$diff" in
          *nan*|*NaN*|*inf*|*Inf*)
            continue
            ;;
        esac

        sum=$(awk -v s="$sum" -v d="$diff" 'BEGIN { printf "%.12f", s + d }')
        case "$sum" in
          *nan*|*NaN*|*inf*|*Inf*)
            return 1
            ;;
        esac

        count=$((count + 1))
      done < <(tail -n +2 -- "$cur_f")

      [ "$count" -gt 0 ] || return 1
      awk -v s="$sum" -v c="$count" 'BEGIN { printf "%.12f", s / c }'
    }

    fit_equation_for_outliers() {
      local outliers=$1
      local log_file=$2
      local equation="" best_eq="" best_score="" cand_score
      local attempt output extra_flags
      local cmd_args

      # The progressively-relaxed attempts differ in which low-bpw / high-ppl points
      # they exclude. Attempt 3 (no filters) keeps the full steep tail and usually
      # hugs best, but for some models a filtered attempt fits better. We therefore
      # evaluate EVERY attempt that passes validation and keep the one that hugs the
      # data closest (lowest score_equation), rather than returning the first valid
      # one -- otherwise a droopy-but-valid early attempt would win over a better fit.
      for attempt in 1 2 3; do
        case $attempt in
          1) extra_flags=(--ignore-bpw-below 1.8 --ignore-ppl-above 20.5) ;;
          2) extra_flags=(--ignore-bpw-below 1.8) ;;
          3) extra_flags=() ;;
        esac

        echo "Trying --ignore-outliers $outliers (attempt $attempt/3)..." >> "$log_file"

        cmd_args=(
          "$CMD"
          --recipe-results-csv "$cur_f"
          --metric-name "perplexity"
          --d-from-lowest 1
          --c-free
          --transforms "identity"
          --ignore-outliers "$outliers"
          --p-grid-max 20
          --p-grid-steps 130
          --penalize-above 15
          --drift-below 15
          --resemblance-metric "abs_mean"
          "${extra_flags[@]}"
          --equation-only
        )

        # run command, capture stdout (suppress stderr). continue on non-zero exit.
        output=$("${cmd_args[@]}" 2>/dev/null) || true

        # extract first line that looks like "y = ..."
        equation=$(printf '%s\n' "$output" | grep -m1 -E '^\s*y\s*=' | sed 's/^[[:space:]]*//')

        if [ -n "$equation" ]; then
          echo "  Found equation: $equation" >> "$log_file"

          if validate_equation "$equation"; then
            cand_score=$(score_equation "$equation" 2>/dev/null) || cand_score=""
            echo "  Validation passed (avg abs error on bpw<4: ${cand_score:-n/a})" >> "$log_file"
            if [ -n "$cand_score" ]; then
              if [ -z "$best_score" ] || [ "$(echo "$cand_score < $best_score" | bc -l 2>/dev/null || echo 0)" = "1" ]; then
                best_score="$cand_score"
                best_eq="$equation"
              fi
            elif [ -z "$best_eq" ]; then
              # score could not be computed but the curve is valid; keep as a fallback
              best_eq="$equation"
            fi
          else
            echo "  Validation failed: curve rises above the data's lower envelope" >> "$log_file"
          fi
        else
          echo "  No equation found" >> "$log_file"
        fi
      done

      if [ -n "$best_eq" ]; then
        echo "  Best attempt for --ignore-outliers $outliers (avg abs error on bpw<4: ${best_score:-n/a})" >> "$log_file"
        printf '%s' "$best_eq"
        return 0
      fi

      return 1
    }

    # Two parallel searches: outliers=0 keeps every (curated) data point so the steep
    # low-bpw tail is never discarded as an "outlier" (it is signal, not noise);
    # outliers=5 is a robustness fallback that drops genuine spikes for messy data.
    tmp0=$(mktemp)
    tmp5=$(mktemp)
    log0=$(mktemp)
    log5=$(mktemp)

    # Run both searches at the same time
    fit_equation_for_outliers 0 "$log0" > "$tmp0" &
    pid0=$!
    fit_equation_for_outliers 5 "$log5" > "$tmp5" &
    pid5=$!

    wait "$pid0"
    wait "$pid5"

    # Show the attempts again so they are visible in the output
    echo "---- Attempt log: --ignore-outliers 0 ----"
    cat "$log0"
    echo "---- Attempt log: --ignore-outliers 5 ----"
    cat "$log5"

    eq0=$(cat "$tmp0")
    eq5=$(cat "$tmp5")

    rm -f "$tmp0" "$tmp5" "$log0" "$log5"

    chosen_equation=""
    chosen_outliers=""
    score0=""
    score5=""

    if [ -n "$eq0" ]; then
      score0=$(score_equation "$eq0") || score0=""
    fi

    if [ -n "$eq5" ]; then
      score5=$(score_equation "$eq5") || score5=""
    fi

    if [ -n "$eq0" ] && [ -n "$score0" ] && [ -n "$eq5" ] && [ -n "$score5" ]; then
      if [ "$(echo "$score0 <= $score5" | bc -l 2>/dev/null || echo 0)" = "1" ]; then
        chosen_equation="$eq0"
        chosen_outliers=0
      else
        chosen_equation="$eq5"
        chosen_outliers=5
      fi
    elif [ -n "$eq0" ] && [ -n "$score0" ]; then
      chosen_equation="$eq0"
      chosen_outliers=0
    elif [ -n "$eq5" ] && [ -n "$score5" ]; then
      chosen_equation="$eq5"
      chosen_outliers=5
    fi

    # if everything failed, default to outliers=0 if eq0 exists
    if [ -z "$chosen_equation" ]; then
      if [ -n "$eq0" ]; then
        chosen_equation="$eq0"
        chosen_outliers=0
      elif [ -n "$eq5" ]; then
        chosen_equation="$eq5"
        chosen_outliers=5
      fi
    fi

    equation="$chosen_equation"

    # --- Fallback to _no-others.csv if nothing found ---
    if [ -z "$equation" ]; then
      fallback_csv="${cur_f%.*}_no-others.csv"

      if [ -f "$fallback_csv" ]; then
        echo "Falling back to '$fallback_csv'..."

        output=$("$CMD" \
          --recipe-results-csv "$fallback_csv" \
          --metric-name "perplexity" \
          --d-from-lowest 1 \
          --c-free \
          --transforms "identity" \
          --p-grid-max 20 \
          --p-grid-steps 130 \
          --penalize-above 15 \
          --drift-below 15 \
          --resemblance-metric "abs_mean" \
          --equation-only 2>/dev/null) || true

        fallback_eq=$(printf '%s\n' "$output" | grep -m1 -E '^\s*y\s*=' | sed 's/^[[:space:]]*//')

        if [ -n "$fallback_eq" ]; then
          echo "Using fallback equation from _no-others.csv unless no equation found, in which case the base will be used"
          equation="$fallback_eq"
        else
          echo "Fallback also failed to produce an equation"
        fi
      fi
    fi

    # Degeneracy guard: discard a flat/wall fit rather than emit a misleading curve.
    # We prefer NO equation, and we do NOT fall back to the base CSV ("others"): if the
    # toolsuite's own good recipes don't span the low-bpw region, wait for better ones.
    degenerate_skip=0
    if [ -n "$equation" ] && is_degenerate "$equation"; then
      echo "Discarding degenerate (flat) equation for '$cur_f': pole is not below the lowest bpw, i.e. insufficient low-bpw good-recipe coverage."
      equation=""
      degenerate_skip=1
    fi

    if [ -n "$equation" ]; then
      echo "Chosen equation from --ignore-outliers $chosen_outliers (avg abs error on bpw<4: ${score0:-n/a} / ${score5:-n/a})"
    else
      echo "No valid equation found for '$cur_f'"
    fi

    if [ -n "$equation" ]; then
      printf '%s:%s\n' "$model_name" "$equation" >> "$OUTFILE"
      echo "Saved: $model_name"
      processed_models[$model_name]=1
      break
    elif [ "$degenerate_skip" = "1" ]; then
      echo "Skipping '$model_name' (no reliable low-bpw fit from good recipes; not falling back to 'others')."
      # Mark processed so the base CSV (which contains 'others') is NOT tried later.
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
