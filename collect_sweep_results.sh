#!/usr/bin/env bash
#***************************************************************#
#** This script is part of Thireus' GGUF Tool Suite.          **#
#** collect_sweep_results.sh is a helper tool that collects   **#
#** the S_PP and S_TG values from bench_sweep_result.* files. **#
#**                                                           **#
#** ********************************************************* **#
#** --------------- Updated: Aug-31-2025 -------------------- **#
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
#** Copyright © 2025 - Thireus.  ₒᵤₜₚᵤₜ ₙₒₙₛₑₙₛₑ ₐₛ 𝒻ₐₛₜ ₐₛ ₚₒₛₛᵢᵦₗₑ **#
#***************************************************************#
#**PLEASE REFER TO THE README FILE FOR ADDITIONAL INFORMATION!**#
#***************************************************************#

# Exit on error, undefined variable, or pipe failure
set -euo pipefail

usage() {
  cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Options:
  --context CONTEXT                     Context value used in sweep filenames (integer). Default: 8192
  --nkv N_KV                            N_KV value to extract from sweep output (integer). Default: 0
  --baseline-pp VALUE                   Global baseline PP (t/s) value for percent-delta computation (float)
  --baseline-tg VALUE                   Global baseline TG (t/s) value for percent-delta computation (float)
  --inject-baseline-pp-qtype QTYPE      For this qtype, inject baseline PP (uses --baseline-pp if provided,
                                        otherwise tries to read bench_sweep_result.baseline.QTYPE.<CONTEXT>.txt)
  --inject-baseline-tg-qtype QTYPE      For this qtype, inject baseline TG (uses --baseline-tg if provided,
                                        otherwise tries to read bench_sweep_result.baseline.QTYPE.<CONTEXT>.txt)
  --auto-baseline QTYPE                 Automatically read bench_sweep_result.baseline.QTYPE.<CONTEXT>.txt to obtain
                                        per-qtype baseline PP and TG (only for the named qtype).
  --hide-empty                          Don't include empty benchmark results to the output csv
  --output-pp FILE                      Path to output PP CSV file (default: pp_results.csv)
  --output-tg FILE                      Path to output TG CSV file (default: tg_results.csv)
  --qtypes Q1,Q2,...                    Comma-separated list of qtypes to use (overrides auto-discovery)
  --group-tensors REG1[,REG2] [REG3,..] Specify one or more group specifications (same syntax as benchmark_each_tensor.sh).
                                        Each argument is a group: comma-separated regexes. If omitted, grouping disabled.
  --expand-groups                       When present, expand groups into individual tensor columns (default: disabled).
  -h, --help                            Show this help message and exit
EOF
}

# ======== USER CONFIGURATION (same regexes as collect_ppl_results.sh) ========

# List of tensor-name regex patterns (Bash regex) to include in the CSV.
# Adjust these as needed.
USER_REGEX=(
  # Token embedding and output tensors (GPU)
  # note token_embd cannot be repacked quant type
  '^output\.weight'
  '^token_embd\.weight'

  # GPU Only
  '^blk\.[0-2]\.ffn_down\.weight'
  '^blk\.[0-2]\.ffn_up\.weight'
  '^blk\.[0-2]\.ffn_gate\.weight'

  ## GPU-loaded ffn_*_shexp
  '^blk\.([3-9]|[1-5][0-9]|60)\.ffn_down_shexp\.weight'
  '^blk\.([3-9]|[1-5][0-9]|60)\.ffn_up_shexp\.weight'
  '^blk\.([3-9]|[1-5][0-9]|60)\.ffn_gate_shexp\.weight'

  ## CPU-loaded ffn_*_exps
  '^blk\.([3-9]|[1-5][0-9]|60)\.ffn_down_exps\.weight'
  '^blk\.([3-9]|[1-5][0-9]|60)\.ffn_up_exps\.weight'
  '^blk\.([3-9]|[1-5][0-9]|60)\.ffn_gate_exps\.weight'
)

# Default output CSV filename (can be overridden via --output-pp and --output-tg)
OUTPUT_PP_CSV="pp_results.csv"
OUTPUT_TG_CSV="tg_results.csv"
# ===========================================================================

# Initialize variables
CONTEXT=8192
N_KV=0
BASELINE_PP=""    # global PP baseline
BASELINE_TG=""    # global TG baseline
BASELINE_PP_QTYPE=""  # qtype to inject PP baseline (or read baseline file if global baseline not provided)
BASELINE_TG_QTYPE=""  # qtype to inject TG baseline
AUTO_BASELINE_QTYPE="" # qtype to auto-read bench_sweep_result.baseline.* file
HIDE_EMPTY=false
qtypes=""
GROUP_TENSORS_RAW=()
GROUP_TENSORS_DISABLED=true
EXPAND_GROUPS=false

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --context)
      shift; CONTEXT="$1"; shift;;
    --nkv)
      shift; N_KV="$1"; shift;;
    --baseline-pp)
      shift; BASELINE_PP="$1"; shift;;
    --baseline-tg)
      shift; BASELINE_TG="$1"; shift;;
    --inject-baseline-pp-qtype)
      shift; BASELINE_PP_QTYPE="$1"; shift;;
    --inject-baseline-tg-qtype)
      shift; BASELINE_TG_QTYPE="$1"; shift;;
    --auto-baseline)
      shift; AUTO_BASELINE_QTYPE="$1"; shift;;
    --hide-empty)
      HIDE_EMPTY=true; shift;;
    --output-pp)
      shift; OUTPUT_PP_CSV="$1"; shift;;
    --output-tg)
      shift; OUTPUT_TG_CSV="$1"; shift;;
    --qtypes)
      shift; qtypes="$1"; shift;;
    --group-tensors)
      shift
      GROUP_TENSORS_RAW=()
      # collect one or more group specs (nargs '+')
      while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
        GROUP_TENSORS_RAW+=("$1")
        shift
      done
      ;;
    --expand-groups)
      EXPAND_GROUPS=true; shift;;
    -h|--help)
      usage; exit 0;;
    *)
      echo "Unknown option: $1" >&2; usage; exit 1;;
  esac
done

# If user explicitly passed single token '[]', treat as disabled (mirrors benchmark_each_tensor.sh behaviour)
if (( ${#GROUP_TENSORS_RAW[@]} == 0 )) || ( (( ${#GROUP_TENSORS_RAW[@]} == 1 )) && [[ "${GROUP_TENSORS_RAW[0]}" == "[]" ]] ); then
  GROUP_TENSORS_DISABLED=true
else
  GROUP_TENSORS_DISABLED=false
fi

# Validate numeric args
if ! [[ "$CONTEXT" =~ ^[0-9]+$ ]]; then
  echo "Error: --context must be an integer." >&2; exit 1
fi
if ! [[ "$N_KV" =~ ^[0-9]+$ ]]; then
  echo "Error: --nkv must be an integer." >&2; exit 1
fi
if [[ -n "$BASELINE_PP" ]] && ! [[ $BASELINE_PP =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
  echo "Error: --baseline-pp must be a number." >&2; exit 1
fi
if [[ -n "$BASELINE_TG" ]] && ! [[ $BASELINE_TG =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
  echo "Error: --baseline-tg must be a number." >&2; exit 1
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting collection of SWEEP results."
echo "Context: $CONTEXT"
echo "Requested N_KV: $N_KV"
[[ -n "$BASELINE_PP" ]] && echo "Using global baseline PP: $BASELINE_PP"
[[ -n "$BASELINE_TG" ]] && echo "Using global baseline TG: $BASELINE_TG"
[[ -n "$BASELINE_PP_QTYPE" ]] && echo "Inject PP baseline for qtype: $BASELINE_PP_QTYPE"
[[ -n "$BASELINE_TG_QTYPE" ]] && echo "Inject TG baseline for qtype: $BASELINE_TG_QTYPE"
[[ -n "$AUTO_BASELINE_QTYPE" ]] && echo "Auto-baseline will attempt to read bench_sweep_result.baseline.${AUTO_BASELINE_QTYPE}.${CONTEXT}.txt"
[[ "$HIDE_EMPTY" == true ]] && echo "Hide empty qtype bench results from the csv: $HIDE_EMPTY"
[[ -n "$qtypes" ]] && echo "Overriding qtypes with: $qtypes"
echo "Output PP CSV: $OUTPUT_PP_CSV"
echo "Output TG CSV: $OUTPUT_TG_CSV"
if [[ "$GROUP_TENSORS_DISABLED" == "true" ]]; then
  echo "Group tensors: DISABLED"
else
  echo "Group tensors: ENABLED; groups:"
  for g in "${GROUP_TENSORS_RAW[@]}"; do echo "  - $g"; done
  if [[ "$EXPAND_GROUPS" == "true" ]]; then
    echo "Group expansion: ENABLED (show all member tensors)"
  else
    echo "Group expansion: DISABLED (show one column per group)"
  fi
fi

# Discover qtypes
declare -a QTYPES
if [[ -n "$qtypes" ]]; then
  IFS=',' read -r -a QTYPES <<< "$qtypes"
else
  for f in tensors.*.map; do
    [[ -f $f ]] || continue
    qtype="${f#tensors.}"
    qtype="${qtype%.map}"
    QTYPES+=("$qtype")
  done
fi

if [[ ${#QTYPES[@]} -eq 0 ]]; then
  echo "Warning: No tensors.*.map files found in current directory and no --qtypes provided. Exiting." >&2
  exit 1
fi

# normalize and sort
IFS=$'\n' sorted_qtypes=($(printf '%s\n' "${QTYPES[@]}" | sort -u))
unset IFS
QTYPES=("${sorted_qtypes[@]}")
echo "Found qtypes: ${QTYPES[*]}"

# gather list of sweep result files in current dir matching context (includes group files)
all_bench_sweep_result_files=$(find . -maxdepth 1 -type f -printf "%f\n" 2>/dev/null | grep -E "^bench_sweep_result\..*\.${CONTEXT}\.txt$" 2>/dev/null || true)

declare -A PP_VALUES   # key: "qtype|tensor_or_group" => S_PP value or "404"
declare -A TG_VALUES   # key: "qtype|tensor_or_group" => S_TG value or "404"
declare -A TENSOR_SET  # tensor_name or group name => 1 (if to include)
declare -A PROCESSED_GROUP_QTYPE  # key: "qtype|groupidx" => 1 when group's results handled for that qtype

# helper: find_group_index_for_tensor <tensor> -> prints group index (0-based) or -1
find_group_index_for_tensor() {
  local tensor="$1"
  if [[ "$GROUP_TENSORS_DISABLED" == "true" ]]; then
    printf '%s\n' -1
    return
  fi
  for idx in "${!GROUP_TENSORS_RAW[@]}"; do
    local group_raw="${GROUP_TENSORS_RAW[$idx]}"
    IFS=',' read -r -a regs <<< "$group_raw"
    for reg in "${regs[@]}"; do
      # trim spaces
      reg="$(sed -E 's/^[[:space:]]+|[[:space:]]+$//g' <<<"$reg")"
      [[ -z "$reg" ]] && continue
      if [[ $tensor =~ $reg ]]; then
        printf '%s\n' "$idx"
        return
      fi
    done
  done
  printf '%s\n' -1
}

# helper: extract S_PP and S_TG from a baseline/sweep file (returns "S_PP|S_TG" or empty string)
extract_pp_tg_from_file() {
  local file="$1"
  # Use same parsing logic as other sweep files; look for row where column 4 equals N_KV
  awk -F'|' -v nkv="$N_KV" '
    function trim(s) { gsub(/^[ \t\r\n]+|[ \t\r\n]+$/, "", s); return s }
    {
      for(i=1;i<=NF;i++) $i=trim($i)
      if (NF >= 8 && $2 ~ /^[0-9]+$/ && $4 == nkv) {
        gsub(/^[ \t]+|[ \t]+$/, "", $6)
        gsub(/^[ \t]+|[ \t]+$/, "", $8)
        print $6 "|" $8
        exit
      }
    }
  ' "$file" 2>/dev/null || true
}

# If grouping is enabled and groups are NOT expanded, and the user does not hide empty columns,
# create column placeholders for each group (group0, group1, ...) so they appear in CSV headers by default.
if [[ "$GROUP_TENSORS_DISABLED" != "true" && "$EXPAND_GROUPS" == "false" && "$HIDE_EMPTY" == "false" ]]; then
  for idx in "${!GROUP_TENSORS_RAW[@]}"; do
    TENSOR_SET["group${idx}"]=1
  done
fi

# If user requested auto-baseline, attempt to read bench_sweep_result.baseline.<qtype>.<CONTEXT>.txt
if [[ -n "$AUTO_BASELINE_QTYPE" ]]; then
  baseline_fname="bench_sweep_result.baseline.${AUTO_BASELINE_QTYPE,,}.${CONTEXT}.txt"
  if printf '%s\n' "$all_bench_sweep_result_files" | grep -qF -- "$baseline_fname"; then
      parsed=$(extract_pp_tg_from_file "./${baseline_fname}" || true)
      if [[ -n "$parsed" ]]; then
        base_pp="${parsed%%|*}"
        base_tg="${parsed#*|}"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Auto-baseline: extracted for qtype=${qtype}: PP=${base_pp}, TG=${base_tg}"
        [[ -n "$BASELINE_PP_QTYPE" && "$AUTO_BASELINE_QTYPE" == "$BASELINE_PP_QTYPE" && -n "$BASELINE_PP" ]] && echo "[$(date '+%Y-%m-%d %H:%M:%S')] BASELINE_PP already user-defined, not replaced!" || { BASELINE_PP=${base_pp} && BASELINE_PP_QTYPE=${AUTO_BASELINE_QTYPE,,} && echo "[$(date '+%Y-%m-%d %H:%M:%S')] BASELINE_PP='$BASELINE_PP' and BASELINE_PP_QTYPE='$BASELINE_PP_QTYPE' have now been set"; }
        [[ -n "$BASELINE_TG_QTYPE" && "$AUTO_BASELINE_QTYPE" == "$BASELINE_TG_QTYPE" && -n "$BASELINE_TG" ]] && echo "[$(date '+%Y-%m-%d %H:%M:%S')] BASELINE_TG already user-defined, not replaced!" || { BASELINE_TG=${base_tg} && BASELINE_TG_QTYPE=${AUTO_BASELINE_QTYPE,,} && echo "[$(date '+%Y-%m-%d %H:%M:%S')] BASELINE_TG='$BASELINE_TG' and BASELINE_TG_QTYPE='$BASELINE_TG_QTYPE' have now been set"; }
      else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Auto-baseline: baseline file exists but no N_KV=${N_KV} row found in $baseline_fname"
      fi
  else
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] Auto-baseline: baseline file $baseline_fname not found for qtype=${qtype}"
  fi
fi

# For each qtype, parse its tensors map
for qtype in "${QTYPES[@]}"; do
  mapfile="tensors.${qtype}.map"
  if [[ ! -f "$mapfile" ]]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Warning: expected map file '$mapfile' not found. Skipping qtype='$qtype'." >&2
    continue
  fi
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Processing map file: $mapfile"

  # read all lines of mapfile into array for flexible scanning/group collection
  mapfile -t MAP_LINES < "$mapfile"

  # Build quick list of tensor names available in this qtype's map
  declare -a TENS_IN_MAP=()
  for line in "${MAP_LINES[@]}"; do
    [[ -z "$line" ]] && continue
    IFS=':' read -r _fname _hash tensor_name _ <<< "$line"
    TENS_IN_MAP+=("$tensor_name")
  done

  # iterate through entries in MAP_LINES
  for line in "${MAP_LINES[@]}"; do
    [[ -z "$line" ]] && continue
    IFS=':' read -r fname file_hash tensor_name _ <<< "$line"

    # match tensor_name against USER_REGEX
    matched=false
    for pat in "${USER_REGEX[@]}"; do
      if [[ $tensor_name =~ $pat ]]; then matched=true; break; fi
    done
    [[ "$matched" == true ]] || continue

    # Determine group membership (if any) for this tensor
    gid=$(find_group_index_for_tensor "$tensor_name")

    # Decide whether to add a column placeholder based on grouping & expansion & hide-empty
    if (( gid >= 0 )); then
      if [[ "$EXPAND_GROUPS" == "true" ]]; then
        # user wants member columns: include the individual tensor as a column unless hide-empty==true
        [[ "$HIDE_EMPTY" == "false" ]] && TENSOR_SET["$tensor_name"]=1
      else
        # user wants group columns: ensure group column is present (unless hide-empty==true)
        [[ "$HIDE_EMPTY" == "false" ]] && TENSOR_SET["group${gid}"]=1
      fi
    else
      # not in a group -> individual tensor column
      [[ "$HIDE_EMPTY" == "false" ]] && TENSOR_SET["$tensor_name"]=1
    fi

    # If grouping is enabled and this tensor belongs to a group, attempt to process the group
    if (( gid >= 0 )); then
      proc_key="${qtype}|${gid}"
      # If this group for this qtype has already been processed (value '1'), skip individual handling.
      # We do NOT skip when the marker is 'MISSING' — that allows falling back to per-tensor files.
      if [[ "${PROCESSED_GROUP_QTYPE[$proc_key]:-}" == "1" ]]; then
        continue
      fi

      # collect all group members present in this qtype's map
      group_raw="${GROUP_TENSORS_RAW[$gid]}"
      IFS=',' read -r -a regs <<< "$group_raw"
      declare -a group_members=()
      for reg in "${regs[@]}"; do
        reg="$(sed -E 's/^[[:space:]]+|[[:space:]]+$//g' <<<"$reg")"
        [[ -z "$reg" ]] && continue
        for t in "${TENS_IN_MAP[@]}"; do
          if [[ $t =~ $reg ]]; then
            if [[ ! " ${group_members[*]} " =~ " $t " ]]; then
              group_members+=("$t")
            fi
          fi
        done
      done

      if (( ${#group_members[@]} == 0 )); then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Warning: no group members found in map for group #${gid} (qtype=${qtype}). Skipping group." >&2
        PROCESSED_GROUP_QTYPE["$proc_key"]=1
        continue
      fi

      # Look for group result file: bench_sweep_result.group{gid}.{qtype}.{CONTEXT}.txt
      group_result_filename="bench_sweep_result.group${gid}.${qtype}.${CONTEXT}.txt"
      # confirm it exists in directory listing
      if ! printf '%s\n' "$all_bench_sweep_result_files" | grep -qF -- "$group_result_filename"; then
        # Only log the "missing group file" message once per (qtype, group).
        if [[ -z "${PROCESSED_GROUP_QTYPE[$proc_key]:-}" ]]; then
          echo "[$(date '+%Y-%m-%d %H:%M:%S')] No group sweep result file found for group #${gid}, qtype=${qtype}: expected '$group_result_filename'. Will fall back to individual tensor files."
          # Mark as 'missing' so we don't re-print this for other members of the same group/qtype.
          PROCESSED_GROUP_QTYPE["$proc_key"]="MISSING"
        fi
        # fall back to per-tensor handling (do not mark as processed=1)
      else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Found group sweep result file: $group_result_filename -> applying to ${#group_members[@]} member(s)."
        result_file="./${group_result_filename}"

        # parse file to extract S_PP and S_TG for requested N_KV
        parsed=$(extract_pp_tg_from_file "$result_file" || true)

        if [[ -n "$parsed" ]]; then
          SPP_VAL="${parsed%%|*}"
          STG_VAL="${parsed#*|}"
          SPP_VAL="$(sed -E 's/^[[:space:]]+|[[:space:]]+$//g' <<<"$SPP_VAL")"
          STG_VAL="$(sed -E 's/^[[:space:]]+|[[:space:]]+$//g' <<<"$STG_VAL")"
          echo "[$(date '+%Y-%m-%d %H:%M:%S')] Extracted group #${gid} (qtype=${qtype}): S_PP=$SPP_VAL, S_TG=$STG_VAL"
        else
          echo "[$(date '+%Y-%m-%d %H:%M:%S')] Warning: no row with N_KV=${N_KV} found in $result_file. Marking 404 for entire group."
          SPP_VAL="404"
          STG_VAL="404"
        fi

        # assign values either to group column (default) or to each member (when expanded)
        if [[ "$EXPAND_GROUPS" == "true" ]]; then
          # assign per-member values
          for gm in "${group_members[@]}"; do
            PP_VALUES["${qtype}|${gm}"]="$SPP_VAL"
            TG_VALUES["${qtype}|${gm}"]="$STG_VAL"
            # ensure tensor column present when hide-empty==true and a result exists
            [[ "$HIDE_EMPTY" == true ]] && TENSOR_SET["$gm"]=1
          done
        else
          # assign to group column key, not individual members
          PP_VALUES["${qtype}|group${gid}"]="$SPP_VAL"
          TG_VALUES["${qtype}|group${gid}"]="$STG_VAL"
          # when hide-empty==true and we found a result, ensure the group column is present
          [[ "$HIDE_EMPTY" == true ]] && TENSOR_SET["group${gid}"]=1
        fi

        PROCESSED_GROUP_QTYPE["$proc_key"]=1
        # done with this group for this qtype
        continue
      fi
    fi

    # If we reach here: either grouping disabled, tensor not in group, OR group file not present -> handle per-tensor file
    # Find matching bench_sweep_result file for this tensor and qtype (individual)
    # regex to match: ^bench_sweep_result\.${tensor_name}\..*\.${CONTEXT}\.txt$
    regex="^bench_sweep_result\.${tensor_name}\..*\.${CONTEXT}\.txt$"
    bench_match=$(printf '%s\n' "$all_bench_sweep_result_files" | grep -m1 -E "$regex" 2>/dev/null || true)

    if [[ -z "$bench_match" ]]; then
      # no individual file: leave empty (maybe other qtypes have it)
      continue
    fi

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Found sweep result file for tensor '$tensor_name': $bench_match"

    # Mark tensor for CSV if hide-empty was true (we found result)
    [[ "$HIDE_EMPTY" == true ]] && TENSOR_SET["$tensor_name"]=1

    # Full path to file in current dir
    result_file="./${bench_match}"

    # parse file: find the row where the N_KV column equals N_KV and extract S_PP (t/s) and S_TG (t/s)
    # file rows look like:
    # |  4096 |   1024 |      0 |    0.786 |  5211.60 |    6.268 |   163.38 |
    # fields (when split by '|'):
    # $2 = PP, $3 = TG, $4 = N_KV, $5 = T_PP s, $6 = S_PP t/s, $7 = T_TG s, $8 = S_TG t/s
    # We'll trim whitespace and match $4 == N_KV requested
    SPP_VAL=""
    STG_VAL=""
    # Use awk to robustly find matching row (skips header & separators)
    parsed=$(awk -F'|' -v nkv="$N_KV" '
      function trim(s) { gsub(/^[ \t\r\n]+|[ \t\r\n]+$/, "", s); return s }
      {
        # trim each field
        for(i=1;i<=NF;i++) $i=trim($i)
        # check we have at least 8 columns and the 4th column equals nkv (and column 2 is numeric)
        if (NF >= 8 && $2 ~ /^[0-9]+$/ && $4 == nk) {
          # print S_PP and S_TG separated by |
          gsub(/^[ \t]+|[ \t]+$/, "", $6)
          gsub(/^[ \t]+|[ \t]+$/, "", $8)
          print $6 "|" $8
          exit
        }
      }
    ' "$result_file" || true)

    if [[ -n "$parsed" ]]; then
      SPP_VAL="${parsed%%|*}"
      STG_VAL="${parsed#*|}"
      # ensure trimmed
      SPP_VAL="$(sed -E 's/^[[:space:]]+|[[:space:]]+$//g' <<<"$SPP_VAL")"
      STG_VAL="$(sed -E 's/^[[:space:]]+|[[:space:]]+$//g' <<<"$STG_VAL")"
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] Extracted for tensor='$tensor_name', qtype='$qtype': S_PP=$SPP_VAL, S_TG=$STG_VAL"
    else
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] Warning: no row with N_KV=${N_KV} found in $result_file. Marking 404."
      SPP_VAL="404"
      STG_VAL="404"
    fi

    PP_VALUES["${qtype}|${tensor_name}"]="$SPP_VAL"
    TG_VALUES["${qtype}|${tensor_name}"]="$STG_VAL"

  done # end iterating MAP_LINES

done # end for qtype

# Build sorted tensor list / group list for headers
tensor_list=("${!TENSOR_SET[@]}")
if [[ ${#tensor_list[@]} -eq 0 ]]; then
  echo "Warning: No tensor names matched USER_REGEX in any map files (or no results found). Exiting." >&2
  exit 1
fi
IFS=$'\n' sorted_tensors=($(printf '%s\n' "${tensor_list[@]}" | sort -Vu))
unset IFS

# Write PP CSV
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Writing PP CSV to $OUTPUT_PP_CSV"

echo "[DEBUG] Writing PP CSV..."

{
  printf 'QTYPE'
  for t in "${sorted_tensors[@]}"; do
    echo "[DEBUG] Header tensor: $t" >&2
    printf ',%s' "$t"
  done
  printf '\n'

  for qtype in "${QTYPES[@]}"; do
    echo "[DEBUG] Writing row for QTYPE: $qtype" >&2
    printf '%s' "$qtype"
    for t in "${sorted_tensors[@]}"; do
      key="${qtype}|${t}"
      val="${PP_VALUES[$key]:-}"
      echo "[DEBUG] Raw value for [$key] = '$val'" >&2

      if [[ -n "$BASELINE_PP" && -n "$val" ]]; then
        if [[ "$val" == "404" ]]; then
          val="404%"
        else
          pct=$(awk -v b="$BASELINE_PP" -v v="$val" 'BEGIN{printf "%+.2f%%", (v-b)/b*100}')
          val="$pct"
        fi
        echo "[DEBUG] Final value for [$key] = '$val'" >&2
      elif [[ -n $BASELINE_PP && "$BASELINE_PP_QTYPE" == "$qtype" ]]; then
        val="0%"
        echo "[DEBUG] Final value set to baseline for [$key] = '$val'" >&2
      fi

      printf ',%s' "$val"
    done
    printf '\n'
  done
} > "$OUTPUT_PP_CSV"

echo "[DEBUG] Writing TG CSV..."

# Write TG CSV
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Writing TG CSV to $OUTPUT_TG_CSV"
{
  printf 'QTYPE'
  for t in "${sorted_tensors[@]}"; do
    echo "[DEBUG] Header tensor: $t" >&2
    printf ',%s' "$t"
  done
  printf '\n'

  for qtype in "${QTYPES[@]}"; do
    echo "[DEBUG] Writing row for QTYPE: $qtype" >&2
    printf '%s' "$qtype"
    for t in "${sorted_tensors[@]}"; do
      key="${qtype}|${t}"
      val="${TG_VALUES[$key]:-}"
      echo "[DEBUG] Raw value for [$key] = '$val'" >&2

      if [[ -n "$BASELINE_TG" && -n "$val" ]]; then
        if [[ "$val" == "404" ]]; then
          val="404%"
        else
          pct=$(awk -v b="$BASELINE_TG" -v v="$val" 'BEGIN{printf "%+.2f%%", (v-b)/b*100}')
          val="$pct"
        fi
        echo "[DEBUG] Final value for [$key] = '$val'" >&2
      elif [[ -n $BASELINE_TG && "$BASELINE_TG_QTYPE" == "$qtype" ]]; then
        val="0%"
        echo "[DEBUG] Final value set to baseline for [$key] = '$val'" >&2
      fi

      printf ',%s' "$val"
    done
    printf '\n'
  done
} > "$OUTPUT_TG_CSV"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Done. CSVs available at: $OUTPUT_PP_CSV and $OUTPUT_TG_CSV"
