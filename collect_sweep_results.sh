#!/usr/bin/env bash
#***************************************************************#
#** This script is part of Thireus' GGUF Tool Suite.          **#
#** collect_sweep_results.sh is a helper tool that collects   **#
#** the S_PP and S_TG values from bench_sweep_result.* files. **#
#**                                                           **#
#** ********************************************************* **#
#** --------------- Updated: Jul-26-2026 -------------------- **#
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

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }

# Usage message
usage() {
  cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Options:
  --context CONTEXT                     Context value used in sweep filenames (integer). Default: 8192
  --nkv N_KV                            N_KV value to extract from sweep output (integer). Default: 0
  --baseline-pp VALUE                   Global baseline PP (t/s) value for percent-delta computation (float)
  --baseline-tg VALUE                   Global baseline TG (t/s) value for percent-delta computation (float)
  --inject-baseline-pp-qtype QTYPE      For this qtype, inject baseline PP. The value is resolved from --baseline-pp, else from --auto-baseline, else from bench_sweep_result.baseline.QTYPE.<CONTEXT>.txt in the current directory; the script errors out if none of those yields a value.
  --inject-baseline-tg-qtype QTYPE      For this qtype, inject baseline TG. The value is resolved from --baseline-tg, else from --auto-baseline, else from bench_sweep_result.baseline.QTYPE.<CONTEXT>.txt in the current directory; the script errors out if none of those yields a value.
  --auto-baseline QTYPE                 Automatically read bench_sweep_result.baseline.QTYPE.<CONTEXT>.txt to obtain
                                        per-qtype baseline PP and TG (only for the named qtype).

  --group-tensors REG1[,REG2] [REG3,..] Specify one or more group specifications (same syntax as benchmark_each_tensor.sh).
                                        Each argument is a group: comma-separated regexes. If omitted, grouping disabled.
  --group-tensors-map FILE              Path to a group mapping file (each line "groupN:regex[,regex2]" or "regex[,regex2]").
                                          This replicates --group-tensors but reads groups from a file. Mutually exclusive
                                          with --group-tensors.
  --groups-only.                        When present, will only collect the group metrics (default: disabled)
  --expand-groups                       When present, expand groups into individual tensor columns (default: disabled)
  --hide-empty                          Don't include empty benchmark results to the output csv
  --output-pp FILE                      Path to output PP CSV file (default: pp_results.csv)
  --output-tg FILE                      Path to output TG CSV file (default: tg_results.csv)
  --qtypes Q1,Q2,...                    Comma-separated list of qtypes to use (overrides auto-discovery)
  --no-percentage                       Disable percent-delta computation (emit raw values; baseline values will be injected as-is)
  -h, --help                            Show this help message and exit

Note: bench_sweep_result.*.txt.unused quarantine files (tensors ignored by the model, produced by benchmark_each_tensor.sh --hotswap) are accepted by default when the normal .txt result is absent, and collected as baseline-equivalent results (PP/TG=baseline).

Note: q*_K and q*_KV quants must be used with a capital "K" and "KV" letters at the end of their name (e.g. q2_K, q6_K, q8_KV). All other quants are lowercase (e.g. bf16, iq3_kt, iq4_k, q8_0, q4_k_r4, q8_k_r8). Map and result filenames are matched case-sensitively, so a mis-cased qtype finds no files.
EOF
}

# ============== USER CONFIGURATION ==============

# List of tensor-name regex patterns (Bash regex) to include in the CSV.
# Adjust these as needed.
USER_REGEX=(
  # Token embedding and output tensors (GPU)
  # note token_embd cannot be repacked quant type
  '^output\.weight$'
  '^token_embd\.weight$'

  # GPU Only
  '^blk\.[0-2]\.ffn_down\.weight$'
  '^blk\.[0-2]\.ffn_up\.weight$'
  '^blk\.[0-2]\.ffn_gate\.weight$'

  ## GPU-loaded ffn_*_shexp
  '^blk\.([3-9]|[1-5][0-9]|60)\.ffn_down_shexp\.weight$'
  '^blk\.([3-9]|[1-5][0-9]|60)\.ffn_up_shexp\.weight$'
  '^blk\.([3-9]|[1-5][0-9]|60)\.ffn_gate_shexp\.weight$'

  ## CPU-loaded ffn_*_exps
  '^blk\.([3-9]|[1-5][0-9]|60)\.ffn_down_exps\.weight$'
  '^blk\.([3-9]|[1-5][0-9]|60)\.ffn_up_exps\.weight$'
  '^blk\.([3-9]|[1-5][0-9]|60)\.ffn_gate_exps\.weight$'
)

# Default output CSV filename (can be overridden via --output-pp and --output-tg)
OUTPUT_PP_CSV="pp_results.csv"
OUTPUT_TG_CSV="tg_results.csv"

# =========== End USER CONFIGURATION ============

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
GROUPS_ONLY=false
EXPAND_GROUPS=false
GROUP_TENSORS_MAP_FILE=""
NO_PERCENTAGE=false

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
      if [[ -z "${2:-}" || "${2:0:2}" == "--" ]]; then
        echo "Error: --qtypes requires an argument (comma-separated list)" >&2; usage; exit 1
      fi
      qtypes="$2"
      shift 2
      ;;
    --group-tensors)
      # collect one or more group specs (nargs '+')
      shift
      GROUP_TENSORS_RAW=()
      if [[ $# -eq 0 || "${1:0:2}" == "--" ]]; then
        echo "Error: --group-tensors requires at least one group specification" >&2; usage; exit 1
      fi
      while [[ $# -gt 0 && "${1:0:2}" != "--" ]]; do
        GROUP_TENSORS_RAW+=("$1")
        shift
      done
      ;;
    --group-tensors-map)
      if [[ -z "${2:-}" || "${2:0:2}" == "--" ]]; then
        echo "Error: --group-tensors-map requires a filename argument" >&2; usage; exit 1
      fi
      GROUP_TENSORS_MAP_FILE="$2"
      shift 2
      ;;
    --groups-only)
      GROUPS_ONLY=true
      shift
      ;;
    --expand-groups)
      EXPAND_GROUPS=true
      shift
      ;;
    --no-percentage)
      NO_PERCENTAGE=true
      shift
      ;;
    -h|--help)
      usage; exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

# Qtype naming rule: q*_K and q*_KV quants must be spelled with a capital "K"
# and "KV" at the end of their name (q2_K, q6_K, q8_KV). All other quants are
# lowercase (bf16, iq3_kt, iq4_k, q8_0, q4_k_r4, q8_k_r8). Map and result files
# are matched case-sensitively, so a mis-cased qtype silently finds nothing.
canonical_qtype() {
  local q="${1,,}"
  if [[ "$q" =~ ^q[^_]+_kv$ ]]; then
    printf '%s' "${q%_kv}_KV"
  elif [[ "$q" =~ ^q[^_]+_k$ ]]; then
    printf '%s' "${q%_k}_K"
  else
    printf '%s' "$q"
  fi
}

warn_if_bad_qtype_casing() {
  local origin="$1" q="$2" canon
  [[ -z "$q" ]] && return 0
  canon="$(canonical_qtype "$q")"
  [[ "$q" == "$canon" ]] && return 0
  echo "⚠️  Warning! ${origin} qtype '$q' does not follow the qtype naming rule - q*_K and q*_KV quants must be used with a capital \"K\" and \"KV\" letters at the end of their name, all other quants are lowercase. Did you mean '$canon'? Map and result filenames are matched case-sensitively, so '$q' will most likely find no files." >&2
}

warn_if_bad_qtype_casing "--inject-baseline-pp-qtype" "$BASELINE_PP_QTYPE"
warn_if_bad_qtype_casing "--inject-baseline-tg-qtype" "$BASELINE_TG_QTYPE"
warn_if_bad_qtype_casing "--auto-baseline" "$AUTO_BASELINE_QTYPE"
if [[ -n "${qtypes:-}" ]]; then
  IFS=',' read -r -a __qtypes_to_check <<< "$qtypes"
  for __q in "${__qtypes_to_check[@]}"; do
    warn_if_bad_qtype_casing "--qtypes" "$__q"
  done
  unset __qtypes_to_check __q
fi

# Ensure user didn't supply both --group-tensors and --group-tensors-map
if [[ -n "${GROUP_TENSORS_MAP_FILE:-}" && ${#GROUP_TENSORS_RAW[@]} -gt 0 ]]; then
  echo "Error: --group-tensors and --group-tensors-map are mutually exclusive. Please provide only one of them." >&2
  exit 1
fi

# If a group mapping file was provided, read it and populate GROUP_TENSORS_RAW.
# File lines can be:
#   group0:^blk\.0\.attn_(k|v)\.weight$
#   group1:^another_regex1$,^another_regex2$
#   or simply:
#   ^blk\.0\.attn_(k|v)\.weight$
# blank lines and lines starting with '#' are ignored.
if [[ -n "${GROUP_TENSORS_MAP_FILE:-}" ]]; then
  if [[ ! -f "$GROUP_TENSORS_MAP_FILE" ]]; then
    echo "Error: group mapping file '$GROUP_TENSORS_MAP_FILE' not found." >&2
    exit 1
  fi

  # Read file, collect named group indices and/or unnamed groups preserving order.
  declare -A __GTMP_idx_map=()
  declare -a __GTMP_idx_list=()
  declare -a __GTMP_ordered_unnamed=()

  while IFS= read -r __line || [[ -n "$__line" ]]; do
    # Trim whitespace
    __line="$(sed -E 's/^[[:space:]]+|[[:space:]]+$//g' <<<"$__line")"
    # skip empty or comment lines
    [[ -z "$__line" ]] && continue
    [[ "${__line:0:1}" == "#" ]] && continue

    if [[ "$__line" == *:* ]]; then
      __prefix="${__line%%:*}"
      __rest="${__line#*:}"
      __rest="$(sed -E 's/^[[:space:]]+|[[:space:]]+$//g' <<<"$__rest")"
      if [[ "$__prefix" =~ ^group([0-9]+)$ ]]; then
        __idx="${BASH_REMATCH[1]}"
        __GTMP_idx_map["$__idx"]="$__rest"
        __GTMP_idx_list+=("$__idx")
      else
        # no groupN prefix, treat entire line after first colon as a single group spec
        __GTMP_ordered_unnamed+=("$__rest")
      fi
    else
      # whole line is a group regex list
      __GTMP_ordered_unnamed+=("$__line")
    fi
  done < "$GROUP_TENSORS_MAP_FILE"

  # Sort numeric indices ascending and build GROUP_TENSORS_RAW
  if [[ ${#__GTMP_idx_list[@]} -gt 0 ]]; then
    # remove duplicates and sort numeric
    IFS=$'\n' __sorted_idx=($(printf '%s\n' "${__GTMP_idx_list[@]}" | sort -n -u))
    unset IFS
    for __i in "${__sorted_idx[@]}"; do
      GROUP_TENSORS_RAW+=("${__GTMP_idx_map[$__i]}")
    done
  fi
  # Append unnamed groups preserving file order
  for __u in "${__GTMP_ordered_unnamed[@]}"; do
    GROUP_TENSORS_RAW+=("$__u")
  done

  # cleanup temp variables
  unset __GTMP_idx_map __GTMP_idx_list __GTMP_ordered_unnamed __sorted_idx __i __u __prefix __rest __line __idx

  if [[ ${#GROUP_TENSORS_RAW[@]} -eq 0 ]]; then
    echo "Warning: group mapping file '$GROUP_TENSORS_MAP_FILE' parsed but no groups found." >&2
  else
    echo "[$(timestamp) Loaded ${#GROUP_TENSORS_RAW[@]} group(s) from mapping file: $GROUP_TENSORS_MAP_FILE"
  fi
fi

# Validate that if user asked to collect groups only, they passed --group-tensors or --group-tensors-map
if [[ "$GROUPS_ONLY" == "true" ]] && [[ ! (-n "${GROUP_TENSORS_MAP_FILE:-}" || ${#GROUP_TENSORS_RAW[@]} -gt 0) ]]; then
  echo "Error: --groups-only requires --group-tensors or --group-tensors-map to be set." >&2
  exit 1
fi

# If the single token '[]' is passed, grouping disabled (mirror benchmark_each_tensor behaviour)
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

# Echo chosen settings
echo "[$(timestamp) Starting collection of SWEEP results."
echo "Context: $CONTEXT"
echo "Requested N_KV: $N_KV"
[[ -n "$BASELINE_PP" ]] && echo "Using global baseline PP: $BASELINE_PP"
[[ -n "$BASELINE_TG" ]] && echo "Using global baseline TG: $BASELINE_TG"
[[ -n "$BASELINE_PP_QTYPE" ]] && echo "Inject PP baseline for qtype: $BASELINE_PP_QTYPE"
[[ -n "$BASELINE_TG_QTYPE" ]] && echo "Inject TG baseline for qtype: $BASELINE_TG_QTYPE"
[[ -n "$AUTO_BASELINE_QTYPE" ]] && echo "Auto-baseline will attempt to read bench_sweep_result.baseline.${AUTO_BASELINE_QTYPE}.${CONTEXT}.txt"
[[ "$HIDE_EMPTY" == true ]] && echo "Hide empty qtype bench results from the csv: $HIDE_EMPTY"
[[ -n "${qtypes:-}" ]] && echo "Overriding qtypes with: $qtypes"
echo "Output PP CSV: $OUTPUT_PP_CSV"
echo "Output TG CSV: $OUTPUT_TG_CSV"
ALL_GROUP_IDS=()
# Split and whitespace-trim every group's regex list ONCE, into a flat table indexed by
# group. This used to be redone - with a sed(1) fork per regex - on every single call to
# find_group_indexes_for_tensor and on every group-member rebuild, i.e. tens of thousands
# of forks per run. Values are identical, this is purely hoisting invariant work.
declare -a GROUP_REGEX_FLAT=()   # all trimmed regexes, concatenated
declare -a GROUP_REGEX_START=()  # group idx -> first offset into GROUP_REGEX_FLAT
declare -a GROUP_REGEX_COUNT=()  # group idx -> number of regexes
for _gr_idx in "${!GROUP_TENSORS_RAW[@]}"; do
  IFS=',' read -r -a _gr_regs <<< "${GROUP_TENSORS_RAW[$_gr_idx]}"
  GROUP_REGEX_START[$_gr_idx]=${#GROUP_REGEX_FLAT[@]}
  _gr_cnt=0
  for _gr_r in ${_gr_regs[@]+"${_gr_regs[@]}"}; do
    _gr_r="${_gr_r#"${_gr_r%%[![:space:]]*}"}"   # ltrim
    _gr_r="${_gr_r%"${_gr_r##*[![:space:]]}"}"   # rtrim
    [[ -z "$_gr_r" ]] && continue
    GROUP_REGEX_FLAT+=("$_gr_r")
    _gr_cnt=$((_gr_cnt + 1))
  done
  GROUP_REGEX_COUNT[$_gr_idx]=$_gr_cnt
done
unset _gr_idx _gr_regs _gr_cnt _gr_r

if [[ "$GROUP_TENSORS_DISABLED" != "true" ]]; then
  echo "Group tensors: ENABLED; groups:"
  gid=0
  for g in "${GROUP_TENSORS_RAW[@]}"; do echo "  - group$gid: $g"; ALL_GROUP_IDS+=("$gid"); gid=$((gid + 1)); done
  if [[ "$GROUPS_ONLY" == "true" ]] && (( ${#ALL_GROUP_IDS[@]} == 0 )); then
    echo "[$(timestamp)] ❌ Error: --groups-only is set but there are no groups set!" >&2
  fi

  if [[ "$EXPAND_GROUPS" == "true" ]]; then
    echo "Group expansion: ENABLED (show all member tensors)"
    if [[ "$GROUPS_ONLY" == "true" ]]; then
      echo "⚠️  Warning! If a group contains tensor(s) presents in other groups the metrics for that tensor will be overwritten by the latest group processed."
    fi
  else
    echo "Group expansion: DISABLED (show one column per group)"
  fi
  if [[ "$GROUPS_ONLY" == "true" ]]; then
    echo "Collecting group metrics only: ENABLED"
  else
    echo "Collecting group metrics only: DISABLED"
  fi
fi
[[ "$NO_PERCENTAGE" == "true" ]] && echo "Percentage computation: DISABLED ( --no-percentage )"

# 1. Discover qtypes by finding tensors.{qtype}.map files in current directory
declare -a QTYPES=()

# Override discovered qtypes if user provided --qtypes
if [[ -n "${qtypes:-}" ]]; then
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
    echo "Warning: No tensors.*.map files found in current directory${qtypes:+ and no valid --qtypes provided}. Exiting." >&2
    exit 1
fi

# Sort qtypes lexically and remove duplicates
IFS=$'\n' sorted_qtypes=($(printf '%s\n' "${QTYPES[@]}" | sort -u))
unset IFS
QTYPES=("${sorted_qtypes[@]}")

echo "Found qtypes: ${QTYPES[*]}"

# The casing warning above is only a warning - it does not stop the run. So that a
# mis-cased --inject-baseline-pp-qtype / --inject-baseline-tg-qtype does not silently
# fail to match any discovered qtype (leaving the baseline uninjected), resolve them
# case-insensitively against the qtypes we actually found and adopt the real spelling.
for __bq_var in BASELINE_PP_QTYPE BASELINE_TG_QTYPE; do
  __bq_val="${!__bq_var}"
  [[ -z "$__bq_val" ]] && continue
  printf '%s\n' "${QTYPES[@]}" | grep -qxF -- "$__bq_val" && continue
  __bq_fixed="$(printf '%s\n' "${QTYPES[@]}" | grep -ixF -- "$__bq_val" | head -n1 || true)"
  if [[ -n "$__bq_fixed" ]]; then
    echo "[$(timestamp) ⚠️  Corrected ${__bq_var} casing: '$__bq_val' -> '$__bq_fixed' (matched a discovered qtype)"
    printf -v "$__bq_var" '%s' "$__bq_fixed"
  fi
done
unset __bq_var __bq_val __bq_fixed

declare -A PP_VALUES   # key: "qtype|tensor_or_group" => S_PP value or "404"
declare -A TG_VALUES   # key: "qtype|tensor_or_group" => S_TG value or "404"
declare -A TENSOR_SET  # tensor_name or group name => 1 (if to include)
declare -A PROCESSED_GROUP_QTYPE  # key: "qtype|groupidx" => 1 when group's results handled for that qtype

# gather list of sweep result files in current dir matching context (includes group files)
bench_files_list=$(
  for f in ./bench_sweep_*.txt; do
    [ -e "$f" ] || continue    # skip non-matching globs
    [ -f "$f" ] && printf '%s\n' "${f##*/}"
  done 2>/dev/null
)
all_bench_sweep_result_files=$(printf '%s\n' "$bench_files_list" | grep -E "^bench_sweep_result\..*\.${CONTEXT}\.txt$" 2>/dev/null || true)

# Also gather .txt.unused quarantine files (tensors ignored by the model, see
# benchmark_each_tensor.sh --hotswap). These are accepted by default: swapping
# such tensors has no effect on the model, so their speed metrics equal the
# baseline's - which is a valid and useful data point.
bench_unused_files_list=$(
  for f in ./bench_sweep_*.txt.unused; do
    [ -e "$f" ] || continue    # skip non-matching globs
    [ -f "$f" ] && printf '%s\n' "${f##*/}"
  done 2>/dev/null
)
all_bench_sweep_unused_result_files=$(printf '%s\n' "$bench_unused_files_list" | grep -E "^bench_sweep_result\..*\.${CONTEXT}\.txt\.unused$" 2>/dev/null || true)
if [[ -n "$all_bench_sweep_unused_result_files" ]]; then
  echo "[$(timestamp) Found $(printf '%s\n' "$all_bench_sweep_unused_result_files" | wc -l | tr -d ' ') .txt.unused result file(s) (tensors ignored by the model); they will be collected as baseline-equivalent results."
fi

# find_group_indexes_for_tensor <tensor> -> prints zero-or-more group indices (one per line)
declare -a GROUP_IDXS_RESULT=()
find_group_indexes_for_tensor() {
  # sets GROUP_IDXS_RESULT instead of printing: being read back through a process
  # substitution cost one fork per tensor per qtype, on top of the sed forks above.
  GROUP_IDXS_RESULT=()
  if [[ "$GROUP_TENSORS_DISABLED" == "true" ]]; then
    # leave GROUP_IDXS_RESULT empty -> caller receives an empty array
    return 0
  fi

  local tensor="$1" idx start count i
  for idx in "${!GROUP_TENSORS_RAW[@]}"; do
    start="${GROUP_REGEX_START[$idx]}"
    count="${GROUP_REGEX_COUNT[$idx]}"
    for (( i = start; i < start + count; i++ )); do
      if [[ $tensor =~ ${GROUP_REGEX_FLAT[$i]} ]]; then
        GROUP_IDXS_RESULT+=("$idx")
        # one matching regex per group is enough -> move to next group
        break
      fi
    done
  done
  return 0
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

# If auto-baseline requested, attempt to read bench_sweep_result.baseline.<qtype>.<CONTEXT>.txt
if [[ -n "$AUTO_BASELINE_QTYPE" ]]; then
  # NOTE: do NOT lowercase the qtype here. benchmark_each_tensor.sh writes this file
  # as bench_sweep_result.baseline.${BASELINE_QTYPE}.<CONTEXT>.txt using the qtype
  # verbatim, so lowercasing it made every q*_K / q*_KV baseline (q2_K, q6_K, q8_KV)
  # impossible to find. Match verbatim first, then fall back to a case-insensitive
  # lookup so a mis-cased --auto-baseline still resolves instead of silently giving up.
  baseline_fname="bench_sweep_result.baseline.${AUTO_BASELINE_QTYPE}.${CONTEXT}.txt"
  if ! grep -qF -- "$baseline_fname" <<< "$all_bench_sweep_result_files"; then
    _ci_fname="$(grep -ixF -- "$baseline_fname" <<< "$all_bench_sweep_result_files" | head -n1 || true)"
    if [[ -n "$_ci_fname" ]]; then
      _fixed_qtype="${_ci_fname#bench_sweep_result.baseline.}"
      _fixed_qtype="${_fixed_qtype%.${CONTEXT}.txt}"
      echo "[$(timestamp) ⚠️  Auto-baseline: corrected qtype casing '${AUTO_BASELINE_QTYPE}' -> '${_fixed_qtype}' to match the on-disk file '${_ci_fname}'"
      AUTO_BASELINE_QTYPE="$_fixed_qtype"
      baseline_fname="$_ci_fname"
    fi
    unset _ci_fname _fixed_qtype
  fi
  if grep -qF -- "$baseline_fname" <<< "$all_bench_sweep_result_files"; then
    parsed=$(extract_pp_tg_from_file "./${baseline_fname}" || true)
    if [[ -n "$parsed" ]]; then
      base_pp="${parsed%%|*}"
      base_tg="${parsed#*|}"
      echo "[$(timestamp) Auto-baseline: extracted for qtype=${AUTO_BASELINE_QTYPE}: PP=${base_pp}, TG=${base_tg}"
      [[ -n "$BASELINE_PP_QTYPE" && "$AUTO_BASELINE_QTYPE" == "$BASELINE_PP_QTYPE" && -n "$BASELINE_PP" ]] && echo "[$(timestamp) BASELINE_PP already user-defined, not replaced!" || { BASELINE_PP=${base_pp} && BASELINE_PP_QTYPE=${AUTO_BASELINE_QTYPE} && echo "[$(timestamp) BASELINE_PP='$BASELINE_PP' and BASELINE_PP_QTYPE='$BASELINE_PP_QTYPE' have now been set"; }
      [[ -n "$BASELINE_TG_QTYPE" && "$AUTO_BASELINE_QTYPE" == "$BASELINE_TG_QTYPE" && -n "$BASELINE_TG" ]] && echo "[$(timestamp) BASELINE_TG already user-defined, not replaced!" || { BASELINE_TG=${base_tg} && BASELINE_TG_QTYPE=${AUTO_BASELINE_QTYPE} && echo "[$(timestamp) BASELINE_TG='$BASELINE_TG' and BASELINE_TG_QTYPE='$BASELINE_TG_QTYPE' have now been set"; }
    else
      echo "[$(timestamp) Auto-baseline: baseline file exists but no N_KV=${N_KV} row found in $baseline_fname"
    fi
  else
    echo "[$(timestamp) Auto-baseline: baseline file $baseline_fname not found for qtype=${AUTO_BASELINE_QTYPE}"
  fi
fi

# --inject-baseline-pp-qtype / --inject-baseline-tg-qtype only make sense if there is
# an actual baseline value to inject. It can come from three places:
#   1. --baseline-pp / --baseline-tg                                  (explicit)
#   2. --auto-baseline QTYPE                                          (resolved just above)
#   3. bench_sweep_result.baseline.<qtype>.<CONTEXT>.txt in this directory
# Source 3 is what the --help text has always promised for these two flags, but it was
# never actually implemented: a lone --inject-baseline-pp-qtype silently produced 404
# for the whole qtype even with the baseline file sitting in the directory. Resolve it
# here, and hard-fail when none of the three yields a value.
for __ibq in PP TG; do
  __ibq_qtype_var="BASELINE_${__ibq}_QTYPE"
  __ibq_val_var="BASELINE_${__ibq}"
  __ibq_qtype="${!__ibq_qtype_var}"
  if [[ -z "$__ibq_qtype" || -n "${!__ibq_val_var}" ]]; then
    continue
  fi
  __ibq_file="bench_sweep_result.baseline.${__ibq_qtype}.${CONTEXT}.txt"
  if [[ -f "$__ibq_file" ]]; then
    __ibq_parsed="$(extract_pp_tg_from_file "./${__ibq_file}" || true)"
    if [[ -n "$__ibq_parsed" ]]; then
      if [[ "$__ibq" == "PP" ]]; then
        printf -v "$__ibq_val_var" '%s' "${__ibq_parsed%%|*}"
      else
        printf -v "$__ibq_val_var" '%s' "${__ibq_parsed#*|}"
      fi
      echo "[$(timestamp) --inject-baseline-${__ibq,,}-qtype: resolved ${__ibq}=${!__ibq_val_var} for qtype='${__ibq_qtype}' from ${__ibq_file}"
    fi
  fi
  if [[ -z "${!__ibq_val_var}" ]]; then
    echo "[$(timestamp) ❌ Error: --inject-baseline-${__ibq,,}-qtype '${__ibq_qtype}' was given but no baseline ${__ibq} value could be resolved for it. Supply one of: --baseline-${__ibq,,} VALUE, or --auto-baseline ${__ibq_qtype}, or place a readable '${__ibq_file}' (with an N_KV=${N_KV} row) in the current directory." >&2
    exit 1
  fi
done
unset __ibq __ibq_qtype_var __ibq_val_var __ibq_qtype __ibq_file __ibq_parsed

# write result into an output array passed by name
remove_items_from_list_lines_inplace() {
  local -n _list="$1"
  local -n _remove="$2"
  local -n _out="$3"

  _out=()
  for item in "${_list[@]}"; do
    local skip=false
    for rm in "${_remove[@]}"; do
      [[ "$item" == "$rm" ]] && { skip=true; break; }
    done
    $skip || _out+=("$item")
  done
}

# 2. For each qtype, parse tensors.{qtype}.map and collect results (with grouping support)
for qtype in "${QTYPES[@]}"; do
  mapfile="tensors.${qtype}.map"
  if [[ ! -f "$mapfile" ]]; then
    echo "[$(timestamp) Warning: expected map file '$mapfile' not found. Skipping qtype='$qtype'." >&2
    continue
  fi
  echo "[$(timestamp) Processing map file: $mapfile"

  # Track unprocessed group ids for this qtype
  QTYPE_REMAINING_GROUP_IDS=("${ALL_GROUP_IDS[@]}")

  # read all lines of mapfile into array for flexible scanning/group collection
  mapfile -t MAP_LINES < "$mapfile"

  # Build quick list of tensor names available in this qtype's map
  declare -a TENS_IN_MAP=()
  for line in "${MAP_LINES[@]}"; do
    [[ -z "$line" ]] && continue
    IFS=':' read -r _fname _hash tensor_name _ <<< "$line"
    TENS_IN_MAP+=("$tensor_name")
  done

  # Precompute each group's member list for THIS qtype's map, once. It depends only on
  # the map and the group regexes, so it is invariant across the map-line loop below.
  # It used to be rebuilt for every map line, scanning all tensors and doing a linear
  # " ${group_members[*]} " substring test per candidate - O(n^2) with a big string
  # rebuild each time. On a 1524-tensor map with a '.*' group that alone took minutes.
  declare -a QT_GROUP_MEMBERS_FLAT=()
  declare -a QT_GROUP_MEMBERS_START=()
  declare -a QT_GROUP_MEMBERS_COUNT=()
  for _qgm_idx in "${!GROUP_TENSORS_RAW[@]}"; do
    QT_GROUP_MEMBERS_START[$_qgm_idx]=${#QT_GROUP_MEMBERS_FLAT[@]}
    _qgm_cnt=0
    declare -A _qgm_seen=()
    _qgm_s="${GROUP_REGEX_START[$_qgm_idx]}"
    _qgm_c="${GROUP_REGEX_COUNT[$_qgm_idx]}"
    for (( _qgm_i = _qgm_s; _qgm_i < _qgm_s + _qgm_c; _qgm_i++ )); do
      _qgm_reg="${GROUP_REGEX_FLAT[$_qgm_i]}"
      for _qgm_t in ${TENS_IN_MAP[@]+"${TENS_IN_MAP[@]}"}; do
        if [[ $_qgm_t =~ $_qgm_reg ]] && [[ -z "${_qgm_seen[$_qgm_t]:-}" ]]; then
          _qgm_seen["$_qgm_t"]=1
          QT_GROUP_MEMBERS_FLAT+=("$_qgm_t")
          _qgm_cnt=$((_qgm_cnt + 1))
        fi
      done
    done
    QT_GROUP_MEMBERS_COUNT[$_qgm_idx]=$_qgm_cnt
    unset _qgm_seen
  done
  unset _qgm_idx _qgm_cnt _qgm_s _qgm_c _qgm_i _qgm_reg _qgm_t
  _gm_cur_key=""   # which (qtype|group) group_members currently holds

  PROCESSED_GROUP_IDS=()

  # iterate through entries in MAP_LINES
  for line in "${MAP_LINES[@]}"; do
    # If we are in --groups-only mode and there are no more groups to process, then we break this loop
    if [[ "$GROUPS_ONLY" == "true" ]] && (( ${#QTYPE_REMAINING_GROUP_IDS[@]} == 0 )); then
      echo "[$(timestamp)] There are no more groups to process. Moving to next qtype..."
      break
    fi

    [[ -z "$line" ]] && continue
    IFS=':' read -r fname file_hash tensor_name _ <<< "$line"

    # match tensor_name against USER_REGEX
    matched=false
    for pat in "${USER_REGEX[@]}"; do
      if [[ $tensor_name =~ $pat ]]; then matched=true; break; fi
    done
    [[ "$matched" == true ]] || continue

    # Determine all group indices for this tensor (could be zero..N)
    find_group_indexes_for_tensor "$tensor_name"
    group_idxs_for_tensor=( ${GROUP_IDXS_RESULT[@]+"${GROUP_IDXS_RESULT[@]}"} )

    # If grouping is enabled and this tensor belongs to one or more group, attempt to process the groups
    if (( ${#group_idxs_for_tensor[@]} > 0 )); then

      # Decide whether to add a column placeholder based on grouping & expansion & hide-empty
      if [[ "$EXPAND_GROUPS" == "true" ]]; then
        # user wants member columns: include the individual tensor as a column unless hide-empty==true
        [[ "$HIDE_EMPTY" == "false" ]] && TENSOR_SET["$tensor_name"]=1
      fi

      # iterate over all groups this tensor belongs to and handle each group separately
      for group_idx_for_tensor in "${group_idxs_for_tensor[@]}"; do
        proc_key="${qtype}|${group_idx_for_tensor}"
        # If this group for this qtype has already been processed (value '1'), skip individual handling.
        # We do NOT skip when the marker is 'MISSING' — that allows falling back to per-tensor files.
        if [[ "${PROCESSED_GROUP_QTYPE[$proc_key]:-}" == "1" ]]; then
          continue
        fi

        # collect all group members present in this qtype's map
        # IMPORTANT: If there is more than one group, this array will be overwritten, which is fine, just make sure to inform the user!
        # Taken from the per-qtype table precomputed above; only re-sliced when the
        # (qtype, group) actually changes, so the common single-group case copies once.
        _gm_key="${qtype}|${group_idx_for_tensor}"
        if [[ "$_gm_cur_key" != "$_gm_key" ]]; then
          _gm_s="${QT_GROUP_MEMBERS_START[$group_idx_for_tensor]}"
          _gm_c="${QT_GROUP_MEMBERS_COUNT[$group_idx_for_tensor]}"
          if (( _gm_c > 0 )); then
            group_members=( "${QT_GROUP_MEMBERS_FLAT[@]:_gm_s:_gm_c}" )
          else
            group_members=()
          fi
          _gm_cur_key="$_gm_key"
        fi

        if (( ${#group_members[@]} == 0 )); then
          echo "[$(timestamp) Warning: no group members found in map for group #${group_idx_for_tensor} (qtype=${qtype}). Skipping group." >&2
          PROCESSED_GROUP_QTYPE["$proc_key"]=1
          PROCESSED_GROUP_IDS+=(${group_idx_for_tensor})
          _tmp_qtype_remaining=()
          remove_items_from_list_lines_inplace QTYPE_REMAINING_GROUP_IDS PROCESSED_GROUP_IDS _tmp_qtype_remaining
          QTYPE_REMAINING_GROUP_IDS=( "${_tmp_qtype_remaining[@]}" )
          unset _tmp_qtype_remaining
          continue
        fi
      done
    # Decide whether to add a column placeholder based on grouping & expansion & hide-empty
    else
      # not in a group -> individual tensor column
      ([[ "$HIDE_EMPTY" == "false" ]] && [[ "$GROUPS_ONLY" != "true" ]]) && TENSOR_SET["$tensor_name"]=1
    fi

    # If grouping is enabled and this tensor belongs to a group, attempt to process the group (continuation)
    if (( ${#group_idxs_for_tensor[@]} > 0 )); then
      # iterate over all groups this tensor belongs to and handle each group separately
      for group_idx_for_tensor in "${group_idxs_for_tensor[@]}"; do
        proc_key="${qtype}|${group_idx_for_tensor}"
        # Look for group result file: bench_sweep_result.group{group_idx_for_tensor}.{qtype}.{CONTEXT}.txt
        group_result_filename="bench_sweep_result.group${group_idx_for_tensor}.${qtype}.${CONTEXT}.txt"
        # Accept the .txt.unused quarantine variant when the normal result is
        # absent (group members ignored by the model: metrics are the
        # baseline's by definition)
        group_result_is_unused=false
        if ! grep -qF -- "$group_result_filename" <<< "$all_bench_sweep_result_files" \
           && grep -qF -- "${group_result_filename}.unused" <<< "$all_bench_sweep_unused_result_files"; then
          group_result_filename="${group_result_filename}.unused"
          group_result_is_unused=true
        fi
        # confirm it exists in directory listing
        if [[ "$group_result_is_unused" == "false" ]] && ! grep -qF -- "$group_result_filename" <<< "$all_bench_sweep_result_files"; then
          # Only log the "missing group file" message once per (qtype, group).
          if [[ -z "${PROCESSED_GROUP_QTYPE[$proc_key]:-}" ]]; then
            echo "[$(timestamp) No group sweep result file found for group #${group_idx_for_tensor}, qtype=${qtype}: expected '$group_result_filename'. Will fall back to individual tensor files (unless --groups-only is enabled)."
            # Mark as 'missing' so we don't re-print this for other members of the same group/qtype.
            PROCESSED_GROUP_QTYPE["$proc_key"]="MISSING"
            PROCESSED_GROUP_IDS+=(${group_idx_for_tensor})
            _tmp_qtype_remaining=()
            remove_items_from_list_lines_inplace QTYPE_REMAINING_GROUP_IDS PROCESSED_GROUP_IDS _tmp_qtype_remaining
            QTYPE_REMAINING_GROUP_IDS=( "${_tmp_qtype_remaining[@]}" )
            unset _tmp_qtype_remaining
          fi
          # fall back to per-tensor handling
        elif [[ -z "${PROCESSED_GROUP_QTYPE[$proc_key]:-}" ]]; then
          echo "[$(timestamp) Found group sweep result file: $group_result_filename -> applying to ${#group_members[@]} member(s)."
          result_file="./${group_result_filename}"

          # parse file to extract S_PP and S_TG for requested N_KV
          parsed=$(extract_pp_tg_from_file "$result_file" || true)

          if [[ -n "$parsed" ]]; then
            SPP_VAL="${parsed%%|*}"
            STG_VAL="${parsed#*|}"
            # trim with parameter expansion rather than a sed(1) fork (this runs per result)
            SPP_VAL="${SPP_VAL#"${SPP_VAL%%[![:space:]]*}"}"; SPP_VAL="${SPP_VAL%"${SPP_VAL##*[![:space:]]}"}"
            STG_VAL="${STG_VAL#"${STG_VAL%%[![:space:]]*}"}"; STG_VAL="${STG_VAL%"${STG_VAL##*[![:space:]]}"}"
            echo "[$(timestamp) Extracted group #${group_idx_for_tensor} (qtype=${qtype}): S_PP=$SPP_VAL, S_TG=$STG_VAL"
          elif [[ "$group_result_is_unused" == "true" && ( -n "${BASELINE_PP:-}" || -n "${BASELINE_TG:-}" ) ]]; then
            SPP_VAL="${BASELINE_PP:-404}"
            STG_VAL="${BASELINE_TG:-404}"
            echo "[$(timestamp) Group #${group_idx_for_tensor} (qtype=${qtype}) is unused by the model; injecting baseline S_PP=$SPP_VAL, S_TG=$STG_VAL"
          else
            echo "[$(timestamp) Warning: no row with N_KV=${N_KV} found in $result_file. Marking 404 for entire group."
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
            PP_VALUES["${qtype}|group${group_idx_for_tensor}"]="$SPP_VAL"
            TG_VALUES["${qtype}|group${group_idx_for_tensor}"]="$STG_VAL"
            # when hide-empty==true and we found a result, ensure the group column is present
            [[ "$HIDE_EMPTY" == true ]] && TENSOR_SET["group${group_idx_for_tensor}"]=1
          fi

          PROCESSED_GROUP_QTYPE["$proc_key"]=1
          PROCESSED_GROUP_IDS+=(${group_idx_for_tensor})
          _tmp_qtype_remaining=()
          remove_items_from_list_lines_inplace QTYPE_REMAINING_GROUP_IDS PROCESSED_GROUP_IDS _tmp_qtype_remaining
          QTYPE_REMAINING_GROUP_IDS=( "${_tmp_qtype_remaining[@]}" )
          unset _tmp_qtype_remaining
          continue
        fi
      done
    fi

    # Skip individual tensors fallback if groups only is used.
    [[ "$GROUPS_ONLY" == "true" ]] && continue

    # Fallback: look for individual per-tensor result
    # If we reach here: either grouping disabled, tensor not in group, OR group file not present -> handle per-tensor file

    # Find matching bench_sweep_result file for this tensor AND qtype (individual).
    # This used to match with the qtype as a wildcard:
    #   ^bench_sweep_result\.${tensor_name}\..*\.${CONTEXT}\.txt$
    # combined with grep -m1, so every qtype column was filled from whichever qtype's
    # file happened to sort first - i.e. individual (non-group) sweep results were
    # silently attributed to the wrong qtype. benchmark_each_tensor.sh writes the qtype
    # into the name (bench_sweep_result.<tensor>.<qtype>.<CONTEXT>.txt), so match it
    # exactly, the same way collect_ppl_results.sh does.
    bench_name="bench_sweep_result.${tensor_name}.${qtype}.${CONTEXT}.txt"
    bench_match=""
    if grep -qxF -- "$bench_name" <<< "$all_bench_sweep_result_files"; then
      bench_match="$bench_name"
    fi

    result_is_unused=false
    if [[ -z "$bench_match" ]]; then
      # Accept the .txt.unused quarantine variant (tensor ignored by the
      # model: metrics are the baseline's by definition)
      if grep -qxF -- "${bench_name}.unused" <<< "$all_bench_sweep_unused_result_files"; then
        bench_match="${bench_name}.unused"
        result_is_unused=true
      else
        # no individual file: leave empty (maybe other qtypes have it)
        continue
      fi
    fi

    echo "[$(timestamp) Found sweep result file for tensor '$tensor_name': $bench_match"
    [[ "$result_is_unused" == "true" ]] && echo "[$(timestamp) Note: '$tensor_name' is unused by the model; its speed metrics equal the baseline's."

    # ensure included if hide-empty true
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
        if (NF >= 8 && $2 ~ /^[0-9]+$/ && $4 == nkv) {
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
      # ensure trimmed (parameter expansion rather than a sed(1) fork - this runs per tensor)
      SPP_VAL="${SPP_VAL#"${SPP_VAL%%[![:space:]]*}"}"; SPP_VAL="${SPP_VAL%"${SPP_VAL##*[![:space:]]}"}"
      STG_VAL="${STG_VAL#"${STG_VAL%%[![:space:]]*}"}"; STG_VAL="${STG_VAL%"${STG_VAL##*[![:space:]]}"}"
      echo "[$(timestamp) Extracted for tensor='$tensor_name', qtype='$qtype': S_PP=$SPP_VAL, S_TG=$STG_VAL"
    elif [[ "$result_is_unused" == "true" && ( -n "${BASELINE_PP:-}" || -n "${BASELINE_TG:-}" ) ]]; then
      SPP_VAL="${BASELINE_PP:-404}"
      STG_VAL="${BASELINE_TG:-404}"
      echo "[$(timestamp) Tensor '$tensor_name' (qtype=${qtype}) is unused by the model; injecting baseline S_PP=$SPP_VAL, S_TG=$STG_VAL"
    else
      echo "[$(timestamp) Warning: no row with N_KV=${N_KV} found in $result_file. Marking 404."
      SPP_VAL="404"
      STG_VAL="404"
    fi

    PP_VALUES["${qtype}|${tensor_name}"]="$SPP_VAL"
    TG_VALUES["${qtype}|${tensor_name}"]="$STG_VAL"

  done # end iterating MAP_LINES
done # end for qtype

# 3. Build sorted list of all tensor names (or groups) for header
tensor_list=("${!TENSOR_SET[@]}")
if [[ ${#tensor_list[@]} -eq 0 ]]; then
  echo "Warning: No tensor names matched USER_REGEX in any map files (or no results found). Exiting." >&2
  exit 1
fi
IFS=$'\n' sorted_tensors=($(printf '%s\n' "${tensor_list[@]}" | sort -Vu))
unset IFS

# 4. Write PP CSV
echo "[$(timestamp) Writing PP CSV to $OUTPUT_PP_CSV"

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
      if [[ -n "$val" ]]; then
        echo "[DEBUG] Raw value for [$key] = '$val'" >&2
      elif [[ "$BASELINE_PP_QTYPE" != "$qtype" || -z "${BASELINE_PP:-}" ]]; then
        echo "[DEBUG] Empty value for [$key] = '$val', will use \"404\" instead" >&2
        val="404"
      else
        echo "[DEBUG] Empty value for [$key] = '$val' (expected for baseline), will use baseline value \"${BASELINE_PP:-}\" instead" >&2
        val="$BASELINE_PP"
      fi

      # If a global baseline exists, compute percent-delta across all qtypes
      # Percentage computation: only when baseline present and --no-percentage not set
      if [[ "$NO_PERCENTAGE" != "true" && -n "${BASELINE_PP:-}" ]]; then
        if [[ "$val" == "404" ]]; then
          val="404%"
        else
          # detect division by zero for baseline (exactly zero numeric)
          if awk -v b="$BASELINE_PP" 'BEGIN{ if ((b+0)==0) exit 0; exit 1 }'; then
            # baseline is numeric zero -> avoid division by zero, mark as 404%
            val="404%"
            echo "[DEBUG] Baseline value is zero, avoiding division by zero for [$key]. Using '404%'" >&2
          else
            pct=$(awk -v b="$BASELINE_PP" -v v="$val" 'BEGIN{printf "%+.2f%%", (v-b)/b*100}')
            val="$pct"
            echo "[DEBUG] Final value for [$key] = '$val'" >&2
          fi
        fi
      elif [[ "$NO_PERCENTAGE" == "true" ]]; then
        # when no-percentage requested: do not compute %, just output raw value.
        # baseline numeric values were already injected earlier (if applicable).
        # make sure baseline qtype row is not forced to "0%"; leave raw value.
        :
      else
        # If baseline present but value empty? handled above; else, if baseline present and this qtype equals baseline qtype,
        # the script previously forced "0%". Keep that behavior only if percentages are enabled.
        if [[ -n "${BASELINE_PP:-}" && "$BASELINE_PP_QTYPE" == "$qtype" && "$NO_PERCENTAGE" != "true" ]]; then
          val="0%"
          echo "[DEBUG] Final value set to baseline for [$key] = '$val'" >&2
        fi
      fi

      printf ',%s' "$val"
    done
    printf '\n'
  done
} > "$OUTPUT_PP_CSV"

echo "[DEBUG] Finished writing PP CSV."

# 5. Write TG CSV
echo "[$(timestamp) Writing TG CSV to $OUTPUT_TG_CSV"

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
      if [[ -n "$val" ]]; then
        echo "[DEBUG] Raw value for [$key] = '$val'" >&2
      elif [[ "$BASELINE_TG_QTYPE" != "$qtype" || -z "${BASELINE_TG:-}" ]]; then
        echo "[DEBUG] Empty value for [$key] = '$val', will use \"404\" instead" >&2
        val="404"
      else
        echo "[DEBUG] Empty value for [$key] = '$val' (expected for baseline), will use baseline value \"${BASELINE_TG:-}\" instead" >&2
        val="$BASELINE_TG"
      fi

      # If a global baseline exists, compute percent-delta across all qtypes
      # Percentage computation: only when baseline present and --no-percentage not set
      if [[ "$NO_PERCENTAGE" != "true" && -n "${BASELINE_TG:-}" ]]; then
        if [[ "$val" == "404" ]]; then
          val="404%"
        else
          # detect division by zero for baseline (exactly zero numeric)
          if awk -v b="$BASELINE_TG" 'BEGIN{ if ((b+0)==0) exit 0; exit 1 }'; then
            # baseline is numeric zero -> avoid division by zero, mark as 404%
            val="404%"
            echo "[DEBUG] Baseline value is zero, avoiding division by zero for [$key]. Using '404%'" >&2
          else
            pct=$(awk -v b="$BASELINE_TG" -v v="$val" 'BEGIN{printf "%+.2f%%", (v-b)/b*100}')
            val="$pct"
            echo "[DEBUG] Final value for [$key] = '$val'" >&2
          fi
        fi
      elif [[ "$NO_PERCENTAGE" == "true" ]]; then
        # when no-percentage requested: do not compute %, just output raw value.
        # baseline numeric values were already injected earlier (if aTGlicable).
        # make sure baseline qtype row is not forced to "0%"; leave raw value.
        :
      else
        # If baseline present but value empty? handled above; else, if baseline present and this qtype equals baseline qtype,
        # the script previously forced "0%". Keep that behavior only if percentages are enabled.
        if [[ -n "${BASELINE_TG:-}" && "$BASELINE_TG_QTYPE" == "$qtype" && "$NO_PERCENTAGE" != "true" ]]; then
          val="0%"
          echo "[DEBUG] Final value set to baseline for [$key] = '$val'" >&2
        fi
      fi

      printf ',%s' "$val"
    done
    printf '\n'
  done
} > "$OUTPUT_TG_CSV"


# 6. The end!

echo "[$(timestamp) All Done."
echo "[$(timestamp) CSVs available at: $OUTPUT_PP_CSV and $OUTPUT_TG_CSV"
exit 0 # ensure a clean exit code regardless of any conditional line added above
