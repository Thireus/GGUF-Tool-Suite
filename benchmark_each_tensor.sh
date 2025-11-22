#!/usr/bin/env bash
#***************************************************************#
#** This script is part of Thireus' GGUF Tool Suite.          **#
#** benchmark_each_tensor.sh is a tool that evaluates the     **#
#** sensitivity to heavy quantisation of each tensor.         **#
#**                                                           **#
#** ********************************************************* **#
#** --------------- Updated: Nov-22-2025 -------------------- **#
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
#** Copyright © 2025 - Thireus.            Fᵢₙₑ₋ₜᵤₙₑ𝒹 ₒₙ 𝒸ₕₐₒₛ. **#
#***************************************************************#
#**PLEASE REFER TO THE README FILE FOR ADDITIONAL INFORMATION!**#
#***************************************************************#

# Exit on error, undefined variable, or pipe failure
set -euo pipefail

# ----------------------------------------------------------------------------
# Add a flag for delayed exit: first SIGINT sets EXIT_PENDING=1, second sends SIGTERM to llama-perplexity and all, third exits immediately
EXIT_PENDING=0
SIGINT_COUNT=0
LLAMA_PID=""
INT_TIMESTAMP=0
# Trap SIGINT (Ctrl+C) to set the flag on first, send SIGTERM on second, send SIGKILL on third, immediate exit on fourth
trap '
  SIGINT_COUNT=$((SIGINT_COUNT+1))
  if [[ $SIGINT_COUNT -eq 1 ]]; then
    EXIT_PENDING=1
    echo "[$(date "+%Y-%m-%d %H:%M:%S")] Received termination signal. Will exit after current operation finishes."
  elif [[ $SIGINT_COUNT -eq 2 ]]; then
    echo "[$(date "+%Y-%m-%d %H:%M:%S")] Received second termination signal. Sending SIGTERM to llama-perplexity."
    [[ -n "$LLAMA_PID" ]] && kill -SIGTERM "$LLAMA_PID"
  elif [[ $SIGINT_COUNT -eq 3 ]]; then
    echo "[$(date "+%Y-%m-%d %H:%M:%S")] Received third termination signal. Sending SIGKILL to llama-perplexity."
    [[ -n "$LLAMA_PID" ]] && kill -SIGKILL "$LLAMA_PID"
  else
    # If you have any cleanup traps for FIFOs etc, they will run because we exit now.
    if [[ $INT_TIMESTAMP -eq 0 ]]; then
      echo "💀 Received fourth termination signal — forwarding SIGINT to the entire process group and for graceful shutdown." >&2
      echo "   Allowing up to 10 seconds for subprocesses to exit cleanly if possible." >&2
      INT_TIMESTAMP=$(date +%s)
    elif [[ $(( $(date +%s) - INT_TIMESTAMP )) -gt 10 ]]; then
      echo "💀 10 seconds elapsed — forcing termination now: sending SIGKILL to the process group (-$$)." >&2
      kill -s KILL -- -$$ 2>/dev/null || true
    fi
    sleep 1 # Slow down the shutdown_on_signal loop
    kill -s INT -- -$$ 2>/dev/null || true
    exit 1
  fi
' SIGINT
# You can leave SIGTERM as before if you prefer it to be immediate:
trap 'echo "[$(date "+%Y-%m-%d %H:%M:%S")] Received termination signal. Exiting immediately."; kill -s KILL -- -$$ 2>/dev/null || true; exit 1' SIGTERM
# ----------------------------------------------------------------------------

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }

# --------------- DETECT & DEFINE SHA256 HELPER ---------------
if command -v sha256sum >/dev/null 2>&1; then
  # GNU coreutils on Linux
  sha256tool=(sha256sum)
  args=()
elif command -v gsha256sum >/dev/null 2>&1; then
  # GNU coreutils on macOS (via Homebrew)
  sha256tool=(gsha256sum)
  args=()
elif command -v shasum >/dev/null 2>&1; then
  # macOS built-in (Perl script)
  sha256tool=(shasum)
  args=(-a 256)
elif command -v openssl >/dev/null 2>&1; then
  # OpenSSL fallback
  sha256tool=(openssl)
  args=(dgst -sha256)
else
  # fallback stub: always errors out
  sha256tool=()
  args=()
fi

# _sha256sum reads either from file (if you pass an arg) or from stdin
_sha256sum() {
  if (( $# > 0 )); then
    # file-mode: pass filename as $1
    "${sha256tool[@]}" "${args[@]}" "$1" | awk '{print $1}'
  else
    # stdin-mode: read data from pipe
    "${sha256tool[@]}" "${args[@]}" | awk '{print $1}'
  fi
}

# --- pure-Bash shuffle replacement for `shuf` ---
shuf() {
  local lines=() line n i j tmp
  # Read all stdin lines into an array
  while IFS= read -r line; do
    lines+=("$line")
  done

  # Fisher–Yates shuffle
  n=${#lines[@]}
  for (( i = n - 1; i > 0; i-- )); do
    j=$(( RANDOM % (i + 1) ))
    tmp=${lines[i]}
    lines[i]=${lines[j]}
    lines[j]=$tmp
  done

  # Print shuffled lines
  for line in "${lines[@]}"; do
    printf '%s\n' "$line"
  done
}

# ================= COMMAND-LINE ARGUMENTS =================
BENCH_CSV=""
BENCH_FROM_QTYPE=""
CUSTOM_QTYPES=()
CUSTOM_CHUNKS=""
CUSTOM_CONTEXT=""
SKIP_GPG=false # If true, skip the gpg signature verification of the signed files

# GROUP_TENSORS_RAW: array of group specifications, each element is a comma-separated
# list of regexes that define that group (e.g. 'a1,a2' or 'blk\..*\.ffn_up_exps.*,blk\..*\.ffn_gate_exps.*')
GROUP_TENSORS_RAW=()
# When the single token '[]' is passed, grouping is disabled
GROUP_TENSORS_DISABLED=false

# Only benchmark groups (requires --group-tensors to be provided)
BENCH_GROUPS_ONLY=false

# Bench mode and sweep context
# BENCH_MODE: 0 = PPL+KLD only (default), 1 = SWEEP only, 2 = PPL+KLD then SWEEP
BENCH_MODE=0

# Control infinite looping (defaults to false)
INFINITE_LOOP=false

# Defines if Kullback–Leibler divergence (KLD) must not be computed (defaults to false)
NO_KLD=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --benchmark-worst-tensors)
      BENCH_CSV="$2"; shift 2;;
    --benchmark-worst-tensors-from-qtype)
      BENCH_FROM_QTYPE="$2"; shift 2;;
    --qtypes)
      shift; while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
        CUSTOM_QTYPES+=("$1"); shift
      done;;
    --chunks)
      CUSTOM_CHUNKS="$2"; shift 2;;
    --skip-gpg)
      SKIP_GPG=true
      shift
      ;;
    --group-tensors)
      shift
      GROUP_TENSORS_RAW=()
      # collect one or more group specs (nargs '+')
      while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
        GROUP_TENSORS_RAW+=("$1"); shift
      done
      ;;
    --benchmark-groups-only)
      # require --group-tensors to be set (validated later after parsing defaults)
      # In this mode, even if individual tensors or all tensors from a different larger group have been benchmarked, the group will still be benchmarked
      # When this mode is not used, the groups will only be benchmarked if both of these statements are true:
      # 1. No other group that contains all the tensors of that group has been benchmarked
      # 2. All individual tensors of that group have never been benchmarked
      # Example: if 
      BENCH_GROUPS_ONLY=true
      shift
      ;;
    --mode)
      BENCH_MODE="$2"; shift 2;;
    --context)
      CUSTOM_CONTEXT="$2"; shift 2;;
    --infinite-loop)
      INFINITE_LOOP=true
      shift
      ;;
    --no-kld)
      NO_KLD=true
      shift
      ;;
    *) echo "Unknown argument: $1" >&2; exit 1;;
  esac
done

# Default grouping if not provided or if the user explicitly passed '[]' as first group, disable grouping
if (( ${#GROUP_TENSORS_RAW[@]} == 0 )) || ((( ${#GROUP_TENSORS_RAW[@]} == 1 )) && [[ "${GROUP_TENSORS_RAW[0]}" == "[]" ]]); then
  GROUP_TENSORS_DISABLED=true
  GROUP_TENSORS_RAW=()
fi

# Validate that if user asked to benchmark groups only, they passed --group-tensors
if [[ "$BENCH_GROUPS_ONLY" == "true" && "$GROUP_TENSORS_DISABLED" == "true" ]]; then
  echo "Error: --benchmark-groups-only requires --group-tensors to be set (provide one or more group regex specifications)." >&2
  exit 1
fi

if [[ -n "$BENCH_CSV" && -z "$BENCH_FROM_QTYPE" ]]; then
  echo "Error: --benchmark-worst-tensors-from-qtype must be provided when --benchmark-worst-tensors is used." >&2
  exit 1
fi

# Validate BENCH_MODE
if ! [[ "$BENCH_MODE" =~ ^[0-2]$ ]]; then
  echo "Error: --mode must be 0 (PPL+KLD), 1 (SWEEP), or 2 (PPL+KLD then SWEEP)." >&2
  exit 1
fi

# ================= USER CONFIGURATION =================

# 1. Remote connection settings for tensor_downloader.sh:
# Please edit tensor_downloader.sh!
# Resolve script directory for locating tensor_downloader.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TENSOR_DOWNLOADER="$SCRIPT_DIR/tensor_downloader.sh"
QUANT_DOWNLOADER="$SCRIPT_DIR/quant_downloader.sh"

if [[ ! -x "$TENSOR_DOWNLOADER" ]]; then
    echo "Error: tensor_downloader.sh not found or not executable at $TENSOR_DOWNLOADER" >&2
    exit 1
fi

run_tensor_downloader() {
  set +e
  "$TENSOR_DOWNLOADER" "$@"
  local ret=$?
  set -e
  return $ret
}

run_quant_downloader() {
  set +e
  "$QUANT_DOWNLOADER" "$@"
  local ret=$?
  set -e
  return $ret
}

# 2. Local directories:
#    Where to store downloaded shard files temporarily:
LOCAL_DOWNLOAD_DIR="./downloaded_shards"
#    Where your local model shards live, so the script can find and replace them for benchmarking.
LOCAL_MODEL_DIR="./"

# 3. USER_REGEX entries include original qtype after first '=', second '=' ensures the script doesn't benchmark those entries but still fetches the right qtype for these tensors
USER_REGEX=(
  # Tensors set to f32 are supposed to be found in any qtype because they are always left unquantised, make sure they are always locked

  # Token embedding and output tensors (GPU)
  # note token_embd cannot be repacked quant type
  '^output\.weight$=q8_0'
  '^output_norm\.weight$=f32=locked'
  # Be extremely careful about this one, especially if benchmarking below iq1_m, since it cannot be quantised to something lower than iq1_m, which is what will be used during benchmarking! Which will introduce incorrect ppl benchmark.
  '^token_embd\.weight$=q8_0'

  # GPU Only - not divisible by 256 so only supports qN_0
  # I recommend against unlocking this tensor, especially since it cannot be quantised lower than iq2_ks, so any benchmark using lower quant than this will be faulty for this tensor
  '^blk\.([0-9]|[1-5][0-9]|60)\.attn_k_b\.weight$=q8_0=locked'

  # GPU Only
  # Best to keep this one locked for Kimi-K2 because it cannot be quantised lower than iq2_ks, so any benchmark using lower quant than this will be faulty for this tensor
  '^blk\.([0-9]|[1-5][0-9]|60)\.attn_v_b\.weight$=q8_0=locked'

  # GPU Only
  '^blk\.([0-9]|[1-5][0-9]|60)\.attn_kv_b\.weight$=q8_0=locked'

  # GPU Only
  '^blk\.([0-9]|[1-5][0-9]|60)\.attn_kv_a_norm\.weight$=f32=locked'
  '^blk\.([0-9]|[1-5][0-9]|60)\.attn_q_a\.weight$=q8_0=locked'
  '^blk\.([0-9]|[1-5][0-9]|60)\.attn_q_a_norm\.weight$=f32=locked'
  '^blk\.([0-9]|[1-5][0-9]|60)\.attn_q_b\.weight$=q8_0=locked'
  '^blk\.([0-9]|[1-5][0-9]|60)\.attn_norm\.weight$=f32=locked'
  '^blk\.([0-9]|[1-5][0-9]|60)\.attn_output\.weight$=q8_0=locked'
  '^blk\.([3-9]|[1-5][0-9]|60)\.exp_probs_b\.bias$=f32=locked'

  # GPU Only
  '^blk\.([0-9]|[1-5][0-9]|60)\.attn_kv_a_mqa\.weight$=q8_0=locked'

  # GPU Only
  '^blk\.[0-2]\.ffn_down\.weight$=q8_0'
  '^blk\.[0-2]\.ffn_up\.weight$=q8_0'
  '^blk\.[0-2]\.ffn_gate\.weight$=q8_0'
  '^blk\.([3-9]|[1-5][0-9]|60)\.ffn_gate_inp\.weight$=q8_0=locked'
  '^blk\.([0-9]|[1-5][0-9]|60)\.ffn_norm\.weight$=q8_0=locked'

  ## GPU-loaded ffn_*_shexp
  '^blk\.([3-9]|[1-5][0-9]|60)\.ffn_down_shexp\.weight$=iq3_xxs'
  '^blk\.([3-9]|[1-5][0-9]|60)\.ffn_up_shexp\.weight$=iq3_xxs'
  '^blk\.([3-9]|[1-5][0-9]|60)\.ffn_gate_shexp\.weight$=iq3_xxs'

  ## CPU-friendly ffn_*_exps
  '^blk\.([3-9]|[1-5][0-9]|60)\.ffn_down_exps\.weight$=iq3_xxs'
  '^blk\.([3-9]|[1-5][0-9]|60)\.ffn_up_exps\.weight$=iq3_xxs'
  '^blk\.([3-9]|[1-5][0-9]|60)\.ffn_gate_exps\.weight$=iq3_xxs'
)

# Extract patterns and associated qtypes
declare -a PATTERNS PATTERN_QTYPES
for entry in "${USER_REGEX[@]}"; do
  IFS='=' read -r pat qtype _locked <<< "$entry"
  PATTERNS+=("$pat")
  PATTERN_QTYPES+=("$qtype")
done
# Derive unique LOCAL_QTYPES sorted, except f32
# Build your LOCAL_QTYPES, removing any “f32”
_LOCAL_QTYPES=( $(printf "%s\n" "${PATTERN_QTYPES[@]}" | sort -u) )
LOCAL_QTYPES=( $(printf "%s\n" "${_LOCAL_QTYPES[@]}" | grep -v '^f32$') )

# Also build USER_REGEX_PATTERNS (left-hand regex portions) for later use
declare -a USER_REGEX_PATTERNS=()
for entry in "${USER_REGEX[@]}"; do
  IFS='=' read -r pat _qrest <<< "$entry"
  USER_REGEX_PATTERNS+=("$pat")
done

# 4. Number of concurrent threads for initial fetch/validation:
N_THREADS=8

# 5. Number of chunks to process for PPL+KLD:
PPL_COMMAND_CHUNKS_TO_PROCESS=${CUSTOM_CHUNKS:-250}

# 6. Max context size for SWEEP:
BENCH_COMMAND_CONTEXT_TO_PROCESS=${CUSTOM_CONTEXT:-8192}

# 7. List of qtypes to process in the loop - it is recommended to assess the tensors.map of these as the quant of some tensors may differ:
# If iq1_s_* is chosen, know that the bench of some tensors like token_embd will be faulty (will be using a higher qtype)
# This is because these tensors have not been quantised to iq1_s due to llama refusal
QTYPES=(${CUSTOM_QTYPES[@]:-"iq1_m_r4" "iq2_k"})

# 8. Baseline QTYPE for baseline PPL+KLD computation, try to best match the recipe's mean quant provided in USER_REGEX
# Try to use the highest baseline you can that fits in your VRAM+RAM
BASELINE_QTYPE="iq3_xxs"

# 9. PPL command template:
# Do not add KLD parameters, they will be automatically added if necessary at the end of this template - See "Add KLD parameter placeholder" section
PPL_COMMAND_TEMPLATE='CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,2,1 ~/ik_llama-main-b3833-65dd65c-bin-win-cuda-12.8-x64/llama-perplexity \
-m {MODEL_FILE} -mla 3 -fa on -amb 1024 -ctk f16 -c 512 -ngl 99 \
-ot "blk\.(3|4|5)\.ffn_.*=CUDA0" -ot "blk\.(6|7|8)\.ffn_.*=CUDA1" -ot "blk\.(9|10)\.ffn_.*=CUDA2" \
-ot exps=CPU -b 4096 -ub 4096 --warmup-batch --no-mmap --threads 36 --main-gpu 0 --seed 1337 \
-f ../../imatrix-calibration-corpus-v02.txt --chunks ${PPL_COMMAND_CHUNKS_TO_PROCESS}'

# 10. SWEEP command template
SWEEP_COMMAND_TEMPLATE='CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 ~/ik_llama-main-b4123-ee719cc-bin-win-cuda-12.8-x64-avx512/llama-sweep-bench \
-m {MODEL_FILE} -mla 3 -fa on -amb 1024 -ctk f16 -ngl 99 \
-b 4096 -ub 4096 --warmup-batch --no-mmap --threads 36 --main-gpu 0 --seed 1337 \
-c ${BENCH_COMMAND_CONTEXT_TO_PROCESS}'

# 11. Pattern to identify the main model shard in LOCAL_MODEL_DIR.
MAIN_SHARD_PATTERN="*-00001-of-*.gguf"

# =============== End USER CONFIGURATION ===============

# Add KLD parameter placeholder to PPL_COMMAND_TEMPLATE if necessary
if [[ "$NO_KLD" == "false" ]] && [[ "$BENCH_MODE" -ne 1 ]]; then
  PPL_COMMAND_TEMPLATE="${PPL_COMMAND_TEMPLATE} {KLD_PARAMETER}"
  PLUS_KLD='+KLD'
  _kld='_kld'
  #echo "[$(timestamp)] KLD computation enabled by default."
else
  PLUS_KLD=''
  _kld=''
  #echo "[$(timestamp)] KLD computation disabled. (--no_kld=true)"
fi

# Verify gpg readiness
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "$SKIP_GPG" != "true" ]]; then
  if [ ! -f "$SCRIPT_DIR/trusted-keys.asc" ]; then
    echo "[$(timestamp)] ❌ Error: trusted-keys.asc not found in the script directory."
    echo "Hint: Provide trusted-keys.asc in the same directory as this script or use the --skip-gpg option to disable gpg signature verification."
    exit 6
  fi
  if command -v gpg >/dev/null 2>&1; then
    # Create a temporary GNUPGHOME
    GNUPG_TMPDIR=$(mktemp -d)
    if [ -z "$GNUPG_TMPDIR" ]; then
      echo "[$(timestamp)] ❌ Error: Failed to create temporary GPG home directory." >&2
      exit 8
    fi
    # Try importing the keys (silently) to check validity
    if ! gpg --homedir "$GNUPG_TMPDIR" --no-default-keyring --import "$SCRIPT_DIR/trusted-keys.asc" > /dev/null 2>&1; then
      echo "[$(timestamp)] ❌ Error: trusted-keys.asc contains missing or invalid GPG public keys."
      echo "Hint: Add valid public keys to this file or re-run with the --skip-gpg option to bypass signature verification."
      [ -n "$GNUPG_TMPDIR" ] && rm -rf "$GNUPG_TMPDIR"
      exit 7
    fi
  else
    echo "[$(timestamp)] ⚠️ Warning: 'gpg' command not found. GPG signature verification skipped." >&2
  fi
fi

# Helper: parse worst-tensor from CSV if requested
if [[ -n "$BENCH_CSV" ]]; then
  echo "[$(timestamp)] Starting worst-tensor selection from CSV: $BENCH_CSV (from qtype=$BENCH_FROM_QTYPE)"
  if [[ ! -f "$BENCH_CSV" ]]; then
    echo "[$(timestamp)] ❌ Error: CSV file '$BENCH_CSV' not found." >&2; exit 1
  fi

  # Read headers and values
  IFS=',' read -r -a hdrs < <(head -n1 "$BENCH_CSV")
  echo "[$(timestamp)] Read ${#hdrs[@]} columns (first is qtype, rest are tensor names)."
  row=$(grep -P "^${BENCH_FROM_QTYPE}," "$BENCH_CSV") || { echo "[$(timestamp)] ❌ Error: qtype '$BENCH_FROM_QTYPE' not in CSV." >&2; exit 1; }
  IFS=',' read -r -a vals <<< "$row"
  echo "[$(timestamp)] Retrieved row for qtype '$BENCH_FROM_QTYPE'."

  # We'll collect both tensor names and their associated qtypes
  declare -a SELECTED_TENSORS=()
  declare -a SELECTED_QTYPES=()

  # Loop over each USER_REGEX entry, splitting into the regex and its qtype
  for entry in "${USER_REGEX[@]}"; do
    IFS='=' read -r pat_regex pat_qtype locked <<< "$entry"
    [[ -n "$locked" ]] && continue # Skip locked tensors
    echo "[$(timestamp)] Evaluating pattern: $pat_regex (qtype=$pat_qtype)"

    max_val=-1
    sel_idx=-1

    # Find the tensor with highest PPL/KLD matching this pattern
    for idx in "${!hdrs[@]}"; do
      name=${hdrs[$idx]}
      if [[ $name =~ $pat_regex ]]; then
        v=${vals[$idx]:-}
        [[ -z "$v" ]] && continue
        echo "[$(timestamp)]  Matched tensor '$name' with PPL/KLD value $v"
        if (( $(bc <<< "$v > $max_val") )); then
          max_val=$v
          sel_idx=$idx
        fi
      fi
    done

    if (( sel_idx >= 0 )); then
      selected_name=${hdrs[$sel_idx]}
      echo "[$(timestamp)] Selected worst tensor for pattern '$pat_regex': $selected_name (PPL/KLD $max_val)"
      SELECTED_TENSORS+=("$selected_name")
      SELECTED_QTYPES+=("$pat_qtype")
    else
      echo "[$(timestamp)] ⚠️ Warning: no tensors matching pattern '$pat_regex' found in CSV; skipping." >&2
    fi
  done

  # Rebuild USER_REGEX to exact tensor=qtype matches
  USER_REGEX=()
  for i in "${!SELECTED_TENSORS[@]}"; do
    tname=${SELECTED_TENSORS[$i]}
    tq=${SELECTED_QTYPES[$i]}
    echo "[$(timestamp)] Finalizing tensor: $tname with qtype: $tq"
    USER_REGEX+=("^${tname}=${tq}") # Don't add ending $, just in case we decide to use tq later after simple = split
  done

  echo "[$(timestamp)] Worst-tensor selection complete. Regex list now contains ${#USER_REGEX[@]} entries."
fi

# Ensure LOCAL_DOWNLOAD_DIR exists
mkdir -p "$LOCAL_DOWNLOAD_DIR"

# Verify LOCAL_MODEL_DIR exists
if [[ ! -d "$LOCAL_MODEL_DIR" ]]; then
    echo "Error: LOCAL_MODEL_DIR '$LOCAL_MODEL_DIR' does not exist or is not a directory." >&2
    exit 1
fi

# Pre-locate the main model shard file in LOCAL_MODEL_DIR
find_main_model_file() {
    # Finds one file matching MAIN_SHARD_PATTERN in LOCAL_MODEL_DIR
    local f
    f=$(find "$LOCAL_MODEL_DIR" -maxdepth 1 \( -type f -o -type l \) -name "$MAIN_SHARD_PATTERN" | head -n1 || true)
    if [[ -z "$f" ]]; then
        return 1
    else
        echo "$f"
        return 0
    fi
}

#echo $(find_main_model_file) | sed -E 's/-[0-9]{5}-of-[0-9]{5}\.gguf$//'
#echo $(find_main_model_file) | sed -nE 's/.*(-[0-9]{5}-of-[0-9]{5}\.gguf)$/\1/p'

# Pre-flight: restore any .gguf.bak back to .gguf before starting
echo "[$(timestamp)] Checking for .gguf.bak files to restore..."
shopt -s nullglob
for bak in "$LOCAL_MODEL_DIR"/*.gguf.bak; do
    # Derive the target .gguf filename
    orig="${bak%.bak}"
    echo "[$(timestamp)] Found backup: $(basename "$bak") -> restoring to $(basename "$orig")"
    # Overwrite any existing .gguf
    mv -f "$bak" "$orig"
done
shopt -u nullglob
echo "[$(timestamp)] Pre-flight restoration complete."

# Initial fetch and validation for each LOCAL_QTYPE
DEFAULT_BASE_FILENAME="" # Helper to obtain the base filename in case no model files present in the working directory
echo "[$(timestamp)] Starting initial validation for _LOCAL_QTYPES='${_LOCAL_QTYPES[*]}'"
declare -a tasks=()
for LOCAL_QTYPE in "${_LOCAL_QTYPES[@]}"; do
  __LOCAL_QTYPE=$LOCAL_QTYPE
  if [[ "$LOCAL_QTYPE" == "f32" ]]; then
    # when f32, replace by bf16
    LOCAL_QTYPE="bf16"
  fi
  echo "[$(timestamp)] Fetching initial tensors.map for LOCAL_QTYPE='$LOCAL_QTYPE'..."
  local_tensors_map="tensors.${LOCAL_QTYPE}.map"
  if run_tensor_downloader "${LOCAL_QTYPE^^}" "0" "." "${local_tensors_map}"; then
    echo "[$(timestamp)] Retrieved initial tensors map: $local_tensors_map"
    # Download the signature
    if [[ "$SKIP_GPG" != "true" ]]; then
      if ! run_tensor_downloader "${LOCAL_QTYPE^^}" -1 . "$local_tensors_map.sig"; then
          echo "[$(timestamp)] ❌ Error: failed to fetch map gpg signature for ${LOCAL_QTYPE^^}" >&2
          [ -n "$GNUPG_TMPDIR" ] && rm -rf "$GNUPG_TMPDIR"
          exit 2
      else
        if gpg --homedir "$GNUPG_TMPDIR" --no-default-keyring --verify "$local_tensors_map.sig" "$local_tensors_map" > /dev/null 2>&1; then
            echo "[$(timestamp)] GPG signature verification successful."
        else
            echo "[$(timestamp)] ❌ Error: GPG signature verification failed for '$local_tensors_map.sig'."
            [ -n "$GNUPG_TMPDIR" ] && rm -rf "$GNUPG_TMPDIR"
            exit 3
        fi
      fi
    fi
  elif [[ $__LOCAL_QTYPE == "f32" ]]; then
    echo "[$(timestamp)] ⚠️ Warning: Could not fetch BF16 tensors.map for LOCAL_QTYPE='$LOCAL_QTYPE'... will try to rely on other map files later." >&2; continue
  else
    echo "[$(timestamp)] ❌ Error: Could not fetch initial tensors.map for LOCAL_QTYPE='$LOCAL_QTYPE'." >&2; [ -n "$GNUPG_TMPDIR" ] && rm -rf "$GNUPG_TMPDIR"; exit 1
  fi
  while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    IFS=':' read -r fname expected_hash tensor_name _shape dtype _rest <<< "$line"
    dtype="${dtype#*=}"
    clean_dtype="${dtype%_r[0-9]}"
    for idx in "${!PATTERNS[@]}"; do
      if [[ "${PATTERN_QTYPES[$idx]}" == "$__LOCAL_QTYPE" ]] && [[ $tensor_name =~ ${PATTERNS[$idx]} ]]; then
        clean___LOCAL_QTYPE="${__LOCAL_QTYPE%_r[0-9]}"
        [[ "${clean_dtype,,}" != "${clean___LOCAL_QTYPE,,}" ]] && echo "[$(timestamp)] ❌ Error: '$local_tensors_map' cannot be used for benchmarking because not pure '$__LOCAL_QTYPE' - tensor '$tensor_name' (user-specified qtype: '$__LOCAL_QTYPE') does not match dtype='$dtype' from tensor map file. Please choose another base qtype." >&2 && exit 9
        tasks+=("$fname:$expected_hash:$__LOCAL_QTYPE")
        break
      fi
    done
  done < "$local_tensors_map"
  # Helper to attempt obtaining the actual name of the file matching BASELINE_QTYPE
  ideal_basename_found=false
  if [[ "${LOCAL_QTYPE^^}" == "${BASELINE_QTYPE^^}" ]]; then
    DEFAULT_BASE_FILENAME="$fname" # Use a filename that corresponds to the BASELINE_QTYPE
    ideal_basename_found=true
  elif [[ "$ideal_basename_found" == false ]] && [[ "${LOCAL_QTYPE^^}" == "BF16" ]]; then
    DEFAULT_BASE_FILENAME="$fname" # Revert to BF16 filename if possible
  elif [[ -z "$DEFAULT_BASE_FILENAME" ]]; then
    DEFAULT_BASE_FILENAME="$fname" # Revert to any filename
  fi
done
# Deduplicate tasks (just in case, but there shouldn't be any...)
mapfile -t tasks < <(printf "%s\n" "${tasks[@]}" | sort -u)

# Attempts to download a shard into LOCAL_DOWNLOAD_DIR and verify its sha256.
# Parameters:
#   $1 = fname
#   $2 = expected_hash
#   $3 = qtype
#   $4 = chunk_id
#   $5 = local_file   (final destination path; used for mv on success)
fetch_and_verify_shard() {
  local fname="$1"
  local expected_hash="$2"
  local qtype="$3"
  local chunk_id="$4"
  local local_file="$5"

  local tmp="${LOCAL_DOWNLOAD_DIR}/$fname"
  local -a candidates

  if [[ "$qtype" == "f32" ]]; then
    # when f32, first try bf16, then all local qtypes
    candidates=( bf16 "${LOCAL_QTYPES[@]}" )
  else
    # otherwise only re-try the original
    candidates=( "$qtype" )
  fi

  while true; do
    rm -f -- "$tmp"

    for try_q in "${candidates[@]}"; do
      echo "[$(timestamp)] Trying remote path with qtype=$try_q…" >&2

      # run_tensor_downloader expects an uppercased qtype in the original code
      if run_tensor_downloader "${try_q^^}" "$chunk_id" "${LOCAL_DOWNLOAD_DIR}" "${fname}"; then

        local new_hash
        # ensure _sha256sum exists (original process_line checked this earlier;
        # keeping a quick check here is defensive but optional)
        if ! command -v _sha256sum &>/dev/null; then
          echo "[$(timestamp)] _sha256sum missing after download; cannot verify $fname." >&2
          rm -f -- "$tmp"
          break  # try next qtype
        fi

        new_hash=$(_sha256sum "$tmp" | cut -d' ' -f1)
        if [[ "$new_hash" == "$expected_hash" ]]; then
          # move into place (atomic replace)
          mv -f -- "$tmp" "$local_file"
          echo "[$(timestamp)] Restored $fname with correct checksum via qtype=$try_q." >&2
          return 0
        else
          echo "[$(timestamp)] Post-fetch mismatch with qtype=$try_q ($new_hash ≠ $expected_hash)." >&2
          rm -f -- "$tmp"
        fi

      else
        echo "[$(timestamp)] Download failed for qtype=$try_q." >&2
      fi
    done

    echo "[$(timestamp)] All qtype attempts failed for $fname. Retrying in 10s…" >&2
    sleep 10
  done
}

# Define global lock file for recipe/download
RECIPE_LOCK_FILE=".recipe_download.lock"
# Remove old lock before declaring function
rm -f "$RECIPE_LOCK_FILE"

process_line() {
    local fname="$1"
    local expected_hash="$2"
    local qtype="$3"
    local main_model_file
    main_model_file="$(find_main_model_file 2>/dev/null || true)"

    # Only try to create lock if main model file is missing and lock does not exist
    if [[ -z "$main_model_file" ]]; then
        local created_lock=false

        # Try to create the lock file only if it does not exist
        if ( set -o noclobber; >"$RECIPE_LOCK_FILE" ) 2>/dev/null; then
            # This process successfully created the lock
            echo "[$(timestamp)] First shard 00001 missing, creating recipe and downloading shards…" >&2
            created_lock=true
        else
            # Lock already exists — this process must wait until it's removed
            while [[ -f "$RECIPE_LOCK_FILE" ]]; do
                sleep 0.1
            done
        fi

        # If this process created the lock, enter critical section
        if [[ "$created_lock" == true ]]; then
            if [[ ! -x "$QUANT_DOWNLOADER" ]]; then
                echo "Error: quant_downloader.sh not found or not executable at $QUANT_DOWNLOADER" >&2
                rm -f "$RECIPE_LOCK_FILE"
                exit 1
            fi

            local TMPDIR
            TMPDIR=$(mktemp -d)
            local OUTPUT_FILE="$TMPDIR/${BASELINE_QTYPE}.recipe"
            for entry in "${USER_REGEX[@]}"; do
                echo "${entry//=locked/}" >> "$OUTPUT_FILE"
            done

            if [[ "$SKIP_GPG" != "false" ]]; then
                _skip_gpg="--skip-gpg"
            else
                _skip_gpg=""
            fi

            # Run the downloader safely, capturing its exit status
            if ! run_quant_downloader "$OUTPUT_FILE" "${_skip_gpg}"; then
                echo "[$(timestamp)] Error: Failed to download model shards!" >&2
                rm -f "$RECIPE_LOCK_FILE"   # release lock
                exit 12
            fi

            # If we reach here, download succeeded
            echo "[$(timestamp)] Model shards successfully downloaded." >&2

            # Download complete — remove lock permanently
            rm -f "$RECIPE_LOCK_FILE"
        fi

        # After waiting or completing critical section, refresh main_model_file
        main_model_file="$(find_main_model_file)"
    fi

    # BASELINE_QTYPE / local file prefix
    local local_file_prefix
    local_file_prefix="$(echo "$main_model_file" | sed -E 's/-[0-9]{5}-of-[0-9]{5}\.gguf$//')"

    # Rest of process_line continues normally
    local local_file_suffix
    local_file_suffix="$(echo "$fname" | sed -nE 's/.*(-[0-9]{5}-of-[0-9]{5}\.gguf)$/\1/p')"
    local chunk_id
    chunk_id="$(echo "$local_file_suffix" | sed -nE 's/.*-([0-9]{5})-of-[0-9]{5}\.gguf$/\1/p')"
    local local_file="$local_file_prefix$local_file_suffix"

    if [[ -f "$local_file" ]]; then
        echo "[$(timestamp)] Checking $fname (qtype=$qtype)…" >&2

        if ! command -v _sha256sum &>/dev/null; then
            echo "[$(timestamp)] _sha256sum missing; skipping check for $fname." >&2
            return
        fi

        local actual_hash
        actual_hash=$(_sha256sum "$local_file" | cut -d' ' -f1)
        if [[ "$actual_hash" == "$expected_hash" ]]; then
            echo "[$(timestamp)] Hash OK for $fname." >&2
            return
        fi

        echo "[$(timestamp)] Hash mismatch ($actual_hash ≠ $expected_hash). Re-fetching $fname…" >&2
    else
        if [[ -z "$local_file_prefix" ]]; then
            echo "[$(timestamp)] Error: First shard 00001 still couldn't be found in the current working directory!" >&2
            exit 13
        else
            echo "[$(timestamp)] Shard $chunk_id is missing. Fetching $fname…" >&2
        fi
    fi

    # Externalized: attempt to fetch & verify
    fetch_and_verify_shard "$fname" "$expected_hash" "$qtype" "$chunk_id" "$local_file"
}

echo "[$(timestamp)] Validating ${#tasks[@]} shards with up to $N_THREADS threads…"

for entry in "${tasks[@]}"; do
  # split at the colon
  IFS=':' read -r fname expected_hash qtype <<< "$entry"

  {
    process_line "$fname" "$expected_hash" "$qtype"
  } 2>&1 &

  # throttle concurrency
  while (( $( (jobs -p 2>/dev/null | wc -l) || echo 0 ) >= N_THREADS )); do
    sleep 0.2
  done
done

wait
rm -f "$RECIPE_LOCK_FILE" # release lock file if still present

echo "[$(timestamp)] Initial validation complete."

# Check availability of required commands
if ! command -v _sha256sum &>/dev/null; then
    echo "Warning: _sha256sum not found; SHA256 verification will be skipped." >&2
    USE_SHA256=false
else
    USE_SHA256=true
fi
# if ! command -v gguf_info.py &>/dev/null; then
#     echo "Warning: gguf_info.py not found in PATH; this script does not need it directly here." >&2
#     # Not strictly required in this script
# fi

#
# --------- New helpers & caches for repeated sha256 calculations and group logic ----------
#

# Cache of computed file hashes keyed by file metadata (device:inode:size:mtime)
declare -A FILE_SHA_CACHE=()

# resolve_link <path> -> prints resolved path (following symlinks). On failure exits the script.
# Tries a few portable methods (realpath, readlink -f, python fallback). If none succeed, the script
# will immediately exit as requested.
resolve_link() {
  local file="$1"
  local resolved=""

  # Prefer realpath if available
  if command -v realpath >/dev/null 2>&1; then
    # use -e if available (GNU realpath supports it), otherwise fall back to plain realpath
    if realpath -e "$file" >/dev/null 2>&1; then
      resolved=$(realpath -e "$file" 2>/dev/null || true)
    else
      resolved=$(realpath "$file" 2>/dev/null || true)
    fi
  fi

  # Try readlink -f if realpath didn't work
  if [[ -z "$resolved" ]] && command -v readlink >/dev/null 2>&1; then
    if readlink -f "$file" >/dev/null 2>&1; then
      resolved=$(readlink -f "$file" 2>/dev/null || true)
    fi
  fi

  # Python fallback (should be available on most systems)
  if [[ -z "$resolved" ]]; then
    if command -v python3 >/dev/null 2>&1; then
      resolved=$(python3 -c "import os,sys;print(os.path.realpath(sys.argv[1]))" "$file" 2>/dev/null || true)
    elif command -v python >/dev/null 2>&1; then
      resolved=$(python -c "import os,sys;print(os.path.realpath(sys.argv[1]))" "$file" 2>/dev/null || true)
    fi
  fi

  # If still empty or non-existent -> error and immediate exit (per request)
  if [[ -z "$resolved" || ! -e "$resolved" ]]; then
    echo "[$(timestamp)] ❌ Error: Failed to resolve symlink/realpath for '$file' or resolved target does not exist." >&2
    exit 22
  fi

  printf '%s' "$resolved"
}

# portable function to get a stable file metadata key
# Now resolves symlinks first (so we operate on the resolved/target file)
get_file_meta_key() {
  local file="$1"

  # Resolve symlink -> we must succeed or exit (resolve_link handles exit on failure)
  local resolved
  resolved=$(resolve_link "$file") || { echo "[$(timestamp)] ❌ Error: resolve_link failed for '$file'." >&2; exit 22; }

  # Use stat on the resolved target. This avoids ambiguity about symlink vs target inode.
  # For GNU stat use -c, for BSD/macOS use -f. Both operate on the resolved path here.
  if stat --version >/dev/null 2>&1; then
    stat -L -c "%d:%i:%s:%Y" "$resolved" 2>/dev/null || echo ""
  else
    stat -L -f "%d:%i:%z:%m" "$resolved" 2>/dev/null || echo ""
  fi
}

# Return cached sha256 if present, otherwise compute and cache it.
cached_sha256() {
  local file="$1"
  # If file doesn't exist (after resolution) return empty (caller handles)
  if [[ ! -e "$file" ]]; then
    echo ""
    return 0
  fi

  # Resolve symlink to ensure we cache/hash the real target
  local resolved
  resolved=$(resolve_link "$file") || { echo "[$(timestamp)] ❌ Error: resolve_link failed for '$file'." >&2; exit 22; }

  local key
  key=$(get_file_meta_key "$resolved")
  if [[ -z "$key" ]]; then
    # fallback: use resolved path as key (less reliable but only used if stat failed)
    key="$resolved"
  fi

  if [[ -n "${FILE_SHA_CACHE[$key]:-}" ]]; then
    printf '%s' "${FILE_SHA_CACHE[$key]}"
    return 0
  fi

  if [[ "$USE_SHA256" == "true" ]]; then
    local h
    h=$(_sha256sum "$resolved" | cut -d' ' -f1)
    FILE_SHA_CACHE[$key]="$h"
    printf '%s' "$h"
    return 0
  else
    printf ''
    return 0
  fi
}

# Compute a stable, order-independent hash for a group's sorted member list
# uses: cksum if available (produces "<checksum> <bytes>"), otherwise falls back to _sha256sum
# NOTE: this implementation avoids creating a temporary file and does the work in-memory via pipes.
compute_group_hash() {
  # $1 = name of array var (passed by name)
  local -n _arr=$1

  # If no members, produce an empty hash deterministically
  if (( ${#_arr[@]} == 0 )); then
    # empty input -> return zero-length sha256 of empty string (or cksum equivalent)
    if command -v cksum >/dev/null 2>&1; then
      # cksum on empty input: "0 0"
      printf '0-0'
    else
      # sha256 of empty string (hex)
      printf '%s' "$(_sha256sum <(printf '') 2>/dev/null || echo 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855')"
    fi
    return 0
  fi

  # Build a sorted stream in-memory and pipe to cksum or sha256
  if command -v cksum >/dev/null 2>&1; then
    # read checksum and bytes from cksum, then print "<checksum>-<bytes>"
    # Use a subshell to ensure the printf + sort are grouped
    local cs bytes
    read cs bytes < <( ( printf '%s\n' "${_arr[@]}" | sort ) | cksum )
    printf '%s-%s' "$cs" "$bytes"
    return 0
  else
    # fallback to sha256 of the sorted stream; ensure we use cut as requested
    ( printf '%s\n' "${_arr[@]}" | sort ) | _sha256sum | cut -d' ' -f1
    return 0
  fi
}

# collect_members_from_regexes <regex_array_name> <out_array_name>
# Generic helper: given an array of regexes, populate out array with all matching tensors
collect_members_from_regexes() {
  local regex_arr_name="$1"
  local out_arr_name="$2"
  local -n _regexes="$regex_arr_name"
  local -n _out_arr="$out_arr_name"    # <- different local name
  _out_arr=()
  for reg in "${_regexes[@]}"; do
    reg="$(sed -E 's/^[[:space:]]+|[[:space:]]+$//g' <<<"$reg")"
    [[ -z "$reg" ]] && continue
    for t in "${!tensor_to_shard[@]}"; do
      if [[ $t =~ $reg ]]; then
        if [[ ! " ${_out_arr[*]} " =~ " $t " ]]; then
          _out_arr+=("$t")
        fi
      fi
    done
  done
}

# collect_group_members <group_idx> <out_array_name>
# populates the out array with all tensors present in current tensor_to_shard that match any regex in the group
collect_group_members() {
  local gidx="$1"
  local -n out_arr=$2
  out_arr=()
  local group_raw="${GROUP_TENSORS_RAW[$gidx]}"
  IFS=',' read -r -a regs <<< "$group_raw"
  # delegate to generic regex-members collector
  collect_members_from_regexes regs out_arr
}

# collect_members_for_user_regex <out_array_name>
# collects members that match any of the USER_REGEX_PATTERNS into out array
collect_members_for_user_regex() {
  local -n out_arr=$1
  collect_members_from_regexes USER_REGEX_PATTERNS out_arr
}

# ---------------- end helper section ---------------------------------------

# ----------------------------------------------------------------
# Seed FILE_SHA_CACHE from initial validation results (tasks array)
# Optimized: run one find to gather local shards, map their "-NNNNN-of-NNNNN.gguf"
# suffix to the local path, then iterate tasks and lookup suffix in the map.
# This avoids a find() for every task.
#
# Each task entry is "fname:expected_hash:qtype".
# We store expected_hash under the same meta-key produced by get_file_meta_key()
# so cached_sha256() will return it later without recomputing.
# ----------------------------------------------------------------
if (( ${#tasks[@]} > 0 )); then
  echo "[$(timestamp)] Seeding FILE_SHA_CACHE from initial tasks (${#tasks[@]} entries) using single-find optimization..."

  # Build suffix -> local file map by running one find across LOCAL_MODEL_DIR.
  # Only files/links at top-level are considered (maxdepth 1), excluding files in LOCAL_DOWNLOAD_DIR.
  declare -A SUFFIX_TO_LOCAL=()
  # Collect candidate files once
  mapfile -t _cand_files < <(find "$LOCAL_MODEL_DIR" -maxdepth 1 \( -type f -o -type l \) -print 2>/dev/null || true)

  if (( ${#_cand_files[@]} == 0 )); then
    echo "[$(timestamp)] ⚠️ Warning: No candidate local model files found in $LOCAL_MODEL_DIR while seeding FILE_SHA_CACHE." >&2
  else
    for cf in "${_cand_files[@]}"; do
      # Skip anything that lives under LOCAL_DOWNLOAD_DIR path (defensive)
      case "$cf" in
        *"$LOCAL_DOWNLOAD_DIR"*) continue ;;
      esac
      # Extract suffix "-NNNNN-of-NNNNN.gguf" if present
      if [[ $cf =~ -([0-9]{5}-of-[0-9]{5})\.gguf$ ]]; then
        suffix="${BASH_REMATCH[1]}.gguf"
        # only add first occurrence (prefer earlier); don't overwrite an existing mapping
        if [[ -z "${SUFFIX_TO_LOCAL[$suffix]:-}" ]]; then
          SUFFIX_TO_LOCAL[$suffix]="$cf"
        fi
      fi
    done
  fi

  # Now iterate tasks and seed cache by lookup instead of running find each time
  for entry in "${tasks[@]}"; do
    IFS=':' read -r fname expected_hash qtype <<< "$entry"
    # sanity check: expected_hash must be present (it should be)
    if [[ -z "$expected_hash" ]]; then
      echo "[$(timestamp)] ❌ Error: task entry '$entry' has empty expected_hash; aborting seed." >&2
      [ -n "$GNUPG_TMPDIR" ] && rm -rf "$GNUPG_TMPDIR"
      exit 20
    fi

    # extract suffix from fname
    if [[ $fname =~ -([0-9]{5}-of-[0-9]{5})\.gguf$ ]]; then
      suffix="${BASH_REMATCH[1]}.gguf"
    else
      echo "[$(timestamp)] ⚠️ Warning: cannot extract suffix from task fname='$fname'; skipping." >&2
      continue
    fi

    local_file="${SUFFIX_TO_LOCAL[$suffix]:-}"
    if [[ -z "$local_file" ]]; then
      # No local file found for this suffix — that's fine, initial validation may have only used downloader tmp dir.
      continue
    fi

    # Ensure the file actually exists (it should), then resolve symlink and compute meta-key
    if [[ ! -e "$local_file" ]]; then
      echo "[$(timestamp)] ⚠️ Warning: mapped local file '$local_file' for suffix '$suffix' does not exist; skipping."
      continue
    fi

    # Resolve symlink -> must succeed or exit (resolve_link handles exiting on failure)
    resolved=$(resolve_link "$local_file") || { echo "[$(timestamp)] ❌ Error: resolve_link failed for '$local_file' while seeding FILE_SHA_CACHE." >&2; exit 22; }

    # Obtain canonical metadata key used by cached_sha256()
    meta_key=$(get_file_meta_key "$resolved")
    if [[ -z "$meta_key" ]]; then
      # fallback to resolved path as a key (rare)
      meta_key="$resolved"
    fi

    # Store expected hash in cache under that key
    FILE_SHA_CACHE["$meta_key"]="$expected_hash"
    echo "[$(timestamp)] Cached expected hash for local shard '$local_file' (suffix='$suffix', key='$meta_key')."
  done
else
  echo "[$(timestamp)] No initial tasks to seed FILE_SHA_CACHE from."
fi
# ----------------------------------------------------------------

# Baseline benchmark (PPL+KLD baseline) — only run if PPL+KLD is enabled (mode == 0 or 2)
# We'll set a flag when baseline exists/done and later inject a PROCESSED_GROUP_COMBOS entry
BASELINE_BENCHMARK_DONE=0
if [[ "$BENCH_MODE" -eq 0 || "$BENCH_MODE" -eq 2 ]]; then
  baseline_result_file="bench_ppl${_kld}_result.baseline.${BASELINE_QTYPE}.${PPL_COMMAND_CHUNKS_TO_PROCESS}.txt"
  baseline_kld_file="bench_kld_result.baseline.${BASELINE_QTYPE}.${PPL_COMMAND_CHUNKS_TO_PROCESS}.bin"
  if [[ ! -f "$baseline_result_file" || ( "$NO_KLD" == "false" && ! -f "$baseline_kld_file" ) ]]; then
      [[ ! -f "$baseline_result_file" ]] && echo "[$(timestamp)] PPL baseline file not yet computed: $baseline_result_file (not found)"
      [[ "$NO_KLD" == "false" && ! -f "$baseline_kld_file" ]] && echo "[$(timestamp)] KLD baseline file not yet computed: $baseline_kld_file (not found)"
      echo "[$(timestamp)] Running baseline PPL$PLUS_KLD benchmark for BASELINE_QTYPE='$BASELINE_QTYPE', chunks=$PPL_COMMAND_CHUNKS_TO_PROCESS."
      main_model_file=$(find_main_model_file) || { echo "Error: main model file not found for baseline." >&2; [ -n "$GNUPG_TMPDIR" ] && rm -rf "$GNUPG_TMPDIR"; exit 1; }
      baseline_cmd="${PPL_COMMAND_TEMPLATE//\{MODEL_FILE\}/$main_model_file}" # Replace model placeholder by actual model file path
      [[ "$NO_KLD" == "false" ]] && baseline_cmd="${baseline_cmd//\{KLD_PARAMETER\}/--kl-divergence-base $baseline_kld_file}" # Replace kld placeholder by actual kld parameter
      eval "$baseline_cmd" > "$baseline_result_file" 2>&1 < /dev/null
      estimate=$(grep "Final estimate" "$baseline_result_file" || true)
      echo "Baseline PPL$PLUS_KLD benchmark completed for chunks=$PPL_COMMAND_CHUNKS_TO_PROCESS: $estimate"
  else
      estimate=$(grep "Final estimate" "$baseline_result_file" || true)
      echo "[$(timestamp)] Baseline PPL$PLUS_KLD benchmark already exists for BASELINE_QTYPE='$BASELINE_QTYPE', chunks=$PPL_COMMAND_CHUNKS_TO_PROCESS: $estimate"
  fi
  BASELINE_BENCHMARK_DONE=1
else
  echo "[$(timestamp)] Skipping baseline PPL$PLUS_KLD benchmark because --mode=${BENCH_MODE} (PPL$PLUS_KLD disabled)."
fi

# Baseline for SWEEP if sweep is enabled
if [[ "$BENCH_MODE" -eq 1 || "$BENCH_MODE" -eq 2 ]]; then
  sweep_baseline_result_file="bench_sweep_result.baseline.${BASELINE_QTYPE}.${BENCH_COMMAND_CONTEXT_TO_PROCESS}.txt"
  if [[ ! -f "$sweep_baseline_result_file" ]]; then
      echo "[$(timestamp)] Running baseline SWEEP benchmark for BASELINE_QTYPE='$BASELINE_QTYPE', context=$BENCH_COMMAND_CONTEXT_TO_PROCESS."
      main_model_file=$(find_main_model_file) || { echo "Error: main model file not found for sweep baseline." >&2; [ -n "$GNUPG_TMPDIR" ] && rm -rf "$GNUPG_TMPDIR"; exit 1; }
      sweep_baseline_cmd="${SWEEP_COMMAND_TEMPLATE//\{MODEL_FILE\}/$main_model_file}"
      # Ensure ${BENCH_COMMAND_CONTEXT_TO_PROCESS} expands in the command by using eval
      eval "$sweep_baseline_cmd" > "$sweep_baseline_result_file" 2>&1 < /dev/null
      echo "Baseline SWEEP benchmark completed for context=$BENCH_COMMAND_CONTEXT_TO_PROCESS."
  else
      echo "Baseline SWEEP benchmark already exists for BASELINE_QTYPE='$BASELINE_QTYPE', context=$BENCH_COMMAND_CONTEXT_TO_PROCESS."
  fi
fi

# Startup logging
echo "[$(timestamp)] Starting benchmark_each_tensor loop."
echo "Local download dir: $LOCAL_DOWNLOAD_DIR"
echo "Local model dir: $LOCAL_MODEL_DIR"
echo "QTypes: ${QTYPES[*]}"
echo "Tensor regex patterns:"
for pat in "${USER_REGEX[@]}"; do
    echo "  - $pat"
done
echo "PPL$PLUS_KLD command template: $PPL_COMMAND_TEMPLATE"
echo "SWEEP command template: $SWEEP_COMMAND_TEMPLATE"
echo "Main shard pattern: $MAIN_SHARD_PATTERN"
echo "Use SHA256 verification: $USE_SHA256"
echo "Benchmark mode: $BENCH_MODE (0=PPL+KLD,1=SWEEP,2=PPL+KLD+SWEEP)"
if [[ "$NO_KLD" == "true" ]]; then
  echo "KLD benchmarking: DISABLED"
else
  echo "KLD benchmarking: ENABLED"
fi
echo "Sweep context (-c): $BENCH_COMMAND_CONTEXT_TO_PROCESS"
ALL_GROUP_IDS=()
if [[ "$GROUP_TENSORS_DISABLED" == "true" ]]; then
  echo "Group tensors: DISABLED"
else
  echo "Group tensors: ENABLED; groups:"
  group_mapping_file="bench_ppl${_kld}_group_mapping.${BASELINE_QTYPE}.${PPL_COMMAND_CHUNKS_TO_PROCESS}.txt"

  # Build the planned mapping content in a temp file (also print the human-friendly lines)
  tmp_mapping="$(mktemp)" || { echo "[$(timestamp)] ❌ Error: mktemp failed."; exit 1; }
  gid=0
  for g in "${GROUP_TENSORS_RAW[@]}"; do
    echo "  - group$gid: $g"
    echo "group$gid:$g" >> "$tmp_mapping"
    ALL_GROUP_IDS+=("$gid")
    gid=$((gid + 1))
  done
  if [[ "$BENCH_GROUPS_ONLY" == "true" ]] && (( ${#ALL_GROUP_IDS[@]} == 0 )); then
    echo "[$(timestamp)] ❌ Error: --benchmark-groups-only is set but there are no groups set!" >&2
  fi

  if [ -e "$group_mapping_file" ]; then
      # If the existing file is byte-equal to the planned content, skip asking and do nothing
      if cmp -s "$tmp_mapping" "$group_mapping_file"; then
          echo "[$(timestamp)] Group mapping file '$group_mapping_file' is identical to planned content; no changes necessary."
          rm -f "$tmp_mapping"
      else
          # different content -> ask user
          read -p "[$(timestamp)] ❓ Question: Group mapping file '$group_mapping_file' already exists and differs. Overwrite? [y/N] " answer
          case "$answer" in
              [Yy]*)
                # overwrite atomically
                mv "$tmp_mapping" "$group_mapping_file"
                echo "[$(timestamp)] Overwrote '$group_mapping_file' with new group mapping."
                ;;
              *)
                echo "Operation cancelled by user. No changes made to '$group_mapping_file'." >&2
                rm -f "$tmp_mapping"
                exit 14
                ;;
          esac
      fi
  else
      # file doesn't exist -> create it from temp
      mv "$tmp_mapping" "$group_mapping_file"
  fi
  
  if [[ "$BENCH_GROUPS_ONLY" == "true" ]]; then
    echo "Benchmark groups only: ENABLED; In this mode, even if all tensors of a group have been benchmarked via another group, the group will still be benchmarked!"
  else
    echo "Benchmark groups only: DISABLED; In this mode, if all tensors of a group have already been benchmarked (either individually or via another group) then the group will not be benchmarked!"
  fi
fi

# helper checks for enabled runs
need_run_ppl() {
  # PPL+KLD if mode==0 or mode==2
  if [[ "$BENCH_MODE" -eq 0 || "$BENCH_MODE" -eq 2 ]]; then return 0; else return 1; fi
}
need_run_sweep() {
  # SWEEP if mode==1 or mode==2
  if [[ "$BENCH_MODE" -eq 1 || "$BENCH_MODE" -eq 2 ]]; then return 0; else return 1; fi
}

shuffle_shards_by_tensor_patterns() {
  local assoc_name=$1
  local out_name=$2

  # nameref only for the input map
  local -n _shard_map=$assoc_name

  # 1) collect & shuffle all shard keys
  local shard_keys=("${!_shard_map[@]}")
  local all_shards
  mapfile -t all_shards < <(printf '%s\n' "${shard_keys[@]}" | shuf)

  # 2) split into non-numeric vs all-numeric shards
  local non_num=() all_num=()
  for shard in "${all_shards[@]}"; do
    IFS=' ' read -r -a tensors <<< "${_shard_map[$shard]}"
    local has_non=0
    for t in "${tensors[@]}"; do
      [[ ! $t =~ [0-9] ]] && { has_non=1; break; }
    done
    (( has_non )) && non_num+=("$shard") || all_num+=("$shard")
  done

  # 3) bucket the all-numeric shards by pattern
  declare -A pat_to_shards
  for shard in "${all_num[@]}"; do
    IFS=' ' read -r -a tensors <<< "${_shard_map[$shard]}"
    for t in "${tensors[@]}"; do
      local pat="^$(sed -E 's/[0-9]+/\\\.[0-9]+\\\./g' <<<"$t")\$"
      local prev="${pat_to_shards[$pat]:-}"
      case " $prev " in
        *" $shard "*) ;;
        *) pat_to_shards[$pat]="$prev $shard" ;;
      esac
    done
  done

  # 4) group patterns by their bucket size
  declare -A size_to_shards
  for pat in "${!pat_to_shards[@]}"; do
    # count how many shards in this pattern
    local bucket="${pat_to_shards[$pat]}"
    local cnt=$(wc -w <<<"$bucket")
    # collect all shards under this size
    size_to_shards[$cnt]="${size_to_shards[$cnt]:-} $bucket"
  done

  # 5) sort sizes ascending
  mapfile -t sizes_sorted < <(
    for size in "${!size_to_shards[@]}"; do
      printf '%s\n' "$size"
    done | sort -n
  )

  # 6) for each size, dedupe, shuffle all shards in that size-group, and collect
  local ordered_num=()
  for size in "${sizes_sorted[@]}"; do
    # split into array and dedupe
    read -r -a all_bucket <<< "${size_to_shards[$size]}"
    # use associative array to unique
    declare -A seen=()
    local unique_bucket=()
    for s in "${all_bucket[@]}"; do
      [[ -z "${seen[$s]:-}" ]] && { seen[$s]=1; unique_bucket+=("$s"); }
    done
    # shuffle the entire size-group at once
    mapfile -t shuffled_bucket < <(printf '%s\n' "${unique_bucket[@]}" | shuf)
    ordered_num+=("${shuffled_bucket[@]}")
  done

  # 7) build result: non-numeric first, then these ordered numeric shards
  local __result=()
  (( ${#non_num[@]} )) && __result+=("${non_num[@]}")
  (( ${#ordered_num[@]} )) && __result+=("${ordered_num[@]}")

  # write back to caller’s array
  eval "${out_name}=(\"\${__result[@]}\")"
}

# shuffle_tensors_by_pattern:
#   $1 = name of input array containing tensor names
#   $2 = name of output array to populate with shuffled tensor list
shuffle_tensors_by_pattern() {
  local in_name=$1
  local out_name=$2

  # nameref for input tensor list
  local -n _tensors=$in_name

  # 1) split into non-numeric vs numeric
  local non_num=() numeric=()
  for t in "${_tensors[@]}"; do
    if [[ $t =~ [0-9] ]]; then
      numeric+=("$t")
    else
      non_num+=("$t")
    fi
  done

  # 2) shuffle non-numeric group
  local shuffled_non_num=()
  if (( ${#non_num[@]} )); then
    mapfile -t shuffled_non_num < <(printf '%s\n' "${non_num[@]}" | shuf)
  fi

  # 3) bucket numeric tensors by pattern (replace digit-runs with \.[0-9]+\.)
  declare -A pat_to_tensors
  for t in "${numeric[@]}"; do
    local pat="^$(sed -E 's/[0-9]+/\\\.[0-9]+\\\./g' <<<"$t")\$"
    local prev="${pat_to_tensors[$pat]:-}"
    case " $prev " in
      *" $t "*) ;;
      *) pat_to_tensors[$pat]="$prev $t" ;;
    esac
  done

  # 4) group patterns by bucket size
  declare -A size_to_tensors
  for pat in "${!pat_to_tensors[@]}"; do
    local bucket="${pat_to_tensors[$pat]}"
    local cnt
    cnt=$(wc -w <<<"$bucket")
    local prev="${size_to_tensors[$cnt]:-}"
    size_to_tensors[$cnt]="$prev $bucket"
  done

  # 5) sort sizes ascending
  mapfile -t sizes_sorted < <(
    for size in "${!size_to_tensors[@]}"; do
      printf '%s\n' "$size"
    done | sort -n
  )

  # 6) for each size, shuffle all tensors in that group and collect
  local ordered_num=()
  for size in "${sizes_sorted[@]}"; do
    read -r -a bucket_all <<< "${size_to_tensors[$size]}"
    mapfile -t bucket_shuf < <(printf '%s\n' "${bucket_all[@]}" | shuf)
    for t in "${bucket_shuf[@]}"; do
      ordered_num+=("$t")
    done
  done

  # 7) combine non-numeric first, then ordered numeric
  local __result=()
  (( ${#shuffled_non_num[@]} )) && __result+=("${shuffled_non_num[@]}")
  (( ${#ordered_num[@]}    )) && __result+=("${ordered_num[@]}")

  # write back to caller’s array
  eval "${out_name}=(\"\${__result[@]}\")"
}

# find_group_indexes_for_tensor <tensor> -> prints zero-or-more group indices (one per line)
find_group_indexes_for_tensor() {
  local tensor="$1"
  # remaining args (if any) are treated as forbidden/processed group IDs
  shift || true
  local -a PROCESSED_GROUP_IDS=()
  if (( $# )); then
    PROCESSED_GROUP_IDS=("$@")
  fi

  if [[ "$GROUP_TENSORS_DISABLED" == "true" ]]; then
    # print nothing -> caller receives an empty array
    return
  fi

  local -a idxs=()
  for idx in "${!GROUP_TENSORS_RAW[@]}"; do
    local group_raw="${GROUP_TENSORS_RAW[$idx]}"

    # skip groups that are in the PROCESSED_GROUP_IDS list
    local skip=false
    for pid in "${PROCESSED_GROUP_IDS[@]}"; do
      [[ "$pid" == "$idx" ]] && { skip=true; break; }
    done
    $skip && continue

    IFS=',' read -r -a regs <<< "$group_raw"
    for reg in "${regs[@]}"; do
      # trim spaces
      reg="$(sed -E 's/^[[:space:]]+|[[:space:]]+$//g' <<<"$reg")"
      [[ -z "$reg" ]] && continue
      if [[ $tensor =~ $reg ]]; then
        idxs+=("$idx")
        # one matching regex per group is enough -> move to next group
        break
      fi
    done
  done

  # print them newline-separated (or nothing if empty)
  for i in "${idxs[@]}"; do
    printf '%s\n' "$i"
  done
}

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

# Instead of a raw `while true; do ... done`, wrap the main body in a function and control
# whether it runs infinitely or just once based on the INFINITE_LOOP flag.
run_main_loop() {
    for qtype in "${QTYPES[@]}"; do
        # Uppercase version for remote directory name
        qtype_up="${qtype^^}"
        local_tensors_map="tensors.${qtype}.map"

        echo "[$(timestamp)] Processing qtype='$qtype'."

        # Track unprocessed group ids for this qtype
        local QTYPE_REMAINING_GROUP_IDS=("${ALL_GROUP_IDS[@]}")

        # Store benchmark counts
        local count_group_ppl_bench_success=0
        local count_group_ppl_bench_failure=0
        local count_group_sweep_bench_success=0
        local count_group_sweep_bench_failure=0
        local count_individual_ppl_bench_success=0
        local count_individual_ppl_bench_failure=0
        local count_individual_sweep_bench_success=0
        local count_individual_sweep_bench_failure=0

        # Attempt to download tensors.map from remote.
        # We first remove any old local copy to ensure we detect absence properly
        rm -f "$local_tensors_map"
        rm -f "$local_tensors_map.sig"
        echo "[$(timestamp)] Fetching remote tensors.map..."
        if run_tensor_downloader "${qtype^^}" "0" "." "${local_tensors_map}"; then
            echo "[$(timestamp)] Retrieved tensors.map to $local_tensors_map"
            # Download the signature
            if [[ "$SKIP_GPG" != "true" ]]; then
              if ! run_tensor_downloader "${qtype^^}" -1 . "$local_tensors_map.sig"; then
                  echo "[$(timestamp)] ❌ Error: failed to fetch map gpg signature for ${qtype^^}" >&2
                  [ -n "$GNUPG_TMPDIR" ] && rm -rf "$GNUPG_TMPDIR"
                  exit 2
              else
                if gpg --homedir "$GNUPG_TMPDIR" --no-default-keyring --verify "$local_tensors_map.sig" "$local_tensors_map" > /dev/null 2>&1; then
                    echo "[$(timestamp)] GPG signature verification successful."
                else
                    echo "[$(timestamp)] ❌ Error: GPG signature verification failed for '$local_tensors_map.sig'."
                    [ -n "$GNUPG_TMPDIR" ] && rm -rf "$GNUPG_TMPDIR"
                    exit 3
                fi
              fi
            fi
        else
            echo "[$(timestamp)] ⚠️ Warning: Could not fetch tensors.map for qtype='$qtype'. Skipping this qtype." >&2
            rm -f "$local_tensors_map"
            continue
        fi

        # Parse tensors.${qtype}.map: lines format assumed: filename.gguf:sha256hash:tensor_name:shape=...:dtype=...:elements=...:bytes=...
        # For each line, if tensor_name matches any USER_REGEX, then schedule that filename for benchmarking.
        declare -A shard_to_tensors=()  # map shard filename -> space-separated list of tensor_names
        declare -A shard_to_hash=()     # map shard filename -> expected hash from map
        declare -A tensor_to_shard=()   # map tensor_name -> shard filename
        while IFS= read -r line; do
            # Skip empty lines
            [[ -z "$line" ]] && continue
            # Split line by colon: filename:hash:tensor_name:...
            IFS=':' read -r fname file_hash tensor_name _shape dtype _rest <<< "$line"
            dtype="${dtype#*=}"
            clean_dtype="${dtype%_r[0-9]}"
            # If tensor_name matches any USER_REGEX, record it under shard fname
            for entry in "${USER_REGEX[@]}"; do
              # split on “=”: LHS is the actual regex, RHS is the qtype (which we ignore here)
              IFS='=' read -r pat _qtype locked <<< "$entry"
              [[ -n "$locked" ]] && continue # Skip locked tensors
              if [[ $tensor_name =~ $pat ]]; then
                clean_qtype="${qtype%_r[0-9]}"
                [[ "${clean_dtype,,}" != "${clean_qtype,,}" ]] && echo "[$(timestamp)] ⚠️ Warning: '$local_tensors_map' cannot be used for benchmarking because not pure '$qtype' - tensor '$tensor_name' (user-specified qtype: '$qtype') does not match dtype='$dtype' from tensor map file. Please choose another target qtype or exclude this tensor. Skipping this qtype." >&2 && break
                # Append tensor_name to shard_to_tensors["$fname"], avoiding duplicates
                if [[ -z "${shard_to_tensors[$fname]:-}" ]]; then
                  shard_to_tensors["$fname"]="$tensor_name"
                else
                  # Check if already present
                  existing="${shard_to_tensors[$fname]}"
                  if [[ ! " $existing " =~ " $tensor_name " ]]; then
                    shard_to_tensors["$fname"]="${existing} $tensor_name"
                  fi
                fi
                # store shard hash and tensor->shard mapping for later group processing
                shard_to_hash["$fname"]="$file_hash"
                tensor_to_shard["$tensor_name"]="$fname"
                break
              fi
            done
        done < "$local_tensors_map"

        if [[ ${#shard_to_tensors[@]} -eq 0 ]]; then
            echo "[$(timestamp)] No matching tensors for qtype='$qtype' in $local_tensors_map. Skipping."
            continue
        fi

        echo "[$(timestamp)] Found ${#shard_to_tensors[@]} shard(s) with matching tensors for qtype='$qtype'."

        # Find main model file once
        main_model_file=$(find_main_model_file) || {
            echo "[$(timestamp)] ❌ Error: Could not find main model shard matching '$MAIN_SHARD_PATTERN' in $LOCAL_MODEL_DIR. Skipping benchmarking." >&2
            continue
        }
        echo "[$(timestamp)] Main model shard for PPL+KLD: $main_model_file"

        # Organise shards so that the ones with tensors that have less layers come first because these tensors cannot be interpolated easily, so it's best to process them first
        shuffle_shards_by_tensor_patterns shard_to_tensors shuffled_shard_keys

        # track individually processed tensors
        declare -A PROCESSED_TENSOR=()
        # track processed group combos by a concise hash (cksum preferred)
        declare -A PROCESSED_GROUP_COMBOS=()

        # If baseline benchmark was done earlier, compute the group combo corresponding to USER_REGEX
        # using the same compute_group_hash logic (i.e., collect actual member tensor names from tensor_to_shard).
        if [[ "${qtype^^}" == "${BASELINE_QTYPE^^}" ]] && [[ "$BASELINE_BENCHMARK_DONE" -eq 1 ]]; then
          tmp_members_group=()
          collect_members_for_user_regex tmp_members_group
          if (( ${#tmp_members_group[@]} > 0 )); then
            group_hash=$(compute_group_hash tmp_members_group)
            PROCESSED_GROUP_COMBOS["$group_hash"]=1
            echo "[$(timestamp)] Injected baseline-based group combo into PROCESSED_GROUP_COMBOS for qtype='$qtype' => ${group_hash}"
            # optionally mark member tensors as processed (unless BENCH_GROUPS_ONLY)
            if [[ "$BENCH_GROUPS_ONLY" != "true" ]]; then
              for t in "${tmp_members_group[@]}"; do PROCESSED_TENSOR["$t"]=1; done
            fi
          else
            echo "[$(timestamp)] ⚠️ Warning: baseline requested to inject group combo but no members were found for USER_REGEX in this qtype map."
          fi
        fi

        if [[ "$GROUP_TENSORS_DISABLED" != "true" ]]; then
          for gidx in "${!GROUP_TENSORS_RAW[@]}"; do
            # collect group members that exist in this tensor_to_shard set
            collect_group_members "$gidx" tmp_members_group
            if (( ${#tmp_members_group[@]} )); then
              # check files for both PPL and SWEEP as required by BENCH_MODE
              ppl_group_file="bench_ppl${_kld}_result.group${gidx}.${qtype}.${PPL_COMMAND_CHUNKS_TO_PROCESS}.txt"
              sweep_group_file="bench_sweep_result.group${gidx}.${qtype}.${BENCH_COMMAND_CONTEXT_TO_PROCESS}.txt"

              ppl_done=false; sweep_done=false
              if need_run_ppl; then [[ -f "$ppl_group_file" ]] && ppl_done=true || ppl_done=false; else ppl_done=true; fi
              if need_run_sweep; then [[ -f "$sweep_group_file" ]] && sweep_done=true || sweep_done=false; else sweep_done=true; fi

              if [[ "$ppl_done" == "true" && "$sweep_done" == "true" ]]; then
                # Only flag individual tensors as already processed when not in --benchmark-groups-only mode
                [[ "$BENCH_GROUPS_ONLY" != "true" ]] && for mt in "${tmp_members_group[@]}"; do PROCESSED_TENSOR["$mt"]=1; done
                # record combo hash so identical exact combos won't be re-run
                group_hash=$(compute_group_hash tmp_members_group)
                PROCESSED_GROUP_COMBOS["${group_hash}"]=1
                echo "[$(timestamp)] Found existing group result(s): group${gidx} -> marking ${#tmp_members_group[@]} member(s) as processed and recording combo (${group_hash})."
              fi
            fi
          done
        fi

        # mark individual per-tensor results too (only if not already marked via a group)
        if [[ "$BENCH_GROUPS_ONLY" == "true" ]]; then
          echo "[$(timestamp)] Skipping all individual tensor benchmarking because --benchmark-groups-only is enabled."
        else
          for t in "${!tensor_to_shard[@]}"; do
            if [[ -z "${PROCESSED_TENSOR[$t]:-}" ]]; then
              ppl_indf="bench_ppl${_kld}_result.${t}.${qtype}.${PPL_COMMAND_CHUNKS_TO_PROCESS}.txt"
              sweep_indf="bench_sweep_result.${t}.${qtype}.${BENCH_COMMAND_CONTEXT_TO_PROCESS}.txt"

              ppl_done=false; sweep_done=false
              if need_run_ppl; then [[ -f "$ppl_indf" ]] && ppl_done=true || ppl_done=false; else ppl_done=true; fi
              if need_run_sweep; then [[ -f "$ppl_indf" ]] && sweep_done=true || sweep_done=false; else sweep_done=true; fi

              if [[ "$ppl_done" == "true" && "$sweep_done" == "true" ]]; then
                PROCESSED_TENSOR["$t"]=1
                echo "[$(timestamp)] Found existing individual result(s): $t -> marking processed."
              fi
            fi
          done
        fi

        local PROCESSED_GROUP_IDS=()

        # Loop over each shard filename and its tensor names
        for shard_fname in "${shuffled_shard_keys[@]}"; do
            # If we are in --benchmark-groups-only mode and there are no more groups to process, then we break this loop
            if [[ "$BENCH_GROUPS_ONLY" == "true" ]] && (( ${#QTYPE_REMAINING_GROUP_IDS[@]} == 0 )); then
              echo "[$(timestamp)] There are no more groups to process. Moving to next qtype..."
              break
            fi

            # For each tensor_name in this shard
            IFS=' ' read -r -a tensor_list <<< "${shard_to_tensors[$shard_fname]}"
            
            # shuffle that array
            shuffle_tensors_by_pattern tensor_list shuffled_tensor_list
            
            for tensor_name in "${shuffled_tensor_list[@]}"; do
                # If we are in --benchmark-groups-only mode and there are no more groups to process, then we break this loop
                if [[ "$BENCH_GROUPS_ONLY" == "true" ]] && (( ${#QTYPE_REMAINING_GROUP_IDS[@]} == 0 )); then
                  break
                fi

                # If this tensor was pre-marked as processed, skip but only if we are not in --benchmark-groups-only mode
                if [[ "$BENCH_GROUPS_ONLY" != "true" ]] && [[ "${PROCESSED_TENSOR[$tensor_name]:-}" == "1" ]]; then
                  continue
                fi

                # Determine all remaining group indices for this tensor (could be zero..N) when we are in --benchmark-groups-only mode only
                if [[ "$BENCH_GROUPS_ONLY" == "true" ]]; then
                  mapfile -t group_idxs_for_tensor < <(find_group_indexes_for_tensor "$tensor_name" "${PROCESSED_GROUP_IDS[@]}")
                else
                  mapfile -t group_idxs_for_tensor < <(find_group_indexes_for_tensor "$tensor_name")
                fi
                
                # If tensor belongs to one or more groups and grouping is enabled, handle group processing
                if (( ${#group_idxs_for_tensor[@]} > 0 )); then
                  # iterate over all groups this tensor belongs to and handle each group separately
                  for group_idx_for_tensor in "${group_idxs_for_tensor[@]}"; do

                    # collect all group members present in tensor_to_shard
                    collect_group_members "$group_idx_for_tensor" group_members

                    # if group has no members in this qtype, skip
                    if (( ${#group_members[@]} == 0 )); then
                      echo "[$(timestamp)] Skipping group #${group_idx_for_tensor} (qtype=$qtype) because no matching member found."
                      continue
                    fi

                    # build canonical sorted group hash (order independent)
                    group_hash=$(compute_group_hash group_members)

                    # if this exact combination for this qtype was processed earlier, mark members and skip
                    if [[ -n "${PROCESSED_GROUP_COMBOS[$group_hash]:-}" ]]; then
                      [[ "$BENCH_GROUPS_ONLY" != "true" ]] && for gm in "${group_members[@]}"; do PROCESSED_TENSOR["$gm"]=1; done
                      echo "[$(timestamp)] Result(s) already exist for group #${group_idx_for_tensor} with tensor='$tensor_name' (combo ${group_hash}), qtype='$qtype' -> skipping."
                      continue
                    fi

                    # NOTE: We will process the group even if never_benched is empty,
                    # provided the exact combo hasn't been processed before. This ensures
                    # a group whose members were processed earlier via a superset group
                    # will still get its own group benchmark run (unless that exact
                    # combination was already run).
                    echo "[$(timestamp)] Tensor '$tensor_name' belongs to group #$group_idx_for_tensor; group has ${#group_members[@]} member(s) (combo=${group_hash})."

                    # compute unique shards needed for the group (use the full group_members list)
                    declare -A shards_needed_map=()
                    declare -A shard_expected_hash=()
                    for t in "${group_members[@]}"; do
                      s="${tensor_to_shard[$t]}"
                      shards_needed_map["$s"]=1
                      shard_expected_hash["$s"]="${shard_to_hash[$s]:-}"
                    done
                    shards_needed=("${!shards_needed_map[@]}")

                    # --- NEW LOGIC: Determine which shards actually need downloading/replacement ---
                    shards_to_fetch=()    # shards that must be downloaded from remote
                    shards_to_replace=()  # shards for which we'll backup & replace
                    shards_already_ok=()  # shards already have expected hash locally (no action)
                    for s in "${shards_needed[@]}"; do
                      # expected hash for this shard
                      expected_hash="${shard_expected_hash[$s]:-}"
                      # if expected_hash is empty -> this is an error, shards should always have a hash
                      if [[ -z "$expected_hash" ]]; then
                        echo "[$(timestamp)] ❌ Error: shard '$s' has no expected hash in map (qtype='$qtype'). Aborting." >&2
                        [ -n "$GNUPG_TMPDIR" ] && rm -rf "$GNUPG_TMPDIR"
                        exit 20
                      fi

                      # try to find a local original file matching the "-NNNNN-of-NNNNN.gguf" suffix
                      if [[ $s =~ -([0-9]{5}-of-[0-9]{5})\.gguf$ ]]; then
                        suffix="${BASH_REMATCH[1]}.gguf"
                        original_file=$(find "$LOCAL_MODEL_DIR" -maxdepth 1 \( -type f -o -type l \) -name "*-${suffix}" | grep -vF "$LOCAL_DOWNLOAD_DIR/" | head -n1 || true)
                        if [[ -n "$original_file" && -f "$original_file" ]]; then
                          actual_hash=$(cached_sha256 "$original_file")
                          if [[ -n "$actual_hash" && "$actual_hash" == "$expected_hash" ]]; then
                            echo "[$(timestamp)] Group #${group_idx_for_tensor}: local shard '$original_file' already matches expected hash for '$s' -> no download/replace needed."
                            # remove any stale backup for this shard as requested
                            backup_file="${original_file}.bak"
                            if [[ -f "$backup_file" ]]; then
                              echo "[$(timestamp)] Removing existing backup file $backup_file (already matched expected hash)."
                              rm -f "$backup_file"
                            fi
                            shards_already_ok+=("$s")
                            continue
                          fi
                        fi
                      fi

                      # otherwise shard must be fetched/replaced
                      shards_to_fetch+=("$s")
                      shards_to_replace+=("$s")
                    done

                    # Download all required shards for the group (with verification) - only for shards_to_fetch
                    group_fetch_ok=true
                    downloaded_shards=()

                    # --- PARALLEL FETCH: spawn N_THREADS-throttled background workers (one per shard) ---
                    # We'll launch a background worker for each shard and throttle using the exact same
                    # jobs-count technique used elsewhere in the script. After launching all workers we
                    # call a single `wait` (no per-PID waits), then validate files and hashes.
                    fetch_worker() {
                      local s="$1"
                      local chunk_id="$2"
                      local expected_hash="$3"
                      local local_shard_tmp="$4"
                      local qtype_local="$5"
                      local gidx_local="$6"

                      # If a temp file already exists, check it first instead of immediately removing it.
                      if [[ -f "$local_shard_tmp" ]]; then
                        if [[ -n "$expected_hash" && "$USE_SHA256" == "true" ]]; then
                          # try compute SHA256
                          actual_hash="$(_sha256sum "$local_shard_tmp" 2>/dev/null | cut -d' ' -f1 || true)"
                          if [[ -n "$actual_hash" && "$actual_hash" == "$expected_hash" ]]; then
                            echo "[$(timestamp)] Group #${gidx_local}: existing tmp $local_shard_tmp matches expected hash; reusing."
                            return 0
                          else
                            if [[ -z "$actual_hash" ]]; then
                              echo "[$(timestamp)] Group #${gidx_local}: could not compute SHA256 for existing tmp $local_shard_tmp; removing and re-downloading." >&2
                            else
                              echo "[$(timestamp)] Group #${gidx_local}: existing tmp $local_shard_tmp hash mismatch ($actual_hash != $expected_hash); removing and re-downloading." >&2
                            fi
                            rm -f -- "$local_shard_tmp"
                          fi
                        else
                          # No hash verification possible; removing and re-downloading.
                          echo "[$(timestamp)] Group #${gidx_local}: found existing tmp $local_shard_tmp but no hash verification available; removing and re-downloading."
                          rm -f -- "$local_shard_tmp"
                        fi
                      fi

                      local fetched=false
                      local retry_before_delete=3
                      while true; do
                        echo "[$(timestamp)] Group #${gidx_local}: fetching shard '$s' (qtype=$qtype_local) ..."
                        if run_tensor_downloader "${qtype_local^^}" "${chunk_id}" "${LOCAL_DOWNLOAD_DIR}" "${s}"; then
                          echo "[$(timestamp)] Group #${gidx_local}: fetched $local_shard_tmp"
                        else
                          echo "[$(timestamp)] ⚠️ Warning: Could not fetch shard '$s' from remote while processing group #${gidx_local}. Aborting fetch for this shard." >&2
                          return 2
                        fi

                        if [[ -n "$expected_hash" && "$USE_SHA256" == "true" ]]; then
                          if ! actual_hash="$(_sha256sum "$local_shard_tmp" 2>/dev/null | cut -d' ' -f1)"; then
                            echo "[$(timestamp)] Group #${gidx_local}: Failed computing SHA256 for $local_shard_tmp. Retrying..." >&2
                            sleep 10
                            continue
                          fi
                          if [[ "$actual_hash" == "$expected_hash" ]]; then
                            fetched=true
                            break
                          else
                            echo "[$(timestamp)] Group #${gidx_local}: SHA256 mismatch for $s: got $actual_hash, expected $expected_hash. Retrying in 10s..." >&2
                            retry_before_delete=$((retry_before_delete - 1))
                            if [[ $retry_before_delete -le 0 ]]; then
                              echo "[$(timestamp)] Too many SHA256 mismatch for $s, removing $local_shard_tmp and resetting retry counter." >&2
                              rm -f -- "$local_shard_tmp"
                              retry_before_delete=3
                            fi
                            sleep 10
                            continue
                          fi
                        else
                          fetched=true
                          break
                        fi
                      done

                      if [[ "$fetched" == "true" ]]; then
                        return 0
                      else
                        return 1
                      fi
                    }

                    # Launch fetch workers using the same jobs-throttle pattern you used for tasks
                    for s in "${shards_to_fetch[@]}"; do
                      local_shard_tmp="${LOCAL_DOWNLOAD_DIR}/${s}"
                      chunk_id="$(echo "$s" | sed -nE 's/.*-([0-9]{5})-of-[0-9]{5}\.gguf$/\1/p')"
                      expected_hash="${shard_expected_hash[$s]:-}"

                      # throttle concurrency to N_THREADS (same approach as initial validation)
                      while (( $( (jobs -p 2>/dev/null | wc -l) || echo 0 ) >= N_THREADS )); do
                        sleep 0.2
                      done

                      # start worker in background (no per-pid tracking)
                      fetch_worker "$s" "$chunk_id" "$expected_hash" "$local_shard_tmp" "$qtype" "$group_idx_for_tensor" &
                    done

                    # Wait for all background fetch workers to finish (single wait as requested)
                    wait

                    # After wait, validate that all expected shard files exist and match expected hashes
                    echo "[$(timestamp)] Group #${group_idx_for_tensor}: verifying all shard hashes... please be patient."
                    for s in "${shards_to_fetch[@]}"; do
                      local_shard_tmp="${LOCAL_DOWNLOAD_DIR}/${s}"
                      expected_hash="${shard_expected_hash[$s]:-}"

                      if [[ ! -f "$local_shard_tmp" ]]; then
                        echo "[$(timestamp)] Group #${group_idx_for_tensor}: Expected downloaded file missing: $local_shard_tmp" >&2
                        group_fetch_ok=false
                        break
                      fi

                      if [[ -n "$expected_hash" && "$USE_SHA256" == "true" ]]; then
                        actual_hash="$(_sha256sum "$local_shard_tmp" 2>/dev/null | cut -d' ' -f1 || true)"
                        if [[ -z "$actual_hash" || "$actual_hash" != "$expected_hash" ]]; then
                          echo "[$(timestamp)] Group #${group_idx_for_tensor}: Post-wait SHA256 mismatch for $local_shard_tmp: got '${actual_hash}', expected '${expected_hash}'." >&2
                          group_fetch_ok=false
                          break
                        else
                          echo "[$(timestamp)] Group #${group_idx_for_tensor}: Hash OK for $local_shard_tmp." >&2
                        fi
                      fi

                      # If we reach here, the downloaded shard is considered valid
                      downloaded_shards+=("$local_shard_tmp")
                    done

                    if [[ "$group_fetch_ok" != true ]]; then
                      echo "[$(timestamp)] ⚠️ Group fetch failed; skipping group #${group_idx_for_tensor} and restoring any partial downloads." >&2
                      for df in "${downloaded_shards[@]:-}"; do rm -f "$df"; done
                      continue
                    fi

                    # Backup originals and replace with downloaded files for all required shards
                    # Only operate on shards_to_replace (shards that need actual replacement)
                    declare -A backups_made=()
                    replace_ok=true
                    for s in "${shards_to_replace[@]}"; do
                      if [[ $s =~ -([0-9]{5}-of-[0-9]{5})\.gguf$ ]]; then
                        suffix="${BASH_REMATCH[1]}.gguf"
                        original_file=$(find "$LOCAL_MODEL_DIR" -maxdepth 1 \( -type f -o -type l \) -name "*-${suffix}" | grep -vF "$LOCAL_DOWNLOAD_DIR/" | head -n1 || true)
                        if [[ -z "$original_file" ]]; then
                          echo "[$(timestamp)] ⚠️ Warning: Could not find local original shard matching '*-${suffix}' in $LOCAL_MODEL_DIR. Aborting group #${group_idx_for_tensor}." >&2
                          replace_ok=false
                          break
                        fi
                        backup_file="${original_file}.bak"
                        if [[ -f "$backup_file" ]]; then
                          echo "[$(timestamp)] Note: Backup already exists: $backup_file of group #${group_idx_for_tensor}. Overwriting."
                          rm -f "$backup_file"
                        fi
                        mv -f "$original_file" "$backup_file"
                        backups_made["$original_file"]="$backup_file"
                        # Move downloaded file into place
                        mv -f "${LOCAL_DOWNLOAD_DIR}/${s}" "$original_file"
                        echo "[$(timestamp)] Replaced original shard $original_file with downloaded shard for group #${group_idx_for_tensor}."
                      else
                        echo "[$(timestamp)] ⚠️ Warning: Could not extract suffix from shard '$s'. Aborting group #${group_idx_for_tensor}." >&2
                        replace_ok=false
                        break
                      fi
                    done

                    if [[ "$replace_ok" != true ]]; then
                      # restore any backups
                      for orig in "${!backups_made[@]}"; do
                        mv -f "${backups_made[$orig]}" "$orig"
                        echo "[$(timestamp)] Restored $orig from backup after failed group #${group_idx_for_tensor} replace."
                      done
                      continue
                    fi

                    # Run benchmark(s) for the group depending on mode
                    group_result_file_ppl="bench_ppl${_kld}_result.group${group_idx_for_tensor}.${qtype}.${PPL_COMMAND_CHUNKS_TO_PROCESS}.txt"
                    group_result_file_sweep="bench_sweep_result.group${group_idx_for_tensor}.${qtype}.${BENCH_COMMAND_CONTEXT_TO_PROCESS}.txt"

                    # Run PPL+KLD first if required
                    if need_run_ppl; then
                      echo "[$(timestamp)] Running PPL$PLUS_KLD for group #${group_idx_for_tensor} -> $group_result_file_ppl"
                      cmd="${PPL_COMMAND_TEMPLATE//\{MODEL_FILE\}/$main_model_file}"
                      if [[ "$NO_KLD" == "false" ]]; then
                        if [[ -z "${baseline_kld_file:-}" ]]; then
                          echo "[$(timestamp)] ❌ Error: KLD baseline file var not defined or empty." >&2; exit 10
                        elif [[ ! -f "$baseline_kld_file" ]]; then
                          echo "[$(timestamp)] ❌ Error: KLD baseline file '$baseline_kld_file' not found." >&2; exit 11
                        else
                          cmd="${cmd//\{KLD_PARAMETER\}/--kl-divergence-base \"$baseline_kld_file\" --kl-divergence}"
                        fi
                      fi
                      if eval "$cmd" > "$group_result_file_ppl" 2>&1 < /dev/null; then
                        echo "[$(timestamp)] Group #${group_idx_for_tensor} PPL$PLUS_KLD finished and saved to $group_result_file_ppl"
                        count_group_ppl_bench_success=$((count_group_ppl_bench_success + 1))
                      else
                        echo "[$(timestamp)] ⚠️ Warning: Group #${group_idx_for_tensor} PPL$PLUS_KLD command exited non-zero. See $group_result_file_ppl for details." >&2
                        count_group_ppl_bench_failure=$((count_group_ppl_bench_failure + 1))
                      fi
                    fi

                    # Run SWEEP if required (and after PPL+KLD if mode==2)
                    if need_run_sweep; then
                      echo "[$(timestamp)] Running SWEEP for group #${group_idx_for_tensor} -> $group_result_file_sweep"
                      cmd="${SWEEP_COMMAND_TEMPLATE//\{MODEL_FILE\}/$main_model_file}"
                      if eval "$cmd" > "$group_result_file_sweep" 2>&1 < /dev/null; then
                        echo "[$(timestamp)] Group #${group_idx_for_tensor} SWEEP finished and saved to $group_result_file_sweep"
                        count_group_sweep_bench_success=$((count_group_sweep_bench_success + 1))
                      else
                        echo "[$(timestamp)] ⚠️ Warning: Group #${group_idx_for_tensor} SWEEP command exited non-zero. See $group_result_file_sweep for details." >&2
                        count_group_sweep_bench_failure=$((count_group_sweep_bench_failure + 1))
                      fi
                    fi

                    # Mark all group members as processed only if all required outputs exist
                    all_done=true
                    if need_run_ppl; then [[ -f "$group_result_file_ppl" ]] || all_done=false; fi
                    if need_run_sweep; then [[ -f "$group_result_file_sweep" ]] || all_done=false; fi

                    if [[ "$all_done" == "true" ]]; then
                      [[ "$BENCH_GROUPS_ONLY" != "true" ]] && for t in "${group_members[@]}"; do PROCESSED_TENSOR["$t"]=1; done
                      # record that this exact combo (sorted member-list) has been processed for this qtype
                      PROCESSED_GROUP_COMBOS["$group_hash"]=1
                      PROCESSED_GROUP_IDS+=(${group_idx_for_tensor})
                      _tmp_qtype_remaining=()
                      remove_items_from_list_lines_inplace QTYPE_REMAINING_GROUP_IDS PROCESSED_GROUP_IDS _tmp_qtype_remaining
                      QTYPE_REMAINING_GROUP_IDS=( "${_tmp_qtype_remaining[@]}" )
                      unset _tmp_qtype_remaining
                      echo "[$(timestamp)] Recorded processed combo for group #${group_idx_for_tensor} (qtype=${qtype}) => ${group_hash}"
                    else
                      echo "[$(timestamp)] ⚠️ Warning: Not all required group result files produced; group #${group_idx_for_tensor} members will remain unmarked for re-run."
                    fi

                    # Restore originals from backups
                    for orig in "${!backups_made[@]}"; do
                      mv -f "${backups_made[$orig]}" "$orig"
                      echo "[$(timestamp)] Restored original shard $orig from backup for group #${group_idx_for_tensor}."
                    done

                    # Clean up any leftover temp files
                    for df in "${downloaded_shards[@]:-}"; do rm -f "$df"; done

                    # If termination requested, exit now
                    if [[ "$EXIT_PENDING" -eq 1 ]]; then
                      echo "[$(timestamp)] 💀 Termination flag detected; exiting group #${group_idx_for_tensor} benchmarking after finishing current operation."
                      [ -n "$GNUPG_TMPDIR" ] && rm -rf "$GNUPG_TMPDIR"
                      exit 0
                    fi
                  done

                  # done group processing; move to next tensor
                  continue
                elif [[ "$BENCH_GROUPS_ONLY" == "true" ]]; then
                  echo '[$(timestamp)] Skipping tensor='$tensor_name' because --benchmark-groups-only mode is used and this tensor is not part of any remaining groups.'
                  continue
                fi

                # Fallback to original single-tensor behaviour
                # Recompute per-tensor result_file (non-group)
                result_file_ppl="bench_ppl${_kld}_result.${tensor_name}.${qtype}.${PPL_COMMAND_CHUNKS_TO_PROCESS}.txt"
                result_file_sweep="bench_sweep_result.${tensor_name}.${qtype}.${BENCH_COMMAND_CONTEXT_TO_PROCESS}.txt"

                # If both required results are already present, skip
                ppl_done=false; sweep_done=false
                if need_run_ppl; then [[ -f "$result_file_ppl" ]] && ppl_done=true || ppl_done=false; else ppl_done=true; fi
                if need_run_sweep; then [[ -f "$result_file_sweep" ]] && sweep_done=true || sweep_done=false; else sweep_done=true; fi

                if [[ "$ppl_done" == "true" && "$sweep_done" == "true" ]]; then
                    echo "[$(timestamp)] Result(s) already exist for tensor='$tensor_name', qtype='$qtype' -> skipping."
                    PROCESSED_TENSOR["$tensor_name"]=1
                    continue
                fi

                echo "[$(timestamp)] Benchmarking tensor='$tensor_name' in shard='$shard_fname' for qtype='$qtype'."

                # Extract expected hash for this shard & tensor: find first matching line in map
                expected_hash=""
                # Use grep to find line starting with "shard_fname:"; then IFS split to get second field
                # Escape shard_fname for grep if needed
                esc_fname="$(printf '%s' "$shard_fname" | sed 's/[][^$.*/]/\\&/g')"
                while IFS= read -r line; do
                    if [[ $line =~ ^${esc_fname}: ]]; then
                        # Split by colon; second field is hash
                        IFS=':' read -r _f hf _t _ <<< "$line"
                        expected_hash="$hf"
                        break
                    fi
                done < "$local_tensors_map"
                if [[ -z "$expected_hash" ]]; then
                    echo "[$(timestamp)] ❌ Error: No hash found for shard='$shard_fname' in map. This is unexpected; aborting." >&2
                    [ -n "$GNUPG_TMPDIR" ] && rm -rf "$GNUPG_TMPDIR"
                    exit 21
                else
                    echo "[$(timestamp)] Expected SHA256 for shard='$shard_fname': $expected_hash"
                fi

                # Now fetch the shard from remote until SHA256 matches (if we have expected_hash)
                local_shard_tmp="${LOCAL_DOWNLOAD_DIR}/${shard_fname}"
                chunk_id="$(echo $shard_fname | sed -nE 's/.*-([0-9]{5})-of-[0-9]{5}\.gguf$/\1/p')"
                # Remove any old local copy
                rm -f "$local_shard_tmp"

                # Locate the local original shard in LOCAL_MODEL_DIR matching the same "-NNNNN-of-NNNNN.gguf"
                # Extract suffix "-NNNNN-of-NNNNN.gguf"
                SKIP_REPLACE=false
                if [[ $shard_fname =~ -([0-9]{5}-of-[0-9]{5})\.gguf$ ]]; then
                    suffix="${BASH_REMATCH[1]}.gguf"
                    # find local original file
                    original_file=$(find "$LOCAL_MODEL_DIR" -maxdepth 1 \( -type f -o -type l \) -name "*-${suffix}" | grep -vF "$LOCAL_DOWNLOAD_DIR/" | head -n1 || true)
                    if [[ -z "$original_file" ]]; then
                        echo "[$(timestamp)] ⚠️ Warning: Could not find local original shard matching '*-${suffix}' in $LOCAL_MODEL_DIR. Skipping this tensor." >&2
                        rm -f "$local_shard_tmp"
                        continue
                    fi
                else
                    echo "[$(timestamp)] ⚠️ Warning: Could not extract suffix from shard_fname='$shard_fname'. Skipping." >&2
                    rm -f "$local_shard_tmp"
                    continue
                fi

                # If we have an expected hash and the local original file already matches it, skip download & replacement.
                if [[ -n "$expected_hash" && -f "$original_file" ]]; then
                  actual_hash=$(cached_sha256 "$original_file")
                  if [[ -n "$actual_hash" && "$actual_hash" == "$expected_hash" ]]; then
                    echo "[$(timestamp)] Local original shard $original_file already matches expected hash; skipping download/replacement for tensor='$tensor_name'."
                    # Remove any stale backup for this shard as requested
                    backup_file="${original_file}.bak"
                    if [[ -f "$backup_file" ]]; then
                      echo "[$(timestamp)] Removing existing backup file $backup_file (already matched expected hash)."
                      rm -f "$backup_file"
                    fi
                    SKIP_REPLACE=true
                  fi
                fi

                # If not skipping, attempt download
                fetch_success=false
                retry_before_delete=3
                if [[ "$SKIP_REPLACE" == "false" ]]; then
                  while true; do
                      echo "[$(timestamp)] Fetching shard from remote: $shard_fname"
                      if run_tensor_downloader "${qtype^^}" "${chunk_id}" "${LOCAL_DOWNLOAD_DIR}" "${shard_fname}"; then
                          echo "[$(timestamp)] Fetched to $local_shard_tmp"
                      else
                          echo "[$(timestamp)] ⚠️ Warning: Could not fetch shard '$shard_fname' from remote. Skipping this tensor." >&2
                          break
                      fi

                      # If we have a hash to verify and _sha256sum available, check
                      if [[ -n "$expected_hash" && "$USE_SHA256" == "true" ]]; then
                          actual_hash="$(_sha256sum "$local_shard_tmp" | cut -d' ' -f1)"
                          if [[ "$actual_hash" == "$expected_hash" ]]; then
                              echo "[$(timestamp)] SHA256 matches for $shard_fname."
                              fetch_success=true
                              break
                          else
                              echo "[$(timestamp)] SHA256 mismatch for $shard_fname: got $actual_hash, expected $expected_hash." >&2
                              retry_before_delete=$((retry_before_delete - 1))
                              if [[ $retry_before_delete -eq 0 ]]; then
                                  echo "[$(timestamp)] Too many SHA256 mismatch for $shard_fname, removing $shard_fname from download directory!" >&2
                                  rm -f "$local_shard_tmp"
                                  retry_before_delete=3
                              fi
                              echo "[$(timestamp)] Retrying fetch after 10s..."
                              sleep 10
                              continue
                          fi
                      else
                          # No hash to check or no _sha256sum: accept fetched file
                          fetch_success=true
                          break
                      fi
                  done

                  if [[ "$fetch_success" != true ]]; then
                      echo "[$(timestamp)] Failed to fetch valid shard for tensor='$tensor_name'. Skipping this tensor." >&2
                      rm -f "$local_shard_tmp"
                      continue
                  fi

                  # Backup original
                  backup_file="${original_file}.bak"
                  if [[ -f "$backup_file" ]]; then
                      echo "[$(timestamp)] Note: Backup file already exists: $backup_file. Overwriting it." >&2
                      rm -f "$backup_file"
                  fi
                  mv -f "$original_file" "$backup_file"
                  echo "[$(timestamp)] Backed up original to $backup_file"

                  # Move downloaded shard into place
                  mv -f "$local_shard_tmp" "$original_file"
                  echo "[$(timestamp)] Replaced original shard with downloaded shard."
                fi

                # Run benchmark(s) according to mode
                # PPL+KLD:
                if need_run_ppl; then
                  echo "[$(timestamp)] Running PPL$PLUS_KLD command for tensor='$tensor_name', qtype='$qtype'... output into $result_file_ppl"
                  cmd="${PPL_COMMAND_TEMPLATE//\{MODEL_FILE\}/$main_model_file}"
                  if [[ "$NO_KLD" == "false" ]]; then
                    if [[ -z "${baseline_kld_file:-}" ]]; then
                      echo "[$(timestamp)] ❌ Error: KLD baseline file var not defined or empty." >&2; exit 10
                    elif [[ ! -f "$baseline_kld_file" ]]; then
                      echo "[$(timestamp)] ❌ Error: KLD baseline file '$baseline_kld_file' not found." >&2; exit 11
                    else
                      cmd="${cmd//\{KLD_PARAMETER\}/--kl-divergence-base \"$baseline_kld_file\" --kl-divergence}"
                    fi
                  fi
                  if eval "$cmd" > "$result_file_ppl" 2>&1 < /dev/null; then
                    echo "[$(timestamp)] 👀 PPL$PLUS_KLD output (stdout+stderr) saved to $result_file_ppl"
                    count_individual_ppl_bench_success=$((count_individual_ppl_bench_success + 1))
                  else
                    echo "[$(timestamp)] ⚠️ Warning: PPL$PLUS_KLD command exited with non-zero status for tensor='$tensor_name'. See $result_file_ppl for details." >&2
                    count_individual_ppl_bench_failure=$((count_individual_ppl_bench_failure + 1))
                  fi
                fi

                # SWEEP:
                if need_run_sweep; then
                  echo "[$(timestamp)] Running SWEEP command for tensor='$tensor_name', qtype='$qtype'..."
                  cmd="${SWEEP_COMMAND_TEMPLATE//\{MODEL_FILE\}/$main_model_file}"
                  if eval "$cmd" > "$result_file_sweep" 2>&1 < /dev/null; then
                    echo "[$(timestamp)] 👀 SWEEP output (stdout+stderr) saved to $result_file_sweep"
                    count_individual_sweep_bench_success=$((count_individual_sweep_bench_success + 1))
                  else
                    echo "[$(timestamp)] ⚠️ Warning: SWEEP command exited with non-zero status for tensor='$tensor_name'. See $result_file_sweep for details." >&2
                    count_individual_sweep_bench_failure=$((count_individual_sweep_bench_failure + 1))
                  fi
                fi

                # Mark processed only if all required outputs exist
                all_done=true
                if need_run_ppl; then [[ -f "$result_file_ppl" ]] || all_done=false; fi
                if need_run_sweep; then [[ -f "$result_file_sweep" ]] || all_done=false; fi

                if [[ "$all_done" == "true" ]]; then
                  PROCESSED_TENSOR["$tensor_name"]=1
                else
                  echo "[$(timestamp)] ⚠️ Warning: Not all required result files produced for '$tensor_name'; it will remain unmarked for re-run."
                fi

                # Restore original shard if we replaced it
                if [[ "$SKIP_REPLACE" == "false" ]]; then
                  mv -f "$backup_file" "$original_file"
                  echo "[$(timestamp)] Restored original shard from backup."
                fi

                # Clean up any leftover in LOCAL_DOWNLOAD_DIR
                rm -f "$local_shard_tmp"

                # If a termination was requested, exit now
                if [[ "$EXIT_PENDING" -eq 1 ]]; then
                  echo "[$(timestamp)] 💀 Termination flag detected; exiting '$tensor_name' benchmarking after finishing current operation."
                  [ -n "$GNUPG_TMPDIR" ] && rm -rf "$GNUPG_TMPDIR"
                  exit 0
                fi

                # End of processing this tensor
            done
        done

        # Optionally remove local_tensors_map if you don't need to keep it
        # rm -f "$local_tensors_map"

        # Summary of benchmarks for this qtype (each line prefixed with timestamp)
        total_all=$((count_group_ppl_bench_success + count_group_ppl_bench_failure + \
                    count_group_sweep_bench_success + count_group_sweep_bench_failure + \
                    count_individual_ppl_bench_success + count_individual_ppl_bench_failure + \
                    count_individual_sweep_bench_success + count_individual_sweep_bench_failure))

        if (( total_all == 0 )); then
          echo "[$(timestamp)] ⚠️ Warning: No benchmarks were run for qtype='$qtype' — either everything was already benchmarked (e.g. baseline) or there was nothing to process."
        fi

        # Print PPL (PPL or PPL+KLD) related summary if PPL mode was enabled
        if need_run_ppl; then
          echo "[$(timestamp)] Summary (PPL${PLUS_KLD} - groups):    success=${count_group_ppl_bench_success}, failure=${count_group_ppl_bench_failure}"
          echo "[$(timestamp)] Summary (PPL${PLUS_KLD} - individual): success=${count_individual_ppl_bench_success}, failure=${count_individual_ppl_bench_failure}"
        fi

        # Print SWEEP related summary if SWEEP mode was enabled
        if need_run_sweep; then
          echo "[$(timestamp)] Summary (SWEEP - groups):    success=${count_group_sweep_bench_success}, failure=${count_group_sweep_bench_failure}"
          echo "[$(timestamp)] Summary (SWEEP - individual): success=${count_individual_sweep_bench_success}, failure=${count_individual_sweep_bench_failure}"
        fi

        # Always print an overall total line for convenience
        echo "[$(timestamp)] Overall benchmark totals: success=$((count_group_ppl_bench_success + count_group_sweep_bench_success + count_individual_ppl_bench_success + count_individual_sweep_bench_success)), failure=$((count_group_ppl_bench_failure + count_group_sweep_bench_failure + count_individual_ppl_bench_failure + count_individual_sweep_bench_failure))"

        # Existing finished marker
        echo "[$(timestamp)] 🎉 Finished qtype='$qtype'."
    done
}

# Run either infinite loop or single run according to INFINITE_LOOP flag
if [[ "$INFINITE_LOOP" == "true" ]]; then
  while true; do
    run_main_loop
    echo "[$(timestamp)] ✅ All qtypes processed. Sleeping 60 seconds before next check..."
    sleep 60
  done
else
  # Single run (no infinite loop)
  run_main_loop
  echo "[$(timestamp)] ✅ All qtypes processed. (single run mode: --infinite-loop=false)"
fi
