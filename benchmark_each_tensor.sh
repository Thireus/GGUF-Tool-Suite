#!/usr/bin/env bash
#***************************************************************#
#** This script is part of Thireus' GGUF Tool Suite.          **#
#** benchmark_each_tensor.sh is a tool that evaluates the     **#
#** sensitivity to heavy quantisation of each tensor.         **#
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
#** Copyright © 2025 - Thireus.            Fᵢₙₑ₋ₜᵤₙₑ𝒹 ₒₙ 𝒸ₕₐₒₛ. **#
#***************************************************************#
#**PLEASE REFER TO THE README FILE FOR ADDITIONAL INFORMATION!**#
#***************************************************************#

# Exit on error, undefined variable, or pipe failure
set -euo pipefail

# ----------------------------------------------------------------------------
# Add a flag for delayed exit: first SIGINT sets EXIT_PENDING=1, second sends SIGTERM to llama-perplexity, third exits immediately
EXIT_PENDING=0
SIGINT_COUNT=0
LLAMA_PID=""
# Trap SIGINT (Ctrl+C) to set the flag on first, send SIGTERM on second, immediate exit on third
trap '
  SIGINT_COUNT=$((SIGINT_COUNT+1))
  if [[ $SIGINT_COUNT -eq 1 ]]; then
    EXIT_PENDING=1
    echo "[$(date "+%Y-%m-%d %H:%M:%S")] Received termination signal. Will exit after current operation finishes."
  elif [[ $SIGINT_COUNT -eq 2 ]]; then
    echo "[$(date "+%Y-%m-%d %H:%M:%S")] Received second termination signal. Sending SIGTERM to llama-perplexity."
    [[ -n "$LLAMA_PID" ]] && kill -SIGTERM "$LLAMA_PID"
  else
    echo "[$(date "+%Y-%m-%d %H:%M:%S")] Received third termination signal. Exiting immediately."
    exit 1
  fi
' SIGINT
# You can leave SIGTERM as before if you prefer it to be immediate:
trap 'echo "[$(date "+%Y-%m-%d %H:%M:%S")] Received termination signal. Exiting immediately."; exit 1' SIGTERM
# ----------------------------------------------------------------------------

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }

# ================= COMMAND-LINE ARGUMENTS =================
BENCH_CSV=""
BENCH_FROM_QTYPE=""
CUSTOM_QTYPES=()
CUSTOM_CHUNKS=""

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
    *) echo "Unknown argument: $1" >&2; exit 1;;
  esac
done

if [[ -n "$BENCH_CSV" && -z "$BENCH_FROM_QTYPE" ]]; then
  echo "Error: --benchmark-worst-tensors-from-qtype must be provided when --benchmark-worst-tensors is used." >&2
  exit 1
fi

# ================= USER CONFIGURATION =================

# 1. Remote connection settings for tensor_downloader.sh:
# Please edit tensor_downloader.sh!
# Resolve script directory for locating tensor_downloader.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TENSOR_DOWNLOADER="$SCRIPT_DIR/tensor_downloader.sh"

if [[ ! -x "$TENSOR_DOWNLOADER" ]]; then
    echo "Error: tensor_downloader.sh not found or not executable at $TENSOR_DOWNLOADER" >&2
    exit 1
fi

run_downloader() {
  set +e
  "$TENSOR_DOWNLOADER" "$@"
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
  '^output\.weight=q8_0'
  '^output_norm\.weight=f32=locked'
  '^token_embd\.weight=q8_0'

  # GPU Only - not divisible by 256 so only supports qN_0
  '^blk\.([0-9]|[1-5][0-9]|60)\.attn_k_b\.weight=q8_0=locked'

  # GPU Only
  '^blk\.([0-9]|[1-5][0-9]|60)\.attn_v_b\.weight=q8_0=locked'

  # GPU Only
  '^blk\.([0-9]|[1-5][0-9]|60)\.attn_kv_b\.weight=q8_0=locked'

  # GPU Only
  '^blk\.([0-9]|[1-5][0-9]|60)\.attn_kv_a_norm\.weight=f32=locked'
  '^blk\.([0-9]|[1-5][0-9]|60)\.attn_q_a\.weight=q8_0=locked'
  '^blk\.([0-9]|[1-5][0-9]|60)\.attn_q_a_norm\.weight=f32=locked'
  '^blk\.([0-9]|[1-5][0-9]|60)\.attn_q_b\.weight=q8_0=locked'
  '^blk\.([0-9]|[1-5][0-9]|60)\.attn_norm\.weight=f32=locked'
  '^blk\.([0-9]|[1-5][0-9]|60)\.attn_output\.weight=q8_0=locked'
  '^blk\.([3-9]|[1-5][0-9]|60)\.exp_probs_b\.bias=f32=locked'

  # GPU Only
  '^blk\.([0-9]|[1-5][0-9]|60)\.attn_kv_a_mqa\.weight=q8_0=locked'

  # GPU Only
  '^blk\.[0-2]\.ffn_down\.weight=q8_0'
  '^blk\.[0-2]\.ffn_up\.weight=q8_0'
  '^blk\.[0-2]\.ffn_gate\.weight=q8_0'
  '^blk\.([3-9]|[1-5][0-9]|60)\.ffn_gate_inp\.weight=q8_0=locked'
  '^blk\.([0-9]|[1-5][0-9]|60)\.ffn_norm\.weight=q8_0=locked'

  ## GPU-loaded ffn_*_shexp
  '^blk\.([3-9]|[1-5][0-9]|60)\.ffn_down_shexp\.weight=iq3_xxs'
  '^blk\.([3-9]|[1-5][0-9]|60)\.ffn_up_shexp\.weight=iq3_xxs'
  '^blk\.([3-9]|[1-5][0-9]|60)\.ffn_gate_shexp\.weight=iq3_xxs'

  ## CPU-loaded ffn_*_exps
  '^blk\.([3-9]|[1-5][0-9]|60)\.ffn_down_exps\.weight=iq3_xxs'
  '^blk\.([3-9]|[1-5][0-9]|60)\.ffn_up_exps\.weight=iq3_xxs'
  '^blk\.([3-9]|[1-5][0-9]|60)\.ffn_gate_exps\.weight=iq3_xxs'
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

# 4. Number of concurrent threads for initial fetch/validation:
N_THREADS=18

# 5. Number of chunks to process for PPL:
PPL_COMMAND_CHUNKS_TO_PROCESS=${CUSTOM_CHUNKS:-250}

# 6. List of qtypes to process in the loop:
QTYPES=(${CUSTOM_QTYPES[@]:-"iq1_m_r4" "iq2_k"})

# 7. Baseline QTYPE for baseline PPL computation
BASELINE_QTYPE="iq3_xxs"

# 8. PPL command template:
PPL_COMMAND_TEMPLATE='CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,2,1 ~/ik_llama-main-b3833-65dd65c-bin-win-cuda-12.8-x64/llama-perplexity \
-m {MODEL_FILE} -mla 3 -fa -amb 1024 -fmoe -ctk f16 -c 512 -ngl 99 \
-ot "blk\.(3|4|5|6)\.ffn_.*=CUDA0" -ot "blk\.(7|8|9)\.ffn_.*=CUDA1" -ot "blk\.(10|11|12)\.ffn_.*=CUDA2" \
-ot exps=CPU -b 4096 -ub 4096 --warmup-batch --no-mmap --threads 36 --main-gpu 0 --seed 1337 \
-f ../../../wiki.test.raw --chunks ${PPL_COMMAND_CHUNKS_TO_PROCESS}'

# 9. Pattern to identify the main model shard in LOCAL_MODEL_DIR.
MAIN_SHARD_PATTERN="*-00001-of-*.gguf"

# =============== End USER CONFIGURATION ===============

# Helper: parse worst-tensor from CSV if requested
if [[ -n "$BENCH_CSV" ]]; then
  echo "[$(timestamp)] Starting worst-tensor selection from CSV: $BENCH_CSV (from qtype=$BENCH_FROM_QTYPE)"
  if [[ ! -f "$BENCH_CSV" ]]; then
    echo "[$(timestamp)] Error: CSV file '$BENCH_CSV' not found." >&2; exit 1
  fi

  # Read headers and values
  IFS=',' read -r -a hdrs < <(head -n1 "$BENCH_CSV")
  echo "[$(timestamp)] Read ${#hdrs[@]} columns (first is qtype, rest are tensor names)."
  row=$(grep -P "^${BENCH_FROM_QTYPE}," "$BENCH_CSV") || { echo "[$(timestamp)] Error: qtype '$BENCH_FROM_QTYPE' not in CSV." >&2; exit 1; }
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

    # Find the tensor with highest PPL matching this pattern
    for idx in "${!hdrs[@]}"; do
      name=${hdrs[$idx]}
      if [[ $name =~ $pat_regex ]]; then
        v=${vals[$idx]:-}
        [[ -z "$v" ]] && continue
        echo "[$(timestamp)]  Matched tensor '$name' with PPL=$v"
        if (( $(bc <<< "$v > $max_val") )); then
          max_val=$v
          sel_idx=$idx
        fi
      fi
    done

    if (( sel_idx >= 0 )); then
      selected_name=${hdrs[$sel_idx]}
      echo "[$(timestamp)] Selected worst tensor for pattern '$pat_regex': $selected_name (PPL=$max_val)"
      SELECTED_TENSORS+=("$selected_name")
      SELECTED_QTYPES+=("$pat_qtype")
    else
      echo "[$(timestamp)] Warning: no tensors matching pattern '$pat_regex' found in CSV; skipping." >&2
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
    f=$(find "$LOCAL_MODEL_DIR" -maxdepth 1 -type f -name "$MAIN_SHARD_PATTERN" | head -n1 || true)
    if [[ -z "$f" ]]; then
        return "$f"
    else
        echo "$f"
        return 0
    fi
}

#echo $(find_main_model_file) | sed -E 's/-[0-9]{5}-of-[0-9]{5}\.gguf$//'
#echo $(find_main_model_file) | sed -nE 's/.*(-[0-9]{5}-of-[0-9]{5}\.gguf)$/\1/p'

# Pre-flight: restore any .gguf.bak back to .gguf before starting
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Checking for .gguf.bak files to restore..."
shopt -s nullglob
for bak in "$LOCAL_MODEL_DIR"/*.gguf.bak; do
    # Derive the target .gguf filename
    orig="${bak%.bak}"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Found backup: $(basename "$bak") -> restoring to $(basename "$orig")"
    # Overwrite any existing .gguf
    mv -f "$bak" "$orig"
done
shopt -u nullglob
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Pre-flight restoration complete."

# Initial fetch and validation for each LOCAL_QTYPE
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
  if run_downloader "${LOCAL_QTYPE^^}" "0" "." "${local_tensors_map}"; then
    echo "[$(timestamp)] Retrieved initial tensors map: $local_tensors_map"
  elif [[ $__LOCAL_QTYPE == "f32" ]]; then
    echo "[$(timestamp)] Warning: Could not fetch BF16 tensors.map for LOCAL_QTYPE='$LOCAL_QTYPE'... will try to rely on other map files later." >&2; continue
  else
    echo "[$(timestamp)] Error: Could not fetch initial tensors.map for LOCAL_QTYPE='$LOCAL_QTYPE'." >&2; exit 1
  fi
  while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    IFS=':' read -r fname expected_hash tensor_name _rest <<< "$line"
    for idx in "${!PATTERNS[@]}"; do
      if [[ "${PATTERN_QTYPES[$idx]}" == "$__LOCAL_QTYPE" ]] && [[ $tensor_name =~ ${PATTERNS[$idx]} ]]; then
        tasks+=("$fname:$expected_hash:$__LOCAL_QTYPE")
        break
      fi
    done
  done < "$local_tensors_map"
done
# Deduplicate tasks (just in case, but there shouldn't be any...)
mapfile -t tasks < <(printf "%s\n" "${tasks[@]}" | sort -u)
echo "[$(timestamp)] Validating \${#tasks[@]} shards with up to $N_THREADS threads…"

# Function to process a single shard line
# Now takes: filename, expected_hash, qtype
process_line() {
  local fname="$1"
  local expected_hash="$2"
  local qtype="$3"
  local local_file_prefix="$(echo $(find_main_model_file) | sed -E 's/-[0-9]{5}-of-[0-9]{5}\.gguf$//')"
  local local_file_suffix="$(echo $fname | sed -nE 's/.*(-[0-9]{5}-of-[0-9]{5}\.gguf)$/\1/p')"
  local chunk_id="$(echo $local_file_suffix | sed -nE 's/.*-([0-9]{5})-of-[0-9]{5}\.gguf$/\1/p')"
  local local_file="$local_file_prefix$local_file_suffix"

  [[ ! -f "$local_file" ]] && return

  echo "[$(timestamp)] Checking $fname (qtype=$qtype)…" >&2

  if ! command -v sha256sum &>/dev/null; then
    echo "[$(timestamp)] sha256sum missing; skipping check for $fname." >&2
    return
  fi

  local actual_hash
  actual_hash=$(sha256sum "$local_file" | cut -d' ' -f1)
  if [[ "$actual_hash" == "$expected_hash" ]]; then
    echo "[$(timestamp)] Hash OK for $fname." >&2
    return
  fi

  echo "[$(timestamp)] Hash mismatch ($actual_hash ≠ $expected_hash). Re‑fetching $fname…" >&2

  local tmp="${LOCAL_DOWNLOAD_DIR}/$fname"
  local -a candidates
  if [[ "$qtype" == "f32" ]]; then
    # when f32, first try bf16, then all local qtypes
    candidates=( bf16 "${LOCAL_QTYPES[@]}" )
  else
    # otherwise only re‑try the original
    candidates=( "$qtype" )
  fi

  while true; do
    rm -f "$tmp"

    for try_q in "${candidates[@]}"; do
      echo "[$(timestamp)] Trying remote path with qtype=$try_q…" >&2

      if run_downloader "${try_q^^}" "$chunk_id" "${LOCAL_DOWNLOAD_DIR}" "${fname}"; then

        local new_hash
        new_hash=$(sha256sum "$tmp" | cut -d' ' -f1)
        if [[ "$new_hash" == "$expected_hash" ]]; then
          mv -f "$tmp" "$local_file"
          echo "[$(timestamp)] Restored $fname with correct checksum via qtype=$try_q." >&2
          return
        else
          echo "[$(timestamp)] Post-fetch mismatch with qtype=$try_q ($new_hash ≠ $expected_hash)." >&2
        fi

      else
        echo "[$(timestamp)] Download failed for qtype=$try_q." >&2
      fi
    done

    echo "[$(timestamp)] All qtype attempts failed for $fname. Retrying in 10s…" >&2
    sleep 10
  done
}

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Validating ${#tasks[@]} shards with up to $N_THREADS threads…"

for entry in "${tasks[@]}"; do
  # split at the colon
  IFS=':' read -r fname expected_hash qtype <<< "$entry"

  {
    process_line "$fname" "$expected_hash" "$qtype"
  } 2>&1 | sed -u 's/^/    /' &

  # throttle concurrency
  while (( $(jobs -p | wc -l) >= N_THREADS )); do
    sleep 0.2
  done
done

wait

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Initial validation complete."

# Check availability of required commands
if ! command -v sha256sum &>/dev/null; then
    echo "Warning: sha256sum not found; SHA256 verification will be skipped." >&2
    USE_SHA256=false
else
    USE_SHA256=true
fi
if ! command -v gguf_info.py &>/dev/null; then
    echo "Warning: gguf_info.py not found in PATH; this script does not need it directly here." >&2
    # Not strictly required in this script
fi

# Baseline benchmark
baseline_result_file="bench_result.baseline.${BASELINE_QTYPE}.${PPL_COMMAND_CHUNKS_TO_PROCESS}.txt"
if [[ ! -f "$baseline_result_file" ]]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running baseline PPL benchmark for BASELINE_QTYPE='$BASELINE_QTYPE', chunks=$PPL_COMMAND_CHUNKS_TO_PROCESS."
    main_model_file=$(find_main_model_file) || { echo "Error: main model file not found for baseline." >&2; exit 1; }
    baseline_cmd="${PPL_COMMAND_TEMPLATE//\{MODEL_FILE\}/$main_model_file}"
    eval "$baseline_cmd" > "$baseline_result_file" 2>&1 < /dev/null
    estimate=$(grep "Final estimate" "$baseline_result_file" || true)
    echo "Baseline benchmark completed for chunks=$PPL_COMMAND_CHUNKS_TO_PROCESS: $estimate"
else
    estimate=$(grep "Final estimate" "$baseline_result_file" || true)
    echo "Baseline benchmark already exists for BASELINE_QTYPE='$BASELINE_QTYPE', chunks=$PPL_COMMAND_CHUNKS_TO_PROCESS: $estimate"
fi

# Startup logging
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting benchmark_each_tensor loop."
echo "Local download dir: $LOCAL_DOWNLOAD_DIR"
echo "Local model dir: $LOCAL_MODEL_DIR"
echo "QTypes: ${QTYPES[*]}"
echo "Tensor regex patterns:"
for pat in "${USER_REGEX[@]}"; do
    echo "  - $pat"
done
echo "PPL command template: $PPL_COMMAND_TEMPLATE"
echo "Main shard pattern: $MAIN_SHARD_PATTERN"
echo "Use SHA256 verification: $USE_SHA256"

# Infinite loop: for each qtype and tensor, until all benchmarks exist; then sleep 60s and repeat
while true; do
    for qtype in "${QTYPES[@]}"; do
        # Uppercase version for remote directory name
        qtype_up="${qtype^^}"
        local_tensors_map="tensors.${qtype}.map"

        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Processing qtype='$qtype'."

        # Attempt to download tensors.map from remote.
        # We first remove any old local copy to ensure we detect absence properly
        rm -f "$local_tensors_map"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Fetching remote tensors.map..."
        if run_downloader "${qtype^^}" "0" "." "${local_tensors_map}"; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Retrieved tensors.map to $local_tensors_map"
        else
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Warning: Could not fetch tensors.map for qtype='$qtype'. Skipping this qtype." >&2
            rm -f "$local_tensors_map"
            continue
        fi

        # Parse tensors.${qtype}.map: lines format assumed: filename.gguf:sha256hash:tensor_name:shape=...:dtype=...:elements=...:bytes=...
        # For each line, if tensor_name matches any USER_REGEX, then schedule that filename for benchmarking.
        declare -A shard_to_tensors=()  # map shard filename -> space-separated list of tensor_names
        while IFS= read -r line; do
            # Skip empty lines
            [[ -z "$line" ]] && continue
            # Split line by colon: filename:hash:tensor_name:...
            IFS=':' read -r fname file_hash tensor_name _rest <<< "$line"
            # If tensor_name matches any USER_REGEX, record it under shard fname
            for entry in "${USER_REGEX[@]}"; do
              # split on “=”: LHS is the actual regex, RHS is the qtype (which we ignore here)
              IFS='=' read -r pat _qtype locked <<< "$entry"
              [[ -n "$locked" ]] && continue # Skip locked tensors
              if [[ $tensor_name =~ $pat ]]; then
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
                break
              fi
            done
        done < "$local_tensors_map"

        if [[ ${#shard_to_tensors[@]} -eq 0 ]]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] No matching tensors for qtype='$qtype' in $local_tensors_map. Skipping."
            continue
        fi

        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Found ${#shard_to_tensors[@]} shard(s) with matching tensors for qtype='$qtype'."

        # Find main model file once
        main_model_file=$(find_main_model_file) || {
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Error: Could not find main model shard matching '$MAIN_SHARD_PATTERN' in $LOCAL_MODEL_DIR. Skipping benchmarking." >&2
            continue
        }
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Main model shard for PPL: $main_model_file"

        # Loop over each shard filename and its tensor names
        for shard_fname in "${!shard_to_tensors[@]}"; do
            # For each tensor_name in this shard
            IFS=' ' read -r -a tensor_list <<< "${shard_to_tensors[$shard_fname]}"
            for tensor_name in "${tensor_list[@]}"; do
                result_file="bench_result.${tensor_name}.${qtype}.${PPL_COMMAND_CHUNKS_TO_PROCESS}.txt"
                if [[ -f "$result_file" ]]; then
                    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Result already exists for tensor='$tensor_name', qtype='$qtype' -> $result_file. Skipping."
                    continue
                fi

                echo "[$(date '+%Y-%m-%d %H:%M:%S')] Benchmarking tensor='$tensor_name' in shard='$shard_fname' for qtype='$qtype'."

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
                    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Warning: No hash found for shard='$shard_fname' in map. Will skip SHA256 check." >&2
                else
                    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Expected SHA256 for shard='$shard_fname': $expected_hash"
                fi

                # Now fetch the shard from remote until SHA256 matches (if we have expected_hash)
                local_shard_tmp="${LOCAL_DOWNLOAD_DIR}/${shard_fname}"
                chunk_id="$(echo $shard_fname | sed -nE 's/.*-([0-9]{5})-of-[0-9]{5}\.gguf$/\1/p')"
                # Remove any old local copy
                rm -f "$local_shard_tmp"

                # Attempt download in a loop if SHA256 mismatch
                fetch_success=false
                while true; do
                    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Fetching shard from remote: $shard_fname"
                    if run_downloader "${qtype^^}" "${chunk_id}" "${LOCAL_DOWNLOAD_DIR}" "${shard_fname}"; then
                        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Fetched to $local_shard_tmp"
                    else
                        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Warning: Could not fetch shard '$shard_fname' from remote. Skipping this tensor." >&2
                        break
                    fi

                    # If we have a hash to verify and sha256sum available, check
                    if [[ -n "$expected_hash" && "$USE_SHA256" == "true" ]]; then
                        actual_hash="$(sha256sum "$local_shard_tmp" | awk '{print $1}')"
                        if [[ "$actual_hash" == "$expected_hash" ]]; then
                            echo "[$(date '+%Y-%m-%d %H:%M:%S')] SHA256 matches for $shard_fname."
                            fetch_success=true
                            break
                        else
                            echo "[$(date '+%Y-%m-%d %H:%M:%S')] SHA256 mismatch for $shard_fname: got $actual_hash, expected $expected_hash." >&2
                            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Retrying fetch after 10s..."
                            sleep 10
                            continue
                        fi
                    else
                        # No hash to check or no sha256sum: accept fetched file
                        fetch_success=true
                        break
                    fi
                done

                if [[ "$fetch_success" != true ]]; then
                    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Failed to fetch valid shard for tensor='$tensor_name'. Skipping this tensor." >&2
                    rm -f "$local_shard_tmp"
                    continue
                fi

                # Locate the local original shard in LOCAL_MODEL_DIR matching the same "-NNNNN-of-NNNNN.gguf"
                # Extract suffix "-NNNNN-of-NNNNN.gguf"
                if [[ $shard_fname =~ -([0-9]{5}-of-[0-9]{5})\.gguf$ ]]; then
                    suffix="${BASH_REMATCH[1]}.gguf"
                    # find local original file
                    original_file=$(find "$LOCAL_MODEL_DIR" -maxdepth 1 -type f -name "*-${suffix}" | grep -vF "$LOCAL_DOWNLOAD_DIR/" | head -n1 || true)
                    if [[ -z "$original_file" ]]; then
                        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Warning: Could not find local original shard matching '*-${suffix}' in $LOCAL_MODEL_DIR. Skipping this tensor." >&2
                        rm -f "$local_shard_tmp"
                        continue
                    fi
                else
                    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Warning: Could not extract suffix from shard_fname='$shard_fname'. Skipping." >&2
                    rm -f "$local_shard_tmp"
                    continue
                fi

                echo "[$(date '+%Y-%m-%d %H:%M:%S')] Found local original shard: $original_file"

                # Backup original
                backup_file="${original_file}.bak"
                if [[ -f "$backup_file" ]]; then
                    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Note: Backup file already exists: $backup_file. Overwriting it." >&2
                    rm -f "$backup_file"
                fi
                mv "$original_file" "$backup_file"
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] Backed up original to $backup_file"

                # Move downloaded shard into place
                mv "$local_shard_tmp" "$original_file"
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] Replaced original shard with downloaded shard."

                # Run PPL command
                # Replace {MODEL_FILE} in template with main_model_file
                cmd="${PPL_COMMAND_TEMPLATE//\{MODEL_FILE\}/$main_model_file}"
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running PPL command for tensor='$tensor_name', qtype='$qtype'..."
                # Redirect stdout and stderr into result file, prevent interactive stdin
                eval "$cmd" > "$result_file" 2>&1 < /dev/null || {
                    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Warning: PPL command exited with non-zero status for tensor='$tensor_name'. See $result_file for details." >&2
                }
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] PPL output (stdout+stderr) saved to $result_file"

                # Restore original shard
                mv "$backup_file" "$original_file"
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] Restored original shard from backup."

                # Clean up any leftover in LOCAL_DOWNLOAD_DIR
                rm -f "$local_shard_tmp"

                # If a termination was requested, exit now
                if [[ "$EXIT_PENDING" -eq 1 ]]; then
                  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Termination flag detected; exiting after finishing current operation."
                  exit 0
                fi

                # End of processing this tensor
            done
        done

        # Optionally remove local_tensors_map if you don't need to keep it
        # rm -f "$local_tensors_map"

        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished qtype='$qtype'."
    done

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] All qtypes processed. Sleeping 60 seconds before next check..."
    sleep 60
done
