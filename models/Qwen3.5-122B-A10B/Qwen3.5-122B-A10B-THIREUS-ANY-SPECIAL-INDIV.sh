#!/usr/bin/env bash
#***************************************************************#
#** This script is part of Thireus' GGUF Tool Suite.          **#
#** $MODEL-THIREUS-ANY-SPECIAL-INDIV.sh used for model        **#
#** quantization purpose. Make sure to adjust USER CONFIG!    **#
#**                                                           **#
#** ********************************************************* **#
#** --------------- Updated: Mar-23-2026 -------------------- **#
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
#** Copyright © 2026 - Thireus.        Cₕₐₜᵦₒₜₛ ₙₑₑ𝒹 ₜₕₑᵣₐₚᵧ ₜₒₒ **#
#***************************************************************#
#**PLEASE REFER TO THE README FILE FOR ADDITIONAL INFORMATION!**#
#***************************************************************#

set -euo pipefail
shopt -s nullglob

if [[ $# -lt 1 ]]; then
  echo "❌ Error: missing quantization dtype argument." >&2
  echo "Usage: $0 <gguf-quant-dtype>" >&2
  exit 20
fi

# ------------------ USER CONFIG ------------------
MODEL="Qwen3.5-122B-A10B"
CHUNKS_TOTAL=880
MAINTAINER="THIREUS" # BF16 repo maintainer
LLAMA_QUANTIZE_BIN="llama-quantize"
# ------------------ USER CONFIG ------------------

# --- BPW lookup table for GGUF quant dtypes ---
declare -A BPW_TABLE=(
  [F32]=32
  [F16]=16
  [BF16]=16
  [Q8_0_R8]=8.5
  [Q8_0]=8.5
  [Q8_K_R8]=8.0625
  [Q8_KV]=8
  [F8]=8
  [IQ6_K]=6.625
  [Q6_K_R4]=6.5625
  [Q6_K]=6.5625
  [Q6_0_R4]=6.5
  [Q6_0]=6.5
  [Q5_1]=6
  [Q5_K_R4]=5.5
  [Q5_K]=5.5
  [Q5_0_R4]=5.5
  [Q5_0]=5.5
  [IQ5_K_R4]=5.5
  [IQ5_K]=5.5
  [IQ5_KS_R4]=5.25
  [IQ5_KS]=5.25
  [Q4_1]=5
  [Q4_K_R4]=4.5
  [Q4_K]=4.5
  [Q4_0_R8]=4.5
  [Q4_0]=4.5
  [IQ4_NL_R4]=4.5
  [IQ4_NL]=4.5
  [IQ4_K_R4]=4.5
  [IQ4_K]=4.5
  [IQ4_XS_R8]=4.25
  [IQ4_XS]=4.25
  [IQ4_KS_R4]=4.25
  [IQ4_KS]=4.25
  [IQ4_KT]=4
  [IQ4_KSS]=4
  [IQ3_KL]=4
  [IQ3_M]=3.66
  [Q3_K_R4]=3.4375
  [Q3_K]=3.4375
  [IQ3_S_R4]=3.4375
  [IQ3_S]=3.4375
  [IQ3_K_R4]=3.4375
  [IQ3_K]=3.4375
  [IQ3_XS]=3.3
  [IQ3_KS]=3.1875
  [IQ3_KT]=3.125
  [IQ3_XXS_R4]=3.0625
  [IQ3_XXS]=3.0625
  [IQ2_M_R4]=2.7
  [IQ2_M]=2.7
  [IQ2_KL]=2.6875
  [Q2_K_R4]=2.625
  [Q2_K]=2.625
  [IQ2_S]=2.5625
  [IQ2_K_R4]=2.375
  [IQ2_K]=2.375
  [IQ2_XS_R4]=2.3125
  [IQ2_XS]=2.3125
  [IQ2_KS]=2.1875
  [IQ2_KT]=2.125
  [IQ2_XXS_R4]=2.0625
  [IQ2_XXS]=2.0625
  [IQ2_BN_R4]=2
  [IQ2_BN]=2
  [IQ1_M_R4]=1.75
  [IQ1_M]=1.75
  [IQ1_KT]=1.75
  [IQ1_BN]=1.625
  [IQ1_S]=1.5625
  [IQ1_S_R4]=1.5
)

FALLBACK_LLAMA_QUANTS=("iq1_s" "iq1_m" "iq2_xxs" "iq2_xs" "iq2_s" "q2_K" "iq3_xxs" "iq3_s" "q3_K" "iq4_xs" "iq4_nl" "q4_0" "q4_K" "q4_1" "q5_0" "q5_K" "q5_1" "q6_K" "q8_0" "bf16")
FALLBACK_IK_QUANTS=("iq1_s_r4" "iq1_s" "iq1_bn" "iq1_kt" "iq1_m" "iq1_m_r4" "iq2_bn" "iq2_bn_r4" "iq2_xxs" "iq2_xxs_r4" "iq2_kt" "iq2_ks" "iq2_xs" "iq2_xs_r4" "iq2_k" "iq2_k_r4" "iq2_s" "q2_K" "q2_k_r4" "iq2_kl" "iq3_xxs" "iq3_xxs_r4" "iq3_kt" "iq3_ks" "iq3_k" "iq3_k_r4" "iq3_s" "iq3_s_r4" "q3_K" "q3_k_r4" "iq4_kss" "iq4_kt" "iq4_ks" "iq4_ks_r4" "iq4_xs" "iq4_xs_r8" "iq4_k" "iq4_k_r4" "iq4_nl" "iq4_nl_r4" "q4_0" "q4_0_r8" "q4_K" "q4_k_r4" "q4_1" "iq5_ks" "iq5_ks_r4" "iq5_k" "iq5_k_r4" "q5_0" "q5_0_r4" "q5_K" "q5_k_r4" "q5_1" "q6_0" "q6_0_r4" "q6_K" "q6_k_r4" "iq6_k" "q8_KV" "q8_k_r8" "q8_0" "q8_0_r8" "bf16")
# ------------------------------------------------------------------------

# --------- DEFAULTS FOR SAFE EARLY EXIT SUMMARY ----------
TOTAL_ATTEMPTS=0
TOTAL_FAILED_ATTEMPTS=0
TOTAL_FALLBACK_ATTEMPTS=0
TOTAL_SUCCESSFUL_SHARDS=0
TOTAL_SUCCESSFUL_TIME=0
TOTAL_ATTEMPT_TIME=0
TOTAL_FALLBACK_HOPS=0
RUN_START_TS=0
RUN_ELAPSED=0
RESUME_START=2
TARGET_DIR=""
TARGET_PREFIX=""
SOURCE_DIR=""
SOURCE_PREFIX=""
STATE_FILE=""
CHUNK_TOTAL_PADDED="$(printf '%05d' "$CHUNKS_TOTAL")"
EXIT_CODE=0

declare -A QTYPE_ATTEMPTS=()
declare -A QTYPE_SUCCESSES=()
declare -A QTYPE_FAILED_ATTEMPTS=()
# --------------------------------------------------------

normalize_qtype() {
  printf '%s' "${1^^}"
}

bpw_of() {
  local qtype_upper
  qtype_upper="$(normalize_qtype "$1")"
  [[ -n "${BPW_TABLE[$qtype_upper]+x}" ]] || return 1
  printf '%s' "${BPW_TABLE[$qtype_upper]}"
}

is_float_greater() {
  awk -v a="$1" -v b="$2" 'BEGIN { exit !(a > b) }'
}

format_duration() {
  local total_seconds=${1:-0}
  local days hours minutes seconds
  days=$(( total_seconds / 86400 ))
  hours=$(( (total_seconds % 86400) / 3600 ))
  minutes=$(( (total_seconds % 3600) / 60 ))
  seconds=$(( total_seconds % 60 ))

  if (( days > 0 )); then
    printf '%dd %02dh %02dm %02ds' "$days" "$hours" "$minutes" "$seconds"
  elif (( hours > 0 )); then
    printf '%dh %02dm %02ds' "$hours" "$minutes" "$seconds"
  elif (( minutes > 0 )); then
    printf '%dm %02ds' "$minutes" "$seconds"
  else
    printf '%ds' "$seconds"
  fi
}

resolve_llama_quantize_bin() {
  resolved="$(command -v "$LLAMA_QUANTIZE_BIN" 2>/dev/null || true)"
  if [[ -z "$resolved" ]]; then
    echo "❌ Error: Provided --llama-quantize binary '$LLAMA_QUANTIZE_BIN' is not executable or not found." >&2
    exit 22
  fi
  LLAMA_QUANTIZE_BIN="$resolved"
}

prompt_delete_latest_shard() {
  local shard_path="$1"
  local shard_id="$2"
  local response

  if [[ ! -t 0 ]]; then
    echo "❌ Error: Cannot prompt for deletion because stdin is not interactive." >&2
    echo "Please delete the shard manually and rerun the script, or run it from an interactive terminal." >&2
    return 1
  fi

  while true; do
    read -r -p "Delete shard $(printf '%05d' "$shard_id") at '$shard_path' and resume from shard $(printf '%05d' "$shard_id")? [y/N]: " response
    case "$response" in
      [yY]|[yY][eE][sS])
        return 0
        ;;
      [nN]|[nN][oO]|"")
        return 1
        ;;
      *)
        echo "Please answer y or n."
        ;;
    esac
  done
}

write_resume_state() {
  local resume_start="$1"
  local latest_shard_path="$2"
  local latest_shard_id="$3"
  local latest_shard_total="$4"

  mkdir -p "$TARGET_DIR"
  {
    printf 'RESUME_START=%q\n' "$resume_start"
    printf 'LATEST_SHARD_PATH=%q\n' "$latest_shard_path"
    printf 'LATEST_SHARD_ID=%q\n' "$latest_shard_id"
    printf 'LATEST_SHARD_TOTAL=%q\n' "$latest_shard_total"
  } > "$STATE_FILE"
}

# ------------------ LLAMA-QUANTIZE SUPPORT CHECK ------------------
# Validate that the provided llama-quantize binary supports --individual-tensors.
if [[ -z "$LLAMA_QUANTIZE_BIN" ]]; then
  echo "❌ Error: --llama-quantize must be provided when quantization from bf16 is required." >&2
  echo "Please obtain a build that includes --individual-tensors support from:" >&2
  echo "  https://github.com/Thireus/ik_llama.cpp/tree/th/quantize_individual_tensors" >&2
  echo "Pre-built releases are at: https://github.com/Thireus/ik_llama.cpp/releases (look for th-quantize_individual_tensors*)." >&2
  exit 21
fi

resolve_llama_quantize_bin

# Check help text for --individual-tensors
matching_help_line="$("$LLAMA_QUANTIZE_BIN" --help 2>&1 | grep -- '--individual-tensors' | head -n1 || true)"
if [[ -z "$matching_help_line" ]]; then
  echo "❌ Error: The provided llama-quantize binary '$LLAMA_QUANTIZE_BIN' does not advertise the --individual-tensors option." >&2
  echo "Please obtain a build with individual-tensors support from:" >&2
  echo "  https://github.com/Thireus/ik_llama.cpp/tree/th/quantize_individual_tensors" >&2
  echo "Pre-built releases are at: https://github.com/Thireus/ik_llama.cpp/releases (look for th-quantize_individual_tensors*)." >&2
  exit 23
fi
# ------------------------------------------------------------------------

count_existing_shards() {
  EXISTING_SHARD_COUNT=0
  HIGHEST_SHARD_ID=""
  HIGHEST_SHARD_TOTAL=""
  HIGHEST_SHARD_PATH=""

  local shard_file shard_base shard_id shard_total
  local -a shard_files=( "$TARGET_DIR/${TARGET_PREFIX}"-*-of-*.gguf )

  for shard_file in "${shard_files[@]}"; do
    [[ -f "$shard_file" ]] || continue
    shard_base="$(basename "$shard_file")"

    if [[ "$shard_base" =~ -([0-9]{5})-of-([0-9]{5})\.gguf$ ]]; then
      shard_id=$((10#${BASH_REMATCH[1]}))
      shard_total=$((10#${BASH_REMATCH[2]}))

      EXISTING_SHARD_COUNT=$((EXISTING_SHARD_COUNT + 1))

      if [[ -z "$HIGHEST_SHARD_ID" || "$shard_id" -gt "$HIGHEST_SHARD_ID" ]]; then
        HIGHEST_SHARD_ID="$shard_id"
        HIGHEST_SHARD_TOTAL="$shard_total"
        HIGHEST_SHARD_PATH="$shard_file"
      fi
    fi
  done
}

cleanup_partial_output() {
  local output_path="$1"
  rm -f -- "$output_path" 2>/dev/null || true
}

next_fallback_qtype() {
  local current_upper="$1"
  local current_bpw candidate_upper candidate_bpw candidate_key
  current_bpw="$(bpw_of "$current_upper")" || return 1

  local -a candidates=()
  local idx
  for idx in "${!FALLBACK_LLAMA_QUANTS[@]}"; do
    candidate_upper="$(normalize_qtype "${FALLBACK_LLAMA_QUANTS[$idx]}")"
    [[ "$candidate_upper" == "BF16" ]] && continue
    candidate_bpw="$(bpw_of "$candidate_upper")" || continue

    if is_float_greater "$candidate_bpw" "$current_bpw"; then
      candidates+=("$candidate_bpw|$idx|$candidate_upper")
    fi
  done

  ((${#candidates[@]} > 0)) || return 1
  candidate_key="$(printf '%s\n' "${candidates[@]}" | sort -t'|' -k1,1n -k2,2n | head -n1)"
  printf '%s' "${candidate_key##*|}"
}

print_summary() {
  local exit_code="${1:-0}"
  local avg_shard_time avg_attempt_time avg_fallback_hops_per_failed

  if (( TOTAL_SUCCESSFUL_SHARDS > 0 )); then
    avg_shard_time=$(( TOTAL_SUCCESSFUL_TIME / TOTAL_SUCCESSFUL_SHARDS ))
  else
    avg_shard_time=0
  fi

  if (( TOTAL_ATTEMPTS > 0 )); then
    avg_attempt_time=$(( TOTAL_ATTEMPT_TIME / TOTAL_ATTEMPTS ))
  else
    avg_attempt_time=0
  fi

  if (( TOTAL_FAILED_ATTEMPTS > 0 )); then
    avg_fallback_hops_per_failed=$(( TOTAL_FALLBACK_HOPS / TOTAL_FAILED_ATTEMPTS ))
  else
    avg_fallback_hops_per_failed=0
  fi

  echo
  echo "====================== RUN SUMMARY ======================"
  echo "Exit status: $exit_code"
  echo "Model: $MODEL"
  echo "Target dtype: $TARGET_QTYPE"
  echo "Output directory: $TARGET_DIR"
  echo "Resume start: $RESUME_START"
  echo "Completed shards: $TOTAL_SUCCESSFUL_SHARDS"
  echo "Total attempts: $TOTAL_ATTEMPTS"
  echo "Failed attempts: $TOTAL_FAILED_ATTEMPTS"
  echo "Fallback attempts: $TOTAL_FALLBACK_ATTEMPTS"
  echo "Fallback hops: $TOTAL_FALLBACK_HOPS"
  echo "Average fallback hops per failed attempt: $avg_fallback_hops_per_failed"
  echo "Total wall time: $(format_duration "$RUN_ELAPSED")"
  echo "Average successful shard time: $(format_duration "$avg_shard_time")"
  echo "Average attempt time: $(format_duration "$avg_attempt_time")"
  echo
  echo "Attempt metrics by qtype:"
  local q q_upper attempts successes failures
  for q in "${FALLBACK_LLAMA_QUANTS[@]}"; do
    q_upper="$(normalize_qtype "$q")"
    attempts="${QTYPE_ATTEMPTS[$q_upper]:-0}"
    successes="${QTYPE_SUCCESSES[$q_upper]:-0}"
    failures="${QTYPE_FAILED_ATTEMPTS[$q_upper]:-0}"
    if (( attempts > 0 || successes > 0 || failures > 0 )); then
      echo "  $q_upper -> attempts: $attempts, successes: $successes, failures: $failures"
    fi
  done
  echo "========================================================="
}

trap 'EXIT_CODE=$?; print_summary "$EXIT_CODE"' EXIT

TARGET_QTYPE="$(normalize_qtype "$1")"

if [[ "$TARGET_QTYPE" == "BF16" ]]; then
  echo "❌ Error: BF16 is reserved as the terminal fallback and will not be attempted by this script." >&2
  exit 24
fi

if [[ -z "${BPW_TABLE[$TARGET_QTYPE]+x}" ]]; then
  echo "❌ Error: Unknown GGUF quant dtype '$1' (normalized: '$TARGET_QTYPE')." >&2
  exit 25
fi

SOURCE_DIR="${MODEL}-${MAINTAINER}-BF16-SPECIAL_SPLIT"
SOURCE_PREFIX="${MODEL}-${MAINTAINER}-BF16-SPECIAL_TENSOR"
TARGET_DIR="${MODEL}-${MAINTAINER}-${TARGET_QTYPE}-SPECIAL_SPLIT"
TARGET_PREFIX="${MODEL}-${MAINTAINER}-${TARGET_QTYPE}-SPECIAL_TENSOR"
STATE_FILE="$TARGET_DIR/.resume_state"

mkdir -p "$TARGET_DIR"

# ------------------ RESUME / EXISTING SHARDS CHECK ------------------
RESUME_START=2
count_existing_shards

if [[ -f "$STATE_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$STATE_FILE"

  if [[ -z "${LATEST_SHARD_PATH:-}" || -z "${RESUME_START:-}" ]]; then
    echo "❌ Error: Existing resume state file is incomplete:" >&2
    echo "  $STATE_FILE" >&2
    exit 34
  fi

  if [[ -e "${LATEST_SHARD_PATH}" ]]; then
    if [[ "${EXISTING_SHARD_COUNT:-0}" -eq 1 && "${HIGHEST_SHARD_ID:-0}" -eq 1 ]]; then
      echo "⚠️  Only shard 00001 exists in '$TARGET_DIR'."
      if prompt_delete_latest_shard "${HIGHEST_SHARD_PATH}" "${HIGHEST_SHARD_ID}"; then
        rm -f -- "${HIGHEST_SHARD_PATH}"
        RESUME_START=2
      else
        echo "⚠️  Leaving shard 00001 in place; exiting without resuming."
        exit 33
      fi
    else
      echo "⚠️  Existing GGUF shards were found in: $TARGET_DIR"
      echo "⚠️  Highest shard id found: $(printf '%05d' "$HIGHEST_SHARD_ID")"
      echo "⚠️  Latest shard path: $LATEST_SHARD_PATH"
      if prompt_delete_latest_shard "$LATEST_SHARD_PATH" "$HIGHEST_SHARD_ID"; then
        rm -f -- "$LATEST_SHARD_PATH"
      else
        echo "⚠️  Delete that shard first, because it is likely corrupted."
        echo "⚠️  Then rerun this script to resume from shard $RESUME_START."
        exit 33
      fi
    fi
  fi
else
  if (( EXISTING_SHARD_COUNT > 0 )); then
    if [[ -n "${HIGHEST_SHARD_ID:-}" && "$HIGHEST_SHARD_ID" -eq 1 ]]; then
      echo "⚠️  Only shard 00001 exists in '$TARGET_DIR'."
      write_resume_state 2 "${HIGHEST_SHARD_PATH}" "${HIGHEST_SHARD_ID}" "${HIGHEST_SHARD_TOTAL}"
      if prompt_delete_latest_shard "${HIGHEST_SHARD_PATH}" "${HIGHEST_SHARD_ID}"; then
        rm -f -- "${HIGHEST_SHARD_PATH}"
        RESUME_START=2
      else
        echo "⚠️  Leaving shard 00001 in place; exiting without resuming."
        exit 33
      fi
    else
      RESUME_START="$HIGHEST_SHARD_ID"
      write_resume_state "$RESUME_START" "${HIGHEST_SHARD_PATH}" "${HIGHEST_SHARD_ID}" "${HIGHEST_SHARD_TOTAL}"

      echo "⚠️  Existing GGUF shards were found in: $TARGET_DIR"
      echo "⚠️  Highest shard id found: $(printf '%05d' "$HIGHEST_SHARD_ID")"
      echo "⚠️  Latest shard path: $HIGHEST_SHARD_PATH"
      if prompt_delete_latest_shard "$HIGHEST_SHARD_PATH" "$HIGHEST_SHARD_ID"; then
        rm -f -- "$HIGHEST_SHARD_PATH"
      else
        echo "⚠️  Delete that shard first, because it is likely corrupted."
        echo "⚠️  Then rerun this script to resume from shard $RESUME_START."
        exit 33
      fi
    fi
  else
    RESUME_START=2
  fi
fi
# -------------------------------------------------------------------

if (( RESUME_START > CHUNKS_TOTAL )); then
  echo "❌ Error: Resume start ($RESUME_START) is greater than CHUNKS_TOTAL ($CHUNKS_TOTAL)." >&2
  exit 26
fi

# The GGUF shard format is: chunkid-of-chuckstotal.gguf
# For the source-side existence check, the total is the configured CHUNKS_TOTAL.
SOURCE_FIRST_SHARD="$SOURCE_DIR/${SOURCE_PREFIX}-00001-of-${CHUNK_TOTAL_PADDED}.gguf"
if [[ ! -f "$SOURCE_FIRST_SHARD" ]]; then
  echo "❌ Error: Required source shard does not exist before looping:" >&2
  echo "  $SOURCE_FIRST_SHARD" >&2
  exit 27
fi

RUN_START_TS=$(date +%s)

for i in $(seq "$RESUME_START" "$CHUNKS_TOTAL"); do
  if (( i > 2 )); then
    skip_first_shard="--skip-first-shard"
  else
    skip_first_shard=""
  fi

  chunk_id_padded="$(printf '%05d' "$i")"

  # The source filename uses the changing chunk id and the fixed user-configured total.
  source_actual_shard="$SOURCE_DIR/${SOURCE_PREFIX}-${chunk_id_padded}-of-${CHUNK_TOTAL_PADDED}.gguf"

  # llama-quantize still receives the base output name; it expands it into
  # ${TARGET_PREFIX}-${chunkid}-of-${chuckstotal}.gguf internally.
  output_shard="$TARGET_DIR/${TARGET_PREFIX}.gguf"
  output_actual_shard="$TARGET_DIR/${TARGET_PREFIX}-${chunk_id_padded}-of-${CHUNK_TOTAL_PADDED}.gguf"

  if [[ ! -f "$source_actual_shard" ]]; then
    echo "❌ Error: Source file for shard $i is missing:" >&2
    echo "  $source_actual_shard" >&2
    exit 29
  fi

  current_qtype="$TARGET_QTYPE"
  shard_overall_start_ts=$(date +%s)
  shard_attempts=0
  shard_used_fallbacks=0
  shard_chain=("$current_qtype")

  while :; do
    if [[ "$current_qtype" == "BF16" ]]; then
      echo "❌ Error: Fallback chain reached BF16 for shard $i. BF16 will not be attempted." >&2
      exit 30
    fi

    shard_attempts=$((shard_attempts + 1))
    TOTAL_ATTEMPTS=$((TOTAL_ATTEMPTS + 1))
    QTYPE_ATTEMPTS["$current_qtype"]=$(( ${QTYPE_ATTEMPTS["$current_qtype"]:-0} + 1 ))

    attempt_start_ts=$(date +%s)
    echo "▶ Shard $i/$CHUNKS_TOTAL: trying ${current_qtype} (attempt $shard_attempts)"

    if "$LLAMA_QUANTIZE_BIN" $skip_first_shard --keep-split \
        --imatrix imatrix_ubergarm.dat \
        --ignore-imatrix-rules \
        --individual-tensors "$i" \
        "$SOURCE_FIRST_SHARD" \
        "$output_shard" \
        "$current_qtype" \
        "$(nproc)"; then
      attempt_elapsed=$(( $(date +%s) - attempt_start_ts ))
      shard_elapsed=$(( $(date +%s) - shard_overall_start_ts ))
      TOTAL_SUCCESSFUL_SHARDS=$((TOTAL_SUCCESSFUL_SHARDS + 1))
      TOTAL_SUCCESSFUL_TIME=$((TOTAL_SUCCESSFUL_TIME + shard_elapsed))
      TOTAL_ATTEMPT_TIME=$((TOTAL_ATTEMPT_TIME + attempt_elapsed))
      QTYPE_SUCCESSES["$current_qtype"]=$(( ${QTYPE_SUCCESSES["$current_qtype"]:-0} + 1 ))

      if (( shard_used_fallbacks > 0 )); then
        echo "⚠️  Shard $i completed after $shard_used_fallbacks fallback hop(s)."
        echo "⚠️  Fallback chain: ${shard_chain[*]}"
      fi

      echo "✅ Shard $i completed with ${current_qtype} in $(format_duration "$shard_elapsed")"
      if [[ -e "$output_actual_shard" ]]; then
        chmod 444 "$output_actual_shard"
      fi
      break
    fi

    attempt_elapsed=$(( $(date +%s) - attempt_start_ts ))
    TOTAL_FAILED_ATTEMPTS=$((TOTAL_FAILED_ATTEMPTS + 1))
    TOTAL_ATTEMPT_TIME=$((TOTAL_ATTEMPT_TIME + attempt_elapsed))
    QTYPE_FAILED_ATTEMPTS["$current_qtype"]=$(( ${QTYPE_FAILED_ATTEMPTS["$current_qtype"]:-0} + 1 ))

    echo "⚠️  Shard $i failed with ${current_qtype} after $(format_duration "$attempt_elapsed")."
    cleanup_partial_output "$output_actual_shard"

    next_qtype="$(next_fallback_qtype "$current_qtype" || true)"
    if [[ -z "$next_qtype" ]]; then
      echo "❌ Error: No higher-BPW fallback remains for ${current_qtype}." >&2
      echo "The fallback list is exhausted before BF16, so this shard cannot continue." >&2
      exit 31
    fi

    if [[ "$next_qtype" == "BF16" ]]; then
      echo "❌ Error: The next fallback after ${current_qtype} would be BF16, but BF16 will not be attempted." >&2
      echo "Please inspect the failing shard manually or choose a different starting quantization." >&2
      exit 32
    fi

    next_bpw="$(bpw_of "$next_qtype")"
    current_bpw="$(bpw_of "$current_qtype")"
    echo "⚠️  Falling back for shard $i: ${current_qtype} (${current_bpw} BPW) -> ${next_qtype} (${next_bpw} BPW)"

    TOTAL_FALLBACK_ATTEMPTS=$((TOTAL_FALLBACK_ATTEMPTS + 1))
    TOTAL_FALLBACK_HOPS=$((TOTAL_FALLBACK_HOPS + 1))
    shard_used_fallbacks=$((shard_used_fallbacks + 1))
    shard_chain+=("$next_qtype")
    current_qtype="$next_qtype"
  done
done

RUN_ELAPSED=$(( $(date +%s) - RUN_START_TS ))

rm -f -- "$STATE_FILE" 2>/dev/null || true

trap - EXIT
print_summary 0
