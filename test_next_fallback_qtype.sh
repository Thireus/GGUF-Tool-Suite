#!/usr/bin/env bash
#***************************************************************#
#** This script is part of Thireus' GGUF Tool Suite.          **#
#** test_next_fallback_qtype.sh is used to test the fallback  **#
#** logic employed by quantize_model.sh upon quant failure.   **#
#**                                                           **#
#** ********************************************************* **#
#** --------------- Updated: Mar-29-2026 -------------------- **#
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
#** Copyright © 2026 - Thireus.           ₒₙₑ ₛₜₑₚ 𝒸ₗₒₛₑᵣ ₜₒ ₐGᵢ **#
#***************************************************************#
#**PLEASE REFER TO THE README FILE FOR ADDITIONAL INFORMATION!**#
#***************************************************************#

set -euo pipefail

# Standalone test harness for next_fallback_qtype.
# Usage:
#   ./test_next_fallback_qtype.sh --current q5_0 --fallbacks q5_0_r4 iq5_k iq5_k_r4 q5_k q5_k_r4
#   ./test_next_fallback_qtype.sh --current iq4_k_r4 --fallbacks iq4_k iq4_ks q4_k q4_k_r4 --chain
#
# You can also source this file and call next_fallback_qtype directly.

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

# Additional scale factor table for qtypes that carry an extra per-row scale.
# Smaller values should be preferred when BPW is the same.
declare -A ADDITIONAL_SCALE_FACTOR_TABLE=(
  [IQ1_BN]=2
  [IQ1_KT]=4
  [IQ2_BN]=4
  [IQ2_BN_R4]=4
  [IQ2_KL]=2
  [IQ2_KS]=2
  [IQ2_KT]=4
  [IQ3_KS]=2
  [IQ3_KT]=4
  [IQ4_KS]=4
  [IQ4_KSS]=4
  [IQ4_KS_R4]=4
  [IQ4_KT]=4
  [IQ5_KS]=4
  [IQ5_KS_R4]=4
  [Q8_KV]=8
  [IQ1_S_R4]=2
  [IQ1_M_R4]=2
  [Q8_KV_R8]=4
)

normalize_qtype() {
  printf '%s' "${1^^}"
}

bpw_of() {
  local q_upper
  q_upper="$(normalize_qtype "$1")"
  [[ -n "${BPW_TABLE[$q_upper]+x}" ]] || return 1
  printf '%s' "${BPW_TABLE[$q_upper]}"
}

scale_factor_of() {
  local q_upper
  q_upper="$(normalize_qtype "$1")"
  if [[ -n "${ADDITIONAL_SCALE_FACTOR_TABLE[$q_upper]+x}" ]]; then
    printf '%s' "${ADDITIONAL_SCALE_FACTOR_TABLE[$q_upper]}"
  else
    printf '%s' 0
  fi
}

qtype_in_array() {
  local needle
  needle="$(normalize_qtype "$1")"
  shift

  local item
  for item in "$@"; do
    [[ "$needle" == "$(normalize_qtype "$item")" ]] && return 0
  done
  return 1
}

qtype_is_r4_or_r8() {
  local q_upper
  q_upper="$(normalize_qtype "$1")"
  [[ "$q_upper" =~ _R[48]$ ]]
}

qtype_base_without_r4_r8() {
  local q_upper
  q_upper="$(normalize_qtype "$1")"
  if qtype_is_r4_or_r8 "$q_upper"; then
    printf '%s' "${q_upper%_*}"
  else
    printf '%s' "$q_upper"
  fi
}

qtype_is_forbidden_fallback() {
  local q_upper base_upper
  q_upper="$(normalize_qtype "$1")"
  base_upper="$(qtype_base_without_r4_r8 "$q_upper")"

  [[ "$q_upper" =~ (_KV|_BN|_N[1-9][0-9]*)$ ]] && return 0
  [[ "$base_upper" =~ (_KV|_BN|_N[1-9][0-9]*)$ ]] && return 0
  return 1
}

is_float_greater() {
  awk -v a="$1" -v b="$2" 'BEGIN { exit !(a > b) }'
}

is_float_greater_or_equal() {
  awk -v a="$1" -v b="$2" 'BEGIN { exit !((a > b) || (a == b)) }'
}

qtype_suffix_rank() {
  local q_upper
  q_upper="$(normalize_qtype "$1")"
  case "$q_upper" in
    *_R4) printf '%s' 1 ;;
    *_R8) printf '%s' 2 ;;
    *)    printf '%s' 0 ;;
  esac
}

next_fallback_qtype() {
  local current_qtype="$1"
  shift

  local current_upper current_bpw current_base current_is_variant
  current_upper="$(normalize_qtype "$current_qtype")"
  current_bpw="$(bpw_of "$current_upper")" || return 1

  current_base="$(qtype_base_without_r4_r8 "$current_upper")"
  current_is_variant=0
  qtype_is_r4_or_r8 "$current_upper" && current_is_variant=1

  local -a candidates=()
  local idx candidate_raw candidate_upper candidate_bpw candidate_base candidate_is_variant
  local same_bpw_rank same_stem_rank stem_variant_rank scale_rank i_rank suffix_group_rank suffix_rank sort_key

  for idx in "$@"; do
    candidate_raw="$idx"
    candidate_upper="$(normalize_qtype "$candidate_raw")"

    [[ "$candidate_upper" == "BF16" ]] && continue
    [[ "$candidate_upper" == "$current_upper" ]] && continue
    qtype_is_forbidden_fallback "$candidate_upper" && continue
    qtype_in_array "$candidate_upper" "${tested_qtypes[@]}" && continue

    candidate_bpw="$(bpw_of "$candidate_upper")" || continue
    is_float_greater_or_equal "$candidate_bpw" "$current_bpw" || continue

    candidate_base="$(qtype_base_without_r4_r8 "$candidate_upper")"
    candidate_is_variant=0
    qtype_is_r4_or_r8 "$candidate_upper" && candidate_is_variant=1

    # Same BPW first; higher BPW only after same BPW candidates are exhausted.
    if [[ "$candidate_bpw" == "$current_bpw" ]]; then
      same_bpw_rank=0
    else
      same_bpw_rank=1
    fi

    # Same stem (base <-> _R4/_R8) is always preferred before other candidates of the same BPW.
    if [[ "$candidate_base" == "$current_base" ]]; then
      same_stem_rank=0
      if (( current_is_variant )); then
        # Current is *_R4 or *_R8: try base first, then other variants.
        if (( candidate_is_variant == 0 )); then
          stem_variant_rank=0
        else
          stem_variant_rank=$((1 + $(qtype_suffix_rank "$candidate_upper")))
        fi
      else
        # Current is base: try suffix variants first, then base-like candidates.
        if (( candidate_is_variant )); then
          stem_variant_rank="$(qtype_suffix_rank "$candidate_upper")"
        else
          stem_variant_rank=99
        fi
      fi
    else
      same_stem_rank=1
      stem_variant_rank=0
    fi

    # Smaller additional scale factor first for qtypes with the same BPW.
    # Qtypes without an additional scale factor are treated as scale factor 0.
    scale_rank="$(scale_factor_of "$candidate_upper")"

    # Within the same BPW, prioritize i* quants over non-i quants.
    if [[ "$candidate_upper" == I* ]]; then
      i_rank=0
    else
      i_rank=1
    fi

    # Within the same BPW, prefer non-_R4/_R8 before _R4/_R8, unless same-stem rules above take precedence.
    if (( candidate_is_variant )); then
      suffix_group_rank=1
      suffix_rank="$(qtype_suffix_rank "$candidate_upper")"
    else
      suffix_group_rank=0
      suffix_rank=0
    fi

    sort_key="$(printf '%d|%d|%d|%d|%d|%d|%d|%s' \
      "$same_bpw_rank" \
      "$same_stem_rank" \
      "$stem_variant_rank" \
      "$scale_rank" \
      "$i_rank" \
      "$suffix_group_rank" \
      "$suffix_rank" \
      "$candidate_upper")"

    candidates+=("$sort_key")
  done

  ((${#candidates[@]} > 0)) || return 1

  local chosen
  chosen="$(
    printf '%s\n' "${candidates[@]}" \
      | sort -t'|' -k1,1n -k2,2n -k3,3n -k4,4n -k5,5n -k6,6n -k7,7n -k8,8 \
      | head -n1
  )"

  printf '%s' "${chosen##*|}"
}

ordered_fallback_qtypes() {
  local current_qtype="$1"
  shift

  local current_upper current_bpw current_base current_is_variant
  current_upper="$(normalize_qtype "$current_qtype")"
  current_bpw="$(bpw_of "$current_upper")" || return 1
  current_base="$(qtype_base_without_r4_r8 "$current_upper")"
  current_is_variant=0
  qtype_is_r4_or_r8 "$current_upper" && current_is_variant=1

  local -a scored=()
  local candidate_raw candidate_upper candidate_bpw candidate_base candidate_is_variant
  local same_bpw_rank same_stem_rank stem_variant_rank scale_rank i_rank suffix_group_rank suffix_rank sort_key

  for candidate_raw in "$@"; do
    candidate_upper="$(normalize_qtype "$candidate_raw")"

    [[ "$candidate_upper" == "BF16" ]] && continue
    [[ "$candidate_upper" == "$current_upper" ]] && continue
    qtype_is_forbidden_fallback "$candidate_upper" && continue
    qtype_in_array "$candidate_upper" "${tested_qtypes[@]}" && continue

    candidate_bpw="$(bpw_of "$candidate_upper")" || continue
    is_float_greater_or_equal "$candidate_bpw" "$current_bpw" || continue

    candidate_base="$(qtype_base_without_r4_r8 "$candidate_upper")"
    candidate_is_variant=0
    qtype_is_r4_or_r8 "$candidate_upper" && candidate_is_variant=1

    # Same BPW first; higher BPW only after same BPW candidates are exhausted.
    if [[ "$candidate_bpw" == "$current_bpw" ]]; then
      same_bpw_rank=0
    else
      same_bpw_rank=1
    fi

    # Same stem (base <-> _R4/_R8) is always preferred before other candidates of the same BPW.
    if [[ "$candidate_base" == "$current_base" ]]; then
      same_stem_rank=0
      if (( current_is_variant )); then
        # Current is *_R4 or *_R8: try base first, then other variants.
        if (( candidate_is_variant == 0 )); then
          stem_variant_rank=0
        else
          stem_variant_rank=$((1 + $(qtype_suffix_rank "$candidate_upper")))
        fi
      else
        # Current is base: try suffix variants first, then base-like candidates.
        if (( candidate_is_variant )); then
          stem_variant_rank="$(qtype_suffix_rank "$candidate_upper")"
        else
          stem_variant_rank=99
        fi
      fi
    else
      same_stem_rank=1
      stem_variant_rank=0
    fi

    # Smaller additional scale factor first for qtypes with the same BPW.
    # Qtypes without an additional scale factor are treated as scale factor 0.
    scale_rank="$(scale_factor_of "$candidate_upper")"

    # Within the same BPW, prioritize i* quants over non-i quants.
    if [[ "$candidate_upper" == I* ]]; then
      i_rank=0
    else
      i_rank=1
    fi

    # Within the same BPW, prefer non-_R4/_R8 before _R4/_R8, unless same-stem rules above take precedence.
    if (( candidate_is_variant )); then
      suffix_group_rank=1
      suffix_rank="$(qtype_suffix_rank "$candidate_upper")"
    else
      suffix_group_rank=0
      suffix_rank=0
    fi

    sort_key="$(printf '%d|%d|%d|%d|%d|%d|%d|%s' \
      "$same_bpw_rank" \
      "$same_stem_rank" \
      "$stem_variant_rank" \
      "$scale_rank" \
      "$i_rank" \
      "$suffix_group_rank" \
      "$suffix_rank" \
      "$candidate_upper")"

    scored+=("$sort_key")
  done

  ((${#scored[@]} > 0)) || return 1

  printf '%s\n' "${scored[@]}" \
    | sort -t'|' -k1,1n -k2,2n -k3,3n -k4,4n -k5,5n -k6,6n -k7,7n -k8,8 \
    | sed 's/^[^|]*|[^|]*|[^|]*|[^|]*|[^|]*|[^|]*|[^|]*|//'
}

usage() {
  cat <<'EOF'
Usage:
  test_next_fallback_qtype.sh --current QTYPE --fallbacks Q1 Q2 Q3... [--chain]

Options:
  --current QTYPE     The qtype that just failed.
  --fallbacks ...     Space-separated fallback pool. Stop at the next option.
  --chain             Print the full ordered eligible fallback chain.
  -h, --help          Show this help.

Examples:
  ./test_next_fallback_qtype.sh --current q5_0 --fallbacks q5_0_r4 iq5_k iq5_k_r4 q5_k q5_k_r4
  ./test_next_fallback_qtype.sh --current iq4_k_r4 --fallbacks iq4_k iq4_ks q4_k q4_k_r4 --chain
EOF
}

CURRENT=""
SHOW_CHAIN=0
FALLBACKS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --current)
      [[ ${2-} == "" || ${2-} == --* ]] && { echo "Error: --current requires a value." >&2; exit 2; }
      CURRENT="$2"
      shift 2
      ;;
    --fallbacks)
      shift
      while [[ $# -gt 0 && $1 != --* ]]; do
        FALLBACKS+=("$1")
        shift
      done
      ;;
    --chain)
      SHOW_CHAIN=1
      shift
      ;;
    *)
      echo "Error: unknown option '$1'." >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$CURRENT" ]]; then
  echo "Error: --current is required." >&2
  usage >&2
  exit 2
fi

if ((${#FALLBACKS[@]} == 0)); then
  echo "Error: --fallbacks requires at least one qtype." >&2
  usage >&2
  exit 2
fi

if (( SHOW_CHAIN )); then
  echo "Current: $(normalize_qtype "$CURRENT")"
  echo "Ordered eligible fallbacks:"
  mapfile -t ordered < <(ordered_fallback_qtypes "$CURRENT" "${FALLBACKS[@]}")
  if ((${#ordered[@]} == 0)); then
    echo "  (none)"
    exit 1
  fi
  for i in "${!ordered[@]}"; do
    q="${ordered[$i]}"
    if bpw_of "$q" >/dev/null 2>&1; then
      printf '  %d. %s (BPW=%s)\n' "$((i + 1))" "$q" "$(bpw_of "$q")"
    else
      printf '  %d. %s\n' "$((i + 1))" "$q"
    fi
  done
else
  if next="$(next_fallback_qtype "$CURRENT" "${FALLBACKS[@]}")"; then
    printf '%s\n' "$next"
  else
    echo "No eligible fallback found." >&2
    exit 1
  fi
fi