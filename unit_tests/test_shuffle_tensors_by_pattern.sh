#!/usr/bin/env bash
#***************************************************************#
#** This script is part of Thireus' GGUF Tool Suite.          **#
#** This is just a test script to check that functions work   **#
#** as intended.                                              **#
#**                                                           **#
#** ********************************************************* **#
#** --------------- Updated: Jul-24-2025 -------------------- **#
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
#** Copyright © 2025 - Thireus.          ₛₑₑ ᵧₒᵤ ₒₙ ᵣ/ₗₒ𝒸ₐₗₗₗₐₘₐ **#
#***************************************************************#
#**PLEASE REFER TO THE README FILE FOR ADDITIONAL INFORMATION!**#
#***************************************************************#

set -euo pipefail

# --- pure‑Bash shuffle replacement for `shuf` ---
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

# shuffle_tensors_by_pattern:
#   $1 = name of input array containing tensor names
#   $2 = name of output array to populate with shuffled tensor list
shuffle_tensors_by_pattern() {
  local in_name=$1
  local out_name=$2

  # nameref for input tensor list
  local -n _tensors=$in_name

  # 1) split into non‑numeric vs numeric
  local non_num=() numeric=()
  for t in "${_tensors[@]}"; do
    if [[ $t =~ [0-9] ]]; then
      numeric+=("$t")
    else
      non_num+=("$t")
    fi
  done

  # 2) shuffle non‑numeric group
  local shuffled_non_num=()
  if (( ${#non_num[@]} )); then
    mapfile -t shuffled_non_num < <(printf '%s\n' "${non_num[@]}" | shuf)
  fi

  # 3) bucket numeric tensors by pattern (replace digit‑runs with \.[0-9]+\.)
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

  # 7) combine non‑numeric first, then ordered numeric
  local __result=()
  (( ${#shuffled_non_num[@]} )) && __result+=("${shuffled_non_num[@]}")
  (( ${#ordered_num[@]}    )) && __result+=("${ordered_num[@]}")

  # write back to caller’s array
  eval "${out_name}=(\"\${__result[@]}\")"
}

# -----------------------
# Test script:

# 1) Define a mixed tensor list
tensor_list=(
  "alpha"
  "blk.1.abc.weight"
  "blk.2.abc.weight"
  "beta"
  "layer.10.conv.weight"
  "layer.11.conv.bias"
  "no_digits_here"
  "block.2.block.bias"
  "block.3.block.bias"
  "gamma"
)

# 2) Invoke the function
shuffle_tensors_by_pattern tensor_list shuffled_tensor_list

# 3) Print results
echo "Input tensor_list:"
printf '  %s\n' "${tensor_list[@]}"
echo
echo "shuffled_tensor_list:"
printf '  %s\n' "${shuffled_tensor_list[@]}"
