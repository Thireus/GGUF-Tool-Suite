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
#** Copyright © 2025 - Thireus.           ᵦₐ𝒸ₖ ᵤₚ ₐₗₗ ₜₕₑ ₜₕᵢₙ𝓰ₛ! **#
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


# --- Mock data
declare -A shard_to_tensors=(
  [shardA]="blk.2.abc.weight blk.3.def.bias"
  [shardB]="xyz weight"
  [shardC]="blk.4.abc.weight blk.5.def.bias"
  [shardD]="12345 67890"
  [shardE]="blk.6.abc.weight blk.7.def.bias"
  [shardF]="blk.8.abc.weight blk.9.def.bias"
  [shardG]="blk.2.eee.weight blk.3.aaa.bias"
  [shardK]="layer.2.conv.bias layer.2.conv.weight"
  [shardL]="blk.1.bob.weight blk.5.bob.bias"
  [shardM]="blk.2.bob.weight blk.6.bob.bias"
  [shardN]="blk.3.alice.weight blk.7.alice.bias"
  [shardO]="blk.4.alice.weight blk.8.alice.bias"
)

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

shuffle_shards_by_tensor_patterns shard_to_tensors shuffled_shard_keys

echo "Final shuffled_shard_keys:"
printf '  %s\n' "${shuffled_shard_keys[@]}"
