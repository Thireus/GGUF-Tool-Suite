#!/usr/bin/env bash
#***************************************************************#
#** This script is part of Thireus' GGUF Tool Suite.          **#
#** monitor_and_clean.sh is a script that auto creates        **#
#** tensors.map files and optionally deletes unused shards.   **#
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
#** Copyright © 2025 - Thireus.          ₜₕᵢₙₖᵢₙ𝓰... ᵦᵤₜ 𝓌ₐᵢₜ.. **#
#***************************************************************#
#**PLEASE REFER TO THE README FILE FOR ADDITIONAL INFORMATION!**#
#***************************************************************#

# Exit on error, undefined variable, or pipe failure
set -euo pipefail

# Trap SIGINT/SIGTERM for clean exit
tap_signal() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Received termination signal. Exiting."
  exit 0
}
trap 'tap_signal' SIGINT SIGTERM

# Resolve script directory for locating create_map_file.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CREATE_MAP_SCRIPT="$SCRIPT_DIR/create_map_file.sh"

if [[ ! -x "$CREATE_MAP_SCRIPT" ]]; then
    echo "Error: create_map_file.sh not found or not executable at $CREATE_MAP_SCRIPT" >&2
    exit 1
fi

#
# USER CONFIGURATION
#

# 1. User-defined regex patterns for tensor names.
#    These are Bash regexes used in [[ $tensor_name =~ $pattern ]].
#    Adjust or extend this list as needed.
USER_REGEX=(
  '^output\.weight'
  '^token_embd\.weight'
  '^blk\.[0-2]\.ffn_down\.weight'
  '^blk\.[0-2]\.ffn_gate\.weight'
  '^blk\.[0-2]\.ffn_up\.weight'
  '^blk\.([3-9]|[1-5][0-9]|60)\.ffn_down_shexp\.weight'
  '^blk\.([3-9]|[1-5][0-9]|60)\.ffn_gate_shexp\.weight'
  '^blk\.([3-9]|[1-5][0-9]|60)\.ffn_up_shexp\.weight'
  '^blk\.([3-9]|[1-5][0-9]|60)\.ffn_down_exps\.weight'
  '^blk\.([3-9]|[1-5][0-9]|60)\.ffn_up_exps\.weight'
  '^blk\.([3-9]|[1-5][0-9]|60)\.ffn_gate_exps\.weight'
)

# 2. Global option: delete shards whose tensor names do NOT match any of the above regexes.
#    Default is "false". Override by exporting DELETE_UNMATCHED_SHARDS=true or editing here.
DELETE_UNMATCHED_SHARDS="${DELETE_UNMATCHED_SHARDS:-false}"

#
# End of user configuration
#

# Base directory to monitor: first argument, default to current directory
BASE_DIR="${1:-.}"

# Verify BASE_DIR exists and is a directory
if [[ ! -d "$BASE_DIR" ]]; then
    echo "Error: '$BASE_DIR' is not a directory or does not exist."
    exit 1
fi

# Regex for shard filenames: -NNNNN-of-NNNNN.gguf
_pattern='^.*-([0-9]{5})-of-\1\.gguf$'

# Startup logs
cat <<-EOF
[$(date '+%Y-%m-%d %H:%M:%S')] Monitoring for SPLIT directories in: $BASE_DIR
Polling every 60 seconds...
DELETE_UNMATCHED_SHARDS is set to: $DELETE_UNMATCHED_SHARDS
EOF
if [[ "$DELETE_UNMATCHED_SHARDS" == "true" ]]; then
    echo "Using USER_REGEX patterns:"
    for pat in "${USER_REGEX[@]}"; do
        echo "  - $pat"
    done
fi

echo

while true; do
    # Find all *_SPLIT dirs
    while IFS= read -r -d '' split_dir; do
        # Skip if no matching shard files
        shopt -s nullglob
        ggufs=("$split_dir"/*.gguf)
        shopt -u nullglob
        # skip empty
        [[ ${#ggufs[@]} -eq 0 ]] && continue
        # skip if no matching pattern shards
        match=false
        for f in "${ggufs[@]}"; do [[ $(basename "$f") =~ $_pattern ]] && match=true && break; done
        [[ "$match" != true ]] && continue

        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Processing directory: $split_dir"

        map_file="$split_dir/tensors.map"
        if [ -f "$map_file" ]; then
            if [[ ! -s "$map_file" ]]; then
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] Empty tensors.map; processing all shards."
            else
                # Check if the file has 400 permissions (i.e., -r--------)
                perms=$(stat -c "%a" "$map_file")
                if [ "$perms" -eq 400 ]; then
                    echo "[$(date '+%Y-%m-%d %H:%M:%S')] tensors.map already produced, skipping directory: $split_dir"
                    continue
                else
                    last_line=$(tail -n1 "$map_file")
                    last_file=${last_line%%:*}
                    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Resuming from shard: $last_file"
                    # drop last line
                    sed -i '$d' "$map_file"
                    # find index in ggufs
                    for i in "${!ggufs[@]}"; do
                        if [[ "$(basename "${ggufs[i]}")" == "$last_file" ]]; then
                            ggufs=("${ggufs[@]:i}"); break
                        fi
                    done
                fi
            fi
        else
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] No tensors.map; processing all shards."
        fi

        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Invoking create_map_file.sh for each shard..."

        # Process each .gguf individually to avoid long arg lists
        for gguf_file in "${ggufs[@]}"; do
            if "$CREATE_MAP_SCRIPT" "$gguf_file"; then
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] create_map_file.sh succeeded on '$(basename "$gguf_file")'."
            else
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: create_map_file.sh failed on '$(basename "$gguf_file")'." >&2
            fi
        done && \
        chmod 400 "$map_file" # Lock file
        
        # ==============================
        # Optional deletion of unmatched shards
        # ==============================
        if [[ "$DELETE_UNMATCHED_SHARDS" == "true" ]]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] DELETE_UNMATCHED_SHARDS=true: Checking shards for deletion..."
            shopt -s nullglob
            for gguf_file in "$split_dir"/*.gguf; do
                fname="$(basename "$gguf_file")"
                # Extract lines in tensors.map for this shard
                matched_any=false
                # Escape fname for grep anchor
                esc_fname="$(printf '%s\n' "$fname" | sed 's/[][^$.*/]/\\&/g')"
                # Read each line for this shard
                while IFS= read -r line; do
                    # Fields: filename:sha256?:tensor_name:...
                    # Extract tensor_name: the third field (if hash present) or second (if empty hash)
                    # We know format is: filename:hash:tensor_name:...
                    # So cut -d: -f3
                    tensor_name="$(printf '%s' "$line" | cut -d: -f3)"
                    # Test tensor_name against USER_REGEX patterns
                    for pattern in "${USER_REGEX[@]}"; do
                        if [[ $tensor_name =~ $pattern ]]; then
                            matched_any=true
                            break 2
                        fi
                    done
                done < <(grep -E "^${esc_fname}:" "$map_file" || true)

                if [[ "$matched_any" == false ]]; then
                    # No tensor_name in this shard matched any allowed pattern -> delete shard
                    if rm -f "$gguf_file"; then
                        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Deleted shard (no matching tensor): '$fname'"
                    else
                        echo "[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: Failed to delete shard '$fname'" >&2
                    fi
                else
                    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Keeping shard (matched pattern): '$fname'"
                fi
            done
            shopt -u nullglob
        else
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] DELETE_UNMATCHED_SHARDS=false: Skipping shard deletion."
        fi
        # ==============================
        # End of optional deletion
        # ==============================

    done < <(find "$BASE_DIR" -type d -name "*_SPLIT" -print0)

    sleep 60
done
