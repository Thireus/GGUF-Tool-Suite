#!/usr/bin/env bash
#***************************************************************#
#** This script is part of Thireus' GGUF Tool Suite.          **#
#** quant_downloader.sh is a tool that downloads GGUF shards  **#
#** from a recipe file containing tensor regexe entries.      **#
#**                                                           **#
#** ********************************************************* **#
#** --------------- Updated: Aug-08-2025 -------------------- **#
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
#** Copyright © 2025 - Thireus.        𝒻ₐᵢₗₑ𝒹 ₜₒ ₐₗₗₒ𝒸ₐₜₑ ᵦᵤ𝒻𝒻ₑᵣ **#
#***************************************************************#
#**PLEASE REFER TO THE README FILE FOR ADDITIONAL INFORMATION!**#
#***************************************************************#

# Exit on error, undefined variable, or pipe failure
set -euo pipefail

# ---------------- SIGNAL HANDLING ----------------
# Ensure child processes are killed on interrupt
# Trap SIGINT and SIGTERM, kill the entire group, but never let kill’s
# “no such process group” error exit the script.
trap_handler() {
  # ignore further SIGTERM so our kill -- -$$ won't re-enter this handler
  trap '' SIGTERM
  echo "[$(timestamp)] Signal caught, forwarding to process group..." >&2
  kill -- -$$ 2>/dev/null || true
  exit 1
}
trap trap_handler SIGINT SIGTERM

# ----------------- DEFAULTS & INITIALIZATION -----------------
MAX_JOBS=8             # Default concurrency level
NEW_MAP=true           # Whether to try obtain the latest map files
FORCE_REDOWNLOAD=false # Whether to redownload all files (maps, shards, first shard)
VERIFY_ONLY=false      # If true, only verify hashes and report errors
BASE_DIR="."           # Base directory for model and download dirs
QTYPE="BF16"           # Default quantization type used for the first shard and filenames
SKIP_GPG=false         # If true, skip the gpg signature verification of the signed files

# --------------------- USAGE & ARG PARSING -------------------
usage() {
  echo "Usage: $0 [options] <recipe-file>" >&2
  echo "  -j, --max-jobs N        Set maximum concurrent downloads (default: $MAX_JOBS)" >&2
  echo "      --no-new-map        Prevent the script from downloading new map files" >&2
  echo "      --force-redownload  Force redownload of all shards and maps, ignoring existing files" >&2
  echo "      --verify            Only verify existing shard hashes; report mismatches; skip downloads" >&2
  echo "      --qtype QUANT       Set quantization type for the first shard and filenames (default: BF16); use highest qtype of the model!" >&2
  echo "      --skip-gpg          Do not verify the gpg signature of the downloaded files" >&2
  echo "  -d, --dest DIR          Base path for model and download dirs (default: .)" >&2
  echo "  <recipe-file>: path to recipe containing USER_REGEX lines (one per tensor; must have .recipe extension)" >&2
  exit 1
}

# Parse arguments (supports GNU long options)
PARSED_OPTS=$(getopt -n "$0" -o j:d: -l max-jobs:,no-new-map,force-redownload,verify,qtype:,skip-gpg,dest:,destination: -- "$@") || usage
eval set -- "$PARSED_OPTS"
while true; do
  case "$1" in
    -j|--max-jobs)
      MAX_JOBS="$2"
      shift 2
      ;;
    --no-new-map)
      NEW_MAP=false
      shift
      ;;
    --force-redownload)
      FORCE_REDOWNLOAD=true
      shift
      ;;
    --verify)
      VERIFY_ONLY=true
      shift
      ;;
    --qtype)
      QTYPE="${2^^}"
      shift 2
      ;;
    --skip-gpg)
      SKIP_GPG=true
      shift
      ;;
    -d|--dest|--destination)
      BASE_DIR="$2"
      shift 2
      ;;
    --)
      shift
      break
      ;;
    *)
      usage
      ;;
  esac
done

# Check recipe-file argument
if [[ $# -ne 1 ]]; then
  usage
fi
RECIPE_FILE="$1"
# Enforce .recipe extension
if [[ "${RECIPE_FILE##*.}" != "recipe" ]]; then
  echo "❌ Error: Recipe file '$RECIPE_FILE' must have a .recipe extension." >&2
  exit 1
fi
if [[ ! -f "$RECIPE_FILE" ]]; then
  echo "❌ Error: Recipe file '$RECIPE_FILE' not found." >&2
  exit 1
fi

# ----------------------- DIRECTORIES -------------------------
# Ensure base directory exists
mkdir -p "$BASE_DIR"
LOCAL_DOWNLOAD_DIR="$BASE_DIR/downloaded_shards"
LOCAL_MODEL_DIR="$BASE_DIR"
mkdir -p "$LOCAL_DOWNLOAD_DIR"

echo "[INFO] Using base directory: $BASE_DIR"
echo "[INFO] Download dir: $LOCAL_DOWNLOAD_DIR"
echo "[INFO] Model dir: $LOCAL_MODEL_DIR"
echo "[INFO] Max jobs: $MAX_JOBS, Obtain new map: $NEW_MAP, Force redownload: $FORCE_REDOWNLOAD, Verify only: $VERIFY_ONLY, Skip signature verification: $SKIP_GPG"

# -------------------- TIMESTAMP FUNCTION ---------------------
timestamp() { date '+%Y-%m-%d %H:%M:%S'; }

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
    # file‑mode: pass filename as $1
    "${sha256tool[@]}" "${args[@]}" "$1" | awk '{print $1}'
  else
    # stdin‑mode: read data from pipe
    "${sha256tool[@]}" "${args[@]}" | awk '{print $1}'
  fi
}

# ------------------ READ USER_REGEX PATTERNS -----------------
declare -a USER_REGEX
while IFS= read -r line || [[ -n "$line" ]]; do
  line="${line##*( )}"
  line="${line%%*( )}"
  [[ -z "$line" || "$line" =~ ^# ]] && continue
  USER_REGEX+=("$line")
done < "$RECIPE_FILE"

if [[ ${#USER_REGEX[@]} -eq 0 ]]; then
  echo "❌ Error: No valid USER_REGEX entries found in '$RECIPE_FILE'." >&2
  exit 1
fi

# ------------------ LOCATE DOWNLOADER ------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TENSOR_DOWNLOADER="$SCRIPT_DIR/tensor_downloader.sh"
if [[ ! -x "$TENSOR_DOWNLOADER" ]]; then
  echo "❌ Error: tensor_downloader.sh not found or not executable at $TENSOR_DOWNLOADER" >&2
  exit 1
fi
run_downloader() {
  set +e
  "$TENSOR_DOWNLOADER" "$@" & CHILD_PID=$!; wait $CHILD_PID; ret=$?; set -e; return $ret
}

# ----------------- VERIFY GPG READINESS ----------------------

if [[ "$SKIP_GPG" != "true" ]]; then
  if [ ! -f "$SCRIPT_DIR/trusted-keys.asc" ]; then
    echo "❌ Error: trusted-keys.asc not found in the script directory."
    echo "Hint: Provide trusted-keys.asc in the same directory as this script or use the --skip-gpg option to disable gpg signature verification."
    exit 6
  fi
  if command -v gpg >/dev/null 2>&1; then
    # Create a temporary GNUPGHOME
    GNUPG_TMPDIR=$(mktemp -d)
    if [ -z "$GNUPG_TMPDIR" ]; then
      echo "❌ Error: Failed to create temporary GPG home directory." >&2
      exit 8
    fi
    # Try importing the keys (silently) to check validity
    if ! gpg --homedir "$GNUPG_TMPDIR" --no-default-keyring --import "$SCRIPT_DIR/trusted-keys.asc" > /dev/null 2>&1; then
      echo "❌ Error: trusted-keys.asc contains missing or invalid GPG public keys."
      echo "Hint: Add valid public keys to this file or re-run with the --skip-gpg option to bypass signature verification."
      [ -n "$GNUPG_TMPDIR" ] && rm -rf "$GNUPG_TMPDIR"
      exit 7
    fi
  else
    echo "⚠️ Warning: 'gpg' command not found. Valid GPG public keys verification skipped." >&2
  fi
fi

# -------------------- HASH & SHARD STORAGE -------------------
declare -A T_HASHES SHARD_ID
set_t_hash() { local key="${1,,}::${2,,}"; T_HASHES["$key"]="$3"; }
get_t_hash() { echo "${T_HASHES["${1,,}::${2,,}"]}"; }
set_shard_id() { SHARD_ID["${1,,}"]="$2"; }
get_shard_id() { echo "${SHARD_ID["${1,,}"]}"; }

# -------- PREPARE QTYPES & PATTERNS --------
declare -a PATTERNS PATTERN_QTYPES
for entry in "${USER_REGEX[@]}"; do
  IFS='=' read -r pat qtype _ <<< "$entry"
  PATTERNS+=("$pat")
  PATTERN_QTYPES+=("$qtype")
done
readarray -t UNIQUE_QTYPES < <(printf "%s
" "${PATTERN_QTYPES[@]}" | sort -u)

# Ensure QTYPE included first
if [[ " ${UNIQUE_QTYPES[*]} " != *"${QTYPE}"* ]]; then
  UNIQUE_QTYPES=("${QTYPE}" "${UNIQUE_QTYPES[@]}")
fi

# --------------- FETCH MAPS & COLLECT ----------------
declare -a TENSORS_TO_FETCH BF16_SHARDS
for _q in "${UNIQUE_QTYPES[@]}"; do
  qtype=${_q^^}
  _qtype=$qtype
  [[ "$qtype" == "F32" ]] && _qtype="${QTYPE}"
  echo "[$(timestamp)] Fetching ${_qtype} tensor map (and gpg signature if enabled) for ${qtype} quants"
  mapfile="tensors.${_qtype,,}.map"
  if [[ "$FORCE_REDOWNLOAD" == true ]]; then
    echo "[$(timestamp)] Force redownload: removing existing map $mapfile and $mapfile.sig"
    rm -f "$mapfile"
    rm -f "$mapfile.sig"
  fi
  if [[ "$NEW_MAP" == true ]]; then
      if [[ -f "$mapfile" ]]; then
          mv -f "$mapfile" "$mapfile.bak"
          if [[ -f "$mapfile.sig" ]]; then
            mv -f "$mapfile.sig" "$mapfile.sig.bak"
          else
            rm -f "$mapfile.sig.bak" # Delete the backup because now the backup may not correspond to the $mapfile.bak
          fi
          if ! run_downloader "$_qtype" 0 . "$mapfile"; then
              echo "⚠️ Warning: failed to fetch map for $_qtype. Using existing map file." >&2
              mv -f "$mapfile.bak" "$mapfile"
              if [[ -f "$mapfile.sig.bak" ]]; then
                mv -f "$mapfile.sig.bak" "$mapfile.sig"
              fi
          else
              # Success: optionally remove backup or keep it
              rm -f "$mapfile.bak"
              # Download the signature
              if [[ "$SKIP_GPG" != "true" ]]; then
                if ! run_downloader "$_qtype" -1 . "$mapfile.sig"; then
                    if [[ -f "$mapfile.sig.bak" ]]; then
                        echo "⚠️ Warning: failed to fetch map gpg signature for $_qtype. Using existing map gpg signature file." >&2
                        mv -f "$mapfile.sig.bak" "$mapfile.sig"
                    else
                        echo "❌ Error: failed to fetch map gpg signature for $_qtype and no existing map gpg signature file present!" >&2
                        [ -n "$GNUPG_TMPDIR" ] && rm -rf "$GNUPG_TMPDIR"
                        exit 3
                    fi
                else
                    # Success: optionally remove backup or keep it
                    rm -f "$mapfile.sig.bak"
                fi
              fi
          fi
      else
          # $mapfile does not exist; just try downloading
          if ! run_downloader "$_qtype" 0 . "$mapfile"; then
              echo "❌ Error: failed to fetch map for $_qtype" >&2
              exit 1
          else
              # Download the signature
              if [[ "$SKIP_GPG" != "true" ]]; then
                rm -f "$LOCAL_MODEL_DIR/$mapfile.sig"
                if ! run_downloader "$_qtype" -1 . "$mapfile.sig"; then
                    echo "❌ Error: failed to fetch map gpg signature for $_qtype" >&2
                    [ -n "$GNUPG_TMPDIR" ] && rm -rf "$GNUPG_TMPDIR"
                    exit 3
                fi
              fi
          fi
      fi
  else
      # NEW_MAP is false; just download normally (which will only happen if the file doesn't already exist)
      if ! run_downloader "$_qtype" 0 . "$mapfile"; then
          echo "❌ Error: failed to fetch map for $_qtype" >&2
          exit 1
      fi
      # Download the signature
      if [[ "$SKIP_GPG" != "true" ]]; then
        if ! run_downloader "$_qtype" -1 . "$mapfile.sig"; then
            echo "❌ Error: failed to fetch map gpg signature for $_qtype" >&2
            [ -n "$GNUPG_TMPDIR" ] && rm -rf "$GNUPG_TMPDIR"
            exit 3
        fi
      fi
  fi

  if [[ "$SKIP_GPG" != "true" ]]; then
    if command -v gpg >/dev/null 2>&1; then
      if [ ! -f "$mapfile" ]; then
          echo "❌ Error: Map file '$mapfile' not found."
          [ -n "$GNUPG_TMPDIR" ] && rm -rf "$GNUPG_TMPDIR"
          exit 5
      fi
      if [ ! -f "$mapfile.sig" ]; then
          echo "❌ Error: Signature file '$mapfile.sig' is missing."
          echo "Hint: To skip GPG verification, re-run this script with the --skip-gpg option."
          [ -n "$GNUPG_TMPDIR" ] && rm -rf "$GNUPG_TMPDIR"
          exit 5
      fi
      if gpg --homedir "$GNUPG_TMPDIR" --no-default-keyring --verify "$mapfile.sig" "$mapfile" > /dev/null 2>&1; then
          echo "[$(timestamp)] GPG signature verification successful."
      else
          echo "❌ Error: GPG signature verification failed for '$mapfile.sig'."
          [ -n "$GNUPG_TMPDIR" ] && rm -rf "$GNUPG_TMPDIR"
          exit 4
      fi
    else
      echo "⚠️ Warning: 'gpg' command not found. Signature verification skipped." >&2
    fi
  fi

  while IFS=: read -r fname hash tname _; do
    if [[ $fname =~ -([0-9]{5})-of-[0-9]{5}\.gguf$ ]]; then
      shard_id=$((10#${BASH_REMATCH[1]}))
      set_shard_id "$tname" "$shard_id"
      set_t_hash "$qtype" "$tname" "$hash"
      if [[ "$qtype" == "${QTYPE}" ]]; then
        BF16_SHARDS+=("$fname")
        TENSORS_TO_FETCH+=("$tname")
      fi
    else
      echo "⚠️ Warning: skipping invalid filename '$fname'" >&2
    fi
  done < "$mapfile"
done

# --------------- CONCURRENCY HELPERS ----------------
wait_for_slot() {
  while (( $(jobs -rp | wc -l) >= MAX_JOBS )); do sleep 0.5; done
}

# ------------- SHARD DOWNLOAD/VERIFY LOGIC --------------
download_shard() {
  local idx="$1"
  local tensor="${TENSORS_TO_FETCH[$idx]}"
  echo "[$(timestamp)] Starting process for tensor='$tensor'"

  chunk_id=$(get_shard_id "$tensor")

  for i in "${!PATTERNS[@]}"; do
    pat="${PATTERNS[$i]}"
    if [[ "$tensor" =~ $pat ]]; then
      qtype="${PATTERN_QTYPES[$i]^^]}"
      dl_type="$qtype"
      [[ "${qtype^^}" == "F32" ]] && dl_type="${QTYPE}"

      local shard_file="${BF16_SHARDS[$idx]}"
      local local_path="$LOCAL_MODEL_DIR/$shard_file"
      local dl_path="$LOCAL_DOWNLOAD_DIR/$shard_file"

      local shard_id=$(echo "$shard_file" | sed -E 's/.*-([0-9]{5})-of-[0-9]{5}\.gguf/\1/')

      got=""
      need_download=false
      if [[ "$FORCE_REDOWNLOAD" == true ]]; then
          echo "[$(timestamp)] Force redownload: removing existing shard $shard_file"
          rm -f "$dl_path" "$local_path" || true
          skip_mv=false
      else
          skip_mv=true
      fi
      while [[ "$need_download" == false ]] && [[ "$got" == "" ]]; do
        if [[ -f "$local_path" ]] || [[ -f "$dl_path" ]]; then
            if [[ -f "$local_path" ]]; then
                _path=$local_path
            else
                _path=$dl_path
                skip_mv=false
            fi
            if command -v _sha256sum &>/dev/null; then
                got=$(_sha256sum "$_path" | cut -d' ' -f1)
                exp=$(get_t_hash "$qtype" "$tensor")
                if [[ "$got" != "$exp" ]]; then
                    echo "[$(timestamp)] Will redownload due to hash mismatch for '$shard_file' - tensor '$tensor' of qtype: '$qtype' ($got != $exp)"
                    rm -f "$dl_path" "$local_path" || true
                    need_download=true
                else
                    echo "[$(timestamp)] File id '$shard_id' - tensor '$tensor' of qtype: '$qtype' hash is valid!"
                fi
            else
                echo "⚠️ Warning: _sha256sum command missing - hash cannot be verified!"
            fi
        else
            need_download=true
        fi

        if [[ "$need_download" == true ]]; then
            echo "[$(timestamp)] Downloading file id '$shard_id' - tensor '$tensor' of qtype: '$qtype' (chunk_id=$chunk_id)"
            until run_downloader "$dl_type" "$chunk_id" "$LOCAL_DOWNLOAD_DIR" "$shard_file"; do
                echo "[$(timestamp)] Download failed; retrying in 10s..."
                sleep 10
            done
            need_download=false
            skip_mv=false
            got=""
        else
          if [[ "$skip_mv" == true ]]; then
              echo "[$(timestamp)] Shard ${shard_file} present and valid - tensor '$tensor' of qtype: '$qtype'"
          else
              mv -f "$dl_path" "$LOCAL_MODEL_DIR/"
              echo "[$(timestamp)] Saved file id '$shard_id' - tensor '$tensor' of qtype: '$qtype'"
          fi
        fi
      done

      break
    fi
  done
}

# ------------------ VERIFY-ONLY MODE -------------------
if [[ "$VERIFY_ONLY" == "true" ]]; then
  echo "[$(timestamp)] VERIFY_ONLY: verifying existing shards"

  # failure marker
  FAIL_MARKER="$BASE_DIR/.verify_failed"
  rm -f "$FAIL_MARKER"

  # helper to cap concurrent jobs
  wait_for_slot() {
    while (( $(jobs -rp | wc -l) >= MAX_JOBS )); do sleep 0.1; done
  }

  SCRIPT_PID=$$

  # 1) check first shard explicitly, in background
  wait_for_slot
  (
    _find=$(find "$LOCAL_MODEL_DIR" -maxdepth 1 -type f -name "*-*-of-*.gguf")
    total=$(echo ${_find} | head -n1 | sed -E 's/.*-[0-9]{5}-of-([0-9]{5})\.gguf/\1/')
    gguf_first=$(basename "$(echo ${_find} | head -n1 \
      | sed "s/-[0-9]\{5\}-of-$total\.gguf$/-00001-of-$total.gguf/")")
    if [[ "$total" != "" && "$gguf_first" != "" && -f "$LOCAL_MODEL_DIR/$gguf_first" ]]; then
      if [[ ! "$gguf_first" =~ "-BF16-" && "${QTYPE}" == "BF16" ]]; then
        echo "[$(timestamp)] VERIFY_ONLY: Error - Non-BF16 qtype detected for first shard, please re-run using the --qtype option!!!"
        kill -s TERM "$SCRIPT_PID"  # Kill parent
        exit 2
      else
        if [[ "$SKIP_GPG" != "true" ]]; then
          if command -v gpg >/dev/null 2>&1; then
            if [ ! -f "$LOCAL_MODEL_DIR/$gguf_first" ]; then
                echo "[$(timestamp)] VERIFY_ONLY: Error - Shard file '$gguf_first' not found."
                kill -s TERM "$SCRIPT_PID"  # Kill parent
                exit 5
            fi
            if [ ! -f "$LOCAL_MODEL_DIR/$gguf_first.sig" ]; then
                echo "[$(timestamp)] VERIFY_ONLY: Error - Signature file '$gguf_first.sig' is missing."
                echo "Hint: To skip GPG verification, re-run this script with the --skip-gpg option."
                kill -s TERM "$SCRIPT_PID"  # Kill parent
                exit 5
            fi
            if gpg --homedir "$GNUPG_TMPDIR" --no-default-keyring --verify "$LOCAL_MODEL_DIR/$gguf_first.sig" "$LOCAL_MODEL_DIR/$gguf_first" > /dev/null 2>&1; then
                echo "[$(timestamp)] GPG signature verification for '$gguf_first.sig' successful."
            else
                echo "[$(timestamp)] VERIFY_ONLY: Error - GPG signature verification failed for '$gguf_first.sig'."
                kill -s TERM "$SCRIPT_PID"  # Kill parent
                exit 4
            fi
          else
            echo "⚠️ Warning: 'gpg' command not found. Signature verification skipped." >&2
          fi
        fi
        echo "[$(timestamp)] OK: $gguf_first"
      fi
    else
      echo "[$(timestamp)] MISSING: $gguf_first"
      touch "$FAIL_MARKER"
    fi
  ) &

  # 2) check each remaining shard, in parallel
  for idx in "${!TENSORS_TO_FETCH[@]}"; do
    wait_for_slot
    (
      for i in "${!PATTERNS[@]}"; do
        pat="${PATTERNS[$i]}"
        if [[ "${TENSORS_TO_FETCH[$idx]}" =~ $pat ]]; then
          qtype="${PATTERN_QTYPES[$i]^^]}"
          shardfile="${BF16_SHARDS[$idx]}"
          local_path="$LOCAL_MODEL_DIR/$shardfile"

          if [[ -f "$local_path" ]] && command -v _sha256sum &>/dev/null; then
            got=$(_sha256sum "$local_path" | cut -d' ' -f1)
            exp=$(get_t_hash "$qtype" "${TENSORS_TO_FETCH[$idx]}")
            if [[ "$got" != "$exp" ]]; then
              echo "[$(timestamp)] WRONG HASH: $shardfile ($got != $exp) - tensor: '${TENSORS_TO_FETCH[$idx]}' - qtype: '$qtype'"
              touch "$FAIL_MARKER"
            else
              echo "[$(timestamp)] OK: $shardfile"
            fi
          else
            echo "[$(timestamp)] MISSING: $shardfile"
            touch "$FAIL_MARKER"
          fi
          break
        fi
      done
    ) &
  done

  # wait for all verifications to finish
  wait

  # Cleanup
  if [[ "$SKIP_GPG" != "true" ]]; then
    if command -v gpg >/dev/null 2>&1; then
      [ -n "$GNUPG_TMPDIR" ] && rm -rf "$GNUPG_TMPDIR"
    fi
  fi

  # summary
  if [[ -f "$FAIL_MARKER" ]]; then
    echo "[$(timestamp)] VERIFY_ONLY: some files missing or with hash mismatch"
    exit 1
  else
    echo "[$(timestamp)] VERIFY_ONLY: all files present and with valid hashes"
    exit 0
  fi
fi

# ------------------ MAIN DOWNLOAD LOOP -------------------
for idx in "${!TENSORS_TO_FETCH[@]}"; do
 wait_for_slot
 download_shard "$idx" &
done
wait

_find=$(find "$LOCAL_MODEL_DIR" -maxdepth 1 -type f -name "*-*-of-*.gguf")
total=$(echo ${_find} | head -n1 | sed -E 's/.*-[0-9]{5}-of-([0-9]{5})\.gguf/\1/')
gguf_first=$(basename "$(echo ${_find} | head -n1 | sed "s/-[0-9]\{5\}-of-$total\.gguf$/-00001-of-$total.gguf/")")

# ------------- FINAL FIRST-SHARD FETCH (non-verify) -----
if [[ "$VERIFY_ONLY" != true ]]; then
  echo "[$(timestamp)] Fetching first shard separately"
  if [[ "$total" != "" ]] && [[ "$gguf_first" != "" ]]; then
    if [[ "$FORCE_REDOWNLOAD" == true ]]; then
      echo "[$(timestamp)] Force redownload: removing existing first shard (and gpg signature)"
      rm -f "$LOCAL_MODEL_DIR/$gguf_first" "$LOCAL_DOWNLOAD_DIR/$gguf_first" || true
      rm -f "$LOCAL_MODEL_DIR/$gguf_first.sig" "$LOCAL_DOWNLOAD_DIR/$gguf_first.sig" || true
    fi
    if ! [ -f "$LOCAL_MODEL_DIR/$gguf_first" ]; then
      until run_downloader "${QTYPE}" 1 "$LOCAL_DOWNLOAD_DIR" "$(basename "$gguf_first")"; do
        echo "[$(timestamp)] First shard download failed; retrying in 10s..."
        sleep 10
      done
      mv -f "$LOCAL_DOWNLOAD_DIR/$(basename "$gguf_first")" "$LOCAL_MODEL_DIR/"
      echo "[$(timestamp)] First shard saved"
      if [[ "$SKIP_GPG" != "true" ]]; then
        until run_downloader "${QTYPE}" -2 "$LOCAL_DOWNLOAD_DIR" "$(basename "$gguf_first.sig")"; do
          echo "[$(timestamp)] First shard signature download failed; retrying in 10s..."
          sleep 10
        done
        mv -f "$LOCAL_DOWNLOAD_DIR/$(basename "$gguf_first.sig")" "$LOCAL_MODEL_DIR/"
        echo "[$(timestamp)] First shard gpg signature saved"
      fi
    else
      echo "[$(timestamp)] First shard already exists"
      if [[ "$SKIP_GPG" != "true" ]]; then
        if ! [ -f "$LOCAL_MODEL_DIR/$gguf_first.sig" ]; then
          until run_downloader "${QTYPE}" -2 "$LOCAL_DOWNLOAD_DIR" "$(basename "$gguf_first.sig")"; do
            echo "[$(timestamp)] First shard gpg signature download failed; retrying in 10s..."
            sleep 10
          done
          mv -f "$LOCAL_DOWNLOAD_DIR/$(basename "$gguf_first.sig")" "$LOCAL_MODEL_DIR/"
          echo "[$(timestamp)] First shard gpg signature saved"
        else
          echo "[$(timestamp)] First shard gpg signature already exists"
        fi
      fi
    fi
  else
    echo "❌ Error: unable to find previous shards..." >&2
  fi
fi

# ------------- FINAL VERIFICATION & SHARD SEQUENCE --------
echo "[$(timestamp)] Verifying shard sequence completeness"
indices=( $(echo "${_find}" | sed -E "s/.*-([0-9]{5})-of-$total\.gguf$/\1/" | sort) )
last_index=${indices[-1]}
first_index=${indices[0]}
count_expected=$((10#$last_index - 10#$first_index + 1))

if [[ ${#indices[@]} -ne $count_expected ]]; then
  echo "❌ Error - $((count_expected - ${#indices[@]})) missing shard(s) between $first_index and $last_index. Verify recipe or rerun." >&2
  echo "Missing indices:"
  # generate the full expected sequence, then filter out the ones you *have*
  seq -f "%05g" "$((10#$first_index))" "$((10#$last_index))" \
    | grep -Fvx -f <(printf "%s\n" "${indices[@]}")
  exit 1
fi

echo "[$(timestamp)] All shards from $first_index to $last_index are present."

if [[ "$SKIP_GPG" != "true" ]]; then
  if command -v gpg >/dev/null 2>&1; then
    if [ ! -f "$LOCAL_MODEL_DIR/$gguf_first" ]; then
        echo "❌ Error: Shard file '$gguf_first' not found."
        [ -n "$GNUPG_TMPDIR" ] && rm -rf "$GNUPG_TMPDIR"
        exit 5
    fi
    if [ ! -f "$LOCAL_MODEL_DIR/$gguf_first.sig" ]; then
        echo "❌ Error: Signature file '$gguf_first.sig' is missing."
        echo "Hint: To skip GPG verification, re-run this script with the --skip-gpg option."
        [ -n "$GNUPG_TMPDIR" ] && rm -rf "$GNUPG_TMPDIR"
        exit 5
    fi
    if gpg --homedir "$GNUPG_TMPDIR" --no-default-keyring --verify "$LOCAL_MODEL_DIR/$gguf_first.sig" "$LOCAL_MODEL_DIR/$gguf_first" > /dev/null 2>&1; then
        echo "[$(timestamp)] GPG signature verification for '$gguf_first.sig' successful."
    else
        echo "❌ Error: GPG signature verification failed for '$gguf_first.sig'."
        [ -n "$GNUPG_TMPDIR" ] && rm -rf "$GNUPG_TMPDIR"
        exit 4
    fi
    [ -n "$GNUPG_TMPDIR" ] && rm -rf "$GNUPG_TMPDIR"
  else
    echo "⚠️ Warning: 'gpg' command not found. Signature verification skipped." >&2
  fi
fi

echo
echo "✅ Download and verification complete. Enjoy!"
