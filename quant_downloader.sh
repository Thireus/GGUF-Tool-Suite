#!/usr/bin/env bash
#***************************************************************#
#** This script is part of Thireus' GGUF Tool Suite.          **#
#** quant_downloader.sh is a tool that obtains GGUF shards    **#
#** from a recipe file containing tensor regex entries.       **#
#**                                                           **#
#** ********************************************************* **#
#** --------------- Updated: Mar-13-2026 -------------------- **#
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
#** Copyright © 2026 - Thireus.        𝒻ₐᵢₗₑ𝒹 ₜₒ ₐₗₗₒ𝒸ₐₜₑ ᵦᵤ𝒻𝒻ₑᵣ **#
#***************************************************************#
#**PLEASE REFER TO THE README FILE FOR ADDITIONAL INFORMATION!**#
#***************************************************************#

# Exit on error, undefined variable, or pipe failure
set -euo pipefail
# ----- Debugging: verbose ERR trap -----
# Capture the exact command line used to start the script (script name + all args).
# We build this safely so it works on Bash 3.2 and preserves arguments with spaces.
ORIGINAL_INVOCATION="$0"
for __arg in "$@"; do
  # protect any dollar signs so later printing doesn't unexpectedly expand things
  # keep it simple and safe for older bash versions
  escaped="${__arg//\$/\\$}"
  ORIGINAL_INVOCATION="${ORIGINAL_INVOCATION} ${escaped}"
done
# Also save the raw args array if you want to inspect individual params later
ORIGINAL_ARGS=("$@")

error_reporting() {
  # save exit status immediately
  local _err="$?"
  # save the command that caused the trap
  local _cmd="${BASH_COMMAND:-}"
  # save shell flags and pid info
  local _shellflags="$-"
  local _pid=$$
  local _ppid=$PPID
  # avoid set -e in trap doing weird things; make trap body robust
  set +e

  {
    printf '\n===== ERR-TRAP - PLEASE REPORT THE ISSUE HERE: https://github.com/Thireus/GGUF-Tool-Suite/issues =====\n' >&2
    printf 'Timestamp: %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" >&2
    printf 'Exit status: %s\n' "$_err" >&2
    printf 'Failed command: %s\n' "$_cmd" >&2

    # Print the exact invocation that started this script (script + parameters)
    printf 'Script invocation (user command): %s\n' "${ORIGINAL_INVOCATION:-unknown}" >&2
    # show the raw args as an indexed list (useful if args contain spaces)
    if [[ ${#ORIGINAL_ARGS[@]} -gt 0 ]]; then
      printf 'Script args (one per line):\n' >&2
      local __i=0
      for __a in "${ORIGINAL_ARGS[@]}"; do
        printf '  [%d] %s\n' "$__i" "$__a" >&2
        __i=$((__i + 1))
      done
    else
      printf 'Script args: (none)\n' >&2
    fi

    # try to show source file and line number where the error happened
    local src="${BASH_SOURCE[1]:-${BASH_SOURCE[0]:-unknown}}"
    local lineno="${BASH_LINENO[0]:-unknown}"
    printf 'Location: %s:%s\n' "$src" "$lineno" >&2

    # Print function/stack trace (if any). Compatible with bash 3.2.
    printf 'Stack trace (most recent call first):\n' >&2
    local i
    for i in "${!FUNCNAME[@]}"; do
      # FUNCNAME[0] is the current function (error_reporting); skip it if desired
      printf '  %d: %s()  at %s:%s\n' "$i" "${FUNCNAME[$i]:-MAIN}" "${BASH_SOURCE[$i]:-?}" "${BASH_LINENO[$i-1]:-?}" >&2
    done

    printf 'Shell flags: %s\n' "$_shellflags" >&2
    printf 'PID: %s  PPID: %s\n' "$_pid" "$_ppid" >&2

    # show active background jobs (if any)
    printf 'Background jobs (jobs -rp): %s\n' "$(jobs -rp 2>/dev/null || true)" >&2

    # dump some useful shell variables for debugging (avoid unbound vars)
    printf 'BASH_COMMAND (raw): %s\n' "$BASH_COMMAND" >&2
    printf 'Last pipeline status: %s\n' "${PIPESTATUS[*]:-unknown}" >&2

    printf '===== END ERR-TRAP =====\n\n' >&2
  } || true

  # restore errexit behaviour for the rest of the script
  set -e
}

# Install the trap. Use single quotes so expansion occurs at trap time, not now.
trap 'error_reporting' ERR
# ----------------------------------------------------------------

# DEBUG function
if [[ -n "${DEBUG:-}" ]]; then
  DEBUG() { printf "DEBUG: %s\n" "$*" >&2; }
else
  DEBUG() { :; }
fi

# ---------------- SIGNAL HANDLING ----------------
# Graceful shutdown on Ctrl+C / SIGTERM
INT_TIMESTAMP=0

shutdown_on_signal() {
  local rc=130  # conventional exit code for SIGINT
  DEBUG "shutdown_on_signal: received signal, initiating graceful shutdown (exit $rc)" >&2

  # Get running child PIDs started by this shell (jobs -rp)
  local pids
  pids="$(jobs -rp 2>/dev/null || true)"

  if [[ -n "${pids:-}" ]]; then
    DEBUG "shutdown_on_signal: killing child PIDs: $pids" >&2

    # ask nicely first
    kill $pids 2>/dev/null || true
    # give them a second to exit gracefully
    sleep 1

    # force-kill any remaining
    pids="$(jobs -rp 2>/dev/null || true)"
    if [[ -n "${pids:-}" ]]; then
      DEBUG "shutdown_on_signal: force-killing remaining PIDs: $pids" >&2
      kill -KILL $pids 2>/dev/null || true
    fi

    # Reap all children (wait without args waits for all children)
    # Use set +e to avoid aborting if wait returns non-zero.
    set +e
    wait
    set -e
  else
    DEBUG "shutdown_on_signal: no running child jobs found" >&2
  fi

  # Optionally print partial state if you maintain WRAPPER_STATUS/WRAPPER_RAW
  if declare -p WRAPPER_STATUS >/dev/null 2>&1; then
    DEBUG "shutdown_on_signal: current WRAPPER_STATUS snapshot:" >&2
    for k in "${!WRAPPER_STATUS[@]:-}"; do
      printf "DEBUG: idx=%s status=%s raw=%s\n" "$k" "${WRAPPER_STATUS[$k]:-}" "${WRAPPER_RAW[$k]:-}" >&2
    done
  fi

  # If you have any cleanup traps for FIFOs etc, they will run because we exit now.
  if [[ $INT_TIMESTAMP -eq 0 ]]; then
    echo "💀 Termination signal received — forwarding SIGINT to the entire process group and starting graceful shutdown (exit ${rc:-?})." >&2
    echo "   Allowing up to 10 seconds for subprocesses to exit cleanly if possible." >&2
    INT_TIMESTAMP=$(date +%s)
  elif [[ $(( $(date +%s) - INT_TIMESTAMP )) -gt 10 ]]; then
    echo "💀 10 seconds elapsed — forcing termination now: sending SIGKILL to the process group (-$$)." >&2
    kill -s KILL -- -$$ 2>/dev/null || true
  fi
  sleep 1 # Slow down the shutdown_on_signal loop
  kill -s INT -- -$$ 2>/dev/null || true
  exit "$rc"
}

# install the trap for INT and TERM
trap 'shutdown_on_signal' INT TERM
# ------------------------------------------------------------

# Record main script PID so subshells/background workers can cause the entire script to exit.
# This is used by exit_from_subprocess() below to ensure that any fatal condition (previously returning 666)
# terminates the whole script rather than just the function or subshell.
SCRIPT_MAIN_PID=$$

# Fatal exit helper: ensures whole script terminates from functions or subshells.
exit_from_subprocess() {
  local _code="${1:-1}"
  kill -TERM "${SCRIPT_MAIN_PID}" 2>/dev/null || true
  # Exit this process (if parent still running, above kills should stop it).
  exit "$_code"
}

# ----------------- DEFAULTS & INITIALIZATION -----------------
MAX_JOBS=8                   # Default concurrency level
NEW_MAP=true                 # Whether to try obtain the latest map files
FORCE_REDOWNLOAD=false       # Whether to redownload all files (maps, shards, first shard)
VERIFY=false                 # If true, only verify hashes and report errors; skip downloads
VERIFY_READONLY=false        # If true, verify in read-only mode: do not create files in the target dir; use temporary workspace
BASE_DIR="."                 # Base directory for model and download dirs
QTYPE="BF16"                 # Default quantization type used for the first shard and filenames - also used for F32 tensors
QTYPE_SPECIFIED=false        # Whether the user explicitly passed --qtype (new flag)
SKIP_GPG=false               # If true, skip the gpg signature verification of the signed files
SKIP_HASH=false              # If true, skip sha256 hash computations and treat them as valid
SPECIAL_NODE_ID=""           # Optional: number of nodes (see --special-node-id). If non-empty, only shards assigned to the deterministic node are downloaded.
TOTAL_NODES=""               # Optional: Total number of nodes (see --total-nodes).
RM_SKIPPED_SHARDS=false      # Optional: Remove shards that are skipped for this node (not performed in --verify mode).
MODEL_NAME=""                # Will be read from $SCRIPT_DIR/download.conf when --special-node-id is used.
RETRY_ATTEMPTS=3             # Retry config for pathological symlink/network cases (only for read ops)
RETRY_DELAY=10               # Retry config for pathological symlink/network cases (only for read ops)
ARCHIVE_COMPRESS=false       # -z / --z-compress : compress .gguf -> .gguf.zbst after verification (and move compressed files)
ARCHIVE_DECOMPRESS=false     # --z-decompress (aka -zd) : accept .gguf.zbst files and decompress them into .gguf (removing .zbst)
ARCHIVE_COMPRESS_OPT=""      # Deprecated single-string fallback; new preferred format: per-tool options via --z-compress-opt 'tool:opts'
ARCHIVE_DECOMPRESS_OPT=""    # Deprecated single-string fallback; new preferred format: per-tool options via --z-decompress-opt 'tool:opts'
ARCHIVE_NOAUTO=false         # --z-noauto: prevent automatic enabling of -z or -zd based on files present
SKIP_FINAL_MESSAGE=false     # If true, skip the final message

# Enforce that any files produced by the downloader that are .gguf or .gguf.zbst must remain symlinks
SYMLINK_ONLY=false

# Only process individual tensors list (numbers, comma separated)
INDIVIDUAL_TENSORS_ENABLED=false
INDIVIDUAL_TENSORS_RAW=""
declare -A IND_TENSOR_SET=()   # filled after reading maps, keys are decimal integers (1-based chunk ids)

QUANTIZE_NTHREADS=0          # --quantize-nthreads : number of threads passed to llama-quantize (0 == auto / number of CPU threads)
QUANTIZE_ALL_SHARDS=false    # --quantize-all-shards : move all fetch items to quantize queue
MAX_QUANTIZE_JOBS=1          # --max-quantize-jobs (-k) : concurrency for quantization (must be >0 and <= MAX_JOBS)
QUANTIZE_JOBS_SPECIFIED=false

# Keep bf16 shards used for quantization (do not delete after quantize)
QUANTIZE_KEEP_BF16=false
# Allow user to specify BF16 download workspace directory
QUANTIZE_BF16_DIR=""

# Flags for gguf file verification behaviour
SKIP_GGUF_VERIFICATION=false   # --skip-gguf-verification : when true, skip additional gguf_info file verification
STRICT_GGUF_VERIFICATION=false # --strict-gguf-verification : when true, treat warnings as fatal (function returns non-zero)
QUANTIZE_F32_WARN_VERIFICATION=false # When set, will warn about quantized tensors when f32 doesn't match the expected qtype

# Options to auto-select tensors/qtypes for quantization and map computation via regex lists
QUANTIZE_TENSORS_REGEX_ENABLED=false
QUANTIZE_TENSORS_REGEX_RAW=""
declare -a QUANTIZE_TENSORS_REGEX=()

QUANTIZE_QTYPES_REGEX_ENABLED=false
QUANTIZE_QTYPES_REGEX_RAW=""
declare -a QUANTIZE_QTYPES_REGEX=()

COMPUTE_QTYPES_REGEX_MAP_ENABLED=false
COMPUTE_QTYPES_REGEX_MAP_RAW=""
declare -a COMPUTE_QTYPES_REGEX_MAP=()

# Default tools and their magic hex (user can override with --z-custom-tools)
# Example magic hex values: zstd -> 28B52FFD, lbzip2 -> 425A68
CUSTOM_TOOLS=("zstd:28B52FFD" "lbzip2:425A68")  # default custom tools if user doesn't override
# arrays filled after parsing: CUSTOM_TOOL_NAMES[], CUSTOM_TOOL_MAGICS[]
CUSTOM_TOOL_NAMES=()
CUSTOM_TOOL_MAGICS=()

# Raw per-tool compress/decompress option specifications (multiple allowed)
# Each entry expected like 'zstd:-19 -D mydict' or 'lbzip2:-9 -u'
COMPRESS_OPTS_RAW=()     # holds 'tool:opts' strings from --z-compress-opt (or defaults)
DECOMPRESS_OPTS_RAW=()   # holds 'tool:opts' strings from --z-decompress-opt (default empty unless user supplies)

# parsed per-tool opts: COMP_OP_TOOL_NAMES/VALUES, DECOMP_OP_TOOL_NAMES/VALUES
COMP_OP_TOOL_NAMES=()
COMP_OP_TOOL_VALUES=()
DECOMP_OP_TOOL_NAMES=()
DECOMP_OP_TOOL_VALUES=()
# -------------------------------------------------------------------------

# --------------------- COMPUTE MAPS OPTIONS -------------------
COMPUTE_MISSING_MAP=false
COMPUTE_ALL_MAP=false
CONVERT_IGNORE_IMATRIX_RULES=false
WITH_IMATRIX_FILE=""           # user must pass a file to --with-imatrix; only the flag (no file) will be passed to convert_map_qtype.py
SKIP_IMATRIX_HASH=false        # --skip-imatrix-hash : when true, skip imatrix hash verification (must be used with --with-imatrix)
IMATRIX_HASH_COMPUTED=""       # computed hash string (sha256) for the user-supplied imatrix file
CONVERT_NO_FALLBACK=false
declare -a CONVERT_FALLBACK_QUANTS=()
declare -a CONVERT_FALLBACK_QUANTS_FORBIDDEN=()
# -------------------------------------------------------------------------

# --------------------- REQUANTIZE & LLAMA-QUANTIZE ----------------------
REQUANTIZE_QUANTIZEONLY_SHARDS=false
LLAMA_QUANTIZE_BIN=""   # path to llama-quantize binary, required when we need to quantize from bf16
# When quantizing from bf16, we will create a temporary bf16 download workspace under $LOCAL_DOWNLOAD_DIR/bf16
BF16_FIRST_OBTAINED=false    # set to true after first-shard retrieval for quantization is done once
# ------------------------------------------------------------------------

# --------------------- QUANTIZE FAILED DOWNLOAD --------------------------
# If non-empty, numeric N: after N failed download attempts for a given tensor, fallback to quantize-from-bf16 for that tensor only.
QUANTIZE_FAILED_DOWNLOAD=""
# -------------------------------------------------------------------------

# --------------------- VERIFICATION RETRIES -----------------------------
# Maximum number of verification attempts for downloaded shards (will retry download until reached)
MAX_FAILED_VERIFICATION=5     # --max-failed-verification : default attempts
# -------------------------------------------------------------------------

# --------------------- USAGE & ARG PARSING -------------------
usage() {
  # Convert boolean defaults into human-friendly ON/OFF strings for usage output
  local _bool_to_onoff
  _bool_to_onoff() {
    if [[ "$1" == "true" ]]; then
      printf "ON"
    else
      printf "OFF"
    fi
  }

  local SKIP_GGUF_DEF
  SKIP_GGUF_DEF="$(_bool_to_onoff "$SKIP_GGUF_VERIFICATION")"
  local STRICT_GGUF_DEF
  STRICT_GGUF_DEF="$(_bool_to_onoff "$STRICT_GGUF_VERIFICATION")"
  local QUANTIZE_F32_WARN_DEF
  QUANTIZE_F32_WARN_DEF="$(_bool_to_onoff "$QUANTIZE_F32_WARN_VERIFICATION")"

  echo "Usage: $0 [options] <recipe-file>" >&2
  echo "       --no-new-map                      Prevent the script from replacing existing map files" >&2
  echo "       --force-redownload                Force redownload of all shards and maps, ignoring existing files" >&2
  echo "       --verify                          Only verify existing shard hashes; report mismatches; skip downloads" >&2
  echo "       --verify-readonly                 Same as --verify but do not create files in the target directory (use a temporary workspace)." >&2
  echo "       --qtype QUANT                     Set quantization type for the first shard and filenames (default: $QTYPE); use highest qtype of the model!" >&2
  echo "                                         NOTE: When --qtype is explicitly provided and the corresponding" >&2
  echo "                                         tensors.<qtype,,>.map and tensors.<qtype,,>.map.sig files are present (<qtype,,> is automatically lowercased)," >&2
  echo "                                         the script will create these helpful symlinks in the model dir:" >&2
  echo "                                           tensors.map -> tensors.<qtype,,>.map" >&2
  echo "                                           tensors.map.sig -> tensors.<qtype,,>.map.sig" >&2
  echo "                                         This makes it easier to use quantized model repositories locally." >&2
  echo "       --skip-gpg                        Do not verify the gpg signature of the downloaded files" >&2
  echo "       --skip-hash                       Do not compute or verify SHA256 hashes; treat all hashes as valid (useful when shards have been quantized with a different imatrix for example)" >&2
  echo "       --skip-imatrix-hash               When used with --with-imatrix, skip imatrix hash computation and mismatch warning; this is different from --skip-hash because achives a different purpose." >&2
  echo "       --skip-gguf-verification          Skip additional gguf_info.py GGUF (important to verify shards locally quantized) checks (default: $SKIP_GGUF_DEF)" >&2
  echo "       --gguf-info-novenv                Do NOT add the '--venv' argument when invoking gguf_info.py (used for gguf-verification). Use this only if you are sure" >&2
  echo "                                         the gguf_info.py script runs correctly on your current Python environment without the virtual-venv helper." >&2
  echo "                                         If you need to install gguf/requirements, recommended commands:" >&2
  echo "                                           pip install \"gguf @ git+https://github.com/ikawrakow/ik_llama.cpp.git@main#subdirectory=gguf-py\" --force; pip install sentencepiece numpy==1.26.4" >&2
  echo "       --strict-gguf-verification        Treat any mismatches/warnings from gguf file verification as fatal (return non-zero) (default: $STRICT_GGUF_DEF)" >&2
  echo "       --special-node-id N               Only process shards assigned to this node for this model." >&2
  echo "       --total-nodes N                   Total number of nodes." >&2
  echo "       --rm-skipped-shards               Remove shards not assigned to this node if present." >&2
  echo "       --z-noauto                        Disable automatic enabling of -z or -zd based on files present in the model dir." >&2
  echo "       --z-custom-tools TOOL:MAGIC_HEX   Single comma-separated string of tool:magic pairs (example: 'zstd:28B52FFD,lbzip2:425A68,brotli:'), with one," >&2
  echo "                                         allowed not to have any magic which must be set at the end (tool selected upon first magic match in list order). Defaults to 'zstd:28B52FFD,lbzip2:425A68'" >&2
  echo "       --symlink-only                    When present, newly downloaded .gguf/.gguf.zbst files that are symlinks will remain symlinks; the script will fail," >&2
  echo "                                         instead of replacing or creating regular files." >&2
  echo "       --individual-tensors LIST         Comma-separated list of tensor numbers to operate on (example: --individual-tensors 1,2,6)." >&2
  echo "                                         When provided the script will SKIP downloading and verifying any shard not listed." >&2
  echo "                                         Note: numbers must be positive integers within the model's shard count and unique." >&2
  echo "       --no-final-message                Do not print the final 'Download and verification complete' message (useful when running nested instances)." >&2
  echo "       --max-failed-verification N       Set the maximum number of verification attempts for downloaded shards (default: $MAX_FAILED_VERIFICATION)." >&2
  echo "  -j,  --max-jobs N                      Set maximum concurrent downloads (default: $MAX_JOBS)" >&2
  echo "  -d,  --dest DIR                        Base path for model and download dirs (default: $BASE_DIR)" >&2
  echo "  -z,  --z-compress                      Compress verified .gguf files to .gguf.zbst (using the best compression produced by --z-custom-tools tools) before," >&2
  echo "                                         moving out of the download dir" >&2
  echo "       --z-compress-opt OPTS             Single comma-separated string of per-tool compress opts: 'zstd:-19,lbzip2:-9 -u,brotli:-Z -n'. Defaults to ," >&2
  echo "                                         'zstd:-19,lbzip2:-9 -u'." >&2
  echo "  -zd, --z-decompress                    Accept .gguf.zbst files: they will be decompressed to .gguf and .gguf.zbst removed." >&2
  echo "       --z-decompress-opt OPTS           Single comma-separated string of per-tool decompress opts (default empty for all tools)." >&2
  echo "  -h,  --help                            Show this help and exit" >&2
  echo "  <recipe-file>: path to recipe containing USER_REGEX lines (one per tensor; must have .recipe extension)" >&2
  echo "" >&2
  echo "Compute-map options:" >&2
  echo "       --compute-missing-map             When set, if a tensors.<qtype>.map file is missing the script will attempt to compute it from tensors.bf16.map using convert_map_qtype.py." >&2
  echo "                                         Computed maps are not gpg-checked and their qtypes will be annotated in the produced recipe with a leading '!'." >&2
  echo "       --compute-all-map                 When set instead of --compute-missing-map, produce all non-bf16 map files via convert_map_qtype.py (mutually exclusive)." >&2
  echo "       --compute-qtypes-regex-map REG1,REG2,...   Instead of downloading map files for matching qtypes, compute them locally via convert_map_qtype.py." >&2
  echo "       --ignore-imatrix-rules            (forwarded to convert_map_qtype.py) Ignore importance-matrix related checks." >&2
  echo "       --with-imatrix IMATRIX_FILE       (forwarded to convert_map_qtype.py) Indicate that an importance matrix file exists. The file must exist on disk." >&2
  echo "       --skip-imatrix-hash               When used with --with-imatrix, skip imatrix hash computation and mismatch warning; this is different from --skip-hash because achives a different purpose." >&2
  echo "       --fallback-quants Q1,Q2,...       (forwarded to convert_map_qtype.py) Comma or space-separated list of qtypes to whitelist for fallback." >&2
  echo "       --fallback-quants-forbidden REGEX1,REGEX2,...  (forwarded to convert_map_qtype.py) Comma or space-separated list of regex patterns matching forbidden fallback qtypes." >&2
  echo "       --no-fallback                     (forwarded to convert_map_qtype.py) Disable fallback behaviour." >&2
  echo "" >&2
  echo "Quantization-from-bf16 options:" >&2
  echo "       --requantize-quantizeonly-shards  Re-quantize computed-map shards or shards marked to be quantized even if they already exist locally (default: do not replace existing files)." >&2
  echo "       --llama-quantize /path/to/bin     Required when the script needs to quantize missing computed-map shards from bf16. The binary must support --individual-tensors." >&2
  echo "                                         Obtain it at https://github.com/Thireus/ik_llama.cpp/tree/th/quantize_individual_tensors" >&2
  echo "                                         Pre-built releases are at: https://github.com/Thireus/ik_llama.cpp/releases (look for th-quantize_individual_tensors*)." >&2
  echo "       --quantize-failed-download N      Fallback: after N failed download attempts for a given tensor, try to quantize that tensor from bf16 instead (requires --llama-quantize)." >&2
  echo "                                         If not provided the script will retry downloads indefinitely (legacy behaviour)." >&2
  echo "       --quantize-nthreads N             Number of threads to pass to llama-quantize (integer >=0). Default $QUANTIZE_NTHREADS (0 == use all CPU threads)." >&2
  echo "       --quantize-all-shards             Quantize all discovered shards from bf16 instead of downloading them." >&2
  echo "       -k, --max-quantize-jobs N         Maximum concurrent quantize jobs (must be >0 and <= -j / --max-jobs)." >&2
  echo "       --quantize-keep-bf16              When used, do NOT delete bf16 shards after quantization (keeps bf16 workspace files)." >&2
  echo "       --quantize-bf16-directory DIR     Specify custom bf16 workspace directory (default: <dest>/downloaded_shards/bf16)." >&2
  echo "       --quantize-f32-warn-verification  Warn about f32 quantized tensors not matching the expected quantization type set at quantization (default: $QUANTIZE_F32_WARN_DEF)" >&2
  echo "       --quantize-tensors-regex REG1,REG2,...     Move tensors whose tensor-names match any provided regex into the quantize queue." >&2
  echo "       --quantize-qtypes-regex REG1,REG2,...      Move tensors whose target qtype (as determined by the recipe patterns) match any provided regex into the quantize." >&2
  echo "" >&2
  echo "Examples:" >&2
  echo "  # Download all model GGUF shards for this DeepSeek-R1-0528 recipe:" >&2
  echo "    ./quant_downloader.sh DeepSeek-R1-0528.THIREUS-1.9413bpw-4.3624ppl.151GB-GGUF_11GB-GPU_140GB-CPU.569b7f6_bb4f3c8.recipe" >&2
  echo "  # Will download GGUF shards and compress them for archival or storage optimisation purpose - make sure to install the mentioned compression tools 'apt-get install lbzip2 brotli zstd'" >&2
  echo "    ./quant_downloader.sh -z --qtype Q8_0_R8 --z-custom-tools 'zstd:28B52FFD,lbzip2:425A68,brotli:' --z-compress-opt 'zstd:-19,lbzip2:-9 -u,brotli:-Z' -j 18 q8_0_r8.recipe" >&2
  echo "  # Automatically decompresses .zbst GGUF shards - must ensure the list of custom tools matches all the tools that may have been used to create the .zbst present on the host repositories" >&2
  echo "    ./quant_downloader.sh -zd --z-custom-tools 'zstd:28B52FFD,lbzip2:425A68,brotli:' my_custom.recipe" >&2
  echo "  # Verify only individual tensors 2,3,1094:" >&2
  echo "    ./quant_downloader.sh --individual-tensors 2,3,1094 --verify my_custom.recipe" >&2
  echo "  # Compute maps of quants that aren't available for download and locally quantize token_embd.weight to q8_0 using Thireus's special llama-quantize:" >&2
  echo "    ./quant_downloader.sh my_custom.recipe --compute-missing-map --llama-quantize ~/thireus_fork/build/bin/llama-quantize --ignore-imatrix-rules -k 4 --with-imatrix /IMATRIX/imatrix-GLM-4.5-Air-BF16.dat --quantize-tensors-regex \"^token_embd.weight\\\$\" --qtype q8_0" >&2
  exit 1
}

# Support user passing the two-letter short "-zd" (convert to long opt) to be compatible
# with the user's requested syntax. This translation happens before getopt so getopt
# doesn't try to interpret '-zd' as '-z -d'.
_preprocess_args=()
for __a in "$@"; do
  if [[ "$__a" == "-zd" ]]; then
    _preprocess_args+=(--z-decompress)
  else
    _preprocess_args+=("$__a")
  fi
done
set -- "${_preprocess_args[@]:-}"

# Parse arguments (supports GNU long options)
PARSED_OPTS=$(getopt -n "$0" -o j:d:hk:z -l max-jobs:,no-new-map,force-redownload,verify,verify-readonly,qtype:,skip-gpg,skip-hash,dest:,destination:,special-node-id:,total-nodes:,rm-skipped-shards,help,z-compress,z-decompress,z-noauto,z-compress-opt:,z-decompress-opt:,z-custom-tools:,symlink-only,individual-tensors:,compute-missing-map,compute-all-map,ignore-imatrix-rules,with-imatrix:,skip-imatrix-hash,fallback-quants:,fallback-quants-forbidden:,no-fallback,requantize-quantizeonly-shards,llama-quantize:,quantize-failed-download:,quantize-nthreads:,quantize-all-shards,max-quantize-jobs:,no-final-message,quantize-keep-bf16,quantize-bf16-directory:,skip-gguf-verification,strict-gguf-verification,quantize-f32-warn-verification,quantize-tensors-regex:,quantize-qtypes-regex:,compute-qtypes-regex-map:,max-failed-verification:,gguf-info-novenv -- "$@") || usage
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
      VERIFY=true
      shift
      ;;
    --verify-readonly)
      VERIFY_READONLY=true
      VERIFY=true   # treat readonly as a variant of verify mode
      shift
      ;;
    --qtype)
      QTYPE="${2^^}"
      QTYPE_SPECIFIED=true
      shift 2
      ;;
    --skip-gpg)
      SKIP_GPG=true
      shift
      ;;
    --skip-hash)
      SKIP_HASH=true
      shift
      ;;
    --special-node-id)
      SPECIAL_NODE_ID="$2"
      shift 2
      ;;
    --total-nodes)
      TOTAL_NODES="$2"
      shift 2
      ;;
    --rm-skipped-shards)
      RM_SKIPPED_SHARDS=true
      shift
      ;;
    -d|--dest|--destination)
      BASE_DIR="$2"
      shift 2
      ;;
    -h|--help)
      usage
      ;;
    -z|--z-compress)
      ARCHIVE_COMPRESS=true
      shift
      ;;
    --z-decompress)
      ARCHIVE_DECOMPRESS=true
      shift
      ;;
    --z-noauto)
      ARCHIVE_NOAUTO=true
      shift
      ;;
    --z-compress-opt)
      # Expect a single comma-separated string in $2: 'zstd:-19,lbzip2:-9 -u'
      IFS=',' read -r -a __comp_parts <<< "$2"
      for __c in "${__comp_parts[@]}"; do
        __c="${__c#"${__c%%[![:space:]]*}"}"
        __c="${__c%"${__c##*[![:space:]]}"}"
        if [[ -n "$__c" ]]; then
          COMPRESS_OPTS_RAW+=("$__c")
        fi
      done
      shift 2
      ;;
    --z-decompress-opt)
      # Expect a single comma-separated string in $2
      IFS=',' read -r -a __decomp_parts <<< "$2"
      for __c in "${__decomp_parts[@]}"; do
        __c="${__c#"${__c%%[![:space:]]*}"}"
        __c="${__c%"${__c##*[![:space:]]}"}"
        if [[ -n "$__c" ]]; then
          DECOMPRESS_OPTS_RAW+=("$__c")
        fi
      done
      shift 2
      ;;
    --z-custom-tools)
      # Accept only a single comma-separated string of tool:magic entries.
      IFS=',' read -r -a __tools_parts <<< "$2"
      for __t in "${__tools_parts[@]}"; do
        # trim whitespace
        __t="${__t#"${__t%%[![:space:]]*}"}"
        __t="${__t%"${__t##*[![:space:]]}"}"
        if [[ -n "$__t" ]]; then
          CUSTOM_TOOLS+=("$__t")
        fi
      done
      shift 2
      ;;
    --symlink-only)
      SYMLINK_ONLY=true
      shift
      ;;
    --individual-tensors)
      INDIVIDUAL_TENSORS_RAW="$2"
      INDIVIDUAL_TENSORS_ENABLED=true
      shift 2
      ;;
    --compute-missing-map)
      COMPUTE_MISSING_MAP=true
      shift
      ;;
    --compute-all-map)
      COMPUTE_ALL_MAP=true
      shift
      ;;
    --ignore-imatrix-rules)
      CONVERT_IGNORE_IMATRIX_RULES=true
      shift
      ;;
    --with-imatrix)
      WITH_IMATRIX_FILE="$2"
      shift 2
      ;;
    --skip-imatrix-hash)
      SKIP_IMATRIX_HASH=true
      shift
      ;;
    --fallback-quants)
      # Accept either comma-separated or space-separated string; split into array
      IFS=$' \t,' read -r -a __fbparts <<< "$2"
      for __p in "${__fbparts[@]}"; do
        [[ -n "$__p" ]] && CONVERT_FALLBACK_QUANTS+=("$__p")
      done
      shift 2
      ;;
    --fallback-quants-forbidden)
      IFS=$' \t,' read -r -a __fbfparts <<< "$2"
      for __p in "${__fbfparts[@]}"; do
        [[ -n "$__p" ]] && CONVERT_FALLBACK_QUANTS_FORBIDDEN+=("$__p")
      done
      shift 2
      ;;
    --no-fallback)
      CONVERT_NO_FALLBACK=true
      shift
      ;;
    --requantize-quantizeonly-shards)
      REQUANTIZE_QUANTIZEONLY_SHARDS=true
      shift
      ;;
    --llama-quantize)
      LLAMA_QUANTIZE_BIN="$2"
      shift 2
      ;;
    --quantize-failed-download)
      QUANTIZE_FAILED_DOWNLOAD="$2"
      shift 2
      ;;
    --quantize-nthreads)
      QUANTIZE_NTHREADS="$2"
      shift 2
      ;;
    --quantize-all-shards)
      QUANTIZE_ALL_SHARDS=true
      shift
      ;;
    -k|--max-quantize-jobs)
      MAX_QUANTIZE_JOBS="$2"
      QUANTIZE_JOBS_SPECIFIED=true
      shift 2
      ;;
    --no-final-message)
      SKIP_FINAL_MESSAGE=true
      shift
      ;;
    --quantize-keep-bf16)
      QUANTIZE_KEEP_BF16=true
      shift
      ;;
    --quantize-bf16-directory)
      QUANTIZE_BF16_DIR="$2"
      shift 2
      ;;
    --skip-gguf-verification)
      SKIP_GGUF_VERIFICATION=true
      shift
      ;;
    --strict-gguf-verification)
      STRICT_GGUF_VERIFICATION=true
      shift
      ;;
    --quantize-f32-warn-verification)
      QUANTIZE_F32_WARN_VERIFICATION=true
      shift
      ;;
    --quantize-tensors-regex)
      QUANTIZE_TENSORS_REGEX_RAW="$2"
      QUANTIZE_TENSORS_REGEX_ENABLED=true
      # parse comma-separated regex list
      IFS=',' read -r -a __qtparts <<< "$2"
      for __r in "${__qtparts[@]}"; do
        __r="${__r#"${__r%%[![:space:]]*}"}"
        __r="${__r%"${__r##*[![:space:]]}"}"
        [[ -n "$__r" ]] && QUANTIZE_TENSORS_REGEX+=("$__r")
      done
      shift 2
      ;;
    --quantize-qtypes-regex)
      QUANTIZE_QTYPES_REGEX_RAW="$2"
      QUANTIZE_QTYPES_REGEX_ENABLED=true
      # parse comma-separated regex list
      IFS=',' read -r -a __qqparts <<< "$2"
      for __r in "${__qqparts[@]}"; do
        __r="${__r#"${__r%%[![:space:]]*}"}"
        __r="${__r%"${__r##*[![:space:]]}"}"
        [[ -n "$__r" ]] && QUANTIZE_QTYPES_REGEX+=("$__r")
      done
      shift 2
      ;;
    --compute-qtypes-regex-map)
      COMPUTE_QTYPES_REGEX_MAP_RAW="$2"
      COMPUTE_QTYPES_REGEX_MAP_ENABLED=true
      # parse comma-separated regex list
      IFS=',' read -r -a __cmpparts <<< "$2"
      for __r in "${__cmpparts[@]}"; do
        __r="${__r#"${__r%%[![:space:]]*}"}"
        __r="${__r%"${__r##*[![:space:]]}"}"
        [[ -n "$__r" ]] && COMPUTE_QTYPES_REGEX_MAP+=("$__r")
      done
      shift 2
      ;;
    --max-failed-verification)
      MAX_FAILED_VERIFICATION="$2"
      shift 2
      ;;
    --gguf-info-novenv)
      GGUF_INFO_NOVENV=true
      shift
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

# Validate compute flags mutual exclusion
if [[ "$COMPUTE_MISSING_MAP" == true && "$COMPUTE_ALL_MAP" == true ]]; then
  echo "❌ Error: --compute-missing-map and --compute-all-map are mutually exclusive" >&2
  exit 1
fi

# New mutual exclusion: compute-qtypes-regex-map cannot be used with compute-all-map
if [[ "$COMPUTE_QTYPES_REGEX_MAP_ENABLED" == true && "$COMPUTE_ALL_MAP" == true ]]; then
  echo "❌ Error: --compute-qtypes-regex-map is mutually exclusive with --compute-all-map" >&2
  exit 1
fi

# Validate --with-imatrix file if provided
if [[ -n "$WITH_IMATRIX_FILE" ]]; then
  if [[ ! -f "$WITH_IMATRIX_FILE" ]]; then
    echo "❌ Error: --with-imatrix specified but the file '$WITH_IMATRIX_FILE' does not exist." >&2
    exit 1
  fi
fi

# Validate skip-imatrix-hash usage
if [[ "$SKIP_IMATRIX_HASH" == true && -z "$WITH_IMATRIX_FILE" ]]; then
  echo "❌ Error: --skip-imatrix-hash can only be used together with --with-imatrix." >&2
  exit 1
fi

if [[ "$ARCHIVE_COMPRESS" == true && "$ARCHIVE_DECOMPRESS" == true ]]; then
  echo "❌ Error: --z-compress and --z-decompress cannot bed used at the same time." >&2
  exit 3
fi

# Validate QUANTIZE_FAILED_DOWNLOAD param (if provided)
if [[ -n "$QUANTIZE_FAILED_DOWNLOAD" ]]; then
  if ! [[ "$QUANTIZE_FAILED_DOWNLOAD" =~ ^[0-9]+$ && "$QUANTIZE_FAILED_DOWNLOAD" -ge 1 ]]; then
    echo "❌ Error: --quantize-failed-download requires a positive integer argument." >&2
    exit 1
  fi
  # Option can only be used with --llama-quantize option
  if [[ -z "$LLAMA_QUANTIZE_BIN" ]]; then
    echo "❌ Error: --quantize-failed-download can only be used when --llama-quantize is provided." >&2
    exit 1
  fi
fi

# Validate MAX_FAILED_VERIFICATION param (if provided)
if ! [[ "$MAX_FAILED_VERIFICATION" =~ ^[0-9]+$ && "$MAX_FAILED_VERIFICATION" -ge 1 ]]; then
  echo "❌ Error: --max-failed-verification requires a positive integer argument." >&2
  exit 1
fi

# Validate QUANTIZE_NTHREADS if provided (must be integer >= 0)
if ! [[ "$QUANTIZE_NTHREADS" =~ ^[0-9]+$ ]]; then
  echo "❌ Error: --quantize-nthreads requires a non-negative integer argument." >&2
  exit 1
fi

# Validate MAX_QUANTIZE_JOBS if provided (integer >0 and <= MAX_JOBS)
if [[ "$QUANTIZE_JOBS_SPECIFIED" == true ]]; then
  if ! [[ "$MAX_QUANTIZE_JOBS" =~ ^[0-9]+$ && "$MAX_QUANTIZE_JOBS" -ge 1 ]]; then
    echo "❌ Error: --max-quantize-jobs requires a positive integer argument." >&2
    exit 1
  fi
  # Ensure MAX_JOBS is numeric and >=1 for comparison
  if ! [[ "$MAX_JOBS" =~ ^[0-9]+$ && "$MAX_JOBS" -ge 1 ]]; then
    echo "❌ Error: invalid -j/--max-jobs value ($MAX_JOBS) when validating --max-quantize-jobs." >&2
    exit 1
  fi
  if (( MAX_QUANTIZE_JOBS > MAX_JOBS )); then
    echo "❌ Error: --max-quantize-jobs ($MAX_QUANTIZE_JOBS) must be less than or equal to --max-jobs ($MAX_JOBS)." >&2
    exit 1
  fi
fi

# ------------------------------------------------------------------------
# Build per-tool compress/decompress option structures.
# If the user didn't supply per-tool options, set sensible defaults.
# ------------------------------------------------------------------------
# If user did not supply any COMPRESS_OPTS_RAW, use default per-tool compress opts:
if [[ ${#COMPRESS_OPTS_RAW[@]} -eq 0 ]]; then
  COMPRESS_OPTS_RAW=("zstd:-19" "lbzip2:-9 -u")
fi
# DECOMP defaults are empty (user may set)
# Now parse CUSTOM_TOOLS list into names+magics
CUSTOM_TOOL_NAMES=()
CUSTOM_TOOL_MAGICS=()
for entry in "${CUSTOM_TOOLS[@]}"; do
  # allow both "tool:hex" or "tool:hex:extra" (we only use first two fields)
  tool_name="${entry%%:*}"
  rest="${entry#*:}"
  magic="${rest%%:*}"
  # normalize
  tool_name="${tool_name,,}" # lowercase for command lookup convenience

  # deduplicate tool_name entries: keep the first and skip duplicates
  skip=false
  for _existing in "${CUSTOM_TOOL_NAMES[@]}"; do
    if [[ "$_existing" == "$tool_name" ]]; then
      skip=true
      break
    fi
  done
  if [[ "$skip" == true ]]; then
    DEBUG "Skipping duplicate custom tool entry for '$tool_name' from CUSTOM_TOOLS" >&2
    continue
  fi

  CUSTOM_TOOL_NAMES+=("$tool_name")
  CUSTOM_TOOL_MAGICS+=("${magic^^}")
done

# parse COMPRESS_OPTS_RAW into COMP_OP_TOOL_NAMES/VALUES
COMP_OP_TOOL_NAMES=()
COMP_OP_TOOL_VALUES=()
for e in "${COMPRESS_OPTS_RAW[@]}"; do
  tool="${e%%:*}"
  opts="${e#*:}"
  tool="${tool,,}"
  # deduplicate entries: only add if tool name not already present in COMP_OP_TOOL_NAMES
  already=false
  for _n in "${COMP_OP_TOOL_NAMES[@]}"; do
    if [[ "$_n" == "$tool" ]]; then
      already=true
      break
    fi
  done
  if [[ "$already" == false ]]; then
    COMP_OP_TOOL_NAMES+=("$tool")
    COMP_OP_TOOL_VALUES+=("$opts")
  else
    # if duplicate, prefer the first occurrence; skip subsequent duplicates quietly
    DEBUG "Skipping duplicate compress opts for tool: $tool" >&2
  fi
done
# parse DECOMP_OPs
DECOMP_OP_TOOL_NAMES=()
DECOMP_OP_TOOL_VALUES=()
for e in "${DECOMPRESS_OPTS_RAW[@]}"; do
  tool="${e%%:*}"
  opts="${e#*:}"
  tool="${tool,,}"
  already=false
  for _n in "${DECOMP_OP_TOOL_NAMES[@]}"; do
    if [[ "$_n" == "$tool" ]]; then
      already=true
      break
    fi
  done
  if [[ "$already" == false ]]; then
    DECOMP_OP_TOOL_NAMES+=("$tool")
    DECOMP_OP_TOOL_VALUES+=("$opts")
  else
    DEBUG "Skipping duplicate decompress opts for tool: $tool" >&2
  fi
done

# helpers to fetch per-tool options (string)
get_compress_opts_for_tool() {
  local t="$1"
  for i in "${!COMP_OP_TOOL_NAMES[@]}"; do
    if [[ "${COMP_OP_TOOL_NAMES[$i]}" == "$t" ]]; then
      echo "${COMP_OP_TOOL_VALUES[$i]}"
      return 0
    fi
  done
  # not found -> empty
  echo ""
  return 0
}
get_decompress_opts_for_tool() {
  local t="$1"
  for i in "${!DECOMP_OP_TOOL_NAMES[@]}"; do
    if [[ "${DECOMP_OP_TOOL_NAMES[$i]}" == "$t" ]]; then
      echo "${DECOMP_OP_TOOL_VALUES[$i]}"
      return 0
    fi
  done
  # not found -> empty
  echo ""
  return 0
}

# ------------------------------------------------------------------------
# If the compression options include a dictionary (-D <path> or -D/path or -D=path)
# on a given tool, and the user did NOT supply -D in that tool's --z-decompress-opt,
# automatically add the same -D <path> to that tool's decompression options so
# decompression used for GPG verification (or stream-hash) uses the same dictionary.
#
# Warn the user when we add it automatically so they can override with
# --z-decompress-opt if they prefer a different decompression configuration.
# ------------------------------------------------------------------------
for i in "${!COMP_OP_TOOL_NAMES[@]}"; do
  tool="${COMP_OP_TOOL_NAMES[$i]}"
  opts="${COMP_OP_TOOL_VALUES[$i]}"

  # check for -D patterns in opts; support forms: -D <arg>, -D=arg, -Darg
  dict_path=""

  # Split opts into an array of tokens without touching positional params
  # Use read -r -a for bash array splitting on IFS (whitespace)
  IFS=$' \t\n' read -r -a tokens <<< "$opts"

  # Walk tokens looking for -D forms
  for (( j=0; j < ${#tokens[@]}; j++ )); do
    token="${tokens[j]}"
    if [[ "$token" == "-D" ]]; then
      # next token is the argument (if present)
      dict_path="${tokens[j+1]:-}"
      break
    elif [[ "$token" == -D=* ]]; then
      dict_path="${token#-D=}"
      break
    elif [[ "${token:0:2}" == "-D" && "${#token}" -gt 2 ]]; then
      dict_path="${token:2}"
      break
    fi
  done

  if [[ -n "$dict_path" ]]; then
    # See if decompress opts for this tool already contain any -D*
    existing="$(get_decompress_opts_for_tool "$tool")"
    hasD=false

    # Split existing into array as well
    IFS=$' \t\n' read -r -a ex_tokens <<< "$existing"
    for ex in "${ex_tokens[@]}"; do
      case "$ex" in
        -D* ) hasD=true; break ;;
      esac
    done

    if [[ "$hasD" == false ]]; then
      echo "⚠️ Warning: compression options for tool '$tool' include a dictionary (-D ${dict_path}) but --z-decompress-opt did not include -D for this tool. Adding same -D option to decompression settings automatically so verification/decompression will use the same dictionary." >&2
      # append to DECOMP_OP_TOOL_NAMES/VALUES arrays
      DECOMP_OP_TOOL_NAMES+=("$tool")
      # preserve any prior decompress opts for tool as empty + add -D
      DECOMP_OP_TOOL_VALUES+=("-D ${dict_path}")
    fi
  fi
done
# ------------------------------------------------------------------------

# ----------------------- DIRECTORIES -------------------------
# If verify-readonly mode is requested, we must not create files in BASE_DIR.
# Prepare a temporary workspace for maps/signatures and fail marker.
VERIFY_TMPDIR=""
MAP_DIR="."   # default map location is current working directory
if [[ "$VERIFY_READONLY" == true ]]; then
  VERIFY_TMPDIR=$(mktemp -d) || { echo "❌ Error: failed to create temporary workspace for --verify-readonly" >&2; exit 8; }
  MAP_DIR="$VERIFY_TMPDIR"
  FAIL_MARKER="$VERIFY_TMPDIR/.verify_failed"
else
  # Ensure base directory exists (only when not verify-readonly)
  mkdir -p "$BASE_DIR"
  FAIL_MARKER="$BASE_DIR/.verify_failed"
fi

LOCAL_DOWNLOAD_DIR="$BASE_DIR/downloaded_shards"
LOCAL_MODEL_DIR="$BASE_DIR"
# Only create the download dir when not in verify-readonly mode (avoid creating dirs in read-only target)
if [[ "$VERIFY_READONLY" != true ]]; then
  mkdir -p "$LOCAL_DOWNLOAD_DIR"
fi

echo "[Info] Using base directory: $BASE_DIR"
echo "[Info] Download dir: $LOCAL_DOWNLOAD_DIR"
echo "[Info] Model dir: $LOCAL_MODEL_DIR"
echo "[Info] Max jobs: $MAX_JOBS, Max quantize jobs: $MAX_QUANTIZE_JOBS, Obtain new map: $NEW_MAP, Force redownload: $FORCE_REDOWNLOAD, Verify only: $VERIFY, Verify-readonly: $VERIFY_READONLY, Skip signature verification: $SKIP_GPG, Skip hash verification: $SKIP_HASH, nthreads: $QUANTIZE_NTHREADS, quantize-all-shards: $QUANTIZE_ALL_SHARDS"

# Prepare gguf_info invocation preference: by default we add "--venv" to gguf_info.py calls.
# The new option --gguf-info-novenv (if set by the user) disables adding the "--venv" argument.
# Default: not set (add --venv).
GGUF_INFO_NOVENV="${GGUF_INFO_NOVENV:-false}"
# GGUF_INFO_EXTRA will be used when calling _python "$gguf_info_script" with extra args.
if [[ "$GGUF_INFO_NOVENV" == true ]]; then
  GGUF_INFO_EXTRA=()
  echo "⚠️ Warning: --gguf-info-novenv selected: the script will NOT pass '--venv' to gguf_info.py. Ensure gguf_info.py runs correctly on your current Python environment." >&2
  echo "   Recommended (if needed): pip install \"gguf @ git+https://github.com/ikawrakow/ik_llama.cpp.git@main#subdirectory=gguf-py\" --force; pip install sentencepiece numpy==1.26.4" >&2
else
  GGUF_INFO_EXTRA=(--venv)
fi

# Automatic selection of z mode based on files present
#
# 1) If neither -z nor -zd specified and neither was auto-disabled (--z-noauto),
#    and there are more .gguf.zbst files than .gguf files in the model dir, auto-enable -z.
# 2) If neither -z nor -zd specified and neither was auto-disabled, and there are more
#    .gguf files than .gguf.zbst files in the model dir, auto-enable -zd.
#    HOWEVER: when running with --verify/--verify-only, prefer enabling -z instead.
# 3) Users can disable this automatic behavior with --z-noauto.
if [[ "$ARCHIVE_COMPRESS" != true && "$ARCHIVE_DECOMPRESS" != true && "$ARCHIVE_NOAUTO" != true ]]; then
  # Count files robustly using nullglob to avoid find portability differences
  shopt -s nullglob 2>/dev/null || true
  gguf_files=( "$LOCAL_MODEL_DIR"/*-*-of-*.gguf )
  zbst_files=(  "$LOCAL_MODEL_DIR"/*-*-of-*.gguf.zbst )
  shopt -u nullglob 2>/dev/null || true

  # Get array lengths (do NOT use ${#array[@]:-0} — some shells reject that)
  count_gguf=${#gguf_files[@]}
  count_zbst=${#zbst_files[@]}

  # Ensure numeric 0 if arrays are unsupported for some reason
  : ${count_gguf:=0}
  : ${count_zbst:=0}

  if (( count_gguf < count_zbst )); then
    ARCHIVE_COMPRESS=true
    echo "⚠️ Warning: Auto-enabled -z (compression) because there are more .gguf.zbst (${count_zbst}) than .gguf (${count_gguf}) files in '$LOCAL_MODEL_DIR'!" >&2
    echo "   If you prefer to keep automatic selection disabled, re-run with --z-noauto." >&2
  elif (( count_zbst > 0 )) && (( count_zbst < count_gguf )); then
    if [[ "$VERIFY" == true ]]; then
      # In verify mode, prefer verifying compressed streams; choose -z
      ARCHIVE_COMPRESS=true
      echo "⚠️ Warning: Auto-enabled -z (verify .gguf.zbst) because --verify mode detected more .gguf (${count_gguf}) than .gguf.zbst (${count_zbst}) files in '$LOCAL_MODEL_DIR'!" >&2
      echo "   If you prefer to keep automatic selection disabled, re-run with --z-noauto." >&2
    else
      ARCHIVE_DECOMPRESS=true
      echo "⚠️ Warning: Auto-enabled -zd (decompress) because there are more .gguf (${count_gguf}) than .gguf.zbst (${count_zbst}) files in '$LOCAL_MODEL_DIR'!" >&2
      echo "   If you prefer to keep automatic selection disabled, re-run with --z-noauto." >&2
    fi
  fi
fi

if [[ "$ARCHIVE_COMPRESS" == true ]]; then
  echo "[Info] compression enabled: verified .gguf files will be (or will remain) compressed to .gguf.zbst"
  # print per-tool compress opts for visibility
  for t in "${CUSTOM_TOOL_NAMES[@]}"; do
    opts="$(get_compress_opts_for_tool "$t")"
    if [[ -n "$opts" ]]; then
      echo "[Info] compress opts for $t: ${opts}"
    fi
  done
elif [[ "$ARCHIVE_DECOMPRESS" == true ]]; then
  echo "[Info] z-decompress mode enabled: will accept .gguf.zbst files and decompress them into .gguf (removing .zbst)"
fi
if [[ "$ARCHIVE_COMPRESS" == true ]] || [[ "$ARCHIVE_DECOMPRESS" == true ]]; then
  # Echo tools on a single line (comma-separated). Use a subshell so we don't change the caller's IFS.
  if [[ ${#CUSTOM_TOOL_NAMES[@]} -gt 0 ]]; then
    ( IFS=','; printf '[Info] compression tools: %s\n' "${CUSTOM_TOOL_NAMES[*]}" )
  else
    echo "[Info] compression tools: (none)"
  fi
  # Show per-tool decompress opts (if any)
  for t in "${CUSTOM_TOOL_NAMES[@]}"; do
    opts="$(get_decompress_opts_for_tool "$t")"
    if [[ -n "$opts" ]]; then
      echo "[Info] decompress opts for $t: ${opts}"
    fi
  done
fi

# If either compression-related option is used (either user-provided or auto-enabled), ensure compression tools are available
if [[ "$ARCHIVE_COMPRESS" == true || "$ARCHIVE_DECOMPRESS" == true ]]; then
  missing_tools=()
  
  # Check each tool
  for t in "${CUSTOM_TOOL_NAMES[@]}"; do
    if ! command -v "$t" >/dev/null 2>&1; then
      missing_tools+=("$t")
    fi
  done
  
  # If any tools are missing, show detailed error and exit
  if [[ ${#missing_tools[@]} -gt 0 ]]; then
    echo "❌ Error: None of these configured compression tools are available: ${missing_tools[*]}" >&2
    exit 2
  fi
fi

# --------------- Incompatibility checks -------------
# -zd (z-decompress) must NOT be used with --verify (or --verify-readonly), because verify may alter files.
if [[ "$ARCHIVE_DECOMPRESS" == true && "$VERIFY" == true ]]; then
  echo "❌ Error: --z-decompress (-zd) is incompatible with --verify/--verify-readonly. Use --z-compress (-z) instead. Usage of --z-decompress-opt is permitted." >&2
  exit 1
fi
# --------------------------------------------------

# -----------------------------------------------------------------
# Filter out accidental empty-string positional arguments (e.g. when caller passes "")
# This preserves existing behavior for legitimate non-empty args but ignores blank placeholders.
NEW_POS_ARGS=()
for _arg in "$@"; do
  if [[ -n "$_arg" ]]; then
    NEW_POS_ARGS+=("$_arg")
  fi
done
# Reset positional parameters to the filtered list (works even if empty)
set -- "${NEW_POS_ARGS[@]:-}"
# -----------------------------------------------------------------

# Check recipe-file argument
if [[ $# -ne 1 ]]; then
  usage
fi
RECIPE_FILE="$1"
# Enforce .recipe extension
if [[ ! "$RECIPE_FILE" == *.recipe && ! "$RECIPE_FILE" == *.recipe.txt ]]; then
  echo "❌ Error: Recipe file '$RECIPE_FILE' must have a .recipe or .recipe.txt extension." >&2
  exit 1
fi
if [[ ! -f "$RECIPE_FILE" ]]; then
  echo "❌ Error: Recipe file '$RECIPE_FILE' not found." >&2
  exit 1
fi

if [[ -n "$SPECIAL_NODE_ID" ]]; then
  if [[ -n "$TOTAL_NODES" ]]; then
    echo "[Info] special-node-id/(total-nodes - 1) set to: $SPECIAL_NODE_ID/$((TOTAL_NODES-1)) (only shards assigned to this model/node pair will be downloaded)"
  else
    echo "❌ Error: --total-nodes N must be specified when using the --special-node-id option!" >&2
    exit 1
  fi
elif [[ -n "$TOTAL_NODES" ]]; then
  echo "❌ Error: --special-node-id N must be specified when using the --total-nodes option!" >&2
  exit 1
fi

# Ensure we clean up the temporary verify/read-only workspace on exit if created.
cleanup_verify_readonly() {
  # remove tempdir if we created it
  if [[ -n "$VERIFY_TMPDIR" && -d "$VERIFY_TMPDIR" ]]; then
    rm -rf "$VERIFY_TMPDIR" || true
  fi
  # also remove GNUPG_TMPDIR if left behind (best-effort)
  if [[ -n "${GNUPG_TMPDIR:-}" && -d "$GNUPG_TMPDIR" ]]; then
    rm -rf "$GNUPG_TMPDIR" || true
  fi
}
# Install EXIT trap to always attempt cleanup (runs after other EXIT handlers; it's ok)
trap 'cleanup_verify_readonly' EXIT

# ----------------------- SYMLINK TRACKING -------------------------
# Track previous symlink targets for downloaded files so we can detect repeated identical symlink sources.
declare -A PREV_SYMLINK_SOURCE=()

# Helper: get absolute path of symlink target if possible (readlink -f), fall back to readlink raw value.
_resolve_symlink_target() {
  local path="$1"
  local resolved=""
  if command -v readlink >/dev/null 2>&1; then
    # Try readlink -f for canonical absolute path (works on GNU)
    if readlink -f "$path" >/dev/null 2>&1; then
      resolved="$(readlink -f "$path" 2>/dev/null || true)"
    else
      # fallback to raw link content and convert relative to absolute based on symlink location
      local target
      target="$(readlink "$path" 2>/dev/null || true)"
      if [[ -z "$target" ]]; then
        resolved=""
      elif [[ "$target" = /* ]]; then
        resolved="$target"
      else
        # relative symlink -> resolve relative to symlink dir
        resolved="$(cd "$(dirname "$path")" >/dev/null 2>&1 && printf '%s/%s' "$(pwd)" "$target" 2>/dev/null || printf '%s/%s' "$(dirname "$path")" "$target")"
      fi
    fi
  else
    # no readlink? degrade gracefully
    resolved="$(readlink "$path" 2>/dev/null || true)"
  fi
  printf '%s' "$resolved"
}

# -------------------- TIMESTAMP FUNCTION ---------------------
timestamp() { date '+%Y-%m-%d %H:%M:%S'; }

# -------------------- HEADER-READ TOOL SELECTION ---------------------
# choose header-read tool (xxd preferred, fallback to od)
USE_XXD=0
USE_OD=0
if command -v xxd >/dev/null 2>&1; then
  USE_XXD=1
fi
if command -v od >/dev/null 2>&1; then
  USE_OD=1
fi
# Use DEBUG helper to print availability (prints only if DEBUG env var set)
DEBUG "xxd available: $([[ $USE_XXD -eq 1 ]] && echo yes || echo no)"
DEBUG "od available: $([[ $USE_OD -eq 1 ]] && echo yes || echo no)"
# Fail early if neither header-read tool is present
if [[ $USE_XXD -eq 0 && $USE_OD -eq 0 ]]; then
  echo "❌ Error: neither 'xxd' nor 'od' is available on PATH. Please install 'xxd' (commonly provided by vim-common) or ensure 'od' (part of coreutils) is present." >&2
  exit 2
fi

# Helper to read first N bytes of a file and return a lowercase hex string without whitespace.
# Arguments:
#   $1 -> file path
#   $2 -> max bytes to read (integer)
# Returns: printed header string
read_file_header() {
  local file="$1"
  local max_bytes="$2"
  local header=""
  if (( USE_XXD )); then
    # xxd produces a clean hex stream (may include trailing newline)
    header="$(xxd -p -l "$max_bytes" -- "$file" 2>/dev/null | tr -d ' \n' | tr '[:upper:]' '[:lower:]' || true)"
  else
    # od prints bytes with spaces; remove spaces/newlines after converting
    # od arguments: -An (no address), -v (show all), -t x1 (hex bytes), -N bytes (limit)
    header="$(od -An -v -t x1 -N "$max_bytes" -- "$file" 2>/dev/null | tr -d ' \n' | tr '[:upper:]' '[:lower:]' || true)"
  fi
  printf '%s' "$header"
}
# ---------------------------------------------------------------------

# --------------- DETECT & DEFINE SHA256 HELPER ---------------
# If SKIP_HASH is set, don't require any sha256 utility and provide a harmless stub that
# returns an empty string (the calling logic will treat skip mode as "passed").
if [[ "$SKIP_HASH" == true ]]; then
  # Define a stub _sha256sum that always succeeds but returns empty (caller will substitute expected hash where needed).
  _sha256sum() {
    # Do nothing, return success with empty output.
    printf ''
    return 0
  }
fi
# If SKIP_IMATRIX_HASH is set, don't require any sha256 utility and provide a harmless stub that
# returns an empty string (the calling logic will treat skip mode as "passed").
if [[ "$SKIP_IMATRIX_HASH" == true ]]; then
  # Define a stub _sha256sum_imatrix that always succeeds but returns empty (caller will substitute expected hash where needed).
  _sha256sum_imatrix() {
    # Do nothing, return success with empty output.
    printf ''
    return 0
  }
fi
if [[ "$SKIP_HASH" == false ]] || [[ -n "$WITH_IMATRIX_FILE" && "$SKIP_IMATRIX_HASH" == false ]]; then
  # Try to find a suitable sha256 utility
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
    # No hashing tool found; leave sha256tool empty to indicate failure
    sha256tool=()
    args=()
  fi

  # Only define functions if we actually found a tool
  if (( ${#sha256tool[@]} > 0 )); then
    if [[ "$SKIP_HASH" == false ]]; then
      # Both hash functions are needed: define real _sha256sum and a wrapper for _sha256sum_imatrix
      _sha256sum() {
        if (( $# > 0 )); then
          # file-mode: pass filename as $1
          "${sha256tool[@]}" "${args[@]}" "$1" | awk '{print $1}'
        else
          # stdin-mode: read data from pipe
          "${sha256tool[@]}" "${args[@]}" | awk '{print $1}'
        fi
      }
      if [[ "$SKIP_IMATRIX_HASH" == false ]]; then
        # Define the _sha256sum_imatrix
        _sha256sum_imatrix() {
          _sha256sum "$@"   # passes along any arguments given to _sha256sum_imatrix
        }
      fi
    else
      # SKIP_HASH is true: need real _sha256sum_imatrix
      _sha256sum_imatrix() {
        if (( $# > 0 )); then
          # file-mode: pass filename as $1
          "${sha256tool[@]}" "${args[@]}" "$1" | awk '{print $1}'
        else
          # stdin-mode: read data from pipe
          "${sha256tool[@]}" "${args[@]}" | awk '{print $1}'
        fi
      }
    fi
  fi
  # If no tool was found, both functions remain undefined, and the warning below will trigger.
fi

command -v _sha256sum &>/dev/null || echo "⚠️ Warning: _sha256sum command missing - hash cannot be verified!" >&2

# -----------------------------------------------------------------
# Compute SHA256 of --with-imatrix file (if provided) using the detected _sha256sum.
# If computation fails (no tool available, SKIP_HASH enabled, or error), warn and mark as not computed.
# This computation happens after _sha256sum is defined so we can reuse it safely.
# -----------------------------------------------------------------
if [[ -n "$WITH_IMATRIX_FILE" && "$SKIP_IMATRIX_HASH" == false ]]; then
  if command -v _sha256sum_imatrix &>/dev/null; then
    # Try to compute the sha256 using the helper. If SKIP_HASH is true or helper returns empty,
    # we will warn the user that hash-checks will be skipped.
    if ! hash_output="$(_sha256sum_imatrix "$WITH_IMATRIX_FILE" 2>/dev/null || true)"; then
      # _sha256sum_imatrix returned non-zero; treat as failure
      echo "⚠️ Warning: failed to compute sha256 for --with-imatrix file '$WITH_IMATRIX_FILE' (hash command returned error). Imatrix hash checks will be skipped." >&2
    else
      # strip leading/trailing whitespace and any trailing filename from common tools
      hash_output="$(printf '%s' "$hash_output" | awk '{print $1}' || true)"
      if [[ -n "$hash_output" ]]; then
        IMATRIX_HASH_COMPUTED="${hash_output,,}"
        echo "[Info] computed sha256 for --with-imatrix: ${IMATRIX_HASH_COMPUTED}"
      elif [[ "$SKIP_HASH" == true ]]; then
        # empty result (because SKIP_HASH stub is active)
        echo "[Info] SKIP-HASH: the imatrix file '$WITH_IMATRIX_FILE' won't be hashed because --skip-hash is used. Imatrix hash checks will be skipped."
      else
        # empty result
        echo "⚠️ Warning: could not compute sha256 for --with-imatrix file '$WITH_IMATRIX_FILE' (no hash produced). Imatrix hash checks will be skipped." >&2
      fi
    fi
  else
    echo "⚠️ Warning: _sha256sum_imatrix command missing - imatrix hash cannot be computed!" >&2
  fi
fi
# -----------------------------------------------------------------

# --------------- DETECT & DEFINE PYTHON HELPER ---------------
# Provide a helper that detects a usable python3 binary and exposes a small wrapper _python()
# similar in spirit to _sha256sum above. This tries:
#  1) python3
#  2) python3.N for common N values
#  3) python (and checks version)
# If no compatible python3 is available, behavior depends on whether the user requested
# compute-map options: if any compute-map option was used, the script exits with an error.
# Otherwise, we automatically enable SKIP_GGUF_VERIFICATION and warn the user that
# gguf verification is disabled because python3 is not available.
PYTHON_BIN=""
PYTHON_MAJOR=0
PYTHON_MINOR=0

_detect_python_binary() {
  # Try common candidates in order of preference.
  local candidates
  # try a few likely python3.x executables (search descending common versions)
  candidates=(python3.16 python3.15 python3.14 python3.13 python3.12 python3.11 python3.10 python3.9 python3.8 python3 python)
  local cand
  for cand in "${candidates[@]}"; do
    if command -v "$cand" >/dev/null 2>&1; then
      # Verify version is Python 3.x
      local ver
      ver="$("$cand" -c 'import sys; v=sys.version_info; print(f"{v[0]}.{v[1]}")' 2>/dev/null || true)"
      if [[ -n "$ver" ]]; then
        PYTHON_MAJOR="${ver%%.*}"
        PYTHON_MINOR="${ver#*.}"
        PYTHON_BIN="$cand"
        return 0
      fi
    fi
  done

  # No python3.* found. Try 'python' and inspect its version.
  if command -v python >/dev/null 2>&1; then
    local verpy
    verpy="$(python -c 'import sys; v=sys.version_info; print(f"{v[0]}.{v[1]}")' 2>/dev/null || true)"
    if [[ -n "$verpy" ]]; then
      PYTHON_MAJOR="${verpy%%.*}"
      PYTHON_MINOR="${verpy#*.}"
      PYTHON_BIN="python"
      return 0
    fi
  fi

  # No usable python found
  PYTHON_BIN=""
  PYTHON_MAJOR=0
  PYTHON_MINOR=0
  return 1
}

# wrapper function to execute the detected python binary (if set)
_python() {
  if [[ -z "${PYTHON_BIN:-}" ]]; then
    return 1
  fi
  "$PYTHON_BIN" "$@"
}

# Run detection now and react according to user options (compute-map and gguf verification)
if _detect_python_binary >/dev/null 2>&1; then
  # If we found some python, ensure it is 3.x
  if [[ "$PYTHON_MAJOR" -lt 3 ]] || ([[ "$PYTHON_MAJOR" -eq 3 ]] && [[ "$PYTHON_MINOR" -lt 8 ]]); then
    # Found python but it's Python 2.x (or weird). This is unacceptable for compute-map / gguf verification.
    echo "⚠️ Warning: Detected Python interpreter '$PYTHON_BIN' with version ${PYTHON_MAJOR}.${PYTHON_MINOR}. A compatible Python 3.8+ is required for compute-map operations and gguf verification." >&2
    if [[ "$COMPUTE_MISSING_MAP" == true || "$COMPUTE_ALL_MAP" == true || "$COMPUTE_QTYPES_REGEX_MAP_ENABLED" == true ]]; then
      echo "❌ Error: Compute-map options were requested but a Python 3.8+ interpreter could not be found. Please install Python 3 higher than 3.8 and re-run the script." >&2
      exit 1
    else
      # Non compute-map usage: fall back to skipping gguf verification
      if [[ "$SKIP_GGUF_VERIFICATION" != true ]]; then
        echo "⚠️ Note: No compatible Python 3.8+ interpreter available — automatically enabling --skip-gguf-verification." >&2
        SKIP_GGUF_VERIFICATION=true
      fi
    fi
  else
    # Found Python 3.x - good to go.
    echo "[Info] Detected python binary (required for GGUF verification and compute-map operations): ${PYTHON_BIN} (version ${PYTHON_MAJOR}.${PYTHON_MINOR})" >&2
  fi
else
  # No python binary detected at all.
  echo "⚠️ Warning: No compatible Python 3.8+ interpreter found on PATH. A compatible Python 3 interpreter is required for compute-map operations and for gguf verification." >&2
  if [[ "$COMPUTE_MISSING_MAP" == true || "$COMPUTE_ALL_MAP" == true || "$COMPUTE_QTYPES_REGEX_MAP_ENABLED" == true ]]; then
    echo "❌ Error: Compute-map options were requested but no Python 3.8+ interpreter is available. Please install Python 3 higher than 3.8 and re-run the script." >&2
    exit 1
  else
    if [[ "$SKIP_GGUF_VERIFICATION" != true ]]; then
      echo "⚠️ Warning: No Python 3.8+ found — automatically enabling --skip-gguf-verification." >&2
      SKIP_GGUF_VERIFICATION=true
    fi
  fi
fi
# -----------------------------------------------------------------

# ----------------------- SYMLINK/RETRY HELPERS (minimal) -----------------------
# retry_exec: run a command repeatedly (exec mode). Usage: retry_exec cmd arg1 arg2 ...
retry_exec() {
  local attempt=1
  while :; do
    if "$@"; then
      return 0
    fi
    if [[ $attempt -ge $RETRY_ATTEMPTS ]]; then
      return 1
    fi
    echo "⚠️ Warning: command failed (attempt $attempt/${RETRY_ATTEMPTS}): $*" >&2
    attempt=$((attempt + 1))
    sleep "$RETRY_DELAY"
  done
}

# retry_capture: run a command and capture stdout with retries. Usage: out="$(retry_capture cmd arg1 ...)"
retry_capture() {
  local attempt=1
  local out
  while :; do
    if out="$("$@" 2>/dev/null)"; then
      printf '%s' "$out"
      return 0
    fi
    if [[ $attempt -ge $RETRY_ATTEMPTS ]]; then
      return 1
    fi
    echo "⚠️ Warning: command failed (attempt $attempt/${RETRY_ATTEMPTS}): $*" >&2
    attempt=$((attempt + 1))
    sleep "$RETRY_DELAY"
  done
}

# ensure_path_available: when a path is a symlink pointing to a network mount
# we retry waiting for the *target* to become available. Returns 0 if path (or
# symlink target) becomes available within RETRY_ATTEMPTS; non-zero otherwise.
ensure_path_available() {
  local path="$1"
  local attempt=1
  while :; do
    if [[ -e "$path" ]]; then
      return 0
    fi
    # If it's a symlink, but target not present, wait and retry
    if [[ -L "$path" ]]; then
      # if target becomes available, -e "$path" above will be true; otherwise we retry
      :
    fi
    if [[ $attempt -ge $RETRY_ATTEMPTS ]]; then
      return 1
    fi
    echo "⚠️ Warning: path not available (attempt $attempt/${RETRY_ATTEMPTS}): $path" >&2
    attempt=$((attempt + 1))
    sleep "$RETRY_DELAY"
  done
}

is_symlink() { [[ -L "$1" ]]; }

# safe_file_exists: when checking file existence, if path is a symlink attempt retries
safe_file_exists() {
  local path="$1"
  if is_symlink "$path"; then
    ensure_path_available "$path" || return 1
  fi
  [[ -f "$path" ]]
}

# safe_sha256sum: ensure path available if symlink, then run _sha256sum (with retries if symlink)
safe_sha256sum() {
  local path="$1"
  if [[ "$SKIP_HASH" == true ]]; then
    # Skip computing hash; caller will treat result as valid (return empty string).
    printf ''
    return 0
  fi
  if is_symlink "$path"; then
    if ! ensure_path_available "$path"; then
      return 1
    fi
    retry_capture _sha256sum "$path"
  else
    _sha256sum "$path"
  fi
}

# safe_stream_sha256_from_z: compute sha256 of the decompressed stream of a .gguf.zbst file without creating a .gguf file.
# Robustly capture tool exit status even when piping (treat decode errors as failures).
safe_stream_sha256_from_z() {
  local z="$1"
  if [[ "$SKIP_HASH" == true ]]; then
    # Skip computing stream hash; return empty so callers can decide behaviour (we typically set got=exp when skip enabled)
    printf ''
    return 0
  fi
  if is_symlink "$z"; then
    ensure_path_available "$z" || return 1
  fi
  local out
  local rc=404
  # Temporarily disable errexit so we can capture the pipeline's exit status cleanly.
  set +e

  # Calculate the maximum number of bytes needed
  local max_magic_bytes=0
  for magic in "${CUSTOM_TOOL_MAGICS[@]}"; do
    local magic_byte_count=$(( ${#magic} / 2 ))
    if [ "$magic_byte_count" -gt "$max_magic_bytes" ]; then
      max_magic_bytes="$magic_byte_count"
    fi
  done

  # Read the first N bytes of the file only once
  local file_header=""
  if [ "$max_magic_bytes" -gt 0 ]; then
    # Use the portable helper that chooses xxd or od
    file_header="$(read_file_header "$1" "$max_magic_bytes")"
  fi

  # Run magic-matching tool to produce the output file.
  local tool=''
  for i in "${!CUSTOM_TOOL_NAMES[@]}"; do
    # Check if file starts with magic
    local magic="${CUSTOM_TOOL_MAGICS[$i]}"
    local magic_byte_count=$(( ${#magic} / 2 ))
    
    # Compare with the pre-read header
    if [ "$magic_byte_count" -le "$max_magic_bytes" ] && [ "${file_header:0:${#magic}}" = "$(echo "$magic" | tr '[:upper:]' '[:lower:]')" ]; then
      tool="${CUSTOM_TOOL_NAMES[$i]}"
      local opts="$(get_decompress_opts_for_tool "$tool")"

      # Run pipeline in a subshell with pipefail so any failure (including tool decoding) sets non-zero exit code.
      echo "[$(timestamp)] z-decompress (stream): ${tool} ${opts} -k -d -c -- \"$z\"" >&2
      out=$(
        (
          set -o pipefail
          ${tool} ${opts} -k -d -c -- "$z" 2>/dev/null | _sha256sum
        )
      )
      
      rc=$?
      # Special condition for lbzip2 exit code 4
      [[ "${tool}" == "lbzip2" && $rc -eq 4 ]] && rc=0

      break # Don't proceed further
    fi
  done

  set -e

  if [[ $rc -ne 0 ]]; then
    # Treat tool decode errors (and any other pipeline failures) as corruption.
    return $rc
  fi

  printf '%s' "$out"
  return 0
}

# ------------------ IMPLEMENT CHECKS FOR QUANTIZED GGUF FILES ----------------
# These functions validate that a quantized .gguf (or a streamed decompressed .zbst) actually
# contains tensors whose dtype/shape/elements/bytes match the map expectations. They rely on
# gguf_info.py being present in the script directory. The functions return non-zero on any
# fatal error (missing gguf_info.py, subprocess failure, or when strict verification is enabled
# and warnings were observed). By default strict verification is OFF (see --strict-gguf-verification).

# check_quantized_gguf: run gguf_info.py on a local .gguf file and validate one tensor entry.
# Args:
#   $1 -> shard_file (path to .gguf)
#   $2 -> shard_id  (zero-padded or numeric chunk id)
#   $3 -> qtype     (target qtype; compared against dtype from gguf_info)
check_quantized_gguf() {
  local shard_file="$1"
  local shard_id_raw="$2"
  local qtype="$3"

  local gguf_info_script="$SCRIPT_DIR/gguf_info.py"
  if [[ ! -f "$gguf_info_script" ]]; then
    echo "[Info] gguf_info.py not found in script directory (${gguf_info_script}); cannot verify quantized shard for ${qtype}." >&2
    return 1
  fi

  if is_symlink "$shard_file"; then
    ensure_path_available "$shard_file" || return 1
  fi
  if [[ ! -f "$shard_file" ]]; then
    echo "❌ Error: quantized shard file not found: $shard_file" >&2
    return 1
  fi

  # run gguf_info.py and capture output
  local info_out
  #echo "$(_python "$gguf_info_script" "-v" "--venv" "$shard_file")" >&2
  if ! info_out="$(_python "$gguf_info_script" "${GGUF_INFO_EXTRA[@]}" "$shard_file" 2>/dev/null)"; then
    echo "❌ Error: gguf_info.py failed for '$shard_file'." >&2
    return 1
  fi

  # normalize shard id to decimal (strip leading zeros)
  local shard_id_dec=$((10#${shard_id_raw})) 2>/dev/null || shard_id_dec="$shard_id_raw"

  # find expected tensor name by reverse lookup in SHARD_ID
  local expected_tensor=""
  for key in "${!SHARD_ID[@]}"; do
    if [[ "${SHARD_ID[$key]}" -eq "$shard_id_dec" ]]; then
      expected_tensor="$key"
      break
    fi
  done

  if [[ -z "$expected_tensor" ]]; then
    echo "⚠️ Warning: could not determine expected tensor name for shard id ${shard_id_raw}; skipping detailed tensor-name check." >&2
    # still proceed to dtype check if possible
  fi

  # Now parse gguf_info output to locate the line for expected tensor (case-insensitive)
  local match_line=""
  if [[ -n "$expected_tensor" ]]; then
    # try to match by exact token at line start (case-insensitive)
    match_line="$(printf '%s\n' "$info_out" | awk -v t="$expected_tensor" 'BEGIN{IGNORECASE=1} { if(tolower($1)==tolower(t)) { print; exit } }')"
    if [[ -z "$match_line" ]]; then
      # fallback: try to find any line containing the tensor name substring
      match_line="$(printf '%s\n' "$info_out" | grep -iF "$expected_tensor" | head -n1 || true)"
    fi
  else
    # no expected tensor known: pick the first data line (skip header)
    match_line="$(printf '%s\n' "$info_out" | sed -n '1!p' | head -n1 || true)"
  fi

  if [[ -z "$match_line" ]]; then
    echo "⚠️ Warning: gguf_info.py output did not contain a matching tensor line for shard id ${shard_id_raw}." >&2
    if [[ "$STRICT_GGUF_VERIFICATION" == true ]]; then
      return 1
    else
      return 0
    fi
  fi

  # now split key=val pairs
  local gguf_name
  gguf_name="$(printf '%s' "$match_line" | awk '{print $1}')"
  local gguf_dtype gguf_elements gguf_bytes gguf_shape
  # simple extraction for dtype (case-sensitive, no normalization)
  gguf_dtype="$(printf '%s\n' "$match_line" | grep -oE 'dtype=[A-Za-z0-9_-]+' | head -n1 | cut -d= -f2 || true)"
  gguf_elements="$(printf '%s\n' "$match_line" | grep -oE 'elements=[0-9]+' | head -n1 | cut -d= -f2 || true)"
  gguf_bytes="$(printf '%s\n' "$match_line" | grep -oE 'bytes=[0-9]+' | head -n1 | cut -d= -f2 || true)"
  gguf_shape="$(printf '%s\n' "$match_line" | grep -oE 'shape=\([^)]*\)' | head -n1 | sed -E 's/^shape=//' || true)"

  local warnings=0

  # dtype comparison
  if [[ -n "$gguf_dtype" ]]; then
    if [[ "${gguf_dtype,,}" != "${qtype,,}" ]] && ([[ "$QUANTIZE_F32_WARN_VERIFICATION" == true ]] || [[ "${gguf_dtype,,}" != "f32" ]]); then
      echo "⚠️ Warning: dtype mismatch from gguf_info for shard ${shard_file} (got='${gguf_dtype,,}' expected='${qtype,,}')." >&2
      warnings=$((warnings + 1))
    fi
  else
    echo "⚠️ Warning: could not determine dtype from gguf_info output for shard ${shard_file}." >&2
    warnings=$((warnings + 1))
  fi

  # tensor name comparison
  if [[ -n "$expected_tensor" ]]; then
    if [[ "${gguf_name,,}" != "${expected_tensor,,}" ]]; then
      echo "⚠️ Warning: tensor name mismatch for shard ${shard_file} (gguf_info: '${gguf_name}' expected: '${expected_tensor}')." >&2
      warnings=$((warnings + 1))
    fi
  else
    echo "⚠️ Warning: tensor name is missing for ${shard_file}!"
  fi

  # compare elements/bytes/shape with map if available
  local map_shape map_elements map_bytes map_dtype map_imatrix
  map_shape="$(get_t_shape "$qtype" "${expected_tensor:-}")"
  map_elements="$(get_t_elements "$qtype" "${expected_tensor:-}")"
  map_bytes="$(get_t_bytes "$qtype" "${expected_tensor:-}")"
  map_dtype="$(get_t_dtype "$qtype" "${expected_tensor:-}")"

  # obtain imatrix hash from map (if available)
  # Only perform the check if:
  #  - the user provided --with-imatrix (WITH_IMATRIX_FILE non-empty)
  #  - we successfully computed its sha256
  #  - and the map contains an imatrix hash for this tensor (non-empty)
  # If SKIP_IMATRIX_HASH is set, skip the verification entirely (user chooses to bypass).
  map_imatrix="$(get_t_imatrix "$qtype" "${expected_tensor:-}" || true)"

  if [[ -n "$map_dtype" && -n "$gguf_dtype" ]]; then
    if [[ "$map_dtype" != "$gguf_dtype" ]] && ([[ "$QUANTIZE_F32_WARN_VERIFICATION" == true ]] || [[ "$gguf_dtype" != "f32" ]]); then
      echo "⚠️ Warning: dtype (map vs gguf) mismatch for ${shard_file}: map='${map_dtype}' gguf='${gguf_dtype}'" >&2
      warnings=$((warnings + 1))
    fi
  else
    echo "⚠️ Warning: dtype (map or gguf) is missing for ${shard_file}!" >&2
  fi

  if [[ -n "$map_elements" && -n "$gguf_elements" ]]; then
    if [[ "$map_elements" != "$gguf_elements" ]]; then
      echo "⚠️ Warning: elements mismatch for ${shard_file}: map='${map_elements}' gguf='${gguf_elements}'" >&2
      warnings=$((warnings + 1))
    fi
  else
    echo "⚠️ Warning: elements (map or gguf) is missing for ${shard_file}!" >&2
  fi

  if [[ -n "$map_bytes" && -n "$gguf_bytes" ]]; then
    if [[ "$map_bytes" != "$gguf_bytes" ]]; then
      echo "⚠️ Warning: bytes mismatch for ${shard_file}: map='${map_bytes}' gguf='${gguf_bytes}'" >&2
      warnings=$((warnings + 1))
    fi
  else
    echo "⚠️ Warning: bytes (map or gguf) is missing for ${shard_file}!" >&2
  fi

  if [[ -n "$map_shape" && -n "$gguf_shape" ]]; then
    if [[ "$map_shape" != "$gguf_shape" ]]; then
      echo "⚠️ Warning: shape mismatch for ${shard_file}: map='${map_shape}' gguf='${gguf_shape}'" >&2
      warnings=$((warnings + 1))
    fi
  else
    echo "⚠️ Warning: shape (map or gguf) is missing for ${shard_file}!" >&2
  fi

  # imatrix verification logic
  if [[ -n "$IMATRIX_HASH_COMPUTED" && -n "$map_imatrix" ]]; then
    if [[ "${map_imatrix,,}" != "${IMATRIX_HASH_COMPUTED,,}" && "$SKIP_IMATRIX_HASH" == false ]]; then
      echo "⚠️ Warning: imatrix hash mismatch for tensor '${expected_tensor}' in ${shard_file}: map='${map_imatrix,,}' provided='${IMATRIX_HASH_COMPUTED,,}'" >&2
      warnings=$((warnings + 1))
    fi
  fi

  if (( warnings > 0 )); then
    if [[ "$STRICT_GGUF_VERIFICATION" == true ]]; then
      echo "❌ Strict quantized file verification enabled: failing due to ${warnings} warning(s) for ${shard_file}." >&2
      return 1
    else
      echo "⚠️ ${warnings} warning(s) observed during quantized file verification for ${shard_file}; continuing (non-strict mode)." >&2
      return 0
    fi
  else
    echo "[$(timestamp)] Quantized file verification enabled: no warnings for ${shard_file} - verification successful." >&2
  fi

  return 0
}

# safe_stream_check_quantized_gguf_from_z: stream-decompress a .gguf.zbst and pipe into gguf_info.py (no persistent decompressed file)
# Args:
#   $1 -> z (path to .gguf.zbst)
#   $2 -> shard_id  (zero-padded or numeric)
#   $3 -> qtype
safe_stream_check_quantized_gguf_from_z() {
  local z="$1"
  local shard_id_raw="$2"
  local qtype="$3"

  local gguf_info_script="$SCRIPT_DIR/gguf_info.py"
  if [[ ! -f "$gguf_info_script" ]]; then
    echo "[Info] gguf_info.py not found in script directory (${gguf_info_script}); cannot verify quantized shard for ${qtype}." >&2
    return 1
  fi

  if is_symlink "$z"; then
    ensure_path_available "$z" || return 1
  fi
  if [[ ! -f "$z" ]]; then
    echo "❌ Error: compressed shard file not found: $z" >&2
    return 1
  fi

  local out
  local rc=404
  set +e

  # Calculate maximum header bytes
  local max_magic_bytes=0
  for magic in "${CUSTOM_TOOL_MAGICS[@]}"; do
    local magic_byte_count=$(( ${#magic} / 2 ))
    if [ "$magic_byte_count" -gt "$max_magic_bytes" ]; then
      max_magic_bytes="$magic_byte_count"
    fi
  done

  local file_header=""
  if [ "$max_magic_bytes" -gt 0 ]; then
    # Use the portable helper that chooses xxd or od
    file_header="$(read_file_header "$z" "$max_magic_bytes")"
  fi

  # Find matching tool & stream-decompress into gguf_info.py reading from stdin (use '-' as filename if supported)
  for i in "${!CUSTOM_TOOL_NAMES[@]}"; do
    local magic="${CUSTOM_TOOL_MAGICS[$i]}"
    local magic_byte_count=$(( ${#magic} / 2 ))
    if [ "$magic_byte_count" -le "$max_magic_bytes" ] && [ "${file_header:0:${#magic}}" = "$(echo "$magic" | tr '[:upper:]' '[:lower:]')" ]; then
      local tool="${CUSTOM_TOOL_NAMES[$i]}"
      local opts
      opts="$(get_decompress_opts_for_tool "$tool")"

      # The pipeline: tool -d -c "$z" 2>/dev/null | python3 gguf_info.py -
      echo "[$(timestamp)] z-decompress (stream->gguf_info): ${tool} ${opts} -k -d -c -- \"$z\" | python3 \"$gguf_info_script\" -" >&2
      out=$(
        (
          set -o pipefail
          ${tool} ${opts} -k -d -c -- "$z" 2>/dev/null | _python "$gguf_info_script" "${GGUF_INFO_EXTRA[@]}" - 2>/dev/null
        )
      )
      rc=$?
      # Special condition for lbzip2 exit code 4
      [[ "${tool}" == "lbzip2" && $rc -eq 4 ]] && rc=0
      break
    fi
  done

  set -e

  if [[ $rc -ne 0 ]]; then
    # If streaming to gguf_info failed, try fallback: decompress to a temp file and run gguf_info.py on that file.
    echo "⚠️ Warning: streaming decompression to gguf_info.py failed (rc=${rc}). Trying fallback via temporary file..." >&2
    tmpf="$(mktemp "${TMPDIR:-/tmp}/ggufinfo.XXXXXX.gguf")"
    if ! decompress_archive_to_file "$z" "$tmpf"; then
      rm -f "$tmpf" || true
      echo "❌ Error: fallback decompression failed for $z" >&2
      return 1
    fi
    if ! out="$(_python "$gguf_info_script" "${GGUF_INFO_EXTRA[@]}" "$tmpf" 2>/dev/null)"; then
      rm -f "$tmpf" || true
      echo "❌ Error: gguf_info.py failed on decompressed temp file for $z" >&2
      return 1
    fi
    rm -f "$tmpf" || true
  fi

  # Now reuse check_quantized_gguf logic but using the captured info_out instead of running gguf_info again.
  # For convenience, write out to a temp file and call check_quantized_gguf by creating a small temp file that contains the gguf content:
  # However the check_quantized_gguf expects a filename; we can emulate by creating a temporary file that stores the captured gguf_info output
  # and then implement an inline validation re-using the same logic as check_quantized_gguf.
  # To avoid code duplication, we will create a temporary file containing the gguf_info output and run a minimal parser similar to check_quantized_gguf.

  # reuse logic: we will parse $out similarly to check_quantized_gguf
  local info_out="$out"

  # normalize shard id to decimal
  local shard_id_dec=$((10#${shard_id_raw})) 2>/dev/null || shard_id_dec="$shard_id_raw"

  # find expected tensor name by reverse lookup in SHARD_ID
  local expected_tensor=""
  for key in "${!SHARD_ID[@]}"; do
    if [[ "${SHARD_ID[$key]}" -eq "$shard_id_dec" ]]; then
      expected_tensor="$key"
      break
    fi
  done

  if [[ -z "$expected_tensor" ]]; then
    echo "⚠️ Warning: could not determine expected tensor name for shard id ${shard_id_raw}; skipping detailed tensor-name check." >&2
  fi

  # find matching line
  local match_line=""
  if [[ -n "$expected_tensor" ]]; then
    match_line="$(printf '%s\n' "$info_out" | awk -v t="$expected_tensor" 'BEGIN{IGNORECASE=1} { if(tolower($1)==tolower(t)) { print; exit } }')"
    if [[ -z "$match_line" ]]; then
      match_line="$(printf '%s\n' "$info_out" | grep -iF "$expected_tensor" | head -n1 || true)"
    fi
  else
    match_line="$(printf '%s\n' "$info_out" | sed -n '1!p' | head -n1 || true)"
  fi

  if [[ -z "$match_line" ]]; then
    echo "⚠️ Warning: gguf_info.py (stream) output did not contain a matching tensor line for shard id ${shard_id_raw}." >&2
    if [[ "$STRICT_GGUF_VERIFICATION" == true ]]; then
      return 1
    else
      return 0
    fi
  fi

  # now split key=val pairs
  local gguf_name
  gguf_name="$(printf '%s' "$match_line" | awk '{print $1}')"
  local gguf_dtype gguf_elements gguf_bytes gguf_shape
  # simple extraction for dtype (case-sensitive, no normalization)
  gguf_dtype="$(printf '%s\n' "$match_line" | grep -oE 'dtype=[A-Za-z0-9_-]+' | head -n1 | cut -d= -f2 || true)"
  gguf_elements="$(printf '%s\n' "$match_line" | grep -oE 'elements=[0-9]+' | head -n1 | cut -d= -f2 || true)"
  gguf_bytes="$(printf '%s\n' "$match_line" | grep -oE 'bytes=[0-9]+' | head -n1 | cut -d= -f2 || true)"
  gguf_shape="$(printf '%s\n' "$match_line" | grep -oE 'shape=\([^)]*\)' | head -n1 | sed -E 's/^shape=//' || true)"

  local warnings=0

  # dtype comparison
  if [[ -n "$gguf_dtype" ]]; then
    if [[ "${gguf_dtype,,}" != "${qtype,,}" ]] && ([[ "$QUANTIZE_F32_WARN_VERIFICATION" == true ]] || [[ "${gguf_dtype,,}" != "f32" ]]); then
      echo "⚠️ Warning: dtype mismatch from gguf_info for compressed shard ${z} (got='${gguf_dtype,,}' expected='${qtype,,}')." >&2
      warnings=$((warnings + 1))
    fi
  else
    echo "⚠️ Warning: could not determine dtype from gguf_info (stream) for shard ${z}." >&2
    warnings=$((warnings + 1))
  fi

  # tensor name comparison
  if [[ -n "$expected_tensor" ]]; then
    if [[ "${gguf_name,,}" != "${expected_tensor,,}" ]]; then
      echo "⚠️ Warning: tensor name mismatch for compressed shard ${z} (gguf_info: '${gguf_name}' expected: '${expected_tensor}')." >&2
      warnings=$((warnings + 1))
    fi
  else
    echo "⚠️ Warning: tensor name is missing for ${z}!" >&2
  fi

  # compare to map
  local map_shape map_elements map_bytes map_dtype map_imatrix
  map_shape="$(get_t_shape "$qtype" "${expected_tensor:-}")"
  map_elements="$(get_t_elements "$qtype" "${expected_tensor:-}")"
  map_bytes="$(get_t_bytes "$qtype" "${expected_tensor:-}")"
  map_dtype="$(get_t_dtype "$qtype" "${expected_tensor:-}")"
  
  # obtain imatrix hash from map (if available) and validate against provided --with-imatrix
  map_imatrix="$(get_t_imatrix "$qtype" "${expected_tensor:-}" || true)"

  if [[ -n "$map_dtype" && -n "$gguf_dtype" ]]; then
    if [[ "$map_dtype" != "$gguf_dtype" ]] && ([[ "$QUANTIZE_F32_WARN_VERIFICATION" == true ]] || [[ "$gguf_dtype" != "f32" ]]); then
      echo "⚠️ Warning: dtype (map vs gguf) mismatch for ${z}: map='${map_dtype}' gguf='${gguf_dtype}'" >&2
      warnings=$((warnings + 1))
    fi
  else
    echo "⚠️ Warning: dtype (map or gguf) is missing for ${z}!" >&2
  fi
  if [[ -n "$map_elements" && -n "$gguf_elements" ]]; then
    if [[ "$map_elements" != "$gguf_elements" ]]; then
      echo "⚠️ Warning: elements mismatch for ${z}: map='${map_elements}' gguf='${gguf_elements}'" >&2
      warnings=$((warnings + 1))
    fi
  else
    echo "⚠️ Warning: elements (map or gguf) is missing for ${z}!" >&2
  fi
  if [[ -n "$map_bytes" && -n "$gguf_bytes" ]]; then
    if [[ "$map_bytes" != "$gguf_bytes" ]]; then
      echo "⚠️ Warning: bytes mismatch for ${z}: map='${map_bytes}' gguf='${gguf_bytes}'" >&2
      warnings=$((warnings + 1))
    fi
  else
    echo "⚠️ Warning: bytes (map or gguf) is missing for ${z}!" >&2
  fi
  if [[ -n "$map_shape" && -n "$gguf_shape" ]]; then
    if [[ "$(printf '%s' "$map_shape" | tr -d '[:space:]')" != "$(printf '%s' "$gguf_shape" | tr -d '[:space:]')" ]]; then
      echo "⚠️ Warning: shape mismatch for ${z}: map='${map_shape}' gguf='${gguf_shape}'" >&2
      warnings=$((warnings + 1))
    fi
  else
    echo "⚠️ Warning: shape (map or gguf) is missing for ${z}!" >&2
  fi

  # imatrix verification logic
  if [[ -n "$IMATRIX_HASH_COMPUTED" && -n "$map_imatrix" ]]; then
    if [[ "${map_imatrix,,}" != "${IMATRIX_HASH_COMPUTED,,}" && "$SKIP_IMATRIX_HASH" == false ]]; then
      echo "⚠️ Warning: imatrix mismatch for tensor '${expected_tensor}' in ${z}: map='${map_imatrix,,}' provided='${IMATRIX_HASH_COMPUTED,,}'" >&2
      warnings=$((warnings + 1))
    fi
  fi

  if (( warnings > 0 )); then
    if [[ "$STRICT_GGUF_VERIFICATION" == true ]]; then
      echo "❌ Strict quantized file verification enabled: failing due to ${warnings} warning(s) for ${z}." >&2
      return 1
    else
      echo "⚠️ ${warnings} warning(s) observed during quantized file verification for ${z}; continuing (non-strict mode)." >&2
      return 0
    fi
  else
    echo "[$(timestamp)] Quantized file verification enabled: no warnings for ${z} - verification successful." >&2
  fi

  return 0
}
# --------------------------------------------------------------------------

# decompress_archive_to_file: write decompressed .gguf from a .gguf.zbst to a target file
# Now checks tool exit code and removes partial output on failure.
# Additionally: If the produced file is size 0 it is considered a decompression failure.
decompress_archive_to_file() {
  local z="$1"
  if is_symlink "$z"; then
    ensure_path_available "$z" || return 1
  fi
  local out="$2"

  # If symlink-only policy active: decompressing a symlinked .gguf.zbst would produce a regular file in this location;
  # refuse because we must not replace or create a regular file when the source is a symlink.
  if [[ "$SYMLINK_ONLY" == true && -L "$z" && "$3" != "skip_symlink_force" ]]; then
    echo "❌ Error: Refusing to decompress '$z' because it is a symlink and --symlink-only is enabled. Decompression would create a regular .gguf file. Remove --symlink-only or provide a non-symlink input." >&2
    exit_from_subprocess 15
  fi

  if is_symlink "$z"; then
    ensure_path_available "$z" || return 1
  fi
  local rc=404
  # Temporarily disable errexit to capture lbzip2 exit code.
  set +e
  rm -f "$out" || true

  # Calculate the maximum number of bytes needed
  local max_magic_bytes=0
  for magic in "${CUSTOM_TOOL_MAGICS[@]}"; do
    local magic_byte_count=$(( ${#magic} / 2 ))
    if [ "$magic_byte_count" -gt "$max_magic_bytes" ]; then
      max_magic_bytes="$magic_byte_count"
    fi
  done

  # Read the first N bytes of the file only once
  local file_header=""
  if [ "$max_magic_bytes" -gt 0 ]; then
    # Use the portable helper that chooses xxd or od
    file_header="$(read_file_header "$1" "$max_magic_bytes")"
  fi

  # Run magic-matching tool to produce the output file.
  local tool=''
  for i in "${!CUSTOM_TOOL_NAMES[@]}"; do
    # Check if file starts with magic
    local magic="${CUSTOM_TOOL_MAGICS[$i]}"
    local magic_byte_count=$(( ${#magic} / 2 ))
    if [ "$magic_byte_count" -le "$max_magic_bytes" ] && [ "${file_header:0:${#magic}}" = "$(echo "$magic" | tr '[:upper:]' '[:lower:]')" ]; then
      tool="${CUSTOM_TOOL_NAMES[$i]}"
      local opts="$(get_decompress_opts_for_tool "$tool")"
      echo "[$(timestamp)] z-decompress: ${tool} ${opts} -k -d -c -- \"$z\" > \"$out\"" >&2
      "${tool}" ${opts} -k -d -c -- "$z" > "$out" 2>/dev/null
      rc=$?
      # Special condition for lbzip2 exit code 4
      [[ "${tool}" == "lbzip2" && $rc -eq 4 ]] && rc=0
      break # Don't proceed further
    fi
  done

  # If tool returned non-zero, treat as failure and remove any partial output.
  if [[ $rc -ne 0 ]]; then
    rm -f "$out" || true
    set -e
    return $rc
  fi

  # tool reported success — but ensure the produced file is not empty.
  if [[ ! -s "$out" ]]; then
    # zero-length file -> treat as decompression failure
    rm -f "$out" || true
    set -e
    return 1
  fi

  # Set restrictive permissions on the valid output.
  chmod 444 "$out" || true
  set -e
  return 0
}

# compress_gguf_to_archive: compress a .gguf into multiple compression formats using get_compress_opts_for_tool, only keep the smallest compressed file and name it .gguf.zbst and remove original on success
compress_gguf_to_archive() {
  local gguf="$1"

  # If symlink-only policy is active and the input file is a symlink we must refuse because compression will create a regular .zbst file.
  if [[ "$SYMLINK_ONLY" == true && -L "$gguf" ]]; then
    echo "❌ Error: Refusing to compress '$gguf' because it is a symlink and --symlink-only is enabled. Compression would create a regular .gguf.zbst file and could alter the symlink source directory. Remove --symlink-only or provide a non-symlink input." >&2
    exit_from_subprocess 14
  fi

  local z="${gguf}.zbst"
  # Use the system lbzip2 (we already checked earlier it's installed when needed)
  rm -f "$z" || true
  local prev_z_tool=''
  local final_z_tool=''

  # Temporary directory to record per-tool exit statuses when running in parallel
  local tmpdir
  tmpdir="$(mktemp -d)" || return 1
  trap 'rm -rf "${tmpdir:-}"' RETURN

  # Start all compression tools in the background so they operate simultaneously.
  for i in "${!CUSTOM_TOOL_NAMES[@]}"; do
    local tool="${CUSTOM_TOOL_NAMES[$i]}"
    local z_tool="${gguf}.${tool}"
    local opts
    opts="$(get_compress_opts_for_tool "$tool")"

    rm -f "$z_tool" || true

    (
      echo "[$(timestamp)] z-compress: ${tool} ${opts} -k -c -- \"$gguf\" > \"$z_tool\"" >&2

      # Some tools don't support symlinks
      if [[ -L "$gguf" ]]; then
        local resolved_gguf=$(readlink -f "$gguf")
      else
        local resolved_gguf="$gguf"
      fi

      if "${tool}" ${opts} -k -c -- "$resolved_gguf" > "$z_tool"; then
        local _rc=0
        chmod 444 "$z_tool" || true
      else
        local _rc=$?
        rm -f "$z_tool" || true
      fi

      printf "%s" "${_rc}" > "${tmpdir}/status_${i}"
    ) &
  done

  wait

  # Check statuses
  for i in "${!CUSTOM_TOOL_NAMES[@]}"; do
    local tool="${CUSTOM_TOOL_NAMES[$i]}"
    local rc
    if [[ -f "${tmpdir}/status_${i}" ]]; then
      rc="$(cat "${tmpdir}/status_${i}")"
    else
      rc=127
    fi

    if [[ "${tool}" == "lbzip2" ]]; then
      if [[ "${rc}" -ne 0 && "${rc}" -ne 4 ]]; then
        echo "❌ Error: ${tool} compression failed for '$gguf' (rc=${rc}), please verify the compression parameters!" >&2
        for j in "${!CUSTOM_TOOL_NAMES[@]}"; do
          rm -f "${gguf}.${CUSTOM_TOOL_NAMES[$j]}" || true
        done
        exit_from_subprocess 18
        return "${rc}"
      fi
    else
      if [[ "${rc}" -ne 0 ]]; then
        echo "❌ Error: ${tool} compression failed for '$gguf' (rc=${rc}), please verify the compression parameters!" >&2
        for j in "${!CUSTOM_TOOL_NAMES[@]}"; do
          rm -f "${gguf}.${CUSTOM_TOOL_NAMES[$j]}" || true
        done
        exit_from_subprocess 19
        return "${rc}"
      fi
    fi
  done

  # Find smallest compressed file
  for i in "${!CUSTOM_TOOL_NAMES[@]}"; do
    local tool="${CUSTOM_TOOL_NAMES[$i]}"
    local z_tool="${gguf}.${tool}"
    if [[ -f "$z_tool" ]]; then
      if [[ -n "$prev_z_tool" && -f "$prev_z_tool" ]]; then
        size_prev_z_tool=$(ls -l "$prev_z_tool" | awk '{print $5}')
        size_z_tool=$(ls -l "$z_tool" | awk '{print $5}')
        if (( size_prev_z_tool > size_z_tool )); then
          rm -f "$prev_z_tool" || true
          final_z_tool="$z_tool"
        else
          rm -f "$z_tool" || true
          final_z_tool="$prev_z_tool"
        fi
      else
        final_z_tool="$z_tool"
      fi
      prev_z_tool="$final_z_tool"
    fi
  done

  if [[ -z "$final_z_tool" || ! -f "$final_z_tool" ]]; then
    echo "❌ Error: No compressed files were produced for '$gguf'." >&2
    return 404
  fi

  mv "$final_z_tool" "$z"
  rm -f "$gguf" || true
  return 0
}

# safe_gpg_verify: run gpg --verify with retry semantics when either file is a symlink
safe_gpg_verify() {
  local sigfile="$1"
  local datafile="$2"
  if is_symlink "$sigfile" || is_symlink "$datafile"; then
    if is_symlink "$sigfile"; then
      ensure_path_available "$sigfile" || return 1
    fi
    if is_symlink "$datafile"; then
      ensure_path_available "$datafile" || return 1
    fi
    retry_exec gpg --homedir "$GNUPG_TMPDIR" --no-default-keyring --verify "$sigfile" "$datafile"
  else
    gpg --homedir "$GNUPG_TMPDIR" --no-default-keyring --verify "$sigfile" "$datafile"
  fi
}
# ----------------------------------------------------------------

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
SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
TENSOR_DOWNLOADER="$SCRIPT_DIR/tensor_downloader.sh"
if [[ ! -x "$TENSOR_DOWNLOADER" ]]; then
  echo "❌ Error: tensor_downloader.sh not found or not executable at $TENSOR_DOWNLOADER" >&2
  exit 1
fi

# Wrap run_downloader to implement symlink-aware behavior and the new --symlink-only policy.
run_downloader() {
  # Original behavior: call the downloader and return its exit status.
  set +e
  "$TENSOR_DOWNLOADER" "$@" & CHILD_PID=$!; wait $CHILD_PID; ret=$?; set -e

  # After downloader returns, perform symlink handling for .gguf / .gguf.zbst files.
  # $@ typically: <qtype> <chunk_id> <destdir> <filename>
  local _args=("$@")
  local _n=${#_args[@]}
  if (( _n >= 1 )); then
    local filename="${_args[$((_n-1))]}"
    local destdir=""
    if (( _n >= 2 )); then
      destdir="${_args[$((_n-2))]}"
    fi
    # Only consider files that are .gguf or .gguf.zbst
    if [[ "$filename" == *.gguf || "$filename" == *.gguf.zbst ]]; then
      # Build full path (if destdir empty, assume current)
      local fullpath
      if [[ -n "$destdir" ]]; then
        fullpath="$destdir/$filename"
      else
        fullpath="./$filename"
      fi

      # If the downloader succeeded, but file is a symlink, process policies
      if [[ -L "$fullpath" ]]; then
        # Resolve symlink target (absolute preferred)
        local symlink_target
        symlink_target="$(_resolve_symlink_target "$fullpath")"

        # Detect repeated identical symlink source across subsequent downloader invocations.
        local prev="${PREV_SYMLINK_SOURCE["$fullpath"]:-}"
        if [[ -n "$prev" && -n "$symlink_target" && "$prev" == "$symlink_target" ]]; then
          echo "❌ Error: Re-downloaded '$fullpath' but it is still a symlink pointing to the same source ('$symlink_target'). This indicates the symlink source is missing or corrupted (or cannot be decompressed if enabled); aborting to avoid infinite retries. Investigate the symlink source directory." >&2
          exit_from_subprocess 10
        fi
        # Record this symlink target for future detection.
        PREV_SYMLINK_SOURCE["$fullpath"]="$symlink_target"

        # Enforce symlink-only policy conditions:
        if [[ "$SYMLINK_ONLY" == true ]]; then
          if [[ "$filename" == *.gguf && "$ARCHIVE_COMPRESS" == true ]]; then
            echo "❌ Error: Download created a .gguf symlink at '$fullpath'. --z (compress) is enabled and --symlink-only is set, so we cannot compress this symlink into a regular .gguf.zbst file. Remove --symlink-only or disable -z to proceed." >&2
            exit_from_subprocess 11
          fi
          if [[ "$filename" == *.gguf.zbst && "$ARCHIVE_DECOMPRESS" == true ]]; then
            echo "❌ Error: Download created a .gguf.zbst symlink at '$fullpath'. -zd (decompress) is enabled and --symlink-only is set, so we cannot decompress this symlink into a regular .gguf file in the working dir. Remove --symlink-only or disable -zd to proceed." >&2
            exit_from_subprocess 12
          fi
        fi
      fi

      # If downloader produced a .gguf.zbst but user has not enabled -z or -zd, that is unexpected:
      if [[ -f "$destdir/$filename" || -L "$destdir/$filename" ]]; then
        if [[ "$filename" == *.gguf.zbst && "$ARCHIVE_COMPRESS" != true && "$ARCHIVE_DECOMPRESS" != true ]]; then
          echo "❌ Error: The downloader produced a compressed file '$destdir/$filename' (.gguf.zbst) but you did not enable -z or -zd. Please rerun the script with either --z-compress (-z) or --z-decompress (-zd) to work with compressed files." >&2
          exit_from_subprocess 13
        fi
      fi
    fi
  fi

  return $ret
}

# ------------------------------------------------------------------------
# Track per-qtype whether the remote repository serves .gguf.zbst files.
# If we detect a successful fetch of a .gguf.zbst for a given qtype, we remember it
# so future downloads for that qtype request zbst directly.
# ------------------------------------------------------------------------
declare -A QTYPE_USES_ZBST=()

# Cleanup any stale qtype marker files from previous runs (startup)
if [[ -d "$LOCAL_MODEL_DIR" ]] && [[ "$VERIFY_READONLY" != true ]]; then
  # Use a safe glob delete (nullglob not required here but keep it robust)
  rm -f "$LOCAL_DOWNLOAD_DIR"/.qtype_zbst_* 2>/dev/null || true
fi

qtype_flagfile() { printf '%s/.qtype_zbst_%s' "$LOCAL_DOWNLOAD_DIR" "$1"; }

mark_qtype_zbst() {
  local q="${1^^}"
  QTYPE_USES_ZBST["$q"]=1   # best-effort in current shell
  # also write a marker visible to other processes
  [[ "$VERIFY_READONLY" != true ]] && touch "$(qtype_flagfile "$q")"
}

qtype_uses_zbst() {
  local q="${1^^}"
  if [[ -n "${QTYPE_USES_ZBST[$q]:-}" ]]; then
    return 0
  fi
  if [[ -f "$(qtype_flagfile "$q")" ]]; then
    # populate in-memory copy for speed
    QTYPE_USES_ZBST["$q"]=1
    return 0
  fi
  return 1
}

# Helper: transform a normal (chunk, filename) request to its .zbst equivalent.
# Accepts chunk and filename, returns transformed_chunk and transformed_filename.
_transform_to_zbst_request() {
  local chunk="$1"
  local filename="$2"

  local out_chunk="$chunk"
  local out_filename="$filename"

  # If chunk already starts with + (zbst positive), leave chunk as-is.
  if [[ "${chunk}" == +* ]]; then
    out_chunk="${chunk}"
  else
    # Numeric positive chunk -> prefix with + and zero-pad to 5 digits as tensor_downloader expects.
    if [[ "$chunk" =~ ^[0-9]+$ ]] && (( 10#$chunk >= 1 )); then
      out_chunk="+$(printf "%05d" "$chunk")"
    fi
  fi

  # Transform the filename suffixes
  if [[ "$filename" == *.gguf ]]; then
    out_filename="${filename%.gguf}.gguf.zbst"
  elif [[ "$filename" == *.gguf.zbst ]]; then
    out_filename="$filename"
  fi

  echo "$out_chunk" "$out_filename"
  return 0
}

# run_downloader_shard: wrapper for run_downloader that attempts a .gguf request first (normal behavior).
# If the requested file isn't produced and we are operating in compression/decompression modes,
# it will automatically attempt the corresponding .gguf.zbst request and, on success, remember that
# this qtype serves zbst files so future downloads for that qtype request zbst directly.
run_downloader_shard() {
  local qtype="$1"
  local chunk="$2"
  local destdir="$3"
  local filename="$4"

  local _qtype="${qtype^^}"

  # Determine whether to attempt zbst-first (if we've recorded zbst for this qtype)
  if qtype_uses_zbst "$_qtype"; then
    # Try zbst-first for this qtype
    read -r zbst_chunk zbst_filename < <(_transform_to_zbst_request "$chunk" "$filename")
    # If transform didn't change anything (e.g. filename not .gguf), fall back to original attempt below
    if [[ "$zbst_chunk" != "$chunk" || "$zbst_filename" != "$filename" ]]; then
      set +e
      run_downloader "$qtype" "$zbst_chunk" "$destdir" "$zbst_filename"; ret=$?
      set -e
      if safe_file_exists "$destdir/$zbst_filename"; then
        # record again (idempotent) and return success
        mark_qtype_zbst "$_qtype"
        return 0
      fi
      # if zbst-first failed, fallthrough to try original normal request
    fi
  fi

  # 1) Try the normal/original request first (preserves existing behavior).
  set +e
  run_downloader "$qtype" "$chunk" "$destdir" "$filename"; ret=$?
  set -e

  if safe_file_exists "$destdir/$filename"; then
    return 0
  fi

  # 2) If we are not operating in any z mode, there's nothing to try.
  if [[ "$ARCHIVE_COMPRESS" != true && "$ARCHIVE_DECOMPRESS" != true ]]; then
    return $ret
  fi

  # 3) Try the zbst variant (only meaningful when original request was for shard files / signatures)
  read -r zbst_chunk zbst_filename < <(_transform_to_zbst_request "$chunk" "$filename")

  # If transformation didn't change anything, nothing more to try.
  if [[ "$zbst_chunk" == "$chunk" && "$zbst_filename" == "$filename" ]]; then
    return $ret
  fi

  # Attempt the .gguf.zbst request
  set +e
  run_downloader "$qtype" "$zbst_chunk" "$destdir" "$zbst_filename"; ret2=$?
  set -e

  if safe_file_exists "$destdir/$zbst_filename"; then
    # record that this qtype serves zbst so future requests use zbst-first
    echo "[$(timestamp)] Note: detected that the '${qtype^^}' repository serves .gguf.zbst files; switching to zbst-mode for subsequent downloads of this qtype." >&2
    mark_qtype_zbst "$_qtype"
  fi

  # Neither attempt produced the expected file.
  return $ret2
}
# ------------------------------------------------------------------------

# ------------------ VERIFY GPG READINESS ----------------------

if [[ "$SKIP_GPG" != true ]]; then
  if [ ! -f "$SCRIPT_DIR/trusted-keys.asc" ]; then
    echo "❌ Error: trusted-keys.asc not found in the script directory." >&2
    echo "Hint: Provide trusted-keys.asc in the same directory as this script or use the --skip-gpg option to disable gpg signature verification." >&2
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
      echo "❌ Error: trusted-keys.asc contains missing or invalid GPG public keys." >&2
      echo "Hint: Add valid public keys to this file or re-run with the --skip-gpg option to bypass signature verification." >&2
      [ -n "$GNUPG_TMPDIR" ] && rm -rf "$GNUPG_TMPDIR"
      exit 7
    fi
  else
    echo "⚠️ Warning: 'gpg' command not found. Valid GPG public keys verification skipped." >&2
  fi
fi

# -------------------- HASH & SHARD STORAGE -------------------
declare -A T_HASHES SHARD_ID
# per-tensor skip-hash markers (used when the map contains an all-zero hash for a tensor)
declare -A T_SKIP_HASH=()
set_t_hash() { local key="${1,,}::${2,,}"; T_HASHES["$key"]="$3"; DEBUG "set_t_hash T_HASHES[${key}]=${T_HASHES["$key"]}"; 
  # Detect all-zero hash (64 zeros) and mark this tensor to skip hash verification.
  local _norm="$(printf '%s' "$3" | tr '[:upper:]' '[:lower:]' | tr -d '[:space:]')"
  if [[ "$_norm" =~ ^0{64}$ ]]; then
    T_SKIP_HASH["$key"]=1
    DEBUG "set_t_hash: detected all-zero hash for $key -> marking to skip per-tensor hash verification"
  else
    # ensure unset if previously set
    unset 'T_SKIP_HASH["$key"]' 2>/dev/null || true
  fi
}
get_t_hash() { [[ "${1^^}" == "F32" ]] && local key="${QTYPE,,}::${2,,}" || local key="${1,,}::${2,,}"; echo "${T_HASHES["$key"]}"; DEBUG "get_t_hash T_HASHES[${key}]=${T_HASHES["$key"]}"; }
set_shard_id() { SHARD_ID["${1,,}"]="$2"; }
get_shard_id() { echo "${SHARD_ID["${1,,}"]}"; }

# New associative arrays to store map-provided metadata for each tensor (shape/elements/bytes/dtype/imatrix)
declare -A T_SHAPE T_ELEMENTS T_BYTES T_DTYPE T_IMATRIX
set_t_shape() { local key="${1,,}::${2,,}"; T_SHAPE["$key"]="$3"; DEBUG "set_t_shape T_SHAPE[${key}]=${T_SHAPE["$key"]}"; }
set_t_elements() { local key="${1,,}::${2,,}"; T_ELEMENTS["$key"]="$3"; DEBUG "set_t_elements T_ELEMENTS[${key}]=${T_ELEMENTS["$key"]}"; }
set_t_bytes() { local key="${1,,}::${2,,}"; T_BYTES["$key"]="$3"; DEBUG "set_t_bytes T_BYTES[${key}]=${T_BYTES["$key"]}"; }
set_t_dtype() { local key="${1,,}::${2,,}"; T_DTYPE["$key"]="$3"; DEBUG "set_t_dtype T_DTYPE[${key}]=${T_DTYPE["$key"]}"; }
set_t_imatrix() { local key="${1,,}::${2,,}"; T_IMATRIX["$key"]="$3"; DEBUG "set_t_imatrix T_IMATRIX[${key}]=${T_IMATRIX["$key"]}"; }

get_t_shape() { [[ "${1^^}" == "F32" ]] && local key="${QTYPE,,}::${2,,}" || local key="${1,,}::${2,,}"; echo "${T_SHAPE["$key"]}"; DEBUG "get_t_shape T_SHAPE[${key}]=${T_SHAPE["$key"]}"; }
get_t_elements() { [[ "${1^^}" == "F32" ]] && local key="${QTYPE,,}::${2,,}" || local key="${1,,}::${2,,}"; echo "${T_ELEMENTS["$key"]}"; DEBUG "get_t_elements T_ELEMENTS[${key}]=${T_ELEMENTS["$key"]}"; }
get_t_bytes() { [[ "${1^^}" == "F32" ]] && local key="${QTYPE,,}::${2,,}" || local key="${1,,}::${2,,}"; echo "${T_BYTES["$key"]}"; DEBUG "get_t_bytes T_BYTES[${key}]=${T_BYTES["$key"]}"; }
get_t_dtype() { [[ "${1^^}" == "F32" ]] && local key="${QTYPE,,}::${2,,}" || local key="${1,,}::${2,,}"; echo "${T_DTYPE["$key"]}"; DEBUG "get_t_dtype T_DTYPE[${key}]=${T_DTYPE["$key"]}"; }
get_t_imatrix() { [[ "${1^^}" == "F32" ]] && local key="${QTYPE,,}::${2,,}" || local key="${1,,}::${2,,}"; echo "${T_IMATRIX["$key"]}"; DEBUG "get_t_imatrix T_IMATRIX[${key}]=${T_IMATRIX["$key"]}"; }

# Helper: determine whether hash verification should be skipped for a specific tensor
# Returns success (0) if we should skip, non-zero otherwise.
should_skip_hash_for() {
  if [[ "$SKIP_HASH" == true ]] || ! command -v _sha256sum &>/dev/null; then
    return 0
  fi
  local q="$1"
  local tensor="$2"
  local key
  if [[ "${q^^}" == "F32" ]]; then
    key="${QTYPE,,}::${tensor,,}"
  else
    key="${q,,}::${tensor,,}"
  fi
  if [[ -n "${T_SKIP_HASH[$key]:-}" ]]; then
    return 0
  fi
  return 1
}

# -------- PREPARE QTYPES & PATTERNS --------
declare -a PATTERNS PATTERN_QTYPES
for entry in "${USER_REGEX[@]}"; do
  IFS='=' read -r pat qtype _ <<< "$entry"
  PATTERNS+=("$pat")
  PATTERN_QTYPES+=("$qtype")
done
readarray -t UNIQUE_QTYPES < <(printf "%s\n" "${PATTERN_QTYPES[@]}" | sort -u)

# Ensure QTYPE is present and is the first element (case-insensitive)
if [[ " ${UNIQUE_QTYPES[*]^^} " != *" ${QTYPE^^} "* ]]; then
  # not present -> prepend
  UNIQUE_QTYPES=("${QTYPE}" "${UNIQUE_QTYPES[@]}")
else
  # present, ensure it's first (case-insensitive match; preserve original casing)
  for i in "${!UNIQUE_QTYPES[@]}"; do
    if [[ "${UNIQUE_QTYPES[$i]^^}" == "${QTYPE^^}" ]]; then
      if [[ $i -ne 0 ]]; then
        val="${UNIQUE_QTYPES[$i]}"
        unset 'UNIQUE_QTYPES[i]'
        UNIQUE_QTYPES=("$val" "${UNIQUE_QTYPES[@]}")
      fi
      break
    fi
  done
fi

# ------------------ IMPLEMENT COMPUTE_MAP_FOR_QTYPE FUNCTION ----------------
# This function will attempt to compute tensors.<qtype>.map from tensors.bf16.map using convert_map_qtype.py
# Return: 0 on success, non-zero on failure.
declare -A COMPUTED_QTYPES=()
declare -A MAP_FILE_INFO=()
declare -A FETCHED_MAPS=()

compute_map_for_qtype() {
  local qtype_raw="$1"
  # normalize to lowercase qtype token (map files use lowercase qtype in filenames)
  local qtype="${qtype_raw,,}"

  # nothing to do for bf16
  if [[ "$qtype" == "bf16" ]]; then
    return 1
  fi

  local convert_script="$SCRIPT_DIR/convert_map_qtype.py"
  if [[ ! -f "$convert_script" ]]; then
    echo "[Info] convert_map_qtype.py not found in script directory (${convert_script}); cannot compute map for ${qtype}." >&2
    return 1
  fi

  local bf16_map_path="$MAP_DIR/tensors.bf16.map"
  # ensure bf16 map exists (attempt to fetch it if missing)
  if [[ ! -f "$bf16_map_path" ]]; then
    echo "[Info] bf16 map missing in map dir (${MAP_DIR}); attempting to fetch bf16 via tensor_downloader." >&2
    # Attempt to fetch bf16 map using existing downloader mechanism
    # We use "BF16" as qtype for the downloader (the surrounding code uses similar pattern)
    if ! run_downloader "BF16" 0 "$MAP_DIR" "tensors.bf16.map"; then
      echo "[Info] Failed to fetch bf16 map; cannot compute map for ${qtype}." >&2
      return 1
    else
      # Download the signature
      if [[ "$SKIP_GPG" != true ]]; then
        # Do NOT remove existing signatures in the target model dir when in verify-readonly mode.
        if [[ "$VERIFY_READONLY" != true ]]; then
          rm -f "$LOCAL_MODEL_DIR/tensors.bf16.map.sig" || true
          sync || true
        else
          DEBUG "verify-readonly: skipping removal of existing signature in LOCAL_MODEL_DIR for $bf16_map_path.sig"
        fi
        if ! run_downloader "BF16" -1 "$MAP_DIR" "tensors.bf16.map.sig"; then
            echo "❌ Error: failed to fetch map gpg signature for BF16" >&2
            [ -n "$GNUPG_TMPDIR" ] && rm -rf "$GNUPG_TMPDIR"
            exit 3
        fi
      fi
    fi
    # ensure file present now
    if [[ ! -f "$bf16_map_path" ]]; then
      echo "[Info] bf16 map still missing after fetch attempt; cannot compute map for ${qtype}." >&2
      return 1
    fi
    # verify gpg signature of the bf16 map
    if [[ "$SKIP_GPG" != true ]]; then
      if command -v gpg >/dev/null 2>&1; then
        if [ ! -f "$bf16_map_path.sig" ]; then
            echo "❌ Error: Signature file '$bf16_map_path.sig' is missing." >&2
            echo "Hint: To skip GPG verification, re-run this script with the --skip-gpg option." >&2
            [ -n "$GNUPG_TMPDIR" ] && rm -rf "$GNUPG_TMPDIR"
            exit 5
        fi
        if safe_gpg_verify "$bf16_map_path.sig" "$bf16_map_path" > /dev/null 2>&1; then
            echo "[$(timestamp)] GPG signature verification successful for '$bf16_map_path'."
        else
            echo "❌ Error: GPG signature verification failed for '$bf16_map_path.sig'." >&2
            [ -n "$GNUPG_TMPDIR" ] && rm -rf "$GNUPG_TMPDIR"
            exit 4
        fi
      else
        echo "⚠️ Warning: 'gpg' command not found. Signature verification skipped." >&2
      fi
    fi
  fi

  # Build convert command
  local cmd=(_python "$convert_script" "$bf16_map_path" --qtype "$qtype")
  # append user-provided convert flags
  if [[ "$CONVERT_IGNORE_IMATRIX_RULES" == true ]]; then
    cmd+=(--ignore-imatrix-rules)
  fi
  if [[ -n "$WITH_IMATRIX_FILE" ]]; then
    # user provided a path; per requirement only pass "--with-imatrix" flag (without the file) to the convert script
    cmd+=(--with-imatrix)
  fi
  if [[ "$CONVERT_NO_FALLBACK" == true ]]; then
    cmd+=(--no-fallback)
  fi
  if [[ ${#CONVERT_FALLBACK_QUANTS[@]} -gt 0 ]]; then
    cmd+=(--fallback-quants)
    for fq in "${CONVERT_FALLBACK_QUANTS[@]}"; do
      cmd+=("$fq")
    done
  fi
  if [[ ${#CONVERT_FALLBACK_QUANTS_FORBIDDEN[@]} -gt 0 ]]; then
    cmd+=(--fallback-quants-forbidden)
    for fqf in "${CONVERT_FALLBACK_QUANTS_FORBIDDEN[@]}"; do
      cmd+=("$fqf")
    done
  fi

  echo "[Info] Attempting to compute map for qtype ${qtype} using convert_map_qtype.py..." >&2
  
  local local_map="$MAP_DIR/tensors.${qtype}.map"
  safe_file_exists "$local_map" && rm -f "$local_map" || true

  # Print debug command quoting elements safely
  echo "[Info] Running: ${cmd[@]}" >&2

  # Run convert script; it should write tensors.<qtype>.map into same dir as bf16_map_path (i.e. $MAP_DIR)
  if [[ -n "${DEBUG:-}" ]]; then
    if ! "${cmd[@]}"; then
      echo "[Info] convert_map_qtype.py failed for qtype ${qtype}" >&2
      return 1
    fi
  else
    if ! "${cmd[@]}" >/dev/null 2>&1; then
      echo "[Info] convert_map_qtype.py failed for qtype ${qtype}" >&2
      return 1
    fi
  fi

  # verify output file exists
  if [[ ! -f "$local_map" ]]; then
    echo "[Info] convert_map_qtype.py did not create expected file ${local_map}" >&2
    return 1
  fi

  # Compute and store sha256 and last_line for the newly created map file
  local sha256sum
  if sha256sum="$(_sha256sum "$local_map" 2>/dev/null)"; then
    sha256sum="${sha256sum%%[^0-9a-fA-F]*}"
  else
    sha256sum="ERROR"
  fi

  local last_line=""
  if last_line="$(tail -n1 "$local_map" 2>/dev/null || true)"; then
    if [[ -n "$last_line" ]]; then
      # extract filename part before first ':' and strip .gguf suffix if present
      last_line="$(printf '%s' "$last_line" | awk -F: '{print $1}' | sed -E 's/\.gguf.*$//')"
    else
      last_line=""
    fi
  else
    last_line=""
  fi

  local map_key="tensors.${qtype}.map"
  local model_name=""
  if [[ -n "$last_line" ]]; then
    model_name="${last_line} (computed)"
  else
    model_name="(computed)"
  fi
  # Store as a string triple (qtype::sha256::modelname) for debugging/inspection later
  MAP_FILE_INFO["$map_key"]="${qtype}::${sha256sum}::${model_name}"

  # Mark as fetched and computed to avoid gpg checks later
  FETCHED_MAPS["$qtype"]=1
  COMPUTED_QTYPES["$qtype"]=1

  echo "[Info] Successfully computed map for qtype ${qtype} -> ${local_map}" >&2
  echo "[Info] This computed map will NOT be gpg-checked." >&2

  return 0
}
# --------------------------------------------------------------------------

# ------------------ FETCH MAPS & COLLECT ----------------
declare -a TENSORS_TO_FETCH SHARD_FILENAMES TENSORS_TO_FETCH_FULL SHARD_FILENAMES_FULL
# Keep track of which mapfiles we've already processed (case-insensitive)
declare -A PROCESSED_MAPFILES=()
for _q in "${UNIQUE_QTYPES[@]}"; do
  qtype=${_q^^}
  _qtype=$qtype
  [[ "$qtype" == "F32" ]] && _qtype="${QTYPE}"
  echo "[$(timestamp)] Fetching ${_qtype} tensor map (and gpg signature if enabled) for ${qtype} quants"

  # canonical mapfile name (lowercase) used as key to prevent duplicate processing
  mapfile="tensors.${_qtype,,}.map"
  mapkey="${mapfile,,}"    # normalized key (all-lowercase)
  mappath="$MAP_DIR/$mapfile"
  mapsigpath="$MAP_DIR/$mapfile.sig"

  # If we've already processed this mapfile, skip the rest of the loop.
  if [[ -n "${PROCESSED_MAPFILES[$mapkey]:-}" ]]; then
    echo "[$(timestamp)] Skipping already-processed mapfile: $mapfile"
    continue
  fi

  # Mark it as being processed now (prevents re-entrance if UNIQUE_QTYPES had duplicates)
  PROCESSED_MAPFILES["$mapkey"]=1
  # If compute-all-map is requested, attempt to compute (only for non-bf16) and skip downloading
  if [[ "$COMPUTE_ALL_MAP" == true ]] && [[ "${_qtype,,}" != "bf16" ]]; then
    echo "[$(timestamp)] compute-all-map requested: attempting to compute map for ${_qtype,,} (skip download)"
    if ! compute_map_for_qtype "${_qtype,,}"; then
      echo "❌ Error: failed to compute map for ${_qtype,,} while --compute-all-map was requested." >&2
      exit 1
    fi
  elif [[ "$COMPUTE_QTYPES_REGEX_MAP_ENABLED" == true && "${_qtype,,}" != "bf16" ]]; then
    matched=false
    for __pat in "${COMPUTE_QTYPES_REGEX_MAP[@]}"; do
      # match against lowercase qtype for robustness (user patterns should take this into account)
      if [[ "${_qtype,,}" =~ ${__pat} ]]; then
        matched=true
        break
      fi
    done
    if [[ "$matched" == true ]]; then
      echo "[$(timestamp)] compute-qtypes-regex-map: qtype '${_qtype,,}' matches provided regex -> attempting to compute map (skip download)"
      if ! compute_map_for_qtype "${_qtype,,}"; then
        echo "❌ Error: failed to compute map for ${_qtype,,} using --compute-qtypes-regex-map." >&2
        exit 1
      fi
    fi
  else
    if [[ "$FORCE_REDOWNLOAD" == true ]]; then
      echo "[$(timestamp)] Force redownload: removing existing map $mapfile and $mapfile.sig"
      rm -f "$mappath" || true
      rm -f "$mapsigpath" || true
      sync || true
    fi
    if [[ "$NEW_MAP" == true ]]; then
        if [[ -f "$mappath" ]]; then
            mv -f "$mappath" "$MAP_DIR/${mapfile}.bak"
            if [[ -f "$mapsigpath" ]]; then
              mv -f "$mapsigpath" "$MAP_DIR/${mapfile}.sig.bak"
            else
              rm -f "$MAP_DIR/${mapfile}.sig.bak" || true # Delete the backup because now the backup may not correspond to the $mapfile.bak
            fi
            if ! run_downloader "$_qtype" 0 "$MAP_DIR" "$mapfile"; then
                echo "⚠️ Warning: failed to fetch map for $_qtype. Using existing map file." >&2
                mv -f "$MAP_DIR/${mapfile}.bak" "$mappath"
                if [[ -f "$MAP_DIR/${mapfile}.sig.bak" ]]; then
                  mv -f "$MAP_DIR/${mapfile}.sig.bak" "$mapsigpath"
                fi
            else
                # Success: optionally remove backup or keep it
                rm -f "$MAP_DIR/${mapfile}.bak" || true
                # Download the signature
                if [[ "$SKIP_GPG" != true ]]; then
                  if ! run_downloader "$_qtype" -1 "$MAP_DIR" "$mapfile.sig"; then
                      if [[ -f "$MAP_DIR/${mapfile}.sig.bak" ]]; then
                          echo "⚠️ Warning: failed to fetch map gpg signature for $_qtype. Using existing map gpg signature file." >&2
                          mv -f "$MAP_DIR/${mapfile}.sig.bak" "$mapsigpath"
                      else
                          # If map download succeeded but signature download failed and this qtype was computed, we may skip signature below.
                          echo "❌ Error: failed to fetch map gpg signature for $_qtype and no existing map gpg signature file present!" >&2
                          [ -n "$GNUPG_TMPDIR" ] && rm -rf "$GNUPG_TMPDIR"
                          # If this qtype was computed, skip; otherwise exit
                          if [[ -n "${COMPUTED_QTYPES[${_qtype,,}]:-}" ]]; then
                            echo "[$(timestamp)] Note: map for ${_qtype,,} was computed locally; skipping signature requirement."
                          else
                            exit 3
                          fi
                      fi
                  else
                      # Success: optionally remove backup or keep it
                      rm -f "$MAP_DIR/${mapfile}.sig.bak" || true
                  fi
                fi
            fi
        else
            # $mapfile does not exist; just try downloading
            if ! run_downloader "$_qtype" 0 "$MAP_DIR" "$mapfile"; then
                echo "❌ Error: failed to fetch map for $_qtype" >&2
                # If this qtype is non-bf16 and compute_missing_map is enabled, attempt compute
                if [[ "${_qtype,,}" != "bf16" && "$COMPUTE_MISSING_MAP" == true ]]; then
                  echo "[$(timestamp)] Attempting to compute map for ${_qtype,,} because download failed and --compute-missing-map is set."
                  if ! compute_map_for_qtype "${_qtype,,}"; then
                    echo "❌ Error: failed to compute map for ${_qtype,,}." >&2
                    exit 1
                  fi
                else
                  # If compute flags not provided, instruct user
                  if [[ "${_qtype,,}" != "bf16" ]]; then
                    echo "❌ Error: failed to fetch map for non-bf16 qtype '${_qtype,,}'. To auto-generate non-bf16 maps from tensors.bf16.map run with --compute-missing-map or --compute-all-map." >&2
                    exit 1
                  else
                    exit 1
                  fi
                fi
            else
                # Download the signature
                if [[ "$SKIP_GPG" != true ]]; then
                  # Do NOT remove existing signatures in the target model dir when in verify-readonly mode.
                  if [[ "$VERIFY_READONLY" != true ]]; then
                    rm -f "$LOCAL_MODEL_DIR/$mapfile.sig" || true
                    sync || true
                  else
                    DEBUG "verify-readonly: skipping removal of existing signature in LOCAL_MODEL_DIR for $mapfile.sig"
                  fi
                  if ! run_downloader "$_qtype" -1 "$MAP_DIR" "$mapfile.sig"; then
                      echo "❌ Error: failed to fetch map gpg signature for $_qtype" >&2
                      [ -n "$GNUPG_TMPDIR" ] && rm -rf "$GNUPG_TMPDIR"
                      exit 3
                  fi
                fi
            fi
        fi
    else
        # NEW_MAP is false; just download normally (which will only happen if the file doesn't already exist)
        if ! run_downloader "$_qtype" 0 "$MAP_DIR" "$mapfile"; then
            echo "❌ Error: failed to fetch map for $_qtype" >&2
            # Attempt compute if allowed
            if [[ "${_qtype,,}" != "bf16" && "$COMPUTE_MISSING_MAP" == true ]]; then
              echo "[$(timestamp)] Attempting to compute map for ${_qtype,,} because download failed and --compute-missing-map is set."
              if ! compute_map_for_qtype "${_qtype,,}"; then
                echo "❌ Error: failed to compute map for ${_qtype,,}." >&2
                exit 1
              fi
            else
              if [[ "${_qtype,,}" != "bf16" ]]; then
                echo "❌ Error: failed to fetch map for non-bf16 qtype '${_qtype,,}'. To auto-generate non-bf16 maps from tensors.bf16.map run with --compute-missing-map or --compute-all-map." >&2
                exit 1
              else
                exit 1
              fi
            fi
        fi
        # Download the signature
        if [[ "$SKIP_GPG" != true ]]; then
          if ! run_downloader "$_qtype" -1 "$MAP_DIR" "$mapfile.sig"; then
              echo "❌ Error: failed to fetch map gpg signature for $_qtype" >&2
              [ -n "$GNUPG_TMPDIR" ] && rm -rf "$GNUPG_TMPDIR"
              exit 3
          fi
        fi
    fi
  fi

  # At this point: either map exists in $mappath (downloaded, restored from bak, or computed by compute_map_for_qtype),
  # or we have earlier exited on error. We now decide whether to GPG-verify it or skip if computed.
  if [[ ! -f "$mappath" ]]; then
    # If somehow still missing, error out
    echo "❌ Error: Map file '$mapfile' not found after download/compute attempts." >&2
    exit 1
  fi

  # -------------------------------
  # Detect placeholder/computed map files and re-compute/check them.
  # Rule: if qtype is NOT bf16 and the FIRST LINE of the map file contains the literal 64-zero string
  # '000000...000' (64 zeros), we treat the map as a previously computed placeholder. In that case:
  #  - We will recompute the map using compute_map_for_qtype().
  #  - If VERIFY_READONLY==true OR NEW_MAP==false:
  #     * compute the new map into a temporary workspace (VERIFY_TMPDIR when available, otherwise mktemp)
  #     * compare with the existing map using cmp -s; if they differ -> fail (exit like a gpg failure)
  #     * if they are identical -> continue (do not replace when verify-readonly; may replace when NEW_MAP==false for idempotency)
  #  - Else (normal mode & NEW_MAP==true): compute_map_for_qtype will write directly into MAP_DIR and the newly
  #    produced map will replace the old one automatically.
  # This preserves existing script comments and behaviour while ensuring computed maps are fresh and validated.
  # -------------------------------
  # Only perform detection for non-bf16 maps
  if [[ -z "${COMPUTED_QTYPES[${_qtype,,}]:-}" ]] && [[ "${_qtype,,}" != "bf16" ]]; then
    # read first line safely
    first_line="$(head -n1 "$mappath" 2>/dev/null || true)"
    if [[ -n "$first_line" ]] && [[ "$first_line" == *"0000000000000000000000000000000000000000000000000000000000000000"* ]]; then
      echo "[$(timestamp)] Detected placeholder/all-zero first-line in map '${mapfile}' -> treating as previously computed map; will recompute to validate/refresh."
      # Decide compute destination
      tmp_compute_dir_created=false
      if [[ "$VERIFY_READONLY" == true ]]; then
        compute_dir="$VERIFY_TMPDIR"
      elif [[ "$NEW_MAP" == false ]]; then
        # Per requirement: --no-new-map should verify existing computed map by recomputing into a temp dir and cmp -s
        compute_dir="$(mktemp -d)" || { echo "❌ Error: failed to create temporary directory for recomputing map." >&2; exit 1; }
        tmp_compute_dir_created=true
      else
        # normal flow: recompute directly in MAP_DIR (will replace existing)
        compute_dir="$MAP_DIR"
      fi

      # Temporarily switch MAP_DIR for compute_map_for_qtype so it writes to desired compute_dir.
      old_map_dir="$MAP_DIR"
      MAP_DIR="$compute_dir"
      if compute_map_for_qtype "${_qtype,,}"; then
        new_map="${MAP_DIR}/tensors.${_qtype,,}.map"
        if [[ ! -f "$new_map" ]]; then
          echo "❌ Error: recompute produced no map file for qtype ${_qtype,,}." >&2
          MAP_DIR="$old_map_dir"
          [[ "$tmp_compute_dir_created" == true ]] && rm -rf "$compute_dir"
          [ -n "$GNUPG_TMPDIR" ] && rm -rf "$GNUPG_TMPDIR"
          exit 1
        fi

        if [[ "$VERIFY_READONLY" == true || "$NEW_MAP" == false ]]; then
          # Compare newly computed map with existing mappath using cmp -s
          if ! cmp -s "$new_map" "$mappath"; then
            echo "❌ Error: recomputed map for '${mapfile}' differs from existing file. Treating as verification failure." >&2
            MAP_DIR="$old_map_dir"
            [[ "$tmp_compute_dir_created" == true ]] && rm -rf "$compute_dir"
            [ -n "$GNUPG_TMPDIR" ] && rm -rf "$GNUPG_TMPDIR"
            # Fail like a GPG signature failure (exit code 4 is used elsewhere for gpg failures)
            exit 4
          else
            echo "[$(timestamp)] Recomputed map for '${mapfile}' matches existing file."
            # If NEW_MAP==false but not verify-readonly, we can optionally replace (identical) the existing file for idempotency.
            if [[ "$NEW_MAP" == false && "$VERIFY_READONLY" != true ]]; then
              # Move (replace) the existing file with the newly computed identical file to maintain freshness.
              mv -f "$new_map" "$mappath"
              echo "[$(timestamp)] Replaced computed map '${mapfile}' with freshly computed identical copy (no-new-map path)."
            fi
          fi
        else
          # Normal non-readonly mode and NEW_MAP==true: the compute_map_for_qtype wrote into MAP_DIR (which we set to compute_dir==MAP_DIR),
          # thus the newly computed map is already in-place. Nothing further required.
          echo "[$(timestamp)] Recomputed map for '${mapfile}' and replaced the previous computed map."
        fi
      else
        echo "❌ Error: failed to recompute map for '${_qtype,,}'." >&2
        MAP_DIR="$old_map_dir"
        [[ "$tmp_compute_dir_created" == true ]] && rm -rf "$compute_dir"
        exit 1
      fi
      # Restore MAP_DIR
      MAP_DIR="$old_map_dir"
      # Cleanup temporary compute dir if we created it
      if [[ "$tmp_compute_dir_created" == true ]]; then
        rm -rf "$compute_dir" || true
      fi
    fi
  fi
  # ------------------------------- end computed-map detection -------------------------------

  # If map was computed, skip gpg verification for this map.
  if [[ "$SKIP_GPG" != true ]]; then
    # Determine canonical qtype key for lookup in COMPUTED_QTYPES
    # _qtype is already set; use lowercase form
    map_qtype_key="${_qtype,,}"
    if [[ -n "${COMPUTED_QTYPES[$map_qtype_key]:-}" ]]; then
      echo "[$(timestamp)] Note: map '${mapfile}' was computed locally; skipping GPG signature verification for it."
    else
      if command -v gpg >/dev/null 2>&1; then
        if [ ! -f "$mapsigpath" ]; then
            echo "❌ Error: Signature file '$mapfile.sig' is missing." >&2
            echo "Hint: To skip GPG verification, re-run this script with the --skip-gpg option." >&2
            [ -n "$GNUPG_TMPDIR" ] && rm -rf "$GNUPG_TMPDIR"
            exit 5
        fi
        if safe_gpg_verify "$mapsigpath" "$mappath" > /dev/null 2>&1; then
            echo "[$(timestamp)] GPG signature verification successful for '$mapfile'."
        else
            echo "❌ Error: GPG signature verification failed for '$mapfile.sig'." >&2
            [ -n "$GNUPG_TMPDIR" ] && rm -rf "$GNUPG_TMPDIR"
            exit 4
        fi
      else
        echo "⚠️ Warning: 'gpg' command not found. Signature verification skipped." >&2
      fi
    fi
  fi

  # If we reach here, parse the map file lines to populate shard lists and hashes
  while IFS=: read -r fname hash tname shape dtype elements bytes imatrix || [[ -n "$fname" ]]; do
    if [[ $fname =~ -([0-9]{5})-of-[0-9]{5}\.gguf$ ]]; then
      shard_id=$((10#${BASH_REMATCH[1]}))
      set_shard_id "$tname" "$shard_id"
      set_t_hash "$qtype" "$tname" "$hash"

      # Attempt to extract additional metadata from the explicit map fields (if present).
      # Map files typically provide shape=...,dtype=...,elements=NNN,bytes=NNN,imatrix=SHA256
      # The read above assigns those colon-separated tokens into variables; we now sanitize them.

      # Normalize/strip leading keys if present (e.g. 'shape=(...)', 'dtype=q8_0', 'elements=123', 'bytes=456', 'imatrix=abc').
      local_dtype="${dtype:-}"
      local_shape="${shape:-}"
      local_elements="${elements:-}"
      local_bytes="${bytes:-}"
      local_imatrix="${imatrix:-}"

      # strip prefix 'dtype=' if present
      if [[ "${local_dtype}" == dtype=* ]]; then
        local_dtype="${local_dtype#dtype=}"
      fi
      # strip prefix 'shape=' if present
      if [[ "${local_shape}" == shape=* ]]; then
        local_shape="${local_shape#shape=}"
      fi
      # strip prefix 'elements=' if present
      if [[ "${local_elements}" == elements=* ]]; then
        local_elements="${local_elements#elements=}"
      fi
      # strip prefix 'bytes=' if present
      if [[ "${local_bytes}" == bytes=* ]]; then
        local_bytes="${local_bytes#bytes=}"
      fi
      # strip prefix 'imatrix=' if present
      if [[ "${local_imatrix}" == imatrix=* ]]; then
        local_imatrix="${local_imatrix#imatrix=}"
      fi

      # store metadata only if non-empty
      if [[ -n "${local_dtype}" ]]; then
        set_t_dtype "$qtype" "$tname" "$local_dtype"
      fi
      if [[ -n "${local_shape}" ]]; then
        set_t_shape "$qtype" "$tname" "$local_shape"
      fi
      if [[ -n "${local_elements}" ]]; then
        set_t_elements "$qtype" "$tname" "$local_elements"
      fi
      if [[ -n "${local_bytes}" ]]; then
        set_t_bytes "$qtype" "$tname" "$local_bytes"
      fi
      if [[ -n "${local_imatrix}" ]]; then
        set_t_imatrix "$qtype" "$tname" "$local_imatrix"
      else
        set_t_imatrix "$qtype" "$tname" "" # Except for imatrix
      fi

      # Filling these lists should only happen once now
      if [[ "$qtype" == "${QTYPE}" ]]; then
        SHARD_FILENAMES+=("$fname")
        TENSORS_TO_FETCH+=("$tname")
      fi
    else
      echo "⚠️ Warning: skipping invalid filename '$fname'" >&2
    fi
  done < "$mappath"
done

# Save original SHARD_FILENAMES and TENSORS_TO_FETCH before any form of filtering (used to construct shard filename with TENSORS_TO_FETCH_FULL[0])
SHARD_FILENAMES_FULL=("${SHARD_FILENAMES[@]}")
TENSORS_TO_FETCH_FULL=("${TENSORS_TO_FETCH[@]}")

# If the user explicitly requested a --qtype, create helpful symlinks in the model dir
# so callers that expect tensors.map / tensors.map.sig (generic names) can work.
# Only create these symlinks when not in --verify-readonly mode (we shouldn't write into
# the model dir when the user asked for readonly verification).
if [[ "$QTYPE_SPECIFIED" == true && "$VERIFY_READONLY" != true ]]; then
  _qtype=${QTYPE,,}
  src_map="$MAP_DIR/tensors.$_qtype.map"
  if safe_file_exists "$src_map"; then
    safe_file_exists "$LOCAL_MODEL_DIR/tensors.$_qtype.map" || cp -p "$src_map" "$LOCAL_MODEL_DIR/tensors.$_qtype.map"
    echo "[$(timestamp)] Ensure tensors.map symlink in model dir pointing to tensors.$_qtype.map"
    rm -f "$LOCAL_MODEL_DIR/tensors.map" || true
    ln -sfn "$src_map" "$LOCAL_MODEL_DIR/tensors.map" || true
    src_sig="$MAP_DIR/tensors.$_qtype.map.sig"
    if safe_file_exists "$src_sig"; then
      safe_file_exists "$LOCAL_MODEL_DIR/tensors.$_qtype.map.sig" || cp -p "$src_sig" "$LOCAL_MODEL_DIR/tensors.$_qtype.map.sig"
      echo "[$(timestamp)] Ensure tensors.map.sig symlink in model dir pointing to tensors.$_qtype.map.sig"
      rm -f "$LOCAL_MODEL_DIR/tensors.map.sig" || true
      ln -sfn "$src_sig" "$LOCAL_MODEL_DIR/tensors.map.sig" || true
    fi
  fi
fi

# ------------------ IMPLEMENT INDIVIDUAL TENSORS VALIDATION ----------------
# If the user specified --individual-tensors, parse and validate the list now that
# SHARD_FILENAMES (hence shard count) is known.
if [[ "$INDIVIDUAL_TENSORS_ENABLED" == true ]]; then
  num_shards=${#SHARD_FILENAMES[@]}
  num_shards=$((num_shards + 1))
  if (( num_shards == 1 )); then
    echo "❌ Error: cannot validate --individual-tensors because no shards were discovered in the maps." >&2
    exit 1
  fi

  # Parse comma-separated list into tokens
  IFS=',' read -r -a __ind_parts <<< "$INDIVIDUAL_TENSORS_RAW"

  if [[ ${#__ind_parts[@]} -eq 0 ]]; then
    echo "❌ Error: --individual-tensors provided but empty." >&2
    exit 1
  fi

  for tok in "${__ind_parts[@]}"; do
    # trim whitespace
    tok="${tok#"${tok%%[![:space:]]*}"}"
    tok="${tok%"${tok##*[![:space:]]}"}"
    if [[ -z "$tok" ]]; then
      echo "❌ Error: empty token in --individual-tensors list." >&2
      exit 1
    fi
    if ! [[ "$tok" =~ ^[0-9]+$ ]]; then
      echo "❌ Error: invalid non-numeric token in --individual-tensors: '$tok'." >&2
      exit 1
    fi
    # convert decimal safely (avoid leading zero octal)
    val=$((10#$tok))
    if (( val < 1 || val > num_shards )); then
      echo "❌ Error: --individual-tensors value $val out of range (must be between 1 and $num_shards)." >&2
      exit 1
    fi
    if [[ -n "${IND_TENSOR_SET[$val]:-}" ]]; then
      echo "❌ Error: duplicate tensor number in --individual-tensors: $val" >&2
      exit 1
    fi
    IND_TENSOR_SET[$val]=1
  done

  # For logging, prepare a sorted, comma-separated presentation
  sorted_list="$(printf "%s\n" "${!IND_TENSOR_SET[@]}" | sort -n | paste -sd, -)"
  echo "[$(timestamp)] --individual-tensors enabled: will only process tensors: $sorted_list"
fi
# --------------------------------------------------------------------------

# -------------------------
# Apply --individual-tensors filter to the arrays we just populated.
# Reasoning:
#  - The map files contain shards for chunk ids starting at 00002 (the special first shard 00001 is not present in maps).
#  - The user supplies tensor numbers like "2,5,..." (these are absolute chunk ids in the model numbering).
#  - We must keep only those map entries whose recorded shard_id (set earlier via set_shard_id)
#    matches one of the user-provided token numbers.
#  - Do NOT error out if the user only requested "1" (the special first shard) because that shard is not part of the map arrays;
#    in that case the arrays will become empty and first-shard logic elsewhere will handle it.
# -------------------------
if [[ "$INDIVIDUAL_TENSORS_ENABLED" == true ]]; then
  DEBUG "Applying --individual-tensors filter to SHARD_FILENAMES / TENSORS_TO_FETCH (keeping only user-selected chunk ids)"

  # Temporary arrays to hold the filtered results
  declare -a __NEW_SHARD_FILENAMES=()
  declare -a __NEW_TENSORS_TO_FETCH=()

  # Iterate over existing arrays and keep only entries whose shard id is present in IND_TENSOR_SET
  for __i in "${!SHARD_FILENAMES[@]}"; do
    __tensor="${TENSORS_TO_FETCH[$__i]:-}"
    # get_shard_id returns the numeric chunk id (1-based as parsed from filename)
    __chunk_id="$(get_shard_id "$__tensor")"

    # If shard_id couldn't be determined, warn and skip
    if [[ -z "$__chunk_id" ]]; then
      #echo "⚠️ Warning: could not determine shard id for tensor='$__tensor' at index $__i; skipping from filtered lists." >&2
      continue
    fi

    # Keep entry if user explicitly requested this chunk id
    if [[ -n "${IND_TENSOR_SET[$__chunk_id]:-}" ]]; then
      __NEW_SHARD_FILENAMES+=("${SHARD_FILENAMES[$__i]}")
      __NEW_TENSORS_TO_FETCH+=("$__tensor")
      DEBUG "Keeping shard index $__i (chunk $__chunk_id) -> ${SHARD_FILENAMES[$__i]}"
    else
      DEBUG "Dropping shard index $__i (chunk $__chunk_id) -> ${SHARD_FILENAMES[$__i]}"
    fi
  done

  # Replace original arrays with filtered ones
  SHARD_FILENAMES=("${__NEW_SHARD_FILENAMES[@]}")
  TENSORS_TO_FETCH=("${__NEW_TENSORS_TO_FETCH[@]}")

  # Logging: if the filter removed everything, that may be because user requested only the special first shard (1).
  if [[ "${#SHARD_FILENAMES[@]}" -eq 0 ]]; then
    echo "[$(timestamp)] Note: --individual-tensors was provided but no matching shards were found in the map entries."
    echo "         This can happen if you requested the special first shard (1), which is not present in the map files."
    echo "         First-shard handling (download/verification/quantize) will be performed separately later if required."
  else
    # For traceability show which zero-based map indexes remain (useful debug output)
    kept_idxs="$(printf "%s\n" "${!SHARD_FILENAMES[@]}" | paste -sd, -)"
    echo "[$(timestamp)] --individual-tensors filter applied: keeping ${#SHARD_FILENAMES[@]} shard(s) from map (map-array indexes: ${kept_idxs})."
  fi
fi

# Prepare dynamic copies
TENSORS_TO_FETCH_DYNAMIC=( "${TENSORS_TO_FETCH[@]}" )
SHARD_FILENAMES_DYNAMIC=( "${SHARD_FILENAMES[@]}" )

# Quantize dynamic lists (initially empty, we'll move computed-qtype shards into them)
TENSORS_TO_QUANTIZE_DYNAMIC=()
SHARD_QUANTIZE_FILENAMES_DYNAMIC=()

# Step 2: Scan original arrays and move computed qtype shards from dynamic fetch lists
# into the quantize dynamic lists (keep indices in sync)
# We iterate original arrays by index to detect computed qtypes, then find the same item
# in the dynamic lists to remove/move it.
for idx in "${!TENSORS_TO_FETCH[@]}"; do
  tensor="${TENSORS_TO_FETCH[$idx]:-}"
  [[ -z "$tensor" ]] && continue
  chunk_id="$(get_shard_id "$tensor")"
  shard_file="${SHARD_FILENAMES[$idx]:-}"

  # Determine target qtype for this tensor
  target_q=""
  for i in "${!PATTERNS[@]}"; do
    if [[ "$tensor" =~ ${PATTERNS[$i]} ]]; then
      target_q="${PATTERN_QTYPES[$i]}"
      break
    fi
  done
  if [[ -z "$target_q" ]]; then
    continue
  fi

  # If computed qtype, move this entry from *_DYNAMIC to *_QUANTIZE_DYNAMIC
  if [[ -n "${COMPUTED_QTYPES[${target_q,,}]:-}" ]]; then
    # Find the index of the matching tensor in TENSORS_TO_FETCH_DYNAMIC
    for d_idx in "${!TENSORS_TO_FETCH_DYNAMIC[@]}"; do
      if [[ "${TENSORS_TO_FETCH_DYNAMIC[$d_idx]}" == "$tensor" ]]; then
        # Append into quantize dynamic arrays
        TENSORS_TO_QUANTIZE_DYNAMIC+=( "$tensor" )
        SHARD_QUANTIZE_FILENAMES_DYNAMIC+=( "$shard_file" )

        # Remove from fetch dynamic arrays while preserving order
        unset 'TENSORS_TO_FETCH_DYNAMIC[d_idx]'
        unset 'SHARD_FILENAMES_DYNAMIC[d_idx]'
        # Re-index arrays
        TENSORS_TO_FETCH_DYNAMIC=( "${TENSORS_TO_FETCH_DYNAMIC[@]}" )
        SHARD_FILENAMES_DYNAMIC=( "${SHARD_FILENAMES_DYNAMIC[@]}" )
        DEBUG "moved computed qtype tensor to quantize queue: tensor='$tensor' chunk_id='$chunk_id' shard_file='$shard_file'"
        break
      fi
    done
  fi
done

# If user requested --quantize-all-shards, move all remaining fetch items into quantize queue.
if [[ "$QUANTIZE_ALL_SHARDS" == true ]]; then
  if ((${#TENSORS_TO_FETCH_DYNAMIC[@]} > 0)); then
    for i in "${!TENSORS_TO_FETCH_DYNAMIC[@]}"; do
      TENSORS_TO_QUANTIZE_DYNAMIC+=( "${TENSORS_TO_FETCH_DYNAMIC[$i]}" )
      SHARD_QUANTIZE_FILENAMES_DYNAMIC+=( "${SHARD_FILENAMES_DYNAMIC[$i]}" )
    done
    TENSORS_TO_FETCH_DYNAMIC=()
    SHARD_FILENAMES_DYNAMIC=()
    echo "[$(timestamp)] --quantize-all-shards: moved all fetch-list tensors into quantize queue (count=${#TENSORS_TO_QUANTIZE_DYNAMIC[@]})"
  else
    echo "[$(timestamp)] --quantize-all-shards: fetch queue empty; nothing to move."
  fi
fi

# ---------------------------
# Apply regex-driven selection to move specific fetch-dynamic items into the quantize queue.
# - If --quantize-tensors-regex is set, any tensor whose name matches any of the provided regexes
#   will be moved to the quantize queue (performed BEFORE MAIN dynamic loop).
# - If --quantize-qtypes-regex is set, any tensor whose target qtype (determined by PATTERNS) matches
#   any of the provided regexes will be moved to the quantize queue.
# Implementation detail: iterate from highest index to lowest to avoid index-shifting bugs when removing items.
# ---------------------------

# Helper: copy an item (by index) from fetch-dynamic -> quantize-dynamic
# Append the moved item so it becomes the last element of the target lists
copy_fetch_index_to_quantize() {
  local idx="$1"
  local tensor="${TENSORS_TO_FETCH_DYNAMIC[$idx]:-}"
  local shard_file="${SHARD_FILENAMES_DYNAMIC[$idx]:-}"
  if [[ -z "$tensor" ]]; then
    return 1
  fi
  # Append to quantize arrays
  TENSORS_TO_QUANTIZE_DYNAMIC=( "${TENSORS_TO_QUANTIZE_DYNAMIC[@]}" "$tensor" )
  SHARD_QUANTIZE_FILENAMES_DYNAMIC=( "${SHARD_QUANTIZE_FILENAMES_DYNAMIC[@]}" "$shard_file" )
  return 0
}

# Helper: move an item (by index) from fetch-dynamic -> quantize-dynamic
# Prepend the moved item so it becomes the first element of the target lists
move_fetch_index_to_quantize() {
  local idx="$1"
  local tensor="${TENSORS_TO_FETCH_DYNAMIC[$idx]:-}"
  local shard_file="${SHARD_FILENAMES_DYNAMIC[$idx]:-}"
  if [[ -z "$tensor" ]]; then
    return 1
  fi
  # Prepend to quantize arrays
  TENSORS_TO_QUANTIZE_DYNAMIC=( "$tensor" "${TENSORS_TO_QUANTIZE_DYNAMIC[@]}" )
  SHARD_QUANTIZE_FILENAMES_DYNAMIC=( "$shard_file" "${SHARD_QUANTIZE_FILENAMES_DYNAMIC[@]}" )
  # Remove from fetch arrays and re-pack to shift indices
  unset 'TENSORS_TO_FETCH_DYNAMIC[idx]'
  unset 'SHARD_FILENAMES_DYNAMIC[idx]'
  TENSORS_TO_FETCH_DYNAMIC=( "${TENSORS_TO_FETCH_DYNAMIC[@]}" )
  SHARD_FILENAMES_DYNAMIC=( "${SHARD_FILENAMES_DYNAMIC[@]}" )
  return 0
}

# Helper: find index in fetch-dynamic by chunk_id and move to quantize dynamic
move_fetch_by_chunk_to_quantize() {
  local chunkid="$1"
  local found_idx=""
  for i in "${!TENSORS_TO_FETCH_DYNAMIC[@]}"; do
    local t="${TENSORS_TO_FETCH_DYNAMIC[$i]}"
    local c
    c="$(get_shard_id "$t")"
    if [[ "$c" == "$chunkid" ]]; then
      found_idx="$i"
      break
    fi
  done
  if [[ -n "$found_idx" ]]; then
    move_fetch_index_to_quantize "$found_idx"
    return 0
  fi
  return 1
}

# Helper: check for any .failed_download.* marker files, move corresponding items
process_failed_download_markers() {
  shopt -s nullglob
  local markers=( "$LOCAL_DOWNLOAD_DIR"/.failed_download.* )
  shopt -u nullglob
  local any_moved=0
  for f in "${markers[@]}"; do
    # filename: $LOCAL_DOWNLOAD_DIR/.failed_download.<chunkid>
    chunk_marker="${f##*.}"   # obtains chunkid (works because marker format is .failed_download.<chunkid>)
    # move matching fetch entry to quantize dynamic list
    if move_fetch_by_chunk_to_quantize "$chunk_marker"; then
      DEBUG "moved failed-download chunk '$chunk_marker' into quantize queue (and removed from fetch-dynamic)"
      any_moved=1
    else
      DEBUG "no matching fetch-dynamic entry found for failed-download chunk '$chunk_marker' (maybe already moved)"
    fi
    # remove the marker
    rm -f -- "$f" 2>/dev/null || true
  done
  return $any_moved
}

# Helper: detect whether a quantize slot is available
quantize_slot_available() {
  shopt -s nullglob
  local q=( "$LOCAL_DOWNLOAD_DIR"/.quantize.* )
  shopt -u nullglob
  local count=${#q[@]}
  if (( count >= MAX_QUANTIZE_JOBS )); then
    return 1
  fi
  return 0
}

# Helper: detect failed quantize markers
failed_quantize_detected() {
  shopt -s nullglob
  local fq=( "$LOCAL_DOWNLOAD_DIR"/.failed_quantize.* )
  shopt -u nullglob
  [[ ${#fq[@]} -gt 0 ]]
}

# Helper to test if a string matches any regex in a given array
_matches_any_regex_in_array() {
  local subject="$1"
  shift
  local -a arr=("$@")
  for pat in "${arr[@]}"; do
    if [[ "$subject" =~ $pat ]]; then
      return 0
    fi
  done
  return 1
}

# Apply quantize-tensors-regex to TENSORS_TO_FETCH_DYNAMIC
if [[ "$QUANTIZE_TENSORS_REGEX_ENABLED" == true && ${#QUANTIZE_TENSORS_REGEX[@]} -gt 0 && ${#TENSORS_TO_FETCH_DYNAMIC[@]} -gt 0 ]]; then
  DEBUG "--quantize-tensors-regex enabled: applying regexes to tensor names"
  # iterate high -> low to avoid skipping after removals
  for (( i=${#TENSORS_TO_FETCH_DYNAMIC[@]}-1; i>=0; i-- )); do
    tname="${TENSORS_TO_FETCH_DYNAMIC[$i]}"
    if _matches_any_regex_in_array "$tname" "${QUANTIZE_TENSORS_REGEX[@]}"; then
      DEBUG "quantize-tensors-regex: moving tensor '$tname' at index $i to quantize queue"
      move_fetch_index_to_quantize "$i"
    fi
  done
fi

# Apply quantize-qtypes-regex to TENSORS_TO_FETCH_DYNAMIC
if [[ "$QUANTIZE_QTYPES_REGEX_ENABLED" == true && ${#QUANTIZE_QTYPES_REGEX[@]} -gt 0 && ${#TENSORS_TO_FETCH_DYNAMIC[@]} -gt 0 ]]; then
  DEBUG "--quantize-qtypes-regex enabled: applying regexes to target qtypes"
  # iterate high -> low to avoid skipping after removals
  for (( i=${#TENSORS_TO_FETCH_DYNAMIC[@]}-1; i>=0; i-- )); do
    tname="${TENSORS_TO_FETCH_DYNAMIC[$i]}"
    # determine target qtype for this tensor by scanning PATTERNS
    found_target_q=""
    for pidx in "${!PATTERNS[@]}"; do
      if [[ "$tname" =~ ${PATTERNS[$pidx]} ]]; then
        found_target_q="${PATTERN_QTYPES[$pidx]}"
        break
      fi
    done
    if [[ -n "$found_target_q" ]]; then
      # match against lowercase form for robustness
      if _matches_any_regex_in_array "${found_target_q,,}" "${QUANTIZE_QTYPES_REGEX[@]}"; then
        DEBUG "quantize-qtypes-regex: moving tensor '$tname' (target qtype='$found_target_q') at index $i to quantize queue"
        move_fetch_index_to_quantize "$i"
      fi
    fi
  done
fi

# ------------------ SPECIAL NODE ASSIGNMENT (if requested) ----------------
# New approach: compute an xxh3-based digest over MODEL_NAME+chunkid and use that
# to decide if *this* runner should download that chunk. This avoids a fixed
# assigned_node and uses per-chunk hashing. The rule implemented:
#   should_process_chunk(chunk_id) -> true if (dec_from_xxhsum(MODEL_NAME+chunkid) % SPECIAL_NODE_ID) == 0
#
# The function below returns success (0) when the chunk should be downloaded by this node.
if [[ -n "$SPECIAL_NODE_ID" ]]; then
  # validate SPECIAL_NODE_ID
  if ! [[ "$SPECIAL_NODE_ID" =~ ^[0-9]+$ ]] || (( SPECIAL_NODE_ID < 0 )); then
    echo "❌ Error: --special-node-id must be a positive integer." >&2
    exit 1
  fi

  # try to read MODEL_NAME from $SCRIPT_DIR/download.conf
  CONFIG_FILE="$SCRIPT_DIR/download.conf"
  MODEL_NAME=""
  MAINTAINER=""
  if [[ -f "$CONFIG_FILE" ]]; then
    # shellcheck source=/dev/null
    source "$CONFIG_FILE"
  fi

  if [[ -z "$MODEL_NAME" ]]; then
    echo "❌ Error: --special-node-id provided but MODEL_NAME could not be read from $DOWNLOAD_CONF" >&2
    exit 1
  fi

  # Ensure xxhsum is available
  if ! command -v xxhsum >/dev/null 2>&1; then
    echo "❌ Error: --special-node-id requires 'xxhsum' to be installed and on PATH." >&2
    echo "Hint: Install xxhash (xxhsum) or omit --special-node-id." >&2
    exit 1
  fi

  # should_process_chunk: returns 0 if this node should process the provided chunk_id
  # Uses: printf '%s' "${MODEL_NAME}-${MAINTAINER}-${QTYPE^^}-SPECIAL_SPLIT${chunk_id}" | xxhsum -H3
  should_process_chunk() {
    local chunk_id="$1"
    chunk_id=$((10#$chunk_id))

    # Concatenate model name + chunk_id (no separator; deterministic)
    local input="${MODEL_NAME}-${MAINTAINER}-${QTYPE^^}-SPECIAL_SPLIT${chunk_id}"

    # Example: echo abc | xxhsum -H3
    local out=$(printf '%s' "$input" | xxhsum -H3 2>/dev/null) || true

    # Extract a hex-like token from the output robustly.
    # Require at least 12 hex chars, then take the first 12.
    local hex=$(printf '%s' "$out" | grep -oE '[0-9a-fA-F]{12,}' | head -n1 || true)

    if [[ -z "$hex" ]]; then
      echo "[$(timestamp)] ⚠️ Warning: failed to parse xxhsum output for chunk_id=$chunk_id; output='$out'." >&2
      return 1
    fi

    # Use only the first 12 hex chars to avoid integer overflow in bash arithmetic
    hex="${hex:0:12}"

    # Convert to decimal and modulo
    if ! [[ "$hex" =~ ^[0-9a-fA-F]+$ ]]; then
      echo "[$(timestamp)] ⚠️ Warning: invalid hex digest ('$hex') for chunk_id=$chunk_id." >&2
      return 1
    fi

    local dec=$((16#$hex))
    local mod=$(( dec % TOTAL_NODES ))

    # We choose the rule: this node handles chunk if mod == SPECIAL_NODE_ID
    if (( mod == SPECIAL_NODE_ID )); then
      return 0
    else
      return 1
    fi
  }

  echo "[$(timestamp)] SPECIAL NODE MODE: MODEL_NAME='$MODEL_NAME' using xxhsum. Only chunks where xxhsum(MODEL_NAME+chunkid) % $TOTAL_NODES == $SPECIAL_NODE_ID will be downloaded/verified by this runner."
fi

# Helper: attempt to redownload the first shard and its signature (unless verify-readonly).
# Returns 0 on success (downloaded & moved into LOCAL_MODEL_DIR), non-zero otherwise.
attempt_redownload_first() {
  if [[ "$VERIFY_READONLY" == true ]]; then
    return 1
  fi

  echo "[$(timestamp)] First shard appears corrupted or invalid — attempting to redownload first shard (and signature if GPG verification enabled)."

  # remove any possibly-broken files (both model dir and download dir)
  rm -f "$LOCAL_MODEL_DIR/$gguf_first" "$LOCAL_MODEL_DIR/$gguf_first.zbst" "$LOCAL_MODEL_DIR/$gguf_first.sig \
        "$LOCAL_DOWNLOAD_DIR/$gguf_first" "$LOCAL_DOWNLOAD_DIR/$gguf_first.zbst" "$LOCAL_DOWNLOAD_DIR/$gguf_first.sig || true

  # Ensure download dir exists for redownload attempts
  mkdir -p "$LOCAL_DOWNLOAD_DIR" 2>/dev/null || true

  # download shard (keep retrying until success)
  until run_downloader_shard "${QTYPE}" 1 "$LOCAL_DOWNLOAD_DIR" "$(basename "$gguf_first")"; do
    echo "[$(timestamp)] First shard download failed; retrying in 10s..." >&2
    sleep 10
  done

  # Move whichever artifact was produced (.gguf OR .gguf.zbst)
  if safe_file_exists "$LOCAL_DOWNLOAD_DIR/$(basename "$gguf_first")"; then
    mv -f "$LOCAL_DOWNLOAD_DIR/$(basename "$gguf_first")" "$LOCAL_MODEL_DIR/" || true
  elif safe_file_exists "$LOCAL_DOWNLOAD_DIR/$(basename "$gguf_first").zbst"; then
    mv -f "$LOCAL_DOWNLOAD_DIR/$(basename "$gguf_first").zbst" "$LOCAL_MODEL_DIR/" || true
  else
    echo "[$(timestamp)] ❌ Error: expected first shard in download dir after redownload but none found." >&2
    return 1
  fi

  # download signature if required
  if [[ "$SKIP_GPG" != true ]]; then
    until run_downloader_shard "${QTYPE}" -2 "$LOCAL_DOWNLOAD_DIR" "$(basename "$gguf_first.sig")"; do
      echo "[$(timestamp)] First shard signature download failed; retrying in 10s..." >&2
      sleep 10
    done

    # Move signature artifact that was procured (.gguf.sig)
    if [[ -f "$LOCAL_DOWNLOAD_DIR/$(basename "$gguf_first.sig")" ]]; then
      mv -f "$LOCAL_DOWNLOAD_DIR/$(basename "$gguf_first.sig")" "$LOCAL_MODEL_DIR/" || true
    else
      echo "[$(timestamp)] ❌ Error: expected first shard signature in download dir after redownload but none found." >&2
      return 1
    fi
  else
    echo "[$(timestamp)] Redownload of first shard completed."
  fi

  return 0
}

# --------------- CONCURRENCY HELPERS ----------------
wait_for_slot() {
  while (( $(jobs -rp | wc -l) >= MAX_JOBS )); do sleep 0.5; done
}
wait_for_slot_quantize() {
  while (( $(jobs -rp | wc -l) >= MAX_QUANTIZE_JOBS )); do sleep 0.5; done
}

# ------------------ LLAMA-QUANTIZE SUPPORT CHECK ------------------
# Validate that the provided llama-quantize binary supports --individual-tensors.
require_llama_quantize_support() {
  if [[ -z "$LLAMA_QUANTIZE_BIN" ]]; then
    echo "❌ Error: --llama-quantize must be provided when quantization from bf16 is required." >&2
    echo "Please obtain a build that includes --individual-tensors support from:" >&2
    echo "  https://github.com/Thireus/ik_llama.cpp/tree/th/quantize_individual_tensors" >&2
    echo "Pre-built releases are at: https://github.com/Thireus/ik_llama.cpp/releases (look for th-quantize_individual_tensors*)." >&2
    exit 21
  fi

  if [[ ! -x "$LLAMA_QUANTIZE_BIN" ]]; then
    echo "❌ Error: Provided --llama-quantize binary '$LLAMA_QUANTIZE_BIN' is not executable or not found." >&2
    exit 22
  fi

  # Check help text for --individual-tensors
  local matching_help_line=$("$LLAMA_QUANTIZE_BIN" --help 2>&1 | grep -- '--individual-tensors' | head -n1 || true)
  if [[ -z "$matching_help_line" ]]; then
    echo "❌ Error: The provided llama-quantize binary '$LLAMA_QUANTIZE_BIN' does not advertise the --individual-tensors option." >&2
    echo "Please obtain a build with individual-tensors support from:" >&2
    echo "  https://github.com/Thireus/ik_llama.cpp/tree/th/quantize_individual_tensors" >&2
    echo "Pre-built releases are at: https://github.com/Thireus/ik_llama.cpp/releases (look for th-quantize_individual_tensors*)." >&2
    exit 23
  fi
}
# ------------------------------------------------------------------------

# ------------------ Quantize-from-bf16 helpers -----------------------
# BF16 workspace (subdir of LOCAL_DOWNLOAD_DIR)
# Allow override from --quantize-bf16-directory; if not provided, default to LOCAL_DOWNLOAD_DIR/bf16
BF16_DOWNLOAD_DIR="${QUANTIZE_BF16_DIR:-$LOCAL_DOWNLOAD_DIR/bf16}"
ensure_bf16_workspace() {
  if [[ "$VERIFY_READONLY" == true ]]; then
    echo "❌ Error: cannot create bf16 workspace in --verify-readonly mode." >&2
    exit 24
  fi
  mkdir -p "$BF16_DOWNLOAD_DIR" || { echo "❌ Error: failed to create bf16 workspace at $BF16_DOWNLOAD_DIR" >&2; exit 25; }
}

# Build the nested invocation args that must be passed to the nested quant_downloader instance.
_build_bf16_nested_args() {
  local args=( "$SCRIPT_DIR/quant_downloader.sh" "-d" "$BF16_DOWNLOAD_DIR" "-j" "1" "--no-final-message" "--qtype" "bf16" )
  # Additional base parameters
  args+=( --no-new-map )
  # Propagate common parameters (if provided)
  if [[ "$FORCE_REDOWNLOAD" == true ]]; then
    args+=( --force-redownload )
  fi
  if [[ "$SKIP_GPG" == true ]]; then
    args+=( --skip-gpg )
  fi
  if [[ "$SKIP_HASH" == true ]]; then
    args+=( --skip-hash )
  fi
  # Propagate user-specified compression-tool config (if provided)
  if [[ "$ARCHIVE_NOAUTO" == true ]]; then
    args+=( --z-noauto )
  fi
  # We must forward --z-custom-tools and --z-decompress / --z-decompress-opt and --symlink-only if the current invocation included them.
  # Check current flags inferred from variables and append accordingly.
  if [[ ${#CUSTOM_TOOLS[@]} -gt 0 ]]; then
    # Reconstruct the original custom-tools string (first set was default + any additions)
    # Keep it simple: join CUSTOM_TOOL_NAMES with their magics in the same order as CUSTOM_TOOL_NAMES/CUSTOM_TOOL_MAGICS.
    local ct=""
    for i in "${!CUSTOM_TOOL_NAMES[@]}"; do
      local n="${CUSTOM_TOOL_NAMES[$i]}"
      local m="${CUSTOM_TOOL_MAGICS[$i]}"
      if [[ -n "$ct" ]]; then ct="$ct,"; fi
      ct="${ct}${n}:${m}"
    done
    args+=( --z-custom-tools "$ct" )
  fi
  # Forward --z-decompress flags if present
  if [[ "$ARCHIVE_DECOMPRESS" == true ]]; then
    args+=( --z-decompress )
  fi
  # Forward any provided decompress opts
  if [[ ${#DECOMP_OP_TOOL_NAMES[@]} -gt 0 ]]; then
    # reconstruct decomp opts string like 'zstd:-X,lbzip2:-Y'
    local dstr=""
    for i in "${!DECOMP_OP_TOOL_NAMES[@]}"; do
      if [[ -n "$dstr" ]]; then dstr="${dstr},"; fi
      dstr="${dstr}${DECOMP_OP_TOOL_NAMES[$i]}:${DECOMP_OP_TOOL_VALUES[$i]}"
    done
    args+=( --z-decompress-opt "$dstr" )
  fi

  # Forward --symlink-only if set
  if [[ "$SYMLINK_ONLY" == true ]]; then
    args+=( --symlink-only )
  fi

  echo "${args[@]}"
}

# Download a single bf16 tensor into BF16_DOWNLOAD_DIR by invoking this script as a nested process.
download_bf16_tensor_via_nested() {
  local tensor_idx="$1"   # numeric chunk id

  # Create a minimal bf16 recipe if not present
  local bf16_recipe="$BF16_DOWNLOAD_DIR/bf16.recipe"
  if [[ ! -f "$bf16_recipe" ]]; then
    printf '.*=bf16\n' > "$bf16_recipe"
  fi

  # Build args
  IFS=' ' read -r -a BF16_ARGS <<< "$(_build_bf16_nested_args)"

  # Append individual-tensors argument and recipe
  BF16_ARGS+=( -j "${MAX_QUANTIZE_JOBS}" --individual-tensors "${tensor_idx}" "$bf16_recipe" ) # -j MAX_QUANTIZE_JOBS is intentional

  echo "[$(timestamp)] Invoking nested quant_downloader to fetch bf16 tensor ${tensor_idx} into ${BF16_DOWNLOAD_DIR}..."
  if ! "${BF16_ARGS[@]}"; then
    echo "❌ Error: nested quant_downloader failed to fetch bf16 tensor ${tensor_idx}." >&2
    return 1
  fi
  return 0
}

# Ensure first shard available for quantization: attempt_redownload_first once (idempotent).
ensure_first_shard_for_quantize() {
  if [[ "$BF16_FIRST_OBTAINED" == true ]]; then
    return 0
  fi

  # Compute gguf_first name based on SHARD_FILENAMES if possible
  num_shards=${#SHARD_FILENAMES_FULL[@]}
  if (( num_shards == 0 )); then
    echo "❌ Error: no shards discovered; cannot determine first-shard name for quantization." >&2
    exit 1
  fi
  # Build total (num_shards + 1) zero-padded
  total=$(printf "%05d" "$((num_shards + 1))")
  # Derive gguf_first by replacing the chunk part in first shard filename
  gguf_first="$(printf '%s' "${SHARD_FILENAMES_FULL[0]}" | sed -E "s/-[0-9]{5}-of-[0-9]{5}\.gguf$/-00001-of-${total}.gguf/")"

  # If the bf16 workspace already contains the first shard, we're good
  if isafe_file_exists "$BF16_DOWNLOAD_DIR/$gguf_first"; then
    BF16_FIRST_OBTAINED=true
    return 0
  fi

  # Attempt to fetch the first shard into LOCAL_MODEL_DIR (one-time operation).
  # This uses existing attempt_redownload_first() helper which downloads into LOCAL_DOWNLOAD_DIR then moves into LOCAL_MODEL_DIR.
  echo "[$(timestamp)] Ensuring first-shard ($gguf_first) is available for quantization (one-time operation)."
  if ! attempt_redownload_first; then
    echo "❌ Error: failed to obtain first shard via attempt_redownload_first()" >&2
    return 1
  fi

  # Now copy or symlink the first shard into BF16_DOWNLOAD_DIR so llama-quantize can run inside that workspace.
  ensure_bf16_workspace
  if isafe_file_exists "$LOCAL_MODEL_DIR/$gguf_first"; then
    cp -f "$LOCAL_MODEL_DIR/$gguf_first" "$BF16_DOWNLOAD_DIR/$gguf_first" || { echo "❌ Error: failed to copy first shard into bf16 workspace" >&2; return 1; }
    chmod 444 "$BF16_DOWNLOAD_DIR/$gguf_first" || true
  elif isafe_file_exists "$LOCAL_MODEL_DIR/$gguf_first.zbst"; then
    # If only compressed exists, decompress to bf16 workspace
    if ! decompress_archive_to_file "$LOCAL_MODEL_DIR/$gguf_first.zbst" "$BF16_DOWNLOAD_DIR/$gguf_first"; then
      echo "❌ Error: failed to decompress first shard into bf16 workspace." >&2
      return 1
    fi
  else
    # If not present in LOCAL_MODEL_DIR (unexpected), attempt to copy from LOCAL_DOWNLOAD_DIR
    if isafe_file_exists "$LOCAL_DOWNLOAD_DIR/$gguf_first"; then
      cp -f "$LOCAL_DOWNLOAD_DIR/$gguf_first" "$BF16_DOWNLOAD_DIR/$gguf_first" || { echo "❌ Error: failed to copy first shard from download dir into bf16 workspace" >&2; return 1; }
      chmod 444 "$BF16_DOWNLOAD_DIR/$gguf_first" || true
    fi
  fi

  BF16_FIRST_OBTAINED=true
  echo "[$(timestamp)] First-shard is present in bf16 workspace: $BF16_DOWNLOAD_DIR/$gguf_first"
  return 0
}

# Run llama-quantize for a given tensor: assumes bf16 input(s) exist in BF16_DOWNLOAD_DIR and first-shard copied there.
quantize_tensor_from_bf16() {
  local chunk_id="$1"         # numeric chunk id
  local output_filename="$2"  # expected output filename (basename)
  local tensor_qtype="$3"     # target qtype
  local first_shard_name="$4" # gguf_first name basename

  require_llama_quantize_support

  # Ensure bf16 workspace and first shard
  # ensure_bf16_workspace
  # if ! ensure_first_shard_for_quantize; then
  #   echo "❌ Error: cannot ensure first shard for quantization" >&2
  #   return 1
  # fi

  # Ensure the specific bf16 tensor file exists (download it if missing)
  # The nested downloader will fetch this chunk into BF16_DOWNLOAD_DIR
  # Note: some repos may serve compressed .zbst; nested invocation propagates z-decompress flags as requested by the user.
  echo "[$(timestamp)] Ensuring bf16 chunk ${chunk_id} is present in $BF16_DOWNLOAD_DIR ..."
  if ! download_bf16_tensor_via_nested "$chunk_id"; then
    echo "❌ Error: Failed to download bf16 chunk ${chunk_id} into ${BF16_DOWNLOAD_DIR}." >&2
    return 1
  fi

  # Build llama-quantize invocation
  local llama_args=()
  llama_args+=( "$LLAMA_QUANTIZE_BIN" --individual-tensors "${chunk_id}" --keep-split --skip-first-shard )
  # forward imatrix-related flags if set
  if [[ "$CONVERT_IGNORE_IMATRIX_RULES" == true ]]; then
    llama_args+=( --ignore-imatrix-rules )
  fi
  if [[ -n "$WITH_IMATRIX_FILE" ]]; then
    llama_args+=( --imatrix "$WITH_IMATRIX_FILE" )
  fi

  # Input and output: "<first_gguf_shard>.gguf <output>.gguf bf16 1"
  # first_shard_name should be basename e.g. "model-00001-of-XXXXX.gguf"
  # Use configured QUANTIZE_NTHREADS instead of hard-coded value (default 0 means auto)
  output_file="$LOCAL_DOWNLOAD_DIR/$(echo "$output_filename" | sed -E 's/-[0-9]{5}-of-[0-9]{5}\.gguf$/.gguf/')"
  llama_args+=( "${first_shard_name}" "${output_file}" "${tensor_qtype}" "${QUANTIZE_NTHREADS}" )

  echo "[$(timestamp)] Running llama-quantize: ${llama_args[*]} from $(pwd)"
  rm -f "${output_file}" || true # Make sure the output_file doesn't already exist
  if ! "${llama_args[@]}"; then
    echo "❌ Error: llama-quantize failed for tensor ${chunk_id} -> ${output_filename}" >&2
    return 1
  fi

  chmod 444 "$LOCAL_DOWNLOAD_DIR/${output_filename}" || true

  return 0
}
# ------------------------------------------------------------------------

# ----------------- SHARD DOWNLOAD/VERIFY LOGIC --------------
download_shard() {
  local tensor="$1"
  local shard_file="$2"
  local chunk_id="$(get_shard_id "$tensor")"

  local local_gguf="$LOCAL_MODEL_DIR/$shard_file"
  local dl_gguf="$LOCAL_DOWNLOAD_DIR/$shard_file"
  local local_z="${local_gguf}.zbst"
  local dl_z="${dl_gguf}.zbst"

  # If individual-tensors list is enabled, skip shards not in set.
  if [[ "$INDIVIDUAL_TENSORS_ENABLED" == true ]]; then
    if [[ -z "${IND_TENSOR_SET[$chunk_id]:-}" ]]; then
      echo "[$(timestamp)] Skipping tensor='$tensor' chunk_id=$chunk_id because it's not in --individual-tensors list."
      return 0
    fi
  fi

  # If special node assignment is enabled, skip shards not assigned to this node
  if [[ -n "$SPECIAL_NODE_ID" ]]; then
    if ! should_process_chunk "$chunk_id"; then
      _del=""
      [[ "$RM_SKIPPED_SHARDS" == true ]] && _del=" and deleting (if present)" && rm -f "$dl_gguf" "$local_gguf" "$dl_z" "$local_z" || true
      echo "[$(timestamp)] Skipping$_del tensor='$tensor' chunk_id=$chunk_id not assigned to this node (xxhsum-based selection)."
      return 0
    else
      echo "[$(timestamp)] Tensor='$tensor' chunk_id=$chunk_id assigned to this node (xxhsum-based selection) — proceeding to HASH verification"
    fi
  else
    echo "[$(timestamp)] Tensor='$tensor' chunk_id=$chunk_id HASH verification — proceeding"
  fi

  for i in "${!PATTERNS[@]}"; do
    local pat="${PATTERNS[$i]}"
    if [[ "$tensor" =~ $pat ]]; then
      local qtype="${PATTERN_QTYPES[$i]^^]}"
      local dl_type="$qtype"
      [[ "${qtype^^}" == "F32" ]] && dl_type="${QTYPE}"

      local shard_id=$(echo "$shard_file" | sed -E 's/.*-([0-9]{5})-of-[0-9]{5}\.gguf/\1/')

      local got=""
      local need_download=false
      local failed_verification_count=0
      local skip_mv=true
      if [[ "$FORCE_REDOWNLOAD" == true ]]; then
          echo "[$(timestamp)] Force redownload: removing existing shard $shard_file"
          rm -f "$dl_gguf" "$local_gguf" "$dl_z" "$local_z" || true
          sync || true
          skip_mv=false
      fi
      local _path=""
      # If ARCHIVE_DECOMPRESS true: if a .zbst exists, decompress to .gguf (overwrite) so script works on .gguf
      if [[ "$ARCHIVE_DECOMPRESS" == true ]]; then
        if safe_file_exists "$local_z"; then
          echo "[$(timestamp)] z-decompress: found $local_z -> decompressing to $local_gguf (overwrite)"
          if ! decompress_archive_to_file "$local_z" "$local_gguf"; then
            echo "[$(timestamp)] ⚠️ decompression failed for $local_z — treating as corrupted and will redownload." >&2
          fi
          rm -f "$local_z" || true
        elif safe_file_exists "$dl_z"; then
          echo "[$(timestamp)] z-decompress: found $dl_z in download dir -> decompressing to $local_gguf (overwrite)"
          if ! decompress_archive_to_file "$dl_z" "$local_gguf"; then
            echo "[$(timestamp)] ⚠️ decompression failed for $dl_z in download dir — treating as corrupted and will redownload." >&2
          fi
          rm -f "$dl_z" || true
        fi
      fi

      while [[ "$need_download" == false ]] && [[ "$got" == "" ]]; do
        # 1) If ARCHIVE_COMPRESS or ARCHIVE_DECOMPRESS is enabled, prefer .gguf.zbst when present and validate from stream
        if safe_file_exists "$local_z" || safe_file_exists "$dl_z"; then
          if [[ "$ARCHIVE_COMPRESS" == true ]]; then
            if safe_file_exists "$local_z"; then
              _path="$local_z"
              skip_mv=true
            else
              _path="$dl_z"
              skip_mv=false
            fi

            if should_skip_hash_for "$qtype" "$tensor"; then
              if [[ "$SKIP_GGUF_VERIFICATION" == false ]]; then
                echo "[$(timestamp)] SKIP-HASH: ${_path} (skipping stream sha256). GGUF verification will be conducted instead."
                if ! check_quantized_gguf "$_path" "$shard_id" "$qtype"; then
                  echo "❌ INVALID GGUF SHARD: ${shard_file} - tensor: '$tensor' - qtype: '$qtype'"
                  got=""
                else
                  echo "[$(timestamp)] GGUF-VERIFICATION OK: ${_path}"
                  # Per-tensor or global skip: treat as valid w/o computing stream hash
                  got="$(get_t_hash "$qtype" "$tensor")"
                fi
              else
                echo "[$(timestamp)] SKIP-VERIFICATION: treating ${_path} as valid (skipping stream sha256). No GGUF verification will be conducted (--skip-gguf-verification is enabled)."
                # Per-tensor or global skip: treat as valid w/o computing stream hash
                got="$(get_t_hash "$qtype" "$tensor")"
              fi
            else
              if got="$(safe_stream_sha256_from_z "$_path")"; then
                got="${got%%[^0-9a-fA-F]*}"
              else
                got=""
              fi
            fi

            local exp="$(get_t_hash "$qtype" "$tensor")"
            if [[ "$got" != "$exp" ]]; then
              if (( failed_verification_count >= MAX_FAILED_VERIFICATION )); then
                echo "[$(timestamp)] ❌ Too many hash/GGUF verification attempt failures (count: $failed_verification_count) for '${_path}' (stream) - tensor '$tensor' of qtype: '$qtype' ($got != $exp)"
                exit_from_subprocess 16
              fi
              failed_verification_count=$((failed_verification_count + 1))
              echo "[$(timestamp)] Will redownload due to hash/GGUF mismatch (count: $failed_verification_count) for '${_path}' (stream) - tensor '$tensor' of qtype: '$qtype' ($got != $exp)"
              rm -f "$dl_z" "$local_z" || true
              sync || true
              need_download=true
            else
              echo "[$(timestamp)] Stream-hash/GGUF OK for '${_path}' - tensor '$tensor' of qtype: '$qtype'"
              # If the valid zbst was in download dir, move to model dir
              if [[ "$skip_mv" == false ]]; then
                mv -f "$dl_z" "$LOCAL_MODEL_DIR/"
                echo "[$(timestamp)] Saved file id '$shard_id' (zbst) - tensor '$tensor' of qtype: '$qtype'"
              fi
              # Ensure we remove any corresponding .gguf if it exists (we keep only .gguf.zbst in compress mode)
              if safe_file_exists "$LOCAL_MODEL_DIR/$shard_file"; then
                rm -f "$LOCAL_MODEL_DIR/$shard_file" || true
                echo "[$(timestamp)] Removed corresponding .gguf for '${shard_file}' because .gguf.zbst is present and valid"
              fi
              # nothing more to do for this shard
            fi
          elif [[ "$ARCHIVE_DECOMPRESS" == true ]]; then
            if safe_file_exists "$local_z"; then
              echo "[$(timestamp)] z-decompress: found $local_z -> decompressing to $local_gguf (overwrite)"
              if ! decompress_archive_to_file "$local_z" "$local_gguf"; then
                echo "[$(timestamp)] ⚠️ decompression failed for $local_z — treating as corrupted and will redownload." >&2
              fi
              rm -f "$local_z" || true
            elif safe_file_exists "$dl_z"; then
              echo "[$(timestamp)] z-decompress: found $dl_z in download dir -> decompressing to $local_gguf (overwrite)"
              if ! decompress_archive_to_file "$dl_z" "$local_gguf"; then
                echo "[$(timestamp)] ⚠️ decompression failed for $dl_z in download dir — treating as corrupted and will redownload." >&2
              fi
              rm -f "$dl_z" || true
            fi
            # Proceed with .gguf present in $local_gguf, assuming decompression was successful
            skip_mv=true
            got=""
          fi
        fi

        # If we didn't find a valid zbst (or compress mode not enabled) or if decompress mode was enabled, check .gguf files
        if [[ -z "$got" ]]; then
          if safe_file_exists "$local_gguf" || safe_file_exists "$dl_gguf"; then
            if safe_file_exists "$local_gguf"; then
              _path=$local_gguf
            else
              _path=$dl_gguf
              skip_mv=false
            fi
            [ "$failed_verification_count" -gt 0 ] && sync || true
            # use safe_sha256sum which will retry if symlink
            if should_skip_hash_for "$qtype" "$tensor"; then
              if [[ "$SKIP_GGUF_VERIFICATION" == false ]]; then
                echo "[$(timestamp)] SKIP-HASH: ${_path} (skipping sha256). GGUF verification will be conducted instead."
                if ! check_quantized_gguf "$_path" "$shard_id" "$qtype"; then
                  echo "❌ INVALID GGUF SHARD: ${shard_file} - tensor: '$tensor' - qtype: '$qtype'"
                  got=""
                else
                  echo "[$(timestamp)] GGUF-VERIFICATION OK: ${_path}"
                  # Per-tensor or global skip: treat as valid w/o computing stream hash
                  got="$(get_t_hash "$qtype" "$tensor")"
                fi
              else
                echo "[$(timestamp)] SKIP-VERIFICATION: treating ${_path} as valid (skipping sha256). No GGUF verification will be conducted (--skip-gguf-verification is enabled)."
                # Per-tensor or global skip: treat as valid w/o computing stream hash
                got="$(get_t_hash "$qtype" "$tensor")"
              fi
            elif got="$(safe_sha256sum "$_path" 2>/dev/null)"; then
              got="${got%%[^0-9a-fA-F]*}"
            else
              got=""
            fi
            local exp="$(get_t_hash "$qtype" "$tensor")"
            if [[ "$got" != "$exp" ]]; then
              if (( failed_verification_count >= MAX_FAILED_VERIFICATION )); then
                echo "[$(timestamp)] ❌ Too many hash/GGUF verification attempt failures (count: $failed_verification_count) for '$shard_file' - tensor '$tensor' of qtype: '$qtype' ($got != $exp)"
                exit_from_subprocess 17
              fi
              failed_verification_count=$((failed_verification_count + 1))
              echo "[$(timestamp)] Will redownload due to hash/GGUF mismatch (count: $failed_verification_count) for '$shard_file' - tensor '$tensor' of qtype: '$qtype' ($got != $exp)"
              rm -f "$dl_gguf" "$local_gguf" || true
              sync || true
              need_download=true
            else
              if should_skip_hash_for "$qtype" "$tensor"; then
                if [[ "$SKIP_GGUF_VERIFICATION" == true ]]; then
                  echo "[$(timestamp)] File id '$shard_id' - tensor '$tensor' of qtype: '$qtype' processed (skipped hash and GGUF verification)!"
                else
                  echo "[$(timestamp)] File id '$shard_id' - tensor '$tensor' of qtype: '$qtype' processed (skipped hash verification)!"
                fi
              else
                echo "[$(timestamp)] File id '$shard_id' - tensor '$tensor' of qtype: '$qtype' hash is valid!"
              fi
              # If compression mode is active, compress and remove the .gguf (only keep .gguf.zbst)
              if [[ "$ARCHIVE_COMPRESS" == true ]]; then
                # compress the validated file in-place (in whichever location it currently resides)
                if [[ "$skip_mv" == true ]]; then
                  echo "[$(timestamp)] z-compress validated ${_path} -> ${_path}.zbst"
                  compress_gguf_to_archive "$_path"
                  # After compressing, ensure any lingering .gguf (could be symlinked elsewhere) is removed
                  rm -f "$LOCAL_MODEL_DIR/$shard_file" || true
                else
                  # file is in download dir; compress there and move .zbst to model dir
                  echo "[$(timestamp)] z-compress validated ${_path} in download dir"
                  compress_gguf_to_archive "$_path"
                  mv -f "${_path}.zbst" "$LOCAL_MODEL_DIR/"
                  echo "[$(timestamp)] Saved file id '$shard_id' (zbst) - tensor '$tensor' of qtype: '$qtype'"
                  # remove any .gguf counterpart in model dir (just in case)
                  rm -f "$LOCAL_MODEL_DIR/$shard_file" || true
                fi
              else
                if [[ "$skip_mv" == false ]]; then
                  mv -f "$dl_gguf" "$LOCAL_MODEL_DIR/"
                  echo "[$(timestamp)] Saved file id '$shard_id' - tensor '$tensor' of qtype: '$qtype'"
                else
                  echo "[$(timestamp)] File id '$shard_id' - tensor '$tensor' of qtype: '$qtype' already present in working directory (nothing to do)."
                fi
              fi
            fi
          else
            need_download=true
          fi
        fi

        if [[ "$need_download" == true ]]; then
          echo "[$(timestamp)] Downloading file id '$shard_id' - tensor '$tensor' of qtype: '$qtype' (chunk_id=$chunk_id)"
          until run_downloader_shard "$dl_type" "$chunk_id" "$LOCAL_DOWNLOAD_DIR" "$shard_file"; do
            echo "[$(timestamp)] Download failed; retrying in 10s..."
            sleep 10
          done
          need_download=false
          skip_mv=false
          got=""
        else
          # If no download was needed and we have already handled saving/compressing above, nothing to do.
          :
        fi
      done

      break
    fi
  done
}

# Prepare failure marker
# (FAIL_MARKER already set earlier depending on verify-readonly / normal mode)
rm -f "$FAIL_MARKER" 2>/dev/null || true

# ------------------ VERIFY-ONLY MODE -------------------
if [[ "$VERIFY" == true ]]; then
  echo "[$(timestamp)] VERIFY: verifying existing shards (readonly mode: $VERIFY_READONLY)"

  # helper to cap concurrent jobs
  wait_for_slot() {
    while (( $(jobs -rp | wc -l) >= MAX_JOBS )); do sleep 0.1; done
  }

  SCRIPT_PID=$$

  # Warn about mismatched file types depending on flags:
  # - If -z is used with --verify: verify only .gguf.zbst; warn if any .gguf present (they will be ignored)
  # - If neither -z nor -zd used with --verify: verify only .gguf; warn if any .gguf.zbst present (they will be ignored)
  if [[ "$ARCHIVE_COMPRESS" == true ]]; then
    # Verify only .gguf.zbst
    if comp=$(find "$LOCAL_MODEL_DIR" -maxdepth 1 -name "*-*-of-*.gguf" -print -quit 2>/dev/null || true); then
      if [[ -n "$comp" ]]; then
        echo "⚠️ Warning: found .gguf files in model dir while --verify + -z is used. These will be ignored; verifying only .gguf.zbst files." >&2
      fi
    fi
  else
    # Verify only .gguf
    if comp=$(find "$LOCAL_MODEL_DIR" -maxdepth 1 -name "*-*-of-*.gguf.zbst" -print -quit 2>/dev/null || true); then
      if [[ -n "$comp" ]]; then
        echo "⚠️ Warning: found .gguf.zbst files in model dir while --verify without -z. These will be ignored; verifying only .gguf files." >&2
      fi
    fi
  fi

  # 1) check first shard explicitly, in background
  wait_for_slot
  (
    # Build files list robustly using shell globbing (nullglob) so we don't depend on find -printf portability.
    shopt -s nullglob 2>/dev/null || true
    files=()
    if [[ "$ARCHIVE_COMPRESS" == true ]]; then
      files=( "$LOCAL_MODEL_DIR"/*-*-of-*.gguf.zbst )
    else
      files=( "$LOCAL_MODEL_DIR"/*-*-of-*.gguf )
    fi
    shopt -u nullglob 2>/dev/null || true

    IFS= read -r first <<< "$(printf '%s\n' "${files[@]:-}" | head -n1 || true)"
    if [[ -n "$first" && "$first" =~ -${QTYPE^^}-SPECIAL_TENSOR-([0-9]{5})-of-([0-9]{5})\.gguf(\.zbst)?$ ]]; then
      total="${BASH_REMATCH[2]}"
    else
      total=""
    fi

    gguf_first=""
    # Determine whether we should perform first-shard GPG download/verification under special-node-mode (does it by default for BF16 models)
    should_verify_first=true
    if [[ -n "$SPECIAL_NODE_ID" && (! "$first" =~ "-BF16-" && "${QTYPE^^}" != "BF16") ]]; then
      if ! should_process_chunk 1; then
        should_verify_first=false
      fi
    fi
    # If individual-tensors is enabled and chunk 1 is not in the list, skip first-shard verification
    if [[ "$INDIVIDUAL_TENSORS_ENABLED" == true ]]; then
      if [[ -z "${IND_TENSOR_SET[1]:-}" ]]; then
        should_verify_first=false
      fi
    fi

    if [[ "$should_verify_first" == true ]]; then
      if [[ -n "$first" ]]; then
        gguf_first=$(basename "$(printf '%s\n' "$first" | sed -E "s/-[0-9]{5}-of-$total\.gguf(\.zbst)?$/-00001-of-$total.gguf/")")
        local_first="$LOCAL_MODEL_DIR/$gguf_first"
        local_first_z="${local_first}.zbst"

        # Enforce expectations: when --verify + -z, first shard MUST be .gguf.zbst (no auto-convert/compress)
        if [[ "$ARCHIVE_COMPRESS" == true ]]; then
          if [[ ! -f "$LOCAL_MODEL_DIR/$gguf_first.zbst" && ! -L "$LOCAL_MODEL_DIR/$gguf_first.zbst" ]]; then
            echo "[$(timestamp)] ❌ VERIFY: Expected first shard '${gguf_first}.zbst' not found; when using --verify with -z the first shard must be .gguf.zbst." >&2
            kill -s TERM "$SCRIPT_PID"  # Kill parent
            exit_from_subprocess 5
          fi
          # Verify signature using a temporary decompressed file (gpg requires a file)
          if [[ "$SKIP_GPG" != true ]]; then
            if command -v gpg >/dev/null 2>&1; then
              # create temp file in writable temp workspace (respect verify-readonly)
              if [[ "$VERIFY_READONLY" == true ]]; then
                tmpf="$(mktemp "$VERIFY_TMPDIR/first.XXXXXX.gguf")"
              else
                tmpf="$(mktemp "$GNUPG_TMPDIR/first.XXXXXX.gguf")"
              fi
              if ! decompress_archive_to_file "$LOCAL_MODEL_DIR/$gguf_first.zbst" "$tmpf" skip_symlink_force; then
                echo "[$(timestamp)] ❌ VERIFY: decompression failed for '$LOCAL_MODEL_DIR/$gguf_first.zbst' (data corruption). Maybe you need to specify --z-decompress-opt? Treating as verification failure." >&2
                rm -f "$tmpf"
                kill -s TERM "$SCRIPT_PID"
                touch "$FAIL_MARKER"
                exit_from_subprocess 1
              fi
              if [ ! -f "$LOCAL_MODEL_DIR/$gguf_first.sig" ]; then
                echo "[$(timestamp)] ❌ VERIFY: Error - Signature file '$gguf_first.sig' is missing." >&2
                kill -s TERM "$SCRIPT_PID"
                rm -f "$tmpf"
                exit_from_subprocess 5
              fi
              if safe_gpg_verify "$LOCAL_MODEL_DIR/$gguf_first.sig" "$tmpf" > /dev/null 2>&1; then
                echo "[$(timestamp)] GPG signature verification for '$gguf_first.sig' successful (via temp decompressed file)."
              else
                echo "[$(timestamp)] ❌ VERIFY: Error - GPG signature verification failed for '$gguf_first.sig'." >&2
                rm -f "$tmpf"
                kill -s TERM "$SCRIPT_PID"
                exit_from_subprocess 4
              fi
              rm -f "$tmpf"
            else
              echo "⚠️ Warning: 'gpg' command not found. Signature verification skipped." >&2
            fi
          fi
          echo "[$(timestamp)] OK: ${gguf_first}.zbst"
        else
          # --verify without -z: expect .gguf present
          if [[ ! -f "$LOCAL_MODEL_DIR/$gguf_first" && ! -L "$LOCAL_MODEL_DIR/$gguf_first" ]]; then
            echo "[$(timestamp)] ❌ VERIFY: Expected first shard '$gguf_first' not found; Additional note: when using --verify without -z the first shard must be .gguf." >&2
            kill -s TERM "$SCRIPT_PID"  # Kill parent
            exit_from_subprocess 5
          fi
          if [[ "$SKIP_GPG" != true ]]; then
            if command -v gpg >/dev/null 2>&1; then
              if [ ! -f "$LOCAL_MODEL_DIR/$gguf_first.sig" ]; then
                echo "[$(timestamp)] ❌ VERIFY: Error - Signature file '$gguf_first.sig' is missing." >&2
                kill -s TERM "$SCRIPT_PID"
                exit_from_subprocess 5
              fi
              if safe_gpg_verify "$LOCAL_MODEL_DIR/$gguf_first.sig" "$LOCAL_MODEL_DIR/$gguf_first" > /dev/null 2>&1; then
                echo "[$(timestamp)] GPG signature verification for '$gguf_first.sig' successful."
              else
                echo "[$(timestamp)] ❌ VERIFY: Error - GPG signature verification failed for '$gguf_first.sig'." >&2
                kill -s TERM "$SCRIPT_PID"
                exit_from_subprocess 4
              fi
            else
              echo "⚠️ Warning: 'gpg' command not found. Signature verification skipped." >&2
            fi
          fi
          echo "[$(timestamp)] OK: $gguf_first"
        fi
      else
        echo "[$(timestamp)] ❌ MISSING: $gguf_first"
        touch "$FAIL_MARKER"
      fi
    else
      echo "[$(timestamp)] Skipping first-shard GPG verification due to special-node-id/xxhsum assignment (not BF16 or assigned node) or --individual-tensors selection."
    fi
  ) &

  i=0
  j=0
  lenf=${#TENSORS_TO_FETCH_DYNAMIC[@]}
  lenq=${#TENSORS_TO_QUANTIZE_DYNAMIC[@]}
  if [ "$lenf" -gt 0 ] && [[ "$SKIP_HASH" == true ]] || ! command -v _sha256sum &>/dev/null; then
    if [[ "$SKIP_GGUF_VERIFICATION" == true ]]; then
      [[ "$SKIP_HASH" == true ]] && echo "[$(timestamp)] --skip-hash is enabled. Hash verifications will be skipped." >&2 || echo "⚠️ Warning: _sha256sum command missing. Hash verifications will be skipped! GGUF verification will happen instead." >&2
    else
      lenq=$((lenq + lenf))
      for (( k=${#TENSORS_TO_FETCH_DYNAMIC[@]}-1; k>=0; k-- )); do
        tname="${TENSORS_TO_FETCH_DYNAMIC[$i]}"
        DEBUG "verify with --skip-hash: moving tensor '$tname' at index $k to quantize queue"
        move_fetch_index_to_quantize "$k"
      done
      lenf=0
      [[ "$SKIP_HASH" == true ]] && echo "[$(timestamp)] --skip-hash is enabled. Hash verifications will be skipped. GGUF verification will happen instead." >&2 || echo "⚠️ Warning: _sha256sum command missing. Hash verifications will be skipped!" >&2
    fi
    turn="quantize"
    quantize_verification_only=1
  elif [ "$lenf" -gt 0 ]; then
    turn="fetch"
    quantize_verification_only=0
  else
    turn="quantize"
    quantize_verification_only=1
  fi
  if [ "$lenq" -gt 0 ] && [[ "$SKIP_GGUF_VERIFICATION" == true ]]; then
    echo "[$(timestamp)] --skip-gguf-verification is enabled. GGUF verification will be skipped." >&2
    turn="fetch"
    fetch_verification_only=1
  elif [ "$lenq" -gt 0 ]; then
    fetch_verification_only=0
  else
    fetch_verification_only=1
  fi
  while (( (!quantize_verification_only && i < lenf) || (!fetch_verification_only && j < lenq) )); do
    if ((!quantize_verification_only && !fetch_verification_only)); then
      # If chosen turn has no items left, switch to the other kind (or break if both empty)
      if [[ $turn == "fetch" && i -ge lenf ]]; then
        if (( j < lenq )); then
          turn="quantize"
        else
          break
        fi
      elif [[ $turn == "quantize" && j -ge lenq ]]; then
        if (( i < lenf )); then
          turn="fetch"
        else
          break
        fi
      fi
    fi

    if [[ $turn == "fetch" ]]; then
      # ---- TENSORS_TO_FETCH_DYNAMIC verify block ----
      wait_for_slot
      (
        tensor="${TENSORS_TO_FETCH_DYNAMIC[$i]}"
        #echo "[$(timestamp)] Checking HASH for tensor='$tensor'"
        chunk_id="$(get_shard_id "$tensor")"

        # If individual-tensors is enabled, skip shards not listed
        proceed=true
        if [[ "$INDIVIDUAL_TENSORS_ENABLED" == true ]]; then
          if [[ -z "${IND_TENSOR_SET[$chunk_id]:-}" ]]; then
            proceed=false
          fi
        fi

        # If special node assignment is enabled, skip shards not assigned to this node
        if [[ "$proceed" == true && -n "$SPECIAL_NODE_ID" ]]; then
          if ! should_process_chunk "$chunk_id"; then
            #echo "[$(timestamp)] Skipping HASH of tensor='$tensor' chunk_id=$chunk_id not assigned to this node (xxhsum-based selection)."
            proceed=false
          else
            echo "[$(timestamp)] Tensor='$tensor' chunk_id=$chunk_id assigned to this node (xxhsum-based selection) — proceeding to HASH"
          fi
        fi
        if [[ "$proceed" == true ]]; then
          for l in "${!PATTERNS[@]}"; do
            pat="${PATTERNS[$l]}"
            if [[ "$tensor" =~ $pat ]]; then
              qtype="${PATTERN_QTYPES[$l]^^]}"
              shardfile="${SHARD_FILENAMES_DYNAMIC[$i]}"
              local_gguf="$LOCAL_MODEL_DIR/$shardfile"
              local_z="${local_gguf}.zbst"

              if [[ "$ARCHIVE_COMPRESS" == true ]]; then
                # In verify-only + -z mode: verify .gguf.zbst only (do not alter .gguf files).
                if safe_file_exists "$local_z"; then
                  if should_skip_hash_for "$qtype" "$tensor"; then
                    if [[ "$SKIP_GGUF_VERIFICATION" == false ]]; then
                      DEBUG "verify with should_skip_hash_for: copying tensor '$tname' at index $i to quantize queue"
                      copy_fetch_index_to_quantize "$i" && lenq=$((lenq + 1)) && fetch_verification_only=0
                      echo "[$(timestamp)] SKIP-HASH: ${shardfile}.zbst (skipping stream sha256). GGUF verification will be conducted instead."
                    else
                      echo "[$(timestamp)] SKIP-VERIFICATION: treating ${shardfile}.zbst as valid (skipping stream sha256). No GGUF verification will be conducted (--skip-gguf-verification is enabled)."
                    fi
                  else
                    got=$(safe_stream_sha256_from_z "$local_z" || true)
                    got="${got%%[^0-9a-fA-F]*}"
                    exp="$(get_t_hash "$qtype" "$tensor")"
                    if [[ "$got" != "$exp" ]]; then
                      echo "[$(timestamp)] ❌ WRONG HASH (stream): ${shardfile}.zbst ($got != $exp) - tensor: '$tensor' - qtype: '$qtype'"
                      touch "$FAIL_MARKER"
                    else
                      echo "[$(timestamp)] HASH OK (stream): ${shardfile}.zbst"
                    fi
                  fi
                elif safe_file_exists "$local_gguf"; then
                  # Found .gguf while user requested -z verify-only: warn & treat as missing .zbst (do NOT compress/modify)
                  echo "[$(timestamp)] ⚠️ WARNING: found ${shardfile} (.gguf) but --verify + -z expects ${shardfile}.zbst; treating as MISSING for verification purposes."
                  touch "$FAIL_MARKER"
                else
                  echo "[$(timestamp)] ❌ MISSING: ${shardfile}.zbst"
                  touch "$FAIL_MARKER"
                fi
              else
                # normal verify-only (no -z): verify .gguf only (do not alter .zbst files)
                if safe_file_exists "$local_gguf"; then
                  if should_skip_hash_for "$qtype" "$tensor"; then
                    if [[ "$SKIP_GGUF_VERIFICATION" == false ]]; then
                      DEBUG "verify with should_skip_hash_for: copying tensor '$tname' at index $i to quantize queue"
                      copy_fetch_index_to_quantize "$i" && lenq=$((lenq + 1)) && fetch_verification_only=0
                      echo "[$(timestamp)] SKIP-HASH: ${shardfile} (skipping sha256). GGUF verification will be conducted instead."
                    else
                      echo "[$(timestamp)] SKIP-VERIFICATION: treating ${shardfile} as valid (skipping sha256). No GGUF verification will be conducted (--skip-gguf-verification is enabled)."
                    fi
                  else
                    got=$(safe_sha256sum "$local_gguf" 2>/dev/null || true)
                    got="${got%%[^0-9a-fA-F]*}"
                    exp="$(get_t_hash "$qtype" "$tensor")"
                    if [[ "$got" != "$exp" ]]; then
                      echo "[$(timestamp)] ❌ WRONG HASH: $shardfile ($got != $exp) - tensor: '$tensor' - qtype: '$qtype'"
                      touch "$FAIL_MARKER"
                    else
                      echo "[$(timestamp)] HASH OK: $shardfile"
                    fi
                  fi
                elif safe_file_exists "$local_z"; then
                  # Found .gguf.zbst while user requested non -z verify-only: warn & treat as missing .gguf
                  echo "[$(timestamp)] ⚠️ WARNING: found ${shardfile}.zbst but --verify without -z expects ${shardfile} (.gguf); treating as MISSING for verification purposes."
                  touch "$FAIL_MARKER"
                else
                  echo "[$(timestamp)] ❌ MISSING: $shardfile"
                  touch "$FAIL_MARKER"
                fi
              fi
              break
            fi
          done
        fi
      ) &

      i=$((i+1))
      ((!fetch_verification_only)) && turn="quantize"
    else
      # ---- TENSORS_TO_QUANTIZE_DYNAMIC verify block ----
      wait_for_slot
      (
        tensor="${TENSORS_TO_QUANTIZE_DYNAMIC[$j]}"
        chunk_id="$(get_shard_id "$tensor")"

        # If individual-tensors is enabled, skip shards not listed
        proceed=true
        if [[ "$INDIVIDUAL_TENSORS_ENABLED" == true ]]; then
          if [[ -z "${IND_TENSOR_SET[$chunk_id]:-}" ]]; then
            proceed=false
          fi
        fi

        # If special node assignment is enabled, skip shards not assigned to this node
        if [[ "$proceed" == true && -n "$SPECIAL_NODE_ID" ]]; then
          if ! should_process_chunk "$chunk_id"; then
            #echo "[$(timestamp)] Skipping gguf verification of tensor='$tensor' chunk_id=$chunk_id not assigned to this node (xxhsum-based selection)."
            proceed=false
          else
            echo "[$(timestamp)] Tensor='$tensor' chunk_id=$chunk_id assigned to this node (xxhsum-based selection) — proceeding to gguf verification"
          fi
        fi
        if [[ "$proceed" == true ]]; then
          for l in "${!PATTERNS[@]}"; do
            pat="${PATTERNS[$l]}"
            if [[ "$tensor" =~ $pat ]]; then
              qtype="${PATTERN_QTYPES[$l]^^]}"
              shardfile="${SHARD_QUANTIZE_FILENAMES_DYNAMIC[$j]}"
              local_gguf="$LOCAL_MODEL_DIR/$shardfile"
              local_z="${local_gguf}.zbst"

              if [[ "$ARCHIVE_COMPRESS" == true ]]; then
                # In verify-only + -z mode: verify .gguf.zbst only (do not alter .gguf files).
                if safe_file_exists "$local_z"; then
                  if ! safe_stream_check_quantized_gguf_from_z "$local_z" "$chunk_id" "$qtype"; then
                    echo "⚠️ INVALID GGUF SHARD (stream): ${shardfile}.zbst - tensor: '$tensor' - qtype: '$qtype'"
                    touch "$FAIL_MARKER"
                  else
                    echo "[$(timestamp)] GGUF-VERIFICATION OK (stream): ${shardfile}.zbst"
                  fi
                elif safe_file_exists "$local_gguf"; then
                  # Found .gguf while user requested -z verify-only: warn & treat as missing .zbst (do NOT compress/modify)
                  echo "[$(timestamp)] ⚠️ WARNING: found ${shardfile} (.gguf) but --verify + -z expects ${shardfile}.zbst; treating as MISSING for verification purposes."
                  touch "$FAIL_MARKER"
                else
                  echo "[$(timestamp)] ❌ MISSING: ${shardfile}.zbst"
                  touch "$FAIL_MARKER"
                fi
              else
                # normal verify-only (no -z): verify .gguf only (do not alter .zbst files)
                if safe_file_exists "$local_gguf"; then
                  if ! check_quantized_gguf "$local_gguf" "$chunk_id" "$qtype"; then
                    echo "⚠️ INVALID GGUF SHARD: ${shardfile} - tensor: '$tensor' - qtype: '$qtype'"
                    touch "$FAIL_MARKER"
                  else
                    echo "[$(timestamp)] GGUF-VERIFICATION OK: ${shardfile}"
                  fi
                elif safe_file_exists "$local_z"; then
                  # Found .gguf.zbst while user requested non -z verify-only: warn & treat as missing .gguf
                  echo "[$(timestamp)] ⚠️ WARNING: found ${shardfile}.zbst but --verify without -z expects ${shardfile} (.gguf); treating as MISSING for verification purposes."
                  touch "$FAIL_MARKER"
                else
                  echo "[$(timestamp)] ❌ MISSING: $shardfile"
                  touch "$FAIL_MARKER"
                fi
              fi
              break
            fi
          done
        fi
      ) &

      j=$((j+1))
      ((!$quantize_verification_only)) && turn="fetch"
    fi
  done

  # wait for all verifications to finish
  wait

  # Cleanup
  if [[ "$SKIP_GPG" != true ]]; then
    if command -v gpg >/dev/null 2>&1; then
      [ -n "$GNUPG_TMPDIR" ] && rm -rf "$GNUPG_TMPDIR"
    fi
  fi
  # Remove qtype marker files
  if [[ "$VERIFY_READONLY" != true ]]; then
    rm -f "$LOCAL_DOWNLOAD_DIR"/.qtype_zbst_* 2>/dev/null || true
  fi

  # summary
  if [[ -f "$FAIL_MARKER" ]]; then
    if [[ "$SKIP_HASH" == true ]]; then
      echo "[$(timestamp)] ❌ VERIFY: some files missing"
    else
      echo "[$(timestamp)] ❌ VERIFY: some files missing or with hash mismatch or invalid gguf"
    fi
    exit 1
  else
    if [[ "$SKIP_HASH" == true ]]; then
      echo "[$(timestamp)] ✅ VERIFY: all files present"
    else
      echo "[$(timestamp)] ✅ VERIFY: all files present and with valid hashes/gguf"
    fi
    rm -f "$FAIL_MARKER" 2>/dev/null || true
    exit 0
  fi
fi

# ----------- Handle computed-map quantization-from-bf16 before starting downloads ----------
# Initialize the quantization-from-bf16 workflow when --llama-quantize is specified 
# For each computed qtype in COMPUTED_QTYPES, check the target shards (all SHARD_FILENAMES/TENSORS_TO_FETCH
# that match the qtype). For any shard that does not exist locally, create it by quantizing from bf16.
# If the shard already exists, skip unless REQUANTIZE_QUANTIZEONLY_SHARDS==true.
if [[ -n "$LLAMA_QUANTIZE_BIN" ]]; then
  echo "[$(timestamp)] Note: detected --llama-quantize. Preparing quantize-from-bf16 workflow."

  # Pre-check: if any missing shard exists and we need to quantize, ensure llama-quantize is usable
  needs_llama=false
  for idx in "${!TENSORS_TO_FETCH[@]}"; do
    tensor="${TENSORS_TO_FETCH[$idx]}"
    chunk_id="$(get_shard_id "$tensor")"
    shard_file="${SHARD_FILENAMES[$idx]}"

    # Determine which pattern matched to get the qtype for this tensor
    matched_q=""
    for i in "${!PATTERNS[@]}"; do
      if [[ "$tensor" =~ ${PATTERNS[$i]} ]]; then
        matched_q="${PATTERN_QTYPES[$i],,}"
        break
      fi
    done
    if [[ -z "$matched_q" ]]; then
      continue
    fi

    if [[ -n "${COMPUTED_QTYPES[${matched_q,,}]:-}" ]]; then
      # This qtype was computed. Check if local/download files exist
      if safe_file_exists "$LOCAL_MODEL_DIR/$shard_file" || safe_file_exists "$LOCAL_DOWNLOAD_DIR/$shard_file" || safe_file_exists "$LOCAL_MODEL_DIR/$shard_file.zbst" || safe_file_exists "$LOCAL_DOWNLOAD_DIR/$shard_file.zbst"; then
        if [[ "$REQUANTIZE_QUANTIZEONLY_SHARDS" == true ]]; then
          needs_llama=true
        else
          # file exists and we're not requantizing -> skip
          :
        fi
      else
        # missing file -> we will need to create via quantization
        needs_llama=true
      fi
    fi
  done

  if [[ "$needs_llama" == true ]]; then
    # Validate llama-quantize presence/support
    require_llama_quantize_support
    echo "[$(timestamp)] llama-quantize binary validated: $LLAMA_QUANTIZE_BIN"
  fi
  
  # Ensure BF16_DOWNLOAD_DIR exists
  ensure_bf16_workspace

  # Ensure bf16 first chunk exists (nested downloader will produce it)
  if ! download_bf16_tensor_via_nested 1; then
    echo "❌ Error: failed to obtain bf16 first chunk required to quantize from bf16" >&2
    exit 1
  fi
fi
# ----------- END quantize-from-bf16 pre-processing ----------

# ------------------ MAIN LOOP (dynamic queues: retry-until-success wrappers, quantize-on-failure) -------------------

# Cleanup any leftover marker files from previous runs
# Use nullglob so that globs that find nothing don't expand into literal patterns.
shopt -s nullglob
rm -f -- "$LOCAL_DOWNLOAD_DIR"/.failed_download.* 2>/dev/null || true
rm -f -- "$LOCAL_DOWNLOAD_DIR"/.quantize.* 2>/dev/null || true
rm -f -- "$LOCAL_DOWNLOAD_DIR"/.failed_quantize.* 2>/dev/null || true
shopt -u nullglob

# Alternation toggle: when both queues non-empty we alternate one fetch, one quantize
alternate_fetch_first=true  # start with fetch (you can change if desired)
toggle="$alternate_fetch_first"

# The loop runs while there are items in either dynamic queue
while ( ((${#TENSORS_TO_FETCH_DYNAMIC[@]} > 0)) || ( ((${#TENSORS_TO_QUANTIZE_DYNAMIC[@]} > 0)) ) ); do

  # First, process any failed-download markers that child download wrappers may have created.
  # This moves their corresponding tensors into the quantize queue.
  process_failed_download_markers || true

  # If any failed quantization markers are present, bail out
  if failed_quantize_detected; then
    echo "❌ Error: one or more quantizations failed (detected $LOCAL_DOWNLOAD_DIR/.failed_quantize.*). Aborting." >&2
    # Show which failed quantize markers exist
    shopt -s nullglob
    for f in "$LOCAL_DOWNLOAD_DIR"/.failed_quantize.*; do
      echo "  failed quantize marker: $f" >&2
    done
    shopt -u nullglob
    exit 1
  fi

  # Decide whether to progress a fetch item or a quantize item:
  # - If quantize queue is non-empty and there's no quantize lock AND either toggle says quantize next or fetch queue is empty,
  #   then attempt to start one quantize.
  # - Otherwise attempt to start a fetch wrapper (download).
  start_quantize_this_round=false
  if ((${#TENSORS_TO_QUANTIZE_DYNAMIC[@]} > 0)); then
    if quantize_slot_available; then
      # If fetch list is empty, always allow quantize processing (but in that case we must use wait_for_slot_quantize())
      if ((${#TENSORS_TO_FETCH_DYNAMIC[@]} == 0)); then
        start_quantize_this_round=true
      else
        # both non-empty: alternate behavior
        if [[ "$toggle" == false ]]; then
          start_quantize_this_round=true
        fi
      fi
    fi
  fi

  if $start_quantize_this_round; then
    # Starting a quantization thread for the first item in the quantize-dynamic list.
    # Use wait_for_slot or wait_for_slot_quantize depending on whether there are other fetch items remaining.
    if ((${#TENSORS_TO_FETCH_DYNAMIC[@]} == 0)); then
      # Only quantize items remain; use mutex waiter to avoid busy-loop.
      wait_for_slot_quantize
    else
      # We still have fetch items and quantization must be exclusive (but quantize thread will touch .quantize.*),
      # still ensure we don't spawn an absurd number of background jobs (use wait_for_slot).
      wait_for_slot
    fi

    # Pop first item from TENSORS_TO_QUANTIZE_DYNAMIC
    qidx=0
    tensor_to_quantize="${TENSORS_TO_QUANTIZE_DYNAMIC[$qidx]}"
    shard_file_to_quantize="${SHARD_QUANTIZE_FILENAMES_DYNAMIC[$qidx]}"
    # remove from the quantize-dynamic arrays (shift)
    unset 'TENSORS_TO_QUANTIZE_DYNAMIC[qidx]'
    unset 'SHARD_QUANTIZE_FILENAMES_DYNAMIC[qidx]'
    TENSORS_TO_QUANTIZE_DYNAMIC=( "${TENSORS_TO_QUANTIZE_DYNAMIC[@]}" )
    SHARD_QUANTIZE_FILENAMES_DYNAMIC=( "${SHARD_QUANTIZE_FILENAMES_DYNAMIC[@]}" )

    # Compute chunk id for markers and logging
    quant_chunk_id="$(get_shard_id "$tensor_to_quantize")"

    local_gguf="$LOCAL_MODEL_DIR/$shard_file_to_quantize"
    dl_gguf="$LOCAL_DOWNLOAD_DIR/$shard_file_to_quantize"
    local_z="${local_gguf}.zbst"
    dl_z="${dl_gguf}.zbst"

    # If individual-tensors list is enabled, skip shards not in set.
    if [[ "$INDIVIDUAL_TENSORS_ENABLED" == true ]]; then
      if [[ -z "${IND_TENSOR_SET[$quant_chunk_id]:-}" ]]; then
        echo "[$(timestamp)] Skipping tensor='$tensor_to_quantize' chunk_id=$quant_chunk_id because it's not in --individual-tensors list."
        continue
      fi
    fi

    # If special node assignment is enabled, skip shards not assigned to this node
    if [[ -n "$SPECIAL_NODE_ID" ]]; then
      if ! should_process_chunk "$quant_chunk_id"; then
        _del=""
        [[ "$RM_SKIPPED_SHARDS" == true ]] && _del=" and deleting (if present)" && rm -f "$dl_gguf" "$local_gguf" "$dl_z" "$local_z" || true
        echo "[$(timestamp)] Skipping$_del tensor='$tensor_to_quantize' chunk_id=$quant_chunk_id not assigned to this node (xxhsum-based selection)."
        continue
      else
        echo "[$(timestamp)] Tensor='$tensor_to_quantize' chunk_id=$quant_chunk_id assigned to this node (xxhsum-based selection) — proceeding to quantization"
      fi
    else
      echo "[$(timestamp)] Tensor='$tensor_to_quantize' chunk_id=$quant_chunk_id quantization — proceeding"
    fi

    # Create quantize lock marker
    touch "$LOCAL_DOWNLOAD_DIR/.quantize.${quant_chunk_id}" || { echo "❌ Error: could not create quantize marker for $quant_chunk_id" >&2; exit 1; }

    DEBUG "starting quantize thread for chunk=${quant_chunk_id} shard_file=${shard_file_to_quantize}"

    (
      # This subshell performs quantization for the given chunk. If it fails, it must create a .failed_quantize.<chunkid> marker.
      set -e
      # Determine target qtype for this tensor
      target_q=""
      for i in "${!PATTERNS[@]}"; do
        if [[ "$tensor_to_quantize" =~ ${PATTERNS[$i]} ]]; then
          target_q="${PATTERN_QTYPES[$i]}"
          break
        fi
      done
      if [[ -z "$target_q" ]]; then
        echo "❌ Error: quantize thread saw tensor with no matching target_q; skipping: $tensor_to_quantize" >&2
        touch "$LOCAL_DOWNLOAD_DIR/.failed_quantize.${quant_chunk_id}" 2>/dev/null || true
        rm -f -- "$LOCAL_DOWNLOAD_DIR/.quantize.${quant_chunk_id}" 2>/dev/null || true
        exit 0
      fi

      skip_mv=true
      if [[ "$REQUANTIZE_QUANTIZEONLY_SHARDS" == true ]]; then
        rm -f "$dl_gguf" "$local_gguf" "$dl_z" "$local_z" || true
        sync || true
        skip_mv=false
        echo "[$(timestamp)] Re-quantizing existing \"$target_q\" quantized-shard (user requested --requantize-quantizeonly-shards): $shard_file_to_quantize"
        # Ensure LLAMA_QUANTIZE_BIN set and usable (already validated above if needs_llama true)
      fi

      _path=""
      # If ARCHIVE_DECOMPRESS true: if a .zbst exists, decompress to .gguf (overwrite) so script works on .gguf
      if [[ "$ARCHIVE_DECOMPRESS" == true ]]; then
        if safe_file_exists "$local_z"; then
          echo "[$(timestamp)] z-decompress: found $local_z -> decompressing to $local_gguf (overwrite)"
          if ! decompress_archive_to_file "$local_z" "$local_gguf"; then
            echo "[$(timestamp)] ⚠️ decompression failed for $local_z — treating as corrupted and will redownload." >&2
          fi
          rm -f "$local_z" || true
        elif safe_file_exists "$dl_z"; then
          echo "[$(timestamp)] z-decompress: found $dl_z in download dir -> decompressing to $local_gguf (overwrite)"
          if ! decompress_archive_to_file "$dl_z" "$local_gguf"; then
            echo "[$(timestamp)] ⚠️ decompression failed for $dl_z in download dir — treating as corrupted and will redownload." >&2
          fi
          rm -f "$dl_z" || true
        fi
      fi

      shard_id=$(echo "$shard_file_to_quantize" | sed -E 's/.*-([0-9]{5})-of-[0-9]{5}\.gguf/\1/')

      # If ARCHIVE_COMPRESS or ARCHIVE_DECOMPRESS is enabled, prefer .gguf.zbst when present and validate from stream
      if safe_file_exists "$local_z" || safe_file_exists "$dl_z"; then
        if [[ "$ARCHIVE_COMPRESS" == true ]]; then
          if safe_file_exists "$local_z"; then
            _path="$local_z"
            skip_mv=true
          else
            _path="$dl_z"
            skip_mv=false
          fi

          if [[ "$SKIP_GGUF_VERIFICATION" == false ]] && ! safe_stream_check_quantized_gguf_from_z "$_path" "$shard_id" "$target_q"; then
            echo "⚠️ Warning: Compressed quantized file id '$shard_id' is already present but appears different than expected. Re-quantization is necessary!" >&2
            rm -f "$dl_z" "$local_z" || true
            sync || true
          else
            # If the valid zbst was in download dir, move to model dir
            if [[ "$skip_mv" == false ]]; then
              mv -f "$dl_z" "$LOCAL_MODEL_DIR/"
              echo "[$(timestamp)] Saved quantized file id '$shard_id' (zbst) - tensor '$tensor_to_quantize' of qtype: '$target_q'"
            else
              echo "[$(timestamp)] Quantized file id '$shard_id' (zbst) - tensor '$tensor_to_quantize' of qtype: '$target_q' already present in working directory (nothing to do)."
            fi
            # Ensure we remove any corresponding .gguf if it exists (we keep only .gguf.zbst in compress mode)
            if safe_file_exists "$LOCAL_MODEL_DIR/$shard_file_to_quantize"; then
              rm -f "$LOCAL_MODEL_DIR/$shard_file_to_quantize" || true
              echo "[$(timestamp)] Removed corresponding .gguf for '${shard_file_to_quantize}' because quantized .gguf.zbst is present and valid"
            fi
            # Quantize succeeded: remove lock marker and exit
            rm -f -- "$LOCAL_DOWNLOAD_DIR/.quantize.${quant_chunk_id}" 2>/dev/null || true
            exit 0
          fi

        elif [[ "$ARCHIVE_DECOMPRESS" == true ]]; then
          if safe_file_exists "$local_z"; then
            echo "[$(timestamp)] z-decompress: found quantized $local_z -> decompressing to $local_gguf (overwrite)"
            if ! decompress_archive_to_file "$local_z" "$local_gguf"; then
              echo "[$(timestamp)] ⚠️ decompression failed for quantized $local_z — treating as corrupted and will re-quantize." >&2
            fi
            rm -f "$local_z" || true
          elif safe_file_exists "$dl_z"; then
            echo "[$(timestamp)] z-decompress: found quantized $dl_z in download dir -> decompressing to $local_gguf (overwrite)"
            if ! decompress_archive_to_file "$dl_z" "$local_gguf"; then
              echo "[$(timestamp)] ⚠️ decompression failed for quantized $dl_z in download dir — treating as corrupted and will re-quantize." >&2
            fi
            rm -f "$dl_z" || true
          fi
          # Proceed with .gguf present in $local_gguf, assuming decompression was successful
        fi
      fi

      # If we didn't find a valid zbst (or compress mode not enabled) or if decompress mode was enabled, check .gguf files
      if safe_file_exists "$local_gguf" || safe_file_exists "$dl_gguf"; then
        if safe_file_exists "$local_gguf"; then
          _path=$local_gguf
        else
          _path=$dl_gguf
          skip_mv=false
        fi
        if [[ "$SKIP_GGUF_VERIFICATION" == false ]] && ! check_quantized_gguf "$_path" "$shard_id" "$target_q"; then
          echo "⚠️ Warning: Quantized file id '$shard_id' is already present but appears different than expected. Re-quantization is necessary!" >&2
            rm -f "$dl_gguf" "$local_gguf" || true
            sync || true
        else
          if [[ "$ARCHIVE_COMPRESS" == true ]]; then
            # compress the validated file in-place (in whichever location it currently resides)
            if [[ "$skip_mv" == true ]]; then
              echo "[$(timestamp)] z-compress validated quantized file ${_path} -> ${_path}.zbst"
              compress_gguf_to_archive "$_path"
              # After compressing, ensure any lingering .gguf (could be symlinked elsewhere) is removed
              rm -f "$local_gguf" || true
            else
              # file is in download dir; compress there and move .zbst to model dir
              echo "[$(timestamp)] z-compress validated quantized file ${_path} in download dir"
              compress_gguf_to_archive "$_path"
              mv -f "${_path}.zbst" "$LOCAL_MODEL_DIR/"
              echo "[$(timestamp)] Saved quantized file id '$shard_id' (zbst) - tensor '$tensor_to_quantize' of qtype: '$target_q'"
              # remove any .gguf counterpart in model dir (just in case)
              rm -f "$local_gguf" || true
            fi
          elif [[ "$skip_mv" == false ]]; then
            mv -f "$dl_gguf" "$LOCAL_MODEL_DIR/"
            echo "[$(timestamp)] Saved quantized file id '$shard_id' - tensor '$tensor_to_quantize' of qtype: '$target_q'"
          else
            echo "[$(timestamp)] Quantized file id '$shard_id' - tensor '$tensor_to_quantize' of qtype: '$target_q' already present in working directory (nothing to do)."
          fi
          # Quantize succeeded: remove lock marker and exit
          rm -f -- "$LOCAL_DOWNLOAD_DIR/.quantize.${quant_chunk_id}" 2>/dev/null || true
          exit 0
        fi
      fi

      echo "[$(timestamp)] Quantized shard missing (or corrupted) -> will quantize to \"$target_q\" from bf16: $shard_file_to_quantize"

      # Perform quantization for this chunk_id into expected filename.
      # Determine gguf_first filename needed by quantize routine
      num_shards=${#SHARD_FILENAMES_FULL[@]}
      total=$(printf "%05d" "$((num_shards + 1))")
      gguf_first="$(printf '%s' "${SHARD_FILENAMES_FULL[0]}" | sed -E "s/-[^-]+-SPECIAL_TENSOR-[0-9]{5}-of-[0-9]{5}\.gguf$/-BF16-SPECIAL_TENSOR-00001-of-${total}.gguf/")"

      # Ensure bf16 chunk exists (nested downloader will produce it)
      if ! download_bf16_tensor_via_nested "$quant_chunk_id"; then
        echo "❌ Error: failed to obtain bf16 chunk ${quant_chunk_id} required to quantize ${shard_file_to_quantize}" >&2
        # Mark failed quantize and exit subshell
        touch "$LOCAL_DOWNLOAD_DIR/.failed_quantize.${quant_chunk_id}" 2>/dev/null || true
        rm -f -- "$LOCAL_DOWNLOAD_DIR/.quantize.${quant_chunk_id}" 2>/dev/null || true
        exit_from_subprocess 30
      fi

      # Run quantize: generate output filename as basename matching expected shard_file
      if ! quantize_tensor_from_bf16 "$quant_chunk_id" "$shard_file_to_quantize" "$target_q" "$BF16_DOWNLOAD_DIR/$gguf_first"; then
        echo "❌ Error: quantization failed for chunk ${quant_chunk_id} -> ${shard_file_to_quantize}" >&2
        touch "$LOCAL_DOWNLOAD_DIR/.failed_quantize.${quant_chunk_id}" 2>/dev/null || true
        rm -f -- "$LOCAL_DOWNLOAD_DIR/.quantize.${quant_chunk_id}" 2>/dev/null || true
        exit_from_subprocess 31
      else
        if safe_file_exists "$LOCAL_DOWNLOAD_DIR/$shard_file_to_quantize"; then
          if [[ "$SKIP_GGUF_VERIFICATION" == false ]]; then
            if ! check_quantized_gguf "$LOCAL_DOWNLOAD_DIR/$shard_file_to_quantize" "$quant_chunk_id" "$target_q"; then
              echo "❌ INVALID GGUF SHARD: ${shard_file_to_quantize} - tensor: '$tensor_to_quantize' - qtype: '$target_q'" >&2
              # Mark failed quantize and exit subshell
              touch "$LOCAL_DOWNLOAD_DIR/.failed_quantize.${quant_chunk_id}" 2>/dev/null || true
              rm -f -- "$LOCAL_DOWNLOAD_DIR/.quantize.${quant_chunk_id}" 2>/dev/null || true
              exit_from_subprocess 32
            else
              echo "[$(timestamp)] GGUF-VERIFICATION OK: ${shard_file_to_quantize}"
            fi
          else
            echo "[$(timestamp)] SKIP-VERIFICATION: treating ${shard_file_to_quantize} as valid. No GGUF verification will be conducted (--skip-gguf-verification is enabled)."
          fi
        else
          echo "❌ Error: quantization didn't produce a file for chunk ${quant_chunk_id} -> ${shard_file_to_quantize}" >&2
          touch "$LOCAL_DOWNLOAD_DIR/.failed_quantize.${quant_chunk_id}" 2>/dev/null || true
          rm -f -- "$LOCAL_DOWNLOAD_DIR/.quantize.${quant_chunk_id}" 2>/dev/null || true
          exit_from_subprocess 38
        fi
      fi

      # Removing the bf16 version of the shard (unless user requested to keep bf16)
      # Extract shard suffix strictly (must match at end)
      shard_suffix="$(printf '%s\n' "$shard_file_to_quantize" | grep -oE -- '-[0-9]{5}-of-[0-9]{5}\.gguf$')" || shard_suffix=""
      if [[ -n "$shard_suffix" ]]; then
          shopt -s nullglob 2>/dev/null || true
          matches=( "$BF16_DOWNLOAD_DIR"/*"$shard_suffix" )
          shopt -u nullglob 2>/dev/null || true
          if [[ ${#matches[@]} -eq 1 ]]; then
              if [[ "$QUANTIZE_KEEP_BF16" == true ]]; then
                  echo "[$(timestamp)] Keeping bf16 version of ${shard_file_to_quantize} due to --quantize-keep-bf16"
              else
                  echo "[$(timestamp)] Cleanup: removing bf16 version of ${shard_file_to_quantize}"
                  rm -f -- "${matches[0]}"
              fi
          else
              echo "[$(timestamp)] Cleanup skipped: expected 1 match, found ${#matches[@]}"
          fi
      else
          echo "[$(timestamp)] Cleanup skipped: invalid shard suffix"
      fi

      # After successful quantization, move output to LOCAL_MODEL_DIR root
      if safe_file_exists "$LOCAL_DOWNLOAD_DIR/$shard_file_to_quantize"; then
        # If user requested archive compression (-z), compress the quantized .gguf into .gguf.zbst
        if [[ "$ARCHIVE_COMPRESS" == true ]]; then
          echo "[$(timestamp)] z-compress: compressing quantized output $LOCAL_DOWNLOAD_DIR/$shard_file_to_quantize"
          if ! compress_gguf_to_archive "$LOCAL_DOWNLOAD_DIR/$shard_file_to_quantize"; then
            echo "❌ Error: compression of quantized output failed for '$shard_file_to_quantize'." >&2
            touch "$LOCAL_DOWNLOAD_DIR/.failed_quantize.${quant_chunk_id}" 2>/dev/null || true
            rm -f -- "$LOCAL_DOWNLOAD_DIR/.quantize.${quant_chunk_id}" 2>/dev/null || true
            exit_from_subprocess 33
          fi
          # Move the produced .zbst to model dir
          if safe_file_exists "$LOCAL_DOWNLOAD_DIR/${shard_file_to_quantize}.zbst"; then
            mv -f "$LOCAL_DOWNLOAD_DIR/${shard_file_to_quantize}.zbst" "$LOCAL_MODEL_DIR/" || { echo "❌ Error: failed to move compressed quantized output to $LOCAL_MODEL_DIR" >&2; touch "$LOCAL_DOWNLOAD_DIR/.failed_quantize.${quant_chunk_id}" 2>/dev/null || true; rm -f -- "$LOCAL_DOWNLOAD_DIR/.quantize.${quant_chunk_id}" 2>/dev/null || true; exit_from_subprocess 34; }
            echo "[$(timestamp)] Quantized tensor ${quant_chunk_id} saved to $LOCAL_MODEL_DIR/${shard_file_to_quantize}.zbst"
          else
            echo "❌ Error: expected compressed quantized output '$LOCAL_DOWNLOAD_DIR/${shard_file_to_quantize}.zbst' not found after compression." >&2
            touch "$LOCAL_DOWNLOAD_DIR/.failed_quantize.${quant_chunk_id}" 2>/dev/null || true
            rm -f -- "$LOCAL_DOWNLOAD_DIR/.quantize.${quant_chunk_id}" 2>/dev/null || true
            exit_from_subprocess 35
          fi
        else
          # No compression requested -> move .gguf directly
          mv -f "$LOCAL_DOWNLOAD_DIR/$shard_file_to_quantize" "$LOCAL_MODEL_DIR/$shard_file_to_quantize" || { echo "❌ Error: failed to move quantized output to $LOCAL_MODEL_DIR" >&2; touch "$LOCAL_DOWNLOAD_DIR/.failed_quantize.${quant_chunk_id}" 2>/dev/null || true; rm -f -- "$LOCAL_DOWNLOAD_DIR/.quantize.${quant_chunk_id}" 2>/dev/null || true; exit_from_subprocess 36; }
          echo "[$(timestamp)] Quantized tensor ${quant_chunk_id} saved to $LOCAL_MODEL_DIR/${shard_file_to_quantize}"
        fi
      else
        echo "❌ Error: expected quantized output '$LOCAL_DOWNLOAD_DIR/${shard_file_to_quantize}' not found after quantize." >&2
        touch "$LOCAL_DOWNLOAD_DIR/.failed_quantize.${quant_chunk_id}" 2>/dev/null || true
        rm -f -- "$LOCAL_DOWNLOAD_DIR/.quantize.${quant_chunk_id}" 2>/dev/null || true
        exit_from_subprocess 37
      fi

      echo "[$(timestamp)] Quantized ${shard_file_to_quantize} produced and stored in $LOCAL_DOWNLOAD_DIR."

      # Quantize succeeded: remove lock marker and exit
      rm -f -- "$LOCAL_DOWNLOAD_DIR/.quantize.${quant_chunk_id}" 2>/dev/null || true
      exit 0
    ) &
    # record pid optionally for debug (no persistent bookkeeping)
    qpid=$!
    DEBUG "spawned quantize pid=$qpid for chunk=${quant_chunk_id}"

    # Flip the alternation toggle when both lists are non-empty
    if ((${#TENSORS_TO_FETCH_DYNAMIC[@]} > 0)) && ((${#TENSORS_TO_QUANTIZE_DYNAMIC[@]} > 0)); then
      # flip toggle so next iteration picks the other queue
      if [[ "$toggle" == true ]]; then toggle=false; else toggle=true; fi
    fi

    # Immediately continue main loop (we will process fetch items in next iteration)
    continue
  fi

  # Otherwise, attempt to start a fetch wrapper (download) for the first fetch-dynamic item
  if ((${#TENSORS_TO_FETCH_DYNAMIC[@]} > 0)); then
    wait_for_slot

    # Pop first item from fetch-dynamic (FIFO)
    idx=0
    current_tensor="${TENSORS_TO_FETCH_DYNAMIC[$idx]}"
    current_shard="${SHARD_FILENAMES_DYNAMIC[$idx]}"
    # Keep values for this subshell and compute chunk
    current_chunk="$(get_shard_id "$current_tensor")"

    # Start the download wrapper as background job
    (
      attempts=0

      # Each wrapper will keep trying until download_shard returns success (0) OR until QUANTIZE_FAILED_DOWNLOAD limit is reached (if set)
      while true; do
        attempts=$((attempts + 1))

        if download_shard "$current_tensor" "$current_shard"; then
          DEBUG "child tensor='${current_tensor}' chunk='${current_chunk}' download_shard succeeded (attempt $attempts)"
          break
        else
          rc=$?
        fi

        # if QUANTIZE_FAILED_DOWNLOAD is set, enforce attempt limit
        if [[ "${QUANTIZE_FAILED_DOWNLOAD}" != "" ]] && (( attempts >= QUANTIZE_FAILED_DOWNLOAD )); then
          echo "[$(timestamp)] ⚠️ Warning: Tensor='${current_tensor}' chunk_id=${current_chunk} download failed with exit $rc (attempt $attempts, reached QUANTIZE_FAILED_DOWNLOAD=${QUANTIZE_FAILED_DOWNLOAD}). Marking for quantization." >&2
          # create failed marker so parent main loop will move this item into quantize queue
          touch "$LOCAL_DOWNLOAD_DIR/.failed_download.${current_chunk}" 2>/dev/null || true
          break
        fi

        # otherwise keep retrying
        echo "[$(timestamp)] ⚠️ Warning: Tensor='${current_tensor}' chunk_id=${current_chunk} download failed with exit $rc (attempt $attempts). Retrying in 10s..." >&2
        sleep 10
      done
    ) &

    # Record child's pid for debug
    pid=$!
    DEBUG "spawned download wrapper for tensor='${current_tensor}' chunk='${current_chunk}' pid=$pid"

    # After spawning a download wrapper, remove that item from the fetch-dynamic arrays
    # because its retrieval is now in-flight (we don't want to spawn duplicate retrievals).
    unset 'TENSORS_TO_FETCH_DYNAMIC[idx]'
    unset 'SHARD_FILENAMES_DYNAMIC[idx]'
    TENSORS_TO_FETCH_DYNAMIC=( "${TENSORS_TO_FETCH_DYNAMIC[@]}" )
    SHARD_FILENAMES_DYNAMIC=( "${SHARD_FILENAMES_DYNAMIC[@]}" )

    # Flip alternation toggle if both queues are non-empty
    if ((${#TENSORS_TO_FETCH_DYNAMIC[@]} > 0)) && ((${#TENSORS_TO_QUANTIZE_DYNAMIC[@]} > 0)); then
      if [[ "$toggle" == true ]]; then toggle=false; else toggle=true; fi
    fi

    # Small pause to let children potentially create .failed_download.* markers quickly (helps prompt movement)
    sleep 0.01

    # Immediately after spawning children, process any failed-download markers created by quick-failing wrappers
    process_failed_download_markers || true

    # Loop continues
    continue
  fi

  # If we reached here, no fetch items were started and no quantize started; sleep briefly to avoid busy-loop
  sleep 0.1
done

# Wait for all background wrapper & quantize jobs to finish
rc=0
set +e
if ! wait; then
  status=$?
  echo "❌ Error: one or more background children exited with non-zero status (wait returned $status). Check child stderr logs for details." >&2
  rc=1
fi
set -e

if (( rc != 0 )); then
  echo "❌ Error: one or more background children exited abnormally." >&2
  exit $rc
fi

# Final cleanup: remove any stray .quantize.* / .failed_download.* / .failed_quantize.* if present
shopt -s nullglob
rm -f -- "$LOCAL_DOWNLOAD_DIR"/.failed_download.* 2>/dev/null || true
rm -f -- "$LOCAL_DOWNLOAD_DIR"/.quantize.* 2>/dev/null || true
rm -f -- "$LOCAL_DOWNLOAD_DIR"/.failed_quantize.* 2>/dev/null || true
shopt -u nullglob

# ------------------ END MAIN LOOP (dynamic queues implemented) -------------------

# ------------- FINAL FIRST-SHARD FETCH (non-verify) -----
# Decide which files to consider for first-shard detection depending on z mode
# Use shell globbing (nullglob) for portability and predictable behavior.
shopt -s nullglob 2>/dev/null || true
if [[ "$ARCHIVE_COMPRESS" == true ]]; then
  files=( "$LOCAL_MODEL_DIR"/*-*-of-*.gguf.zbst )
elif [[ "$ARCHIVE_DECOMPRESS" == true ]]; then
  # prefer already-decompressed .gguf if present, otherwise accept .gguf.zbst
  files=( "$LOCAL_MODEL_DIR"/*-*-of-*.gguf "$LOCAL_MODEL_DIR"/*-*-of-*.gguf.zbst )
else
  files=( "$LOCAL_MODEL_DIR"/*-*-of-*.gguf )
fi
shopt -u nullglob 2>/dev/null || true

IFS= read -r first <<< "$(printf '%s\n' "${files[@]:-}" | head -n1 || true)"
if [[ -n "$first" && "$first" =~ -${QTYPE^^}-SPECIAL_TENSOR-([0-9]{5})-of-([0-9]{5})\.gguf(\.zbst)?$ ]]; then
  total="${BASH_REMATCH[2]}"
  gguf_first=$(basename "$(printf '%s\n' "$first" | sed -E "s/-[0-9]{5}-of-$total\.gguf(\.zbst)?$/-00001-of-$total.gguf/")")
else
  # Attempt to build file name from .map file instead
  num_shards=${#SHARD_FILENAMES_FULL[@]}
  total="$(printf "%05d" "$((num_shards + 1))")"
  gguf_first=$(basename "$(printf '%s\n' "${SHARD_FILENAMES_FULL[0]}" | sed -E "s/-[0-9]{5}-of-$total\.gguf(\.zbst)?$/-00001-of-$total.gguf/")")
fi

# Determine whether we should perform first-shard GPG download/verification under special-node-mode (does it by default for BF16 models)
should_verify_first=true
_del=""
if [[ -n "$SPECIAL_NODE_ID" && (! "$first" =~ "-BF16-" && "${QTYPE^^}" != "BF16") ]]; then
  if ! should_process_chunk 1; then
    [[ "$RM_SKIPPED_SHARDS" == true ]] && _del=" and deleting (if present)" && rm -f "$LOCAL_MODEL_DIR/$gguf_first" "$LOCAL_DOWNLOAD_DIR/$gguf_first" "$LOCAL_MODEL_DIR/$gguf_first.sig" "$LOCAL_DOWNLOAD_DIR/$gguf_first.sig" || true
    should_verify_first=false
  fi
fi

# If individual-tensors is enabled and 1 isn't included, skip verifying/downloading first shard here
if [[ "$INDIVIDUAL_TENSORS_ENABLED" == true && -z "${IND_TENSOR_SET[1]:-}" ]]; then
  should_verify_first=false
fi

if [[ "$should_verify_first" == true ]]; then
  if [[ "$VERIFY" != true ]]; then
    echo "[$(timestamp)] Fetching first shard separately"
    if [[ "$total" != "" ]] && [[ "$gguf_first" != "" ]]; then
      if [[ "$FORCE_REDOWNLOAD" == true ]]; then
        echo "[$(timestamp)] Force redownload: removing existing first shard (and gpg signature)"
        rm -f "$LOCAL_MODEL_DIR/$gguf_first" "$LOCAL_DOWNLOAD_DIR/$gguf_first" || true
        rm -f "$LOCAL_MODEL_DIR/$gguf_first.sig" "$LOCAL_DOWNLOAD_DIR/$gguf_first.sig" || true
        rm -f "$LOCAL_MODEL_DIR/$gguf_first.zbst" "$LOCAL_DOWNLOAD_DIR/$gguf_first.zbst" || true
        sync || true
      fi
      if ! [ -f "$LOCAL_MODEL_DIR/$gguf_first" ] && ! [ -f "$LOCAL_MODEL_DIR/$gguf_first.zbst" ]; then
        until run_downloader_shard "${QTYPE}" 1 "$LOCAL_DOWNLOAD_DIR" "$(basename "$gguf_first")"; do
          echo "[$(timestamp)] First shard download failed; retrying in 10s..."
          sleep 10
        done

        # Move whichever file was produced (.gguf or .gguf.zbst)
        if safe_file_exists "$LOCAL_DOWNLOAD_DIR/$(basename "$gguf_first")"; then
          mv -f "$LOCAL_DOWNLOAD_DIR/$(basename "$gguf_first")" "$LOCAL_MODEL_DIR/"
        elif safe_file_exists "$LOCAL_DOWNLOAD_DIR/$(basename "$gguf_first").zbst"; then
          mv -f "$LOCAL_DOWNLOAD_DIR/$(basename "$gguf_first").zbst" "$LOCAL_MODEL_DIR/"
        else
          echo "[$(timestamp)] ❌ Error: expected first shard in download dir but none found after download." >&2
          touch "$FAIL_MARKER"
          exit 1
        fi

        echo "[$(timestamp)] First shard saved"
        if [[ "$SKIP_GPG" != true ]]; then
          until run_downloader_shard "${QTYPE}" -2 "$LOCAL_DOWNLOAD_DIR" "$(basename "$gguf_first.sig")"; do
            echo "[$(timestamp)] First shard signature download failed; retrying in 10s..."
            sleep 10
          done

          # Move file that was procurred .gguf.sig
          if safe_file_exists "$LOCAL_DOWNLOAD_DIR/$(basename "$gguf_first.sig")"; then
            mv -f "$LOCAL_DOWNLOAD_DIR/$(basename "$gguf_first.sig")" "$LOCAL_MODEL_DIR/"
          else
            echo "[$(timestamp)] ❌ Error: expected first shard signature in download dir but none found after download." >&2
            touch "$FAIL_MARKER"
            exit 1
          fi

          echo "[$(timestamp)] First shard gpg signature saved"
        fi
      else
        echo "[$(timestamp)] First shard already exists"
        if [[ "$SKIP_GPG" != true ]]; then
          if ! [ -f "$LOCAL_MODEL_DIR/$gguf_first.sig" ]; then
            until run_downloader_shard "${QTYPE}" -2 "$LOCAL_DOWNLOAD_DIR" "$(basename "$gguf_first.sig")"; do
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

      # After ensuring first shard and signature present, perform GPG verification if required.
      if [[ "$SKIP_GPG" != true ]]; then
        if command -v gpg >/dev/null 2>&1; then
          # We'll attempt verification and on corruption/verify failure (when not verify-readonly) we will redownload and retry.
          # Prefer verifying the actual .gguf file if present (this avoids trying to decompress a missing/non-updated .gguf.zbst).
          if safe_file_exists "$LOCAL_MODEL_DIR/$gguf_first"; then
            # verify using the .gguf that we just ensured is present
            while :; do
              if [ ! -f "$LOCAL_MODEL_DIR/$gguf_first.sig" ]; then
                echo "[$(timestamp)] ⚠️ Signature file '$gguf_first.sig' is missing — treating as corrupted and will redownload." >&2
                if attempt_redownload_first; then
                  continue
                else
                  echo "❌ Error: Signature file '$gguf_first.sig' is missing." >&2
                  [ -n "$GNUPG_TMPDIR" ] && rm -rf "$GNUPG_TMPDIR"
                  touch "$FAIL_MARKER"
                  exit 5
                fi
              fi
              if safe_gpg_verify "$LOCAL_MODEL_DIR/$gguf_first.sig" "$LOCAL_MODEL_DIR/$gguf_first" > /dev/null 2>&1; then
                echo "[$(timestamp)] First shard gpg signature verification successful."
                break
              else
                echo "[$(timestamp)] ⚠️ GPG signature verification failed for '$gguf_first.sig' — treating as corrupted and will redownload." >&2
                if attempt_redownload_first; then
                  continue
                else
                  echo "❌ Error: GPG signature verification failed for '$gguf_first.sig'." >&2
                  [ -n "$GNUPG_TMPDIR" ] && rm -rf "$GNUPG_TMPDIR"
                  touch "$FAIL_MARKER"
                  exit 4
                fi
              fi
            done
          elif safe_file_exists "$LOCAL_MODEL_DIR/$gguf_first.zbst"; then
            # If only a .zbst exists, decompress to temp (or to local model directory if -zd) and verify.
            while :; do
              if [[ "$VERIFY_READONLY" == true ]]; then
                tmpf="$(mktemp "$VERIFY_TMPDIR/first.XXXXXX.gguf")"
              elif [[ "$ARCHIVE_DECOMPRESS" == true ]] || ([[ "$ARCHIVE_COMPRESS" == true ]] && [[ ! -f "$LOCAL_MODEL_DIR/$gguf_first.zbst" ]]); then
                tmpf="$LOCAL_MODEL_DIR/$gguf_first"
              else
                tmpf="$(mktemp "$GNUPG_TMPDIR/first.XXXXXX.gguf")"
              fi

              if safe_file_exists "$LOCAL_MODEL_DIR/$gguf_first.zbst" && ! decompress_archive_to_file "$LOCAL_MODEL_DIR/$gguf_first.zbst" "$tmpf" skip_symlink_force; then
                echo "[$(timestamp)] ⚠️ decompression failed for '$LOCAL_MODEL_DIR/$gguf_first.zbst' (data corruption) — treating as corrupted and will redownload." >&2
                rm -f "$tmpf"
                if attempt_redownload_first; then
                  # redownloaded, retry loop
                  continue
                else
                  # cannot redownload (readonly) -> mark failure and exit
                  echo "❌ Error: decompression failed for '$LOCAL_MODEL_DIR/$gguf_first.zbst' (data corruption)." >&2
                  [ -n "$GNUPG_TMPDIR" ] && rm -rf "$GNUPG_TMPDIR"
                  touch "$FAIL_MARKER"
                  exit 1
                fi
              fi

              if [ ! -f "$LOCAL_MODEL_DIR/$gguf_first.sig" ]; then
                echo "[$(timestamp)] ⚠️ Signature file '$gguf_first.sig' is missing — treating as corrupted and will redownload." >&2
                rm -f "$tmpf"
                if attempt_redownload_first; then
                  continue
                else
                  echo "❌ Error: Signature file '$gguf_first.sig' is missing." >&2
                  [ -n "$GNUPG_TMPDIR" ] && rm -rf "$GNUPG_TMPDIR"
                  touch "$FAIL_MARKER"
                  exit 5
                fi
              fi

              if safe_gpg_verify "$LOCAL_MODEL_DIR/$gguf_first.sig" "$tmpf" > /dev/null 2>&1; then
                if [[ "$tmpf" != "$LOCAL_MODEL_DIR/$gguf_first" ]]; then
                  echo "[$(timestamp)] First shard gpg signature verification succeeded (via temp decompressed file)."
                  rm -f "$tmpf"
                else
                  echo "[$(timestamp)] First shard gpg signature verification succeeded."
                fi
                # If both .gguf and .gguf.zbst existed, remove .gguf to keep only .gguf.zbst in compress mode
                if [[ "$ARCHIVE_COMPRESS" == true && -f "$LOCAL_MODEL_DIR/$gguf_first" ]]; then
                  if [ ! -f "$LOCAL_MODEL_DIR/$gguf_first.zbst" ]; then
                    echo "[$(timestamp)] z-compress validated \"$LOCAL_MODEL_DIR/$gguf_first\""
                    compress_gguf_to_archive "$LOCAL_MODEL_DIR/$gguf_first"
                  else
                    rm -f "$LOCAL_MODEL_DIR/$gguf_first" || true
                    echo "[$(timestamp)] Removed corresponding .gguf for '$gguf_first' because .gguf.zbst is present and valid"
                  fi
                fi
                break
              else
                echo "[$(timestamp)] ⚠️ First shard gpg signature verification failed for '$gguf_first.sig'." >&2
                rm -f "$tmpf"
                if attempt_redownload_first; then
                  continue
                else
                  echo "❌ Error: GPG signature verification failed for '$gguf_first.sig'." >&2
                  [ -n "$GNUPG_TMPDIR" ] && rm -rf "$GNUPG_TMPDIR"
                  touch "$FAIL_MARKER"
                  exit 4
                fi
              fi
            done
          else
            echo "❌ Error: unable to find previous shards..." >&2
            touch "$FAIL_MARKER"
          fi
        else
          echo "⚠️ Warning: 'gpg' command not found. Signature verification skipped." >&2
        fi
      fi

      # After verification, if compression is enabled, ensure only .zbst remains (compress if necessary)
      if [[ "$ARCHIVE_COMPRESS" == true ]]; then
        if safe_file_exists "$LOCAL_MODEL_DIR/$gguf_first"; then
          echo "[$(timestamp)] z-compress verified first shard to .zbst"
          compress_gguf_to_archive "$LOCAL_MODEL_DIR/$gguf_first"
        fi
      fi

    else
      echo "[$(timestamp)] Skipping$_del first-shard signature download/verification due to special-node-id/xxhsum assignment (not BF16 or assigned node) or --individual-tensors selection."
    fi
  fi
else
  echo "[$(timestamp)] Skipping$_del first-shard signature download/verification due to special-node-id/xxhsum assignment (not BF16 or assigned node) or --individual-tensors selection."
fi

# ------------- FINAL VERIFICATION & SHARD SEQUENCE --------
# If individual-tensors mode is enabled the user requested skipping full sequence verification.
if [[ "$INDIVIDUAL_TENSORS_ENABLED" == true ]]; then
  echo "[$(timestamp)] Skipping full shard sequence verification because --individual-tensors was provided."
else
  echo "[$(timestamp)] Verifying shard sequence completeness"
  # Build full_indices from the files list we assembled above (respecting z modes)
  full_indices=()
  if [[ "$ARCHIVE_COMPRESS" == true ]]; then
    # only consider .gguf.zbst for completeness
    shopt -s nullglob 2>/dev/null || true
    files=( "$LOCAL_MODEL_DIR"/*-*-of-*.gguf.zbst )
    shopt -u nullglob 2>/dev/null || true
  elif [[ "$ARCHIVE_DECOMPRESS" == true ]]; then
    # if z-decompress, prefer .gguf files (we expect decompressed .gguf). If .gguf absent, consider .gguf.zbst
    shopt -s nullglob 2>/dev/null || true
    files=( "$LOCAL_MODEL_DIR"/*-*-of-*.gguf )
    if [[ ${#files[@]} -eq 0 ]]; then
      files=( "$LOCAL_MODEL_DIR"/*-*-of-*.gguf.zbst )
    fi
    shopt -u nullglob 2>/dev/null || true
  else
    shopt -s nullglob 2>/dev/null || true
    files=( "$LOCAL_MODEL_DIR"/*-*-of-*.gguf )
    shopt -u nullglob 2>/dev/null || true
  fi

  for f in "${files[@]}"; do
    base=$(basename "$f")
    if [[ "$base" =~ -${QTYPE^^}-SPECIAL_TENSOR-([0-9]{5})-of-([0-9]{5})\.gguf(\.zbst)?$ ]]; then
      full_indices+=( "${BASH_REMATCH[1]}" )
      # capture total if not already set
      if [[ -z "${total:-}" ]]; then
        total="${BASH_REMATCH[2]}"
      fi
    fi
  done

  # sort the indices
  if [[ ${#full_indices[@]} -gt 0 ]]; then
    IFS=$'\n' full_indices=($(printf "%s\n" "${full_indices[@]}" | sort))
  else
    echo "❌ Error: no shard files found to verify." >&2
    touch "$FAIL_MARKER"
    exit 1
  fi

  # Convert first/last from the full list (zero-padded strings)
  full_first_z="${full_indices[0]}"
  full_last_z="${full_indices[-1]}"
  full_first=$((10#$full_first_z))
  full_last=$((10#$full_last_z))

  if [[ -z "$SPECIAL_NODE_ID" ]]; then
    # unchanged legacy behavior when not using special-node mode:
    last_index=${full_last_z}
    first_index=${full_first_z}
    count_expected=$((10#$last_index - 10#$first_index + 1))

    if [[ ${#full_indices[@]} -ne $count_expected ]]; then
      echo "❌ Error - $((count_expected - ${#full_indices[@]})) missing shard(s) between $first_index and $last_index. Verify recipe or rerun." >&2
      echo "Missing indices:"
      seq -f "%05g" "$full_first" "$full_last" \
        | grep -Fvx -f <(printf "%s\n" "${full_indices[@]}")
      touch "$FAIL_MARKER"
      exit 1
    fi

    echo "[$(timestamp)] All shards from ${first_index} to ${last_index} are present."
  else
    # SPECIAL_NODE_ID mode: compute which indices this node SHOULD process
    expected_for_node=()
    # iterate the *full* range and test each index with should_process_chunk
    while IFS= read -r idx_z; do
      # convert to integer for should_process_chunk
      if should_process_chunk "$idx_z"; then
        expected_for_node+=("$idx_z")
      fi
    done < <(seq -f "%05g" "$full_first" "$full_last")

    # Build actual present-for-node list by filtering full_indices with should_process_chunk
    present_for_node=()
    for idx_z in "${full_indices[@]}"; do
      if should_process_chunk "$idx_z"; then
        present_for_node+=("$idx_z")
      fi
    done

    count_expected=${#expected_for_node[@]}
    count_present=${#present_for_node[@]}

    if (( count_present != count_expected )); then
      # find missing expected entries (expected - present)
      echo "❌ Error - $((count_expected - count_present)) missing shard(s) assigned to this node between ${full_first_z} and ${full_last_z}. Verify recipe or rerun." >&2
      echo "Missing indices (zero-padded):"
      # print expected list and subtract present list
      printf "%s\n" "${expected_for_node[@]}" | grep -Fvx -f <(printf "%s\n" "${present_for_node[@]}")
      touch "$FAIL_MARKER"
      exit 1
    fi

    # For logging, show the node-specific first/last
    first_index="${expected_for_node[0]}"
    last_index="${expected_for_node[-1]}"
    echo "[$(timestamp)] All assigned shards for this node present: ${first_index} .. ${last_index} (count=${count_present})"
  fi
fi

if [[ "$SKIP_GPG" != true ]]; then
  # Decide whether to run final GPG verification for first-shard
  should_verify_first=true
  if [[ -n "$SPECIAL_NODE_ID" && (! "$first" =~ "-BF16-" && "${QTYPE^^}" != "BF16") ]]; then
    if ! should_process_chunk 1; then
      should_verify_first=false
    fi
  fi
  if [[ "$INDIVIDUAL_TENSORS_ENABLED" == true && -z "${IND_TENSOR_SET[1]:-}" ]]; then
    should_verify_first=false
  fi
  if [[ "$should_verify_first" == true ]]; then
    if command -v gpg >/dev/null 2>&1; then
      if ([[ "$ARCHIVE_COMPRESS" == false ]] && [ ! -f "$LOCAL_MODEL_DIR/$gguf_first" ]) || ([[ "$ARCHIVE_COMPRESS" == true ]] && [ ! -f "$LOCAL_MODEL_DIR/$gguf_first.zbst" ]); then
          echo "❌ Error: Shard file '$gguf_first' not found." >&2
          [ -n "$GNUPG_TMPDIR" ] && rm -rf "$GNUPG_TMPDIR"
          touch "$FAIL_MARKER"
          exit 5
      fi
      if [ ! -f "$LOCAL_MODEL_DIR/$gguf_first.sig" ]; then
          echo "❌ Error: Signature file '$gguf_first.sig' is missing." >&2
          echo "Hint: To skip GPG verification, re-run this script with the --skip-gpg option." >&2
          [ -n "$GNUPG_TMPDIR" ] && rm -rf "$GNUPG_TMPDIR"
          touch "$FAIL_MARKER"
          exit 5
      fi

      # If .zbst exists and we're in compress mode, decompress to a temp file for gpg verify
      if safe_file_exists "$LOCAL_MODEL_DIR/$gguf_first.zbst"; then
        # create temp file in writable temp workspace (respect verify-readonly)
        if [[ "$VERIFY_READONLY" == true ]]; then
          tmpf="$(mktemp "$VERIFY_TMPDIR/first.XXXXXX.gguf")"
        else
          tmpf="$(mktemp "$GNUPG_TMPDIR/first.XXXXXX.gguf")"
        fi
        if ! decompress_archive_to_file "$LOCAL_MODEL_DIR/$gguf_first.zbst" "$tmpf" skip_symlink_force; then
          echo "❌ Error: decompression failed for '$LOCAL_MODEL_DIR/$gguf_first.zbst' (data corruption)." >&2
          rm -f "$tmpf"
          [ -n "$GNUPG_TMPDIR" ] && rm -rf "$GNUPG_TMPDIR"
          touch "$FAIL_MARKER"
          exit 1
        fi
        if safe_gpg_verify "$LOCAL_MODEL_DIR/$gguf_first.sig" "$tmpf" > /dev/null 2>&1; then
            echo "[$(timestamp)] GPG signature verification for '$gguf_first.sig' successful (via temp decompressed file)."
            # If both .gguf and .gguf.zbst existed, remove .gguf to keep only .gguf.zbst in compress mode
            if [[ "$ARCHIVE_COMPRESS" == true && -f "$LOCAL_MODEL_DIR/$gguf_first" ]]; then
              rm -f "$LOCAL_MODEL_DIR/$gguf_first" || true
              echo "[$(timestamp)] Removed corresponding .gguf for '$gguf_first' because .gguf.zbst is present and valid"
            fi
        else
            echo "❌ Error: GPG signature verification failed for '$gguf_first.sig'." >&2
            rm -f "$tmpf"
            [ -n "$GNUPG_TMPDIR" ] && rm -rf "$GNUPG_TMPDIR"
            touch "$FAIL_MARKER"
            exit 4
        fi
        rm -f "$tmpf"
        # If z-decompress requested, convert to regular .gguf permanently
        if [[ "$ARCHIVE_DECOMPRESS" == true ]]; then
          if ! decompress_archive_to_file "$LOCAL_MODEL_DIR/$gguf_first.zbst" "$LOCAL_MODEL_DIR/$gguf_first"; then
            echo "❌ Error: decompression failed while converting '$LOCAL_MODEL_DIR/$gguf_first.zbst' -> '$LOCAL_MODEL_DIR/$gguf_first' (data corruption)." >&2
            rm -f "$LOCAL_MODEL_DIR/$gguf_first" || true
            touch "$FAIL_MARKER"
            exit 1
          fi
          rm -f "$LOCAL_MODEL_DIR/$gguf_first.zbst" || true
          echo "[$(timestamp)] z-decompress: converted first shard .zbst -> .gguf"
        fi
      else
        if safe_gpg_verify "$LOCAL_MODEL_DIR/$gguf_first.sig" "$LOCAL_MODEL_DIR/$gguf_first" > /dev/null 2>&1; then
            echo "[$(timestamp)] GPG signature verification for '$gguf_first.sig' successful."
        else
            echo "❌ Error: GPG signature verification failed for '$gguf_first.sig'." >&2
            [ -n "$GNUPG_TMPDIR" ] && rm -rf "$GNUPG_TMPDIR"
            touch "$FAIL_MARKER"
            exit 4
        fi
      fi
    else
      echo "⚠️ Warning: 'gpg' command not found. Signature verification skipped." >&2
    fi
  else
    echo "[$(timestamp)] Skipping final first-shard GPG verification due to special-node-id/xxhsum assignment (not BF16 or assigned node) or --individual-tensors selection."
  fi
  [ -n "$GNUPG_TMPDIR" ] && rm -rf "$GNUPG_TMPDIR"
fi

# Remove qtype marker files
rm -f "$LOCAL_DOWNLOAD_DIR"/.qtype_zbst_* 2>/dev/null || true

# Remove bf16 directory unless keeping it was requested and only if it is $LOCAL_DOWNLOAD_DIR/bf16
if [[ "$QUANTIZE_KEEP_BF16" == false ]]; then
  if [[ "$BF16_DOWNLOAD_DIR" == "$LOCAL_DOWNLOAD_DIR/bf16" ]]; then
    rm -rf "$BF16_DOWNLOAD_DIR" 2>/dev/null || true
  else
    echo "⚠️ Custom BF16 directory used for quantization not removed for safety reasons. Please consider removing it manually if needed." >&2
  fi
fi

echo
rm -f "$FAIL_MARKER" 2>/dev/null || true
if [[ "${SKIP_FINAL_MESSAGE:-false}" != true ]]; then
  echo "✅ Download and verification complete. Enjoy!"
fi
