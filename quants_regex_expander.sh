#!/usr/bin/env bash
#***************************************************************#
#** This script is part of Thireus' GGUF Tool Suite.          **#
#** quants_regex_expander.sh is a basic tool to expand tensor **#
#** regex for troubleshooting purpose.                        **#
#**                                                           **#
#** ********************************************************* **#
#** --------------- Updated: Dec-27-2025 -------------------- **#
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
#** Copyright © 2025 - Thireus.             Gᵣₑₑ𝒹ᵧ ₛₜₑₐₗₜ₉₁ 𝒻ₜ𝓌 **#
#***************************************************************#
#**PLEASE REFER TO THE README FILE FOR ADDITIONAL INFORMATION!**#
#***************************************************************#

set -euo pipefail

# Toggle debug by exporting DEBUG=1
_debug() {
  [[ "${DEBUG:-0}" -ne 1 ]] && return
  printf '[DEBUG] %s\n' "$*" >&2
}

# echo "$(for f in `ls DeepSeek-R1-0528-DQ4_K_R4-*.gguf`; do gguf_info.py "$f"; done)" | grep 'dtype=' | awk -F $'\t' '{print $1 "=" $3}' | sed 's/=dtype=/=/g' | sed 's/\./\\./g'
custom="
## Quant mix recipe created using Thireus' GGUF Tool Suite - https://gguf.thireus.com/
# Model name: Kimi-K2-Thinking
# Link to the original model: https://huggingface.co/moonshotai/Kimi-K2-Thinking

## Model head & embeddings — qbits: 32 8 
^output_norm\.weight$=f32
^output\.weight$=q8_0
^token_embd\.weight$=q8_0

## Special attention kernels — single-quant only (llama-quantize takes care of it) — qbits: 8 
^blk\.([0-9]|[1-5][0-9]|60)\.attn_k_b\.weight$=q8_0

## Multi-headed attention parameters — qbits: 32 8 6 
^blk\.([0-9]|[1-5][0-9]|60)\.attn_norm\.weight$=f32
^blk\.(39|4[0-2]|4[4-9]|5[0-4]|56|58)\.attn_kv_a_mqa\.weight$=iq6_k
^blk\.(39|4[0-2]|4[4-9]|5[0-4]|56|58)\.attn_q_a\.weight$=iq6_k
^blk\.(39|4[0-2]|4[4-9]|5[0-4]|56|58)\.attn_q_b\.weight$=iq6_k
^blk\.([0-9]|[1-2][0-9]|3[0-8]|43|55|57|59|60)\.attn_kv_a_mqa\.weight$=q8_0
^blk\.([0-9]|[1-2][0-9]|3[0-8]|43|55|57|59|60)\.attn_output\.weight$=q8_0
^blk\.([0-9]|[1-2][0-9]|3[0-8]|43|55|57|59|60)\.attn_q_a\.weight$=q8_0
^blk\.([0-9]|[1-5][0-9]|60)\.attn_q_a_norm\.weight$=f32
^blk\.(39|4[0-2]|4[4-9]|5[0-4]|56|58)\.attn_v_b\.weight$=iq6_k
^blk\.([0-9]|[1-2][0-9]|3[0-8]|43|55|57|59|60)\.attn_q_b\.weight$=q8_0
^blk\.([0-9]|[1-2][0-9]|3[0-8]|43|55|57|59|60)\.attn_v_b\.weight$=q8_0
^blk\.(39|4[0-2]|4[4-9]|5[0-4]|56|58)\.attn_output\.weight$=iq6_k
^blk\.([0-9]|[1-5][0-9]|60)\.attn_kv_a_norm\.weight$=f32

## Dense Feed-Forward Network weights — qbits: 6 
^blk\.0\.ffn_up\.weight$=iq6_k
^blk\.0\.ffn_down\.weight$=iq6_k

## MoE Gating & Routing — qbits: 32 
^blk\.([1-9]|[1-5][0-9]|60)\.ffn_gate_inp\.weight$=f32
^blk\.([1-9]|[1-5][0-9]|60)\.exp_probs_b\.bias$=f32

## Gating network — qbits: 6 
^blk\.0\.ffn_gate\.weight$=iq6_k

## Misc / Other tensors — qbits: 32 
^blk\.([0-9]|[1-5][0-9]|60)\.ffn_norm\.weight$=f32

## GPU-loaded - MoE Shared Experts Feed-Forward Network - ffn_*_shexp
# ffn_down_shexp — down-projection (shared experts) — qbits: 8 6 5 
^blk\.60\.ffn_down_shexp\.weight$=q8_0
^blk\.([1-9]|[1-3][0-9]|4[0-5]|4[7-9]|5[0-9])\.ffn_down_shexp\.weight$=iq6_k
^blk\.46\.ffn_down_shexp\.weight$=iq5_k_r4

# ffn_up_shexp — up-projection (shared experts) — qbits: 8 6 5 
^blk\.60\.ffn_up_shexp\.weight$=q8_0
^blk\.([1-9]|[1-3][0-9]|4[0-5]|4[7-9]|5[0-9])\.ffn_up_shexp\.weight$=iq6_k
^blk\.46\.ffn_up_shexp\.weight$=iq5_k_r4

# ffn_gate_shexp — gating network (shared experts) — qbits: 8 6 5 
^blk\.60\.ffn_gate_shexp\.weight$=q8_0
^blk\.([1-9]|[1-3][0-9]|4[0-5]|4[7-9]|5[0-9])\.ffn_gate_shexp\.weight$=iq6_k
^blk\.46\.ffn_gate_shexp\.weight$=iq5_k_r4

## CPU-friendly - MoE Per-expert Feed-Forward Network - ffn_*_exps
# ffn_down_exps — down-projection (per-expert) — qbits: 2 1 
^blk\.([2-9]|1[0-9]|22|3[0-9]|4[0-8])\.ffn_down_exps\.weight$=iq2_kt
^blk\.(1|2[0-1]|2[3-9]|49|5[0-9]|60)\.ffn_down_exps\.weight$=iq1_kt

# ffn_up_exps — up-projection (per-expert) — qbits: 2 1 
^blk\.([2-9]|1[0-4]|3[5-9]|4[0-9]|51)\.ffn_up_exps\.weight$=iq2_kt
^blk\.(1|1[5-9]|2[0-9]|3[0-4]|50|5[2-9]|60)\.ffn_up_exps\.weight$=iq1_kt

# ffn_gate_exps — gating network (per-expert) — qbits: 2 1 
^blk\.([2-9]|1[0-4]|3[5-9]|4[0-9]|51)\.ffn_gate_exps\.weight$=iq2_kt
^blk\.(1|1[5-9]|2[0-9]|3[0-4]|50|5[2-9]|60)\.ffn_gate_exps\.weight$=iq1_kt

## Summary of tensor sizes per class
# GPU Total: 11.03 GiB (91.5%) | 12.05 GiB max, if all were q8_0 | 8.11 GiB min, if all were iq5_k_r4
# CPU Total: 230.34 GiB (91.8%) | 251.02 GiB max, if all were iq2_kt | 206.72 GiB min, if all were iq1_kt
# GPU+CPU Total: 241.38 GiB (91.7%)

## Summary of tensor counts and bpw per qtype
#
# GPU-loaded quants:
# QTYPE		Count	BPW	Assigned GiB	% Assigned	Max GiB (all)
# +f32       	365	32    	  0.62 GiB	-		-
# *+q8_0    	61 	8.5   	  0.25 GiB	-		-
# q8_0      	225	8.5   	  6.59 GiB	58.9%		11.18
# iq6_k     	262	6.625 	  3.55 GiB	40.7%		8.72
# iq5_k_r4  	3  	5.5   	  0.03 GiB	0.4%		7.24
#
# CPU-friendly quants:
# QTYPE		Count	BPW	Assigned GiB	% Assigned	Max GiB (all)
# iq2_kt    	96 	2.125 	133.88 GiB	53.3%		251.02
# iq1_kt    	84 	1.75  	 96.47 GiB	46.7%		206.72
#
# -Average BPW: 2.0201
#
# -Notes:
# - '+' means user-defined pre-assigned tensors, or tensor missing from csv data or f32 tensors
# - '*' means fallback tensors: these tensors were present in the map(s) with a different dtype than the originally-intended qtype;
#   they have been grouped and displayed as '*<qtype>' above to show the final (map-observed) qtype and sizes separately.
# - WARNING: 61 tensor assignments were substituted to the dtype actually present in their tensor map files. 
#   This may change the final size relative to the expected thresholds and chosen quants. 
#   To disable automatic map-based fallbacks and preserve the script's original assigned qtypes exactly, re-run with --no-fallback.
# - Recipe produced on the 2025-12-24 10:48:41 UTC+0000 using Thireus' GGUF tools (https://gguf.thireus.com/)
# - Script SHA-256: 569b7f6a3239c9173d71ca1fadf34222607d72a2cfed2c284b42633e95b4a627
# - Calibration dataset 'kld_results.csv' SHA-256: b5a7341cb88828e49793a106f43de1312681162500304922e26ab9cc30292203
# - tensors.bf16.map SHA-256: b92fda9c9fcee52780a84dbc1772dbb9b967a3ca5901d981258257bdd5bf30ca
# - tensors.bf16.map model name: Kimi-K2-Thinking-THIREUS-BF16-SPECIAL_TENSOR-01097-of-01097
# - tensors.q8_0.map SHA-256: 4b8106b4c534dfa9474a9881ce1959328e1bbb142307aa0e89bbcf8a6d58d10e
# - tensors.q8_0.map model name: Kimi-K2-Thinking-THIREUS-Q8_0-SPECIAL_TENSOR-01097-of-01097
# - tensors.iq6_k.map SHA-256: bbc3a96641b51468cccb20b9647abe6a7daa7c4cbf139d8a353b713d8446a9e6
# - tensors.iq6_k.map model name: Kimi-K2-Thinking-THIREUS-IQ6_K-SPECIAL_TENSOR-01097-of-01097
# - tensors.iq5_k_r4.map SHA-256: be395c01a3dbc2fc43d7020bdb18c4148e25971325874046d36b451618378ffb
# - tensors.iq5_k_r4.map model name: Kimi-K2-Thinking-THIREUS-IQ5_K_R4-SPECIAL_TENSOR-01097-of-01097
# - tensors.iq2_kt.map SHA-256: baa743da76e757d237f1145d258c7505d6345c1ccd098d939ecf24c178cf8070
# - tensors.iq2_kt.map model name: Kimi-K2-Thinking-THIREUS-IQ2_KT-SPECIAL_TENSOR-01097-of-01097
# - tensors.iq1_kt.map SHA-256: be45178bc1c81f002e2f8a490cfad46ef4d771d2240a2a7a0de2e4279be56faa
# - tensors.iq1_kt.map model name: Kimi-K2-Thinking-THIREUS-IQ1_KT-SPECIAL_TENSOR-01097-of-01097
# - GPG signatures: PASSED
# - Command used:
# ../../quant_assign.py kld_results.csv --tolerance 0.01 --cpu-irq-k 1.5 --gpu-irq-k 1.5 --gpu-assign-qtype iq6_k \
# --cpu-tensors-max-size 230 --gpu-tensors-max-size 95% --exponential-factor 8 --cpu-tensors \
# 'blk\..*\.ffn_down_exps\.weight' 'blk\..*\.ffn_up_exps\.weight' 'blk\..*\.ffn_gate_exps\.weight' --gpu-tensors '.*' \
# --cpu-quants iq2_kt iq1_kt --gpu-quants q8_0 iq5_k_r4 iq6_k --harmonize-tensors \
# '^blk\..*\.ffn_up_exps.*,blk\..*\.ffn_gate_exps.*' --harmonization-technique 3

## THE END!
"

# ---------------------------------------------------------------------------------------------
# expand_ranges(): read stdin, separate regex range entries if not supported by llama-quantize
# ---------------------------------------------------------------------------------------------
expand_ranges() {
  while IFS= read -r input; do
    local prefix body suffix

    # Check for parentheses
    if [[ "$input" =~ \(.*\) ]]; then
      prefix="${input%%(*}"
      body="${input#*\(}"
      body="${body%\)*}"
      suffix="${input##*\)}"
    else
      prefix=""
      body="${input}"
      suffix=""
    fi

    # Convert [a-b] to {a..b}
    body_expanded=$(echo "$body" | sed -E 's/\[([0-9]+)-([0-9]+)\]/{\1..\2}/g')

    # Always split on | and process each line
    echo "$body_expanded" | tr '|' '\n' | while IFS= read -r part; do
      # Use brace expansion only if there are braces
      if [[ "$part" == *\{*..*\}* ]]; then
        # Escape backslashes before eval
        escaped_part="${part//\\/\\\\}"
        eval "expanded=( $escaped_part )"
        for e in "${expanded[@]}"; do
          printf '%s%s%s\n' "$prefix" "$e" "$suffix"
        done
      else
        # No brace expansion needed
        printf '%s%s%s\n' "$prefix" "$part" "$suffix"
      fi
    done
  done
}

echo "$custom" | grep -v '^#' | grep -v '^$' | expand_ranges
