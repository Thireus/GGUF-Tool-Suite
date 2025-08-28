#!/usr/bin/env bash
#***************************************************************#
#** This script is part of Thireus' GGUF Tool Suite.          **#
#** imatrix_Qwen3-4B-Thinking-2507.sh used to create the      **#
#** imatrix file.                                             **#
#**                                                           **#
#** ********************************************************* **#
#** --------------- Updated: Aug-25-2025 -------------------- **#
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
#** Copyright © 2025 - Thireus.        Cₕₐₜᵦₒₜₛ ₙₑₑ𝒹 ₜₕₑᵣₐₚᵧ ₜₒₒ **#
#***************************************************************#
#**PLEASE REFER TO THE README FILE FOR ADDITIONAL INFORMATION!**#
#***************************************************************#

set -euo pipefail

# Fixed argument list too long and too many open files
ulimit -S -s unlimited
ulimit -n 99999

curl -L https://gist.githubusercontent.com/ubergarm/edfeb3ff9c6ec8b49e88cdf627b0711a/raw/ba5b01b6960a86874592f5913e283746ff734483/ubergarm-imatrix-calibration-corpus-v02.txt -o ubergarm-imatrix-calibration-corpus-v02.txt

# See instructions on https://github.com/ikawrakow/ik_llama.cpp/discussions/434
llama-imatrix \
    --verbosity 1 \
    -m ~/AI/Qwen3-4B-Thinking-2507/Qwen3-4B-Thinking-2507-THIREUS-BF16-SPECIAL_SPLIT/Qwen3-4B-Thinking-2507-THIREUS-BF16-SPECIAL_TENSOR-00001-of-00399.gguf \
    -f ubergarm-imatrix-calibration-corpus-v02.txt \
    -o ./ubergarm_imatrix.dat \
    -ngl 99 \
    --layer-similarity \
    --ctx-size 512 \
    --threads 32
