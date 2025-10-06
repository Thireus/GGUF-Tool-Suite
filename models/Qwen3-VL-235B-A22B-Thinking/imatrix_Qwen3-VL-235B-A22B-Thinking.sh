#!/usr/bin/env bash
#***************************************************************#
#** This script is part of Thireus' GGUF Tool Suite.          **#
#** imatrix_Qwen3-VL-235B-A22B-Thinking.sh used to create the **#
#** imatrix file.                                             **#
#**                                                           **#
#** ********************************************************* **#
#** --------------- Updated: Oct-06-2025 -------------------- **#
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
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3 ~/ik_llama-tr-qwen3-vl-b4184-2b0ce8a-bin-win-cuda-12.8-x64-avx512/llama-imatrix  \
    --verbosity 1 \
    -m Qwen3-VL-235B-A22B-Thinking-THIREUS-BF16-SPECIAL_TENSOR-00001-of-01132.gguf \
    -f ubergarm-imatrix-calibration-corpus-v02.txt \
    -o ./imatrix_ubergarm.dat \
    -ngl 99 \
    --layer-similarity \
    --ctx-size 512 \
    --threads 18 --warmup-batch --no-mmap --main-gpu 0 -mla 3 -fa -amb 1024 -fmoe \
    -ot "blk\.([0-9]|1[0-8])\.ffn_.*=CUDA0" \
    -ot "blk\.(19|2[0-9]|3[0-7])\.ffn_.*=CUDA1" \
    -ot "blk\.(38|39|4[0-9]|5[0-6])\.ffn_.*=CUDA2" \
    -ot "blk\.(5[7-9]|6[0-1])\.ffn_.*=CUDA3" \
    -ot exps=CPU -b 4096 -ub 4096
