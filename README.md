# GGUF Tool Suite

**GGUF Tool Suite** is a set of flexible utilities that enables users to experiment with and create custom GGUF quantization blends. It simplifies the process of mixing quant formats (like `iq3_xxs`, `iq4_nl`, etc.) to:

- Optimize performance  
- Reduce model size  
- Preserve accuracy across different hardware and use cases

---

### ✅ Supported models

*In theory, any model supported by llama.cpp is also supported by this tool suite. However, models that are not explicitely in the models/ folder would require additional efforts such as benchmarking and quantizing the model tensors. This table provides an overview of the models officially supported.*

| **Model** | **Calibration Data** | **Quantized Shards** | **Google Colabs** | **Evaluated** | **Comments** |
|---|---|---|---|---|---|
| [DeepSeek-R1-0528](https://huggingface.co/collections/Thireus/deepseek-r1-0528-thireus-special-split-68725429aceffbd1094bdd29) | ✅ Complete | ✅ Complete | ✅ Tested and Working | ✅ Yes | Works like a charm. When the quant_assign settings are right, it produces recipes with better ppl than any other reputable GGUFs. |
| [DeepSeek-TNG-R1T2-Chimera](https://huggingface.co/collections/Thireus/deepseek-tng-r1t2-chimera-thireus-special-split-68737c17e50ee1d7fb0fc474) | ✅ Complete | ✅ Complete | ✅ Tested and Working | ⚠️ Not personally | Should not be any different than DeepSeek-R1-0528. |
| [DeepSeek-V3-0324](https://huggingface.co/collections/Thireus/deepseek-v3-0324-thireus-special-split-6885d5fba52645afa377cc79) | ❌ Not started | ⚠️ Work in progress... | ⚠️ Untested | ⚠️ Not evaluated | Should not be any different than DeepSeek-R1-0528. |
| [Kimi-K2-Instruct](https://huggingface.co/collections/Thireus/kimi-k2-instruct-thireus-special-split-68778e3701cf3e80574185e2) | ✅ Complete | ✅ Complete | ✅ Tested and Working | ✅ Tested and Working | Examples provided. It would appear that it does really well on _kt quants, likely because this is the target quant that was used for the calibration data. I may need to redo the calibration data using iq1_s_r4 to verify this theory. |
| [Qwen3-235B-A22B-Instruct-2507](https://huggingface.co/collections/Thireus/qwen3-235b-a22b-instruct-2507-thireus-special-split-68813c81fa03b52fe0d6f4e9) | ✅ Complete | ✅ Best effort (a few quants are still missing) | ⚠️ Untested | ⚠️ Not evaluated | All you need is available to produce quant mixes, but not personally tested. |
| [Qwen3-235B-A22B-Thinking-2507](https://huggingface.co/collections/Thireus/qwen3-235b-a22b-thinking-2507-thireus-special-split-688368c416e0f03b1853b10f) | ✅ Complete | ✅ Best effort (a few quants are still missing) | ⚠️ Untested | ⚠️ Not evaluated | All you need is available to produce quant mixes, but not personally tested. |
| [Qwen3-Coder-480B-A35B-Instruct](https://huggingface.co/collections/Thireus/qwen3-coder-480b-a35b-instruct-thireus-special-split-68813cb65745c166d0386e91) | ✅ Complete | ✅ Best effort (a few quants are still missing) | ⚠️ Untested | ⚠️ Not evaluated | All you need is available to produce quant mixes, but not personally tested. |
| [GLM-4.5](https://huggingface.co/collections/Thireus/glm-45-thireus-special-split-6888e23853f18dd2d57a295b) | ❌ Not started | ⚠️ Work in progress... | ⚠️ Untested | ⚠️ Not evaluated | Support in `llama.cpp` is currently broken – see the discussion in [PR #14939](https://github.com/ggml-org/llama.cpp/pull/14939). Support in `ik_llama.cpp` is working but experimental, refer to [ikawrakow/ik_llama.cpp#668](https://github.com/ikawrakow/ik_llama.cpp/pull/668) |

### ⚠️ Requirements

- You **must compile the latest `ik_llama.cpp`** with `-DGGML_MAX_CONTEXTS=2048` - see pull requests: [#611](https://github.com/ikawrakow/ik_llama.cpp/pull/611), [#620](https://github.com/ikawrakow/ik_llama.cpp/pull/620) and [#622](https://github.com/ikawrakow/ik_llama.cpp/pull/622), and if using `llama.cpp` then make sure to apply these code changes. Note that compatibility with `llama.cpp` is **not guaranteed**.
- If you are on Windows, you will also need [this patch](https://github.com/ikawrakow/ik_llama.cpp/pull/620).
- Source code and Windows builds of `ik_llama.cpp` with pre-patched `GGML_MAX_CONTEXTS` and `ulimit` are available at:  
  👉 https://github.com/Thireus/ik_llama.cpp  
- Official repo:  
  👉 https://github.com/ikawrakow/ik_llama.cpp  

### 🧠 Important: Linux `ulimit` Patch

Split models with a large number of files may **fail to load** unless you increase file descriptor limits.  
Run the following command on Linux/macOS **before launching llama binaries**:

```bash
# Lifts "too many open files" limitation
ulimit -n 99999
```

---

## 📁 Recipe Examples

Examples are included in the `recipe_examples` folder. Have a look at the file name or inside the recipe files to see the VRAM and RAM requirements of each.

> ⚠️ You’re encouraged to build your own recipes tailored to your setup rather than relying on others'.

---

## 📥 Download Model Shards from a Recipe

```bash
git clone https://github.com/Thireus/GGUF-Tool-Suite
cd GGUF-Tool-Suite
# Make sure to copy the relevant download.conf for the model before running quant_assign.py
rm -f download.conf
# Use the download.conf of the chosen model
cp -f models/DeepSeek-R1-0528/download.conf .
mkdir -p kitchen && cd kitchen
../quant_downloader.sh ../recipe_examples/DeepSeek-R1-0528.THIREUS-3.4064bpw-3.3372ppl.242GB-GGUF_11GB-GPU_231GB-CPU.254e1cf_c044584.recipe
```

> 💡 **Pro tip**: Re-running `quant_downloader.sh` in the same directory will only download the **missing/different shards** from your current quant mix.

---

## 🧠 Run a Downloaded Model (Example)

```bash
~/ik_llama-main-b3904-41a9c8a-bin-win-cuda-12.8-x64-avx512/llama-cli \
  -m DeepSeek-R1-0528-THIREUS-BF16-SPECIAL_TENSOR-00001-of-01148.gguf \
  -mla 3 -fa -amb 1024 -fmoe -ctk f16 -c 16384 -ngl 99 \
  -ot "blk\.(3|4|5|6)\.ffn_.*=CUDA0" \
  -ot "blk\.(7|8|9)\.ffn_.*=CUDA1" \
  -ot "blk\.(10|11|12)\.ffn_.*=CUDA2" \
  -ot exps=CPU -b 4096 -ub 4096 --warmup-batch --no-mmap --threads 36 \
  --main-gpu 0 \
  -p '<｜begin▁of▁sentence｜><｜User｜>What is the solution of x+5=-2?<｜Assistant｜><think>\n'
```

---

## 🛠️ Generate a Custom Recipe for Your Config

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Thireus/GGUF-Tool-Suite/blob/main/quant_recipe_pipeline.ipynb)

```bash
# Make sure to copy the relevant download.conf and ppl_results.csv for the model before running quant_assign.py
rm -f download.conf ppl_results.csv
# Use the download.conf and ppl_results.csv of the chosen model
cp -f models/DeepSeek-R1-0528/download.conf .
cp -f models/DeepSeek-R1-0528/ppl_results.csv .
# Run the quant_assign.py script (adjust the parameters to match your configuration and target model)
python quant_assign.py ppl_results.csv \
  --gpu-tensors '.*' \
  --cpu-tensors 'blk\.([3-9]|[1-5][0-9]|60)\.ffn_down_exps\.weight' \
                 'blk\.([3-9]|[1-5][0-9]|60)\.ffn_up_exps\.weight' \
                 'blk\.([3-9]|[1-5][0-9]|60)\.ffn_gate_exps\.weight' \
  --cpu-quants iq4_ks iq3_k iq2_k iq1_m_r4 \
  --gpu-quants q8_0 iq5_k_r4 iq6_k \
  --cpu-tensors-max-size 230 \
  --gpu-tensors-max-size 95% \
  --tolerance 0.01 \
  --exponential-factor 8 \
  --gpu-assign-qtype iq4_xs \
  --gpu-assign-tensors 'blk\.([0-9]|[1-5][0-9]|60)\.attn_k_b\.weight=q8_0' \
  | ./quants_regex_merger.sh \
    --model-name "recipe_examples/DeepSeek-R1-0528" \
    --add-ppl 0 \
    --model-link "https://huggingface.co/deepseek-ai/DeepSeek-R1-0528"
```

> 🔧 **Adjust parameters** such as `--cpu-tensors-max-size` or `--gpu-quants` as needed for your specific hardware.

---

## 📊 About `ppl_results.csv`

The file `ppl_results.csv` contains **individual tensor-level PPL benchmarks**, for example for **DeepSeek-R1-0528**:

- `Q8_0` (GPU tensors) + `IQ3-XXS` (CPU tensors)
- Target model: **DeepSeek-R1-0528**
- Quantization degradation reference: `IQ1-M-R4`

This is the **core file** used to determine optimal quant mix strategies.  
> ⚠️ Generating this CSV took **several days of GPU + CPU compute time**.

- `IQ3-XXS` was chosen for CPU tensors as it fits within **256GB RAM**
- Scripts used to generate (edit the "USER CONFIGURATION" section in the bash scripts as needed):

```bash
# Make sure to copy the relevant download.conf for the model before running quant_assign.py
rm -f download.conf
# Use the download.conf of the chosen model
cp -f models/DeepSeek-R1-0528/download.conf .
# Make sure to adjust all configuration settings from both of these scripts, such as the most important USER_REGEX variable
./benchmark_each_tensor.sh --qtypes iq1_m_r4
./collect_ppl_results.sh --chunks 250 --qtypes iq1_m_r4
```

📄 An article explaining the methodology is **coming soon**.

---

## 🙏 Acknowledgements

Big thanks to **ubergarm** for his support and for providing the invaluable **`imatrix` files**.

📄 Ubergarm's `imatrix` for DeepSeek-R1-0528 can be found here:  
🔗 [imatrix_DeepSeek-R1-0528_ubergarm.dat](https://huggingface.co/ubergarm/DeepSeek-R1-0528-GGUF/blob/main/imatrix-DeepSeek-R1-0528.dat)

📄 Ubergarm's `imatrix` for DeepSeek-TNG-R1T2-Chimera can be found here:  
🔗 [imatrix_DeepSeek-TNG-R1T2-Chimera_r1t2_ubergarm.dat](https://huggingface.co/ubergarm/DeepSeek-TNG-R1T2-Chimera-GGUF/blob/main/imatrix-DeepSeek-TNG-R1T2-Chimera-Q8_0.dat)

📄 Ubergarm's `imatrix` for Kimi-K2-Instruct can be found here:  
🔗 [imatrix_Kimi-K2-Instruct_ubergarm.dat](https://huggingface.co/ubergarm/Kimi-K2-Instruct-GGUF/blob/main/imatrix-Kimi-K2-Instruct-Q8_0.dat)

📄 Ubergarm's `imatrix` for Qwen3-235B-A22B-Instruct-2507 can be found here:  
🔗 [imatrix_Qwen3-235B-A22B-Instruct-2507_ubergarm.dat](https://huggingface.co/ubergarm/Qwen3-235B-A22B-Instruct-2507-GGUF/blob/main/imatrix-eaddario-combined-all-medium-Qwen3-235B-A22B-Instruct-2507-BF16.dat)

📄 Ubergarm's `imatrix` for Qwen3-235B-A22B-Thinking-2507 can be found here:  
🔗 [imatrix_Qwen3-235B-A22B-Thinking-2507_ubergarm.dat](https://huggingface.co/ubergarm/Qwen3-235B-A22B-Thinking-2507-GGUF/blob/main/imatrix-Qwen3-235B-A22B-Thinking-2507-BF16.dat)

📄 Ubergarm's `imatrix` for Qwen3-Coder-480B-A35B-Instruct can be found here:  
🔗 [imatrix_Qwen3-Coder-480B-A35B-Instruct_ubergarm.dat](https://huggingface.co/ubergarm/Qwen3-Coder-480B-A35B-Instruct-GGUF/blob/main/imatrix-Qwen3-Coder-480B-A35B-Instruct-Q8_0.dat)

📄 Ubergarm's `imatrix` for DeepSeek-V3-0324 can be found here:  
🔗 [imatrix_DeepSeek-V3-0324_ubergarm.dat](https://huggingface.co/ubergarm/DeepSeek-V3-0324-GGUF/blob/main/DeepSeek-V3-0324.imatrix)

📄 Ubergarm's `imatrix` for GLM-4.5 can be found here:  
🔗 [imatrix_GLM-4.5_ubergarm.dat](https://huggingface.co/ubergarm/GLM-4.5-GGUF/blob/main/imatrix-GLM-4.5-BF16.dat)

Also sincere thanks to **ikawrakow** and all **co-authors** of `ik_llama.cpp` for making this entire toolchain possible.

---

## 📜 License & Attribution

Any **use, reproduction, or modification** of this software **must give clear and visible credit** to **Thireus** and the **GGUF Tool Suite**.  
See the [LICENSE](./LICENSE) file for more details.

🔗 https://gguf.thireus.com/
