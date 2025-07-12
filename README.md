# GGUF Tool Suite

**GGUF Tool Suite** is a set of flexible utilities that enables users to experiment with and create custom GGUF quantization blends. It simplifies the process of mixing quant formats (like `iq3_xxs`, `iq4_nl`, etc.) to:

- Optimize performance  
- Reduce model size  
- Preserve accuracy across different hardware and use cases

---

## 🚀 MVP1 Release

> This release is MVP1 and has only been tested on  
> **[DeepSeek-R1-0528](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528)**.

### ⚠️ Requirements

- You **must use `ik_llama.cpp`** with [this patch](https://github.com/Thireus/ik_llama.cpp/commit/a66490410a366a9605234b94d67f3d9b7b389140) (compatibility with `llama.cpp` is **not guaranteed**).
- Source code and Windows builds of `ik_llama.cpp` with pre-patched `GGML_MAX_CONTEXTS` and `ulimit` are available at:  
  👉 https://github.com/Thireus/ik_llama.cpp  
- Official repo:  
  👉 https://github.com/ikawrakow/ik_llama.cpp  

### 🧠 Important: Linux `ulimit` Patch

Split models with a large number of files may **fail to load** unless you increase file descriptor limits.  
Run the following command **before launching llama binaries**:

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

The file `ppl_results.csv` contains **individual tensor-level PPL benchmarks** for:

- `Q8_0` (GPU tensors) + `IQ3-XXS` (CPU tensors)
- Target model: **DeepSeek-R1-0528**
- Quantization degradation reference: `IQ1-M-R4`

This is the **core file** used to determine optimal quant mix strategies.  
> ⚠️ Generating this CSV took **several days of GPU + CPU compute time**.

- `IQ3-XXS` was chosen for CPU tensors as it fits within **256GB RAM**
- Scripts used to generate:

```bash
./benchmark_each_tensor.sh --qtypes iq1_m_r4
./collect_ppl_results.sh --chunks 250 --qtypes iq1_m_r4
```

📄 An article explaining the methodology is **coming soon**.

---

## 🙏 Acknowledgements

Big thanks to **ubergarm** for his support and for providing the invaluable **`imatrix` file**.

📄 Ubergarm's `imatrix` can be found here:  
🔗 [imatrix_ubergarm.dat](https://huggingface.co/ubergarm/DeepSeek-R1-0528-GGUF/blob/main/imatrix-DeepSeek-R1-0528.dat)

Also sincere thanks to **ikawrakow** and all **co-authors** of `ik_llama.cpp` for making this entire toolchain possible.

---

## 📜 License & Attribution

Any **use, reproduction, or modification** of this software **must give clear and visible credit** to **Thireus** and the **GGUF Tool Suite**.  
See the [LICENSE](./LICENSE) file for more details.

🔗 https://gguf.thireus.com/
