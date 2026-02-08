---
license: mit
---
# Qwen3-4B-Thinking-2507

## 🤔 What is this [HuggingFace repository](https://huggingface.co/Thireus/Qwen3-4B-Thinking-2507-THIREUS-BF16-SPECIAL_SPLIT/) about?

This repository provides **GGUF-quantized tensors** for the Qwen3-4B-Thinking-2507 model (official repo: https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507). These GGUF shards are designed to be used with **Thireus’ GGUF Tool Suite** (https://gguf.thireus.com), a collection of tools that automatically finds the perplexity-optimal mix of quantizations for any given VRAM and RAM target. With this GGUF Tool Suite, you can produce your own Dynamic 3.0 Quants recipes and achieve optimum accuracy & SOTA quantization performance.

- 📖 Read more: https://github.com/Thireus/GGUF-Tool-Suite  
- 🔍 Example of GGUF recipes: https://github.com/Thireus/GGUF-Tool-Suite/tree/main/recipe_examples  
- 🍳 Cook your own recipe files: https://gguf.thireus.com/quant_assign.html  
- ☁️ Download GGUF models from recipe files: https://gguf.thireus.com/quant_downloader.html  
- 📂 Browse available models: https://gguf.thireus.com  

*tl;dr: Expand the details section below*
<details>

```
cd ~

# Make sure to install all ik_llama.cpp compilation dependencies...
apt install python3-dev python3-pip python3-venv python3-wheel python3-setuptools git acl netcat-openbsd cmake # pipx

# Obtain ik_llama's Thireus version - Windows/macOS/Linux builds available at https://github.com/Thireus/ik_llama.cpp/releases
git clone https://github.com/Thireus/ik_llama.cpp
cd ik_llama.cpp
git pull
# Build ik_llama.cpp
cmake -B build -DGGML_AVX=ON -DGGML_AVX2=ON -DLLAMA_CURL=OFF -DGGML_MAX_CONTEXTS=2048
cmake --build build --config Release -j16
cd ..

# Obtain Thireus' GGUF-Tool-Suite
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/Thireus/GGUF-Tool-Suite

# Download model quant mix from recipe file - you can also try the web version: https://gguf.thireus.com/quant_downloader.html
cd GGUF-Tool-Suite
rm -f download.conf # Make sure to copy the relevant download.conf for the model before running quant_assign.py
cp -f models/Qwen3-4B-Thinking-2507/download.conf . # Use the download.conf of the chosen model
mkdir -p kitchen && cd kitchen
../quant_downloader.sh ../recipe_examples/ik_llama.cpp_recipes/Qwen3-4B-Thinking-2507.ROOT-4.2498bpw-10.9335ppl.1GB-GGUF_0GB-GPU_1GB-CPU.9888e4b_9193781.recipe

# Other recipe examples can be found at https://github.com/Thireus/GGUF-Tool-Suite/tree/main/recipe_examples

# Launch ik_llama's llama-cli:
ulimit -n 9999 # Lifts "too many open files" limitation on Linux
~/ik_llama.cpp/build/bin/llama-server \
  -m Qwen3-4B-Thinking-2507-THIREUS-BF16-SPECIAL_TENSOR-00001-of-00399.gguf \
  -fa auto -amb 1024 -ctk q8_0 -c 32768 -ngl 99 \
  -b 4096 -ub 4096 --warmup-batch --no-mmap --threads 1 \
  --main-gpu 0
```

</details>

---

## ❓ Why does this Tool Suite exist?

1. **Compatibility & Speed** – [unsloth](https://huggingface.co/unsloth)’s dynamic quants may not always work optimally with `ik_llama.cpp`.  
2. **Custom Rig Fit** – No off-the-shelf GGUF model perfectly matched my VRAM/RAM setup, so I built a way to tailor models and leverage extra VRAM/RAM to reduce perplexity.  
3. **Automated PPL-Optimal Quantization** – To my knowledge, there was no open source flexible, automated method to minimize perplexity for any bits-per-weight (bpw) target—so I created one with excellent results!  

---

## 📊 How does it compare to other GGUFs?

Here’s how Qwen3-4B-Thinking-2507 quantized with **Thireus’ GGUF Tool Suite** stacks up against other quantizers (lower perplexity = better at equal or lower bpw):

![PPLs Compared With Others](https://github.com/Thireus/GGUF-Tool-Suite/raw/main/ppl_graphs/Qwen3-4B-Thinking-2507.svg)

> _Note: The `recipe_examples` files illustrate good recipes. The Tool Suite computes the optimal ppl/bpw curve for you — just specify your target RAM, VRAM, and quant types, and `quant_assign.py` finds the best mix._  

More perplexity/bpw graphs for other supported models: https://github.com/Thireus/GGUF-Tool-Suite/tree/main/ppl_graphs  

*All PPL benchmarks are computed with the parameters `-ctk f16 -c 512 -b 4096 -ub 4096`. Changing any of these parameters will alter the PPL. In particular, reducing `-b 4096 -ub 4096` increases the PPL, while increasing them decreases the PPL.*

---

## 🚀 How do I get started?

Check out the [GGUF Tool Suite README](https://github.com/Thireus/GGUF-Tool-Suite) — focus on these sections:

1. ⚠️ **Requirements** – Which `ik_llama.cpp` (or `llama.cpp`) version to use and how to compile.  
   - Windows binaries (no patching needed) at: https://github.com/Thireus/ik_llama.cpp/releases  
2. 📥 **Download Model Shards** – Use `quant_downloader.sh` or [quant_downloader.html](https://gguf.thireus.com/quant_downloader.html) to fetch GGUF shards from any recipe.  
   - Recipe examples: https://github.com/Thireus/GGUF-Tool-Suite/tree/main/recipe_examples  
3. 🧠 **Run a Downloaded Model** – Sample usage with `llama-cli`.  
4. 🛠️ **Generate a Custom Recipe** – Produce recipes tailored to your VRAM/RAM target usage for optimum perplexity.  

---

## ✅ Supported Models

Supported models are listed under `models/` in the [Tool Suite Github repo](https://github.com/Thireus/GGUF-Tool-Suite/tree/main/models). Presence of `ppl_results.csv` indicates official support and compatibility with `quant_assign.py`.

---

## 🤷‍♂️ Will I release baked dynamic quant GGUFs?

No, because I believe in **tailored quantization** for each user’s hardware. If you prefer ready-made shards, you are welcome to merge them via `llama-gguf-split --merge`, or request someone to publish them, or rely on generic GGUF dynamic quants such as [unsloth](https://huggingface.co/unsloth)'s.

Instead, I prefer to share examples of recipes so users can see exactly how they were produced (command included inside these recipe files) and tweak them for their own rigs. The `quant_downloader.sh` script or [quant_downloader.html](https://gguf.thireus.com/quant_downloader.html) (web port of this script) handles automatic fetching and verification of each shard. Note that recipes provided by [Ubergarm](https://huggingface.co/ubergarm) on his model cards are also compatible with `quant_downloader.sh` and [quant_downloader.html](https://gguf.thireus.com/quant_downloader.html), providing a "SPECIAL_SPLIT" version of these models exists (see https://gguf.thireus.com/).

Users who don’t trust the GGUF shards on HuggingFace can also quantize their own by passing recipe lines to `llama-quantize --custom-q` ([see example](https://github.com/Thireus/GGUF-Tool-Suite/blob/main/models/DeepSeek-R1-0528/DeepSeek-R1-0528-THIREUS-ANY-SPECIAL.sh#L482-L486)). Run `llama-quantize --help` to list compatible quants for `quant_assign.py`. This approach is especially useful if you prefer `llama.cpp` over `ik_llama.cpp`.  

---

## 📦 What’s in this repository?

- **00001 GGUF header shard** – Contains metadata (tokens, chat template, tensor count, etc.). This metadata can be explored directly from the HuggingFace web interface after clicking on that shard.  
- **Tensor shards** – Each shard holds one tensor; see `tensors.map` for names, quant types, sizes, SHA-256 hash, shard IDs, etc.  
- **GPG-signed files** – `tensors.map` and header shard are signed with the key in [trusted-keys.asc](https://github.com/Thireus/GGUF-Tool-Suite/blob/main/trusted-keys.asc) for tamper detection.  
- **Security note** – Some papers about various ways to attack GGUFs and LLMs are available online, such as https://arxiv.org/abs/2505.23786, and there are also more classic security exploits like CVE-2024-23496 and CVE-2024-25664 through CVE-2024-25668. Only use GGUFs from reputable, trusted authors—or alternatively self-quantize—to avoid potential exploits. 

---

## 💡 Pro Tips

You can easily download the BF16 model version to quantize your own shards:

```
mkdir kitchen  
echo '.*=bf16' > kitchen/bf16.recipe  
cd kitchen
../quant_downloader.sh bf16.recipe  
```

You can also quantize individual BF16 tensors without the need to download every BF16 .gguf shard:

BF16 model shards can also be individually quantized using a special version of ik_llama.cpp's `llama-quantize` utility which comes with the `--individual-tensors` option.

- Source code: https://github.com/Thireus/ik_llama.cpp/tree/th/quantize_individual_tensors
- Builds (macOS, Windows and Linux): https://github.com/Thireus/ik_llama.cpp/releases/tag/th-quantize_individual_tensors-b4210-7a44805

Usage example:
```
./llama-quantize --keep-split --imatrix imatrix_ubergarm.dat --individual-tensors 2,3,1094 Kimi-K2-Thinking-THIREUS-BF16-SPECIAL_TENSOR-00001-of-01097.gguf my_new_shards.gguf iq3_s 12
```

For more information about how to use it: https://github.com/Thireus/GGUF-Tool-Suite/issues/45

Enjoy optimized quantization! 🎉
