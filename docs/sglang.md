# Produce SGLang recipes and checkpoints (NVFP4 / FP8)

_In this section I will explain how to produce a recipe for the quant types SGLang loads itself (NVFP4, FP8) and how to turn that recipe into a checkpoint SGLang serves directly, without GGUF. The idea is the same as for GGUF recipes: the calibration data tells `quant_assign.py` how sensitive each tensor is, and it spends your size budget where it hurts the least. Everything is produced from the same BF16 GGUF shards this suite uses everywhere else, so if you have made GGUF recipes for a model you already have all you need._

Before we get started I feel it's important to explain a few points:

1. Why SGLang at all? Speed and concurrency. [SGLang](https://github.com/sgl-project/sglang) serves many users at once and, on Blackwell GPUs, its NVFP4 and FP8 kernels are the fastest way to run a model that fits in VRAM. The trade-off is quality per byte: the smallest type SGLang loads is 4-bit (NVFP4, about 4.5 bits per weight once the scales are counted), so there is no 1, 2 or 3-bit option and none of the quality-per-byte tricks ik_llama.cpp offers. If your model already fits nicely on your GPUs at 4 to 8 bits, SGLang is the better engine for serving it. If you need to squeeze a big model into a small space, stick to GGUF recipes.

2. SGLang does not read GGUF files. It reads safetensors checkpoints whose tensors are already stored in the format its kernels consume: NVFP4 (4 bits with a per-16 FP8 block scale and a per-tensor scale), FP8 (per tensor, or per 128x128 block), and BF16. On Blackwell GPUs the NVFP4 tensor cores are what make these formats fast. `sglang_write.py` reads the BF16 GGUF shards, undoes what `convert_hf_to_gguf.py` did to them, and writes such a checkpoint.

3. NVFP4 and per-tensor FP8 need one static number per tensor, the `input_scale`: the largest activation that tensor ever sees, measured by running text through the BF16 model. It cannot be computed from the weights. `sglang_calibrate.py` measures it for you in a couple of minutes on one GPU, on the same calibration text this suite uses for imatrix and KLD. I compared the result with the scales published by others and there is nothing to choose between them, see [About the shipped recipes](#about-the-shipped-recipes).

4. Unlike GGUF quants, NVFP4 and per-tensor FP8 quantize the activations at runtime too (W4A4 and W8A8), which costs quality a weight-only measurement cannot see. This is why [About the shipped recipes](#about-the-shipped-recipes) reports every number twice: what the weights alone cost, and what you actually get in the engine.

5. The pool has only three rungs, 4.5, 8 and 16 bits, so the per-format degradation rows the assigner needs are a property of the format rather than of the model. They are built into the tool (measured in SGLang on Qwen3.8-27B) and you do not have to benchmark them for your model. I measured that claim rather than assumed it, see [Adding a new model](#optional-adding-a-new-model).

## Requirements

**IMPORTANT: FOLLOW THIS GUIDE STEP-BY-STEP. DO NOT SKIP ANY NON-OPTIONAL PART. DO NOT TAKE SHORTCUTS. READ CAREFULLY WHAT IS WRITTEN AT ANY STEP.**

Hardware:

```
CPU: YES (the recipe and the checkpoint are produced on CPU)
GPU: YES (any CUDA GPU with 24GB to measure the activation scales, a Blackwell one to serve the result)
```

Note: I personally own 3xRTX6000Pro + 1xRTX5090. The checkpoint writer needs about 8GB of RAM whatever the model size, the shards are streamed: a 27B checkpoint takes a few minutes to write, a 358B one about 70 minutes. The scale measurement streams the model layer by layer too, so it fits a 24GB GPU for any model size; on my RTX 5090 it takes 2 minutes for Qwen3.8-27B.

Make sure you edit and paste these env variables in any terminal session you'll be using:

```
WORKING_DIRECTORY='/AI' # Full path please!
MODEL='Qwen3.8-27B'
MAINTAINER='THIREUS' # Or use your name!
SIZE='17.41GB' # The size budget for the quantized tensors, see the note on units below
HF_REPO='Qwen/Qwen3.8-27B' # The official HuggingFace repository of the model
```

Note: `GB` is decimal (17.41GB is 17,410,000,000 bytes, which is what the shipped recipe footers record). `GiB`, `MB`, a percentage of the all-BF16 size, or a bare byte count with `B` all work too. The GLM rows at the end show the budget in GB with the exact byte argument that produced the recipe next to it.

Note: The model folder under `models/$MODEL` must already hold `kld_results.csv` and `group0/tensors.bf16.map`, the same inputs you use to make GGUF recipes. If it does not, [benchmark the model](https://github.com/Thireus/GGUF-Tool-Suite/blob/main/docs/Benchmarking%20models%20-%20How.md) first. Nothing else is needed per model, in particular no SGLang-specific benchmark.

## Know your quants

Quant types supported by SGLang for this tool suite:

```
sgl_nvfp4 sgl_nvfp4a16 sgl_fp8 sgl_fp8_pb_wo sgl_mxfp8 sgl_int4_g128 sgl_bf16
```

`sgl_nvfp4` is the fast one on Blackwell (4.5 bits per weight, activations quantized too). `sgl_fp8` is per-tensor FP8 (8 bits, W8A8), `sgl_fp8_pb_wo` is block-scaled FP8 (weight-only on paper, but the kernel quantizes the activations per token group inside the GEMM, so it measures the same). `sgl_bf16` is the escape hatch for what must stay exact. The token embedding can hold NVFP4 or BF16. Which one depends on two separate things: the architecture, which the tool knows (GLM's SGLang model file loads a BF16 table only, so GLM recipes keep it BF16), and your engine build, which is yours to pick. A stock SGLang builds the Qwen3.5 table without a quant config, so an NVFP4 embedding there needs the change carried on [my SGLang fork](https://github.com/Thireus/sglang) (upstream did not take it); `--nextn-optimization on` is the alternative, it keeps the embedding BF16 so it loads everywhere. `sgl_nvfp4a16` (NVFP4 weights, BF16 activations), `sgl_mxfp8` and `sgl_int4_g128` are accepted by the tools, but the `sglang` speed profile does not choose them and none of the shipped recipes uses one.

Important: You will notice that these types are lowercase and start with `sgl_`. Do not mix them with GGUF quant types in one recipe, the assigner refuses such a pool.

## Prepare the environment

```
apt-get install screen gpg curl git python3 python3-dev python3-pip python3-venv # Run as root
```

Note: `python3-dev` is not optional, the scale measurement compiles a small CUDA shim on the fly and needs the Python headers. Use CPython 3.12: on 3.13 and later `pgpy` installs but cannot import, and the recipe step silently records `GPG signatures: DISABLED` in its footer where a good run records `PASSED`.

Create the working directory (where all files will be downloaded and produced):

```
mkdir -p "$WORKING_DIRECTORY"
```

Obtain the GGUF-Tool-Suite and prepare it for the chosen `$MODEL`:

```
cd "$WORKING_DIRECTORY"
GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://github.com/Thireus/GGUF-Tool-Suite/
cd GGUF-Tool-Suite && git pull # Update it
rm -f download.conf && cp -rf models/"$MODEL"/download.conf .
python3 -m venv .venv && . .venv/bin/activate && pip install pandas numpy pgpy safetensors torch transformers
```

Note: `torch` and `transformers` are only needed by the scale measurement and the checkpoint writer; the recipe itself needs pandas and numpy like every other recipe in this suite. This venv produces the recipe, the scales and the checkpoint; it does not serve them. SGLang has its own install (`pip install "sglang[all]"` in a venv of its own, or [my fork](https://github.com/Thireus/sglang), see [Serve the checkpoint](#serve-the-checkpoint)) and nothing below writes into it. Activate this venv in every terminal session you use for the steps below.

Download the BF16 version of the model, one tensor per shard, exactly as you would to [quantize it from BF16](https://github.com/Thireus/GGUF-Tool-Suite/blob/main/docs/Quantize%20from%20BF16.md):

```
cd "$WORKING_DIRECTORY"
export PATH="$WORKING_DIRECTORY"/GGUF-Tool-Suite/:$PATH
mkdir "$MODEL"-"${MAINTAINER^^}"-BF16-SPECIAL_SPLIT && \
cd "$MODEL"-"${MAINTAINER^^}"-BF16-SPECIAL_SPLIT && \
echo '.*=bf16' > bf16.recipe && \
quant_downloader.sh bf16.recipe --qtype BF16
```

The models Thireus provides are on [HuggingFace](https://huggingface.co/Thireus/collections) and mirrored on [gguf.thireus.com](https://gguf.thireus.com). If you converted the model yourself following [Convert model to BF16](https://github.com/Thireus/GGUF-Tool-Suite/blob/main/docs/Convert%20model%20to%20BF16.md), point the commands below at that split instead.

A model converted with `--no-mtp` keeps its draft head, and a vision model its vision tower, in splits of their own (`mtp-` and `mmproj-`), and SGLang needs both. Download them next to the main split when this suite has a folder for them (Qwen3.8-27B has both, GLM-4.7 has neither, its draft head is in the main split):

```
cd "$WORKING_DIRECTORY" && \
for part in mtp mmproj; do if [ -f GGUF-Tool-Suite/models/"$part"-"$MODEL"/download.conf ]; then mkdir -p "$part"-"$MODEL"-"${MAINTAINER^^}"-BF16-SPECIAL_SPLIT && cd "$part"-"$MODEL"-"${MAINTAINER^^}"-BF16-SPECIAL_SPLIT && echo '.*=bf16' > bf16.recipe && GGUF_DOWNLOAD_CONF="$WORKING_DIRECTORY"/GGUF-Tool-Suite/models/"$part"-"$MODEL"/download.conf quant_downloader.sh bf16.recipe --qtype BF16; cd "$WORKING_DIRECTORY"; fi; done
```

Note: `quant_downloader.sh` leaves a `tensors.map` beside the shards, which is the `tensors.bf16.map` of the model. The writer reads the list of tensors from it, so a recipe file written against `group0/tensors.bf16.map` applies to the split as is.

A GGUF carries the weights and a lot of metadata, but not the small files SGLang and transformers parse: `config.json`, the tokenizer, the chat template. They are the non-LFS files of the model's own repository and come to a few MB. Obtain them (not every model ships every one of these files; `curl` prints `curl: (22) The requested URL returned error: 404` for each one that is missing, which is expected, Qwen3.8-27B has no `special_tokens_map.json` and lands 10 of the 11):

```
mkdir -p "$WORKING_DIRECTORY"/huggingface/"$MODEL" && cd "$WORKING_DIRECTORY"/huggingface/"$MODEL" && \
for f in config.json generation_config.json tokenizer.json tokenizer_config.json vocab.json merges.txt special_tokens_map.json chat_template.jinja preprocessor_config.json video_preprocessor_config.json model.safetensors.index.json; do curl -fsSL -o "$f" https://huggingface.co/"$HF_REPO"/resolve/main/"$f" || rm -f "$f"; done && ls
```

Note: Everything you hand to the writer is checked against the GGUF's own metadata (layer count, hidden size, head counts, expert count, vocabulary size, the architecture itself), so somebody else's `config.json` is caught in the first second rather than on the GPU.

Then, obtain the `imatrix-calibration-corpus-v02.txt` file which is used to measure the activation scales (the same text this suite uses for imatrix and KLD, see why [here](https://github.com/Thireus/GGUF-Tool-Suite/discussions/23#discussioncomment-14764941)):

```
cd "$WORKING_DIRECTORY" && \
curl -L 'https://gist.githubusercontent.com/ubergarm/edfeb3ff9c6ec8b49e88cdf627b0711a/raw/ba5b01b6960a86874592f5913e283746ff734483/ubergarm-imatrix-calibration-corpus-v02.txt' -o imatrix-calibration-corpus-v02.txt
```

## Produce the recipe

Run this from your model folder. The only thing you pick is the size:

```
cd "$WORKING_DIRECTORY"/GGUF-Tool-Suite && . .venv/bin/activate && cd models/"$MODEL" && \
../../quant_assign.py kld_results.csv --speed-profile sglang --gpu-tensors-max-size "$SIZE" --hf-files "$WORKING_DIRECTORY"/huggingface/"$MODEL" | ../../quants_regex_merger.sh --model-name "$MODEL" --model-link https://huggingface.co/"$HF_REPO" && \
RECIPE="$(ls -t "$MODEL".*.recipe | head -1)" && echo "$RECIPE"
```

`--gpu-tensors-max-size` is the most important parameter, and pretty much the only one you need to think about: it is the size budget of the quantized tensors. It matters so much more than for GGUF recipes because SGLang has very few quant types to choose from: a tensor is either 4-bit (NVFP4), 8-bit (FP8) or left in BF16. There is no q3, no q6, no intermediate types and no variants of the same bits per weight, so the whole recipe boils down to which tensors deserve 8 bits or BF16 within your budget, and the calibration data is what decides that.

`--speed-profile sglang` takes care of the rest: it picks the quant types SGLang can load, detects the model architecture from the tensor names, keeps fused tensors on the same type, prices each type with the degradation rows it carries (measured in SGLang, no benchmark of your own needed) and chooses the speed budgets so that your bytes are spent first and speed is maximised second. Everything it decided is printed as `[Preset]` lines and written in the recipe footer, together with the full command that reproduces the run. `--hf-files` gives it the model's `config.json`, which it reads for two facts the tensor names do not carry: the number of experts routed per token on a MoE model (the speed model needs it), and whether the model has a draft head.

Note: always pipe the output through `quants_regex_merger` as above. `quant_assign.py` prints one line per tensor; the merger turns that into the compact recipe file and names it for you (model, bits per weight, size, hashes), and that file is exactly what the checkpoint writer takes. `--add-ppl` puts the measured perplexity in the filename like every other recipe in this repo (perplexity on `wiki.test.raw`, 512-token context, scored the way `llama-perplexity` scores it; for SGLang recipes I measure it with the checkpoint served in SGLang; the two engines agree to 0.03% on identical weights). The predicted KLD stays in the footer; `--add-kld` can put it in the name too.

Tip: leave the speed budgets alone. The rule is one sentence: spend the size you asked for, then be as fast as possible. `--prefill-budget` and `--decode-budget` default to `auto`, which sweeps a handful of options and keeps the fastest recipe that still spends at least 98% of your size budget. `--fill-fraction` changes that 98%.

## (optional) Speculative decoding (NEXTN / EAGLE)

If your model has an MTP or NEXTN head and you want speculative decoding on a stock SGLang, add `--nextn-optimization on` to the command above. It keeps the token embedding, and the draft head's own table, in BF16, which a stock SGLang needs to share them with the draft (and, on Qwen3.5, to load the embedding at all, see above). It costs a bit of your size budget and some quality, so it is off by default. Leave it off if you serve on my fork, which shares a quantized embedding with the draft. The preset prints a reminder when it detects a draft head so you can decide.

## Measure the activation scales

This is the one step that needs a GPU before serving. `sglang_calibrate.py` runs the calibration text through the BF16 model, one layer at a time, and records for every linear the largest activation it saw. The result is a tiny scale set (a few hundred KB) the writer reads:

```
cd "$WORKING_DIRECTORY"/GGUF-Tool-Suite && . .venv/bin/activate && \
./sglang_calibrate.py --gpu 0 --source "$WORKING_DIRECTORY"/"$MODEL"-"${MAINTAINER^^}"-BF16-SPECIAL_SPLIT --hf-files "$WORKING_DIRECTORY"/huggingface/"$MODEL" --text "$WORKING_DIRECTORY"/imatrix-calibration-corpus-v02.txt --tokens 131072 --out "$WORKING_DIRECTORY"/"$MODEL"-"${MAINTAINER^^}"-SCALES
```

Note: `--gpu 0` is the CUDA index of the GPU to use; every other GPU is hidden from the process, and the tool never picks one on its own (`--gpu cpu` walks on the CPU, slowly). 131,072 tokens is what the shipped scale sets use (2 minutes on an RTX 5090 for a 27B model). 32,768 is enough if you are in a hurry; the measured difference between the two is inside the noise, see the table at the end. What is not fine is calibrating on the text you evaluate with: the tool refuses `wiki.test.raw` by name and by hash, keep it that way.

Note: The scale sets of the shipped recipes are in `models/$MODEL/sglang_scales/`. Use that directory instead of `"$MODEL"-"${MAINTAINER^^}"-SCALES` below and your checkpoint will be byte for byte the one I measured (give the build the same `--name` too, the writer stores it in every shard's header). Your own measurement will not reproduce the shipped set exactly: the pip line above pins nothing, and a different torch or transformers moves a minority of the values by a fraction of a percent (114 of 497 on Qwen3.8-27B between torch 2.13 with transformers 5.16 and torch 2.14 with transformers 5.17). The checkpoint changes, the measured perplexity does not, see [About the shipped recipes](#about-the-shipped-recipes).

## Build the checkpoint

A recipe is a text file of `^tensor$=qtype` lines. `sglang_write.py` turns one into a checkpoint SGLang can load, on CPU:

```
cd "$WORKING_DIRECTORY"/GGUF-Tool-Suite && . .venv/bin/activate && \
./sglang_write.py --build --source "$WORKING_DIRECTORY"/"$MODEL"-"${MAINTAINER^^}"-BF16-SPECIAL_SPLIT --hf-files "$WORKING_DIRECTORY"/huggingface/"$MODEL" --recipe models/"$MODEL"/"$RECIPE" --input-scales-from "$WORKING_DIRECTORY"/"$MODEL"-"${MAINTAINER^^}"-SCALES --out "$WORKING_DIRECTORY"/"$MODEL"-"${MAINTAINER^^}"-SGLANG --name "$MODEL"-"${MAINTAINER^^}"-SGLANG
```

Then check it before any GPU sees it:

```
./sglang_write.py --verify --ckpt "$WORKING_DIRECTORY"/"$MODEL"-"${MAINTAINER^^}"-SGLANG --source "$WORKING_DIRECTORY"/"$MODEL"-"${MAINTAINER^^}"-BF16-SPECIAL_SPLIT --hf-files "$WORKING_DIRECTORY"/huggingface/"$MODEL"
```

Note: The architecture is read from the files you point at (`config.json` in `--hf-files`, `general.architecture` and the tensor names in the split) and the writer refuses when two of those disagree. The vision tower and the draft head of a model converted with `--no-mtp` live in their own `mmproj-` and `mtp-` splits; the writer finds them beside the main split (that is what the companion download above is for) and refuses to write a checkpoint that is quietly missing a part the architecture keeps in a split of its own.

Note: The checkpoint is built from the recipe's assignments, so re-representing a recipe (running it through the merger again) never changes the checkpoint. `--verify` checks every tensor's shape, type and scales against the BF16 source and dequantizes a sample of them to bound the error. `BUILD.json` in the checkpoint records what was read, what was written, the sha256 of every shard, and the few norm gammas the GGUF could not represent exactly (on Qwen3.8-27B 36 of 694,784, each within 6e-08 of the original; the GGUF holds a correctly rounded `1 + gamma`, which is what llama.cpp uses, so nothing is lost anywhere).

## (optional) Cross-check against a published checkpoint

The writer's codec is ModelOpt's, bit for bit, and `--crosscheck` proves it on your machine: every tensor that both your checkpoint and a published NVFP4 checkpoint of the same model store as NVFP4 must be byte-identical. For Qwen3.8-27B that checkpoint is [RadixArk's Qwen3.8-27B-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-27B-NVFP4) (credits to them, about 20 GB); place it under `"$WORKING_DIRECTORY"/huggingface/Qwen3.8-27B-NVFP4` (for example with `hf download RadixArk/Qwen3.8-27B-NVFP4 --local-dir "$WORKING_DIRECTORY"/huggingface/Qwen3.8-27B-NVFP4`) and run:

```
./sglang_write.py --crosscheck --ckpt "$WORKING_DIRECTORY"/"$MODEL"-"${MAINTAINER^^}"-SGLANG --against "$WORKING_DIRECTORY"/huggingface/Qwen3.8-27B-NVFP4
```

Note: The activation scales are excluded from that comparison by construction, they are yours; the packed 4-bit codes, the block scales and the per-tensor weight scales are what must match, and on the shipped recipes they match on every one of the tensors the two have in common.

## Serve the checkpoint

```
python -m sglang.launch_server --model-path "$WORKING_DIRECTORY"/"$MODEL"-"${MAINTAINER^^}"-SGLANG --quantization modelopt_mixed --host 127.0.0.1
```

The server names the model by the path you gave it, so the first request to make is a GET of `/v1/models` on the port it prints at startup; pass `--served-model-name` for a shorter id.

Note: `modelopt_mixed` is what `hf_quant_config.json` declares and a stock SGLang loads it. What a stock SGLang does not load is an NVFP4 token embedding on Qwen3.5, and the shipped Qwen recipes quantize it: serve those on [my SGLang fork](https://github.com/Thireus/sglang) (its `main` is upstream plus that change), or build your own recipe with `--nextn-optimization on` so the embedding stays BF16. The NEXTN speed numbers below also need the fork, which shares the quantized embedding with the draft head.

## (optional) Ship it: one repository per SGLang qtype

_A GGUF model ships from `<MODEL>-<MAINTAINER>-<QTYPE>-SPECIAL_SPLIT` repositories, one file per tensor, and `quant_downloader.sh` cooks any recipe out of them by fetching each tensor from the repository of the qtype the recipe gave it. SGLang checkpoints ship exactly the same way: `--split` writes one repository per SGLang qtype, holding the whole model at that type, one `.safetensors` per GGUF tensor carrying that module's HF tensors (`weight`, `weight_scale`, `weight_scale_2`, `input_scale`, whichever the format registers). The numbering is the model's own GGUF chunk numbering, so a recipe written against `tensors.bf16.map` addresses these repositories unchanged._

Write the repositories, one command per qtype; each writes the main split and its `mtp-`/`mmproj-` companions:

```
cd "$WORKING_DIRECTORY"/GGUF-Tool-Suite && . .venv/bin/activate && \
for q in sgl_bf16 $(sed -n 's/.*=\(sgl_[a-z0-9_]*\)$/\1/p' models/"$MODEL"/"$RECIPE" | sort -u); do ./sglang_write.py --split --qtype $q --source "$WORKING_DIRECTORY"/"$MODEL"-"${MAINTAINER^^}"-BF16-SPECIAL_SPLIT --hf-files "$WORKING_DIRECTORY"/huggingface/"$MODEL" --input-scales-from "$WORKING_DIRECTORY"/"$MODEL"-"${MAINTAINER^^}"-SCALES --amax-cache "$WORKING_DIRECTORY"/"$MODEL"-amax.json --out "$WORKING_DIRECTORY"/sgl-repos; done
```

The loop writes `sgl_bf16` (what an unnamed tensor is fetched as) plus every qtype your recipe names; the shipped 17.41GB recipe uses `sgl_fp8_pb_wo` on 101 modules as well as `sgl_nvfp4`, `sgl_fp8` and `sgl_bf16`. Each output directory carries its files, a `tensors.<qtype>.map` (and a `tensors.map` copy), and a `SPLIT.json` census. Shard `00001` is the metadata shard, exactly as in a GGUF split: no weights, and in the main split the checkpoint's small text files (`config.json`, the tokenizer, `chat_template.jinja`, the preprocessor configs and the source `model.safetensors.index.json`), so everything a checkpoint needs travels through the one downloader and is hashed like every other file. Sign `tensors.map` and shard `00001` before publishing, as you do for a GGUF split (`gpg --detach-sign --armor -o tensors.map.sig tensors.map`, and the same for the shard). Until you do, `quant_downloader.sh` refuses the repository with `failed to fetch map gpg signature`; pass `--skip-gpg` to read an unsigned one you just wrote, and point `download.conf` at it with `DOWNLOAD_ORDER=(SYMLINK)` and `SYMLINK_FOLDERS=("$WORKING_DIRECTORY/sgl-repos/")` while it is not published.

Download a recipe's pieces. `--qtype sgl_bf16` is required and is what a tensor the recipe does not name is fetched as, because in the SGLang container BF16 is encoded by absence and `quant_assign.py --ignore-f32` leaves the norms and the 1-D tensors out of the recipe. One run per part, as on the GGUF side, each with its own `download.conf`:

```
cd "$WORKING_DIRECTORY" && \
mkdir -p "$MODEL"-"${MAINTAINER^^}"-SGLANG-SPLIT && cd "$MODEL"-"${MAINTAINER^^}"-SGLANG-SPLIT && \
GGUF_DOWNLOAD_CONF="$WORKING_DIRECTORY"/GGUF-Tool-Suite/models/"$MODEL"/download.conf quant_downloader.sh "$WORKING_DIRECTORY"/GGUF-Tool-Suite/models/"$MODEL"/"$RECIPE" --qtype sgl_bf16 -j 8
```

Then the same command in `mtp-"$MODEL"-"${MAINTAINER^^}"-SGLANG-SPLIT` and `mmproj-"$MODEL"-"${MAINTAINER^^}"-SGLANG-SPLIT` with `models/mtp-"$MODEL"/download.conf` and `models/mmproj-"$MODEL"/download.conf`, when the model has those parts. Everything the GGUF path does, this path does: the sha256 of every file against the map, the GPG signature on the map and on shard `00001`, resume, `--verify`, `--individual-tensors`, `-z`/`-zd`, `--special-node-id`. What it does not do is the GGUF-only extras (`gguf_info.py` verification, `--compute-*-map`, quantize-from-bf16), which have nothing to inspect in a safetensors file and are refused rather than ignored.

Assemble, then serve as above:

```
cd "$WORKING_DIRECTORY"/GGUF-Tool-Suite && . .venv/bin/activate && \
./sglang_write.py --assemble "$WORKING_DIRECTORY"/"$MODEL"-"${MAINTAINER^^}"-SGLANG-SPLIT "$WORKING_DIRECTORY"/mtp-"$MODEL"-"${MAINTAINER^^}"-SGLANG-SPLIT "$WORKING_DIRECTORY"/mmproj-"$MODEL"-"${MAINTAINER^^}"-SGLANG-SPLIT --out "$WORKING_DIRECTORY"/"$MODEL"-"${MAINTAINER^^}"-SGLANG-ASSEMBLED --name "$MODEL"-"${MAINTAINER^^}"-SGLANG
```

That writes `model-NNNNN-of-MMMMM.safetensors`, `model.safetensors.index.json`, `hf_quant_config.json` with the `quantized_layers` list read off the files themselves, `config.json` with the `quantization_config` edited in, and the tokenizer and preprocessor files out of the metadata shard. It copies bytes and quantizes nothing, so it needs no torch, and it refuses when a part the metadata shard's index names is missing from the directories you gave it. Measured on Qwen3.8-27B with the shipped 4.5101 bpw recipe: the assembled checkpoint is identical to the one `--build` writes from the same BF16 source in all 2,404 tensors and in all four shard files, byte for byte, provided you pass `--build`'s `--name`: the writer stores it as `producer` in every shard header, so a different name changes the four sha256 and nothing else.

## (optional) Adding a new model

The SGLang path needs three things per model, and none of them is a GPU benchmark:

1. Its own `kld_results.csv` and `group0/tensors.bf16.map`, which you already have if you made GGUF recipes for it.

2. The per-format degradation rows. These are a property of the format, not of the model, so you do not have to supply them: they are built into the tool, measured in SGLang on Qwen3.8-27B, and the preset prices your pool with them from a model folder alone. The recipe footer names them and the model they were measured on, so a prediction never reads as a measurement of yours. If you do measure your own, put them in `group0/kld_results_sglang.csv` and the preset prefers them; `--quant-degradation-csv` still overrides everything. Both of those warn, because neither changes the recipe. I tested three sources of those rows on Qwen3.8-27B: the measured ones, an interpolation of the model's own GGUF rows, and the round-to-nearest estimate of `fp4_rtn_metric.py`. They give byte-identical recipes at 25 of 27 sizes between 14.5 and 20 GB, and at the other two the checkpoints measure the same in the engine. With three rungs only the ranking of the types matters, so what the measured rows buy is the absolute predicted number in the footer (the substitutes under-report it five to seven times), not a better recipe. They live apart from `group0/kld_results.csv` because they are not the same kind of number: the GGUF rows are weight-only `llama-perplexity` measurements at full coverage, while four of the seven `sgl_*` rows also carry the activation quantization the kernels do at runtime, and every one of them is divided by the share of the sensitivity mass its checkpoint covers.

3. An entry in the architecture table so the tools know the model's tensor names, the fused groups, and how `convert_hf_to_gguf.py` transformed each tensor on the way to GGUF. This is a small declarative table (`sglang_native.py` and `sglang_gguf.py`), not a benchmark. If the architecture is unknown the tools refuse rather than guess; pass `--sgl-arch` to name it or add the entry. The entry also says whether the model's SGLang file can load a quantized token embedding table: GLM's builds it without a `quant_config`, so it can only load BF16 there.

## About the shipped recipes

The recipes in `recipe_examples/sglang_recipes/` are made by the one-line command above; each footer records the exact command it was produced with, so you can re-run that line from the model folder and get the same assignments back (the filename differs only by the perplexity, which `--add-ppl` puts there once measured). The four Qwen3.8-27B ones are the sizes I use myself, and I call them F, E2, P and B2:

- `F` is [`Qwen3.8-27B.THIREUS-4.5101bpw-7.4384ppl.14GB-GGUF_14GB-GPU_0GB-CPU.a0e9e82_4ca1dfe.recipe`](../recipe_examples/sglang_recipes/Qwen3.8-27B.THIREUS-4.5101bpw-7.4384ppl.14GB-GGUF_14GB-GPU_0GB-CPU.a0e9e82_4ca1dfe.recipe)
- `E2` is [`Qwen3.8-27B.THIREUS-4.8217bpw-7.2763ppl.15GB-GGUF_15GB-GPU_0GB-CPU.a0e9e82_fde61ad.recipe`](../recipe_examples/sglang_recipes/Qwen3.8-27B.THIREUS-4.8217bpw-7.2763ppl.15GB-GGUF_15GB-GPU_0GB-CPU.a0e9e82_fde61ad.recipe)
- `P` is [`Qwen3.8-27B.THIREUS-5.1767bpw-7.1846ppl.16GB-GGUF_16GB-GPU_0GB-CPU.a0e9e82_e555fc0.recipe`](../recipe_examples/sglang_recipes/Qwen3.8-27B.THIREUS-5.1767bpw-7.1846ppl.16GB-GGUF_16GB-GPU_0GB-CPU.a0e9e82_e555fc0.recipe)
- `B2` is [`Qwen3.8-27B.THIREUS-5.6648bpw-7.0780ppl.17GB-GGUF_17GB-GPU_0GB-CPU.a0e9e82_4dd8078.recipe`](../recipe_examples/sglang_recipes/Qwen3.8-27B.THIREUS-5.6648bpw-7.0780ppl.17GB-GGUF_17GB-GPU_0GB-CPU.a0e9e82_4dd8078.recipe)

Two KLD columns, because they answer two different questions. "Weights only" is the recipe's quantized weights put back into BF16 and measured with `llama-perplexity --kl-divergence` exactly like every GGUF recipe in this repo (250 chunks of the calibration corpus against the BF16 model), so it is directly comparable with the ik_llama.cpp recipes and the group0 table. "In SGLang" is the checkpoint measured in the engine over 2,550 positions of the same corpus, which adds the FP4 quantization of the activations the NVFP4 kernels perform at runtime; it is what you actually get. The two perplexity columns are the same split on `wiki.test.raw` (580 chunks of 512 tokens; the BF16 model reads 6.9548): weights only with ik_llama.cpp, and in SGLang with the activations quantized too, which costs 0.7% to 2.6% of perplexity here, most at the smallest size. The KLD columns are on the calibration corpus and the perplexity columns on wikitext, and wikitext reads the same weights about twice as hard, so do not convert one into the other.

| name | `--gpu-tensors-max-size` | bpw | KLD, weights only | KLD in SGLang | PPL, weights only | PPL in SGLang | prefill tok/s, 1 / 8 streams | decode tok/s, no speculation, 1 / 8 | decode tok/s, NEXTN, 1 / 8 |
|---|---|---|---|---|---|---|---|---|---|
| [F](../recipe_examples/sglang_recipes/Qwen3.8-27B.THIREUS-4.5101bpw-7.4384ppl.14GB-GGUF_14GB-GPU_0GB-CPU.a0e9e82_4ca1dfe.recipe) | 15.1GB | 4.51 | 0.0388 | 0.0590 | 7.2674 | 7.4384 | 11,039 / 14,175 | 88 / 566 | 161 / 925 |
| [E2](../recipe_examples/sglang_recipes/Qwen3.8-27B.THIREUS-4.8217bpw-7.2763ppl.15GB-GGUF_15GB-GPU_0GB-CPU.a0e9e82_fde61ad.recipe) | 16.209GB | 4.82 | 0.0276 | 0.0444 | 7.1557 | 7.2763 | 10,782 / 13,714 | 84 / 543 | 147 / 844 |
| [P](../recipe_examples/sglang_recipes/Qwen3.8-27B.THIREUS-5.1767bpw-7.1846ppl.16GB-GGUF_16GB-GPU_0GB-CPU.a0e9e82_e555fc0.recipe) | 17.41GB | 5.18 | 0.0222 | 0.0349 | 7.0944 | 7.1846 | 10,447 / 12,984 | 78 / 513 | 143 / 811 |
| [B2](../recipe_examples/sglang_recipes/Qwen3.8-27B.THIREUS-5.6648bpw-7.0780ppl.17GB-GGUF_17GB-GPU_0GB-CPU.a0e9e82_4dd8078.recipe) | 19.044GB | 5.66 | 0.0162 | 0.0266 | 7.0338 | 7.0780 | 9,799 / 11,963 | 72 / 474 | 136 / 783 |
| [RadixArk's Qwen3.8-27B-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-27B-NVFP4), for reference | 20.15GB | 5.99 | 0.0260 | 0.0427 | 7.1526 | 7.2809 | 9,766 / 12,333 | 76 / 473 | 149 / 834 |

The `14GB` to `17GB` in the filenames is the merger's rounded on-disk size, not the budget; the budget is the column above. For scale, the same measurement gives the GGUF quants of this model q4_K 0.0071 at 4.5 bpw, q5_K 0.0026 at 5.5 bpw and q3_K 0.0280 at 3.4 bpw, with one caveat in the GGUF side's favour: those quants were made with an imatrix built on the same corpus the KLD is measured on, which reads a k-quant low (held-out measurements by others put Q4_K_M for a model this size at 0.010 to 0.022). Taken together with what other labs measure for NVFP4, the honest summary is that NVFP4 weights, which use no calibration data and no scale search, sit about one to one and a half bits behind an imatrix k-quant, and the activation quantization adds more on top. What it buys is speed on Blackwell (FP4 tensor cores, and the SGLang serving stack with NEXTN), and within the SGLang world these recipes beat RadixArk's NVFP4 checkpoint on quality at a smaller size. If quality per byte is what you want and ik_llama.cpp speed is enough, make a GGUF recipe instead.

Speeds were measured on one RTX PRO 6000 Blackwell 96 GB (sm_120) with SGLang 0.5.19 (commit d06f3bec8), 1 and 8 concurrent streams, the NEXTN column with 3 speculative steps and 4 draft tokens. The speeds are for that GPU and engine build; another GPU, in particular one without NVFP4 tensor cores, will not see them and may not prefer the same recipe, while the sizes and the KLD hold anywhere. Re-measured on a later build of my fork (0dda0b38) with the checkpoints built from the shipped scale set, on an otherwise idle machine: plain decode reproduces the table (F 88 / 566, B2 72 / 474), and NEXTN reads F 153 / 899, E2 146 / 835, P 135 / 829, B2 133 / 790 and RadixArk 147 / 810, within noise of the table at 8 streams and up to 6% under it at one stream, where two passes cannot separate the difference from the run-to-run spread. The two engine builds were also compared request by request on identical prompts and differ by 0.07%, so an earlier reading of NEXTN 4% to 10% slower on the newer build came from a busy host during that measurement, not from the engine or the recipes (the scales do not touch speed: the recipes' bytes and kernels are the same). The predicted KLD the tool prints is on the in-SGLang scale and runs a little conservative (0.064, 0.050, 0.041, 0.031 for these four).

About the activation scales. The numbers above are measured with the scale set the guide produces, `models/Qwen3.8-27B/sglang_scales/` (131,072 tokens of the imatrix corpus). Before settling on it I measured the `F` recipe (every one of its 402 modules is NVFP4, so it is the recipe where the scale matters most) six times, changing nothing but where the scales came from: RadixArk's checkpoint, and `sglang_calibrate.py` on 8,192, 32,768 and 131,072 tokens of the imatrix corpus, on 32,768 tokens of `wiki.train.raw`, and on a held-out slice of the imatrix corpus. Served in SGLang, none of the five differs from RadixArk's beyond the noise of the corpus sample, and the same holds for all four recipes rebuilt with the shipped set and paired against their RadixArk-scaled twins over the same positions and the same chunks (the largest paired t over eight comparisons is 1.46). The text matters more than the length: at the same 32,768 tokens the wikitext set is the worst of the six and both imatrix sets beat it, which is what you expect of a statistic that lives in the tail of the distribution. The measurements are recorded in the calibration notes (the CALIBRATION.json next to the shipped scale set carries the chosen corpus and token budget).

The three GLM-4.7 recipes are priced off the built-in rows too, the preset says so on stderr and in the footer. Two tensors stay BF16 by the architecture's advisory, and the recipe footer names both: the token embedding, because GLM's SGLang model file cannot load a quantized one, and the output head, because I measured what quantizing it costs (below). That is also why the smallest recipe is 203.9 GB and not smaller. Measured in SGLang the same way as the Qwen ones (2,550 positions of the calibration corpus for the KLD, 565 chunks of `wiki.test.raw` for the perplexity), with the shipped scale set `models/GLM-4.7/sglang_scales/`:

| name | `--gpu-tensors-max-size` | bpw | KLD in SGLang | PPL in SGLang |
|---|---|---|---|---|
| [floor](../recipe_examples/sglang_recipes/GLM-4.7.THIREUS-4.5522bpw-4.3640ppl.189GB-GGUF_189GB-GPU_0GB-CPU.2bc59af_bf855df.recipe) | 203.90GB (203900926828B) | 4.55 | 0.0911 | 4.3640 |
| [mid](../recipe_examples/sglang_recipes/GLM-4.7.THIREUS-4.7166bpw-4.1510ppl.196GB-GGUF_196GB-GPU_0GB-CPU.2bc59af_fb79dca.recipe) | 211.28GB (211275244537B) | 4.72 | 0.0630 | 4.1510 |
| [top](../recipe_examples/sglang_recipes/GLM-4.7.THIREUS-4.8190bpw-4.1396ppl.201GB-GGUF_201GB-GPU_0GB-CPU.2bc59af_fbe1406.recipe) | 220.00GB (220000000000B) | 4.82 | 0.0632 | 4.1396 |
| [Salyut1's GLM-4.7-NVFP4](https://huggingface.co/Salyut1/GLM-4.7-NVFP4), for reference | 200.79GB (as published, no draft head) | 4.55 | 0.0912 | 4.3389 |

GLM pays more for the activation quantization than Qwen does, and the tool's predictions for GLM run optimistic (the built-in rows are a property of the format, not of the model). About the head: the first builds of these recipes quantized `output.weight` (NVFP4 in floor and mid, FP8 in top), and served in SGLang the floor read 0.5% higher in perplexity and 0.005 higher in KLD than the same build with the head in BF16 (paired over the same chunks and positions, t = 11 and 6), for 1.1 GB saved out of 210. So the head is pinned, and the recipes say so. About the scales: Salyut1's checkpoint calibrated its scales on cnn_dailymail and Nemotron post-training text with a budget they do not state; the shipped set walks the whole imatrix corpus (413,696 tokens, 19 minutes on an RTX 5090) and covers the tail at least as well by the only proxy two sets allow. Served, the floor built with the shipped set reads 4.3640 against 4.3389 with Salyut1's scales on the same weights (paired t = 1.8 on the perplexity, 0.0 on the KLD), and mid and top, whose assignments moved with the head pin, sit within the same noise (t = 1.1 and 1.3 on the perplexity, 0.9 and 1.7 on the KLD). Note that the KLD positions lie inside the corpus the scales were fitted on, so the perplexity column is the out-of-sample one. The floor's weights are Salyut1's: every packed tensor and every weight scale of the 174,281 tensors the two checkpoints have in common is byte-identical (the writer's codec is ModelOpt's), what differs is the activation scales, plus the NEXTN draft head (9.96 GB, layer 92) which their checkpoint does not ship, and a set of FP8 KV-cache scale scalars that all read 1.0 and change nothing; the bpw is the same method as the Qwen row, the quantised language model's bytes over its parameters. Mid and top are what the size buys on top of that floor. One caveat when you set these against GLM-4.7's own group0 table: that table was measured against an iq6_k baseline rather than BF16, which reads every row higher (its q8_0 sits at 0.0098), so it is not the same axis; the weights-only GLM measurements of the earlier recipe generation are described in `models/GLM-4.7/group0/sglang_notes.txt`.

The measurements behind all these numbers were made in SGLang on the lanes described in `models/<model>/group0/sglang_notes.txt`.
