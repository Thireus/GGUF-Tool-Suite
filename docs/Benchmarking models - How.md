# Benchmarking models - How?

_In this section I will explain how to benchmark a model and produce the KLD (or PPL) calibration data `kld_results.csv` required by `quant_assign.py` to produce optimum quant mix `.recipe` files. I will also cover producing the optional (but very useful) degradation data `group0/kld_results.csv` employed by the greedy quant distribution technique to optimally spread the quant assignment distribution._

Before we get started I feel it's important to explain a few points:

1. [GGUF](https://huggingface.co/docs/hub/gguf) files are composed primarily of [tensors](https://en.wikipedia.org/wiki/Tensor), which can be thought of as large contiguous blocks of floating-point values. These tensors store the learned weights of the model. The only portion of a GGUF file that is not tensor data is the header section at the beginning. This section contains metadata such as the number of tensors, their names and shapes, model configuration, and additional information like the Jinja template. For practical reasons, I prefer to separate this metadata into its own .gguf shard. This allows metadata changes—such as updating the Jinja template—to be made by modifying only a small, dedicated file, rather than requiring updates to the much larger tensor data files.

2. Each tensor represents a set of learned parameters that transform inputs into outputs during inference. In essence, tensors define how data flows through the model and how intermediate representations are computed. Quantization aims to reduce the memory footprint of these tensors by lowering the numerical precision of their values (for example, converting from 16-bit floating point to lower-bit representations). The goal is to preserve behavior as closely as possible to the original, unquantized model while significantly reducing storage and computational requirements. The quantization types used in implementations such as [llama.cpp](https://github.com/ggml-org/llama.cpp) and [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) achieve this through specialized mathematical algorithms. These algorithms compress tensor data by approximating the original values with lower-precision formats, while maintaining outputs within a measurable and acceptable margin of error compared to the full-precision model. This is that margin of error that the `benchmark_each_tensor.sh` measures for each tensor of the model. These measurements are then used to produce a `kld_results.csv` (or `ppl_results.csv`) file, with the help of `collect_ppl_results.sh`, which lists which tensors are more sensitive to quantization than others.

3. Almost everyone is creating "dynamic" GGUFs, but not all go through the effort of carefully determining which specific tensor could use one quantization type over another with a measurable evidence. This process is tedious, time-consuming, and requires detailed analysis. Many quantization approaches rely on general heuristics. For example, it is commonly assumed that tensors like `token_embd.weight` and `output.weight` require higher precision due to their importance. Others apply rules based on tensor roles, having observed that tensors such as `ffn_down_exps.weight` tend to benefit more from higher precision than `ffn_up_exps.weight` or `ffn_gate_exps.weight`. These approaches result in internally consistent—but still heuristic—methodologies. Some go further by applying per-layer strategies, based on the observation that certain layers (often early or late in the network) may be less sensitive to quantization. Others attempt to analyze tensor statistics to estimate which tensors are more error-prone and should therefore retain higher precision. However, none of these approaches fully measure the real impact of quantization at a per-tensor level during actual inference. Instead, they rely on approximations, assumptions, or partial metrics rather than directly optimizing for end-use behavior. In practice, this is understandable—performing such fine-grained analysis is expensive and time-intensive. For a community that has historically been eager for GGUF releases, quickly producing "good enough" GGUFs has been the norm. That said, "good enough" is not the standard I’m aiming for. With GGUF-Tool-Suite, the goal is to change this paradigm by providing both the methodology and tooling to automatically and systematically determine the optimal combination of tensor quantization types for any given model and target size. The objective is to make it possible for **anybody** to generate truly optimized "dynamic quants" that maximize quality within a specified size constraint—without relying on guesswork or static heuristics.

4. Not all tensors can be quantized to every quantization type, and not all tensors should be quantized at all. For example, smaller tensors stored in full f32 precision are often critical to model stability and are typically best left unmodified. Additionally, certain tensors—such as `token_embd.weight`—may have shapes that are incompatible with specific quantization types. The `llama-quantize` utility provided in [llama.cpp](https://github.com/ggml-org/llama.cpp) and [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) includes a set of built-in checks and fallback rules. When an incompatibility is detected, these rules may automatically override the user-selected quantization type and substitute a different one. These safeguards are often well-justified, as they help prevent outright invalid tensor representations. However, in some cases, they act conservatively to prevent severe degradation in model quality. This behavior runs counter to the goals of the GGUF-Tool-Suite methodology, which is to explicitly measure the impact of quantization decisions rather than rely on implicit safeguards. For this reason, I have removed some of these checks in this [ik_llama.cpp fork](https://github.com/Thireus/ik_llama.cpp/tree/th/quantize_individual_tensors), which I use to quantize models. This allows full control over per-tensor quantization choices, enabling direct measurement of their effects.

## Requirements

Hardware:

```
CPU: YES (if model doesn't fit into VRAM)
GPU: YES
```

Note: I personally own 3xRTX6000Pro + 1xRTX5090 and a i9-7980xe with 256GB of dual-channel DDR4. This isn't because producing calibration data requires such hardware, but because for my use case I need to run large LLMs.

If you own a GPU with 16GB of VRAM and 32GB of DDR4, this should be plenty to benchmark models that can already fit your hardware when quantized to a decent quant type - considering that benchmarking only requires a tiny context window of 512 tokens. For example, 
[Qwen3.5-27B](https://huggingface.co/Thireus/Qwen3.5-27B-THIREUS-BF16-SPECIAL_SPLIT/tree/main) which has a BF16 size of 53GB can be benchmarked on a 16GB VRAM + 32GB RAM - the way this is done is: number one, offload layers to the CPU and number two choose a `BASELINE_QTYPE` that isn't BF16, for example [IQ5_KS](https://huggingface.co/Thireus/Qwen3.5-27B-THIREUS-IQ5_KS-SPECIAL_SPLIT/tree/main).

Let's get into it... Make sure you edit and paste these variables in your terminal:

```
WORKING_DIRECTORY='/AI' # Full path please!
MODEL='Qwen3.5-0.8B'
MAINTAINER='THIREUS' # Or use your name!
BASELINE_QTYPE="bf16" # The highest "pure" quant of this model your hardware can run
TARGET_QTYPE="iq1_kt" # The lowest "pure" quant of this model
```

Important: You will notice that `q` quants end with uppercase letters such as `_K` or `_KV`. This is very important to get this right! Quant types are often case-sensitive!

The `TARGET_QTYPE` variable defines the quantization type that each benchmarking run will drop the invividually assessed tensors to.

Important: `BASELINE_QTYPE` and `TARGET_QTYPE` must be chosen with as few quant impurities as possible. You can use [gguf.thireus.com](https://gguf.thireus.com/) to identify which quants are impure (they are highlighted in orange and red), alternatively you can identify this yourself via the tensors.map files produced that record if tensor quantization fallback to other quant types have occured. I also recommend against using `*_BN` quants. Soft-impure quants (quant type replaced by row-interleaved or non-ri equivalent) are always better than hard-impure quants. Finally, the largest bpw gap there is between `BASELINE_QTYPE` and `TARGET_QTYPE`, the better the calibration data will be.

## Prepare the environment

```
apt-get install screen gpg curl lbzip2 python3 # Run as root
```

Create the working directory (where all files will be downloaded and produced):

```
mkdir -p "$WORKING_DIRECTORY"
```

Obtain the GGUF-Tool-Suite and prepare it for the chosen `$MODEL`:

```
cd "$WORKING_DIRECTORY" && \
GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://github.com/Thireus/GGUF-Tool-Suite/ "$MODEL"-BENCH
cd "$MODEL"-BENCH && git pull # Update it
rm -f download.conf && cp -rf models/"$MODEL"/download.conf .
```

Download the `$BASELINE_QTYPE` of our model using `quant_downloader.sh`:

```
cd "$WORKING_DIRECTORY" && \
cd "$MODEL"-BENCH && \
export PATH="$WORKING_DIRECTORY"/"$MODEL"-BENCH/:$PATH && \
mkdir "$MODEL"-"${MAINTAINER^^}"-"${BASELINE_QTYPE^^}"-SPECIAL_SPLIT && \
cd "$MODEL"-"${MAINTAINER^^}"-"${BASELINE_QTYPE^^}"-SPECIAL_SPLIT && \
echo ".*=$BASELINE_QTYPE" > "$BASELINE_QTYPE".recipe && \
quant_downloader.sh "$BASELINE_QTYPE".recipe --qtype "$BASELINE_QTYPE"
```

## Configure the benchmark script

Now that we have a `$BASELINE_QTYPE` GGUF sharded version of our model with tensors.map we can proceed to configuring the `benchmark_each_tensor.sh` script.

First we need to list all the tensors of the model and the relevant assigned quants, while locking in place certain tensors we do not which to benchmark, such as the `f32` ones. Inject this into the `USER_REGEX` section of the `benchmark_each_tensor.sh` script and set the `BASELINE_QTYPE`:

```
cd "$WORKING_DIRECTORY" && \
cd "$MODEL"-BENCH && \
cd "$MODEL"-"${MAINTAINER^^}"-"${BASELINE_QTYPE^^}"-SPECIAL_SPLIT && \
cat tensors.map | cut -d: -f 3,5 | sed 's/:dtype/$/' | sed 's/\./\\\./g' | quants_regex_merger.sh --model-name "$MODEL" --no-file > "$BASELINE_QTYPE".recipe && \
cat "$BASELINE_QTYPE".recipe | sed 's/=f32/=f32=locked/g' | sed "s/\^.*/'&'/" > ${BASELINE_QTYPE^^}_USER_REGEX.txt && \
sed -i '/^USER_REGEX=(/,/^[[:space:]]*)/{//!d}; /^USER_REGEX=(/r '"${BASELINE_QTYPE^^}"'_USER_REGEX.txt' "$WORKING_DIRECTORY"/"$MODEL"-BENCH/benchmark_each_tensor.sh && \
sed -i '/^BASELINE_QTYPE=/c\BASELINE_QTYPE="'"$BASELINE_QTYPE"'"' "$WORKING_DIRECTORY"/"$MODEL"-BENCH/benchmark_each_tensor.sh
```

Then, you will need ik_llama.cpp's `llama-perplexity` binary which can be found pre-compiled at [Thireus' fork of ik_llama.cpp](https://github.com/Thireus/ik_llama.cpp/tags) - look for the `main*` tags. Alternatively, compile it yourself as instructed [here](https://github.com/ikawrakow/ik_llama.cpp/blob/main/docs/build.md), but make sure you use the `-DGGML_MAX_CONTEXTS=2048` cmake option which lifts the limit of the number of .gguf shards ik_llama.cpp can load at once - not using this compilation option (combined with `ulimit -n 9999`) will result in sharded models failing to load.

Finally, we need to edit the `PPL_COMMAND_TEMPLATE` found in the `benchmark_each_tensor.sh` script. This template is what the script will use to execute `llama-perplexity` and compute the [KLD](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence) and [PPL](https://en.wikipedia.org/wiki/Perplexity) metrics of the model after each tensor quantization gets dropped from `BASELINE_QTYPE` to `TARGET_QTYPE`. To find the best template to use you must identify which `llama-perplexity` parameters give you the fastest benchmarking speed. I recommend to play around with the parameters of ik_llama.cpp's `llama-perplexity` and the curent `$MODEL` you've downloaded. For example, with trial and error I found that the best benchmarking speed for my hardware for the `$MODEL` were achieved when using the following `llama-perplexity` parameters (with both GPU and CPU offloading):

```
MODEL_FILE=$(ls "$MODEL"-*-SPECIAL_TENSOR-00001-of-*.gguf) && \
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,2,1 llama-perplexity \
-m $MODEL_FILE -mla 3 -fa on -amb 1024 -ctk f16 -c 512 -ngl 99 \
-ot "blk\.(3|4|5)\.ffn_.*=CUDA0" -ot "blk\.(6|7|8)\.ffn_.*=CUDA1" -ot "blk\.(9|10)\.ffn_.*=CUDA2" \
-ot exps=CPU -b 512 -ub 512 --warmup-batch --no-mmap --threads $(nproc) --main-gpu 0 --seed 1337 \
-f imatrix-calibration-corpus-v02.txt --chunks 250
```

You will see in the logs the ETA, the aim being to reduce that ETA as much as possible:

```
perplexity: 25.21 seconds per pass - ETA 13.12 minutes
```

Note: If you are using KLD benchmarking (default behaviour of the script and recommended for more accurate calibration results), you must use `-b 512 -ub 512` because those are the values that will sadly be enforced by `llama-perplexity` no matter what else you choose.

Once you have identified the best parameters, you must manually edit the `PPL_COMMAND_TEMPLATE` template variable of the `benchmark_each_tensor.sh` script. The location of the script is here:

```
ls -l "$WORKING_DIRECTORY"/"$MODEL"-BENCH/benchmark_each_tensor.sh
```

Open the script with your favourite editor, locate the line that states `# 9. PPL command template:`. Underneath that line you'll find `PPL_COMMAND_TEMPLATE='...`, replace the content of that multi-line variable with your command but make sure you add at the end of it these additional parameters: `-f imatrix-calibration-corpus-v02.txt --chunks ${PPL_COMMAND_CHUNKS_TO_PROCESS}`. Also, the `$MODEL_FILE` variable set before can be replaced by `{MODEL_FILE}` which the script will automatically replace by the model's first shard. So, our example becomes:

```
PPL_COMMAND_TEMPLATE='CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,2,1 llama-perplexity \
-m {MODEL_FILE} -mla 3 -fa on -amb 1024 -ctk f16 -c 512 -ngl 99 \
-ot "blk\.(3|4|5)\.ffn_.*=CUDA0" -ot "blk\.(6|7|8)\.ffn_.*=CUDA1" -ot "blk\.(9|10)\.ffn_.*=CUDA2" \
-ot exps=CPU -b 512 -ub 512 --warmup-batch --no-mmap --threads 36 --main-gpu 0 --seed 1337 \
-f imatrix-calibration-corpus-v02.txt --chunks ${PPL_COMMAND_CHUNKS_TO_PROCESS}'
```

Make sure you get this `PPL_COMMAND_TEMPLATE` right. If you plan on tweaking it later in the middle of a benchmarking session, you will likely have to redo benchmarking from the beginning because different parameters likely lead to slightly different metrics - which is going to corrupt the calibration data.

Obtain the `imatrix-calibration-corpus-v02.txt` file which is used for KLD and PPL calibration data benchmarking (see why [here](https://github.com/Thireus/GGUF-Tool-Suite/discussions/23#discussioncomment-14764941)):

```
cd "$WORKING_DIRECTORY" && \
curl -L 'https://gist.githubusercontent.com/ubergarm/edfeb3ff9c6ec8b49e88cdf627b0711a/raw/ba5b01b6960a86874592f5913e283746ff734483/ubergarm-imatrix-calibration-corpus-v02.txt' -o imatrix-calibration-corpus-v02.txt && \
cd "$MODEL"-BENCH && \
cd "$MODEL"-"${MAINTAINER^^}"-"${BASELINE_QTYPE^^}"-SPECIAL_SPLIT && \
cp -f "$WORKING_DIRECTORY"/imatrix-calibration-corpus-v02.txt .
```

Important: Double check that the additional `-f imatrix-calibration-corpus-v02.txt --chunks ${PPL_COMMAND_CHUNKS_TO_PROCESS}` parameters are present at the end of your `PPL_COMMAND_TEMPLATE` variable.

## (optional) Reduce benchmarking costs

_We'll assume you've already made sure all your VRAM has been maxed out, since GPU VRAM speed is superor to CPU RAM speed. So, we're not going to touch the `PPL_COMMAND_TEMPLATE` at this point._

Let's identify tensors that can be benchmarked together:

```
cd "$WORKING_DIRECTORY" && \
cd "$MODEL"-BENCH && \
cd "$MODEL"-"${MAINTAINER^^}"-"${BASELINE_QTYPE^^}"-SPECIAL_SPLIT && \
python3 ../model_tensor_sizes.py --sort "$BASELINE_QTYPE".recipe tensors."$BASELINE_QTYPE".map | grep -v '=f32'
```

Example of output:

```
 406.00 GB  ^blk\.([3-9]|[1-5][0-9]|60)\.ffn_down_exps\.weight$=bf16
 406.00 GB  ^blk\.([3-9]|[1-5][0-9]|60)\.ffn_up_exps\.weight$=bf16
 406.00 GB  ^blk\.([3-9]|[1-5][0-9]|60)\.ffn_gate_exps\.weight$=bf16
  13.34 GB  ^blk\.([0-9]|[1-5][0-9]|60)\.attn_output\.weight$=bf16
   4.29 GB  ^blk\.([0-9]|[1-5][0-9]|60)\.attn_q_b\.weight$=bf16
   1.73 GB  ^token_embd\.weight$=bf16
   1.73 GB  ^output\.weight$=bf16
   1.59 GB  ^blk\.([3-9]|[1-5][0-9]|60)\.ffn_down_shexp\.weight$=bf16
   1.59 GB  ^blk\.([3-9]|[1-5][0-9]|60)\.ffn_up_shexp\.weight$=bf16
   1.59 GB  ^blk\.([3-9]|[1-5][0-9]|60)\.ffn_gate_shexp\.weight$=bf16
   1.25 GB  ^blk\.([0-9]|[1-5][0-9]|60)\.attn_q_a\.weight$=bf16
 976.00 MB  ^blk\.([0-9]|[1-5][0-9]|60)\.attn_k_b\.weight$=bf16
 976.00 MB  ^blk\.([0-9]|[1-5][0-9]|60)\.attn_v_b\.weight$=bf16
 756.00 MB  ^blk\.[0-2]\.ffn_down\.weight$=bf16
 756.00 MB  ^blk\.[0-2]\.ffn_up\.weight$=bf16
 756.00 MB  ^blk\.[0-2]\.ffn_gate\.weight$=bf16
 480.38 MB  ^blk\.([0-9]|[1-5][0-9]|60)\.attn_kv_a_mqa\.weight$=bf16
```

From here we can decide to bundle together the `ffn_.*_shexp.weight` tensors of the same layer, because they of the same tensor class (they work together), they are relatively small in size and they are many, and we don't want to spend that much time benchmarking them knowing they don't heavily contribute to the model size. For similar reasons, we can decide to bundle together `attn_q_a.weight` and `attn_q_b.weight` as well as `attn_k_b.weight` and `attn_v_b.weight`. Doing so will drastically reduce the number of benchmark rounds we have to complete to produce the callibration data of our model - we have effectively eliminated `2*58+61+61 = 238` rounds. Bundling tensors together also means their quantization sensitivity will be measured to be greated compared to when measured individually, which means if we do not apply a correction factor `c * kld[t1+t2] = kld[t1] + kld[t2]` they will be assigned quants that are of higher precision than they should, which we're just fine with because of their small size and negligible impact on the overall model size.

This approach helps reducing electricity cost and time while minimizing the impact on the calibration data quality.

The `--group-tensors` that will be used with the `benchmark_each_tensor.sh` script can be constructed using this command, make sure to edit the `BUNDLED_TENSORS` to your own liking:

```
BUNDLED_TENSORS=(
'^blk\.([3-9]|[1-5][0-9]|60)\.ffn_.*_shexp\.weight$'
'^blk\.([0-9]|[1-5][0-9]|60)\.attn_(q_a|q_b)\.weight$'
'^blk\.([0-9]|[1-5][0-9]|60)\.attn_(k_b|v_b)\.weight$'
) && \
echo '--group-tensors \' && for t in "${BUNDLED_TENSORS[@]}"; do us=$'\x1f'; tmp=${t/\\./$us}; tmp=${tmp/\\./$us}; p1=${tmp%%$us*}; r1=${tmp#*$us}; p2=${r1%%$us*}; second=${r1#*$us}; first=${p1}${us}${p2}${us}; first=${first//$us/\\.}; echo "$first" | quants_regex_expander.sh | while IFS= read -r x; do printf "'%s%s' \\\\\n" "$x" "$second"; done; done | head -c -2
```

Which would output for example:

```
--group-tensors \
'^blk\.3\.ffn_.*_shexp\.weight$' \
'^blk\.4\.ffn_.*_shexp\.weight$' \
...
'^blk\.59\.ffn_.*_shexp\.weight$' \
'^blk\.60\.ffn_.*_shexp\.weight$' \
'^blk\.0\.attn_(q_a|q_b)\.weight$' \
'^blk\.1\.attn_(q_a|q_b)\.weight$' \
...
'^blk\.59\.attn_(q_a|q_b)\.weight$' \
'^blk\.60\.attn_(q_a|q_b)\.weight$' \
'^blk\.0\.attn_(k_b|v_b)\.weight$' \
'^blk\.1\.attn_(k_b|v_b)\.weight$' \
...
'^blk\.59\.attn_(k_b|v_b)\.weight$' \
'^blk\.60\.attn_(k_b|v_b)\.weight$'
```

Note: I have intentionally trimmed the output with `...`. You can find the complete version [here](https://github.com/Thireus/GGUF-Tool-Suite/issues/34#issuecomment-3761825253).

Important: When using the `--group-tensors` parameter, the following file will be produced to keep record of the individual groups used during benchmarking: `bench_ppl*group_mapping.*.txt`. This file is later used when benchmark results are collected with `collect_ppl_results.sh` to assign the bundled benchmark results to the relevant individual tensors when `--expand-groups` is used.

## Run the calibration data benchmark

I recommend running the benchmarking script into a screen because this is a task that takes several days (sometimes weeks) to complete. During that time I would encourage you to avoid using your GPUs as any variation of VRAM usage could cause benchmarks to fail. Should the `benchmark_each_tensor.sh` script be killed, the next time it is launched with the same parameters it can self-identify which tensors (or group of tensors) have already benchmarked and resume from there. There is also a smart random tensor section implemented into the script that identifies which tensors to pick up for benchmarking first.

Run the following command to start benchmarking:

```
cd "$WORKING_DIRECTORY" && \
cd "$MODEL"-BENCH && \
export PATH="$WORKING_DIRECTORY"/"$MODEL"-BENCH/:$PATH && \
cd "$MODEL"-"${MAINTAINER^^}"-"${BASELINE_QTYPE^^}"-SPECIAL_SPLIT && \
benchmark_each_tensor.sh --qtypes "$TARGET_QTYPE" --chunks 250
```

Note: If you also identified tensors that can be benchmarked together from the optional `Reduce benchmarking costs` section, make sure to append to the `benchmark_each_tensor.sh` parameters the additional `--group-tensors` parameter.

Important: I strongly advise against reducing the chunks to a value below `250`. This value was [assessed](https://github.com/Thireus/GGUF-Tool-Suite/issues/34#issuecomment-3519158063) to be the general minimum decent KLD/PPL that won't hurt the calibration data. However, I recommend bumping this value to towards its maximum if possible.

It will take time... and could be optimised if we had a `llama-perplexity` binary that could only swap the necessary tensors after each benchmark round instead of unloading/reloading the entire model... I have opened [an issue](https://github.com/Thireus/GGUF-Tool-Suite/issues/30) in this regard if someone is willing to create a PR.

Once benchmarking is completed you will see the following message appear:

```
✅ All qtypes processed. (single run mode: --infinite-loop=false)
```

However, this isn't the end of the story as we still need to validate that all benchmark logs are present and valid.

Their filenames are as follow:

```
bench_ppl(_kld)_result.TENSOR_OR_GROUP_NAME.TARGET_QTYPE.CHUNKS.txt
```

Additionally, the baseline benchmark file can also be found, it always only contains PPL results (as KLD for the baseline is 0):

```
bench_ppl(_kld)_result.baseline.BASELINE_QTYPE.CHUNKS.txt
```

and if you've used the `--group-tensors` parameter, you will find the following file which contains the list of group identifiers for your bundled tensors:

```
bench_ppl(_kld)_group_mapping.TARGET_QTYPE.CHUNKS.txt
```

Use the following command to count how many tensors/groups were benchmarked:

```
cd "$WORKING_DIRECTORY" && \
cd "$MODEL"-BENCH && \
export PATH="$WORKING_DIRECTORY"/"$MODEL"-BENCH/:$PATH && \
cd "$MODEL"-"${MAINTAINER^^}"-"${BASELINE_QTYPE^^}"-SPECIAL_SPLIT && \
ls bench_*result\.*.txt | wc -l
```

Use the following command to identify which benchmark files are corrupted (because don't contain results), which you will need to delete:

```
cd "$WORKING_DIRECTORY" && \
cd "$MODEL"-BENCH && \
cd "$MODEL"-"${MAINTAINER^^}"-"${BASELINE_QTYPE^^}"-SPECIAL_SPLIT && \
grep -L -E 'KL divergence statistics|Final estimate: PPL over' bench_*result\.*.txt
```

If you've identified that some benchmark files are missing or that you've had to delete some, you must run the same `benchmark_each_tensor.sh` with the exact same parameters to resume benchmarking the remaining tensors/groups.

## Prepare the benchmark results collection

_Similar to how we've configured the `benchmark_each_tensor.sh`, the `collect_ppl_results.sh` script which is used to produce the `kld_results.csv` and `ppl_results.csv` files needs to be configured.

Run this command to replace the `USER_REGEX` tensors to collect in the `collect_ppl_results.sh` script: 

```
cd "$WORKING_DIRECTORY" && \
cd "$MODEL"-BENCH && \
cd "$MODEL"-"${MAINTAINER^^}"-"${BASELINE_QTYPE^^}"-SPECIAL_SPLIT && \
sed -i '/^USER_REGEX=(/,/^[[:space:]]*)/{//!d}; /^USER_REGEX=(/r /dev/fd/3' 3< <(cat "${BASELINE_QTYPE^^}_USER_REGEX.txt" | grep -v '=locked' | sed "s/=.*'$/'/g") "$WORKING_DIRECTORY"/"$MODEL"-BENCH/collect_ppl_results.sh && sed -i '/^BASELINE_QTYPE=/c\BASELINE_QTYPE="'"$BASELINE_QTYPE"'"' "$WORKING_DIRECTORY"/"$MODEL"-BENCH/collect_ppl_results.sh
```

## Collect the calibration data results

_At this point it is assumed you have successfully ran the `benchmark_each_tensor.sh` script and identified that all benchmark files are present and valid. And that you've also configured the `collect_ppl_results.sh` script._

Collect the benchmark results using this command:

```
cd "$WORKING_DIRECTORY" && \
cd "$MODEL"-BENCH && \
export PATH="$WORKING_DIRECTORY"/"$MODEL"-BENCH/:$PATH && \
cd "$MODEL"-"${MAINTAINER^^}"-"${BASELINE_QTYPE^^}"-SPECIAL_SPLIT && \
collect_ppl_results.sh --qtypes "$TARGET_QTYPE" --chunks 250 --no-percentage --auto-baseline "$BASELINE_QTYPE"
```

Important: If you have used the `--group-tensors` parameter when running the `benchmark_each_tensor.sh` script, you must add to the following two additional parameters to the `collect_ppl_results.sh` script:

```
--expand-groups --group-tensors-map bench_*_group_mapping.$BASELINE_QTYPE.250.txt
```

The `collect_ppl_results.sh ` script will output both a `kld_results.csv` and a `ppl_results.csv` file. Before claiming victory, we need to check if they contain all the expected results.

Note: Running the `collect_ppl_results.sh` script again will overwrite the produced .csv files.

Use the following command to identify which tensors (or groups) failed to be benchmarked:

```
cd "$WORKING_DIRECTORY" && \
cd "$MODEL"-BENCH && \
cd "$MODEL"-"${MAINTAINER^^}"-"${BASELINE_QTYPE^^}"-SPECIAL_SPLIT && \
awk -F',' 'NR==1 {for (i=1;i<=NF;i++) h[i]=$i; next} {for (i=1;i<=NF;i++) if ($i=="" || $i=="404" || $i=="404%") print h[i]}' kld_results.csv
```

If this command doesn't return anything, then you have successfully produced `kld_results.csv` and `ppl_results.csv` calibration data files. If not, then you must run the same `benchmark_each_tensor.sh` with the exact same parameters to resume benchmarking these remaining tensors/groups.

You can find the calibration data of models that have already been benchmarked in the https://github.com/Thireus/GGUF-Tool-Suite/tree/main/models directories.

## (optional) Run the degradation data benchmark

_The `quant_assign.py` methodology that produces the best recipe with minimal user input is the one called `greedy` which is enabled with the `--use-greedy-quant-assign` parameter. However, this technique relies on a secondary benchmark data (called group0 degradation data) which is used to identify how wide or narrow the range of quant assignments for a given target size has to be and which quantization types to priotise over others due to better quality/size ratio. Without this degradation data, `quant_assign.py` uses the degradation data of `Qwen3-4B-Thinking-2507`, which is already a fine degradation data for most model, but often requires guessing the ideal `--exponential-factor` to bend the metrics to best fit an hypothetical degradation of the target model. For this reason, while this is an optional step, I still recommend producing such data. We will assume here that you have already configured both `quant_assign.py` and `collect_ppl_results.sh` as instructed in the previous steps - no new configuration is necessary._

The group0 degradation data benchmark is produced by benchmarking all "pure" quants of a model. That means, computing the KLD (or PPL, although doesn't work well) of a model that has all its quantizable tensors set to a specific quantization type, and doing this for as many quantization types as possible. It is called group0 because we effectively set `--group-tensors '.*'` to the `benchmark_each_tensor.sh` script - meaning bundling all quantizable tensors into one single group (group ID 0).

Important: Before going any further, make sure you move any existing results in a safe place so they don't get mixed up with new benchmark data.

To move and backup your previous benchmark results, use the following command - the baseline results can be kept:

```
cd "$WORKING_DIRECTORY" && \
cd "$MODEL"-BENCH && \
cd "$MODEL"-"${MAINTAINER^^}"-"${BASELINE_QTYPE^^}"-SPECIAL_SPLIT && \
BACKUP_DATETIME=$(date +"%Y%m%d_%H%M%S") && \
mkdir -p "BACKUP_$BACKUP_DATETIME" && \
mv *.csv bench_*.txt "BACKUP_$BACKUP_DATETIME" && \
cp "BACKUP_$BACKUP_DATETIME"/bench_*result.baseline.*.txt .
```

Quant types supported by [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) for this tool suite:

```
iq1_s_r4 iq1_s iq1_bn iq1_kt iq1_m iq1_m_r4 iq2_bn iq2_bn_r4 iq2_xxs iq2_xxs_r4 iq2_kt iq2_ks iq2_xs iq2_xs_r4 iq2_k iq2_k_r4 iq2_s q2_K q2_k_r4 iq2_kl iq3_xxs iq3_xxs_r4 iq3_kt iq3_ks iq3_k iq3_k_r4 iq3_s iq3_s_r4 q3_K q3_k_r4 iq4_kss iq4_kt iq4_ks iq4_ks_r4 iq4_xs iq4_xs_r8 iq4_k iq4_k_r4 iq4_nl iq4_nl_r4 q4_0 q4_0_r8 q4_K q4_k_r4 q4_1 iq5_ks iq5_ks_r4 iq5_k iq5_k_r4 q5_0 q5_0_r4 q5_K q5_k_r4 q5_1 q6_0 q6_0_r4 q6_K q6_k_r4 iq6_k q8_KV q8_k_r8 q8_0 q8_0_r8 bf16
```

Ideally we'd want to obtain the group0 metrics for all possible quantization types (which is often what I do for small models as the cost is relatively low). However, we can take a few shortcuts to save on costs. First of all, apart from the `iq1_s/iq1_s_r4` pair, every other row-interleaved quants observe the same bpw and theoretical KLD/PPL results - which means we can safely exclude every `_r4` and `_r8` quants, providing the base quant exists - it is also a good strategy are in most cases these row-interleaved quants can be very slow. Other quants that can be excluded are the less-used ones such as `_KV` (very slow too) and `_BN` quants.

That leaves us with the following list:

```
iq1_s_r4 iq1_s iq1_kt iq1_m iq2_xxs iq2_kt iq2_ks iq2_xs iq2_k iq2_s q2_K iq2_kl iq3_xxs iq3_kt iq3_ks iq3_k iq3_s q3_K iq4_kss iq4_kt iq4_ks iq4_xs iq4_k iq4_nl q4_0 q4_K q4_1 iq5_ks iq5_k q5_0 q5_K q5_1 q6_0 q6_K iq6_k q8_0 bf16
```

Note: I could also recommend to further refine this list and select only quantization types that are not "hard-impure" - You can use [gguf.thireus.com](https://gguf.thireus.com/) to identify which quants are hard-impure (they are highlighted in red), alternatively you can identify this yourself via the tensors.map files produced that record if tensor quantization fallback to other quant types (that aren't base or row-interleaved, which are fine) have occured.

From there, you will need to assess how far in this list you can go while still being able to load models quantized to these types. For example, if your `BASELINE_QTYPE` is `q5_K` then you probably won't be able to load and benchmark `q5_1` and beyond since their bpw is higher than your baseline quantization type (which is already the max your hardware supports). If you decide to change your `BASELINE_QTYPE` for this benchmark, remember you will also need to edit the `BASELINE_QTYPE` found under the "# 8. Baseline QTYPE for baseline PPL+KLD computation" section of the `benchmark_each_tensor.sh` script.

To run the group0 benchmarking, run the following command:

```
cd "$WORKING_DIRECTORY" && \
cd "$MODEL"-BENCH && \
export PATH="$WORKING_DIRECTORY"/"$MODEL"-BENCH/:$PATH && \
cd "$MODEL"-"${MAINTAINER^^}"-"${BASELINE_QTYPE^^}"-SPECIAL_SPLIT && \
benchmark_each_tensor.sh --chunks 250 --group-tensors '.*' --benchmark-groups-only --qtypes iq1_s_r4 iq1_s iq1_kt iq1_m iq2_xxs iq2_kt iq2_ks iq2_xs iq2_k iq2_s q2_K iq2_kl iq3_xxs iq3_kt iq3_ks iq3_k iq3_s q3_K iq4_kss iq4_kt iq4_ks iq4_xs iq4_k iq4_nl q4_0 q4_K q4_1 iq5_ks iq5_k q5_0 q5_K q5_1 q6_0 q6_K iq6_k q8_0 bf16
```

Note: Trim the list of qtypes to the qtypes your hardware can handle, otherwise you will end up with empty benchmark results for these qtypes.

Once benchmarking is completed you will see the following message appear:

```
✅ All qtypes processed. (single run mode: --infinite-loop=false)
```

Like before, we still need to validate that all benchmark logs are present and valid.

Their filenames are as follow:

```
bench_ppl(_kld)_result.group0.TARGET_QTYPE.CHUNKS.txt
```

Use the following command to count how many tensors/groups were benchmarked:

```
cd "$WORKING_DIRECTORY" && \
cd "$MODEL"-BENCH && \
export PATH="$WORKING_DIRECTORY"/"$MODEL"-BENCH/:$PATH && \
cd "$MODEL"-"${MAINTAINER^^}"-"${BASELINE_QTYPE^^}"-SPECIAL_SPLIT && \
ls bench_*result\.group0\.*.txt | wc -l
```

Use the following command to identify which benchmark files are corrupted (because don't contain results), which you will need to delete:

```
cd "$WORKING_DIRECTORY" && \
cd "$MODEL"-BENCH && \
cd "$MODEL"-"${MAINTAINER^^}"-"${BASELINE_QTYPE^^}"-SPECIAL_SPLIT && \
grep -L -E 'KL divergence statistics|Final estimate: PPL over' bench_*result\.group0\.*.txt
```

If you've identified that some benchmark files are missing or that you've had to delete some, you must run the same `benchmark_each_tensor.sh` with the exact same parameters to resume benchmarking the remaining tensors/groups.

## (optional) Collect the group0 degradation data results

_This step is required if you wish to collect the benchmarked group0 degradation data results and produce the `group0/kld_results.csv` and `group0/ppl_results.csv` files._

Collect the benchmark results using this command:

```
cd "$WORKING_DIRECTORY" && \
cd "$MODEL"-BENCH && \
export PATH="$WORKING_DIRECTORY"/"$MODEL"-BENCH/:$PATH && \
cd "$MODEL"-"${MAINTAINER^^}"-"${BASELINE_QTYPE^^}"-SPECIAL_SPLIT && \
collect_ppl_results.sh --chunks 250 --group-tensors '.*' --groups-only --no-percentage --auto-baseline "$BASELINE_QTYPE"
```

This script will output both a `kld_results.csv` and a `ppl_results.csv` file. Before claiming victory, we need to check if they contain all the expected results.

Use the following command to identify which quantization types failed to be benchmarked:

```
cd "$WORKING_DIRECTORY" && \
cd "$MODEL"-BENCH && \
cd "$MODEL"-"${MAINTAINER^^}"-"${BASELINE_QTYPE^^}"-SPECIAL_SPLIT && \
awk -F',' 'NR==1 {for (i=1;i<=NF;i++) h[i]=$i; next} {for (i=2;i<=NF;i++) if ($i=="" || $i=="404" || $i=="404%") print $1}' kld_results.csv
```

If this command doesn't return anything, then you have successfully produced `kld_results.csv` and `ppl_results.csv` calibration data files. If not, then you must run the same `benchmark_each_tensor.sh` with the exact same parameters to resume benchmarking these remaining quantization types.

### Interpolate the partial degradation data

_At this point you have likely performed a group0 degradation data benchmark against a subset of the full 64 quantization types. That means your results are partial and need to be interpolated to cover the complete range of quantization types._

Let's start by renaming the group0 degradation data results:

```
cd "$WORKING_DIRECTORY" && \
cd "$MODEL"-BENCH && \
cd "$MODEL"-"${MAINTAINER^^}"-"${BASELINE_QTYPE^^}"-SPECIAL_SPLIT && \
mv ppl_results.csv ppl_results_partial.csv && \
mv kld_results.csv kld_results_partial.csv
```

Now we must find the equation of the mean curve of our kld results:

```
cd "$WORKING_DIRECTORY" && \
cd "$MODEL"-BENCH && \
cd "$MODEL"-"${MAINTAINER^^}"-"${BASELINE_QTYPE^^}"-SPECIAL_SPLIT && \
python3 ../model_tensor_bpw_metric.py --results-csv kld_results_partial.csv --c-free --exclude-qtypes '.*_bn.*$' --transforms "identity" --ignore-outliers 50 --allow-impure-map --p-grid-max 15 --p-grid-steps 100 --d-from-lowest 1 --penalize-above 15 --resemblance-metric r2 --equation-only --plot
```

Copy-paste this equation obtained into the following command (complete the `...`):

```
cd "$WORKING_DIRECTORY" && \
cd "$MODEL"-BENCH && \
cd "$MODEL"-"${MAINTAINER^^}"-"${BASELINE_QTYPE^^}"-SPECIAL_SPLIT && \
mkdir -p group0 && \
python3 ../group0_enricher.py --output-csv group0/kld_results.csv --target-csv kld_results_partial.csv --target-mean-equation "y = ..."
```

Note: By default the `group0_enricher.py` script uses the degradation data and mean curve equation of `Qwen3-4B-Thinking-2507`. But you can pass as parameters other more suitable degradation data and mean curves using the `--reference-csv` and `--reference-mean-equation` parameters, ideally from the same model architecture as demonstrated [here](https://github.com/Thireus/GGUF-Tool-Suite/blob/main/models/Qwen3.5-35B-A3B/group0/notes.txt).

You will usually find how Thireus' degradation data has been enriched in the `notes.txt` file present in the `group0` folder of each model, for example [here](https://github.com/Thireus/GGUF-Tool-Suite/blob/main/models/Qwen3.5-35B-A3B/group0/notes.txt).

Congratulations, if you've made it thus far it means you have successfully produced a `group0/kld_results.csv` file which is to be used with the `quant_assign.py` script and is loaded via the `--quant-degradation-csv group0/kld_results.csv` parameter.