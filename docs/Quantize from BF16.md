# Quantize from BF16

_In this section I will explain how to quantize a [BF16 model](https://github.com/Thireus/GGUF-Tool-Suite/blob/main/docs/Convert_model_to_BF16.md) to produce 'pure' (or as pure as possible) quants for this GGUF-Tool-Suite: with one tensor per gguf, and first gguf only containing model's metadata - useful when the model needs to be updated as only the relevant files will be altered. Any script in the GGUF Tool Suite assumes the repositories of sharded GGUF are in such format and with specific filename. This section is not about quantizing a model from a mixture of quants or from a recipe file!_

## Requirements

Hardware:

```
CPU: YES
GPU: NO
```

_I like to use Hetzner servers, you can find suitable and cheap options [here](https://www.hetzner.com/sb/#search=5950X&drives_size_from=3500&drives_size_to=22000&cpuType=AMD). I recommend using RAID0 and Debian 13._

Make sure you edit and paste these variables in your terminal:

```
WORKING_DIRECTORY='/AI' # Full path please!
MODEL='Qwen3.5-0.8B'
MAINTAINER='THIREUS' # Or use your name!
IMATRIX='imatrix_ubergarm.dat' # Full path! Tutorial to obtain them will be covered separately.
QTYPE="q2_K" # The target quantization type
BF16_URL='https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/main/Qwen3.5-0.8B-BF16.gguf?download=true' # Optional, for Option 3
```

## Know your quants

[ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) offers a much wider and modern range of quantization types than [llama.cpp](https://github.com/ggml-org/llama.cpp), this not only helps with CPU inference speed but also enables a more granular range quant mix combination which helps produce quantized models of higher quality. For theis reason, I can only encourage users to choose [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) whenever possible.

Quant types supported by [llama.cpp](https://github.com/ggml-org/llama.cpp) for this tool suite:

```
iq1_s iq1_m iq2_xxs iq2_xs iq2_s q2_K iq3_xxs iq3_s q3_K iq4_xs iq4_nl q4_0 q4_K q4_1 q5_0 q5_K q5_1 q6_K q8_0 bf16
```

Quant types supported by [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) for this tool suite:

```
iq1_s_r4 iq1_s iq1_bn iq1_kt iq1_m iq1_m_r4 iq2_bn iq2_bn_r4 iq2_xxs iq2_xxs_r4 iq2_kt iq2_ks iq2_xs iq2_xs_r4 iq2_k iq2_k_r4 iq2_s q2_K q2_k_r4 iq2_kl iq3_xxs iq3_xxs_r4 iq3_kt iq3_ks iq3_k iq3_k_r4 iq3_s iq3_s_r4 q3_K q3_k_r4 iq4_kss iq4_kt iq4_ks iq4_ks_r4 iq4_xs iq4_xs_r8 iq4_k iq4_k_r4 iq4_nl iq4_nl_r4 q4_0 q4_0_r8 q4_K q4_k_r4 q4_1 iq5_ks iq5_ks_r4 iq5_k iq5_k_r4 q5_0 q5_0_r4 q5_K q5_k_r4 q5_1 q6_0 q6_0_r4 q6_K q6_k_r4 iq6_k q8_KV q8_k_r8 q8_0 q8_0_r8 bf16
```

A more detailed and exhaustive table is available [here](https://github.com/Thireus/GGUF-Tool-Suite/discussions/53).

Important: You will notice that `q` quants end with uppercase letters such as `_K` or `_KV`. This is very important to get this right! Quant types are often case-sensitive!

## Prepare the environment

We first start by obtaining or creating the BF16 version of our model. I like to use git lfs for this purpose but you can also use [HuggingFace cli tool](https://huggingface.co/docs/huggingface_hub/en/guides/cli) instead. Make sure you have sufficient disk space.


```
apt-get install python3-dev python3-pip python3-venv python3-wheel python3-setuptools git cmake build-essential git-lfs pipx ccache gpg screen lbzip2 curl lbzip2 # Run as root
```

Create the working directory (where all files will be downloaded and produced):

```
mkdir -p "$WORKING_DIRECTORY"
```

You have 3 options to get a BF16 version that works with the GGUF-Tool-Suite. Choose only one of them!

### Option 1:

To create the BF16 version of the model, please follow the steps [here](https://github.com/Thireus/GGUF-Tool-Suite/blob/main/docs/Convert_model_to_BF16.md).

### Option 2:

To obtain the GGUF-Tool-Suite ready BF16 version of your model, use `quant_downloader.sh` as instructed below.

Obtain the GGUF-Tool-Suite and prepare it for the chosen `$MODEL`:

```
cd "$WORKING_DIRECTORY"
GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://github.com/Thireus/GGUF-Tool-Suite/
cd GGUF-Tool-Suite && git pull # Update it
rm -f download.conf && cp -rf models/"$MODEL"/download.conf .
```

Download the BF16 version of a supported model:

```
cd "$WORKING_DIRECTORY"
export PATH="$WORKING_DIRECTORY"/GGUF-Tool-Suite/:$PATH
mkdir "$MODEL"-"${MAINTAINER^^}"-BF16-SPECIAL_SPLIT && \
cd "$MODEL"-"${MAINTAINER^^}"-BF16-SPECIAL_SPLIT && \
echo '.*=bf16' > bf16.recipe && \
quant_downloader.sh bf16.recipe --qtype BF16
```

The models Thireus provides are on [HuggingFace](https://huggingface.co/Thireus/collections) and mirrored on [gguf.thireus.com](https://gguf.thireus.com).

### Option 3:

Obtain a BF16 model repo from HuggingFace:

```
cd "$WORKING_DIRECTORY"
mkdir -p huggingface && cd huggingface && \
curl -L "$BF16_URL" -o "$MODEL"-BF16.gguf
```

Compile llama.cpp's llama-gguf-split:

```
cd "$WORKING_DIRECTORY"
git clone --depth 1 https://github.com/ggml-org/llama.cpp --recursive
cd llama.cpp && git pull # Update it
cmake -B build # -DGGML_AVX=ON -DGGML_AVX2=ON
cmake --build build --config Release -j$(nproc) --target llama-gguf-split
```

Split the BF16 model:

```
cd "$WORKING_DIRECTORY"
export PATH="$WORKING_DIRECTORY"/llama.cpp/build/bin/:$PATH && \
mkdir "$MODEL"-"${MAINTAINER^^}"-BF16-SPECIAL_SPLIT && \
llama-gguf-split --split --no-tensor-first-split --split-max-tensors 1 "$MODEL"-BF16.gguf /"$WORKING_DIRECTORY"/"$MODEL"-"${MAINTAINER^^}"-BF16-SPECIAL_SPLIT/model_name && \
cd /"$WORKING_DIRECTORY"/"$MODEL"-"${MAINTAINER^^}"-BF16-SPECIAL_SPLIT && \
for f in $(ls); do mv -f $f $(echo $f | sed "s/model_name/$MODEL-"${MAINTAINER^^}"-BF16-SPECIAL_TENSOR/g"); done
```

Note: You can also have a look at the `monitor_and_split.sh` which automates splitting models.

You will also need to produce the tensors.map file, enrich it with the imatrix hash and produce the GPG signatures. These steps are detailed [here](https://github.com/Thireus/GGUF-Tool-Suite/blob/main/docs/Convert_model_to_BF16.md).

## Quantize from BF16

Now that we have a BF16 GGUF sharded version of our model with tensors.map we can proceed to quantizing it.

Obtain the GGUF-Tool-Suite:

```
cd "$WORKING_DIRECTORY"
GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://github.com/Thireus/GGUF-Tool-Suite/
cd GGUF-Tool-Suite && git pull # Update it
```

Compile Thireus' special ik_llama.cpp llama-quantize version:

```
cd "$WORKING_DIRECTORY"
git clone --depth 1 https://github.com/Thireus/ik_llama.cpp --recursive ik_llama.cpp_indiv && \
cd ik_llama.cpp_indiv && git checkout th/quantize_individual_tensors && \
git pull # Update if required
cmake -B build -DGGML_MAX_CONTEXTS=2048 # -DGGML_AVX=ON -DGGML_AVX2=ON && \
cmake --build build --config Release -j$(nproc) --target llama-quantize
```

Note: Alternatively you can obtain builds which are available under the "th-quantize_individual_tensors" tag [here](https://github.com/Thireus/ik_llama.cpp/tags) - look for the "th-quantize_individual_tensors" tag, not "main"!


Quantize the model to the chosen `$QTYPE`:

```
cd "$WORKING_DIRECTORY"
export PATH=~/AI/ik_llama.cpp_indiv/build/bin/:$PATH
mkdir ${MODEL}-${MAINTAINER}-${QTYPE^^}-SPECIAL_SPLIT && \
quantize_model.sh --imatrix "$IMATRIX" --model "$MODEL" --qtype "$QTYPE" --destination-dir "${MODEL}-${MAINTAINER}-${QTYPE^^}-SPECIAL_SPLIT"
```

Note: Some tensors cannot always be quantized to the target `$QTYPE`. `quantize_model.sh` as well as `llama-quantize` employ special rules to fallback to the nearest and safe quantization type.

Tip: If you kill `quantize_model.sh` and run it again for the same qtype, it will safely resume quantizing from the last quantized tensor.

You will also need to produce the tensors.map file, enrich it with the imatrix hash and produce the GPG signatures. These steps are detailed [here](https://github.com/Thireus/GGUF-Tool-Suite/blob/main/docs/Convert_model_to_BF16.md).