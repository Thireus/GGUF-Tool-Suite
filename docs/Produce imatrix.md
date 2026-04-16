# Produce imatrix

_In this section I will explain how produce the imatrix file. I will assume that you have already followed the documentation to produce a [BF16 GGUF of your model](https://github.com/Thireus/GGUF-Tool-Suite/blob/main/docs/Convert%20model%20to%20BF16.md) or that, due to hardware constrains, you are limited to use another high quality quantization type [obtained from BF16](https://github.com/Thireus/GGUF-Tool-Suite/blob/main/docs/Quantize%20from%20BF16.md) such as Q8\_0 (please, ensure no imatrix file was used during quantization)._

## Requirements

**IMPORTANT: FOLLOW THIS GUIDE STEP-BY-STEP. DO NOT SKIP ANY NON-OPTIONAL PART. DO NOT TAKE SHORTCUTS. READ CAREFULLY WHAT IS WRITTEN AT ANY STEP.**

Hardware:

```
CPU: YES
GPU: NO (but suggested)
```

_I like to use Hetzner servers, you can find suitable and cheap options [here](https://www.hetzner.com/sb/#search=5950X&drives_size_from=3500&drives_size_to=22000&cpuType=AMD). I recommend using RAID0 and Debian 13._

Make sure you edit and paste these env variables in any terminal session you'll be using:

```
WORKING_DIRECTORY='/AI' # Full path please!
MODEL_GGUF="$WORKING_DIRECTORY/Qwen3.5-0.8B-THIREUS-BF16-SPECIAL_SPLIT/Qwen3.5-0.8B-THIREUS-BF16-SPECIAL_TENSOR-00001-of-00321.gguf" # Full path
IMATRIX='imatrix_ubergarm.dat' # Full path where the imatrix file will be written to! Tutorial to obtain them will be covered separately.
```

## Prepare the environment

Create the working directory where the imatrix file will be produced:

```
mkdir -p "$WORKING_DIRECTORY"
```

Either obtain `llama-imatrix` from https://github.com/Thireus/ik_llama.cpp/releases, or compile it from `ik_llama.cpp`, for example:

```
cd "$WORKING_DIRECTORY"
git clone --depth 1 https://github.com/Thireus/ik_llama.cpp --recursive ik_llama.cpp && \
cd ik_llama.cpp && \
git pull # Update if required
cmake -B build -DGGML_MAX_CONTEXTS=2048 # -DGGML_AVX=ON -DGGML_AVX2=ON && \
cmake --build build --config Release -j$(nproc) --target llama-imatrix
```

Important: If you are using GPU(s), ensure your compilation options take advantage of it. See: https://github.com/ikawrakow/ik_llama.cpp/blob/main/docs/build.md.

Note: Thireus' fork is used for convenience, but you can use the official https://github.com/ikawrakow/ik_llama.cpp instead.

Then, obtain the `imatrix-calibration-corpus-v02.txt` file which is used for KLD and PPL calibration data benchmarking (see why [here](https://github.com/Thireus/GGUF-Tool-Suite/discussions/23#discussioncomment-14764941)):

```
cd "$WORKING_DIRECTORY" && \
curl -L 'https://gist.githubusercontent.com/ubergarm/edfeb3ff9c6ec8b49e88cdf627b0711a/raw/ba5b01b6960a86874592f5913e283746ff734483/ubergarm-imatrix-calibration-corpus-v02.txt' -o imatrix-calibration-corpus-v02.txt
```

## Product the imatrix file

To produce an imatrix of reasonable quality we will assume you are either using `BF16` or another near-original-quality quantization type of your model that fits your hardware.

```
cd "$WORKING_DIRECTORY" && \
export PATH="$WORKING_DIRECTORY"/ik_llama.cpp/build/bin/:$PATH && \
llama-imatrix --verbosity 1 -m ${MODEL_GGUF} -f ubergarm-imatrix-calibration-corpus-v02.txt -o ${IMATRIX} -ngl 99 --layer-similarity --ctx-size 512 --threads $(nproc)
```

Important: Adjust the `-ngl` option and related `ik_llama.cpp` parameters to take advantage of your GPU(s) if you are using any.

Note: @ubergarm wrote a more detailed guide on https://github.com/ikawrakow/ik_llama.cpp/discussions/434, check it out!

The `llama-imatrix` process can take between a few hours up to a couple of days. Once completed, the imatrix file should be available at the following location: `${IMATRIX}`.