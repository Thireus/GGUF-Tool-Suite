# Convert model to BF16

_In this section I will explain how to convert an original model made of .safetensors to a BF16 gguf model split into chunks: with one tensor per gguf, and first gguf only containing model's metadata - useful when the model needs to be updated as only the relevant files will be altered. Any script in the GGUF Tool Suite assumes the repositories of sharded GGUF are in such format and with specific filename._

## Requirements

Hardware:

```
CPU: YES
GPU: NO
```

_I like to use Hetzner servers, you can find suitable and cheap options [here](https://www.hetzner.com/sb/#search=5950X&drives_size_from=3500&drives_size_to=22000&cpuType=AMD). I recommend using RAID0 and Debian 13._

Make sure you edit and paste these variables in your terminal:

```
MODEL='Qwen3.5-0.8B'
MODEL_URL='https://huggingface.co/Qwen/Qwen3.5-0.8B'
WORKING_DIRECTORY='/AI' # FUll path please!
MAINTAINER='THIREUS'
```

## Prepare the environment

We first start by obtaining the original model .safetensors and config files. I like to use git lfs for this purpose but you can also use [HuggingFace cli tool](https://huggingface.co/docs/huggingface_hub/en/guides/cli) instead. Make sure you have sufficient disk space.

```
apt-get install screen python3-dev python3-pip python3-venv python3-wheel python3-setuptools git cmake build-essential git-lfs pipx ccache # Run as root
```

Obtain original model data:

```
cd "$WORKING_DIRECTORY"
mkdir -p huggingface && cd huggingface && \
git lfs clone --depth 1 "$MODEL_URL" "$MODEL" && cd "$MODEL" && rm -rf "$MODEL"/.git
```

Obtain llama.cpp:

```
cd "$WORKING_DIRECTORY"
git clone --depth 1 https://github.com/ggml-org/llama.cpp --recursive
cd llama.cpp && git pull # Update it
```

Create Python environment:

```
pipx install uv
cd "$WORKING_DIRECTORY"
uv venv ./venv --python 3.12 --python-preference=only-managed
```

Install Python requirements:

```
cd "$WORKING_DIRECTORY"
# Activate env
source venv/bin/activate && \
cd llama.cpp && \
uv pip install -r requirements/requirements-convert_hf_to_gguf.txt --prerelease=allow --index-strategy unsafe-best-match && \
uv pip install -U git+https://github.com/huggingface/transformers.git
```

## Convert to BF16 GGUF

Convert model to BF16 GGUF:

```
cd "$WORKING_DIRECTORY"
mkdir -p /"$WORKING_DIRECTORY"/"$MODEL"-"$MAINTAINER"-BF16-SPECIAL_SPLIT
# Activate env
source venv/bin/activate && \
cd llama.cpp && \
ulimit -n 99999 && \
python convert_hf_to_gguf.py \
     --outtype bf16 \
     --outfile /"$WORKING_DIRECTORY"/"$MODEL"-"$MAINTAINER"-BF16-SPECIAL_SPLIT/model_name-THIREUS-BF16-SPECIAL_TENSOR \
     --no-tensor-first-split --split-max-tensors 1 \
     /"$WORKING_DIRECTORY"/huggingface/"$MODEL" 
cd /"$WORKING_DIRECTORY"/"$MODEL"-"$MAINTAINER"-BF16-SPECIAL_SPLIT && \
for f in $(ls); do mv -f $f $(echo $f | sed "s/model_name/$MODEL/g"); done
```

## Produce tensors.map

Obtain GGUF-Tool-Suite:

```
cd "$WORKING_DIRECTORY"
GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://github.com/Thireus/GGUF-Tool-Suite/
cd GGUF-Tool-Suite && git pull # Update it
```

Create tensors.map file:

```
# Prefer running these in a screen; monitor_and_clean.sh auto creates tensors.map on new repos
cd "$WORKING_DIRECTORY"
cd GGUF-Tool-Suite && git pull # Update it
# Activate env
cd "$WORKING_DIRECTORY" && \
source venv/bin/activate && \
ulimit -n 99999 && \
export PATH="$WORKING_DIRECTORY"/GGUF-Tool-Suite/:$PATH && \
cd "$WORKING_DIRECTORY" && \
monitor_and_clean.sh .
```

## Produce GPG signatures

