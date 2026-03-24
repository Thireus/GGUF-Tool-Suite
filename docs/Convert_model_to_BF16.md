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
WORKING_DIRECTORY='/AI' # Full path please!
MODEL='Qwen3.5-0.8B'
MODEL_URL='https://huggingface.co/Qwen/Qwen3.5-0.8B'
MAINTAINER='YOUR_NAME'
MAINTAINER_EMAIL='your@email.com'
IMATRIX='imatrix_ubergarm.dat' # Full path! Tutorial to obtain them will be covered separately.
```

## Prepare the environment

We first start by obtaining the original model .safetensors and config files. I like to use git lfs for this purpose but you can also use [HuggingFace cli tool](https://huggingface.co/docs/huggingface_hub/en/guides/cli) instead. Make sure you have sufficient disk space.

```
apt-get install python3-dev python3-pip python3-venv python3-wheel python3-setuptools git cmake build-essential git-lfs pipx ccache gpg screen zlib1g-dev libxml2-dev libssl-dev libgmp-dev libmpfr-dev # Run as root
```

Create the working directory (where all files will be downloaded and produced):

```
mkdir -p "$WORKING_DIRECTORY"
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

Build and install triton-cpu (might not be required) :

```
cd "$WORKING_DIRECTORY"
# Activate env
source venv/bin/activate && \
uv pip install ninja cmake wheel setuptools pybind11 && \
git clone --depth 1 https://github.com/triton-lang/triton-cpu --recursive && \
cd triton-cpu && \
sed -i '/CMAKE_CXX_FLAGS/s/-Werror //g' CMakeLists.txt && \
sed -i '/^if (dnnl_FOUND)/,/^endif()/ s/^/# /' third_party/cpu/CMakeLists.txt && \
MAX_JOBS=$(nproc) uv pip install -e python --no-build-isolation # Be patient, "Preparing Packages" downloads a lot of stuff before build begins...
```

Note: See [here](https://github.com/ikawrakow/ik_llama.cpp/issues/383#issuecomment-2865306085) for patch explanation.


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
     --outfile /"$WORKING_DIRECTORY"/"$MODEL"-"$MAINTAINER"-BF16-SPECIAL_SPLIT/model_name \
     --no-tensor-first-split --split-max-tensors 1 \
     /"$WORKING_DIRECTORY"/huggingface/"$MODEL" 
cd /"$WORKING_DIRECTORY"/"$MODEL"-"$MAINTAINER"-BF16-SPECIAL_SPLIT && \
for f in $(ls); do mv -f $f $(echo $f | sed "s/model_name/$MODEL-$MAINTAINER-BF16-SPECIAL_TENSOR/g"); done
```

_Note: Some models will require more steps. You'll need to dig into github/reddit/hf._

For example, DeepSeek and Kimi-K2 FP8 needs to be cast to BF16 safetensors before it can be converted to BF16 GGUF:

```
cd "$WORKING_DIRECTORY"
# Activate env
source venv/bin/activate
git clone --depth 1 https://github.com/deepseek-ai/DeepSeek-V3.git && \
cd DeepSeek-V3/inference && \
sed -i 's/device="cuda"/device="cpu"/g' fp8_cast_bf16.py && \
mv /"$WORKING_DIRECTORY"/huggingface/"$MODEL" /"$WORKING_DIRECTORY"/huggingface/"$MODEL"-FP8 && \
mkdir /"$WORKING_DIRECTORY"/huggingface/"$MODEL" && \
python fp8_cast_bf16.py \
      --input-fp8-hf-path /"$WORKING_DIRECTORY"/huggingface/"$MODEL"-FP8/ \
      --output-bf16-hf-path /"$WORKING_DIRECTORY"/huggingface/"$MODEL"/ 2>&1 | tee -a fp8_cast_bf16-Kimi-K2-Instruct.log
cp /"$WORKING_DIRECTORY"/huggingface/"$MODEL"-FP8/config.json /"$WORKING_DIRECTORY"/huggingface/"$MODEL"/
cp /"$WORKING_DIRECTORY"/huggingface/"$MODEL"-FP8/generation_config.json /"$WORKING_DIRECTORY"/huggingface/"$MODEL"/
cp /"$WORKING_DIRECTORY"/huggingface/"$MODEL"-FP8/tokenizer_config.json /"$WORKING_DIRECTORY"/huggingface/"$MODEL"/
cp /"$WORKING_DIRECTORY"/huggingface/"$MODEL"-FP8/*.py /"$WORKING_DIRECTORY"/huggingface/"$MODEL"/
cp /"$WORKING_DIRECTORY"/huggingface/"$MODEL"-FP8/*.model /"$WORKING_DIRECTORY"/huggingface/"$MODEL"/
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

## (optional) Enrich tensors.map with imatrix hash

Obtain the GGUF-Tool-Suite:

```
cd "$WORKING_DIRECTORY"
GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://github.com/Thireus/GGUF-Tool-Suite/
cd GGUF-Tool-Suite && git pull # Update it
```

Enrich tensors.map that are ready with imatrix hash:

```
cd "$WORKING_DIRECTORY"
export PATH="$WORKING_DIRECTORY"/GGUF-Tool-Suite/:$PATH && \
for q in $(ls -l */tensors.map | sed "s/.*-$MAINTAINER-//g" | cut -d'-' -f1); do cd "$WORKING_DIRECTORY" && sed -n '/:imatrix=/q1; $q0' ${MODEL}-${MAINTAINER}-${q^^}-SPECIAL_SPLIT/tensors.map && tail -n1 ${MODEL}-${MAINTAINER}-${q^^}-SPECIAL_SPLIT/tensors.map \
| sed -nE '/-([0-9]+)-of-\1\.gguf:/q0; q1' && imatrix_tensors.py --map-file ${MODEL}-${MAINTAINER}-${q^^}-SPECIAL_SPLIT/tensors.map --output-map-file tensors.map imatrix_ubergarm.dat && mv -f tensors.map ${MODEL}-${MAINTAINER}-${q^^}-SPECIAL_SPLIT/tensors.map; done
```

## (optional) Produce GPG signatures

Create a GPG signing key:

```
MAINTAINER_NAME="$(echo "$MAINTAINER" | awk '{for(i=1;i<=NF;i++){ $i=toupper(substr($i,1,1)) tolower(substr($i,2)) } print}')"
gpg --batch --pinentry-mode ask --quick-gen-key "$MAINTAINER_NAME <$MAINTAINER_EMAIL>" rsa4096 sign 0
```

_Export your public key and send it over in an [issue](https://github.com/Thireus/GGUF-Tool-Suite/issues) if you would like it to be added to GGUF-Tool-Suite's [trusted-keys.asc](https://github.com/Thireus/GGUF-Tool-Suite/blob/main/trusted-keys.asc). This will enable other users to use and verify your GGUF shards using the GGUF-Tool-Suite tools._

Obtain the GGUF-Tool-Suite:

```
cd "$WORKING_DIRECTORY"
GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://github.com/Thireus/GGUF-Tool-Suite/
cd GGUF-Tool-Suite && git pull # Update it
```

Produce GPG signatures:

```
cd "$WORKING_DIRECTORY"
cd GGUF-Tool-Suite && git pull # Update it
# Activate env
cd "$WORKING_DIRECTORY" && \
source venv/bin/activate && \
export PATH="$WORKING_DIRECTORY"/GGUF-Tool-Suite/helpers/:$PATH && \
cd "$WORKING_DIRECTORY" && \
for q in $(ls -l */tensors.map | sed "s/.*-$MAINTAINER-//g" | cut -d'-' -f1); do cd "$WORKING_DIRECTORY" && d="${MODEL}-THIREUS-${q^^}-SPECIAL_SPLIT" && ls "$d"/*.sig >/dev/null 2>&1 && echo "Skipping $d: .sig files found — run prepare_model.sh -p $d manually to replace signatures" || prepare_model.sh -p "$d"; done
```