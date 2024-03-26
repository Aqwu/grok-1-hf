# Grok-1 Convert to HF Transformers format

Unofficial dequantized weight of grok-1 in HF Transformers format.

In WSL2 Ubuntu 22.04, convert to HF format, Trading time and storage space for GPU.

Based on your hardware configuration, the required time could be very long, possibly exceeding 12 hours?

Refer to the [code1](https://gist.github.com/chu-tianxiang/ec310e15d56949fd0f351cb5f65ee7a1).
[code2](https://huggingface.co/keyfan/grok-1-hf).

Due to insufficient memory and GPU, everything needs to run on WSL2 virtual memory and under the CPU in Python. 

The entire process may require 2TB of memory, so you will need a 2TB virtual space. 
On the C drive, you need to have 2TB of free space.

The converted model file is about 590GB, so you will also need 600GB of space.

Open Windows Explorer, enter %UserProfile% in the address bar and press Enter. 
In that directory, create a file named .wslconfig and write the following content.
Set the memory to 50% of the existing memory.
```
[wsl2]
memory=64GB
swap=2048GB
localhostForwarding=true
```

Then, run
```
wsl --shutdown 
```

Restart WSL2 Ubuntu 22.04.

```
conda create -yn grok-1-hf
conda activate grok-1-hf

pip uninstall torch torchvision torchaudio
pip uninstall dm_haiku jax jaxlib numpy sentencepiece

pip install torch torchvision torchaudio

pip install dm_haiku==0.0.12
pip install jaxlib -U https://storage.googleapis.com/jax-releases/nocuda/jaxlib-0.4.25-cp310-cp310-manylinux2014_x86_64.whl
pip install jax==0.4.25

pip install numpy==1.26.4
pip install sentencepiece==0.2.0

git clone https://github.com/Aqwu/grok-1-hf
export jax_platform_name="cpu"

python convert_hf.py

```
# gguf
```

git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
mkdir build
cd build
cmake ..

make
cd bin

python convert-hf-to-gguf.py grok-1-hf/ --outfile grok-1-hf/ggml-model-f16.gguf --outtype f16
./quantize grok-1-hf/ggml-model-f16.gguf  grok-1-hf/ggml-model-q2_k_s.gguf q2_k_s

./main -m grok-1/ggml-model-f16.gguf -p "The answer to life the universe and everything is of course" -s 1 -n 128

# chat
./main -m grok-1-hf/ggml-model-q2_k_s.gguf -n 256 --repeat_penalty 1.0 --color -i -r "User:" -f ../../prompts/chat-with-bob.txt

```

# windows gguf cuda
```

git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
mkdir build
cd build

cmake .. -DLLAMA_CUDA=ON
cmake --build . --config Release
cd bin

python convert-hf-to-gguf.py grok-1-hf/ --outfile grok-1-hf/ggml-model-f16.gguf --outtype f16
quantize grok-1-hf/ggml-model-f16.gguf  grok-1-hf/ggml-model-q2_k_s.gguf q2_k_s

gguf-split  --split-max-tensors 256 grok-1-hf/ggml-model-q2_k_s.gguf grok-1-hf/grok-1-q2_k_s

main -m grok-1-hf/grok-1-q2_k_s-00001-of-00009.gguf -p "The answer to life the universe and everything is of course" -s 1 -n 128

# chat
main -m grok-1-hf/grok-1-q2_k_s-00001-of-00009.gguf -n 256 --repeat_penalty 1.0 --color -i -r "User:" -f ../../prompts/chat-with-bob.txt

```

# Grok-1

This repository contains JAX example code for loading and running the Grok-1 open-weights model.

Make sure to download the checkpoint and place the `ckpt-0` directory in `checkpoints` - see [Downloading the weights](#downloading-the-weights)

Then, run

```shell
pip install -r requirements.txt
python run.py
```

to test the code.

The script loads the checkpoint and samples from the model on a test input.

Due to the large size of the model (314B parameters), a machine with enough GPU memory is required to test the model with the example code.
The implementation of the MoE layer in this repository is not efficient. The implementation was chosen to avoid the need for custom kernels to validate the correctness of the model.

# Model Specifications

Grok-1 is currently designed with the following specifications:

- **Parameters:** 314B
- **Architecture:** Mixture of 8 Experts (MoE)
- **Experts Utilization:** 2 experts used per token
- **Layers:** 64
- **Attention Heads:** 48 for queries, 8 for keys/values
- **Embedding Size:** 6,144
- **Tokenization:** SentencePiece tokenizer with 131,072 tokens
- **Additional Features:**
  - Rotary embeddings (RoPE)
  - Supports activation sharding and 8-bit quantization
- **Maximum Sequence Length (context):** 8,192 tokens

# Downloading the weights

You can download the weights using a torrent client and this magnet link:

```
magnet:?xt=urn:btih:5f96d43576e3d386c9ba65b883210a393b68210e&tr=https%3A%2F%2Facademictorrents.com%2Fannounce.php&tr=udp%3A%2F%2Ftracker.coppersurfer.tk%3A6969&tr=udp%3A%2F%2Ftracker.opentrackr.org%3A1337%2Fannounce
```

or directly using [HuggingFace 🤗 Hub](https://huggingface.co/xai-org/grok-1):
```
git clone https://github.com/xai-org/grok-1.git && cd grok-1
pip install huggingface_hub[hf_transfer]
huggingface-cli download xai-org/grok-1 --repo-type model --include ckpt-0/* --local-dir checkpoints --local-dir-use-symlinks False
```


# License

The code and associated Grok-1 weights in this release are licensed under the
Apache 2.0 license. The license only applies to the source files in this
repository and the model weights of Grok-1.
