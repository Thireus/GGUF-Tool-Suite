#!/usr/bin/env python3
#***************************************************************#
#** This script is part of Thireus' GGUF Tool Suite.          **#
#** gguf_info.py is a useful tool that inspects tensors from  **#
#** GGUF files.                                               **#
#**                                                           **#
#** ********************************************************* **#
#** --------------- Updated: Jul-10-2025 -------------------- **#
#** ********************************************************* **#
#**                                                           **#
#** Author: Thireus <gguf@thireus.com>                        **#
#**                                                           **#
#** https://gguf.thireus.com/                                 **#
#** Thireus' GGUF Tool Suite - Quantize LLMs Like a Chef       **#
#**                                  ·     ·       ·~°          **#
#**     Λ,,Λ             ₚₚₗ  ·° ᵍᵍᵐˡ   · ɪᴋ_ʟʟᴀᴍᴀ.ᴄᴘᴘ°   ᴮᶠ¹⁶ ·  **#
#**    (:·ω·)       。··°      ·   ɢɢᴜғ   ·°·  ₕᵤ𝓰𝓰ᵢₙ𝓰𝒻ₐ𝒸ₑ   ·°   **#
#**    /    o―ヽニニフ))             · · ɪǫ3_xxs      ~·°        **#
#**    し―-J                                                   **#
#**                                                           **#
#** Copyright © 2025 - Thireus.                ₜₑₙₛₒᵣ ₛₑₐₛₒₙᵢₙ𝓰 **#
#***************************************************************#
#**PLEASE REFER TO THE README FILE FOR ADDITIONAL INFORMATION!**#
#***************************************************************#

# Requirements:
# If using llama.cpp: pip install gguf
# If using ik_llama.cpp: pip install "gguf @ git+https://github.com/ikawrakow/ik_llama.cpp.git@main#subdirectory=gguf-py" --force; pip install sentencepiece numpy==1.26.4

import sys
from pathlib import Path

# import the GGUF reader
from gguf.gguf_reader import GGUFReader

def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} path/to/model.gguf", file=sys.stderr)
        sys.exit(1)

    gguf_path = Path(sys.argv[1])
    reader   = GGUFReader(gguf_path)   # loads metadata & tensor index :contentReference[oaicite:0]{index=0}

    print(f"=== Tensors in {gguf_path.name} ===")
    # reader.tensors is a list of TensorEntry objects :contentReference[oaicite:1]{index=1}
    for tensor in reader.tensors:
        name = tensor.name
    
        # --- Shape: convert tensor.shape (array-like) into a Python tuple of ints
        try:
            shape = tuple(int(dim) for dim in tensor.shape)
        except Exception:
            shape = tuple(tensor.shape)
    
        # --- Dtype / quantization type: use the enum name
        # e.g. tensor.tensor_type.name might be "Q8_0", "F16", etc.
        dtype = tensor.tensor_type.name.lower()  # keep uppercase like "Q8_0"; use .lower() if you prefer "q8_0"
    
        # --- Number of elements:
        if hasattr(tensor, 'n_elements'):
            try:
                elements = int(tensor.n_elements)
            except Exception:
                # fallback to computing from shape
                elements = 1
                for dim in shape:
                    elements *= dim
        else:
            # compute product of dims
            elements = 1
            for dim in shape:
                elements *= dim
    
        # --- Number of bytes:
        if hasattr(tensor, 'n_bytes'):
            try:
                byte_count = int(tensor.n_bytes)
            except Exception:
                # fallback to data buffer size
                try:
                    byte_count = tensor.data.nbytes
                except Exception:
                    byte_count = None
        else:
            # fallback: if tensor.data is a NumPy array or memmap:
            try:
                byte_count = tensor.data.nbytes
            except Exception:
                byte_count = None
    
        # Format byte_count if None
        byte_str = str(byte_count) if byte_count is not None else "unknown"
    
        print(f"{name}\tshape={shape}\tdtype={dtype}\telements={elements}\tbytes={byte_str}")

if __name__ == "__main__":
    main()
