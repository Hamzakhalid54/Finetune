#!/bin/bash
# Nanoswarm GGUF Quantization Pipeline

# Stop script on first error
set -e 

echo "🧹 Step 0: Forcing the removal of broken torchvision to fix HuggingFace imports..."
pip uninstall -y torchvision || true

echo "🔨 Step 1: Fusing MLX DoRA adapters into a dense Hugging Face model..."
# MLX-LM provides a seamless command to fuse adapter weights back into a base model
python3 -m mlx_lm.fuse --model "unsloth/Llama-3.2-1B-Instruct" --adapter-path "nanoswarm_lora_mlx" --save-path "nanoswarm_fused_hf"

echo "📥 Step 2: Cloning the official llama.cpp library for GGUF conversion..."
if [ ! -d "llama.cpp" ]; then
    git clone https://github.com/ggerganov/llama.cpp.git
fi

echo "⚙️ Step 3: Converting fused model into an F16 GGUF..."
# Intentionally removed the volatile llama.cpp requirements installer to protect your environment dependencies
python3 llama.cpp/convert_hf_to_gguf.py nanoswarm_fused_hf --outfile nanoswarm_f16.gguf --outtype f16

echo "🗜️ Step 4: Compiling llama.cpp Quantize tool natively (via CMake)..."
# Automatically install CMake into your Python virtual environment if missing
pip install cmake
cd llama.cpp
cmake -B build
cmake --build build --config Release -t llama-quantize
cd ..

echo "📉 Step 5: Quantizing down to Q4_K_M (4-bit)..."
./llama.cpp/build/bin/llama-quantize nanoswarm_f16.gguf nanoswarm_q4_k_m.gguf Q4_K_M

echo "🧹 Step 6: Cleaning up temporary raw F16 file..."
rm nanoswarm_f16.gguf

echo "✅ Quantization complete! The highly optimized model 'nanoswarm_q4_k_m.gguf' is ready!"
