# scripts/train.py
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
import argparse
import sys
import os
import json
import subprocess

# Set up argument parsing - Defaulting to MLX for your Mac speed
parser = argparse.ArgumentParser(description="NanoSwarm Training Script")
parser.add_argument("--backend", type=str, default="mlx", choices=["peft", "mlx"], 
                    help="Choose 'mlx' for native Apple Silicon speed or 'peft' for PyTorch")
args = parser.parse_args()

cot_prompt = """Below is an instruction describing a task. Write a detailed response showing your chain of thought (reasoning steps), followed by the final answer.

### Instruction:
{}

### Response (Chain of Thought):
{}"""

# ==============================================================================
# MLX IMPLEMENTATION (THE FAST WAY)
# ==============================================================================
if args.backend == "mlx":
    print("🍏 Step 1: Initializing Native Apple MLX Training Pipeline...")
    
    # MLX-LM is picky about folder structure, let's set it up
    os.makedirs("mlx_data", exist_ok=True)
    
    tokenizer = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-1B-Instruct")
    eos_token = tokenizer.eos_token if tokenizer.eos_token else "<|end_of_text|>"
    
    print("📚 Step 2: Re-formatting Data for MLX...")
    with open("data/train.jsonl", "r") as f_in, open("mlx_data/train.jsonl", "w") as f_out:
        for line in f_in:
            data = json.loads(line)
            # Combine instruction and output into the 'text' field MLX expects
            formatted_text = cot_prompt.format(data["instruction"], data["output"]) + eos_token
            f_out.write(json.dumps({"text": formatted_text}) + "\n")
            
    # Create a dummy validation set so MLX doesn't complain
    with open("mlx_data/valid.jsonl", "w") as f_out:
        f_out.write(json.dumps({"text": formatted_text}) + "\n")
        
    print("🚀 Step 3: Launching MLX-LM LoRA Training on GPU...")
    # This calls the Apple-optimized training engine directly
    mlx_cmd = [
        sys.executable, "-m", "mlx_lm", "lora",
        "--model", "unsloth/Llama-3.2-1B-Instruct",
        "--train",
        "--data", "mlx_data",
        "--batch-size", "1",
        "--num-layers", "16",
        "--fine-tune-type", "dora",  # <--- Upgraded to DoRA here!
        "--iters", "500", 
        "--learning-rate", "1e-5",
        "--adapter-path", "nanoswarm_lora_mlx" # Naming the output folder
    ]
    
    try:
        subprocess.run(mlx_cmd, check=True)
        print("✅ MLX Training Complete! Adapters saved in 'nanoswarm_lora_mlx'.")
    except subprocess.CalledProcessError as e:
        print(f"❌ MLX Training failed: {e}")
    sys.exit(0)

# ==============================================================================
# PYTORCH (PEFT) MPS IMPLEMENTATION (THE FALLBACK)
# ==============================================================================
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"🚀 Step 1: Loading Llama-3.2-1B on Mac {device}...")

model_id = "unsloth/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map=device,
    torch_dtype=torch.float16,
    use_cache=False
)
model.gradient_checkpointing_enable()

print("⚙️ Step 2: Configuring LoRA...")
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

print("📚 Step 3: Mapping Dataset...")
dataset = load_dataset("json", data_files="data/train.jsonl", split="train")

def format_func(batch):
    return {"text": [cot_prompt.format(i, o) + tokenizer.eos_token for i, o in zip(batch["instruction"], batch["output"])]}

dataset = dataset.map(format_func, batched=True)

print("🔥 Step 4: Starting SFT...")
# FIXED SFTConfig by passing dataset_text_field and max_seq_length correctly
sft_config = SFTConfig(
    dataset_text_field="text",
    max_seq_length=1024,
    output_dir="outputs",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    max_steps=100,
    learning_rate=2e-4,
    optim="adamw_torch",
    report_to="none",
    save_strategy="no"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=sft_config,
)

trainer.train()
trainer.model.save_pretrained("nanoswarm_lora")
print("✅ Training Complete!")