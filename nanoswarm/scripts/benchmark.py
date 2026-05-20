# scripts/benchmark.py
import time
import json
import psutil
import os
import argparse

# Set up argument parsing to choose which stage we are testing
parser = argparse.ArgumentParser()
parser.add_argument("--stage", type=str, choices=["raw", "trained", "gguf"], required=True, help="Testing 'raw', 'trained' (MLX Adapters), or 'gguf' model")
args = parser.parse_args()

# The logic test questions
test_prompts = [
    "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
    "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",
    "Analyze if a 2-story building with 500kg load capacity per floor can support a 1000kg water tank on the roof."
]

print(f"--- Starting Benchmark for Stage: {args.stage.upper()} ---")

# Measure baseline memory
process = psutil.Process(os.getpid())
start_mem = process.memory_info().rss / (1024 * 1024)

start_time = time.time()
total_tokens = 0

# Load the appropriate model
if args.stage == "raw":
    from transformers import pipeline
    print("Loading Raw Llama-3.2-1B...")
    pipe = pipeline("text-generation", model="unsloth/Llama-3.2-1B-Instruct", device_map="auto")
    
    for prompt in test_prompts:
        result = pipe(prompt, max_new_tokens=100)[0]['generated_text']
        total_tokens += 100 # Approx for calculation
        print(f"\n[Prompt]: {prompt}\n[Answer]: {result[len(prompt):].strip()}")

elif args.stage == "trained":
    import mlx_lm
    print("Loading Native MLX Model with your fresh DoRA Adapters...")
    
    # mlx_lm natively fuses the adapters to the base model on the fly during load!
    model, tk = mlx_lm.load("unsloth/Llama-3.2-1B-Instruct", adapter_path="nanoswarm_lora_mlx")
    
    for prompt in test_prompts:
        # Prompt it exactly how it was trained
        formatted_prompt = f"Below is an instruction describing a task. Write a detailed response showing your chain of thought (reasoning steps), followed by the final answer.\n\n### Instruction:\n{prompt}\n\n### Response (Chain of Thought):\n"
        
        # Generation
        response = mlx_lm.generate(model, tk, prompt=formatted_prompt, max_tokens=150, verbose=False)
        total_tokens += 150 # Est. for metric math
        print(f"\n[Prompt]: {prompt}\n[Answer]: {response.strip()}")

elif args.stage == "gguf":
    from llama_cpp import Llama
    print("Loading Quantized NanoSwarm GGUF...")
    llm = Llama(model_path="nanoswarm_q4_k_m.gguf", n_ctx=2048, verbose=False)
    
    for prompt in test_prompts:
        result = llm(f"Q: {prompt} A: ", max_tokens=100)
        answer = result['choices'][0]['text'].strip()
        total_tokens += result['usage']['completion_tokens']
        print(f"\n[Prompt]: {prompt}\n[Answer]: {answer}")

end_time = time.time()
end_mem = process.memory_info().rss / (1024 * 1024)

# Calculate Metrics
time_taken = end_time - start_time
tps = total_tokens / time_taken if time_taken > 0 else 0
ram_used = end_mem - start_mem

metrics = {
    "stage": args.stage,
    "time_seconds": round(time_taken, 2),
    "tokens_per_second": round(tps, 2),
    "ram_used_mb": round(ram_used, 2)
}

# Save to file
with open("../metrics_report.jsonl", "a") as f:
    f.write(json.dumps(metrics) + "\n")

print(f"\n--- Metrics Saved! TPS: {metrics['tokens_per_second']}, RAM: {metrics['ram_used_mb']}MB ---")