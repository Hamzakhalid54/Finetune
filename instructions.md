# PROJECT: NanoSwarm-1B (Agentic Logic Core)
# ARCHITECTURE: Multi-Agent System (MAS)
# TECH STACK: Llama-3.2-1B, Unsloth, LangGraph, DSPy, GGUF-Imatrix

## 1. Core Objectives
- Build a 3-agent swarm (Planner, Specialist, Auditor).
- Optimize for <2GB RAM footprint using 4-bit Quantization.
- Implement self-correcting logic loops via LangGraph & DSPy Assertions.
- Use Unsloth for high-efficiency fine-tuning on CoT (Chain of Thought) data.

## 2. Directory Structure
/nanoswarm
├── /core
│   ├── agents.py       # DSPy Signatures & Modules
│   ├── graph.py        # LangGraph State Machine
│   └── state.py        # Pydantic state definitions
├── /scripts
│   ├── train.py        # Unsloth Fine-tuning script
│   └── quantize.sh     # Llama.cpp quantization pipeline
├── requirements.txt    # [unsloth, dspy-ai, langgraph, llama-cpp-python, gradio]
└── app.py              # Main Entry & Gradio UI

## 3. Agentic Logic
1. **Planner**: Decomposes user query into logical steps.
2. **Specialist**: Executes each step (Fine-tuned for domain expertise).
3. **Auditor**: Validates Specialist output. If invalid, triggers a recursive loop back to Specialist.

## 4. Technical Constraints
- Model: Llama-3.2-1B (Base) -> LoRA fine-tune.
- Quantization: GGUF Q4_K_M with Importance Matrix.
- Logic: DSPy `dspy.Assert` for real-time validation.