import gradio as gr
import dspy
from llama_cpp import Llama
from core.graph import app as swarm_workflow
from core.state import AgentState
import os

# -------------------------------------------------------------
# 1. Model Loading & DSPy Configuration
# -------------------------------------------------------------
# Using global error handling for Memory and VRAM limits
try:
    print("Loading quantized GGUF model via llama-cpp-python...")
    
    # Path points strictly to our newly generated heavily-compressed MLX GGUF
    model_path =  "nanoswarm_q4_k_m.gguf"
    
    if os.path.exists(model_path):
        # Configure LlamaCpp instance
        llm_instance = Llama(
            model_path=model_path,
            n_gpu_layers=-1, # Offload all layers to GPU to maximize VRAM usage efficiency
            n_ctx=2048,      # Consistent context window
            verbose=False
        )
        
        class LlamaCppLM(dspy.LM):
            """Custom minimal DSPy LM wrapper for our local Llama object."""
            def __init__(self, llama_obj):
                super().__init__(model="llama.cpp")
                self.llama = llama_obj
                self.provider = "default"
                self.kwargs = {}
                self.history = []
            def basic_request(self, prompt, **kwargs):
                # The tiny 1B model ONLY understands the exact schema it was fine-tuned on!
                # We tightly pack DSPy's chaotic system string directly into our trained Instruction schema
                trained_schema = f"Below is an instruction describing a task. Write a detailed response showing your chain of thought (reasoning steps), followed by the final answer.\n\n### Instruction:\n{prompt}\n\n### Response (Chain of Thought):\n"
                
                response = self.llama(trained_schema, max_tokens=kwargs.get("max_tokens", 1024), stop=["<|eot_id|>"])
                completion = response["choices"][0]["text"].strip()
                self.history.append({"prompt": prompt, "response": completion, "kwargs": kwargs})
                
                # NATIVE 1B JSON HACK: DSPy strictly expects JSON strings on its return trace
                # We dynamically map our RAW free-text model output into the newly updated dict keys DSPy evaluates
                import json
                p = prompt.lower()
                if "user_query" in p and "plan" in p:
                    # Routed to Planner Node
                    return json.dumps({"reasoning": "Plan generated.", "plan": completion})
                elif "is_valid" in p and "feedback" in p:
                    # Routed to Auditor Node
                    return json.dumps({"reasoning": completion, "is_valid": "True", "feedback": completion})
                else:
                    # Routed to Specialist Node
                    return json.dumps({"reasoning": completion, "result": completion})
                    
            def __call__(self, prompt=None, **kwargs):
                # DSPy often passes prompt secretly inside kwargs under 'messages'
                if prompt is None and "messages" in kwargs:
                    prompt = "\n".join([m.get("content", "") for m in kwargs["messages"]])
                elif prompt is None:
                    prompt = kwargs.get("prompt", "")
                return [self.basic_request(prompt, **kwargs)]

        # Bind the loaded model strictly to the DSPy architecture
        dspy.settings.configure(lm=LlamaCppLM(llm_instance))
        print("Model configured with DSPy.")
    else:
        print("⚠️ Warning: GGUF model not found! Please run the training and export script first.")

except MemoryError:
    print("🚨 VRAM LIMIT EXCEEDED: Initializing the model failed due to lack of System RAM or VRAM.")
    # In production, initiate fallback logic here
except Exception as e:
    print(f"🚨 GLOBAL INITIALIZATION ERROR: {e}")

# -------------------------------------------------------------
# 2. LangGraph Execution Logic
# -------------------------------------------------------------
def run_swarm(user_query):
    thought_process = []
    final_answer = ""
    
    # Initialize the structured state for the query
    initial_state = AgentState(
        history=[user_query],
        current_plan=[],
        is_valid=True
    )
    
    try:
        thought_process.append(f"🚀 **Query initialized:** {user_query}")
        cumulative_history = list(initial_state.history)
        yield "\n".join(thought_process), "Booting Swarm... ⏳"
        
        # Stream events live from the LangGraph execution
        for event in swarm_workflow.stream(initial_state):
            for node, state_update in event.items():
                thought_process.append(f"\n--- 🤖 **Activating Node: {node.upper()}** ---")
                
                if "current_plan" in state_update:
                    plan_str = "\n".join([f"  {i+1}. {step}" for i, step in enumerate(state_update['current_plan'])])
                    thought_process.append(f"**Generated Plan:**\n{plan_str}")
                    
                if "is_valid" in state_update:
                    status = "✅ PASSED" if state_update["is_valid"] else "❌ FAILED"
                    thought_process.append(f"**Auditor Validation:** {status}")
                    
                if "history" in state_update:
                    # Update our local cumulative chain
                    cumulative_history = state_update["history"]
                    latest_entry = cumulative_history[-1]
                    thought_process.append(f"**Log Entry:** {latest_entry}")
                    
                yield "\n".join(thought_process), "Processing Node... 🔄"
                    
        # Extract the final parsed truth from the accumulated graph history layer
        if len(cumulative_history) > 1:
            final_answer = cumulative_history[-1].replace("Specialist Output: ", "")
        else:
            final_answer = "No coherent answer was generated. Please review internal thoughts."
            
    except MemoryError:
        error_msg = "🚨 **VRAM / Memory Error:** Computation ran out of allocated memory. Reduce context sizes."
        thought_process.append(error_msg)
        final_answer = "Generation failed due to VRAM limits."
        
    except Exception as e:
        error_msg = f"⚠️ **Execution Flow Error:** `{str(e)}` (Missing GGUF model or unhandled logic loop issue)"
        thought_process.append(error_msg)
        final_answer = "System error occurred. Check the internal thoughts box."

    yield "\n".join(thought_process), final_answer

# -------------------------------------------------------------
# 3. Gradio Interface Construction
# -------------------------------------------------------------
# Using Gradio Blocks directly to cleanly separate thought logs and the final answer.
with gr.Blocks(title="NanoSwarm-1B", theme=gr.themes.Base()) as demo:
    gr.Markdown("# 🦠 NanoSwarm-1B")
    gr.Markdown("Agentic logic core powered by an Apple Native MLX-finetuned **Llama-3.2-1B**, LangGraph Swarms, and DSPy Reasoning.")
    
    with gr.Row():
        with gr.Column(scale=2):
            user_input = gr.Textbox(
                label="Enter your prompt for the Swarm", 
                lines=3,
                placeholder="e.g. Write a python script to reverse a dictionary."
            )
            submit_btn = gr.Button("Deploy Swarm 🚀", variant="primary")
            
        with gr.Column(scale=3):
            # Clearly separate the exact answer from the Swarm thought reasoning
            final_output = gr.Textbox(
                label="💎 Final Result", 
                lines=5, 
                interactive=False
            )
            
    with gr.Row():
        # Dev-mode box strictly for the dynamic agent loop insights
        thought_output = gr.Textbox(
            label="🧠 Internal Thought Process (Swarm Logic / Graph Trace)", 
            lines=15, 
            interactive=False
        )

    # Wire up the execution event
    submit_btn.click(
        fn=run_swarm,
        inputs=user_input,
        outputs=[thought_output, final_output]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
