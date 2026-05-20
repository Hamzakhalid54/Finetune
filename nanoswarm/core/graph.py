from langgraph.graph import StateGraph, END, START
from .state import AgentState
from .agents import Planner, Specialist, Auditor

# Initialize the DSPy agents
planner_agent = Planner()
specialist_agent = Specialist()
auditor_agent = Auditor()

def planner_node(state: AgentState):
    """Node A: Planner"""
    # Assuming initial query is in history or will be passed implicitly
    user_query = state.history[0] if state.history else "Execute task"
    
    result = planner_agent(user_query=user_query)
    
    # Store the plan steps
    steps = [step.strip() for step in result.plan.split('\n') if step.strip()]
    return {"current_plan": steps}

def specialist_node(state: AgentState):
    """Node B: Specialist"""
    step_idx = getattr(state, "current_step_index", 0)
    step_instruction = state.current_plan[step_idx] if state.current_plan and step_idx < len(state.current_plan) else "Final verification"
    
    # 3. Pass Memory fully to Specialist exactly as requested mathematically!
    history_str = "\n".join(state.history)
    result = specialist_agent(step_instruction=step_instruction, history=history_str)
    
    # Stage the raw result in memory (we'll properly commit it in the Auditor if it passes)
    new_history = list(state.history) + [f"UNVERIFIED_TEMP_OUTPUT: {result.result}"]
    return {"history": new_history}

def auditor_node(state: AgentState):
    """Node C: Auditor"""
    step_idx = getattr(state, "current_step_index", 0)
    step_instruction = state.current_plan[step_idx] if state.current_plan and step_idx < len(state.current_plan) else "Final verification"
    
    # Pull out that temporary staged result
    staged_memory = state.history[-1]
    specialist_output = staged_memory.replace("UNVERIFIED_TEMP_OUTPUT: ", "")
    
    # 4. Auditor Checks
    result = auditor_agent(step_instruction=step_instruction, specialist_output=specialist_output)
    is_valid = str(result.is_valid).strip().lower() == 'true'
    
    updates = {"is_valid": is_valid}
    new_history = list(state.history)[:-1] # Pop the staged result out
    
    if is_valid:
        # 5. UPDATE MEMORY: The critical part defined in your pseudo code!
        new_history.append(f"Step: {step_instruction}\nResult: {specialist_output}\n")
        updates["history"] = new_history
        updates["current_step_index"] = step_idx + 1 # Move to the next plan step!
    else:
        new_history.append(f"Error caught by Auditor on Step {step_idx + 1}: {result.feedback}")
        updates["history"] = new_history
        
    return updates

def should_continue(state: AgentState):
    """Decides if the LangGraph loop should self-correct, loop to next step, or finish."""
    if not state.is_valid:
        return "continue" # Fail: self-correct the exact same step right now!
        
    step_idx = getattr(state, "current_step_index", 0)
    if state.current_plan and step_idx < len(state.current_plan):
        return "continue" # Success: Iterate to the next step instruction
    else:
        return "end" # Fully Exhausted & Passed!

# Initialize state graph
workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("planner", planner_node)
workflow.add_node("specialist", specialist_node)
workflow.add_node("auditor", auditor_node)

# Connect Edges
workflow.add_edge(START, "planner")
workflow.add_edge("planner", "specialist")
workflow.add_edge("specialist", "auditor")

# Conditional Edge for Self-Correction
workflow.add_conditional_edges(
    "auditor",
    should_continue,
    {
        "continue": "specialist",
        "end": END
    }
)

app = workflow.compile()
