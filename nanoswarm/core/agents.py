import dspy

class PlannerSignature(dspy.Signature):
    """### Instruction: Break the math problem into a Chain of Thought plan."""
    user_query = dspy.InputField()
    plan = dspy.OutputField(desc="Numbered logic steps")

class SpecialistSignature(dspy.Signature):
    """
    ### Instruction: Solve this step. Use the Memory to keep variables consistent.
    ### Format: 
    Reasoning: [Show work]
    Answer: [Final value]
    """
    step_instruction = dspy.InputField()
    history = dspy.InputField(desc="Memory of all previous steps and facts")
    result = dspy.OutputField()

class AuditorSignature(dspy.Signature):
    """### Instruction: If Reasoning and Answer contradict, set is_valid=False."""
    step_instruction = dspy.InputField()
    specialist_output = dspy.InputField()
    is_valid = dspy.OutputField()
    feedback = dspy.OutputField(desc="Explain the calculation error")

# DSPy Modules using ChainOfThought for deep reasoning
class Planner(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought(PlannerSignature)
        
    def forward(self, user_query):
        return self.prog(user_query=user_query)

class Specialist(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought(SpecialistSignature)
        
    def forward(self, step_instruction, history):
        return self.prog(step_instruction=step_instruction, history=history)

class Auditor(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought(AuditorSignature)
        
    def forward(self, step_instruction, specialist_output):
        return self.prog(step_instruction=step_instruction, specialist_output=specialist_output)
