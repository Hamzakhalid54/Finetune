from pydantic import BaseModel, Field
from typing import List, Optional

class AgentState(BaseModel):
    history: List[str] = Field(default_factory=list, description="History of interactions and executed steps")
    current_plan: Optional[List[str]] = Field(default=None, description="The current plan broken down into executable steps")
    current_step_index: int = Field(default=0, description="Index of the currently executing step in the plan")
    is_valid: bool = Field(default=True, description="Indicates if the current state or last simulated output is valid")
