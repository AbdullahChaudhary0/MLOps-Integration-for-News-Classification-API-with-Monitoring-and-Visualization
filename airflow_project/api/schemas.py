from pydantic import BaseModel, Field
from typing import Dict, List

class SingleTextInput(BaseModel):
    text: str = Field(..., example="This is a sample headline")

class BatchTextInput(BaseModel):
    texts: List[str] = Field(..., example=["Headline 1", "Headline 2"])

class PredictionOutput(BaseModel):
    label: str = Field(..., example="DIVORCE")
    probabilities: Dict[str, float] = Field(
        ..., 
        description="Probabilities for top predicted classes",
        example={"DIVORCE": 0.6, "CRIME": 0.2, "POLITICS": 0.1, "SPORTS": 0.05, "BUSINESS": 0.05}
    )
