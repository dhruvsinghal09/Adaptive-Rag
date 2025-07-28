from pydantic import BaseModel, Field


class VerificationResult(BaseModel):
    faithful: bool = Field(description="True if answer is supported by the context.")
    explanation: str = Field(description="Brief reasoning.")