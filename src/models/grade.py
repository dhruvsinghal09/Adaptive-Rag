from pydantic import BaseModel, Field


class Grade(BaseModel):
    binary_score:str=Field(description="relevance score 'yes' or 'no'")