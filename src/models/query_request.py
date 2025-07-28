from pydantic import BaseModel


class QueryRequest(BaseModel):
    query: str
    session_id: str