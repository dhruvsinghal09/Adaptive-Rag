from fastapi import APIRouter
from pydantic import BaseModel

from src.memory.chat_history import ChatHistory
from src.rag.graph_builder import builder

router = APIRouter()

class QueryRequest(BaseModel):
    query: str
    token: str

@router.post("/rag/query")
async def rag_query(req: QueryRequest):
    chat_history=ChatHistory.get_session_history(req.token)
    chat_history.add_user_message(req.query)
    result = builder.invoke({
        "messages": chat_history.messages
    })
    output_text = result["messages"][-1].content if isinstance(result, dict) else str(result)

    chat_history.add_ai_message(output_text)

    return {"result": result["messages"][-1]}
