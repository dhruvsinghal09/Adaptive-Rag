from fastapi import APIRouter
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel

from src.memory.chat_history_mongo import ChatHistory
from src.memory.chathistory_in_memory import ChatInMemoryHistory
from src.rag.graph_builder import builder

router = APIRouter()

class QueryRequest(BaseModel):
    query: str
    session_id: str

@router.post("/rag/query")
async def rag_query(req: QueryRequest):
    chat_history=ChatHistory.get_session_history(req.session_id)
    #chat_history=ChatInMemoryHistory.get_session_history(req.token)
    await chat_history.add_message(HumanMessage(content=req.query))

    # Fetch full history
    messages = await chat_history.get_messages()
    result = builder.invoke({
        "messages": messages
    })
    output_text = result["messages"][-1].content

    # Save assistant message
    await chat_history.add_message(AIMessage(content=output_text))

    return {"result": result["messages"][-1]}
