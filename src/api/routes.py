"""
API routes for RAG operations.
"""

from fastapi import APIRouter, UploadFile, File, Header
from langchain_core.messages import HumanMessage, AIMessage

from src.memory.chat_history_mongo import ChatHistory
from src.models.query_request import QueryRequest
from src.rag.document_upload import documents
from src.rag.graph_builder import builder

router = APIRouter()


@router.post("/rag/query")
async def rag_query(
    req: QueryRequest,
    tenant_id: str = Header("default", alias="X-Tenant-Id")
):
    """
    Process a RAG query and return the result.

    Args:
        req: The query request containing query text and session_id.

    Returns:
        The generated response from the RAG pipeline.
    """
    #chat_history=ChatInMemoryHistory.get_session_history(req.token)
    chat_history = ChatHistory.get_session_history(req.session_id)
    await chat_history.add_message(HumanMessage(content=req.query))

    # Fetch full history
    messages = await chat_history.get_messages()
    result = builder.invoke({
        "messages": messages,
        "session_id": req.session_id,
        "tenant_id": tenant_id,
    })
    last_message = result["messages"][-1]
    output_text = last_message.content
    citations = getattr(last_message, "additional_kwargs", {}).get("citations", [])

    # Save assistant message
    await chat_history.add_message(AIMessage(content=output_text))

    return {
        "result": {
            "type": "ai",
            "content": output_text,
            "citations": citations
        }
    }


@router.post("/rag/documents/upload")
async def upload_file(
    file: UploadFile = File(...),
    description: str = Header(..., alias="X-Description"),
    tenant_id: str = Header("default", alias="X-Tenant-Id"),
    session_id: str = Header("global", alias="X-Session-Id")
):
    """
    Upload a document for RAG processing.

    Args:
        file: The file to upload (PDF or TXT).
        description: Document description provided via header.

    Returns:
        Upload status.
    """
    status_upload = documents(
        description=description,
        file=file,
        tenant_id=tenant_id,
        session_id=session_id
    )
    return {"status": status_upload}

