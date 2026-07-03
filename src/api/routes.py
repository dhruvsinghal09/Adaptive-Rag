"""
API routes for Retrieval-Augmented Generation (RAG) operations.
"""

import logging

from fastapi import APIRouter, File, Header, HTTPException, UploadFile
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel

from src.memory.chat_history_mongo import ChatHistory
from src.models.query_request import QueryRequest
from src.rag.document_upload import documents
from src.rag.graph_builder import builder

logger = logging.getLogger(__name__)

router = APIRouter()


class QueryResponse(BaseModel):
    """Response model for query endpoint."""

    result: dict


class UploadResponse(BaseModel):
    """Response model for upload endpoint."""

    status: bool


@router.post(
    "/rag/query",
    response_model=QueryResponse,
    summary="Process a RAG query",
)
async def rag_query(req: QueryRequest) -> QueryResponse:
    """
    Process a Retrieval-Augmented Generation query.

    Retrieves chat history, invokes the LangGraph workflow,
    stores the assistant response, and returns the generated answer.

    Args:
        req: Query request containing query text and session ID.

    Returns:
        QueryResponse containing the generated response.

    Raises:
        HTTPException: If query processing fails.
    """
    logger.info("Received query for session '%s'.", req.session_id)

    try:
        chat_history = ChatHistory.get_session_history(req.session_id)

        await chat_history.add_message(
            HumanMessage(content=req.query)
        )

        messages = await chat_history.get_messages()

        result = builder.invoke(
            {
                "messages": messages,
            }
        )

        output_message = result["messages"][-1]

        await chat_history.add_message(
            AIMessage(content=output_message.content)
        )

        logger.info(
            "Successfully processed query for session '%s'.",
            req.session_id,
        )

        return QueryResponse(
            result={
                "type": "ai",
                "content": output_message.content,
            }
        )

    except Exception as exc:
        logger.exception("Failed to process query.")

        raise HTTPException(
            status_code=500,
            detail="Unable to process the query at this time.",
        ) from exc


@router.post(
    "/rag/documents/upload",
    response_model=UploadResponse,
    summary="Upload a document",
)
async def upload_file(
    file: UploadFile = File(...),
    description: str = Header(..., alias="X-Description"),
) -> UploadResponse:
    """
    Upload and index a document.

    Args:
        file: Uploaded PDF or TXT document.
        description: Description supplied through the
            X-Description header.

    Returns:
        UploadResponse indicating upload status.

    Raises:
        HTTPException:
            If document upload fails.
    """
    logger.info(
        "Received upload request for '%s'.",
        file.filename,
    )

    try:
        upload_status = documents(
            description=description,
            file=file,
        )

        logger.info(
            "Document '%s' uploaded successfully.",
            file.filename,
        )

        return UploadResponse(status=upload_status)

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception("Unexpected upload error.")

        raise HTTPException(
            status_code=500,
            detail="Unable to upload document.",
        ) from exc