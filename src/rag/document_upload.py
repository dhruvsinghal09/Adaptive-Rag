"""
Document upload and processing module.
"""

import os
import tempfile
import uuid

from fastapi import UploadFile, File
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.rag.retriever_setup import retriever_chain
from src.tools.common_tools import enhance_description_with_llm


def documents(
    description: str,
    file: UploadFile = File(...),
    tenant_id: str | None = None,
    session_id: str | None = None
):
    """
    Process and upload a document for RAG.

    Validates file type, loads content, enhances description, chunks documents,
    and stores them in the vector database.

    Args:
        description: User-provided document description.
        file: The uploaded file (PDF or TXT).
        tenant_id: Logical tenant for document isolation.
        session_id: Logical session/user scope for document isolation.

    Returns:
        Boolean indicating success of the upload process.

    Raises:
        HTTPException: If file type is not supported or loading fails.
    """
    filename = file.filename
    print(filename)
    if not filename.endswith(".pdf") and not filename.endswith(".txt"):
        from fastapi import HTTPException
        raise HTTPException(
            status_code=400,
            detail="Only PDF and TXT files are supported"
        )

    file_bytes = file.file.read()

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=os.path.splitext(filename)[1]
    ) as tmp_file:
        tmp_file.write(file_bytes)
        tmp_path = tmp_file.name

    if filename.endswith(".pdf"):
        loader = PyPDFLoader(tmp_path)
    else:
        loader = TextLoader(tmp_path, encoding="utf-8")

    try:
        docs = loader.load()
    except Exception as e:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=500,
            detail=f"Error loading file: {e}"
        )
    finally:
        os.unlink(tmp_path)

    description_llm = enhance_description_with_llm(description)

    # Split documents into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    chunks = splitter.split_documents(docs)

    doc_id = str(uuid.uuid4())
    for idx, chunk in enumerate(chunks):
        chunk.metadata = chunk.metadata or {}
        chunk.metadata.update({
            "chunk_id": idx,
            "document_id": doc_id,
            "filename": filename,
            "document_description": description_llm,
            "tenant_id": tenant_id or "default",
            "session_id": session_id or "global",
        })

    return retriever_chain(
        chunks=chunks,
        tenant_id=tenant_id,
        session_id=session_id
    )




