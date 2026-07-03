"""
Document upload and processing module.

This module handles document validation, temporary storage, loading,
description enhancement, document chunking, and ingestion into the
retrieval pipeline.
"""

import logging
import tempfile
from pathlib import Path

from fastapi import File, HTTPException, UploadFile
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.rag.retriever_setup import retriever_chain
from src.tools.common_tools import enhance_description_with_llm

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".txt"}
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
DESCRIPTION_FILE = Path("description.txt")


def _validate_file(filename: str, file_bytes: bytes) -> str:
    """
    Validate uploaded file.

    Args:
        filename: Name of the uploaded file.
        file_bytes: Raw file content.

    Returns:
        File extension.

    Raises:
        HTTPException: If file type is unsupported or file is empty.
    """
    extension = Path(filename).suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Only PDF and TXT files are supported.",
        )

    if not file_bytes:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty.",
        )

    return extension


def _save_temp_file(file_bytes: bytes, extension: str) -> str:
    """
    Save uploaded content to a temporary file.

    Args:
        file_bytes: Raw uploaded file content.
        extension: File extension.

    Returns:
        Path to the temporary file.
    """
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=extension,
    ) as temp_file:
        temp_file.write(file_bytes)
        return temp_file.name


def _load_documents(temp_path: str, extension: str):
    """
    Load documents using the appropriate LangChain loader.

    Args:
        temp_path: Path to temporary file.
        extension: Uploaded file extension.

    Returns:
        Loaded LangChain documents.

    Raises:
        HTTPException: If document loading fails.
    """
    try:
        if extension == ".pdf":
            loader = PyPDFLoader(temp_path)
        else:
            loader = TextLoader(temp_path, encoding="utf-8")

        return loader.load()

    except Exception as exc:
        logger.exception("Failed to load uploaded document.")
        raise HTTPException(
            status_code=500,
            detail=f"Error loading file: {exc}",
        ) from exc


def _save_enhanced_description(description: str) -> None:
    """
    Enhance and persist the document description.

    Args:
        description: User-provided document description.
    """
    enhanced_description = enhance_description_with_llm(description)

    DESCRIPTION_FILE.write_text(
        enhanced_description,
        encoding="utf-8",
    )

    logger.info("Enhanced document description saved successfully.")


def _split_documents(documents):
    """
    Split documents into chunks.

    Args:
        documents: Loaded LangChain documents.

    Returns:
        List of document chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    return splitter.split_documents(documents)


def documents(
    description: str,
    file: UploadFile = File(...),
):
    """
    Process and upload a document into the RAG pipeline.

    The workflow consists of:

    1. Validate uploaded file.
    2. Save file temporarily.
    3. Load document.
    4. Enhance and store document description.
    5. Split document into chunks.
    6. Store chunks in the vector database.

    Args:
        description: User-provided description of the document.
        file: Uploaded PDF or TXT document.

    Returns:
        Result returned by the retriever pipeline.

    Raises:
        HTTPException:
            * 400 if the uploaded file is invalid.
            * 500 if document processing fails.
    """
    filename = file.filename or ""

    logger.info("Processing uploaded file: %s", filename)

    file_bytes = file.file.read()

    extension = _validate_file(filename, file_bytes)

    temp_path = _save_temp_file(file_bytes, extension)

    try:
        loaded_documents = _load_documents(
            temp_path,
            extension,
        )
    finally:
        temp_file = Path(temp_path)

        if temp_file.exists():
            temp_file.unlink()

    _save_enhanced_description(description)

    chunks = _split_documents(loaded_documents)

    logger.info(
        "Document processed successfully. Generated %d chunks.",
        len(chunks),
    )

    return retriever_chain(chunks)