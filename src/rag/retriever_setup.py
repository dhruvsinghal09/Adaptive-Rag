"""
Retriever setup and vector store configuration.
"""

from langchain_core.documents import Document
from langchain_core.tools import create_retriever_tool
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

from src.core.config import settings
from src.rag.scope import get_collection_name

embeddings = OpenAIEmbeddings()

def _get_qdrant_client() -> QdrantClient:
    """Create Qdrant client using configured URL/API key."""
    return QdrantClient(
        url=settings.QDRANT_URL,
        api_key=settings.QDRANT_API_KEY,
        timeout=30.0
    )


def _ensure_collection(collection_name: str) -> None:
    """Create collection if absent using embedding dimensions."""
    client = _get_qdrant_client()
    existing = {c.name for c in client.get_collections().collections}
    if collection_name in existing:
        return

    vector_dim = len(embeddings.embed_query("dimension_probe"))
    client.create_collection(
        collection_name=collection_name,
        vectors_config=qdrant_models.VectorParams(
            size=vector_dim,
            distance=qdrant_models.Distance.COSINE
        )
    )


def retriever_chain(
    chunks: list[Document],
    tenant_id: str | None = None,
    session_id: str | None = None
):
    """
    Initialize and store documents in Qdrant vector database.

    Args:
        chunks: List of document chunks to store.
        tenant_id: Logical tenant for multi-tenant indexing.
        session_id: Logical session/user scope for collection partition.

    Returns:
        Boolean indicating success of the operation.
    """
    try:
        collection_name = get_collection_name(tenant_id, session_id)
        _ensure_collection(collection_name)
        QdrantVectorStore.from_documents(
            documents=chunks,
            embedding=embeddings,
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
            collection_name=collection_name
        )
        print(f"Qdrant collection '{collection_name}' updated")
        print(f"Vectorstore contains {len(chunks)} document chunks")
        return True
    except Exception as e:
        print(f"Error storing documents in Qdrant: {e}")
        return False


def get_vectorstore(tenant_id: str | None = None, session_id: str | None = None):
    """Get Qdrant vector store for a tenant/session collection."""
    collection_name = get_collection_name(tenant_id, session_id)
    _ensure_collection(collection_name)
    return QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        url=settings.QDRANT_URL,
        api_key=settings.QDRANT_API_KEY,
        collection_name=collection_name
    )


def retrieve_documents(
    query: str,
    tenant_id: str | None = None,
    session_id: str | None = None,
    k: int = 4
) -> list[Document]:
    """Similarity-search documents for citation extraction and context."""
    vectorstore = get_vectorstore(tenant_id=tenant_id, session_id=session_id)
    return vectorstore.similarity_search(query=query, k=k)


def get_retriever(tenant_id: str | None = None, session_id: str | None = None):
    """
    Get a retriever tool connected to the Qdrant vector store.

    Returns:
        A LangChain retriever tool configured for the vector store.

    Raises:
        Exception: If vector store initialization fails.
    """
    try:
        collection_name = get_collection_name(tenant_id, session_id)
        retriever = get_vectorstore(tenant_id, session_id).as_retriever(search_kwargs={"k": 4})

        retriever_tool = create_retriever_tool(
            retriever,
            "retriever_customer_uploaded_documents",
            "Use this tool only for questions that should be answered from uploaded documents. "
            f"Current retrieval scope collection: {collection_name}."
        )

        return retriever_tool

    except Exception as e:
        print(f"Error initializing retriever: {e}")
        raise Exception(e)
