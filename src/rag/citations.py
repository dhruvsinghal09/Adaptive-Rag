"""
Helpers for extracting and rendering source citations.
"""


def extract_citations(docs) -> list[dict]:
    """
    Build a unique citation list from retrieved documents.
    """
    citations = []
    seen = set()
    for doc in docs:
        metadata = getattr(doc, "metadata", {}) or {}
        cite = {
            "document_id": metadata.get("document_id"),
            "chunk_id": metadata.get("chunk_id"),
            "filename": metadata.get("filename", metadata.get("source", "unknown")),
            "page": metadata.get("page"),
        }
        key = (cite["document_id"], cite["chunk_id"], cite["filename"], cite["page"])
        if key not in seen:
            seen.add(key)
            citations.append(cite)
    return citations


def render_citations(citations: list[dict]) -> str:
    """
    Render citations as a markdown-friendly source section.
    """
    if not citations:
        return ""

    lines = []
    for citation in citations:
        filename = citation.get("filename", "unknown")
        page = citation.get("page")
        chunk_id = citation.get("chunk_id")
        parts = [filename]
        if page is not None:
            parts.append(f"page {page}")
        if chunk_id is not None:
            parts.append(f"chunk {chunk_id}")
        lines.append("- " + ", ".join(parts))
    return "Sources:\n" + "\n".join(lines)
