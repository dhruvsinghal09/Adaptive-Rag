"""
Helpers for tenant/session scoped collection naming.
"""

import re


def safe_collection_part(value: str | None, default: str) -> str:
    """Normalize values for valid Qdrant collection-name parts."""
    normalized = (value or default).lower()
    return re.sub(r"[^a-zA-Z0-9_-]", "_", normalized)


def get_collection_name(tenant_id: str | None = None, session_id: str | None = None) -> str:
    """Build tenant/session scoped collection name."""
    tenant = safe_collection_part(tenant_id, "default")
    session = safe_collection_part(session_id, "global")
    return f"adaptive_rag__{tenant}__{session}"
