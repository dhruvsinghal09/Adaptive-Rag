"""
Request models for Retrieval-Augmented Generation APIs.
"""

from pydantic import BaseModel, Field, field_validator


class QueryRequest(BaseModel):
    """
    Request model for RAG queries.

    Attributes:
        query: User question.
        session_id: Conversation session identifier.
    """

    query: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="User query.",
        examples=["What is machine learning?"],
    )

    session_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Unique session identifier.",
        examples=["user_session_123"],
    )

    @field_validator("query", "session_id")
    @classmethod
    def strip_whitespace(cls, value: str) -> str:
        """
        Remove leading and trailing whitespace.

        Args:
            value: Input string.

        Returns:
            Cleaned string.

        Raises:
            ValueError:
                If the resulting string is empty.
        """
        cleaned = value.strip()

        if not cleaned:
            raise ValueError(
                "Field cannot contain only whitespace."
            )

        return cleaned