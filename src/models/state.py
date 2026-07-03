"""
State model for the LangGraph-based RAG workflow.
"""

from typing import Annotated, Optional, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages


class State(TypedDict):
    """
    Shared graph state.

    Attributes:
        messages:
            Complete conversation history.

        binary_score:
            Relevance grading result.

        route:
            Selected execution route.

        latest_query:
            Most recent user query.
    """

    messages: Annotated[list[AnyMessage], add_messages]

    binary_score: Optional[str]

    route: Optional[str]

    latest_query: Optional[str]