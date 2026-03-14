"""
ReAct agent setup for document retrieval and question answering.
"""

from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate

from src.config.settings import Config
from src.llms.openai import llm
from src.rag.retriever_setup import get_retriever

config = Config()


def build_agent_executor(
    tenant_id: str | None = None,
    session_id: str | None = None
) -> AgentExecutor:
    """
    Build a fresh ReAct AgentExecutor with the latest retriever tool.

    This must be created per invocation so newly uploaded documents are
    immediately visible to retrieval.
    """
    tools = [get_retriever(tenant_id=tenant_id, session_id=session_id)]

    prompt = ChatPromptTemplate.from_messages([
        ("system", config.prompt("system_prompt")),
        ("human", "{input}"),
        ("ai", "{agent_scratchpad}")
    ])

    react_agent = create_react_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=react_agent,
        tools=tools,
        handle_parsing_errors=True,
        max_iterations=2,
        verbose=True,
        return_intermediate_steps=True
    )
