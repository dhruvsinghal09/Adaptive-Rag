from typing import Literal

from langchain_core.prompts import PromptTemplate

from src.config.settings import Config
#from src.llms.openai import llm
from src.llms.groq import llm
from src.models.state import State
from src.models.verification_result import VerificationResult

config = Config()

def routing_tool(state:State) -> Literal["retriever","general_llm","web_search"]:
    """This tool will be used to route the graph to retriever node or web_search node
    Args:
        state (State): the current state of the graph
    """
    if state["route"]=="index":
        return "retriever"
    elif state["route"]=="general":
        return "general_llm"
    else:
        return "web_search"

def doc_tool(state:State) -> Literal["rewrite", "generate"]:
    """This is a tool which will be used to determine that if the query needs rewriting or not according to the binary_score"""
    score = state["binary_score"]
    print(f"[doc_tool] Routing based on score: {score}")
    if score == "yes":
        return "generate"
    else:
        return "rewrite"

def verify_answer(state:State) -> Literal["__end__","generate"]:
    """Check whether the final answer is faithful to the retrieved context."""
    if state["route"]=="general":
        return "__end__"
    else:
        question = state["latest_query"]
        context = state["messages"][-1].content
        final_asnwer = state["messages"][-1].content

        verify_prompt = PromptTemplate(
            template=config.prompt("verify_prompt"),
            input_variables=["question", "context", "final_answer"]
        )
        llm_with_verification = llm.with_structured_output(VerificationResult)

        verify_chain = verify_prompt | llm_with_verification

        result = verify_chain.invoke({"question": question, "context": context, "final_answer": final_asnwer})
        if result.faithful:
            return "__end__"
        else:
            print("generating again as faithful is false.")
            return "generate"