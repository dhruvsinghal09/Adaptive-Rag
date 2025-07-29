import pathlib

from langchain_community.tools import TavilySearchResults
from langchain_core.messages import AIMessage
from langchain_core.prompts import PromptTemplate
from langgraph.constants import START, END
from langgraph.graph.state import CompiledStateGraph, StateGraph

from src.rag.reAct_agent import agent_executor
from src.rag.retriever_setup import retriever
from src.config.settings import Config
#from src.llms.openai import llm
from src.llms.groq import llm
from src.models.grade import Grade
from src.models.route_identifier import RouteIdentifier
from src.models.state import State
from src.tools.graph_tools import routing_tool, doc_tool, verify_answer

config = Config()


# Node implementations
def query_classifier(state: State):
    """This is the node to classify the query if it is related to index or not
    Args:
        state (State): the current state of the graph
    """
    question = state["messages"][-1].content
    top_docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in top_docs])
    print("context received is")
    print(context)
    llm_with_structured_output = llm.with_structured_output(RouteIdentifier)
    classify_prompt = PromptTemplate(
        template=config.prompt("classify_prompt"),
        input_variables=["question", "context"]
    )
    chain = classify_prompt | llm_with_structured_output
    result = chain.invoke({"question": question, "context": context})
    print("result received is in query classifier")
    print(result.route)
    return {"messages": state["messages"], "route": result.route, "latest_query": question}

def general_llm(state: State):
    """This nodes fetches general common knowledge result from the llm
    Args:
        state (State): the current state of the graph
    """
    result=llm.invoke(state["messages"])
    print("inside general llm")
    print(result)
    return {"messages":result}


def retriever_node(state: State):
    """This node will be used to retriever the result from the vectorStores
    Args:
        state (State): the current state of the graph
    """
    messages = state["latest_query"]
    result = agent_executor.invoke({"input": messages})

    # just to check the tool was called
    # Extract tool calls
    intermediate_steps = result.get("intermediate_steps", [])
    tool_calls = []
    if intermediate_steps:
        for action, tool_result in intermediate_steps:
            tool_calls.append({
                "tool": action.tool,
                "input": action.tool_input,
            })
    new_message = AIMessage(
        content=result["output"],
        additional_kwargs={"tool_calls": tool_calls},
    )
    return {
        "messages": [new_message]
    }


def grade(state: State):
    """This node will be used to grade the result from the vectorStores
    Args:
        state (State): the current state of the graph
    """
    grading_prompt = PromptTemplate(
        template=config.prompt("grading_prompt"),
        input_variables=["question", "context"]
    )
    context = state["messages"][-1].content
    question = state["latest_query"]

    llm_with_grade = llm.with_structured_output(Grade)

    chain_graded = grading_prompt | llm_with_grade
    result = chain_graded.invoke({"question": question, "context": context})

    print(result)
    return {"messages": state["messages"], "binary_score": result.binary_score}


def rewrite_query(state: State):
    """This node will rewrite the query to get the better results
    Args:
        state (State): State of the question
    """

    query = state["latest_query"]
    rewrite_prompt = PromptTemplate(
        template=config.prompt("rewrite_prompt"),
        input_variables=["query"]
    )
    chain = rewrite_prompt | llm
    result = chain.invoke({"query": query})
    print(result)
    return {
        "latest_query": result.content
    }


def generate(state: State):
    """This node will generate the answer for the user in the best and suitable way possible.
    Args:
        state (State): State of the question
    """
    context = state["messages"][-1].content

    generate_prompt = PromptTemplate(
        template=config.prompt("generate_prompt"),
        input_variables=["context"]
    )

    generate_chain = generate_prompt | llm
    result = generate_chain.invoke({"context": context})

    return {"messages": [{"role": "assistant", "content": result.content}]}


def web_search(state: State):
    """This node will search the web for the rewritten query"""

    # Initialize the Tavily tool
    search_tool = TavilySearchResults()

    # Search a query
    result = search_tool.invoke(state["latest_query"])

    contents = [item["content"] for item in result if "content" in item]
    print(contents)
    return {
        "messages": [{"role": "assistant", "content": "\n\n".join(contents)}]
    }


graph = StateGraph(State)

graph.add_node("query_analysis", query_classifier)
graph.add_node("retriever", retriever_node)
graph.add_node("grade", grade)
graph.add_node("generate", generate)
graph.add_node("rewrite", rewrite_query)
graph.add_node("web_search", web_search)
graph.add_node("general_llm", general_llm)

graph.add_edge(START, "query_analysis")
graph.add_edge("web_search", "generate")
graph.add_edge("retriever", "grade")
graph.add_edge("rewrite", "retriever")
graph.add_conditional_edges("query_analysis", routing_tool)
graph.add_conditional_edges("grade", doc_tool)
graph.add_edge("generate",END)
#graph.add_conditional_edges("generate", verify_answer)
graph.add_edge("general_llm", END)

builder=graph.compile()

#png_data = builder.get_graph().draw_mermaid_png()

# Save it to a file
#output_path = pathlib.Path("adaptive_RAG.png")
#output_path.write_bytes(png_data)
