from langchain.agents import create_react_agent, AgentExecutor

from src.config.settings import Config
#from src.llms.openai import llm
from src.llms.groq import llm
from src.rag.retriever_setup import retriever_tool

config = Config()

tools = [retriever_tool]

# Create ReAct agent and executor
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", config.prompt("system_prompt")),
    ("human", "{input}"),
    ("ai", "{agent_scratchpad}")
])
react_agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=react_agent, tools=tools, handle_parsing_errors=True, max_iterations=2,
                               verbose=True,return_intermediate_steps=True)
