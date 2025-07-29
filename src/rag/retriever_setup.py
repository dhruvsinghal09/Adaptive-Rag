import os

from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import create_retriever_tool

load_dotenv()
os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY", "")
urls = [
    "https://langchain-ai.github.io/langgraph/concepts/why-langgraph/",
    "https://langchain-ai.github.io/langgraph/tutorials/workflows/",
    "https://langchain-ai.github.io/langgraph/how-tos/graph-api/#map-reduce-and-the-send-api"
]

loader = WebBaseLoader(urls)
docs = [loader.load() for url in urls]
docs_items = [item for sublist in docs for item in sublist]

chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150).split_documents(docs_items)

embeddings = OpenAIEmbeddings()
vectorStore = FAISS.from_documents(chunks, embeddings)
retriever = vectorStore.as_retriever()

retriever_tool = create_retriever_tool(
    retriever,
    "retriever_vectorstore_langgraph",
    (
        "Use this tool **only** to answer questions about LangGraph documentation."
        "Don't use this tool to answer anything else"
    )
)
