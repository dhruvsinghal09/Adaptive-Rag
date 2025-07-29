import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()
api_key=os.getenv("GROQ_API_KEY")
#llm=ChatGroq(model="llama-3.1-8b-instant",api_key=api_key)
llm=ChatGroq(model="gemma2-9b-it",api_key=api_key)