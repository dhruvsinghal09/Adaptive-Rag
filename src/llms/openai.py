import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")