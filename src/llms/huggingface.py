import torch
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline

pipe = pipeline(
    task="text-generation",
    model="HuggingFaceTB/SmolLM3-3B",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

llm = HuggingFacePipeline(pipeline=pipe)
