from fastapi import FastAPI
from src.api.routes import router

app = FastAPI(title="Adaptive RAG API")
app.include_router(router)

@app.get("/")
async def root():
    return {"message": "Adaptive RAG API is running"}
