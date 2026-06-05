from fastapi import FastAPI
from api.routes import router

app = FastAPI(
    title="RAG Document Q&A API",
    description="Upload a PDF and ask questions about it",
    version="1.0.0"
)

app.include_router(router)

@app.get("/")
def root():
    return {"status": "ok", "message": "RAG Document Q&A API is running"}