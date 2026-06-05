import os
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel

from rag.loader import load_pdf
from rag.chunker import chunk_pages
from rag.embedder import build_index, load_index

router = APIRouter()

# In-memory state — holds index + chunks for the current session
_state = {"index": None, "chunks": None}


class QuestionRequest(BaseModel):
    question: str


class AnswerResponse(BaseModel):
    question: str
    answer: str
    sources: list[dict]


@router.post("/upload", summary="Upload a PDF and build the vector index")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    # Save uploaded file to data/
    save_path = f"data/{file.filename}"
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Run RAG pipeline
    try:
        pages = load_pdf(save_path)
        chunks = chunk_pages(pages)
        index, chunks = build_index(chunks)
        _state["index"] = index
        _state["chunks"] = chunks
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "message": "PDF processed successfully",
        "filename": file.filename,
        "pages": len(pages),
        "chunks": len(chunks)
    }


@router.post("/ask", response_model=AnswerResponse, summary="Ask a question about the uploaded PDF")
async def ask_question(request: QuestionRequest):
    if _state["index"] is None:
        # Try loading saved index from disk
        try:
            index, chunks = load_index()
            _state["index"] = index
            _state["chunks"] = chunks
        except FileNotFoundError:
            raise HTTPException(
                status_code=400,
                detail="No document loaded. Please upload a PDF first via /upload"
            )

    from rag.retriever import ask
    try:
        result = ask(request.question, _state["index"], _state["chunks"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return result