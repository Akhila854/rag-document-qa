import faiss
import numpy as np
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
VECTOR_STORE_DIR = Path("vector_store")


def build_index(chunks: list[dict], index_name: str = "index") -> tuple:
    """
    Embed chunks and build a FAISS index.

    Args:
        chunks: output from chunk_pages()
        index_name: filename prefix for saved index files

    Returns:
        (faiss_index, chunks) tuple
    """
    print(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    texts = [c["text"] for c in chunks]

    print(f"Embedding {len(texts)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    # Normalise for cosine similarity
    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]
    print(f"Embedding dimension: {dimension}")

    index = faiss.IndexFlatIP(dimension)  # Inner product = cosine after normalisation
    index.add(embeddings)

    print(f"FAISS index built: {index.ntotal} vectors")

    # Save index and chunk metadata to disk
    VECTOR_STORE_DIR.mkdir(exist_ok=True)
    faiss.write_index(index, str(VECTOR_STORE_DIR / f"{index_name}.faiss"))
    with open(VECTOR_STORE_DIR / f"{index_name}.pkl", "wb") as f:
        pickle.dump(chunks, f)

    print(f"Saved to vector_store/{index_name}.faiss + .pkl")
    return index, chunks


def load_index(index_name: str = "index") -> tuple:
    """
    Load a saved FAISS index and chunk metadata from disk.

    Returns:
        (faiss_index, chunks) tuple
    """
    index_path = VECTOR_STORE_DIR / f"{index_name}.faiss"
    chunks_path = VECTOR_STORE_DIR / f"{index_name}.pkl"

    if not index_path.exists():
        raise FileNotFoundError(f"No saved index at {index_path}. Run build_index() first.")

    index = faiss.read_index(str(index_path))
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)

    print(f"Loaded index: {index.ntotal} vectors, {len(chunks)} chunks")
    return index, chunks


if __name__ == "__main__":
    from rag.loader import load_pdf
    from rag.chunker import chunk_pages

    pages = load_pdf("data/attention_is_all_you_need.pdf")
    chunks = chunk_pages(pages)
    index, chunks = build_index(chunks)

    print(f"\nIndex stats:")
    print(f"  Total vectors : {index.ntotal}")
    print(f"  Dimensions    : {index.d}")
    print(f"  Chunks stored : {len(chunks)}")