from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_pages(pages: list[dict], chunk_size: int = 500, chunk_overlap: int = 50) -> list[dict]:
    """
    Split page texts into smaller overlapping chunks.

    Args:
        pages: output from load_pdf() — list of {page, text, source}
        chunk_size: max characters per chunk
        chunk_overlap: overlap between consecutive chunks

    Returns:
        list of dicts:
        [
            {"chunk_id": 0, "page": 1, "source": "file.pdf", "text": "..."},
            ...
        ]
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = []
    chunk_id = 0

    for page in pages:
        splits = splitter.split_text(page["text"])
        for split in splits:
            chunks.append({
                "chunk_id": chunk_id,
                "page": page["page"],
                "source": page["source"],
                "text": split.strip()
            })
            chunk_id += 1

    print(f"Chunking complete: {len(pages)} pages → {len(chunks)} chunks")
    print(f"Settings: chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
    return chunks


if __name__ == "__main__":
    from rag.loader import load_pdf

    pages = load_pdf("data/attention_is_all_you_need.pdf")
    chunks = chunk_pages(pages)

    print(f"\nSample chunk:\n{'-'*40}")
    print(f"Chunk ID : {chunks[5]['chunk_id']}")
    print(f"Page     : {chunks[5]['page']}")
    print(f"Source   : {chunks[5]['source']}")
    print(f"Text     :\n{chunks[5]['text']}")
    print(f"\nShortest chunk: {min(len(c['text']) for c in chunks)} chars")
    print(f"Longest chunk:  {max(len(c['text']) for c in chunks)} chars")
    print(f"Avg chunk size: {int(sum(len(c['text']) for c in chunks) / len(chunks))} chars")