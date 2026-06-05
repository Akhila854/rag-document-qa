import fitz  # PyMuPDF
from pathlib import Path


def load_pdf(pdf_path: str) -> list[dict]:
    """
    Extract text from a PDF file, page by page.

    Returns a list of dicts:
    [
        {"page": 1, "text": "...", "source": "filename.pdf"},
        ...
    ]
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"File is not a PDF: {pdf_path}")

    doc = fitz.open(pdf_path)
    pages = []
    total_pages = len(doc)

    for page_num in range(total_pages):
        page = doc[page_num]
        text = page.get_text().strip()

        if text:  # skip blank pages
            pages.append({
                "page": page_num + 1,
                "text": text,
                "source": path.name
            })

    doc.close()

    print(f"Loaded '{path.name}': {total_pages} pages, {len(pages)} non-blank pages extracted")
    return pages


if __name__ == "__main__":
    import sys
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "data/attention_is_all_you_need.pdf"
    pages = load_pdf(pdf_path)
    print(f"\nFirst page preview:\n{'-'*40}")
    print(pages[0]["text"][:500])
    print(f"\nTotal pages with text: {len(pages)}")