from pypdf import PdfReader
from pathlib import Path
from config import DATA_DIR, OUTPUT_DIR

def extract_pdf_text(pdf_path: str) -> str:
    """Extract text from Supernote PDF exports."""
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    """Simple chunking."""
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        if chunk.strip():
            chunks.append(chunk)
    return chunks

if __name__ == "__main__":
    pdf_files = list(DATA_DIR.glob("*.pdf"))
    if pdf_files:
        text = extract_pdf_text(pdf_files[0])
        chunks = chunk_text(text)
        print(f"Extracted {len(chunks)} chunks from {pdf_files[0].name}")
