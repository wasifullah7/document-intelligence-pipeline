import json
import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, Query, UploadFile, File, HTTPException
from pydantic import BaseModel

from docpipeline.pipeline import process_folder
from docpipeline.pipeline import search as pipeline_search
from docpipeline.ingestor import extract_text_from_file
from docpipeline.classifier import classify_document
from docpipeline.extractor import extract_fields

app = FastAPI(
    title="Document Intelligence API",
    description=(
        "Local AI pipeline: classify documents (Invoice / Resume / Utility Bill / Other), "
        "extract structured fields, and search by meaning — no cloud APIs required."
    ),
    version="1.0.0",
)


class ProcessRequest(BaseModel):
    folder: str = "./documents"


@app.get("/health", summary="Health check")
def health() -> dict:
    return {"status": "ok"}


@app.post("/process", summary="Ingest, classify, and extract a folder of documents")
def process(request: ProcessRequest) -> dict:
    output_file = os.getenv("OUTPUT_FILE", "output.json")
    chroma_path = os.getenv("CHROMA_PATH", None)
    results = process_folder(request.folder, output_file=output_file, chroma_path=chroma_path)
    return {"status": "success", "processed": len(results), "results": results}


@app.get("/results", summary="Return the full output.json contents")
def get_results() -> dict:
    output_file = os.getenv("OUTPUT_FILE", "output.json")
    path = Path(output_file)
    if not path.exists():
        return {"error": "No results yet. Call POST /process first."}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@app.get("/documents", summary="List processed documents with their class labels")
def list_documents() -> dict:
    output_file = os.getenv("OUTPUT_FILE", "output.json")
    path = Path(output_file)
    if not path.exists():
        return {"documents": []}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {
        "documents": [
            {"filename": k, "class": v.get("class")} for k, v in data.items()
        ]
    }


@app.post("/analyze", summary="Upload a single PDF or TXT and get instant results")
def analyze_file(file: UploadFile = File(..., description="PDF or TXT file to analyze")) -> dict:
    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".pdf", ".txt"}:
        raise HTTPException(status_code=400, detail="Only .pdf and .txt files are supported.")

    raw_bytes = file.file.read()

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(raw_bytes)
        tmp_path = tmp.name

    try:
        text = extract_text_from_file(tmp_path)
        # Gather page/image stats to produce a useful error if text is empty
        if not text.strip() and suffix == ".pdf":
            import fitz as _fitz
            _doc = _fitz.open(tmp_path)
            page_count = len(_doc)
            has_images = any(_doc[i].get_images() for i in range(min(page_count, 3)))
            _doc.close()
            if has_images:
                detail = (
                    f"'{file.filename}' appears to be a scanned image PDF ({page_count} page(s), "
                    "no selectable text). Tesseract OCR is not installed on this server. "
                    "Please use a text-based PDF, or install Tesseract and restart."
                )
            else:
                detail = (
                    f"'{file.filename}' has {page_count} page(s) but no extractable text. "
                    "The file may be empty, password-protected, or corrupt."
                )
            raise HTTPException(status_code=422, detail=detail)
    finally:
        os.unlink(tmp_path)

    doc_class = classify_document(text)
    fields = extract_fields(text, doc_class)
    return {"filename": file.filename, "class": doc_class, **fields}


@app.get("/search", summary="Semantic search across all indexed documents")
def search_documents(
    q: str = Query(..., description="Natural language search query"),
    top_k: int = Query(5, ge=1, le=20, description="Number of results to return"),
) -> dict:
    chroma_path = os.getenv("CHROMA_PATH", None)
    results = pipeline_search(q, top_k=top_k, chroma_path=chroma_path)
    return {"query": q, "results": results}
