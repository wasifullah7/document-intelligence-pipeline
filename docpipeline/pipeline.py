import json
import os

from docpipeline.ingestor import ingest_folder
from docpipeline.classifier import classify_document
from docpipeline.extractor import extract_fields
from docpipeline.retriever import index_documents
from docpipeline.retriever import search as _search

DEFAULT_OUTPUT_FILE = "output.json"


def process_folder(
    folder: str,
    output_file: str | None = None,
    chroma_path: str | None = None,
) -> dict:
    """
    Full pipeline: ingest → classify → extract → index → save output.json.
    Returns the results dict.
    """
    output_path = output_file or os.getenv("OUTPUT_FILE", DEFAULT_OUTPUT_FILE)

    # 1. Ingest all documents in the folder
    documents = ingest_folder(folder)

    # 2. Classify + extract structured fields for each document
    results: dict = {}
    for filename, text in documents.items():
        doc_class = classify_document(text)
        fields = extract_fields(text, doc_class)
        results[filename] = {"class": doc_class, **fields}

    # 3. Persist to output.json
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # 4. Index all documents for semantic search
    index_documents(documents, chroma_path=chroma_path)

    return results


def search(query: str, top_k: int = 5, chroma_path: str | None = None) -> list[dict]:
    """Semantic search — delegates to retriever."""
    return _search(query, top_k=top_k, chroma_path=chroma_path)
