from docpipeline.retriever import index_documents, search

SAMPLE_DOCS = {
    "invoice_jan.txt": "Invoice for ACME Corp. Total amount due: $500. Payment due January 2024.",
    "resume_john.txt": "John Doe. Software Engineer. 5 years of Python experience.",
    "utility_feb.txt": "Electric bill. Account ACC-001. Usage: 300 kWh. Amount due February 2024.",
}


def test_index_and_search_returns_results(tmp_path):
    chroma = str(tmp_path / "chroma")
    index_documents(SAMPLE_DOCS, chroma_path=chroma)
    results = search("payments due in January", top_k=3, chroma_path=chroma)
    assert len(results) > 0
    assert all("filename" in r for r in results)
    assert all("score" in r for r in results)
    assert all("excerpt" in r for r in results)


def test_search_returns_relevant_result(tmp_path):
    chroma = str(tmp_path / "chroma")
    index_documents(SAMPLE_DOCS, chroma_path=chroma)
    results = search("Python software engineer resume", top_k=3, chroma_path=chroma)
    filenames = [r["filename"] for r in results]
    assert "resume_john.txt" in filenames


def test_search_top_k_limits_results(tmp_path):
    chroma = str(tmp_path / "chroma")
    index_documents(SAMPLE_DOCS, chroma_path=chroma)
    results = search("document", top_k=2, chroma_path=chroma)
    assert len(results) <= 2


def test_index_documents_upsert_is_idempotent(tmp_path):
    chroma = str(tmp_path / "chroma")
    index_documents(SAMPLE_DOCS, chroma_path=chroma)
    index_documents(SAMPLE_DOCS, chroma_path=chroma)  # second call must not error
    results = search("invoice", top_k=3, chroma_path=chroma)
    assert len(results) > 0
