import json
import os
import fitz
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path):
    os.environ["CHROMA_PATH"] = str(tmp_path / "chroma")
    os.environ["OUTPUT_FILE"] = str(tmp_path / "output.json")
    os.environ["DOCUMENTS_FOLDER"] = str(tmp_path / "docs")
    os.makedirs(str(tmp_path / "docs"), exist_ok=True)

    # Create a sample invoice PDF
    pdf_path = str(tmp_path / "docs" / "invoice1.pdf")
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text(
        (50, 100),
        "Invoice Number: INV-TEST-001\nBill To: Test Corp\nTotal Amount: $999.00\nAmount Due: $999.00",
        fontsize=11,
    )
    doc.save(pdf_path)
    doc.close()

    from docpipeline.api import app
    yield TestClient(app)

    # Cleanup env vars after each test
    for key in ("CHROMA_PATH", "OUTPUT_FILE", "DOCUMENTS_FOLDER"):
        os.environ.pop(key, None)


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_process_endpoint(client):
    docs_folder = os.environ.get("DOCUMENTS_FOLDER", "./documents")
    response = client.post("/process", json={"folder": docs_folder})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert body["processed"] >= 1
    assert "invoice1.pdf" in body["results"]


def test_results_endpoint_after_process(client):
    docs_folder = os.environ.get("DOCUMENTS_FOLDER", "./documents")
    client.post("/process", json={"folder": docs_folder})
    response = client.get("/results")
    assert response.status_code == 200
    data = response.json()
    assert "invoice1.pdf" in data


def test_documents_endpoint_after_process(client):
    docs_folder = os.environ.get("DOCUMENTS_FOLDER", "./documents")
    client.post("/process", json={"folder": docs_folder})
    response = client.get("/documents")
    assert response.status_code == 200
    docs = response.json()["documents"]
    assert len(docs) >= 1
    assert all("filename" in d and "class" in d for d in docs)


def test_search_endpoint_after_process(client):
    docs_folder = os.environ.get("DOCUMENTS_FOLDER", "./documents")
    client.post("/process", json={"folder": docs_folder})
    response = client.get("/search", params={"q": "invoice payment"})
    assert response.status_code == 200
    body = response.json()
    assert "results" in body
    assert len(body["results"]) >= 1


def test_results_before_process_returns_error(client):
    response = client.get("/results")
    assert response.status_code == 200
    assert "error" in response.json()
