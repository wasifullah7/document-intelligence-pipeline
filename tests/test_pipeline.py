import json
import os
import fitz
from docpipeline.pipeline import process_folder, search


def make_temp_pdf(folder: str, filename: str, text: str) -> str:
    path = os.path.join(folder, filename)
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 100), text, fontsize=11)
    doc.save(path)
    doc.close()
    return path


def test_process_folder_produces_output_json(tmp_path):
    docs_dir = str(tmp_path / "docs")
    os.makedirs(docs_dir)
    make_temp_pdf(
        docs_dir, "invoice1.pdf",
        "Invoice Number: INV-001\nBill To: ACME Corp\nTotal Amount: $500.00\nAmount Due: $500.00",
    )

    output_path = str(tmp_path / "output.json")
    chroma = str(tmp_path / "chroma")
    results = process_folder(docs_dir, output_file=output_path, chroma_path=chroma)

    assert "invoice1.pdf" in results
    assert results["invoice1.pdf"]["class"] == "Invoice"
    assert os.path.exists(output_path)

    with open(output_path) as f:
        saved = json.load(f)
    assert "invoice1.pdf" in saved


def test_process_folder_classifies_resume(tmp_path):
    docs_dir = str(tmp_path / "docs")
    os.makedirs(docs_dir)
    make_temp_pdf(
        docs_dir, "resume1.pdf",
        "Jane Doe\njane@example.com\nWork Experience\nSoftware Engineer 5 years of experience\nSkills: Python",
    )

    results = process_folder(
        docs_dir,
        output_file=str(tmp_path / "output.json"),
        chroma_path=str(tmp_path / "chroma"),
    )
    assert results["resume1.pdf"]["class"] == "Resume"


def test_search_after_process(tmp_path):
    docs_dir = str(tmp_path / "docs")
    os.makedirs(docs_dir)
    make_temp_pdf(
        docs_dir, "invoice1.pdf",
        "Invoice Number: INV-001\nBill To: ACME Corp\nTotal Amount: $500.00\nAmount Due: $500.00",
    )

    chroma = str(tmp_path / "chroma")
    process_folder(docs_dir, output_file=str(tmp_path / "output.json"), chroma_path=chroma)

    results = search("invoice payment", chroma_path=chroma)
    assert len(results) > 0
