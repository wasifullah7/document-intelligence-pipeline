import fitz
import tempfile
import os
import shutil
from docpipeline.ingestor import extract_text_from_file, ingest_folder


def make_temp_pdf(text: str) -> str:
    """Helper: create a single-page PDF with given text, return its path."""
    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    tmp_path = tmp.name
    tmp.close()  # Release Windows file lock before PyMuPDF writes
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 100), text, fontsize=12)
    doc.save(tmp_path)
    doc.close()
    return tmp_path


def test_extract_text_from_pdf():
    path = make_temp_pdf("Invoice Number: INV-001\nTotal: $500.00")
    try:
        result = extract_text_from_file(path)
        assert "INV-001" in result
        assert "500.00" in result
    finally:
        os.unlink(path)


def test_extract_text_from_txt():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write("Name: John Doe\nEmail: john@example.com")
        path = f.name
    try:
        result = extract_text_from_file(path)
        assert "John Doe" in result
        assert "john@example.com" in result
    finally:
        os.unlink(path)


def test_extract_text_unsupported_extension_returns_empty():
    with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as f:
        path = f.name
    try:
        result = extract_text_from_file(path)
        assert result == ""
    finally:
        os.unlink(path)


def test_ingest_folder_returns_dict():
    with tempfile.TemporaryDirectory() as tmpdir:
        pdf_path = make_temp_pdf("Invoice Number: INV-999")
        shutil.copy(pdf_path, os.path.join(tmpdir, "invoice.pdf"))
        os.unlink(pdf_path)

        txt_path = os.path.join(tmpdir, "resume.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("John Doe\nSoftware Engineer\n5 years of experience")

        result = ingest_folder(tmpdir)
        assert "invoice.pdf" in result
        assert "resume.txt" in result
        assert "INV-999" in result["invoice.pdf"]
        assert "John Doe" in result["resume.txt"]


def test_ingest_folder_skips_non_documents():
    with tempfile.TemporaryDirectory() as tmpdir:
        jpg_path = os.path.join(tmpdir, "image.jpg")
        with open(jpg_path, "w") as f:
            f.write("fake image")
        result = ingest_folder(tmpdir)
        assert result == {}


def test_ingest_folder_raises_on_missing_folder():
    import pytest
    with pytest.raises(ValueError, match="Not a valid folder"):
        ingest_folder("/nonexistent/path/xyz")
