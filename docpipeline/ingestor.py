import fitz
from functools import lru_cache
from pathlib import Path

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".text"}


@lru_cache(maxsize=1)
def _get_easyocr_reader():
    import easyocr
    return easyocr.Reader(["en"], gpu=False, verbose=False)


def _ocr_page_image(page) -> str:
    """Render page to image and run easyocr on it — no Tesseract needed."""
    try:
        pix = page.get_pixmap(dpi=200)
        img_bytes = pix.tobytes("png")
        reader = _get_easyocr_reader()
        results = reader.readtext(img_bytes, detail=0, paragraph=True)
        return "\n".join(results)
    except Exception:
        return ""


def extract_text_from_pdf(path: str) -> str:
    doc = fitz.open(path)
    pages_text = []
    for page in doc:
        text = page.get_text()
        if not text.strip():
            # Tesseract fallback first, then easyocr
            try:
                text = page.get_textpage_ocr(language="eng").extractText()
            except Exception:
                text = ""
            if not text.strip():
                text = _ocr_page_image(page)
        pages_text.append(text)
    doc.close()
    return "\n".join(pages_text).strip()


def extract_text_from_file(path: str) -> str:
    p = Path(path)
    ext = p.suffix.lower()
    if ext == ".pdf":
        return extract_text_from_pdf(path)
    if ext in {".txt", ".text"}:
        return p.read_text(encoding="utf-8", errors="ignore").strip()
    return ""


def ingest_folder(folder: str) -> dict[str, str]:
    folder_path = Path(folder)
    if not folder_path.is_dir():
        raise ValueError(f"Not a valid folder: {folder}")
    results = {}
    for file in sorted(folder_path.iterdir()):
        if file.suffix.lower() in SUPPORTED_EXTENSIONS:
            text = extract_text_from_file(str(file))
            if text:
                results[file.name] = text
    return results
