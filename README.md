# Document Intelligence Pipeline

A fully **local** AI system that ingests PDF/text documents, classifies them, extracts structured fields, and provides semantic search — served via a FastAPI REST API with Swagger UI. No cloud APIs. No internet required at runtime.

---

## Architecture

```
PDF / TXT files
      │
      ▼
  ingestor.py       ← PyMuPDF: extract clean text
      │
      ▼
  classifier.py     ← Keyword rules → zero-shot NLI (deberta-v3-xsmall)
      │
      ▼
  extractor.py      ← Regex + spaCy NER → structured fields
      │
      ▼
  output.json       ← Final deliverable

  retriever.py      ← bge-small-en-v1.5 + ChromaDB → semantic index
      │
      ▼
  api.py            ← FastAPI: /process /search /results /documents
```

---

## Tech Stack

| Component | Library | Model / Size |
|-----------|---------|-------------|
| PDF extraction | PyMuPDF 1.24 | — |
| Classification | Transformers 4.44 | MoritzLaurer/deberta-v3-xsmall-zeroshot (~142 MB) |
| NER | spaCy 3.7 | en_core_web_sm (~12 MB) |
| Embeddings | SentenceTransformers 3.1 | BAAI/bge-small-en-v1.5 (~133 MB) |
| Vector DB | ChromaDB 0.5 | — |
| API | FastAPI 0.115 + uvicorn | — |

**Total model footprint: ~287 MB. Runs entirely on CPU.**

---

## Installation

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

AI models (deberta + bge) are downloaded automatically on first run and cached locally by HuggingFace/SentenceTransformers. All subsequent runs are fully offline.

---

## Running the API

```bash
python main.py
```

Open Swagger UI: **http://localhost:8000/docs**

---

## Usage

### 1. Process a folder of documents

Place your PDF or `.txt` files into the `documents/` folder, then call:

**Swagger UI → POST /process**
```json
{ "folder": "./documents" }
```

Or via curl:
```bash
curl -X POST http://localhost:8000/process \
     -H "Content-Type: application/json" \
     -d '{"folder": "./documents"}'
```

### 2. View all results

```
GET /results     → full output.json as JSON
GET /documents   → list of filenames + class labels
```

### 3. Semantic search

```
GET /search?q=payments+due+in+January
GET /search?q=Python+developer+5+years+experience&top_k=3
```

---

## Output Format (output.json)

```json
{
  "invoice_1.pdf": {
    "class": "Invoice",
    "invoice_number": "INV-1234",
    "date": "2025-01-01",
    "company": "ACME Ltd.",
    "total_amount": 350.50
  },
  "resume_1.pdf": {
    "class": "Resume",
    "name": "John Doe",
    "email": "john@example.com",
    "phone": "123-456-7890",
    "experience_years": 5
  },
  "utility_1.pdf": {
    "class": "Utility Bill",
    "account_number": "ACC-789456",
    "date": "2025-03-01",
    "usage_kwh": 350.0,
    "amount_due": 87.50
  }
}
```

---

## Classification Logic

Documents are classified using a three-tier strategy:

1. **Keyword rules** — regex patterns for clear signals (Invoice, Resume, Utility Bill). Fast, zero ML compute.
2. **Zero-shot NLI** — `deberta-v3-xsmall-zeroshot` for ambiguous documents. No training data needed.
3. **Confidence gate** — scores below 0.45 → `Unclassifiable`.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/process` | Process a folder → classify + extract + index |
| `GET` | `/results` | Return full output.json |
| `GET` | `/documents` | List filenames + class labels |
| `GET` | `/search?q=...` | Semantic search (optional `&top_k=N`) |

Full interactive docs at `http://localhost:8000/docs` (Swagger UI).

---

## Running Tests

```bash
pytest tests/ -v
```

Expected: 36 tests, all passing.

---

## Supported Document Types

| Class | Extracted Fields |
|-------|-----------------|
| Invoice | invoice_number, date, company, total_amount |
| Resume | name, email, phone, experience_years |
| Utility Bill | account_number, date, usage_kwh, amount_due |
| Other | (none) |
| Unclassifiable | (none) |
