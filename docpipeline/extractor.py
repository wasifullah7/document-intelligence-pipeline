import re
from functools import lru_cache

from dateutil import parser as du_parser
import dateparser
import dateparser.search
import spacy


@lru_cache(maxsize=1)
def _get_nlp():
    return spacy.load("en_core_web_sm")


# ── Generic field extractors ──────────────────────────────────────────────────

def extract_email(text: str) -> str | None:
    match = re.search(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", text)
    return match.group() if match else None


def extract_phone(text: str) -> str | None:
    match = re.search(r"(\+?1?\s?)?(\(?\d{3}\)?[\s.\-]?\d{3}[\s.\-]?\d{4})", text)
    return match.group().strip() if match else None


def extract_date(text: str) -> str | None:
    """Parse a date from a short text snippet (e.g., a label value like 'January 15, 2024')."""
    try:
        dt = du_parser.parse(text, fuzzy=True)
        return dt.strftime("%Y-%m-%d")
    except (ValueError, OverflowError):
        pass
    results = dateparser.search.search_dates(text, settings={"PREFER_DAY_OF_MONTH": "first"})
    if results:
        return results[0][1].strftime("%Y-%m-%d")
    return None


# Explicit date patterns for scanning full document text — avoids fuzzy misfires
_DATE_SCAN_PATTERNS = [
    r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r"\.?\s+\d{1,2},?\s+\d{4}\b",          # January 15, 2024
    r"\b\d{1,2}\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r"\.?\s+\d{4}\b",                        # 15 January 2024
    r"\b\d{4}[-/]\d{2}[-/]\d{2}\b",         # 2024-01-15
    r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b",  # 01/15/2024
]


def _find_first_date(text: str) -> str | None:
    """Scan full document text for explicit date patterns — never fuzzy-parses the whole text."""
    for pattern in _DATE_SCAN_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                dt = du_parser.parse(match.group(), fuzzy=False)
                return dt.strftime("%Y-%m-%d")
            except (ValueError, OverflowError):
                continue
    return None


def extract_amount(text: str) -> float | None:
    # Dollar sign + number
    match = re.search(r"\$\s*([\d,]+(?:\.\d{1,2})?)", text)
    if match:
        return float(match.group(1).replace(",", ""))
    # Plain number followed by optional currency hint
    match = re.search(r"([\d,]+\.\d{2})\s*(?:USD|EUR|GBP)?", text)
    if match:
        return float(match.group(1).replace(",", ""))
    return None


# ── Per-class extractors ──────────────────────────────────────────────────────

def _extract_invoice_fields(text: str) -> dict:
    # invoice_number — match "Invoice Number: INV-001", "Invoice: INV-001", "INVOICE: INV-100"
    inv_match = re.search(
        r"(?:invoice\s*(?:#|no\.?|number|:)\s*:?\s*)([A-Z]{2,5}[-\s][\w\-]{1,20}|\d{3,}[\w\-]*)",
        text,
        re.IGNORECASE,
    )
    invoice_number = inv_match.group(1).strip() if inv_match else None

    # date — look for labelled date line first, fall back to explicit pattern scan
    date_match = re.search(r"(?:^|\b)date\s*:?\s*([A-Za-z0-9,\s]+\d{4})", text, re.IGNORECASE)
    date = extract_date(date_match.group(1)) if date_match else None
    if date is None:
        date = _find_first_date(text)

    # company: spaCy ORG first, then regex label fallback
    nlp = _get_nlp()
    doc = nlp(text[:500])
    company = next((ent.text for ent in doc.ents if ent.label_ == "ORG"), None)
    if not company:
        co_match = re.search(
            r"(?:company|bill\s+to|billed?\s+(?:to|by|from)|from|vendor|seller)\s*:?\s*([A-Z][^\n,]{2,40})",
            text,
            re.IGNORECASE,
        )
        company = co_match.group(1).strip() if co_match else None

    # total_amount — specific labels first, then standalone "total" requiring $ sign
    total_match = re.search(
        r"(?:total\s+(?:amount|due)|amount\s+due|balance\s+due)\s*:?\s*\$?\s*([\d,]+(?:\.\d{1,2})?)",
        text,
        re.IGNORECASE,
    )
    if not total_match:
        total_match = re.search(
            r"\btotal\b\s*:?\s*\$\s*([\d,]+(?:\.\d{1,2})?)",
            text,
            re.IGNORECASE,
        )
    total_amount = (
        float(total_match.group(1).replace(",", "")) if total_match else extract_amount(text)
    )

    return {
        "invoice_number": invoice_number,
        "date": date,
        "company": company,
        "total_amount": total_amount,
    }


def _normalize_ocr_email(text: str) -> str:
    """Fix OCR-injected spaces inside email addresses: 'wasif wwez@ gmail.com' → 'wasifwwez@gmail.com'."""
    # "wasif wwez@" → "wasifwwez@"  (word before token-ending-with-@)
    text = re.sub(r'(\w+)\s+(\w+@)', r'\1\2', text)
    # "wasifwwez@ gmail.com" → "wasifwwez@gmail.com"  (token ending with @, space, domain)
    text = re.sub(r'(\w+@)\s+(\S+)', r'\1\2', text)
    # "user@ gmail.com" → "user@gmail.com"  (@ followed by space then domain)
    text = re.sub(r'@\s+(\w)', r'@\1', text)
    # "@gmail. com" → "@gmail.com"  (split TLD)
    text = re.sub(r'(\.\w+)\s+(com|net|org|edu|io|co|gov|mil)\b', r'\1\2', text, flags=re.IGNORECASE)
    return text


def _extract_resume_fields(text: str) -> dict:
    # Normalize OCR artifacts before extraction
    text_clean = _normalize_ocr_email(text)
    email = extract_email(text_clean)
    phone = extract_phone(text)

    # name — four-tier cascade, most reliable first
    name = None

    # Tier 1: leading UPPERCASE words on the very first line (handles OCR output like
    # "WASIF ULLAH wasif@email.com +92..." where name and contact run together).
    # Only match words that START with uppercase — stops before lowercase words.
    first_line = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
    leading = re.match(r'^([A-Z][A-Za-z]{1,}(?:\s+[A-Z][A-Za-z]{1,}){1,2})', first_line)
    if leading and len(leading.group(1).split()) >= 2:
        candidate = leading.group(1)
        if not re.match(r'^(resume|curriculum|vitae|profile|summary|objective|contact)$',
                        candidate.split()[0], re.IGNORECASE):
            name = candidate.title()

    # Tier 2: spaCy PERSON entity on first 300 chars
    if not name:
        nlp = _get_nlp()
        doc = nlp(text[:300])
        raw_name = next((ent.text for ent in doc.ents if ent.label_ == "PERSON"), None)
        if raw_name and re.match(r"^[A-Za-z][A-Za-z\s\.\-']{1,40}$", raw_name.strip()):
            # Reject single-word place/generic names that spaCy misclassifies
            spacy_words = raw_name.strip().split()
            if len(spacy_words) >= 2:
                name = raw_name.strip().title()

    # Tier 3: scan first 5 lines for a clean 2–4 word capitalised line
    if not name:
        first_lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        for line in first_lines[:5]:
            # Skip single-word all-caps headers ("RESUME", "CV", "EDUCATION")
            words = line.split()
            if line.isupper() and len(words) <= 1:
                continue
            # Skip lines that contain email/phone/URL fragments
            if re.search(r'[@:/]|\d{5,}', line):
                continue
            if 2 <= len(words) <= 4 and words[0][0].isupper():
                name = line.title()
                break

    # Tier 4: take first 2–3 pure-alpha tokens anywhere in first 200 chars
    if not name:
        tokens = re.findall(r'\b[A-Za-z]{2,}\b', text[:200])
        candidates = [t for t in tokens
                      if not re.match(r'^(mr|ms|dr|prof|the|and|or|in|at|of|from|for|to)$', t, re.IGNORECASE)]
        if len(candidates) >= 2:
            name = ' '.join(candidates[:2]).title()

    # experience_years: "X years of experience" or "X+ years experience"
    exp_match = re.search(
        r"(\d+)\+?\s+years?\s+(?:of\s+)?(?:experience|exp\.?)",
        text,
        re.IGNORECASE,
    )
    experience_years = int(exp_match.group(1)) if exp_match else None

    return {
        "name": name,
        "email": email,
        "phone": phone,
        "experience_years": experience_years,
    }


def _extract_utility_fields(text: str) -> dict:
    # account_number — capture then require at least one digit (guards against column-header matches)
    acc_match = re.search(
        r"(?:account\s*(?:#|no\.?|number)\s*:?\s*)([A-Z0-9][\w\-]{2,20})",
        text,
        re.IGNORECASE,
    )
    if acc_match:
        candidate = acc_match.group(1).strip()
        account_number = candidate if re.search(r"\d", candidate) else None
    else:
        account_number = None

    # date
    date_match = re.search(
        r"(?:date|dated|bill\s+date|service\s+date)\s*:?\s*([A-Za-z0-9,\s]+\d{4})",
        text,
        re.IGNORECASE,
    )
    date = extract_date(date_match.group(1)) if date_match else None
    if date is None:
        date = _find_first_date(text)

    # usage_kwh
    kwh_match = re.search(r"([\d,]+(?:\.\d+)?)\s*kWh", text, re.IGNORECASE)
    usage_kwh = float(kwh_match.group(1).replace(",", "")) if kwh_match else None

    # amount_due
    due_match = re.search(
        r"(?:amount\s+due|total\s+due|balance\s+due|payment\s+due)\s*:?\s*\$?\s*([\d,]+(?:\.\d{1,2})?)",
        text,
        re.IGNORECASE,
    )
    amount_due = (
        float(due_match.group(1).replace(",", "")) if due_match else extract_amount(text)
    )

    return {
        "account_number": account_number,
        "date": date,
        "usage_kwh": usage_kwh,
        "amount_due": amount_due,
    }


# ── Public API ────────────────────────────────────────────────────────────────

def extract_fields(text: str, doc_class: str) -> dict:
    """Dispatch to the correct extractor based on document class."""
    if doc_class == "Invoice":
        return _extract_invoice_fields(text)
    if doc_class == "Resume":
        return _extract_resume_fields(text)
    if doc_class == "Utility Bill":
        return _extract_utility_fields(text)
    return {}
