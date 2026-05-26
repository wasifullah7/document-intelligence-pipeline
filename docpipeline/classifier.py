import re
from functools import lru_cache
from transformers import pipeline as hf_pipeline

# Keyword patterns per class — need >= 2 matches for high-confidence classification
KEYWORD_RULES: dict[str, list[str]] = {
    "Invoice": [
        r"invoice\s*(?:#|number|no\.?|:)",
        r"bill\s+to",
        r"amount\s+due",
        r"total\s+amount",
        r"\binv[-\s]?\d+",
        r"purchase\s+order",
        r"payment\s+terms",
        r"ship\s+to",
    ],
    "Resume": [
        r"work\s+experience",
        r"work\s+history",
        r"curriculum\s+vitae",
        r"\d+\s+years?\s+of\s+(?:relevant\s+)?experience",
        r"(?:bachelor|master|b\.s\.|m\.s\.|b\.a\.|ph\.d)",
        r"(?:professional\s+)?(?:profile|summary|objective)\s*\n",
        r"(?:contact|personal)\s+(?:info(?:rmation)?|details)",
    ],
    "Utility Bill": [
        r"\bkwh\b",
        r"meter\s+reading",
        r"service\s+address",
        r"electricity\s+bill|gas\s+bill|water\s+bill|utility\s+bill",
        r"account\s+(?:#|number|no\.?)",
        r"\btherms?\b",
        r"billing\s+statement",
        r"service\s+period",
        r"customer\s+(?:#|number|no\.?)",
        r"\bgas\s+or\s+electric\b",
    ],
}

LABEL_MAP = {
    "invoice": "Invoice",
    "resume": "Resume",
    "utility bill": "Utility Bill",
    "other document": "Other",
}

NLI_CANDIDATE_LABELS = list(LABEL_MAP.keys())
NLI_HYPOTHESIS = "This document is a {}."
CONFIDENCE_THRESHOLD = 0.45


@lru_cache(maxsize=1)
def _get_nli_classifier():
    return hf_pipeline(
        "zero-shot-classification",
        model="MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33",
        device=-1,  # CPU
    )


def classify_by_keywords(text: str) -> str | None:
    """Return a class label if >= 2 keyword patterns match, else None."""
    text_lower = text.lower()
    match_counts: dict[str, int] = {}
    for label, patterns in KEYWORD_RULES.items():
        count = sum(1 for p in patterns if re.search(p, text_lower))
        if count >= 2:
            match_counts[label] = count
    if match_counts:
        return max(match_counts, key=match_counts.get)
    return None


def classify_document(text: str) -> str:
    """
    Three-tier classification:
    1. Keyword rules  — fast, deterministic, handles clear cases
    2. Zero-shot NLI  — for ambiguous documents
    3. Confidence gate — scores < 0.45 → Unclassifiable
    """
    # Tier 1 — keywords
    label = classify_by_keywords(text)
    if label:
        return label

    # Tier 2 — NLI model on first 2000 chars (~512 deberta tokens, model's full window)
    classifier = _get_nli_classifier()
    result = classifier(
        text[:2000],
        candidate_labels=NLI_CANDIDATE_LABELS,
        hypothesis_template=NLI_HYPOTHESIS,
    )
    top_label: str = result["labels"][0]
    top_score: float = result["scores"][0]

    # Tier 3 — confidence gate
    if top_score < CONFIDENCE_THRESHOLD:
        return "Unclassifiable"

    return LABEL_MAP.get(top_label, "Other")
