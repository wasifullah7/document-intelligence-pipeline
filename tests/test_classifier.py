from docpipeline.classifier import classify_by_keywords, classify_document

INVOICE_TEXT = """
INVOICE
Invoice Number: INV-2024-001
Bill To: ACME Corporation
Date: January 15, 2024
Amount Due: $1,250.00
Total Amount: $1,250.00
"""

RESUME_TEXT = """
John Doe
john.doe@email.com | (555) 123-4567

Work Experience
Software Engineer at Google — 2019 to present
Education
B.S. Computer Science, MIT 2018
Skills: Python, FastAPI, Machine Learning
5 years of experience
"""

UTILITY_TEXT = """
ELECTRICITY BILL
Account Number: ACC-789456
Service Address: 123 Main St
Meter Reading: 1245 kWh
Amount Due: $145.50
Due Date: February 1, 2024
"""

OTHER_TEXT = """
Meeting notes from the Q3 planning session.
Agenda: roadmap review, team sync, budget allocation.
No specific document type indicators present.
"""


def test_keyword_classify_invoice():
    assert classify_by_keywords(INVOICE_TEXT) == "Invoice"


def test_keyword_classify_resume():
    assert classify_by_keywords(RESUME_TEXT) == "Resume"


def test_keyword_classify_utility():
    assert classify_by_keywords(UTILITY_TEXT) == "Utility Bill"


def test_keyword_classify_returns_none_for_ambiguous():
    assert classify_by_keywords("This is a simple memo.") is None


def test_classify_document_invoice():
    assert classify_document(INVOICE_TEXT) == "Invoice"


def test_classify_document_resume():
    assert classify_document(RESUME_TEXT) == "Resume"


def test_classify_document_utility():
    assert classify_document(UTILITY_TEXT) == "Utility Bill"


def test_classify_document_returns_valid_label():
    result = classify_document(OTHER_TEXT)
    assert result in {"Invoice", "Resume", "Utility Bill", "Other", "Unclassifiable"}
