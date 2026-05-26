from docpipeline.extractor import (
    extract_email,
    extract_phone,
    extract_amount,
    extract_fields,
    _find_first_date,
)

INVOICE_TEXT = """
INVOICE
Invoice Number: INV-2024-001
Date: January 15, 2024
Company: ACME Corporation
Bill To: John Smith
Total Amount: $1,250.00
"""

RESUME_TEXT = """
Jane Doe
jane.doe@example.com
(555) 987-6543

Software Engineer with 7 years of experience
Work Experience: Senior Developer at TechCorp 2017-present
Education: B.S. Computer Science
Skills: Python, FastAPI
"""

UTILITY_TEXT = """
ELECTRICITY BILL
Account Number: ACC-789456
Service Date: March 1, 2024
Usage: 350 kWh
Amount Due: $87.50
"""


def test_extract_email():
    assert extract_email("Contact: user@example.com for info") == "user@example.com"


def test_extract_email_none_when_missing():
    assert extract_email("No email address here") is None


def test_extract_phone():
    result = extract_phone("Call us at (555) 123-4567 anytime")
    assert result is not None
    assert "555" in result


def test_extract_amount_with_dollar_sign():
    assert extract_amount("Total: $1,250.00") == 1250.0


def test_extract_amount_without_dollar_sign():
    assert extract_amount("Balance: 87.50 USD") == 87.50


def test_extract_invoice_fields():
    result = extract_fields(INVOICE_TEXT, "Invoice")
    assert result["invoice_number"] == "INV-2024-001"
    assert result["total_amount"] == 1250.0
    assert result["company"] is not None
    assert result["date"] is not None


def test_extract_company_from_bill_to_label():
    text = "Invoice Number: INV-5678\nBill To: ACME Ltd.\nDate: 2025-01-01\nTotal Amount: $350.50"
    result = extract_fields(text, "Invoice")
    assert result["company"] is not None
    assert "ACME" in result["company"]


def test_extract_resume_fields():
    result = extract_fields(RESUME_TEXT, "Resume")
    assert result["email"] == "jane.doe@example.com"
    assert result["experience_years"] == 7
    assert result["name"] is not None


def test_extract_utility_fields():
    result = extract_fields(UTILITY_TEXT, "Utility Bill")
    assert result["account_number"] == "ACC-789456"
    assert result["usage_kwh"] == 350.0
    assert result["amount_due"] == 87.50


def test_extract_other_returns_empty():
    assert extract_fields("Random text with no structure.", "Other") == {}


def test_extract_unclassifiable_returns_empty():
    assert extract_fields("Random text.", "Unclassifiable") == {}


def test_find_first_date_iso_format():
    assert _find_first_date("Issued on 2024-01-15 for services") == "2024-01-15"


def test_find_first_date_long_format():
    result = _find_first_date("Account opened January 15, 2024 balance due")
    assert result == "2024-01-15"


def test_find_first_date_returns_none_for_no_date():
    assert _find_first_date("No dates in this text at all") is None


def test_find_first_date_ignores_non_date_numbers():
    # Invoice number INV-20240115 should NOT be parsed as a date
    result = _find_first_date("Invoice: INV-20240115 Amount: $500")
    assert result is None


def test_resume_name_skips_all_caps_header():
    text = "RESUME\nJane Doe\njane@example.com\n7 years of experience"
    result = extract_fields(text, "Resume")
    # The header "RESUME" must NOT be returned as the person's name
    assert result["name"] is not None
    assert result["name"] != "RESUME"
    assert "Jane" in result["name"]
