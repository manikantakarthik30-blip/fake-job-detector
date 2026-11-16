import pytest
from app_core import MAX_ROWS, rule_based_prediction, validate_csv


def test_csv_schema_and_empty_rows():
    frame = validate_csv(b"title,description\nEngineer,Build software\n,\n")
    assert len(frame) == 1
    assert frame.loc[0, "title"] == "Engineer"


def test_rejects_garbage_and_empty_csv():
    with pytest.raises(ValueError, match="job-text column"):
        validate_csv(b"foo,bar\na,b\n")
    with pytest.raises(ValueError, match="empty"):
        validate_csv(b"")


def test_row_limit():
    data = "title\n" + "Engineer\n" * (MAX_ROWS + 1)
    with pytest.raises(ValueError, match="20,000-row"):
        validate_csv(data.encode())


def test_fallback_reasons():
    result = rule_based_prediction({"title": "Remote entry", "description": "Pay registration fee", "has_company_logo": 0})
    assert result["verdict"] == "LIKELY_FRAUDULENT"
    assert any("registration fee" in reason for reason in result["reasons"])
    assert any("logo" in reason.lower() for reason in result["reasons"])
