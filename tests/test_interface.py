"""Tests for the LLM interface: parsing accuracy and edge-case handling.

The LLM callable is injected as a fake, so these tests exercise the real
extraction/validation/response pipeline without network access or API keys.
"""

import json

from src.llm_interface import (
    answer,
    extract_json,
    missing_required,
    normalize_features,
    parse_query,
)


def fake_llm(payload: dict):
    """Build a fake LLM that always returns the given JSON payload."""

    def _llm(messages, **kwargs):
        return json.dumps(payload)

    return _llm


def test_parse_query_extracts_and_normalizes_features():
    llm = fake_llm(
        {
            "sex": "f",
            "age": 24,
            "height": 175,
            "weight": 63,
            "noc": "usa",
            "sport": "swimming",
            "season": None,
            "year": 2016,
            "out_of_scope": False,
        }
    )
    features = parse_query(
        "I'm a 24-year-old female swimmer from the USA, 175cm and 63kg", llm
    )
    assert features["sex"] == "F"
    assert features["age"] == 24
    assert features["noc"] == "USA"
    assert features["sport"] == "Swimming"
    assert features["season"] == "Summer"  # inferred from the sport
    assert not features["out_of_scope"]
    assert missing_required(features) == []


def test_incomplete_input_triggers_clarifying_question():
    llm = fake_llm(
        {"sex": None, "age": 30, "noc": None, "sport": "Judo", "out_of_scope": False}
    )
    result = answer("How likely is a 30-year-old judoka to medal?", bundle=None, llm=llm)
    assert result["prediction"] is None
    reply = result["reply"].lower()
    assert "sex" in reply and "country" in reply


def test_out_of_scope_query_is_declined():
    llm = fake_llm({"out_of_scope": True})
    result = answer("What's the capital of France?", bundle=None, llm=llm)
    assert result["prediction"] is None
    assert "outside" in result["reply"].lower()


def test_garbage_llm_output_handled_gracefully():
    def bad_llm(messages, **kwargs):
        return "I cannot help with that."

    result = answer("blah", bundle=None, llm=bad_llm)
    assert result["prediction"] is None
    assert "couldn't understand" in result["reply"].lower()


def test_extract_json_tolerates_code_fences():
    text = "Here you go:\n```json\n{\"sex\": \"M\", \"age\": 20}\n```"
    assert extract_json(text) == {"sex": "M", "age": 20}


def test_normalize_rejects_invalid_values():
    features = normalize_features(
        {"sex": "male?", "age": -5, "noc": "US", "sport": "  ", "year": 3000}
    )
    assert features["sex"] is None
    assert features["age"] is None
    assert features["noc"] is None
    assert features["sport"] is None
