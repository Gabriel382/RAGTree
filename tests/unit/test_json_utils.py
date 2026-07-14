"""Unit tests for robust JSON extraction and relation normalization."""

from ragtree.generation.json_utils import extract_first_json, normalize_relations


def test_extract_plain_json():
    assert extract_first_json('{"a": 1}') == {"a": 1}


def test_extract_json_from_markdown_fence():
    text = 'Sure! Here is the result:\n```json\n{"CAUSES": [["E1", "E2"]]}\n```\nDone.'
    assert extract_first_json(text) == {"CAUSES": [["E1", "E2"]]}


def test_extract_json_embedded_in_prose():
    text = 'The relations are {"CAUSES": []} as requested.'
    assert extract_first_json(text) == {"CAUSES": []}


def test_extract_returns_none_for_garbage():
    assert extract_first_json("no json here { broken") is None
    assert extract_first_json("") is None


def test_normalize_fills_missing_and_drops_unknown_types():
    raw = {"CAUSES": [["E1", "E2"]], "HALLUCINATED": [["E1", "E3"]]}
    normalized = normalize_relations(raw, ["CAUSES", "PRECONDITION"])
    assert normalized == {"CAUSES": [["E1", "E2"]], "PRECONDITION": []}


def test_normalize_skips_malformed_pairs():
    raw = {"CAUSES": [["E1", "E2"], ["only-one"], "junk", ["E1", 42], ["E3", "E4"]]}
    normalized = normalize_relations(raw, ["CAUSES"])
    assert normalized == {"CAUSES": [["E1", "E2"], ["E3", "E4"]]}


def test_normalize_without_schema_keeps_source_keys():
    raw = {"X": [["a", "b"]], "Y": "not a list"}
    assert normalize_relations(raw) == {"X": [["a", "b"]], "Y": []}


def test_normalize_non_dict_input():
    assert normalize_relations(None, ["CAUSES"]) == {"CAUSES": []}
    assert normalize_relations([1, 2], ["CAUSES"]) == {"CAUSES": []}
