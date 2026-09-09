"""Surviving Qwen normalization and script entrypoint contracts."""

import importlib
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.postprocess import qwen_normalization as rules
from scripts.postprocess.check_llm_result_match import (
    EXPECTED_FORMATS as FILTER_FORMATS,
)


@pytest.mark.parametrize(
    "module_name,class_name",
    [
        ("qwen2_audio_json_output_standardizer", "JsonOutputStandardizer"),
        ("qwen2_audio_jsonl_standardizer_batch", "BatchInferenceStandardizer"),
    ],
)
def test_standardizer_rules_without_models(monkeypatch, module_name, class_name):
    module = importlib.import_module("scripts.postprocess." + module_name)
    monkeypatch.setattr(module, "AutoTokenizer", None)
    monkeypatch.setattr(module, "AutoModelForCausalLM", None)
    standardizer = getattr(module, class_name)()
    assert not standardizer.llm_available
    for key, schema in rules.EXPECTED_FORMATS.items():
        for category in schema.get("categories", []):
            assert standardizer._rules_based_standardize(
                key, category
            ) == rules.rules_based_standardize(key, category)
    for source, translated in rules.TRANSLATION_DICT.items():
        # Preserve ordered replacement, including historical partial terms.
        expected = {
            "很快": "很Fast",
            "很慢": "很Slow",
            "很高": "很High",
            "很低": "很Low",
            "不清晰": "不High clarity",
        }.get(source, translated)
        assert standardizer._translate_non_english(source) == expected
    assert (
        standardizer._rules_based_standardize("qwen_language", "spoken in Polish")
        == "Polish"
    )
    assert (
        standardizer._rules_based_standardize("qwen_overlapping_speech", "no overlap")
        == "No overlap"
    )
    assert (
        standardizer._rules_based_standardize("unknown", "unrecognized")
        == "unrecognized"
    )
    number_key = next(
        key
        for key, schema in rules.EXPECTED_FORMATS.items()
        if schema["type"] == "number"
    )
    for raw, expected in [
        ("99 speakers", 10),
        ("0", 1),
        ("two speakers", 2),
        ("???", 1),
    ]:
        assert standardizer._rules_based_standardize(number_key, raw) == expected
    # The LLM path historically does not clamp numeric output.
    assert standardizer._process_llm_output(number_key, "Answer: 99") == 99
    assert standardizer._process_llm_output(number_key, "```") == 1
    assert standardizer._process_llm_output("unknown", "Category: Happy") == "Happy"


def test_filter_retains_historical_subset():
    assert set(FILTER_FORMATS) == set(rules.EXPECTED_FORMATS) - {
        "qwen_overlapping_speech"
    }


@pytest.mark.parametrize(
    "script",
    [
        "qwen2_audio_json_output_standardizer.py",
        "qwen2_audio_jsonl_standardizer_batch.py",
    ],
)
def test_direct_script_import(script, tmp_path):
    path = Path(__file__).resolve().parents[1] / "scripts/postprocess" / script
    result = subprocess.run(
        [sys.executable, str(path), "--help"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "usage:" in result.stdout
