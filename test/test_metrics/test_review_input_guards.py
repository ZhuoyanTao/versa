"""Regression checks for invalid collections and nonmatching reference audio."""

import importlib
from unittest.mock import Mock

import numpy as np
import pytest


@pytest.mark.parametrize(
    "module_name,class_name",
    [
        ("fad", "FadMetric"),
        ("individual_fad", "IndividualFadMetric"),
        ("kid", "KidMetric"),
    ],
)
@pytest.mark.parametrize("source", ["references", "metadata", "config", "predictions"])
def test_empty_collections_rejected_before_backend(module_name, class_name, source):
    """Empty explicit inputs must not fall through to another baseline or backend."""
    module = importlib.import_module(f"versa.corpus_metrics.{module_name}")
    metric = object.__new__(getattr(module, class_name))
    metric.io = "dir"
    metric.baseline = {"a": "a.wav", "b": "b.wav"}
    metric.module = Mock()
    metric.cache_dir = "unused"
    metric.use_inf = False
    references = None
    metadata = {}
    predictions = {"p": "p.wav", "q": "q.wav"}
    if source == "references":
        references = {}
        metadata["baseline_files"] = metric.baseline
    elif source == "metadata":
        metadata["baseline_files"] = {}
    elif source == "config":
        metric.baseline = {}
    else:
        predictions = {}
    with pytest.raises(ValueError, match="non-empty|at least 2"):
        metric.compute(predictions, references, metadata)
    assert metric.module.mock_calls == []


@pytest.mark.parametrize("test_length", [0, 4])
def test_noresqa_empty_reference_rejected(test_length):
    """Reject an empty reference before attempting to repeat it."""
    from versa.utterance_metrics.noresqa_utils.noresqa_utils import check_size

    with pytest.raises(ValueError, match="non-empty reference"):
        check_size(np.array([]), np.zeros(test_length))


@pytest.mark.parametrize(
    "length,expected", [(1, [1]), (2, [1, 2]), (5, [1, 2, 1, 2, 1])]
)
def test_noresqa_reference_duration_adjustment(length, expected):
    """Preserve truncation, equal durations, and repetition for valid references."""
    from versa.utterance_metrics.noresqa_utils.noresqa_utils import check_size

    audio_test = np.zeros(length)
    reference, result = check_size(np.array([1, 2]), audio_test)
    np.testing.assert_array_equal(reference, expected)
    assert result is audio_test
