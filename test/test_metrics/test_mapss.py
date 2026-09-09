from types import SimpleNamespace

import numpy as np
import pytest

from versa.definition import MetricCategory, MetricRegistry, MetricType
from versa.metric_discovery import create_metric_discovery_registry, describe_metric
from versa.utterance_metrics import mapss


class FakeSummary:
    def __init__(self, rows):
        self.rows = rows

    def to_dict(self, orient):
        assert orient == "index"
        return self.rows


class FakeResult:
    def __init__(self):
        self.summary = FakeSummary(
            {
                "Vocals": {"ps": 0.75, "pm": 0.8, "ps_frames": 10, "pm_frames": 9},
                "Drums": {"ps": 0.5, "pm": 0.6, "ps_frames": 8, "pm_frames": 7},
            }
        )
        self.saved = None

    def save(self, directory, plot):
        self.saved = (directory, plot)


def test_mapss_metric_forwards_ordered_sources_and_retains_frames(
    monkeypatch, tmp_path
):
    call = SimpleNamespace(kwargs=None)
    result = FakeResult()

    def fake_mapss(**kwargs):
        call.kwargs = kwargs
        return result

    monkeypatch.setattr(mapss, "mapss_compute", fake_mapss)
    metric = mapss.MapssMetric(
        {
            "source_names": ["Vocals", "Drums"],
            "cache_dir": tmp_path,
            "model": "raw",
            "add_ci": False,
            "use_gpu": False,
        }
    )
    predictions = [np.ones(8000), np.ones(8000) * 2]
    references = [np.ones(8000) * 3, np.ones(8000) * 4]

    scores = metric.compute(
        predictions,
        references,
        metadata={"key": "mixture/one", "sample_rate": 16000},
    )

    assert call.kwargs["output"] is predictions
    assert call.kwargs["reference"] is references
    assert call.kwargs["source_names"] == ["Vocals", "Drums"]
    assert call.kwargs["model"] == "raw"
    assert call.kwargs["max_gpus"] == 0
    assert scores == {
        "mapss_ps_vocals": 0.75,
        "mapss_pm_vocals": 0.8,
        "mapss_ps_drums": 0.5,
        "mapss_pm_drums": 0.6,
        "mapss_result_dir": str(tmp_path / "results" / "mixture_one-207dc6d4"),
    }
    assert result.saved == (tmp_path / "results" / "mixture_one-207dc6d4", False)


@pytest.mark.parametrize(
    ("predictions", "references", "message"),
    [
        ([np.ones(10)], [np.ones(10)], "at least two ordered predicted"),
        ([np.ones(10), np.ones(10)], [np.ones(10)], "at least two ordered reference"),
        (
            [np.ones(10), np.ones(10), np.ones(10)],
            [np.ones(10), np.ones(10)],
            "same number",
        ),
    ],
)
def test_mapss_metric_validates_source_pairs(
    monkeypatch, predictions, references, message
):
    monkeypatch.setattr(mapss, "mapss_compute", lambda **kwargs: FakeResult())
    metric = mapss.MapssMetric()

    with pytest.raises(ValueError, match=message):
        metric.compute(predictions, references)


def test_mapss_setup_reports_missing_optional_dependency(monkeypatch):
    monkeypatch.setattr(mapss, "mapss_compute", None)

    with pytest.raises(ImportError, match=r"tools/install_mapss\.sh"):
        mapss.MapssMetric()


def test_mapss_registration_metadata():
    registry = MetricRegistry()

    mapss.register_mapss_metric(registry)
    metadata = registry.get_metadata("MAPSS")

    assert metadata.name == "mapss"
    assert metadata.category == MetricCategory.DEPENDENT
    assert metadata.metric_type == MetricType.DICT
    assert metadata.requires_reference
    assert metadata.requires_multiple_sources
    assert metadata.gpu_compatible
    assert not metadata.auto_install
    assert metadata.dependencies == ["mapss"]
    assert registry.get_metric("mapss_measures") is mapss.MapssMetric


def test_mapss_is_discoverable_without_optional_backend():
    registry = create_metric_discovery_registry()

    metadata = registry.get_metadata("mapss")
    description = describe_metric(registry, "mapss")

    assert metadata.requires_multiple_sources
    assert metadata.dependencies == ["mapss"]
    assert "requires_multiple_sources: true" in description


def test_mapss_rejects_colliding_normalized_source_names():
    summary = FakeSummary(
        {
            "lead vocal": {"ps": 1, "pm": 1, "ps_frames": 1, "pm_frames": 1},
            "lead_vocal": {"ps": 1, "pm": 1, "ps_frames": 1, "pm_frames": 1},
        }
    )

    with pytest.raises(ValueError, match="remain unique"):
        mapss._summary_scores(summary)
