import json
import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest

from versa import config_validation, scorer_shared
from versa.bin.scorer import (
    _text_required_multi_source_metrics,
    get_parser,
    main as scorer_main,
)
from versa.definition import (
    BaseMetric,
    MetricCategory,
    MetricMetadata,
    MetricRegistry,
    MetricType,
)
from versa.scorer_shared import VersaScorer, find_files


class OrderedSourceMetric(BaseMetric):
    calls = []

    def _setup(self):
        pass

    def compute(self, predictions, references=None, metadata=None):
        self.calls.append((predictions, references, metadata))
        return {"ordered_source_score": float(len(predictions))}

    def get_metadata(self):
        return MetricMetadata(
            name="ordered_source",
            category=MetricCategory.DEPENDENT,
            metric_type=MetricType.DICT,
            requires_reference=True,
            requires_text=False,
            gpu_compatible=False,
            auto_install=False,
            dependencies=[],
            description="Dependency-light ordered source test metric.",
            requires_multiple_sources=True,
        )


def _source_mappings():
    sample_files = list(find_files("test/test_samples/test2").values())
    sample_file = sample_files[0]
    keys = ["mixture-a", "mixture-b"]
    mapping = {key: sample_file for key in keys}
    return [dict(mapping), dict(mapping)], [dict(mapping), dict(mapping)]


def test_multi_source_parser_accepts_ordered_scp_lists():
    args = get_parser().parse_args(
        [
            "--pred_sources",
            "pred-1.scp",
            "pred-2.scp",
            "--gt_sources",
            "ref-1.scp",
            "ref-2.scp",
        ]
    )

    assert args.pred_sources == ["pred-1.scp", "pred-2.scp"]
    assert args.gt_sources == ["ref-1.scp", "ref-2.scp"]


def test_multi_source_rejects_only_metrics_that_require_text():
    metadata = OrderedSourceMetric().get_metadata()
    score_config = [{"name": "ordered_source"}]
    registry = MetricRegistry()
    registry.register(OrderedSourceMetric, metadata)

    assert _text_required_multi_source_metrics(score_config, registry) == []

    text_registry = MetricRegistry()
    text_registry.register(OrderedSourceMetric, replace(metadata, requires_text=True))
    assert _text_required_multi_source_metrics(score_config, text_registry) == [
        "ordered_source"
    ]


def test_multi_source_cli_rejects_text_metric_before_generic_validation(
    monkeypatch, tmp_path, capsys
):
    registry = MetricRegistry()
    registry.register(
        OrderedSourceMetric,
        replace(OrderedSourceMetric().get_metadata(), requires_text=True),
    )
    monkeypatch.setattr(
        scorer_shared,
        "VersaScorer",
        lambda: SimpleNamespace(registry=registry),
    )

    def unexpected_validation(*args, **kwargs):
        pytest.fail("generic validation ran before multi-source text validation")

    monkeypatch.setattr(
        config_validation, "validate_score_config", unexpected_validation
    )
    config_path = tmp_path / "text_metric.yaml"
    config_path.write_text("- name: ordered_source\n", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "versa-scorer",
            "--score_config",
            str(config_path),
            "--pred_sources",
            "pred-1.scp",
            "pred-2.scp",
            "--gt_sources",
            "ref-1.scp",
            "ref-2.scp",
        ],
    )

    with pytest.raises(SystemExit) as error:
        scorer_main()

    assert error.value.code == 2
    assert (
        "does not yet support metrics requiring reference text"
        in capsys.readouterr().err
    )


def test_multi_source_pipeline_preserves_order_and_writes_jsonl(tmp_path):
    registry = MetricRegistry()
    metadata = OrderedSourceMetric().get_metadata()
    registry.register(OrderedSourceMetric, metadata)
    scorer = VersaScorer(registry)
    suite = scorer.load_metrics([{"name": "ordered_source"}], use_gt=True)
    predictions, references = _source_mappings()
    output_file = tmp_path / "mapss.jsonl"

    OrderedSourceMetric.calls = []
    scores = scorer.score_multi_source_utterances(
        predictions,
        suite,
        references,
        output_file=str(output_file),
        io="soundfile",
    )

    assert scores == [
        {"key": "mixture-a", "ordered_source_score": 2.0},
        {"key": "mixture-b", "ordered_source_score": 2.0},
    ]
    assert len(OrderedSourceMetric.calls) == 2
    assert all(
        len(call[0]) == 2 and len(call[1]) == 2 for call in OrderedSourceMetric.calls
    )
    assert all(call[2]["sample_rate"] == 16000 for call in OrderedSourceMetric.calls)
    assert [
        json.loads(line)
        for line in output_file.read_text(encoding="utf-8").splitlines()
    ] == scores


def test_multi_source_pipeline_rejects_key_mismatch():
    registry = MetricRegistry()
    metadata = OrderedSourceMetric().get_metadata()
    registry.register(OrderedSourceMetric, metadata)
    scorer = VersaScorer(registry)
    suite = scorer.load_metrics([{"name": "ordered_source"}], use_gt=True)

    with pytest.raises(ValueError, match="missing keys"):
        scorer.score_multi_source_utterances(
            [{"a": "one.wav"}, {"a": "two.wav"}],
            suite,
            [{"a": "one.wav"}, {}],
            io="soundfile",
        )
