#!/usr/bin/env python3

# Copyright 2026 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Optional MAPSS integration for ordered source-separation outputs."""

import hashlib
import re
from pathlib import Path

from versa.definition import BaseMetric, MetricCategory, MetricMetadata, MetricType

try:
    from mapss import mapss as mapss_compute
except ImportError:
    mapss_compute = None


def _require_mapss():
    """Raise an actionable ImportError when the optional MAPSS backend is absent."""
    if mapss_compute is None:
        raise ImportError(
            "MAPSS is an optional dependency. Install the pinned backend with "
            "`tools/install_mapss.sh`. MAPSS 1.1.2 supports Python 3.10-3.12."
        )


def _safe_path_component(value):
    """Replace unsafe path characters and provide ``mixture`` for an empty result."""
    component = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return component or "mixture"


def _result_directory_name(value):
    """Append an eight-character key hash to a sanitized result directory name."""
    text = str(value)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
    return f"{_safe_path_component(text)}-{digest}"


def _summary_scores(summary):
    """Convert MAPSS's diagnostic summary table to stable VERSA keys."""
    rows = summary.to_dict(orient="index")
    normalized_names = {}
    for source_name in rows:
        normalized = _safe_path_component(source_name).lower()
        if normalized in normalized_names:
            raise ValueError(
                "MAPSS source names must remain unique after normalization; "
                f"{source_name!r} conflicts with {normalized_names[normalized]!r}"
            )
        normalized_names[normalized] = source_name

    scores = {}
    for normalized, source_name in normalized_names.items():
        row = rows[source_name]
        scores[f"mapss_ps_{normalized}"] = float(row["ps"])
        scores[f"mapss_pm_{normalized}"] = float(row["pm"])
    return scores


class MapssMetric(BaseMetric):
    """Manifold-based perceptual assessment for ordered separated sources."""

    def _setup(self):
        """Validate backend availability and retain model and artifact settings.

        The backend model is invoked during compute; results are saved under
        ``cache_dir/results`` (default ``versa_cache/mapss/results``)."""
        _require_mapss()
        self.model = self.config.get("model", "wav2vec2")
        self.layer = self.config.get("layer")
        self.alpha = self.config.get("alpha", 1.0)
        self.add_ci = self.config.get("add_ci", True)
        self.seed = self.config.get("seed", 42)
        self.length_policy = self.config.get("length_policy", "error")
        self.verbose = self.config.get("verbose", False)
        self.source_names = self.config.get("source_names")
        cache_dir = Path(self.config.get("cache_dir", "versa_cache/mapss"))
        self.results_dir = cache_dir / "results"
        self.max_gpus = int(self.config.get("use_gpu", False))

    def compute(self, predictions, references=None, metadata=None):
        """Score ordered separated sources and save MAPSS diagnostic artifacts.

        Args:
            predictions: List or tuple of at least two mono source waveforms.
            references: Equally sized ordered reference list; source i must match
                prediction i. This wrapper does not search for a permutation.
            metadata: Optional ``sample_rate`` in Hz (default 16000) and mixture
                ``key`` used to derive the artifact directory.

        Returns:
            Per-source ``mapss_ps_<source>`` and ``mapss_pm_<source>`` floats from
            the backend summary, plus ``mapss_result_dir``. Values retain backend
            scaling; this wrapper does not clip or normalize them.

        Raises:
            ValueError: Source collections are invalid or normalized names collide.

        Backend model loading and result saving may download assets or write files.
        Length handling and confidence intervals follow the configured MAPSS policy."""
        if not isinstance(predictions, (list, tuple)) or len(predictions) < 2:
            raise ValueError(
                "MAPSS requires at least two ordered predicted source waveforms"
            )
        if not isinstance(references, (list, tuple)) or len(references) < 2:
            raise ValueError(
                "MAPSS requires at least two ordered reference source waveforms"
            )
        if len(predictions) != len(references):
            raise ValueError(
                "MAPSS requires the same number of ordered predicted and reference "
                "sources"
            )

        metadata = metadata or {}
        result = mapss_compute(
            reference=references,
            output=predictions,
            sample_rate=metadata.get("sample_rate", 16000),
            source_names=self.source_names,
            model=self.model,
            layer=self.layer,
            alpha=self.alpha,
            add_ci=self.add_ci,
            seed=self.seed,
            max_gpus=self.max_gpus,
            length_policy=self.length_policy,
            verbose=self.verbose,
        )

        scores = _summary_scores(result.summary)
        result_dir = self.results_dir / _result_directory_name(
            metadata.get("key", "mixture")
        )
        result.save(result_dir, plot=False)
        scores["mapss_result_dir"] = str(result_dir)
        return scores

    def get_metadata(self):
        """Return the MAPSS input requirements and backend provenance."""
        return _mapss_metadata()


def _mapss_metadata():
    """Declare ordered-reference requirements and optional MAPSS dependencies."""
    return MetricMetadata(
        name="mapss",
        category=MetricCategory.DEPENDENT,
        metric_type=MetricType.DICT,
        requires_reference=True,
        requires_text=False,
        gpu_compatible=True,
        auto_install=False,
        dependencies=["mapss"],
        description=(
            "MAPSS perceptual separation and match measures for ordered source pairs"
        ),
        paper_reference="https://arxiv.org/abs/2509.09212",
        implementation_source="https://github.com/Amir-Ivry/MAPSS-measures",
        requires_multiple_sources=True,
    )


def register_mapss_metric(registry):
    """Register the MAPSS wrapper and its public aliases in the supplied registry."""
    registry.register(
        MapssMetric,
        _mapss_metadata(),
        aliases=["MAPSS", "mapss_measures"],
    )
