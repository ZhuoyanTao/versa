#!/usr/bin/env python3

# Copyright 2024 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Per-file Frechet audio distances against a reference collection."""
import logging
from pathlib import Path

import numpy as np
from tqdm import tqdm

from versa.definition import BaseMetric, MetricCategory, MetricMetadata, MetricType
from versa.scorer_shared import audio_loader_setup

try:
    from fadtk.fad_versa import (
        FrechetAudioDistance,
        calc_embd_statistics,
        calc_frechet_distance,
    )
    from fadtk.model_loader import get_model
except ImportError:
    FrechetAudioDistance = None
    calc_embd_statistics = None
    calc_frechet_distance = None
    get_model = None
    logging.warning(
        "FADTK is not installed. Please install it following `tools/install_fadtk.sh`"
    )


def _load_audio_collection(audio, io):
    """Use an existing keyed audio mapping or load one with the selected I/O mode."""
    if isinstance(audio, dict):
        return audio
    return audio_loader_setup(audio, io)


def _individual_fad_metadata():
    """Return registry metadata describing individual fad inputs and dependencies."""
    return MetricMetadata(
        name="individual_fad",
        category=MetricCategory.DISTRIBUTIONAL,
        metric_type=MetricType.DICT,
        requires_reference=True,
        requires_text=False,
        gpu_compatible=True,
        auto_install=False,
        dependencies=["fadtk"],
        description=(
            "Frechet Audio Distance from each generated audio file to a "
            "reference audio distribution"
        ),
        paper_reference="https://arxiv.org/abs/1812.08466",
        implementation_source="https://github.com/ftshijt/fadtk",
    )


class IndividualFadMetric(BaseMetric):
    """Per-file Frechet Audio Distance against a reference collection."""

    def _setup(self):
        """Require FADTK, load the configured embedding model, and retain cache/I/O settings."""
        if (
            get_model is None
            or FrechetAudioDistance is None
            or calc_embd_statistics is None
            or calc_frechet_distance is None
        ):
            raise ModuleNotFoundError(
                "FADTK is not installed. Please install it following `tools/install_fadtk.sh`"
            )

        self.cache_dir = self.config.get("cache_dir", "versa_cache/individual_fad")
        self.io = self.config.get("io", "kaldi")
        self.baseline = self.config.get("baseline")
        self.embedding = self.config.get("fad_embedding", "default")
        self.module = FrechetAudioDistance(
            ml=get_model(self.embedding),
            load_model=True,
        )

    def compute(self, predictions, references=None, metadata=None):
        """Compare prediction and reference audio collections through cached embeddings.

        Inputs are keyed mappings or paths interpreted by the configured io mode.
        Use references, then metadata baseline_files, then the configured baseline.
        Sample rates and channel processing are delegated to FADTK audio loading.
        Cache embeddings in baseline/eval subdirectories under cache_dir.

        Return ``individual_fad`` mapping each prediction key to its distance
        from the pooled reference embedding distribution.
        Distance values retain backend scaling without clipping. Missing prediction
        or baseline inputs raise ValueError; backend loading/statistics errors propagate.
        """
        if predictions is None:
            raise ValueError("Individual FAD requires prediction audio files")

        metadata = metadata or {}
        baseline = references or metadata.get("baseline_files") or self.baseline
        if baseline is None:
            raise ValueError(
                "Individual FAD requires reference or baseline audio files"
            )

        baseline_files = _load_audio_collection(baseline, self.io)
        eval_files = _load_audio_collection(predictions, self.io)

        baseline_cache = Path(self.cache_dir) / "baseline"
        eval_cache = Path(self.cache_dir) / "eval"

        logging.info("[Individual FAD] caching baseline embeddings...")
        for key in tqdm(baseline_files.keys()):
            self.module.cache_embedding_file(key, baseline_files[key], baseline_cache)
        logging.info("[Individual FAD] Finished caching baseline embeddings.")

        logging.info("[Individual FAD] caching eval embeddings...")
        for key in tqdm(eval_files.keys()):
            self.module.cache_embedding_file(key, eval_files[key], eval_cache)
        logging.info("[Individual FAD] Finished caching eval embeddings.")

        baseline_embeddings = [
            self.module.read_embedding_file(key, baseline_cache)
            for key in baseline_files.keys()
        ]
        mu_base, cov_base = calc_embd_statistics(
            np.concatenate(baseline_embeddings, axis=0)
        )

        scores = {}
        for key in tqdm(eval_files.keys(), desc="Calculating individual FAD"):
            embd = self.module.read_embedding_file(key, eval_cache)
            mu_eval, cov_eval = calc_embd_statistics(embd)
            scores[key] = float(
                calc_frechet_distance(mu_base, cov_base, mu_eval, cov_eval)
            )

        return {"individual_fad": scores}

    def get_metadata(self):
        """Return input requirements and provenance for this metric configuration."""
        return _individual_fad_metadata()


def register_individual_fad_metric(registry):
    """Register individual Frechet Audio Distance with the registry."""
    registry.register(
        IndividualFadMetric,
        _individual_fad_metadata(),
        aliases=["per_file_fad", "individual_fad_metric"],
    )
