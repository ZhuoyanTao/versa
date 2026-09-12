#!/usr/bin/env python3

# Copyright 2024 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""TorchAudio SQUIM objective and subjective speech quality models."""
import logging

import numpy as np
import torch

try:
    import torchaudio.functional as F
    from torchaudio.pipelines import SQUIM_OBJECTIVE, SQUIM_SUBJECTIVE
except ImportError:
    logging.warning(
        "Import error. Please install pesq, pystoi, torchaudio for torch squim"
    )
    F = None
    SQUIM_OBJECTIVE = None
    SQUIM_SUBJECTIVE = None

from versa.metric_metadata import _squim_metadata
from versa.definition import BaseMetric

SQUIM_AVAILABLE = SQUIM_OBJECTIVE is not None and SQUIM_SUBJECTIVE is not None


def is_squim_available():
    """Return whether SQUIM and its required metric dependencies were imported."""
    return SQUIM_AVAILABLE


def squim_metric(pred_x, gt_x, fs):
    """
    Reference:
    Kumar et al., "TorchAudio-Squim: Reference-less Speech Quality and
    Intelligibility measures in TorchAudio", ICASSP 2023.
    https://pytorch.org/audio/main/tutorials/squim_tutorial.html

    """
    gt_x = torch.from_numpy(gt_x)
    pred_x = torch.from_numpy(pred_x)

    if fs != 16000:
        gt_x = F.resample(gt_x, fs, 16000)
        pred_x = F.resample(pred_x, fs, 16000)

    gt_x = gt_x.unsqueeze(0).float()
    pred_x = pred_x.unsqueeze(0).float()

    subjective_model = SQUIM_SUBJECTIVE.get_model()
    torch_squim_mos = subjective_model(pred_x, gt_x)

    return {"torch_squim_mos": torch_squim_mos.detach().numpy()[0]}


def squim_metric_no_ref(pred_x, fs):
    """
    Reference:
    Kumar et al., "TorchAudio-Squim: Reference-less Speech Quality and
    Intelligibility measures in TorchAudio", ICASSP 2023.
    https://pytorch.org/audio/main/tutorials/squim_tutorial.html

    """
    pred_x = torch.from_numpy(pred_x)
    if fs != 16000:
        pred_x = F.resample(pred_x, fs, 16000)

    pred_x = pred_x.unsqueeze(0).float()

    objective_model = SQUIM_OBJECTIVE.get_model()
    torch_squim_stoi, torch_squim_pesq, torch_squim_si_sdr = objective_model(pred_x)

    return {
        "torch_squim_stoi": torch_squim_stoi.detach().numpy()[0],
        "torch_squim_pesq": torch_squim_pesq.detach().numpy()[0],
        "torch_squim_si_sdr": torch_squim_si_sdr.detach().numpy()[0],
    }


class SquimMetric(BaseMetric):
    """TorchAudio-SQUIM speech quality metric."""

    def _setup(self):
        """Validate mode, set the global Torch Hub cache, and load a SQUIM model."""
        if not SQUIM_AVAILABLE:
            raise ImportError(
                "SQUIM is not available. Please install pesq, pystoi, and torchaudio"
            )
        self.mode = self.config.get("mode", "no_ref")
        self.cache_dir = self.config.get("cache_dir", "versa_cache/torch")
        torch.hub.set_dir(self.cache_dir)
        if self.mode not in {"ref", "no_ref"}:
            raise ValueError(f"Invalid SQUIM mode: {self.mode}")
        if self.mode == "ref":
            self.model = SQUIM_SUBJECTIVE.get_model()
        else:
            self.model = SQUIM_OBJECTIVE.get_model()

    def compute(self, predictions, references=None, metadata=None):
        """Predict SQUIM speech quality from mono waveform arrays.

        Read sample_rate in Hz from metadata (default 16000), resample to 16 kHz,
        and add a batch axis without mixing channels. Ref mode requires a second
        waveform and returns torch_squim_mos; no_ref mode returns torch_squim_stoi,
        torch_squim_pesq, and torch_squim_si_sdr. Missing required audio raises
        ValueError. Estimates retain backend scaling without clipping."""
        if predictions is None:
            raise ValueError("Predicted signal must be provided")
        if self.mode == "ref" and references is None:
            raise ValueError("Reference signal must be provided for SQUIM ref mode")

        fs = metadata.get("sample_rate", 16000) if metadata else 16000
        pred_x = torch.from_numpy(np.asarray(predictions))
        if fs != 16000:
            pred_x = F.resample(pred_x, fs, 16000)
        pred_x = pred_x.unsqueeze(0).float()

        if self.mode == "ref":
            gt_x = torch.from_numpy(np.asarray(references))
            if fs != 16000:
                gt_x = F.resample(gt_x, fs, 16000)
            gt_x = gt_x.unsqueeze(0).float()
            torch_squim_mos = self.model(pred_x, gt_x)
            return {"torch_squim_mos": torch_squim_mos.detach().numpy()[0]}

        torch_squim_stoi, torch_squim_pesq, torch_squim_si_sdr = self.model(pred_x)
        return {
            "torch_squim_stoi": torch_squim_stoi.detach().numpy()[0],
            "torch_squim_pesq": torch_squim_pesq.detach().numpy()[0],
            "torch_squim_si_sdr": torch_squim_si_sdr.detach().numpy()[0],
        }

    def get_metadata(self):
        """Return input requirements and provenance for this metric configuration."""
        return _squim_metadata(f"squim_{self.mode}", self.mode)


class SquimRefMetric(SquimMetric):
    """Reference-based TorchAudio-SQUIM MOS metric."""

    def _setup(self):
        """Default to subjective reference mode before loading the SQUIM model."""
        self.config = {**self.config, "mode": self.config.get("mode", "ref")}
        super()._setup()


class SquimNoRefMetric(SquimMetric):
    """Reference-less TorchAudio-SQUIM objective metrics."""

    def _setup(self):
        """Default to objective reference-free mode before loading the SQUIM model."""
        self.config = {**self.config, "mode": self.config.get("mode", "no_ref")}
        super()._setup()


def register_squim_metric(registry):
    """Register TorchAudio-SQUIM metrics with the registry."""
    registry.register(
        SquimRefMetric,
        _squim_metadata("squim_ref", "ref"),
        aliases=["torch_squim_mos"],
    )
    registry.register(
        SquimNoRefMetric,
        _squim_metadata("squim_no_ref", "no_ref"),
        aliases=["squim", "torch_squim_objective"],
    )


if __name__ == "__main__":
    a = np.random.random(16000)
    b = np.random.random(16000)
    metric = SquimRefMetric()
    scores = metric.compute(a, b, metadata={"sample_rate": 16000})
    print(scores)
