#!/usr/bin/env python3

# Copyright 2024 Jiatong Shi
# Mainly adpated from ESPnet-SE (https://github.com/espnet/espnet.git)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Reference-based SDR, SIR, SAR, SI-SNR, and CI-SDR scoring."""
import ci_sdr
import fast_bss_eval
import numpy as np
import torch
from mir_eval.separation import bss_eval_sources

from versa.definition import BaseMetric, MetricCategory, MetricMetadata, MetricType


def calculate_si_snr(pred_x, gt_x, zero_mean=None, clamp_db=None, pairwise=False):
    # TODO(jiatong): pass zero_mean and clamp_db setup to the function
    """Return scalar SI-SNR in dB by negating fast_bss_eval loss.

    Inputs are NumPy source/sample arrays yielding a single loss value.
    Forward zero_mean, clamp_db, and pairwise to the backend; a multi-element
    loss cannot be converted to the scalar returned by this implementation."""
    pred_x = torch.from_numpy(pred_x).float()
    gt_x = torch.from_numpy(gt_x).float()

    si_snr_loss = fast_bss_eval.si_sdr_loss(
        est=pred_x,
        ref=gt_x,
        zero_mean=zero_mean,
        clamp_db=clamp_db,
        pairwise=pairwise,
    )
    return -float(si_snr_loss)


def calculate_ci_sdr(pred_x, gt_x, filter_length=512):
    # TODO(jiatong): pass filter_length to the function
    """Return scalar CI-SDR in dB with the given filter length in samples.

    Source/sample NumPy arrays must yield one loss value. Source permutation
    is disabled; multi-element backend losses cannot be converted to float."""
    pred_x = torch.from_numpy(pred_x).float()
    gt_x = torch.from_numpy(gt_x).float()

    ci_sdr_loss = ci_sdr.pt.ci_sdr_loss(
        pred_x, gt_x, compute_permutation=False, filter_length=filter_length
    )
    return -float(ci_sdr_loss)


def signal_metric(pred_x, gt_x):
    # Expected input: (channel, samples)
    """Return sdr, sir, sar, si_snr, and ci_sdr in dB for a single-source pair.

    Accept mono ``(samples,)`` or ``(1, samples)`` arrays and truncate unequal
    lengths to the shorter input. No resampling, channel mixing, or source
    permutation is performed. Although 2D input is accepted, multiple sources
    are unsupported by the scalar SI-SNR/CI-SDR helpers. Scores are not clipped."""
    if pred_x.ndim == 1:
        pred_x = pred_x[np.newaxis, :]
    if gt_x.ndim == 1:
        gt_x = gt_x[np.newaxis, :]
    if pred_x.shape[1] != gt_x.shape[1]:
        min_audio_length = min(pred_x.shape[1], gt_x.shape[1])
        pred_x = pred_x[:, :min_audio_length]
        gt_x = gt_x[:, :min_audio_length]
    sdr, sir, sar, _ = bss_eval_sources(gt_x, pred_x, compute_permutation=False)
    si_snr = calculate_si_snr(pred_x, gt_x)
    ci_sdr = calculate_ci_sdr(pred_x, gt_x)
    return {
        "sdr": sdr[0],
        "sir": sir[0],
        "sar": sar[0],
        "si_snr": si_snr,
        "ci_sdr": ci_sdr,
    }


class SignalMetric(BaseMetric):
    """Reference-based signal distortion metrics."""

    def _setup(self):
        """Use the stateless signal-metric functions without model initialization."""
        pass

    def compute(self, predictions, references=None, metadata=None):
        """Score required single-source prediction/reference arrays with ``signal_metric``.

        Inputs must already share a sample rate; metadata is unused. Return five
        signal ratios in dB. Missing either array raises ValueError; backend
        validation errors propagate."""
        if predictions is None:
            raise ValueError("Predicted signal must be provided")
        if references is None:
            raise ValueError("Reference signal must be provided")
        return signal_metric(np.asarray(predictions), np.asarray(references))

    def get_metadata(self):
        """Return input requirements and provenance for this metric configuration."""
        return _signal_metadata()


def _signal_metadata():
    """Return registry metadata describing signal metric inputs and dependencies."""
    return MetricMetadata(
        name="signal_metric",
        category=MetricCategory.DEPENDENT,
        metric_type=MetricType.DICT,
        requires_reference=True,
        requires_text=False,
        gpu_compatible=False,
        auto_install=False,
        dependencies=["ci_sdr", "fast_bss_eval", "mir_eval", "numpy", "torch"],
        description="Reference-based SDR, SIR, SAR, SI-SNR, and CI-SDR metrics",
        implementation_source="https://github.com/espnet/espnet",
    )


def register_signal_metric(registry):
    """Register signal distortion metrics with the registry."""
    registry.register(
        SignalMetric,
        _signal_metadata(),
        aliases=["signal", "snr_related"],
    )


# debug code
if __name__ == "__main__":
    a = np.random.random(16000)
    b = np.random.random(16000)
    metric = SignalMetric()
    print("metrics: {}".format(metric.compute(a, b)))
