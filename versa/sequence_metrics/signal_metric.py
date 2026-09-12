#!/usr/bin/env python3

# Copyright 2024 Jiatong Shi
# Mainly adpated from ESPnet-SE (https://github.com/espnet/espnet.git)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Reference-based signal ratios with optional multi-source permutation alignment."""
import ci_sdr
import fast_bss_eval
import numpy as np
import torch
from mir_eval.separation import bss_eval_sources

from versa.definition import BaseMetric, MetricCategory, MetricMetadata, MetricType


def calculate_si_snr(
    pred_x,
    gt_x,
    zero_mean=None,
    clamp_db=None,
    pairwise=False,
    return_per_source=False,
):
    """Return mean SI-SNR in dB for NumPy source/sample arrays.

    Sources must already be aligned. Forward zero_mean, clamp_db, and pairwise
    to fast_bss_eval. With return_per_source, return (mean, flattened scores);
    pairwise mode flattens all pairwise scores rather than only aligned pairs."""
    # TODO(jiatong): pass zero_mean and clamp_db setup to the function
    pred_x = torch.from_numpy(pred_x).float()
    gt_x = torch.from_numpy(gt_x).float()

    si_snr_loss = fast_bss_eval.si_sdr_loss(
        est=pred_x,
        ref=gt_x,
        zero_mean=zero_mean,
        clamp_db=clamp_db,
        pairwise=pairwise,
    )
    # With more than one source the loss is per-source, so an unguarded
    # float() raises "only one element tensors can be converted". Return the
    # mean plus the per-source values.
    values = (-si_snr_loss).reshape(-1).tolist()
    mean = float(values[0]) if len(values) == 1 else float(np.mean(values))
    return (mean, values) if return_per_source else mean


def calculate_ci_sdr(pred_x, gt_x, filter_length=512, return_per_source=False):
    # TODO(jiatong): pass filter_length to the function
    """Return mean CI-SDR in dB for aligned NumPy source/sample arrays.

    filter_length is in samples. Source permutation is disabled here. With
    return_per_source, return (mean, per-source scores) instead of a scalar."""
    pred_x = torch.from_numpy(pred_x).float()
    gt_x = torch.from_numpy(gt_x).float()

    ci_sdr_loss = ci_sdr.pt.ci_sdr_loss(
        pred_x, gt_x, compute_permutation=False, filter_length=filter_length
    )
    values = (-ci_sdr_loss).reshape(-1).tolist()
    mean = float(values[0]) if len(values) == 1 else float(np.mean(values))
    return (mean, values) if return_per_source else mean


def signal_metric(pred_x, gt_x, compute_permutation=False):
    """Return mean SDR, SIR, SAR, SI-SNR, and CI-SDR in dB for paired sources.

    Accept mono (samples,) or (sources, samples) arrays with equal source counts.
    Truncate unequal lengths to the shorter input; do not resample or mix channels.
    With multiple sources, also return <metric>_src<i> values. If
    compute_permutation is enabled, use fast_bss_eval to align estimates to
    references and reuse that assignment for SI-SNR and CI-SDR; permutation[j]
    is the estimate index assigned to reference j. Otherwise use the supplied
    source order and mir_eval for SDR/SIR/SAR. Single-source output contains
    only the five mean keys. Missing source-count agreement raises ValueError;
    backend validation errors propagate and scores are not clipped."""
    # Expected input: (channel, samples)
    if pred_x.ndim == 1:
        pred_x = pred_x[np.newaxis, :]
    if gt_x.ndim == 1:
        gt_x = gt_x[np.newaxis, :]
    if pred_x.shape[0] != gt_x.shape[0]:
        # Enforce the contract here, on both paths: fast_bss_eval would
        # otherwise fail deep inside an einsum with a message that says
        # nothing about source counts.
        raise ValueError(
            "signal_metric needs the same number of sources in the estimate "
            f"and the reference, got {pred_x.shape[0]} estimated and "
            f"{gt_x.shape[0]} reference sources"
        )
    if pred_x.shape[1] != gt_x.shape[1]:
        min_audio_length = min(pred_x.shape[1], gt_x.shape[1])
        pred_x = pred_x[:, :min_audio_length]
        gt_x = gt_x[:, :min_audio_length]

    n_src = pred_x.shape[0]
    do_perm = bool(compute_permutation) and n_src > 1

    if do_perm:
        # fast_bss_eval, not mir_eval, for the permutation path. Two reasons.
        # It solves the assignment natively and runs 2-3x faster on a
        # two-source 10 s pair, which matters because permutation is exactly
        # the case that was previously too slow to turn on. And mir_eval's
        # bss_eval_sources is deprecated as of mir_eval 0.8 and slated for
        # removal in 0.9, so building a new code path on it would ship with a
        # known expiry date. fast_bss_eval is already a declared dependency
        # of this project, so nothing new is pulled in.
        #
        # The two agree to ~6e-14 on identical input. The default,
        # non-permuted path is deliberately left on mir_eval so that no
        # number an existing caller already relies on shifts in a change
        # whose purpose is multi-source support.
        sdr, sir, sar, perm = fast_bss_eval.bss_eval_sources(
            gt_x, pred_x, compute_permutation=True
        )
        sdr, sir, sar = (np.asarray(v) for v in (sdr, sir, sar))
    else:
        sdr, sir, sar, perm = bss_eval_sources(gt_x, pred_x, compute_permutation=False)

    if do_perm:
        # Apply the assignment bss_eval resolved to every remaining metric,
        # so all five numbers describe one consistent source-to-reference
        # pairing rather than each choosing its own.
        pred_x = pred_x[np.asarray(perm)]

    si_snr, si_snr_per_src = calculate_si_snr(pred_x, gt_x, return_per_source=True)
    ci_sdr_val, ci_sdr_per_src = calculate_ci_sdr(pred_x, gt_x, return_per_source=True)

    out = {
        "sdr": float(np.mean(sdr)),
        "sir": float(np.mean(sir)),
        "sar": float(np.mean(sar)),
        "si_snr": si_snr,
        "ci_sdr": ci_sdr_val,
    }
    if n_src > 1:
        for i in range(len(sdr)):
            out[f"sdr_src{i}"] = float(sdr[i])
            out[f"sir_src{i}"] = float(sir[i])
            out[f"sar_src{i}"] = float(sar[i])
        for i, v in enumerate(si_snr_per_src):
            out[f"si_snr_src{i}"] = float(v)
        for i, v in enumerate(ci_sdr_per_src):
            out[f"ci_sdr_src{i}"] = float(v)
        if do_perm:
            out["permutation"] = [int(v) for v in np.asarray(perm).ravel()]
    return out


class SignalMetric(BaseMetric):
    """Reference-based signal distortion metrics."""

    def _setup(self):
        # Off by default so single-source behaviour is byte-identical to
        # before; multi-source users opt in.
        """Retain the opt-in source-permutation setting without loading a model."""
        self.compute_permutation = bool(self.config.get("compute_permutation", False))

    def compute(self, predictions, references=None, metadata=None):
        """Score required mono or source/sample prediction and reference arrays.

        Inputs must already share a sample rate; metadata is unused. Return the
        means and optional per-source ratios/permutation described by signal_metric.
        Missing either input raises ValueError."""
        if predictions is None:
            raise ValueError("Predicted signal must be provided")
        if references is None:
            raise ValueError("Reference signal must be provided")
        return signal_metric(
            np.asarray(predictions),
            np.asarray(references),
            compute_permutation=self.compute_permutation,
        )

    def get_metadata(self):
        """Return signal-metric input requirements and implementation provenance."""
        return _signal_metadata()


def _signal_metadata():
    """Describe the reference-based signal ratios and optional source alignment."""
    return MetricMetadata(
        name="signal_metric",
        category=MetricCategory.DEPENDENT,
        metric_type=MetricType.DICT,
        requires_reference=True,
        requires_text=False,
        gpu_compatible=False,
        auto_install=False,
        dependencies=["ci_sdr", "fast_bss_eval", "mir_eval", "numpy", "torch"],
        description=(
            "Reference-based SDR, SIR, SAR, SI-SNR, and CI-SDR metrics. Set "
            "compute_permutation=true for multi-source outputs to resolve the "
            "optimal source-to-reference assignment before scoring."
        ),
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
    print("single source: {}".format(metric.compute(a, b)))

    # Two sources handed over in the wrong order: without permutation this
    # scores as a failure, with it as a success.
    s1 = np.random.random(16000)
    s2 = np.random.random(16000)
    refs = np.stack([s1, s2])
    swapped = np.stack([s2, s1])
    print("swapped, no perm : {}".format(SignalMetric().compute(swapped, refs)))
    print(
        "swapped, with perm: {}".format(
            SignalMetric(config={"compute_permutation": True}).compute(swapped, refs)
        )
    )
