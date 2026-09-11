#!/usr/bin/env python3

# Copyright 2026 Zhuoyan Tao
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Tests for multi-source and permutation-invariant signal metrics."""

import numpy as np
import pytest

from versa.sequence_metrics.signal_metric import (
    SignalMetric,
    calculate_ci_sdr,
    calculate_si_snr,
    signal_metric,
)

# ---------------------------------------------------------------------------
# backward compatibility
# ---------------------------------------------------------------------------


def test_single_source_helpers_still_return_float():
    """The per-source return is opt-in, so existing callers are unaffected."""
    rng = np.random.default_rng(0)
    a = rng.random(16000).astype("float32")
    b = rng.random(16000).astype("float32")
    assert isinstance(calculate_si_snr(a, b), float)
    assert isinstance(calculate_ci_sdr(a, b), float)
    mean, per_src = calculate_si_snr(a, b, return_per_source=True)
    assert isinstance(mean, float) and len(per_src) == 1


def test_single_source_keys_unchanged():
    """A single-source call returns exactly the original five keys."""
    rng = np.random.default_rng(1)
    a = rng.random(8000).astype("float32")
    b = rng.random(8000).astype("float32")
    out = signal_metric(a, b)
    assert set(out) == {"sdr", "sir", "sar", "si_snr", "ci_sdr"}


def test_permutation_off_by_default():
    """Default behaviour must not change for anyone."""
    metric = SignalMetric()
    assert metric.compute_permutation is False


# ---------------------------------------------------------------------------
# multi-source support
# ---------------------------------------------------------------------------


def test_multi_source_does_not_raise():
    """Two sources used to raise: float() on a multi-element tensor."""
    rng = np.random.default_rng(2)
    refs = rng.random((2, 8000)).astype("float32")
    preds = rng.random((2, 8000)).astype("float32")
    out = signal_metric(preds, refs)
    assert "sdr_src0" in out and "sdr_src1" in out
    assert "si_snr_src0" in out and "ci_sdr_src1" in out


def test_permutation_recovers_swapped_sources():
    """A correct separation emitted in the wrong order is a success, not a
    failure. Without permutation it scores as though it had failed."""
    rng = np.random.default_rng(3)
    s1 = rng.standard_normal(8000).astype("float32")
    s2 = rng.standard_normal(8000).astype("float32")
    refs = np.stack([s1, s2])
    swapped = np.stack([s2, s1])

    without = signal_metric(swapped, refs, compute_permutation=False)
    with_perm = signal_metric(swapped, refs, compute_permutation=True)

    # the swapped estimate IS the reference set, so resolving the assignment
    # should improve SDR by a wide margin
    assert with_perm["sdr"] > without["sdr"] + 20.0
    assert with_perm["permutation"] in ([1, 0], [1, 0])


def test_permutation_noop_when_already_aligned():
    """Correctly ordered sources must not be made worse by enabling PIT.

    The two paths use different bss_eval backends -- mir_eval when no
    permutation is requested, fast_bss_eval when one is -- and they solve for
    the 512-tap distortion filter by different linear algebra. On a pair this
    close (~40 dB) that estimation is ill-conditioned, so the last digits
    genuinely differ: 40.2755 against 40.2803 for this seed. The tolerance is
    therefore 0.05 dB rather than the 1e-6 a single shared backend allowed,
    still orders of magnitude tighter than any difference a wrong permutation
    would produce.
    """
    rng = np.random.default_rng(4)
    refs = rng.standard_normal((2, 8000)).astype("float32")
    preds = refs + 0.01 * rng.standard_normal((2, 8000)).astype("float32")
    without = signal_metric(preds, refs, compute_permutation=False)
    with_perm = signal_metric(preds, refs, compute_permutation=True)
    assert with_perm["sdr"] >= without["sdr"] - 0.05
    assert with_perm["permutation"] == [0, 1]


def test_permutation_ignored_for_single_source():
    """No permutation exists for one source; the flag must be harmless."""
    rng = np.random.default_rng(5)
    a = rng.random(8000).astype("float32")
    b = rng.random(8000).astype("float32")
    out = signal_metric(a, b, compute_permutation=True)
    assert "permutation" not in out


def test_permutation_path_agrees_with_mir_eval():
    """The PIT path uses fast_bss_eval; it must not silently disagree.

    Swapping the backend for the permutation path is only defensible if it
    reproduces the reference implementation, so this pins the agreement
    rather than trusting it. mir_eval is asked for the same assignment
    explicitly, and the two SDR sets are compared as sorted multisets since
    each reports in its own source order.
    """
    from mir_eval.separation import bss_eval_sources as mir_bss

    rng = np.random.default_rng(7)
    src = rng.standard_normal((2, 16000))
    mix = np.array([[0.9, 0.3], [0.25, 0.95]]) @ src
    swapped = mix[::-1]

    ours = signal_metric(swapped, src, compute_permutation=True)
    mir_sdr, _, _, mir_perm = mir_bss(src, swapped, compute_permutation=True)

    assert ours["permutation"] == [int(v) for v in np.asarray(mir_perm).ravel()]
    got = sorted(ours[f"sdr_src{i}"] for i in range(2))
    want = sorted(float(v) for v in mir_sdr)
    assert np.allclose(got, want, atol=1e-6), (got, want)


def test_default_path_untouched_by_the_backend_swap():
    """compute_permutation=False must still go through mir_eval unchanged."""
    from mir_eval.separation import bss_eval_sources as mir_bss

    rng = np.random.default_rng(8)
    src = rng.standard_normal((2, 16000))
    est = src + 0.05 * rng.standard_normal((2, 16000))

    out = signal_metric(est, src, compute_permutation=False)
    mir_sdr, _, _, _ = mir_bss(src, est, compute_permutation=False)
    assert out["sdr"] == pytest.approx(float(np.mean(mir_sdr)), abs=1e-9)


def test_permutation_three_source_cyclic_shift():
    """Two sources cannot tell a permutation from its inverse; three can.

    With references [s1, s2, s3] and estimates [s2, s3, s1], the assignment
    that aligns the estimate is perm = [2, 0, 1] (estimate perm[j] belongs to
    reference j). Applying the inverse instead would leave every source
    mismatched, so this pins the convention the backend returns.
    """
    rng = np.random.RandomState(0)
    T = 16000

    def narrowband(freq):
        kernel = np.hanning(64) * np.cos(2 * np.pi * freq * np.arange(64) / 16000)
        return np.convolve(rng.randn(T), kernel, "same") / 10

    refs = np.stack([narrowband(500), narrowband(2000), narrowband(4000)])
    est = np.stack([refs[1], refs[2], refs[0]]) + 0.01 * rng.randn(3, T)

    out = signal_metric(est, refs, compute_permutation=True)
    assert out["permutation"] == [2, 0, 1]
    for i in range(3):
        assert out[f"si_snr_src{i}"] > 20.0
        assert out[f"sdr_src{i}"] > 20.0
    # and without the flag the same input scores as a failure
    assert signal_metric(est, refs)["si_snr"] < 0.0


@pytest.mark.parametrize("compute_permutation", [False, True])
@pytest.mark.parametrize("n_est, n_ref", [(2, 3), (3, 2), (1, 2), (2, 1)])
def test_mismatched_source_count_raises_clearly(compute_permutation, n_est, n_ref):
    """The contract is enforced before either backend sees the arrays, so the
    error is the same clear ValueError with or without permutation instead
    of an einsum failure from fast_bss_eval."""
    rng = np.random.RandomState(0)
    est = rng.random((n_est, 8000))
    ref = rng.random((n_ref, 8000))
    with pytest.raises(ValueError, match=rf"{n_est} estimated and {n_ref} reference"):
        signal_metric(est, ref, compute_permutation=compute_permutation)
