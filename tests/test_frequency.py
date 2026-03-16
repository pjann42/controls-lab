import numpy as np
import pytest
import control as ctrl
from core.frequency import (
    compute_frequency_response,
    compute_margins,
    smart_autoscale,
    align_phase_axis_45_deg,
)


# --- compute_frequency_response ---

def test_frequency_response_shape():
    G = ctrl.TransferFunction([1], [1, 1])
    w, mag, phase = compute_frequency_response(G)
    assert w.shape == mag.shape == phase.shape
    assert len(w) == 1000


def test_frequency_response_dc_gain():
    # G(s) = 1/(s+1) → |G(0)| = 1 → 0 dB
    G = ctrl.TransferFunction([1], [1, 1])
    w = np.array([1e-4])
    _, mag, _ = compute_frequency_response(G, w)
    assert abs(mag[0]) < 0.1  # close to 0 dB at very low frequency


def test_frequency_response_custom_w():
    G = ctrl.TransferFunction([1], [1, 2, 1])
    w = np.logspace(-1, 1, 50)
    w_out, mag, phase = compute_frequency_response(G, w)
    assert len(w_out) == 50


# --- compute_margins ---

def test_margins_stable_system():
    G = ctrl.TransferFunction([1], [1, 2, 1])
    gm, pm, wg, wp = compute_margins(G)
    # For a stable system with enough margin, pm should be positive
    assert np.isnan(gm) or gm > 0
    assert np.isnan(pm) or pm > 0


def test_margins_pure_integrator():
    # 1/s has infinite gain margin
    G = ctrl.TransferFunction([1], [1, 0])
    gm, pm, wg, wp = compute_margins(G)
    # Infinite gm → should be np.nan after sanitize
    assert np.isnan(gm)


# --- smart_autoscale ---

def test_small_range_adds_fixed_padding():
    y = np.array([0.0, 1.0, 2.0])  # range = 2 < 5
    y_min, y_max = smart_autoscale(y)
    assert y_min < 0
    assert y_max > 2


def test_normal_range_adds_fractional_padding():
    y = np.linspace(0, 100, 1000)  # range = 100 ≥ 5
    y_min, y_max = smart_autoscale(y, padding_factor=0.1)
    assert y_min < 0
    assert y_max > 100


def test_steady_state_uses_symmetric_range():
    # Mostly flat → steady-state branch
    y = np.concatenate([np.linspace(0, 10, 10), np.full(990, 10.0)])
    y_min, y_max = smart_autoscale(y)
    center = (y_min + y_max) / 2
    assert abs(center - 5.0) < 3.0  # roughly centered


# --- align_phase_axis_45_deg ---

def test_tick_values_are_multiples_of_45():
    _, _, ticks = align_phase_axis_45_deg(-180, 0)
    for tick in ticks:
        assert tick % 45 == 0


def test_aligned_min_lte_input_min():
    aligned_min, _, _ = align_phase_axis_45_deg(-91, 0)
    assert aligned_min <= -91


def test_aligned_max_gte_input_max():
    _, aligned_max, _ = align_phase_axis_45_deg(-180, 10)
    assert aligned_max >= 10
