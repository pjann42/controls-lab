import numpy as np
import pytest
from core.stability import classify_stability


def test_empty_poles_returns_undefined():
    stability, _, _ = classify_stability(np.array([]))
    assert stability == "Undefined"


def test_all_left_poles_asymptotically_stable():
    poles = np.array([-1.0, -2.0, -0.5 + 1j, -0.5 - 1j])
    stability, _, _ = classify_stability(poles)
    assert stability == "Asymptotically Stable"


def test_right_half_plane_pole_unstable():
    poles = np.array([-1.0, 1.0])
    stability, _, _ = classify_stability(poles)
    assert stability == "Unstable"


def test_repeated_imaginary_poles_unstable():
    poles = np.array([1j, 1j, -1.0])
    stability, desc, _ = classify_stability(poles)
    assert stability == "Unstable"
    assert "repeated" in desc.lower()


def test_multiple_origin_poles_unstable():
    poles = np.array([0.0, 0.0, -1.0])
    stability, _, _ = classify_stability(poles)
    assert stability == "Unstable"


def test_single_origin_pole_marginally_stable():
    poles = np.array([0.0, -1.0, -2.0])
    stability, desc, _ = classify_stability(poles)
    assert stability == "Marginally Stable"
    assert "origin" in desc.lower()


def test_simple_imaginary_poles_marginally_stable():
    poles = np.array([1j, -1j])
    stability, _, _ = classify_stability(poles)
    assert stability == "Marginally Stable"


def test_purely_stable_complex_poles():
    poles = np.array([-1 + 2j, -1 - 2j])
    stability, _, _ = classify_stability(poles)
    assert stability == "Asymptotically Stable"
