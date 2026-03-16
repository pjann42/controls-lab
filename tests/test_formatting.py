import pytest
from core.formatting import format_polynomial, format_metric, clean_coefficients


# --- format_polynomial ---

def test_empty_coefficients():
    assert format_polynomial([]) == "0"


def test_single_constant():
    assert format_polynomial([5]) == "5"


def test_first_order():
    assert format_polynomial([1, 0]) == "s"


def test_first_order_with_constant():
    result = format_polynomial([1, 1])
    assert "s" in result
    assert "1" in result


def test_negative_leading_coefficient():
    result = format_polynomial([-1, 0])
    assert result.startswith("-")
    assert "s" in result


def test_negative_intermediate_term():
    # x^2 - x + 1
    result = format_polynomial([1, -1, 1])
    assert " - " in result
    # Must NOT contain "+ -"
    assert "+ -" not in result


def test_unity_coefficient_omitted_for_s():
    # coefficient 1 on s^1 should render as "s" not "1s"
    result = format_polynomial([1, 0, 0])
    assert "1s" not in result
    assert "s^{2}" in result


def test_negative_one_coefficient_omitted():
    # coefficient -1 on s^1 should render as "-s" not "-1s"
    result = format_polynomial([-1, 0])
    assert "-1s" not in result
    assert result == "-s"


def test_zero_coefficients_skipped():
    result = format_polynomial([1, 0, 1])
    # Middle zero should not appear as a term
    assert "0s" not in result
    parts = result.split()
    # Should be "s^{2} + 1" — 3 tokens
    assert len(parts) == 3


def test_all_zero_coefficients():
    assert format_polynomial([0, 0, 0]) == "0"


# --- format_metric ---

def test_format_metric_normal():
    assert format_metric(1.23456) == "1.235"


def test_format_metric_integer():
    assert format_metric(2) == "2.000"


def test_format_metric_none_returns_fallback():
    assert format_metric(None) == "N/A"


def test_format_metric_string_na_returns_fallback():
    assert format_metric("N/A") == "N/A"


def test_format_metric_custom_format():
    assert format_metric(3.14159, fmt=".1f") == "3.1"


def test_format_metric_custom_fallback():
    assert format_metric(None, fallback="—") == "—"


# --- clean_coefficients ---

def test_clean_removes_near_zero_noise():
    noisy = [1.0, 2e-15, 1.0]
    cleaned = clean_coefficients(noisy)
    assert cleaned[1] == 0.0


def test_clean_preserves_real_coefficients():
    coeffs = [1.0, -2.0, 1.0]
    cleaned = clean_coefficients(coeffs)
    assert cleaned == [1.0, -2.0, 1.0]


def test_clean_empty_list():
    assert clean_coefficients([]) == []


def test_clean_all_zeros():
    assert clean_coefficients([0.0, 0.0]) == [0.0, 0.0]


def test_clean_small_but_significant_coefficients():
    # 1e-6 relative to 1.0 is above default tol (1e-9) — must be preserved
    coeffs = [1.0, 1e-6, 1.0]
    cleaned = clean_coefficients(coeffs)
    assert cleaned[1] != 0.0


# --- scientific notation never in LaTeX output ---

def test_no_scientific_notation_in_polynomial():
    noisy = clean_coefficients([1.0, 2e-15, 1.0])
    result = format_polynomial(noisy)
    assert "e" not in result
    assert "E" not in result


def test_large_coefficient_no_scientific_notation():
    result = format_polynomial([100000.0, 1.0])
    assert "e" not in result
    assert "E" not in result
