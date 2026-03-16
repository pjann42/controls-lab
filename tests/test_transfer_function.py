import numpy as np
import pytest
from core.transfer_function import validate_and_create_system


def test_empty_numerator_returns_invalid():
    result = validate_and_create_system([], [1, 1])
    assert result["valid"] is False
    assert "Empty" in result["message"]


def test_empty_denominator_returns_invalid():
    result = validate_and_create_system([1], [])
    assert result["valid"] is False


def test_improper_system_detected():
    # num degree 2 > den degree 1 → improper
    result = validate_and_create_system([1, 0, 0], [1, 1])
    assert result["valid"] is False
    assert result.get("improper_system") is True
    assert result["num_degree"] == 2
    assert result["den_degree"] == 1


def test_valid_first_order_system():
    result = validate_and_create_system([1], [1, 1])
    assert result["valid"] is True
    assert result["system"] is not None
    assert result["num_degree"] == 0
    assert result["den_degree"] == 1


def test_valid_second_order_system():
    result = validate_and_create_system([1], [1, 2, 1])
    assert result["valid"] is True
    assert len(result["poles"]) == 2


def test_identical_coefficients_flagged():
    result = validate_and_create_system([1, 1], [1, 1])
    assert result["valid"] is True
    assert result["identical_coeffs"] is True


def test_leading_zeros_stripped():
    # [0, 1] should be treated as degree-0 numerator
    result = validate_and_create_system([0, 1], [1, 2, 1])
    assert result["valid"] is True
    assert result["num_degree"] == 0


def test_biproper_system_valid():
    # num and den same degree → proper (biproper)
    result = validate_and_create_system([2, 1], [1, 3])
    assert result["valid"] is True


def test_invalid_coefficients_string_raises_gracefully():
    # Should not propagate an exception
    result = validate_and_create_system(["a", "b"], [1, 1])
    assert result["valid"] is False
