import numpy as np
import control as ctrl


def validate_and_create_system(num_coeffs, den_coeffs):
    """
    Validate transfer function realizability and create system if valid.

    Args:
        num_coeffs: Numerator coefficients (list or array)
        den_coeffs: Denominator coefficients (list or array)

    Returns:
        dict with keys: valid, message, system, and optional metadata
    """
    if not num_coeffs or not den_coeffs:
        return {"valid": False, "message": "Empty coefficient arrays provided", "system": None}

    try:
        num_array = np.array(num_coeffs, dtype=float)
        den_array = np.array(den_coeffs, dtype=float)

        num_degree = len(num_array) - 1
        den_degree = len(den_array) - 1

        # Strip leading zeros for accurate degree calculation
        while len(num_array) > 1 and abs(num_array[0]) < 1e-10:
            num_array = num_array[1:]
            num_degree -= 1

        while len(den_array) > 1 and abs(den_array[0]) < 1e-10:
            den_array = den_array[1:]
            den_degree -= 1

        # Causal (proper) system check: numerator degree <= denominator degree
        if num_degree > den_degree:
            return {
                "valid": False,
                "message": "The system is non-realizable and has no physical meaning (Improper System)",
                "system": None,
                "num_degree": num_degree,
                "den_degree": den_degree,
                "improper_system": True,
            }

        G = ctrl.TransferFunction(num_array, den_array)

        return {
            "valid": True,
            "message": "System is realizable and physically meaningful",
            "system": G,
            "num_degree": num_degree,
            "den_degree": den_degree,
            "poles": ctrl.pole(G),
            "zeros": ctrl.zero(G),
            "identical_coeffs": np.array_equal(num_array, den_array),
        }

    except Exception as e:
        return {"valid": False, "message": f"Error creating system: {e}", "system": None}
