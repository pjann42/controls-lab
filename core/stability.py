import numpy as np


def classify_stability(poles):
    """
    Classify system stability based on closed-loop pole positions in the s-plane.

    Returns:
        tuple: (stability_class, short_description, detailed_description)
    """
    if poles.size == 0:
        return "Static Gain", "No poles — pure gain", "System is a static gain (no dynamics)"

    left_poles = poles[poles.real < 0]
    right_poles = poles[poles.real > 0]
    imag_poles = poles[np.isclose(poles.real, 0, atol=1e-10)]

    # Detect repeated poles on the imaginary axis
    imag_pole_counts: dict[str, int] = {}
    for pole in imag_poles:
        key = f"{pole.real:.6f},{pole.imag:.6f}"
        imag_pole_counts[key] = imag_pole_counts.get(key, 0) + 1
    repeated_imag_poles = any(count > 1 for count in imag_pole_counts.values())

    origin_poles = poles[
        np.isclose(poles.real, 0, atol=1e-10) & np.isclose(poles.imag, 0, atol=1e-10)
    ]
    origin_pole_count = len(origin_poles)

    if len(right_poles) > 0:
        return (
            "Unstable",
            f"Pole(s) in right half-plane: {len(right_poles)}",
            f"System has {len(right_poles)} pole(s) with Re(s) > 0",
        )

    if repeated_imag_poles:
        return (
            "Unstable",
            "Repeated poles on imaginary axis",
            "System has repeated poles on jω axis (including origin)",
        )

    if origin_pole_count > 1:
        return (
            "Unstable",
            f"Multiple poles at origin: {origin_pole_count}",
            f"System has {origin_pole_count} poles at s=0 (integrators)",
        )

    if len(left_poles) == poles.size:
        return (
            "Asymptotically Stable",
            f"All {len(left_poles)} poles in left half-plane",
            "All poles have Re(s) < 0",
        )

    if origin_pole_count == 1 and len(left_poles) == poles.size - 1:
        return (
            "Marginally Stable",
            "Single pole at origin",
            f"Single integrator (s=0) + {len(left_poles)} stable poles. "
            "Open-loop response to step is unbounded (ramp) — expected for integrating components.",
        )

    if len(imag_poles) > 0 and not repeated_imag_poles:
        return (
            "Marginally Stable",
            f"Non-repeated poles on imaginary axis: {len(imag_poles)}",
            f"System has {len(imag_poles)} simple pole(s) on jω axis",
        )

    return "Undefined", "Mixed pole configuration", "Complex pole arrangement requiring detailed analysis"
