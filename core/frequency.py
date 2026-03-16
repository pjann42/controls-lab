import numpy as np
import control as ctrl


def compute_frequency_response(G, w=None):
    """
    Compute Bode magnitude (dB) and phase (degrees) for transfer function G.

    Args:
        G: control.TransferFunction
        w: frequency array in rad/s (optional; defaults to logspace(-2, 2, 1000))

    Returns:
        tuple: (w, mag_db, phase_deg)
    """
    if w is None:
        w = np.logspace(-2, 2, 1000)

    G_num, G_den = ctrl.tfdata(G)
    num = G_num[0][0]
    den = G_den[0][0]

    s = 1j * w
    G_jw = np.polyval(num, s) / np.polyval(den, s)

    mag_db = 20 * np.log10(np.abs(G_jw))
    phase_deg = np.degrees(np.unwrap(np.arctan2(np.imag(G_jw), np.real(G_jw))))

    return w, mag_db, phase_deg


def compute_margins(G):
    """
    Compute gain margin, phase margin, and crossover frequencies for G.

    Returns:
        tuple: (gm, pm, wg, wp) — infinite or non-numeric values are returned as np.nan
    """
    try:
        gm, pm, wg, wp = ctrl.margin(G)
    except Exception:
        return np.nan, np.nan, np.nan, np.nan

    def _sanitize(val):
        if not isinstance(val, (int, float)) or np.isinf(val):
            return np.nan
        return float(val)

    return _sanitize(gm), _sanitize(pm), _sanitize(wg), _sanitize(wp)


def smart_autoscale(y_data, padding_factor=0.1, steady_state_threshold=0.01):
    """
    Return Y-axis limits that avoid excessive jitter in steady-state regions.

    Args:
        y_data: 1-D array of values to scale
        padding_factor: fractional padding around the data range
        steady_state_threshold: fraction of range below which a change is "steady"

    Returns:
        tuple: (y_min, y_max)
    """
    y_min, y_max = float(np.min(y_data)), float(np.max(y_data))
    data_range = y_max - y_min

    # Very small range — add fixed padding for visual clarity
    if data_range < 5:
        return y_min - 2, y_max + 2

    y_diff = np.abs(np.diff(y_data))
    steady_state_mask = y_diff < (data_range * steady_state_threshold)

    if np.sum(steady_state_mask) > len(y_data) * 0.3:
        # Mostly steady-state: use extended symmetric range
        extended_range = data_range * (1 + padding_factor * 2)
        center = (y_min + y_max) / 2
        return center - extended_range / 2, center + extended_range / 2

    y_padding = data_range * padding_factor
    return y_min - y_padding, y_max + y_padding


def align_phase_axis_45_deg(phase_min, phase_max):
    """
    Snap phase axis limits to 45-degree multiples and generate tick values.

    Returns:
        tuple: (aligned_min, aligned_max, tick_values_list)
    """
    range_extension = 45
    aligned_min = round((phase_min - range_extension) / 45) * 45
    aligned_max = round((phase_max + range_extension) / 45) * 45
    tick_values = np.arange(aligned_min, aligned_max + 45, 45)
    return aligned_min, aligned_max, tick_values.tolist()
