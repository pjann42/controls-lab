import numpy as np
import control as ctrl

# Number of frequency points used throughout the app.
_N_POINTS = 1000


def _adaptive_frequency_grid(G, n_points=_N_POINTS):
    """Return a log-spaced frequency grid adapted to the poles and zeros of G.

    The range is determined by taking the minimum and maximum of the
    pole/zero magnitudes (ignoring values at the origin which would give
    zero magnitude) and then adding 2 decades of margin on each side.
    If no finite, non-zero poles or zeros exist the function falls back to
    the legacy ``[1e-2, 1e2]`` decade range.

    Args:
        G: control.TransferFunction
        n_points: number of frequency points (default 1000)

    Returns:
        numpy.ndarray of length ``n_points``
    """
    poles = ctrl.poles(G)
    zeros = ctrl.zeros(G)

    magnitudes = []
    for v in np.concatenate([poles, zeros]):
        m = abs(v)
        if m > 0 and np.isfinite(m):
            magnitudes.append(m)

    if magnitudes:
        w_min = min(magnitudes) / 1e2   # 2 decades below smallest feature
        w_max = max(magnitudes) * 1e2   # 2 decades above largest feature
        # Clamp to sensible absolute limits so purely-DC systems still plot
        w_min = max(w_min, 1e-4)
        w_max = min(w_max, 1e6)
    else:
        # No finite non-zero poles or zeros — use legacy range
        w_min, w_max = 1e-2, 1e2

    return np.logspace(np.log10(w_min), np.log10(w_max), n_points)


def compute_frequency_response(G, w=None):
    """Compute Bode magnitude (dB) and phase (degrees) for transfer function G.

    When ``w`` is *not* supplied the frequency grid is chosen adaptively
    based on the pole and zero locations of ``G`` — 2 decades of margin are
    added on each side of the smallest and largest pole/zero magnitudes.
    The grid always contains exactly 1000 points.

    Args:
        G: control.TransferFunction
        w: frequency array in rad/s (optional)

    Returns:
        tuple: (w, mag_db, phase_deg)
    """
    if w is None:
        w = _adaptive_frequency_grid(G, n_points=_N_POINTS)

    G_num, G_den = ctrl.tfdata(G)
    num = G_num[0][0]
    den = G_den[0][0]

    s = 1j * w
    G_jw = np.polyval(num, s) / np.polyval(den, s)

    mag_db = 20 * np.log10(np.abs(G_jw))
    phase_deg = np.degrees(np.unwrap(np.arctan2(np.imag(G_jw), np.real(G_jw))))

    return w, mag_db, phase_deg


def compute_margins(G):
    """Compute gain margin, phase margin, and crossover frequencies for G.

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
    """Return Y-axis limits that avoid excessive jitter in steady-state regions.

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
    """Snap phase axis limits to 45-degree multiples and generate tick values.

    Returns:
        tuple: (aligned_min, aligned_max, tick_values_list)
    """
    range_extension = 45
    aligned_min = round((phase_min - range_extension) / 45) * 45
    aligned_max = round((phase_max + range_extension) / 45) * 45
    tick_values = np.arange(aligned_min, aligned_max + 45, 45)
    return aligned_min, aligned_max, tick_values.tolist()
