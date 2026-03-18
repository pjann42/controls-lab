"""
Plot builders for Control Systems Lab.

All functions return ``plotly.graph_objects.Figure`` objects.
They must not import or call streamlit — the caller is responsible for
rendering via ``st.plotly_chart``.
"""

import numpy as np
import plotly.graph_objects as go


def build_pz_figure(p, z, title, pole_color="red", zero_color="blue"):
    """Return a Plotly pole-zero map figure.

    Args:
        p: array of pole values (complex)
        z: array of zero values (complex)
        title: string label used in trace names and the figure title
        pole_color: colour for pole markers
        zero_color: colour for zero markers

    Returns:
        go.Figure
    """
    fig = go.Figure()

    if p.size > 0:
        fig.add_trace(go.Scatter(
            x=[v.real for v in p],
            y=[v.imag for v in p],
            mode="markers",
            name=f"{title} Poles",
            marker=dict(symbol="x", size=12, color=pole_color, line=dict(width=2)),
        ))

    if z.size > 0:
        fig.add_trace(go.Scatter(
            x=[v.real for v in z],
            y=[v.imag for v in z],
            mode="markers",
            name=f"{title} Zeros",
            marker=dict(symbol="circle", size=10, color=zero_color, line=dict(width=2)),
        ))

    fig.add_hline(y=0, line_dash="dot", line_color="gray")
    fig.add_vline(x=0, line_dash="dot", line_color="gray")

    all_vals = np.concatenate([
        [abs(v.real) for v in p], [abs(v.imag) for v in p],
        [abs(v.real) for v in z], [abs(v.imag) for v in z],
        [1.0],
    ])
    limit = float(np.max(all_vals)) * 1.2

    fig.update_layout(
        title=f"{title} Pole-Zero Map",
        xaxis_title="Real Axis",
        yaxis_title="Imaginary Axis",
        xaxis=dict(range=[-limit, limit]),
        yaxis=dict(range=[-limit, limit]),
        template="plotly_white",
        showlegend=True,
    )
    return fig


def build_step_figure(t, y, stability_class):
    """Return a Plotly step-response figure.

    Args:
        t: time array
        y: response array
        stability_class: one of "Asymptotically Stable", "Marginally Stable", "Unstable"

    Returns:
        go.Figure
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t, y=y,
        mode="lines",
        name="Step Response",
        line=dict(color="royalblue", width=2),
    ))

    if stability_class == "Asymptotically Stable":
        fig.add_hline(
            y=float(y[-1]),
            line_dash="dot",
            line_color="gray",
            annotation_text=f"Steady state: {y[-1]:.3f}",
        )

    fig.update_layout(
        title="Step Response",
        xaxis_title="Time (s)",
        yaxis_title="Amplitude",
        hovermode="x unified",
        template="plotly_white",
        height=350,
    )
    return fig


def build_magnitude_figure(w, mag_db, mag_ymin, mag_ymax, gm, wg):
    """Return a Plotly Bode magnitude figure.

    Args:
        w: frequency array (rad/s)
        mag_db: magnitude array (dB)
        mag_ymin: lower y-axis limit
        mag_ymax: upper y-axis limit
        gm: gain margin (np.nan if unavailable)
        wg: gain crossover frequency (np.nan if unavailable)

    Returns:
        go.Figure
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=w, y=mag_db,
        mode="lines",
        name="Magnitude",
        line=dict(color="royalblue", width=2),
    ))

    if not np.isnan(gm) and not np.isnan(wg):
        fig.add_vline(x=wg, line_dash="dash", line_color="crimson", line_width=1.5)

    fig.add_hline(
        y=0,
        line_dash="dot",
        line_color="gray",
        annotation_text="0 dB",
        annotation_position="bottom right",
    )

    fig.update_layout(
        xaxis=dict(type="log", title="Frequency (rad/s)"),
        yaxis=dict(title="Magnitude (dB)", range=[mag_ymin, mag_ymax]),
        template="plotly_white",
        height=300,
        margin=dict(t=30),
        hovermode="x unified",
    )
    return fig


def build_phase_figure(w, phase_deg, phase_ymin_aligned, phase_ymax_aligned, phase_ticks_45, pm, wp):
    """Return a Plotly Bode phase figure.

    Args:
        w: frequency array (rad/s)
        phase_deg: phase array (degrees)
        phase_ymin_aligned: lower y-axis limit (snapped to 45° multiple)
        phase_ymax_aligned: upper y-axis limit (snapped to 45° multiple)
        phase_ticks_45: list of tick values at 45° intervals
        pm: phase margin (np.nan if unavailable)
        wp: phase crossover frequency (np.nan if unavailable)

    Returns:
        go.Figure
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=w, y=phase_deg,
        mode="lines",
        name="Phase",
        line=dict(color="seagreen", width=2),
    ))

    if not np.isnan(pm) and not np.isnan(wp):
        fig.add_vline(x=wp, line_dash="dash", line_color="crimson", line_width=1.5)

    fig.add_hline(
        y=-180,
        line_dash="dot",
        line_color="gray",
        annotation_text="-180°",
        annotation_position="bottom right",
    )

    fig.update_layout(
        xaxis=dict(type="log", title="Frequency (rad/s)"),
        yaxis=dict(
            title="Phase (degrees)",
            range=[phase_ymin_aligned, phase_ymax_aligned],
            tickmode="array",
            tickvals=phase_ticks_45,
            ticktext=[f"{int(t)}°" for t in phase_ticks_45],
        ),
        template="plotly_white",
        height=300,
        margin=dict(t=30),
        hovermode="x unified",
    )
    return fig
