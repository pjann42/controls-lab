import numpy as np
import streamlit as st
import control as ctrl
import plotly.graph_objects as go

from core.transfer_function import validate_and_create_system
from core.stability import classify_stability
from core.frequency import compute_frequency_response, compute_margins, smart_autoscale, align_phase_axis_45_deg
from core.formatting import format_polynomial, format_metric, clean_coefficients

# NumPy 2.0 compatibility shim
np.NaN = np.nan
np.Inf = np.inf

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Control Systems Laboratory", layout="wide", initial_sidebar_state="expanded")

# ---------------------------------------------------------------------------
# Responsive CSS — columns stack on mobile
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@media (max-width: 768px) {
    /* Stack all column groups vertically */
    [data-testid="stHorizontalBlock"] {
        flex-wrap: wrap !important;
    }
    [data-testid="stColumn"] {
        width: 100% !important;
        flex: 1 1 100% !important;
        min-width: 100% !important;
    }
    /* Tighter card padding on small screens */
    [data-testid="stVerticalBlockBorderWrapper"] {
        padding: 0.5rem !important;
    }
    /* Reduce chart height hints */
    .js-plotly-plot {
        max-width: 100% !important;
    }
    /* Title font size */
    h1 { font-size: 1.4rem !important; }
    h2 { font-size: 1.1rem !important; }
    h3 { font-size: 1rem !important;  }
}
</style>
""", unsafe_allow_html=True)

st.title("Control Systems Laboratory")
st.markdown("Analyze open-loop transfer functions G(s) under unity feedback")

# ---------------------------------------------------------------------------
# Sidebar — system configuration
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    # --- System mode ---
    mode = st.radio(
        "System Mode",
        ["Custom Transfer Function", "Canonical Second-Order System"],
        label_visibility="collapsed",
    )

    # --- Plant G(s) ---
    with st.container(border=True):
        st.markdown("**Plant G(s)**")

        if mode == "Custom Transfer Function":
            numerator_input = st.text_input(
                "Numerator", "1",
                help="Comma-separated coefficients (highest power first), e.g. 1,2",
            )
            denominator_input = st.text_input(
                "Denominator", "1,1",
                help="Comma-separated coefficients (highest power first), e.g. 1,3,2",
            )

            try:
                num = [float(x.strip()) for x in numerator_input.split(",") if x.strip()]
                den = [float(x.strip()) for x in denominator_input.split(",") if x.strip()]
            except ValueError:
                st.error("Invalid coefficient format")
                st.stop()
        else:
            zeta_col, wn_col = st.columns(2)
            with zeta_col:
                zeta_input = st.text_input("ζ", "0.5", help="Damping ratio (≥ 0)")
            with wn_col:
                wn_input = st.text_input("ωn", "1.0", help="Natural frequency (> 0)")

            try:
                zeta = float(zeta_input.strip())
                wn = float(wn_input.strip())
            except ValueError:
                st.error("Invalid input format.")
                st.stop()

            if zeta < 0:
                st.error("Damping ratio must be non-negative")
                st.stop()
            if wn <= 0:
                st.error("Natural frequency must be positive")
                st.stop()

            num = [wn**2]
            den = [1, 2 * zeta * wn, wn**2]

    # --- Controller C(s) ---
    with st.container(border=True):
        use_controller = st.checkbox("**Controller C(s)**")

        controller_num = [1]
        controller_den = [1]

        if use_controller:
            ctrl_num_input = st.text_input(
                "Numerator", "1",
                help="Comma-separated coefficients", key="ctrl_num",
            )
            ctrl_den_input = st.text_input(
                "Denominator", "1",
                help="Comma-separated coefficients", key="ctrl_den",
            )

            try:
                controller_num = [float(x.strip()) for x in ctrl_num_input.split(",") if x.strip()]
                controller_den = [float(x.strip()) for x in ctrl_den_input.split(",") if x.strip()]
            except ValueError:
                st.error("Invalid controller coefficient format")
                st.stop()
        else:
            st.caption("Check to add C(s) in series with G(s).")

    calculate_button = st.button("🔍 Calculate", type="primary", use_container_width=True)

    st.caption("⚠️ Under active development. Results should be validated against reference tools before use.")

    if not calculate_button:
        st.stop()

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
try:
    # Validate plant G(s)
    validation = validate_and_create_system(num, den)

    if not validation["valid"]:
        st.error(f"⚠️ {validation['message']}")

        if validation.get("improper_system"):
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Open-Loop Transfer Function G(s)")
                st.latex(f"G(s) = \\frac{{{format_polynomial(num)}}}{{{format_polynomial(den)}}}")
            with col2:
                st.subheader("Closed-Loop Transfer Function T(s)")
                st.latex(r"T(s) = \frac{G(s)}{1 + G(s)}")

            st.markdown(
                f"""
                **Improper Transfer Function**: Your system has numerator degree {validation['num_degree']}
                and denominator degree {validation['den_degree']}. Since
                {validation['num_degree']} > {validation['den_degree']},
                this system is **physically impossible**.

                An improper transfer function violates causality. To fix it, add poles or reduce the
                numerator order.
                """
            )
        st.stop()

    G = validation["system"]

    # Validate and build controller C(s)
    if use_controller:
        ctrl_validation = validate_and_create_system(controller_num, controller_den)
        if not ctrl_validation["valid"]:
            st.error(f"⚠️ Controller C(s): {ctrl_validation['message']}")
            st.stop()
        C = ctrl_validation["system"]
        GC = ctrl.series(G, C)
    else:
        C = None
        GC = G

    T = ctrl.feedback(GC, 1)

    # -----------------------------------------------------------------------
    # Transfer function display — always two columns: Open-Loop | Closed-Loop
    # -----------------------------------------------------------------------
    T_num, T_den = ctrl.tfdata(T)
    T_num_str = format_polynomial(clean_coefficients(T_num[0][0].tolist()))
    T_den_str = format_polynomial(clean_coefficients(T_den[0][0].tolist()))

    GC_num_data, GC_den_data = ctrl.tfdata(GC)
    GC_num_str = format_polynomial(clean_coefficients(GC_num_data[0][0].tolist()))
    GC_den_str = format_polynomial(clean_coefficients(GC_den_data[0][0].tolist()))

    def _tf_card(title, latex_str):
        with st.container(border=True):
            st.markdown(f"<div style='text-align:center; font-weight:600;'>{title}</div>", unsafe_allow_html=True)
            st.latex(latex_str)

    if use_controller:
        c1, c2 = st.columns(2)
        with c1:
            _tf_card("Plant G(s)", f"G(s) = \\frac{{{format_polynomial(num)}}}{{{format_polynomial(den)}}}")
        with c2:
            _tf_card("Controller C(s)",
                     f"C(s) = \\frac{{{format_polynomial(controller_num)}}}{{{format_polynomial(controller_den)}}}")

        c3, c4 = st.columns(2)
        with c3:
            _tf_card("Open-Loop G(s)C(s)", f"G(s)C(s) = \\frac{{{GC_num_str}}}{{{GC_den_str}}}")
        with c4:
            _tf_card("Closed-Loop T(s)", f"T(s) = \\frac{{{T_num_str}}}{{{T_den_str}}}")
    else:
        c1, c2 = st.columns(2)
        with c1:
            ol_latex = f"G(s) = \\frac{{{format_polynomial(num)}}}{{{format_polynomial(den)}}}"
            if validation.get("identical_coeffs"):
                ol_latex += " = 1"
            _tf_card("Open-Loop G(s)", ol_latex)
        with c2:
            cl_latex = f"T(s) = \\frac{{{T_num_str}}}{{{T_den_str}}}"
            if validation.get("identical_coeffs"):
                cl_latex += " = 0.5"
            _tf_card("Closed-Loop T(s)", cl_latex)

    st.info(f"✅ {validation['message']}")

    # Unity-gain special case (only valid without controller)
    if validation.get("identical_coeffs") and not use_controller:
        st.markdown(
            r"""
            **Unity Gain System Analysis**: This transfer function simplifies to $H(s) = 1$.
            With unity feedback the closed-loop gain is:

            $$T(s) = \frac{H(s)}{1 + H(s)} = \frac{1}{2} = 0.5$$

            meaning the output follows the input with half the amplitude.
            """
        )
        st.stop()

    # ---------------------------------------------------------------------------
    # Stability Analysis — all components
    # ---------------------------------------------------------------------------
    st.header("Stability Analysis")

    # Classify each component
    g_stab,  _, g_det  = classify_stability(ctrl.poles(G))
    gc_stab, _, gc_det = classify_stability(ctrl.poles(GC))
    cl_stab, _, cl_det = classify_stability(ctrl.poles(T))
    if use_controller:
        c_stab, _, c_det = classify_stability(ctrl.poles(C))

    def _stab_card(label, stab_class, details):
        icons = {
            "Asymptotically Stable": ("✅", "green"),
            "Marginally Stable":     ("⚠️", "orange"),
            "Unstable":              ("❌", "red"),
        }
        icon, color = icons.get(stab_class, ("ℹ️", "gray"))
        with st.container(border=True):
            st.markdown(
                "<div style='display:flex; flex-direction:column; align-items:center; "
                "justify-content:center; min-height:90px; padding:8px 4px; text-align:center;'>"
                f"<div style='font-weight:600; font-size:0.82rem; text-transform:uppercase; "
                f"letter-spacing:0.04em; color:#555; margin-bottom:6px'>{label}</div>"
                f"<div style='font-size:1.5rem; line-height:1'>{icon}</div>"
                f"<div style='color:{color}; font-weight:700; font-size:0.88rem; margin-top:4px'>{stab_class}</div>"
                f"<div style='color:#999; font-size:0.74rem; margin-top:4px'>{details}</div>"
                "</div>",
                unsafe_allow_html=True,
            )

    if use_controller:
        # 2 per row → wraps to 2×2 on mobile via CSS
        ca, cb = st.columns(2)
        with ca: _stab_card("Plant G(s)",         g_stab,  g_det)
        with cb: _stab_card("Controller C(s)",    c_stab,  c_det)
        cc, cd = st.columns(2)
        with cc: _stab_card("Open-Loop G(s)C(s)", gc_stab, gc_det)
        with cd: _stab_card("Closed-Loop T(s)",   cl_stab, cl_det)
    else:
        ca, cb, cc = st.columns(3)
        with ca: _stab_card("Plant G(s)",       g_stab,  g_det)
        with cb: _stab_card("Open-Loop G(s)",   gc_stab, gc_det)
        with cc: _stab_card("Closed-Loop T(s)", cl_stab, cl_det)

    # Contextual diagnostic messages
    if use_controller:
        if g_stab == "Unstable" and cl_stab == "Asymptotically Stable":
            st.success("✅ The controller successfully stabilizes the unstable plant.")
        if g_stab == "Unstable" and cl_stab == "Unstable":
            st.error("❌ The controller fails to stabilize the unstable plant.")
        if c_stab == "Unstable" and cl_stab == "Asymptotically Stable":
            st.warning("⚠️ The controller itself is unstable — the closed loop is stable, but this is fragile in practice.")
        if c_stab == "Unstable" and cl_stab == "Unstable":
            st.error("❌ Both the controller and the plant are unstable, and the closed loop did not recover.")
    if gc_stab == "Unstable" and cl_stab == "Asymptotically Stable":
        st.info("ℹ️ The open loop is unstable. Gain/Phase margins from Bode require Nyquist analysis for correct interpretation.")
    if cl_stab == "Marginally Stable":
        st.warning("⚠️ The closed loop is marginally stable — sustained oscillations may occur for certain inputs.")

    # ---------------------------------------------------------------------------
    # Pole-Zero Maps
    # ---------------------------------------------------------------------------
    def _make_pz_figure(p, z, title, pole_color="red", zero_color="blue"):
        fig = go.Figure()
        if p.size > 0:
            fig.add_trace(go.Scatter(
                x=[v.real for v in p], y=[v.imag for v in p],
                mode="markers", name=f"{title} Poles",
                marker=dict(symbol="x", size=12, color=pole_color, line=dict(width=2)),
            ))
        if z.size > 0:
            fig.add_trace(go.Scatter(
                x=[v.real for v in z], y=[v.imag for v in z],
                mode="markers", name=f"{title} Zeros",
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
            xaxis_title="Real Axis", yaxis_title="Imaginary Axis",
            xaxis=dict(range=[-limit, limit]),
            yaxis=dict(range=[-limit, limit]),
            template="plotly_white", showlegend=True,
        )
        return fig

    def _metric_card(label, value):
        with st.container(border=True):
            st.markdown(
                "<div style='"
                "display:flex; flex-direction:column; align-items:center; "
                "justify-content:center; min-height:68px; padding:6px 4px;'>"
                f"<div style='color:#888; font-size:0.74rem; text-transform:uppercase; "
                f"letter-spacing:0.04em; margin-bottom:6px; text-align:center'>{label}</div>"
                f"<div style='font-size:1.2rem; font-weight:700; text-align:center; "
                f"line-height:1.2'>{value}</div>"
                "</div>",
                unsafe_allow_html=True,
            )

    def _pz_tab(label, sys_obj, pole_color="red", zero_color="blue"):
        p, z = ctrl.poles(sys_obj), ctrl.zeros(sys_obj)
        chart_col, info_col = st.columns([3, 1])
        with chart_col:
            fig = _make_pz_figure(p, z, label, pole_color, zero_color)
            fig.update_layout(height=380)
            st.plotly_chart(fig, use_container_width=True)
        with info_col:
            with st.container(border=True):
                st.markdown("**Poles**")
                if p.size > 0:
                    for i, v in enumerate(p):
                        st.caption(f"{i+1}: `{v:.3f}`")
                else:
                    st.caption("None")
            with st.container(border=True):
                st.markdown("**Zeros**")
                if z.size > 0:
                    for i, v in enumerate(z):
                        st.caption(f"{i+1}: `{v:.3f}`")
                else:
                    st.caption("None")

    ol_tab_label = "Open-Loop G(s)C(s)" if use_controller else "Open-Loop G(s)"
    tab_ol, tab_cl = st.tabs([ol_tab_label, "Closed-Loop T(s)"])
    with tab_ol: _pz_tab(ol_tab_label, GC, "red", "blue")
    with tab_cl: _pz_tab("Closed-Loop", T, "darkred", "darkblue")

    # convenience alias used by time-domain and Bode sections below
    stability_class = cl_stab

    # ---------------------------------------------------------------------------
    # Time Domain Analysis (stable systems only)
    # ---------------------------------------------------------------------------
    st.divider()
    st.header("Time Domain Analysis")
    st.caption("Using Closed-Loop Transfer Function T(s)")

    if stability_class == "Unstable":
        st.error("⚠️ Skipped — step response diverges for unstable closed-loop systems")
    else:
        time = np.linspace(0, 10, 1000)
        t, y = ctrl.step_response(T, time)
        step_info = ctrl.step_info(T)

        # Only show steady-state line for asymptotically stable systems
        fig_step = go.Figure()
        fig_step.add_trace(go.Scatter(
            x=t, y=y, mode="lines", name="Step Response",
            line=dict(color="royalblue", width=2),
        ))
        if stability_class == "Asymptotically Stable":
            fig_step.add_hline(y=float(y[-1]), line_dash="dot", line_color="gray",
                               annotation_text=f"Steady state: {y[-1]:.3f}")
        fig_step.update_layout(
            title="Step Response",
            xaxis_title="Time (s)",
            yaxis_title="Amplitude",
            hovermode="x unified",
            template="plotly_white",
            height=350,
        )
        st.plotly_chart(fig_step, use_container_width=True)

        st.subheader("Step Response Metrics")

        if stability_class == "Marginally Stable":
            st.info(
                "ℹ️ Only Peak metrics are shown — Rise Time, Settling Time, Overshoot and "
                "Steady-State Value are not physically meaningful for marginally stable systems."
            )
            # Only Peak and Peak Time make sense
            raw = [
                ("Peak",      format_metric(step_info.get("Peak"))),
                ("Peak Time", f"{format_metric(step_info.get('PeakTime'))} s"),
            ]
        else:
            # Asymptotically Stable: all 6 metrics, drop any that are N/A
            raw = [
                ("Rise Time",     f"{format_metric(step_info.get('RiseTime'))} s"),
                ("Settling Time", f"{format_metric(step_info.get('SettlingTime'))} s"),
                ("Overshoot",     f"{format_metric(step_info.get('Overshoot', 0))}%"),
                ("Peak",          format_metric(step_info.get("Peak"))),
                ("Peak Time",     f"{format_metric(step_info.get('PeakTime'))} s"),
                ("Steady State",  format_metric(y[-1])),
            ]

        valid = [(lbl, val) for lbl, val in raw if "N/A" not in val]
        if valid:
            # Max 3 per row so cards stay readable on mobile
            for row_start in range(0, len(valid), 3):
                row = valid[row_start:row_start + 3]
                cols = st.columns(len(row))
                for col, (lbl, val) in zip(cols, row):
                    with col:
                        _metric_card(lbl, val)

    # ---------------------------------------------------------------------------
    # Frequency Domain Analysis
    # ---------------------------------------------------------------------------
    st.divider()
    st.header("Frequency Domain Analysis")
    ol_caption = "Using Open-Loop G(s)C(s)" if use_controller else "Using Open-Loop G(s)"
    st.caption(ol_caption)

    if gc_stab == "Unstable":
        st.warning("⚠️ Open loop is unstable — Gain/Phase margins require Nyquist analysis for correct interpretation")

    w, mag_db, phase_deg = compute_frequency_response(GC)
    gm, pm, wg, wp = compute_margins(GC)

    mag_ymin, mag_ymax = smart_autoscale(mag_db, padding_factor=0.15, steady_state_threshold=0.02)
    phase_ymin, phase_ymax = smart_autoscale(phase_deg, padding_factor=0.15, steady_state_threshold=0.02)
    phase_ymin_aligned, phase_ymax_aligned, phase_ticks_45 = align_phase_axis_45_deg(phase_ymin, phase_ymax)

    # — Magnitude —
    st.subheader("Magnitude")
    fig_mag = go.Figure()
    fig_mag.add_trace(go.Scatter(x=w, y=mag_db, mode="lines", name="Magnitude",
                                 line=dict(color="royalblue", width=2)))
    if not np.isnan(gm) and not np.isnan(wg):
        fig_mag.add_vline(x=wg, line_dash="dash", line_color="crimson", line_width=1.5)
    fig_mag.add_hline(y=0, line_dash="dot", line_color="gray", annotation_text="0 dB",
                      annotation_position="bottom right")
    fig_mag.update_layout(
        xaxis=dict(type="log", title="Frequency (rad/s)"),
        yaxis=dict(title="Magnitude (dB)", range=[mag_ymin, mag_ymax]),
        template="plotly_white", height=300, margin=dict(t=30),
        hovermode="x unified",
    )
    st.plotly_chart(fig_mag, use_container_width=True)

    # — Phase —
    st.subheader("Phase")
    fig_phase = go.Figure()
    fig_phase.add_trace(go.Scatter(x=w, y=phase_deg, mode="lines", name="Phase",
                                   line=dict(color="seagreen", width=2)))
    if not np.isnan(pm) and not np.isnan(wp):
        fig_phase.add_vline(x=wp, line_dash="dash", line_color="crimson", line_width=1.5)
    fig_phase.add_hline(y=-180, line_dash="dot", line_color="gray", annotation_text="-180°",
                        annotation_position="bottom right")
    fig_phase.update_layout(
        xaxis=dict(type="log", title="Frequency (rad/s)"),
        yaxis=dict(
            title="Phase (degrees)",
            range=[phase_ymin_aligned, phase_ymax_aligned],
            tickmode="array",
            tickvals=phase_ticks_45,
            ticktext=[f"{int(t)}°" for t in phase_ticks_45],
        ),
        template="plotly_white", height=300, margin=dict(t=30),
        hovermode="x unified",
    )
    st.plotly_chart(fig_phase, use_container_width=True)

    # — Stability Margins —
    st.subheader("Stability Margins")
    sm1, sm2, sm3 = st.columns(3)
    with sm1: _metric_card("Gain Margin",          f"{20*np.log10(gm):.3f} dB" if not np.isnan(gm) else "∞")
    with sm2: _metric_card("Phase Margin",         f"{pm:.3f}°"                if not np.isnan(pm) else "N/A")
    with sm3: _metric_card("Crossover Frequency",  f"{wp:.3f} rad/s"           if not np.isnan(wp) else "N/A")

except Exception as e:
    st.error(f"Error: {e}")
    st.stop()
