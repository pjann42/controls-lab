import numpy as np
import streamlit as st
import control as ctrl

from ui.plots import build_pz_figure


def _tf_card(title, latex_str):
    """Render a bordered card with a centred title and a LaTeX expression."""
    with st.container(border=True):
        st.markdown(
            f"<div style='text-align:center; font-weight:600;'>{title}</div>",
            unsafe_allow_html=True,
        )
        st.latex(latex_str)


def _stab_card(label, stab_class, details):
    """Render a bordered stability-status card with icon and colour coding."""
    icons = {
        "Asymptotically Stable": ("✅", "green"),
        "Marginally Stable":     ("ℹ️", "#4a9eff"),
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


def _metric_card(label, value):
    """Render a small centred metric card with a label and a value."""
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


def _make_pz_figure(p, z, title, pole_color="red", zero_color="blue"):
    """Build and return a Plotly pole-zero map figure."""
    return build_pz_figure(p, z, title, pole_color=pole_color, zero_color=zero_color)


def _pz_tab(label, sys_obj, pole_color="red", zero_color="blue"):
    """Render the pole-zero map tab: chart on the left, numeric list on the right."""
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
