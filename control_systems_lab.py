import streamlit as st
import numpy as np
import control as ctrl
import plotly.graph_objects as go
import plotly.subplots as sp
from plotly.subplots import make_subplots

# Ensure compatibility with NumPy 2.0
np.NaN = np.nan
np.Inf = np.inf

st.set_page_config(page_title="Control Systems Laboratory", layout="wide", initial_sidebar_state="expanded")

st.title("Control Systems Laboratory")
st.markdown("Analyze open-loop transfer functions G(s) under unity feedback")

with st.sidebar:
    st.header("System Configuration")
    
    mode = st.radio("System Mode", ["Custom Transfer Function", "Canonical Second-Order System"])
    
    if mode == "Custom Transfer Function":
        st.subheader("Transfer Function Coefficients")
        numerator_input = st.text_input("Numerator Coefficients", "1", help="Enter comma-separated values")
        denominator_input = st.text_input("Denominator Coefficients", "1,1", help="Enter comma-separated values")
        
        try:
            num = [float(x.strip()) for x in numerator_input.split(',') if x.strip()]
            den = [float(x.strip()) for x in denominator_input.split(',') if x.strip()]
        except:
            st.error("Invalid coefficient format")
            st.stop()
    else:
        st.subheader("Second-Order Parameters")
        zeta = st.slider("Damping Ratio (ζ)", 0.0, 2.0, 0.5, 0.01)
        wn = st.slider("Natural Frequency (ωn)", 0.1, 10.0, 1.0, 0.1)
        
        num = [wn**2]
        den = [1, 2*zeta*wn, wn**2]
    
    st.markdown("---")
    calculate_button = st.button("🔍 Calculate System Analysis", type="primary")
    
    if not calculate_button:
        st.stop()

def format_polynomial(coeffs, variable='s'):
    if len(coeffs) == 0:
        return "0"
    
    terms = []
    degree = len(coeffs) - 1
    
    for i, coeff in enumerate(coeffs):
        if coeff == 0:
            continue
            
        power = degree - i
        if power == 0:
            term = f"{coeff}"
        elif power == 1:
            term = f"{coeff}{variable}" if coeff != 1 else variable
        else:
            term = f"{coeff}{variable}^{{{power}}}" if coeff != 1 else f"{variable}^{{{power}}}"
        
        terms.append(term)
    
    return " + ".join(terms) if terms else "0"

def classify_stability(poles):
    """
    Classify system stability based on pole positions in the s-plane.
    
    Returns:
        tuple: (stability_class, description, details)
    """
    if poles.size == 0:
        return "Undefined", "No poles found", "System has no poles to analyze"
    
    # Count poles by region
    left_poles = poles[poles.real < 0]
    right_poles = poles[poles.real > 0]
    imag_poles = poles[np.isclose(poles.real, 0, atol=1e-10)]
    
    # Count repeated poles on imaginary axis
    imag_pole_counts = {}
    for pole in imag_poles:
        pole_key = f"{pole.real:.6f},{pole.imag:.6f}"
        imag_pole_counts[pole_key] = imag_pole_counts.get(pole_key, 0) + 1
    
    repeated_imag_poles = any(count > 1 for count in imag_pole_counts.values())
    
    # Check for pole at origin (s=0)
    origin_poles = poles[np.isclose(poles.real, 0, atol=1e-10) & np.isclose(poles.imag, 0, atol=1e-10)]
    origin_pole_count = len(origin_poles)
    
    # Stability classification logic
    if len(right_poles) > 0:
        return "Unstable", f"Pole(s) in right half-plane: {len(right_poles)}", \
               f"System has {len(right_poles)} pole(s) with Re(s) > 0"
    
    elif repeated_imag_poles:
        return "Unstable", "Repeated poles on imaginary axis", \
               f"System has repeated poles on jω axis (including origin)"
    
    elif origin_pole_count > 1:
        return "Unstable", f"Multiple poles at origin: {origin_pole_count}", \
               f"System has {origin_pole_count} poles at s=0 (integrators)"
    
    elif len(left_poles) == poles.size:
        return "Asymptotically Stable", f"All {len(left_poles)} poles in left half-plane", \
               f"All poles have Re(s) < 0"
    
    elif origin_pole_count == 1 and len(left_poles) == poles.size - 1:
        return "Marginally Stable", "Single pole at origin", \
               f"Single integrator (s=0) + {len(left_poles)} stable poles. Note: Step input causes unbounded output"
    
    elif len(imag_poles) > 0 and not repeated_imag_poles:
        return "Marginally Stable", f"Non-repeated poles on imaginary axis: {len(imag_poles)}", \
               f"System has {len(imag_poles)} simple pole(s) on jω axis"
    
    else:
        return "Undefined", "Mixed pole configuration", "Complex pole arrangement requiring detailed analysis"

try:
    G = ctrl.TransferFunction(num, den)
    T = ctrl.feedback(G, 1)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Open-Loop Transfer Function G(s)")
        num_str = format_polynomial(num)
        den_str = format_polynomial(den)
        st.latex(f"G(s) = \\frac{{{num_str}}}{{{den_str}}}")
        
    with col2:
        st.subheader("Closed-Loop Transfer Function T(s)")
        # Get closed-loop numerator and denominator
        T_num, T_den = ctrl.tfdata(T)
        T_num_str = format_polynomial(T_num[0][0].tolist())
        T_den_str = format_polynomial(T_den[0][0].tolist())
        st.latex(f"T(s) = \\frac{{{T_num_str}}}{{{T_den_str}}}")
    
    st.header("Time Domain Analysis")
    st.caption("Using Closed-Loop Transfer Function T(s)")
    
    time = np.linspace(0, 10, 1000)
    t, y = ctrl.step_response(T, time)
    
    step_info = ctrl.step_info(T)
    
    col3, col4 = st.columns([2, 1])
    
    with col3:
        fig_step = go.Figure()
        fig_step.add_trace(go.Scatter(x=t, y=y, mode='lines', name='Step Response', line=dict(color='blue', width=2)))
        fig_step.update_layout(
            title="Step Response",
            xaxis_title="Time (s)",
            yaxis_title="Amplitude",
            hovermode='x unified',
            template='plotly_white'
        )
        st.plotly_chart(fig_step, use_container_width=True)
    
    with col4:
        st.subheader("Step Response Metrics")
        st.metric("Rise Time", f"{step_info.get('RiseTime', 'N/A'):.3f} s")
        st.metric("Settling Time", f"{step_info.get('SettlingTime', 'N/A'):.3f} s")
        st.metric("Overshoot", f"{step_info.get('Overshoot', 0):.3f}%")
        st.metric("Peak", f"{step_info.get('Peak', 'N/A'):.3f}")
        st.metric("Peak Time", f"{step_info.get('PeakTime', 'N/A'):.3f} s")
    
    st.header("Frequency Domain Analysis")
    st.caption("Using Open-Loop Transfer Function G(s)")
    
    # Generate frequency response data using proper method
    w = np.logspace(-2, 2, 1000)  # 0.01 to 100 rad/s
    mag, phase, w_out = ctrl.bode(G, w, plot=False)
    
    # The phase from bode is already in degrees
    mag_db = 20 * np.log10(mag)
    phase_deg = phase
    
    try:
        gm, pm, wg, wp = ctrl.margin(G)
    except:
        gm, pm, wg, wp = np.nan, np.nan, np.nan, np.nan

    # Handle infinite values
    if not isinstance(gm, (int, float)) or np.isinf(gm):
        gm = np.nan
    if not isinstance(pm, (int, float)) or np.isinf(pm):
        pm = np.nan
    if not isinstance(wg, (int, float)) or np.isinf(wg):
        wg = np.nan
    if not isinstance(wp, (int, float)) or np.isinf(wp):
        wp = np.nan
    
    fig_bode = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Magnitude', 'Phase'),
        vertical_spacing=0.1
    )
    
    fig_bode.add_trace(
        go.Scatter(x=w, y=mag_db, mode='lines', name='Magnitude', line=dict(color='blue')),
        row=1, col=1
    )
    
    if not np.isnan(gm) and not np.isnan(wg):
        fig_bode.add_vline(x=wg, line_dash="dash", line_color="red", annotation_text=f"GM: {20*np.log10(gm):.1f} dB", row=1, col=1)
    
    fig_bode.add_hline(y=0, line_dash="dot", line_color="gray", row=1, col=1)
    
    fig_bode.add_trace(
        go.Scatter(x=w, y=phase_deg, mode='lines', name='Phase', line=dict(color='green')),
        row=2, col=1
    )
    
    if not np.isnan(pm) and not np.isnan(wp):
        fig_bode.add_vline(x=wp, line_dash="dash", line_color="red", annotation_text=f"PM: {pm:.1f}°", row=2, col=1)
    
    fig_bode.add_hline(y=-180, line_dash="dot", line_color="gray", row=2, col=1)
    
    fig_bode.update_xaxes(type="log", title_text="Frequency (rad/s)", row=1, col=1)
    fig_bode.update_xaxes(type="log", title_text="Frequency (rad/s)", row=2, col=1)
    fig_bode.update_yaxes(title_text="Magnitude (dB)", row=1, col=1)
    fig_bode.update_yaxes(title_text="Phase (degrees)", row=2, col=1)
    fig_bode.update_layout(template='plotly_white', height=600)
    
    st.plotly_chart(fig_bode, use_container_width=True)
    
    col5, col6, col7 = st.columns(3)
    with col5:
        if not np.isnan(gm):
            st.metric("Gain Margin", f"{20*np.log10(gm):.3f} dB")
        else:
            st.metric("Gain Margin", "∞")
    with col6:
        if not np.isnan(pm):
            st.metric("Phase Margin", f"{pm:.3f}°")
        else:
            st.metric("Phase Margin", "N/A")
    with col7:
        if not np.isnan(wp):
            st.metric("Crossover Frequency", f"{wp:.3f} rad/s")
        else:
            st.metric("Crossover Frequency", "N/A")
    
    st.header("Stability Analysis")
    st.caption("Using Open-Loop Transfer Function G(s)")
    
    poles = ctrl.pole(G)
    zeros = ctrl.zero(G)
    
    # Classify stability
    stability_class, stability_desc, stability_details = classify_stability(poles)
    
    # Display stability result
    if stability_class == "Asymptotically Stable":
        st.success(f"✅ System is {stability_class}")
    elif stability_class == "Marginally Stable":
        st.warning(f"⚠️ System is {stability_class}")
    elif stability_class == "Unstable":
        st.error(f"❌ System is {stability_class}")
    else:
        st.info(f"ℹ️ System is {stability_class}")
    
    st.caption(stability_details)
    
    fig_pz = go.Figure()
    
    if poles.size > 0:
        fig_pz.add_trace(go.Scatter(
            x=[p.real for p in poles],
            y=[p.imag for p in poles],
            mode='markers',
            name='Poles',
            marker=dict(symbol='x', size=12, color='red', line=dict(width=2))
        ))
    
    if zeros.size > 0:
        fig_pz.add_trace(go.Scatter(
            x=[z.real for z in zeros],
            y=[z.imag for z in zeros],
            mode='markers',
            name='Zeros',
            marker=dict(symbol='circle', size=10, color='blue', line=dict(width=2))
        ))
    
    fig_pz.add_hline(y=0, line_dash="dot", line_color="gray")
    fig_pz.add_vline(x=0, line_dash="dot", line_color="gray")
    
    max_val = max(max(abs(p.real) for p in poles) if poles.size > 0 else 1, 
                  max(abs(p.imag) for p in poles) if poles.size > 0 else 1,
                  max(abs(z.real) for z in zeros) if zeros.size > 0 else 1,
                  max(abs(z.imag) for z in zeros) if zeros.size > 0 else 1) * 1.2
    
    fig_pz.update_layout(
        title="Pole-Zero Map",
        xaxis_title="Real Axis",
        yaxis_title="Imaginary Axis",
        xaxis=dict(range=[-max_val, max_val]),
        yaxis=dict(range=[-max_val, max_val]),
        template='plotly_white',
        showlegend=True
    )
    
    st.plotly_chart(fig_pz, use_container_width=True)
    
    col8, col9 = st.columns(2)
    with col8:
        st.subheader("System Poles")
        if poles.size > 0:
            for i, pole in enumerate(poles):
                st.write(f"Pole {i+1}: {pole:.3f}")
        else:
            st.write("No poles")
    
    with col9:
        st.subheader("System Zeros")
        if zeros.size > 0:
            for i, zero in enumerate(zeros):
                st.write(f"Zero {i+1}: {zero:.3f}")
        else:
            st.write("No zeros")

except Exception as e:
    st.error(f"Error: {str(e)}")
    st.stop()
