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
        st.info("Click the Calculate button to perform system analysis")
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
        st.latex(f"T(s) = \\frac{{G(s)}}{{1 + G(s)}}")
    
    st.header("Time Domain Analysis")
    
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
        st.metric("Overshoot", f"{step_info.get('Overshoot', 0):.1f}%")
        st.metric("Peak", f"{step_info.get('Peak', 'N/A'):.3f}")
        st.metric("Peak Time", f"{step_info.get('PeakTime', 'N/A'):.3f} s")
    
    st.header("Frequency Domain Analysis")
    
    w, mag, phase = ctrl.bode_plot(G, plot=False)
    w = w[1:]
    mag = mag[1:]
    phase = phase[1:]
    
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
        go.Scatter(x=w, y=20*np.log10(mag), mode='lines', name='Magnitude', line=dict(color='blue')),
        row=1, col=1
    )
    
    if not np.isnan(gm) and not np.isnan(wg):
        fig_bode.add_vline(x=wg, line_dash="dash", line_color="red", annotation_text=f"GM: {20*np.log10(gm):.1f} dB", row=1, col=1)
    
    fig_bode.add_hline(y=0, line_dash="dot", line_color="gray", row=1, col=1)
    
    fig_bode.add_trace(
        go.Scatter(x=w, y=np.degrees(phase), mode='lines', name='Phase', line=dict(color='green')),
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
            st.metric("Gain Margin", f"{20*np.log10(gm):.2f} dB")
        else:
            st.metric("Gain Margin", "∞")
    with col6:
        if not np.isnan(pm):
            st.metric("Phase Margin", f"{pm:.2f}°")
        else:
            st.metric("Phase Margin", "N/A")
    with col7:
        if not np.isnan(wp):
            st.metric("Crossover Frequency", f"{wp:.3f} rad/s")
        else:
            st.metric("Crossover Frequency", "N/A")
    
    st.header("Stability Analysis")
    
    poles = ctrl.pole(G)
    zeros = ctrl.zero(G)
    
    fig_pz = go.Figure()
    
    if len(poles) > 0:
        fig_pz.add_trace(go.Scatter(
            x=[p.real for p in poles],
            y=[p.imag for p in poles],
            mode='markers',
            name='Poles',
            marker=dict(symbol='x', size=12, color='red', line=dict(width=2))
        ))
    
    if len(zeros) > 0:
        fig_pz.add_trace(go.Scatter(
            x=[z.real for z in zeros],
            y=[z.imag for z in zeros],
            mode='markers',
            name='Zeros',
            marker=dict(symbol='circle', size=10, color='blue', line=dict(width=2))
        ))
    
    fig_pz.add_hline(y=0, line_dash="dot", line_color="gray")
    fig_pz.add_vline(x=0, line_dash="dot", line_color="gray")
    
    max_val = max(max(abs(p.real) for p in poles) if poles else 1, 
                  max(abs(p.imag) for p in poles) if poles else 1,
                  max(abs(z.real) for z in zeros) if zeros else 1,
                  max(abs(z.imag) for z in zeros) if zeros else 1) * 1.2
    
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
        for i, pole in enumerate(poles):
            st.write(f"Pole {i+1}: {pole:.3f}")
    
    with col9:
        st.subheader("System Zeros")
        for i, zero in enumerate(zeros):
            st.write(f"Zero {i+1}: {zero:.3f}")
    
    stable = all(p.real < 0 for p in poles)
    if stable:
        st.success("✅ System is STABLE")
    else:
        st.error("❌ System is UNSTABLE")

except Exception as e:
    st.error(f"Error: {str(e)}")
    st.stop()
