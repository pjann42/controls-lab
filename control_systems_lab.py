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
        zeta_input = st.text_input("Damping Ratio (ζ)", "0.5", help="Enter damping ratio value")
        wn_input = st.text_input("Natural Frequency (ωn)", "1.0", help="Enter natural frequency value")
        
        try:
            zeta = float(zeta_input.strip())
            wn = float(wn_input.strip())
            
            if zeta < 0:
                st.error("Damping ratio must be non-negative")
                st.stop()
            if wn <= 0:
                st.error("Natural frequency must be positive")
                st.stop()
                
            num = [wn**2]
            den = [1, 2*zeta*wn, wn**2]
        except ValueError:
            st.error("Invalid input format. Please enter numeric values.")
            st.stop()
    
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
    
    st.header("Stability Analysis")
    st.caption("Using Closed-Loop Transfer Function T(s)")
    
    poles = ctrl.pole(T)  # Use closed-loop poles for stability analysis
    zeros = ctrl.zero(T)  # Use closed-loop zeros
    
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
    
    # Create tabs for Open-Loop and Closed-Loop pole-zero maps
    tab1, tab2 = st.tabs(["Open-Loop Pole-Zero Map", "Closed-Loop Pole-Zero Map"])
    
    with tab1:
        st.subheader("Open-Loop System G(s)")
        
        # Open-loop poles and zeros
        ol_poles = ctrl.pole(G)
        ol_zeros = ctrl.zero(G)
        
        fig_pz_ol = go.Figure()
        
        if ol_poles.size > 0:
            fig_pz_ol.add_trace(go.Scatter(
                x=[p.real for p in ol_poles],
                y=[p.imag for p in ol_poles],
                mode='markers',
                name='Open-Loop Poles',
                marker=dict(symbol='x', size=12, color='red', line=dict(width=2))
            ))
        
        if ol_zeros.size > 0:
            fig_pz_ol.add_trace(go.Scatter(
                x=[z.real for z in ol_zeros],
                y=[z.imag for z in ol_zeros],
                mode='markers',
                name='Open-Loop Zeros',
                marker=dict(symbol='circle', size=10, color='blue', line=dict(width=2))
            ))
        
        fig_pz_ol.add_hline(y=0, line_dash="dot", line_color="gray")
        fig_pz_ol.add_vline(x=0, line_dash="dot", line_color="gray")
        
        max_val_ol = max(max(abs(p.real) for p in ol_poles) if ol_poles.size > 0 else 1, 
                         max(abs(p.imag) for p in ol_poles) if ol_poles.size > 0 else 1,
                         max(abs(z.real) for z in ol_zeros) if ol_zeros.size > 0 else 1,
                         max(abs(z.imag) for z in ol_zeros) if ol_zeros.size > 0 else 1) * 1.2
        
        fig_pz_ol.update_layout(
            title="Open-Loop Pole-Zero Map",
            xaxis_title="Real Axis",
            yaxis_title="Imaginary Axis",
            xaxis=dict(range=[-max_val_ol, max_val_ol]),
            yaxis=dict(range=[-max_val_ol, max_val_ol]),
            template='plotly_white',
            showlegend=True
        )
        
        st.plotly_chart(fig_pz_ol, use_container_width=True)
        
        col8, col9 = st.columns(2)
        with col8:
            st.subheader("Open-Loop Poles")
            if ol_poles.size > 0:
                for i, pole in enumerate(ol_poles):
                    st.write(f"Pole {i+1}: {pole:.3f}")
            else:
                st.write("No poles")
        
        with col9:
            st.subheader("Open-Loop Zeros")
            if ol_zeros.size > 0:
                for i, zero in enumerate(ol_zeros):
                    st.write(f"Zero {i+1}: {zero:.3f}")
            else:
                st.write("No zeros")
    
    with tab2:
        st.subheader("Closed-Loop System T(s)")
        
        # Closed-loop poles and zeros (for stability analysis)
        cl_poles = ctrl.pole(T)
        cl_zeros = ctrl.zero(T)
        
        fig_pz_cl = go.Figure()
        
        if cl_poles.size > 0:
            fig_pz_cl.add_trace(go.Scatter(
                x=[p.real for p in cl_poles],
                y=[p.imag for p in cl_poles],
                mode='markers',
                name='Closed-Loop Poles',
                marker=dict(symbol='x', size=12, color='darkred', line=dict(width=2))
            ))
        
        if cl_zeros.size > 0:
            fig_pz_cl.add_trace(go.Scatter(
                x=[z.real for z in cl_zeros],
                y=[z.imag for z in cl_zeros],
                mode='markers',
                name='Closed-Loop Zeros',
                marker=dict(symbol='circle', size=10, color='darkblue', line=dict(width=2))
            ))
        
        fig_pz_cl.add_hline(y=0, line_dash="dot", line_color="gray")
        fig_pz_cl.add_vline(x=0, line_dash="dot", line_color="gray")
        
        max_val_cl = max(max(abs(p.real) for p in cl_poles) if cl_poles.size > 0 else 1, 
                         max(abs(p.imag) for p in cl_poles) if cl_poles.size > 0 else 1,
                         max(abs(z.real) for z in cl_zeros) if cl_zeros.size > 0 else 1,
                         max(abs(z.imag) for z in cl_zeros) if cl_zeros.size > 0 else 1) * 1.2
        
        fig_pz_cl.update_layout(
            title="Closed-Loop Pole-Zero Map",
            xaxis_title="Real Axis",
            yaxis_title="Imaginary Axis",
            xaxis=dict(range=[-max_val_cl, max_val_cl]),
            yaxis=dict(range=[-max_val_cl, max_val_cl]),
            template='plotly_white',
            showlegend=True
        )
        
        st.plotly_chart(fig_pz_cl, use_container_width=True)
        
        col10, col11 = st.columns(2)
        with col10:
            st.subheader("Closed-Loop Poles")
            if cl_poles.size > 0:
                for i, pole in enumerate(cl_poles):
                    st.write(f"Pole {i+1}: {pole:.3f}")
            else:
                st.write("No poles")
        
        with col11:
            st.subheader("Closed-Loop Zeros")
            if cl_zeros.size > 0:
                for i, zero in enumerate(cl_zeros):
                    st.write(f"Zero {i+1}: {zero:.3f}")
            else:
                st.write("No zeros")
    
    # Only show Time Domain Analysis for stable systems
    if stability_class != "Unstable":
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
    else:
        st.warning("⚠️ Time Domain Analysis skipped for unstable systems")
        st.caption("Step response analysis is not meaningful for unstable systems")
    
    st.header("Frequency Domain Analysis")
    st.caption("Using Open-Loop Transfer Function G(s)")
    
    # Generate frequency response data using proper method
    w = np.logspace(-2, 2, 1000)  # 0.01 to 100 rad/s
    
    # Get transfer function coefficients for manual calculation
    G_num, G_den = ctrl.tfdata(G)
    G_num = G_num[0][0]
    G_den = G_den[0][0]
    
    # Manual frequency response calculation
    s = 1j * w  # s = jω
    G_jw_num = np.polyval(G_num, s)
    G_jw_den = np.polyval(G_den, s)
    G_jw = G_jw_num / G_jw_den
    
    # Convert magnitude to dB
    mag_abs = np.abs(G_jw)
    mag_db = 20 * np.log10(mag_abs)
    
    # Calculate phase using atan2 for correct quadrant logic
    phase_rad = np.arctan2(np.imag(G_jw), np.real(G_jw))
    
    # Phase unwrapping to remove 2π discontinuities
    phase_unwrapped = np.unwrap(phase_rad)
    
    # Convert to degrees
    phase_deg = np.degrees(phase_unwrapped)
    
    # Smart autoscaling function for Bode plots
    def smart_autoscale(y_data, padding_factor=0.1, steady_state_threshold=0.01):
        """
        Calculate smart Y-axis limits that avoid jittering in steady-state regions.
        
        Args:
            y_data: Data array for axis scaling
            padding_factor: Percentage of padding around data (default 10%)
            steady_state_threshold: Threshold for detecting steady-state (default 1%)
        
        Returns:
            tuple: (y_min, y_max) smart axis limits
        """
        y_min, y_max = np.min(y_data), np.max(y_data)
        data_range = y_max - y_min
        
        # Detect steady-state regions (small changes)
        y_diff = np.abs(np.diff(y_data))
        steady_state_mask = y_diff < (data_range * steady_state_threshold)
        
        if np.sum(steady_state_mask) > len(y_data) * 0.3:  # If 30%+ is steady-state
            # Use extended range with hysteresis for steady-state regions
            extended_range = data_range * (1 + padding_factor * 2)
            center = (y_min + y_max) / 2
            y_min_smart = center - extended_range / 2
            y_max_smart = center + extended_range / 2
        else:
            # Normal padding for dynamic regions
            y_padding = data_range * padding_factor
            y_min_smart = y_min - y_padding
            y_max_smart = y_max + y_padding
        
        # Add minimum range constraints for visual clarity
        if data_range < 5:  # For very small ranges
            y_min_smart = y_min - 2
            y_max_smart = y_max + 2
        
        return y_min_smart, y_max_smart
    
    # Calculate smart axis limits
    mag_ymin, mag_ymax = smart_autoscale(mag_db, padding_factor=0.15, steady_state_threshold=0.02)
    phase_ymin, phase_ymax = smart_autoscale(phase_deg, padding_factor=0.15, steady_state_threshold=0.02)
    
    # Force phase axis to strict 45-degree intervals
    def align_phase_axis_45_deg(phase_min, phase_max):
        """
        Align phase axis limits to strict 45-degree intervals.
        
        Args:
            phase_min: Minimum phase value in degrees
            phase_max: Maximum phase value in degrees
        
        Returns:
            tuple: (aligned_min, aligned_max, tick_values)
        """
        # Extend range to ensure 45-degree coverage
        range_extension = 45  # One extra 45-degree step on each side
        
        # Snap min down to nearest 45-degree multiple
        aligned_min = round((phase_min - range_extension) / 45) * 45
        
        # Snap max up to nearest 45-degree multiple
        aligned_max = round((phase_max + range_extension) / 45) * 45
        
        # Generate tick marks at exact 45-degree intervals
        tick_values = np.arange(aligned_min, aligned_max + 45, 45)
        
        return aligned_min, aligned_max, tick_values.tolist()
    
    # Apply 45-degree alignment to phase axis
    phase_ymin_aligned, phase_ymax_aligned, phase_ticks_45 = align_phase_axis_45_deg(phase_ymin, phase_ymax)
    
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
    fig_bode.update_yaxes(title_text="Magnitude (dB)", row=1, col=1, range=[mag_ymin, mag_ymax])
    fig_bode.update_yaxes(title_text="Phase (degrees)", row=2, col=1, 
                         range=[phase_ymin_aligned, phase_ymax_aligned],
                         tickmode='array',
                         tickvals=phase_ticks_45,
                         ticktext=[f"{int(tick)}°" for tick in phase_ticks_45])
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

except Exception as e:
    st.error(f"Error: {str(e)}")
    st.stop()
