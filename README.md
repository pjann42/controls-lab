# Control Systems Laboratory

An interactive web tool for analyzing linear time-invariant (LTI) control systems under unity feedback. Built with Python and Streamlit.

**[Launch the app](https://controls-lab-n3q4fdygrll6th7wuve5xh.streamlit.app/)**

> **Note:** This project is under active development. Some features may be incomplete, and the tool has not yet undergone full validation testing.

---

## About

I am an undergraduate Control Engineering student and this is a personal project I have been developing to support the study and analysis of control systems. The goal is to provide an accessible, browser-based environment where transfer functions can be quickly analyzed without the need for MATLAB or similar proprietary software.

The tool is aimed at students and practitioners who want to explore system behavior — stability, transient response, and frequency characteristics — in an intuitive way.

---

## Features

- **Transfer function input** — custom numerator/denominator coefficients or canonical second-order system parameters (ζ, ωn)
- **Optional controller C(s)** — series connection with the plant G(s)
- **Stability analysis** — pole classification for plant, controller, open-loop, and closed-loop systems
- **Pole-zero maps** — interactive plots for open-loop and closed-loop systems
- **Time domain analysis** — unit step response with metrics (rise time, settling time, overshoot, peak, steady-state)
- **Frequency domain analysis** — Bode magnitude and phase plots with gain and phase margins
- **Mobile-friendly** — responsive layout that adapts to smaller screens

---

## Tech Stack

| Library | Purpose |
|---|---|
| [Streamlit](https://streamlit.io) | Web UI framework |
| [python-control](https://python-control.readthedocs.io) | Control systems computations |
| [Plotly](https://plotly.com/python/) | Interactive charts |
| [NumPy](https://numpy.org) | Numerical computation |

---

## Getting Started

### Prerequisites

- Python 3.9+
- pip or conda

### Installation

```bash
git clone https://github.com/pjann42/Control-Systems-Lab.git
cd Control-Systems-Lab
pip install -r requirements.txt
```

### Running

```bash
python -m streamlit run control_systems_lab.py
```

Then open `http://localhost:8501` in your browser.

---

## Usage

1. Select the **system mode**: custom transfer function or canonical second-order
2. Enter the **plant G(s)** coefficients (comma-separated, highest power first)
3. Optionally enable and configure a **controller C(s)**
4. Click **Calculate** to run the full analysis

---

## Known Limitations & Roadmap

This tool is still a work in progress. Areas that need further development:

- [ ] Validation testing against established references (MATLAB, Scilab)
- [ ] Root locus plot
- [ ] Nyquist diagram
- [ ] Support for state-space representation
- [ ] PID controller design assistant
- [ ] Improved handling of edge cases (repeated poles, pure integrators)
- [ ] Export results to PDF or CSV

---

## Disclaimer

This tool is provided as-is for educational purposes. Results should be verified against established software before use in any engineering application. The author assumes no responsibility for errors or inaccuracies.

---

## Author

Developed by **Pedro** — Control Engineering undergraduate student.

Feel free to open an issue or submit a pull request if you find a bug or want to contribute.
