"""
Interactive GUI for EEG Synthesis Framework - Enhanced Version

Run with: streamlit run gui_app.py
"""

from __future__ import annotations

import streamlit as st
import numpy as np
import io
import json

from pipeline.orchestrator import Pipeline
from core.types import BandSpec
from blocks.generators import BandGenerator, ColoredNoise
from blocks.envelopes import BurstyEnvelope
from blocks.transforms import PACModulate
from blocks.spatial import Reference
from blocks.artifacts import EOGArtifacts, EMGArtifacts, LineNoise, BaselineDrift
from blocks.filters import BandpassFilter, NotchFilter, LowpassFilter
from blocks.observables import MultiBandPower, ValidationChecks, BandSignalExtractor, TapSignal
from eeg_io.export import (
    plot_psd, plot_timeseries, plot_stacked_channels,
    plot_band_powers, plot_filtered_comparison, plot_band_filtered_signals
)

st.set_page_config(page_title="EEG Synthesis", page_icon="brain", layout="wide")

# Initialize session state
if "generated" not in st.session_state:
    st.session_state.generated = False

# Title
st.title("EEG Synthesis Framework - Enhanced")
st.markdown("**Interactive brainwave generator with advanced preprocessing and analysis**")

# Sidebar — Global Settings
st.sidebar.title("Global Settings")
seed = st.sidebar.number_input("Random seed", value=42, step=1)
n_channels = st.sidebar.slider("Channels", 1, 32, 8, 1)
sfreq = st.sidebar.selectbox("Sampling rate (Hz)", [250.0, 500.0, 1000.0], index=1)
duration = st.sidebar.slider("Duration (s)", 5, 120, 30, 5)

# Preprocessing options
st.sidebar.markdown("---")
st.sidebar.title("Preprocessing")
enable_bandpass = st.sidebar.checkbox("Bandpass filter", value=False)
if enable_bandpass:
    bp_low = st.sidebar.slider("Bandpass low (Hz)", 0.1, 20.0, 0.5, 0.1)
    bp_high = st.sidebar.slider("Bandpass high (Hz)", 20.0, 100.0, 45.0, 1.0)

enable_notch = st.sidebar.checkbox("Notch filter (line noise)", value=False)
if enable_notch:
    notch_freq = st.sidebar.selectbox("Notch frequency", [50.0, 60.0], index=1)

enable_lowpass = st.sidebar.checkbox("Lowpass filter (smoothing)", value=False)
if enable_lowpass:
    lp_cutoff = st.sidebar.slider("Lowpass cutoff (Hz)", 5.0, 50.0, 20.0, 1.0)

# Analysis options
st.sidebar.markdown("---")
st.sidebar.title("Analysis")
enable_bandpower = st.sidebar.checkbox("Multi-band power analysis", value=True)
enable_validation = st.sidebar.checkbox("Signal validation", value=True)
enable_band_extraction = st.sidebar.checkbox("Extract band signals", value=True)


# Band amplitudes 
delta_amp: float = 0.0
theta_amp: float = 0.0
alpha_amp: float = 0.0
sigma_amp: float = 0.0
beta_amp: float = 0.0
gamma_amp: float = 0.0

# Envelope / PAC defaults
burst_rate: float = 12.0
burst_dur: float = 2.0
pac_strength: float = 0.0

# Artifact defaults
eog_rate: float = 15.0
emg_rate: float = 8.0
line_amp: float = 0.0

# Preprocessing parameter defaults (avoid "possibly unbound" warnings)
notch_freq: float = 60.0
bp_low: float = 0.5
bp_high: float = 45.0
lp_cutoff: float = 20.0

# Pipeline object for the exception handler
pipe = None


# Band configuration
st.header("Frequency Bands")
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Delta (δ)")
    delta_enable = st.checkbox("Enable Delta", value=False)
    if delta_enable:
        delta_amp = st.slider("Delta amplitude (μV)", 1.0, 50.0, 20.0, 1.0)

with col2:
    st.subheader("Theta (θ)")
    theta_enable = st.checkbox("Enable Theta", value=False)
    if theta_enable:
        theta_amp = st.slider("Theta amplitude (μV)", 1.0, 50.0, 15.0, 1.0)

with col3:
    st.subheader("Alpha (α)")
    alpha_enable = st.checkbox("Enable Alpha", value=True)
    if alpha_enable:
        alpha_amp = st.slider("Alpha amplitude (μV)", 1.0, 50.0, 25.0, 1.0)

col4, col5, col6 = st.columns(3)

with col4:
    st.subheader("Beta (β)")
    beta_enable = st.checkbox("Enable Beta", value=True)
    if beta_enable:
        beta_amp = st.slider("Beta amplitude (μV)", 1.0, 50.0, 8.0, 1.0)

with col5:
    st.subheader("Gamma (γ)")
    gamma_enable = st.checkbox("Enable Gamma", value=False)
    if gamma_enable:
        gamma_amp = st.slider("Gamma amplitude (μV)", 1.0, 20.0, 5.0, 1.0)

with col6:
    st.subheader("Sigma (σ)")
    sigma_enable = st.checkbox("Enable Sigma", value=False)
    if sigma_enable:
        sigma_amp = st.slider("Sigma amplitude (μV)", 1.0, 30.0, 10.0, 1.0)

# Envelope
st.header("Envelope Modulation")
alpha_env = st.selectbox("Alpha envelope", ["constant", "bursty"], index=1 if alpha_enable else 0)
if alpha_env == "bursty":
    col1, col2 = st.columns(2)
    with col1:
        burst_rate = st.slider("Bursts/min", 1.0, 30.0, 12.0, 1.0)
    with col2:
        burst_dur = st.slider("Duration (s)", 0.5, 5.0, 2.0, 0.1)

# PAC
st.header("Phase-Amplitude Coupling")
pac_enable = st.checkbox("Enable PAC (Theta-Gamma)", value=False)
if pac_enable:
    pac_strength = st.slider("PAC strength", 0.0, 1.0, 0.6, 0.05)

# Artifacts
st.header("Artifacts & Noise")
col1, col2, col3, col4 = st.columns(4)

with col1:
    eog_enable = st.checkbox("EOG (blinks)", value=True)
    if eog_enable:
        eog_rate = st.slider("EOG rate/min", 5.0, 40.0, 20.0, 1.0)

with col2:
    emg_enable = st.checkbox("EMG (muscle)", value=True)
    if emg_enable:
        emg_rate = st.slider("EMG rate/min", 2.0, 30.0, 10.0, 1.0)

with col3:
    line_noise_enable = st.checkbox("Line noise", value=False)
    if line_noise_enable:
        line_amp = st.slider("Line noise (μV)", 0.5, 10.0, 2.0, 0.5)

with col4:
    noise_level = st.slider("Pink noise (μV)", 0.0, 10.0, 2.0, 0.5)

# Generate button
if st.button("Generate EEG", type="primary", use_container_width=True):
    with st.spinner("Generating synthetic EEG..."):
        try:
            # Build pipeline
            pipe = Pipeline(backend="numpy", seed=int(seed))

            first = True

            # Delta
            if delta_enable:
                delta = BandSpec("delta", 0.5, 4.0, delta_amp, 2)
                pipe.add("delta", BandGenerator(delta), save_component=True,
                         accumulate="replace" if first else "add")
                first = False

            # Theta (needed for PAC)
            if theta_enable or pac_enable:
                theta_amp_use = theta_amp if theta_enable else 15.0
                theta = BandSpec("theta", 4.0, 8.0, theta_amp_use, 2)
                pipe.add("theta", BandGenerator(theta), save_component=True,
                         accumulate="replace" if first else "add")
                first = False

            # Alpha
            if alpha_enable:
                alpha = BandSpec("alpha", 8.0, 13.0, alpha_amp, 3)
                pipe.add("alpha", BandGenerator(alpha), save_component=True,
                         accumulate="replace" if first else "add")

                if alpha_env == "bursty":
                    pipe.add(
                        "alpha_env",
                        BurstyEnvelope(
                            rate_per_min=float(burst_rate),
                            duration_sec=float(burst_dur),
                            normalize_post=True,
                            target_rms_uv=float(alpha_amp),
                        ),
                    )
                first = False

            # Sigma
            if sigma_enable:
                sigma = BandSpec("sigma", 12.0, 16.0, sigma_amp, 2)
                pipe.add("sigma", BandGenerator(sigma), save_component=True,
                         accumulate="replace" if first else "add")
                first = False

            # Beta
            if beta_enable:
                beta = BandSpec("beta", 13.0, 30.0, beta_amp, 2)
                pipe.add("beta", BandGenerator(beta), save_component=True,
                         accumulate="replace" if first else "add")
                first = False

            # Gamma + PAC modulation
            if gamma_enable or pac_enable:
                gamma_amp_use = gamma_amp if gamma_enable else 5.0
                gamma = BandSpec("gamma", 30.0, 80.0, gamma_amp_use, 3)
                pipe.add("gamma", BandGenerator(gamma), save_component=True, accumulate="add")
                
                if pac_enable:
                    pipe.add("pac", PACModulate("theta", "gamma", strength=float(pac_strength)))

            # Noise
            if noise_level > 0:
                pipe.add("noise", ColoredNoise(beta=1.0, rms_uv=float(noise_level)), accumulate="add")

            # Artifacts
            if eog_enable:
                pipe.add("eog", EOGArtifacts(rate_per_min=float(eog_rate)))
            if emg_enable:
                pipe.add("emg", EMGArtifacts(rate_per_min=float(emg_rate)))
            if line_noise_enable:
                pipe.add("line_noise", LineNoise(amplitude_uv=float(line_amp)))

            # Tap signal before preprocessing for before/after comparison
            any_preprocessing = enable_notch or enable_bandpass or enable_lowpass
            if any_preprocessing:
                pipe.add("pre_filter", TapSignal("pre_filter"))

            # Preprocessing
            if enable_notch:
                pipe.add("notch_filter", NotchFilter(freq_hz=float(notch_freq)))
            
            if enable_bandpass:
                pipe.add("bandpass_filter", BandpassFilter(band_hz=(float(bp_low), float(bp_high))))
            
            if enable_lowpass:
                pipe.add("lowpass_filter", LowpassFilter(cutoff_hz=float(lp_cutoff)))

            # Analysis
            if enable_bandpower:
                pipe.add("multiband_power", MultiBandPower())
            
            if enable_validation:
                pipe.add("validation", ValidationChecks())
            
            if enable_band_extraction:
                pipe.add("band_extractor", BandSignalExtractor())

            # Run pipeline
            result = pipe.run(duration=float(duration), sfreq=float(sfreq), n_channels=int(n_channels))
            raw = pipe.get_mne_raw(result)

            # Store in session state
            st.session_state.result = result
            st.session_state.raw = raw
            st.session_state.generated = True
            
            # Store run configuration for consistent display
            st.session_state.run_config = {
                "enable_validation": enable_validation,
                "enable_bandpower": enable_bandpower,
                "enable_band_extraction": enable_band_extraction,
                "enable_lowpass": enable_lowpass,
                "enable_bandpass": enable_bandpass,
                "enable_notch": enable_notch,
                "any_preprocessing": any_preprocessing,
            }

            st.success("EEG generated successfully!")
            st.rerun()

        except Exception as e:
            st.error(f"Error during generation: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
            
            # Show pipeline configuration for debugging
            if pipe is not None:
                st.markdown("### Pipeline Configuration")
                st.write("Steps added:")
                for step in pipe.steps:
                    st.write(f"- {step.name}: {step.block.__class__.__name__} (accumulate={step.accumulate})")

# Display results
if st.session_state.generated:
    st.markdown("---")
    st.header("Results")

    result = st.session_state.result
    raw = st.session_state.raw
    cfg = st.session_state.get("run_config", {})

    # Metrics
    dur_sec = float(len(result['times'])) / float(result['sfreq'])
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Duration", f"{dur_sec:.1f}s")
    col2.metric("Channels", int(result['n_channels']))
    col3.metric("Sampling Rate", f"{result['sfreq']:.0f} Hz")
    col4.metric("Events", len(result['events']))

    # Validation metrics if available
    if cfg.get("enable_validation", False):
        for key, value in result['metadata'].items():
            if isinstance(value, dict) and 'check_rms_uv_mean' in value:
                st.markdown("### Signal Quality Validation")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("RMS (μV)", f"{value['check_rms_uv_mean']:.2f}")
                col2.metric("Peak-to-Peak (μV)", f"{value['check_p2p_uv_mean']:.2f}")
                col3.metric("RMS OK %", f"{value['check_rms_uv_ok_frac']*100:.1f}%")
                col4.metric("P2P OK %", f"{value['check_p2p_uv_ok_frac']*100:.1f}%")
                break

    st.markdown("---")

    # Tabs for different visualizations
    tabs = st.tabs(["Overview", "Band Analysis", "Detailed Plots", "Download"])

    with tabs[0]:
        # PSD Plot
        st.subheader("Power Spectral Density")
        fig_psd = plot_psd(raw, fmin=0.5, fmax=100.0)
        st.pyplot(fig_psd, clear_figure=True)

        # Time Series
        st.subheader("Time Series (Channel 1)")
        fig_ts = plot_timeseries(result['signal'], result['times'], channel_idx=0)
        st.pyplot(fig_ts, clear_figure=True)

    with tabs[1]:
        # Band power plot
        if cfg.get("enable_bandpower", False):
            st.subheader("Power in EEG Bands")
            for key, value in result['metadata'].items():
                if isinstance(value, dict) and "band_powers" in value:
                    fig_bp = plot_band_powers(value["band_powers"])
                    st.pyplot(fig_bp, clear_figure=True)
                    
                    # Show table in canonical order
                    st.markdown("#### Band Power Values")
                    import pandas as pd
                    
                    # Canonical band order
                    canonical_order = ["delta", "theta", "alpha", "sigma", "beta", "gamma"]
                    bands_present = [b for b in canonical_order if b in value["band_powers"]]
                    
                    bp_df = pd.DataFrame({
                        'Band': [k.capitalize() for k in bands_present],
                        'Mean Power': [value["band_powers"][k]["mean"] for k in bands_present]
                    })
                    st.dataframe(bp_df, use_container_width=True)
                    break
        
        # Band filtered signals
        if cfg.get("enable_band_extraction", False) and "components" in result:
            st.subheader("Filtered Signals by Band")
            band_signals = {k: v for k, v in result["components"].items() if "_signal" in k}
            if band_signals:
                fig_bands = plot_band_filtered_signals(band_signals, result['times'], channel_idx=0)
                st.pyplot(fig_bands, clear_figure=True)

    with tabs[2]:
        # Stacked Channels
        st.subheader("All Channels (Stacked)")
        fig_stack = plot_stacked_channels(result['signal'], result['times'], offset_uv=150.0)
        st.pyplot(fig_stack, clear_figure=True)

        # Filtered comparison if preprocessing was applied
        if cfg.get("any_preprocessing", False) and "pre_filter" in result.get("components", {}):
            st.subheader("Original vs Filtered Comparison")
            original = result["components"]["pre_filter"]
            filtered = result["signal"]
            fig_comp = plot_filtered_comparison(original, filtered, result['times'], 
                                               channel_idx=0, title="Signal Filtering Effect")
            st.pyplot(fig_comp, clear_figure=True)

    with tabs[3]:
        # Downloads
        st.subheader("Download Results")

        col1, col2, col3 = st.columns(3)

        with col1:
            # CSV
            ch_names = [f"EEG{i+1:02d}" for i in range(result['n_channels'])]
            header = "time," + ",".join(ch_names)
            data = np.column_stack([result['times'], result['signal'].T])
            csv_buf = io.StringIO()
            np.savetxt(csv_buf, data, delimiter=",", header=header, comments="", fmt="%.9e")
            st.download_button(
                "CSV",
                data=csv_buf.getvalue().encode("utf-8"),
                file_name="synthetic_eeg.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with col2:
            # Events
            if result['events']:
                st.download_button(
                    "Events JSON",
                    data=json.dumps(result['events'], indent=2).encode("utf-8"),
                    file_name="events.json",
                    mime="application/json",
                    use_container_width=True,
                )

        with col3:
            # Metadata with proper JSON serialization
            def _jsonify_meta(obj):
                import numpy as np
                if isinstance(obj, np.generic):
                    return obj.item()
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                try:
                    json.dumps(obj)
                    return obj
                except (TypeError, ValueError):
                    return str(obj)
            
            st.download_button(
                "Metadata",
                data=json.dumps(result['metadata'], indent=2, default=_jsonify_meta).encode("utf-8"),
                file_name="metadata.json",
                mime="application/json",
                use_container_width=True,
            )

else:
    st.info("Configure parameters in the sidebar and click **Generate EEG** to start")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 20px;'>
    <p><strong>EEG Synthesis Framework v2.1</strong></p>
    <p>Enhanced with preprocessing, multi-band analysis, and validation tools</p>
</div>
""", unsafe_allow_html=True)




