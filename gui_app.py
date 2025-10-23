"""
Interactive GUI for EEG Synthesis Framework

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
from eeg_io.export import plot_psd, plot_timeseries, plot_stacked_channels

st.set_page_config(page_title="EEG Synthesis", page_icon="🧠", layout="wide")

# Initialize session state
if "generated" not in st.session_state:
    st.session_state.generated = False

# Title
st.title("EEG Synthesis Framework")
st.markdown("**Interactive brainwave generator**")

# Sidebar — Global Settings
st.sidebar.title("Global Settings")
seed = st.sidebar.number_input("Random seed", value=42, step=1)
n_channels = st.sidebar.slider("Channels", 1, 32, 8, 1)
sfreq = st.sidebar.selectbox("Sampling rate (Hz)", [250.0, 500.0, 1000.0], index=1)
duration = st.sidebar.slider("Duration (s)", 5, 120, 30, 5)

# ------------------------------
# Predefine safe defaults (silence Pylance "possibly unbound")
# ------------------------------

alpha_amp: float = 0.0
beta_amp: float = 0.0
burst_rate: float = 12.0
burst_dur: float = 2.0
pac_strength: float = 0.0
eog_rate: float = 15.0
emg_rate: float = 8.0

# Band configuration
st.header("Frequency Bands")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Alpha (α)")
    alpha_enable = st.checkbox("Enable Alpha", value=True)
    if alpha_enable:
        alpha_amp = st.slider("Alpha amplitude (μV)", 1.0, 50.0, 25.0, 1.0)

with col2:
    st.subheader("Beta (β)")
    beta_enable = st.checkbox("Enable Beta", value=True)
    if beta_enable:
        beta_amp = st.slider("Beta amplitude (μV)", 1.0, 50.0, 8.0, 1.0)

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
col1, col2, col3 = st.columns(3)

with col1:
    eog_enable = st.checkbox("EOG (blinks)", value=True)
    if eog_enable:
        eog_rate = st.slider("EOG rate/min", 5.0, 40.0, 20.0, 1.0)

with col2:
    emg_enable = st.checkbox("EMG (muscle)", value=True)
    if emg_enable:
        emg_rate = st.slider("EMG rate/min", 2.0, 30.0, 10.0, 1.0)

with col3:
    noise_level = st.slider("Noise (μV)", 0.0, 10.0, 2.0, 0.5)

# Generate button
if st.button("Generate EEG", type="primary", use_container_width=True):
    with st.spinner("Generating synthetic EEG..."):
        try:
            # Build pipeline
            pipe = Pipeline(backend="numpy", seed=int(seed))

            first = True

            # PAC: add theta carrier first
            if pac_enable:
                theta = BandSpec("theta", 4.0, 8.0, 15.0, 2)
                pipe.add("theta", BandGenerator(theta), save_component=True,
                         accumulate="replace" if first else "add")
                first = False

            # Alpha
            if alpha_enable:
                alpha = BandSpec("alpha", 8.0, 12.0, alpha_amp, 3)
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

            # Beta
            if beta_enable:
                beta = BandSpec("beta", 13.0, 30.0, beta_amp, 2)
                pipe.add("beta", BandGenerator(beta), save_component=True,
                         accumulate="replace" if first else "add")
                first = False

            # Gamma + PAC modulation
            if pac_enable:
                gamma = BandSpec("gamma", 30.0, 80.0, 5.0, 3)
                pipe.add("gamma", BandGenerator(gamma), save_component=True, accumulate="add")
                pipe.add("pac", PACModulate("theta", "gamma", strength=float(pac_strength)))

            # Noise
            if noise_level > 0:
                pipe.add("noise", ColoredNoise(beta=1.0, rms_uv=float(noise_level)), accumulate="add")

            # Artifacts
            if eog_enable:
                pipe.add("eog", EOGArtifacts(rate_per_min=float(eog_rate)))
            if emg_enable:
                pipe.add("emg", EMGArtifacts(rate_per_min=float(emg_rate)))

            # Run pipeline
            result = pipe.run(duration=float(duration), sfreq=float(sfreq), n_channels=int(n_channels))
            raw = pipe.get_mne_raw(result)

            # Store in session state
            st.session_state.result = result
            st.session_state.raw = raw
            st.session_state.generated = True

            st.success("✓ EEG generated successfully!")
            st.rerun()

        except Exception as e:
            st.error(f"Error during generation: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

# Display results
if st.session_state.generated:
    st.markdown("---")
    st.header("Results")

    result = st.session_state.result
    raw = st.session_state.raw

    # Metrics (compute duration from times to avoid relying on metadata)
    dur_sec = float(len(result['times'])) / float(result['sfreq'])
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Duration", f"{dur_sec:.1f}s")
    col2.metric("Channels", int(result['n_channels']))
    col3.metric("Sampling Rate", f"{result['sfreq']:.0f} Hz")
    col4.metric("Events", len(result['events']))

    st.markdown("---")

    # PSD Plot
    st.subheader("Power Spectral Density")
    fig_psd = plot_psd(raw, fmin=0.5, fmax=100.0)
    st.pyplot(fig_psd, clear_figure=True)

    # Time Series
    st.subheader("Time Series (Channel 1)")
    fig_ts = plot_timeseries(result['signal'], result['times'], channel_idx=0)
    st.pyplot(fig_ts, clear_figure=True)

    # Stacked Channels
    st.subheader("All Channels (Stacked)")
    fig_stack = plot_stacked_channels(result['signal'], result['times'], offset_uv=150.0)
    st.pyplot(fig_stack, clear_figure=True)

    # Downloads
    st.markdown("---")
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
        # Metadata
        st.download_button(
            "ℹ Metadata",
            data=json.dumps(result['metadata'], indent=2, default=str).encode("utf-8"),
            file_name="metadata.json",
            mime="application/json",
            use_container_width=True,
        )

else:
    st.info("Configure parameters above and click **Generate EEG**")




