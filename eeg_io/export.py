"""
Export and I/O utilities for synthetic EEG data.
"""

import numpy as np
import json
import os
from typing import Dict, Any, List, Optional, Tuple
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def save_to_csv(
    signal: np.ndarray,
    times: np.ndarray,
    filename: str,
    ch_names: Optional[List[str]] = None
):
    """Save signal to CSV file"""
    n_channels = signal.shape[0]
    
    if ch_names is None:
        ch_names = [f"EEG{i+1:02d}" for i in range(n_channels)]
    
    header = "time," + ",".join(ch_names)
    data = np.column_stack([times, signal.T])
    
    np.savetxt(filename, data, delimiter=",", header=header, comments="", fmt="%.9e")
    print(f"Saved CSV: {filename}")


def save_events(events: List[Dict], filename: str):
    """Save events to JSON file"""
    def _jsonify(obj):
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Type not serializable: {type(obj)}")
    
    with open(filename, "w") as f:
        json.dump(events, f, indent=2, default=_jsonify)
    
    print(f"Saved events: {filename}")


def plot_psd(
    raw,
    fmin: float = 0.5,
    fmax: float = 100.0,
    figsize: Tuple = (12, 6),
    max_channels: int = 8
):
    """Plot power spectral density"""
    psd = raw.compute_psd(fmin=fmin, fmax=fmax, verbose=False)
    psds, freqs = psd.get_data(return_freqs=True)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    n_plot = min(max_channels, len(raw.ch_names))
    for i in range(n_plot):
        ax.semilogy(freqs, psds[i] * 1e12, alpha=0.7, label=f"Ch{i+1}")
    
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (μV²/Hz)")
    ax.set_title("Power Spectral Density")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=4, fontsize=8, loc="upper right")
    
    return fig


def plot_timeseries(
    signal: np.ndarray,
    times: np.ndarray,
    channel_idx: int = 0,
    figsize: Tuple = (12, 4)
):
    """Plot single channel time series"""
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(times, signal[channel_idx] * 1e6, lw=0.8, color='steelblue')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (μV)")
    ax.set_title(f"Channel {channel_idx + 1} — Time Series")
    ax.grid(True, alpha=0.3)
    
    return fig


def plot_stacked_channels(
    signal: np.ndarray,
    times: np.ndarray,
    offset_uv: float = 150.0,
    figsize: Tuple = (14, 8)
):
    """Plot all channels stacked"""
    fig, ax = plt.subplots(figsize=figsize)
    
    n_channels = signal.shape[0]
    
    for i in range(n_channels):
        ax.plot(times, signal[i] * 1e6 - i * offset_uv, lw=0.6, label=f"Ch{i+1}")
    
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (μV, stacked)")
    ax.set_title("All Channels — Stacked View")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=8, fontsize=8, loc="upper right")
    
    return fig


def create_output_bundle(
    result: Dict[str, Any],
    raw,
    output_dir: str,
    prefix: str = "eeg_synth"
):
    """Create complete output bundle with all files"""
    os.makedirs(output_dir, exist_ok=True)
    
    paths = {}
    
    # Save CSV
    csv_path = os.path.join(output_dir, f"{prefix}.csv")
    save_to_csv(result["signal"], result["times"], csv_path)
    paths["csv"] = csv_path
    
    # Save events
    if result["events"]:
        events_path = os.path.join(output_dir, f"{prefix}_events.json")
        save_events(result["events"], events_path)
        paths["events"] = events_path
    
    # Save FIF
    fif_path = os.path.join(output_dir, f"{prefix}.fif")
    raw.save(fif_path, overwrite=True, verbose=False)
    paths["fif"] = fif_path
    print(f"Saved FIF: {fif_path}")
    
    # Create figures
    figures = []
    
    fig_psd = plot_psd(raw)
    figures.append(fig_psd)
    
    fig_ts = plot_timeseries(result["signal"], result["times"], channel_idx=0)
    figures.append(fig_ts)
    
    fig_stack = plot_stacked_channels(result["signal"], result["times"])
    figures.append(fig_stack)
    
    # Save figures as PNG
    for i, fig in enumerate(figures):
        png_path = os.path.join(output_dir, f"{prefix}_fig{i+1}.png")
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    
    # Save PDF
    pdf_path = os.path.join(output_dir, f"{prefix}_figures.pdf")
    
    # Recreate figures for PDF
    figures = [
        plot_psd(raw),
        plot_timeseries(result["signal"], result["times"]),
        plot_stacked_channels(result["signal"], result["times"])
    ]
    
    with PdfPages(pdf_path) as pdf:
        for fig in figures:
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
    
    paths["pdf"] = pdf_path
    print(f"Saved PDF: {pdf_path}")
    
    print(f"\n✓ Output bundle created: {output_dir}")
    
    return paths