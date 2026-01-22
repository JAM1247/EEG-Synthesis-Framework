"""
Demo script showcasing enhanced EEG synthesis features
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from pipeline.orchestrator import Pipeline
from core.types import BandSpec
from blocks.generators import BandGenerator, ColoredNoise
from blocks.envelopes import BurstyEnvelope
from blocks.transforms import PACModulate
from blocks.artifacts import EOGArtifacts, EMGArtifacts, LineNoise
from blocks.filters import BandpassFilter, NotchFilter, LowpassFilter
from blocks.observables import MultiBandPower, ValidationChecks, BandSignalExtractor
from eeg_io.export import (
    create_output_bundle,
    plot_band_powers,
    plot_filtered_comparison,
    plot_band_filtered_signals
)


def demo_basic_generation():
    """Demo 1: Basic EEG generation with multiple bands"""
    print("\n" + "="*60)
    print("DEMO 1: Basic EEG Generation")
    print("="*60)
    
    pipe = Pipeline(backend="numpy", seed=42)
    
    # Add multiple bands
    alpha = BandSpec("alpha", 8.0, 12.0, 25.0, 3)
    beta = BandSpec("beta", 13.0, 30.0, 8.0, 2)
    
    pipe.add("alpha", BandGenerator(alpha), save_component=True, accumulate="replace")
    pipe.add("beta", BandGenerator(beta), save_component=True, accumulate="add")
    
    # Add artifacts
    pipe.add("eog", EOGArtifacts(rate_per_min=15.0))
    pipe.add("noise", ColoredNoise(beta=1.0, rms_uv=2.0), accumulate="add")
    
    result = pipe.run(duration=30.0, sfreq=500.0, n_channels=8)
    raw = pipe.get_mne_raw(result)
    
    print(f"Generated {result['n_channels']} channels, {len(result['times'])} samples")
    print(f"Events recorded: {len(result['events'])}")
    
    return result, raw


def demo_preprocessing():
    """Demo 2: Preprocessing with filters"""
    print("\n" + "="*60)
    print("DEMO 2: Preprocessing with Filters")
    print("="*60)
    
    pipe = Pipeline(backend="numpy", seed=42)
    
    # Generate signal with line noise
    alpha = BandSpec("alpha", 8.0, 12.0, 25.0, 3)
    pipe.add("alpha", BandGenerator(alpha), save_component=True)
    pipe.add("line_noise", LineNoise(freq_hz=60.0, amplitude_uv=5.0), accumulate="add")
    pipe.add("noise", ColoredNoise(beta=1.0, rms_uv=3.0), accumulate="add")
    
    # Apply filters
    pipe.add("notch", NotchFilter(freq_hz=60.0, q=30.0))
    pipe.add("bandpass", BandpassFilter(band_hz=(0.5, 45.0)))
    pipe.add("lowpass", LowpassFilter(cutoff_hz=20.0))
    
    result = pipe.run(duration=30.0, sfreq=500.0, n_channels=4)
    raw = pipe.get_mne_raw(result)
    
    print("Preprocessing applied:")
    print("  - Notch filter at 60 Hz")
    print("  - Bandpass filter 0.5-45 Hz")
    print("  - Lowpass filter at 20 Hz")
    
    return result, raw


def demo_band_analysis():
    """Demo 3: Multi-band power analysis"""
    print("\n" + "="*60)
    print("DEMO 3: Multi-Band Power Analysis")
    print("="*60)
    
    pipe = Pipeline(backend="numpy", seed=42)
    
    # Generate all standard bands
    delta = BandSpec("delta", 0.5, 4.0, 20.0, 2)
    theta = BandSpec("theta", 4.0, 8.0, 15.0, 2)
    alpha = BandSpec("alpha", 8.0, 13.0, 25.0, 3)
    beta = BandSpec("beta", 13.0, 30.0, 8.0, 2)
    gamma = BandSpec("gamma", 30.0, 80.0, 5.0, 3)
    
    pipe.add("delta", BandGenerator(delta), save_component=True, accumulate="replace")
    pipe.add("theta", BandGenerator(theta), save_component=True, accumulate="add")
    pipe.add("alpha", BandGenerator(alpha), save_component=True, accumulate="add")
    pipe.add("beta", BandGenerator(beta), save_component=True, accumulate="add")
    pipe.add("gamma", BandGenerator(gamma), save_component=True, accumulate="add")
    
    # Add analysis blocks
    pipe.add("multiband_power", MultiBandPower())
    pipe.add("validation", ValidationChecks())
    pipe.add("band_extractor", BandSignalExtractor())
    
    result = pipe.run(duration=30.0, sfreq=500.0, n_channels=8)
    raw = pipe.get_mne_raw(result)
    
    # Print results
    for key, value in result['metadata'].items():
        if isinstance(value, dict) and "band_powers" in value:
            print("\nBand Powers (mean across channels):")
            for band, data in value["band_powers"].items():
                print(f"  {band.capitalize():8s}: {data['mean']:10.2f} µV²")
        
        if isinstance(value, dict) and "check_rms_uv_mean" in value:
            print(f"\nValidation:")
            print(f"  RMS: {value['check_rms_uv_mean']:.2f} µV")
            print(f"  Peak-to-Peak: {value['check_p2p_uv_mean']:.2f} µV")
            print(f"  RMS OK: {value['check_rms_uv_ok_frac']*100:.1f}%")
            print(f"  P2P OK: {value['check_p2p_uv_ok_frac']*100:.1f}%")
    
    return result, raw


def demo_pac_modulation():
    """Demo 4: Phase-Amplitude Coupling"""
    print("\n" + "="*60)
    print("DEMO 4: Phase-Amplitude Coupling (Theta-Gamma)")
    print("="*60)
    
    pipe = Pipeline(backend="numpy", seed=42)
    
    # Generate carrier (theta) and modulated (gamma) signals
    theta = BandSpec("theta", 4.0, 8.0, 15.0, 2)
    gamma = BandSpec("gamma", 30.0, 80.0, 5.0, 3)
    
    pipe.add("theta", BandGenerator(theta), save_component=True, accumulate="replace")
    pipe.add("gamma", BandGenerator(gamma), save_component=True, accumulate="add")
    pipe.add("pac", PACModulate("theta", "gamma", strength=0.7))
    
    result = pipe.run(duration=30.0, sfreq=500.0, n_channels=4)
    raw = pipe.get_mne_raw(result)
    
    print("PAC modulation applied:")
    print("  Carrier: Theta (4-8 Hz)")
    print("  Modulated: Gamma (30-80 Hz)")
    print("  Strength: 0.7")
    
    return result, raw


def demo_bursty_envelope():
    """Demo 5: Bursty envelope (sleep spindles)"""
    print("\n" + "="*60)
    print("DEMO 5: Bursty Envelope (Sleep Spindles)")
    print("="*60)
    
    pipe = Pipeline(backend="numpy", seed=42)
    
    # Generate alpha with bursty envelope
    alpha = BandSpec("alpha", 8.0, 12.0, 25.0, 3)
    pipe.add("alpha", BandGenerator(alpha), save_component=True)
    pipe.add(
        "bursty_envelope",
        BurstyEnvelope(
            rate_per_min=12.0,
            duration_sec=2.0,
            amp_factor=2.5,
            normalize_post=True,
            target_rms_uv=25.0
        )
    )
    
    result = pipe.run(duration=30.0, sfreq=500.0, n_channels=4)
    raw = pipe.get_mne_raw(result)
    
    # Count burst events
    burst_events = [e for e in result['events'] if e['type'] == 'burst']
    print(f"Generated {len(burst_events)} bursts")
    print(f"Average burst rate: {len(burst_events)/(30.0/60.0):.1f} per minute")
    
    return result, raw


def demo_complete_pipeline():
    """Demo 6: Complete realistic pipeline"""
    print("\n" + "="*60)
    print("DEMO 6: Complete Realistic EEG Pipeline")
    print("="*60)
    
    pipe = Pipeline(backend="numpy", seed=42)
    
    # Generate all bands
    delta = BandSpec("delta", 0.5, 4.0, 20.0, 2)
    theta = BandSpec("theta", 4.0, 8.0, 15.0, 2)
    alpha = BandSpec("alpha", 8.0, 13.0, 25.0, 3)
    beta = BandSpec("beta", 13.0, 30.0, 8.0, 2)
    gamma = BandSpec("gamma", 30.0, 80.0, 5.0, 3)
    
    pipe.add("delta", BandGenerator(delta), save_component=True, accumulate="replace")
    pipe.add("theta", BandGenerator(theta), save_component=True, accumulate="add")
    pipe.add("alpha", BandGenerator(alpha), save_component=True, accumulate="add")
    pipe.add("beta", BandGenerator(beta), save_component=True, accumulate="add")
    pipe.add("gamma", BandGenerator(gamma), save_component=True, accumulate="add")
    
    # Add bursty alpha
    pipe.add(
        "alpha_bursts",
        BurstyEnvelope(rate_per_min=12.0, duration_sec=2.0, normalize_post=True, target_rms_uv=25.0)
    )
    
    # Add PAC
    pipe.add("pac", PACModulate("theta", "gamma", strength=0.6))
    
    # Add artifacts
    pipe.add("eog", EOGArtifacts(rate_per_min=15.0))
    pipe.add("emg", EMGArtifacts(rate_per_min=8.0))
    pipe.add("line_noise", LineNoise(freq_hz=60.0, amplitude_uv=2.0), accumulate="add")
    pipe.add("noise", ColoredNoise(beta=1.0, rms_uv=2.0), accumulate="add")
    
    # Preprocessing
    pipe.add("notch", NotchFilter(freq_hz=60.0))
    pipe.add("bandpass", BandpassFilter(band_hz=(0.5, 45.0)))
    
    # Analysis
    pipe.add("multiband_power", MultiBandPower())
    pipe.add("validation", ValidationChecks())
    pipe.add("band_extractor", BandSignalExtractor())
    
    result = pipe.run(duration=30.0, sfreq=500.0, n_channels=8)
    raw = pipe.get_mne_raw(result)
    
    print(f"\nGeneration complete:")
    print(f"  Channels: {result['n_channels']}")
    print(f"  Duration: {len(result['times'])/result['sfreq']:.1f}s")
    print(f"  Events: {len(result['events'])}")
    print(f"  Artifacts: EOG, EMG, Line noise, Pink noise")
    print(f"  Preprocessing: Notch + Bandpass")
    print(f"  Analysis: Band powers, Validation, Band extraction")
    
    return result, raw


def main():
    """Run all demos"""
    print("\n" + "="*60)
    print(" EEG SYNTHESIS FRAMEWORK - ENHANCED DEMOS")
    print("="*60)
    
    # Create output directory (relative path, not hardcoded)
    output_dir = Path("output/demos")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run demos
    demos = [
        ("basic", demo_basic_generation),
        ("preprocessing", demo_preprocessing),
        ("band_analysis", demo_band_analysis),
        ("pac", demo_pac_modulation),
        ("bursty", demo_bursty_envelope),
        ("complete", demo_complete_pipeline),
    ]
    
    for name, demo_func in demos:
        result, raw = demo_func()
        
        # Save output bundle
        paths = create_output_bundle(
            result, raw, 
            output_dir=str(output_dir / name),
            prefix=f"demo_{name}"
        )
    
    print("\n" + "="*60)
    print(f"All demos complete! Results saved to: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()


    