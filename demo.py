


"""
Complete demonstration of the modular EEG synthesis framework.
"""

import os
from pipeline.orchestrator import Pipeline
from core.types import BandSpec
from blocks.generators import BandGenerator, ColoredNoise
from blocks.envelopes import BurstyEnvelope
from blocks.transforms import PACModulate
from blocks.spatial import Reference
from blocks.artifacts import EOGArtifacts, EMGArtifacts, LineNoise, BaselineDrift
from eeg_io.export import create_output_bundle

# Create output directory
os.makedirs("output", exist_ok=True)

print("\n" + "="*60)
print("DEMO 1: Basic Alpha with Bursts")
print("="*60)

pipe = Pipeline(backend="numpy", seed=42)
alpha = BandSpec("alpha", 8.0, 12.0, 25.0, 3)
pipe.add("alpha", BandGenerator(alpha), save_component=True)
pipe.add("envelope", BurstyEnvelope(rate_per_min=12, duration_sec=2.0, normalize_post=True, target_rms_uv=25.0))

result = pipe.run(duration=10.0, sfreq=500.0, n_channels=8)
raw = pipe.get_mne_raw(result)
create_output_bundle(result, raw, "output/demo1", "alpha_bursts")

print("✓ Demo 1 complete: output/demo1/")

print("\n" + "="*60)
print("DEMO 2: Multi-band with PAC")
print("="*60)

pipe = Pipeline(backend="numpy", seed=123)
theta = BandSpec("theta", 4.0, 8.0, 15.0, 2)
gamma = BandSpec("gamma", 30.0, 80.0, 5.0, 3)
pipe.add("theta", BandGenerator(theta), save_component=True)
pipe.add("gamma", BandGenerator(gamma), save_component=True, accumulate="add")
pipe.add("pac", PACModulate("theta", "gamma", strength=0.7))

result = pipe.run(duration=20.0, sfreq=500.0, n_channels=8)
raw = pipe.get_mne_raw(result)
create_output_bundle(result, raw, "output/demo2", "pac")

print("✓ Demo 2 complete: output/demo2/")

print("\n" + "="*60)
print("✓ ALL DEMOS COMPLETE!")
print("="*60)
print("\nCheck the 'output/' directory for generated files.")















