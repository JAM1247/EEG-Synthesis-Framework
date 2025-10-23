

"""
Quickstart: Generate your first synthetic EEG in 10 lines of code
"""

from pipeline.orchestrator import Pipeline
from core.types import BandSpec
from blocks.generators import BandGenerator
from blocks.envelopes import BurstyEnvelope
from blocks.artifacts import EOGArtifacts, LineNoise
from eeg_io.export import create_output_bundle

# Create pipeline
pipe = Pipeline(backend="numpy", seed=42)

# Add alpha band with bursty envelope
alpha = BandSpec(name="alpha", freq_low=8.0, freq_high=12.0, amplitude_uv=25.0, num_partials=3)
pipe.add("alpha", BandGenerator(alpha), save_component=True)
pipe.add("envelope", BurstyEnvelope(rate_per_min=12, duration_sec=2.0, normalize_post=True, target_rms_uv=25.0))

# Add some realism
pipe.add("line_noise", LineNoise(freq_hz=60.0, amplitude_uv=2.0))
pipe.add("blinks", EOGArtifacts(rate_per_min=15.0, channels=[0, 1]))

# Generate 30 seconds of 8-channel EEG at 500 Hz
result = pipe.run(duration=30.0, sfreq=500.0, n_channels=8)
raw = pipe.get_mne_raw(result)

# Export everything
paths = create_output_bundle(result, raw, output_dir="output/quickstart", prefix="my_first_eeg")

print("\n✓ Generated synthetic EEG!")
print(f"✓ Output saved to: output/quickstart/")
print(f"✓ Files created: CSV, FIF, events, figures")

