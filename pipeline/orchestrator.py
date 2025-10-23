# pipeline/orchestrator.py
"""
Pipeline orchestration and execution.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from core.types import Block, BlockOutput, SignalContext
from core.backend import Backend, RNGManager


@dataclass
class BlockStep:
    """A single block in the pipeline"""
    name: str
    block: Block
    params: Dict[str, Any]
    save_component: bool = False
    component_name: str = ""   # always a str
    accumulate: str = "replace"


class Pipeline:
    """Composable EEG signal processing pipeline"""

    def __init__(self, backend: str = "numpy", seed: int = 42):
        self.backend_name = backend
        self.backend = Backend(backend)
        self.seed = seed
        self.rng_manager = RNGManager(self.backend, seed)
        self.steps: List[BlockStep] = []
        self.context: Optional[SignalContext] = None

    def add(
        self,
        name: str,
        block: Block,
        save_component: bool = False,
        component_name: Optional[str] = None,
        accumulate: str = "replace",
        **params,
    ):
        """Add a block to the pipeline"""
        step = BlockStep(
            name=name,
            block=block,
            params=params,
            save_component=save_component,
            component_name=(component_name or name),  # now guaranteed str
            accumulate=accumulate,
        )
        self.steps.append(step)
        return self

    def run(
        self,
        duration: float,
        sfreq: float,
        n_channels: int,
        **initial_context,
    ) -> Dict[str, Any]:
        """Execute the pipeline"""
        # Initialize context
        self.context = SignalContext.create(duration, sfreq, n_channels, self.backend_name)

        # Base metadata useful to downstream UIs / exports
        self.context.metadata.update({
            "duration": duration,
            "sfreq": sfreq,
            "n_channels": n_channels,
            "backend": self.backend_name,
            "seed": self.seed,
        })
        # Add user-provided context/metadata
        for k, v in initial_context.items():
            self.context.metadata[k] = v

        # Initialize signal accumulator
        current_signal = self.backend.zeros((n_channels, self.context.n_samples))

        # Execute pipeline
        for step in self.steps:
            # Prepare block context
            block_context = {
                "signal": current_signal,
                "duration": duration,
                "sfreq": sfreq,
                "n_channels": n_channels,
                "n_samples": self.context.n_samples,
                "times": self.context.times,
                "nyquist": self.context.nyquist,
                "backend": self.backend_name,
                "components": self.context.components,
                **step.params,
                **self.context.metadata,
            }

            # Execute block — blocks seed their own RNG from `key`, and `None` is fine
            key = None
            output: BlockOutput = step.block(key=key, context=block_context)

            # Process output
            if "signal" in output.data:
                new_signal = output.data["signal"]

                # Accumulate
                if step.accumulate == "replace":
                    current_signal = new_signal
                elif step.accumulate == "add":
                    current_signal = current_signal + new_signal
                elif step.accumulate == "multiply":
                    current_signal = current_signal * new_signal

            # Save component
            if step.save_component and "signal" in output.data:
                self.context.components[step.component_name] = output.data["signal"]

            # Collect events
            if output.events:
                self.context.events.extend(output.events)

            # Update metadata
            self.context.metadata[f"{step.name}_meta"] = output.metadata

        # Store final signal
        self.context.signals["mixed"] = current_signal

        return {
            "signal": current_signal,
            "components": self.context.components,
            "events": self.context.events,
            "metadata": self.context.metadata,
            "times": self.context.times,
            "sfreq": sfreq,
            "n_channels": n_channels,
        }

    def get_mne_raw(self, result: Dict[str, Any], ch_names: Optional[List[str]] = None):
        """Convert pipeline result to MNE Raw object"""
        import mne

        signal = result["signal"]
        sfreq = result["sfreq"]
        n_channels = result["n_channels"]

        # Convert to numpy if needed
        if self.backend_name == "jax":
            signal = self.backend.to_numpy(signal)

        # Generate channel names
        if ch_names is None:
            ch_names = [f"EEG{i+1:02d}" for i in range(n_channels)]

        # Create MNE info
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")

        # Create Raw object
        raw = mne.io.RawArray(signal, info, verbose=False)

        # Add annotations from events
        if result["events"]:
            onset = [e["onset"] for e in result["events"]]
            duration = [e.get("duration", 0) for e in result["events"]]
            description = [e["type"] for e in result["events"]]
            raw.set_annotations(mne.Annotations(onset, duration, description))

        return raw
