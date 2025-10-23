

"""
Test suite for EEG Synthesis Framework
"""

import numpy as np
import sys


def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...", end=" ")
    try:
        from core.types import BandSpec, BlockOutput
        from core.backend import Backend, RNGManager
        from blocks.generators import BandGenerator, ColoredNoise
        from blocks.envelopes import BurstyEnvelope
        from blocks.transforms import PACModulate
        from blocks.spatial import Reference
        from blocks.artifacts import EOGArtifacts, EMGArtifacts, LineNoise
        from pipeline.orchestrator import Pipeline
        from eeg_io.export import create_output_bundle
        print("✓ PASS")
        return True
    except Exception as e:
        print(f"✗ FAIL: {e}")
        return False


def test_basic_generation():
    """Test basic signal generation"""
    print("Testing basic generation...", end=" ")
    try:
        from pipeline.orchestrator import Pipeline
        from core.types import BandSpec
        from blocks.generators import BandGenerator
        
        pipe = Pipeline(backend="numpy", seed=42)
        spec = BandSpec("test", 8.0, 12.0, 10.0)
        pipe.add("test", BandGenerator(spec))
        
        result = pipe.run(duration=1.0, sfreq=250.0, n_channels=2)
        
        assert result["signal"].shape == (2, 250)
        assert not np.any(np.isnan(result["signal"]))
        assert not np.any(np.isinf(result["signal"]))
        
        print("✓ PASS")
        return True
    except Exception as e:
        print(f"✗ FAIL: {e}")
        return False


def test_determinism():
    """Test that same seed produces same output"""
    print("Testing determinism...", end=" ")
    try:
        from pipeline.orchestrator import Pipeline
        from core.types import BandSpec
        from blocks.generators import BandGenerator
        
        def generate(seed):
            pipe = Pipeline(backend="numpy", seed=seed)
            spec = BandSpec("test", 10.0, 12.0, 15.0)
            pipe.add("test", BandGenerator(spec))
            result = pipe.run(duration=2.0, sfreq=250.0, n_channels=2)
            return result["signal"]
        
        sig1 = generate(42)
        sig2 = generate(42)
        sig3 = generate(999)
        
        # Same seed -> identical
        assert np.allclose(sig1, sig2), "Same seed produced different outputs"
        
        # Different seed -> different
        assert not np.allclose(sig1, sig3), "Different seeds produced identical outputs"
        
        print("✓ PASS")
        return True
    except Exception as e:
        print(f"✗ FAIL: {e}")
        return False


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print("EEG SYNTHESIS FRAMEWORK - TEST SUITE")
    print("="*60 + "\n")
    
    tests = [
        test_imports,
        test_basic_generation,
        test_determinism,
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "="*60)
    passed = sum(results)
    total = len(results)
    print(f"RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All tests passed!")
        print("="*60 + "\n")
        return 0
    else:
        print(f"✗ {total - passed} test(s) failed")
        print("="*60 + "\n")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())



