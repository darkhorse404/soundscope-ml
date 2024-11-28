import pytest
import numpy as np
from src.data_processing.audio_processor import AudioProcessor

def test_load_audio():
    audio_processor = AudioProcessor()
    # Use a sample file (ensure you have a test file in your test environment)
    audio, sr = audio_processor.load_audio('tests/test_audio.wav')
    assert audio is not None
    assert sr == 22050  # Ensure sample rate is as expected

def test_standardize_audio():
    audio_processor = AudioProcessor()
    audio = np.random.randn(100000)  # Simulated random audio
    standardized_audio = audio_processor.standardize_audio(audio)
    assert len(standardized_audio) == 22050  # Check standard length
    assert np.allclose(np.max(standardized_audio), 1.0, atol=1e-5)  # Check normalization
