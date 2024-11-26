import librosa
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class AudioProcessor:
    """Handle audio file processing and manipulation."""
    
    def __init__(self, sample_rate: int = 22050, duration: float = 5.0):
        """
        Initialize AudioProcessor.
        
        Args:
            sample_rate (int): Target sample rate for audio processing
            duration (float): Target duration for audio clips in seconds
        """
        self.sample_rate = sample_rate
        self.duration = duration
        self.samples = int(sample_rate * duration)
    
    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """
        Load audio file with error handling.
        
        Args:
            file_path (str): Path to audio file
            
        Returns:
            Tuple[np.ndarray, int]: Audio signal and sample rate
        """
        try:
            audio, sr = librosa.load(file_path, sr=self.sample_rate)
            return self.standardize_audio(audio)
        except Exception as e:
            logger.error(f"Error loading audio file {file_path}: {str(e)}")
            raise
    
    def standardize_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Standardize audio length and amplitude.
        
        Args:
            audio (np.ndarray): Input audio signal
            
        Returns:
            np.ndarray: Standardized audio signal
        """
        # Pad or truncate to standard length
        if len(audio) < self.samples:
            audio = np.pad(audio, (0, self.samples - len(audio)))
        else:
            audio = audio[:self.samples]
        
        # Normalize amplitude
        audio = librosa.util.normalize(audio)
        
        return audio
    
    def save_audio(self, audio: np.ndarray, file_path: str) -> None:
        """
        Save audio to file.
        
        Args:
            audio (np.ndarray): Audio signal to save
            file_path (str): Output file path
        """
        try:
            sf.write(file_path, audio, self.sample_rate)
        except Exception as e:
            logger.error(f"Error saving audio file {file_path}: {str(e)}")
            raise
