import numpy as np
import librosa
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class FeatureExtractor:
    """Extract audio features for sound classification."""
    
    def __init__(self):
        """Initialize FeatureExtractor with default parameters."""
        self.feature_functions = {
            'mfcc': self._extract_mfcc,
            'spectral_centroid': self._extract_spectral_centroid,
            'chroma': self._extract_chroma,
            'zero_crossing_rate': self._extract_zero_crossing_rate,
            'spectral_rolloff': self._extract_spectral_rolloff
        }
    
    def extract_all_features(self, audio: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
        """
        Extract all available features from audio signal.
        
        Args:
            audio (np.ndarray): Audio signal
            sr (int): Sample rate
            
        Returns:
            Dict[str, np.ndarray]: Dictionary of features
        """
        features = {}
        for name, func in self.feature_functions.items():
            try:
                features[name] = func(audio, sr)
            except Exception as e:
                logger.error(f"Error extracting {name} features: {str(e)}")
                raise
        return features
    
    def _extract_mfcc(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Extract Mel-frequency cepstral coefficients."""
        return librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    
    def _extract_spectral_centroid(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Extract spectral centroid."""
        return librosa.feature.spectral_centroid(y=audio, sr=sr)
    
    def _extract_chroma(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Extract chromagram."""
        return librosa.feature.chroma_stft(y=audio, sr=sr)
    
    def _extract_zero_crossing_rate(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Extract zero crossing rate."""
        return librosa.feature.zero_crossing_rate(audio)
    
    def _extract_spectral_rolloff(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Extract spectral rolloff."""
        return librosa.feature.spectral_rolloff(y=audio, sr=sr)
    
    @staticmethod
    def compute_statistics(features: np.ndarray) -> Dict[str, float]:
        """
        Compute statistical measures for features.
        
        Args:
            features (np.ndarray): Input features
            
        Returns:
            Dict[str, float]: Statistical measures
        """
        return {
            'mean': np.mean(features),
            'std': np.std(features),
            'max': np.max(features),
            'min': np.min(features),
            'median': np.median(features),
            'skew': float(stats.skew(features.ravel()))
        }
