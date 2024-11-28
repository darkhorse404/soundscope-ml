import time
import numpy as np
from typing import Optional
import logging
from src.data_processing.audio_processor import AudioProcessor
from src.models.classifier import SoundClassifier
from src.utils.visualization import SoundVisualizer
import os

logger = logging.getLogger(__name__)

class UrbanSoundMonitor:
    """Real-time urban sound monitoring system."""
    
    def __init__(self, duration: float = 5.0, sample_rate: int = 22050):
        """
        Initialize the sound monitor.
        
        Args:
            duration (float): Duration of each monitored sound clip in seconds
            sample_rate (int): Sample rate for audio processing
        """
        self.duration = duration
        self.sample_rate = sample_rate
        self.audio_processor = AudioProcessor(sample_rate=sample_rate, duration=duration)
        self.classifier = SoundClassifier()
        self.visualizer = SoundVisualizer()
        self.model_loaded = False

    def load_model(self, model_path: str) -> None:
        """Load a pre-trained model for classification."""
        try:
            self.classifier.load_model(model_path)
            self.model_loaded = True
            logger.info(f"Model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def start_monitoring(self, duration_minutes: int = 60, save_path: Optional[str] = None) -> None:
        """
        Start the monitoring process.
        
        Args:
            duration_minutes (int): Duration for which monitoring should run in minutes
            save_path (Optional[str]): Path to save monitored sound clips
        """
        if not self.model_loaded:
            logger.error("Model must be loaded before starting monitoring.")
            return

        logger.info(f"Starting sound monitoring for {duration_minutes} minutes.")
        end_time = time.time() + duration_minutes * 60
        
        while time.time() < end_time:
            try:
                # Capture audio and classify
                audio_signal, _ = self.audio_processor.load_audio('path_to_audio_file')
                features = self.extract_features(audio_signal)
                prediction = self.classifier.predict(features)
                
                # Log and visualize the result
                logger.info(f"Predicted sound class: {prediction}")
                
                if save_path:
                    self.audio_processor.save_audio(audio_signal, os.path.join(save_path, f"monitoring_clip_{int(time.time())}.wav"))
                
                # Visualize the waveform and spectrogram
                self.visualizer.plot_waveform(audio_signal, self.sample_rate)
                self.visualizer.plot_spectrogram(audio_signal, self.sample_rate)
                
                # Sleep before next monitoring interval
                time.sleep(self.duration)
            
            except Exception as e:
                logger.error(f"Error during monitoring: {str(e)}")
    
    def extract_features(self, audio_signal: np.ndarray) -> np.ndarray:
        """Extract features for classification."""
        try:
            features = self.audio_processor.standardize_audio(audio_signal)
            return features
        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            raise
    
    def generate_report(self, save_path: str) -> None:
        """
        Generate and save a report on the environmental sound monitoring.
        
        Args:
            save_path (str): Path to save the generated report
        """
        # For simplicity, this is a placeholder. You can extend it with actual reporting features.
        logger.info(f"Generating environmental report and saving to {save_path}")
        with open(save_path, 'w') as f:
            f.write("Urban Sound Monitoring Report\n")
            f.write("Detailed report of monitoring session\n")
