import os
import tempfile
import numpy as np
import librosa
import parselmouth
from scipy import signal
from typing import Dict, List, Tuple, Optional
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioAnalyzer:
    """Core audio analysis utilities for baby cry analysis"""
    
    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate
        self.frame_length = 2048
        self.hop_length = 512
    
    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """Load audio file and return waveform and sample rate"""
        try:
            y, sr = librosa.load(audio_path, sr=self.sample_rate)
            return y, sr
        except Exception as e:
            logger.error(f"Error loading audio: {e}")
            raise
    
    def extract_f0(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Extract fundamental frequency using librosa"""
        f0 = librosa.yin(y, fmin=80, fmax=2000, sr=sr, 
                        frame_length=self.frame_length, hop_length=self.hop_length)
        return f0
    
    def extract_f0_parselmouth(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Extract F0 using Parselmouth (Praat) for more accurate analysis"""
        # Create temporary file for Parselmouth
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            import soundfile as sf
            sf.write(tmp_file.name, y, sr)
            
            try:
                sound = parselmouth.Sound(tmp_file.name)
                pitch = sound.to_pitch(time_step=0.01, pitch_floor=75.0, pitch_ceiling=2000.0)
                f0_values = pitch.selected_array['frequency']
                
                # Remove unvoiced frames (0 Hz)
                f0_values[f0_values == 0] = np.nan
                return f0_values
            finally:
                os.unlink(tmp_file.name)
    
    def extract_spectral_features(self, y: np.ndarray, sr: int) -> Dict:
        """Extract spectral features like centroid, rolloff, etc."""
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        spectral_flatness = librosa.feature.spectral_flatness(y=y)[0]
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        
        return {
            'spectral_centroid': spectral_centroids,
            'spectral_rolloff': spectral_rolloff,
            'spectral_flatness': spectral_flatness,
            'mfccs': mfccs
        }
    
    def extract_intensity(self, y: np.ndarray) -> np.ndarray:
        """Calculate RMS energy as intensity measure"""
        rms = librosa.feature.rms(y=y, frame_length=self.frame_length, 
                                 hop_length=self.hop_length)[0]
        return rms
    
    def calculate_hnr(self, y: np.ndarray, sr: int) -> float:
        """Calculate Harmonics-to-Noise Ratio using Parselmouth"""
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            import soundfile as sf
            sf.write(tmp_file.name, y, sr)
            
            try:
                sound = parselmouth.Sound(tmp_file.name)
                harmonicity = sound.to_harmonicity()
                hnr_values = harmonicity.values[harmonicity.values != -200]  # Remove undefined values
                return np.nanmean(hnr_values) if len(hnr_values) > 0 else 0.0
            finally:
                os.unlink(tmp_file.name)
    
    def calculate_jitter_shimmer(self, y: np.ndarray, sr: int) -> Tuple[float, float]:
        """Calculate jitter and shimmer using Parselmouth"""
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            import soundfile as sf
            sf.write(tmp_file.name, y, sr)
            
            try:
                sound = parselmouth.Sound(tmp_file.name)
                
                # Calculate jitter
                point_process = parselmouth.praat.call(sound, "To PointProcess (periodic, cc)", 75, 600)
                jitter = parselmouth.praat.call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
                
                # Calculate shimmer
                shimmer = parselmouth.praat.call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
                
                return jitter * 100, shimmer * 100  # Convert to percentages
            except Exception as e:
                logger.warning(f"Error calculating jitter/shimmer: {e}")
                return 0.0, 0.0
            finally:
                os.unlink(tmp_file.name)
    
    def detect_voiced_segments(self, y: np.ndarray, sr: int, 
                             min_duration: float = 0.1) -> List[Tuple[float, float]]:
        """Detect voiced segments in audio"""
        f0 = self.extract_f0_parselmouth(y, sr)
        
        # Find voiced frames (non-NaN F0 values)
        voiced_mask = ~np.isnan(f0)
        
        # Convert to time segments
        time_step = 0.01  # Parselmouth default
        segments = []
        
        if not np.any(voiced_mask):
            return segments
        
        # Find continuous voiced regions
        voiced_diff = np.diff(np.concatenate(([False], voiced_mask, [False])).astype(int))
        starts = np.where(voiced_diff == 1)[0] * time_step
        ends = np.where(voiced_diff == -1)[0] * time_step
        
        # Filter by minimum duration
        for start, end in zip(starts, ends):
            if end - start >= min_duration:
                segments.append((start, end))
        
        return segments
    
    def extract_formants(self, y: np.ndarray, sr: int) -> Dict:
        """Extract formant frequencies"""
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            import soundfile as sf
            sf.write(tmp_file.name, y, sr)
            
            try:
                sound = parselmouth.Sound(tmp_file.name)
                formants = sound.to_formant_burg()
                
                # Extract F1, F2, F3, F4
                f1_values = []
                f2_values = []
                f3_values = []
                f4_values = []
                
                for i in range(formants.get_number_of_frames()):
                    time = formants.get_time_from_frame_number(i + 1)
                    f1 = formants.get_value_at_time(1, time)
                    f2 = formants.get_value_at_time(2, time)
                    f3 = formants.get_value_at_time(3, time)
                    f4 = formants.get_value_at_time(4, time)
                    
                    if not np.isnan(f1): f1_values.append(f1)
                    if not np.isnan(f2): f2_values.append(f2)
                    if not np.isnan(f3): f3_values.append(f3)
                    if not np.isnan(f4): f4_values.append(f4)
                
                return {
                    'F1': np.array(f1_values),
                    'F2': np.array(f2_values),
                    'F3': np.array(f3_values),
                    'F4': np.array(f4_values)
                }
            except Exception as e:
                logger.warning(f"Error extracting formants: {e}")
                return {'F1': np.array([]), 'F2': np.array([]), 'F3': np.array([]), 'F4': np.array([])}
            finally:
                os.unlink(tmp_file.name)
    
    def calculate_spectral_bands_energy(self, y: np.ndarray, sr: int) -> Dict:
        """Calculate energy in specific frequency bands"""
        stft = librosa.stft(y, hop_length=self.hop_length, n_fft=self.frame_length)
        magnitude = np.abs(stft)
        
        # Frequency bins
        freqs = librosa.fft_frequencies(sr=sr, n_fft=self.frame_length)
        
        # Define bands
        bands = {
            'nasal_250_300': (250, 300),
            'mid_500_1500': (500, 1500),
            'low_mid_200_800': (200, 800),
            'high_800_2000': (800, 2000),
            'low_spectral_peak': (0, 700)
        }
        
        energy_bands = {}
        for band_name, (low_freq, high_freq) in bands.items():
            band_mask = (freqs >= low_freq) & (freqs <= high_freq)
            band_energy = np.mean(magnitude[band_mask, :], axis=0)
            energy_bands[band_name] = band_energy
        
        return energy_bands