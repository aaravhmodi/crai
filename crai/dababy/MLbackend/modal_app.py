import modal
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
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import shutil
import base64
import io

# Modal configuration
image = modal.Image.debian_slim(python_version="3.11").pip_install([
    "fastapi==0.104.1",
    "librosa==0.10.1",
    "numpy==1.24.3",
    "scipy==1.11.4",
    "scikit-learn==1.3.2",
    "parselmouth==0.4.3",
    "soundfile==0.12.1",
    "python-multipart==0.0.6",
    "pydantic==2.4.2"
])

app_modal = modal.App("baby-cry-analyzer", image=image)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for blob data
class AudioBlobRequest(BaseModel):
    audio_data: str  # base64 encoded audio data
    format: str = "wav"  # audio format (wav, mp3, etc.)

class AudioBlobWithParamsRequest(BaseModel):
    audio_data: str
    format: str = "wav"
    age_months: Optional[int] = None
    baseline_rms: Optional[float] = None

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


class BabyCryDetector:
    """Specialized detector for various baby cry patterns and medical indicators"""
    
    def __init__(self):
        self.analyzer = AudioAnalyzer()
        
        # Thresholds and parameters
        self.hyperphonation_f0_threshold = 1000  # Hz
        self.hyperphonation_duration_threshold = 0.15  # seconds
        
        self.hoarseness_hnr_threshold = 6.0  # dB
        self.hoarseness_jitter_threshold = 1.5  # %
        self.hoarseness_shimmer_threshold = 4.0  # %
        
        self.cri_du_chat_f0_mean_threshold = 800  # Hz
        self.cri_du_chat_f0_sd_threshold = 80  # Hz
        self.cri_du_chat_burst_min = 0.1  # seconds
        self.cri_du_chat_burst_max = 0.5  # seconds
        
        # Baselines for comparison (should be updated based on data)
        self.baseline_rms = 0.1  # Example baseline
        self.baseline_rms_std = 0.02  # Example standard deviation
    
    def detect_baby_cry(self, audio_path: str) -> Dict:
        """Main endpoint to detect if audio contains baby crying"""
        try:
            y, sr = self.analyzer.load_audio(audio_path)
            
            # Basic cry detection based on spectral and temporal features
            spectral_features = self.analyzer.extract_spectral_features(y, sr)
            f0 = self.analyzer.extract_f0_parselmouth(y, sr)
            intensity = self.analyzer.extract_intensity(y)
            
            # Simple heuristic for cry detection
            # High F0, high intensity variation, specific spectral characteristics
            mean_f0 = np.nanmean(f0)
            f0_std = np.nanstd(f0)
            mean_intensity = np.mean(intensity)
            intensity_std = np.std(intensity)
            
            # Cry detection logic (simplified)
            is_cry = (
                mean_f0 > 300 and mean_f0 < 2000 and  # Typical cry F0 range
                f0_std > 50 and  # Variable pitch
                intensity_std > 0.01 and  # Variable intensity
                mean_intensity > 0.05  # Sufficient loudness
            )
            
            return {
                "is_baby_cry": bool(is_cry),
                "confidence": float(min(1.0, (mean_f0 / 500) * (intensity_std * 100))),
                "mean_f0": float(mean_f0) if not np.isnan(mean_f0) else None,
                "f0_std": float(f0_std) if not np.isnan(f0_std) else None,
                "mean_intensity": float(mean_intensity),
                "intensity_std": float(intensity_std)
            }
            
        except Exception as e:
            logger.error(f"Error in baby cry detection: {e}")
            return {"is_baby_cry": False, "error": str(e)}
    
    def detect_hyperphonation(self, audio_path: str) -> Dict:
        """Detect hyperphonation: voiced frame with F0 > 1000 Hz for ≥150 ms"""
        try:
            y, sr = self.analyzer.load_audio(audio_path)
            f0 = self.analyzer.extract_f0_parselmouth(y, sr)
            
            # Find frames with F0 > 1000 Hz
            high_f0_mask = f0 > self.hyperphonation_f0_threshold
            
            if not np.any(high_f0_mask):
                return {
                    "neuro_distress_candidate": False,
                    "max_duration": 0.0,
                    "episodes_count": 0
                }
            
            # Find continuous episodes
            time_step = 0.01  # Parselmouth default
            episodes = []
            
            # Find continuous high F0 regions
            high_f0_diff = np.diff(np.concatenate(([False], high_f0_mask, [False])).astype(int))
            starts = np.where(high_f0_diff == 1)[0] * time_step
            ends = np.where(high_f0_diff == -1)[0] * time_step
            
            for start, end in zip(starts, ends):
                duration = end - start
                if duration >= self.hyperphonation_duration_threshold:
                    episodes.append(duration)
            
            neuro_distress_candidate = len(episodes) > 0
            max_duration = max(episodes) if episodes else 0.0
            
            return {
                "neuro_distress_candidate": neuro_distress_candidate,
                "max_duration": float(max_duration),
                "episodes_count": len(episodes),
                "episodes_durations": [float(d) for d in episodes]
            }
            
        except Exception as e:
            logger.error(f"Error in hyperphonation detection: {e}")
            return {"neuro_distress_candidate": False, "error": str(e)}
    
    def detect_hoarseness(self, audio_path: str) -> Dict:
        """Detect hoarseness: low HNR, high jitter/shimmer"""
        try:
            y, sr = self.analyzer.load_audio(audio_path)
            
            # Calculate voice quality measures
            hnr = self.analyzer.calculate_hnr(y, sr)
            jitter, shimmer = self.analyzer.calculate_jitter_shimmer(y, sr)
            
            # Check hoarseness criteria
            low_hnr = hnr < self.hoarseness_hnr_threshold
            high_jitter = jitter > self.hoarseness_jitter_threshold
            high_shimmer = shimmer > self.hoarseness_shimmer_threshold
            
            hoarseness_like = low_hnr or high_jitter or high_shimmer
            
            recommendations = []
            if hoarseness_like:
                recommendations.append("Consider laryngeal issues / hypothyroidism check")
            
            return {
                "hoarseness_like": hoarseness_like,
                "hnr": float(hnr),
                "jitter_percent": float(jitter),
                "shimmer_percent": float(shimmer),
                "criteria_met": {
                    "low_hnr": low_hnr,
                    "high_jitter": high_jitter,
                    "high_shimmer": high_shimmer
                },
                "recommendations": recommendations
            }
            
        except Exception as e:
            logger.error(f"Error in hoarseness detection: {e}")
            return {"hoarseness_like": False, "error": str(e)}
    
    def detect_cri_du_chat_pattern(self, audio_path: str) -> Dict:
        """Detect cri du chat pattern: mean F0 ≥ 800 Hz, low F0 SD, specific burst characteristics"""
        try:
            y, sr = self.analyzer.load_audio(audio_path)
            f0 = self.analyzer.extract_f0_parselmouth(y, sr)
            voiced_segments = self.analyzer.detect_voiced_segments(y, sr)
            
            # Calculate F0 statistics
            mean_f0 = np.nanmean(f0)
            f0_sd = np.nanstd(f0)
            
            # Analyze burst characteristics
            burst_durations = [end - start for start, end in voiced_segments]
            valid_bursts = [d for d in burst_durations 
                          if self.cri_du_chat_burst_min <= d <= self.cri_du_chat_burst_max]
            
            # Calculate repetition rate (approximate)
            if len(voiced_segments) > 1:
                intervals = [voiced_segments[i+1][0] - voiced_segments[i][1] 
                           for i in range(len(voiced_segments)-1)]
                mean_interval = np.mean(intervals) if intervals else 0
                repetition_rate = 1.0 / mean_interval if mean_interval > 0 else 0
            else:
                repetition_rate = 0
            
            # Check criteria
            high_f0 = mean_f0 >= self.cri_du_chat_f0_mean_threshold
            low_f0_variability = f0_sd < self.cri_du_chat_f0_sd_threshold
            good_burst_pattern = len(valid_bursts) > 0
            good_repetition_rate = 1.0 <= repetition_rate <= 2.0
            
            cri_du_chat_pattern = (high_f0 and low_f0_variability and 
                                 good_burst_pattern and good_repetition_rate)
            
            return {
                "cri_du_chat_pattern": cri_du_chat_pattern,
                "mean_f0": float(mean_f0) if not np.isnan(mean_f0) else None,
                "f0_sd": float(f0_sd) if not np.isnan(f0_sd) else None,
                "repetition_rate": float(repetition_rate),
                "burst_durations": [float(d) for d in burst_durations],
                "valid_bursts_count": len(valid_bursts),
                "criteria_met": {
                    "high_f0": high_f0,
                    "low_f0_variability": low_f0_variability,
                    "good_burst_pattern": good_burst_pattern,
                    "good_repetition_rate": good_repetition_rate
                }
            }
            
        except Exception as e:
            logger.error(f"Error in cri du chat detection: {e}")
            return {"cri_du_chat_pattern": False, "error": str(e)}
    
    def detect_weak_cry(self, audio_path: str, baseline_rms: Optional[float] = None) -> Dict:
        """Detect weak cry pattern: low RMS, short voiced segments, long pauses"""
        try:
            y, sr = self.analyzer.load_audio(audio_path)
            intensity = self.analyzer.extract_intensity(y)
            voiced_segments = self.analyzer.detect_voiced_segments(y, sr)
            
            # Use provided baseline or default
            if baseline_rms is None:
                baseline_rms = self.baseline_rms
            
            mean_rms = np.mean(intensity)
            rms_threshold = baseline_rms - (2 * self.baseline_rms_std)
            
            # Check for low RMS
            low_rms = mean_rms < rms_threshold
            
            # Check for short voiced segments
            voiced_durations = [end - start for start, end in voiced_segments]
            short_voiced_segments = np.mean([d < 0.3 for d in voiced_durations]) > 0.5
            
            # Check for long pauses
            if len(voiced_segments) > 1:
                pauses = [voiced_segments[i+1][0] - voiced_segments[i][1] 
                         for i in range(len(voiced_segments)-1)]
                long_pauses = np.mean([p > 1.0 for p in pauses]) > 0.3
            else:
                long_pauses = False
            
            weak_cry = low_rms and short_voiced_segments and long_pauses
            
            recommendations = []
            if weak_cry:
                recommendations.append("Consider SMA/botulism screen note")
            
            return {
                "weak_cry": weak_cry,
                "mean_rms": float(mean_rms),
                "rms_threshold": float(rms_threshold),
                "voiced_durations": [float(d) for d in voiced_durations],
                "mean_voiced_duration": float(np.mean(voiced_durations)) if voiced_durations else 0.0,
                "criteria_met": {
                    "low_rms": low_rms,
                    "short_voiced_segments": short_voiced_segments,
                    "long_pauses": long_pauses
                },
                "recommendations": recommendations
            }
            
        except Exception as e:
            logger.error(f"Error in weak cry detection: {e}")
            return {"weak_cry": False, "error": str(e)}
    
    def detect_grunting(self, audio_path: str) -> Dict:
        """Detect grunting: repetitive expiratory pulses with specific characteristics"""
        try:
            y, sr = self.analyzer.load_audio(audio_path)
            
            # Extract spectral features for low-mid frequency analysis
            energy_bands = self.analyzer.calculate_spectral_bands_energy(y, sr)
            low_mid_energy = energy_bands['low_mid_200_800']
            
            # Detect pulses using envelope
            envelope = np.abs(y)
            smoothed_envelope = np.convolve(envelope, np.ones(int(sr*0.01))/int(sr*0.01), mode='same')
            
            # Find peaks (potential pulses)
            from scipy.signal import find_peaks
            peaks, properties = find_peaks(smoothed_envelope, 
                                         height=np.mean(smoothed_envelope) * 1.5,
                                         distance=int(sr * 0.1))  # Min 100ms apart
            
            # Calculate pulse durations and intervals
            pulse_times = peaks / sr
            
            if len(pulse_times) > 1:
                intervals = np.diff(pulse_times)
                pulse_rate = 1.0 / np.mean(intervals) if len(intervals) > 0 else 0
                
                # Check for breathing cadence (0.5-1.2 Hz)
                good_rate = 0.5 <= pulse_rate <= 1.2
                
                # Check for strong low-mid harmonics
                strong_low_mid = np.mean(low_mid_energy) > np.mean(smoothed_envelope) * 0.7
                
                # Check pulse duration (150-350ms estimated from peak properties)
                pulse_durations = []
                for i, peak in enumerate(peaks):
                    # Estimate pulse duration from peak width
                    left_base = properties.get('left_bases', [peak])[i] if 'left_bases' in properties else peak - int(sr*0.1)
                    right_base = properties.get('right_bases', [peak])[i] if 'right_bases' in properties else peak + int(sr*0.1)
                    duration = (right_base - left_base) / sr
                    pulse_durations.append(duration)
                
                good_duration = np.mean([0.15 <= d <= 0.35 for d in pulse_durations]) > 0.5
                
                resp_distress = good_rate and strong_low_mid and good_duration and len(pulse_times) >= 3
            else:
                pulse_rate = 0
                good_rate = False
                strong_low_mid = False
                good_duration = False
                resp_distress = False
                pulse_durations = []
            
            return {
                "resp_distress": resp_distress,
                "pulse_rate": float(pulse_rate),
                "pulse_count": len(pulse_times),
                "pulse_durations": [float(d) for d in pulse_durations],
                "criteria_met": {
                    "good_rate": good_rate,
                    "strong_low_mid": strong_low_mid,
                    "good_duration": good_duration
                }
            }
            
        except Exception as e:
            logger.error(f"Error in grunting detection: {e}")
            return {"resp_distress": False, "error": str(e)}
    
    def detect_serious_illness_alert(self, audio_path: str) -> Dict:
        """Detect sepsis/meningitis patterns: continuous cry or moaning timbre"""
        try:
            y, sr = self.analyzer.load_audio(audio_path)
            duration = len(y) / sr
            
            # Check for continuous cry (>5-10 minutes)
            continuous_cry = duration > 300  # 5 minutes in seconds
            
            # Analyze spectral characteristics for moaning timbre
            spectral_features = self.analyzer.extract_spectral_features(y, sr)
            energy_bands = self.analyzer.calculate_spectral_bands_energy(y, sr)
            
            # Check for spectral peak < 500-700 Hz
            low_spectral_peak = np.mean(spectral_features['spectral_centroid']) < 600
            
            # Check for high spectral flatness (moaning timbre)
            high_flatness = np.mean(spectral_features['spectral_flatness']) > 0.7
            
            # Check for very high-pitched inconsolable cry
            f0 = self.analyzer.extract_f0_parselmouth(y, sr)
            very_high_pitch = np.nanmean(f0) > 1200 if not np.all(np.isnan(f0)) else False
            
            # Moaning pattern detection
            moaning_pattern = low_spectral_peak and high_flatness
            
            # Overall serious illness alert
            serious_illness_alert = (
                (continuous_cry and moaning_pattern) or
                (very_high_pitch and continuous_cry)
            )
            
            alert_messages = []
            if serious_illness_alert:
                alert_messages.append("SEEK IMMEDIATE MEDICAL CARE")
                if moaning_pattern:
                    alert_messages.append("Possible signs of serious illness detected")
                if very_high_pitch:
                    alert_messages.append("Very high-pitched inconsolable cry detected")
            
            return {
                "serious_illness_alert": serious_illness_alert,
                "duration_minutes": float(duration / 60),
                "continuous_cry": continuous_cry,
                "moaning_pattern": moaning_pattern,
                "very_high_pitch": very_high_pitch,
                "mean_f0": float(np.nanmean(f0)) if not np.all(np.isnan(f0)) else None,
                "mean_spectral_centroid": float(np.mean(spectral_features['spectral_centroid'])),
                "mean_spectral_flatness": float(np.mean(spectral_features['spectral_flatness'])),
                "alert_messages": alert_messages
            }
            
        except Exception as e:
            logger.error(f"Error in serious illness detection: {e}")
            return {"serious_illness_alert": False, "error": str(e)}


# Initialize FastAPI app
web_app = FastAPI(
    title="Baby Cry Analysis API",
    description="API for analyzing baby cries and detecting medical indicators",
    version="1.0.0"
)

# Add CORS middleware
web_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize detector (will be done in Modal function)
detector = None

async def save_upload_file(upload_file: UploadFile) -> str:
    """Save uploaded file to temporary location"""
    try:
        # Create a temporary file
        suffix = os.path.splitext(upload_file.filename)[1] if upload_file.filename else '.wav'
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            shutil.copyfileobj(upload_file.file, tmp_file)
            return tmp_file.name
    except Exception as e:
        logger.error(f"Error saving upload file: {e}")
        raise HTTPException(status_code=500, detail="Error processing uploaded file")

def save_blob_to_file(audio_data: str, format: str = "wav") -> str:
    """Save base64 encoded audio blob to temporary file"""
    try:
        # Decode base64 audio data
        audio_bytes = base64.b64decode(audio_data)
        
        # Create temporary file with appropriate extension
        if not format.startswith('.'):
            format = f'.{format}'
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=format) as tmp_file:
            tmp_file.write(audio_bytes)
            return tmp_file.name
    except Exception as e:
        logger.error(f"Error saving blob to file: {e}")
        raise HTTPException(status_code=500, detail="Error processing audio blob")

@web_app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Baby Cry Analysis API",
        "version": "1.0.0",
        "endpoints": {
            "baby_cry_detection_file": "/detect-cry",
            "baby_cry_detection_blob": "/detect-cry-blob",
            "hyperphonation_file": "/analyze/hyperphonation",
            "hyperphonation_blob": "/analyze/hyperphonation-blob",
            "hoarseness_file": "/analyze/hoarseness",
            "hoarseness_blob": "/analyze/hoarseness-blob",
            "cri_du_chat_file": "/analyze/cri-du-chat",
            "cri_du_chat_blob": "/analyze/cri-du-chat-blob",
            "weak_cry_file": "/analyze/weak-cry",
            "weak_cry_blob": "/analyze/weak-cry-blob",
            "grunting_file": "/analyze/grunting",
            "grunting_blob": "/analyze/grunting-blob",
            "serious_illness_file": "/analyze/serious-illness",
            "serious_illness_blob": "/analyze/serious-illness-blob",
            "comprehensive_file": "/analyze/comprehensive",
            "comprehensive_blob": "/analyze/comprehensive-blob"
        },
        "supported_formats": ["wav", "mp3", "m4a", "flac", "ogg"],
        "blob_format": "base64 encoded audio data"
    }

@web_app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

@web_app.post("/detect-cry")
async def detect_baby_cry(file: UploadFile = File(...)):
    """Detect if audio contains baby crying"""
    global detector
    if detector is None:
        detector = BabyCryDetector()
    
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Validate file type
    allowed_extensions = ['.wav', '.mp3', '.m4a', '.flac', '.ogg']
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file format. Allowed: {', '.join(allowed_extensions)}"
        )
    
    temp_file_path = None
    try:
        # Save uploaded file
        temp_file_path = await save_upload_file(file)
        
        # Analyze audio
        result = detector.detect_baby_cry(temp_file_path)
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"Error in cry detection: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

@web_app.post("/analyze/hyperphonation")
async def analyze_hyperphonation(file: UploadFile = File(...)):
    """Analyze for hyperphonation patterns (neuro distress indicator)"""
    global detector
    if detector is None:
        detector = BabyCryDetector()
    
    temp_file_path = None
    try:
        temp_file_path = await save_upload_file(file)
        result = detector.detect_hyperphonation(temp_file_path)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error in hyperphonation analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

@web_app.post("/analyze/hoarseness")
async def analyze_hoarseness(file: UploadFile = File(...)):
    """Analyze for hoarseness patterns (laryngeal issues indicator)"""
    global detector
    if detector is None:
        detector = BabyCryDetector()
    
    temp_file_path = None
    try:
        temp_file_path = await save_upload_file(file)
        result = detector.detect_hoarseness(temp_file_path)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error in hoarseness analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

@web_app.post("/analyze/comprehensive")
async def comprehensive_analysis(
    file: UploadFile = File(...),
    age_months: Optional[int] = Form(None),
    baseline_rms: Optional[float] = Form(None)
):
    """Run comprehensive analysis across all detection methods"""
    global detector
    if detector is None:
        detector = BabyCryDetector()
    
    temp_file_path = None
    try:
        temp_file_path = await save_upload_file(file)
        
        # Run all analyses
        results = {
            "baby_cry_detection": detector.detect_baby_cry(temp_file_path),
            "hyperphonation": detector.detect_hyperphonation(temp_file_path),
            "hoarseness": detector.detect_hoarseness(temp_file_path),
            "cri_du_chat_pattern": detector.detect_cri_du_chat_pattern(temp_file_path),
            "weak_cry": detector.detect_weak_cry(temp_file_path, baseline_rms),
            "grunting": detector.detect_grunting(temp_file_path),
            "serious_illness_alert": detector.detect_serious_illness_alert(temp_file_path)
        }
        
        # Generate summary
        alerts = []
        recommendations = []
        
        if results["baby_cry_detection"].get("is_baby_cry"):
            if results["hyperphonation"].get("neuro_distress_candidate"):
                alerts.append("Hyperphonation detected - consider neurological assessment")
            
            if results["hoarseness"].get("hoarseness_like"):
                recommendations.extend(results["hoarseness"].get("recommendations", []))
            
            if results["cri_du_chat_pattern"].get("cri_du_chat_pattern"):
                alerts.append("Cri du chat pattern detected")
            
            if results["weak_cry"].get("weak_cry"):
                recommendations.extend(results["weak_cry"].get("recommendations", []))
            
            if results["grunting"].get("resp_distress"):
                alerts.append("Respiratory distress signs detected")
            
            if results["serious_illness_alert"].get("serious_illness_alert"):
                alerts.extend(results["serious_illness_alert"].get("alert_messages", []))
        
        results["summary"] = {
            "alerts": alerts,
            "recommendations": recommendations,
            "analysis_timestamp": "now"
        }
        
        return JSONResponse(content=results)
        
    except Exception as e:
        logger.error(f"Error in comprehensive analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)


# Blob-based endpoints (accept base64 encoded audio data)

@web_app.post("/detect-cry-blob")
async def detect_baby_cry_blob(request: AudioBlobRequest):
    """Detect if audio contains baby crying - blob version"""
    global detector
    if detector is None:
        detector = BabyCryDetector()
    
    temp_file_path = None
    try:
        # Save blob to temporary file
        temp_file_path = save_blob_to_file(request.audio_data, request.format)
        
        # Analyze audio
        result = detector.detect_baby_cry(temp_file_path)
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"Error in cry detection: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

@web_app.post("/analyze/hyperphonation-blob")
async def analyze_hyperphonation_blob(request: AudioBlobRequest):
    """Analyze for hyperphonation patterns - blob version"""
    global detector
    if detector is None:
        detector = BabyCryDetector()
    
    temp_file_path = None
    try:
        temp_file_path = save_blob_to_file(request.audio_data, request.format)
        result = detector.detect_hyperphonation(temp_file_path)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error in hyperphonation analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

@web_app.post("/analyze/hoarseness-blob")
async def analyze_hoarseness_blob(request: AudioBlobRequest):
    """Analyze for hoarseness patterns - blob version"""
    global detector
    if detector is None:
        detector = BabyCryDetector()
    
    temp_file_path = None
    try:
        temp_file_path = save_blob_to_file(request.audio_data, request.format)
        result = detector.detect_hoarseness(temp_file_path)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error in hoarseness analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

@web_app.post("/analyze/cri-du-chat-blob")
async def analyze_cri_du_chat_blob(request: AudioBlobRequest):
    """Analyze for cri du chat syndrome patterns - blob version"""
    global detector
    if detector is None:
        detector = BabyCryDetector()
    
    temp_file_path = None
    try:
        temp_file_path = save_blob_to_file(request.audio_data, request.format)
        result = detector.detect_cri_du_chat_pattern(temp_file_path)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error in cri du chat analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

@web_app.post("/analyze/weak-cry-blob")
async def analyze_weak_cry_blob(request: AudioBlobWithParamsRequest):
    """Analyze for weak cry patterns - blob version"""
    global detector
    if detector is None:
        detector = BabyCryDetector()
    
    temp_file_path = None
    try:
        temp_file_path = save_blob_to_file(request.audio_data, request.format)
        result = detector.detect_weak_cry(temp_file_path, request.baseline_rms)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error in weak cry analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

@web_app.post("/analyze/grunting-blob")
async def analyze_grunting_blob(request: AudioBlobRequest):
    """Analyze for grunting patterns - blob version"""
    global detector
    if detector is None:
        detector = BabyCryDetector()
    
    temp_file_path = None
    try:
        temp_file_path = save_blob_to_file(request.audio_data, request.format)
        result = detector.detect_grunting(temp_file_path)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error in grunting analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

@web_app.post("/analyze/serious-illness-blob")
async def analyze_serious_illness_blob(request: AudioBlobRequest):
    """Analyze for serious illness alert patterns - blob version"""
    global detector
    if detector is None:
        detector = BabyCryDetector()
    
    temp_file_path = None
    try:
        temp_file_path = save_blob_to_file(request.audio_data, request.format)
        result = detector.detect_serious_illness_alert(temp_file_path)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error in serious illness analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

@web_app.post("/analyze/comprehensive-blob")
async def comprehensive_analysis_blob(request: AudioBlobWithParamsRequest):
    """Run comprehensive analysis across all detection methods - blob version"""
    global detector
    if detector is None:
        detector = BabyCryDetector()
    
    temp_file_path = None
    try:
        temp_file_path = save_blob_to_file(request.audio_data, request.format)
        
        # Run all analyses
        results = {
            "baby_cry_detection": detector.detect_baby_cry(temp_file_path),
            "hyperphonation": detector.detect_hyperphonation(temp_file_path),
            "hoarseness": detector.detect_hoarseness(temp_file_path),
            "cri_du_chat_pattern": detector.detect_cri_du_chat_pattern(temp_file_path),
            "weak_cry": detector.detect_weak_cry(temp_file_path, request.baseline_rms),
            "grunting": detector.detect_grunting(temp_file_path),
            "serious_illness_alert": detector.detect_serious_illness_alert(temp_file_path)
        }
        
        # Generate summary
        alerts = []
        recommendations = []
        
        if results["baby_cry_detection"].get("is_baby_cry"):
            if results["hyperphonation"].get("neuro_distress_candidate"):
                alerts.append("Hyperphonation detected - consider neurological assessment")
            
            if results["hoarseness"].get("hoarseness_like"):
                recommendations.extend(results["hoarseness"].get("recommendations", []))
            
            if results["cri_du_chat_pattern"].get("cri_du_chat_pattern"):
                alerts.append("Cri du chat pattern detected")
            
            if results["weak_cry"].get("weak_cry"):
                recommendations.extend(results["weak_cry"].get("recommendations", []))
            
            if results["grunting"].get("resp_distress"):
                alerts.append("Respiratory distress signs detected")
            
            if results["serious_illness_alert"].get("serious_illness_alert"):
                alerts.extend(results["serious_illness_alert"].get("alert_messages", []))
        
        results["summary"] = {
            "alerts": alerts,
            "recommendations": recommendations,
            "analysis_timestamp": "now"
        }
        
        return JSONResponse(content=results)
        
    except Exception as e:
        logger.error(f"Error in comprehensive analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)


@app_modal.function()
@modal.asgi_app()
def fastapi_app():
    return web_app


# For local development
if __name__ == "__main__":
    import uvicorn
    detector = BabyCryDetector()
    uvicorn.run(web_app, host="0.0.0.0", port=8000)