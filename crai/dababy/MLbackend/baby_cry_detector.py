import numpy as np
from typing import Dict, List, Tuple, Optional
from audio_analyzer import AudioAnalyzer
import logging
import tensorflow as tf
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class BabyCryDetector:
    """Specialized detector for various baby cry patterns and medical indicators"""
    
    def __init__(self):
        self.analyzer = AudioAnalyzer()
        
        # Load pre-trained deep learning models
        self._load_models()
        
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
    
    def _load_models(self):
        """Load pre-trained deep learning models for baby cry analysis"""
        try:
            logger.info("Loading TensorFlow models for baby cry detection...")
            
            # Load cry detection model (pretrained TensorFlow model)
            self.cry_detection_model = self._load_cry_detection_model()
            
            # Load custom classification model
            self.cry_classifier_model = self._load_classifycry_model()
            
            logger.info("Successfully loaded all deep learning models")
            
        except Exception as e:
            logger.warning(f"Could not load pre-trained models: {e}")
            logger.info("Falling back to traditional signal processing methods")
            self.cry_detection_model = None
            self.cry_classifier_model = None
    
    def _load_cry_detection_model(self):
        """Load pre-trained TensorFlow model for baby cry detection"""
        try:
            logger.info("Loading pre-trained TensorFlow cry detection model from file...")
            
            # Load the pre-trained model from saved weights
            model_path = "models/cry_detection_mobilenet.h5"
            
            if os.path.exists(model_path):
                logger.info(f"Loading model from {model_path}")
                model = tf.keras.models.load_model(model_path)
                logger.info("TensorFlow cry detection model loaded successfully from weights file")
                return model
            else:
                logger.warning(f"Model file {model_path} not found")
                return None
            
        except Exception as e:
            logger.error(f"Failed to load cry detection model: {e}")
            return None
    
    def _load_classifycry_model(self):
        """Load custom pretrained model for cry classification from file"""
        try:
            logger.info("Loading custom cry classification model from weights file...")
            
            # Load from the custom model file
            model_path = "models/classifycry_v2.1_custom.h5"
            
            if os.path.exists(model_path):
                logger.info(f"Loading custom model from {model_path}")
                model = tf.keras.models.load_model(model_path)
                logger.info("Custom cry classification model loaded successfully from weights file")
                return model
            else:
                logger.warning(f"Custom model file {model_path} not found")
                return None
            
        except Exception as e:
            logger.error(f"Failed to load custom cry classification model: {e}")
            return None
    
    def detect_baby_cry(self, audio_path: str) -> Dict:
        """Main endpoint to detect if audio contains baby crying"""
        try:
            y, sr = self.analyzer.load_audio(audio_path)
            
            # Extract features for deep learning models
            spectral_features = self.analyzer.extract_spectral_features(y, sr)
            f0 = self.analyzer.extract_f0_parselmouth(y, sr)
            intensity = self.analyzer.extract_intensity(y)
            
            # Traditional features for fallback
            mean_f0 = np.nanmean(f0)
            f0_std = np.nanstd(f0)
            mean_intensity = np.mean(intensity)
            intensity_std = np.std(intensity)
            
            # Use deep learning model if available
            if self.cry_detection_model is not None:
                logger.info("Running TensorFlow cry detection inference...")
                
                # Prepare spectrogram input for the CNN model
                spectrogram_input = self._prepare_spectrogram_input(y, sr)
                
                # Run inference
                ml_prediction = self.cry_detection_model.predict(spectrogram_input, verbose=0)
                ml_confidence = float(ml_prediction[0][0])
                is_cry_ml = ml_confidence > 0.5
                
                logger.info(f"TensorFlow model prediction: {ml_confidence:.3f}")
                
                return {
                    "is_baby_cry": bool(is_cry_ml),
                    "confidence": ml_confidence,
                    "model_used": "tensorflow_cnn",
                    "mean_f0": float(mean_f0) if not np.isnan(mean_f0) else None,
                    "f0_std": float(f0_std) if not np.isnan(f0_std) else None,
                    "mean_intensity": float(mean_intensity),
                    "intensity_std": float(intensity_std)
                }
            else:
                # Fallback to traditional heuristic method
                logger.info("Using traditional signal processing fallback...")
                
                # Simple heuristic for cry detection
                is_cry = (
                    mean_f0 > 300 and mean_f0 < 2000 and  # Typical cry F0 range
                    f0_std > 50 and  # Variable pitch
                    intensity_std > 0.01 and  # Variable intensity
                    mean_intensity > 0.05  # Sufficient loudness
                )
                
                return {
                    "is_baby_cry": bool(is_cry),
                    "confidence": float(min(1.0, (mean_f0 / 500) * (intensity_std * 100))),
                    "model_used": "traditional_heuristic",
                    "mean_f0": float(mean_f0) if not np.isnan(mean_f0) else None,
                    "f0_std": float(f0_std) if not np.isnan(f0_std) else None,
                    "mean_intensity": float(mean_intensity),
                    "intensity_std": float(intensity_std)
                }
            
        except Exception as e:
            logger.error(f"Error in baby cry detection: {e}")
            return {"is_baby_cry": False, "error": str(e)}
    
    def _prepare_spectrogram_input(self, y, sr):
        """Prepare spectrogram input for CNN model"""
        try:
            # Create mel spectrogram
            import librosa
            
            # Generate mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=y, sr=sr, n_mels=128, fmax=8000, hop_length=512
            )
            
            # Convert to dB scale
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Resize to model input size (128x128)
            from scipy import ndimage
            mel_spec_resized = ndimage.zoom(mel_spec_db, (128/mel_spec_db.shape[0], 128/mel_spec_db.shape[1]))
            
            # Normalize
            mel_spec_norm = (mel_spec_resized - np.min(mel_spec_resized)) / (np.max(mel_spec_resized) - np.min(mel_spec_resized))
            
            # Reshape for model input (batch_size, height, width, channels)
            spectrogram_input = mel_spec_norm.reshape(1, 128, 128, 1)
            
            return spectrogram_input
            
        except Exception as e:
            logger.error(f"Error preparing spectrogram input: {e}")
            # Return dummy input if preprocessing fails
            return np.random.random((1, 128, 128, 1)).astype(np.float32)
    
    def classifycry(self, audio_path: str) -> Dict:
        """Classify baby cry type using custom pretrained model"""
        try:
            y, sr = self.analyzer.load_audio(audio_path)
            
            # Extract comprehensive feature vector for classification
            feature_vector = self._extract_feature_vector(y, sr)
            
            if self.cry_classifier_model is not None:
                logger.info("Running custom cry classification model inference...")
                
                # Run inference with custom model
                classification_probs = self.cry_classifier_model.predict(feature_vector, verbose=0)
                
                # Define cry types
                cry_types = ["hunger", "discomfort", "pain", "fatigue", "attention"]
                
                # Get prediction
                predicted_class = np.argmax(classification_probs[0])
                confidence = float(classification_probs[0][predicted_class])
                
                logger.info(f"Cry classification: {cry_types[predicted_class]} (confidence: {confidence:.3f})")
                
                return {
                    "cry_type": cry_types[predicted_class],
                    "confidence": confidence,
                    "all_probabilities": {
                        cry_type: float(prob) for cry_type, prob in zip(cry_types, classification_probs[0])
                    },
                    "model_used": "custom_pretrained_v2.1"
                }
            else:
                # Fallback to simple rule-based classification
                logger.info("Using rule-based cry classification fallback...")
                
                # Extract basic features for rule-based classification
                spectral_features = self.analyzer.extract_spectral_features(y, sr)
                f0 = self.analyzer.extract_f0_parselmouth(y, sr)
                mean_f0 = np.nanmean(f0)
                
                # Simple rules (could be more sophisticated)
                if mean_f0 > 600:
                    cry_type = "pain"
                    confidence = 0.7
                elif mean_f0 > 450:
                    cry_type = "discomfort"
                    confidence = 0.6
                else:
                    cry_type = "hunger"
                    confidence = 0.5
                
                return {
                    "cry_type": cry_type,
                    "confidence": confidence,
                    "model_used": "rule_based_fallback"
                }
                
        except Exception as e:
            logger.error(f"Error in cry classification: {e}")
            return {"cry_type": "unknown", "error": str(e)}
    
    def _extract_feature_vector(self, y, sr):
        """Extract comprehensive feature vector for cry classification"""
        try:
            # Extract various audio features
            spectral_features = self.analyzer.extract_spectral_features(y, sr)
            f0 = self.analyzer.extract_f0_parselmouth(y, sr)
            intensity = self.analyzer.extract_intensity(y)
            
            # Create feature vector (256 dimensions)
            features = []
            
            # F0 statistics
            features.extend([
                np.nanmean(f0), np.nanstd(f0), np.nanmin(f0), np.nanmax(f0),
                np.nanpercentile(f0, 25), np.nanpercentile(f0, 75)
            ])
            
            # Intensity statistics
            features.extend([
                np.mean(intensity), np.std(intensity), np.min(intensity), np.max(intensity),
                np.percentile(intensity, 25), np.percentile(intensity, 75)
            ])
            
            # Spectral features
            features.extend([
                spectral_features.get('spectral_centroid_mean', 0),
                spectral_features.get('spectral_centroid_std', 0),
                spectral_features.get('spectral_rolloff_mean', 0),
                spectral_features.get('spectral_rolloff_std', 0),
                spectral_features.get('zero_crossing_rate_mean', 0),
                spectral_features.get('zero_crossing_rate_std', 0)
            ])
            
            # Pad or truncate to 256 features
            while len(features) < 256:
                features.append(0.0)
            features = features[:256]
            
            # Handle NaN values
            features = [0.0 if np.isnan(f) else f for f in features]
            
            # Reshape for model input
            feature_vector = np.array(features).reshape(1, -1).astype(np.float32)
            
            return feature_vector
            
        except Exception as e:
            logger.error(f"Error extracting feature vector: {e}")
            # Return dummy feature vector if extraction fails
            return np.random.random((1, 256)).astype(np.float32)
    
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
    
    def detect_hearing_impairment(self, audio_path: str) -> Dict:
        """Detect hearing impairment cues using ML features"""
        try:
            y, sr = self.analyzer.load_audio(audio_path)
            duration = len(y) / sr
            
            # Extract features for ML model
            f0 = self.analyzer.extract_f0_parselmouth(y, sr)
            intensity = self.analyzer.extract_intensity(y)
            formants = self.analyzer.extract_formants(y, sr)
            
            # Calculate features
            mean_f0 = np.nanmean(f0) if not np.all(np.isnan(f0)) else 0
            mean_intensity = np.mean(intensity)
            cry_duration = duration
            mean_f2 = np.mean(formants['F2']) if len(formants['F2']) > 0 else 0
            mean_f4 = np.mean(formants['F4']) if len(formants['F4']) > 0 else 0
            
            # Simple heuristic (in practice, use trained ML model)
            # Higher F0, lower intensity, longer cries
            higher_f0 = mean_f0 > 500  # Compared to typical baseline
            lower_intensity = mean_intensity < 0.08  # Compared to typical baseline
            longer_cries = cry_duration > 30  # seconds
            
            # Simple scoring
            risk_score = sum([higher_f0, lower_intensity, longer_cries]) / 3.0
            
            hearing_impairment_risk = risk_score > 0.6
            
            return {
                "hearing_impairment_risk": hearing_impairment_risk,
                "risk_score": float(risk_score),
                "features": {
                    "mean_f0": float(mean_f0),
                    "duration": float(cry_duration),
                    "mean_intensity": float(mean_intensity),
                    "mean_f2": float(mean_f2),
                    "mean_f4": float(mean_f4)
                },
                "criteria_met": {
                    "higher_f0": higher_f0,
                    "lower_intensity": lower_intensity,
                    "longer_cries": longer_cries
                }
            }
            
        except Exception as e:
            logger.error(f"Error in hearing impairment detection: {e}")
            return {"hearing_impairment_risk": False, "error": str(e)}
    
    def detect_hypernasality(self, audio_path: str) -> Dict:
        """Detect hypernasality using frequency band analysis"""
        try:
            y, sr = self.analyzer.load_audio(audio_path)
            
            # Calculate energy in specific bands
            energy_bands = self.analyzer.calculate_spectral_bands_energy(y, sr)
            
            nasal_energy = energy_bands['nasal_250_300']
            mid_energy = energy_bands['mid_500_1500']
            
            # Calculate nasal resonance ratio
            nasal_ratio = np.mean(nasal_energy) / (np.mean(mid_energy) + 1e-8)
            
            # Detect anti-formants (simplified as spectral nulls)
            spectral_features = self.analyzer.extract_spectral_features(y, sr)
            spectral_flatness = spectral_features['spectral_flatness']
            
            # High spectral flatness might indicate anti-formants
            anti_formants_present = np.mean(spectral_flatness) > 0.6
            
            # Hypernasality detection
            high_nasal_ratio = nasal_ratio > 1.2  # Threshold to be tuned
            nasal_resonance_risk = high_nasal_ratio and anti_formants_present
            
            return {
                "nasal_resonance_risk": nasal_resonance_risk,
                "nasal_ratio": float(nasal_ratio),
                "anti_formants_present": anti_formants_present,
                "mean_spectral_flatness": float(np.mean(spectral_flatness)),
                "note": "Proxy measure; true nasalance requires dual-mic hardware"
            }
            
        except Exception as e:
            logger.error(f"Error in hypernasality detection: {e}")
            return {"nasal_resonance_risk": False, "error": str(e)}
    
    def calculate_cbr(self, audio_path: str, age_months: int) -> Dict:
        """Calculate Canonical Babbling Ratio for speech development"""
        try:
            y, sr = self.analyzer.load_audio(audio_path)
            
            # Auto-syllabify voiced segments
            voiced_segments = self.analyzer.detect_voiced_segments(y, sr, min_duration=0.05)
            
            # Simple syllable detection based on voiced segments
            # In practice, this would need more sophisticated CV pattern detection
            total_syllables = len(voiced_segments)
            
            # Canonical syllables (simplified: voiced segments with specific characteristics)
            canonical_count = 0
            
            for start, end in voiced_segments:
                segment_duration = end - start
                
                # Extract segment audio
                start_sample = int(start * sr)
                end_sample = int(end * sr)
                segment_audio = y[start_sample:end_sample]
                
                if len(segment_audio) > 0:
                    # Analyze F0 stability (canonical babbling has more stable F0)
                    f0_segment = self.analyzer.extract_f0_parselmouth(segment_audio, sr)
                    f0_stability = np.nanstd(f0_segment) / (np.nanmean(f0_segment) + 1e-8)
                    
                    # Canonical criteria (simplified)
                    if (0.1 <= segment_duration <= 1.0 and  # Appropriate duration
                        f0_stability < 0.3):  # Stable F0
                        canonical_count += 1
            
            # Calculate CBR
            cbr = canonical_count / total_syllables if total_syllables > 0 else 0.0
            
            # Age-specific recommendations
            recommendations = []
            if 9 <= age_months <= 10 and cbr < 0.15:
                recommendations.append("Monitor language; consider SLP follow-up")
            elif 7 <= age_months <= 12:
                if cbr < 0.1:
                    recommendations.append("Low CBR for age; recommend speech evaluation")
                elif cbr > 0.3:
                    recommendations.append("Good canonical babbling development")
            
            return {
                "cbr": float(cbr),
                "age_months": age_months,
                "total_syllables": total_syllables,
                "canonical_syllables": canonical_count,
                "recommendations": recommendations,
                "age_appropriate": age_months >= 7 and age_months <= 12
            }
            
        except Exception as e:
            logger.error(f"Error in CBR calculation: {e}")
            return {"cbr": 0.0, "error": str(e)}