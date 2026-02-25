"""
Dataset Configuration Classes

Centralized configuration for different datasets.
Each config class contains all dataset-specific parameters with documentation.
"""

from dataclasses import dataclass
from typing import Dict, Any
import json

@dataclass
class Dutch30Config:
    """
    Configuration for Dutch30 dataset processing.
    
    All parameters are based on:
    - Dataset specifications (sampling rates, hardware)
    - Predecessors' preprocessing pipeline
    - Standard signal processing parameters
    
    References:
    - Paper: "Speech detection from intracranial EEG..."
    - Code: Preprocessing scripts from original Dutch30 analysis
    """
    
    # ============================================================
    # HARDWARE / DATASET SPECIFICATIONS
    # ============================================================
    
    # Audio parameters
    audio_sr: int = 48000           # Raw audio sampling rate (microphone)
    audio_target_sr: int = 16000    # Downsampled rate for spectrograms
    
    # EEG parameters  
    eeg_sr: int = 1024             # EEG sampling rate (after downsampling)
    
    # ============================================================
    # FEATURE EXTRACTION PARAMETERS
    # From predecessors' preprocessing pipeline
    # ============================================================
    
    window_length: float = 0.030    # 30ms window for features
    frameshift: float = 0.006       # 5ms frameshift between windows
    mel_num_filters: int = 23      # Mel filterbank filters
    # Fixed window for feature extraction (normalizes segment lengths)
    fixed_feature_window_ms = 100  # 100ms window
    fixed_feature_samples = 102    # At 1024 Hz: 102 samples
    
    
    # ============================================================
    # TEMPORAL CONTEXT
    # From predecessors' analysis
    # ============================================================
    
    model_order: int = 10          # Temporal context windows
    step_size: int = 5             # Skip frames for stacking
    # Results in: 21 windows from -500ms to +500ms
    
    
    # ============================================================
    # CHANNEL PROCESSING
    # ============================================================
    
    #min_channels: int = 10         # Minimum channels per patient
    #target_channels: int = 133     # Standardized channel count
    
    
    # ============================================================
    # PHONEME PROCESSING
    # ============================================================
    
    min_phoneme_duration: float = 0.025   # 50ms minimum
    max_phoneme_duration: float = 0.40   # 400ms maximum
    min_silence_duration: float = 0.20   # 200ms for baseline
    
    # Length normalization for neural features
    boundary_detection_method = 'wav2vec'  # Options: 'rms', 'wav2vec'
    min_eeg_samples_for_features: int = 40  # Minimum EEG samples for extractHG
    
    """
        phoneme duration
        Short consonants (stops, t, k, p): 30-80ms
        Regular consonants (s, f, n): 80-150ms
        Short vowels (ə, ɪ): 50-120ms
        Regular vowels (a, e, o): 100-200ms
        Long vowels/diphthongs (aː, eː, ɛi): 150-400ms
        Rarely >400ms (unless emphatic or includes silence)
    """
    
    # ============================================================
    # SILENCE DETECTION
    # From predecessors' speech detection
    # ============================================================
    
    silence_threshold_factor: float = 0.45
    # threshold = (max_energy + min_energy) * 0.45
    electrode_exclusion_file: str = "electrode_exclusions.json"
    
    # ============================================================
    # AUDIO NORMALIZATION
    # ============================================================
    
    int16_max: int = 32767  # Max value for 16-bit signed integer
    # Converts float audio [-1, 1] to int16 for mel spectrogram
    
    
    # ============================================================
    # CHANNEL QUALITY FILTERING
    # ============================================================
    
    channel_outlier_threshold: float = 3.0    # Channels with std > median * threshold are excluded
    channel_flat_threshold: float = 0.1       # Channels with std < median * threshold are excluded  
    channel_kurtosis_threshold: float = 5.0   # Channels with kurtosis > median * threshold are excluded
    min_channels_to_keep: int = 20            # Minimum channels to retain per patient
    
   
    # ============================================================
    # DERIVED PROPERTIES
    # ============================================================
    
    @property
    def melspec_frame_duration(self) -> float:
        """Duration of each mel spectrogram frame in seconds"""
        return self.frameshift
    
    @property
    def min_samples_for_extraction(self) -> int:
        """Minimum EEG samples needed for feature extraction"""
        return int((self.window_length + self.frameshift) * self.eeg_sr)
    
    @property
    def temporal_context_range(self) -> tuple:
        """Time range covered by temporal stacking (ms)"""
        range_ms = self.model_order * self.step_size * self.frameshift * 1000
        return (-range_ms, range_ms)
    
    # ============================================================
    # UTILITY METHODS
    # ============================================================
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            'audio_sr': self.audio_sr,
            'audio_target_sr': self.audio_target_sr,
            'eeg_sr': self.eeg_sr,
            'window_length': self.window_length,
            'frameshift': self.frameshift,
            'mel_num_filters': self.mel_num_filters,
            'model_order': self.model_order,
            'step_size': self.step_size,
            'min_phoneme_duration': self.min_phoneme_duration,
            'max_phoneme_duration': self.max_phoneme_duration,
            'min_silence_duration': self.min_silence_duration,
            'silence_threshold_factor': self.silence_threshold_factor,
            'int16_max': self.int16_max,
        }
    
    def save(self, filepath: str):
        """Save config to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str):
        """Load config from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls(**data)
    
    def __str__(self):
        """Pretty print configuration"""
        lines = ["Dutch30 Configuration:"]
        for key, value in self.to_dict().items():
            lines.append(f"  {key}: {value}")
        return "\n".join(lines)


# For future extensibility
@dataclass 
class Dutch10Config(Dutch30Config):
    """
    Configuration for Dutch10 dataset.
    Inherits from Dutch30Config, override differences if any.
    """
    # Override if Dutch10 has different parameters
    pass