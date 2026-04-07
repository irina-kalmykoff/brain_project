"""
Dataset Configuration Classes

Centralized configuration for different datasets.
Each config class contains all dataset-specific parameters with documentation.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Tuple
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

    # Default parameters for extract_features.py standalone usage
    default_model_order: int = 4   # Temporal context windows (standalone)
    default_step_size: int = 5     # Skip frames (standalone)

    # ============================================================
    # FREQUENCY BANDS
    # Used by acoustic change detectors and custom_decoder
    # ============================================================

    frequency_bands: Dict[str, Tuple[int, int]] = field(default_factory=lambda: {
        'delta': (1, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'low_gamma': (30, 70),
        'high_gamma': (70, 170),
    })

    # High gamma filter boundaries (Hz) for extract_features bandpass
    high_gamma_low: float = 70.0
    high_gamma_high: float = 170.0
    notch_50hz_band: Tuple[float, float] = (98.0, 102.0)   # 1st harmonic of 50Hz mains
    notch_150hz_band: Tuple[float, float] = (148.0, 152.0)  # 3rd harmonic

    # ============================================================
    # ACOUSTIC CHANGE DETECTION
    # ============================================================

    smoothing_window: int = 3              # Gaussian filter sigma for distance smoothing
    peak_threshold: float = 0.75           # Peak detection threshold
    spectral_k_factor: float = 1.5         # threshold = median + k * MAD (spectral boundaries)
    rms_k_factor: float = 1.2              # k factor for RMS-based boundaries
    sentence_k_factor: float = 1.0         # k factor for sentence segmentation
    onset_threshold_fraction: float = 0.15  # Fraction of max RMS change for onset detection
    peak_prominence: float = 0.0          # Minimum prominence for peak detection
    threshold_reduction_factor: float = 0.7 # Factor to reduce threshold when too few boundaries found
    welch_nperseg: int = 256               # Window size for Welch PSD estimation

    # RMS boundary detection
    rms_hop_ms: float = 5.0               # 5ms hop length for RMS computation
    rms_frame_ms: float = 20.0            # 20ms frame length for RMS computation
    rms_smoothing_sigma: float = 2.0      # Gaussian smoothing for RMS
    rms_change_smoothing_sigma: float = 1.5  # Gaussian smoothing for RMS change

    # ============================================================
    # WAV2VEC PARAMETERS
    # ============================================================

    wav2vec_fps: int = 50                  # Wav2vec output frame rate
    wav2vec_decimate_factor: int = 3       # Downsample factor for wav2vec input (48kHz -> 16kHz)

    # Wav2vec gaussian smoothing
    wav2vec_word_boundary_sigma: float = 0    # sigma for detect_boundaries (word-level)
    wav2vec_sentence_sigma: float = 0         # sigma for segment_sentence_by_wav2vec
    wav2vec_phoneme_sigma: float = 0          # sigma for _adaptive_peak_detection

    # Smoothing filter type: 'gaussian', 'savgol', or 'none'
    wav2vec_smoothing_filter: str = 'gaussian'
    # Savgol parameters (only used when filter='savgol')
    wav2vec_savgol_window: int = 7       # must be odd, window length
    wav2vec_savgol_polyorder: int = 3    # polynomial order, must be < window
    # Median filter parameter (only used when filter='median')
    wav2vec_median_size: int = 3         # kernel size, must be odd

    # ============================================================
    # TRAIN/TEST SPLIT DEFAULTS
    # ============================================================

    default_train_fraction: float = 0.7
    default_random_seed: int = 42

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

    # Adaptive peak detection tuning (used by _adaptive_peak_detection)
    adaptive_threshold_factors: list = None   # None = default [0.6, 0.5, ..., 1.2]; best: [1.3]
    adaptive_prominence_factor: float = 0.005  # prominence = factor * max(distances); tuned from 0.01

    # Word-level boundary detection tuning (used by segment_sentence_by_wav2vec)
    word_threshold_factors: list = field(default_factory=lambda: [0.0])  # [0.0] = no height filter; None = default k=1.0
    word_prominence_factor: float = 0.0    # prominence = factor * max(distances); 0 = no filter
    
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
            'frequency_bands': self.frequency_bands,
            'high_gamma_low': self.high_gamma_low,
            'high_gamma_high': self.high_gamma_high,
            'spectral_k_factor': self.spectral_k_factor,
            'rms_k_factor': self.rms_k_factor,
            'peak_prominence': self.peak_prominence,
            'wav2vec_fps': self.wav2vec_fps,
            'default_train_fraction': self.default_train_fraction,
            'default_random_seed': self.default_random_seed,
        }
    
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