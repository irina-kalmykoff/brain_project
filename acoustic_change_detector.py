import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.signal import find_peaks, savgol_filter
from numpy.lib.stride_tricks import sliding_window_view
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
from scipy.signal import welch
import gc
import os
import numpy as np
import torch
from transformers import Wav2Vec2Model, Wav2Vec2Processor
import librosa
from collections import Counter, defaultdict

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from extract_features import extractHG
from debugger import DebugMixin
from phonetic_dictionary import PhoneticDictionary
from dataset_config import Dutch30Config



class AcousticChangeDetector(DebugMixin):
    """
    Detects acoustic changes in spectrograms to identify potential phoneme boundaries.
    Uses unsupervised methods to find significant changes in acoustic features.
    """
    
    def __init__(self, config: Dutch30Config, distance_metric='cosine', smoothing_window=3, peak_threshold=0.75, 
                 decoder=None, debug_mode=None, phonetic_dict=None, feature_extraction_method='high_gamma', 
                 use_rms_boundaries=True, use_multifeature=False, use_wav2vec=False):
        
        """
        Initialize with parameters to control boundary detection sensitivity.
        """
        # Initialize the DebugMixin
        super().__init__(class_name="AcousticChangeDetector", debug_mode=False)
        
        self.config = config
        if debug_mode is not None:
            self.DEBUG_MODE = debug_mode
        self.log(f"Initialized with DEBUG_MODE={self.DEBUG_MODE}")

        self.distance_metric = distance_metric
        self.smoothing_window = smoothing_window
        self.peak_threshold = peak_threshold 
        self.decoder = decoder
        self.phonetic_dict = phonetic_dict or PhoneticDictionary()
        self.feature_extraction_method = feature_extraction_method
        
        self.use_rms_boundaries = use_rms_boundaries 
        self.use_multifeature = use_multifeature    
        
        self.log(f"Using feature extraction method: {self.feature_extraction_method}")
        self.use_wav2vec = use_wav2vec
        if self.use_wav2vec:
            self.log("Initializing wav2vec model for boundary detection...")
            self.wav2vec_processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-large-xlsr-53") #Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
            self.wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-xlsr-53") #Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base", use_safetensors=True)
            self.wav2vec_model.eval()
            self.log("Wav2vec model initialized successfully")
        
        self.frameshift = config.frameshift
        self.eeg_sr = config.eeg_sr
        self.window_length = config.window_length

    def _smooth_distances(self, distances, sigma):
        """Smooth distance curve using configured filter type.

        Args:
            distances: 1D array of distances.
            sigma: gaussian sigma (only used for gaussian filter).
        Returns:
            Smoothed distances array.
        """
        filter_type = getattr(self.config, 'wav2vec_smoothing_filter', 'gaussian')

        if filter_type == 'none' or (filter_type == 'gaussian' and sigma <= 0):
            return distances.copy()
        elif filter_type == 'savgol':
            window = getattr(self.config, 'wav2vec_savgol_window', 7)
            polyorder = getattr(self.config, 'wav2vec_savgol_polyorder', 3)
            window = min(window, len(distances))
            if window % 2 == 0:
                window -= 1
            if window <= polyorder:
                return distances.copy()
            return savgol_filter(distances, window, polyorder)
        elif filter_type == 'median':
            size = getattr(self.config, 'wav2vec_median_size', 3)
            size = min(size, len(distances))
            if size % 2 == 0:
                size -= 1
            if size < 1:
                return distances.copy()
            return median_filter(distances, size=size)
        else:  # gaussian
            return gaussian_filter1d(distances, sigma=sigma)

    def _extract_band_power_features(self, eeg_segment: np.ndarray) -> np.ndarray:
        """
        Extract power in delta, theta, alpha, beta, low_gamma, and high_gamma bands.
        Returns fixed-length feature vector (1, channels×6).
        """        
        
        bands = self.config.frequency_bands

        n_channels = eeg_segment.shape[1]
        band_features = []

        for ch in range(n_channels):
            freqs, psd = welch(
                eeg_segment[:, ch],
                fs=self.config.eeg_sr,
                nperseg=min(self.config.welch_nperseg, eeg_segment.shape[0])
            )
            
            for band_name, (low, high) in bands.items():
                band_mask = (freqs >= low) & (freqs < high)
                if np.any(band_mask):
                    band_power = np.mean(psd[band_mask])
                else:
                    band_power = 0.0
                band_features.append(band_power)
        
        return np.array(band_features).reshape(1, -1)
        
    def _extract_temporal_stats_features(self, eeg_segment: np.ndarray) -> np.ndarray:
        """
        Extract mean, std, max, min for each channel.
        Returns fixed-length feature vector (1, channels×4).
        """
        n_channels = eeg_segment.shape[1]
        stat_features = []
        
        for ch in range(n_channels):
            signal = eeg_segment[:, ch]
            stat_features.extend([
                np.mean(signal),
                np.std(signal),
                np.max(signal),
                np.min(signal)
            ])
        
        return np.array(stat_features).reshape(1, -1)

    def _extract_combined_features(self, eeg_segment: np.ndarray) -> np.ndarray:
        """
        Combine band powers and temporal statistics.
        Returns fixed-length feature vector (1, channels×10).
        """
        band_feats = self._extract_band_power_features(eeg_segment)
        stat_feats = self._extract_temporal_stats_features(eeg_segment)
        
        return np.concatenate([band_feats, stat_feats], axis=1)
    
    def count_phonemes(self, word):
        """ Count the number of phonemes in a word based on its transcription."""
        return self.phonetic_dict.count_phonemes(word)
    
    def compute_frame_distances(self, spectrogram):
        """
        Calculate distances between consecutive frames in the spectrogram.
        
        Parameters:
        -----------
        spectrogram : ndarray
            Mel spectrogram (time frames × frequency bins)
            
        Returns:
        --------
        distances : ndarray
            Array of distances between consecutive frames
        """
        # Initialize distances array
        num_frames = spectrogram.shape[0]
        distances = np.zeros(num_frames - 1)
        
        # Select distance metric
        if self.distance_metric == 'cosine':
            distance_func = cosine
        elif self.distance_metric == 'euclidean':
            distance_func = euclidean
        elif self.distance_metric == 'kl_divergence':
            def kl_div(p, q):
                # Small constant to avoid division by zero
                epsilon = 1e-10
                p = np.clip(p, epsilon, None)
                q = np.clip(q, epsilon, None)
                # Normalize to ensure they sum to 1
                p = p / np.sum(p)
                q = q / np.sum(q)
                return np.sum(p * np.log(p / q))
            distance_func = kl_div
        else:
            raise ValueError(f"Unsupported distance metric: {self.distance_metric}")
        
        # Vectorized frame-to-frame distance computation
        if self.distance_metric == 'cosine':
            norms = np.linalg.norm(spectrogram, axis=1, keepdims=True)
            norms = np.clip(norms, 1e-10, None)
            normalized = spectrogram / norms
            distances = 1.0 - np.sum(normalized[:-1] * normalized[1:], axis=1)
        elif self.distance_metric == 'euclidean':
            distances = np.sqrt(np.sum((spectrogram[1:] - spectrogram[:-1])**2, axis=1))
        elif self.distance_metric == 'kl_divergence':
            epsilon = 1e-10
            s = np.clip(spectrogram, epsilon, None)
            row_sums = s.sum(axis=1, keepdims=True)
            s = s / row_sums
            distances = np.sum(s[:-1] * np.log(s[:-1] / s[1:]), axis=1)

        return distances
    
    def compute_spectral_flux(self, spectrogram):
        """
        Calculate spectral flux between consecutive frames.
        
        Parameters:
        -----------
        spectrogram : ndarray
            Mel spectrogram (time frames × frequency bins)
            
        Returns:
        --------
        flux : ndarray
            Spectral flux between consecutive frames
        """
        # Vectorized spectral flux (sum of positive differences)
        diff = spectrogram[1:] - spectrogram[:-1]
        flux = np.sum(np.maximum(0, diff), axis=1)

        return flux
    
    def compute_energy_contour(self, spectrogram):
        """
        Calculate energy contour of the spectrogram.
        
        Parameters:
        -----------
        spectrogram : ndarray
            Mel spectrogram (time frames × frequency bins)
            
        Returns:
        --------
        energy : ndarray
            Energy contour
        """
        # Calculate frame energy (sum of squared values)
        energy = np.sum(spectrogram ** 2, axis=1)
        
        # Normalize
        energy = energy / np.max(energy)
        
        return energy
    
    def enhance_transitions(self, distances, flux=None, energy=None):
        """
        Apply signal processing techniques to enhance transition points.
        
        Parameters:
        -----------
        distances : ndarray
            Frame-to-frame distances
        flux : ndarray or None
            Spectral flux if available
        energy : ndarray or None
            Energy contour if available
            
        Returns:
        --------
        enhanced : ndarray
            Enhanced distance curve with transitions emphasized
        """
        # Create a copy to avoid modifying the original
        enhanced = distances.copy()
        
        # Normalize distances to [0, 1]
        if np.max(enhanced) > 0:
            enhanced = enhanced / np.max(enhanced)
        
        # Apply smoothing
        if self.smoothing_window > 1:
            enhanced = gaussian_filter1d(enhanced, sigma=self.smoothing_window / 3)
        
        # Compute derivative to emphasize changes
        derivative = np.gradient(enhanced)
        
        # Enhance with derivative
        enhanced = enhanced + np.abs(derivative)
        
        # Incorporate additional features if available
        if flux is not None:
            # Normalize flux
            flux_norm = flux / np.max(flux) if np.max(flux) > 0 else flux
            # Combine with enhanced distances
            enhanced = enhanced + flux_norm
        
        if energy is not None:
            # Use negative gradient of energy (energy dips often indicate boundaries)
            energy_grad = -np.gradient(energy)
            energy_grad = np.maximum(0, energy_grad)  # Keep only negative gradients
            # Normalize
            energy_grad = energy_grad / np.max(energy_grad) if np.max(energy_grad) > 0 else energy_grad
            # Add to enhanced distances
            enhanced = enhanced + energy_grad[:-1]  # Adjust length to match distances
        
        # Final normalization
        if np.max(enhanced) > 0:
            enhanced = enhanced / np.max(enhanced)
        
        return enhanced
    
    def detect_peaks(self, enhanced_distances, n_phonemes=None, frameshift=None, participant_id = None, word=None, word_position=None):
        """
        Detect peaks in the distance curve as potential boundaries.
        
        Parameters:
        -----------
        enhanced_distances : ndarray
            Enhanced frame-to-frame distances
        n_phonemes : int or None
            Number of expected phonemes (if known from transcription)
        frameshift : float
            Time between consecutive frames in seconds
            
        Returns:
        --------
        boundaries : ndarray
            Indices of detected boundaries
        """
        if frameshift is None:
            frameshift = self.config.frameshift
        
        # Calculate minimum distance between peaks in frames
        min_dist_frames = max(1, int(self.config.min_phoneme_duration / self.config.frameshift))
        
        # Use median-based threshold instead of max-based        
        median_val = np.median(enhanced_distances)
        mad = np.median(np.abs(enhanced_distances - median_val))  # Median Absolute Deviation
        
        # Adaptive threshold: median + k * MAD
        k = self.config.spectral_k_factor
        height = median_val + k * mad

        # Ensure minimum threshold
        min_height = 0.1 * np.max(enhanced_distances)
        height = max(height, min_height)

        self.debug(f"  Peak detection threshold: {height:.4f} (median: {median_val:.4f}, MAD: {mad:.4f})")

        # Find all peaks above threshold
        peaks, properties = find_peaks(
            enhanced_distances,
            height=height,
            distance=min_dist_frames,
            prominence=self.config.peak_prominence
        )
        
        self.debug(f"  Found {len(peaks)} candidate peaks")
        
        # If number of phonemes is known, select the best ones
        if n_phonemes is not None and n_phonemes > 1:
            n_boundaries = n_phonemes - 1
            
            if len(peaks) > n_boundaries:
                # Use prominence instead of just height
                if 'prominences' in properties:
                    peak_scores = properties['prominences']
                else:
                    peak_scores = enhanced_distances[peaks]
                
                # Select peaks with highest scores
                strongest_indices = np.argsort(peak_scores)[-n_boundaries:]
                peaks = peaks[strongest_indices]
                peaks = np.sort(peaks)
                self.debug(f"  Selected {len(peaks)} peaks based on prominence")
                
            elif len(peaks) < n_boundaries:
                # Try multiple threshold reductions
                patient_info = f" (Patient {participant_id})" if participant_id else ""
                word_info = f" for word '{word}'" if word else ""
                position_info = f" [position {word_position}]" if word_position is not None else ""
                self.log(f"  Need {n_boundaries} peaks but only found {len(peaks)}{word_info}{patient_info}")
                
                for attempt in range(3):
                    height = height * 0.6  # More aggressive reduction
                    self.debug(f"    Attempt {attempt+1}: lowering threshold to {height:.4f}")
                    
                    peaks, properties = find_peaks(
                        enhanced_distances,
                        height=height,
                        distance=min_dist_frames,
                        prominence=self.config.peak_prominence
                    )
                    
                    self.debug(f"    Found {len(peaks)} peaks")
                    
                    if len(peaks) >= n_boundaries:
                        # Select best ones
                        if 'prominences' in properties:
                            peak_scores = properties['prominences']
                        else:
                            peak_scores = enhanced_distances[peaks]
                        
                        strongest_indices = np.argsort(peak_scores)[-n_boundaries:]
                        peaks = peaks[strongest_indices]
                        peaks = np.sort(peaks)
                        break        

        # Boundaries are the frame indices where segments start/end
        boundaries = np.array([0] + list(peaks) + [len(enhanced_distances)])
        
        self.debug(f"  Final boundaries: {boundaries}")
        
        return boundaries
    
    def refine_boundaries(self, boundaries, spectrogram):
        """
        Refine boundary positions using additional constraints.
        
        Parameters:
        -----------
        boundaries : ndarray
            Initial boundary indices
        spectrogram : ndarray
            Original mel spectrogram
        frameshift : float
            Time between consecutive frames in seconds
            
        Returns:
        --------
        refined_boundaries : ndarray
            Refined boundary indices after applying constraints
        """
        if len(boundaries) <= 2:
            return boundaries  # No refinement needed
        
        # Calculate energy contour
        energy = self.compute_energy_contour(spectrogram)
        
        # Calculate minimum and maximum segment duration in frames
        min_frames = max(1, int(self.config.min_phoneme_duration / self.config.frameshift))
        max_frames = max(min_frames + 1, int(self.config.max_phoneme_duration / self.config.frameshift))
        
        # Initialize refined boundaries
        refined = [boundaries[0]]  # Keep the first boundary
        
        # Process internal boundaries
        for i in range(1, len(boundaries) - 1):
            current = boundaries[i]
            
            # Check if segment is too short
            if current - refined[-1] < min_frames:
                continue  # Skip this boundary
            
            # Look for energy minimum near the boundary
            search_radius = min(3, min_frames // 2)  # Look within a small window
            search_start = max(0, current - search_radius)
            search_end = min(len(energy), current + search_radius + 1)
            
            if search_start < search_end and search_end <= len(energy):
                # Find local energy minimum
                local_energy = energy[search_start:search_end]
                if len(local_energy) > 0:
                    local_min_idx = np.argmin(local_energy)
                    refined_pos = search_start + local_min_idx
                    refined.append(refined_pos)
                else:
                    refined.append(current)  # Keep original if search window is invalid
            else:
                refined.append(current)  # Keep original if search window is invalid
        
        # Add the last boundary
        refined.append(boundaries[-1])
        
        # Convert to array
        refined = np.array(refined)
        
        # Ensure refined boundaries are unique and sorted
        refined = np.unique(refined)
        
        return refined
    
    def detect_boundaries(self, spectrogram, word=None, phonetic_transcription=None, 
                     participant_id=None, word_position=None, 
                     use_multifeature=True,
                     use_rms_boundaries=True,
                     audio_segment=None, audio_sr=None):
        """Main method to detect phoneme boundaries in a word spectrogram."""
        self.debug(f"Detecting boundaries for word: {word if word else 'unknown'}")
        
        # Determine number of phonemes
        n_phonemes = None
        if word is not None:
            n_phonemes = self.count_phonemes(word)
            self.debug(f"Estimated {n_phonemes} phonemes for '{word}'")
        
        # STEP 1: Choose boundary detection method
        if use_rms_boundaries and audio_segment is not None:
            # RMS-based detection
            self.debug("Using RMS-based boundary detection")
            
            boundaries_original, rms_change = self.compute_rms_boundaries(
                audio_segment,
                audio_sr if audio_sr else self.config.audio_sr,
                n_phonemes=n_phonemes
            )
            
            # For compatibility
            distances = np.zeros(len(boundaries_original) - 1)
            energy = np.sum(spectrogram ** 2, axis=1)
            enhanced_distances = rms_change
            feature_dict = {'rms_change': rms_change}
            
            # Skip refinement for RMS (already precise)
            boundaries_final = boundaries_original
            
        elif self.use_wav2vec:
            # Wav2vec-based detection
            self.debug("Using wav2vec-based boundary detection")
            
            if audio_segment is None:
                raise ValueError("Audio waveform required when use_wav2vec=True")
            
            # Extract wav2vec features
            wav2vec_features = self.extract_wav2vec_features(audio_segment, audio_sr)
            
            # Calculate distances in wav2vec space
            distances = self.compute_wav2vec_distances(wav2vec_features)
            
            # Optional: smooth the distances
            enhanced_distances = self._smooth_distances(distances, self.config.wav2vec_word_boundary_sigma)
            
            # Adaptive peak detection for wav2vec
            if n_phonemes is not None and n_phonemes > 1:
                boundaries_original = self._adaptive_peak_detection(
                    enhanced_distances,
                    n_phonemes,
                    participant_id=participant_id,
                    word=word
                )
            else:
                boundaries_original = self.detect_peaks(
                    enhanced_distances, 
                    n_phonemes, 
                    participant_id=participant_id, 
                    word=word,
                    word_position=word_position
                )
            
            # For wav2vec, skip refinement since features are already optimized
            boundaries_final = boundaries_original
            
            # For compatibility
            energy = np.zeros(len(boundaries_final))
            feature_dict = {'wav2vec_distances': distances}
            
        elif use_multifeature:
            # Multi-feature fusion
            self.debug("Using multi-feature fusion for boundary detection")
            enhanced_distances, feature_dict = self.compute_multifeature_distances(
                spectrogram, 
                audio_segment=audio_segment,
                audio_sr=audio_sr if audio_sr else self.config.audio_sr
            )
            
            distances = feature_dict.get('spectral_distance', enhanced_distances)
            energy = np.sum(spectrogram ** 2, axis=1)
            
            # Detect peaks
            boundaries_original = self.detect_peaks(
                enhanced_distances, 
                n_phonemes, 
                participant_id=participant_id, 
                word=word,
                word_position=word_position
            )
            
            # Refine boundaries
            boundaries_refined = self.refine_boundaries(boundaries_original, spectrogram)
            
            # Decide which to use
            original_durations = [(boundaries_original[i+1] - boundaries_original[i]) * self.config.frameshift 
                                  for i in range(len(boundaries_original) - 1)]
            refined_durations = [(boundaries_refined[i+1] - boundaries_refined[i]) * self.config.frameshift 
                                 for i in range(len(boundaries_refined) - 1)]
            
            if len(original_durations) > 1:
                original_cv = np.std(original_durations) / np.mean(original_durations) * 100
                refined_cv = np.std(refined_durations) / np.mean(refined_durations) * 100
                
                if refined_cv <= original_cv * 1.2:
                    boundaries_final = boundaries_refined
                    self.debug(f"  ✓ Using refined boundaries (CV: {original_cv:.1f}% → {refined_cv:.1f}%)")
                else:
                    boundaries_final = boundaries_original
            else:
                boundaries_final = boundaries_refined
        
        else:
            # Original spectral distance method
            self.debug("Using original spectral distance method")
            distances = self.compute_frame_distances(spectrogram)
            flux = self.compute_spectral_flux(spectrogram)
            energy = self.compute_energy_contour(spectrogram)
            enhanced_distances = self.enhance_transitions(distances, flux, energy)
            feature_dict = None
            
            boundaries_original = self.detect_peaks(
                enhanced_distances, 
                n_phonemes, 
                participant_id=participant_id, 
                word=word,
                word_position=word_position
            )
            
            boundaries_refined = self.refine_boundaries(boundaries_original, spectrogram)
            
            original_durations = [(boundaries_original[i+1] - boundaries_original[i]) * self.config.frameshift 
                                  for i in range(len(boundaries_original) - 1)]
            refined_durations = [(boundaries_refined[i+1] - boundaries_refined[i]) * self.config.frameshift 
                                 for i in range(len(boundaries_refined) - 1)]
            
            if len(original_durations) > 1:
                original_cv = np.std(original_durations) / np.mean(original_durations) * 100
                refined_cv = np.std(refined_durations) / np.mean(refined_durations) * 100
                
                if refined_cv <= original_cv * 1.2:
                    boundaries_final = boundaries_refined
                    self.debug(f"  ✓ Using refined boundaries (CV: {original_cv:.1f}% → {refined_cv:.1f}%)")
                else:
                    boundaries_final = boundaries_original
            else:
                boundaries_final = boundaries_refined
        
        # STEP 2: Extract segments using final boundaries
        segments = []
        for i in range(len(boundaries_final) - 1):
            start = boundaries_final[i]
            end = boundaries_final[i + 1]
            
            start = max(0, start)
            end = min(spectrogram.shape[0], end)
            
            if start < end:
                segment = spectrogram[start:end]
                segments.append(segment)
        
        # STEP 3: Calculate boundary times
        boundary_times = boundaries_final * self.config.frameshift    
        boundary_samples = np.round(boundary_times * self.config.eeg_sr).astype(int)
        
        # Update boundaries_final to match adjusted samples
        boundaries_final = np.round(boundary_samples / self.config.eeg_sr / self.config.frameshift).astype(int)     
        

        # STEP 4: Create result
        result = {
            'boundaries': boundaries_final,
            'boundary_samples': boundary_samples,
            'boundary_times': boundary_times,
            'segments': segments,
            'distances': distances,
            'enhanced_distances': enhanced_distances,
            'word': word,
            'n_phonemes': n_phonemes,
            'energy': energy,
            'feature_dict': feature_dict,
            'method': 'rms' if (use_rms_boundaries and audio_segment is not None) else 
                  'wav2vec' if self.use_wav2vec else
                  'multifeature' if use_multifeature else 
                  'spectral'
        }
        
        return result
        
    def detect_speech_onset_rms(self, audio_segment, audio_sr, threshold_factor=0.15):
        """
        Detect speech onset using RMS energy.
        
        Parameters:
        -----------
        audio_segment : ndarray
            Raw audio waveform
        audio_sr : int
            Audio sampling rate
        threshold_factor : float
            Fraction of max RMS to use as threshold (0.1-0.3 typical)
            
        Returns:
        --------
        onset_sample : int
            Sample index where speech starts
        onset_time : float
            Time in seconds where speech starts
        """
        # Compute RMS in frames
        hop_length = int(0.010 * audio_sr)  # 10ms hop
        frame_length = int(0.025 * audio_sr)  # 25ms frame
        
        rms = librosa.feature.rms(
            y=audio_segment,
            frame_length=frame_length,
            hop_length=hop_length
        )[0]
        
        # Compute RMS change (derivative)
        rms_change = np.abs(np.gradient(rms))
        
        # Find threshold
        max_rms_change = np.max(rms_change)
        threshold = max_rms_change * threshold_factor
        
        # Find first point above threshold
        above_threshold = np.where(rms_change > threshold)[0]
        
        if len(above_threshold) > 0:
            onset_frame = above_threshold[0]
            onset_sample = onset_frame * hop_length
            onset_time = onset_sample / audio_sr
            
            self.debug(f"Detected speech onset at {onset_time:.3f}s (sample {onset_sample})")
            return onset_sample, onset_time
        else:
            self.debug("No clear speech onset detected, using start of segment")
            return 0, 0.0
    
    def process_word_segment(self, word_segment: dict, participant_id: str = None) -> dict:
        """
        Process a word segment from segment_data_by_words output.
        """
        word = word_segment.get('word', None)
        
        # Get spectrogram if available
        if 'spectrogram_segment' in word_segment:
            spectrogram = word_segment['spectrogram_segment']
        else:
            self.debug("Warning: No spectrogram found in word segment")
            return None
        
        # Detect boundaries
        result = self.detect_boundaries(spectrogram, word, frameshift=self.config.frameshift)
        
        # Add additional metadata
        result['participant_id'] = participant_id
        result['word_onset_sample'] = word_segment.get('onset_sample', None)
        result['word_offset_sample'] = word_segment.get('offset_sample', None)
        
        # Extract EEG segments if available
        if 'eeg_segment' in word_segment:
            eeg = word_segment['eeg_segment']
            
            # Calculate EEG sample indices for boundaries
            eeg_sr = word_segment.get('eeg_sr', self.config.eeg_sr)  # Default EEG sampling rate
            boundary_samples = np.round(result['boundary_times'] * eeg_sr).astype(int)
            
            # Extract EEG segments
            eeg_segments = []
            for i in range(len(boundary_samples) - 1):
                start = boundary_samples[i]
                end = boundary_samples[i + 1]
                
                # Ensure valid indices
                start = max(0, start)
                end = min(eeg.shape[0], end)
                
                if start < end:
                    eeg_segment = eeg[start:end]
                    eeg_segments.append(eeg_segment)
            
            result['eeg_segments'] = eeg_segments
            result['boundary_samples'] = boundary_samples
        
        return result
    
    def process_word_segments_dict(self, word_segments_dict: dict, participant_id: str) -> dict:
        """
        Process all word segments for a participant.
        """
        self.debug(f"Processing word segments for {participant_id}")
        
        # Get metadata
        metadata = word_segments_dict.get('metadata', {})
        frameshift = metadata.get('frameshift', self.config.frameshift)
        
        # Process each word
        results = {}
        
        for word, word_info in word_segments_dict.get('words', {}).items():
            self.debug(f"Processing word: {word}")
            
            word_results = []
            
            # Process each instance of the word
            for instance in word_info.get('instances', []):
                instance_result = self.process_word_segment(
                    instance, 
                    participant_id=participant_id,
                    frameshift=frameshift
                )
                
                if instance_result is not None:
                    word_results.append(instance_result)
            
            if word_results:
                results[word] = word_results
        
        return results
    
    def process_batch(self, batch: dict, skip_mismatches: bool = True) -> dict:
        """
        Process a batch of data from get_data_batch to extract phoneme-level features.
    
        Parameters:
        -----------
        batch : Output from get_data_batch function
        skip_mismatches :  If True, skip words where detected segments don't match expected phonemes
        """
        self.debug(f"Processing batch with {len(batch.get('words', []))} instances")
        
        if 'eeg_segments' in batch and batch['eeg_segments']:
            self.debug(f"  First EEG segment shape: {batch['eeg_segments'][0].shape}")
        if 'spectrogram_segments' in batch and batch['spectrogram_segments']:
            self.debug(f"  First spectrogram shape: {batch['spectrogram_segments'][0].shape}")
        
        # Initialize structures for phoneme-level data
        phoneme_eeg_segments = []
        phoneme_spectrogram_segments = []
        phoneme_labels = []
        phoneme_words = []  # Original words these phonemes come from
        phoneme_positions = []  # Position within word
        phoneme_participant_ids = []
        phoneme_durations_samples = []
        phoneme_instance_indices = []
        word_boundaries = []  # Store phoneme boundaries for each word
        
        total_words = 0
        mismatched_words = 0
        perfect_match_words = 0
        
  
        # Process each instance in the batch
        for i, word in enumerate(batch.get('words', [])):
            if i % 100 == 0:
                self.debug(f"Processing instance {i}/{len(batch.get('words', []))}: {word}")
            
            # Get data for this instance
            eeg_segment = batch['eeg_segments'][i] if 'eeg_segments' in batch else None
            spectrogram_segment = batch['spectrogram_segments'][i] if 'spectrogram_segments' in batch else None
            participant_id = batch['participant_ids'][i] if 'participant_ids' in batch else None
            instance_idx = batch['instance_indices'][i] if 'instance_indices' in batch else None
        
            # validate spectrogram size
            if spectrogram_segment is None:
                self.debug(f"Skipping word '{word}': spectrogram is None")
                continue
            
            # Need at least 3 frames for proper boundary detection
            if spectrogram_segment.shape[0] < 3:
                self.debug(f"Skipping word '{word}': only {spectrogram_segment.shape[0]} frames (need ≥3)")
                continue
    
            if spectrogram_segment.shape[0] < 2:  # Need at least 2 frames for distances
                self.debug(f"Skipping word '{word}': spectrogram too short ({spectrogram_segment.shape[0]} frames, need ≥2)")
                continue        
            
            # Apply phoneme segmentation if requested
            if word is not None:
                
                #count words for stats
                if word in self.phonetic_dict:
                    total_words += 1 
                    
                # Detect boundaries
                audio_segment = batch['audio_segments'][i] if 'audio_segments' in batch and i < len(batch['audio_segments']) else None

                result = self.detect_boundaries(
                    spectrogram=spectrogram_segment,
                    word=word,
                    participant_id=participant_id, 
                    word_position=i,
                    use_multifeature=self.use_multifeature,   
                    use_rms_boundaries=self.use_rms_boundaries,                     
                    audio_segment=audio_segment,   
                    audio_sr=self.config.audio_sr  
                )

                # Check if word should be dropped due to invalid boundaries
                if result.get('drop_word', False):
                    self.debug(f"  Dropping word '{word}' from {participant_id}: {result.get('reason', 'invalid segments')}")
                    continue

                # Store boundaries
                word_boundaries.append(result['boundaries'])
                
                extended_segments = None
                if eeg_segment is not None and result.get('boundary_samples') is not None:
                    min_samples = self.config.min_eeg_samples_for_features
                    
                    extended_segments = self._extend_short_segments(
                        result['boundary_samples'],
                        eeg_segment.shape[0],
                        min_samples
                    )
                
                # Extract phoneme segments
                segments = result['segments']
                
                # Try to get phoneme transcription
                if word in self.phonetic_dict:
                    # Use the phonetic dictionary's extract_phonemes method
                    phonemes = self.phonetic_dict.extract_phonemes(word)
                    
                    # Handle mismatch between segments and phonemes
                    if len(segments) == len(phonemes):
                        # Perfect match, use direct mapping
                        perfect_match_words += 1
                        for j, (phoneme, segment) in enumerate(zip(phonemes, segments)):
                            phoneme_spectrogram_segments.append(segment)
                            phoneme_labels.append(phoneme)
                            phoneme_words.append(word)
                            phoneme_positions.append(j)
                            phoneme_participant_ids.append(participant_id)
                            phoneme_instance_indices.append(instance_idx) 
                            
                            # Extract corresponding EEG segment if available
                            if eeg_segment is not None and result.get('boundary_samples') is not None:
                                boundaries = result['boundary_samples']
                                if extended_segments is not None and j < len(extended_segments):
                                    start, end = extended_segments[j]
                                    
                                    if start < end:
                                        raw_segment = eeg_segment[start:end]
                                        phoneme_durations_samples.append(end - start)
                                        # Normalize to fixed window size
                                        #fixed_segment = self._extract_fixed_window(
                                        #    raw_segment, 
                                        #    self.config.fixed_feature_samples
                                        #)
                                        phoneme_eeg_segments.append(raw_segment)
                                    else:
                                        phoneme_eeg_segments.append(np.array([]).reshape(0, eeg_segment.shape[1]))
                                        phoneme_durations_samples.append(0)
                                else:
                                    phoneme_eeg_segments.append(np.array([]).reshape(0, eeg_segment.shape[1]))
                                    phoneme_durations_samples.append(0)
                            else:
                                # No EEG data - append empty array with correct shape
                                if eeg_segment is not None:
                                    phoneme_eeg_segments.append(np.array([]).reshape(0, eeg_segment.shape[1]))
                                    phoneme_durations_samples.append(0)
                                else:
                                    phoneme_eeg_segments.append(np.array([]))
                                    phoneme_durations_samples.append(0)
                    else:
                        mismatched_words += 1
                        self.debug(
                            f"Mismatch for word '{word}': "
                            f"{len(phonemes)} phonemes but {len(segments)} segments"
                        )

                        if len(segments) == 0 or len(phonemes) == 0:
                            continue

                        seg_positions = np.linspace(0, len(phonemes) - 1, len(segments))
                        assigned_labels = [
                            phonemes[int(round(pos))] for pos in seg_positions
                        ]

                        for j, (segment, phoneme) in enumerate(
                            zip(segments, assigned_labels)
                        ):
                            phoneme_spectrogram_segments.append(segment)
                            phoneme_labels.append(phoneme)
                            phoneme_words.append(word)
                            phoneme_positions.append(j)
                            phoneme_participant_ids.append(participant_id)
                            phoneme_instance_indices.append(instance_idx)

                            if (eeg_segment is not None
                                    and result.get('boundary_samples') is not None):
                                if (extended_segments is not None
                                        and j < len(extended_segments)):
                                    start, end = extended_segments[j]
                                    start = max(0, start)
                                    end = min(eeg_segment.shape[0], end)

                                    if start < end:
                                        phoneme_eeg_segments.append(
                                            eeg_segment[start:end]
                                        )
                                        phoneme_durations_samples.append(end - start)
                                    else:
                                        phoneme_eeg_segments.append(
                                            np.array([]).reshape(0, eeg_segment.shape[1])
                                        )
                                        phoneme_durations_samples.append(0)
                                else:
                                    phoneme_eeg_segments.append(
                                        np.array([]).reshape(0, eeg_segment.shape[1])
                                    )
                                    phoneme_durations_samples.append(0)
                            else:
                                if eeg_segment is not None:
                                    phoneme_eeg_segments.append(
                                        np.array([]).reshape(0, eeg_segment.shape[1])
                                    )
                                else:
                                    phoneme_eeg_segments.append(np.array([]))
                                phoneme_durations_samples.append(0)
                else:
                    # No transcription available
                    self.debug(f"No transcription for word '{word}'")
                    
                    # Still add segments with unknown phoneme labels
                    for j, segment in enumerate(segments):
                        phoneme_spectrogram_segments.append(segment)
                        # Use '?' as placeholder for unknown phoneme
                        phoneme_labels.append('?')
                        phoneme_words.append(word)
                        phoneme_positions.append(j)
                        phoneme_participant_ids.append(participant_id)
                        phoneme_instance_indices.append(instance_idx) 
                        
                        # Extract corresponding EEG segment if available
                        if eeg_segment is not None and result.get('boundary_samples') is not None:
                            boundaries = result['boundary_samples']
                            if extended_segments is not None and j < len(extended_segments):
                                start, end = extended_segments[j]
                                
                                if start < end:
                                    raw_segment = eeg_segment[start:end]
                                    phoneme_durations_samples.append(end - start)
                                    # Normalize to fixed window size
                                    #fixed_segment = self._extract_fixed_window(
                                    #    raw_segment, 
                                    #    self.config.fixed_feature_samples
                                    #)
                                    phoneme_eeg_segments.append(raw_segment)
                                else:
                                    phoneme_eeg_segments.append(np.array([]).reshape(0, eeg_segment.shape[1]))
                                    phoneme_durations_samples.append(0)
                            else:
                                # extended_segments is None or j out of range
                                phoneme_eeg_segments.append(np.array([]).reshape(0, eeg_segment.shape[1]))
                                phoneme_durations_samples.append(0)
                        else:
                            # No EEG data available
                            if eeg_segment is not None:
                                phoneme_eeg_segments.append(np.array([]).reshape(0, eeg_segment.shape[1]))
                                phoneme_durations_samples.append(0)
                            else:
                                phoneme_eeg_segments.append(np.array([]))
                                phoneme_durations_samples.append(0)
            else:
                # Use whole segments as is (no phoneme segmentation)
                if spectrogram_segment is not None:
                    phoneme_spectrogram_segments.append(spectrogram_segment)
                    phoneme_labels.append(word)  # Use word as the label
                    phoneme_words.append(word)
                    phoneme_positions.append(0)  # Single position
                    phoneme_participant_ids.append(participant_id)
                    phoneme_instance_indices.append(instance_idx)
                    
                    if eeg_segment is not None:
                        phoneme_eeg_segments.append(eeg_segment)
        
        # Create enhanced batch
        enhanced_batch = {
            'phoneme_spectrogram_segments': phoneme_spectrogram_segments,
            'phoneme_labels': phoneme_labels,
            'phoneme_words': phoneme_words,
            'phoneme_positions': phoneme_positions,
            'phoneme_participant_ids': phoneme_participant_ids,
            'phoneme_durations_samples': phoneme_durations_samples, 
            'phoneme_instance_indices': phoneme_instance_indices, 
            'word_boundaries': word_boundaries,
            'original_batch': batch
        }
        
        # Add EEG segments if available
        if phoneme_eeg_segments:
            enhanced_batch['phoneme_eeg_segments'] = phoneme_eeg_segments
        
        # Add metadata
        enhanced_batch['metadata'] = {
            'phoneme_count': len(phoneme_labels),
            'unique_phonemes': len(set(phoneme_labels)),
            'total_words': total_words,
            'mismatched_words': mismatched_words,
            'perfect_match_words': perfect_match_words,
            'mismatch_rate': mismatched_words / total_words if total_words > 0 else 0
        }
        
        self.debug(f"Enhanced batch contains {enhanced_batch['metadata']['phoneme_count']} phoneme segments")
        self.debug(f"Found {enhanced_batch['metadata']['unique_phonemes']} unique phonemes")
        
        return enhanced_batch
    
    def accumulate_phoneme_data(self, split_result, batch_size=32, feature_extraction_method='high_gamma', 
                                batch_type='train') -> dict:
        """
        Accumulate phoneme data from multiple batches for training.
        
        Parameters:
        -----------
        num_batches : int
            Number of batches to process
        batch_size : int
            Size of each batch
        feature_extraction_method : str
            Method for feature extraction

        """
        
        # STEP 1: Collect all available instances (without loading data yet)
        all_instances = []
        word_segments_dict = split_result['word_segments_dict']
        
        for pid in split_result[batch_type]:
            if pid not in word_segments_dict:
                continue
            
            for word, indices in split_result[batch_type][pid].items():
                for idx in indices:
                    all_instances.append({
                        'participant_id': pid,
                        'word': word,
                        'instance_index': idx
                    })
        
        # STEP 2: Calculate number of batches
        num_batches = (len(all_instances) + batch_size - 1) // batch_size
        
        self.log(f"Processing {len(all_instances)} instances in {num_batches} batches (no replacement)")
        
        # Initialize accumulated data structures
        accumulated_features = []
        accumulated_spectrograms = []
        accumulated_labels = []
        accumulated_words = []
        accumulated_participant_ids = []
        accumulated_positions = []
        accumulated_durations_samples = []
        accumulated_instance_indices = []
        # vars for statistics
        total_words_processed = 0
        total_mismatches = 0
        total_perfect_matches = 0
        
    
        # Process multiple batches
        for batch_num in range(num_batches):
            self.log(f"Processing batch {batch_num+1}/{num_batches}")
            
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, len(all_instances))
            batch_instances = all_instances[start_idx:end_idx]            
        
            # Build batch from specific instances
            print(f"          Building batch from {len(batch_instances)} instances...")
            batch = self._build_batch_from_instances(batch_instances, word_segments_dict)
            print(f"          Batch built: {len(batch.get('words', []))} words")
            
            # Process batch to get phoneme-level data
            print(f"          Processing batch (phoneme detection)...")
            phoneme_batch = self.process_batch(batch)
            print(f"          Batch processed")
            
            # Prepare features for model training
            print(f"          Preparing training data...")
            phoneme_data = self.prepare_phoneme_training_data(
                phoneme_batch,
                feature_extraction_method=feature_extraction_method
            )
            print(f"          Training data prepared: {len(phoneme_data['features'])} features")
            
             # Accumulate mismatch statistics
            if 'total_words' in phoneme_batch['metadata']:
                total_words_processed += phoneme_batch['metadata']['total_words']
                total_mismatches += phoneme_batch['metadata']['mismatched_words']
                total_perfect_matches += phoneme_batch['metadata']['perfect_match_words']
            
            # Accumulate data
            accumulated_features.extend(phoneme_data['features'])
            accumulated_instance_indices.extend(phoneme_data['phoneme_instance_indices'])
            
            if 'spectrograms' in phoneme_data and phoneme_data['spectrograms']:
                accumulated_spectrograms.extend(phoneme_data['spectrograms'])
                accumulated_labels.extend(phoneme_data['phoneme_labels'])
                accumulated_words.extend(phoneme_data['phoneme_words'])
                accumulated_participant_ids.extend(phoneme_data['phoneme_participant_ids'])
                accumulated_positions.extend(phoneme_data.get('phoneme_positions', 
                                    [0] * len(phoneme_data['phoneme_labels'])))
            self.log(f"Accumulated {len(accumulated_features)} phoneme segments so far")
            
            if 'phoneme_durations_samples' in phoneme_data:
                accumulated_durations_samples.extend(phoneme_data['phoneme_durations_samples'])

            del batch, phoneme_batch, phoneme_data
            gc.collect()

        # Create result dictionary
        accumulated_data = {
            'features': accumulated_features,
            'spectrograms': accumulated_spectrograms if accumulated_spectrograms else None,
            'phoneme_labels': accumulated_labels,
            'phoneme_words': accumulated_words,
            'phoneme_participant_ids': accumulated_participant_ids,
            'phoneme_durations_samples': accumulated_durations_samples,
            'phoneme_instance_indices': accumulated_instance_indices,
            'phoneme_positions': accumulated_positions,
            'metadata': {
                'feature_extraction_method': feature_extraction_method,
                'n_phonemes': len(accumulated_labels),
                'unique_phonemes': len(set(accumulated_labels)),
                'n_batches': num_batches,
                'batch_size': batch_size
            }
        }
        
        self.log(f"Accumulated data from {num_batches} batches:")
        self.log(f"Total phoneme segments: {accumulated_data['metadata']['n_phonemes']}")
        self.log(f"Unique phonemes: {accumulated_data['metadata']['unique_phonemes']}")
        
        self.log("\n" + "="*60)
        self.log("PHONEME DETECTION SUMMARY")
        self.log("="*60)
        self.log(f"Total words processed: {total_words_processed}")
        if total_words_processed > 0:
            self.log(f"Perfect matches: {total_perfect_matches} ({total_perfect_matches/total_words_processed*100:.1f}%)")
            self.log(f"Mismatches: {total_mismatches} ({total_mismatches/total_words_processed*100:.1f}%)")
        else:
            self.log(f"Perfect matches: {total_perfect_matches} (N/A)")
            self.log(f"Mismatches: {total_mismatches} (N/A)")
        self.log(f"Unknown phonemes ('?'): {accumulated_labels.count('?')}")
        self.log("="*60 + "\n")

        return accumulated_data
    
    def prepare_phoneme_training_data(self, enhanced_batch: dict, **kwargs):
        """
        Prepare phoneme-level data for model training.
            
        Parameters:
        -----------
        enhanced_batch : Output from process_batch function
        feature_extraction_method : Method to use for feature extraction ('high_gamma', 'multi_band', etc.)
               
        Returns: Dictionary containing processed data ready for phoneme-based model training
        """
        # Use decoder's config as base, override with kwargs
        
        if self.decoder is not None and hasattr(self.decoder, 'config'):
            config_dict = self.decoder.config.to_dict()
            config_dict.update(kwargs)
        else:
            config_dict = kwargs
                
        # Extract parameters from config
        feature_extraction_method = config_dict.get('feature_extraction_method', 'high_gamma')
            
        # Check if we have EEG segments
        if 'phoneme_eeg_segments' not in enhanced_batch or not enhanced_batch['phoneme_eeg_segments']:
            raise ValueError("No EEG segments found in enhanced batch")
            
        # Filter out empty segments
        valid_indices = []
        for i, eeg in enumerate(enhanced_batch['phoneme_eeg_segments']):
            if eeg is not None and isinstance(eeg, np.ndarray) and eeg.size > 0:
                valid_indices.append(i)
            
        self.debug(f"Found {len(valid_indices)} valid phoneme segments out of {len(enhanced_batch['phoneme_eeg_segments'])}")            
           
        # FIRST PASS: Extract all features to determine expected dimensions
        patient_features = defaultdict(list)
    
        for idx in valid_indices:
            eeg = enhanced_batch['phoneme_eeg_segments'][idx]
            pid = enhanced_batch['phoneme_participant_ids'][idx]
            
            # Minimum samples needed for extractHG to produce at least 1 frame
            min_samples_for_extractHG = int(self.config.window_length * self.config.eeg_sr) + 1

            min_samples = max(
                int(self.config.min_phoneme_duration * self.config.eeg_sr),
                min_samples_for_extractHG
            )

            # Check for invalid segments
            if eeg is None:
                self.debug(f"Skipping segment {idx}: None")
                continue

            if eeg.ndim != 2:
                self.debug(f"Skipping segment {idx}: wrong dimensions ({eeg.ndim}D, expected 2D)")
                continue

            if eeg.shape[0] <= 0 or eeg.shape[1] <= 0:
                self.debug(f"Skipping segment {idx}: invalid shape {eeg.shape}")
                continue

            if eeg.shape[0] < min_samples:
                self.debug(f"Skipping segment {idx}: too short ({eeg.shape[0]} samples, need {min_samples})")
                continue
            
        
            try:
                
                # Extract features
                if self.decoder is not None:
                    if feature_extraction_method == 'high_gamma':
                        feat = extractHG(eeg, self.config.eeg_sr,
                                         windowLength=self.config.window_length,
                                         frameshift=self.config.frameshift)
                    elif feature_extraction_method == 'multi_band':
                        feat = self.decoder.custom_feature_extraction(eeg, self.config.eeg_sr, method='multi_band')
                    
                    elif feature_extraction_method == 'band_powers':  
                        feat = self._extract_band_power_features(eeg) 
                        
                    elif feature_extraction_method == 'combined':  
                        feat = self._extract_combined_features(eeg)
            
                    elif hasattr(self.decoder, 'custom_feature_extraction'):
                        feat = self.decoder.custom_feature_extraction(eeg, self.config.eeg_sr, method=feature_extraction_method)
                    else:
                        raise ValueError(f"Unknown feature extraction method: {feature_extraction_method}")
                
                else:
                    raise ValueError("No decoder available for feature extraction")
                   
                if feat.shape[0] == 0:
                    word = enhanced_batch['phoneme_words'][idx]
                    phoneme = enhanced_batch['phoneme_labels'][idx]
                    self.debug(f"Skipping segment {idx}: extractHG returned 0 frames - "
                               f"Patient {pid}, phoneme '{phoneme}' in word '{word}'")
                    continue
                
                patient_features[pid].append({
                    'idx': idx,
                    'feat': feat,
                    'label': enhanced_batch['phoneme_labels'][idx],
                    'word': enhanced_batch['phoneme_words'][idx],
                    'position': enhanced_batch['phoneme_positions'][idx],
                    'spectrogram': enhanced_batch['phoneme_spectrogram_segments'][idx],
                    'instance_idx': enhanced_batch['phoneme_instance_indices'][idx]
                })
                
            except Exception as e:
                self.log(f"Error processing segment {idx}: {e}")
            
        if not patient_features:
            self.log("No features extracted successfully")
            return {
                    'features': [],
                    'spectrograms': [],
                    'phoneme_labels': [],
                    'phoneme_words': [],
                    'phoneme_positions': [],
                    'phoneme_participant_ids': [],
                    'metadata': {
                        'feature_extraction_method': feature_extraction_method,
                        'n_phonemes': 0,
                        'unique_phonemes': 0
                    }
                }
            
            
        # SECOND PASS: Filter features and build final lists
        features = []
        spectrograms = []
        phoneme_labels = []
        phoneme_words = []
        phoneme_positions = []
        phoneme_participant_ids = []
        phoneme_instance_indices = []
        
        for pid, patient_data in patient_features.items():
            # Find expected dimension for this patient
            feature_dims = [d['feat'].shape[1] for d in patient_data]
            expected_dim = Counter(feature_dims).most_common(1)[0][0]
            
            self.debug(f"Patient {pid}: expected dimension = {expected_dim} (from {len(patient_data)} segments)")
            
            for data in patient_data:
                features.append(data['feat'])
                #aligned_feat = data['feat'][:5, :] if data['feat'].shape[0] >= 5 else np.pad(
                #    data['feat'], 
                #    ((0, 5 - data['feat'].shape[0]), (0, 0)), 
                #    mode='edge'
                #)
                #features.append(aligned_feat)
                spectrograms.append(data['spectrogram'])
                phoneme_labels.append(data['label'])
                phoneme_words.append(data['word'])
                phoneme_positions.append(data['position'])
                phoneme_participant_ids.append(pid)
                phoneme_instance_indices.append(data['instance_idx'])
        
        self.debug(f"Processed {len(features)} segments successfully")
          
            
        # Create result dictionary
        result = {
                'features': features,
                'spectrograms': spectrograms,
                'phoneme_labels': phoneme_labels,
                'phoneme_words': phoneme_words,
                'phoneme_positions': phoneme_positions,
                'phoneme_participant_ids': phoneme_participant_ids,
                'phoneme_instance_indices': phoneme_instance_indices, 
                'phoneme_durations_samples': [enhanced_batch['phoneme_durations_samples'][idx] 
                                   for idx in valid_indices],
                'metadata': {
                    'feature_extraction_method': feature_extraction_method,
                    'n_phonemes': len(phoneme_labels),
                    'unique_phonemes': len(set(phoneme_labels))
                }
            }
            
        self.debug(f"Prepared data for {result['metadata']['n_phonemes']} phoneme segments")
        self.debug(f"Found {result['metadata']['unique_phonemes']} unique phonemes")
            
        return result              
        
    def _build_batch_from_instances(self, instances, word_segments_dict):
        """helper class for accumulate_phoneme_data()"""
        batch = {
            'words': [],
            'eeg_segments': [],
            'spectrogram_segments': [],
            'audio_segments': [],
            'audio_sr': [],
            'participant_ids': [],
            'instance_indices': []
        }
        
        for inst in instances:
            pid = inst['participant_id']
            word = inst['word']
            idx = inst['instance_index']
                
            # Get the actual data from word_segments_dict
            try:
                word_data = word_segments_dict[pid]['words'][word]['instances'][idx]
                    
                batch['words'].append(word)
                batch['eeg_segments'].append(word_data['eeg_segment'])
                batch['spectrogram_segments'].append(word_data['spectrogram_segment'])
                batch['audio_segments'].append(word_data.get('audio_segment'))  
                batch['participant_ids'].append(pid)
                batch['instance_indices'].append(idx)
            except (KeyError, IndexError) as e:
                self.log(f"Warning: Could not load instance {pid}/{word}/{idx}: {e}")
                continue
            
        return batch
        
    def _detect_boundaries_rms(self, spectrogram, n_phonemes, audio_segment, audio_sr, participant_id=None, word=None):
        """
        RMS-based boundary detection as fallback.
        
        Args:
            spectrogram: Spectrogram array.
            n_phonemes: Expected number of phonemes.
            audio_segment: Audio waveform.
            audio_sr: Audio sample rate.
            participant_id: For logging.
            word: For logging.
            
        Returns:
            Dict with 'boundaries' and 'boundary_samples'.
        """
        # Use existing compute_rms_boundaries method
        boundaries, rms_change = self.compute_rms_boundaries(
            audio_segment,
            audio_sr,
            n_phonemes=n_phonemes
        )
        
        # Convert to EEG samples
        boundary_times = boundaries * self.config.frameshift
        boundary_samples = np.round(boundary_times * self.config.eeg_sr).astype(int)

        
        # Update boundaries to match
        boundaries = np.round(boundary_samples / self.config.eeg_sr / self.config.frameshift).astype(int)
        
        return {
            'boundaries': boundaries,
            'boundary_samples': boundary_samples
        }
        
    def compute_rms_boundaries(self, audio_segment, audio_sr, n_phonemes=None):
        """
        Detect phoneme boundaries using RMS energy changes.
        """
        self.debug("Computing RMS-based boundaries...")
        
        # Compute RMS with fine temporal resolution
        hop_length = int(self.config.rms_hop_ms / 1000.0 * audio_sr)
        frame_length = int(self.config.rms_frame_ms / 1000.0 * audio_sr)
        
        # Manual RMS computation (replaces librosa.feature.rms)
        from numpy.lib.stride_tricks import sliding_window_view

        audio_float = audio_segment.astype(np.float32)
        n_frames = 1 + (len(audio_float) - frame_length) // hop_length

        if len(audio_float) >= frame_length:
            frames = sliding_window_view(audio_float, frame_length)[::hop_length]
            rms = np.sqrt(np.mean(frames ** 2, axis=1))
        else:
            # Audio too short, single frame
            rms = np.array([np.sqrt(np.mean(audio_float ** 2))])
        
        # Smooth RMS
        rms_smoothed = gaussian_filter1d(rms, sigma=self.config.rms_smoothing_sigma)

        # Compute RMS change
        rms_change = np.abs(np.gradient(rms_smoothed))
        rms_change_smoothed = gaussian_filter1d(rms_change, sigma=self.config.rms_change_smoothing_sigma)

        # STEP 1: Detect speech onset (first significant peak)
        max_rms_change = np.max(rms_change_smoothed)
        onset_threshold = max_rms_change * self.config.onset_threshold_fraction
        
        # Find first point above onset threshold
        onset_candidates = np.where(rms_change_smoothed > onset_threshold)[0]
        
        if len(onset_candidates) > 0:
            onset_frame = onset_candidates[0]
            self.debug(f"  Speech onset detected at frame {onset_frame} ({onset_frame * hop_length / audio_sr:.3f}s)")
        else:
            onset_frame = 0
            self.debug(f"  No clear onset, using start of segment")
        
        # STEP 2: Find internal phoneme boundaries (after onset)
        # Use adaptive threshold based on median + MAD
        median_val = np.median(rms_change_smoothed)
        mad = np.median(np.abs(rms_change_smoothed - median_val))
        
        k = self.config.rms_k_factor
        threshold = median_val + k * mad
        
        # Minimum distance between boundaries
        min_phoneme_duration_sec = self.config.min_phoneme_duration
        min_distance_frames = int(min_phoneme_duration_sec / (hop_length / audio_sr))
        
        self.debug(f"  Internal boundary threshold: {threshold:.4f} (median: {median_val:.4f}, MAD: {mad:.4f})")
        
        # Find peaks AFTER onset
        
        
        # Search for peaks starting from onset
        search_start = max(0, onset_frame - 2)  # Start slightly before onset
        rms_change_after_onset = rms_change_smoothed[search_start:]
        
        peaks_relative, properties = find_peaks(
            rms_change_after_onset,
            height=threshold,
            distance=min_distance_frames,
            prominence=0.02 * np.max(rms_change_after_onset)
        )
        
        # Convert back to absolute frame indices
        peaks = peaks_relative + search_start
        
        self.debug(f"  Found {len(peaks)} internal boundaries (peaks after onset)")
        
        # STEP 3: Adjust number of boundaries if we know expected phonemes
        if n_phonemes is not None and n_phonemes > 1:
            n_boundaries_needed = n_phonemes - 1
            
            self.debug(f"  Need {n_boundaries_needed} internal boundaries for {n_phonemes} phonemes")
            
            if len(peaks) > n_boundaries_needed:
                # Keep strongest peaks
                peak_heights = rms_change_smoothed[peaks]
                strongest_indices = np.argsort(peak_heights)[-n_boundaries_needed:]
                peaks = peaks[strongest_indices]
                peaks = np.sort(peaks)
                self.debug(f"  Selected {len(peaks)} strongest peaks")
                
            elif len(peaks) < n_boundaries_needed:
                # Lower threshold to find more
                self.debug(f"  Only found {len(peaks)} peaks, need {n_boundaries_needed}")
                
                for attempt in range(3):
                    threshold *= self.config.threshold_reduction_factor
                    self.debug(f"    Attempt {attempt+1}: lowering threshold to {threshold:.4f}")
                    
                    peaks_relative, properties = find_peaks(
                        rms_change_after_onset,
                        height=threshold,
                        distance=min_distance_frames,
                        prominence=0.01 * np.max(rms_change_after_onset)
                    )
                    
                    peaks = peaks_relative + search_start
                    self.debug(f"    Found {len(peaks)} peaks")
                    
                    if len(peaks) >= n_boundaries_needed:
                        peak_heights = rms_change_smoothed[peaks]
                        strongest_indices = np.argsort(peak_heights)[-n_boundaries_needed:]
                        peaks = peaks[strongest_indices]
                        peaks = np.sort(peaks)
                        break
        
        # STEP 4: Convert from audio frames to spectrogram frames
        spec_hop_samples = int(self.config.frameshift * audio_sr)
        
        # Convert onset
        onset_spec_frame = int(onset_frame * hop_length / spec_hop_samples)
        
        # Convert internal boundaries
        spec_frame_boundaries = np.round(peaks * hop_length / spec_hop_samples).astype(int)
        
        # IMPORTANT: Use onset as first boundary (not 0)
        # This captures the first phoneme properly
        boundaries = np.concatenate(
            [[onset_spec_frame],  # Start at detected onset, not 0
             spec_frame_boundaries, 
             [int(len(audio_segment) / spec_hop_samples)]]  # End of segment
        )
        
        # Remove duplicates and ensure sorted
        boundaries = np.unique(boundaries)
        
        self.debug(f"  Final boundaries (spec frames): {boundaries}")
        self.debug(f"  Number of segments: {len(boundaries) - 1}")
        
        return boundaries, rms_change_smoothed
        
    def segment_sentence_by_rms(self, audio_sentence: np.ndarray, audio_sr: int, 
                            words: list, phonetic_dict) -> dict:
        """
        Segment entire sentence using RMS peaks, then group into words.
        
        Parameters:
        -----------
        audio_sentence : ndarray
            Full sentence audio
        audio_sr : int
            Audio sampling rate
        words : list
            List of words in sentence
        phonetic_dict : PhoneticDictionary
            For looking up expected phoneme counts
            
        Returns:
        --------
        dict with 'word_boundaries', 'phoneme_boundaries', 'word_segments'
        """
        self.debug(f"Segmenting sentence: {words}")
        
        # Get expected phoneme counts for each word
        word_phoneme_counts = []
        for word in words:
            phonemes = phonetic_dict.extract_phonemes(word)
            count = len(phonemes) if phonemes else 3
            word_phoneme_counts.append(count)
        
        total_phonemes = sum(word_phoneme_counts)
        self.debug(f"  Expected: {total_phonemes} total phonemes across {len(words)} words")
        
        # Compute RMS change for entire sentence
        hop_length = int(self.config.rms_hop_ms / 1000.0 * audio_sr)
        frame_length = int(self.config.rms_frame_ms / 1000.0 * audio_sr)

        rms = librosa.feature.rms(
            y=audio_sentence,
            frame_length=frame_length,
            hop_length=hop_length
        )[0]

        from scipy.ndimage import gaussian_filter1d
        rms_smoothed = gaussian_filter1d(rms, sigma=self.config.rms_smoothing_sigma)
        rms_change = np.abs(np.gradient(rms_smoothed))
        rms_change_smoothed = gaussian_filter1d(rms_change, sigma=self.config.rms_change_smoothing_sigma)

        # Find speech onset and offset
        max_rms_change = np.max(rms_change_smoothed)
        onset_threshold = max_rms_change * self.config.onset_threshold_fraction
        above_onset = np.where(rms_change_smoothed > onset_threshold)[0]
        
        if len(above_onset) > 0:
            speech_start = above_onset[0]
            speech_end = above_onset[-1]
        else:
            speech_start = 0
            speech_end = len(rms_change_smoothed) - 1
        
        # Focus on speech region
        rms_change_speech = rms_change_smoothed[speech_start:speech_end]
        
        # Find ALL internal boundaries
        median_val = np.median(rms_change_speech)
        mad = np.median(np.abs(rms_change_speech - median_val))
        threshold = median_val + self.config.sentence_k_factor * mad
        
        min_phoneme_duration_sec = self.config.min_phoneme_duration
        min_distance_frames = int(min_phoneme_duration_sec / (hop_length / audio_sr))
        
        # Find peaks
        n_boundaries_needed = total_phonemes - 1
        
        for attempt in range(5):
            peaks, _ = find_peaks(
                rms_change_speech,
                height=threshold,
                distance=min_distance_frames,
                prominence=0.01 * np.max(rms_change_speech)
            )
            
            self.debug(f"  Attempt {attempt+1}: threshold={threshold:.4f}, found {len(peaks)} peaks (need {n_boundaries_needed})")
            
            if len(peaks) >= n_boundaries_needed:
                # Keep strongest n peaks
                if len(peaks) > n_boundaries_needed:
                    peak_heights = rms_change_speech[peaks]
                    strongest = np.argsort(peak_heights)[-n_boundaries_needed:]
                    peaks = peaks[strongest]
                    peaks = np.sort(peaks)
                break
            else:
                threshold *= 0.75  # Lower threshold
        
        # Convert peaks to audio samples
        peaks_absolute = peaks + speech_start
        onset_sample = speech_start * hop_length
        
        phoneme_boundaries_samples = [onset_sample]
        for peak in peaks_absolute:
            phoneme_boundaries_samples.append(peak * hop_length)
        phoneme_boundaries_samples.append(len(audio_sentence))
        
        phoneme_boundaries_samples = np.array(phoneme_boundaries_samples)
        
        # Now group phonemes into words based on expected counts
        word_boundaries_samples = [phoneme_boundaries_samples[0]]
        
        phoneme_idx = 0
        for word_idx, n_phonemes in enumerate(word_phoneme_counts):
            # This word should span n_phonemes
            phoneme_idx += n_phonemes
            
            # Word ends at the boundary after these phonemes
            if phoneme_idx < len(phoneme_boundaries_samples):
                word_boundaries_samples.append(phoneme_boundaries_samples[phoneme_idx])
            else:
                # Last word - use end of sentence
                word_boundaries_samples.append(phoneme_boundaries_samples[-1])
        
        word_boundaries_samples = np.array(word_boundaries_samples)
        
        # Extract word segments
        word_segments = []
        for i in range(len(word_boundaries_samples) - 1):
            start = int(word_boundaries_samples[i])
            end = int(word_boundaries_samples[i + 1])
            word_segments.append(audio_sentence[start:end])
        
        self.debug(f"  Result: {len(word_segments)} words extracted")
        
        return {
            'word_boundaries_samples': word_boundaries_samples,
            'phoneme_boundaries_samples': phoneme_boundaries_samples,
            'word_segments': word_segments,
            'rms_change': rms_change_smoothed,
            'speech_start': speech_start,
            'speech_end': speech_end
        }
        
    def segment_sentence_by_wav2vec(
        self,
        audio_signal,
        sample_rate,
        word_list,
        patient_id,
        sentence_id,
        sigma=0
    ):
        """
        Segment sentence audio into word boundaries using wav2vec2 distance peaks.
        
        Args:
            audio_signal: Audio time series
            sample_rate: Audio sampling rate in Hz
            word_list: List of expected words in order
            patient_id: Patient identifier
            sentence_id: Sentence identifier
            sigma: Gaussian smoothing parameter (0 = no smoothing)
            
        Returns:
            List of tuples: [(start_ms, end_ms), ...] for each word boundary
        """
        from dataset_config import WAV2VEC_MODEL_NAME
        
        n_words = len(word_list)
        n_boundaries_needed = n_words - 1
        
        wav2vec2_distances = self._compute_wav2vec2_distances(
            audio_signal, 
            sample_rate
        )
        
        if sigma > 0:
            from scipy.ndimage import gaussian_filter1d
            wav2vec2_distances = gaussian_filter1d(
                wav2vec2_distances, 
                sigma=sigma
            )
        
        peak_prominence = self.config.get('peak_prominence', 0.0)
        peaks, properties = find_peaks(
            wav2vec2_distances,
            prominence=peak_prominence,
            height=0
        )
        
        print(f"words expected: {n_words}")
        print(f"peaks found: {len(peaks)}")
        
        if len(peaks) == 0:
            audio_duration_ms = int((len(audio_signal) / sample_rate) * 1000)
            print(f"word segments returned: 1")
            print(f"  '{word_list[0]}': {audio_duration_ms}ms")
            return [(0, audio_duration_ms)]
        
        if len(peaks) < n_boundaries_needed:
            print(f"WARNING: Found {len(peaks)} peaks but need {n_boundaries_needed} boundaries")
        
        peak_times_ms = (peaks / len(wav2vec2_distances)) * (len(audio_signal) / sample_rate) * 1000
        
        selected_boundaries = self._select_word_boundaries(
            peak_times_ms,
            n_boundaries_needed
        )
        
        audio_duration_ms = int((len(audio_signal) / sample_rate) * 1000)
        word_segments = []
        start_ms = 0
        
        for boundary_ms in selected_boundaries:
            word_segments.append((int(start_ms), int(boundary_ms)))
            start_ms = boundary_ms
        
        word_segments.append((int(start_ms), audio_duration_ms))
        
        print(f"word segments returned: {len(word_segments)}")
        for idx, (word, (start_ms, end_ms)) in enumerate(zip(word_list, word_segments)):
            duration_ms = end_ms - start_ms
            print(f"  '{word}': {duration_ms}ms")
        
        return word_segments
        
    def extract_wav2vec_features(self, audio_segment, audio_sr):
        """
        Extract wav2vec 2.0 features from audio.
        
        Returns:
            np.ndarray: (n_frames, 768) - contextualized audio features
        """
        # Initialize models once (cache them)
        if not hasattr(self, 'wav2vec_processor'):
            #self.wav2vec_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
            #self.wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")

            self.wav2vec_processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-large-xlsr-53")
            self.wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-xlsr-53")
            self.wav2vec_model.eval()
        
        # CRITICAL: Resample to 16kHz if needed
        target_sr = self.config.audio_target_sr
        if audio_sr != target_sr:
            from scipy.signal import resample_poly
            downsample_factor = int(audio_sr / target_sr)
            audio_resampled = resample_poly(audio_segment.astype(np.float32), up=1, down=downsample_factor)
            self.debug(f"Resampled audio: {audio_sr}Hz → {target_sr}Hz")
        else:
            audio_resampled = audio_segment
        
        # Preprocess audio
        inputs = self.wav2vec_processor(
            audio_resampled,  # Use resampled audio
            sampling_rate=target_sr,  
            return_tensors="pt"
        )
        
        # Extract features
        with torch.no_grad():
            outputs = self.wav2vec_model(**inputs)
            features = outputs.last_hidden_state.squeeze(0).numpy()
            
        return features  # Shape: (time_frames, 768)
        
    def compute_wav2vec_distances(self, wav2vec_features):
        """
        Compute frame-to-frame distances in wav2vec feature space.
        
        Parameters:
        -----------
        wav2vec_features : np.ndarray
            Shape (n_frames, 768)
            
        Returns:
        --------
        np.ndarray: (n_frames-1,) - distance between consecutive frames
        """
        # Compute Euclidean distance between consecutive frames
        distances = np.sqrt(np.sum((wav2vec_features[1:] - wav2vec_features[:-1])**2, axis=1))
        
        # Alternative: Cosine distance
        # from scipy.spatial.distance import cosine
        # distances = np.array([cosine(wav2vec_features[i], wav2vec_features[i+1]) 
        #                      for i in range(len(wav2vec_features)-1)])
        
        return distances
    
    def _adaptive_peak_detection(self, distances, n_phonemes, participant_id=None, word=None):
        """
        Adaptively find peaks by adjusting threshold until we get the right number.
        
        Args:
            distances: Array of frame-to-frame distances.
            n_phonemes: Expected number of phonemes.
            participant_id: For logging.
            word: For logging.
            
        Returns:
            Array of boundary indices including start (0) and end.
        """
        from scipy.signal import find_peaks
        from scipy.ndimage import gaussian_filter1d
        
        n_boundaries_needed = n_phonemes - 1
        
        distances_smooth = self._smooth_distances(distances, self.config.wav2vec_phoneme_sigma)
        mean_dist = np.mean(distances_smooth)
        std_dist = np.std(distances_smooth)
        
        min_dist_frames = max(1, int(self.config.min_phoneme_duration / self.config.frameshift))
        
        best_peaks = None
        best_diff = float('inf')
        
        threshold_factors = getattr(self.config, 'adaptive_threshold_factors', None)
        if threshold_factors is None:
            threshold_factors = [0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.7, 0.8, 0.9, 1.0, 1.2]
        
        for factor in threshold_factors:
            threshold = mean_dist + factor * std_dist
            
            peaks, properties = find_peaks(
                distances_smooth,
                height=threshold,
                distance=min_dist_frames,
                prominence=getattr(self.config, 'adaptive_prominence_factor', 0.01) * np.max(distances_smooth)
            )

            diff = abs(len(peaks) - n_boundaries_needed)
            
            if diff < best_diff:
                best_diff = diff
                best_peaks = peaks
                
                if diff == 0:
                    self.debug(f"  Found exact match with threshold factor {factor}")
                    break
            
            if len(peaks) == n_boundaries_needed:
                break
            elif len(peaks) > n_boundaries_needed:
                peak_heights = distances_smooth[peaks]
                strongest_indices = np.argsort(peak_heights)[-n_boundaries_needed:]
                best_peaks = np.sort(peaks[strongest_indices])
                best_diff = 0
                break
        
        if best_peaks is None or len(best_peaks) == 0:
            self.debug(f"  No peaks found, using equal spacing for '{word}'")
            best_peaks = np.linspace(0, len(distances), n_phonemes + 1)[1:-1].astype(int)
        
        if best_diff > 0:
            if len(best_peaks) < n_boundaries_needed:
                self.log(f"  Need {n_boundaries_needed} peaks but only found {len(best_peaks)} for word '{word}' (Patient {participant_id})")
            
            if len(best_peaks) > n_boundaries_needed:
                peak_heights = distances_smooth[best_peaks]
                strongest_indices = np.argsort(peak_heights)[-n_boundaries_needed:]
                best_peaks = np.sort(best_peaks[strongest_indices])
        
        boundaries = np.concatenate([[0], best_peaks, [len(distances)]])
        
        return boundaries
        
    def _extend_short_segments(self, boundary_samples, eeg_length, min_samples):
        """
        Extend short segments by creating overlaps with neighbors.
        
        Args:
            boundary_samples: Array of boundary positions in EEG samples.
            eeg_length: Total length of EEG segment.
            min_samples: Minimum samples per segment.
            
        Returns:
            List of (start, end) tuples for each segment, may have overlaps.
        """
        n_segments = len(boundary_samples) - 1
        segments = []
        
        for i in range(n_segments):
            start = boundary_samples[i]
            end = boundary_samples[i + 1]
            duration = end - start
            
            if duration >= min_samples:
                # Segment is long enough, use as-is
                segments.append((start, end))
            else:
                # Segment too short - extend symmetrically
                shortfall = min_samples - duration
                extend_before = shortfall // 2
                extend_after = shortfall - extend_before
                
                new_start = max(0, start - extend_before)
                new_end = min(eeg_length, end + extend_after)
                
                # If still too short (at boundaries), extend more in available direction
                if new_end - new_start < min_samples:
                    if new_start == 0:
                        new_end = min(eeg_length, new_start + min_samples)
                    elif new_end == eeg_length:
                        new_start = max(0, new_end - min_samples)
                
                segments.append((new_start, new_end))
                self.debug(f"  Segment {i}: extended [{start}:{end}] -> [{new_start}:{new_end}]")
        
        return segments
    
    def _extract_fixed_window(self, eeg_segment, target_samples):
        """
        Extract a fixed-size window from the center of the segment.
        Short segments are padded with edge values, long segments are truncated.
        
        Args:
            eeg_segment: EEG data array (n_samples, n_channels)
            target_samples: Fixed window size to extract
            
        Returns:
            Fixed-size EEG segment (target_samples, n_channels)
        """
        n_samples = eeg_segment.shape[0]
        
        if n_samples == target_samples:
            return eeg_segment
        
        elif n_samples > target_samples:
            # Truncate from center
            start = (n_samples - target_samples) // 2
            return eeg_segment[start:start + target_samples]
        
        else:
            # Pad with edge values
            pad_total = target_samples - n_samples
            pad_before = pad_total // 2
            pad_after = pad_total - pad_before
            return np.pad(eeg_segment, ((pad_before, pad_after), (0, 0)), mode='edge')