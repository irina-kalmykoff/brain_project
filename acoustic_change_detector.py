import matplotlib.pyplot as plt
from scipy import signal
from scipy.spatial.distance import cosine, euclidean
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, savgol_filter
from sklearn.preprocessing import normalize
import os
import numpy as np
from collections import Counter, defaultdict

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from extract_features import extractHG
from debugger import DebugMixin
from phonetic_dictionary import PhoneticDictionary


class AcousticChangeDetector(DebugMixin):
    """
    Detects acoustic changes in spectrograms to identify potential phoneme boundaries.
    Uses unsupervised methods to find significant changes in acoustic features.
    """
    
    def __init__(self, min_segment_duration=0.06, max_segment_duration=0.4,     
                 distance_metric='cosine', smoothing_window=3, peak_threshold=0.75, 
                 decoder=None, debug_mode=None, phonetic_dict=None, feature_extraction_method='high_gamma'):
        """
        phoneme duration
        Short consonants (stops, t, k, p): 30-80ms
        Regular consonants (s, f, n): 80-150ms
        Short vowels (ə, ɪ): 50-120ms
        Regular vowels (a, e, o): 100-200ms
        Long vowels/diphthongs (aː, eː, ɛi): 150-400ms
        Rarely >400ms (unless emphatic or includes silence)
        """
        
        """
        Initialize with parameters to control boundary detection sensitivity.
        """
        # Initialize the DebugMixin
        super().__init__(class_name="AcousticChangeDetector", debug_mode=False)
        
        if debug_mode is not None:
            self.DEBUG_MODE = debug_mode
        self.log(f"Initialized with DEBUG_MODE={self.DEBUG_MODE}")
        

        self.min_segment_duration = min_segment_duration
        self.max_segment_duration = max_segment_duration
        self.distance_metric = distance_metric
        self.smoothing_window = smoothing_window
        self.peak_threshold = peak_threshold 
        self.decoder = decoder
        self.phonetic_dict = phonetic_dict or PhoneticDictionary()
        # Store the feature extraction method
        self.feature_extraction_method = feature_extraction_method
        self.log(f"Using feature extraction method: {self.feature_extraction_method}")
    
    def count_phonemes(self, word):
        """
        Count the number of phonemes in a word based on its transcription.
        
        """
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
        
        # Calculate frame-to-frame distances
        for i in range(num_frames - 1):
            distances[i] = distance_func(spectrogram[i], spectrogram[i+1])
        
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
        # Initialize flux array
        num_frames = spectrogram.shape[0]
        flux = np.zeros(num_frames - 1)
        
        # Calculate spectral flux (sum of positive differences)
        for i in range(num_frames - 1):
            diff = spectrogram[i+1] - spectrogram[i]
            flux[i] = np.sum(np.maximum(0, diff))  # Half-wave rectification
        
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
    
    def detect_peaks(self, enhanced_distances, n_phonemes=None, frameshift=0.01, participant_id = None, word=None, word_position=None):
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
        # Calculate minimum distance between peaks in frames
        min_dist_frames = max(1, int(self.min_segment_duration / frameshift))
        
        # Use median-based threshold instead of max-based
        
        median_val = np.median(enhanced_distances)
        mad = np.median(np.abs(enhanced_distances - median_val))  # Median Absolute Deviation
        
        # Adaptive threshold: median + k * MAD
        k = 1.5  # 2 - strict, 1.5 detects subtler transisions
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
            prominence=0.05 # previously 0.1, cahnged to detect smaller peaks 
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
                self.log(f"  Need {n_boundaries} peaks but only found {len(peaks)}{word_info}{patient_info}{position_info}")
                
                for attempt in range(3):
                    height = height * 0.6  # More aggressive reduction
                    self.debug(f"    Attempt {attempt+1}: lowering threshold to {height:.4f}")
                    
                    peaks, properties = find_peaks(
                        enhanced_distances,
                        height=height,
                        distance=min_dist_frames,
                        prominence=0.05  # Lower prominence requirement
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
    
    def refine_boundaries(self, boundaries, spectrogram, frameshift=0.01):
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
        min_frames = max(1, int(self.min_segment_duration / frameshift))
        max_frames = max(min_frames + 1, int(self.max_segment_duration / frameshift))
        
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
                         frameshift=0.01, participant_id = None, word_position=None):
        """
        Main method to detect phoneme boundaries in a word spectrogram.
        
        Parameters:
        -----------
        spectrogram : ndarray
            Mel spectrogram of the word
        word : str or None
            The word being analyzed
        phonetic_transcription : str or None
            Phonetic transcription if available
        frameshift : float
            Time between consecutive frames in seconds
            
        Returns:
        --------
        dict
            Dictionary containing:
            - 'boundaries': list of boundary indices
            - 'boundary_times': list of boundary times in seconds
            - 'segments': list of spectrogram segments
            - 'distances': frame-to-frame distances
            - 'enhanced_distances': enhanced distance curve
            - 'word': original word
            - 'n_phonemes': estimated number of phonemes
        """
        self.debug(f"Detecting boundaries for word: {word if word else 'unknown'}")
        
        # Determine number of phonemes
        n_phonemes = None
        if word is not None:
            n_phonemes = self.count_phonemes(word)
            n_phonemes = self.count_phonemes(word)
            self.debug(f"Estimated {n_phonemes} phonemes for '{word}'")
        
        # Calculate frame-to-frame distances
        distances = self.compute_frame_distances(spectrogram)
        
        # Calculate additional features
        flux = self.compute_spectral_flux(spectrogram)
        energy = self.compute_energy_contour(spectrogram)
        
        # Enhance transitions
        enhanced_distances = self.enhance_transitions(distances, flux, energy)
        
        # Detect peaks
        boundaries = self.detect_peaks(
            enhanced_distances, 
            n_phonemes, 
            frameshift, 
            participant_id=participant_id, 
            word=word,
            word_position=word_position
        )
        
        # Refine boundaries
        refined_boundaries = self.refine_boundaries(boundaries, spectrogram, frameshift)
        
        # Extract segments
        segments = []
        for i in range(len(refined_boundaries) - 1):
            start = refined_boundaries[i]
            end = refined_boundaries[i + 1]
            
            # Ensure valid indices
            start = max(0, start)
            end = min(spectrogram.shape[0], end)
            
            if start < end:
                segment = spectrogram[start:end]
                segments.append(segment)
        
        # Calculate boundary times
        boundary_times = refined_boundaries * frameshift        


        # Create result dictionary
        result = {
            'boundaries': refined_boundaries,
            'boundary_times': boundary_times,
            'segments': segments,
            'distances': distances,
            'enhanced_distances': enhanced_distances,
            'word': word,
            'n_phonemes': n_phonemes,
            'energy': energy
        }
        
        eeg_sr = 1024  # EEG sampling rate
        boundary_samples = np.round(boundary_times * eeg_sr).astype(int)
        result['boundary_samples'] = boundary_samples
        
        return result
    
    def process_word_segment(self, word_segment, participant_id=None, frameshift=0.01):
        """
        Process a word segment from segment_data_by_words output.
        
        Parameters:
        -----------
        word_segment : dict
            Word segment dictionary from segment_data_by_words
        participant_id : str or None
            Participant ID for reference
        frameshift : float
            Time between consecutive frames in seconds
            
        Returns:
        --------
        dict
            Dictionary containing phoneme segmentation results
        """
        word = word_segment.get('word', None)
        
        # Get spectrogram if available
        if 'spectrogram_segment' in word_segment:
            spectrogram = word_segment['spectrogram_segment']
        else:
            self.debug("Warning: No spectrogram found in word segment")
            return None
        
        # Detect boundaries
        result = self.detect_boundaries(spectrogram, word, frameshift=frameshift)
        
        # Add additional metadata
        result['participant_id'] = participant_id
        result['word_onset_sample'] = word_segment.get('onset_sample', None)
        result['word_offset_sample'] = word_segment.get('offset_sample', None)
        
        # Extract EEG segments if available
        if 'eeg_segment' in word_segment:
            eeg = word_segment['eeg_segment']
            
            # Calculate EEG sample indices for boundaries
            eeg_sr = word_segment.get('eeg_sr', 1024)  # Default EEG sampling rate
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
    
    def process_word_segments_dict(self, word_segments_dict, participant_id):
        """
        Process all word segments for a participant.
        
        Parameters:
        -----------
        word_segments_dict : dict
            Output from segment_data_by_words
        participant_id : str
            Participant ID
            
        Returns:
        --------
        dict
            Dictionary containing phoneme segmentation results for all words
        """
        self.debug(f"Processing word segments for {participant_id}")
        
        # Get metadata
        metadata = word_segments_dict.get('metadata', {})
        frameshift = metadata.get('frameshift', 0.01)
        
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
    
    def process_batch(self, batch, apply_segmentation=True, detect_phonemes=True, frameshift=0.01, skip_mismatches=True):
        """
        Process a batch of data from get_data_batch to extract phoneme-level features.
    
        Parameters:
        -----------
        batch : dict
            Output from get_data_batch function
        apply_segmentation : bool
            Whether to apply phoneme segmentation
        detect_phonemes : bool
            Whether to try detecting phonemes
        frameshift : float
            Time between consecutive frames in seconds
        skip_mismatches : bool
            If True, skip words where detected segments don't match expected phonemes
            
        Returns:
        --------
        dict
            Enhanced batch with phoneme-level segmentation
        """
        self.debug(f"Processing batch with {len(batch.get('words', []))} instances")
        self.debug(f"Original batch keys: {list(batch.keys())}")
        self.debug(f"EEG segments in batch: {'eeg_segments' in batch}")
        self.debug(f"Spectrogram segments in batch: {'spectrogram_segments' in batch}")
        
        
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
        word_boundaries = []  # Store phoneme boundaries for each word
        total_words = 0
        mismatched_words = 0
        perfect_match_words = 0
  
        # Process each instance in the batch
        for i, word in enumerate(batch.get('words', [])):
            if i % 10 == 0:
                self.debug(f"Processing instance {i}/{len(batch.get('words', []))}: {word}")
            
            # Get data for this instance
            eeg_segment = batch['eeg_segments'][i] if 'eeg_segments' in batch else None
            spectrogram_segment = batch['spectrogram_segments'][i] if 'spectrogram_segments' in batch else None
            participant_id = batch['participant_ids'][i] if 'participant_ids' in batch else None
        
            # Skip if required data is missing
            if spectrogram_segment is None:
                continue
            
            # Apply phoneme segmentation if requested
            if apply_segmentation and detect_phonemes and word is not None:
                
                #count words for stats
                if word in self.phonetic_dict:
                    total_words += 1 
                    
                # Detect boundaries
                result = self.detect_boundaries(
                    spectrogram=spectrogram_segment,
                    word=word,
                    frameshift=frameshift,
                    participant_id=participant_id, 
                    word_position=i
                )
                
                # Store boundaries
                word_boundaries.append(result['boundaries'])
                
                # Extract phoneme segments
                segments = result['segments']
                
                # Try to get phoneme transcription
                if word in self.phonetic_dict:
                    transcription = self.phonetic_dict[word]
                    # Clean transcription
                    cleaned = transcription.replace('ˈ', '').replace('(', '').replace(')', '').replace("'", '')
                    
                    # Extract individual phonemes
                    phonemes = []
                    i = 0
                    while i < len(cleaned):
                        # Check for complex phonemes (digraphs, diphthongs)
                        complex_found = False
                        # Sort by length to prevent substring matches
                        sorted_complex = sorted(['ɛi', 'œy', 'ɑu', 'ɵ:', 'ɛ:', 'a:', 'o:', 'e:', 'øk', 'ɔf', 
                            'ts', 'ŋk', 'sx', 'ɔx', 'ɪx', 'aː', 'eː', 'iː', 'oː', 'uː', 
                            'yː', 'øː', 'tʋ', 'ʋɑ', 'ʋɪ', 'ʋə'],
                           key=len, reverse=True)
                           
                        for cp in sorted_complex:
                            if i + len(cp) <= len(cleaned) and cleaned[i:i+len(cp)] == cp:
                                phonemes.append(cp)
                                i += len(cp)
                                complex_found = True
                                break
                        
                        # Check for length markers
                        if not complex_found and i + 1 < len(cleaned) and cleaned[i+1] == 'ː':
                            phonemes.append(cleaned[i:i+2])
                            i += 2
                            complex_found = True
                        
                        if not complex_found:
                            phonemes.append(cleaned[i])
                            i += 1
                    
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
                            
                            # Extract corresponding EEG segment if available
                            if eeg_segment is not None and result.get('boundary_samples') is not None:
                                boundaries = result['boundary_samples']
                                if j < len(boundaries) - 1:
                                    start = max(0, boundaries[j])
                                    end = min(eeg_segment.shape[0], boundaries[j + 1])
                                    
                                    if start < end:
                                        phoneme_eeg_segments.append(eeg_segment[start:end])
                                    else:
                                        phoneme_eeg_segments.append(np.array([]).reshape(0, eeg_segment.shape[1]))  # Empty with correct channels
                                else:
                                    phoneme_eeg_segments.append(np.array([]).reshape(0, eeg_segment.shape[1]))
                            else:
                                # No EEG data - append empty array with correct shape
                                if eeg_segment is not None:
                                    phoneme_eeg_segments.append(np.array([]).reshape(0, eeg_segment.shape[1]))
                                else:
                                    phoneme_eeg_segments.append(np.array([]))
                    else:
                        # Mismatch - use a simple distribution based on segment count
                        mismatched_words += 1
                        self.debug(f"Mismatch for word '{word}': {len(phonemes)} phonemes but {len(segments)} segments")
                        
                        # Still add segments with unknown phoneme labels
                        for j, segment in enumerate(segments):
                            phoneme_spectrogram_segments.append(segment)
                            # Use '?' as placeholder for unknown phoneme
                            phoneme_labels.append('?')
                            phoneme_words.append(word)
                            phoneme_positions.append(j)
                            phoneme_participant_ids.append(participant_id)
                            
                            # Extract corresponding EEG segment if available
                            if eeg_segment is not None and result.get('boundary_samples') is not None:
                                boundaries = result['boundary_samples']
                                if j < len(boundaries) - 1:
                                    start = boundaries[j]
                                    end = boundaries[j + 1]
                                    
                                    # Ensure valid indices
                                    start = max(0, start)
                                    end = min(eeg_segment.shape[0], end)
                                    
                                    if start < end:
                                        phoneme_eeg_segments.append(eeg_segment[start:end])
                                    else:
                                        phoneme_eeg_segments.append(np.array([]))
                                else:
                                    phoneme_eeg_segments.append(np.array([]))
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
                        
                        # Extract corresponding EEG segment if available
                        if eeg_segment is not None and result.get('boundary_samples') is not None:
                            boundaries = result['boundary_samples']
                            if j < len(boundaries) - 1:
                                start = boundaries[j]
                                end = boundaries[j + 1]
                                
                                # Ensure valid indices
                                start = max(0, start)
                                end = min(eeg_segment.shape[0], end)
                                
                                if start < end:
                                    phoneme_eeg_segments.append(eeg_segment[start:end])
                                else:
                                    phoneme_eeg_segments.append(np.array([]))
                            else:
                                phoneme_eeg_segments.append(np.array([]))
            else:
                # Use whole segments as is (no phoneme segmentation)
                if spectrogram_segment is not None:
                    phoneme_spectrogram_segments.append(spectrogram_segment)
                    phoneme_labels.append(word)  # Use word as the label
                    phoneme_words.append(word)
                    phoneme_positions.append(0)  # Single position
                    phoneme_participant_ids.append(participant_id)
                    
                    if eeg_segment is not None:
                        phoneme_eeg_segments.append(eeg_segment)
        
        # Create enhanced batch
        enhanced_batch = {
            'phoneme_spectrogram_segments': phoneme_spectrogram_segments,
            'phoneme_labels': phoneme_labels,
            'phoneme_words': phoneme_words,
            'phoneme_positions': phoneme_positions,
            'phoneme_participant_ids': phoneme_participant_ids,
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
            'segmentation_applied': apply_segmentation,
            'phoneme_detection_applied': detect_phonemes,
            'total_words': total_words,
            'mismatched_words': mismatched_words,
            'perfect_match_words': perfect_match_words,
            'mismatch_rate': mismatched_words / total_words if total_words > 0 else 0
        }
        
        self.debug(f"Enhanced batch contains {enhanced_batch['metadata']['phoneme_count']} phoneme segments")
        self.debug(f"Found {enhanced_batch['metadata']['unique_phonemes']} unique phonemes")
        
        return enhanced_batch
    
    def accumulate_phoneme_data(self, split_result, batch_size=32,
                                feature_extraction_method='high_gamma', 
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
            batch = self._build_batch_from_instances(batch_instances, word_segments_dict)

        
            # Get a new batch of training data
            #batch = self.decoder.get_data_batch(
            #    split_result=self.split_result,
            #    batch_type=batch_type,
            #    balanced_sampling=True,
            #    batch_size=batch_size,
            #    random_seed=42
            #)
            
            # Process batch to get phoneme-level data
            phoneme_batch = self.process_batch(batch)
            
            # Prepare features for model training
            phoneme_data = self.prepare_phoneme_training_data(
                phoneme_batch,
                feature_extraction_method=feature_extraction_method,
                standardize_channels=True,
                standardize_values=False
            )
            
             # Accumulate mismatch statistics
            if 'total_words' in phoneme_batch['metadata']:
                total_words_processed += phoneme_batch['metadata']['total_words']
                total_mismatches += phoneme_batch['metadata']['mismatched_words']
                total_perfect_matches += phoneme_batch['metadata']['perfect_match_words']

            
            # Accumulate data
            accumulated_features.extend(phoneme_data['features'])
            if 'spectrograms' in phoneme_data and phoneme_data['spectrograms']:
                accumulated_spectrograms.extend(phoneme_data['spectrograms'])
            accumulated_labels.extend(phoneme_data['phoneme_labels'])
            accumulated_words.extend(phoneme_data['phoneme_words'])
            accumulated_participant_ids.extend(phoneme_data['phoneme_participant_ids'])
            accumulated_positions.extend(phoneme_data.get('phoneme_positions', 
                                    [0] * len(phoneme_data['phoneme_labels'])))
            self.log(f"Accumulated {len(accumulated_features)} phoneme segments so far")
        
        # Create result dictionary
        accumulated_data = {
            'features': accumulated_features,
            'spectrograms': accumulated_spectrograms if accumulated_spectrograms else None,
            'phoneme_labels': accumulated_labels,
            'phoneme_words': accumulated_words,
            'phoneme_participant_ids': accumulated_participant_ids,
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
    
    def prepare_phoneme_training_data(self, enhanced_batch, **kwargs):
        """
        Prepare phoneme-level data for model training.
            
        Parameters:
        -----------
        enhanced_batch : dict
                Output from process_batch function
        feature_extraction_method : str
                Method to use for feature extraction ('high_gamma', 'multi_band', etc.)
        standardize : bool
                Whether to standardize features
        pca_components : int or None
                Number of PCA components to use. If None, don't use PCA.
                
        Returns: Dictionary containing processed data ready for phoneme-based model training
        """
        # Use decoder's config as base, override with kwargs
        if self.decoder is not None and hasattr(self.decoder, 'config'):
            config = self.decoder.config.copy()
            config.update(kwargs)
        else:
            # Fallback if no decoder is provided
            config = {
                    'feature_extraction_method': 'high_gamma',
                    'standardize_channels': True,   # RENAMED from just using standardize
                    'standardize_values': True,     # RENAMED - this is the problematic one
                    'pca_components': 150
            }
            config.update(kwargs)
                
        # Extract parameters from config
        feature_extraction_method = config.get('feature_extraction_method', 'high_gamma')
        standardize_channels = config.get('standardize_channels', True)  # NEW
        standardize_values = config.get('standardize_values', True) 
        pca_components = config.get('pca_components', None)
            
        self.debug(f"Preparing phoneme training data with PCA components={pca_components}")
            
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
            
            min_samples = 70  # Minimum to get at least 1 frame from extractHG
            if eeg.size == 0 or eeg.shape[0] < min_samples:
                self.debug(f"Skipping segment {idx}: too short ({eeg.shape[0]} samples)")
                continue
        
            try:
                # Extract features
                if self.decoder is not None:
                    if feature_extraction_method == 'high_gamma':
                        feat = extractHG(eeg, 1024)
                    elif feature_extraction_method == 'multi_band':
                        feat = self.decoder.custom_feature_extraction(eeg, 1024, method='multi_band')
                    elif hasattr(self.decoder, 'custom_feature_extraction'):
                        feat = self.decoder.custom_feature_extraction(eeg, 1024, method=feature_extraction_method)
                    else:
                        raise ValueError(f"Unknown feature extraction method: {feature_extraction_method}")
                
                else:
                    raise ValueError("No decoder available for feature extraction")
#added log printout                    
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
                    'spectrogram': enhanced_batch['phoneme_spectrogram_segments'][idx]
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
                        'standardized': standardize,
                        'pca_components': pca_components,
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
        
        for pid, patient_data in patient_features.items():
            # Find expected dimension for this patient
            feature_dims = [d['feat'].shape[1] for d in patient_data]
            expected_dim = Counter(feature_dims).most_common(1)[0][0]
            
            self.debug(f"Patient {pid}: expected dimension = {expected_dim} (from {len(patient_data)} segments)")
            
            for data in patient_data:
                features.append(data['feat'])
                spectrograms.append(data['spectrogram'])
                phoneme_labels.append(data['label'])
                phoneme_words.append(data['word'])
                phoneme_positions.append(data['position'])
                phoneme_participant_ids.append(pid)
        
        self.debug(f"Processed {len(features)} segments successfully")
            
        # Standardize feature shapes zero padding the channels that are missing to make all data have 133 channels 
        # (patients with less than 75 channels were filtered out before)
        
        if standardize_channels and features:  # ADD this condition
            max_channels = 133
            self.debug(f"Standardizing channels to {max_channels}")
        
            standardized_features = []
            for feat in features:
                result = np.zeros((feat.shape[0], max_channels))
                min_channels = min(feat.shape[1], max_channels)
                result[:, :min_channels] = feat[:, :min_channels]
                standardized_features.append(result)

            features = standardized_features
        
        # Standardize features if requested
        
        if standardize_values and features:
            scaler = StandardScaler()
            self.debug(f"Standardizing feature values (mean=0, std=1)")
            
            for i in range(len(features)):
                # Standardize each feature set independently
                features[i] = scaler.fit_transform(features[i])
            
        # Apply PCA if requested
        if pca_components is not None and pca_components > 0 and features:
            
            pca = PCA(n_components=min(pca_components, min(f.shape[1] for f in features)))
            
            # Stack all features for fitting
            try:
                all_features = np.vstack(features)
                pca.fit(all_features)
                
                # Transform all features
                for i in range(len(features)):
                    features[i] = pca.transform(features[i])
                
                # Store in decoder if available
                if self.decoder is not None and hasattr(self.decoder, 'pca_models'):
                    key = f"{feature_extraction_method}_phoneme"
                    self.decoder.pca_models[key] = pca
                    self.debug(f"Stored PCA model in decoder")
            except ValueError as e:
                self.log(f"Error during PCA: {e}. Continuing without PCA")
            
        # Create result dictionary
        result = {
                'features': features,
                'spectrograms': spectrograms,
                'phoneme_labels': phoneme_labels,
                'phoneme_words': phoneme_words,
                'phoneme_positions': phoneme_positions,
                'phoneme_participant_ids': phoneme_participant_ids,
                'metadata': {
                    'feature_extraction_method': feature_extraction_method,
                    'standardized_channels': standardize_channels,  # RENAMED
                    'standardized_values': standardize_values, 
                    'pca_components': pca_components,
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
            'participant_ids': []
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
                batch['participant_ids'].append(pid)
            except (KeyError, IndexError) as e:
                self.log(f"Warning: Could not load instance {pid}/{word}/{idx}: {e}")
                continue
        
        return batch

    