import matplotlib.pyplot as plt
from scipy import signal
from scipy.spatial.distance import cosine, euclidean
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, savgol_filter
from sklearn.preprocessing import normalize
import os
import numpy as np

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
    
    def __init__(self, min_segment_duration=0.03, max_segment_duration=0.3, 
                 distance_metric='cosine', smoothing_window=3, peak_threshold=0.5, 
                 decoder=None, debug_mode=None, phonetic_dict=None):
        """
        Initialize with parameters to control boundary detection sensitivity.
        """
        # Initialize the DebugMixin
        super().__init__(class_name="AcousticChangeDetector", debug_mode=False)
        
        if debug_mode is not None:
            self.DEBUG_MODE = debug_mode
        self.log(f"Initialized with DEBUG_MODE={self.DEBUG_MODE}")
        
        self.decoder = decoder
        self.min_segment_duration = min_segment_duration
        self.max_segment_duration = max_segment_duration
        self.distance_metric = distance_metric
        self.smoothing_window = smoothing_window
        self.peak_threshold = peak_threshold 
        self.phonetic_dict = phonetic_dict or PhoneticDictionary()
    
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
    
    def detect_peaks(self, enhanced_distances, n_phonemes=None, frameshift=0.01):
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
        
        # Adaptive height threshold
        height = self.peak_threshold * np.max(enhanced_distances) if np.max(enhanced_distances) > 0 else 0.1
        
        # Find all peaks
        peaks, _ = find_peaks(enhanced_distances, height=height, distance=min_dist_frames)
        
        # If number of phonemes is known, select the n_phonemes-1 highest peaks
        if n_phonemes is not None and n_phonemes > 1:
            n_boundaries = n_phonemes - 1
            
            # If we found too many peaks, select the strongest ones
            if len(peaks) > n_boundaries:
                peak_heights = enhanced_distances[peaks]
                strongest_indices = np.argsort(peak_heights)[-n_boundaries:]
                peaks = peaks[strongest_indices]
                peaks = np.sort(peaks)  # Sort in ascending order
            
            # If we found too few peaks, lower the threshold and try again
            elif len(peaks) < n_boundaries:
                attempts = 0
                while len(peaks) < n_boundaries and attempts < 3:
                    height = height * 0.7  # Lower threshold
                    peaks, _ = find_peaks(enhanced_distances, height=height, distance=min_dist_frames)
                    attempts += 1
                
                # If still too many, select strongest ones
                if len(peaks) > n_boundaries:
                    peak_heights = enhanced_distances[peaks]
                    strongest_indices = np.argsort(peak_heights)[-n_boundaries:]
                    peaks = peaks[strongest_indices]
                    peaks = np.sort(peaks)  # Sort in ascending order
        
        # Convert to array and add boundaries at the beginning and end
        boundaries = np.array([0] + list(peaks + 1) + [len(enhanced_distances) + 1])
        
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
                         frameshift=0.01):
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
            self.debug(f"Estimated {n_phonemes} phonemes for '{word}'")
        
        # Calculate frame-to-frame distances
        distances = self.compute_frame_distances(spectrogram)
        
        # Calculate additional features
        flux = self.compute_spectral_flux(spectrogram)
        energy = self.compute_energy_contour(spectrogram)
        
        # Enhance transitions
        enhanced_distances = self.enhance_transitions(distances, flux, energy)
        
        # Detect peaks
        boundaries = self.detect_peaks(enhanced_distances, n_phonemes, frameshift)
        
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
    
    def process_batch(self, batch, apply_segmentation=True, detect_phonemes=True, frameshift=0.01):
        """
        Process a batch of data from get_data_batch to extract phoneme-level features.
        
        Parameters:
        -----------
        batch : dict
            Output from get_data_batch function
        apply_segmentation : bool
            Whether to apply phoneme segmentation
        detect_phonemes : bool
            Whether to try detecting phonemes (requires words to be provided)
        frameshift : float
            Time between consecutive frames in seconds
            
        Returns:
        --------
        dict
            Enhanced batch with phoneme-level segmentation
        """
        self.debug(f"Processing batch with {len(batch.get('words', []))} instances")
        self.debug(f"Original batch keys: {list(batch.keys())}")
        self.debug(f"EEG segments in batch: {'eeg_segments' in batch}")
        self.debug(f"Spectrogram segments in batch: {'spectrogram_segments' in batch}")
        
        
        # Initialize structures for phoneme-level data
        phoneme_eeg_segments = []
        phoneme_spectrogram_segments = []
        phoneme_labels = []
        phoneme_words = []  # Original words these phonemes come from
        phoneme_positions = []  # Position within word
        phoneme_participant_ids = []
        word_boundaries = []  # Store phoneme boundaries for each word
        
        # Process each instance in the batch
        for i, word in enumerate(batch.get('words', [])):
            if i % 10 == 0:
                self.debug(f"Processing instance {i}/{len(batch.get('words', []))}: {word}")
            
            # Get data for this instance
            eeg_segment = batch['eeg_segments'][i] if 'eeg_segments' in batch else None
            
            if eeg_segment is not None:
                phoneme_eeg_segments.append(eeg_segment)
                self.debug(f"Added EEG segment with shape {eeg_segment.shape}")
            else:
                self.debug("No EEG segment available for this phoneme")

            
            spectrogram_segment = batch['spectrogram_segments'][i] if 'spectrogram_segments' in batch else None
            participant_id = batch['participant_ids'][i] if 'participant_ids' in batch else None
            
            # Skip if required data is missing
            if spectrogram_segment is None:
                continue
            
            # Apply phoneme segmentation if requested
            if apply_segmentation and detect_phonemes and word is not None:
                # Detect boundaries
                result = self.detect_boundaries(
                    spectrogram=spectrogram_segment,
                    word=word,
                    frameshift=frameshift
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
                                    start = boundaries[j]
                                    end = boundaries[j + 1]
                                    
                                    # Ensure valid indices
                                    start = max(0, start)
                                    end = min(eeg_segment.shape[0], end)
                                    
                                    if start < end:
                                        phoneme_eeg_segments.append(eeg_segment[start:end])
                                    else:
                                        # Use empty segment as placeholder
                                        phoneme_eeg_segments.append(np.array([]))
                                else:
                                    phoneme_eeg_segments.append(np.array([]))
                    else:
                        # Mismatch - use a simple distribution based on segment count
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
            'phoneme_detection_applied': detect_phonemes
        }
        
        self.debug(f"Enhanced batch contains {enhanced_batch['metadata']['phoneme_count']} phoneme segments")
        self.debug(f"Found {enhanced_batch['metadata']['unique_phonemes']} unique phonemes")
        
        return enhanced_batch
    
    def accumulate_phoneme_data(self, num_batches=10, batch_size=32,
                                feature_extraction_method='high_gamma', 
                                batch_type='train'):
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
            
        Returns:
        --------
        dict
            Accumulated training data
        """
        # Initialize accumulated data structures
        accumulated_features = []
        accumulated_spectrograms = []
        accumulated_labels = []
        accumulated_words = []
        accumulated_participant_ids = []
        
        self.log(f"Accumulating {batch_type} data from {num_batches} batches...")
        
        # Process multiple batches
        for i in range(num_batches):
            self.log(f"Processing batch {i+1}/{num_batches}")
            
            # Get a new batch of training data
            batch = self.decoder.get_data_batch(
                split_result=self.split_result,
                batch_type=batch_type,
                balanced_sampling=True,
                batch_size=batch_size,
                random_seed=None  # Use different random seed each time
            )
            
            # Process batch to get phoneme-level data
            phoneme_batch = self.process_batch(batch)
            
            # Prepare features for model training
            phoneme_data = self.prepare_phoneme_training_data(
                phoneme_batch,
                feature_extraction_method=feature_extraction_method
            )
            
            # Accumulate data
            accumulated_features.extend(phoneme_data['features'])
            if 'spectrograms' in phoneme_data and phoneme_data['spectrograms']:
                accumulated_spectrograms.extend(phoneme_data['spectrograms'])
            accumulated_labels.extend(phoneme_data['phoneme_labels'])
            accumulated_words.extend(phoneme_data['phoneme_words'])
            accumulated_participant_ids.extend(phoneme_data['phoneme_participant_ids'])
            
            self.log(f"Accumulated {len(accumulated_features)} phoneme segments so far")
        
        # Create result dictionary
        accumulated_data = {
            'features': accumulated_features,
            'spectrograms': accumulated_spectrograms if accumulated_spectrograms else None,
            'phoneme_labels': accumulated_labels,
            'phoneme_words': accumulated_words,
            'phoneme_participant_ids': accumulated_participant_ids,
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
            
        Returns:
        --------
        dict
            Dictionary containing processed data ready for phoneme-based model training
        """

        # Use decoder's config as base, override with kwargs
        if self.decoder is not None:
            config = self.decoder.config.copy()
            config.update(kwargs)
        else:
            # Fallback if no decoder is provided
            config = {
                'feature_extraction_method': 'high_gamma',
                'standardize': True,
                'pca_components': 50
            }
            config.update(kwargs)
            
        # Extract parameters from config
        feature_extraction_method = config['feature_extraction_method']
        standardize = config['standardize']
        pca_components = config['pca_components']
        
        self.debug(f"Preparing phoneme training data with PCA components={pca_components}")
    
        
        try:
            from extract_features import extractHG
            self.debug("Successfully imported extractHG function")
        except ImportError as e:
            self.log(f"Error importing extractHG: {e}")
            raise
        
        # Check if we have EEG segments
        if 'phoneme_eeg_segments' not in enhanced_batch or not enhanced_batch['phoneme_eeg_segments']:
            raise ValueError("No EEG segments found in enhanced batch")
        
        # Filter out empty segments
        valid_indices = []
        for i, eeg in enumerate(enhanced_batch['phoneme_eeg_segments']):
            if eeg is not None and isinstance(eeg, np.ndarray) and eeg.size > 0:
                valid_indices.append(i)
        
        self.debug(f"Found {len(valid_indices)} valid phoneme segments out of {len(enhanced_batch['phoneme_eeg_segments'])}")
        
        # Extract features from valid segments
        features = []
        spectrograms = []
        phoneme_labels = []
        phoneme_words = []
        phoneme_positions = []
        phoneme_participant_ids = []
        
        # Get references to original data
        eeg_segments = [enhanced_batch['phoneme_eeg_segments'][i] for i in valid_indices]
        spectrogram_segments = [enhanced_batch['phoneme_spectrogram_segments'][i] for i in valid_indices]
        labels = [enhanced_batch['phoneme_labels'][i] for i in valid_indices]
        words = [enhanced_batch['phoneme_words'][i] for i in valid_indices]
        positions = [enhanced_batch['phoneme_positions'][i] for i in valid_indices]
        participant_ids = [enhanced_batch['phoneme_participant_ids'][i] for i in valid_indices]
        
        # Process each valid segment
        for i, eeg in enumerate(eeg_segments):
            self.debug(f"Processing segment {i} with shape {eeg.shape}")
            try:
                # Extract features using the decoder
                if self.decoder is not None:
                
                    if feature_extraction_method == 'high_gamma':
                        # Use the directly imported extractHG function
                        feat = extractHG(eeg, 1024)  # Using imported function

                    elif feature_extraction_method == 'multi_band' and self.decoder is not None:
                        feat = self.decoder.custom_feature_extraction(eeg, 1024, method='multi_band')
                    
                    else:
                        raise ValueError(f"Unknown feature extraction method: {feature_extraction_method}")
                        
                else:
                    raise ValueError("No decoder available for feature extraction")
                
                # Add extracted features
                features.append(feat)
                spectrograms.append(spectrogram_segments[i])
                phoneme_labels.append(labels[i])
                phoneme_words.append(words[i])
                phoneme_positions.append(positions[i])
                phoneme_participant_ids.append(participant_ids[i])
            except Exception as e:
                self.log(f"Error processing segment {i}: {e}")
                
        self.debug(f"Processed {len(features)} segments successfully")
        
        # Standardize features if requested
        if standardize and features:
            scaler = StandardScaler()
            for i in range(len(features)):
                # Standardize each feature set independently
                features[i] = scaler.fit_transform(features[i])
        
        # Apply PCA if requested
        if pca_components is not None and pca_components > 0 and features:
            pca = PCA(n_components=pca_components)
            for i in range(len(features)):
                # Apply PCA if we have enough samples
                if features[i].shape[0] > pca_components and features[i].shape[1] > pca_components:
                    features[i] = pca.fit_transform(features[i])
        
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
                'standardized': standardize,
                'pca_components': pca_components,
                'n_phonemes': len(phoneme_labels),
                'unique_phonemes': len(set(phoneme_labels))
            }
        }
        
        self.debug(f"Prepared data for {result['metadata']['n_phonemes']} phoneme segments")
        self.debug(f"Found {result['metadata']['unique_phonemes']} unique phonemes")
        
        
        # Use the decoder's PCA model if available
        if self.decoder is not None and pca_components is not None:
            pca_model = self.decoder.get_pca_model(feature_extraction_method, 'phoneme')
            if pca_model is not None:
                # Reuse existing PCA model
                self.debug("Using existing PCA model from decoder")
                # Apply the PCA model
                # ...
            else:
                # Create new PCA model and store it in decoder
                self.debug("Creating new PCA model")
                # Fit new PCA model
                # ...
                # Store in decoder
                key = f"{feature_extraction_method}_phoneme"
                self.decoder.pca_models[key] = pca
        
        return result