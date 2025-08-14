import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
from scipy.interpolate import interp1d
import os
from collections import Counter, defaultdict
from debugger import DebugMixin

class PhonemeValidator(DebugMixin):
    """
    Tools for validating and visualizing phoneme segmentation quality.
    Works with the AcousticChangeDetector to evaluate phoneme boundary detection.
    """
    
    def __init__(self, detector, debug_mode=None):
            """
            Initialize the validator with a reference to the AcousticChangeDetector.
            
            Parameters:
            -----------
            detector : AcousticChangeDetector
                The acoustic change detector to use for accessing phonetic dictionary
                and other shared resources
            debug_mode : bool or None
                Whether to enable debug mode
            """
            # Initialize the DebugMixin
            super().__init__(class_name="PhonemeValidator", debug_mode=False)
            
            if debug_mode is not None:
                self.DEBUG_MODE = debug_mode
            self.log(f"Initialized with DEBUG_MODE={self.DEBUG_MODE}")
            

            self.detector = detector         
            # Access the phonetic dictionary from the detector
            self.phonetic_dict = detector.phonetic_dict
            
            # Dutch phoneme map
            self.dutch_phoneme_map = {
                # Basic vowels
                'a': 'a', 'e': 'ə', 'i': 'i', 'o': 'o', 'u': 'u', 
                # Diphthongs
                'ij': 'ɛi', 'ei': 'ɛi', 'ui': 'œy', 'eu': 'ø', 'oe': 'u', 
                'ou': 'ɑu', 'au': 'ɑu', 'ie': 'i', 'ee': 'e', 'oo': 'o', 'uu': 'y',
                # Consonants
                'b': 'b', 'c': 'k', 'd': 'd', 'f': 'f', 'g': 'x', 'h': 'h', 'j': 'j', 'k': 'k',
                'l': 'l', 'm': 'm', 'n': 'n', 'p': 'p', 'r': 'r', 's': 's', 't': 't', 'v': 'v',
                'w': 'w', 'z': 'z', 'ch': 'x', 'ng': 'ŋ', 'nk': 'ŋk', 'sch': 'sx',
                # Additional special cases
                'aa': 'a:', 'ee': 'e:', 'oo': 'o:', 'uu': 'y:',  # Long vowels
                'aai': 'aj', 'ooi': 'oj', 'oei': 'uj',  # Vowel + j combinations
                'eeuw': 'ew', 'ieuw': 'iw', 'uw': 'yw',  # Vowel + w combinations
            }
            
            # Reverse map for lookups
            self.phoneme_to_letter = {}
            for letter, phoneme in self.dutch_phoneme_map.items():
                if phoneme not in self.phoneme_to_letter:
                    self.phoneme_to_letter[phoneme] = letter
            
    def extract_phoneme_segments(self, all_results):
        """
        Extract phoneme segments from multiple word results.
        
        Parameters:
        -----------
        all_results : dict
            Dictionary of results from AcousticChangeDetector.process_word_segments_dict
            
        Returns:
        --------
        dict
            Dictionary mapping phonemes to their segments across different words
        """
        phoneme_segments = {}
        
        # Process each word
        for word, word_results in all_results.items():
            if word not in self.phonetic_dict:
                continue
                
            transcription = self.phonetic_dict[word]
            # Clean transcription
            cleaned = transcription.replace('ˈ', '').replace('(', '').replace(')', '').replace("'", '')
            
            # Try to extract individual phonemes from the transcription
            # This is a rough approximation
            phonemes = []
            i = 0
            while i < len(cleaned):
                # Check for complex phonemes (digraphs, diphthongs)
                complex_found = False
                for cp in ['ɛi', 'œy', 'ɑu', 'ɵ:', 'ɛ:', 'a:', 'o:', 'e:', 'øk', 'ɔf', 'ts', 'ŋk', 'sx', 'ɔx', 'ɪx']:
                    if i + len(cp) <= len(cleaned) and cleaned[i:i+len(cp)] == cp:
                        phonemes.append(cp)
                        i += len(cp)
                        complex_found = True
                        break
                
                if not complex_found:
                    phonemes.append(cleaned[i])
                    i += 1
            
            # Process each instance of the word
            for result in word_results:
                if 'segments' not in result or not result['segments']:
                    continue
                    
                segments = result['segments']
                
                # Check if number of segments matches number of phonemes
                if len(segments) == len(phonemes):
                    # Associate each segment with its phoneme
                    for i, (phoneme, segment) in enumerate(zip(phonemes, segments)):
                        if phoneme not in phoneme_segments:
                            phoneme_segments[phoneme] = []
                        
                        phoneme_segments[phoneme].append({
                            'segment': segment,
                            'word': word,
                            'position': i,
                            'participant_id': result.get('participant_id', None),
                            'instance_index': result.get('instance_index', None)
                        })
                elif len(segments) > 0:
                    # If mismatch, still try to use the segments by assuming roughly equal distribution
                    n_expected = len(phonemes)
                    n_actual = len(segments)
                    
                    # Distribute segments according to relative phoneme durations
                    # This is a simple heuristic based on typical phoneme durations
                    
                    # Estimate duration weights
                    weights = []
                    for phoneme in phonemes:
                        # Vowels typically longer than consonants
                        if phoneme in 'aeiouəɑɛɪɔʊʌœyɵ':
                            weights.append(1.5)
                        # Diphthongs longer than simple vowels
                        elif phoneme in ['ɛi', 'œy', 'ɑu']:
                            weights.append(2.0)
                        # Fricatives often longer than stops
                        elif phoneme in 'fvszʃʒxh':
                            weights.append(1.2)
                        else:
                            weights.append(1.0)
                    
                    # Normalize weights to sum to n_actual
                    weights = np.array(weights) * (n_actual / sum(weights))
                    
                    # Convert to segment counts (ensuring we use exactly n_actual segments)
                    counts = np.round(weights).astype(int)
                    # Adjust to ensure sum equals n_actual
                    while sum(counts) < n_actual:
                        # Add to the phoneme with largest fractional part
                        fractional = weights - counts
                        counts[np.argmax(fractional)] += 1
                    while sum(counts) > n_actual:
                        # Subtract from the phoneme with smallest fractional part
                        fractional = counts - weights
                        nonzero = np.where(counts > 0)[0]
                        counts[nonzero[np.argmax(fractional[nonzero])]] -= 1
                    
                    # Assign segments to phonemes
                    segment_index = 0
                    for i, (phoneme, count) in enumerate(zip(phonemes, counts)):
                        if phoneme not in phoneme_segments:
                            phoneme_segments[phoneme] = []
                        
                        # For phonemes assigned multiple segments, average them
                        if count > 0:
                            combined_segment = segments[segment_index]
                            for j in range(1, count):
                                if segment_index + j < len(segments):
                                    # Stack additional segments if shapes match
                                    if combined_segment.shape[1] == segments[segment_index + j].shape[1]:
                                        combined_segment = np.vstack((combined_segment, segments[segment_index + j]))
                            
                            phoneme_segments[phoneme].append({
                                'segment': combined_segment,
                                'word': word,
                                'position': i,
                                'participant_id': result.get('participant_id', None),
                                'instance_index': result.get('instance_index', None),
                                'segment_count': count
                            })
                            
                            segment_index += count
        
        return phoneme_segments
    
    def visualize_phoneme_segments(self, phoneme_segments, phoneme=None, max_examples=10, 
                                 normalize=True, save_path=None):
        """
        Visualize segments for a specific phoneme across different words.
        
        Parameters:
        -----------
        phoneme_segments : dict
            Output from extract_phoneme_segments
        phoneme : str or None
            Specific phoneme to visualize. If None, will visualize the phoneme with most examples.
        max_examples : int
            Maximum number of examples to show
        normalize : bool
            Whether to normalize segment durations for comparison
        save_path : str or None
            Path to save the visualization
            
        Returns:
        --------
        fig : matplotlib.figure.Figure
            The created figure
        """
        # If no phoneme specified, use the one with most examples
        if phoneme is None:
            phoneme = max(phoneme_segments.keys(), key=lambda p: len(phoneme_segments[p]))
        
        if phoneme not in phoneme_segments:
            raise ValueError(f"Phoneme '{phoneme}' not found in segments")
        
        segments = phoneme_segments[phoneme]
        
        # Limit number of examples
        if len(segments) > max_examples:
            # Select a diverse set of examples from different words
            words = set(s['word'] for s in segments)
            selected = []
            
            # Try to include at least one example from each word
            for word in words:
                word_segments = [s for s in segments if s['word'] == word]
                if word_segments and len(selected) < max_examples:
                    selected.append(word_segments[0])
            
            selected_indices = set(id(s) for s in selected)
            # Fill remaining slots with random selections
            
            remaining = [s for s in segments if id(s) not in selected_indices]
            if remaining and len(selected) < max_examples:
                selected.extend(np.random.choice(
                    remaining, 
                    min(max_examples - len(selected), len(remaining)), 
                    replace=False
                ).tolist())
            
            segments = selected
        
        n_examples = len(segments)
        
        # Create figure
        fig, axs = plt.subplots(n_examples, 1, figsize=(10, 2 * n_examples))
        if n_examples == 1:
            axs = [axs]
        
        # Normalize time dimension if requested
        if normalize:
            # Find the target length (median of segment lengths)
            target_length = int(np.median([s['segment'].shape[0] for s in segments]))
            
            # Ensure target length is reasonable
            target_length = max(5, min(50, target_length))
        
        # Plot each segment
        for i, segment_info in enumerate(segments):
            segment = segment_info['segment']
            word = segment_info['word']
            position = segment_info['position']
            participant_id = segment_info['participant_id']
            
            # Normalize time dimension if requested
            if normalize and segment.shape[0] != target_length:
                # Simple resampling by linear interpolation
                # Create interpolator for each frequency bin
                time_orig = np.arange(segment.shape[0])
                time_new = np.linspace(0, segment.shape[0] - 1, target_length)
                
                normalized_segment = np.zeros((target_length, segment.shape[1]))
                for j in range(segment.shape[1]):
                    f = interp1d(time_orig, segment[:, j], bounds_error=False, fill_value="extrapolate")
                    normalized_segment[:, j] = f(time_new)
                
                segment = normalized_segment
            
            # Plot the segment
            im = axs[i].imshow(segment.T, aspect='auto', origin='lower', cmap='viridis')
            
            # Add word and position information
            title = f"Phoneme '{phoneme}' in '{word}' (position {position+1})"
            if participant_id:
                title += f" - {participant_id}"
            axs[i].set_title(title)
            
            # Add colorbar
            plt.colorbar(im, ax=axs[i], orientation='horizontal', pad=0.05, aspect=40)
            
            axs[i].set_ylabel('Mel Frequency Bin')
            if i == n_examples - 1:
                axs[i].set_xlabel('Frame' + (' (Normalized)' if normalize else ''))
        
        plt.tight_layout()
        
        # Save figure if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def compare_phoneme_across_words(self, all_results, target_phoneme, 
                                    max_words=5, max_examples_per_word=3,
                                    normalize=True, save_path=None):
        """
        Compare the same phoneme across different words.
        
        Parameters:
        -----------
        all_results : dict
            Dictionary of results from AcousticChangeDetector.process_word_segments_dict
        target_phoneme : str
            Phoneme to compare
        max_words : int
            Maximum number of different words to include
        max_examples_per_word : int
            Maximum number of examples per word
        normalize : bool
            Whether to normalize segment durations for comparison
        save_path : str or None
            Path to save the visualization
            
        Returns:
        --------
        fig : matplotlib.figure.Figure
            The created figure
        """
        # Extract phoneme segments
        phoneme_segments = self.extract_phoneme_segments(all_results)
        
        if target_phoneme not in phoneme_segments:
            raise ValueError(f"Phoneme '{target_phoneme}' not found in segments")
        
        segments = phoneme_segments[target_phoneme]
        
        # Group by word
        word_segments = {}
        for segment in segments:
            word = segment['word']
            if word not in word_segments:
                word_segments[word] = []
            word_segments[word].append(segment)
        
        # Select words to include
        words = list(word_segments.keys())
        if len(words) > max_words:
            words = words[:max_words]
        
        # Create figure
        n_rows = len(words)
        n_cols = max_examples_per_word
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 2 * n_rows))
        
        # Handle case with single row or column
        if n_rows == 1 and n_cols == 1:
            axs = [[axs]]
        elif n_rows == 1:
            axs = [axs]
        elif n_cols == 1:
            axs = [[ax] for ax in axs]
        
        # Normalize time dimension if requested
        target_length = None
        if normalize:
            # Find the target length (median of all segment lengths)
            all_lengths = [s['segment'].shape[0] for word in words 
                          for s in word_segments[word][:max_examples_per_word]]
            target_length = int(np.median(all_lengths))
            
            # Ensure target length is reasonable
            target_length = max(5, min(50, target_length))
        
        # Plot each word's examples
        for i, word in enumerate(words):
            segments = word_segments[word][:max_examples_per_word]
            
            for j in range(n_cols):
                if j < len(segments):
                    segment_info = segments[j]
                    segment = segment_info['segment']
                    position = segment_info['position']
                    
                    # Normalize time dimension if requested
                    if normalize and target_length is not None and segment.shape[0] != target_length:
                        # Simple resampling by linear interpolation
                        time_orig = np.arange(segment.shape[0])
                        time_new = np.linspace(0, segment.shape[0] - 1, target_length)
                        
                        normalized_segment = np.zeros((target_length, segment.shape[1]))
                        for k in range(segment.shape[1]):
                            f = interp1d(time_orig, segment[:, k], bounds_error=False, fill_value="extrapolate")
                            normalized_segment[:, k] = f(time_new)
                        
                        segment = normalized_segment
                    
                    # Plot the segment
                    im = axs[i][j].imshow(segment.T, aspect='auto', origin='lower', cmap='viridis')
                    
                    # Add position information
                    axs[i][j].set_title(f"Position {position+1}")
                    
                    # Only add y-label for first column
                    if j == 0:
                        axs[i][j].set_ylabel(f"'{word}'")
                    
                    # Only add x-label for last row
                    if i == n_rows - 1:
                        axs[i][j].set_xlabel('Frame' + (' (Normalized)' if normalize else ''))
                else:
                    # Hide unused subplots
                    axs[i][j].axis('off')
        
        # Add overall title
        plt.suptitle(f"Comparison of Phoneme '{target_phoneme}' Across Different Words", fontsize=16)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for suptitle
        
        # Save figure if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig

    def visualize_boundaries(self, result, save_path=None):
        """
        Visualize detected boundaries on the spectrogram.
        
        Parameters:
        -----------
        result : dict
            Result dictionary from detect_boundaries
        save_path : str or None
            Path to save the visualization
            
        Returns:
        --------
        fig : matplotlib.figure.Figure
            The created figure
        """
        # Extract data from result
        spectrogram = result['segments'][0] if result['segments'] else None
        for segment in result['segments'][1:]:
            if spectrogram is not None and segment is not None:
                spectrogram = np.vstack((spectrogram, segment))
        
        boundaries = result['boundaries']
        distances = result['distances']
        enhanced_distances = result['enhanced_distances']
        word = result['word']
        energy = result.get('energy', None)
        
        # Create figure
        fig, axs = plt.subplots(3, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 1, 1]})
        
        # Plot spectrogram
        if spectrogram is not None:
            axs[0].imshow(spectrogram.T, aspect='auto', origin='lower', cmap='viridis')
            axs[0].set_title(f'Mel Spectrogram with Phoneme Boundaries - Word: {word}')
            
            # Plot boundaries on spectrogram
            for boundary in boundaries:
                if 0 <= boundary < spectrogram.shape[0]:
                    axs[0].axvline(x=boundary, color='r', linestyle='--', alpha=0.7)
            
            # Add phoneme labels if we have transcription
            if word in self.phonetic_dict:
                transcription = self.phonetic_dict[word]
                # Clean transcription and estimate phoneme count
                cleaned = transcription.replace('ˈ', '').replace('(', '').replace(')', '').replace("'", '')
                phoneme_count = self.detector.count_phonemes(word)
                
                if len(result['segments']) == phoneme_count:
                    # Add phoneme labels at the center of each segment
                    segment_starts = boundaries[:-1]
                    segment_ends = boundaries[1:]
                    
                    # Try to extract individual phonemes from the transcription
                    # This is a very rough approximation
                    phonemes = []
                    i = 0
                    while i < len(cleaned):
                        # Check for complex phonemes (digraphs, diphthongs)
                        complex_found = False
                        for cp in ['ɛi', 'œy', 'ɑu', 'ɵ:', 'ɛ:', 'a:', 'o:', 'e:', 'øk', 'ɔf', 'ts', 'ŋk', 'sx', 'ɔx', 'ɪx']:
                            if i + len(cp) <= len(cleaned) and cleaned[i:i+len(cp)] == cp:
                                phonemes.append(cp)
                                i += len(cp)
                                complex_found = True
                                break
                        
                        if not complex_found:
                            phonemes.append(cleaned[i])
                            i += 1
                    
                    # Ensure we have the correct number of phonemes
                    if len(phonemes) > phoneme_count:
                        phonemes = phonemes[:phoneme_count]
                    elif len(phonemes) < phoneme_count:
                        # Pad with question marks if we couldn't identify enough phonemes
                        phonemes.extend(['?'] * (phoneme_count - len(phonemes)))
                    
                    # Add phoneme labels
                    for i in range(len(result['segments'])):
                        segment_center = (segment_starts[i] + segment_ends[i]) / 2
                        if i < len(phonemes):
                            axs[0].text(segment_center, spectrogram.shape[1] * 0.9, phonemes[i], 
                                      horizontalalignment='center', verticalalignment='center',
                                      color='white', fontsize=12, fontweight='bold',
                                      bbox=dict(facecolor='black', alpha=0.6))
            
            axs[0].set_ylabel('Mel Frequency Bin')
            axs[0].set_xlabel('Frame')
        
        # Plot distances
        if distances is not None and len(distances) > 0:
            axs[1].plot(np.arange(len(distances)), distances, label='Frame Distances')
            axs[1].plot(np.arange(len(enhanced_distances)), enhanced_distances, 
                      'g-', label='Enhanced Distances')
            
            # Plot boundaries on distances
            for boundary in boundaries:
                if boundary > 0 and boundary <= len(distances):
                    axs[1].axvline(x=boundary-1, color='r', linestyle='--', alpha=0.7)
            
            axs[1].set_title('Frame-to-Frame Distances')
            axs[1].set_xlabel('Frame')
            axs[1].set_ylabel('Distance')
            axs[1].legend()
        
        # Plot energy contour
        if energy is not None:
            axs[2].plot(np.arange(len(energy)), energy, 'b-', label='Energy Contour')
            
            # Plot boundaries on energy
            for boundary in boundaries:
                if boundary < len(energy):
                    axs[2].axvline(x=boundary, color='r', linestyle='--', alpha=0.7)
            
            axs[2].set_title('Energy Contour')
            axs[2].set_xlabel('Frame')
            axs[2].set_ylabel('Energy')
            axs[2].legend()
        
        plt.tight_layout()
        
        # Save figure if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
        
    def analyze_specific_word(self, word, all_results, normalize=True, save_path=None):
        """
        Analyze all instances of a specific word to compare segmentation consistency.
        
        Parameters:
        -----------
        word : str
            The word to analyze
        all_results : dict
            Dictionary of results from process_word_segments_dict
        normalize : bool
            Whether to normalize segment durations for comparison
        save_path : str or None
            Path to save the visualization
            
        Returns:
        --------
        fig : matplotlib.figure.Figure
            The created figure
        """
        # Filter results for the target word
        word_instances = []
        
        for participant_id, participant_results in all_results.items():
            if word in participant_results:
                word_instances.extend(participant_results[word])
        
        if not word_instances:
            raise ValueError(f"No instances found for word '{word}'")
        
        # Limit to a reasonable number of instances
        max_instances = 10
        if len(word_instances) > max_instances:
            word_instances = np.random.choice(word_instances, max_instances, replace=False)
        
        n_instances = len(word_instances)
        
        # Create figure
        fig_height = 3 * n_instances
        fig, axs = plt.subplots(n_instances, 1, figsize=(12, fig_height))
        
        # Ensure axs is always a list
        if n_instances == 1:
            axs = [axs]
        
        # Get phonetic transcription
        if word in self.phonetic_dict:
            transcription = self.phonetic_dict[word]
            # Extract phonemes
            cleaned = transcription.replace('ˈ', '').replace('(', '').replace(')', '').replace("'", '')
            
            # Try to extract individual phonemes from the transcription
            phonemes = []
            i = 0
            while i < len(cleaned):
                # Check for complex phonemes (digraphs, diphthongs)
                complex_found = False
                for cp in ['ɛi', 'œy', 'ɑu', 'ɵ:', 'ɛ:', 'a:', 'o:', 'e:', 'øk', 'ɔf', 'ts', 'ŋk', 'sx', 'ɔx', 'ɪx']:
                    if i + len(cp) <= len(cleaned) and cleaned[i:i+len(cp)] == cp:
                        phonemes.append(cp)
                        i += len(cp)
                        complex_found = True
                        break
                
                if not complex_found:
                    phonemes.append(cleaned[i])
                    i += 1
        else:
            phonemes = None
        
        # Plot each instance
        for i, instance in enumerate(word_instances):
            # Get participant info
            participant_id = instance.get('participant_id', 'unknown')
            
            # Combine segments to reconstruct the full spectrogram
            if 'segments' in instance and instance['segments']:
                spectrogram = instance['segments'][0]
                for segment in instance['segments'][1:]:
                    spectrogram = np.vstack((spectrogram, segment))
            else:
                continue
            
            # Plot spectrogram
            im = axs[i].imshow(spectrogram.T, aspect='auto', origin='lower', cmap='viridis')
            
            # Plot boundaries
            boundaries = instance.get('boundaries', [])
            for boundary in boundaries:
                if 0 <= boundary < spectrogram.shape[0]:
                    axs[i].axvline(x=boundary, color='r', linestyle='--', alpha=0.7)
            
            # Add phoneme labels if available
            if phonemes is not None and len(instance['segments']) == len(phonemes):
                segment_starts = boundaries[:-1]
                segment_ends = boundaries[1:]
                
                for j in range(len(instance['segments'])):
                    segment_center = (segment_starts[j] + segment_ends[j]) / 2
                    if j < len(phonemes):
                        axs[i].text(segment_center, spectrogram.shape[1] * 0.9, phonemes[j], 
                                  horizontalalignment='center', verticalalignment='center',
                                  color='white', fontsize=12, fontweight='bold',
                                  bbox=dict(facecolor='black', alpha=0.6))
            
            # Add colorbar
            plt.colorbar(im, ax=axs[i], orientation='horizontal', pad=0.05, aspect=40)
            
            # Add title
            axs[i].set_title(f"Word: '{word}' - Participant: {participant_id}")
            
            # Add labels
            axs[i].set_ylabel('Mel Frequency Bin')
            if i == n_instances - 1:
                axs[i].set_xlabel('Frame')
        
        plt.tight_layout()
        
        # Save figure if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
        
    def analyze_phoneme_inventory(self, all_results, min_occurrences=3, save_path=None):
        """
        Analyze the complete phoneme inventory and their detection statistics.
        
        Parameters:
        -----------
        all_results : dict
            Dictionary of results from process_word_segments_dict
        min_occurrences : int
            Minimum number of occurrences required for a phoneme to be included
        save_path : str or None
            Path to save the visualization
            
        Returns:
        --------
        tuple
            (fig, phoneme_inventory) containing the figure and phoneme inventory dict
        """
        # Extract phoneme segments
        phoneme_segments = self.extract_phoneme_segments(all_results)
        
        # Collect statistics for each phoneme
        phoneme_inventory = {}
        
        for phoneme, segments in phoneme_segments.items():
            if len(segments) < min_occurrences:
                continue
            
            # Calculate metrics
            durations = [s['segment'].shape[0] for s in segments]
            words = [s['word'] for s in segments]
            positions = [s['position'] for s in segments]
            
            # Store statistics
            phoneme_inventory[phoneme] = {
                'count': len(segments),
                'unique_words': len(set(words)),
                'mean_duration': np.mean(durations),
                'std_duration': np.std(durations),
                'min_duration': np.min(durations),
                'max_duration': np.max(durations),
                'positions': sorted(set(positions)),
                'words': sorted(set(words))
            }
        
        # Categorize phonemes
        vowels = [p for p in phoneme_inventory if p in 'aeiouəɑɛɪɔʊʌœyɵ' or p in ['ɛi', 'œy', 'ɑu']]
        consonants = [p for p in phoneme_inventory if p not in vowels]
        
        # Create figure
        fig, axs = plt.subplots(2, 1, figsize=(14, 10))
        
        # Sort by count
        vowels_sorted = sorted(vowels, key=lambda p: phoneme_inventory[p]['count'], reverse=True)
        consonants_sorted = sorted(consonants, key=lambda p: phoneme_inventory[p]['count'], reverse=True)
        
        # Plot vowels
        vowel_counts = [phoneme_inventory[p]['count'] for p in vowels_sorted]
        vowel_durations = [phoneme_inventory[p]['mean_duration'] for p in vowels_sorted]
        vowel_colors = plt.cm.viridis(np.linspace(0, 1, len(vowels_sorted)))
        
        axs[0].bar(vowels_sorted, vowel_counts, color=vowel_colors)
        axs[0].set_title('Vowel Phoneme Inventory')
        axs[0].set_ylabel('Occurrence Count')
        
        # Add duration as text
        for i, p in enumerate(vowels_sorted):
            axs[0].text(i, vowel_counts[i] + 1, f"{vowel_durations[i]:.1f} frames", 
                      ha='center', va='bottom', fontsize=8, rotation=45)
        
        # Plot consonants
        consonant_counts = [phoneme_inventory[p]['count'] for p in consonants_sorted]
        consonant_durations = [phoneme_inventory[p]['mean_duration'] for p in consonants_sorted]
        consonant_colors = plt.cm.plasma(np.linspace(0, 1, len(consonants_sorted)))
        
        axs[1].bar(consonants_sorted, consonant_counts, color=consonant_colors)
        axs[1].set_title('Consonant Phoneme Inventory')
        axs[1].set_ylabel('Occurrence Count')
        
        # Add duration as text
        for i, p in enumerate(consonants_sorted):
            axs[1].text(i, consonant_counts[i] + 1, f"{consonant_durations[i]:.1f} frames", 
                      ha='center', va='bottom', fontsize=8, rotation=45)
        
        plt.tight_layout()
        
        # Save figure if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig, phoneme_inventory        
       
    def validate_phoneme_consistency(self, all_results, min_occurrences=3):
        """
        Validate the consistency of phoneme segments across different words.
        
        Parameters:
        -----------
        all_results : dict
            Dictionary of results from process_word_segments_dict
        min_occurrences : int
            Minimum number of occurrences required for a phoneme to be analyzed
            
        Returns:
        --------
        dict
            Dictionary containing validation metrics for each phoneme
        """
        # Extract phoneme segments
        phoneme_segments = self.extract_phoneme_segments(all_results)
        
        # Initialize results
        validation_results = {}
        
        # For each phoneme with sufficient occurrences
        for phoneme, segments in phoneme_segments.items():
            if len(segments) < min_occurrences:
                continue
            
            # Calculate consistency metrics
            
            # 1. Duration consistency
            durations = [s['segment'].shape[0] for s in segments]
            duration_mean = np.mean(durations)
            duration_std = np.std(durations)
            duration_cv = duration_std / duration_mean if duration_mean > 0 else float('inf')
            
            # 2. Spectral similarity
            # Normalize all segments to the same length for comparison
            target_length = int(np.median(durations))
            
            normalized_segments = []
            for segment_info in segments:
                segment = segment_info['segment']
                
                if segment.shape[0] != target_length:
                    # Resample to target length
                    from scipy.interpolate import interp1d
                    
                    time_orig = np.arange(segment.shape[0])
                    time_new = np.linspace(0, segment.shape[0] - 1, target_length)
                    
                    normalized_segment = np.zeros((target_length, segment.shape[1]))
                    for j in range(segment.shape[1]):
                        f = interp1d(time_orig, segment[:, j], bounds_error=False, fill_value="extrapolate")
                        normalized_segment[:, j] = f(time_new)
                    
                    normalized_segments.append(normalized_segment)
                else:
                    normalized_segments.append(segment)
            
            # Calculate average segment (centroid)
            if normalized_segments:
                centroid = np.mean(normalized_segments, axis=0)
                
                # Calculate distance from each segment to centroid
                distances = []
                for segment in normalized_segments:
                    # Flatten segments for distance calculation
                    flat_segment = segment.flatten()
                    flat_centroid = centroid.flatten()
                    
                    # Use cosine distance
                    distance = cosine(flat_segment, flat_centroid)
                    distances.append(distance)
                
                spectral_similarity = 1 - np.mean(distances)  # Higher is more similar
            else:
                spectral_similarity = 0
            
            # 3. Word context analysis
            words = [s['word'] for s in segments]
            positions = [s['position'] for s in segments]
            
            # Check positional consistency
            position_counts = {}
            for position in positions:
                if position not in position_counts:
                    position_counts[position] = 0
                position_counts[position] += 1
            
            most_common_position = max(position_counts.items(), key=lambda x: x[1])[0]
            position_consistency = position_counts[most_common_position] / len(positions)
            
            # Store results
            validation_results[phoneme] = {
                'count': len(segments),
                'duration_mean': duration_mean,
                'duration_std': duration_std,
                'duration_cv': duration_cv,  # Coefficient of variation (lower is more consistent)
                'spectral_similarity': spectral_similarity,
                'position_consistency': position_consistency,
                'common_words': sorted(set(words)),
                'most_common_position': most_common_position
            }
        
        return validation_results    
        
    def extract_phoneme_segments_from_batch(self, phoneme_batch):
        """
        Extract phoneme segments from a processed batch with phoneme-level data.
        
        Parameters:
        -----------
        phoneme_batch : dict
            The output of process_batch with phoneme-level data
            
        Returns:
        --------
        dict
            Dictionary mapping phonemes to their segments
        """
        self.debug("Extracting phoneme segments from processed batch")
        
        # Initialize results dictionary
        phoneme_segments = {}
        
        # Check if we have all the necessary data
        required_keys = ['phoneme_labels', 'phoneme_spectrogram_segments', 'phoneme_words']
        if not all(key in phoneme_batch for key in required_keys):
            missing = [key for key in required_keys if key not in phoneme_batch]
            self.debug(f"Missing required keys: {missing}")
            return {}
        
        # Get data from the batch
        labels = phoneme_batch['phoneme_labels']
        spectrogram_segments = phoneme_batch['phoneme_spectrogram_segments']
        words = phoneme_batch['phoneme_words']
        positions = phoneme_batch.get('phoneme_positions', [0] * len(labels))
        participant_ids = phoneme_batch.get('phoneme_participant_ids', ['unknown'] * len(labels))
        
        self.debug(f"Processing {len(labels)} phoneme segments")
        
        # Track statistics
        total_phonemes = 0
        known_phonemes = 0
        
        # Process each phoneme segment
        for i, (label, segment, word, position, participant_id) in enumerate(
            zip(labels, spectrogram_segments, words, positions, participant_ids)
        ):
            # Skip segments with unknown phonemes
            if label == '?':
                continue
                
            total_phonemes += 1
            
            # Add this segment to the appropriate phoneme entry
            if label not in phoneme_segments:
                phoneme_segments[label] = []
            
            # Create the segment information
            segment_info = {
                'segment': segment,
                'word': word,
                'position': position,
                'participant_id': participant_id
            }
            
            # Add to the segments for this phoneme
            phoneme_segments[label].append(segment_info)
            known_phonemes += 1
        
        # Log statistics
        self.debug(f"Processed {total_phonemes} phoneme segments, found {known_phonemes} valid phoneme segments")
        self.debug(f"Extracted {len(phoneme_segments)} unique phonemes")
        
        if phoneme_segments:
            # List phonemes with their counts
            phoneme_counts = {p: len(s) for p, s in phoneme_segments.items()}
            sorted_phonemes = sorted(phoneme_counts.items(), key=lambda x: x[1], reverse=True)
            
            self.debug("Phoneme counts:")
            for phoneme, count in sorted_phonemes[:10]:  # Show top 10
                self.debug(f"  {phoneme}: {count} segments")
        
        return phoneme_segments
        
        
    def resolve_unknown_phonemes(self, phoneme_batch):
        """
        Attempt to resolve unknown phonemes ('?') based on word and position.
        
        Parameters:
        -----------
        phoneme_batch : dict
            The phoneme batch with unknown phonemes to resolve
            
        Returns:
        --------
        dict
            Copy of the phoneme batch with resolved phonemes
        """
        import copy
        
        # Create a deep copy to avoid modifying the original
        resolved_batch = copy.deepcopy(phoneme_batch)
        
        # Get relevant data
        labels = resolved_batch['phoneme_labels']
        words = resolved_batch['phoneme_words']
        positions = resolved_batch['phoneme_positions']
        
        # Track statistics
        unknown_count = labels.count('?')
        resolved_count = 0
        
        self.debug(f"Attempting to resolve {unknown_count} unknown phonemes")
        
        # Process each phoneme
        for i, (label, word, position) in enumerate(zip(labels, words, positions)):
            if label != '?':
                continue
                
            # Try to resolve based on word and position
            resolved = self._infer_phoneme_from_word(word, position)
            
            if resolved:
                labels[i] = resolved
                resolved_count += 1
                self.debug(f"Resolved phoneme at position {position} in '{word}' as '{resolved}'")
        
        self.debug(f"Resolved {resolved_count} out of {unknown_count} unknown phonemes")
        
        return resolved_batch

    def _infer_phoneme_from_word(self, word, position):
        """
        Infer a phoneme based on word and position.
        
        Parameters:
        -----------
        word : str
            The word containing the phoneme
        position : int
            The position of the phoneme in the word
            
        Returns:
        --------
        str or None
            The inferred phoneme, or None if it couldn't be inferred
        """
        # 1. Check if we have a transcription for this word
        if word in self.phonetic_dict:
            transcription = self.phonetic_dict[word]
            cleaned = transcription.replace('ˈ', '').replace('(', '').replace(')', '').replace("'", '')
            
            # Extract phonemes
            phonemes = []
            i = 0
            while i < len(cleaned):
                # Check for complex phonemes
                complex_found = False
                for cp in ['ɛi', 'œy', 'ɑu', 'ɵ:', 'ɛ:', 'a:', 'o:', 'e:', 'øk', 'ɔf', 'ts', 'ŋk', 'sx', 'ɔx', 'ɪx']:
                    if i + len(cp) <= len(cleaned) and cleaned[i:i+len(cp)] == cp:
                        phonemes.append(cp)
                        i += len(cp)
                        complex_found = True
                        break
                
                if not complex_found:
                    phonemes.append(cleaned[i])
                    i += 1
            
            # Check if position is valid
            if 0 <= position < len(phonemes):
                return phonemes[position]
        
        # 2. If no transcription, try to infer based on Dutch spelling rules
        # This is a simplistic approach and won't be perfect
        if 0 <= position < len(word):
            # Get the character at this position
            char = word[position].lower()
            
            # Check if it's in our phoneme map
            if char in self.dutch_phoneme_map:
                return self.dutch_phoneme_map[char]
            
            # Check for digraphs/trigraphs
            if position < len(word) - 1:
                digraph = word[position:position+2].lower()
                if digraph in self.dutch_phoneme_map:
                    return self.dutch_phoneme_map[digraph]
                    
            if position < len(word) - 2:
                trigraph = word[position:position+3].lower()
                if trigraph in self.dutch_phoneme_map:
                    return self.dutch_phoneme_map[trigraph]
        
        # 3. Fallback: guess based on common Dutch phonemes
        # For now, return None
        return None
        
    def visualize_phoneme_spectrograms(self, phoneme_data, target_phonemes=None, max_examples=5, 
                                  participant_filter=None, save_dir=None, show_stats=True):
        """
        Visualize mel spectrograms for specific phonemes before model processing.
        
        Parameters:
        -----------
        phoneme_data : dict
            Dictionary containing phoneme data with keys:
            - 'phoneme_labels': List of phoneme labels
            - 'spectrograms': List of spectrogram arrays for each phoneme
            - 'phoneme_words': List of words each phoneme comes from
            - 'phoneme_participant_ids': List of participant IDs for each phoneme
        target_phonemes : list or None
            List of specific phonemes to visualize. If None, selects most frequent phonemes.
        max_examples : int
            Maximum number of examples to show per phoneme
        participant_filter : str or None
            If provided, only show examples from this participant
        save_dir : str or None
            Directory to save visualizations. If None, uses default.
        show_stats : bool
            Whether to show statistics about each phoneme's occurrence
        
        Returns:
        --------
        dict
            Dictionary mapping phonemes to their visualization paths
        """

        
        def sanitize_filename(phoneme):
            """Convert phonetic symbols to safe filenames"""
            # Map of phonetic symbols to safe equivalents
            replacements = {
                 'ɛ': 'e_open',
                'œy': 'oe_y',
                'ɑ': 'a_back',
                'ə': 'schwa',
                'ŋ': 'ng',
                'ɪ': 'i_small',
                'ʃ': 'sh',
                'ʒ': 'zh',
                'ɔ': 'o_open',
                'ʊ': 'u_small',
                'ʌ': 'v_turned',
                'ɵ': 'o_barred',
                'ː': '_long',  # Handle length marker
                'ɣ': 'gamma',
                'χ': 'chi',
                'ʁ': 'inverted_r',
                'ɦ': 'h_voiced',
                'ʔ': 'glottal_stop',
                'ʋ': 'v_labiodental'
            }
            
            # Replace known special characters
            safe_name = phoneme
            for char, replacement in replacements.items():
                if char in safe_name:
                    safe_name = safe_name.replace(char, replacement)
            
            # Remove any remaining special characters
            safe_name = ''.join(c if c.isalnum() or c in '_-' else '_' for c in safe_name)
            
            return safe_name
        
        # Create save directory if needed
        if save_dir is None:
            save_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results', 'phoneme_visualizations')
        os.makedirs(save_dir, exist_ok=True)
        
        # Extract data
        phoneme_labels = phoneme_data['phoneme_labels']
        spectrograms = phoneme_data['spectrograms']
        words = phoneme_data.get('phoneme_words', ['unknown'] * len(phoneme_labels))
        participants = phoneme_data.get('phoneme_participant_ids', ['unknown'] * len(phoneme_labels))
        positions = phoneme_data.get('phoneme_positions', [0] * len(phoneme_labels))
        
        # Count phoneme occurrences
        phoneme_counter = Counter(phoneme_labels)
        self.log(f"Found {len(phoneme_counter)} unique phonemes in the data")
        
        # If no target phonemes specified, use the most frequent ones
        if target_phonemes is None:
            target_phonemes = [p for p, _ in phoneme_counter.most_common(20)]
        
        # Group examples by phoneme
        phoneme_examples = defaultdict(list)
        for i, (phoneme, spec, word, participant, position) in enumerate(
            zip(phoneme_labels, spectrograms, words, participants, positions)):
            
            # Apply participant filter if specified
            if participant_filter is not None and participant != participant_filter:
                continue
            
            # Only collect examples for target phonemes
            if phoneme in target_phonemes:
                phoneme_examples[phoneme].append({
                    'index': i,
                    'spectrogram': spec,
                    'word': word,
                    'participant': participant,
                    'position': position
                })
        
        # Create visualizations for each phoneme
        visualization_paths = {}
        
        # Track phoneme stats
        phoneme_stats = {}
        
        for phoneme in target_phonemes:
            examples = phoneme_examples[phoneme]
            
            # Skip if no examples found
            if not examples:
                self.log(f"No examples found for phoneme '{phoneme}'")
                continue
            
            # Limit to max examples
            if len(examples) > max_examples:
                # Select diverse examples from different participants if possible
                participant_set = set(ex['participant'] for ex in examples)
                
                if len(participant_set) > 1:
                    # Try to get examples from different participants
                    selected_examples = []
                    for participant in participant_set:
                        participant_examples = [ex for ex in examples if ex['participant'] == participant]
                        if participant_examples:
                            selected_examples.append(participant_examples[0])
                        if len(selected_examples) >= max_examples:
                            break
                    
                    # If we still need more, add randomly
                    if len(selected_examples) < max_examples:
                        remaining = [ex for ex in examples if ex not in selected_examples]
                        selected_examples.extend(remaining[:max_examples-len(selected_examples)])
                    
                    examples = selected_examples[:max_examples]
                else:
                    # Just take the first max_examples
                    examples = examples[:max_examples]
            
            # Calculate statistics
            specs = [ex['spectrogram'] for ex in examples]
            phoneme_stats[phoneme] = {
                'count': phoneme_counter[phoneme],
                'mean_duration': np.mean([spec.shape[0] for spec in specs]),
                'std_duration': np.std([spec.shape[0] for spec in specs]),
                'participants': len(set(ex['participant'] for ex in examples)),
                'words': len(set(ex['word'] for ex in examples))
            }
            
            # Create visualization
            n_rows = len(examples)
            
            # Three column layout: spectrogram, frequency profile, time profile
            fig, axs = plt.subplots(n_rows, 3, figsize=(15, 4 * n_rows), 
                                   gridspec_kw={'width_ratios': [3, 1, 3]})
            
            # For single example case
            if n_rows == 1:
                axs = [axs]
            
            for i, example in enumerate(examples):
                spec = example['spectrogram']
                word = example['word']
                participant = example['participant']
                position = example['position']
                
                # 1. Plot spectrogram
                im = axs[i][0].imshow(spec.T, aspect='auto', origin='lower', cmap='viridis')
                axs[i][0].set_title(f"Phoneme '{phoneme}' in word '{word}' (Participant: {participant})")
                axs[i][0].set_ylabel('Frequency Bin')
                axs[i][0].set_xlabel('Time Frame')
                
                # Add colorbar
                plt.colorbar(im, ax=axs[i][0])
                
                # 2. Plot frequency profile (amplitude spectrum)
                # For mel spectrograms, the values represent log-scaled energy
                # Taking the absolute values gives us amplitude-like representation
                amplitude_spectrum = np.mean(np.abs(spec), axis=0)
                freq_bins = np.arange(len(amplitude_spectrum))

                # Plot with frequencies on x-axis (horizontal) and amplitude on y-axis (vertical)
                axs[i][1].plot(freq_bins, amplitude_spectrum, color='#e74c3c')
                axs[i][1].set_title('Amplitude Spectrum')
                axs[i][1].set_xlabel('Frequency Bin')
                axs[i][1].set_ylabel('Amplitude')
                axs[i][1].grid(True, linestyle='--', alpha=0.7)

                # Add a thin horizontal line at y=0 for reference
                axs[i][1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
                
                # 3. Plot time evolution of key frequency bands
                # Select a few frequency bands to display
                if spec.shape[1] > 5:
                    # Select evenly spaced frequency bands
                    band_indices = np.linspace(0, spec.shape[1]-1, 5, dtype=int)
                    for j, band_idx in enumerate(band_indices):
                        # Get a color from the viridis colormap
                        color_val = j / (len(band_indices) - 1)
                        color = plt.cm.viridis(color_val)
                        axs[i][2].plot(spec[:, band_idx], 
                                      label=f'Band {band_idx}', 
                                      color=color, 
                                      alpha=0.8)
                    
                    axs[i][2].set_title('Time Evolution of Key Frequency Bands')
                    axs[i][2].set_xlabel('Time Frame')
                    axs[i][2].set_ylabel('Energy')
                    axs[i][2].legend(loc='upper right')
                    axs[i][2].grid(True, linestyle='--', alpha=0.7)
                else:
                    # Not enough frequency bands, show average energy over time
                    axs[i][2].plot(np.mean(spec, axis=1), color='#2980b9')
                    axs[i][2].set_title('Average Energy Over Time')
                    axs[i][2].set_xlabel('Time Frame')
                    axs[i][2].set_ylabel('Energy')
                    axs[i][2].grid(True, linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            
            # Add summary statistics at the bottom if requested
            if show_stats:
                stats = phoneme_stats[phoneme]
                stats_text = (
                    f"Summary for phoneme '{phoneme}':\n"
                    f"Total occurrences: {stats['count']}\n"
                    f"Mean duration: {stats['mean_duration']:.2f} frames (±{stats['std_duration']:.2f})\n"
                    f"Found in {stats['words']} different words across {stats['participants']} participants"
                )
                plt.figtext(0.5, 0.01, stats_text, ha='center', fontsize=12, 
                           bbox={'facecolor': 'lightgray', 'alpha': 0.5, 'pad': 5})
                plt.subplots_adjust(bottom=0.15)
            
            # Save the figure with a sanitized filename
            safe_phoneme = sanitize_filename(phoneme)
            save_path = os.path.join(save_dir, f"phoneme_{safe_phoneme}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            self.log(f"Visualization for phoneme '{phoneme}' saved to {save_path}")
            visualization_paths[phoneme] = save_path
        
        # Create a summary visualization showing phoneme frequency
        plt.figure(figsize=(12, 6))
        
        # Get counts for target phonemes
        targets = [p for p in target_phonemes if p in phoneme_counter]
        counts = [phoneme_counter[p] for p in targets]
        
        # Sort by frequency
        sorted_indices = np.argsort(counts)[::-1]
        sorted_phonemes = [targets[i] for i in sorted_indices]
        sorted_counts = [counts[i] for i in sorted_indices]
        
        # Create bar chart
        bars = plt.bar(sorted_phonemes, sorted_counts, color='#3498db')
        
        # Add count labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f"{int(height)}", ha='center', va='bottom')
        
        plt.title('Phoneme Frequency Distribution')
        plt.xlabel('Phoneme')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save the frequency chart
        freq_path = os.path.join(save_dir, "phoneme_frequency.png")
        plt.savefig(freq_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create a phoneme comparison visualization
        if len(target_phonemes) > 1:
            # Extract average spectrograms for each phoneme
            avg_specs = {}
            for phoneme in target_phonemes:
                if phoneme in phoneme_examples and phoneme_examples[phoneme]:
                    # Get all spectrograms for this phoneme
                    specs = [ex['spectrogram'] for ex in phoneme_examples[phoneme]]
                    
                    # Standardize lengths to the median length
                    lengths = [spec.shape[0] for spec in specs]
                    median_length = int(np.median(lengths))
                    
                    # Resize spectrograms to median length
                    resized_specs = []
                    for spec in specs:
                        if spec.shape[0] > median_length:
                            resized_specs.append(spec[:median_length])
                        elif spec.shape[0] < median_length:
                            # Pad with zeros
                            padding = np.zeros((median_length - spec.shape[0], spec.shape[1]))
                            resized_specs.append(np.vstack([spec, padding]))
                        else:
                            resized_specs.append(spec)
                    
                    # Calculate average spectrogram
                    avg_spec = np.mean(resized_specs, axis=0)
                    avg_specs[phoneme] = avg_spec
            
            # Create comparison visualization
            n_phonemes = len(avg_specs)
            if n_phonemes > 0:
                # Calculate number of rows and columns for grid layout
                n_cols = min(3, n_phonemes)
                n_rows = (n_phonemes + n_cols - 1) // n_cols
                
                fig, axs = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
                
                # Handle single row/column cases
                if n_rows * n_cols == 1:
                    axs = np.array([[axs]])
                elif n_rows == 1 or n_cols == 1:
                    axs = axs.reshape(n_rows, n_cols)
                
                # Plot each average spectrogram
                for i, (phoneme, avg_spec) in enumerate(avg_specs.items()):
                    row = i // n_cols
                    col = i % n_cols
                    
                    im = axs[row, col].imshow(avg_spec.T, aspect='auto', origin='lower', cmap='viridis')
                    axs[row, col].set_title(f"Average Spectrogram: '{phoneme}' (n={phoneme_counter[phoneme]})")
                    axs[row, col].set_ylabel('Frequency Bin')
                    axs[row, col].set_xlabel('Time Frame')
                    
                    # Add colorbar
                    plt.colorbar(im, ax=axs[row, col])
                
                # Hide empty subplots
                for i in range(len(avg_specs), n_rows * n_cols):
                    row = i // n_cols
                    col = i % n_cols
                    axs[row, col].axis('off')
                
                plt.tight_layout()
                
                # Save the comparison
                comparison_path = os.path.join(save_dir, "phoneme_comparison.png")
                plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
                
                self.log(f"Phoneme comparison visualization saved to {comparison_path}")
                visualization_paths['comparison'] = comparison_path
        
        self.log(f"Phoneme visualizations completed. Results saved to {save_dir}")
        
        return visualization_paths