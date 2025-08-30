from brain_audio_decoder import BrainAudioDecoder
from extract_features import extractHG, stackFeatures, extractMelSpecs, downsampleLabels, nameVector

import pandas as pd
import numpy as np
import os
import random


from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import scipy
from scipy.stats import pearsonr
from scipy.signal import detrend

import matplotlib.pyplot as plt

from collections import defaultdict
from pynwb import NWBHDF5IO       
from debugger import DebugMixin
        
        
class CustomBrainAudioDecoder(BrainAudioDecoder, DebugMixin):
    
    """
    Extended version of BrainAudioDecoder with additional methods
    for experimentation and improvements.
    """
    
    def __init__(self, path_bids, path_output, path_results, debug_mode=False, **kwargs):
        """Initialize with parent class parameters and additional options"""
        # Initialize parent BrainAudioDecoder
        BrainAudioDecoder.__init__(self, path_bids, path_output, path_results, **kwargs)
        
        # Initialize DebugMixin
        DebugMixin.__init__(self, class_name="CustomBrainAudioDecoder", debug_mode=debug_mode)
        
        self.log(f"Initializing CustomBrainAudioDecoder with debug_mode={self.DEBUG_MODE}")
        
        # Store additional models for comparison
        self.models = {
            'linear': self.model,  # The default LinearRegression model
            'ridge': Ridge(alpha=1.0),
            'lasso': Lasso(alpha=0.1),
            'elastic_net': ElasticNet(alpha=0.1, l1_ratio=0.5),
            'random_forest': RandomForestRegressor(n_estimators=100, max_depth=10),
            'svr': SVR(kernel='linear'),
            'mlp': MLPRegressor(hidden_layer_sizes=(100,), max_iter=500)
        }
        
        # Results for each model
        self.model_results = {}
        self.debug("Initialized additional models for comparison")
    
    def train_test_with_model(self, participant_id, model_name, save_audio=False):
        """
        Train and test with a specific model
        
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Available models: {list(self.models.keys())}")
        
        print(f"Training and testing {model_name} model for {participant_id}...")
        
        # Save current model
        original_model = self.model
        
        # Set model to use
        self.model = self.models[model_name]
        
        # Train and test
        results = self.train_test_model(participant_id, save_audio=save_audio)
        
        # Store results
        if participant_id not in self.model_results:
            self.model_results[participant_id] = {}
        
        self.model_results[participant_id][model_name] = results
        
        # Restore original model
        self.model = original_model
        
        return results
  
    def custom_feature_extraction(self, eeg, eeg_sr, method='high_gamma'):
        """
        Custom feature extraction method
        
        Parameters:
        -----------
        eeg : array
            EEG time series
        eeg_sr : int
            Sampling rate of EEG
        method : str
            Feature extraction method
            
        Returns:
        --------
        array
            Extracted features
        """
        if method == 'high_gamma':
            # Use original high gamma extraction
            return extractHG(eeg, eeg_sr, windowLength=self.win_length, frameshift=self.frameshift)
        elif method == 'multi_band':

            # Extract multiple frequency bands
            
            # Define frequency bands
            bands = {
                'delta': (1, 4),
                'theta': (4, 8),
                'alpha': (8, 13),
                'beta': (13, 30),
                'gamma': (30, 70),
                'high_gamma': (70, 170)
            }
            
            # Initialize features
            num_windows = int(np.floor((eeg.shape[0] - self.win_length * eeg_sr) / (self.frameshift * eeg_sr)))
            all_features = np.zeros((num_windows, eeg.shape[1] * len(bands)))
            
            # Extract features for each band
            for i, (band_name, (low_freq, high_freq)) in enumerate(bands.items()):
                # Linear detrend
                data = scipy.signal.detrend(eeg, axis=0)
                
                # Filter for band
                sos = scipy.signal.iirfilter(
                    4, [low_freq / (eeg_sr / 2), high_freq / (eeg_sr / 2)],
                    btype='bandpass', output='sos'
                )
                data = scipy.signal.sosfiltfilt(sos, data, axis=0)
                
                # If high gamma, attenuate line noise
                if band_name == 'high_gamma':
                    # Attenuate first harmonic of line noise
                    sos = scipy.signal.iirfilter(
                        4, [98 / (eeg_sr / 2), 102 / (eeg_sr / 2)],
                        btype='bandstop', output='sos'
                    )
                    data = scipy.signal.sosfiltfilt(sos, data, axis=0)
                    
                    # Attenuate second harmonic of line noise
                    sos = scipy.signal.iirfilter(
                        4, [148 / (eeg_sr / 2), 152 / (eeg_sr / 2)],
                        btype='bandstop', output='sos'
                    )
                    data = scipy.signal.sosfiltfilt(sos, data, axis=0)
                
                # Get envelope
                data = np.abs(scipy.signal.hilbert(data, axis=0))
                
                # Extract windows
                start_idx = i * eeg.shape[1]
                end_idx = (i + 1) * eeg.shape[1]
                
                for win in range(num_windows):
                    start = int(np.floor((win * self.frameshift) * eeg_sr))
                    stop = int(np.floor(start + self.win_length * eeg_sr))
                    all_features[win, start_idx:end_idx] = np.mean(data[start:stop, :], axis=0)
            
            return all_features
        else:
            raise ValueError(f"Unknown feature extraction method: {method}")
    
    def extract_features_with_custom_method(self, participant_id, method='high_gamma'):
        """
        Extract features using custom method
        
        Parameters:
        -----------
        participant_id : str
            Participant ID (e.g., 'sub-01')
        method : str
            Feature extraction method: 'high_gamma' or 'multi_band'
            
        Returns:
        --------
        tuple
            (features, spectrogram, words, feature_names)
        """
        self.log(f"Extracting features for {participant_id} using {method} method...")
        
        # Load data
        eeg, eeg_sr, audio, audio_sr, words, channels = self._load_participant_data(participant_id)
        
        # Extract features with custom method
        feat = self.custom_feature_extraction(eeg, eeg_sr, method=method)
        
        # Stack features with temporal context
        feat = stackFeatures(feat, modelOrder=self.model_order, stepSize=self.step_size)
        
        # Process audio
        scaled, audio_sr = self._process_audio(audio, audio_sr)
        
        # Extract mel spectrogram
        mel_spec = extractMelSpecs(
            scaled, 
            audio_sr, 
            windowLength=self.win_length, 
            frameshift=self.frameshift
        )
        
        # Align features
        words, mel_spec, feat = self._align_features(feat, mel_spec, words, eeg_sr)
        
        # Create feature names based on method
        if method == 'high_gamma':
            feature_names = nameVector(channels[:, None], modelOrder=self.model_order)
        elif method == 'multi_band':
            # Create names for each band
            bands = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'high_gamma']
            all_names = []
            
            for band in bands:
                for ch in channels:
                    all_names.append(f"{ch}_{band}")
                    
            feature_names = nameVector(np.array(all_names)[:, None], modelOrder=self.model_order)
        
        # Save features
        self._save_features(participant_id, feat, mel_spec, words, feature_names, method)
        
        return feat, mel_spec, words, feature_names
        
    def stratify_participants_by_channel_quality(self, channel_correlation_threshold=0.1):
        """
        Stratify participants based on the quality and relevance of their EEG channels.
        
        Parameters:
        -----------
        channel_correlation_threshold : float
            Minimum correlation coefficient for a channel to be considered relevant
        verbose : bool
            Whether to print progress information
        
        Returns:
        --------
        participant_strata : dict
            Dictionary with participant stratification information
            Format:
            {
                'high_quality': list,  # Participants with many relevant channels
                'medium_quality': list,  # Participants with average number of relevant channels
                'low_quality': list,  # Participants with few relevant channels
                'participant_metrics': {
                    'participant_id': {
                        'relevant_channel_count': int,
                        'relevant_channels': list,
                        'mean_correlation': float,
                        'quality_score': float
                    },
                    # More participants...
                },
                'metadata': {
                    'channel_correlation_threshold': float,
                    'total_participants': int
                }
            }
        """

        self.debug("Stratifying participants based on channel quality...")
        
        # Get all participants
        participant_ids = [f'sub-{i:02d}' for i in range(1, 11)]
        
        # Metrics to track for each participant
        participant_metrics = {}
        
        # 1. Analyze channel quality for each participant
        for participant_id in participant_ids:
            self.debug(f"Analyzing channels for {participant_id}...")
            
            # Check if channel analysis results exist
            result_path = os.path.join(self.path_results, 'channel_analysis', 
                                      f'{participant_id}_channel_correlations.npy')
            
            if not os.path.exists(result_path):
                self.debug(f"Starting segmentation for {participant_id}")
                continue
            
            # Load channel correlations
            channel_results = np.load(result_path, allow_pickle=True).item()
            
            # Extract relevant channels (above threshold)
            relevant_channels = []
            correlations = []
            
            for channel_name, channel_info in channel_results.items():
                if 'correlation' in channel_info and not np.isnan(channel_info['correlation']):
                    correlation = channel_info['correlation']
                    correlations.append(correlation)
                    
                    if correlation >= channel_correlation_threshold:
                        relevant_channels.append({
                            'name': channel_name,
                            'correlation': correlation,
                            'region': channel_info.get('region', 'Unknown')
                        })
            
            # Calculate metrics
            mean_correlation = np.mean(correlations) if correlations else 0
            relevant_count = len(relevant_channels)
            
            # Sort relevant channels by correlation (highest first)
            relevant_channels = sorted(relevant_channels, key=lambda x: x['correlation'], reverse=True)
            
            # Calculate a quality score (combination of number of relevant channels and their correlations)
            # This score prioritizes both quantity and quality of channels
            quality_score = relevant_count * mean_correlation
            
            # Store metrics
            participant_metrics[participant_id] = {
                'relevant_channel_count': relevant_count,
                'relevant_channels': relevant_channels,
                'mean_correlation': mean_correlation,
                'quality_score': quality_score
            }
        
        if not participant_metrics:
            self.debug("No channel analysis results found for any participant.")
            return None
        
        # 2. Calculate statistics for stratification
        quality_scores = [metrics['quality_score'] for metrics in participant_metrics.values()]
        
        # Define thresholds for stratification
        # Using percentiles to create roughly equal-sized groups
        if len(quality_scores) >= 6:  # If we have enough participants for 3 groups
            low_threshold = np.percentile(quality_scores, 33)
            high_threshold = np.percentile(quality_scores, 67)
        else:  # For fewer participants, use a simpler median split
            low_threshold = np.median(quality_scores) * 0.8
            high_threshold = np.median(quality_scores) * 1.2
        
        # 3. Stratify participants
        strata = {
            'high_quality': [],
            'medium_quality': [],
            'low_quality': []
        }
        
        for participant_id, metrics in participant_metrics.items():
            
            score = metrics['quality_score']
            if score >= high_threshold:
                strata['high_quality'].append(participant_id)
            elif score >= low_threshold:
                strata['medium_quality'].append(participant_id)
            else:
                strata['low_quality'].append(participant_id)
        
        # 4. Create the final structure
        participant_strata = {
            'high_quality': strata['high_quality'],
            'medium_quality': strata['medium_quality'],
            'low_quality': strata['low_quality'],
            'participant_metrics': participant_metrics,
            'metadata': {
                'channel_correlation_threshold': channel_correlation_threshold,
                'total_participants': len(participant_metrics),
                'quality_score_thresholds': {
                    'low_threshold': low_threshold,
                    'high_threshold': high_threshold
                }
            }
        }
        
        # 5. Print statistics
        self.log("\nParticipant stratification results:")
        self.log(f"  Participants with most relevant channels: {len(strata['high_quality'])}")
        self.log(f"  Participants with relevant channels: {len(strata['medium_quality'])}")
        self.log(f"  Participants with least relevant channels: {len(strata['low_quality'])}")
        self.log("\nTop participants by channel quality:")
            
        # Sort participants by quality score and print top 3
        sorted_participants = sorted(participant_metrics.items(), 
                                       key=lambda x: x[1]['quality_score'], 
                                       reverse=True)
            
        for i, (p_id, metrics) in enumerate(sorted_participants[:3]):
            self.debug(f"  {i+1}. {p_id}: {metrics['relevant_channel_count']} relevant channels, "
                     f"mean correlation: {metrics['mean_correlation']:.4f}, "
                     f"quality score: {metrics['quality_score']:.4f}")
            
        if len(sorted_participants) > 3:
                self.debug("  ...")
            
            # Print bottom participant
        p_id, metrics = sorted_participants[-1]
        self.debug(f"  {len(sorted_participants)}. {p_id}: {metrics['relevant_channel_count']} relevant channels, "
                 f"mean correlation: {metrics['mean_correlation']:.4f}, "
                 f"quality score: {metrics['quality_score']:.4f}")
        
        return participant_strata
                
    def create_stratified_cross_word_split(self, participant_strata, word_segments_dict=None, 
                               test_ratio=0.2, min_word_freq=1, random_seed=42):
        """
        Create train/test splits using cross-word validation while maintaining balance
        across participant quality strata.
        
        Parameters:
        -----------
        participant_strata : dict
            Output from stratify_participants_by_channel_quality method
        word_segments_dict : dict or None
            Dictionary mapping participant_id to word_segments (output from segment_data_by_words)
            If None, will call segment_data_by_words for each participant
        test_ratio : float
            Proportion of words to use for testing (0.0-1.0)
        min_word_freq : int
            Minimum frequency for a word to be included in the split
        random_seed : int
            Seed for random number generator to ensure reproducibility
        
        Returns:
        --------
        dict
            Dictionary containing train/test splits and statistics
        """
        # Set random seed for reproducibility
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # If no word segments dictionary provided, create one
        if word_segments_dict is None:
            word_segments_dict = {}
            
        # Get participants from each stratum
        high_quality_participants = participant_strata.get('high_quality', [])
        medium_quality_participants = participant_strata.get('medium_quality', [])
        low_quality_participants = participant_strata.get('low_quality', [])
        
        # Combine all strata
        all_participants = []
        all_participants.extend(high_quality_participants)
        all_participants.extend(medium_quality_participants)
        all_participants.extend(low_quality_participants)
        
        if not all_participants:
            all_participants = [f'sub-{i:02d}' for i in range(1, 11)]
        
        # Segment data for each participant if not already provided
        if not word_segments_dict:
            for participant_id in all_participants:
                try:
                    self.log(f"Segmenting data for {participant_id}...")
                    word_segments = self.segment_data_by_words(
                        participant_id=participant_id,
                        min_word_freq=min_word_freq
                    )
                    if word_segments:
                        word_segments_dict[participant_id] = word_segments
                except Exception as e:
                    self.log(f"Error segmenting data for {participant_id}: {e}")
        
        # Initialize train/test split structures
        train_split = {}
        test_split = {}
        
        # Track statistics
        strata_statistics = {
            'high_quality': {'train': 0, 'test': 0},
            'medium_quality': {'train': 0, 'test': 0},
            'low_quality': {'train': 0, 'test': 0}
        }
        
        # Process each participant
        total_train = 0
        total_test = 0
        
        for participant_id, word_segments in word_segments_dict.items():
            # Determine participant stratum
            participant_stratum = 'medium_quality'  # Default
            if participant_id in high_quality_participants:
                participant_stratum = 'high_quality'
            elif participant_id in low_quality_participants:
                participant_stratum = 'low_quality'
            
            # Initialize splits for this participant
            train_split[participant_id] = {}
            test_split[participant_id] = {}
            
            # Get all words for this participant
            all_words = list(word_segments['words'].keys())
            
            # Skip if no words
            if not all_words:
                continue
                
            # Determine number of test words
            # Adjust test ratio based on participant quality to ensure balanced representation
            adjusted_test_ratio = test_ratio
            if participant_stratum == 'high_quality':
                # Ensure high quality participants have good representation in both sets
                adjusted_test_ratio = test_ratio
            elif participant_stratum == 'medium_quality':
                # Standard ratio for medium quality
                adjusted_test_ratio = test_ratio
            else:
                # Slightly increase test representation for low quality participants
                adjusted_test_ratio = test_ratio * 1.1
                
            # Ensure test_ratio remains valid
            adjusted_test_ratio = min(0.5, max(0.1, adjusted_test_ratio))
            
            # Calculate number of test words
            num_test_words = max(1, int(len(all_words) * adjusted_test_ratio))
            num_test_words = min(num_test_words, len(all_words) - 1)  # Ensure at least one word for training
            
            # Randomly select test words
            test_words = random.sample(all_words, num_test_words)
            train_words = [w for w in all_words if w not in test_words]
            
            # Assign words to train/test splits
            for word in train_words:
                # Get all instances for this word
                instances = word_segments['words'][word]['instances']
                # Add all instances to training set
                train_split[participant_id][word] = list(range(len(instances)))
                
                # Update statistics
                strata_statistics[participant_stratum]['train'] += len(instances)
                total_train += len(instances)
            
            for word in test_words:
                # Get all instances for this word
                instances = word_segments['words'][word]['instances']
                # Add all instances to test set
                test_split[participant_id][word] = list(range(len(instances)))
                
                # Update statistics
                strata_statistics[participant_stratum]['test'] += len(instances)
                total_test += len(instances)
        
        # Calculate overall statistics
        total_strata_train = sum(stats['train'] for stats in strata_statistics.values())
        total_strata_test = sum(stats['test'] for stats in strata_statistics.values())
        
        # Calculate proportion of each stratum in train and test sets
        strata_proportions = {
            'train': {},
            'test': {}
        }
        
        for stratum, counts in strata_statistics.items():
            if total_strata_train > 0:
                strata_proportions['train'][stratum] = counts['train'] / total_strata_train
            else:
                strata_proportions['train'][stratum] = 0
                
            if total_strata_test > 0:
                strata_proportions['test'][stratum] = counts['test'] / total_strata_test
            else:
                strata_proportions['test'][stratum] = 0
        
        # Create final result structure
        split_result = {
            'train': train_split,
            'test': test_split,
            'statistics': {
                'total_train_instances': total_train,
                'total_test_instances': total_test,
                'participants_by_strata': strata_statistics,
                'strata_proportions': strata_proportions,
            },
            'word_segments_dict': word_segments_dict  # Include the word segments for reference
        }
        
        # Print summary statistics
        if total_train + total_test > 0:
            self.log(f"Total train instances: {total_train}")
            self.log(f"Total test instances: {total_test}")
            self.log(f"Train/Test ratio: {total_train/(total_train+total_test):.2f}/{total_test/(total_train+total_test):.2f}")
        else:
            self.log("No instances found for train or test sets!")
            self.log("Please check your data and parameters:")
            self.log(f"- min_word_freq: {min_word_freq}")
            self.log(f"- Number of participants: {len(word_segments_dict)}")
        
        self.log("\nParticipants by strata:")
        for stratum, counts in strata_statistics.items():
            if counts['train'] + counts['test'] > 0:
                self.log(f"  {stratum}: {counts['train']} train, {counts['test']} test")
                self.log(f"    Train proportion: {strata_proportions['train'][stratum]:.2f}")
                self.log(f"    Test proportion: {strata_proportions['test'][stratum]:.2f}")
        
        return split_result
    
    def segment_data_by_words(self, participant_id, pre_onset_ms=200, post_offset_ms=200, 
                     min_word_freq=1, handle_overlaps='adjust'):
        """
        Segment EEG and audio data by words for a participant, creating a structured dataset
        suitable for stratified sampling.
        
        Parameters:
        -----------
        participant_id : str
            Participant ID (e.g., 'sub-01')
        pre_onset_ms : int
            Milliseconds to include before word onset
        post_offset_ms : int
            Milliseconds to include after word offset
        min_word_freq : int
            Minimum frequency of a word to be included
        verbose : bool
            Whether to print progress information
        handle_overlaps : str
            How to handle overlapping segments:
            - 'adjust': Adjust window sizes to avoid overlaps
            - 'flag': Flag overlapping segments in metadata
            - 'skip': Skip overlapping segments
            - 'allow': Allow overlaps (original behavior)
        
        Returns:
        --------
        word_segments : dict
            Structured dictionary with word segments
        """
        import gc
        self.debug(f"Segmenting data by words for {participant_id}...")
        
        # 1. Load the necessary data
        try:
            # Load EEG data
            io = NWBHDF5IO(
                os.path.join(self.path_bids, participant_id, 'ieeg', 
                            f'{participant_id}_task-wordProduction_ieeg.nwb'), 
                'r'
            )
            nwbfile = io.read()
            eeg_data = nwbfile.acquisition['iEEG'].data[:]
            eeg_sr = 1024  # Default sampling rate for this dataset
            
            # Load audio data
            audio_data = nwbfile.acquisition['Audio'].data[:]
            audio_sr = 48000  # Default sampling rate for audio
            
            # Load word markers
            word_markers = nwbfile.acquisition['Stimulus'].data[:]
            word_markers = np.array(word_markers, dtype=str)
            
            self.debug(f"Loaded {len(word_markers)} word markers")
            self.debug(f"First 10 word markers: {word_markers[:10]}")
            
            io.close()
            gc.collect()
            
            # Load channel names if available
            channels_df = pd.read_csv(
                os.path.join(self.path_bids, participant_id, 'ieeg', 
                            f'{participant_id}_task-wordProduction_channels.tsv'), 
                delimiter='\t'
            )
            channel_names = channels_df['name'].values
            
            # Load spectrogram if available
            spec_path = os.path.join(self.path_output, f'{participant_id}_spec.npy')
            if os.path.exists(spec_path):
                spectrogram = np.load(spec_path)
                spec_available = True
            else:
                spectrogram = None
                spec_available = False
                self.debug("Spectrogram not found, proceeding without it.")
        
        except MemoryError as e:
            self.log(f"Memory error for {participant_id}: {e}")
            # Try to free memory and continue
            gc.collect()
            return None
        
        # 2. Identify word boundaries
        # Convert pre/post duration to samples
        pre_onset_samples_eeg = int(pre_onset_ms / 1000 * eeg_sr)
        post_offset_samples_eeg = int(post_offset_ms / 1000 * eeg_sr)
        
        pre_onset_samples_audio = int(pre_onset_ms / 1000 * audio_sr)
        post_offset_samples_audio = int(post_offset_ms / 1000 * audio_sr)
        
        # Find word onsets (when words change)
        word_onsets = []
        current_word = ""
        
        for i, word in enumerate(word_markers):
            # Ensure word is a string
            if not isinstance(word, str):
                word = str(word).strip()
                
            # Skip empty strings
            if word == "":
                continue
                
            # Detect word changes (onsets)
            if word != current_word:
                word_onsets.append((i, word))
                current_word = word
        
        self.debug(f"Found {len(word_onsets)} word onsets")
        self.debug(f"First 5 word onsets: {word_onsets[:5]}")
        
        # 3. Extract word instances
        word_instances = defaultdict(list)
        overlapping_segments = 0
        adjusted_segments = 0
        skipped_segments = 0
        
        # First pass - collect all segment boundaries
        segment_boundaries = []
        
        for i, (onset_idx, word) in enumerate(word_onsets):
            # Determine word offset
            if i < len(word_onsets) - 1:
                offset_idx = word_onsets[i+1][0] - 1
            else:
                # Last word goes to the end
                offset_idx = len(word_markers) - 1
            
            # Calculate segment boundaries with pre/post windows
            eeg_start = max(0, onset_idx - pre_onset_samples_eeg)
            eeg_end = min(eeg_data.shape[0], offset_idx + post_offset_samples_eeg + 1)
            
            segment_boundaries.append({
                'word': word,
                'onset_idx': onset_idx,
                'offset_idx': offset_idx,
                'eeg_start': eeg_start,
                'eeg_end': eeg_end,
                'original_eeg_start': eeg_start,
                'original_eeg_end': eeg_end,
                'adjusted': False,
                'overlapping': False,
                'skipped': False
            })
        
        # Second pass - check for overlaps and adjust if needed
        if handle_overlaps != 'allow':
            for i in range(len(segment_boundaries) - 1):
                current = segment_boundaries[i]
                next_seg = segment_boundaries[i + 1]
                
                # Check if segments overlap
                if current['eeg_end'] > next_seg['eeg_start']:
                    # Flag as overlapping
                    current['overlapping'] = True
                    next_seg['overlapping'] = True
                    overlapping_segments += 1
                    
                    if handle_overlaps == 'adjust':
                        # Find midpoint between current word end and next word start
                        midpoint = (current['offset_idx'] + next_seg['onset_idx']) // 2
                        
                        # Adjust boundaries to meet at midpoint
                        current['eeg_end'] = midpoint
                        next_seg['eeg_start'] = midpoint
                        
                        # Mark as adjusted
                        current['adjusted'] = True
                        next_seg['adjusted'] = True
                        adjusted_segments += 1
                        
                    elif handle_overlaps == 'skip':
                        # Mark shorter segment to be skipped
                        current_duration = current['offset_idx'] - current['onset_idx']
                        next_duration = next_seg['offset_idx'] - next_seg['onset_idx']
                        
                        if current_duration < next_duration:
                            current['skipped'] = True
                        else:
                            next_seg['skipped'] = True
                        
                        skipped_segments += 1
        
        # Third pass - extract segments based on adjusted boundaries
        for segment in segment_boundaries:
            # Skip if marked for skipping
            if segment['skipped']:
                continue
            
            word = segment['word']
            onset_idx = segment['onset_idx']
            offset_idx = segment['offset_idx']
            eeg_start = segment['eeg_start']
            eeg_end = segment['eeg_end']
            
            # Calculate duration
            duration_samples = offset_idx - onset_idx + 1
            duration_ms = duration_samples * (1000 / eeg_sr)
            
            # Calculate corresponding audio indices
            audio_ratio = audio_sr / eeg_sr
            audio_start = int(eeg_start * audio_ratio)
            audio_end = int(eeg_end * audio_ratio)
            
            # Extract segments
            eeg_segment = eeg_data[eeg_start:eeg_end, :]
            
            # Check if audio indices are valid
            if audio_start < audio_data.shape[0] and audio_end <= audio_data.shape[0]:
                audio_segment = audio_data[audio_start:audio_end]
            else:
                # Handle edge case where audio indices are out of bounds
                audio_segment = np.zeros(1)  # Empty placeholder
                self.debug(f"Warning: Audio segment out of bounds for word '{word}' at index {onset_idx}")
            
            # Extract spectrogram segment if available
            spec_segment = None  # Initialize with None
            if spec_available:
                # Spectrogram has fewer time points than EEG due to windowing
                # Need to map EEG indices to spectrogram indices
                if hasattr(self, 'win_length') and hasattr(self, 'frameshift'):
                    # Calculate spectrogram indices based on windowing parameters
                    spec_ratio = (self.frameshift * eeg_sr)  # Samples per spectrogram frame
                    spec_start = max(0, int(eeg_start / spec_ratio))
                    spec_end = min(spectrogram.shape[0], int(eeg_end / spec_ratio))
                    
                    if spec_start < spectrogram.shape[0] and spec_end <= spectrogram.shape[0]:
                        spec_segment = spectrogram[spec_start:spec_end, :]
                    else:
                        spec_segment = np.zeros((1, spectrogram.shape[1]))  # Empty placeholder
                else:
                    # If windowing parameters are not available, use a rough approximation
                    spec_ratio = eeg_data.shape[0] / spectrogram.shape[0]
                    spec_start = max(0, int(eeg_start / spec_ratio))
                    spec_end = min(spectrogram.shape[0], int(eeg_end / spec_ratio))
                    
                    if spec_start < spectrogram.shape[0] and spec_end <= spectrogram.shape[0]:
                        spec_segment = spectrogram[spec_start:spec_end, :]
                    else:
                        spec_segment = np.zeros((1, spectrogram.shape[1]))  # Empty placeholder
            
            # Store the instance
            instance = {
                'onset_sample': onset_idx,
                'offset_sample': offset_idx,
                'eeg_start': eeg_start,
                'eeg_end': eeg_end,
                'audio_start': audio_start,
                'audio_end': audio_end,
                'duration_samples': duration_samples,
                'duration_ms': duration_ms,
                'eeg_segment': eeg_segment,
                'audio_segment': audio_segment,
                'overlapping': segment['overlapping'],
                'adjusted': segment['adjusted'],
                'original_eeg_start': segment['original_eeg_start'],
                'original_eeg_end': segment['original_eeg_end']
            }
            
            if spec_available and spec_segment is not None:
                instance['spectrogram_segment'] = spec_segment
            
            word_instances[word].append(instance)
        
        # 4. Create the data structure
        # Filter words by minimum frequency
        filtered_words = {word: instances for word, instances in word_instances.items() 
                         if len(instances) >= min_word_freq}
        
        self.debug(f"Found {len(filtered_words)} unique words with at least {min_word_freq} occurrences")
        if filtered_words:
            # Print some examples of the found words
            example_words = list(filtered_words.keys())[:5]
            for word in example_words:
                self.debug(f"Word '{word}': {len(filtered_words[word])} instances")
        
        # Create stratification groups based on word frequency
        word_counts = {word: len(instances) for word, instances in filtered_words.items()}
        
        # Define stratification groups
        stratification_groups = {
            'high_freq': [],    # > 10 occurrences
            'medium_freq': [],  # 5-10 occurrences
            'low_freq': []      # 1-4 occurrences
        }
        
        for word, count in word_counts.items():
            if count > 10:
                stratification_groups['high_freq'].append(word)
            elif count >= 5:
                stratification_groups['medium_freq'].append(word)
            else:
                stratification_groups['low_freq'].append(word)
        
        # 5. Build the final structure
        word_segments = {
            'metadata': {
                'participant_id': participant_id,
                'eeg_sr': eeg_sr,
                'audio_sr': audio_sr,
                'pre_onset_ms': pre_onset_ms,
                'post_offset_ms': post_offset_ms,
                'total_word_instances': sum(len(instances) for instances in filtered_words.values()),
                'channel_names': channel_names.tolist() if hasattr(channel_names, 'tolist') else channel_names,
                'spectrogram_available': spec_available,
                'overlap_handling': handle_overlaps,
                'overlap_statistics': {
                    'overlapping_segments': overlapping_segments,
                    'adjusted_segments': adjusted_segments,
                    'skipped_segments': skipped_segments
                }
            },
            'words': {
                word: {
                    'count': len(instances),
                    'instances': instances
                } for word, instances in filtered_words.items()
            },
            'word_list': list(filtered_words.keys()),
            'word_counts': word_counts,
            'stratification_groups': stratification_groups
        }
        
        # 6. Print statistics
        self.debug(f"Word statistics:")
        self.debug(f"  Total unique words: {len(filtered_words)}")
        self.debug(f"  Total word instances: {word_segments['metadata']['total_word_instances']}")
            
        # Print overlap statistics if any
        if overlapping_segments > 0:
            self.debug(f"\nOverlap statistics:")
            self.debug(f"  Overlapping segments detected: {overlapping_segments}")
            if handle_overlaps == 'adjust':
                self.debug(f"  Segments adjusted to avoid overlap: {adjusted_segments}")
            elif handle_overlaps == 'skip':
                self.debug(f"  Segments skipped due to overlap: {skipped_segments}")
            
        # Print a few examples of the most frequent words
        sorted_words = sorted(word_segments['word_counts'].items(), key=lambda x: x[1], reverse=True)
        if sorted_words:
            self.log(f"\n  Top 5 most frequent words:")
            for i, (word, count) in enumerate(sorted_words[:5]):
                self.log(f"    {i+1}. '{word}': {count} instances")
                
            if sorted_words:
                most_freq_word = sorted_words[0][0]
                self.log(f"\nExample EEG segment shape: {filtered_words[most_freq_word][0]['eeg_segment'].shape}")
                if spec_available and 'spectrogram_segment' in filtered_words[most_freq_word][0]:
                    self.log(f"Example spectrogram segment shape: {filtered_words[most_freq_word][0]['spectrogram_segment'].shape}")
        
        return word_segments
      
    def create_train_test_split(self, participant_strata, word_segments_dict=None, 
                               test_ratio=0.2, balanced_by_word_freq=True, 
                               min_word_freq=2, random_seed=42):
        """
        Create train/test splits for brain decoding, ensuring balanced representation
        across participant quality strata and word frequencies.
        
        Parameters:
        -----------
        self : CustomBrainAudioDecoder
            Instance of the CustomBrainAudioDecoder class
        participant_strata : dict
            Output from stratify_participants_by_channel_quality method
        word_segments_dict : dict or None
            Dictionary mapping participant_id to word_segments (output from segment_data_by_words)
            If None, will call segment_data_by_words for each participant
        test_ratio : float
            Proportion of data to use for testing (0.0-1.0)
        balanced_by_word_freq : bool
            Whether to balance the split based on word frequencies
        min_word_freq : int
            Minimum frequency for a word to be included in the split
        random_seed : int
            Seed for random number generator to ensure reproducibility
        
        Returns:
        --------
        dict
            Dictionary containing train/test splits and statistics
            Format:
            {
                'train': {
                    'participant_id': {
                        'word': [instance_indices],
                        ...
                    },
                    ...
                },
                'test': {
                    'participant_id': {
                        'word': [instance_indices],
                        ...
                    },
                    ...
                },
                'statistics': {
                    'total_train_instances': int,
                    'total_test_instances': int,
                    'participants_by_strata': dict,
                    'words_by_frequency': dict
                }
            }
        """
        # Set random seed for reproducibility
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # Group participants by strata
        all_participants = []
        if participant_strata:
            all_participants.extend(participant_strata.get('high_quality', []))
            all_participants.extend(participant_strata.get('medium_quality', []))
            all_participants.extend(participant_strata.get('low_quality', []))
        else:
            # If no strata available, use all participants
            all_participants = [f'sub-{i:02d}' for i in range(1, 11)]
            
        # Segment data for each participant
        for participant_id in all_participants:
            try:
                self.log(f"Segmenting data for {participant_id}...")
                word_segments = self.segment_data_by_words(
                    participant_id=participant_id,
                    min_word_freq=min_word_freq
                )
                if word_segments:
                    word_segments_dict[participant_id] = word_segments
            except Exception as e:
                self.log(f"Error segmenting data for {participant_id}: {e}")
        
        # Initialize train/test split structures
        train_split = {}
        test_split = {}
        
        # Track statistics for each stratum
        strata_statistics = {
            'high_quality': {'train': 0, 'test': 0},
            'medium_quality': {'train': 0, 'test': 0},
            'low_quality': {'train': 0, 'test': 0}
        }
        
        # Track word frequencies across all participants
        all_words = defaultdict(int)
        for participant_id, word_segments in word_segments_dict.items():
            for word, count in word_segments['word_counts'].items():
                all_words[word] += count
        
        # Group words by frequency
        word_frequency_groups = {
            'high_freq': [],    # > 10 occurrences
            'medium_freq': [],  # 5-10 occurrences
            'low_freq': []      # 1-4 occurrences
        }
        
        for word, count in all_words.items():
            if count > 10:
                word_frequency_groups['high_freq'].append(word)
            elif count >= 5:
                word_frequency_groups['medium_freq'].append(word)
            else:
                word_frequency_groups['low_freq'].append(word)
        
        # Initialize counts
        total_train = 0
        total_test = 0
        
        # Process each participant
        for participant_id, word_segments in word_segments_dict.items():
            # Determine participant stratum
            participant_stratum = 'medium_quality'  # Default
            if participant_strata:
                if participant_id in participant_strata.get('high_quality', []):
                    participant_stratum = 'high_quality'
                elif participant_id in participant_strata.get('low_quality', []):
                    participant_stratum = 'low_quality'
            
            # Initialize splits for this participant
            train_split[participant_id] = {}
            test_split[participant_id] = {}
            
            # Process each word for this participant
            for word, word_info in word_segments['words'].items():
                instances = word_info['instances']
                instance_count = len(instances)
                
                # Skip words with too few instances
                if instance_count < min_word_freq:
                    continue
                
                # Determine word frequency group
                word_freq_group = 'low_freq'
                if word in word_frequency_groups['high_freq']:
                    word_freq_group = 'high_freq'
                elif word in word_frequency_groups['medium_freq']:
                    word_freq_group = 'medium_freq'
                
                # Calculate number of test instances
                # Balanced sampling based on word frequency and participant quality
                if balanced_by_word_freq:
                    # Adjust test ratio based on word frequency and participant quality
                    # Higher quality participants and high frequency words get more test samples
                    if word_freq_group == 'high_freq' and participant_stratum == 'high_quality':
                        adjusted_test_ratio = test_ratio * 1.2  # Increase test ratio
                    elif word_freq_group == 'low_freq' and participant_stratum == 'low_quality':
                        adjusted_test_ratio = test_ratio * 0.8  # Decrease test ratio
                    else:
                        adjusted_test_ratio = test_ratio
                    
                    # Ensure at least one test instance for each word
                    num_test = max(1, int(instance_count * adjusted_test_ratio))
                else:
                    # Simple split without balancing
                    num_test = max(1, int(instance_count * test_ratio))
                
                # Ensure we don't take too many test samples
                num_test = min(num_test, instance_count - 1)
                
                # Randomly select test indices
                all_indices = list(range(instance_count))
                test_indices = sorted(random.sample(all_indices, num_test))
                train_indices = sorted(set(all_indices) - set(test_indices))
                
                # Store splits
                train_split[participant_id][word] = train_indices
                test_split[participant_id][word] = test_indices
                
                # Update statistics
                strata_statistics[participant_stratum]['train'] += len(train_indices)
                strata_statistics[participant_stratum]['test'] += len(test_indices)
                total_train += len(train_indices)
                total_test += len(test_indices)
        
        # Create final result structure
        split_result = {
            'train': train_split,
            'test': test_split,
            'statistics': {
                'total_train_instances': total_train,
                'total_test_instances': total_test,
                'participants_by_strata': strata_statistics,
                'words_by_frequency': {
                    'high_freq': len(word_frequency_groups['high_freq']),
                    'medium_freq': len(word_frequency_groups['medium_freq']),
                    'low_freq': len(word_frequency_groups['low_freq'])
                },
                'word_frequency_groups': word_frequency_groups
            },
            'word_segments_dict': word_segments_dict  # Include the word segments for reference
        }
        
        # Print summary statistics
        if total_train + total_test > 0:
            self.log(f"Total train instances: {total_train}")
            self.log(f"Total test instances: {total_test}")
            self.log(f"Train/Test ratio: {total_train/(total_train+total_test):.2f}/{total_test/(total_train+total_test):.2f}")
        else:
            self.log("No instances found for train or test sets!")
            self.log("Please check your data and parameters:")
            self.log(f"- min_word_freq: {min_word_freq}")
            self.log(f"- Number of participants: {len(word_segments_dict)}")
        
        self.log("\nParticipants by strata:")
        for stratum, counts in strata_statistics.items():
            if counts['train'] + counts['test'] > 0:
                self.log(f"  {stratum}: {counts['train']} train, {counts['test']} test")
        
        self.log("\nWords by frequency:")
        for freq_group, count in split_result['statistics']['words_by_frequency'].items():
            self.log(f"  {freq_group}: {count} words")
        
        return split_result

    def get_data_batch(self, split_result, batch_type='train', participant_ids=None, 
                      max_instances_per_word=None, balanced_sampling=True,
                      batch_size=32, random_seed=None):
        """
        Get a batch of data from the train/test split for model training or evaluation.
        
        Parameters:
        -----------
        split_result : dict
            Output from create_train_test_split function
        batch_type : str
            'train' or 'test'
        participant_ids : list or None
            List of participant IDs to include. If None, use all participants.
        max_instances_per_word : int or None
            Maximum number of instances to use per word. If None, use all available.
        balanced_sampling : bool
            Whether to sample evenly across participants and words
        batch_size : int
            Number of instances to include in the batch
        random_seed : int or None
            Seed for random number generator. If None, use different sampling each time.
        
        Returns:
        --------
        dict
            Dictionary containing batch data
            Format:
            {
                'eeg_segments': list of arrays,
                'audio_segments': list of arrays,
                'spectrogram_segments': list of arrays (if available),
                'words': list of strings,
                'participant_ids': list of strings,
                'metadata': dict with batch information
            }
        """
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
        
        # Get the correct split
        if batch_type not in ['train', 'test']:
            raise ValueError(f"Invalid batch_type: {batch_type}. Must be 'train' or 'test'.")
        
        split = split_result[batch_type]
        word_segments_dict = split_result['word_segments_dict']
        
        # Filter by participant IDs if specified
        if participant_ids is None:
            participant_ids = list(split.keys())
        else:
            # Ensure all requested participants are in the split
            participant_ids = [p_id for p_id in participant_ids if p_id in split]
            if not participant_ids:
                raise ValueError(f"None of the requested participants found in the {batch_type} split")
        
        # Collect all available instances
        available_instances = []
        
        for participant_id in participant_ids:
            participant_split = split[participant_id]
            word_segments = word_segments_dict[participant_id]
            
            for word, indices in participant_split.items():
                # Apply max_instances_per_word limit if specified
                if max_instances_per_word is not None and len(indices) > max_instances_per_word:
                    if balanced_sampling:
                        # Randomly select max_instances_per_word indices
                        selected_indices = random.sample(indices, max_instances_per_word)
                    else:
                        # Take the first max_instances_per_word indices
                        selected_indices = indices[:max_instances_per_word]
                else:
                    selected_indices = indices
                
                # Add each instance to the available list
                for idx in selected_indices:
                    available_instances.append({
                        'participant_id': participant_id,
                        'word': word,
                        'instance_index': idx,
                        'instance': word_segments['words'][word]['instances'][idx]
                    })
        
        # Perform balanced sampling if requested
        if balanced_sampling:
            # Group instances by participant and word
            by_participant = defaultdict(list)
            by_word = defaultdict(list)
            
            for i, instance_info in enumerate(available_instances):
                by_participant[instance_info['participant_id']].append(i)
                by_word[instance_info['word']].append(i)
            
            # Determine sampling strategy
            if len(by_participant) > 1 and len(by_word) > 1:
                # Sample evenly across both participants and words
                sampled_indices = []
                remaining_slots = min(batch_size, len(available_instances))
                
                # First, ensure each participant and word is represented
                for p_id in by_participant:
                    if remaining_slots <= 0:
                        break
                    sampled_indices.append(random.choice(by_participant[p_id]))
                    remaining_slots -= 1
                
                for word in by_word:
                    if remaining_slots <= 0:
                        break
                    word_indices = [idx for idx in by_word[word] if idx not in sampled_indices]
                    if word_indices:
                        sampled_indices.append(random.choice(word_indices))
                        remaining_slots -= 1
                
                # Fill remaining slots randomly
                remaining_indices = [i for i in range(len(available_instances)) if i not in sampled_indices]
                if remaining_indices and remaining_slots > 0:
                    additional_indices = random.sample(
                        remaining_indices, 
                        min(remaining_slots, len(remaining_indices))
                    )
                    sampled_indices.extend(additional_indices)
                
                # Select the sampled instances
                batch_instances = [available_instances[i] for i in sampled_indices]
            else:
                # Simple random sampling
                batch_instances = random.sample(
                    available_instances, 
                    min(batch_size, len(available_instances))
                )
        else:
            # Simple random sampling without balancing
            batch_instances = random.sample(
                available_instances, 
                min(batch_size, len(available_instances))
            )
        
        # Extract data from selected instances
        eeg_segments = []
        audio_segments = []
        spectrogram_segments = []
        words = []
        participant_ids_batch = []
        
        for instance_info in batch_instances:
            instance = instance_info['instance']
            
            # Add data
            eeg_segments.append(instance['eeg_segment'])
            audio_segments.append(instance['audio_segment'])
            
            # Add spectrogram if available
            if 'spectrogram_segment' in instance:
                spectrogram_segments.append(instance['spectrogram_segment'])
            
            # Add metadata
            words.append(instance_info['word'])
            participant_ids_batch.append(instance_info['participant_id'])
        
        # Create batch dictionary
        batch = {
            'eeg_segments': eeg_segments,
            'audio_segments': audio_segments,
            'words': words,
            'participant_ids': participant_ids_batch,
            'metadata': {
                'batch_type': batch_type,
                'batch_size': len(batch_instances),
                'balanced_sampling': balanced_sampling
            }
        }
        
        # Add spectrogram segments if available
        if spectrogram_segments:
            batch['spectrogram_segments'] = spectrogram_segments
        
        return batch

    def prepare_word_model_data(self, batch, feature_extraction_method='high_gamma', 
                          temporal_context=True, model_order=4, step_size=5,
                          standardize=True, pca_components=50):
        """
        Prepare data batch for model training/evaluation by extracting features
        and performing necessary preprocessing.
        
        Parameters:
        -----------
        batch : dict
            Output from get_data_batch function
        self : CustomBrainAudioDecoder
            Instance of the CustomBrainAudioDecoder class
        feature_extraction_method : str
            Method to use for feature extraction ('high_gamma', 'multi_band', etc.)
        temporal_context : bool
            Whether to add temporal context to features
        model_order : int
            Number of temporal contexts to include (if temporal_context=True)
        step_size : int
            Step size for temporal contexts (if temporal_context=True)
        standardize : bool
            Whether to standardize features (z-score normalization)
        pca_components : int or None
            Number of PCA components to use. If None, don't use PCA.
        
        Returns:
        --------
        dict
            Dictionary containing processed data ready for model input
        """

        
        # Extract features from EEG segments
        all_features = []
        all_spectrograms = []
        
        for i, eeg_segment in enumerate(batch['eeg_segments']):
            # Extract features based on method
            if feature_extraction_method == 'high_gamma':
                # Use high gamma band power
                features = extractHG(
                    eeg_segment, 
                    batch['metadata'].get('eeg_sr', 1024),
                    windowLength=self.win_length,
                    frameshift=self.frameshift
                )
            elif feature_extraction_method == 'multi_band':
                # Use custom multi-band feature extraction
                features = self.custom_feature_extraction(
                    eeg_segment,
                    batch['metadata'].get('eeg_sr', 1024),
                    method='multi_band'
                )
            else:
                raise ValueError(f"Unknown feature extraction method: {feature_extraction_method}")
            
            # Add temporal context if requested
            if temporal_context and features.shape[0] > model_order * step_size * 2:
                features = stackFeatures(features, modelOrder=model_order, stepSize=step_size)
            
            # Store features
            all_features.append(features)
            
            # Use provided spectrograms if available, otherwise we'd need to extract them
            if 'spectrogram_segments' in batch:
                all_spectrograms.append(batch['spectrogram_segments'][i])
        
        # Prepare data dictionary
        data = {
            'features': all_features,
            'spectrograms': all_spectrograms if all_spectrograms else None,
            'words': batch['words'],
            'participant_ids': batch['participant_ids'],
            'metadata': {
                'feature_extraction_method': feature_extraction_method,
                'temporal_context': temporal_context,
                'model_order': model_order if temporal_context else None,
                'step_size': step_size if temporal_context else None,
                'standardized': standardize,
                'pca_components': pca_components,
                'batch_metadata': batch['metadata']
            }
        }
        
        # Standardize features if requested
        if standardize:
            scaler = StandardScaler()
            
            # Standardize each feature set separately
            for i in range(len(all_features)):
                data['features'][i] = scaler.fit_transform(all_features[i])
        
        # Apply PCA if requested
        if pca_components is not None and pca_components > 0:
            pca = PCA(n_components=pca_components)
            
            # Apply PCA to each feature set separately
            for i in range(len(data['features'])):
                # Ensure we have enough samples for PCA
                if data['features'][i].shape[0] > pca_components and data['features'][i].shape[1] > pca_components:
                    data['features'][i] = pca.fit_transform(data['features'][i])
                else:
                    # Skip PCA for this segment
                    data['metadata']['pca_skipped_segments'] = data['metadata'].get('pca_skipped_segments', 0) + 1
        
        return data
        
        
    