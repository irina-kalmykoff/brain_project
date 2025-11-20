import os
import glob
import numpy as np
import json
import h5py
import pickle
import traceback

import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional

from datetime import datetime
from collections import Counter, defaultdict

from scipy.interpolate import interp1d
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

#from diverse_models import SimplePhonemeModels
from acoustic_change_detector import AcousticChangeDetector
from custom_decoder import CustomBrainAudioDecoder
from markov_phoneme_model import MarkovPhonemeModel
from phonetic_dictionary import PhoneticDictionary
from phoneme_validator import PhonemeValidator
from debugger import DebugMixin



class UnifiedPhonemePipeline(CustomBrainAudioDecoder, DebugMixin):
    """
    Unified pipeline for phoneme processing with all steps integrated
    and full access to intermediate results.
    """
    
    def __init__(self, path_bids, path_output, path_results, 
                 feature_extraction_method='high_gamma',
                 unknown_keep_ratio=0.1, pca_components=100, 
                 use_phoneme_groups=True,
                 channel_correlation_threshold=0.1,
                 prioritize_regions=True,
                 channel_selection='all', #'motor'/'best_correlation'
                 use_rest_periods=False,
                 rest_normalization=True,
                 rest_as_class=False,
                 debug_mode=False, **kwargs):
        
        # Initialize CustomBrainAudioDecoder first
        CustomBrainAudioDecoder.__init__(self, path_bids, path_output, path_results, 
                                        debug_mode=debug_mode, **kwargs)
                                        
        # Then DebugMixin
        DebugMixin.__init__(self, class_name="UnifiedPhonemePipeline", debug_mode=debug_mode)

        
        self.path_bids = path_bids
        self.path_output = path_output
        self.path_results = path_results
        self.feature_extraction_method = feature_extraction_method
        self.unknown_keep_ratio = unknown_keep_ratio
        self.pca_components = pca_components
        self.use_phoneme_groups = use_phoneme_groups
        self.channel_correlation_threshold = channel_correlation_threshold
        self.prioritize_regions = prioritize_regions
        self.channel_selection = channel_selection
        self.use_rest_periods = use_rest_periods
        self.rest_normalization = rest_normalization
        self.rest_as_class = rest_as_class
        self.extraction_params = kwargs
        self.step_outputs = {}
        
        # Initialize phonetic dictionary once

        self.phonetic_dict = PhoneticDictionary(debug_mode=debug_mode)
        self.phonetic_dict.add_phoneme_groups()
        
        self.config = {
            'pca_components': pca_components,
            'feature_extraction_method': feature_extraction_method,
            'temporal_context': True,
            'standardize': True,
            'frameshift': 0.01,
            'win_length': 0.05,
            'use_rest_periods': use_rest_periods,  
            'rest_normalization': rest_normalization,
            'rest_as_class': rest_as_class
        }
        self.config.update(self.extraction_params)
        
        self.log(f"Pipeline initialized: {self.feature_extraction_method}, PCA={pca_components}, groups={use_phoneme_groups}")

    
    def step1_initialize_decoder(self):
        """Step 1: Initialize the custom decoder with PCA configuration"""

        
        # Create config dictionary for decoder
        config = {
            'pca_components': self.pca_components,
            'feature_extraction_method': self.feature_extraction_method,
            'temporal_context': True,
            'standardize': True,
            'use_rest_periods': self.use_rest_periods,  # ADD THIS
            'rest_normalization': self.rest_normalization,
            'rest_as_class': self.rest_as_class
        }
        
        self.custom_decoder = CustomBrainAudioDecoder(
            path_bids=self.path_bids,
            path_output=self.path_output,
            path_results=self.path_results,
            debug_mode=self.DEBUG_MODE,
            config=config  # Pass the config to the decoder
        )
        
        self.step_outputs['decoder'] = self.custom_decoder
        print(f"Step 1: Decoder initialized with PCA components={self.pca_components}, rest_periods={self.use_rest_periods}")
        return self.custom_decoder
    
    def step2_stratify_participants(self, channel_correlation_threshold= None, 
                                        prioritize_regions=None, 
                                       channel_selection=None):
        """
        Step 2: Filter participants and select channels.
        
        Two separate filters:
        1. Patient filter: Keep only patients with good electrode coverage
        2. Channel filter: Within selected patients, use only relevant channels
        """
        
        if channel_correlation_threshold is None:
            channel_correlation_threshold = self.channel_correlation_threshold
        if prioritize_regions is None:
            prioritize_regions = self.prioritize_regions        
        if channel_selection is None:
            channel_selection = getattr(self, 'channel_selection', 'all')
        
        # Basic quality stratification
        self.participant_strata = self.custom_decoder.stratify_participants_by_channel_quality(
            channel_correlation_threshold=channel_correlation_threshold
        )
        
        # Load channel analysis ONCE
        all_results_path = os.path.join(self.path_results, 'channel_analysis', 
                                       'all_participants_channel_correlations.npy')
        
        if os.path.exists(all_results_path):
            all_results = np.load(all_results_path, allow_pickle=True).item()
        else:
            self.log("Computing channel analysis...")
            all_results = self.custom_decoder.analyze_channels_across_participants(
                participant_ids=list(self.participant_strata.keys())
            )
        
        # STEP 1: Filter PARTICIPANTS if prioritize_regions=True
        if prioritize_regions:
            participant_scores = {}
            for participant_id in self.participant_strata:
                if participant_id in all_results:
                    score = self._score_participant_from_channel_results(all_results[participant_id])
                    participant_scores[participant_id] = score
                    self.log(f"{participant_id}: region score = {score:.3f}")
            
            # Keep only high-scoring participants
            min_region_score = 0.3
            self.participant_strata = {
                pid: strata for pid, strata in self.participant_strata.items()
                if participant_scores.get(pid, 0) >= min_region_score
            }
            self.log(f"Filtered to {len(self.participant_strata)} participants with good coverage")
        
        # STEP 2: Select CHANNELS within remaining participants
        if channel_selection != 'all':
            self.participant_channel_masks = {}
            
            for pid in self.participant_strata:  # Only for selected participants
                if pid not in all_results:
                    continue
                
                channels = all_results[pid]
                
                if channel_selection == 'motor':
                    # Select motor/frontal channels only
                    selected_indices = [
                        ch_data['index'] for ch_name, ch_data in channels.items()
                        if any(term in ch_data.get('region', '').lower() 
                               for term in ['central', 'frontal', 'precentral', 'postcentral'])
                    ]
                elif channel_selection == 'best_correlation':
                    # Select top N channels by correlation
                    sorted_channels = sorted(channels.items(), 
                                           key=lambda x: x[1].get('correlation', 0), 
                                           reverse=True)
                    selected_indices = [ch[1]['index'] for ch in sorted_channels[:50]]
                
                self.participant_channel_masks[pid] = selected_indices
                self.log(f"{pid}: Using {len(selected_indices)} {channel_selection} channels")
            
            # Pass to decoder
            if hasattr(self, 'custom_decoder'):
                self.custom_decoder.participant_channel_masks = self.participant_channel_masks
        
        self.step_outputs['participant_strata'] = self.participant_strata
        return self.participant_strata
    
    def step3_create_split(self, test_ratio=0.2, min_word_freq=1, random_seed=42):
        """Step 3: Create train/test split"""
        self.split_result = self.custom_decoder.create_stratified_cross_word_split(
            participant_strata=self.participant_strata,
            test_ratio=test_ratio,
            min_word_freq=min_word_freq,
            random_seed=random_seed
        )
        
        self.step_outputs['split_result'] = self.split_result
        self.log("Step 3: Train/test split created")
        return self.split_result
    
    def step4_initialize_detector(self):
        """Step 4: Initialize detector with PCA configuration"""
        
        self.detector = AcousticChangeDetector(
            min_segment_duration=0.03,
            max_segment_duration=0.3,
            distance_metric='cosine',
            smoothing_window=3,
            peak_threshold=0.5,
            decoder=self.custom_decoder,  # Passing the decoder with PCA config
            feature_extraction_method=self.feature_extraction_method,
            phonetic_dict=self.phonetic_dict if hasattr(self, 'phonetic_dict') else None,
            debug_mode=self.DEBUG_MODE
        )
        
        self.log(f"Setting up acoustic change detector with method: {self.feature_extraction_method}")
        # Only set split_result if it exists
        if hasattr(self, 'split_result'):
            self.detector.split_result = self.split_result
        else:
            self.log("Warning: No split_result found for detector")
        
        self.step_outputs['detector'] = self.detector
        
        self.debug("Step 4: Detector initialized")
        return self.detector
    
    def step5_accumulate_data(self, train_batches=5, test_batches=3, batch_size=100):
        """Step 5: Accumulate training and test data"""
        
        # Check if detector and split_result are available
        if not hasattr(self, 'detector') or self.detector is None:
            self.log("ERROR: Detector not initialized. Initializing now...")
            self.step4_initialize_detector()
        
        if not hasattr(self.detector, 'split_result') or self.detector.split_result is None:
            if hasattr(self, 'split_result'):
                self.detector.split_result = self.split_result
            else:
                self.log("ERROR: No split_result available for detector")
                return None, None
        
        # Check if decoder is properly set in detector
        if not hasattr(self.detector, 'decoder') or self.detector.decoder is None:
            if hasattr(self, 'custom_decoder'):
                self.detector.decoder = self.custom_decoder
            else:
                self.log("ERROR: No decoder available for detector")
                return None, None
        
        # Training data
        self.debug("Accumulating training data...")
        try:
            self.train = self.detector.accumulate_phoneme_data(
                num_batches=train_batches,
                batch_size=batch_size,
                feature_extraction_method=self.feature_extraction_method,
                batch_type='train'
            )
            
            if not self.train or 'features' not in self.train or not self.train['features']:
                self.log("ERROR: Training data accumulation failed - empty or missing features")
                return None, None
                
            self.log(f"Successfully accumulated {len(self.train['features'])} training samples")
            
        except Exception as e:
            self.log(f"Error accumulating training data: {e}")
            traceback.print_exc()
            return None, None
        
        # Test data
        self.debug("Accumulating test data...")
        try:
            self.test = self.detector.accumulate_phoneme_data(
                num_batches=test_batches,
                batch_size=batch_size,
                feature_extraction_method=self.feature_extraction_method,
                batch_type='test'
            )
            
            if not self.test or 'features' not in self.test or not self.test['features']:
                self.log("ERROR: Test data accumulation failed - empty or missing features")
                return self.train, None
                
            self.log(f"Successfully accumulated {len(self.test['features'])} test samples")
            
        except Exception as e:
            self.log(f"Error accumulating test data: {e}")
            traceback.print_exc()
            return self.train, None
        
        self.step_outputs['train_raw'] = self.train
        self.step_outputs['test_raw'] = self.test
        
        self.debug(f"Step 5: Data accumulated - {len(self.train['features'])} train, {len(self.test['features'])} test")
        return self.train, self.test
    
    def step6_resolve_unknowns(self):
        """Step 6: Initialize validator to resolve unknown phonemes"""        
        
        validator = PhonemeValidator(
            phonetic_dict=self.phonetic_dict,
            debug_mode=self.DEBUG_MODE
        )
        
        self.step_outputs['validator'] = self.validator
        self.log("Step 6: Validator initialized")
        
        # Resolve training data
        if self.train['phoneme_labels'].count('?') > 0:
            self.log(f"Resolving {self.train['phoneme_labels'].count('?')} unknown phonemes in training...")
            
            train_converted = {
                'phoneme_labels': self.train['phoneme_labels'],
                'phoneme_spectrogram_segments': self.train.get('spectrograms', []),
                'phoneme_words': self.train['phoneme_words'],
                'phoneme_positions': self.train.get('phoneme_positions', [0] * len(self.train['phoneme_labels'])),
                'phoneme_participant_ids': self.train.get('phoneme_participant_ids', ['unknown'] * len(self.train['phoneme_labels']))
            }
            
            resolved_train = self.validator.resolve_unknown_phonemes(train_converted)
            self.train['phoneme_labels'] = resolved_train['phoneme_labels']
            self.debug(f"Training after resolution: {self.train['phoneme_labels'].count('?')} unknown phonemes")
        
        # Resolve test data
        if self.test['phoneme_labels'].count('?') > 0:
            self.debug(f"Resolving {self.test['phoneme_labels'].count('?')} unknown phonemes in test...")
            
            test_converted = {
                'phoneme_labels': self.test['phoneme_labels'],
                'phoneme_spectrogram_segments': self.test.get('spectrograms', []),
                'phoneme_words': self.test['phoneme_words'],
                'phoneme_positions': self.test.get('phoneme_positions', [0] * len(self.test['phoneme_labels'])),
                'phoneme_participant_ids': self.test.get('phoneme_participant_ids', ['unknown'] * len(self.test['phoneme_labels']))
            }
            
            resolved_test = self.validator.resolve_unknown_phonemes(test_converted)
            self.test['phoneme_labels'] = resolved_test['phoneme_labels']
            self.log(f"Test after resolution: {self.test['phoneme_labels'].count('?')} unknown phonemes")
        
        
        # Add a check to warn but continue if there are still unknown phonemes
        unknown_train = self.train['phoneme_labels'].count('?')
        unknown_test = self.test['phoneme_labels'].count('?')
        
        if unknown_train > 0 or unknown_test > 0:
            self.log(f"WARNING: There are still {unknown_train} unknown phonemes in training and {unknown_test} in test.")
            self.log(f"The pipeline will continue, but some phonemes will be treated as the '?' class.")

        self.step_outputs['train_resolved'] = self.train
        self.step_outputs['test_resolved'] = self.test
        
        self.debug("Step 6: Unknown phonemes resolved")
        return self.train, self.test
    
    def checkpoint_after_step6(self):
        """Save a checkpoint after step 6 is completed."""
        
        # Validate that we have actual data before saving
        if not hasattr(self, 'train') or self.train is None:
            self.log("WARNING: No training data to checkpoint")
            return None
        
        if 'features' not in self.train or not self.train['features']:
            self.log("WARNING: Training data is empty, not saving checkpoint")
            return None
        
        # Create a timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create checkpoint filename
        filename = f"pipeline_{self.feature_extraction_method}_pca{self.pca_components}_after_step6_{timestamp}.pkl"
        filepath = os.path.join(self.path_results, filename)
        
        # Save the checkpoint
        self.log(f"Saving checkpoint after step 6: {filename}")
        
        try:
            # Save metadata and smaller components first
            metadata = {
                'method': self.feature_extraction_method,
                'pca_components': self.pca_components,
                'timestamp': timestamp,
                'stage': 'after_step6',
                'train_samples': len(self.train['features']) if self.train and 'features' in self.train else 0,
                'test_samples': len(self.test['features']) if self.test and 'features' in self.test else 0
            }
            
            # Only save if we have actual data
            if metadata['train_samples'] == 0:
                self.log("ERROR: Cannot save checkpoint with 0 training samples")
                return None
            
            # Save large data in separate files
            if self.train and self.train.get('features'):
                train_file = filepath.replace('.pkl', '_train.h5')
                self._save_data_to_h5(self.train, train_file)
                metadata['train_file'] = os.path.basename(train_file)
            
            if self.test and self.test.get('features'):
                test_file = filepath.replace('.pkl', '_test.h5')
                self._save_data_to_h5(self.test, test_file)
                metadata['test_file'] = os.path.basename(test_file)
            
            # Save metadata and other small objects
            with open(filepath, 'wb') as f:
                pickle.dump({
                    'metadata': metadata,
                    'validator': getattr(self, 'validator', None),
                    'phonetic_dict': getattr(self, 'phonetic_dict', None),
                }, f)
            
            self.log(f"Checkpoint saved: {filename} (train: {metadata['train_samples']}, test: {metadata['test_samples']} samples)")
            return filepath
        
        except Exception as e:
            self.log(f"Error saving checkpoint: {e}")
            return None

    def try_load_checkpoint(self):
        """
        Try to load a checkpoint for the current configuration.
        """
        
        # Look for checkpoint files matching the current configuration
        pattern = f"pipeline_{self.feature_extraction_method}_pca{self.pca_components}_after_step6_*.pkl"
        matching_files = glob.glob(os.path.join(self.path_results, pattern))
        
        if not matching_files:
            self.log(f"No checkpoint found for {self.feature_extraction_method} with PCA={self.pca_components}")
            return False
        
        for file in matching_files:
            file_size = os.path.getsize(file) / (1024 * 1024)  # Size in MB
            mod_time = datetime.fromtimestamp(os.path.getmtime(file)).strftime('%Y-%m-%d %H:%M:%S')
            self.log(f"  - {os.path.basename(file)}, Size: {file_size:.2f} MB, Modified: {mod_time}")
        
        # Sort by modification time (newest first)
        matching_files.sort(key=lambda f: os.path.getmtime(f), reverse=True)
        newest_checkpoint = matching_files[0]
        
        try:
            self.log(f"Loading checkpoint: {os.path.basename(newest_checkpoint)}")
            
            # Load checkpoint
            with open(newest_checkpoint, 'rb') as f:
                data = pickle.load(f)
            
            # Restore state
            metadata = data.get('metadata', {})
            self.feature_extraction_method = metadata.get('method', self.feature_extraction_method)
            self.pca_components = metadata.get('pca_components', self.pca_components)
            
            # Load large data from h5 files if available
            if 'train_file' in metadata:
                train_file = os.path.join(self.path_results, metadata['train_file'])
                self.train = self._load_data_from_h5(train_file)
                self.log(f"Loaded train data from {os.path.basename(train_file)}")
            elif 'train' in data:
                self.train = data['train']
            
            if 'test_file' in metadata:
                test_file = os.path.join(self.path_results, metadata['test_file'])
                self.test = self._load_data_from_h5(test_file)
                self.log(f"Loaded test data from {os.path.basename(test_file)}")
            elif 'test' in data:
                self.test = data['test']
            
            if 'validator' in data:
                self.validator = data['validator']
            
            self.log(f"Checkpoint loaded successfully: {os.path.basename(newest_checkpoint)}")
            return True
        except Exception as e:
            self.log(f"Error loading checkpoint: {e}")
            return False
        
    def run_step1_to_step6(self, force_reprocess=False):
        """
        Run steps 1-6 with checkpoint support.
        
        Parameters:
        -----------
        force_reprocess : bool
            If True, reprocess data even if checkpoint exists
        """
        # Check if we can load a checkpoint
        if not force_reprocess and self.try_load_checkpoint():
            self.log("Loaded checkpoint after step 6 - skipping steps 1-6")
            return self
        
        try:
            # Step 1: Initialize
            self.log("Step 1: Initializing")
            self.step1_initialize_decoder()
            
            # Step 2: Set up acoustic change detector
            self.log("Step 2: Setting up acoustic change detector")
            self.step2_stratify_participants()            
            
            # Step 3: Set up decoder
            self.log("Step 3: Setting up decoder")
            self.step3_create_split()
            
            # Step 4: Process batches
            self.log("Step 4: Processing batches")
            self.step4_initialize_detector()
            
            # Step 5: Accumulate data
            self.log("Step 5: Accumulating data")
            self.train, self.test = self.step5_accumulate_data()
            
            # Step 6: Validate phonemes and set up phonetic dictionary
            self.log("Step 6: Validating phonemes")
            self.step6_resolve_unknowns()
            
            # Verify that train and test were set properly
            if not hasattr(self, 'train') or self.train is None:
                self.log("ERROR: train data not set after step 5-6")
                raise ValueError("Train data not set properly")
                
            # Save checkpoint after step 6
            self.checkpoint_after_step6()
            
        except Exception as e:
            self.log(f"Error in steps 1-6: {str(e)}")
            traceback.print_exc()
            # Rethrow the error if critical
            if "Train data not set properly" in str(e):
                raise
        
        return self
    
    def step7_filter_unknowns(self, unknown_keep_ratio=None):
        """Step 7: Filter unknown phonemes from training data"""        

        if unknown_keep_ratio is None:
            unknown_keep_ratio = self.unknown_keep_ratio
        
        if not hasattr(self, 'train') or self.train is None:
            self.log("Warning: No training data available for filtering")
            return None
        
        # Simple filtering - work with phonemes, not groups
        filtered_features = []
        filtered_labels = []
        filtered_words = []
        filtered_participants = []
        
        for i, label in enumerate(self.train['phoneme_labels']):
            # Keep all known phonemes, subsample unknowns
            if label != '?' or np.random.random() < unknown_keep_ratio:
                filtered_features.append(self.train['features'][i])
                filtered_labels.append(label)
                
                if 'phoneme_words' in self.train:
                    filtered_words.append(self.train['phoneme_words'][i])
                if 'phoneme_participant_ids' in self.train:
                    filtered_participants.append(self.train['phoneme_participant_ids'][i])
        
        self.log(f"Filtered training: {len(filtered_features)} samples (from {len(self.train['features'])})")
        
        # Store filtered data with PHONEME labels
        self.train_filtered = {
            'features': filtered_features,
            'phoneme_labels': filtered_labels  # Still phonemes!
        }
        
        if filtered_words:
            self.train_filtered['phoneme_words'] = filtered_words
        if filtered_participants:
            self.train_filtered['phoneme_participant_ids'] = filtered_participants
        
        return self.train_filtered
    
    def step8_convert_to_groups(self):
        """Step 8: Convert phonemes to groups if use_phoneme_groups is True"""
        if not self.use_phoneme_groups:
            self.log("Step 8: Skipping group conversion (use_phoneme_groups=False)")
            return
        
        self.log("Step 8: Converting phonemes to groups...")
        
        # ONE conversion method used here
        if hasattr(self, 'train') and self.train:
            self.train = self._convert_labels_to_groups(self.train)
        
        if hasattr(self, 'test') and self.test:
            self.test = self._convert_labels_to_groups(self.test)
        
        if hasattr(self, 'train_filtered') and self.train_filtered:
            self.train_filtered = self._convert_labels_to_groups(self.train_filtered)
        
        # Log the results
        if self.train:
            from collections import Counter
            train_groups = Counter(self.train['phoneme_labels'])
            self.log(f"Train data: {len(train_groups)} groups, {dict(train_groups)}")
        
        self.log("Step 8: Conversion to groups complete")
        return self
    
    # for neural networks in diverse models class:
    def run_steps_4_to_7(self):
        """Run only steps 4-7 (method-specific steps)."""
        try:
            # Check if required objects from steps 1-3 exist
            if not hasattr(self, 'custom_decoder') or self.custom_decoder is None:
                self.log("ERROR: Decoder not initialized. Initializing now...")
                self.step1_initialize_decoder()
                
            if not hasattr(self, 'participant_strata') or self.participant_strata is None:
                self.log("ERROR: Participants not stratified. Running step 2...")
                self.step2_stratify_participants()
                
            if not hasattr(self, 'split_result') or self.split_result is None:
                self.log("ERROR: Train/test split not created. Running step 3...")
                self.step3_create_split()
            
            # Step 4: Initialize detector
            self.log("Step 4: Initializing detector")
            self.step4_initialize_detector()
            
            # Step 5: Accumulate data
            self.log("Step 5: Accumulating data")
            train_data, test_data = self.step5_accumulate_data()
            
            # Check if data accumulation was successful
            if train_data is None or test_data is None:
                self.log("ERROR: Data accumulation failed in step 5")
                return False
            
            # Store the data properly
            self.train = train_data
            self.test = test_data
            
            # Step 6: Resolve unknowns
            self.log("Step 6: Resolving unknowns")
            self.step6_resolve_unknowns()
            
            # Save checkpoint after step 6
            try:
                self.checkpoint_after_step6()
            except MemoryError:
                self.log("Memory error during checkpoint - continuing without saving")
            except Exception as e:
                self.log(f"Error saving checkpoint: {e}")
            
            # Step 7: Filter unknowns
            self.log("Step 7: Filtering unknowns")
            self.step7_filter_unknowns()
            
            self.log("Steps 4-7 completed successfully")
            return True
        
        except Exception as e:
            self.log(f"Error in steps 4-7: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_step7_and_beyond(self):
        """Run steps 7 and beyond."""
        # Step 7: Resolve unknown phonemes
        self.log("Step 7: Resolving unknown phonemes")
        self.step7_filter_unknowns()
        self.step8_convert_to_groups()
        
        return self
    
    def run_all_steps(self):
        """Run all steps in sequence with detailed error handling"""
        print("="*60)
        print(f"RUNNING COMPLETE PIPELINE FOR {self.feature_extraction_method}")
        print("="*60)
        
        # Step 1-6
        try:
            self.log("Running steps 1-6...")
            self.run_step1_to_step6()
            
            # Verify data was properly set
            if not hasattr(self, 'train') or self.train is None:
                self.log("ERROR: train data not available after steps 1-6")
                raise ValueError("Steps 1-6 did not properly set train data")
                
            # Print data statistics for debugging
            if hasattr(self, 'train') and 'features' in self.train:
                train_samples = len(self.train['features'])
                unique_phonemes = len(set(self.train['phoneme_labels'])) if 'phoneme_labels' in self.train else 0
                self.log(f"Train data: {train_samples} samples")
                self.log(f"Unique phonemes: {unique_phonemes}")
            else:
                self.log("WARNING: train data structure is incomplete")
                
        except MemoryError as e:
            self.log(f"Memory error in steps 1-6: {e}")
            self.log("Continuing without saving checkpoint...")
        except Exception as e:
            import traceback
            self.log(f"Error in steps 1-6: {e}")
            traceback.print_exc()
            print("\n" + "="*60)
            print("PIPELINE FAILED IN STEPS 1-6 - See logs for details")
            print("="*60)
            return self
        
        # Step 7 and beyond
        try:
            self.log("Running step 7 and beyond...")
            self.run_step7_and_beyond()
            success = True
        except Exception as e:
            import traceback
            self.log(f"Error in step 7 and beyond: {e}")
            traceback.print_exc()
            success = False
        
        print("\n" + "="*60)
        if success:
            print("PIPELINE COMPLETE")
        else:
            print("PIPELINE FAILED - See logs for details")
        print("="*60)
        
        return self
    
    # Add method to change PCA components
    def set_pca_components(self, pca_components):
        """Update PCA components value and propagate to decoder if initialized"""
        self.pca_components = pca_components
        
        if hasattr(self, 'custom_decoder') and self.custom_decoder is not None:
            self.custom_decoder.update_config(pca_components=pca_components)
            self.log(f"Updated PCA components to {pca_components} in decoder")
            
            # Ensure the detector has the updated decoder reference
            if hasattr(self, 'detector') and self.detector is not None:
                # Reset the detector to use the updated decoder
                self.detector.decoder = self.custom_decoder
                self.log(f"Updated detector with new PCA config")
        
        self.log(f"Set pipeline PCA components to {pca_components}")
        return self
    
    def optimize_pca_components(self, component_values=[10, 20, 30, 40, 50], 
                          train_batches=2, test_batches=1):
        """Find optimal PCA components by testing different values"""

        
        self.log(f"Optimizing PCA components with values: {component_values}")
        
        results = {}
        
        # Step 1: Initialize decoder
        self.step1_initialize_decoder()
        
        # Step 2-3: Setup data split
        self.step2_stratify_participants()
        self.step3_create_split()
        
        for n_components in component_values:
            self.log(f"Testing with {n_components} PCA components...")
            
            # Update PCA components in decoder
            self.set_pca_components(n_components)
            
            # Initialize detector with updated decoder
            self.step4_initialize_detector()
            
            # Accumulate data with this PCA setting
            train_data, test_data = self.step5_accumulate_data(
                train_batches=train_batches,
                test_batches=test_batches
            )
            
            # Process data for classification
            # Find the feature with the smallest flattened length
            min_length = min(f.size for f in train_data['features'])
            
            # Create X_train by truncating or padding features to the same length
            X_train = []
            for f in train_data['features']:
                # Flatten the feature
                flat_f = f.flatten()
                
                # Truncate or pad to min_length
                if flat_f.size > min_length:
                    X_train.append(flat_f[:min_length])
                elif flat_f.size < min_length:
                    padded = np.zeros(min_length)
                    padded[:flat_f.size] = flat_f
                    X_train.append(padded)
                else:
                    X_train.append(flat_f)
                    
            X_train = np.array(X_train)
            y_train = train_data['phoneme_labels']
            
            # Do the same for test data
            min_length_test = min(f.size for f in test_data['features'])
            # Use the smaller of the two minimum lengths
            min_length = min(min_length, min_length_test)
            
            X_test = []
            for f in test_data['features']:
                flat_f = f.flatten()
                if flat_f.size > min_length:
                    X_test.append(flat_f[:min_length])
                elif flat_f.size < min_length:
                    padded = np.zeros(min_length)
                    padded[:flat_f.size] = flat_f
                    X_test.append(padded)
                else:
                    X_test.append(flat_f)
                    
            X_test = np.array(X_test)
            y_test = test_data['phoneme_labels']
            
            self.log(f"Prepared feature arrays: X_train shape={X_train.shape}, X_test shape={X_test.shape}")
            
            # Train a simple model
            try:
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                
                # Evaluate
                train_acc = model.score(X_train, y_train)
                test_acc = model.score(X_test, y_test)
                
                results[n_components] = {
                    'train_acc': train_acc,
                    'test_acc': test_acc
                }
                
                self.log(f"PCA={n_components}: train={train_acc:.4f}, test={test_acc:.4f}")
            except Exception as e:
                self.log(f"Error training model with {n_components} components: {e}")
                results[n_components] = {
                    'train_acc': 0.0,
                    'test_acc': 0.0
                }
        
        # Find best component value (defaulting to the first one if all failed)
        if results:
            best_components = max(results, key=lambda x: results[x]['test_acc'])
            
            # Set to best value
            self.set_pca_components(best_components)
            
            self.log(f"Best PCA components: {best_components} "
                    f"(train: {results[best_components]['train_acc']:.4f}, "
                    f"test: {results[best_components]['test_acc']:.4f})")
        else:
            self.log("No successful PCA component configurations found")
            best_components = component_values[0]  # Default to first value
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = os.path.join(self.path_results, f"pca_optimization_{timestamp}.json")
        
        with open(results_path, 'w') as f:
            json.dump({
                'results': {str(k): v for k, v in results.items()},
                'best_components': best_components
            }, f, indent=2)
        
        return best_components, results
            
    def get_training_data(self, filtered=True):
        """Get training data - already converted if needed"""
        if filtered and hasattr(self, 'train_filtered'):
            return self.train_filtered
        return self.train

    def get_test_data(self):
        """Get test data - already converted if needed"""
        return self.test
            
    def balance_training_data(self, train_data, balance_strategy='oversample', group_weights=None):
        """
        Balance training data across labels (works for both phonemes and groups).
        """
       
        # Check what kind of labels we're working with
        sample_labels = train_data['phoneme_labels'][:5] if train_data['phoneme_labels'] else []
        groups_list = {'front_vowels', 'back_vowels', 'labial', 'alveolar', 
                       'alveolar plosive', 'alveolar fricative', 'alveolar other',
                       'palatal', 'dorsal', 'glottal', 'unknown', 'marker'}
        
        # Determine if we're working with groups or phonemes
        working_with_groups = any(label in groups_list for label in sample_labels)
        
        # Organize data by labels
        label_to_indices = {}
        
        if working_with_groups:
            # Already groups - use labels directly
            self.log("Working with group labels")
            for i, label in enumerate(train_data['phoneme_labels']):
                if label not in label_to_indices:
                    label_to_indices[label] = []
                label_to_indices[label].append(i)
        else:
            # Phonemes - map to groups for balancing
            self.log("Working with phoneme labels, mapping to groups for balancing")
            if not hasattr(self.phonetic_dict, 'phoneme_to_group'):
                self.phonetic_dict.add_phoneme_groups()
                
            for i, label in enumerate(train_data['phoneme_labels']):
                group = self.phonetic_dict.phoneme_to_group.get(label, 'unknown')
                if group not in label_to_indices:
                    label_to_indices[group] = []
                label_to_indices[group].append(i)
        
        # Count samples per label/group
        label_counts = {label: len(indices) for label, indices in label_to_indices.items()}
        self.log(f"Original distribution: {label_counts}")
        
        # Filter out empty groups
        label_to_indices = {k: v for k, v in label_to_indices.items() if v}
        
        if not label_to_indices:
            self.log("ERROR: No valid data found!")
            return train_data
        
        # Balance the data
        balanced_indices = []
        
        if balance_strategy == 'oversample':
            # Find the maximum group size
            max_count = max(len(indices) for indices in label_to_indices.values())
            
            for label, indices in label_to_indices.items():
                # Oversample to match max_count
                sampled = list(indices)
                while len(sampled) < max_count:
                    n_needed = min(len(indices), max_count - len(sampled))
                    sampled.extend(np.random.choice(indices, n_needed, replace=True).tolist())
                balanced_indices.extend(sampled[:max_count])
                
        elif balance_strategy == 'undersample':
            # Find the minimum group size
            min_count = min(len(indices) for indices in label_to_indices.values())
            
            for label, indices in label_to_indices.items():
                # Randomly sample min_count items
                sampled = np.random.choice(indices, min(min_count, len(indices)), replace=False)
                balanced_indices.extend(sampled.tolist())
                
        elif balance_strategy == 'weighted':
            # Use custom weights or default weights
            if group_weights is None:
                # Default weights for groups
                group_weights = {
                    'front_vowels': 1.0,
                    'back_vowels': 1.0,
                    'labial': 2.0,
                    'alveolar': 2.0,
                    'alveolar plosive': 2.0,
                    'alveolar fricative': 2.0,
                    'alveolar other': 2.0,
                    'palatal': 2.5,
                    'dorsal': 2.5,
                    'glottal': 3.0,
                    'unknown': 0.5,
                    'marker': 0.5
                }
            
            # Calculate target count
            target_count = int(np.mean([len(indices) for indices in label_to_indices.values()]))
            
            for label, indices in label_to_indices.items():
                weight = group_weights.get(label, 1.0)
                target = max(1, int(target_count * weight))
                
                if len(indices) < target:
                    # Oversample
                    sampled = list(indices)
                    while len(sampled) < target:
                        n_needed = min(len(indices), target - len(sampled))
                        sampled.extend(np.random.choice(indices, n_needed, replace=True).tolist())
                    balanced_indices.extend(sampled[:target])
                else:
                    # Undersample
                    sampled = np.random.choice(indices, min(target, len(indices)), replace=False)
                    balanced_indices.extend(sampled.tolist())
        
        else:
            # No balancing - return as is
            return train_data
        
        # Shuffle the balanced indices
        np.random.shuffle(balanced_indices)
        
        # Create balanced dataset
        balanced_data = {
            'features': [train_data['features'][i] for i in balanced_indices],
            'phoneme_labels': [train_data['phoneme_labels'][i] for i in balanced_indices]
        }
        
        # Add optional fields if they exist
        for field in ['phoneme_words', 'phoneme_participant_ids']:
            if field in train_data:
                balanced_data[field] = [train_data[field][i] for i in balanced_indices]
        
        # Log new distribution
        new_counts = Counter(balanced_data['phoneme_labels'])
        self.log(f"Balanced distribution: {dict(new_counts)}")
        
        return balanced_data
    
    def convert_data_to_groups(self, data):
        """One-time conversion to groups"""
        if not data or 'phoneme_labels' not in data:
            return data
            
        converted = data.copy()
        new_labels = []
        
        for label in data['phoneme_labels']:
            # Use the phonetic dictionary mapping
            group = self.phonetic_dict.phoneme_to_group.get(label, 'unknown')
            new_labels.append(group)
        
        converted['phoneme_labels'] = new_labels
        self.log(f"Converted {len(set(data['phoneme_labels']))} phonemes to {len(set(new_labels))} groups")
        
        return converted
    
    def _convert_labels_to_groups(self, data):
        """THE ONLY conversion method - converts phoneme labels to group labels"""
        if not data or 'phoneme_labels' not in data:
            return data
        
        converted = data.copy()
        new_labels = []
        unmapped = set()
        
        for label in data['phoneme_labels']:
            group = self.phonetic_dict.phoneme_to_group.get(label, 'unknown')
            if group == 'unknown' and label != '?':
                unmapped.add(label)
            new_labels.append(group)
        
        if unmapped:
            self.log(f"Unmapped phonemes: {unmapped}")
        
        converted['phoneme_labels'] = new_labels
        return converted
        
    def _score_participant_from_channel_results(self, channel_results):
        """
        Score participant based on their channel analysis results.
        Uses both correlation values and region information.
        """
        region_weights = {
            'precentral': 1.0,
            'postcentral': 0.8,
            'frontal': 0.7,
            'central': 0.7,
            'subcentral': 0.6,
            'insula': 0.6,
            'parietal': 0.5,
            'temporal': 0.3,
            'white-matter': 0.1,
            'unknown': 0.1
        }
        
        total_score = 0
        total_channels = 0
        
        for channel_name, channel_data in channel_results.items():
            correlation = channel_data.get('correlation', 0)
            region = channel_data.get('region', 'Unknown').lower()
            
            # Skip NaN correlations
            if np.isnan(correlation):
                continue
            
            # Find best matching region weight
            region_score = 0.1  # Default
            for key, weight in region_weights.items():
                if key in region:
                    region_score = max(region_score, weight)
            
            # Combine correlation and region information
            # High correlation in motor regions is best
            # High correlation in temporal regions is less useful
            channel_score = correlation * region_score
            
            total_score += channel_score
            total_channels += 1
        
        return total_score / total_channels if total_channels > 0 else 0
        
    def check_step(self, step_name):
        """Check the output of any step"""
        if step_name in self.step_outputs:
            output = self.step_outputs[step_name]
            print(f"\n=== Checking {step_name} ===")
            
            if isinstance(output, dict):
                if 'features' in output:
                    print(f"  Features: {len(output['features'])} samples")
                    print(f"  First feature shape: {output['features'][0].shape}")
                if 'phoneme_labels' in output:
                    print(f"  Labels: {len(output['phoneme_labels'])} total")
                    label_counts = Counter(output['phoneme_labels'])
                    print(f"  Unique labels: {len(label_counts)}")
                    print(f"  Top 5 labels: {label_counts.most_common(5)}")
                    print(f"  Unknown count: {output['phoneme_labels'].count('?')}")
            else:
                print(f"  Type: {type(output)}")
            
            return output
        else:
            print(f"Step '{step_name}' not found. Available: {list(self.step_outputs.keys())}")
            return None    

        
        if method == 'high_gamma':
            # Your existing high gamma extraction
            features = extractHG(eeg, eeg_sr, **kwargs)
            return features
        
        elif method == 'multi_band':
            # Extract power in multiple frequency bands
            # Delta (1-4 Hz), Theta (4-8 Hz), Alpha (8-13 Hz), Beta (13-30 Hz), Gamma (30-70 Hz), High Gamma (70-150 Hz)
            bands = kwargs.get('bands', [(1, 4), (4, 8), (8, 13), (13, 30), (30, 70), (70, 150)])
            
            # Get window parameters
            win_length = kwargs.get('win_length', self.win_length)
            frameshift = kwargs.get('frameshift', self.frameshift)
            
            # Extract features for each band
            all_features = []
            for band_low, band_high in bands:
                # Apply bandpass filter
                from scipy.signal import butter, filtfilt
                
                # Design filter
                nyquist = eeg_sr / 2
                low = band_low / nyquist
                high = band_high / nyquist
                
                # Apply filter using filtfilt for zero-phase filtering
                b, a = butter(4, [low, high], btype='band')
                filtered = filtfilt(b, a, eeg, axis=0)
                
                # Extract power using short-time windows
                from scipy.signal import spectrogram
                
                # Convert window length from seconds to samples
                win_samples = int(win_length * eeg_sr)
                shift_samples = int(frameshift * eeg_sr)
                
                # Initialize feature array
                n_frames = 1 + (filtered.shape[0] - win_samples) // shift_samples
                features = np.zeros((n_frames, filtered.shape[1]))
                
                # Extract power in each window
                for i in range(n_frames):
                    start = i * shift_samples
                    end = start + win_samples
                    window = filtered[start:end, :]
                    # Calculate power (mean squared amplitude)
                    features[i, :] = np.mean(window ** 2, axis=0)
                
                all_features.append(features)
            
            # Concatenate features from all bands
            multi_band_features = np.concatenate(all_features, axis=1)
            return multi_band_features
        
        elif method == 'spectral':
            # Extract detailed spectral features
            # Use spectrogram and extract frequency domain statistics
            
            # Get window parameters
            win_length = kwargs.get('win_length', self.win_length)
            frameshift = kwargs.get('frameshift', self.frameshift)
            n_fft = kwargs.get('n_fft', 256)  # FFT size
            
            # Convert window length from seconds to samples
            win_samples = int(win_length * eeg_sr)
            shift_samples = int(frameshift * eeg_sr)
            
            # Number of frequency bins to keep (low frequencies are more important)
            n_freqs = kwargs.get('n_freqs', 30)
            
            # Initialize feature array
            n_frames = 1 + (eeg.shape[0] - win_samples) // shift_samples
            n_channels = eeg.shape[1]
            features = np.zeros((n_frames, n_freqs * n_channels))
            
            # Compute spectrogram for each channel
            for ch in range(n_channels):
                channel_data = eeg[:, ch]
                
                # Compute spectrogram
                from scipy.signal import spectrogram
                freqs, times, spec = spectrogram(
                    channel_data, 
                    fs=eeg_sr, 
                    nperseg=win_samples,
                    noverlap=win_samples - shift_samples,
                    nfft=n_fft,
                    return_onesided=True
                )
                
                actual_freqs = spec.shape[0]
                if actual_freqs < n_freqs:
                    print(f"Warning: Only {actual_freqs} frequency bins available, requested {n_freqs}")
                    n_freqs = actual_freqs  # Adjust to available frequencies

                
                # Keep only the first n_freqs frequency bins
                spec = spec[:n_freqs, :]
                
                # Convert to dB scale
                spec_db = 10 * np.log10(spec + 1e-10)
                
                # Reshape to (time, freq) for this channel
                ch_features = spec_db.T
                
                # Store in the feature array
                features[:, ch * n_freqs:(ch + 1) * n_freqs] = ch_features
            
            return features
        
        elif method == 'wavelet':
            # Wavelet-based time-frequency analysis
            # Extract wavelet coefficients at multiple scales
            
            import pywt
            
            # Get parameters
            wavelet = kwargs.get('wavelet', 'db4')  # Wavelet type
            scales = kwargs.get('scales', [2, 4, 8, 16, 32])  # Wavelet scales
            
            # Convert window length from seconds to samples
            win_length = kwargs.get('win_length', self.win_length)
            frameshift = kwargs.get('frameshift', self.frameshift)
            win_samples = int(win_length * eeg_sr)
            shift_samples = int(frameshift * eeg_sr)
            
            # Initialize feature array
            n_frames = 1 + (eeg.shape[0] - win_samples) // shift_samples
            n_channels = eeg.shape[1]
            n_scales = len(scales)
            features = np.zeros((n_frames, n_channels * n_scales))
            
            # Process each channel
            for ch in range(n_channels):
                channel_data = eeg[:, ch]
                
                # Compute continuous wavelet transform
                coefs = []
                for scale in scales:
                    # Compute CWT at this scale
                    coef, _ = pywt.cwt(channel_data, [scale], wavelet)
                    # Get magnitude of coefficients
                    coef_mag = np.abs(coef[0])
                    coefs.append(coef_mag)
                
                # Extract windowed features
                for i in range(n_frames):
                    start = i * shift_samples
                    end = start + win_samples
                    
                    # Get wavelet power in this window for each scale
                    for s, scale in enumerate(scales):
                        scale_power = np.mean(coefs[s][start:end] ** 2)
                        features[i, ch * n_scales + s] = scale_power
            
            return features
        
        elif method == 'mfcc':
            # MFCC-inspired features for neural signals
            # Similar to speech processing but adapted for brain signals
            
            from python_speech_features import mfcc
            
            # Get parameters
            n_mfcc = kwargs.get('n_mfcc', 13)  # Number of coefficients
            
            # Window parameters
            win_length = kwargs.get('win_length', self.win_length)
            frameshift = kwargs.get('frameshift', self.frameshift)
            
            # Initialize feature array
            n_channels = eeg.shape[1]
            
            # Process each channel
            all_features = []
            for ch in range(n_channels):
                channel_data = eeg[:, ch]
                
                # Compute MFCC features
                # Note: We're applying speech processing to brain signals as an experiment
                # The frequency range is different, but the principle is similar
                mfcc_feat = mfcc(
                    channel_data, 
                    samplerate=eeg_sr, 
                    winlen=win_length,
                    winstep=frameshift,
                    numcep=n_mfcc,
                    nfilt=26,  # Number of filters
                    nfft=512,  # FFT size
                    lowfreq=1,  # Lowest frequency
                    highfreq=eeg_sr/2 - 1  # Highest frequency
                )
                
                all_features.append(mfcc_feat)
            
            # Combine features from all channels
            # Different approach: keep channels separate in time, concatenate in feature dimension
            n_frames = all_features[0].shape[0]
            combined_features = np.zeros((n_frames, n_channels * n_mfcc))
            
            for ch in range(n_channels):
                combined_features[:, ch * n_mfcc:(ch + 1) * n_mfcc] = all_features[ch]
            
            return combined_features
        
        else:
            raise ValueError(f"Unknown feature extraction method: {method}")
    
    def save(self):
        """Save pipeline data with feature extraction method in filename."""
        
        # Create a timestamp for versioning
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Filename includes extraction method, PCA components, and timestamp
        filename = f'pipeline_{self.feature_extraction_method}_pca{self.pca_components}_{timestamp}.pkl'
        filepath = os.path.join(self.path_results, filename)
        
        # Prepare metadata with detailed information
        metadata = {
            'method': self.feature_extraction_method,
            'pca_components': self.pca_components,
            'timestamp': timestamp,
            'created_at': datetime.now().isoformat(),
            'config': self.config if hasattr(self, 'config') else None,
            # Add statistics about the data
            'train_samples': len(self.train['features']) if hasattr(self, 'train') and 'features' in self.train else 0,
            'test_samples': len(self.test['features']) if hasattr(self, 'test') and 'features' in self.test else 0,
            'filtered_samples': len(self.train_filtered['features']) if hasattr(self, 'train_filtered') and 'features' in self.train_filtered else 0,
            'unique_phonemes': len(set(self.train['phoneme_labels'])) if hasattr(self, 'train') and 'phoneme_labels' in self.train else 0
        }
        
        # Save data with metadata
        with open(filepath, 'wb') as f:
            pickle.dump({
                'metadata': metadata,
                'train': getattr(self, 'train', None),
                'test': getattr(self, 'test', None),
                'train_filtered': getattr(self, 'train_filtered', None)
            }, f)
        
        # Also save a JSON metadata file for easy inspection
        metadata_file = filepath.replace('.pkl', '_metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        self.log(f"Saved: {filename}")
        self.log(f"Method: {self.feature_extraction_method}, PCA: {self.pca_components}")
        self.log(f"Train: {metadata['train_samples']} samples, Test: {metadata['test_samples']} samples")
        
        return filepath

    def load(self, filepath):
        """Load pipeline data."""
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        # Handle both old and new format files
        if 'metadata' in data:
            # New format with metadata
            metadata = data['metadata']
            self.feature_extraction_method = metadata['method']
            self.pca_components = metadata['pca_components']
            if hasattr(self, 'config') and metadata.get('config'):
                self.config.update(metadata['config'])
        else:
            # Old format without metadata
            self.feature_extraction_method = data.get('method', 'high_gamma')
            self.pca_components = data.get('pca_components', 50)
        
        # Load data
        self.train = data.get('train', None)
        self.test = data.get('test', None)
        self.train_filtered = data.get('train_filtered', None)
        
        # Print summary
        train_samples = len(self.train['features']) if self.train and 'features' in self.train else 0
        test_samples = len(self.test['features']) if self.test and 'features' in self.test else 0
        
        self.log(f"Loaded {self.feature_extraction_method} pipeline with PCA components={self.pca_components}")
        self.log(f"Train: {train_samples} samples, Test: {test_samples} samples")
        
        return self

    @staticmethod
    def list_saved(directory):
        """List saved pipelines by extraction method and PCA components with detailed info."""
        
        files = [f for f in os.listdir(directory) if f.startswith('pipeline_') and f.endswith('.pkl')]
        
        if not files:
            print("No saved pipeline files found.")
            return
        
        # Format as a table
        print("\nSaved Pipelines:")
        print("-" * 80)
        print(f"{'Filename':<40} {'Method':<10} {'PCA':<5} {'Train':<6} {'Test':<6} {'Date':<19}")
        print("-" * 80)
        
        for f in sorted(files, key=lambda x: os.path.getmtime(os.path.join(directory, x)), reverse=True):
            try:
                with open(os.path.join(directory, f), 'rb') as file:
                    data = pickle.load(file)
                    
                    # Handle both old and new format files
                    if 'metadata' in data:
                        # New format with metadata
                        metadata = data['metadata']
                        method = metadata.get('method', 'unknown')
                        pca = metadata.get('pca_components', 'unknown')
                        train_size = metadata.get('train_samples', 0)
                        test_size = metadata.get('test_samples', 0)
                        timestamp = metadata.get('created_at', 'unknown')
                        if timestamp != 'unknown':
                            try:
                                timestamp = datetime.fromisoformat(timestamp).strftime('%Y-%m-%d %H:%M')
                            except:
                                pass
                    else:
                        # Old format
                        method = data.get('method', 'unknown')
                        pca = data.get('pca_components', 'unknown')
                        train_size = len(data['train']['features']) if data.get('train') else 0
                        test_size = len(data['test']['features']) if data.get('test') else 0
                        timestamp = datetime.fromtimestamp(os.path.getmtime(os.path.join(directory, f))).strftime('%Y-%m-%d %H:%M')
                    
                    print(f"{f:<40} {method:<10} {pca:<5} {train_size:<6} {test_size:<6} {timestamp:<19}")
            except Exception as e:
                print(f"{f:<40} Error reading: {str(e)}")
        
        print("-" * 80)

    @staticmethod
    def load_saved(directory, method='high_gamma', pca_components=None, newest=True):
        
        # Find all matching files
        matching_files = []
        
        for f in os.listdir(directory):
            if not (f.startswith('pipeline_') and f.endswith('.pkl')):
                continue
                
            # Check if method matches
            if f'pipeline_{method}' not in f:
                continue
                
            # Check if PCA components match (if specified)
            if pca_components is not None and f'_pca{pca_components}_' not in f:
                continue
                
            matching_files.append(f)
        
        if not matching_files:
            print(f"No matching pipelines found for method={method}, pca_components={pca_components}")
            raise FileNotFoundError(f"No matching pipeline files found")
        
        if newest:
            # Sort by modification time (newest first)
            matching_files.sort(key=lambda f: os.path.getmtime(os.path.join(directory, f)), reverse=True)
            selected_file = matching_files[0]
            print(f"Loading newest matching pipeline: {selected_file}")
        else:
            # Try to find the one with highest test accuracy
            best_accuracy = -1
            best_file = None
            
            for f in matching_files:
                try:
                    with open(os.path.join(directory, f), 'rb') as file:
                        data = pickle.load(file)
                        
                        # TODO: Implement logic to determine accuracy from saved data
                        # For now, just use newest as fallback
                        best_file = f
                        break
                except:
                    continue
            
            if best_file:
                selected_file = best_file
                print(f"Loading best performing pipeline: {selected_file}")
            else:
                # Fallback to newest
                matching_files.sort(key=lambda f: os.path.getmtime(os.path.join(directory, f)), reverse=True)
                selected_file = matching_files[0]
                print(f"Loading newest matching pipeline: {selected_file}")
        
        # Load the selected pipeline
        filepath = os.path.join(directory, selected_file)
        
        pipeline = UnifiedPhonemePipeline(directory, directory, directory)
        pipeline.load(filepath)
        
        return pipeline
        
    def visualize_model_results(self, model, eval_results, title_prefix="Model", 
                           save_dir=None, show_plot=True, figsize=(7, 5)):
        """
        Visualize model results including transition matrix and confusion matrix.
        
        """
        
        # Use specified save directory, or model's output directory, or pipeline's results directory
        if save_dir is None:
            save_dir = getattr(model, 'output_dir', self.path_results)
        
        # Ensure save directory exists
        os.makedirs(save_dir, exist_ok=True)
        
        saved_files = {}
        timestamp = getattr(self, '_viz_timestamp', None)
        if timestamp is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._viz_timestamp = timestamp  # Store for consistent naming
        
        # Create a subdirectory for this visualization session
        viz_dir = os.path.join(save_dir, f"model_viz_{timestamp}")
        os.makedirs(viz_dir, exist_ok=True)
        
        self.debug(f"Visualizing model results for {title_prefix} in {viz_dir}")
        
        # Visualize transition matrix if model has the method
        if hasattr(model, 'visualize_transitions'):
            try:
                transition_path = model.visualize_transitions(
                    save_dir=viz_dir,
                    title_prefix=title_prefix
                )
                saved_files['transition_matrix'] = transition_path
                self.log(f"Transition matrix saved to {transition_path}")
            except Exception as e:
                self.log(f"Error visualizing transitions: {e}")
        
        # Plot confusion matrix if available
        if eval_results is not None and 'confusion_matrix' in eval_results:
            try:
                conf_matrix = eval_results['confusion_matrix']
                
                # Get labels from model if available
                if hasattr(model, 'group_encoder') and hasattr(model.group_encoder, 'classes_'):
                    labels = list(model.group_encoder.classes_)
                elif hasattr(model, 'label_encoder') and hasattr(model.label_encoder, 'classes_'):
                    labels = list(model.label_encoder.classes_)
                elif 'labels' in eval_results:
                    labels = eval_results['labels']
                else:
                    labels = [str(i) for i in range(conf_matrix.shape[0])]
                
                plt.figure(figsize=figsize)
                plt.imshow(conf_matrix, interpolation='nearest', cmap='Blues')
                plt.title(f'{title_prefix} Confusion Matrix')
                plt.colorbar()
                
                # Handle large number of labels by skipping some
                n_labels = len(labels)
                if n_labels > 20:
                    # Show a subset of labels (max 20)
                    step = max(1, n_labels // 20)
                    tick_marks = np.arange(0, n_labels, step)
                    plt.xticks(tick_marks, [labels[i] for i in tick_marks], rotation=45)
                    plt.yticks(tick_marks, [labels[i] for i in tick_marks])
                else:
                    tick_marks = np.arange(len(labels))
                    plt.xticks(tick_marks, labels, rotation=45)
                    plt.yticks(tick_marks, labels)
                
                # Add text annotations if matrix is not too large
                if conf_matrix.shape[0] <= 20 and conf_matrix.shape[1] <= 20:
                    thresh = conf_matrix.max() / 2.
                    for i in range(conf_matrix.shape[0]):
                        for j in range(conf_matrix.shape[1]):
                            plt.text(j, i, format(conf_matrix[i, j], 'd'),
                                    ha="center", va="center",
                                    color="white" if conf_matrix[i, j] > thresh else "black")
                
                plt.tight_layout()
                plt.ylabel('True Label')
                plt.xlabel('Predicted Label')
                
                # Save figure
                confusion_path = os.path.join(viz_dir, f'{title_prefix.lower().replace(" ", "_")}_confusion_matrix.png')
                plt.savefig(confusion_path, dpi=300, bbox_inches='tight')
                saved_files['confusion_matrix'] = confusion_path
                self.log(f"Confusion matrix saved to {confusion_path}")
                
                if show_plot:
                    plt.show()
                else:
                    plt.close()
                    
            except Exception as e:
                self.log(f"Error visualizing confusion matrix: {e}")
        
        # Plot accuracy metrics if available
        if eval_results is not None:
            try:
                # Collect available metrics
                metrics = {}
                if 'accuracy' in eval_results:
                    metrics['Accuracy'] = eval_results['accuracy']
                if 'train_acc' in eval_results:
                    metrics['Train Accuracy'] = eval_results['train_acc']
                if 'test_acc' in eval_results:
                    metrics['Test Accuracy'] = eval_results['test_acc']
                if 'val_acc' in eval_results:
                    metrics['Validation Accuracy'] = eval_results['val_acc']
                
                if metrics:
                    # Create a bar chart of metrics
                    plt.figure(figsize=(3, 2))
                    names = list(metrics.keys())
                    values = [metrics[name] for name in names]
                    plt.bar(names, values, color='cornflowerblue')
                    plt.title(f'{title_prefix} Performance Metrics')
                    plt.ylim(0, 1.0)
                    plt.ylabel('Score')
                    plt.grid(axis='y', linestyle='--', alpha=0.7)
                    
                    # Add text annotations
                    for i, v in enumerate(values):
                        plt.text(i, v + 0.02, f'{v:.4f}', 
                                ha='center', va='bottom', fontweight='bold')
                    
                    # Save figure
                    metrics_path = os.path.join(viz_dir, f'{title_prefix.lower().replace(" ", "_")}_metrics.png')
                    plt.savefig(metrics_path, dpi=300, bbox_inches='tight')
                    saved_files['metrics'] = metrics_path
                    self.log(f"Metrics plot saved to {metrics_path}")
                    
                    if show_plot:
                        plt.show()
                    else:
                        plt.close()
                        
            except Exception as e:
                self.log(f"Error visualizing metrics: {e}")
        
        # Add a method to compare multiple models if results are stored
        if not hasattr(self, '_model_results'):
            self._model_results = {}
        
        # Store these results for potential comparison later
        model_key = title_prefix.lower().replace(" ", "_")
        self._model_results[model_key] = {
            'eval_results': eval_results,
            'title': title_prefix
        }
        
        return saved_files

    def compare_models(self, save_dir=None, show_plot=True, figsize=(12, 8)):
        """
        Compare results from multiple models that have been visualized.

        """
        
        if not hasattr(self, '_model_results') or not self._model_results:
            self.log("No model results available for comparison")
            return {}
        
        # Use pipeline's results directory if not specified
        if save_dir is None:
            save_dir = self.path_results
        
        # Create a timestamp for the comparison

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create a subdirectory for this comparison
        compare_dir = os.path.join(save_dir, f"model_comparison_{timestamp}")
        os.makedirs(compare_dir, exist_ok=True)
        
        saved_files = {}
        
        # Collect accuracy metrics across models
        model_names = []
        accuracies = []
        
        for model_key, data in self._model_results.items():
            eval_results = data['eval_results']
            title = data['title']
            
            if eval_results and 'accuracy' in eval_results:
                model_names.append(title)
                accuracies.append(eval_results['accuracy'])
        
        if accuracies:
            # Create comparison bar chart
            plt.figure(figsize=figsize)
            plt.bar(model_names, accuracies, color=['cornflowerblue', 'lightcoral', 'mediumseagreen', 'gold'][:len(model_names)])
            plt.title(f'Model Accuracy Comparison')
            plt.ylim(0, 1.0)
            plt.ylabel('Accuracy')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Add text annotations
            for i, v in enumerate(accuracies):
                plt.text(i, v + 0.02, f'{v:.4f}', 
                        ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            
            # Save figure
            comparison_path = os.path.join(compare_dir, f'model_accuracy_comparison.png')
            plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
            saved_files['accuracy_comparison'] = comparison_path
            self.log(f"Model comparison saved to {comparison_path}")
            
            if show_plot:
                plt.show()
            else:
                plt.close()
        
        return saved_files        

    def _save_data_to_h5(self, data_dict, filepath):
        """Save a dictionary of data to an HDF5 file to avoid memory issues."""
        with h5py.File(filepath, 'w') as f:
            # Save feature arrays
            if 'features' in data_dict and data_dict['features']:
                features_grp = f.create_group('features')
                for i, feat in enumerate(data_dict['features']):
                    features_grp.create_dataset(f'feature_{i}', data=feat)
                
                # Store the count
                f.attrs['feature_count'] = len(data_dict['features'])
            
            # Save spectrogram arrays if present
            if 'spectrograms' in data_dict and data_dict['spectrograms']:
                spec_grp = f.create_group('spectrograms')
                for i, spec in enumerate(data_dict['spectrograms']):
                    spec_grp.create_dataset(f'spec_{i}', data=spec)
                
                # Store the count
                f.attrs['spectrogram_count'] = len(data_dict['spectrograms'])
            
            # Save string data as attributes or as datasets with special encoding
            if 'phoneme_labels' in data_dict:
                # Convert list of strings to a single string with delimiter
                labels_str = '|'.join(data_dict['phoneme_labels'])
                f.attrs['phoneme_labels'] = labels_str
            
            if 'phoneme_words' in data_dict:
                words_str = '|'.join(data_dict['phoneme_words'])
                f.attrs['phoneme_words'] = words_str
            
            if 'phoneme_participant_ids' in data_dict:
                ids_str = '|'.join(data_dict['phoneme_participant_ids'])
                f.attrs['phoneme_participant_ids'] = ids_str
            
            # Save metadata
            if 'metadata' in data_dict:
                meta_grp = f.create_group('metadata')
                for key, value in data_dict['metadata'].items():
                    if isinstance(value, (int, float, str, bool)):
                        meta_grp.attrs[key] = value
                    elif isinstance(value, dict):
                        # For nested dictionaries, store as JSON string
                        import json
                        meta_grp.attrs[key] = json.dumps(value)
                        
    def _load_data_from_h5(self, filepath):
        """Load data from an HDF5 file."""
        if not os.path.exists(filepath):
            self.log(f"H5 file not found: {filepath}")
            return None
        
        try:
            data_dict = {}
            with h5py.File(filepath, 'r') as f:
                # Load feature arrays
                if 'features' in f:
                    features_grp = f['features']
                    feature_count = f.attrs.get('feature_count', len(features_grp))
                    features = []
                    
                    for i in range(feature_count):
                        feat_name = f'feature_{i}'
                        if feat_name in features_grp:
                            features.append(features_grp[feat_name][()])
                    
                    data_dict['features'] = features
                
                # Load spectrogram arrays
                if 'spectrograms' in f:
                    spec_grp = f['spectrograms']
                    spec_count = f.attrs.get('spectrogram_count', len(spec_grp))
                    spectrograms = []
                    
                    for i in range(spec_count):
                        spec_name = f'spec_{i}'
                        if spec_name in spec_grp:
                            spectrograms.append(spec_grp[spec_name][()])
                    
                    data_dict['spectrograms'] = spectrograms
                
                # Load string data from attributes
                if 'phoneme_labels' in f.attrs:
                    labels_str = f.attrs['phoneme_labels']
                    data_dict['phoneme_labels'] = labels_str.split('|')
                
                if 'phoneme_words' in f.attrs:
                    words_str = f.attrs['phoneme_words']
                    data_dict['phoneme_words'] = words_str.split('|')
                
                if 'phoneme_participant_ids' in f.attrs:
                    ids_str = f.attrs['phoneme_participant_ids']
                    data_dict['phoneme_participant_ids'] = ids_str.split('|')
                
                # Load metadata
                if 'metadata' in f:
                    meta_grp = f['metadata']
                    metadata = {}
                    
                    for key in meta_grp.attrs:
                        value = meta_grp.attrs[key]
                        # Check if it's a JSON string
                        if isinstance(value, str) and value.startswith('{') and value.endswith('}'):
                            try:
                                import json
                                metadata[key] = json.loads(value)
                            except:
                                metadata[key] = value
                        else:
                            metadata[key] = value
                    
                    data_dict['metadata'] = metadata
            
            return data_dict
        
        except Exception as e:
            self.log(f"Error loading H5 file {filepath}: {e}")
            return None
            
