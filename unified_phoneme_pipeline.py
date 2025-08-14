import os
import numpy as np
import h5py
import json
from datetime import datetime
import pickle
from collections import Counter, defaultdict
from scipy.interpolate import interp1d
import pickle
from datetime import datetime
from debugger import DebugMixin




class UnifiedPhonemePipeline(DebugMixin):
    """
    Unified pipeline for phoneme processing with all steps integrated
    and full access to intermediate results.
    """
    
    def __init__(self, path_bids, path_output, path_results, unknown_keep_ratio=0.1, feature_extraction_method='high_gamma', debug_mode=False):
        
        
        # Initialize the DebugMixin
        super().__init__(class_name="UnifiedPhonemePipeline", debug_mode=False)
            
        if debug_mode is not None:
            self.DEBUG_MODE = debug_mode
        self.log(f"Initialized with DEBUG_MODE={self.DEBUG_MODE}")            
            
        self.path_bids = path_bids
        self.path_output = path_output
        self.path_results = path_results
        self.feature_extraction_method = feature_extraction_method
        self.unknown_keep_ratio = unknown_keep_ratio        
        
        # Store outputs from each step for inspection
        self.step_outputs = {}
        
        print("Pipeline initialized")
    
    def step1_initialize_decoder(self):
        """Step 1: Initialize the custom decoder"""
        from custom_decoder import CustomBrainAudioDecoder
        
        self.custom_decoder = CustomBrainAudioDecoder(
            path_bids=self.path_bids,
            path_output=self.path_output,
            path_results=self.path_results,
            debug_mode=self.debug
        )
        
        self.step_outputs['decoder'] = self.custom_decoder
        print("✓ Step 1: Decoder initialized")
        return self.custom_decoder
    
    def step2_stratify_participants(self, channel_correlation_threshold=0.1):
        """Step 2: Stratify participants"""
        print("Stratifying participants...")
        
        self.participant_strata = self.custom_decoder.stratify_participants_by_channel_quality(
            channel_correlation_threshold=channel_correlation_threshold
        )
        
        self.step_outputs['participant_strata'] = self.participant_strata
        print("✓ Step 2: Participants stratified")
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
        print("✓ Step 3: Train/test split created")
        return self.split_result
    
    def step4_initialize_detector(self):
        """Step 4: Initialize detector"""
        from acoustic_change_detector import AcousticChangeDetector
        
        self.detector = AcousticChangeDetector(
            min_segment_duration=0.03,
            max_segment_duration=0.3,
            distance_metric='cosine',
            smoothing_window=3,
            peak_threshold=0.5,
            decoder=self.custom_decoder,
            debug_mode=False  # Keep False as in your notebook
        )
        
        # Only set split_result if it exists
        if hasattr(self, 'split_result'):
            self.detector.split_result = self.split_result
        else:
            self.log("Warning: No split_result found for detector")
        
        self.step_outputs['detector'] = self.detector
        
        self.debug("✓ Step 4: Detector initialized")
        return self.detector
    
    def step5_accumulate_data(self, train_batches=5, test_batches=3, 
                             batch_size=32):
        """Step 5: Accumulate training and test data"""
        
        # Training data
        self.debug("Accumulating training data...")
        self.train = self.detector.accumulate_phoneme_data(
            num_batches=train_batches,
            batch_size=batch_size,
            feature_extraction_method=self.feature_extraction_method,
            batch_type='train'
        )
        
        # Test data
        self.debug("Accumulating test data...")
        self.test = self.detector.accumulate_phoneme_data(
            num_batches=test_batches,
            batch_size=batch_size,
            feature_extraction_method=self.feature_extraction_method,
            batch_type='test'
        )
        
        self.step_outputs['train_raw'] = self.train
        self.step_outputs['test_raw'] = self.test
        
        self.debug(f"Step 5: Data accumulated - {len(self.train['features'])} train, {len(self.test['features'])} test")
        return self.train, self.test
    
    def step6_resolve_unknowns(self):
        """Step 6: Initialize validator to resolve unknown phonemes"""
        from phoneme_validator import PhonemeValidator
        
        self.validator = PhonemeValidator(detector=self.detector)
        if self.debug:
            self.validator.enable_debug()
        
        self.step_outputs['validator'] = self.validator
        self.log("✓ Step 6: Validator initialized")
        
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
        
        self.step_outputs['train_resolved'] = self.train
        self.step_outputs['test_resolved'] = self.test
        
        self.debug("✓ Step 7: Unknown phonemes resolved")
        return self.train, self.test
    
    def step7_filter_unknowns(self, unknown_keep_ratio=None):
        """Step 7: Setup phonetic dictionary with groups"""
        # Use class default if not specified
        if unknown_keep_ratio is None:
            unknown_keep_ratio = getattr(self, 'unknown_keep_ratio', 0.1)
        
        self.phonetic_dict = self.detector.phonetic_dict
        
        if not hasattr(self.phonetic_dict, 'phoneme_groups'):
            self.phonetic_dict.add_phoneme_groups()
        
        self.step_outputs['phonetic_dict'] = self.phonetic_dict
        self.debug("Step 7: Phonetic dictionary setup")
                
        # Initialize the Markov model just to get phoneme_to_group mapping
        from markov_phoneme_model import MarkovPhonemeModel
        temp_model = MarkovPhonemeModel(phonetic_dict=self.phonetic_dict)
        
        # Filter training data
        filtered_features = []
        filtered_labels = []
        filtered_words = []
        filtered_participants = []
        
        for i, label in enumerate(self.train['phoneme_labels']):
            group = temp_model.phoneme_to_group.get(label, 'unknown')
            
            # Keep all non-unknown, but only keep 10% of unknown
            if group != 'unknown' or np.random.random() < unknown_keep_ratio:
                filtered_features.append(self.train['features'][i])
                filtered_labels.append(label)  # Keep as PHONEME, not group!
                filtered_words.append(self.train['phoneme_words'][i])
                filtered_participants.append(self.train['phoneme_participant_ids'][i])
        
        self.log(f"Filtered training: {len(filtered_features)} samples (from {len(self.train['features'])})")
        
        # Store filtered data
        self.train_filtered = {
            'features': filtered_features,
            'phoneme_labels': filtered_labels,
            'phoneme_words': filtered_words,
            'phoneme_participant_ids': filtered_participants
        }
        
        self.step_outputs['train_filtered'] = self.train_filtered
        
        self.debug("Step 7: Unknowns filtered")
        return self.train_filtered
    
    def run_all_steps(self):
        """Run all steps in sequence"""
        print("="*60)
        print("RUNNING COMPLETE PIPELINE")
        print("="*60)
        
        self.step1_initialize_decoder()
        self.step2_stratify_participants()
        self.step3_create_split()
        self.step4_initialize_detector()
        self.step5_accumulate_data()
        self.step6_resolve_unknowns()
        self.step7_filter_unknowns()
        
        print("\n" + "="*60)
        print("PIPELINE COMPLETE")
        print("="*60)
        
        return self
    
    def get_training_data(self, filtered=True):
        """Get training data for model"""
        if filtered and hasattr(self, 'train_filtered'):
            return self.train_filtered
        else:
            return self.train
    
    def get_test_data(self):
        """Get test data for model"""
        return self.test
    
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

    def save(self):
        """Save pipeline data with feature extraction method in filename."""
        
        # Filename includes extraction method
        filename = f'pipeline_{self.feature_extraction_method}.pkl'
        filepath = os.path.join(self.path_results, filename)
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'method': self.feature_extraction_method,
                'train': getattr(self, 'train', None),
                'test': getattr(self, 'test', None),
                'train_filtered': getattr(self, 'train_filtered', None)
            }, f)
        
        print(f"Saved: {filename}")
        return filepath
    
    def load(self, filepath):
        """Load pipeline data."""
        import pickle
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.feature_extraction_method = data['method']
        self.train = data['train']
        self.test = data['test']
        self.train_filtered = data['train_filtered']
        
        print(f"Loaded {data['method']} pipeline")
        return self
    
    @staticmethod
    def list_saved(directory):
        """List saved pipelines by extraction method."""
        
        files = [f for f in os.listdir(directory) if f.startswith('pipeline_') and f.endswith('.pkl')]
        
        for f in sorted(files):
            try:
                with open(os.path.join(directory, f), 'rb') as file:
                    data = pickle.load(file)
                    method = data.get('method', 'unknown')
                    train_size = len(data['train']['features']) if data.get('train') else 0
                    print(f"{f}: {method} - {train_size} samples")
            except:
                print(f"{f}: error reading")
    
    @staticmethod
    def load_saved(directory, method='high_gamma'):
        """Load the pipeline for a specific extraction method."""
        
        # Find files matching the method
        filename = f'pipeline_{method}.pkl'
        filepath = os.path.join(directory, filename)
        
        if not os.path.exists(filepath):
            print(f"No {method} pipeline found")
            raise FileNotFoundError(f"Pipeline file {filename} not found")
        
        # Load it
        pipeline = UnifiedPhonemePipeline(directory, directory, directory)
        pipeline.load(filepath)
        print(f"Loaded {filename}")
        return pipeline