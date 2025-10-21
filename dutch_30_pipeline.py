from pipeline import UnifiedPhonemePipeline
from phonetic_dictionary import PhoneticDictionary
import glob
import os
import re
import json
import pickle
from datetime import datetime
from collections import Counter, defaultdict
from debugger import DebugMixin
import numpy as np
from extract_features import extractHG, extractMelSpecs
from config import BIDS_PATH, OUTPUT_PATH, RESULTS_PATH, DUTCH_30_PATH, DUTCH_10_PATH, get_dataset_paths

class Dutch30Pipeline(UnifiedPhonemePipeline, DebugMixin):
    """Extend the pipeline for Dutch30 data"""
    
    def __init__(self, dutch30_extractor, **kwargs):
        
        super().__init__(
            path_bids=dutch30_extractor.data_dir, 
            path_output=dutch30_extractor.results_dir,
            path_results=dutch30_extractor.results_dir,
            **kwargs
        )
        self.class_name = "Dutch30Pipeline" 
        self.dutch30_extractor = dutch30_extractor
        self.phonetic_dict = PhoneticDictionary()
        self.phonetic_dict.add_phoneme_groups()
    
    def step1_load_dutch30_data(self, sample_fraction=0.2):
        """Replace step1 - load Dutch30 data instead of initializing decoder"""
        self.log("Step 1: Loading Dutch30 raw data...")
        
        # Get patient split
        split_info = self.dutch30_extractor.create_patient_split()
        
        # We'll process raw data through normal pipeline
        # Don't convert to "pipeline format" with pre-computed features
        self.split_info = split_info
        
        self.log(f"Patients: {len(split_info['train'])} train, "
              f"{len(split_info['val'])} val, {len(split_info['test'])} test")
        
        return self
    
    def _convert_to_pipeline_format(self, X, y, split_name):
        """Convert Dutch30 data to pipeline format"""
        # y contains words, we need to map to phonemes
        phoneme_labels = []
        phoneme_words = []
        features = []
        
        for i, label in enumerate(y):
            if isinstance(label, bytes):
                label = label.decode()
            
            # Clean the label - remove \r and whitespace
            label = label.strip()
        
            # Map word to phonemes (simplified - takes first phoneme)
            if label and label in self.phonetic_dict:
                phonemes = self.phonetic_dict.extract_phonemes(label)
                phoneme_labels.append(phonemes[0] if phonemes else 'unknown')
            else:
                phoneme_labels.append('unknown')
            
            phoneme_words.append(label)
            features.append(X[i])
        
        unique_words = set(phoneme_words[:100])
        self.debug(f"  Sample words in {split_name}: {list(unique_words)[:5]}")
        unique_phonemes = set(phoneme_labels[:100])
        self.debug(f"  Sample phonemes in {split_name}: {list(unique_phonemes)[:5]}")
        
        return {
            'features': features,
            'phoneme_labels': phoneme_labels,
            'phoneme_words': phoneme_words,
            'split': split_name
        }
    
    def step2_3_use_existing_split(self):
        """Use existing split with actual channel quality stratification"""
        self.log("Steps 2-3")
        
        split_info = self.dutch30_extractor.create_patient_split()
        channel_dir = os.path.join(self.path_results, 'channel_analysis')
        
        # Calculate quality scores for all patients
        patient_scores = {}
        for pid in split_info['train'] + split_info['val'] + split_info['test']:
            result_path = os.path.join(channel_dir, f'{pid}_channel_correlations.npy')
            if os.path.exists(result_path):
                data = np.load(result_path, allow_pickle=True).item()
                correlations = [ch['correlation'] for ch in data.values() 
                              if not np.isnan(ch['correlation'])]
                if correlations:
                    patient_scores[pid] = np.mean(correlations)
        
        # Calculate thresholds
        scores = list(patient_scores.values())
        low_thresh = np.percentile(scores, 33)
        high_thresh = np.percentile(scores, 67)
        
        # Assign quality strata
        self.participant_strata = {}
        for pid in split_info['train'] + split_info['val'] + split_info['test']:
            if pid in patient_scores:
                if patient_scores[pid] >= high_thresh:
                    self.participant_strata[pid] = 'high_quality'
                elif patient_scores[pid] >= low_thresh:
                    self.participant_strata[pid] = 'medium_quality'
                else:
                    self.participant_strata[pid] = 'low_quality'
            else:
                self.participant_strata[pid] = 'low_quality'  # No data = low quality
        
        # Create split_result
        self.split_result = {
            'train': {pid: {} for pid in split_info['train']},
            'test': {pid: {} for pid in split_info['test']},
            'val': {pid: {} for pid in split_info['val']},
            'word_segments_dict': {}
        }
        
        successfully_loaded = []
        failed_patients = []
        
        
        # Populate word segments for sampled patients
        for pid in split_info['train'] + split_info['test'] + split_info['val']:
            try:
                #self.log(f"Segmenting data for {pid}...")
                # CHECK CHANNEL COUNT FIRST
                eeg_path = os.path.join(self.path_bids, f'{pid}_sEEG.npy')
                if os.path.exists(eeg_path):
                    eeg = np.load(eeg_path)
                    n_channels = eeg.shape[1]
                    
                    # Exclude patients with too few channels
                    if n_channels < 75:
                        self.log(f"Excluding {pid}: only {n_channels} channels (< 75 threshold)")
                        failed_patients.append(pid)
                        continue  # Skip to next patient
                    
                    self.log(f"Segmenting data for {pid} ({n_channels} channels)...")
                else:
                    self.log(f"Warning: No raw data file for {pid}")
                    failed_patients.append(pid)
                    continue
                
                word_segments = self.segment_data_by_words(pid)
                
                if word_segments and 'words' in word_segments:
                    self.split_result['word_segments_dict'][pid] = word_segments
                    successfully_loaded.append(pid)
                else:
                    self.log(f"Warning: No word segments for {pid}")
                    failed_patients.append(pid)
                    
            except MemoryError as e:
                self.log(f"Memory error for {pid}: {e}. Skipping...")
                failed_patients.append(pid)
            except Exception as e:
                self.log(f"Error segmenting {pid}: {e}. Skipping...")
                failed_patients.append(pid)
        
        # *** REMOVE FAILED PATIENTS FROM SPLITS ***
        for pid in failed_patients:
            # Remove from train/test/val splits
            if pid in self.split_result['train']:
                del self.split_result['train'][pid]
            if pid in self.split_result['test']:
                del self.split_result['test'][pid]
            if pid in self.split_result['val']:
                del self.split_result['val'][pid]
        
        self.log(f"Successfully loaded: {len(successfully_loaded)} patients")
        self.log(f"Failed/excluded: {len(failed_patients)} patients: {failed_patients}")

        quality_distribution = Counter()
        for pid in successfully_loaded:
            if pid in self.participant_strata:
                quality_distribution[self.participant_strata[pid]] += 1

        self.log(f"Quality distribution (after exclusion): {quality_distribution}")
        self.log(f"  High quality: {quality_distribution.get('high_quality', 0)} patients")
        self.log(f"  Medium quality: {quality_distribution.get('medium_quality', 0)} patients")
        self.log(f"  Low quality: {quality_distribution.get('low_quality', 0)} patients")

        # Populate word-to-instance mappings for parent's get_data_batch
        for pid in successfully_loaded:  # *** ONLY USE SUCCESSFULLY LOADED ***
            segments = self.split_result['word_segments_dict'][pid]
            
            # Determine which split this patient is in
            if pid in split_info['train']:
                split_type = 'train'
            elif pid in split_info['test']:
                split_type = 'test'
            else:
                split_type = 'val'
            
            # Create word-to-instance mappings
            for word, word_data in segments['words'].items():  # Now it's a dict
                if word not in self.split_result[split_type][pid]:
                    self.split_result[split_type][pid][word] = []
                num_instances = len(word_data['instances'])
                self.split_result[split_type][pid][word].extend(range(num_instances))
                
        return self

    def segment_data_by_words(self, participant_id):
        """
        Segment raw EEG by words (like Dutch10 does)
        """
        # Load raw data
        raw_data = self.dutch30_extractor.load_patient_raw_data(participant_id)
        
        eeg = raw_data['eeg']
        stimuli = raw_data['stimuli']
        audio = raw_data['audio']
        eeg_sr = raw_data['eeg_sr']
        
        # Find word boundaries in stimuli
        word_segments = self._segment_by_word_markers(eeg, stimuli, audio, eeg_sr, participant_id)
        
        return word_segments
    
    def _segment_by_word_markers(self, eeg, stimuli, audio, eeg_sr, participant_id):
        """
        Segment continuous recording into words.
        Dutch30: stimuli contains sentences repeated across samples.
        """
        # Find sentence boundaries (where stimuli changes)
        sentence_boundaries = []
        current_sentence = None
        
        for i, stim in enumerate(stimuli):
            sentence = stim.decode() if isinstance(stim, bytes) else str(stim)
            sentence = sentence.strip()
            
            if sentence != current_sentence:
                if current_sentence is not None:
                    sentence_boundaries.append({
                        'sentence': current_sentence,
                        'start': sentence_start,
                        'end': i
                    })
                current_sentence = sentence
                sentence_start = i
        
        # Add final sentence
        if current_sentence is not None:
            sentence_boundaries.append({
                'sentence': current_sentence,
                'start': sentence_start,
                'end': len(stimuli)
            })
        
        self.debug(f"Found {len(sentence_boundaries)} sentences")
        
        # Split each sentence into words and create segments
        all_words = []
        all_eeg_segments = []
        all_spec_segments = []
        all_audio_segments = [] 
        
        for sent_info in sentence_boundaries:
            sentence = sent_info['sentence']
            sent_start = sent_info['start']
            sent_end = sent_info['end']
            
            # Clean and split sentence into words
            # Remove punctuation, split on whitespace
            cleaned = re.sub(r'["""„"''\r\n]+', '', sentence)
            words = [w for w in cleaned.split() if w]
            
            if not words:
                continue
            
            # Divide sentence time equally among words
            sent_duration = sent_end - sent_start
            samples_per_word = sent_duration // len(words)

            
            for word_idx, word in enumerate(words):
                word_start = sent_start + (word_idx * samples_per_word)
                word_end = sent_start + ((word_idx + 1) * samples_per_word)
                if word_idx == len(words) - 1:  # Last word gets remainder
                    word_end = sent_end
                
                # Extract EEG segment
                eeg_segment = eeg[word_start:word_end]
                
                # Extract audio segment
                audio_start = int(word_start * len(audio) / len(eeg))
                audio_end = int(word_end * len(audio) / len(eeg))
                audio_segment = audio[audio_start:audio_end].copy()
                scaled_audio = np.int16(audio_segment * 32767)
                all_audio_segments.append(audio_segment)
                
                # Compute spectrogram
                spec_segment = extractMelSpecs(
                    scaled_audio,
                    48000,
                    windowLength=0.05,
                    frameshift=0.01
                )
                
                all_words.append(word)
                all_eeg_segments.append(eeg_segment)
                all_spec_segments.append(spec_segment)
                
        
        words_dict = {}
        for i, word in enumerate(all_words):
            if word not in words_dict:
                words_dict[word] = {'instances': []}
            words_dict[word]['instances'].append({
                'eeg_segment': all_eeg_segments[i],
                'spectrogram_segment': all_spec_segments[i], 
                'audio_segment': all_audio_segments[i]
            })
        
        self.debug(f"Extracted {len(all_words)} word segments")
        
        return {
            'words': words_dict,
            'words_list': all_words,
            'eeg_segments': all_eeg_segments,
            'spectrogram_segments': all_spec_segments,
            'audio_segments': all_audio_segments,  
            'participant_id': participant_id
        }
    
    def _create_segments_from_features(self, features, words):
        word_instances = {}
        current_word = None
        word_start = 0
        
        for i, word in enumerate(words):
            if word != current_word:
                if current_word and current_word.strip():
                    if current_word not in word_instances:
                        word_instances[current_word] = []
                    
                    segment_features = np.array(features[word_start:i])
                    word_instances[current_word].append({
                        'onset_sample': word_start,
                        'offset_sample': i,
                        'eeg_segment': segment_features.copy(),
                        'audio_segment': segment_features.copy(),
                        'spectrogram_segment': segment_features.copy(),
                        'duration_samples': i - word_start,
                        'duration_ms': (i - word_start) * 10
                    })
                
                current_word = word
                word_start = i
        
        # Handle last word
        if current_word and current_word.strip():
            if current_word not in word_instances:
                word_instances[current_word] = []
            segment_features = np.array(features[word_start:])
            word_instances[current_word].append({
                'onset_sample': word_start,
                'offset_sample': len(words),
                'eeg_segment': segment_features.copy(),
                'audio_segment': segment_features.copy(),
                'spectrogram_segment': segment_features.copy(),
                'duration_samples': len(words) - word_start,
                'duration_ms': (len(words) - word_start) * 10
            })
        
        return {
            'words': {word: {'instances': instances} 
                     for word, instances in word_instances.items()},
            'metadata': {
                'participant_id': 'dutch30_patient',
                'total_word_instances': sum(len(inst) for inst in word_instances.values())
            }
        }
    
    def step4_custom_detector(self):
        """Initialize detector without BIDS decoder"""
        print("Step 4: Initializing detector...")
        from acoustic_change_detector import AcousticChangeDetector
        
        self.detector = AcousticChangeDetector(
            min_segment_duration=0.03,
            max_segment_duration=0.3,
            phonetic_dict=self.phonetic_dict,
            debug_mode=self.DEBUG_MODE
        )
        
        # Critical: set decoder to self
        self.detector.decoder = self
        #self.detector.split_result = self.split_result
        
        return self.detector

    def step5_accumulate_data_dutch30(self, sample_fraction):
        """Accumulate all available data for Dutch30"""
        
        # Calculate total samples and batches needed
        train_samples = 0
        test_samples = 0
        val_samples = 0
        
        word_segments_dict = self.split_result['word_segments_dict']
        
        # Count train samples
        for pid in self.split_result['train']:
            if pid in word_segments_dict:
                for word, indices in self.split_result['train'][pid].items():
                    train_samples += len(indices)
        
        # Count test samples
        for pid in self.split_result['test']:
            if pid in word_segments_dict:
                for word, indices in self.split_result['test'][pid].items():
                    test_samples += len(indices)
        
        # Count val samples
        if 'val' in self.split_result:
            for pid in self.split_result['val']:
                if pid in word_segments_dict:
                    for word, indices in self.split_result['val'][pid].items():
                        val_samples += len(indices)
        
        self.debug(f"Available samples: train={train_samples}, test={test_samples}, val={val_samples}")
        
        # NOW use your flexible batch sizing logic
        if train_samples < 5000:
            batch_size = 32
        elif train_samples < 20000:
            batch_size = 64
        else:
            batch_size = 128
 
        
        self.debug(f"Using batch_size={batch_size}")    
        
        # Call parent's step5 with calculated batches
        self.train = self.detector.accumulate_phoneme_data(
            split_result=self.split_result,
            batch_size=batch_size,
            feature_extraction_method=self.feature_extraction_method,
            batch_type='train'
        )
        
        self.test = self.detector.accumulate_phoneme_data(
            split_result=self.split_result,
            batch_size=batch_size,
            feature_extraction_method=self.feature_extraction_method,
            batch_type='test'
        )
        
        # Add val accumulation after test data in parent method
        self.val = self.detector.accumulate_phoneme_data(
            split_result=self.split_result,
            batch_size=batch_size,
            feature_extraction_method=self.feature_extraction_method,
            batch_type='val'  # Requires 'val' in split_result
        )
        
        self.debug(f"Train phonemes: {set(self.train['phoneme_labels'])}")
        self.debug(f"Test phonemes: {set(self.test['phoneme_labels'])}")
        
        unknown_words = set()
        for i, label in enumerate(self.train['phoneme_labels']):
            if label == '?':
                unknown_words.add(self.train['phoneme_words'][i])
        self.debug(f"Words without phoneme mappings: {unknown_words}")

        # STANDARDIZE AT END OF STEP 5
        self.log("Standardizing features to (195, 133) at end of step 5")
        
        # Store original shapes for reference (optional but useful)
        self._original_train_shapes = [f.shape for f in self.train['features']]
        self._original_test_shapes = [f.shape for f in self.test['features']]
        
        # Now standardize
        self.standardize_feature_shapes(target_channels=133)
        
        self.log(f"Step 5 complete: train={len(self.train['features'])}, "
                 f"test={len(self.test['features'])}, val={len(self.val['features'])} samples")        
        
        return self.train, self.test, self.val
        
    def step6_resolve_unknowns(self):
        # Check what fields your accumulated data has
        print("Train data keys:", self.train.keys())
        print("Sample phoneme_labels:", self.train['phoneme_labels'][:5])
        print("Unknown count:", self.train['phoneme_labels'].count('?'))

        """Step 6: Initialize validator to resolve unknown phonemes"""        
        self.train, self.test = super().step6_resolve_unknowns()
        return self.train, self.test
    
    def step8_convert_to_groups(self):
        """Step 8: Convert phonemes to groups if use_phoneme_groups is True"""
        if not self.use_phoneme_groups:
            self.log("Step 8: Skipping group conversion (use_phoneme_groups=False)")
            return
        
        self.log("Step 8: Converting phonemes to groups...")
        
        # SAVE ORIGINAL PHONEME DATA FIRST
        if hasattr(self, 'train') and self.train:
            self.train_phonemes = {k: v.copy() if isinstance(v, list) else v 
                                   for k, v in self.train.items()}
        
        if hasattr(self, 'test') and self.test:
            self.test_phonemes = {k: v.copy() if isinstance(v, list) else v 
                                  for k, v in self.test.items()}
        
        # NOW convert to groups
        if hasattr(self, 'train') and self.train:
            self.train = self._convert_labels_to_groups(self.train)
        
        if hasattr(self, 'test') and self.test:
            self.test = self._convert_labels_to_groups(self.test)
        
        # Also save group versions
        self.train_groups = self.train
        self.test_groups = self.test
        
        # Log the results
        if self.train:
            from collections import Counter
            train_groups = Counter(self.train['phoneme_labels'])
            self.log(f"Train data: {len(train_groups)} groups, {dict(train_groups)}")
        
        self.log("Step 8: Conversion to groups complete")
        return self
    
    def analyze_dutch30_channels(self):
        """Run channel analysis for Dutch30 patients"""
        
        os.makedirs(os.path.join(self.path_results, 'channel_analysis'), exist_ok=True)
        
        split_info = self.dutch30_extractor.create_patient_split()
        all_patients = split_info['train'] + split_info['val'] + split_info['test']
        
        for pid in all_patients:
            result_path = os.path.join(self.path_results, 'channel_analysis', 
                                      f'{pid}_channel_correlations.npy')
            
            if os.path.exists(result_path):
                print(f"{pid}: Already analyzed")
                continue
                
            print(f"Analyzing {pid}...")
            
            # Load raw EEG data for this patient
            eeg_path = os.path.join(self.dutch30_extractor.data_dir, f'{pid}_sEEG.npy')
            stimuli_path = os.path.join(self.dutch30_extractor.data_dir, f'{pid}_stimuli.npy')
            
            if not os.path.exists(eeg_path):
                print(f"  {pid}: EEG file not found")
                continue
            
            eeg = np.load(eeg_path)
            stimuli = np.load(stimuli_path, allow_pickle=True)
            
            # Create spectrogram from stimuli (as proxy for reconstruction target)
            labels = downsampleLabels(stimuli, self.dutch30_extractor.sampling_rate)
            
            # Analyze each channel
            channel_results = {}
            n_channels = eeg.shape[1]
            
            for ch_idx in range(n_channels):
                # Extract features for single channel
                single_chan = eeg[:, [ch_idx]]
                feat = extractHG(single_chan, self.dutch30_extractor.sampling_rate)
                feat = stackFeatures(feat, modelOrder=4, stepSize=5)
                
                # Quick correlation test (simplified)
                if feat.shape[0] > 100:
                    # Use a simple correlation metric
                    correlation = np.random.random()  # Replace with actual correlation calculation
                    channel_results[f'CH{ch_idx:03d}'] = {
                        'correlation': correlation,
                        'region': 'Unknown',
                        'index': ch_idx
                    }
            
            np.save(result_path, channel_results)
            print(f"  {pid}: Analyzed {n_channels} channels")     
            
    def get_data_batch(self, split_result, batch_type='train', **kwargs):
        """Override to handle flat list format"""
        word_segments = split_result['word_segments_dict']
        
        # Convert to expected format on-the-fly
        for pid, segments in word_segments.items():
            if isinstance(segments['words'], list):
                # Convert flat lists to nested dict
                words_dict = {}
                for i, word in enumerate(segments['words']):
                    if word not in words_dict:
                        words_dict[word] = {'instances': []}
                    words_dict[word]['instances'].append({
                        'eeg_segment': segments['eeg_segments'][i],
                        'spectrogram_segment': segments['spectrogram_segments'][i]
                    })
                segments['words'] = words_dict
        
        return super().get_data_batch(split_result, batch_type, **kwargs)
        
    def checkpoint_after_step6(self, sample_fraction=None):
        """Save checkpoint with sample fraction in filename"""
        
        if not hasattr(self, 'train') or self.train is None:
            self.log("WARNING: No training data to checkpoint")
            return None
        
        if 'features' not in self.train or not self.train['features']:
            self.log("WARNING: Training data is empty, not saving checkpoint")
            return None
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Include sample fraction in filename
        fraction_str = f"_sample{int(sample_fraction*100)}" if sample_fraction else ""
        filename = f"pipeline_{self.feature_extraction_method}_pca{self.pca_components}{fraction_str}_after_step6_{timestamp}.pkl"
        filepath = os.path.join(self.path_results, filename)
        
        self.log(f"Saving checkpoint: {filename}")
        
        try:
            metadata = {
                'method': self.feature_extraction_method,
                'pca_components': self.pca_components,
                'sample_fraction': sample_fraction,
                'timestamp': timestamp,
                'stage': 'after_step6',
                'train_samples': len(self.train['features']),
                'test_samples': len(self.test['features']) if self.test else 0
            }
            
            # Save data to HDF5
            train_file = filepath.replace('.pkl', '_train.h5')
            self._save_data_to_h5(self.train, train_file)
            metadata['train_file'] = os.path.basename(train_file)
            
            if self.test:
                test_file = filepath.replace('.pkl', '_test.h5')
                self._save_data_to_h5(self.test, test_file)
                metadata['test_file'] = os.path.basename(test_file)
            
            # Save only metadata (avoid unpickleable objects)
            with open(filepath, 'wb') as f:
                pickle.dump({'metadata': metadata}, f)
            
            self.log(f"Checkpoint saved: {filename}")
            return filepath
            
        except Exception as e:
            self.log(f"Error saving checkpoint: {e}")
            return None

    def try_load_checkpoint(self, sample_fraction=None):
        """Load checkpoint matching current configuration and sample fraction"""
        
        # Include sample fraction in pattern if specified
        fraction_str = f"_sample{int(sample_fraction*100)}" if sample_fraction else ""
        pattern = f"pipeline_{self.feature_extraction_method}_pca{self.pca_components}{fraction_str}_after_step6_*.pkl"
        matching_files = glob.glob(os.path.join(self.path_results, pattern))
        
        if not matching_files:
            self.log(f"No checkpoint found for {self.feature_extraction_method}, PCA={self.pca_components}, sample={sample_fraction}")
            return False
        
        matching_files.sort(key=lambda f: os.path.getmtime(f), reverse=True)
        newest_checkpoint = matching_files[0]
        
        try:
            self.log(f"Loading checkpoint: {os.path.basename(newest_checkpoint)}")
            
            with open(newest_checkpoint, 'rb') as f:
                data = pickle.load(f)
            
            metadata = data.get('metadata', {})
            
            # Load data from h5 files
            if 'train_file' in metadata:
                train_file = os.path.join(self.path_results, metadata['train_file'])
                self.train = self._load_data_from_h5(train_file)
            
            if 'test_file' in metadata:
                test_file = os.path.join(self.path_results, metadata['test_file'])
                self.test = self._load_data_from_h5(test_file)
            
            # Load val if exists
            val_file = newest_checkpoint.replace('.pkl', '_val.h5')
            if os.path.exists(val_file):
                self.val = self._load_data_from_h5(val_file)
            
            self.log(f"Checkpoint loaded: train={len(self.train['features'])}, test={len(self.test['features'])} samples")
            return True
            
        except Exception as e:
            self.log(f"Error loading checkpoint: {e}")
            return False

    def run_step1_to_step6(self, sample_fraction=0.0001, force_reprocess=False):
        """Run Dutch30-specific steps 1-6"""
        
        if not force_reprocess and self.try_load_checkpoint(sample_fraction):
            self.log("Loaded checkpoint - skipping steps 1-6")
            return self
        
        try:
            # Dutch30 custom steps
            self.step1_load_dutch30_data(sample_fraction)
            self.step2_3_use_existing_split()
            self.step4_custom_detector()
            
            # Modified step 5 for Dutch30
            self.train, self.test, self.val = self.step5_accumulate_data_dutch30(sample_fraction)
            
            # Reuse parent's step 6
            self.step6_resolve_unknowns()
            
            # Save checkpoint with sample fraction
            self.checkpoint_after_step6(sample_fraction)
            
        except Exception as e:
            self.log(f"Error in Dutch30 steps 1-6: {e}")
            raise
        
        return self

    def run_step7_and_beyond(self):
        """Run steps 7 and beyond."""
        self.log("Step 7: Filtering unknowns")
        self.step7_filter_unknowns()
        
        return self            
    
    # 18.10.2025 temp debugging method
    def debug_sentence_parsing(self, participant_id, max_samples=10):
        """
        Comprehensive debug to understand sentence → word → phoneme parsing
        Add this method to your Dutch30Pipeline class
        """
        self.log("\n" + "="*80)
        self.log(f"SENTENCE PARSING DEBUG: {participant_id}")
        self.log("="*80)
        
        # Load raw data
        raw_data = self.dutch30_extractor.load_patient_raw_data(participant_id)
        eeg = raw_data['eeg']
        stimuli = raw_data['stimuli']
        audio = raw_data['audio']
        
        # 1. STIMULI ANALYSIS
        self.log("\n[1] STIMULI STRUCTURE")
        self.log("-" * 80)
        unique_stimuli = np.unique(stimuli)
        self.log(f"Total unique stimuli: {len(unique_stimuli)}")
        self.log(f"Total stimuli instances: {len(stimuli)}")
        
        # Categorize by word count
        by_word_count = defaultdict(list)
        
        for label in unique_stimuli[:max_samples]:
            label_str = label.decode() if isinstance(label, bytes) else str(label)
            word_count = len(label_str.split())
            by_word_count[word_count].append(label_str)
        
        for word_count in sorted(by_word_count.keys()):
            samples = by_word_count[word_count]
            self.log(f"\n  {word_count}-word stimuli ({len(samples)} samples):")
            for sample in samples[:3]:
                self.log(f"    '{sample}'")
        
        # 2. WORD SEGMENTATION
        self.log("\n[2] WORD SEGMENTATION PROCESS")
        self.log("-" * 80)
        
        # Simulate what _segment_by_word_markers does
        word_segments = []
        
        for i, label in enumerate(stimuli[:max_samples]):
            label_str = label.decode() if isinstance(label, bytes) else str(label)
            words_in_label = label_str.strip().split()
            
            self.log(f"\nStimulus {i}: '{label_str}'")
            self.log(f"  → Splits into {len(words_in_label)} words: {words_in_label}")
            
            # Check if this creates issues
            if len(words_in_label) > 1:
                self.log(f"MULTI-WORD STIMULUS - needs splitting!")
            
        # 3. PHONEME LOOKUP
        print("\n[3] PHONEME LOOKUP")
        print("-" * 80)
        
        for i, label in enumerate(unique_stimuli[:max_samples]):
            label_str = label.decode() if isinstance(label, bytes) else str(label)
            words = label_str.strip().split()
            
            self.log(f"\nStimulus: '{label_str}'")
            for word in words:
                phonemes = self.phonetic_dict.extract_phonemes(word)
                if phonemes:
                    self.log(f"  '{word}' → {phonemes} ({len(phonemes)} phonemes)")
                else:
                    self.log(f"  '{word}' → NOT FOUND in dictionary")
        
        # 4. BOUNDARY DETECTION SIMULATION
        print("\n[4] BOUNDARY DETECTION SIMULATION")
        print("-" * 80)
        
        # Get actual word segments
        word_result = self.segment_data_by_words(participant_id)
        
        words_list = word_result.get('words_list', [])
        specs_list = word_result.get('spectrogram_segments', [])

        for i, (word, spec) in enumerate(zip(
            words_list[:max_samples],
            specs_list[:max_samples]
        )):
            self.log(f"\nWord {i}: '{word}'")
            self.log(f"  Spectrogram shape: {spec.shape}")
            
            # Get expected phonemes
            phonemes = self.phonetic_dict.extract_phonemes(word)
            if phonemes:
                self.log(f"  Expected phonemes: {phonemes} ({len(phonemes)} phonemes)")
                
                # Initialize if not exists
                if not hasattr(self, 'detector'):
                    self.step4_custom_detector()
    
                # Simulate boundary detection
                boundary_result = self.detector.detect_boundaries(
                    spectrogram=spec,
                    word=word,
                    frameshift=0.01
                )
                
                detected_count = len(boundary_result['segments'])
                expected_count = len(phonemes)
                
                self.log(f"  Detected boundaries: {detected_count} segments")
                
                if detected_count != expected_count:
                    self.log(f"  MISMATCH: Expected {expected_count}, got {detected_count}")
                    self.log(f"    Segment lengths: {[seg.shape[0] for seg in boundary_result['segments']]}")
                else:
                    self.log(f"  Match: {detected_count} segments for {expected_count} phonemes")
            else:
                self.log(f"  Word not in dictionary")
        
        # 5. SUMMARY STATISTICS
        print("\n[5] SUMMARY")
        print("-" * 80)
        
        # Count multi-word vs single-word stimuli
        multi_word_count = sum(1 for s in stimuli if len((s.decode() if isinstance(s, bytes) else str(s)).split()) > 1)
        single_word_count = len(stimuli) - multi_word_count
        
        self.log(f"Single-word stimuli: {single_word_count} ({100*single_word_count/len(stimuli):.1f}%)")
        self.log(f"Multi-word stimuli:  {multi_word_count} ({100*multi_word_count/len(stimuli):.1f}%)")
        
        # Check dictionary coverage
        unique_words = set()
        for s in unique_stimuli:
            label_str = s.decode() if isinstance(s, bytes) else str(s)
            unique_words.update(label_str.split())
        
        found_words = sum(1 for w in unique_words if self.phonetic_dict.extract_phonemes(w))
        self.log(f"\nDictionary coverage: {found_words}/{len(unique_words)} unique words ({100*found_words/len(unique_words):.1f}%)")
        
        print("\n" + "="*80)