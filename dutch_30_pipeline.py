from pipeline import UnifiedPhonemePipeline
from phonetic_dictionary import PhoneticDictionary
from scipy.signal import hilbert

import os
import re
import json
import glob

import pickle
import numpy as np

from datetime import datetime
from collections import Counter, defaultdict
from debugger import DebugMixin
from phoneme_validator import PhonemeValidator

from scipy.signal import decimate
from extract_features import extractHG, extractMelSpecs
from acoustic_change_detector import AcousticChangeDetector
from config import BIDS_PATH, OUTPUT_PATH, RESULTS_PATH, DUTCH_30_PATH, DUTCH_10_PATH, get_dataset_paths
from dataset_config import Dutch30Config

class Dutch30Pipeline(UnifiedPhonemePipeline, DebugMixin):
    """Extend the pipeline for Dutch30 data"""

    
    def __init__(self, dutch30_extractor, config: Dutch30Config = None,
                        decoder=None, feature_extraction_method='high_gamma',
                        pca_components=100, use_phoneme_groups=False, 
                        debug_mode=False, use_rms_boundaries=True, use_multifeature=False,
                        **kwargs):
        
        super().__init__(
            path_bids=dutch30_extractor.data_dir, 
            path_output=dutch30_extractor.results_dir,
            path_results=dutch30_extractor.results_dir,
            feature_extraction_method=feature_extraction_method,
            pca_components=pca_components,
            use_phoneme_groups=use_phoneme_groups,
            debug_mode=debug_mode,
            **kwargs
        )
        self.class_name = "Dutch30Pipeline" 
        self.dutch30_extractor = dutch30_extractor
        self.phonetic_dict = PhoneticDictionary()
        self.phonetic_dict.add_phoneme_groups()
        self.config = config if config is not None else Dutch30Config()
        self.use_rms_boundaries = use_rms_boundaries
        self.use_multifeature = use_multifeature
        
        # Log config if in debug mode
        self.debug(str(self.config))
        self.log(f"Pipeline initialized: {feature_extraction_method}, PCA={pca_components}, groups={use_phoneme_groups}")
        self.log(f"Boundary detection: RMS={use_rms_boundaries}, MultiFeature={use_multifeature}")
        
        # Initialize detector with config
        self.detector = AcousticChangeDetector(
            config=self.config,
            feature_extraction_method=self.feature_extraction_method,
            use_rms_boundaries=self.use_rms_boundaries,     
            use_multifeature=self.use_multifeature         
        )
    
    def step1_load_dutch30_data(self, num_patients=None, patient_ids=None, patient_range=None):
        """
        Load data from specified patients.
        
        Parameters:
        -----------
        num_patients : int or None
            Load first N patients (e.g., 3 → P01, P02, P03)
        patient_ids : list or None
            Specific patient IDs (e.g., ['P01', 'P10', 'P20'])
        patient_range : tuple or None
            Range of patients (e.g., (10, 20) → P10 through P20 inclusive)
        """
        self.log(f"Step 1: Loading Dutch30...")
        
        all_patient_ids = ['P01', 'P02', 'P03', 'P04', 'P06', 'P07', 'P08', 
                           'P09', 'P10', 'P11', 'P12', 'P13', 'P14', 'P15',
                           'P16', 'P17', 'P20', 'P21', 'P22', 'P23', 'P24',
                           'P25', 'P26', 'P27', 'P28', 'P29', 'P30']
        
        # Determine which patients to use
        if patient_ids is not None:
            # Use explicit list
            selected_patients = [pid for pid in patient_ids if pid in all_patient_ids]
            self.log(f"  Using specified patients: {selected_patients}")
            
        elif patient_range is not None:
            # Use range (start, end) inclusive
            start, end = patient_range
            selected_patients = []
            for pid in all_patient_ids:
                # Extract number from PXX format
                num = int(pid[1:])
                if start <= num <= end:
                    selected_patients.append(pid)
            self.log(f"  Using patients P{start:02d} to P{end:02d}: {selected_patients}")
            
        elif num_patients is not None:
            # Use first N patients
            selected_patients = all_patient_ids[:num_patients]
            self.log(f"  Using first {num_patients} patients: {selected_patients}")
            
        else:
            # Use all patients
            selected_patients = all_patient_ids
            self.log(f"  Using all {len(selected_patients)} patients")
        
        if not selected_patients:
            raise ValueError("No valid patients selected!")
        
        # Store selected patients for later use
        self.selected_patients = selected_patients
        
        return self
    
    def step2_split_by_instances(self, train_fraction=0.7, random_seed=42):
        """Split each patient's word instances into train/test."""
        np.random.seed(random_seed)
        
        self.split_result = {'train': {}, 'test': {}, 'word_segments_dict': {}}
        self.patient_baselines = {}
        
        patient_ids = self.selected_patients if hasattr(self, 'selected_patients') else ['P01', 'P02', 'P03', 'P04', 'P06', 'P07', 'P08', 
                                                                                       'P09', 'P10', 'P11', 'P12', 'P13', 'P14', 'P15',
                                                                                       'P16', 'P17', 'P20', 'P21', 'P22', 'P23', 'P24',
                                                                                       'P25', 'P26', 'P27', 'P28', 'P29', 'P30']
        
        for pid in patient_ids:
            try:
                # Load raw data to extract baseline
                raw_data = self.dutch30_extractor.load_patient_raw_data(pid)
                eeg = raw_data['eeg']
                audio = raw_data['audio']
                
                # Extract baseline from silence
                baseline = self._extract_baseline_from_silence(audio, eeg)
                self.patient_baselines[pid] = baseline
                
                # Segment words
                word_segments = self.segment_data_by_words(pid)
                self.split_result['word_segments_dict'][pid] = word_segments
                
            except Exception as e:
                self.log(f"Failed to process {pid}: {e}")
                continue
            
            self.split_result['train'][pid] = {}
            self.split_result['test'][pid] = {}
            
            for word, word_data in word_segments['words'].items():
                num_instances = len(word_data['instances'])
                if num_instances == 0:
                    continue
                
                indices = np.arange(num_instances)
                np.random.shuffle(indices)
                
                # CHANGE THIS PART:
                if num_instances == 1:
                    # Randomly assign to train or test
                    if np.random.random() < train_fraction:
                        self.split_result['train'][pid][word] = [0]
                    else:
                        self.split_result['test'][pid][word] = [0]
                else:
                    n_train = max(1, int(num_instances * train_fraction))
                    self.split_result['train'][pid][word] = indices[:n_train].tolist()
                    self.split_result['test'][pid][word] = indices[n_train:].tolist()
            
            train_total = sum(len(v) for v in self.split_result['train'][pid].values())
            test_total = sum(len(v) for v in self.split_result['test'][pid].values())
            self.log(f"{pid}: {train_total} train, {test_total} test, baseline: {baseline.shape}")
        
        return self.split_result
    
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
        
        # Extract baseline from silence
        baseline = self._extract_baseline_from_silence(audio, eeg)
    
        # Find word boundaries in stimuli
        word_segments = self._segment_by_word_markers(eeg, stimuli, audio, eeg_sr, participant_id)
        
        # Add baseline to result
        word_segments['baseline'] = baseline
        
        return word_segments
    
    def _segment_by_word_markers(self, eeg: np.ndarray, stimuli: np.ndarray, audio: np.ndarray, 
                             eeg_sr: int, participant_id: str) -> dict:
        """
        Segment data by processing sentences, then extracting individual words.
        
        Flow:
        1. Identify sentence boundaries from stimuli
        2. For each sentence: use RMS to find word boundaries
        3. Extract individual word segments (EEG, audio, spectrogram)
        4. Store words in dictionary
        """
        
        # ===================================================================
        # STEP 1: IDENTIFY SENTENCES FROM STIMULI
        # ===================================================================
        sentence_list = []
        current_sentence_text = None
        current_sentence_start_idx = 0
        
        for stim_idx, stim in enumerate(stimuli):
            sentence_text = stim.decode() if isinstance(stim, bytes) else str(stim)
            sentence_text = sentence_text.strip()
            
            # New sentence detected
            if sentence_text != current_sentence_text:
                # Save previous sentence
                if current_sentence_text is not None:
                    sentence_list.append({
                        'text': current_sentence_text,
                        'stim_start_idx': current_sentence_start_idx,
                        'stim_end_idx': stim_idx
                    })
                
                # Start tracking new sentence
                current_sentence_text = sentence_text
                current_sentence_start_idx = stim_idx
        
        # Save final sentence
        if current_sentence_text is not None:
            sentence_list.append({
                'text': current_sentence_text,
                'stim_start_idx': current_sentence_start_idx,
                'stim_end_idx': len(stimuli)
            })
        
        self.debug(f"Identified {len(sentence_list)} sentences")
        
        # Step2: Storage for all extracted words
        all_word_texts = []
        all_word_eeg_segments = []
        all_word_spec_segments = []
        all_word_audio_segments = []
        
        for sent_info in sentence_list:
            sentence_text = sent_info['text']
            sent_stim_start = sent_info['stim_start_idx']
            sent_stim_end = sent_info['stim_end_idx']
            
            # -----------------------------------------------------------
            # 2A. Parse sentence text into individual words
            # -----------------------------------------------------------
            cleaned_sentence = re.sub(r'["""„"''\r\n]+', '', sentence_text)
            word_texts = [w for w in cleaned_sentence.split() if w]
            
            if not word_texts:
                self.debug(f"Skipping empty sentence")
                continue
            
            self.debug(f"Processing sentence: '{sentence_text}' → {len(word_texts)} words")
            
            # -----------------------------------------------------------
            # 2B. Extract sentence-level audio
            # -----------------------------------------------------------
            audio_start_sent = int(sent_stim_start * len(audio) / len(eeg))
            audio_end_sent = int(sent_stim_end * len(audio) / len(eeg))
            audio_sent = audio[audio_start_sent:audio_end_sent].copy()
            
            # -----------------------------------------------------------
            # 2C. Use RMS to find word boundaries WITHIN this sentence
            # -----------------------------------------------------------
            try:
                rms_result = self.detector.segment_sentence_by_rms(
                    audio_sentence=audio_sent,
                    audio_sr=self.config.audio_sr,
                    words=word_texts,
                    phonetic_dict=self.phonetic_dict
                )
                
                word_boundaries_in_sent = rms_result['word_boundaries_samples']
                word_audio_segments = rms_result['word_segments']
                
            except Exception as e:
                self.debug(f"Failed RMS segmentation for '{sentence_text}': {e}")
                continue
            
            # -----------------------------------------------------------
            # 2D. Extract each word's data from sentence
            # -----------------------------------------------------------
            for word_idx, word_text in enumerate(word_texts):
                # Get word boundaries (in samples, relative to sentence audio)
                word_audio_start_in_sent = int(word_boundaries_in_sent[word_idx])
                word_audio_end_in_sent = int(word_boundaries_in_sent[word_idx + 1])
                
                # Map word audio boundaries to absolute audio indices
                word_audio_start_abs = audio_start_sent + word_audio_start_in_sent
                word_audio_end_abs = audio_start_sent + word_audio_end_in_sent
                
                # Map word audio boundaries to EEG indices
                word_fraction_start = word_audio_start_in_sent / len(audio_sent)
                word_fraction_end = word_audio_end_in_sent / len(audio_sent)
                
                sent_duration_eeg = sent_stim_end - sent_stim_start
                word_eeg_start = sent_stim_start + int(word_fraction_start * sent_duration_eeg)
                word_eeg_end = sent_stim_start + int(word_fraction_end * sent_duration_eeg)
                
                # Extract word EEG segment
                word_eeg = eeg[word_eeg_start:word_eeg_end]
                word_audio = word_audio_segments[word_idx]
                
                # -----------------------------------------------------------
                # 2E. Validate word segment lengths
                # -----------------------------------------------------------
                min_eeg_frames = int(self.config.min_phoneme_duration * eeg_sr)
                if len(word_eeg) < min_eeg_frames:
                    self.debug(f"Skipping '{word_text}': EEG too short "
                              f"({len(word_eeg)/eeg_sr*1000:.0f}ms < {min_eeg_frames/eeg_sr*1000:.0f}ms)")
                    continue
                
                min_audio_samples = int((self.config.window_length + self.config.frameshift) * self.config.audio_sr)
                if len(word_audio) < min_audio_samples:
                    self.debug(f"Skipping '{word_text}': audio too short")
                    continue
                
                # -----------------------------------------------------------
                # 2F. Create spectrogram for this word
                # -----------------------------------------------------------
                try:
                    # Downsample audio
                    audio_down = decimate(
                        word_audio,
                        int(self.config.audio_sr / self.config.audio_target_sr)
                    )
                    
                    # Check downsampled length
                    min_audio_samples_down = int((self.config.window_length + self.config.frameshift) 
                                                * self.config.audio_target_sr)
                    if len(audio_down) < min_audio_samples_down:
                        self.debug(f"Skipping '{word_text}': downsampled audio too short")
                        continue
                    
                    # Normalize and convert to int16
                    audio_normalized = audio_down / np.max(np.abs(audio_down) + 1e-10)
                    audio_int16 = np.int16(audio_normalized * self.config.int16_max)
                    
                    # Extract mel spectrogram
                    word_spec = extractMelSpecs(
                        audio_int16,
                        self.config.audio_target_sr,
                        windowLength=self.config.window_length,
                        frameshift=self.config.frameshift,
                        numFilter=self.config.mel_num_filters
                    )
                    
                except Exception as e:
                    self.debug(f"Skipping '{word_text}': spectrogram failed ({e})")
                    continue
                
                # Validate spectrogram
                if word_spec.shape[0] < 3:
                    self.debug(f"Skipping '{word_text}': spectrogram only {word_spec.shape[0]} frames")
                    continue
                
                # -----------------------------------------------------------
                # 2G. Store validated word segments
                # -----------------------------------------------------------
                all_word_texts.append(word_text)
                all_word_eeg_segments.append(word_eeg)
                all_word_spec_segments.append(word_spec)
                all_word_audio_segments.append(word_audio)  # Original, not downsampled
        
        # ===================================================================
        # STEP 3: ORGANIZE WORDS INTO DICTIONARY
        # ===================================================================
        words_dict = {}
        
        for i, word_text in enumerate(all_word_texts):
            if word_text not in words_dict:
                words_dict[word_text] = {'instances': []}
            
            words_dict[word_text]['instances'].append({
                'eeg_segment': all_word_eeg_segments[i],
                'spectrogram_segment': all_word_spec_segments[i],
                'audio_segment': all_word_audio_segments[i]
            })
        
        self.debug(f"Successfully extracted {len(all_word_texts)} word segments from {len(sentence_list)} sentences")
        self.debug(f"Unique words: {len(words_dict)}")
        
        # ===================================================================
        # RETURN: Word-level data organized by unique words
        # ===================================================================
        return {
            'words': words_dict,                          # Dict: word → instances
            'words_list': all_word_texts,                 # List: all words in order
            'eeg_segments': all_word_eeg_segments,        # List: EEG per word
            'spectrogram_segments': all_word_spec_segments,  # List: spec per word
            'audio_segments': all_word_audio_segments,    # List: audio per word
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
        self.log("Step 4: Initializing detector...")
                
        self.detector = AcousticChangeDetector(
            config=self.config,
            phonetic_dict=self.phonetic_dict,
            debug_mode=self.DEBUG_MODE,
            feature_extraction_method=self.feature_extraction_method,
            use_wav2vec=True
        )
        
        self.detector.decoder = self
                
        return self.detector

    def step5_accumulate_data_dutch30(self):
        """Accumulate all available data for Dutch30"""
        

        # Calculate total samples and batches needed
        train_samples = 0
        test_samples = 0
        
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
        
        
        self.debug(f"Available samples: train={train_samples}, test={test_samples}")
        
        self.log(f"\nStep 5 starting:")
        self.log(f"  Train patients: {list(self.split_result['train'].keys())}")
        self.log(f"  Available samples: train={train_samples}, test={test_samples}")
        
        # flexible batch sizing logic
        if train_samples < 5000:
            batch_size = 128
        elif train_samples < 20000:
            batch_size = 256
        else:
            batch_size = 128
        
        self.debug(f"Using batch_size={batch_size}")    
        
        # Call parent's step5 with calculated batches
        self.train = self.detector.accumulate_phoneme_data(
            split_result=self.split_result,
            batch_size=batch_size,
            feature_extraction_method=self.feature_extraction_method,
            batch_type='train', 
            standardize_channels=self.config.standardize_channels,
            standardize_values=self.config.standardize_values   
        )
        self.log(f"  Train accumulated: {len(self.train['features'])} samples, {len(set(self.train['phoneme_labels']))} phonemes")
        
        self.test = self.detector.accumulate_phoneme_data(
            split_result=self.split_result,
            batch_size=batch_size,
            feature_extraction_method=self.feature_extraction_method,
            batch_type='test', 
            standardize_channels=self.config.standardize_channels,
            standardize_values=self.config.standardize_values   
        )        
        
        if self.feature_extraction_method in ['high_gamma', 'multi_band']:
            # Per-patient trimming
            self._trim_edge_phonemes_per_patient()
            
            # Duration filtering
            self.train = self.filter_valid_phonemes(
                dataset='train', 
                min_duration=self.config.min_phoneme_duration, 
                max_duration=self.config.max_phoneme_duration
            )
            self.log(f"  After filtering: {len(self.train['features'])} samples")
            
            self.test = self.filter_valid_phonemes(
                dataset='test',
                min_duration=self.config.min_phoneme_duration,
                max_duration=self.config.max_phoneme_duration
            )
            
            if hasattr(self, 'val'):
                self.val = self.filter_valid_phonemes(
                    dataset='val',
                    min_duration=self.config.min_phoneme_duration,
                    max_duration=self.config.max_phoneme_duration
                )
        else:
            self.log(f"Skipping trimming for '{self.feature_extraction_method}'")
            
        # Subtract baselines (already extracted in step 2!)
        if hasattr(self, 'patient_baselines'):
            self.train = self.subtract_baseline(self.train, 'train', self.patient_baselines)
            self.log(f"  After baseline: {len(self.train['features'])} samples")
            self.test = self.subtract_baseline(self.test, 'test', self.patient_baselines)
        
        self.debug(f"Train phonemes: {set(self.train['phoneme_labels'])}")
        self.debug(f"Test phonemes: {set(self.test['phoneme_labels'])}")
        
        unknown_words = set()
        for i, label in enumerate(self.train['phoneme_labels']):
            if label == '?':
                unknown_words.add(self.train['phoneme_words'][i])
        self.debug(f"Words without phoneme mappings: {unknown_words}")
        
        self.log(f"Step 5 complete: train={len(self.train['features'])}, "
                 f"test={len(self.test['features'])} samples")        
        
        return self.train, self.test
        
    def analyze_phoneme_lengths(self):
        """Analyze phoneme length distribution"""
        phoneme_lengths = defaultdict(list)
        
        for i, label in enumerate(self.train['phoneme_labels']):
            feat = self.train['features'][i]
            n_frames = feat.shape[0]
            duration = n_frames * self.config.frameshift
            phoneme_lengths[label].append(duration)
        
       
        for phoneme in sorted(phoneme_lengths.keys()):
            durations = phoneme_lengths[phoneme]
            self.log(f"\n{phoneme}: {len(durations)} samples")
            self.log(f"  Duration range: {min(durations):.3f}s - {max(durations):.3f}s")
            self.log(f"  Mean ± Std: {np.mean(durations):.3f}s ± {np.std(durations):.3f}s")
            self.log(f"  CV: {(np.std(durations)/np.mean(durations)*100):.1f}%")
            
            if (np.std(durations)/np.mean(durations)) > self.config.max_phoneme_duration:
                self.log(f"HIGH LENGTH VARIATION!")

    def dutch30_step6_resolve_unknowns(self):
        """Step 6: Initialize validator to resolve unknown phonemes (Dutch30-specific)"""
        
        self.log(f"Train data keys: {self.train.keys()}")
        self.log(f"Sample phoneme_labels: {self.train['phoneme_labels'][:5]}")
        self.log(f"Unknown count: {self.train['phoneme_labels'].count('?')}")
        
        # Initialize validator with detector
        validator = PhonemeValidator(
            detector=self.detector,
            debug_mode=self.DEBUG_MODE
        )
        
        self.validator = validator
        
        if self.DEBUG_MODE:
            self.validator.enable_debug()
        
        self.log("Step 6: Validator initialized")
        
        # Resolve unknowns in training data 
        unknown_count = self.train['phoneme_labels'].count('?')
        if unknown_count > 0:
            self.log(f"Resolving {unknown_count} unknown phonemes in training...")
            
            self.train = self.validator.resolve_unknown_phonemes(
                self.train
            )
        
        # Resolve unknowns in test data
        if self.test is not None and len(self.test.get('phoneme_labels', [])) > 0:
            test_unknown = self.test['phoneme_labels'].count('?')
            self.log(f"Test unknowns: {test_unknown}")
            
            if test_unknown > 0:
                self.test = self.validator.resolve_unknown_phonemes(self.test)
        
        # Check remaining unknowns
        train_unknown_after = self.train['phoneme_labels'].count('?')
        test_unknown_after = self.test['phoneme_labels'].count('?') if self.test else 0
        
        if train_unknown_after > 0 or test_unknown_after > 0:
            self.log(f"WARNING: Still {train_unknown_after} unknown in train, {test_unknown_after} in test")
        
        self.log(f"Step 6 complete: {len(self.train['features'])} train, {len(self.test['features']) if self.test else 0} test")
        self.log(f"  Unknown remaining: {train_unknown_after} train, {test_unknown_after} test")
        
        return self.train, self.test
    
    def analyze_dutch30_channels(self):
        """Run channel analysis for Dutch30 patients"""
        
        os.makedirs(os.path.join(self.path_results, 'channel_analysis'), exist_ok=True)
        
        split_info = self.split_info
        all_patients = split_info['train'] + split_info['val'] + split_info['test']
        
        for pid in all_patients:
            result_path = os.path.join(self.path_results, 'channel_analysis', 
                                      f'{pid}_channel_correlations.npy')
            
            if os.path.exists(result_path):
                self.log(f"{pid}: Already analyzed")
                continue
                
            self.log(f"Analyzing {pid}...")
            
            # Load raw EEG data for this patient
            eeg_path = os.path.join(self.dutch30_extractor.data_dir, f'{pid}_sEEG.npy')
            stimuli_path = os.path.join(self.dutch30_extractor.data_dir, f'{pid}_stimuli.npy')
            
            if not os.path.exists(eeg_path):
                self.log(f"  {pid}: EEG file not found")
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
            self.log(f"  {pid}: Analyzed {n_channels} channels")     
            
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
        
    '''
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
'''
    def run_step1_to_step6(self, sample_fraction=0.0001, force_reprocess=False):
        """Run Dutch30-specific steps 1-6"""
        
        if not force_reprocess and self.try_load_checkpoint(sample_fraction):
            self.log("Loaded checkpoint - skipping steps 1-6")
            return self
        
        try:
            # Dutch30 custom steps
            self.step1_load_dutch30_data(sample_fraction)
            self.step2_split_by_instances()
            self.step2_3_use_existing_split()
            self.step4_custom_detector()
            
            # Modified step 5 for Dutch30
            self.train, self.test, self.val = self.step5_accumulate_data_dutch30()
            
            # Reuse parent's step 6
            self.dutch30_step6_resolve_unknowns()
            
            # Save checkpoint with sample fraction
            #self.checkpoint_after_step6(sample_fraction)
            
        except Exception as e:
            self.log(f"Error in Dutch30 steps 1-6: {e}")
            raise
        
        return self

    def debug_sentence_parsing(self, participant_id, max_samples=10):
        """
        Comprehensive debug to understand sentence → word → phoneme parsing
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
        self.log("\n[3] PHONEME LOOKUP")
        self.log("-" * 80)
        
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
        self.log("\n[4] BOUNDARY DETECTION SIMULATION")
        self.log("-" * 80)
        
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
                    word=word
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
        self.log("\n[5] SUMMARY")
        self.log("-" * 80)
        
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
        
        self.log("\n" + "="*80)
        
    '''
    def trim_edge_phonemes_to_mean(self, dataset='train'):
        """
        Trim first and last phonemes to match mean duration of middle phonemes.
        """
       
        data = getattr(self, dataset)
        
        features = data['features']
        positions = data.get('phoneme_positions', [0] * len(features))
        words = data.get('phoneme_words', ['unknown'] * len(features))
        
        # Step 1: Calculate mean length of middle phonemes
        middle_lengths = []
        
        for i, feat in enumerate(features):
            position = positions[i]
            word = words[i]
            
            # Get expected phoneme count for this word
            if hasattr(self, 'detector') and hasattr(self.detector, 'phonetic_dict'):
                try:
                    expected_phonemes = self.detector.phonetic_dict.extract_phonemes(word)
                    n_phonemes = len(expected_phonemes)
                except:
                    n_phonemes = None
            else:
                n_phonemes = None
            
            # Check if it's a middle phoneme
            if n_phonemes and n_phonemes > 2:  # Only for words with 3+ phonemes
                if position > 0 and position < n_phonemes - 1:
                    middle_lengths.append(feat.shape[0])
        
        if len(middle_lengths) == 0:
            self.debug("No middle phonemes found - skipping trim")
            return
        
        target_length = int(np.mean(middle_lengths))
        self.log(f"Middle phonemes: {len(middle_lengths)} samples")
        self.log(f"Mean length: {target_length} frames ({target_length * self.config.frameshift:.3f}s)")
        self.log(f"Std: {np.std(middle_lengths):.1f} frames")
        
        # Step 2: Trim first and last phonemes
        trimmed_features = []
        trim_stats = {'first_trimmed': 0, 'last_trimmed': 0, 'unchanged': 0}
        
        for i, feat in enumerate(features):
            position = positions[i]
            word = words[i]
            current_length = feat.shape[0]
            
            # Get expected phoneme count
            if hasattr(self, 'detector') and hasattr(self.detector, 'phonetic_dict'):
                expected_phonemes = self.detector.phonetic_dict.extract_phonemes(word)
                # Filter out '?' phonemes
                valid_phonemes = [p for p in expected_phonemes if p != '?']
                n_phonemes = len(valid_phonemes) if valid_phonemes else None                

            else:
                n_phonemes = None
            
            # Determine if first or last
            is_first = (n_phonemes and position == 0)
            is_last = (n_phonemes and n_phonemes > 1 and position == n_phonemes - 1)
            
            if is_first and current_length > target_length:
                # Trim from START (keep end that borders next phoneme)
                trim_amount = current_length - target_length
                trimmed_feat = feat[trim_amount:, :]  # Keep the end
                trimmed_features.append(trimmed_feat)
                trim_stats['first_trimmed'] += 1
                
            elif is_last and current_length > target_length:
                # Trim from END (keep start that borders previous phoneme)
                trimmed_feat = feat[:target_length, :]  # Keep the start
                trimmed_features.append(trimmed_feat)
                trim_stats['last_trimmed'] += 1
                
            else:
                # Keep as is (middle phonemes or already short enough)
                trimmed_features.append(feat)
                trim_stats['unchanged'] += 1
        
        # Step 3: Update the dataset
        data['features'] = trimmed_features
        
        # Print summary
        self.log(f"\nTrimming summary:")
        self.log(f"  First phonemes trimmed: {trim_stats['first_trimmed']}")
        self.log(f"  Last phonemes trimmed: {trim_stats['last_trimmed']}")
        self.log(f"  Unchanged: {trim_stats['unchanged']}")
        
        # Recalculate statistics after trimming
        new_lengths = [f.shape[0] for f in trimmed_features]
        self.log(f"\nAfter trimming:")
        self.log(f"  Mean length: {np.mean(new_lengths):.1f} frames ({np.mean(new_lengths) * self.config.frameshift:.3f}s)")
        self.log(f"  Std: {np.std(new_lengths):.1f} frames")
        self.log(f"  CV: {(np.std(new_lengths) / np.mean(new_lengths) * 100):.1f}%")
        
        return data
        '''
        
    def _trim_edge_phonemes_per_patient(self):
        """Calculate per-patient targets from train, apply to all datasets."""
        
        # Calculate targets from train only
        patient_targets = {}
        for pid in set(self.train['phoneme_participant_ids']):
            middle_lengths = []
            for i, p in enumerate(self.train['phoneme_participant_ids']):
                if p != pid:
                    continue
                
                feat = self.train['features'][i]
                pos = self.train['phoneme_positions'][i]
                word = self.train['phoneme_words'][i]
                
                phonemes = self.detector.phonetic_dict.extract_phonemes(word)
                n_phonemes = len([p for p in phonemes if p != '?']) if phonemes else None
                
                if n_phonemes and n_phonemes > 2 and 0 < pos < n_phonemes - 1:
                    middle_lengths.append(feat.shape[0])
            
            if middle_lengths:
                patient_targets[pid] = int(np.mean(middle_lengths))
        
        # Apply to all datasets
        for dataset_name in ['train', 'test', 'val']:
            if not hasattr(self, dataset_name):
                continue
            
            data = getattr(self, dataset_name)
            trimmed = []
            
            for i, feat in enumerate(data['features']):
                pid = data['phoneme_participant_ids'][i]
                pos = data['phoneme_positions'][i]
                word = data['phoneme_words'][i]
                
                if pid not in patient_targets:
                    trimmed.append(feat)
                    continue
                
                target = patient_targets[pid]
                phonemes = self.detector.phonetic_dict.extract_phonemes(word)
                n = len([p for p in phonemes if p != '?']) if phonemes else None
                
                is_first = n and pos == 0
                is_last = n and n > 1 and pos == n - 1
                
                if is_first and feat.shape[0] > target:
                    trimmed.append(feat[-target:, :])
                elif is_last and feat.shape[0] > target:
                    trimmed.append(feat[:target, :])
                else:
                    trimmed.append(feat)
            
            data['features'] = trimmed
            self.log(f"  {dataset_name}: Trimmed to per-patient targets")
        
    def filter_valid_phonemes(self, dataset: str ='train', min_duration: float = None, max_duration: float = None) -> dict:
        """
        Remove phonemes with invalid durations.
        
        Parameters:
        -----------
        dataset : str
            'train', 'test', or 'val'

        """
        min_duration = self.config.min_phoneme_duration
        max_duration = self.config.max_phoneme_duration
        self.log(f"\nFiltering {dataset} phonemes by duration [{min_duration}, {max_duration}]s")
        
        data = getattr(self, dataset)  # Get self.train or self.test or self.val
        
        valid_indices = []
        
        for i, feat in enumerate(data['features']):
            duration = feat.shape[0] * self.config.frameshift  # frames to seconds
            if min_duration <= duration <= max_duration:
                valid_indices.append(i)
        
        removed = len(data['features']) - len(valid_indices)
        self.log(f"Filtered out {removed}/{len(data['features'])} phonemes "
                 f"({removed/len(data['features'])*100:.1f}%) outside duration range")
        
        # Return filtered data
        return {
            'features': [data['features'][i] for i in valid_indices],
            'phoneme_labels': [data['phoneme_labels'][i] for i in valid_indices],
            'phoneme_words': [data['phoneme_words'][i] for i in valid_indices],
            'phoneme_positions': [data['phoneme_positions'][i] for i in valid_indices] if 'phoneme_positions' in data else [],
            'phoneme_participant_ids': [data['phoneme_participant_ids'][i] for i in valid_indices] if 'phoneme_participant_ids' in data else [],
            'spectrograms': [data['spectrograms'][i] for i in valid_indices] if 'spectrograms' in data else [],
            'metadata': data.get('metadata', {})
        }
        
    def _extract_baseline_from_silence(self, audio, eeg):
        """
        Extract baseline from silence periods using audio energy threshold.
        """
        
        # Process audio
        audio_down  = decimate(audio, int(self.config.audio_sr / self.config.audio_target_sr))
        scaled      = np.int16(audio_down / np.max(np.abs(audio_down)) * self.config.int16_max)
        
        window_length   = self.config.window_length
        frameshift      = self.config.frameshift
        eeg_sr          = self.config.eeg_sr
        
        # Extract mel spectrogram
        melspec = extractMelSpecs(
            scaled, 
            self.config.audio_target_sr,
            windowLength=window_length,
            frameshift=frameshift,
            numFilter=self.config.mel_num_filters
        )
        
        # Detect silence
        spec_avg = np.mean(melspec, axis=1)
        threshold = (np.max(spec_avg) + np.min(spec_avg)) * self.config.silence_threshold_factor
        is_silence = spec_avg < threshold
        
        # Find CONTINUOUS silence blocks (not individual frames)
        silence_blocks = []
        in_silence = False
        block_start = None
        
        for i, silent in enumerate(is_silence):
            if silent and not in_silence:
                # Start of silence block
                block_start = i
                in_silence = True
            elif not silent and in_silence:
                # End of silence block
                block_end = i
                silence_blocks.append((block_start, block_end))
                in_silence = False
        
        # Handle case where recording ends in silence
        if in_silence:
            silence_blocks.append((block_start, len(is_silence)))
        
        self.debug(f"Found {len(silence_blocks)} silence blocks")
        
        # Extract EEG from silence blocks (only if long enough)
        baseline_features = []
        min_silence_duration = self.config.min_silence_duration
        
        for block_start, block_end in silence_blocks:
            # Duration in seconds
            block_duration = (block_end - block_start) * frameshift  # 10ms frameshift
            
            if block_duration < min_silence_duration:
                continue  # Skip short silence blocks
            
            # Convert to EEG samples
            eeg_start = int(block_start * frameshift * eeg_sr)
            eeg_end = int(block_end * frameshift * eeg_sr)
            
            if eeg_end <= len(eeg):
                silence_eeg = eeg[eeg_start:eeg_end]
                
                # Check if segment is long enough for extractHG
                min_samples = int((window_length + frameshift) * eeg_sr)  # windowLength + frameshift
                if len(silence_eeg) < min_samples:
                    continue
                
                try:
                    # Extract high-gamma features from silence
                    silence_feat = extractHG(
                        silence_eeg, 
                        eeg_sr,
                        windowLength = window_length,
                        frameshift = frameshift
                    )
                    
                    if silence_feat.shape[0] > 0:
                        baseline_features.append(silence_feat)
                except Exception as e:
                    self.debug(f"Error extracting features from silence block: {e}")
                    continue
        
        # Average all silence
        if len(baseline_features) > 0:
            all_silence = np.vstack(baseline_features)
            baseline = np.mean(all_silence, axis=0)
            self.debug(f"Baseline: {len(baseline_features)} blocks, {all_silence.shape[0]} frames total")
            return baseline
        else:
            self.debug("Warning: No usable silence blocks found!")
            return None
            
    def subtract_baseline(self, data, dataset_name, baseline_dict):
        """
        Subtract per-patient baseline from features.
        Automatically pads baseline to match feature dimensions.
        """
        self.log(f"\nSubtracting baseline for {dataset_name}:")
        
        features = data['features']
        participants = data.get('phoneme_participant_ids', [])
        
        corrected_features = []
        corrected_count = 0
        
        for i, feat in enumerate(features):
            pid = participants[i] if i < len(participants) else None
            
            if pid in baseline_dict:
                baseline = baseline_dict[pid]  # Shape: (n_channels,)
                
                # Pad baseline to match feature dimensions if needed
                n_feat_channels = feat.shape[1]
                n_baseline_channels = baseline.shape[0]
                
                if n_baseline_channels < n_feat_channels:
                    # Pad baseline with zeros to match
                    padded_baseline = np.zeros(n_feat_channels)
                    padded_baseline[:n_baseline_channels] = baseline
                    baseline = padded_baseline
                    self.debug(f"Padded baseline for {pid}: {n_baseline_channels} → {n_feat_channels} channels")
                elif n_baseline_channels > n_feat_channels:
                    # Trim baseline (shouldn't happen but handle it)
                    baseline = baseline[:n_feat_channels]
                    self.debug(f"Trimmed baseline for {pid}: {n_baseline_channels} → {n_feat_channels} channels")
                
                # Subtract baseline from all time frames
                corrected_feat = feat - baseline
                corrected_features.append(corrected_feat)
                corrected_count += 1
            else:
                # No baseline for this patient - keep as is
                corrected_features.append(feat)
        
        data['features'] = corrected_features
        
        self.log(f"  Features corrected: {corrected_count}/{len(features)}")
        self.log(f"  Features unchanged: {len(features) - corrected_count}")
        
        return data
        
    def sample_split(self, split_info, sample_fraction):
        """Sample patients from each split proportionally"""
        sampled_split = {}
        
        for split_type in ['train', 'val', 'test']:
            full_list = split_info[split_type]
            n_total = len(full_list)
            n_sample = max(1, int(n_total * sample_fraction))
            
            sampled_split[split_type] = full_list[:n_sample]
            self.log(f"  {split_type}: {n_sample}/{n_total} patients")
        
        return sampled_split
        
    def _extract_baseline_from_silence(self, audio, eeg):
        """Extract baseline EEG from silence periods in audio."""
        window_size = int(0.5 * self.config.audio_sr)  # 500ms
        silence_threshold = np.sqrt(np.mean(audio**2)) * 0.1
        
        segment_averages = []  # Store averages, not raw segments
        
        for i in range(0, len(audio) - window_size, window_size):
            window_rms = np.sqrt(np.mean(audio[i:i+window_size]**2))
            if window_rms < silence_threshold:
                eeg_start = int(i * len(eeg) / len(audio))
                eeg_end = int((i + window_size) * len(eeg) / len(audio))
                
                # Average THIS segment immediately, don't store raw data
                segment_avg = np.mean(eeg[eeg_start:eeg_end], axis=0)
                segment_averages.append(segment_avg)
        
        if segment_averages:
            # Average across segment averages (much smaller!)
            baseline = np.mean(segment_averages, axis=0)
        else:
            baseline = np.zeros(eeg.shape[1])
        
        return baseline
