from pipeline import UnifiedPhonemePipeline
from phonetic_dictionary import PhoneticDictionary
from scipy.signal import hilbert

import os
import re
import json
import glob
import string

import pickle
import numpy as np
import matplotlib.pyplot as plt

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
                        use_wav2vec=False, subtract_baseline=True, 
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
        self.subtract_baseline_flag = subtract_baseline
        self.use_wav2vec = use_wav2vec
        
        # Log config if in debug mode
        self.debug(str(self.config))
        self.log(f"Pipeline initialized: {feature_extraction_method}, PCA={pca_components}, groups={use_phoneme_groups}")
        self.log(f"Baseline subtraction: {subtract_baseline}")
        self.log(f"Boundary detection: RMS={use_rms_boundaries}, MultiFeature={use_multifeature}")
        
        # Initialize detector with config
        self.detector = AcousticChangeDetector(
            config=self.config,
            feature_extraction_method=self.feature_extraction_method,
            use_rms_boundaries=self.use_rms_boundaries,     
            use_multifeature=self.use_multifeature,
            use_wav2vec = self.use_wav2vec             
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
        # Clear all previous state
        self.patient_data = {}
        self.patient_baselines = {}
        
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
            # Set per-patient seed so split is identical regardless of other patients
            patient_seed = random_seed + int(pid[1:])  # P06 -> 42 + 6 = 48
            np.random.seed(patient_seed)
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
        
        print("\n=== step2_split_by_instances complete ===")
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
        2. For each sentence: use RMS or wav2vec to find word boundaries
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
        
        
        # ===================================================================
        # STEP 1.5: PRE-RESAMPLE AUDIO FOR WAV2VEC (ONCE, NOT PER SENTENCE)
        # ===================================================================
        audio_resampled_16k = None
        resample_ratio = 1.0
        
        if self.use_wav2vec:
            from scipy.signal import resample_poly
            downsample_factor = int(self.config.audio_sr / self.config.audio_target_sr)  # 48000/16000 = 3
            audio_resampled_16k = resample_poly(audio.astype(np.float32), up=1, down=downsample_factor)
            resample_ratio = len(audio_resampled_16k) / len(audio)
        
        self.debug(f"Identified {len(sentence_list)} sentences")
        
        # ===================================================================
        # STEP 2: PROCESS EACH SENTENCE
        # ===================================================================
        all_word_texts = []
        all_word_eeg_segments = []
        all_word_spec_segments = []
        all_word_audio_segments = []
        
        for sent_idx, sent_info in enumerate(sentence_list):
            sentence_text = sent_info['text']
            sent_stim_start = sent_info['stim_start_idx']
            sent_stim_end = sent_info['stim_end_idx']
            
            # -----------------------------------------------------------
            # 2A. Parse sentence text into individual words
            # -----------------------------------------------------------
            cleaned_sentence = re.sub(r'["""„"''\r\n]+', '', sentence_text)
            #word_texts = [w for w in cleaned_sentence.split() if w]
            word_texts = [w.strip(string.punctuation).lower() for w in cleaned_sentence.split() if w.strip(string.punctuation)]
            
            if not word_texts:
                #self.debug(f"Skipping empty sentence")
                continue
            
            self.debug(f"Processing sentence: '{sentence_text}' -> {len(word_texts)} words")
            
            # -----------------------------------------------------------
            # 2B. Extract sentence-level audio (original sample rate)
            # -----------------------------------------------------------
            audio_start_sent = int(sent_stim_start * len(audio) / len(eeg))
            audio_end_sent = int(sent_stim_end * len(audio) / len(eeg))
            audio_sent = audio[audio_start_sent:audio_end_sent].copy()
            
            # -----------------------------------------------------------
            # 2C. Find word boundaries WITHIN this sentence
            # -----------------------------------------------------------
            try:
              
                if self.use_wav2vec:
                    # Slice the pre-resampled audio for this sentence
                    audio_start_16k = int(audio_start_sent * resample_ratio)
                    audio_end_16k = int(audio_end_sent * resample_ratio)
                    audio_sent_16k = audio_resampled_16k[audio_start_16k:audio_end_16k]
                    
                    result = self.detector.segment_sentence_by_wav2vec(
                        audio_sentence=audio_sent_16k,
                        audio_sr=self.config.audio_target_sr,  # Already 16kHz
                        words=word_texts,
                        phonetic_dict=self.phonetic_dict
                    )

                    
                    # Scale boundaries back to original sample rate
                    word_boundaries_in_sent = (result['word_boundaries_samples'] / resample_ratio).astype(int)
                    
                    # Re-extract word segments from original audio (not resampled)
                    word_audio_segments = []
                    for i in range(len(word_boundaries_in_sent) - 1):
                        start = int(word_boundaries_in_sent[i])
                        end = int(word_boundaries_in_sent[i + 1])
                        word_audio_segments.append(audio_sent[start:end])
                        
                else:
                    result = self.detector.segment_sentence_by_rms(
                        audio_sentence=audio_sent,
                        audio_sr=self.config.audio_sr,
                        words=word_texts,
                        phonetic_dict=self.phonetic_dict
                    )
                    
                    word_boundaries_in_sent = result['word_boundaries_samples']
                    word_audio_segments = result['word_segments']
                
            except Exception as e:
                print(f"        FAILED: {e}")
                self.debug(f"Failed segmentation for '{sentence_text}': {e}")
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
                all_word_audio_segments.append(word_audio)
        
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
            'words': words_dict,
            'words_list': all_word_texts,
            'eeg_segments': all_word_eeg_segments,
            'spectrogram_segments': all_word_spec_segments,
            'audio_segments': all_word_audio_segments,
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
    
    def step3_load_channel_exclusions(self, exclusions_path):
        """
        Load channel exclusions from JSON file.
        
        Args:
            exclusions_path: Path to JSON file with format:
                            {"P01": [5, 12, 45], "P02": [3, 99], ...}
                            Values are lists of channel indices to EXCLUDE.
        
        Returns:
            dict: channel_masks per patient
        """
        import json
        
        self.log("Step 3: Loading channel exclusions...")
        
        # Load exclusions
        with open(exclusions_path, 'r') as f:
            manual_exclusions = json.load(f)
        
        self.log(f"  Loaded exclusions for {len(manual_exclusions)} patients")
        
        word_segments_dict = self.split_result['word_segments_dict']
        self.channel_masks = {}
        
        for pid in sorted(word_segments_dict.keys()):
            words_data = word_segments_dict[pid]['words']
            
            # Get channel count from first valid EEG segment
            n_channels = None
            for word, word_info in words_data.items():
                for instance in word_info['instances']:
                    eeg = instance['eeg_segment']
                    if eeg is not None and eeg.size > 0:
                        n_channels = eeg.shape[1]
                        break
                if n_channels is not None:
                    break
            
            if n_channels is None:
                self.log(f"  {pid}: No EEG data found, skipping")
                continue
            
            # Get exclusions for this patient (empty list if not specified)
            exclude_indices = manual_exclusions.get(pid, [])
            
            # Validate indices are in range
            exclude_indices = [i for i in exclude_indices if 0 <= i < n_channels]
            keep_indices = [i for i in range(n_channels) if i not in exclude_indices]
            
            self.channel_masks[pid] = {
                'exclude_indices': exclude_indices,
                'keep_indices': keep_indices,
                'n_original': n_channels,
                'n_kept': len(keep_indices),
                'n_excluded': len(exclude_indices)
            }
            
            self.log(f"  {pid}: {len(keep_indices)}/{n_channels} channels kept")
        
        return self.channel_masks
        
    def apply_channel_exclusions(self):
        """
        Apply loaded channel exclusions to EEG data in word_segments_dict.
        Call this AFTER step3_load_channel_exclusions.
        """
        if not hasattr(self, 'channel_masks') or not self.channel_masks:
            self.log("ERROR: Run step3_load_channel_exclusions first")
            return
        
        self.log("\nApplying channel exclusions...")
        
        for pid, mask in self.channel_masks.items():
            if mask['n_excluded'] == 0:
                continue
            
            keep = mask['keep_indices']
            words_data = self.split_result['word_segments_dict'][pid]['words']
            
            for word_info in words_data.values():
                for instance in word_info['instances']:
                    if instance['eeg_segment'] is not None:
                        instance['eeg_segment'] = instance['eeg_segment'][:, keep]
            
            self.log(f"  {pid}: {mask['n_original']} -> {mask['n_kept']} channels")
        
    def get_channel_counts(self):
        """Return current channel count per patient."""
        counts = {}
        for pid, patient_data in self.split_result['word_segments_dict'].items():
            first_word = list(patient_data['words'].keys())[0]
            eeg = patient_data['words'][first_word]['instances'][0]['eeg_segment']
            counts[pid] = eeg.shape[1]
        return counts

    def print_channel_counts(self):
        """Print channel counts for all patients."""
        counts = self.get_channel_counts()
        print(f"{'Patient':<10} {'Channels':<10}")
        print("-"*20)
        for pid in sorted(counts.keys()):
            print(f"{pid:<10} {counts[pid]:<10}")
    
    def step4_custom_detector(self):
        """Initialize detector without BIDS decoder"""
        self.log("Step 4: Initializing detector...")
                
        self.detector = AcousticChangeDetector(
            config=self.config,
            phonetic_dict=self.phonetic_dict,
            debug_mode=self.DEBUG_MODE,
            feature_extraction_method=self.feature_extraction_method,
            use_rms_boundaries=self.use_rms_boundaries,
            use_multifeature=self.use_multifeature,
            use_wav2vec=self.use_wav2vec, 
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
            batch_size = 256
        elif train_samples < 20000:
            batch_size = 512
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
        self.log(f"  Train accumulated: {len(self.train['features'])} samples, {len(set(self.train['phoneme_labels']))} phonemes")
    
        self.test = self.detector.accumulate_phoneme_data(
            split_result=self.split_result,
            batch_size=batch_size,
            feature_extraction_method=self.feature_extraction_method,
            batch_type='test'
        )        
        
        train_pids = set(self.train['phoneme_participant_ids'])
        for pid in sorted(train_pids):
            pid_features = [self.train['features'][i] for i, p in enumerate(self.train['phoneme_participant_ids']) if p == pid]
        
        if self.feature_extraction_method in ['high_gamma', 'multi_band']:

            # Per-patient trimming
            self._trim_edge_phonemes_per_patient()

            for pid in sorted(train_pids):
                pid_features = [self.train['features'][i] for i, p in enumerate(self.train['phoneme_participant_ids']) if p == pid]

            
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

            train_pids = set(self.train['phoneme_participant_ids'])
            for pid in sorted(train_pids):
                pid_features = [self.train['features'][i] for i, p in enumerate(self.train['phoneme_participant_ids']) if p == pid]
                    
            if hasattr(self, 'val'):
                self.val = self.filter_valid_phonemes(
                    dataset='val',
                    min_duration=self.config.min_phoneme_duration,
                    max_duration=self.config.max_phoneme_duration
                )
        else:
            self.log(f"Skipping trimming for '{self.feature_extraction_method}'")
            
            
        if self.pca_components is not None and self.pca_components > 0:
            self.train, self.test = self._apply_pca_per_patient(
                self.train, self.test, self.pca_components
            )
            
        # Subtract baselines (already extracted in step 2)
        if self.subtract_baseline_flag and hasattr(self, 'patient_baselines'):             
            
            self.train = self.subtract_baseline(self.train, 'train', self.patient_baselines)
            self.log(f"  After baseline: {len(self.train['features'])} samples")
            self.test = self.subtract_baseline(self.test, 'test', self.patient_baselines)

            train_pids = set(self.train['phoneme_participant_ids'])
            for pid in sorted(train_pids):
                pid_features = [self.train['features'][i] for i, p in enumerate(self.train['phoneme_participant_ids']) if p == pid]
                
        else:
            self.log(f"  Baseline subtraction: skipped (subtract_baseline={self.subtract_baseline_flag})")

        
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

    def step5b_normalize_lengths(self, target_frames=None, use_augmentation=True, 
                             balance_classes=True, n_chunks=5, random_state=42):
        """
        Normalize feature lengths with optional balanced augmentation.
        
        Args:
            target_frames: Target number of frames (default from config)
            use_augmentation: Whether to apply stretch augmentation
            balance_classes: Whether to balance phoneme classes
            n_chunks: Number of chunks for balancing
            random_state: For reproducibility
        """
        if target_frames is None:
            target_frames = self.config.target_frames
        
        stretch_factors = self.config.augment_stretch_factors if use_augmentation else (1.0,)
        
        self.log(f"Normalizing features: target_frames={target_frames}, augment={use_augmentation}, balance={balance_classes}")
        
        if balance_classes and use_augmentation:
            # Balanced augmentation (recommended for training)
            self.train = self._normalize_with_balanced_augmentation(
                self.train, 
                target_frames, 
                stretch_factors,
                n_chunks=n_chunks,
                random_state=random_state
            )
        elif use_augmentation:
            # Regular augmentation (all samples)
            self.train = self._normalize_with_augmentation(
                self.train, 
                target_frames, 
                stretch_factors
            )
        else:
            # No augmentation, just normalize lengths
            self.train = self._normalize_with_augmentation(
                self.train, 
                target_frames, 
                (1.0,)  # Only factor 1.0 = no stretch
            )
        
        # Test data: normalize only, no augmentation, no balancing
        self.test = self._normalize_with_augmentation(
            self.test, 
            target_frames, 
            (1.0,)
        )
        
        return self
    
    def _normalize_segments(self, data, target_frames):
        """
        Resample all segments to target_frames using scipy.signal.resample.
        
        Args:
            data: Dictionary with 'features' list and metadata lists.
            target_frames: Target number of frames.
            
        Returns:
            Modified data dictionary with resampled features.
        """
        from scipy.signal import resample
        
        normalized_features = []
        
        for feat in data['features']:
            n_frames, n_channels = feat.shape
            
            if n_frames == target_frames:
                normalized_features.append(feat)
            else:
                # Resample along time axis (axis=0)
                resampled = resample(feat, target_frames, axis=0)
                normalized_features.append(resampled)
        
        data['features'] = normalized_features
        return data
    
    def _normalize_with_augmentation(self, data, target_frames, stretch_factors):
        """
        Resample segments with augmentation: create multiple versions
        at different effective durations.
        
        For each sample, creates len(stretch_factors) versions by first
        resampling to target_frames * stretch_factor, then to target_frames.
        This simulates the phoneme being spoken faster or slower.
        
        Args:
            data: Dictionary with 'features' list and metadata lists.
            target_frames: Base target number of frames.
            stretch_factors: Tuple of factors (e.g., (0.8, 1.0, 1.2)).
            
        Returns:
            Modified data dictionary with augmented samples.
        """
        from scipy.signal import resample
        
        augmented_features = []
        augmented_labels = []
        augmented_words = []
        augmented_positions = []
        augmented_pids = []
        
        # Handle optional keys
        has_specs = 'spectrograms' in data and len(data.get('spectrograms', [])) > 0
        augmented_specs = [] if has_specs else None
        
        for i, feat in enumerate(data['features']):
            n_frames, n_channels = feat.shape
            
            for factor in stretch_factors:
                # Stretch factor < 1 means fewer frames (faster speech)
                # Stretch factor > 1 means more frames (slower speech)
                intermediate_frames = int(target_frames * factor)
                intermediate_frames = max(2, intermediate_frames)  # Minimum 2 frames
                
                # First resample to intermediate length
                if n_frames != intermediate_frames:
                    intermediate = resample(feat, intermediate_frames, axis=0)
                else:
                    intermediate = feat
                
                # Then resample to final target (only if different)
                if intermediate_frames != target_frames:
                    final = resample(intermediate, target_frames, axis=0)
                else:
                    final = intermediate
                
                augmented_features.append(final)
                augmented_labels.append(data['phoneme_labels'][i])
                augmented_words.append(data['phoneme_words'][i])
                augmented_positions.append(data['phoneme_positions'][i])
                augmented_pids.append(data['phoneme_participant_ids'][i])
                
                if has_specs:
                    augmented_specs.append(data['spectrograms'][i])
        
        # Build output dictionary
        result = {
            'features': augmented_features,
            'phoneme_labels': augmented_labels,
            'phoneme_words': augmented_words,
            'phoneme_positions': augmented_positions,
            'phoneme_participant_ids': augmented_pids,
        }
        
        if has_specs:
            result['spectrograms'] = augmented_specs
        
        # Copy any other keys that exist
        for key in data:
            if key not in result:
                result[key] = data[key]
        
        return result
        
    def _normalize_with_balanced_augmentation(self, data, target_frames, stretch_factors, 
                                           n_chunks=5, random_state=42):
        """
        Normalize all samples and augment ONLY underrepresented phonemes per patient.
        
        Strategy:
        1. Normalize ALL samples to target_frames
        2. Per patient: find max phoneme count
        3. For phonemes below max, add augmented samples to reach max
        4. Result: original data + augmented samples for rare phonemes
        """
        from scipy.signal import resample
        from collections import Counter, defaultdict
        
        rng = np.random.default_rng(random_state)
        
        labels = data['phoneme_labels']
        pids = data['phoneme_participant_ids']
        
        self.log(f"Original total samples: {len(labels)}")
        self.log(f"Original distribution: {dict(Counter(labels))}")
        
        # First: normalize ALL original samples (no augmentation yet)
        normalized_features = []
        for feat in data['features']:
            if feat.ndim == 1:
                feat = feat.reshape(-1, 1)
            n_frames = feat.shape[0]
            if n_frames != target_frames:
                normalized_features.append(resample(feat, target_frames, axis=0))
            else:
                normalized_features.append(feat.copy())
        
        # Start with ALL original data
        result_features = list(normalized_features)
        result_labels = list(data['phoneme_labels'])
        result_words = list(data['phoneme_words'])
        result_positions = list(data['phoneme_positions'])
        result_pids = list(data['phoneme_participant_ids'])
        
        has_specs = 'spectrograms' in data and len(data.get('spectrograms', [])) > 0
        result_specs = list(data['spectrograms']) if has_specs else None
        
        # Group indices by patient
        patient_indices = defaultdict(list)
        for i, pid in enumerate(pids):
            patient_indices[pid].append(i)
        
        # Augment underrepresented phonemes per patient
        total_augmented = 0
        
        for pid, indices in patient_indices.items():
            # Get phoneme counts for this patient
            patient_labels = [labels[i] for i in indices]
            counts = Counter(patient_labels)
            
            if len(counts) == 0:
                continue
            
            max_count = max(counts.values())
            
            # Build index lists per phoneme for this patient
            phoneme_indices = defaultdict(list)
            for i in indices:
                phoneme_indices[labels[i]].append(i)
            
            patient_augmented = 0
            
            # Augment phonemes that are below max_count
            for label, ph_indices in phoneme_indices.items():
                current_count = len(ph_indices)
                n_needed = max_count - current_count
                
                if n_needed <= 0:
                    continue  # This phoneme already at max, skip
                
                # Shuffle indices for variety
                ph_indices_shuffled = ph_indices.copy()
                rng.shuffle(ph_indices_shuffled)
                
                # Generate augmented samples
                augmented_count = 0
                sample_idx = 0
                
                while augmented_count < n_needed:
                    # Cycle through available samples
                    i = ph_indices_shuffled[sample_idx % len(ph_indices_shuffled)]
                    sample_idx += 1
                    
                    # Pick a random stretch factor (excluding 1.0 to avoid duplicates)
                    aug_factors = [f for f in stretch_factors if f != 1.0]
                    if not aug_factors:
                        aug_factors = stretch_factors
                    factor = rng.choice(aug_factors)
                    
                    feat = normalized_features[i]
                    n_frames = feat.shape[0]
                    
                    # Apply stretch augmentation
                    intermediate_frames = max(2, int(target_frames * factor))
                    
                    if n_frames != intermediate_frames:
                        intermediate = resample(feat, intermediate_frames, axis=0)
                    else:
                        intermediate = feat.copy()
                    
                    if intermediate_frames != target_frames:
                        final = resample(intermediate, target_frames, axis=0)
                    else:
                        final = intermediate
                    
                    # Add augmented sample
                    result_features.append(final)
                    result_labels.append(data['phoneme_labels'][i])
                    result_words.append(data['phoneme_words'][i])
                    result_positions.append(data['phoneme_positions'][i])
                    result_pids.append(data['phoneme_participant_ids'][i])
                    
                    if has_specs:
                        result_specs.append(data['spectrograms'][i])
                    
                    augmented_count += 1
                    patient_augmented += 1
            
            if patient_augmented > 0:
                self.log(f"  {pid}: added {patient_augmented} augmented samples (max_count={max_count})")
            
            total_augmented += patient_augmented
        
        self.log(f"Total augmented samples added: {total_augmented}")
        
        # Shuffle final result
        n_samples = len(result_labels)
        shuffle_idx = rng.permutation(n_samples)
        
        result = {
            'features': [result_features[i] for i in shuffle_idx],
            'phoneme_labels': [result_labels[i] for i in shuffle_idx],
            'phoneme_words': [result_words[i] for i in shuffle_idx],
            'phoneme_positions': [result_positions[i] for i in shuffle_idx],
            'phoneme_participant_ids': [result_pids[i] for i in shuffle_idx],
        }
        
        if has_specs:
            result['spectrograms'] = [result_specs[i] for i in shuffle_idx]
        
        # Copy any other keys
        for key in data:
            if key not in result:
                result[key] = data[key]
        
        # Log final distribution
        final_counts = Counter(result['phoneme_labels'])
        self.log(f"Final distribution: {dict(final_counts)}")
        self.log(f"Total samples: {len(labels)} -> {len(result['phoneme_labels'])} (+{total_augmented} augmented)")
        
        return result
    
    
    def dutch30_step6_resolve_unknowns(self):
        """Step 6: Initialize validator to resolve unknown phonemes (Dutch30-specific)"""
        
        self.log(f"Train data keys: {self.train.keys()}")
       # self.log(f"Sample phoneme_labels: {self.train['phoneme_labels'][:5]}")
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
    
    def step8_group_phonemes(self):
        """
        Convert raw phoneme labels to phoneme groups.
        
        This step maps individual phonemes (e.g., 'p', 't', 'k') to 
        articulatory groups (e.g., 'plosive').
        
        Must be called AFTER step 7 (resolve_unknowns).
        Stores original labels in 'phoneme_labels_raw' before converting.
        """
        self.log("Step 8: Grouping phonemes.")
        
        # Get phoneme-to-group mapping from phonetic dictionary
        if not hasattr(self.detector.phonetic_dict, 'phoneme_to_group'):
            self.detector.phonetic_dict.add_phoneme_groups()
        
        phoneme_to_group = self.detector.phonetic_dict.phoneme_to_group
        
        # Process each dataset
        for dataset_name in ['train', 'test', 'val']:
            if not hasattr(self, dataset_name):
                continue
            
            data = getattr(self, dataset_name)
            
            if 'phoneme_labels' not in data:
                continue
            
            # Store original raw labels
            data['phoneme_labels_raw'] = data['phoneme_labels'].copy()
            
            # Convert to groups
            grouped_labels = []
            unknown_count = 0
            
            for label in data['phoneme_labels']:
                if label in phoneme_to_group:
                    grouped_labels.append(phoneme_to_group[label])
                elif label in ('?', 'unknown', ''):
                    grouped_labels.append('unknown')
                    unknown_count += 1
                else:
                    # Phoneme not in mapping - mark as unknown
                    grouped_labels.append('unknown')
                    unknown_count += 1
            
            data['phoneme_labels'] = grouped_labels
            
            # Log statistics
            unique_raw = len(set(data['phoneme_labels_raw']))
            unique_grouped = len(set(grouped_labels))
            self.log(f"  {dataset_name}: {unique_raw} raw phonemes -> {unique_grouped} groups")
            
            if unknown_count > 0:
                self.log(f"    {unknown_count} samples mapped to 'unknown'")
        
        # Store flag indicating grouping was applied
        self.phonemes_grouped = True
        
        # Log group distribution for train
        if hasattr(self, 'train') and 'phoneme_labels' in self.train:
            from collections import Counter
            group_counts = Counter(self.train['phoneme_labels'])
            self.log(f"  Train group distribution:")
            for group, count in sorted(group_counts.items(), key=lambda x: -x[1]):
                self.log(f"    {group}: {count}")
        
        self.log("Step 8 complete: Phonemes grouped")
        
    def revert_to_raw_phonemes(self):
        """
        Revert grouped labels back to raw phoneme labels.
        
        Only works if step8_group_phonemes was previously called.
        """
        for dataset_name in ['train', 'test', 'val']:
            if not hasattr(self, dataset_name):
                continue
            
            data = getattr(self, dataset_name)
            
            if 'phoneme_labels_raw' in data:
                data['phoneme_labels'] = data['phoneme_labels_raw'].copy()
                self.log(f"  {dataset_name}: Reverted to raw phoneme labels")
        
        self.phonemes_grouped = False
        self.log("Reverted to raw phonemes")
    
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
                'test_samples': len(self.test['features']) if self.test else 0,
                'split_result': self.split_result if hasattr(self, 'split_result') else None,
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
            self.split_result = metadata.get('split_result', None)
            
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
                n_phonemes = len([ph for ph in phonemes if p != '?']) if phonemes else None
                
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
        min_duration = min_duration or self.config.min_phoneme_duration
        max_duration = max_duration or self.config.max_phoneme_duration
        self.log(f"\nFiltering {dataset} phonemes by duration [{min_duration}, {max_duration}]s")
        
        data = getattr(self, dataset)
        
        features = data['features']
        labels = data['phoneme_labels']
        words = data['phoneme_words']
        positions = data['phoneme_positions']
        pids = data['phoneme_participant_ids']
        specs = data.get('spectrograms', [])
        
        # First pass: check duration validity for each phoneme
        is_valid_duration = []
        for i, feat in enumerate(features):
            if 'phoneme_durations_samples' in data and i < len(data['phoneme_durations_samples']):
                duration = data['phoneme_durations_samples'][i] / self.config.eeg_sr
            else:
                # Fallback to frame-based calculation (less accurate)
                duration = feat.shape[0] * self.config.frameshift
            is_valid_duration.append(min_duration <= duration <= max_duration)
        
        # Second pass: group by word instance and keep only if ALL phonemes valid
        valid_indices = []
        i = 0
        
        instances_total = 0
        instances_dropped = 0
        
        while i < len(features):
            # Find word instance boundary
            start = i
            word = words[i]
            pid = pids[i]
            
            # Find end of this instance (next position=0 or different word/patient)
            i += 1
            while i < len(features):
                if positions[i] == 0:  # New instance starts
                    break
                if words[i] != word or pids[i] != pid:  # Different word/patient
                    break
                i += 1
            
            # Check if ALL phonemes in this instance are valid
            instance_indices = list(range(start, i))
            all_valid = all(is_valid_duration[j] for j in instance_indices)
            
            instances_total += 1
            
            if all_valid:
                valid_indices.extend(instance_indices)
            else:
                instances_dropped += 1
        
        removed_phonemes = len(features) - len(valid_indices)
        self.log(f"Dropped {instances_dropped}/{instances_total} word instances ({100*instances_dropped/instances_total:.1f}%)")
        self.log(f"Removed {removed_phonemes}/{len(features)} phonemes ({100*removed_phonemes/len(features):.1f}%)")
        
        return {
            'features': [features[i] for i in valid_indices],
            'phoneme_labels': [labels[i] for i in valid_indices],
            'phoneme_words': [words[i] for i in valid_indices],
            'phoneme_positions': [positions[i] for i in valid_indices],
            'phoneme_participant_ids': [pids[i] for i in valid_indices],
            'phoneme_durations_samples': [data['phoneme_durations_samples'][i] for i in valid_indices] 
                                  if 'phoneme_durations_samples' in data else [],
            'spectrograms': [specs[i] for i in valid_indices] if specs else [],
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
        Handles different feature shapes automatically.
        """
        self.log(f"\nSubtracting baseline for {dataset_name}:")
        
        features = data['features']
        participants = data.get('phoneme_participant_ids', [])
        
        corrected_features = []
        corrected_count = 0
        skipped_count = 0
        
        for i, feat in enumerate(features):
            pid = participants[i] if i < len(participants) else None
            
            if pid in baseline_dict and baseline_dict[pid] is not None:
                baseline = baseline_dict[pid]
                
                # Get feature dimension (handle both 1D and 2D)
                if feat.ndim == 1:
                    n_feat_dim = feat.shape[0]
                else:
                    n_feat_dim = feat.shape[1]
                
                n_baseline_dim = baseline.shape[0]
                
                # Match dimensions
                if n_baseline_dim != n_feat_dim:
                    if n_baseline_dim < n_feat_dim:
                        # Pad baseline
                        padded_baseline = np.zeros(n_feat_dim)
                        padded_baseline[:n_baseline_dim] = baseline
                        baseline = padded_baseline
                    else:
                        # Trim baseline
                        baseline = baseline[:n_feat_dim]
                
                # Subtract
                if feat.ndim == 1:
                    corrected_feat = feat - baseline
                else:
                    corrected_feat = feat - baseline  # Broadcasting handles (n_frames, n_features) - (n_features,)
                
                corrected_features.append(corrected_feat)
                corrected_count += 1
            else:
                corrected_features.append(feat)
                skipped_count += 1
        
        data['features'] = corrected_features
        
        self.log(f"  Baseline corrected: {corrected_count}/{len(features)}")
        self.log(f"  Skipped (no baseline): {skipped_count}")
        
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
        """
        Extract baseline EEG from silence periods in audio.
        Uses the same feature extraction method as the main pipeline.
        """
        window_size = int(0.5 * self.config.audio_sr)  # 500ms
        silence_threshold = np.sqrt(np.mean(audio**2)) * 0.1
        
        baseline_features = []
        
        for i in range(0, len(audio) - window_size, window_size):
            window_rms = np.sqrt(np.mean(audio[i:i+window_size]**2))
            if window_rms < silence_threshold:
                eeg_start = int(i * len(eeg) / len(audio))
                eeg_end = int((i + window_size) * len(eeg) / len(audio))
                
                eeg_segment = eeg[eeg_start:eeg_end]
                
                # Skip if segment too short
                min_samples = int((self.config.window_length + self.config.frameshift) * self.config.eeg_sr)
                if len(eeg_segment) < min_samples:
                    continue
                
                # Extract features using the SAME method as main pipeline
                try:
                    if self.feature_extraction_method == 'high_gamma':
                        feat = extractHG(
                            eeg_segment, 
                            self.config.eeg_sr,
                            windowLength=self.config.window_length,
                            frameshift=self.config.frameshift
                        )
                    elif self.feature_extraction_method == 'band_powers':
                        feat = self.detector._extract_band_power_features(eeg_segment)
                    elif self.feature_extraction_method == 'hjorth':
                        feat = self.detector._extract_hjorth_features(eeg_segment)
                    elif self.feature_extraction_method == 'temporal_dynamics':
                        feat = self.detector._extract_temporal_dynamics_features(eeg_segment)
                    elif self.feature_extraction_method == 'band_power_hjorth':
                        feat = self.detector._extract_band_power_hjorth_features(eeg_segment)
                    else:
                        # Default to high gamma
                        feat = extractHG(
                            eeg_segment,
                            self.config.eeg_sr,
                            windowLength=self.config.window_length,
                            frameshift=self.config.frameshift
                        )
                    
                    if feat is not None and feat.shape[0] > 0:
                        # Average across time if multi-frame
                        if feat.ndim > 1 and feat.shape[0] > 1:
                            feat_avg = np.mean(feat, axis=0)
                        else:
                            feat_avg = feat.flatten()
                        baseline_features.append(feat_avg)
                        
                except Exception as e:
                    self.debug(f"Error extracting baseline features: {e}")
                    continue
        
        if baseline_features:
            baseline = np.mean(baseline_features, axis=0)
            self.debug(f"Baseline extracted: {len(baseline_features)} segments, shape {baseline.shape}")
        else:
            # Return zeros matching expected feature dimension
            if self.feature_extraction_method == 'band_powers':
                n_features = eeg.shape[1] * 6  # channels x 6 bands
            elif self.feature_extraction_method == 'hjorth':
                n_features = eeg.shape[1] * 3  # channels x 3 params
            elif self.feature_extraction_method == 'temporal_dynamics':
                n_features = eeg.shape[1] * 4  # channels x 4 features
            elif self.feature_extraction_method == 'band_power_hjorth':
                n_features = eeg.shape[1] * 9  # channels x (6 + 3)
            else:
                n_features = eeg.shape[1]  # high gamma: just channels
            
            baseline = np.zeros(n_features)
            self.debug(f"Warning: No usable silence blocks, using zero baseline shape {baseline.shape}")
        
        return baseline

    def _apply_pca_per_patient(self, train_data, test_data, pca_components):
        """Fit PCA on train, transform both train and test."""
        from sklearn.decomposition import PCA
        
        # Group by patient
        train_by_patient = {}
        for i, pid in enumerate(train_data['phoneme_participant_ids']):
            if pid not in train_by_patient:
                train_by_patient[pid] = []
            train_by_patient[pid].append(i)
        
        test_by_patient = {}
        for i, pid in enumerate(test_data['phoneme_participant_ids']):
            if pid not in test_by_patient:
                test_by_patient[pid] = []
            test_by_patient[pid].append(i)
        
        # Fit and transform per patient
        for pid in train_by_patient:
            train_indices = train_by_patient[pid]
            train_features = [train_data['features'][i] for i in train_indices]
            
            # Stack all frames for this patient's train data
            all_train = np.vstack(train_features)
            
            # Fit PCA on TRAIN only
            n_comp = min(pca_components, all_train.shape[1])
            pca = PCA(n_components=n_comp)
            pca.fit(all_train)
            
            self.log(f"PCA for {pid}: fit on {all_train.shape}, explained var={pca.explained_variance_ratio_.sum():.3f}")
            
            # Transform train
            for i in train_indices:
                train_data['features'][i] = pca.transform(train_data['features'][i])
            
            # Transform test using SAME PCA
            if pid in test_by_patient:
                for i in test_by_patient[pid]:
                    test_data['features'][i] = pca.transform(test_data['features'][i])
        
        return train_data, test_data
        

    