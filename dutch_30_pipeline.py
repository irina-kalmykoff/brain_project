from pipeline import UnifiedPhonemePipeline
from phonetic_dictionary import PhoneticDictionary
import os
import re
import json
import glob
import string

import gc
import pickle
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
from collections import Counter, defaultdict, OrderedDict
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
                        use_phoneme_groups=False, 
                        debug_mode=False, use_rms_boundaries=True, use_multifeature=False,
                        use_wav2vec=False, subtract_baseline=True, 
                        **kwargs):
        
        super().__init__(
            path_bids=dutch30_extractor.data_dir, 
            path_output=dutch30_extractor.results_dir,
            path_results=dutch30_extractor.results_dir,
            feature_extraction_method=feature_extraction_method,
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
        self.log(f"Pipeline initialized: {feature_extraction_method}, groups={use_phoneme_groups}")
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
    
    def step2_split_by_instances(self, train_fraction=None, random_seed=None):
        """Split each patient's word instances into train/test."""
        if train_fraction is None:
            train_fraction = self.config.default_train_fraction
        if random_seed is None:
            random_seed = self.config.default_random_seed

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
            
            patient_num = int(pid[1:])
            is_sentence_patient = patient_num > 20

            if is_sentence_patient:
                # Split by sentence presentation, not by word instance.
                # Group word instances by which sentence presentation they came from.
                sentence_to_word_instances = defaultdict(list)
                for word, word_data in word_segments['words'].items():
                    for inst_idx, instance in enumerate(word_data['instances']):
                        sent_key = (instance['sentence_text'], instance['sentence_idx'])
                        sentence_to_word_instances[sent_key].append((word, inst_idx))

                # Group sentence presentations by sentence text to find repetitions.
                text_to_presentations = defaultdict(list)
                for sent_key in sentence_to_word_instances:
                    sent_text, sent_idx = sent_key
                    text_to_presentations[sent_text].append(sent_key)

                # Split presentations of each sentence into train/test.
                train_presentations = set()
                test_presentations = set()
                for sent_text, presentations in text_to_presentations.items():
                    n_pres = len(presentations)
                    pres_indices = np.arange(n_pres)
                    np.random.shuffle(pres_indices)

                    if n_pres == 1:
                        if np.random.random() < train_fraction:
                            train_presentations.add(presentations[pres_indices[0]])
                        else:
                            test_presentations.add(presentations[pres_indices[0]])
                    else:
                        n_train = max(1, int(n_pres * train_fraction))
                        for idx in pres_indices[:n_train]:
                            train_presentations.add(presentations[idx])
                        for idx in pres_indices[n_train:]:
                            test_presentations.add(presentations[idx])

                # Map back to word instances.
                for sent_key in train_presentations:
                    for word, inst_idx in sentence_to_word_instances[sent_key]:
                        if word not in self.split_result['train'][pid]:
                            self.split_result['train'][pid][word] = []
                        self.split_result['train'][pid][word].append(inst_idx)

                for sent_key in test_presentations:
                    for word, inst_idx in sentence_to_word_instances[sent_key]:
                        if word not in self.split_result['test'][pid]:
                            self.split_result['test'][pid][word] = []
                        self.split_result['test'][pid][word].append(inst_idx)

            else:
                # Word/mixed patients: split by word instance (existing logic).
                for word, word_data in word_segments['words'].items():
                    num_instances = len(word_data['instances'])
                    if num_instances == 0:
                        continue

                    indices = np.arange(num_instances)
                    np.random.shuffle(indices)

                    if num_instances == 1:
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

            del raw_data, eeg, audio
            gc.collect()
        
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
        all_word_sentence_indices = []    
        all_word_sentence_texts = []     
        
        for sent_idx, sent_info in enumerate(sentence_list):
            sentence_text = sent_info['text']
            sent_stim_start = sent_info['stim_start_idx']
            sent_stim_end = sent_info['stim_end_idx']
            
            # -----------------------------------------------------------
            # 2A. Parse sentence text into individual words
            # -----------------------------------------------------------
            # Remove Unicode quotes and control characters from the sentence
            cleaned_sentence = re.sub(
                r'[\u2018\u2019\u201A\u201B\u2032\u2035'   # single curly/prime quotes
                r'\u201C\u201D\u201E\u201F\u2033\u2036'    # double curly/prime quotes
                r'"""„\'\'\r\n]+', '', sentence_text)
            # Strip all punctuation (ASCII + Unicode quotes) from each word
            strip_chars = string.punctuation + '\u2018\u2019\u201C\u201D\u201E\u201F""„"'''
            word_texts = [w.strip(strip_chars).lower()
                          for w in cleaned_sentence.split()
                          if w.strip(strip_chars)]
            
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
                all_word_sentence_indices.append(sent_idx)      
                all_word_sentence_texts.append(sentence_text)    
        
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
                'audio_segment': all_word_audio_segments[i],
                'sentence_idx': all_word_sentence_indices[i],      
                'sentence_text': all_word_sentence_texts[i],        
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
            'participant_id': participant_id,
            'sentence_list': sentence_list,                       
            'word_sentence_indices': all_word_sentence_indices,     
            'word_sentence_texts': all_word_sentence_texts,         
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

    def save_checkpoint_after_step3(pipeline, filepath='checkpoint_after_step3.pkl'):
        """Save pipeline state after steps 1-3 (load data, split, channel exclusions).
        
        This saves the heavy data that takes minutes to load from NWB files,
        so you can skip steps 1-3 on subsequent runs.
        """
        state = {
            'split_result': pipeline.split_result,
            'patient_data': getattr(pipeline, 'patient_data', None),
            'patient_baselines': getattr(pipeline, 'patient_baselines', None),
            'channel_exclusions': getattr(pipeline, 'channel_exclusions', None),
        }
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"Step 3 checkpoint saved: {filepath} ({size_mb:.1f} MB)")

    def load_checkpoint_after_step3(pipeline, filepath='checkpoint_after_step3.pkl'):
        """Load pipeline state from step 3 checkpoint.
        
        Returns True if loaded, False if file not found.
        After loading, continue with step4_custom_detector().
        """
        if not os.path.exists(filepath):
            print(f"No step 3 checkpoint found at {filepath}")
            return False
        
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        pipeline.split_result = state['split_result']
        if state['patient_data'] is not None:
            pipeline.patient_data = state['patient_data']
        if state['patient_baselines'] is not None:
            pipeline.patient_baselines = state['patient_baselines']
        if state['channel_exclusions'] is not None:
            pipeline.channel_exclusions = state['channel_exclusions']
        
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"Step 3 checkpoint loaded: {filepath} ({size_mb:.1f} MB)")
        return True

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
    
    def step5a_filter_by_frame_count(self, min_frames=2, max_frames=25):
        """Filter phonemes by HG frame count.

        Removes phonemes with too few or too many frames before
        stacking or resampling. Call after step5, before step5b.

        Args:
            min_frames: int, minimum HG frames to keep.
            max_frames: int, maximum HG frames to keep.
        """
        self.log(f"\nStep 5a: Filter by frame count "
                 f"(keep {min_frames}-{max_frames} frames)")

        for dataset_name, data in [('train', self.train), ('test', self.test)]:
            if data is None:
                continue

            n_before = len(data['features'])
            keep = []

            for i, feat in enumerate(data['features']):
                if hasattr(feat, 'ndim') and feat.ndim == 2:
                    n_frames = feat.shape[0]
                    if min_frames <= n_frames <= max_frames:
                        keep.append(i)
                else:
                    keep.append(i)

            for key in data:
                if isinstance(data[key], list) and len(data[key]) == n_before:
                    data[key] = [data[key][i] for i in keep]

            n_removed = n_before - len(keep)
            self.log(f"  {dataset_name}: {n_before} -> {len(keep)} "
                     f"(removed {n_removed})")
                     
    def step5b_normalize_feature_lengths(self, target_frames=10):
        """Resample all features to fixed frame count, then flatten.

        Alternative A for fixed-length features. Resamples each
        phoneme's HG array to target_frames, then flattens.
        One sample per phoneme.

        Call after step5_accumulate_data_dutch30.
        Mutually exclusive with step5b_stack_features.

        Args:
            target_frames: int, target number of frames.
        """
        from scipy.signal import resample

        self.log(f"\nStep 5b: Normalize feature lengths "
                 f"(target={target_frames} frames)")

        for dataset_name, data in [('train', self.train), ('test', self.test)]:
            if data is None:
                continue

            n_before = len(data['features'])
            normalized = []
            for feat in data['features']:
                if feat.ndim == 1:
                    normalized.append(feat)
                    continue
                if feat.shape[0] == target_frames:
                    normalized.append(feat.flatten())
                else:
                    resampled = resample(feat, target_frames, axis=0)
                    normalized.append(resampled.flatten())

            data['features'] = normalized

            n_channels = (normalized[0].shape[0] // target_frames
                          if normalized else 0)
            self.log(f"  {dataset_name}: {n_before} samples, "
                     f"feature dim = {target_frames} x {n_channels} "
                     f"= {target_frames * n_channels}")

    def step5b_stack_features(self, model_order=4, step_size=5):
        """Apply temporal context stacking per word instance.

        Alternative B for fixed-length features. Reconstructs the
        continuous HG stream per word instance, applies stackFeatures,
        then assigns each stacked frame its phoneme label. Produces
        multiple training samples per phoneme.

        Call after step5_accumulate_data_dutch30.
        Mutually exclusive with step5b_normalize_feature_lengths.

        Args:
            model_order: int, temporal context steps before and after.
                With step_size=5 and frameshift=0.01, model_order=4
                gives +/- 200ms context (9 frames stacked).
            step_size: int, frame skip between context steps.
        """
        from extract_features import stackFeatures

        n_context = 2 * model_order + 1
        margin = model_order * step_size

        self.log(f"\nStep 5b: Stack features "
                 f"(order={model_order}, step={step_size}, "
                 f"context={n_context} frames, "
                 f"+/- {margin * self.config.frameshift * 1000:.0f}ms)")

        for dataset_name, data in [('train', self.train), ('test', self.test)]:
            if data is None:
                continue

            features = data['features']
            labels = data['phoneme_labels']
            words = data['phoneme_words']
            positions = data['phoneme_positions']
            pids = data['phoneme_participant_ids']
            has_durations = ('phoneme_durations_samples' in data
                            and data['phoneme_durations_samples'])

            n_input = len(features)
            if n_input == 0:
                continue

            # Check features are 2D (not already flattened)
            if features[0].ndim == 1:
                self.log(f"  {dataset_name}: features already flattened, "
                         f"cannot stack. Run before normalize.")
                continue

            n_channels = features[0].shape[1]

            # Detect word instance boundaries
            instance_groups = []
            current_start = 0
            
            for i in range(1, n_input):
                is_boundary = (
                    positions[i] == 0
                    or words[i] != words[i - 1]
                    or pids[i] != pids[i - 1]
                )
                if is_boundary:
                    instance_groups.append((current_start, i))
                    current_start = i
            instance_groups.append((current_start, n_input))

            new_features = []
            new_instance_ids = []
            new_labels = []
            new_words = []
            new_positions = []
            new_pids = []
            new_durations = []
            n_skipped_short = 0
            n_skipped_unknown = 0

            instance_counter = 0
            for start, end in instance_groups:
                # Collect phoneme segments for this instance
                instance_feats = []
                instance_labels = []
                offset = 0
                skip_instance = False
                instance_id = f"{pids[start]}_{instance_counter}"
                instance_counter += 1

                for i in range(start, end):
                    feat = features[i]
                    if feat.ndim != 2:
                        skip_instance = True
                        break
                    n_frames = feat.shape[0]
                    instance_feats.append(feat)
                    instance_labels.append(
                        (labels[i], offset, offset + n_frames, i)
                    )
                    offset += n_frames

                if skip_instance or not instance_feats:
                    continue

                # Concatenate into continuous stream
                continuous = np.concatenate(instance_feats, axis=0)
                total_frames = continuous.shape[0]
                min_frames = 2 * margin + 1

                if total_frames < min_frames:
                    # Pad edges to allow stacking
                    pad_needed = min_frames - total_frames
                    pad_before = pad_needed // 2
                    pad_after = pad_needed - pad_before
                    continuous = np.pad(
                        continuous,
                        ((pad_before, pad_after), (0, 0)),
                        mode='edge',
                    )
                    # Adjust label offsets
                    instance_labels = [
                        (lbl, s + pad_before, e + pad_before, orig_i)
                        for lbl, s, e, orig_i in instance_labels
                    ]
                    adjusted_margin = margin
                else:
                    pad_before = 0
                    adjusted_margin = margin

                stacked = stackFeatures(
                    continuous,
                    modelOrder=model_order,
                    stepSize=step_size,
                )

                if stacked.shape[0] == 0:
                    n_skipped_short += 1
                    continue

                # Map each stacked frame to its phoneme label
                stacked_offset = adjusted_margin
                for frame_idx in range(stacked.shape[0]):
                    original_pos = stacked_offset + frame_idx
                    assigned = False
                    for lbl, seg_start, seg_end, orig_i in instance_labels:
                        if seg_start <= original_pos < seg_end:
                            if lbl == '?':
                                n_skipped_unknown += 1
                                assigned = True
                                break
                            new_features.append(stacked[frame_idx])
                            new_instance_ids.append(instance_id)
                            new_labels.append(lbl)
                            new_words.append(words[orig_i])
                            new_positions.append(positions[orig_i])
                            new_pids.append(pids[orig_i])
                            if has_durations:
                                new_durations.append(
                                    data['phoneme_durations_samples'][orig_i]
                                    if orig_i < len(data['phoneme_durations_samples'])
                                    else 0
                                )
                            assigned = True
                            break
                    if not assigned:
                        # Frame falls outside any phoneme (edge artifact)
                        pass

            expected_dim = n_channels * n_context
            multiplier = (len(new_features) / n_input
                          if n_input > 0 else 0)

            self.log(f"  {dataset_name}: {n_input} phonemes -> "
                     f"{len(new_features)} stacked frames "
                     f"({multiplier:.1f}x)")
            self.log(f"    Feature dim: {n_channels} ch x "
                     f"{n_context} context = {expected_dim}")
            self.log(f"    Instances: {len(instance_groups)}, "
                     f"skipped short: {n_skipped_short}, "
                     f"skipped unknown: {n_skipped_unknown}")

            data['features'] = new_features
            data['phoneme_instance_ids'] = new_instance_ids
            data['phoneme_labels'] = new_labels
            data['phoneme_words'] = new_words
            data['phoneme_positions'] = new_positions
            data['phoneme_participant_ids'] = new_pids
            if has_durations:
                data['phoneme_durations_samples'] = new_durations
    
    def checkpoint_after_step5(self, sample_fraction=None, step5b_method=None,
                            model_order=None, step_size=None, target_frames=None):
        """Save checkpoint after step 5 (before stacking/resampling).

        Features are still 2D (n_frames, n_channels) arrays,
        allowing experimentation with different step 5b options.

        Args:
            sample_fraction: float or None, for filename.

        Returns:
            str or None, filepath if saved.
        """
        if not hasattr(self, 'train') or self.train is None:
            self.log("WARNING: No training data to checkpoint")
            return None

        if 'features' not in self.train or not self.train['features']:
            self.log("WARNING: Training data is empty")
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fraction_str = f"_sample{int(sample_fraction*100)}" if sample_fraction else ""
        if step5b_method == 'stack' and model_order is not None and step_size is not None:
            step5b_str = f"_stack_o{model_order}_s{step_size}"
        elif step5b_method == 'normalize' and target_frames is not None:
            step5b_str = f"_norm_f{target_frames}"
        else:
            step5b_str = ""

        # Build filter info string from detector config
        filter_type = getattr(self.detector.config, 'wav2vec_smoothing_filter', 'gaussian')
        if filter_type == 'gaussian':
            sigma = getattr(self.detector.config, 'wav2vec_phoneme_sigma', 0.5)
            filter_str = f"_filt-gauss-s{sigma}"
        elif filter_type == 'savgol':
            win = getattr(self.detector.config, 'wav2vec_savgol_window', 7)
            poly = getattr(self.detector.config, 'wav2vec_savgol_polyorder', 3)
            filter_str = f"_filt-savgol-w{win}-p{poly}"
        elif filter_type == 'median':
            size = getattr(self.detector.config, 'wav2vec_median_size', 3)
            filter_str = f"_filt-median-k{size}"
        elif filter_type == 'none':
            filter_str = "_filt-none"
        else:
            filter_str = f"_filt-{filter_type}"

        filename = (f"pipeline_{self.feature_extraction_method}"
            f"{fraction_str}{step5b_str}{filter_str}_after_step5_{timestamp}.pkl")
        filepath = os.path.join(self.path_results, filename)

        self.log(f"Saving step5 checkpoint: {filename}")

        try:
            # Check if features are still 2D
            first_feat = self.train['features'][0]
            feat_info = (f"{first_feat.shape}" if hasattr(first_feat, 'shape')
                         else "unknown")

            metadata = {
                'method': self.feature_extraction_method,
                'sample_fraction': sample_fraction,
                'timestamp': timestamp,
                'stage': 'after_step5',
                'feature_shape': feat_info,
                'step5b_method': step5b_method,
                'model_order': model_order,
                'step_size': step_size,
                'target_frames': target_frames,
                'train_samples': len(self.train['features']),
                'test_samples': (len(self.test['features'])
                                 if self.test else 0),
                'smoothing_filter': filter_type,
                'smoothing_params': {
                    'sigma': getattr(self.detector.config, 'wav2vec_phoneme_sigma', None),
                    'savgol_window': getattr(self.detector.config, 'wav2vec_savgol_window', None),
                    'savgol_polyorder': getattr(self.detector.config, 'wav2vec_savgol_polyorder', None),
                    'median_size': getattr(self.detector.config, 'wav2vec_median_size', None),
                },
                'split_result': (self.split_result
                                 if hasattr(self, 'split_result') else None),
            }

            train_file = filepath.replace('.pkl', '_train.h5')
            self._save_data_to_h5(self.train, train_file)
            metadata['train_file'] = os.path.basename(train_file)

            if self.test:
                test_file = filepath.replace('.pkl', '_test.h5')
                self._save_data_to_h5(self.test, test_file)
                metadata['test_file'] = os.path.basename(test_file)

            with open(filepath, 'wb') as f:
                pickle.dump({'metadata': metadata}, f)

            self.log(f"Step5 checkpoint saved: {filename}")
            self.log(f"  Feature shape: {feat_info} (2D = ready for stacking/resampling)")
            return filepath

        except Exception as e:
            self.log(f"Error saving checkpoint: {e}")
            return None
            
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
        """Build the grouped-label columns alongside the raw labels.

        After this step every dataset dict has three label keys:

        - ``phoneme_labels_raw``   – always the original per-phoneme labels
        - ``phoneme_labels_grouped`` – always the group-mapped labels
        - ``phoneme_labels``       – alias pointing at whichever is active

        The active view defaults to **grouped** for backward compatibility.
        Call :meth:`use_phoneme_labels` / :meth:`use_group_labels` to switch
        without re-running the step.

        Safe to call more than once — raw labels are preserved on the first
        call and never overwritten.
        """
        self.log("Step 8: Grouping phonemes.")

        # Get phoneme-to-group mapping from phonetic dictionary
        if not hasattr(self.detector.phonetic_dict, 'phoneme_to_group'):
            self.detector.phonetic_dict.add_phoneme_groups()

        phoneme_to_group = self.detector.phonetic_dict.phoneme_to_group

        for dataset_name in ['train', 'test', 'val']:
            if not hasattr(self, dataset_name):
                continue

            data = getattr(self, dataset_name)

            if 'phoneme_labels' not in data:
                continue

            # Preserve the original raw labels exactly once.
            if 'phoneme_labels_raw' not in data:
                data['phoneme_labels_raw'] = data['phoneme_labels'].copy()

            # Always build groups from the raw labels (never from a
            # previous grouped version), so this is idempotent.
            grouped_labels = []
            unknown_count = 0

            for label in data['phoneme_labels_raw']:
                if label in phoneme_to_group:
                    grouped_labels.append(phoneme_to_group[label])
                elif label in ('?', 'unknown', ''):
                    grouped_labels.append('unknown')
                    unknown_count += 1
                else:
                    grouped_labels.append('unknown')
                    unknown_count += 1

            data['phoneme_labels_grouped'] = grouped_labels

            # Default active view: grouped (backward compatible).
            data['phoneme_labels'] = grouped_labels

            # Log statistics
            unique_raw = len(set(data['phoneme_labels_raw']))
            unique_grouped = len(set(grouped_labels))
            self.log(f"  {dataset_name}: {unique_raw} raw phonemes -> {unique_grouped} groups")

            if unknown_count > 0:
                self.log(f"    {unknown_count} samples mapped to 'unknown'")

        self.phonemes_grouped = True

        # Log group distribution for train
        if hasattr(self, 'train') and 'phoneme_labels' in self.train:
            from collections import Counter
            group_counts = Counter(self.train['phoneme_labels'])
            self.log(f"  Train group distribution:")
            for group, count in sorted(group_counts.items(), key=lambda x: -x[1]):
                self.log(f"    {group}: {count}")

        self.log("Step 8 complete: Phonemes grouped")

    def use_phoneme_labels(self):
        """Switch ``phoneme_labels`` to the raw per-phoneme labels."""
        for dataset_name in ['train', 'test', 'val']:
            if not hasattr(self, dataset_name):
                continue
            data = getattr(self, dataset_name)
            if 'phoneme_labels_raw' in data:
                data['phoneme_labels'] = data['phoneme_labels_raw']
        self.phonemes_grouped = False
        self.log("Switched to raw phoneme labels")

    def use_group_labels(self):
        """Switch ``phoneme_labels`` to the grouped labels."""
        for dataset_name in ['train', 'test', 'val']:
            if not hasattr(self, dataset_name):
                continue
            data = getattr(self, dataset_name)
            if 'phoneme_labels_grouped' in data:
                data['phoneme_labels'] = data['phoneme_labels_grouped']
            else:
                self.log(f"  Warning: {dataset_name} has no grouped labels — run step8_group_phonemes() first")
                return
        self.phonemes_grouped = True
        self.log("Switched to grouped phoneme labels")
        
    def step9_train_and_evaluate(self, model_factory, model_params=None,
                                 use_viterbi=True, min_train=10, min_test=5):
        """Train a per-patient model and evaluate on test set.

        Model-agnostic: accepts any callable that returns a model with
        train(features, phoneme_labels) and predict(features) methods.

        When use_viterbi is True and phoneme_instance_ids are available,
        predictions are grouped by word instance. Classifier probabilities
        are averaged per phoneme position within each instance, then
        Viterbi decodes the phoneme-level sequence using corpus transition
        probabilities.

        Args:
            model_factory: callable that accepts **model_params and returns
                a model instance. Must support train() and predict().
            model_params: dict of keyword arguments passed to model_factory.
                Defaults to empty dict.
            use_viterbi: bool, passed to model.predict() if supported.
            min_train: int, skip patients with fewer training samples.
            min_test: int, skip patients with fewer test samples.

        Returns:
            dict mapping patient_id to result dict with keys: model,
            accuracy, train_size, test_size, n_classes, predictions,
            true_labels.
        """
        if model_params is None:
            model_params = {}

        self.patient_results = {}
        self.model_factory = model_factory
        self.model_params = model_params

        patient_ids = sorted(set(self.train["phoneme_participant_ids"]))

        self.log(f"Step 9: Training {model_factory.__name__} "
                 f"on {len(patient_ids)} patients")
        self.log(f"  Params: {model_params}")

        for pid in patient_ids:
            train_mask = [
                p == pid for p in self.train["phoneme_participant_ids"]
            ]
            test_mask = [
                p == pid for p in self.test["phoneme_participant_ids"]
            ]

            train_feat = [
                self.train["features"][i]
                for i, m in enumerate(train_mask) if m
            ]
            train_labels = [
                self.train["phoneme_labels"][i]
                for i, m in enumerate(train_mask) if m
            ]
            test_indices = [i for i, m in enumerate(test_mask) if m]
            test_feat = [self.test["features"][i] for i in test_indices]
            test_labels = [self.test["phoneme_labels"][i] for i in test_indices]

            if len(train_feat) < min_train or len(test_feat) < min_test:
                self.log(f"  {pid}: skipped (train={len(train_feat)}, "
                         f"test={len(test_feat)})")
                continue

            # Create and train model
            model = model_factory(**model_params)
            model.train(features=train_feat, phoneme_labels=train_labels)

            has_instance_ids = 'phoneme_instance_ids' in self.test
            preds_no_viterbi = None

            if use_viterbi and has_instance_ids:
                # Group test samples by word instance for sequential decoding
                instance_order = OrderedDict()
                for list_pos, data_idx in enumerate(test_indices):
                    inst_id = self.test['phoneme_instance_ids'][data_idx]
                    if inst_id not in instance_order:
                        instance_order[inst_id] = []
                    instance_order[inst_id].append(list_pos)

                # Predict each word instance as a sequence
                preds = [None] * len(test_feat)
                preds_no_viterbi = [None] * len(test_feat)
                for inst_id, positions in instance_order.items():
                    inst_features = [test_feat[p] for p in positions]
                    inst_positions = [
                        self.test['phoneme_positions'][test_indices[p]]
                        for p in positions
                    ]

                    # Get raw classifier probabilities (no Viterbi)
                    inst_preds, inst_probs = model.predict(
                        inst_features, use_viterbi=False
                    )
                    
                    for p, pred in zip(positions, inst_preds):
                        preds_no_viterbi[p] = pred

                    # Aggregate probabilities per phoneme position
                    pos_probs = OrderedDict()
                    pos_frames = OrderedDict()
                    for frame_idx, pos in enumerate(inst_positions):
                        if pos not in pos_probs:
                            pos_probs[pos] = []
                            pos_frames[pos] = []
                        pos_probs[pos].append(inst_probs[frame_idx])
                        pos_frames[pos].append(frame_idx)

                    # Average probabilities per position
                    agg_probs = np.array([
                        np.mean(pos_probs[pos], axis=0)
                        for pos in pos_probs
                    ])

                    # Run Viterbi on phoneme-level sequence
                    path = model._viterbi_decode(agg_probs)
                    phoneme_preds = [
                        model.trained_classes[idx] for idx in path
                    ]

                    # Map back to frame-level predictions
                    for pos_idx, pos in enumerate(pos_probs):
                        for frame_idx in pos_frames[pos]:
                            preds[positions[frame_idx]] = phoneme_preds[pos_idx]
            else:
                # Flat prediction without Viterbi
                try:
                    preds, probs = model.predict(
                        test_feat, use_viterbi=False
                    )
                except TypeError:
                    preds, probs = model.predict(test_feat)

            # Flatten nested predictions if needed
            if preds and isinstance(preds[0], list):
                preds = [
                    p[0] if len(p) > 0 else "?" for p in preds
                ]
            preds = [
                str(p) if not isinstance(p, str) else p for p in preds
            ]

            # Calculate accuracy
            correct = sum(
                1 for p, t in zip(preds, test_labels) if p == t
            )
            accuracy = correct / len(test_labels)

            # Chance level for this patient
            label_counts = Counter(test_labels)
            chance = max(label_counts.values()) / len(test_labels)
            lift = accuracy / chance if chance > 0 else 0

            self.patient_results[pid] = {
                "model": model,
                "accuracy": accuracy,
                "chance": chance,
                "lift": lift,
                "train_size": len(train_feat),
                "test_size": len(test_feat),
                "n_classes": len(set(train_labels)),
                "predictions": preds,
                "predictions_no_viterbi": preds_no_viterbi if use_viterbi and has_instance_ids else None,
                "true_labels": test_labels,
            }

            self.log(f"  {pid}: acc={accuracy:.3f} "
                     f"chance={chance:.3f} lift={lift:.2f}x "
                     f"({len(set(train_labels))} classes)")

        # Summary
        accs = [r["accuracy"] for r in self.patient_results.values()]
        lifts = [r["lift"] for r in self.patient_results.values()]

        # Group means
        groups = {"P01-P10": [], "P11-P20": [], "P21-P30": []}
        for pid, r in self.patient_results.items():
            num = int(pid[1:])
            if num <= 10:
                groups["P01-P10"].append(r["accuracy"])
            elif num <= 20:
                groups["P11-P20"].append(r["accuracy"])
            else:
                groups["P21-P30"].append(r["accuracy"])

        self.log(f"\n  Overall: {np.mean(accs):.4f} +/- {np.std(accs):.4f}")
        self.log(f"  Mean lift: {np.mean(lifts):.2f}x")
        for group, group_accs in groups.items():
            if group_accs:
                self.log(f"  {group}: {np.mean(group_accs):.4f}")

        self.log("Step 9 complete")

        return self.patient_results

    def step10_visualize_patient(self, pid, show_table=True, show_predictions=False, preds_no_viterbi=None, min_class_samples=5):
        """Visualize per-patient classification results.

        Produces a figure with train/test distribution, per-phoneme
        precision/recall, and confusion matrices. When preds_no_viterbi
        is provided, adds pre/post Viterbi comparison panels.

        Args:
            pid: str, patient ID to visualize.
            show_table: bool, print per-phoneme metrics table.
            show_predictions: bool, print word-level prediction comparison.
            preds_no_viterbi: list or None, predictions before Viterbi
                decoding. When provided, shows pre/post comparison.

        Requires step 9 to have been run first.
        """
        from sklearn.metrics import confusion_matrix
        from matplotlib.patches import Rectangle
        from collections import Counter
        import matplotlib.pyplot as plt

        if not hasattr(self, "patient_results") or pid not in self.patient_results:
            self.log(f"{pid}: no results found. Run step 9 first.")
            return

        train_mask = [p == pid for p in self.train["phoneme_participant_ids"]]
        test_mask = [p == pid for p in self.test["phoneme_participant_ids"]]
        train_labels = [
            self.train["phoneme_labels"][i]
            for i, m in enumerate(train_mask) if m
        ]

        preds = self.patient_results[pid]["predictions"]
        test_labels = self.patient_results[pid]["true_labels"]

        if preds_no_viterbi is None and "predictions_no_viterbi" in self.patient_results[pid]:
            preds_no_viterbi = self.patient_results[pid]["predictions_no_viterbi"]

        has_viterbi_comparison = preds_no_viterbi is not None

        # build confusion data from post-viterbi predictions
        confusion_data = {}
        for true_label, pred_label in zip(test_labels, preds):
            if true_label not in confusion_data:
                confusion_data[true_label] = Counter()
            confusion_data[true_label][pred_label] += 1

        train_counts = Counter(train_labels)
        test_counts = Counter(test_labels)
        all_labels = sorted(set(list(train_counts.keys()) + list(test_counts.keys())))
        test_phonemes = sorted(test_counts.keys())
        unique_labels = sorted(set(list(test_labels) + list(preds)))

        def compute_metrics(predictions, phonemes, true_labels):
            """Compute per-phoneme precision and recall.

            Args:
                predictions: list of predicted labels.
                phonemes: list of phoneme labels to evaluate.
                true_labels: list of true labels.

            Returns:
                dict mapping phoneme to dict with recall, precision, support.
            """
            metrics = {}
            for p in phonemes:
                true_mask = [l == p for l in true_labels]
                correct = sum(
                    1 for i, m in enumerate(true_mask) if m and predictions[i] == p
                )
                total_true = sum(true_mask)
                recall = correct / total_true if total_true > 0 else 0

                pred_mask = [pr == p for pr in predictions]
                total_pred = sum(pred_mask)
                precision = correct / total_pred if total_pred > 0 else 0

                metrics[p] = {
                    "recall": recall,
                    "precision": precision,
                    "support": total_true,
                }
            return metrics

        phoneme_metrics_post = compute_metrics(preds, test_phonemes, test_labels)

        if has_viterbi_comparison:
            phoneme_metrics_pre = compute_metrics(
                preds_no_viterbi, test_phonemes, test_labels
            )

        # confusion matrices post-viterbi
        cm = confusion_matrix(test_labels, preds, labels=unique_labels)
        cm_recall = cm.astype("float") / (cm.sum(axis=1, keepdims=True) + 1e-10)
        cm_precision = cm.astype("float") / (cm.sum(axis=0, keepdims=True) + 1e-10)

        # confusion matrix pre-viterbi
        if has_viterbi_comparison:
            unique_labels_pre = sorted(set(list(test_labels) + list(preds_no_viterbi)))
            cm_pre = confusion_matrix(
                test_labels, preds_no_viterbi, labels=unique_labels_pre
            )
            cm_pre_recall = cm_pre.astype("float") / (
                cm_pre.sum(axis=1, keepdims=True) + 1e-10
            )

        # layout
        n_cols = 3 if has_viterbi_comparison else 2
        fig, axes = plt.subplots(2, n_cols, figsize=(8 * n_cols, 12))

        acc = self.patient_results[pid]["accuracy"]
        lift = self.patient_results[pid]["lift"]

        if has_viterbi_comparison:
            correct_pre = sum(1 for p, t in zip(preds_no_viterbi, test_labels) if p == t)
            acc_pre = correct_pre / len(test_labels)
            n_classes = len(set(test_labels))
            chance = 1.0 / n_classes if n_classes > 0 else 0
            lift_pre = acc_pre / chance if chance > 0 else 0
            fig.suptitle(
                f"{pid} - pre-Viterbi: Accuracy={acc_pre:.3f}, Lift={lift_pre:.2f}x"
                f"   |   post-Viterbi: Accuracy={acc:.3f}, Lift={lift:.2f}x",
                fontsize=13, fontweight="bold",
            )
        else:
            fig.suptitle(
                f"{pid} - Accuracy: {acc:.3f}, Lift: {lift:.2f}x",
                fontsize=14, fontweight="bold",
            )

        # 1. train/test distribution
        ax1 = axes[0, 0]
        x = np.arange(len(all_labels))
        width = 0.35
        ax1.bar(
            x - width / 2,
            [train_counts.get(p, 0) for p in all_labels],
            width, label="Train", color="cornflowerblue",
        )
        ax1.bar(
            x + width / 2,
            [test_counts.get(p, 0) for p in all_labels],
            width, label="Test", color="coral",
        )
        ax1.axhline(
            min_class_samples,
            color="red", linestyle="--", linewidth=1.5,
            label=f"min samples ({min_class_samples})"
        )
        ax1.set_xticks(x)
        ax1.set_xticklabels(all_labels, rotation=90, fontsize=8)
        ax1.set_title(
            f"Distribution (train={len(train_labels)}, test={len(test_labels)})"
        )
        ax1.set_ylabel("Count")
        ax1.legend()

        def draw_metrics(ax, metrics, phonemes, overall_acc, title):
            """Draw per-phoneme precision/recall bar chart.

            Args:
                ax: matplotlib axis to draw on.
                metrics: dict of per-phoneme metrics.
                phonemes: ordered list of phonemes to plot.
                overall_acc: float, overall accuracy for reference line.
                title: str, axis title.
            """
            x = np.arange(len(phonemes))
            width = 0.35
            recalls = [metrics[p]["recall"] for p in phonemes]
            precisions = [metrics[p]["precision"] for p in phonemes]
            ax.bar(x - width / 2, recalls, width, label="Recall", color="steelblue")
            ax.bar(x + width / 2, precisions, width, label="Precision", color="darkorange")
            ax.set_xticks(x)
            ax.set_xticklabels(phonemes, rotation=90, fontsize=8)
            ax.set_title(title)
            ax.set_ylim([0, 1])
            ax.axhline(
                overall_acc, color="red", linestyle="--", alpha=0.5, label="Overall Acc"
            )
            ax.legend(loc="upper right", fontsize=8)
            ax.set_ylabel("Score")

        # 2. per-class metrics
        if has_viterbi_comparison:
            draw_metrics(
                axes[0, 1], phoneme_metrics_pre, test_phonemes, acc_pre,
                "Per-Class Metrics (pre-Viterbi)"
            )
            draw_metrics(
                axes[0, 2], phoneme_metrics_post, test_phonemes, acc,
                "Per-Class Metrics (post-Viterbi)"
            )
        else:
            draw_metrics(
                axes[0, 1], phoneme_metrics_post, test_phonemes, acc,
                "Per-Class Metrics"
            )

        def draw_confusion(ax, matrix, matrix_norm, labels, title, cmap, cbar_label):
            """Draw a normalised confusion matrix.

            Args:
                ax: matplotlib axis to draw on.
                matrix: raw count confusion matrix.
                matrix_norm: normalised confusion matrix.
                labels: list of label strings.
                title: str, axis title.
                cmap: str, matplotlib colormap name.
                cbar_label: str, colorbar label.
            """
            im = ax.imshow(matrix_norm, cmap=cmap, vmin=0, vmax=1)
            n = len(labels)
            fs = max(5, min(8, 100 // n))
            for i in range(n):
                for j in range(n):
                    val = matrix[i, j]
                    if val > 0:
                        color = "white" if matrix_norm[i, j] > 0.5 else "black"
                        ax.text(
                            j, i, str(val), ha="center", va="center",
                            color=color, fontsize=fs,
                        )
            for i in range(n):
                ax.add_patch(Rectangle(
                    (i - 0.5, i - 0.5), 1, 1,
                    fill=False, edgecolor="red", linewidth=1,
                ))
            ax.set_xticks(range(n))
            ax.set_yticks(range(n))
            ax.set_xticklabels(labels, rotation=90, fontsize=fs)
            ax.set_yticklabels(labels, fontsize=fs)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_title(title)
            plt.colorbar(im, ax=ax, label=cbar_label, fraction=0.046)

        # 3. confusion matrix recall-normalised pre-viterbi
        if has_viterbi_comparison:
            draw_confusion(
                axes[1, 0], cm_pre, cm_pre_recall, unique_labels_pre,
                "Confusion Matrix - Recall normalised (pre-Viterbi)",
                "Blues", "Recall"
            )
        else:
            draw_confusion(
                axes[1, 0], cm, cm_recall, unique_labels,
                "Confusion Matrix - Recall normalised",
                "Blues", "Recall"
            )

        # 4. confusion matrix recall-normalised post-viterbi
        draw_confusion(
            axes[1, 1], cm, cm_recall, unique_labels,
            "Confusion Matrix - Recall normalised (post-Viterbi)"
            if has_viterbi_comparison else "Confusion Matrix - Recall normalised",
            "Blues", "Recall"
        )

        # 5. confusion matrix precision-normalised post-viterbi
        if has_viterbi_comparison:
            draw_confusion(
                axes[1, 2], cm, cm_precision, unique_labels,
                "Confusion Matrix - Precision normalised (post-Viterbi)",
                "Greens", "Precision"
            )


        plt.tight_layout()
        plt.show()

        # print table
        if show_table:
            self.log(f"{pid} - per-class metrics (post-Viterbi)")
            self.log(
                f"{'label':<12} {'recall':<8} {'prec':<8} "
                f"{'count':<8} {'top 3 confusions'}"
            )
            for p in test_phonemes:
                m = phoneme_metrics_post[p]
                if p in confusion_data:
                    confusions = confusion_data[p].copy()
                    confusions.pop(p, None)
                    top = confusions.most_common(3)
                    conf_str = ", ".join(
                        [f"{pred}({cnt})" for pred, cnt in top]
                    ) if top else "-"
                else:
                    conf_str = "-"
                self.log(
                    f"{p:<12} {m['recall']:>6.2f}  {m['precision']:>6.2f}  "
                    f"{m['support']:>6}  {conf_str}"
                )
            mean_recall = np.mean([m["recall"] for m in phoneme_metrics_post.values()])
            mean_precision = np.mean([m["precision"] for m in phoneme_metrics_post.values()])
            self.log(f"{'macro':<12} {mean_recall:>6.2f}  {mean_precision:>6.2f}")

        if show_predictions:
            self.log(f"{pid} - predictions vs true labels")
            words = [
                self.test["phoneme_words"][i]
                for i, m in enumerate(test_mask) if m
            ]
            current_word = None
            word_trues = []
            word_preds = []
            self.log(f"{'word/sentence':<40} {'true':<15} {'pred':<15} {'match'}")
            for i, (w, t, p) in enumerate(zip(words, test_labels, preds)):
                if w != current_word:
                    if current_word is not None:
                        display = current_word[:38]
                        for j, (tr, pr) in enumerate(zip(word_trues, word_preds)):
                            match = "ok" if tr == pr else ""
                            if j == 0:
                                print(f"{display:<40} {tr:<15} {pr:<15} {match}")
                            else:
                                print(f"{'':<40} {tr:<15} {pr:<15} {match}")
                    current_word = w
                    word_trues = []
                    word_preds = []
                word_trues.append(t)
                word_preds.append(p)
            if current_word is not None:
                display = current_word[:38]
                for j, (tr, pr) in enumerate(zip(word_trues, word_preds)):
                    match = "ok" if tr == pr else ""
                    if j == 0:
                        print(f"{display:<40} {tr:<15} {pr:<15} {match}")
                    else:
                        print(f"{'':<40} {tr:<15} {pr:<15} {match}")
            correct = sum(1 for t, p in zip(test_labels, preds) if t == p)
            self.log(
                f"\n{correct}/{len(test_labels)} correct "
                f"({correct/len(test_labels)*100:.1f}%)"
            )
        
    def step10_visualize_group(self, patient_ids=None, show_table=False):
        """Visualize results for a group of patients.

        Args:
            patient_ids: list of str, patient IDs to visualize.
                Defaults to all patients in patient_results.
                Can also pass a group name: 'mixed', 'word', 'sentence'.
            show_table: bool, print per-class metrics table for each.
        """
        if not hasattr(self, "patient_results"):
            self.log("No results found. Run step 9 first.")
            return

        # Handle group shortcuts
        if isinstance(patient_ids, str):
            if patient_ids == "mixed":
                patient_ids = [f"P{i:02d}" for i in range(1, 11) if i != 5]
            elif patient_ids == "word":
                patient_ids = [f"P{i:02d}" for i in range(11, 21) if i not in (18, 19)]
            elif patient_ids == "sentence":
                patient_ids = [f"P{i:02d}" for i in range(21, 31)]
            else:
                patient_ids = [patient_ids]

        if patient_ids is None:
            patient_ids = sorted(self.patient_results.keys())

        # Filter to patients that have results
        valid = [p for p in patient_ids if p in self.patient_results]
        skipped = [p for p in patient_ids if p not in self.patient_results]

        if skipped:
            self.log(f"Skipped (no results): {skipped}")

        for pid in valid:
            self.step10_visualize_patient(pid, show_table=show_table)
        
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
        # Detect which step5b was used
        first_feat = self.train['features'][0]
        if hasattr(first_feat, 'ndim') and first_feat.ndim == 2:
            step5b_str = "_raw"
        elif hasattr(first_feat, 'shape'):
            dim = first_feat.shape[0]
            # Guess from dimension
            step5b_str = f"_dim{dim}"
        else:
            dim = len(first_feat)
            step5b_str = f"_dim{dim}"

        filename = (f"pipeline_{self.feature_extraction_method}"
                    f"{fraction_str}{step5b_str}"
                    f"_after_step6_{timestamp}.pkl")
                    
        filepath = os.path.join(self.path_results, filename)
        
        self.log(f"Saving checkpoint: {filename}")
        
        try:
            metadata = {
                'method': self.feature_extraction_method,
                'sample_fraction': sample_fraction,
                'step5b_info': step5b_str,
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

    def try_load_checkpoint(self, sample_fraction=None, stage=None,
                         step5b_method=None, model_order=None,
                         step_size=None, target_frames=None):
        """Load checkpoint matching current configuration.

        Args:
            sample_fraction: float or None, filter by sample fraction.
            stage: str or None, 'after_step5' or 'after_step6'.
                If None, tries step6 first, then step5.

        Returns:
            bool, True if loaded.
        """
        fraction_str = (f"_sample{int(sample_fraction*100)}"
                if sample_fraction else "")

        if step5b_method == 'stack' and model_order is not None and step_size is not None:
            step5b_str = f"_stack_o{model_order}_s{step_size}"
        elif step5b_method == 'normalize' and target_frames is not None:
            step5b_str = f"_norm_f{target_frames}"
        else:
            step5b_str = "*"

        # Build filter pattern for matching
        filter_type = getattr(self.detector.config, 'wav2vec_smoothing_filter', 'gaussian')
        if filter_type == 'gaussian':
            sigma = getattr(self.detector.config, 'wav2vec_phoneme_sigma', 0.5)
            filter_str = f"_filt-gauss-s{sigma}"
        elif filter_type == 'savgol':
            win = getattr(self.detector.config, 'wav2vec_savgol_window', 7)
            poly = getattr(self.detector.config, 'wav2vec_savgol_polyorder', 3)
            filter_str = f"_filt-savgol-w{win}-p{poly}"
        elif filter_type == 'median':
            size = getattr(self.detector.config, 'wav2vec_median_size', 3)
            filter_str = f"_filt-median-k{size}"
        elif filter_type == 'none':
            filter_str = "_filt-none"
        else:
            filter_str = "_filt-*"

        if stage is not None:
            stages = [stage]
        else:
            stages = ['after_step6', 'after_step5']

        for try_stage in stages:
            pattern = (f"pipeline_{self.feature_extraction_method}"
                       f"{fraction_str}{step5b_str}{filter_str}_{try_stage}_*.pkl")
            matching_files = glob.glob(
                os.path.join(self.path_results, pattern)
            )

            if not matching_files:
                continue

            matching_files.sort(
                key=lambda f: os.path.getmtime(f), reverse=True
            )
            newest_checkpoint = matching_files[0]

            try:
                self.log(f"Loading checkpoint: "
                         f"{os.path.basename(newest_checkpoint)}")

                with open(newest_checkpoint, 'rb') as f:
                    data = pickle.load(f)

                metadata = data.get('metadata', {})
                self.split_result = metadata.get('split_result', None)

                if 'train_file' in metadata:
                    train_file = os.path.join(
                        self.path_results, metadata['train_file']
                    )
                    self.train = self._load_data_from_h5(train_file)

                if 'test_file' in metadata:
                    test_file = os.path.join(
                        self.path_results, metadata['test_file']
                    )
                    self.test = self._load_data_from_h5(test_file)

                val_file = newest_checkpoint.replace('.pkl', '_val.h5')
                if os.path.exists(val_file):
                    self.val = self._load_data_from_h5(val_file)

                loaded_stage = metadata.get('stage', 'unknown')
                step5b_info = metadata.get('step5b_info', '')
                feat_shape = metadata.get('feature_shape', '')

                self.log(f"Checkpoint loaded:")
                self.log(f"  Stage: {loaded_stage}")
                self.log(f"  Train: {len(self.train['features'])} samples")
                self.log(f"  Test: {len(self.test['features'])} samples")
                if step5b_info:
                    self.log(f"  Step5b: {step5b_info}")
                if feat_shape:
                    self.log(f"  Feature shape: {feat_shape}")

                return True

            except Exception as e:
                self.log(f"Error loading checkpoint: {e}")
                return False

        self.log(f"No checkpoint found for "
                 f"{self.feature_extraction_method}, "
                 f"sample={sample_fraction}, stage={stage}")
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
    