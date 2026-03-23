# Converted from parse_features_of_30_patients_wav2vec information loss.ipynb

import os
import gc
import glob
import json
#import h5py
import numpy as np
import pickle
import pandas as pd
#from IPython.display import Audio, display
from collections import Counter, defaultdict
from pynwb import NWBHDF5IO
from datetime import datetime
import scipy.signal
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, silhouette_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cosine, euclidean
from scipy.signal import decimate

from extract_features import extractHG, stackFeatures, downsampleLabels
from brain_audio_decoder import BrainAudioDecoder
from custom_decoder import CustomBrainAudioDecoder
from brain_audio_decoder_viz import BrainAudioDecoderViz
from acoustic_change_detector import AcousticChangeDetector
from phoneme_validator import PhonemeValidator
from phonetic_dictionary import PhoneticDictionary
#from feature_vizualizer import PhonemeFeatureVisualizer
from markov_phoneme_model import MarkovPhonemeModel
from extract_features import extractHG, downsampleLabels, extractMelSpecs
from pipeline import UnifiedPhonemePipeline
from config import BIDS_PATH, OUTPUT_PATH, RESULTS_PATH, DUTCH_30_PATH, DUTCH_10_PATH, get_dataset_paths

from dutch_30_pipeline import Dutch30Pipeline
from dutch_30_feature_extractor import Dutch30FeatureExtractor
from phoneme_detection_diagnostic import Dutch30PhonemeDetectionDiagnostic 
from dataset_config import Dutch30Config

from transformers import Wav2Vec2Model, Wav2Vec2Processor
import torch

dutch30_dir = DUTCH_30_PATH

# List all .npy files for one patient
patient_files = glob.glob(os.path.join(dutch30_dir, 'P01*.npy'))

# Check we're using the right paths
print(f"BIDS path: {BIDS_PATH}")
print(f"Output path: {OUTPUT_PATH}")
print(f"Results path: {RESULTS_PATH}")
# Define paths
path_bids = BIDS_PATH # './SingleWordProductionDutch-iBIDS'  # Path to the BIDS dataset
path_output = OUTPUT_PATH #'./features'  # Path to save extracted features
path_results = RESULTS_PATH #'./results'  # Path to save results
paths_30 = get_dataset_paths('dutch30')

extractor = Dutch30FeatureExtractor()

pipeline = Dutch30Pipeline(
        dutch30_extractor=extractor,
        debug_mode=False,
        pca_components= None, #100,
        feature_extraction_method = 'high_gamma',# 'high_gamma', #'band_powers', #'band_power_hjorth', # 'hjorth', #'band_powers',# 'hjorth', #'high_gamma', # 'band_powers', # 'band_power_hjorth'
        use_rms_boundaries=False,   
        use_multifeature=False,
        use_wav2vec=True,
        subtract_baseline=False,
        #baseline_method = 'band_powers' #'feature_matched', 'band_powers', 'raw'
    )

sample_fraction = 1
patient_range = (1,30)

# Try to load checkpoint
print(f"Attempting to load checkpoint (sample_fraction={sample_fraction})...")
    
if pipeline.try_load_checkpoint(sample_fraction=sample_fraction):
    print(f"Checkpoint loaded successfully!")
    print(f"  Train samples: {len(pipeline.train.get('features', []))}")
    print(f"  Test samples: {len(pipeline.test.get('features', []))}")
    
else: # No checkpoint found - run all steps
    print(f"No checkpoint found. Running pipeline steps...")
    
    print(f"\n  Step 1: Loading data (patients {patient_range})...")

    # Load pre-trained wav2vec model
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base", use_safetensors=True)
    config = Dutch30Config()
    extractor = Dutch30FeatureExtractor()
    pipeline.step1_load_dutch30_data(patient_range=(1,30))
    pipeline.split_result = None
    pipeline.step2_split_by_instances();
    pipeline.print_channel_counts()
    pipeline.step3_load_channel_exclusions('channel_exclusions.json')
    pipeline.apply_channel_exclusions()
    pipeline.print_channel_counts()
    pipeline.step4_custom_detector()
    pipeline.step5_accumulate_data_dutch30()
    #pipeline.step5b_normalize_lengths(target_frames=10, use_augmentation=True, balance_classes=True, n_chunks=5)
    #pipeline.step5b_normalize_lengths(target_frames=10, use_augmentation=False, balance_classes=True, n_chunks=5)
    pipeline.dutch30_step6_resolve_unknowns()
    #pipeline.checkpoint_after_step6(sample_fraction=sample_fraction)
    pipeline.step7_filter_unknowns(unknown_keep_ratio=0.0025);
    

    print(f"  Train samples: {len(pipeline.train.get('features', []))}")
    print(f"  Test samples: {len(pipeline.test.get('features', []))}")   

from scipy.signal import resample

# Pick target based on distribution
target = 10 

def normalize_feature_lengths(data, target_frames):
    """Resample all features to fixed frame count, then flatten.
    
    Args:
        data: dict with 'features' list of (n_frames, n_channels) arrays.
        target_frames: int, target number of frames.
    
    Returns:
        Modified data dict with features as 1D vectors of length
        target_frames * n_channels.
    """
    from scipy.signal import resample
    
    normalized = []
    for feat in data['features']:
        if feat.shape[0] == target_frames:
            normalized.append(feat.flatten())
        else:
            resampled = resample(feat, target_frames, axis=0)
            normalized.append(resampled.flatten())
    
    data['features'] = normalized
    return data
    
normalize_feature_lengths(pipeline.train, target)
normalize_feature_lengths(pipeline.test, target)

# Verify
print(f"Train feature shape: {pipeline.train['features'][0].shape}") # (target * n_channels,) e.g. (1300,) for 10 frames x 130 channels

# pipeline.checkpoint_after_step6(sample_fraction=sample_fraction)

# #check where method is used
# import os

# search_term = "standardize_channels"
# project_path = r"D:\Documents\UM DACS\bachelor\UM DACS\bachelor\mozg\code\SingleWordProductionDutch_step2"

# for filename in os.listdir(project_path):
#     if filename.endswith('.py'):
#         filepath = os.path.join(project_path, filename)
#         with open(filepath, 'r', encoding='utf-8') as f:
#             lines = f.readlines()
#             for line_num, line in enumerate(lines, 1):
#                 if search_term in line and 'def ' not in line and not line.strip().startswith('#'):
#                     print(f"{filename}:{line_num}: {line.strip()}")

"""Diagnose per-patient data loss in the phoneme pipeline train set.

Tracks three categories of loss:
1. Word-level drops during segmentation (short EEG, short audio, 
   spectrogram too short, sentence segmentation failures)
2. Boundary count mismatches (detected segments != dictionary phonemes)
3. Feature extraction drops (short neural signal segments, extractHG 
   producing 0 frames)

Run after step2 completes (needs split_result with word_segments_dict).

Usage:
    report = diagnose_train_loss(pipeline)
"""

import numpy as np
from collections import defaultdict, Counter
from dataset_config import Dutch30Config


def diagnose_train_loss(pipeline, split_type='train'):
    """Diagnose where phoneme samples are lost per patient.

    Args:
        pipeline: Dutch30Pipeline instance after step2_split_by_instances.
        split_type: 'train' or 'test'.

    Returns:
        dict: Per-patient loss report with counts at each stage.
    """
    import re
    import string
    from scipy.signal import decimate
    from extract_features import extractHG, extractMelSpecs

    config = pipeline.config
    split_result = pipeline.split_result
    word_segments_dict = split_result['word_segments_dict']
    phonetic_dict = pipeline.phonetic_dict

    report = {}

    for pid in sorted(split_result[split_type].keys()):
        if pid not in word_segments_dict:
            continue

        patient_report = {
            'expected_word_instances': 0,
            'expected_phonemes': 0,
            'drop_word_not_in_dict': 0,
            'drop_word_not_in_dict_words': [],
            'drop_eeg_too_short': 0,
            'drop_audio_too_short': 0,
            'drop_audio_downsample_too_short': 0,
            'drop_spectrogram_failed': 0,
            'drop_spectrogram_too_few_frames': 0,
            'words_reaching_boundary_detection': 0,
            'phonemes_reaching_boundary_detection': 0,
            'boundary_mismatch_words': 0,
            'boundary_mismatch_phonemes_lost': 0,
            'boundary_perfect_match_words': 0,
            'boundary_perfect_match_phonemes': 0,
            'boundary_mismatch_details': [],
            'drop_empty_neural_segments': 0,
            'drop_neural_too_short_for_features': 0,
            'drop_feature_extraction_failed': 0,
            'drop_feature_extraction_zero_frames': 0,
            'phonemes_surviving_feature_extraction': 0,
        }

        words_and_indices = split_result[split_type][pid]

        # Count expected totals
        for word, indices in words_and_indices.items():
            patient_report['expected_word_instances'] += len(indices)
            phonemes = phonetic_dict.extract_phonemes(word)
            if phonemes:
                patient_report['expected_phonemes'] += len(phonemes) * len(indices)
            else:
                patient_report['drop_word_not_in_dict'] += len(indices)
                patient_report['drop_word_not_in_dict_words'].append(word)

        # Now replay the segmentation checks per word instance
        eeg_sr = config.eeg_sr
        min_eeg_frames = int(config.min_phoneme_duration * eeg_sr)
        min_audio_samples = int(
            (config.window_length + config.frameshift) * config.audio_sr
        )
        min_audio_samples_down = int(
            (config.window_length + config.frameshift) * config.audio_target_sr
        )
        downsample_factor = int(config.audio_sr / config.audio_target_sr)
        min_samples_for_hg = int(config.window_length * eeg_sr) + 1
        min_samples = max(
            int(config.min_phoneme_duration * eeg_sr),
            min_samples_for_hg
        )

        words_data = word_segments_dict[pid]['words']

        for word, indices in words_and_indices.items():
            if word not in words_data:
                # Word was dropped entirely during sentence segmentation
                # (wav2vec/RMS failure or short segment filtering)
                phonemes = phonetic_dict.extract_phonemes(word)
                n_ph = len(phonemes) if phonemes else 0
                patient_report['drop_eeg_too_short'] += len(indices)
                continue

            word_info = words_data[word]
            phonemes = phonetic_dict.extract_phonemes(word)
            n_phonemes = len(phonemes) if phonemes else 0

            for idx in indices:
                if idx >= len(word_info['instances']):
                    # Instance index out of range -- split references
                    # a presentation that was dropped during segmentation
                    patient_report['drop_eeg_too_short'] += 1
                    continue

                instance = word_info['instances'][idx]
                eeg_segment = instance['eeg_segment']
                audio_segment = instance.get('audio_segment')
                spec_segment = instance.get('spectrogram_segment')

                # -- Check 1: EEG too short --
                if eeg_segment is None or len(eeg_segment) < min_eeg_frames:
                    patient_report['drop_eeg_too_short'] += 1
                    continue

                # -- Check 2: Audio too short --
                if audio_segment is None or len(audio_segment) < min_audio_samples:
                    patient_report['drop_audio_too_short'] += 1
                    continue

                # -- Check 3: Downsampled audio too short --
                try:
                    audio_down = decimate(
                        audio_segment,
                        downsample_factor
                    )
                    if len(audio_down) < min_audio_samples_down:
                        patient_report['drop_audio_downsample_too_short'] += 1
                        continue
                except Exception:
                    patient_report['drop_audio_downsample_too_short'] += 1
                    continue

                # -- Check 4: Spectrogram --
                if spec_segment is None:
                    patient_report['drop_spectrogram_failed'] += 1
                    continue

                if spec_segment.shape[0] < 3:
                    patient_report['drop_spectrogram_too_few_frames'] += 1
                    continue

                # This word instance reached boundary detection
                patient_report['words_reaching_boundary_detection'] += 1
                patient_report['phonemes_reaching_boundary_detection'] += n_phonemes

                if not phonemes:
                    continue

                # -- Check 5: Boundary count mismatch --
                # Replay boundary detection on the acoustic signal
                try:
                    result = pipeline.detector.detect_boundaries(
                        spectrogram=spec_segment,
                        word=word,
                        participant_id=pid,
                        word_position=0,
                        use_multifeature=pipeline.use_multifeature,
                        use_rms_boundaries=pipeline.use_rms_boundaries,
                        audio_segment=audio_segment,
                        audio_sr=config.audio_sr
                    )
                except Exception:
                    # Boundary detection itself failed
                    patient_report['boundary_mismatch_words'] += 1
                    patient_report['boundary_mismatch_phonemes_lost'] += n_phonemes
                    continue

                if result.get('drop_word', False):
                    patient_report['boundary_mismatch_words'] += 1
                    patient_report['boundary_mismatch_phonemes_lost'] += n_phonemes
                    continue

                n_segments = len(result['segments'])
                if n_segments != n_phonemes:
                    patient_report['boundary_mismatch_words'] += 1
                    patient_report['boundary_mismatch_phonemes_lost'] += n_phonemes
                    patient_report['boundary_mismatch_details'].append(
                        f"{word}: {n_phonemes} phonemes, {n_segments} segments"
                    )
                    continue

                patient_report['boundary_perfect_match_words'] += 1
                patient_report['boundary_perfect_match_phonemes'] += n_phonemes

                # -- Check 6: Per-phoneme neural signal segment validity --
                boundary_samples = result.get('boundary_samples')
                if boundary_samples is None:
                    patient_report['drop_empty_neural_segments'] += n_phonemes
                    continue

                for j in range(n_phonemes):
                    if j >= len(boundary_samples) - 1:
                        patient_report['drop_empty_neural_segments'] += 1
                        continue

                    start = int(boundary_samples[j])
                    end = int(boundary_samples[j + 1])

                    if start >= end or end > eeg_segment.shape[0]:
                        patient_report['drop_empty_neural_segments'] += 1
                        continue

                    seg = eeg_segment[start:end]

                    if seg.shape[0] < min_samples:
                        patient_report['drop_neural_too_short_for_features'] += 1
                        continue

                    # -- Check 7: Feature extraction --
                    try:
                        feat = extractHG(
                            seg, eeg_sr,
                            windowLength=config.window_length,
                            frameshift=config.frameshift
                        )
                        if feat.shape[0] == 0:
                            patient_report['drop_feature_extraction_zero_frames'] += 1
                            continue
                    except Exception:
                        patient_report['drop_feature_extraction_failed'] += 1
                        continue

                    patient_report['phonemes_surviving_feature_extraction'] += 1

        # Compute summary
        expected = patient_report['expected_phonemes']
        surviving = patient_report['phonemes_surviving_feature_extraction']
        patient_report['total_phonemes_lost'] = expected - surviving
        patient_report['survival_rate'] = (
            surviving / expected * 100 if expected > 0 else 0
        )

        report[pid] = patient_report

    # Print summary table
    _print_report(report)

    return report


def _print_report(report):
    """Print a per-patient summary table of data loss.

    Args:
        report: Dict from diagnose_train_loss.
    """
    print("\n" + "=" * 120)
    print("PER-PATIENT DATA LOSS REPORT (TRAIN SET)")
    print("=" * 120)

    header = (
        f"{'Patient':<8} {'Expected':>8} {'Survive':>8} {'Rate':>6} "
        f"| {'NotInDict':>9} {'EEG short':>9} {'Audio':>6} {'Spec':>5} "
        f"| {'BndMatch':>8} {'BndMiss':>7} {'MissPh':>6} "
        f"| {'EmptySeg':>8} {'ShortNN':>7} {'FeatFail':>8} {'0frame':>6}"
    )
    print(header)
    print("-" * 120)

    totals = defaultdict(int)

    for pid in sorted(report.keys()):
        r = report[pid]
        print(
            f"{pid:<8} {r['expected_phonemes']:>8} "
            f"{r['phonemes_surviving_feature_extraction']:>8} "
            f"{r['survival_rate']:>5.1f}% "
            f"| {r['drop_word_not_in_dict']:>9} "
            f"{r['drop_eeg_too_short']:>9} "
            f"{r['drop_audio_too_short']:>6} "
            f"{r['drop_spectrogram_too_few_frames']:>5} "
            f"| {r['boundary_perfect_match_words']:>8} "
            f"{r['boundary_mismatch_words']:>7} "
            f"{r['boundary_mismatch_phonemes_lost']:>6} "
            f"| {r['drop_empty_neural_segments']:>8} "
            f"{r['drop_neural_too_short_for_features']:>7} "
            f"{r['drop_feature_extraction_failed']:>8} "
            f"{r['drop_feature_extraction_zero_frames']:>6}"
        )

        for key in r:
            if isinstance(r[key], (int, float)) and key not in (
                'survival_rate',
            ):
                totals[key] += r[key]

    print("-" * 120)
    total_expected = totals['expected_phonemes']
    total_surviving = totals['phonemes_surviving_feature_extraction']
    total_rate = total_surviving / total_expected * 100 if total_expected > 0 else 0
    print(
        f"{'TOTAL':<8} {total_expected:>8} "
        f"{total_surviving:>8} "
        f"{total_rate:>5.1f}% "
        f"| {totals['drop_word_not_in_dict']:>9} "
        f"{totals['drop_eeg_too_short']:>9} "
        f"{totals['drop_audio_too_short']:>6} "
        f"{totals['drop_spectrogram_too_few_frames']:>5} "
        f"| {totals['boundary_perfect_match_words']:>8} "
        f"{totals['boundary_mismatch_words']:>7} "
        f"{totals['boundary_mismatch_phonemes_lost']:>6} "
        f"| {totals['drop_empty_neural_segments']:>8} "
        f"{totals['drop_neural_too_short_for_features']:>7} "
        f"{totals['drop_feature_extraction_failed']:>8} "
        f"{totals['drop_feature_extraction_zero_frames']:>6}"
    )

    print("\n" + "=" * 120)
    print("COLUMN LEGEND")
    print("=" * 120)
    print("  Expected    : Total phonemes expected from dictionary * train instances")
    print("  Survive     : Phonemes that produce valid neural signal features")
    print("  Rate        : Survival percentage")
    print("  NotInDict   : Word instances where word has no dictionary entry")
    print("  EEG short   : Word instances dropped for short neural signal segments")
    print("  Audio       : Word instances dropped for short acoustic signal")
    print("  Spec        : Word instances dropped for <3 spectrogram frames")
    print("  BndMatch    : Word instances with perfect boundary count match")
    print("  BndMiss     : Word instances where boundary count != phoneme count")
    print("  MissPh      : Phonemes lost due to boundary mismatches")
    print("  EmptySeg    : Phonemes with empty/invalid neural signal boundaries")
    print("  ShortNN     : Phonemes where neural signal too short for feature extraction")
    print("  FeatFail    : Phonemes where feature extraction threw an exception")
    print("  0frame      : Phonemes where extractHG returned 0 frames")
    print("=" * 120)


def print_worst_mismatch_words(report, top_n=10):
    """Print the most common boundary-mismatch words across all patients.

    Args:
        report: Dict from diagnose_train_loss.
        top_n: Number of top offending words to show.
    """
    all_details = []
    for pid, r in report.items():
        for detail in r['boundary_mismatch_details']:
            all_details.append(f"{pid} {detail}")

    word_counts = Counter()
    for detail in all_details:
        # Extract word from "PXX word: N phonemes, M segments"
        parts = detail.split(':')[0].split()
        if len(parts) >= 2:
            word_counts[parts[1]] += 1

    print(f"\nTop {top_n} boundary-mismatch words:")
    print(f"{'Word':<25} {'Occurrences':>12}")
    print("-" * 40)
    for word, count in word_counts.most_common(top_n):
        print(f"{word:<25} {count:>12}")

report = diagnose_train_loss(pipeline)

"""Diagnose feature imprecision per patient.

Two sections:
1. Phoneme length variance: how variable are neural signal segment
   durations for the same phoneme across repetitions?
2. Pooling impact: how much information does statistical pooling
   (mean/std/min/max) destroy compared to the raw temporal features?

Run after step5 completes.

Usage:
    from diagnose_imprecision import diagnose_imprecision
    results = diagnose_imprecision(pipeline)
"""

import numpy as np
from collections import defaultdict
from dataset_config import Dutch30Config


def diagnose_imprecision(pipeline):
    """Quantify imprecision per patient.

    Args:
        pipeline: Dutch30Pipeline instance after step5.

    Returns:
        dict: Per-patient imprecision metrics.
    """
    train = pipeline.train

    print(f"Train data keys: {list(train.keys())}")
    print(f"Total samples: {len(train['features'])}")

    # Check actual feature shapes
    shapes = [f.shape for f in train['features'][:10]]
    print(f"First 10 feature shapes: {shapes}")
    all_lengths = [f.shape[0] for f in train['features']]
    n_unique = len(set(all_lengths))
    print(f"Unique frame counts: {n_unique} (range {min(all_lengths)}-{max(all_lengths)})")
    if n_unique == 1:
        print("All features are fixed-length. Pooling would use FLATTEN, not statistical summary.")
    else:
        print("Features are variable-length. Pooling would use STATISTICAL SUMMARY (mean/std/min/max).")

    results = {}
    patient_ids = sorted(set(train['phoneme_participant_ids']))

    for pid in patient_ids:
        mask = [p == pid for p in train['phoneme_participant_ids']]
        indices = [i for i, m in enumerate(mask) if m]

        features = [train['features'][i] for i in indices]
        labels = [train['phoneme_labels'][i] for i in indices]
        words = [train['phoneme_words'][i] for i in indices]

        results[pid] = {
            'n_samples': len(indices),
            'length_variance': _measure_length_variance(features, labels, words),
            'pooling_impact': _measure_pooling_impact(features, labels),
        }

        if hasattr(pipeline, 'patient_baselines') and pid in pipeline.patient_baselines:
            results[pid]['baseline_impact'] = _measure_baseline_impact(
                features, pipeline.patient_baselines[pid]
            )

    _print_report(results)
    return results


def _measure_length_variance(features, labels, words):
    """Measure how much neural signal segment lengths vary per phoneme.

    Groups features by phoneme label and computes frame count statistics.

    Args:
        features: List of feature arrays.
        labels: List of phoneme labels.
        words: List of word strings.

    Returns:
        dict with per-phoneme length statistics.
    """
    # Frame counts per phoneme
    phoneme_frames = defaultdict(list)
    for i in range(len(features)):
        phoneme_frames[labels[i]].append(features[i].shape[0])

    # Per-phoneme stats
    phoneme_stats = {}
    for phoneme, frames in phoneme_frames.items():
        frames_arr = np.array(frames)
        mean_f = np.mean(frames_arr)
        std_f = np.std(frames_arr)
        phoneme_stats[phoneme] = {
            'count': len(frames),
            'mean_frames': float(mean_f),
            'std_frames': float(std_f),
            'min_frames': int(np.min(frames_arr)),
            'max_frames': int(np.max(frames_arr)),
            'cv': float(std_f / mean_f) if mean_f > 0 else 0,
            'range_ratio': float(np.max(frames_arr) / np.min(frames_arr)) if np.min(frames_arr) > 0 else float('inf'),
        }

    # Overall stats
    all_frames = [f.shape[0] for f in features]
    all_cvs = [s['cv'] for s in phoneme_stats.values() if s['count'] >= 2]

    return {
        'overall_mean_frames': float(np.mean(all_frames)),
        'overall_std_frames': float(np.std(all_frames)),
        'overall_min_frames': int(np.min(all_frames)),
        'overall_max_frames': int(np.max(all_frames)),
        'n_unique_lengths': len(set(all_frames)),
        'mean_phoneme_cv': float(np.mean(all_cvs)) if all_cvs else 0,
        'phoneme_stats': phoneme_stats,
    }


def _measure_pooling_impact(features, labels):
    """Measure information loss from statistical pooling.

    For each feature (n_frames, n_channels):
    - Temporal variance ratio: what fraction of total variance is along time axis

    Also computes per-phoneme averages so you can see which phonemes
    suffer most from pooling.

    Args:
        features: List of feature arrays.
        labels: List of phoneme labels.

    Returns:
        dict with pooling impact metrics.
    """
    temporal_var_ratios = []
    phoneme_temporal_var = defaultdict(list)
    frame_counts = []

    for i, feat in enumerate(features):
        if feat.ndim != 2 or feat.shape[0] < 2:
            continue

        n_frames, n_channels = feat.shape
        frame_counts.append(n_frames)

        # Temporal variance: per-channel variance across frames, averaged
        per_ch_var = np.var(feat, axis=0)  # variance across time per channel
        mean_temporal_var = np.mean(per_ch_var)

        # Total variance across entire feature matrix
        total_var = np.var(feat)

        if total_var > 0:
            ratio = mean_temporal_var / total_var
            temporal_var_ratios.append(ratio)
            phoneme_temporal_var[labels[i]].append(ratio)

    # Phoneme-level summary
    phoneme_pooling = {}
    for phoneme, ratios in phoneme_temporal_var.items():
        phoneme_pooling[phoneme] = {
            'mean_temporal_var_ratio': float(np.mean(ratios)),
            'count': len(ratios),
        }

    return {
        'would_pool': len(set(frame_counts)) > 1,
        'n_unique_lengths': len(set(frame_counts)),
        'mean_temporal_var_ratio': float(np.mean(temporal_var_ratios)) if temporal_var_ratios else 0,
        'phoneme_pooling': phoneme_pooling,
    }


def _measure_baseline_impact(features, baseline):
    """Measure baseline magnitude relative to signal.

    Args:
        features: List of feature arrays (already baseline-subtracted).
        baseline: Baseline vector for this patient.

    Returns:
        dict with baseline impact metrics.
    """
    baseline_norm = np.linalg.norm(baseline)
    signal_norms = []

    for feat in features:
        if feat.ndim != 2:
            continue
        frame_norms = np.linalg.norm(feat, axis=1)
        signal_norms.append(np.mean(frame_norms))

    if not signal_norms:
        return {'baseline_norm': float(baseline_norm)}

    mean_sig = np.mean(signal_norms)
    return {
        'baseline_norm': float(baseline_norm),
        'mean_signal_norm': float(mean_sig),
        'ratio': float(baseline_norm / mean_sig) if mean_sig > 0 else 0,
        'dominates': float(baseline_norm / mean_sig) > 1.0 if mean_sig > 0 else False,
    }


def _print_report(results):
    """Print per-patient report.

    Args:
        results: Dict from diagnose_imprecision.
    """
    print("\n" + "=" * 100)
    print("SECTION 1: PHONEME LENGTH VARIANCE")
    print("How variable are neural signal segment durations per patient?")
    print("=" * 100)

    print(f"\n{'Patient':<8} {'N':>6} {'MeanFr':>7} {'StdFr':>6} "
          f"{'Min':>4} {'Max':>4} {'Unique':>6} {'PhonCV':>7}")
    print("-" * 55)

    for pid in sorted(results.keys()):
        lv = results[pid]['length_variance']
        print(f"{pid:<8} {results[pid]['n_samples']:>6} "
              f"{lv['overall_mean_frames']:>7.1f} "
              f"{lv['overall_std_frames']:>6.1f} "
              f"{lv['overall_min_frames']:>4} "
              f"{lv['overall_max_frames']:>4} "
              f"{lv['n_unique_lengths']:>6} "
              f"{lv['mean_phoneme_cv']:>7.3f}")

    # Per-phoneme detail for a representative patient
    # Pick the patient with most samples
    biggest_pid = max(results.keys(), key=lambda p: results[p]['n_samples'])
    _print_phoneme_detail(biggest_pid, results[biggest_pid])

    print("\n" + "=" * 100)
    print("SECTION 2: POOLING IMPACT")
    print("Would statistical pooling be used? How much temporal info would be lost?")
    print("=" * 100)

    print(f"\n{'Patient':<8} {'N':>6} {'WouldPool':>9} {'UniqLen':>7} "
          f"{'TempVarR':>8} {'Interpretation':>20}")
    print("-" * 65)

    for pid in sorted(results.keys()):
        pi = results[pid]['pooling_impact']
        tvr = pi['mean_temporal_var_ratio']
        if tvr < 0.05:
            interp = "pooling ~lossless"
        elif tvr < 0.15:
            interp = "pooling low loss"
        elif tvr < 0.30:
            interp = "pooling moderate loss"
        else:
            interp = "pooling HIGH loss"

        print(f"{pid:<8} {results[pid]['n_samples']:>6} "
              f"{'YES' if pi['would_pool'] else 'no':>9} "
              f"{pi['n_unique_lengths']:>7} "
              f"{tvr:>8.3f} "
              f"{interp:>20}")

    # Baseline impact
    has_baseline = any('baseline_impact' in r for r in results.values())
    if has_baseline:
        print("\n" + "=" * 100)
        print("SECTION 3: BASELINE IMPACT")
        print("Is baseline subtraction magnitude appropriate relative to signal?")
        print("=" * 100)

        print(f"\n{'Patient':<8} {'BslnNorm':>9} {'SigNorm':>8} "
              f"{'Ratio':>7} {'Status':>12}")
        print("-" * 50)
        for pid in sorted(results.keys()):
            bi = results[pid].get('baseline_impact', {})
            if bi:
                ratio = bi.get('ratio', 0)
                if ratio > 1.0:
                    status = "DOMINATES"
                elif ratio > 0.5:
                    status = "high"
                elif ratio > 0.1:
                    status = "normal"
                else:
                    status = "negligible"
                print(f"{pid:<8} {bi.get('baseline_norm', 0):>9.2f} "
                      f"{bi.get('mean_signal_norm', 0):>8.2f} "
                      f"{ratio:>7.3f} {status:>12}")

    print("\n" + "=" * 100)
    print("INTERPRETATION")
    print("=" * 100)
    print("  PhonCV  : Mean coefficient of variation of frame counts per phoneme.")
    print("            High = same phoneme gets very different segment lengths")
    print("            across repetitions. Caused by boundary detection noise.")
    print("  TempVarR: Fraction of feature variance that is temporal (across frames).")
    print("            High = pooling destroys more information.")
    print("            Low = most variance is across channels, pooling is ~lossless.")
    print("  Ratio   : Baseline norm / signal norm. >1 = baseline dominates.")
    print("=" * 100)


def _print_phoneme_detail(pid, patient_result):
    """Print per-phoneme length variance for one patient.

    Args:
        pid: Patient ID string.
        patient_result: Result dict for this patient.
    """
    print(f"\n  Per-phoneme detail for {pid} "
          f"(most samples, {patient_result['n_samples']} total):")

    stats = patient_result['length_variance']['phoneme_stats']
    print(f"  {'Phoneme':<8} {'N':>5} {'Mean':>6} {'Std':>6} "
          f"{'Min':>4} {'Max':>4} {'CV':>6} {'MaxMin':>6}")
    print(f"  " + "-" * 50)

    # Sort by CV descending
    sorted_phonemes = sorted(stats.items(), key=lambda x: -x[1]['cv'])
    for phoneme, s in sorted_phonemes:
        if s['count'] < 2:
            continue
        print(f"  {phoneme:<8} {s['count']:>5} {s['mean_frames']:>6.1f} "
              f"{s['std_frames']:>6.1f} {s['min_frames']:>4} "
              f"{s['max_frames']:>4} {s['cv']:>6.2f} "
              f"{s['range_ratio']:>6.1f}x")

results = diagnose_imprecision(pipeline)

# What does pipeline.train actually look like RIGHT NOW?
print(f"Shape of first 5 features:")
for i in range(5):
    print(f"  {pipeline.train['features'][i].shape}")
print(f"Total: {len(pipeline.train['features'])}")

# Experiment logger
import json
import os
from datetime import datetime
import numpy as np


PATIENT_GROUPS = {
    'P01-P10': ['P01', 'P02', 'P03', 'P04', 'P06', 'P07', 'P08', 'P09', 'P10'],
    'P11-P20': ['P11', 'P12', 'P13', 'P14', 'P15', 'P16', 'P17', 'P20'],
    'P21-P30': ['P21', 'P22', 'P23', 'P24', 'P25', 'P26', 'P27', 'P28', 'P29', 'P30'],
}


def run_experiment(pipeline, order=3, class_weight='balanced', use_groups=False,
                   classifier_type='random_forest', use_viterbi=False):
    """
    Run a single experiment with given parameters.
    Returns experiment name, params dict, and per-patient results dict.

    Args:
        pipeline: Pipeline with train/test data.
        order: Markov chain order.
        class_weight: 'balanced', 'balanced_subsample', or None.
        use_groups: Whether to use phoneme groups.
        classifier_type: 'random_forest', 'extra_trees', etc.
        use_viterbi: Whether to use Viterbi decoding.

    Returns:
        Tuple of (name, params, results).
    """
    from markov_phoneme_model import MarkovPhonemeModel

    # Build name automatically from parameters
    weight_str = str(class_weight) if class_weight else 'none'
    name = f"{classifier_type}_o{order}_w{weight_str}"
    if use_viterbi:
        name += "_viterbi"

    params = {
        'order': order,
        'class_weight': str(class_weight),
        'use_groups': use_groups,
        'classifier_type': classifier_type,
        'use_viterbi': use_viterbi,
    }

    print(f"\nRunning: {name}")
    print(f"  Params: {params}")

    results = {}
    for pid in sorted(set(pipeline.train['phoneme_participant_ids'])):
        train_mask = [p == pid for p in pipeline.train['phoneme_participant_ids']]
        test_mask = [p == pid for p in pipeline.test['phoneme_participant_ids']]
        train_feat = [pipeline.train['features'][i] for i, m in enumerate(train_mask) if m]
        train_labels = [pipeline.train['phoneme_labels'][i] for i, m in enumerate(train_mask) if m]
        test_feat = [pipeline.test['features'][i] for i, m in enumerate(test_mask) if m]
        test_labels = [pipeline.test['phoneme_labels'][i] for i, m in enumerate(test_mask) if m]

        if len(train_feat) < 10 or len(test_feat) < 5:
            continue

        model = MarkovPhonemeModel(
            phonetic_dict=pipeline.detector.phonetic_dict,
            order=order,
            use_groups=use_groups,
            class_weight=class_weight,
            classifier_type=classifier_type,
        )
        model.train(features=train_feat, phoneme_labels=train_labels)

        preds, _ = model.predict(test_feat, use_viterbi=use_viterbi)
        correct = sum(1 for p, t in zip(preds, test_labels) if p == t)
        accuracy = correct / len(test_labels)

        results[pid] = {
            'accuracy': accuracy,
            'train_size': len(train_feat),
            'test_size': len(test_feat),
            'n_classes': len(set(train_labels)),
            'predictions': preds,
            'true_labels': test_labels,
            'model': model,
        }

    return name, params, results


class ExperimentLogger:
    """
    Simple experiment logger that stores a flat CSV-like table.

    Each row = one experiment.
    Columns = experiment name, params, per-patient accuracies, group means.
    """

    def __init__(self, log_file='experiments.json'):
        """
        Initialize the experiment logger.

        Args:
            log_file: Path to JSON file for storing experiments.
        """
        self.log_file = log_file
        self.experiments = []

        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                self.experiments = json.load(f)
            print(f"Loaded {len(self.experiments)} previous experiments from {log_file}")

    def log(self, name, params, results):
        """
        Log an experiment from run_experiment output.

        Args:
            name: Experiment name (auto-generated by run_experiment).
            params: Dict of parameters.
            results: Dict of per-patient results from run_experiment.
        """
        # Extract only accuracies per patient (no model objects)
        patient_accs = {}
        for pid, pr in results.items():
            patient_accs[pid] = round(pr['accuracy'], 4)

        # Compute group means
        group_means = {}
        for group_name, patients in PATIENT_GROUPS.items():
            accs = [patient_accs[pid] for pid in patients if pid in patient_accs]
            if accs:
                group_means[group_name] = round(np.mean(accs), 4)
            else:
                group_means[group_name] = None

        all_accs = list(patient_accs.values())
        overall = round(np.mean(all_accs), 4) if all_accs else 0

        entry = {
            'name': name,
            'timestamp': datetime.now().isoformat()[:19],
            'params': params,
            'patients': patient_accs,
            'group_means': group_means,
            'overall': overall,
        }

        self.experiments.append(entry)

        with open(self.log_file, 'w') as f:
            json.dump(self.experiments, f, indent=2)

        print(f"  Saved: {name} (overall={overall})")

    def print_table(self, last_n=None):
        """
        Print a simple comparison table.

        Args:
            last_n: Show last N experiments (None = all).
        """
        experiments = self.experiments[-last_n:] if last_n else self.experiments

        if not experiments:
            print("No experiments.")
            return

        # Collect all patient IDs
        all_patients = set()
        for exp in experiments:
            all_patients.update(exp['patients'].keys())

        # Print header
        exp_names = [exp['name'][:20] for exp in experiments]
        header_width = max(10, max(len(n) for n in exp_names) + 2)

        print("\n" + "=" * (12 + header_width * len(experiments)))
        row = f"{'Patient':<12}"
        for name in exp_names:
            row += f"{name:<{header_width}}"
        print(row)
        print("-" * (12 + header_width * len(experiments)))

        # Print by group
        for group_name, patients in PATIENT_GROUPS.items():
            group_patients = [p for p in patients if p in all_patients]
            if not group_patients:
                continue

            for pid in group_patients:
                row = f"  {pid:<10}"
                for exp in experiments:
                    acc = exp['patients'].get(pid)
                    if acc is not None:
                        row += f"{acc:<{header_width}.4f}"
                    else:
                        row += f"{'N/A':<{header_width}}"
                print(row)

            # Group mean
            row = f"  {'MEAN':<10}"
            for exp in experiments:
                gm = exp['group_means'].get(group_name)
                if gm is not None:
                    row += f"{gm:<{header_width}.4f}"
                else:
                    row += f"{'N/A':<{header_width}}"
            print(row)
            print()

        # Overall
        print("-" * (12 + header_width * len(experiments)))
        row = f"{'OVERALL':<12}"
        for exp in experiments:
            row += f"{exp['overall']:<{header_width}.4f}"
        print(row)
        print("=" * (12 + header_width * len(experiments)))

    def clear(self):
        """Clear all experiments."""
        confirm = input(f"Delete all {len(self.experiments)} experiments? (yes/no): ")
        if confirm.lower() == 'yes':
            self.experiments = []
            if os.path.exists(self.log_file):
                os.remove(self.log_file)
            print("Cleared.")

    def remove_last(self, n=1):
        """
        Remove last N experiments.

        Args:
            n: Number of experiments to remove from the end.
        """
        removed = self.experiments[-n:]
        self.experiments = self.experiments[:-n]
        with open(self.log_file, 'w') as f:
            json.dump(self.experiments, f, indent=2)
        for exp in removed:
            print(f"  Removed: {exp['name']}")

# def train_and_evaluate(pipeline, use_groups=False, order=3, class_weight='balanced', 
#                        classifier_type='dtw_knn'):
#     """Train per patient and return results."""
#     from collections import Counter
    
#     results = {}
#     for pid in sorted(set(pipeline.train['phoneme_participant_ids'])):
#         # Filter data
#         train_mask = [p == pid for p in pipeline.train['phoneme_participant_ids']]
#         test_mask = [p == pid for p in pipeline.test['phoneme_participant_ids']]
#         train_feat = [pipeline.train['features'][i] for i, m in enumerate(train_mask) if m]
#         train_labels = [pipeline.train['phoneme_labels'][i] for i, m in enumerate(train_mask) if m]
#         test_feat = [pipeline.test['features'][i] for i, m in enumerate(test_mask) if m]
#         test_labels = [pipeline.test['phoneme_labels'][i] for i, m in enumerate(test_mask) if m]
        
#         if len(train_feat) < 10 or len(test_feat) < 5:
#             continue
        
#         # Train model with passed parameters
#         model = MarkovPhonemeModel(
#             phonetic_dict=pipeline.detector.phonetic_dict,
#             order=order,
#             use_groups=use_groups,
#             class_weight=class_weight,
#             classifier_type=classifier_type  # ADD THIS
#         )
#         model.train(features=train_feat, phoneme_labels=train_labels)
        
#         # Predict and calculate accuracy
#         preds, _ = model.predict(test_feat, use_viterbi=True)
#         correct = sum(1 for p, t in zip(preds, test_labels) if p == t)
#         accuracy = correct / len(test_labels)
        
#         results[pid] = {
#             'model': model,
#             'accuracy': accuracy,
#             'train_size': len(train_feat),
#             'test_size': len(test_feat),
#             'n_classes': len(set(train_labels)),
#             'predictions': preds,
#             'true_labels': test_labels
#         }
#         print(f"  {pid}: Acc={accuracy:.3f} ({len(set(train_labels))} classes)")
    
#     accs = [r['accuracy'] for r in results.values()]
#     print(f"\n  Mean: {np.mean(accs):.3f} +/- {np.std(accs):.3f}")
#     return results

# def visualize_patient_model(pid, patient_results, pipeline, show_table=True):
#     """
#     Detailed analysis for one patient.
    
#     Args:
#         pid: Patient ID
#         patient_results: Results dict from train_and_evaluate
#         pipeline: Pipeline with train/test data
#         show_table: Whether to print per-phoneme accuracy table
#     """
#     from sklearn.metrics import confusion_matrix, precision_score, recall_score
#     from matplotlib.patches import Rectangle
#     from collections import Counter
#     import numpy as np
#     import matplotlib.pyplot as plt

#     if pid not in patient_results:
#         print(f"{pid} not found")
#         return

#     # Filter data
#     train_mask = [p == pid for p in pipeline.train['phoneme_participant_ids']]
#     test_mask = [p == pid for p in pipeline.test['phoneme_participant_ids']]

#     train_labels = [pipeline.train['phoneme_labels'][i] for i, m in enumerate(train_mask) if m]
#     train_feat = [pipeline.train['features'][i] for i, m in enumerate(train_mask) if m]
#     test_feat = [pipeline.test['features'][i] for i, m in enumerate(test_mask) if m]
#     test_labels = [pipeline.test['phoneme_labels'][i] for i, m in enumerate(test_mask) if m]

#     # Use stored predictions if available
#     if 'predictions' in patient_results[pid]:
#         preds = patient_results[pid]['predictions']
#         test_labels = patient_results[pid].get('true_labels', test_labels)
#     else:
#         model = patient_results[pid]['model']
#         preds, _ = model.predict(test_feat, use_viterbi=True)

#     # Flatten preds if nested
#     if preds and isinstance(preds[0], list):
#         preds = [p[0] if len(p) > 0 else '?' for p in preds]
    
#     preds = [str(p) if not isinstance(p, str) else p for p in preds]

#     # Build confusion data
#     confusion_data = {}
#     for true_label, pred_label in zip(test_labels, preds):
#         if true_label not in confusion_data:
#             confusion_data[true_label] = Counter()
#         confusion_data[true_label][pred_label] += 1

#     # Get labels
#     train_counts = Counter(train_labels)
#     test_counts = Counter(test_labels)
#     all_phonemes = sorted(set(list(train_counts.keys()) + list(test_counts.keys())))
#     test_phonemes = sorted(test_counts.keys())
#     unique_labels = sorted(set(list(test_labels) + list(preds)))

#     # Calculate per-phoneme metrics
#     phoneme_metrics = {}
#     for p in test_phonemes:
#         # Accuracy (recall): correct predictions for this class / total of this class
#         true_mask = [l == p for l in test_labels]
#         correct = sum(1 for i, m in enumerate(true_mask) if m and preds[i] == p)
#         total_true = sum(true_mask)
#         recall = correct / total_true if total_true > 0 else 0
        
#         # Precision: correct predictions for this class / total predicted as this class
#         pred_mask = [pr == p for pr in preds]
#         total_pred = sum(pred_mask)
#         precision = correct / total_pred if total_pred > 0 else 0
        
#         # F1
#         f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
#         phoneme_metrics[p] = {
#             'recall': recall,
#             'precision': precision,
#             'f1': f1,
#             'support': total_true
#         }

#     # Compute confusion matrices
#     cm = confusion_matrix(test_labels, preds, labels=unique_labels)
#     cm_recall = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-10)  # Row-normalized
#     cm_precision = cm.astype('float') / (cm.sum(axis=0, keepdims=True) + 1e-10)  # Col-normalized

#     # Setup figure - 2x2 layout
#     fig, axes = plt.subplots(2, 2, figsize=(16, 12))
#     fig.suptitle(f'{pid} - Accuracy: {patient_results[pid]["accuracy"]:.3f}', fontsize=14, fontweight='bold')

#     # 1. Top-left: Train/test distribution
#     ax1 = axes[0, 0]
#     x = np.arange(len(all_phonemes))
#     width = 0.35
#     ax1.bar(x - width/2, [train_counts.get(p, 0) for p in all_phonemes], width, label='Train', color='cornflowerblue')
#     ax1.bar(x + width/2, [test_counts.get(p, 0) for p in all_phonemes], width, label='Test', color='coral')
#     ax1.set_xticks(x)
#     ax1.set_xticklabels(all_phonemes, rotation=90, fontsize=8)
#     ax1.set_title(f'Distribution (train={len(train_labels)}, test={len(test_labels)})')
#     ax1.set_ylabel('Count')
#     ax1.legend()

#     # 2. Top-right: Per-phoneme Precision, Recall, F1
#     ax2 = axes[0, 1]
#     x = np.arange(len(test_phonemes))
#     width = 0.25
    
#     recalls = [phoneme_metrics[p]['recall'] for p in test_phonemes]
#     precisions = [phoneme_metrics[p]['precision'] for p in test_phonemes]
#     f1s = [phoneme_metrics[p]['f1'] for p in test_phonemes]
    
#     ax2.bar(x - width, recalls, width, label='Recall', color='steelblue')
#     ax2.bar(x, precisions, width, label='Precision', color='darkorange')
#     ax2.bar(x + width, f1s, width, label='F1', color='green')
    
#     ax2.set_xticks(x)
#     ax2.set_xticklabels(test_phonemes, rotation=90, fontsize=8)
#     ax2.set_title('Per-Phoneme Metrics')
#     ax2.set_ylim([0, 1])
#     ax2.axhline(patient_results[pid]['accuracy'], color='red', linestyle='--', alpha=0.5, label='Overall Acc')
#     ax2.legend(loc='upper right', fontsize=8)
#     ax2.set_ylabel('Score')

#     # 3. Bottom-left: Confusion matrix (counts)
#     ax3 = axes[1, 0]
#     im3 = ax3.imshow(cm_recall, cmap='Blues', vmin=0, vmax=1)
    
#     # Add cell values
#     n_labels = len(unique_labels)
#     fontsize = max(5, min(8, 100 // n_labels))
#     for i in range(n_labels):
#         for j in range(n_labels):
#             val = cm[i, j]
#             if val > 0:
#                 color = 'white' if cm_recall[i, j] > 0.5 else 'black'
#                 ax3.text(j, i, str(val), ha='center', va='center', color=color, fontsize=fontsize)
    
#     # Highlight diagonal
#     for i in range(n_labels):
#         ax3.add_patch(Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, edgecolor='red', linewidth=1))
    
#     ax3.set_xticks(range(n_labels))
#     ax3.set_yticks(range(n_labels))
#     ax3.set_xticklabels(unique_labels, rotation=90, fontsize=fontsize)
#     ax3.set_yticklabels(unique_labels, fontsize=fontsize)
#     ax3.set_xlabel('Predicted')
#     ax3.set_ylabel('True')
#     ax3.set_title('Confusion Matrix (Recall normalized)')
#     plt.colorbar(im3, ax=ax3, label='Recall', fraction=0.046)

#     # 4. Bottom-right: Precision-normalized confusion matrix
#     ax4 = axes[1, 1]
#     im4 = ax4.imshow(cm_precision, cmap='Greens', vmin=0, vmax=1)
    
#     for i in range(n_labels):
#         for j in range(n_labels):
#             val = cm[i, j]
#             if val > 0:
#                 color = 'white' if cm_precision[i, j] > 0.5 else 'black'
#                 ax4.text(j, i, str(val), ha='center', va='center', color=color, fontsize=fontsize)
    
#     for i in range(n_labels):
#         ax4.add_patch(Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, edgecolor='darkred', linewidth=1))
    
#     ax4.set_xticks(range(n_labels))
#     ax4.set_yticks(range(n_labels))
#     ax4.set_xticklabels(unique_labels, rotation=90, fontsize=fontsize)
#     ax4.set_yticklabels(unique_labels, fontsize=fontsize)
#     ax4.set_xlabel('Predicted')
#     ax4.set_ylabel('True')
#     ax4.set_title('Confusion Matrix (Precision normalized)')
#     plt.colorbar(im4, ax=ax4, label='Precision', fraction=0.046)

#     plt.tight_layout()
#     plt.show()

#     # Print table if requested
#     if show_table:
#         print(f"\n{'='*80}")
#         print(f"{pid} - PER-PHONEME METRICS")
#         print(f"{'='*80}")
#         print(f"{'Phoneme':<8} {'Recall':<8} {'Prec':<8} {'F1':<8} {'Count':<8} {'Top 3 Confusions'}")
#         print('-'*80)

#         for p in test_phonemes:
#             m = phoneme_metrics[p]
            
#             if p in confusion_data:
#                 confusions = confusion_data[p].copy()
#                 confusions.pop(p, None)
#                 top_confusions = confusions.most_common(3)
#                 confusion_str = ', '.join([f"{pred}({cnt})" for pred, cnt in top_confusions]) if top_confusions else '-'
#             else:
#                 confusion_str = '-'

#             print(f"{p:<8} {m['recall']:>6.2f}  {m['precision']:>6.2f}  {m['f1']:>6.2f}  {m['support']:>6}  {confusion_str}")

#         # Print summary
#         mean_recall = np.mean([m['recall'] for m in phoneme_metrics.values()])
#         mean_precision = np.mean([m['precision'] for m in phoneme_metrics.values()])
#         mean_f1 = np.mean([m['f1'] for m in phoneme_metrics.values()])
        
#         print('-'*80)
#         print(f"{'MACRO':<8} {mean_recall:>6.2f}  {mean_precision:>6.2f}  {mean_f1:>6.2f}")
#         print(f"{'='*80}\n")

# import importlib
# import dutch_30_pipeline
# importlib.reload(dutch_30_pipeline)

# import importlib
# import dutch_30_pipeline
# importlib.reload(dutch_30_pipeline)

# # Bind all step methods from the reloaded class to existing instance
# for name in dir(dutch_30_pipeline.Dutch30Pipeline):
#     if name.startswith('step'):
#         method = getattr(dutch_30_pipeline.Dutch30Pipeline, name)
#         if callable(method):
#             setattr(pipeline, name, method.__get__(pipeline))

from scipy.signal import resample

# Pick target based on your distribution
target = 10  # adjust after seeing distribution

normalize_feature_lengths(pipeline.train, target)
normalize_feature_lengths(pipeline.test, target)

# Verify
print(f"Train feature shape: {pipeline.train['features'][0].shape}")
# Should be (target * n_channels,) e.g. (1300,) for 10 frames x 130 channels

pipeline.step9_train_and_evaluate = dutch_30_pipeline.Dutch30Pipeline.step9_train_and_evaluate.__get__(pipeline)
pipeline.step10_visualize_patient = dutch_30_pipeline.Dutch30Pipeline.step10_visualize_patient.__get__(pipeline)

# With MarkovPhonemeModel
from markov_phoneme_model import MarkovPhonemeModel

results = pipeline.step9_train_and_evaluate(
    model_factory=MarkovPhonemeModel,
    model_params={
        "phonetic_dict": pipeline.phonetic_dict,
        "order": 1,
        "classifier_type": "logistic_regression",
        "class_weight": None,
        "use_groups": False,
    },
    use_viterbi=False,
)

pipeline.step10_visualize_group(['P01', 'P02', 'P03', 'P04', 'P06', 'P07', 'P08', 'P09', 'P10', 'P11', 'P12', 'P13', 'P14', 'P15', 'P16', 'P17', 'P20', 'P21', 'P22', 'P23', 'P24', 'P25', 'P26', 'P27', 'P28', 'P29', 'P30'], show_table=False)

# Train on raw phonemes
#importlib.reload(markov_phoneme_model)
# raw_results = train_and_evaluate(pipeline, use_groups=False)

frame_counts = [f.shape[0] for f in pipeline.train['features']]
print(f"min={min(frame_counts)}, max={max(frame_counts)}, "
      f"median={np.median(frame_counts):.0f}, mean={np.mean(frame_counts):.0f}")

# #Phoneme Feature Inspector

# """Diagnostic visualization for inspecting the neural (EEG-derived) feature
# representations that the pipeline produces per phoneme per patient.

# Intended usage:
#     After step5 (accumulate) or step5b (normalize), plug this into the
#     pipeline to visualize what the classifier actually receives.

#     inspector = PhonemeFeatureInspector(pipeline)
#     inspector.report_feature_structure()
#     inspector.plot_top_phoneme_signals(n_top=3, dataset="train")
#     inspector.plot_all_samples_pooled_representation(dataset="train")
# """

# import os
# import numpy as np
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt

# from collections import Counter, defaultdict
# from dataset_config import Dutch30Config


# class PhonemeFeatureInspector:
#     """Inspect and visualize neural features extracted per phoneme.

#     This class operates on the pipeline's accumulated data dictionaries
#     (self.train / self.test) and produces per-patient visualizations
#     of the EEG-derived feature representations.

#     Args:
#         pipeline: A Dutch30Pipeline instance that has completed at least
#             step5 (so that pipeline.train and pipeline.test exist).
#         output_dir: Directory to save figures. Defaults to
#             pipeline.path_results / 'feature_inspection'.
#     """

#     # -- visual constants (no icons, no emoji) --
#     COLORS_PHONEME = [
#         "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
#         "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
#     ]
#     SAMPLE_ALPHA = 0.15
#     MEAN_LINEWIDTH = 2.0
#     FIG_DPI = 150

#     def __init__(self, pipeline, output_dir=None):
#         self.pipeline = pipeline
#         self.config = getattr(pipeline, "config", Dutch30Config())

#         if output_dir is None:
#             self.output_dir = os.path.join(
#                 getattr(pipeline, "path_results", "."),
#                 "feature_inspection",
#             )
#         else:
#             self.output_dir = output_dir

#         os.makedirs(self.output_dir, exist_ok=True)

#     # ------------------------------------------------------------------
#     # 1. Textual report: what the features actually are
#     # ------------------------------------------------------------------

#     def report_feature_structure(self, dataset="train"):
#         """Print a summary of the neural feature structure.

#         Args:
#             dataset: 'train' or 'test'.
#         """
#         data = self._get_data(dataset)
#         features = data["features"]
#         labels = data["phoneme_labels"]
#         pids = data["phoneme_participant_ids"]

#         print("=" * 70)
#         print("NEURAL FEATURE STRUCTURE REPORT")
#         print("=" * 70)

#         # Global shape info
#         shapes = [np.array(f).shape for f in features]
#         ndims = set(len(s) for s in shapes)
#         print(f"Dataset           : {dataset}")
#         print(f"Total samples     : {len(features)}")
#         print(f"Unique phonemes   : {len(set(labels))}")
#         print(f"Unique patients   : {len(set(pids))}")
#         print(f"Feature ndim(s)   : {ndims}")

#         if 2 in ndims:
#             two_d = [s for s in shapes if len(s) == 2]
#             frame_counts = [s[0] for s in two_d]
#             channel_counts = [s[1] for s in two_d]
#             print(
#                 f"Frames per sample : min={min(frame_counts)}, "
#                 f"max={max(frame_counts)}, median={int(np.median(frame_counts))}"
#             )
#             print(
#                 f"Channels (dim-1)  : min={min(channel_counts)}, "
#                 f"max={max(channel_counts)}"
#             )
#             print()
#             print("INTERPRETATION:")
#             print(
#                 "  Each feature array has shape (n_frames, n_channels)."
#             )
#             print(
#                 "  - n_frames  = number of time windows "
#                 f"(frameshift = {self.config.frameshift}s "
#                 f"= {self.config.frameshift * 1000:.0f}ms each)."
#             )
#             print(
#                 "  - n_channels = number of EEG electrode channels "
#                 "after exclusions."
#             )
#             print(
#                 "  Each value is the HIGH-GAMMA ENVELOPE POWER for that "
#                 "channel at that time window."
#             )
#             print(
#                 "  This is NOT a single mean amplitude -- it is a full "
#                 "time-by-channel matrix of neural activity."
#             )
#         elif 1 in ndims:
#             lengths = [s[0] for s in shapes if len(s) == 1]
#             print(
#                 f"Vector length     : min={min(lengths)}, "
#                 f"max={max(lengths)}"
#             )
#             print()
#             print("INTERPRETATION:")
#             print("  Features have been pooled to 1-D vectors.")
#         print("=" * 70)

#         # Per-patient breakdown
#         print()
#         print("PER-PATIENT BREAKDOWN:")
#         print(f"{'Patient':<10} {'Samples':<10} {'Shape example':<25} {'Phonemes'}")
#         print("-" * 70)
#         pid_indices = defaultdict(list)
#         for i, pid in enumerate(pids):
#             pid_indices[pid].append(i)

#         for pid in sorted(pid_indices.keys()):
#             idxs = pid_indices[pid]
#             example_shape = np.array(features[idxs[0]]).shape
#             unique_ph = len(set(labels[i] for i in idxs))
#             print(f"{pid:<10} {len(idxs):<10} {str(example_shape):<25} {unique_ph}")
#         print()

#     # ------------------------------------------------------------------
#     # 2. Per-patient: neural signal traces for top-N phonemes
#     # ------------------------------------------------------------------

#     def plot_top_phoneme_signals(self, n_top=3, dataset="train",
#                                 max_samples_per_phoneme=20,
#                                 max_channels_to_show=5):
#         """Plot raw neural feature traces for the most frequent phonemes.

#         For each patient, selects the top n_top phonemes by count and
#         plots the high-gamma time courses across a few channels.

#         One figure is saved per patient.

#         Args:
#             n_top: Number of top phonemes to show.
#             dataset: 'train' or 'test'.
#             max_samples_per_phoneme: Cap on overlaid sample traces.
#             max_channels_to_show: Number of EEG channels to plot.
#         """
#         data = self._get_data(dataset)
#         features = data["features"]
#         labels = data["phoneme_labels"]
#         pids = data["phoneme_participant_ids"]

#         pid_phoneme_indices = self._group_by_patient_and_phoneme(
#             pids, labels
#         )

#         for pid in sorted(pid_phoneme_indices.keys()):
#             phoneme_map = pid_phoneme_indices[pid]
#             counts = {ph: len(idxs) for ph, idxs in phoneme_map.items()}
#             top_phonemes = sorted(counts, key=counts.get, reverse=True)[
#                 :n_top
#             ]

#             # Determine number of channels from first sample
#             first_idx = phoneme_map[top_phonemes[0]][0]
#             feat_example = np.array(features[first_idx])
#             if feat_example.ndim == 1:
#                 print(
#                     f"  {pid}: features are 1-D, cannot plot time traces. "
#                     "Skipping."
#                 )
#                 continue

#             n_channels_total = feat_example.shape[1]
#             n_ch = min(max_channels_to_show, n_channels_total)

#             # Pick channels spread evenly across the array
#             channel_indices = np.linspace(
#                 0, n_channels_total - 1, n_ch, dtype=int
#             )

#             fig, axes = plt.subplots(
#                 n_top,
#                 n_ch,
#                 figsize=(4 * n_ch, 3 * n_top),
#                 squeeze=False,
#                 sharex=True,
#             )
#             fig.suptitle(
#                 f"{pid} -- Neural (EEG high-gamma) traces for "
#                 f"top {n_top} phonemes ({dataset})",
#                 fontsize=12,
#                 y=1.02,
#             )

#             time_axis_ms = None

#             for row, phoneme in enumerate(top_phonemes):
#                 idxs = phoneme_map[phoneme][:max_samples_per_phoneme]
#                 color = self.COLORS_PHONEME[row % len(self.COLORS_PHONEME)]

#                 for col, ch_idx in enumerate(channel_indices):
#                     ax = axes[row, col]

#                     all_traces = []
#                     for sample_idx in idxs:
#                         feat = np.array(features[sample_idx])
#                         trace = feat[:, ch_idx]
#                         n_frames = len(trace)
#                         t_ms = (
#                             np.arange(n_frames)
#                             * self.config.frameshift
#                             * 1000
#                         )
#                         ax.plot(
#                             t_ms,
#                             trace,
#                             color=color,
#                             alpha=self.SAMPLE_ALPHA,
#                             linewidth=0.8,
#                         )
#                         all_traces.append(trace)

#                     # Mean trace
#                     if all_traces:
#                         min_len = min(len(t) for t in all_traces)
#                         trimmed = np.array([t[:min_len] for t in all_traces])
#                         mean_trace = np.mean(trimmed, axis=0)
#                         t_mean = (
#                             np.arange(min_len)
#                             * self.config.frameshift
#                             * 1000
#                         )
#                         ax.plot(
#                             t_mean,
#                             mean_trace,
#                             color=color,
#                             linewidth=self.MEAN_LINEWIDTH,
#                             label=f"mean (n={len(idxs)})",
#                         )

#                     if row == 0:
#                         ax.set_title(f"ch {ch_idx}", fontsize=9)
#                     if col == 0:
#                         ax.set_ylabel(
#                             f"/{phoneme}/ (n={counts[phoneme]})",
#                             fontsize=9,
#                         )
#                     if row == n_top - 1:
#                         ax.set_xlabel("Time (ms)")
#                     ax.legend(fontsize=7, loc="upper right")

#             fig.tight_layout()
#             save_path = os.path.join(
#                 self.output_dir,
#                 f"{pid}_top{n_top}_neural_traces_{dataset}.png",
#             )
#             fig.savefig(save_path, dpi=self.FIG_DPI, bbox_inches="tight")
#             plt.close(fig)
#             print(f"  Saved: {save_path}")

#     # ------------------------------------------------------------------
#     # 3. Pooled representation for ALL samples
#     # ------------------------------------------------------------------

#     def plot_all_samples_pooled_representation(self, dataset="train",
#                                                pool_method="concat_mean_std"):
#         """Show what each sample looks like after statistical pooling.

#         Since the classifier likely receives a fixed-length vector per
#         sample, this method applies a pooling strategy and then
#         visualises the resulting vectors as a heatmap (sorted by phoneme
#         label) per patient.

#         Two pooling methods are supported:
#             'mean'            : np.mean(feat, axis=0)  -- simple temporal average
#             'concat_mean_std' : np.concatenate([mean, std], axis=0)

#         One figure is saved per patient.

#         Args:
#             dataset: 'train' or 'test'.
#             pool_method: 'mean' or 'concat_mean_std'.
#         """
#         data = self._get_data(dataset)
#         features = data["features"]
#         labels = data["phoneme_labels"]
#         pids = data["phoneme_participant_ids"]

#         pid_indices = defaultdict(list)
#         for i, pid in enumerate(pids):
#             pid_indices[pid].append(i)

#         for pid in sorted(pid_indices.keys()):
#             idxs = pid_indices[pid]

#             vectors = []
#             sample_labels = []

#             for i in idxs:
#                 feat = np.array(features[i])
#                 if feat.ndim == 1:
#                     vec = feat
#                 elif feat.ndim == 2:
#                     if pool_method == "mean":
#                         vec = np.mean(feat, axis=0)
#                     elif pool_method == "concat_mean_std":
#                         vec = np.concatenate(
#                             [np.mean(feat, axis=0), np.std(feat, axis=0)]
#                         )
#                     else:
#                         raise ValueError(
#                             f"Unknown pool_method: {pool_method}"
#                         )
#                 else:
#                     continue

#                 vectors.append(vec)
#                 sample_labels.append(labels[i])

#             if not vectors:
#                 continue

#             # Sort by phoneme label for visual grouping
#             sort_order = np.argsort(sample_labels)
#             sorted_labels = [sample_labels[j] for j in sort_order]
#             matrix = np.array([vectors[j] for j in sort_order])

#             # Find phoneme boundaries for tick placement
#             label_positions = []
#             label_names = []
#             prev_label = None
#             for row_idx, lbl in enumerate(sorted_labels):
#                 if lbl != prev_label:
#                     label_positions.append(row_idx)
#                     label_names.append(lbl)
#                     prev_label = lbl

#             # Normalize columns for visualization
#             col_mean = matrix.mean(axis=0, keepdims=True)
#             col_std = matrix.std(axis=0, keepdims=True) + 1e-10
#             matrix_norm = (matrix - col_mean) / col_std

#             fig, ax = plt.subplots(
#                 figsize=(max(8, matrix.shape[1] * 0.05), max(6, len(vectors) * 0.025))
#             )

#             n_channels = matrix.shape[1]
#             if pool_method == "concat_mean_std":
#                 half = n_channels // 2
#                 xlabel_text = (
#                     f"Feature index  "
#                     f"[0..{half-1}] = temporal mean per channel  |  "
#                     f"[{half}..{n_channels-1}] = temporal std per channel"
#                 )
#             else:
#                 xlabel_text = "Feature index (temporal mean per channel)"

#             im = ax.imshow(
#                 matrix_norm,
#                 aspect="auto",
#                 cmap="RdBu_r",
#                 interpolation="nearest",
#                 vmin=-2,
#                 vmax=2,
#             )
#             ax.set_yticks(label_positions)
#             ax.set_yticklabels(label_names, fontsize=7)
#             ax.set_xlabel(xlabel_text, fontsize=8)
#             ax.set_ylabel("Sample (sorted by phoneme)")
#             ax.set_title(
#                 f"{pid} -- Pooled neural features ({pool_method}), "
#                 f"{len(vectors)} samples, {dataset}",
#                 fontsize=10,
#             )
#             fig.colorbar(im, ax=ax, label="z-scored value", shrink=0.6)

#             fig.tight_layout()
#             save_path = os.path.join(
#                 self.output_dir,
#                 f"{pid}_pooled_{pool_method}_{dataset}.png",
#             )
#             fig.savefig(save_path, dpi=self.FIG_DPI, bbox_inches="tight")
#             plt.close(fig)
#             print(f"  Saved: {save_path}")

#     # ------------------------------------------------------------------
#     # 4. Per-phoneme distribution box plot
#     # ------------------------------------------------------------------

#     def plot_phoneme_mean_amplitude_distribution(self, dataset="train",
#                                                  n_top=10):
#         """Box plot of mean neural amplitude per phoneme per patient.

#         For each sample, computes np.mean(feat) -- the grand mean across
#         all time frames and all channels -- and groups by phoneme label.

#         Args:
#             dataset: 'train' or 'test'.
#             n_top: Show only the top-N most frequent phonemes.
#         """
#         data = self._get_data(dataset)
#         features = data["features"]
#         labels = data["phoneme_labels"]
#         pids = data["phoneme_participant_ids"]

#         pid_indices = defaultdict(list)
#         for i, pid in enumerate(pids):
#             pid_indices[pid].append(i)

#         for pid in sorted(pid_indices.keys()):
#             idxs = pid_indices[pid]
#             ph_counts = Counter(labels[i] for i in idxs)
#             top_phs = [
#                 ph
#                 for ph, _ in ph_counts.most_common(n_top)
#             ]

#             box_data = []
#             box_labels = []
#             for ph in top_phs:
#                 vals = []
#                 for i in idxs:
#                     if labels[i] != ph:
#                         continue
#                     feat = np.array(features[i])
#                     vals.append(float(np.mean(feat)))
#                 box_data.append(vals)
#                 box_labels.append(f"/{ph}/ (n={len(vals)})")

#             if not box_data:
#                 continue

#             fig, ax = plt.subplots(figsize=(max(6, len(top_phs) * 0.8), 5))
#             bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True)
#             for patch, color in zip(
#                 bp["boxes"],
#                 self.COLORS_PHONEME * 3,
#             ):
#                 patch.set_facecolor(color)
#                 patch.set_alpha(0.5)

#             ax.set_ylabel("Grand mean neural amplitude (high-gamma)")
#             ax.set_xlabel("Phoneme")
#             ax.set_title(
#                 f"{pid} -- Mean neural amplitude per phoneme ({dataset})",
#                 fontsize=10,
#             )
#             plt.xticks(rotation=45, ha="right")
#             fig.tight_layout()

#             save_path = os.path.join(
#                 self.output_dir,
#                 f"{pid}_phoneme_amplitude_boxplot_{dataset}.png",
#             )
#             fig.savefig(save_path, dpi=self.FIG_DPI, bbox_inches="tight")
#             plt.close(fig)
#             print(f"  Saved: {save_path}")

#     # ------------------------------------------------------------------
#     # helpers
#     # ------------------------------------------------------------------

#     def _get_data(self, dataset):
#         """Retrieve the pipeline's data dict for the given split.

#         Args:
#             dataset: 'train' or 'test'.

#         Returns:
#             dict with keys 'features', 'phoneme_labels', etc.

#         Raises:
#             ValueError: If the requested dataset is not available.
#         """
#         data = getattr(self.pipeline, dataset, None)
#         if data is None or "features" not in data:
#             raise ValueError(
#                 f"Pipeline has no '{dataset}' data. "
#                 "Run at least step5 before inspection."
#             )
#         return data

#     def _group_by_patient_and_phoneme(self, pids, labels):
#         """Group sample indices by (patient, phoneme).

#         Args:
#             pids: List of patient IDs parallel to features.
#             labels: List of phoneme labels parallel to features.

#         Returns:
#             dict[str, dict[str, list[int]]]: pid -> phoneme -> [indices].
#         """
#         result = defaultdict(lambda: defaultdict(list))
#         for i, (pid, label) in enumerate(zip(pids, labels)):
#             result[pid][label].append(i)
#         return result

# # from phoneme_feature_inspector import PhonemeFeatureInspector
# inspector = PhonemeFeatureInspector(pipeline)
# # inspector.report_feature_structure()
# inspector.plot_top_phoneme_signals(n_top=3, dataset="train")
# # inspector.plot_all_samples_pooled_representation(dataset="train")
# # inspector.plot_phoneme_mean_amplitude_distribution(dataset="train")

# # hmm
# %matplotlib inline
# for pid in sorted(raw_results.keys()):
#     visualize_patient_model(pid, raw_results, pipeline, show_table = False)

logger = ExperimentLogger('my_experiments.json')

# Run and log - parameters are used directly, no mismatch possible
# Test 1: RF balanced
name, params, results = run_experiment(pipeline, order=1, class_weight=None, classifier_type='dtw_knn', use_viterbi=False)
logger.log(name, params, results)

# Test 2: RF no weights
# name, params, results = run_experiment(pipeline, order=2, class_weight=None, classifier_type='dtw_knn', use_viterbi=False) #dynamic time warping knn
# logger.log(name, params, results)

name, params, results = run_experiment(pipeline, order=1, class_weight=None, classifier_type='logistic_regression', use_viterbi=False)
logger.log(name, params, results)

# name, params, results = run_experiment(pipeline, order=2, class_weight=None, classifier_type='logistic_regression', use_viterbi=False)
# logger.log(name, params, results)

# name, params, results = run_experiment(pipeline, order=3, class_weight=None)
# logger.log(name, params, results)

# name, params, results = run_experiment(pipeline, order=2, classifier_type='extra_trees')
# logger.log(name, params, results)

# # Print comparison
# logger.print_table()

# # Oops, last one was wrong
# logger.remove_last()

"""Boundary precision diagnostic for transition features.

Assesses whether phoneme boundaries are accurate enough to produce
consistent neural features across instances of the same transition type.
Computes within-class vs between-class cosine distances and a
discriminability index (Fisher ratio) per transition type per patient.

If boundaries are correct, instances of the same transition type should
have more similar neural features (lower within-class distance) than
instances of different transition types (higher between-class distance).

Usage:
    from boundary_diagnostic import run_boundary_diagnostic
    run_boundary_diagnostic(pipeline, patient_ids=["P23", "P22"])
"""

import numpy as np

from collections import defaultdict
from scipy.spatial.distance import cosine, pdist, cdist


def summarize_features(feat_2d):
    """Reduce a variable-length feature array to a fixed-length summary.

    Computes channel-wise mean and standard deviation across time frames,
    producing a (2 * n_channels,) vector.

    Args:
        feat_2d: np.array of shape (n_frames, n_channels).

    Returns:
        np.array of shape (2 * n_channels,).
    """
    if feat_2d.ndim != 2 or feat_2d.shape[0] == 0:
        return None
    channel_mean = np.mean(feat_2d, axis=0)
    channel_std = np.std(feat_2d, axis=0)
    return np.concatenate([channel_mean, channel_std])


def compute_discriminability(pipeline, pid, batch_type="train",
                             min_samples=8, top_n=15):
    """Compute within-class and between-class distances for a patient.

    For each transition type with enough samples, computes:
    - Within-class distance: mean pairwise cosine distance among
      instances of that type.
    - Between-class distance: mean cosine distance between that type's
      instances and instances of other types.
    - Fisher ratio: (between - within) / within. Higher is better.
      Values > 1 suggest the class is discriminable. Values < 0.5
      suggest boundaries are too noisy.

    Args:
        pipeline: Dutch30Pipeline with train/test data loaded.
        pid: str, patient ID.
        batch_type: str, 'train' or 'test'.
        min_samples: int, minimum instances per transition type.
        top_n: int, maximum number of transition types to analyze.

    Returns:
        dict with per-type metrics and summary statistics.
    """
    data = pipeline.train if batch_type == "train" else pipeline.test

    # Filter for this patient
    mask = [p == pid for p in data["phoneme_participant_ids"]]
    features = [data["features"][i] for i, m in enumerate(mask) if m]
    labels = [data["phoneme_labels"][i] for i, m in enumerate(mask) if m]

    # Summarize variable-length features to fixed vectors
    by_class = defaultdict(list)
    for feat, label in zip(features, labels):
        summary = summarize_features(feat)
        if summary is not None:
            by_class[label].append(summary)

    # Filter classes with enough samples
    valid_classes = {
        label: np.array(vecs)
        for label, vecs in by_class.items()
        if len(vecs) >= min_samples
    }

    if len(valid_classes) < 2:
        return None

    # Take top_n classes by sample count
    sorted_classes = sorted(
        valid_classes.items(), key=lambda x: -len(x[1])
    )[:top_n]

    # Compute within-class and between-class distances
    results = {}
    all_other_vecs = []
    class_names = []
    class_vecs = []

    for label, vecs in sorted_classes:
        class_names.append(label)
        class_vecs.append(vecs)
        all_other_vecs.extend(vecs)

    for i, (label, vecs) in enumerate(sorted_classes):
        n = len(vecs)

        # Within-class: pairwise cosine distances
        if n < 2:
            within_dist = 0.0
        else:
            within_dists = pdist(vecs, metric="cosine")
            within_dist = np.mean(within_dists)
            within_std = np.std(within_dists)

        # Between-class: distance to all instances of other classes
        other_vecs = []
        for j, (other_label, other_v) in enumerate(sorted_classes):
            if i != j:
                other_vecs.extend(other_v)

        if other_vecs:
            other_vecs = np.array(other_vecs)
            between_dists = cdist(vecs, other_vecs, metric="cosine")
            between_dist = np.mean(between_dists)
            between_std = np.std(between_dists)
        else:
            between_dist = 0.0
            between_std = 0.0

        # Fisher ratio
        fisher = ((between_dist - within_dist) / within_dist
                  if within_dist > 1e-10 else 0.0)

        results[label] = {
            "n_samples": n,
            "within_dist": within_dist,
            "within_std": within_std if n >= 2 else 0.0,
            "between_dist": between_dist,
            "between_std": between_std,
            "fisher_ratio": fisher,
        }

    return results


def run_boundary_diagnostic(pipeline, patient_ids=None,
                            batch_type="train", min_samples=8,
                            top_n=15):
    """Run boundary precision diagnostic across patients.

    Args:
        pipeline: Dutch30Pipeline with train/test data loaded.
        patient_ids: list of patient IDs. Defaults to all patients
            in the data.
        batch_type: str, 'train' or 'test'.
        min_samples: int, minimum instances per transition type.
        top_n: int, max transition types to analyze per patient.
    """
    if patient_ids is None:
        patient_ids = sorted(set(
            pipeline.train["phoneme_participant_ids"]
        ))

    print("=" * 80)
    print("BOUNDARY PRECISION DIAGNOSTIC")
    print("=" * 80)
    print(f"Metric: cosine distance (0 = identical, 1 = orthogonal)")
    print(f"Fisher ratio: (between - within) / within")
    print(f"  > 1.0 = good discriminability")
    print(f"  0.5-1.0 = moderate")
    print(f"  < 0.5 = poor (boundaries likely too noisy)")

    patient_summaries = {}

    for pid in patient_ids:
        results = compute_discriminability(
            pipeline, pid, batch_type, min_samples, top_n
        )

        if results is None:
            print(f"\n{pid}: insufficient data (< {min_samples} per class)")
            continue

        # Sort by fisher ratio
        sorted_results = sorted(
            results.items(), key=lambda x: -x[1]["fisher_ratio"]
        )

        mean_fisher = np.mean([r["fisher_ratio"] for _, r in sorted_results])
        mean_within = np.mean([r["within_dist"] for _, r in sorted_results])
        mean_between = np.mean([r["between_dist"] for _, r in sorted_results])

        patient_summaries[pid] = {
            "mean_fisher": mean_fisher,
            "mean_within": mean_within,
            "mean_between": mean_between,
            "n_classes": len(results),
        }

        print(f"\n{'=' * 70}")
        print(f"{pid} -- {len(results)} transition types analyzed")
        print(f"  Mean within-class dist: {mean_within:.4f}")
        print(f"  Mean between-class dist: {mean_between:.4f}")
        print(f"  Mean Fisher ratio: {mean_fisher:.3f}")
        print(f"{'=' * 70}")

        print(f"{'Type':<12} {'N':>4} {'Within':>8} {'Between':>8} "
              f"{'Fisher':>8} {'Quality'}")
        print("-" * 60)

        for label, metrics in sorted_results:
            fisher = metrics["fisher_ratio"]
            if fisher > 1.0:
                quality = "good"
            elif fisher > 0.5:
                quality = "moderate"
            elif fisher > 0.2:
                quality = "weak"
            else:
                quality = "poor"

            print(f"{label:<12} {metrics['n_samples']:>4} "
                  f"{metrics['within_dist']:>8.4f} "
                  f"{metrics['between_dist']:>8.4f} "
                  f"{metrics['fisher_ratio']:>8.3f} {quality}")

    # Cross-patient summary
    if patient_summaries:
        print(f"\n{'=' * 70}")
        print("CROSS-PATIENT SUMMARY")
        print(f"{'=' * 70}")
        print(f"{'Patient':<10} {'Classes':>8} {'Within':>8} "
              f"{'Between':>8} {'Fisher':>8}")
        print("-" * 50)

        for pid in sorted(patient_summaries.keys()):
            s = patient_summaries[pid]
            print(f"{pid:<10} {s['n_classes']:>8} "
                  f"{s['mean_within']:>8.4f} "
                  f"{s['mean_between']:>8.4f} "
                  f"{s['mean_fisher']:>8.3f}")

import numpy as np
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

def test_extended_hyperparameters(pid, train_data, test_data):
    """Test extended RF hyperparameters for one patient."""
    
    train_mask = [p == pid for p in train_data['phoneme_participant_ids']]
    train_feat = [train_data['features'][i] for i, m in enumerate(train_mask) if m]
    train_labels = [train_data['phoneme_labels'][i] for i, m in enumerate(train_mask) if m]
    
    test_mask = [p == pid for p in test_data['phoneme_participant_ids']]
    test_feat = [test_data['features'][i] for i, m in enumerate(test_mask) if m]
    test_labels = [test_data['phoneme_labels'][i] for i, m in enumerate(test_mask) if m]
    
    if len(train_feat) < 30 or len(test_feat) < 10:
        return None
    
    X_train = np.array([f.flatten() for f in train_feat])
    X_test = np.array([f.flatten() for f in test_feat])
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    baseline = 1.0 / len(set(train_labels))
    
    # Extended configurations
    configs = {
        # Current Markov (baseline)
        'markov_current': {
            'n_estimators': 200, 'max_depth': 20, 'min_samples_leaf': 2,
            'class_weight': 'balanced', 'random_state': 42, 'n_jobs': -1
        },
        # Best from before
        'more_trees_500': {
            'n_estimators': 500, 'max_depth': None, 'min_samples_leaf': 1,
            'class_weight': 'balanced', 'random_state': 42, 'n_jobs': -1
        },
        # Even more trees
        'more_trees_1000': {
            'n_estimators': 1000, 'max_depth': None, 'min_samples_leaf': 1,
            'class_weight': 'balanced', 'random_state': 42, 'n_jobs': -1
        },
        # 500 trees, no class weight
        'trees_500_no_cw': {
            'n_estimators': 500, 'max_depth': None, 'min_samples_leaf': 1,
            'class_weight': None, 'random_state': 42, 'n_jobs': -1
        },
        # 1000 trees, no class weight
        'trees_1000_no_cw': {
            'n_estimators': 1000, 'max_depth': None, 'min_samples_leaf': 1,
            'class_weight': None, 'random_state': 42, 'n_jobs': -1
        },
        # Try balanced_subsample
        'balanced_subsample': {
            'n_estimators': 500, 'max_depth': None, 'min_samples_leaf': 1,
            'class_weight': 'balanced_subsample', 'random_state': 42, 'n_jobs': -1
        },
        # Slight depth limit (might help generalization)
        'depth_200': {
            'n_estimators': 500, 'max_depth': 200, 'min_samples_leaf': 1,
            'class_weight': 'balanced', 'random_state': 42, 'n_jobs': -1
        },
        # min_samples_leaf=2 with more trees
        'trees_500_leaf2': {
            'n_estimators': 500, 'max_depth': None, 'min_samples_leaf': 2,
            'class_weight': 'balanced', 'random_state': 42, 'n_jobs': -1
        },
    }
    
    results = {}
    for name, params in configs.items():
        clf = RandomForestClassifier(**params)
        clf.fit(X_train_scaled, train_labels)
        acc = clf.score(X_test_scaled, test_labels)
        results[name] = {
            'accuracy': acc,
            'lift': acc / baseline
        }
    
    return {
        'pid': pid,
        'n_train': len(train_feat),
        'n_test': len(test_feat),
        'n_classes': len(set(train_labels)),
        'baseline': baseline,
        'results': results
    }


patients_to_test = ['P21', 'P22', 'P23', 'P24', 'P25', 'P26', 'P27', 'P28', 'P29', 'P30']

print("="*120)
print("EXTENDED HYPERPARAMETER COMPARISON FOR PATIENTS P21-P30")
print("="*120)

all_results = []

for pid in patients_to_test:
    result = test_extended_hyperparameters(pid, pipeline.train, pipeline.test)
    if result:
        all_results.append(result)

config_names = list(all_results[0]['results'].keys())

# Lift table
print("\n" + "="*120)
print("LIFT BY PATIENT AND CONFIGURATION")
print("="*120)

header = f"{'Patient':<8}"
for name in config_names:
    header += f"{name[:15]:<17}"
print(header)
print("-" * (8 + 17 * len(config_names)))

for result in all_results:
    row = f"{result['pid']:<8}"
    for name in config_names:
        lift = result['results'][name]['lift']
        row += f"{lift:<17.2f}"
    print(row)

print("-" * (8 + 17 * len(config_names)))
row = f"{'MEAN':<8}"
for name in config_names:
    mean_lift = np.mean([r['results'][name]['lift'] for r in all_results])
    row += f"{mean_lift:<17.2f}"
print(row)

# Summary
print("\n" + "="*120)
print("RANKED BY MEAN LIFT")
print("="*120)

ranked = []
for name in config_names:
    mean_lift = np.mean([r['results'][name]['lift'] for r in all_results])
    mean_acc = np.mean([r['results'][name]['accuracy'] for r in all_results])
    ranked.append((name, mean_lift, mean_acc))

ranked.sort(key=lambda x: x[1], reverse=True)

print(f"\n{'Rank':<6} {'Config':<25} {'Mean Lift':<12} {'Mean Acc':<12}")
print("-" * 55)
for i, (name, lift, acc) in enumerate(ranked, 1):
    print(f"{i:<6} {name:<25} {lift:<12.2f}x {acc:<12.3f}")

def compare_markov_classifiers(pipeline):
    """Compare different classifiers within Markov framework."""
    
    print("="*80)
    print("COMPARING CLASSIFIERS WITHIN MARKOV MODEL")
    print("="*80)
    
    # Test on a few patients
    test_pids = ['P21', 'P22', 'P23', 'P24', 'P25', 'P26', 'P27', 'P28',  'P29', 'P30']
    
    classifiers_to_test = ['dtw_knn', 'knn', 'random_forest', 'rf', 'svm', 'logistic']
    
    results = {clf: [] for clf in classifiers_to_test}
    
    for pid in test_pids:
        train_mask = [p == pid for p in pipeline.train['phoneme_participant_ids']]
        train_feat = [pipeline.train['features'][i] for i, m in enumerate(train_mask) if m]
        train_labels = [pipeline.train['phoneme_labels'][i] for i, m in enumerate(train_mask) if m]
        
        test_mask = [p == pid for p in pipeline.test['phoneme_participant_ids']]
        test_feat = [pipeline.test['features'][i] for i, m in enumerate(test_mask) if m]
        test_labels = [pipeline.test['phoneme_labels'][i] for i, m in enumerate(test_mask) if m]
        
        if len(train_feat) < 30 or len(test_feat) < 10:
            continue
        
        baseline = 1.0 / len(set(train_labels))
        
        print(f"\n{pid}:")
        
        for clf_type in classifiers_to_test:
            try:
                model = MarkovPhonemeModel(
                    phonetic_dict=pipeline.detector.phonetic_dict,
                    order=3,
                    use_groups=False,
                    class_weight='balanced',
                    classifier_type=clf_type
                )
                model.train(features=train_feat, phoneme_labels=train_labels)
                preds, _ = model.predict(test_feat, use_viterbi=True)
                
                acc = sum(1 for p, t in zip(preds, test_labels) if p == t) / len(test_labels)
                lift = acc / baseline
                
                results[clf_type].append({'pid': pid, 'acc': acc, 'lift': lift})
                print(f"  {clf_type}: {acc:.3f} ({lift:.2f}x)")
                
            except Exception as e:
                print(f"  {clf_type}: ERROR - {e}")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    for clf_type, res in results.items():
        if res:
            mean_acc = np.mean([r['acc'] for r in res])
            mean_lift = np.mean([r['lift'] for r in res])
            print(f"{clf_type}: Mean acc={mean_acc:.3f}, Mean lift={mean_lift:.2f}x")


compare_markov_classifiers(pipeline)

import numpy as np
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

def test_hyperparameters_for_patient(pid, train_data, test_data):
    """Test different RF hyperparameters for one patient."""
    
    # Get patient data
    train_mask = [p == pid for p in train_data['phoneme_participant_ids']]
    train_feat = [train_data['features'][i] for i, m in enumerate(train_mask) if m]
    train_labels = [train_data['phoneme_labels'][i] for i, m in enumerate(train_mask) if m]
    
    test_mask = [p == pid for p in test_data['phoneme_participant_ids']]
    test_feat = [test_data['features'][i] for i, m in enumerate(test_mask) if m]
    test_labels = [test_data['phoneme_labels'][i] for i, m in enumerate(test_mask) if m]
    
    if len(train_feat) < 30 or len(test_feat) < 10:
        return None
    
    # Prepare features
    X_train = np.array([f.flatten() for f in train_feat])
    X_test = np.array([f.flatten() for f in test_feat])
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    baseline = 1.0 / len(set(train_labels))
    
    # Hyperparameter configurations to test
    configs = {
        'default': {
            'n_estimators': 100, 
            'random_state': 42, 
            'n_jobs': -1
        },
        'markov_current': {
            'n_estimators': 200, 
            'max_depth': 20, 
            'min_samples_leaf': 2,
            'class_weight': 'balanced', 
            'random_state': 42, 
            'n_jobs': -1
        },
        'no_depth_limit': {
            'n_estimators': 200, 
            'max_depth': None, 
            'min_samples_leaf': 2,
            'class_weight': 'balanced', 
            'random_state': 42, 
            'n_jobs': -1
        },
        'depth_50': {
            'n_estimators': 200, 
            'max_depth': 50, 
            'min_samples_leaf': 2,
            'class_weight': 'balanced', 
            'random_state': 42, 
            'n_jobs': -1
        },
        'depth_100': {
            'n_estimators': 200, 
            'max_depth': 100, 
            'min_samples_leaf': 2,
            'class_weight': 'balanced', 
            'random_state': 42, 
            'n_jobs': -1
        },
        'no_restrictions': {
            'n_estimators': 200, 
            'max_depth': None, 
            'min_samples_leaf': 1,
            'class_weight': 'balanced', 
            'random_state': 42, 
            'n_jobs': -1
        },
        'no_class_weight': {
            'n_estimators': 200, 
            'max_depth': None, 
            'min_samples_leaf': 1,
            'random_state': 42, 
            'n_jobs': -1
        },
        'more_trees': {
            'n_estimators': 500, 
            'max_depth': None, 
            'min_samples_leaf': 1,
            'class_weight': 'balanced', 
            'random_state': 42, 
            'n_jobs': -1
        },
    }
    
    results = {}
    for name, params in configs.items():
        clf = RandomForestClassifier(**params)
        clf.fit(X_train_scaled, train_labels)
        acc = clf.score(X_test_scaled, test_labels)
        results[name] = {
            'accuracy': acc,
            'lift': acc / baseline,
            'baseline': baseline
        }
    
    return {
        'pid': pid,
        'n_train': len(train_feat),
        'n_test': len(test_feat),
        'n_classes': len(set(train_labels)),
        'baseline': baseline,
        'results': results
    }


# Test patients P21-P30
patients_to_test = ['P21', 'P22', 'P23', 'P24', 'P25', 'P26', 'P27', 'P28', 'P29', 'P30']

print("="*100)
print("HYPERPARAMETER COMPARISON FOR PATIENTS P21-P30")
print("="*100)

all_results = []

for pid in patients_to_test:
    result = test_hyperparameters_for_patient(pid, pipeline.train, pipeline.test)
    if result:
        all_results.append(result)
        print(f"\n{pid}: {result['n_train']} train, {result['n_test']} test, {result['n_classes']} classes")

# Create summary table
config_names = list(all_results[0]['results'].keys())

print("\n" + "="*100)
print("ACCURACY BY PATIENT AND CONFIGURATION")
print("="*100)

# Header
header = f"{'Patient':<8}"
for name in config_names:
    header += f"{name[:12]:<14}"
print(header)
print("-" * (8 + 14 * len(config_names)))

# Data rows
for result in all_results:
    row = f"{result['pid']:<8}"
    for name in config_names:
        acc = result['results'][name]['accuracy']
        row += f"{acc:<14.3f}"
    print(row)

# Mean row
print("-" * (8 + 14 * len(config_names)))
row = f"{'MEAN':<8}"
for name in config_names:
    mean_acc = np.mean([r['results'][name]['accuracy'] for r in all_results])
    row += f"{mean_acc:<14.3f}"
print(row)


print("\n" + "="*100)
print("LIFT BY PATIENT AND CONFIGURATION")
print("="*100)

# Header
print(header)
print("-" * (8 + 14 * len(config_names)))

# Data rows
for result in all_results:
    row = f"{result['pid']:<8}"
    for name in config_names:
        lift = result['results'][name]['lift']
        row += f"{lift:<14.2f}"
    print(row)

# Mean row
print("-" * (8 + 14 * len(config_names)))
row = f"{'MEAN':<8}"
for name in config_names:
    mean_lift = np.mean([r['results'][name]['lift'] for r in all_results])
    row += f"{mean_lift:<14.2f}"
print(row)


# Find best config per patient
print("\n" + "="*100)
print("BEST CONFIGURATION PER PATIENT")
print("="*100)

print(f"\n{'Patient':<10} {'Best Config':<20} {'Accuracy':<12} {'Lift':<10}")
print("-" * 55)

best_config_counts = Counter()

for result in all_results:
    best_name = max(result['results'].keys(), key=lambda x: result['results'][x]['accuracy'])
    best_acc = result['results'][best_name]['accuracy']
    best_lift = result['results'][best_name]['lift']
    best_config_counts[best_name] += 1
    print(f"{result['pid']:<10} {best_name:<20} {best_acc:<12.3f} {best_lift:<10.2f}x")

print("\n" + "="*100)
print("SUMMARY: Which config wins most often?")
print("="*100)

for name, count in best_config_counts.most_common():
    mean_lift = np.mean([r['results'][name]['lift'] for r in all_results])
    print(f"  {name}: {count} patients, mean lift = {mean_lift:.2f}x")


# Recommendation
print("\n" + "="*100)
print("RECOMMENDATION FOR MARKOV MODEL")
print("="*100)

# Find overall best config by mean lift
best_overall = max(config_names, key=lambda x: np.mean([r['results'][x]['lift'] for r in all_results]))
best_lift = np.mean([r['results'][best_overall]['lift'] for r in all_results])
current_lift = np.mean([r['results']['markov_current']['lift'] for r in all_results])

print(f"""
Current Markov config:
  max_depth=20, min_samples_leaf=2, class_weight='balanced'
  Mean lift: {current_lift:.2f}x

Recommended config: {best_overall}
  Mean lift: {best_lift:.2f}x

Improvement: {best_lift - current_lift:.2f}x lift increase ({100*(best_lift-current_lift)/current_lift:.0f}% improvement)
""")

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.stats import f_oneway, ttest_ind

pid = 'P29'
mask = [p == pid for p in pipeline.train['phoneme_participant_ids']]
patient_features = [f for f, m in zip(pipeline.train['features'], mask) if m]
patient_labels = [l for l, m in zip(pipeline.train['phoneme_labels'], mask) if m]

phoneme_features = defaultdict(list)
for feat, label in zip(patient_features, patient_labels):
    phoneme_features[label].append(feat)


# ============================================================
# Different alignment strategies
# ============================================================

def align_to_peak(feat, frames_left, frames_right):
    """Align to maximum amplitude frame."""
    temporal_profile = feat.mean(axis=1)
    peak_idx = np.argmax(temporal_profile)
    if peak_idx < frames_left or (feat.shape[0] - peak_idx - 1) < frames_right:
        return None
    return feat[peak_idx - frames_left : peak_idx + frames_right + 1, :]


def align_to_min(feat, frames_left, frames_right):
    """Align to minimum amplitude frame."""
    temporal_profile = feat.mean(axis=1)
    min_idx = np.argmin(temporal_profile)
    if min_idx < frames_left or (feat.shape[0] - min_idx - 1) < frames_right:
        return None
    return feat[min_idx - frames_left : min_idx + frames_right + 1, :]


def align_to_max_derivative(feat, frames_left, frames_right):
    """Align to maximum change (steepest rise)."""
    temporal_profile = feat.mean(axis=1)
    if len(temporal_profile) < 3:
        return None
    derivative = np.diff(temporal_profile)
    max_deriv_idx = np.argmax(derivative) + 1  # +1 because diff shortens by 1
    if max_deriv_idx < frames_left or (feat.shape[0] - max_deriv_idx - 1) < frames_right:
        return None
    return feat[max_deriv_idx - frames_left : max_deriv_idx + frames_right + 1, :]


def align_to_min_derivative(feat, frames_left, frames_right):
    """Align to steepest fall."""
    temporal_profile = feat.mean(axis=1)
    if len(temporal_profile) < 3:
        return None
    derivative = np.diff(temporal_profile)
    min_deriv_idx = np.argmin(derivative) + 1
    if min_deriv_idx < frames_left or (feat.shape[0] - min_deriv_idx - 1) < frames_right:
        return None
    return feat[min_deriv_idx - frames_left : min_deriv_idx + frames_right + 1, :]


def align_to_max_variance(feat, frames_left, frames_right):
    """Align to frame with highest variance across channels."""
    variance_profile = feat.var(axis=1)
    max_var_idx = np.argmax(variance_profile)
    if max_var_idx < frames_left or (feat.shape[0] - max_var_idx - 1) < frames_right:
        return None
    return feat[max_var_idx - frames_left : max_var_idx + frames_right + 1, :]


def align_from_start(feat, total_frames):
    """Take first N frames (onset alignment)."""
    if feat.shape[0] < total_frames:
        return None
    return feat[:total_frames, :]


def align_from_end(feat, total_frames):
    """Take last N frames (offset alignment)."""
    if feat.shape[0] < total_frames:
        return None
    return feat[-total_frames:, :]


def align_center(feat, total_frames):
    """Take center N frames."""
    if feat.shape[0] < total_frames:
        return None
    start = (feat.shape[0] - total_frames) // 2
    return feat[start : start + total_frames, :]


def evaluate_alignment(align_func, phoneme_features, min_samples=5, **kwargs):
    """Evaluate an alignment strategy."""
    phonemes_aligned = {}
    
    for phoneme, feats in phoneme_features.items():
        aligned = [align_func(f, **kwargs) for f in feats]
        aligned = [a for a in aligned if a is not None]
        if len(aligned) >= min_samples:
            phonemes_aligned[phoneme] = aligned
    
    if len(phonemes_aligned) < 2:
        return None
    
    # Get means per phoneme
    phoneme_means = {}
    all_means = []
    all_labels = []
    
    for phoneme, aligned_list in phonemes_aligned.items():
        means = [f.mean() for f in aligned_list]
        phoneme_means[phoneme] = {
            'means': means,
            'mean': np.mean(means),
            'std': np.std(means),
            'n': len(means)
        }
        all_means.extend(means)
        all_labels.extend([phoneme] * len(means))
    
    # ANOVA
    groups = [phoneme_means[p]['means'] for p in phoneme_means]
    f_stat, p_value = f_oneway(*groups)
    
    # Eta-squared
    grand_mean = np.mean(all_means)
    ss_between = sum(len(phoneme_means[p]['means']) * (phoneme_means[p]['mean'] - grand_mean)**2 
                     for p in phoneme_means)
    ss_total = sum((x - grand_mean)**2 for x in all_means)
    eta_squared = ss_between / ss_total if ss_total > 0 else 0
    
    # Count significant pairs
    phoneme_list = list(phoneme_means.keys())
    n_sig_pairs = 0
    n_total_pairs = 0
    
    for i, p1 in enumerate(phoneme_list):
        for j, p2 in enumerate(phoneme_list):
            if i < j:
                _, pval = ttest_ind(phoneme_means[p1]['means'], phoneme_means[p2]['means'])
                n_total_pairs += 1
                if pval < 0.05:
                    n_sig_pairs += 1
    
    return {
        'n_phonemes': len(phonemes_aligned),
        'n_samples': len(all_means),
        'f_stat': f_stat,
        'p_value': p_value,
        'eta_squared': eta_squared,
        'n_sig_pairs': n_sig_pairs,
        'n_total_pairs': n_total_pairs,
        'pct_sig_pairs': 100 * n_sig_pairs / n_total_pairs if n_total_pairs > 0 else 0,
        'phoneme_means': phoneme_means
    }


# ============================================================
# Test different alignment strategies
# ============================================================
print("="*100)
print("COMPARING ALIGNMENT STRATEGIES")
print("="*100)

results = []

# Test different window sizes for peak alignment
for fl, fr in [(1, 1), (2, 2), (3, 3), (4, 4), (2, 3), (3, 2), (1, 3), (3, 1)]:
    r = evaluate_alignment(align_to_peak, phoneme_features, frames_left=fl, frames_right=fr)
    if r:
        results.append({
            'method': f'Peak ({fl}L+{fr}R)',
            'total_frames': fl + 1 + fr,
            **r
        })

# Test different alignment points
for fl, fr in [(2, 2)]:
    r = evaluate_alignment(align_to_min, phoneme_features, frames_left=fl, frames_right=fr)
    if r:
        results.append({'method': f'Min ({fl}L+{fr}R)', 'total_frames': fl + 1 + fr, **r})
    
    r = evaluate_alignment(align_to_max_derivative, phoneme_features, frames_left=fl, frames_right=fr)
    if r:
        results.append({'method': f'MaxDeriv ({fl}L+{fr}R)', 'total_frames': fl + 1 + fr, **r})
    
    r = evaluate_alignment(align_to_min_derivative, phoneme_features, frames_left=fl, frames_right=fr)
    if r:
        results.append({'method': f'MinDeriv ({fl}L+{fr}R)', 'total_frames': fl + 1 + fr, **r})
    
    r = evaluate_alignment(align_to_max_variance, phoneme_features, frames_left=fl, frames_right=fr)
    if r:
        results.append({'method': f'MaxVar ({fl}L+{fr}R)', 'total_frames': fl + 1 + fr, **r})

# Test onset/offset/center alignment
for total in [5, 7, 9]:
    r = evaluate_alignment(align_from_start, phoneme_features, total_frames=total)
    if r:
        results.append({'method': f'Onset ({total}f)', 'total_frames': total, **r})
    
    r = evaluate_alignment(align_from_end, phoneme_features, total_frames=total)
    if r:
        results.append({'method': f'Offset ({total}f)', 'total_frames': total, **r})
    
    r = evaluate_alignment(align_center, phoneme_features, total_frames=total)
    if r:
        results.append({'method': f'Center ({total}f)', 'total_frames': total, **r})

# Sort by eta-squared (variance explained)
results.sort(key=lambda x: x['eta_squared'], reverse=True)

print(f"\n{'Method':<22} {'Frames':<8} {'Phonemes':<10} {'Samples':<10} {'Eta-sq':<10} {'ANOVA p':<12} {'Sig pairs %':<12}")
print("-" * 100)

for r in results:
    sig = "***" if r['p_value'] < 0.001 else "**" if r['p_value'] < 0.01 else "*" if r['p_value'] < 0.05 else ""
    print(f"{r['method']:<22} {r['total_frames']:<8} {r['n_phonemes']:<10} {r['n_samples']:<10} "
          f"{r['eta_squared']:<10.4f} {r['p_value']:<12.4f} {r['pct_sig_pairs']:<10.1f}% {sig}")


# ============================================================
# Visualize top 3 methods
# ============================================================
top_methods = results[:3]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, r in enumerate(top_methods):
    ax = axes[idx]
    
    # Sort phonemes by mean
    phoneme_stats = [(p, r['phoneme_means'][p]) for p in r['phoneme_means']]
    phoneme_stats.sort(key=lambda x: x[1]['mean'], reverse=True)
    
    phonemes = [p for p, _ in phoneme_stats]
    means = [s['mean'] for _, s in phoneme_stats]
    stds = [s['std'] for _, s in phoneme_stats]
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(phonemes)))
    
    ax.bar(range(len(phonemes)), means, yerr=stds, capsize=3,
           color=colors, alpha=0.7, edgecolor='black')
    
    ax.set_xticks(range(len(phonemes)))
    ax.set_xticklabels([f"'{p}'" for p in phonemes], rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Mean amplitude')
    ax.set_title(f"{r['method']}\nEta-sq={r['eta_squared']:.4f}, Sig pairs={r['pct_sig_pairs']:.1f}%")
    ax.grid(True, alpha=0.3, axis='y')

plt.suptitle(f'{pid}: Top 3 alignment strategies\n(Higher eta-squared = more variance explained by phoneme identity)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{pid}_alignment_comparison.png', dpi=150)
plt.show()


# ============================================================
# Detailed analysis of best method
# ============================================================
best = results[0]
print("\n" + "="*100)
print(f"BEST METHOD: {best['method']}")
print("="*100)

print(f"\nEta-squared: {best['eta_squared']:.4f} ({100*best['eta_squared']:.1f}% variance explained)")
print(f"ANOVA p-value: {best['p_value']:.6f}")
print(f"Significant pairs: {best['n_sig_pairs']} / {best['n_total_pairs']} ({best['pct_sig_pairs']:.1f}%)")

print(f"\n{'Phoneme':<10} {'N':<8} {'Mean':<12} {'Std':<12}")
print("-" * 45)

phoneme_stats = [(p, best['phoneme_means'][p]) for p in best['phoneme_means']]
phoneme_stats.sort(key=lambda x: x[1]['mean'], reverse=True)

for p, s in phoneme_stats:
    print(f"'{p}'       {s['n']:<8} {s['mean']:<12.4f} {s['std']:<12.4f}")

# Check the shape of features in your pipeline
print("Sample feature shapes:")
for i in range(5):
    print(f"  Sample {i}: {pipeline.train['features'][i].shape}")

# If all shapes are (5, n_channels) → alignment is in pipeline
# If shapes vary → alignment is NOT in pipeline

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler

def align_from_start(feat, total_frames=5):
    """Onset alignment - take first N frames with padding if needed."""
    if feat.shape[0] < total_frames:
        padding = np.tile(feat[-1:, :], (total_frames - feat.shape[0], 1))
        return np.vstack([feat, padding])
    return feat[:total_frames, :]


# ============================================================
# Collect final results for all patients
# ============================================================
all_patients = sorted(set(pipeline.train['phoneme_participant_ids']))

results = []
for pid in all_patients:
    mask = [p == pid for p in pipeline.train['phoneme_participant_ids']]
    patient_features = [f for f, m in zip(pipeline.train['features'], mask) if m]
    patient_labels = [l for l, m in zip(pipeline.train['phoneme_labels'], mask) if m]
    
    # Align
    aligned = [align_from_start(f, 5) for f in patient_features]
    
    # Filter classes
    class_counts = Counter(patient_labels)
    valid_classes = {c for c, count in class_counts.items() if count >= 5}
    
    X = np.array([a.flatten() for a, l in zip(aligned, patient_labels) if l in valid_classes])
    y = np.array([l for l in patient_labels if l in valid_classes])
    
    if len(X) < 30 or len(valid_classes) < 3:
        continue
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    n_splits = min(5, min(Counter(y).values()))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring='accuracy')
    
    baseline = 1.0 / len(valid_classes)
    
    results.append({
        'pid': pid,
        'n_samples': len(y),
        'n_classes': len(valid_classes),
        'accuracy': scores.mean(),
        'std': scores.std(),
        'baseline': baseline,
        'lift': scores.mean() / baseline
    })

results.sort(key=lambda x: x['lift'], reverse=True)


# ============================================================
# Create thesis-ready figure
# ============================================================
fig = plt.figure(figsize=(16, 10))

# Layout: 2x2 grid
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

# Plot 1: Bar chart - Accuracy vs Baseline by patient
ax1 = fig.add_subplot(gs[0, 0])
pids = [r['pid'] for r in results]
accs = [r['accuracy'] for r in results]
baselines = [r['baseline'] for r in results]

x = np.arange(len(pids))
width = 0.35

bars1 = ax1.bar(x - width/2, accs, width, label='Model accuracy', color='steelblue', alpha=0.8)
bars2 = ax1.bar(x + width/2, baselines, width, label='Random baseline', color='lightcoral', alpha=0.8)

ax1.set_xticks(x)
ax1.set_xticklabels(pids, rotation=45, ha='right', fontsize=8)
ax1.set_ylabel('Accuracy')
ax1.set_title('A) Classification Accuracy vs Random Baseline')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3, axis='y')

# Plot 2: Lift by patient
ax2 = fig.add_subplot(gs[0, 1])
lifts = [r['lift'] for r in results]
colors = ['green' if l > 2.0 else 'steelblue' if l > 1.5 else 'orange' for l in lifts]

ax2.bar(x, lifts, color=colors, alpha=0.8, edgecolor='black')
ax2.axhline(1.0, color='red', linestyle='--', linewidth=2, label='Chance level (1.0x)')
ax2.axhline(2.0, color='green', linestyle=':', linewidth=1, alpha=0.5, label='2x lift')

ax2.set_xticks(x)
ax2.set_xticklabels(pids, rotation=45, ha='right', fontsize=8)
ax2.set_ylabel('Lift (Accuracy / Random Baseline)')
ax2.set_title('B) Classification Lift Over Random Baseline')
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3, axis='y')

# Plot 3: Histogram of lift values
ax3 = fig.add_subplot(gs[1, 0])
ax3.hist(lifts, bins=12, edgecolor='black', alpha=0.7, color='steelblue')
ax3.axvline(1.0, color='red', linestyle='--', linewidth=2, label='Chance (1.0x)')
ax3.axvline(np.mean(lifts), color='green', linestyle='-', linewidth=2, 
            label=f'Mean ({np.mean(lifts):.2f}x)')
ax3.axvline(np.median(lifts), color='orange', linestyle='-', linewidth=2,
            label=f'Median ({np.median(lifts):.2f}x)')

ax3.set_xlabel('Lift (Accuracy / Random Baseline)')
ax3.set_ylabel('Number of Patients')
ax3.set_title('C) Distribution of Classification Lift')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Summary statistics table
ax4 = fig.add_subplot(gs[1, 1])
ax4.axis('off')

# Create summary text
summary_text = f"""
PHONEME CLASSIFICATION RESULTS
{'='*50}

Method: Onset alignment (first 5 frames = 30ms)
Features: Flattened high-gamma envelope
Classifier: RandomForest (100 trees, 5-fold CV)

OVERALL PERFORMANCE (n={len(results)} patients):
  Mean accuracy:     {np.mean(accs):.1%} +/- {np.std(accs):.1%}
  Mean lift:         {np.mean(lifts):.2f}x
  Median lift:       {np.median(lifts):.2f}x
  Min lift:          {min(lifts):.2f}x ({results[-1]['pid']})
  Max lift:          {max(lifts):.2f}x ({results[0]['pid']})

PATIENTS ABOVE THRESHOLD:
  Lift > 1.0x:       {sum(1 for l in lifts if l > 1.0)}/{len(lifts)} ({100*sum(1 for l in lifts if l > 1.0)/len(lifts):.0f}%)
  Lift > 1.5x:       {sum(1 for l in lifts if l > 1.5)}/{len(lifts)} ({100*sum(1 for l in lifts if l > 1.5)/len(lifts):.0f}%)
  Lift > 2.0x:       {sum(1 for l in lifts if l > 2.0)}/{len(lifts)} ({100*sum(1 for l in lifts if l > 2.0)/len(lifts):.0f}%)

TOP 5 PATIENTS:
  1. {results[0]['pid']}: {results[0]['accuracy']:.1%} acc, {results[0]['lift']:.2f}x lift
  2. {results[1]['pid']}: {results[1]['accuracy']:.1%} acc, {results[1]['lift']:.2f}x lift
  3. {results[2]['pid']}: {results[2]['accuracy']:.1%} acc, {results[2]['lift']:.2f}x lift
  4. {results[3]['pid']}: {results[3]['accuracy']:.1%} acc, {results[3]['lift']:.2f}x lift
  5. {results[4]['pid']}: {results[4]['accuracy']:.1%} acc, {results[4]['lift']:.2f}x lift
"""

ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle('Phoneme Classification from High-Gamma Neural Features\n(Dutch30 Dataset, Onset Alignment)',
             fontsize=14, fontweight='bold')
plt.savefig('thesis_figure_phoneme_classification.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\n{'Patient':<8} {'Samples':<10} {'Classes':<10} {'Accuracy':<12} {'Baseline':<12} {'Lift':<10}")
print("-" * 65)

for r in results:
    print(f"{r['pid']:<8} {r['n_samples']:<10} {r['n_classes']:<10} "
          f"{r['accuracy']:<12.3f} {r['baseline']:<12.3f} {r['lift']:<10.2f}x")

print("-" * 65)
print(f"{'MEAN':<8} {np.mean([r['n_samples'] for r in results]):<10.0f} "
      f"{np.mean([r['n_classes'] for r in results]):<10.1f} "
      f"{np.mean(accs):<12.3f} {np.mean(baselines):<12.3f} {np.mean(lifts):<10.2f}x")

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import resample


def extract_example_features(pipeline, phoneme_target='a', n_examples=5):
    """Extract example features for a specific phoneme.
    
    Args:
        pipeline: Your Dutch30Pipeline with loaded data.
        phoneme_target: Phoneme to extract examples for.
        n_examples: Number of examples to extract.
        
    Returns:
        List of feature arrays (n_frames, n_channels).
    """
    features = []
    labels = pipeline.train['phoneme_labels']
    feats = pipeline.train['features']
    
    for i, label in enumerate(labels):
        if label == phoneme_target:
            features.append(feats[i])
            if len(features) >= n_examples:
                break
    
    return features


def pool_features(feat):
    """Statistical pooling: collapse time to mean/std/min/max per channel.
    
    Args:
        feat: Array of shape (n_frames, n_channels).
        
    Returns:
        Array of shape (n_channels * 4,).
    """
    mean = feat.mean(axis=0)
    std = feat.std(axis=0)
    min_val = feat.min(axis=0)
    max_val = feat.max(axis=0)
    return np.concatenate([mean, std, min_val, max_val])


def resample_and_flatten(feat, target_frames=10):
    """Resample to fixed frames, then flatten.
    
    Args:
        feat: Array of shape (n_frames, n_channels).
        target_frames: Number of frames to resample to.
        
    Returns:
        Array of shape (target_frames * n_channels,).
    """
    if feat.shape[0] != target_frames:
        resampled = resample(feat, target_frames, axis=0)
    else:
        resampled = feat
    return resampled.flatten(), resampled


def visualize_single_feature(feat, title="Feature matrix"):
    """Visualize a single feature matrix as heatmap.
    
    Args:
        feat: Array of shape (n_frames, n_channels).
        title: Plot title.
    """
    plt.figure(figsize=(12, 4))
    plt.imshow(feat.T, aspect='auto', cmap='viridis', origin='lower')
    plt.colorbar(label='Amplitude')
    plt.xlabel('Frame (time)')
    plt.ylabel('Channel')
    plt.title(f'{title}\nShape: {feat.shape}')
    plt.tight_layout()
    plt.show()


def visualize_information_loss(feat, target_frames=10):
    """Show what each processing approach preserves and loses.
    
    Args:
        feat: Original feature array (n_frames, n_channels).
        target_frames: Target frames for resampling.
    """
    n_frames, n_channels = feat.shape
    
    # Process with each method
    flattened, resampled = resample_and_flatten(feat, target_frames)
    pooled = pool_features(feat)
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # Row 1: Feature matrices
    # Original
    im1 = axes[0, 0].imshow(feat.T, aspect='auto', cmap='viridis', origin='lower')
    axes[0, 0].set_title(f'Original\n({n_frames} frames x {n_channels} ch)')
    axes[0, 0].set_xlabel('Frame')
    axes[0, 0].set_ylabel('Channel')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Resampled
    im2 = axes[0, 1].imshow(resampled.T, aspect='auto', cmap='viridis', origin='lower')
    axes[0, 1].set_title(f'Resampled\n({target_frames} frames x {n_channels} ch)')
    axes[0, 1].set_xlabel('Frame')
    axes[0, 1].set_ylabel('Channel')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Pooled (reshape for visualization)
    pooled_reshaped = pooled.reshape(4, n_channels)  # 4 stats x n_channels
    im3 = axes[0, 2].imshow(pooled_reshaped, aspect='auto', cmap='viridis', origin='lower')
    axes[0, 2].set_title(f'Pooled statistics\n(4 stats x {n_channels} ch)')
    axes[0, 2].set_xlabel('Channel')
    axes[0, 2].set_ylabel('Statistic')
    axes[0, 2].set_yticks([0, 1, 2, 3])
    axes[0, 2].set_yticklabels(['mean', 'std', 'min', 'max'])
    plt.colorbar(im3, ax=axes[0, 2])
    
    # Row 2: What can be reconstructed
    # Original temporal profile (mean across channels)
    temporal_profile = feat.mean(axis=1)
    axes[1, 0].plot(temporal_profile, 'b-', linewidth=2)
    axes[1, 0].set_title('Original temporal profile\n(mean across channels)')
    axes[1, 0].set_xlabel('Frame')
    axes[1, 0].set_ylabel('Mean amplitude')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Resampled temporal profile
    resampled_profile = resampled.mean(axis=1)
    axes[1, 1].plot(resampled_profile, 'g-', linewidth=2)
    axes[1, 1].plot(np.linspace(0, target_frames-1, n_frames), temporal_profile, 
                    'b--', alpha=0.5, label='Original (scaled)')
    axes[1, 1].set_title('Resampled temporal profile\n(interpolated)')
    axes[1, 1].set_xlabel('Frame')
    axes[1, 1].set_ylabel('Mean amplitude')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Pooled: no temporal information, show what we have
    mean_val = pooled[:n_channels].mean()
    std_val = pooled[n_channels:2*n_channels].mean()
    min_val = pooled[2*n_channels:3*n_channels].mean()
    max_val = pooled[3*n_channels:].mean()
    
    axes[1, 2].bar(['mean', 'std', 'min', 'max'], [mean_val, std_val, min_val, max_val])
    axes[1, 2].set_title('Pooled: temporal info LOST\n(only these 4 values per channel)')
    axes[1, 2].set_ylabel('Average across channels')
    axes[1, 2].axhline(y=temporal_profile.mean(), color='b', linestyle='--', 
                       label='True mean', alpha=0.7)
    axes[1, 2].legend()
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print("\n" + "="*60)
    print("INFORMATION PRESERVED BY EACH METHOD")
    print("="*60)
    print(f"\nOriginal: {n_frames} frames x {n_channels} channels = {n_frames * n_channels} values")
    print(f"  - Full temporal dynamics preserved")
    print(f"  - All frame-channel relationships preserved")
    print(f"\nResampled + Flattened: {target_frames} x {n_channels} = {target_frames * n_channels} values")
    print(f"  - Temporal order PRESERVED (frame 0 = onset, frame 9 = offset)")
    print(f"  - Interpolated to fixed length")
    print(f"  - Classifier can learn 'channel X at frame Y matters'")
    print(f"\nPooled: 4 x {n_channels} = {4 * n_channels} values")
    print(f"  - Temporal order LOST")
    print(f"  - Only knows mean/std/min/max per channel")
    print(f"  - Cannot learn 'channel X at frame Y matters'")
    print(f"  - CAN learn 'channel X has high variance' or 'channel X peaks high'")


def compare_two_phonemes(feat1, feat2, label1, label2, target_frames=10):
    """Compare how two phonemes differ and what each method preserves.
    
    Args:
        feat1: Feature array for phoneme 1.
        feat2: Feature array for phoneme 2.
        label1: Label for phoneme 1.
        label2: Label for phoneme 2.
        target_frames: Target frames for resampling.
    """
    n_channels = feat1.shape[1]
    
    # Process both
    _, resampled1 = resample_and_flatten(feat1, target_frames)
    _, resampled2 = resample_and_flatten(feat2, target_frames)
    pooled1 = pool_features(feat1)
    pooled2 = pool_features(feat2)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # Temporal profiles
    axes[0, 0].plot(feat1.mean(axis=1), 'b-', linewidth=2, label=label1)
    axes[0, 0].plot(feat2.mean(axis=1), 'r-', linewidth=2, label=label2)
    axes[0, 0].set_title('Original temporal profiles')
    axes[0, 0].set_xlabel('Frame')
    axes[0, 0].set_ylabel('Mean amplitude')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(resampled1.mean(axis=1), 'b-', linewidth=2, label=label1)
    axes[0, 1].plot(resampled2.mean(axis=1), 'r-', linewidth=2, label=label2)
    axes[0, 1].set_title('Resampled profiles (aligned)')
    axes[0, 1].set_xlabel('Frame')
    axes[0, 1].set_ylabel('Mean amplitude')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Pooled comparison
    x = np.arange(4)
    width = 0.35
    stats1 = [pooled1[:n_channels].mean(), pooled1[n_channels:2*n_channels].mean(),
              pooled1[2*n_channels:3*n_channels].mean(), pooled1[3*n_channels:].mean()]
    stats2 = [pooled2[:n_channels].mean(), pooled2[n_channels:2*n_channels].mean(),
              pooled2[2*n_channels:3*n_channels].mean(), pooled2[3*n_channels:].mean()]
    
    axes[0, 2].bar(x - width/2, stats1, width, label=label1)
    axes[0, 2].bar(x + width/2, stats2, width, label=label2)
    axes[0, 2].set_title('Pooled statistics')
    axes[0, 2].set_xticks(x)
    axes[0, 2].set_xticklabels(['mean', 'std', 'min', 'max'])
    axes[0, 2].legend()
    
    # Difference maps
    diff_original = np.abs(resample(feat1, target_frames, axis=0) - 
                          resample(feat2, target_frames, axis=0))
    im1 = axes[1, 0].imshow(diff_original.T, aspect='auto', cmap='hot', origin='lower')
    axes[1, 0].set_title(f'|{label1} - {label2}| difference map')
    axes[1, 0].set_xlabel('Frame')
    axes[1, 0].set_ylabel('Channel')
    plt.colorbar(im1, ax=axes[1, 0])
    
    # Frame-wise difference (what flatten preserves)
    frame_diff = diff_original.mean(axis=1)
    axes[1, 1].bar(range(target_frames), frame_diff)
    axes[1, 1].set_title('Per-frame difference\n(PRESERVED by flatten)')
    axes[1, 1].set_xlabel('Frame')
    axes[1, 1].set_ylabel('Mean |difference|')
    
    # Channel-wise difference (what pooling preserves)
    channel_diff = diff_original.mean(axis=0)
    axes[1, 2].bar(range(n_channels), channel_diff)
    axes[1, 2].set_title('Per-channel difference\n(Partially preserved by pooling)')
    axes[1, 2].set_xlabel('Channel')
    axes[1, 2].set_ylabel('Mean |difference|')
    
    plt.tight_layout()
    plt.show()
    
    # Key insight
    print("\n" + "="*60)
    print("KEY INSIGHT")
    print("="*60)
    print(f"\nFrame with biggest difference: Frame {np.argmax(frame_diff)}")
    print(f"  - Flatten PRESERVES this: classifier can learn 'frame {np.argmax(frame_diff)} differs'")
    print(f"  - Pooling LOSES this: only knows overall statistics")
    print(f"\nIf phonemes differ mainly in WHEN activation happens (not overall level),")
    print(f"pooling will fail to capture the difference.")


def run_diagnostic(pipeline, phoneme1='a', phoneme2='p', target_frames=10):
    """Run the full diagnostic on your pipeline data.
    
    Args:
        pipeline: Your Dutch30Pipeline with loaded train data.
        phoneme1: First phoneme to analyze.
        phoneme2: Second phoneme to compare.
        target_frames: Target frames for resampling comparison.
    """
    print("="*60)
    print("FEATURE INFORMATION DIAGNOSTIC")
    print("="*60)
    
    # Get example features
    feats1 = extract_example_features(pipeline, phoneme1, n_examples=3)
    feats2 = extract_example_features(pipeline, phoneme2, n_examples=3)
    
    if not feats1:
        print(f"No examples found for phoneme '{phoneme1}'")
        return
    if not feats2:
        print(f"No examples found for phoneme '{phoneme2}'")
        return
    
    print(f"\nFound {len(feats1)} examples of '{phoneme1}', {len(feats2)} examples of '{phoneme2}'")
    print(f"Feature shapes: {[f.shape for f in feats1[:3]]}")
    
    # Single feature visualization
    print("\n--- Single feature analysis ---")
    visualize_information_loss(feats1[0], target_frames)
    
    # Two phoneme comparison
    print("\n--- Two phoneme comparison ---")
    compare_two_phonemes(feats1[0], feats2[0], phoneme1, phoneme2, target_frames)

run_diagnostic(pipeline, phoneme1='n', phoneme2='t', target_frames=10)

# Store raw labels count before grouping
raw_phoneme_count = len(set(pipeline.train['phoneme_labels']))

# Apply grouping
pipeline.step7_filter_unknowns(unknown_keep_ratio=0);
pipeline.step8_group_phonemes()

grouped_phoneme_count = len(set(pipeline.train['phoneme_labels']))
print(f"\nReduced from {raw_phoneme_count} phonemes to {grouped_phoneme_count} groups")
grouped_results = train_and_evaluate(pipeline, use_groups=False)  # use_groups=False because labels are already grouped

raw_phonemes = set(pipeline.train['phoneme_labels_raw'])
print(f"Raw phonemes in data ({len(raw_phonemes)}):")
print(sorted(raw_phonemes))

# Get the mapping
phoneme_to_group = pipeline.detector.phonetic_dict.phoneme_to_group
print(f"\nPhonemes in mapping ({len(phoneme_to_group)}):")
print(sorted(phoneme_to_group.keys()))

# Find which ones are NOT in the mapping
not_mapped = [p for p in raw_phonemes if p not in phoneme_to_group]
print(f"\nNOT mapped ({len(not_mapped)}):")
print(sorted(not_mapped))

# Find which ones ARE mapped
mapped = [p for p in raw_phonemes if p in phoneme_to_group]
print(f"\nMapped ({len(mapped)}):")
for p in sorted(mapped):
    print(f"  '{p}' -> '{phoneme_to_group[p]}'")

for pid in sorted(grouped_results.keys()):
    visualize_patient_model(pid, grouped_results, pipeline)

import warnings
warnings.filterwarnings('ignore', message='.*number of unique classes.*')

def diagnose_feature_quality(pipeline, n_phonemes_to_check=10):
    """
    Check if features show ANY systematic differences between phonemes.
    """
    import numpy as np
    from collections import defaultdict, Counter
    from scipy.stats import ttest_ind
    import matplotlib.pyplot as plt
    
    phoneme_features = defaultdict(list)
    
    for feat, label in zip(pipeline.train['features'], pipeline.train['phoneme_labels']):
        if label == '?' or label == 'unknown':
            continue
        if feat.ndim > 1:
            feat_flat = np.mean(feat, axis=0)
        else:
            feat_flat = feat
        if not np.any(np.isnan(feat_flat)) and not np.any(np.isinf(feat_flat)):
            phoneme_features[label].append(feat_flat)
    
    # Filter to consistent shapes per phoneme
    phoneme_features_clean = {}
    
    for phoneme, features in phoneme_features.items():
        shape_counts = Counter(f.shape[0] for f in features)
        if not shape_counts:
            continue
        most_common_len = max(shape_counts, key=shape_counts.get)
        filtered = [f for f in features if f.shape[0] == most_common_len]
        
        if len(filtered) >= 10:
            phoneme_features_clean[phoneme] = np.array(filtered)
    
    print(f"Phonemes with consistent features: {len(phoneme_features_clean)}")
    
    # Get phonemes with most samples
    sorted_phonemes = sorted(phoneme_features_clean.keys(), key=lambda x: -len(phoneme_features_clean[x]))
    top_phonemes = sorted_phonemes[:n_phonemes_to_check]
    
    print(f"Top {len(top_phonemes)} phonemes by sample count:")
    for p in top_phonemes:
        print(f"  {p}: {len(phoneme_features_clean[p])} samples, {phoneme_features_clean[p].shape[1]} features")
    
    print("\nChecking feature separability between top phonemes:")
    print("=" * 70)
    
    pair_results = []
    
    for i, p1 in enumerate(top_phonemes):
        for p2 in top_phonemes[i+1:]:
            features_p1 = phoneme_features_clean[p1]
            features_p2 = phoneme_features_clean[p2]
            
            # Ensure same feature dimension
            min_dim = min(features_p1.shape[1], features_p2.shape[1])
            features_p1 = features_p1[:, :min_dim]
            features_p2 = features_p2[:, :min_dim]
            
            # Count significant features
            n_significant = 0
            for feat_idx in range(min_dim):
                stat, pval = ttest_ind(features_p1[:, feat_idx], features_p2[:, feat_idx])
                if pval < 0.01:
                    n_significant += 1
            
            pct_significant = n_significant / min_dim * 100
            pair_results.append((p1, p2, pct_significant, len(phoneme_features_clean[p1]), len(phoneme_features_clean[p2])))
    
    # Sort by percentage of significant features
    pair_results.sort(key=lambda x: -x[2])
    
    print(f"\n{'Phoneme 1':<10} {'Phoneme 2':<10} {'% Sig. Features':<15} {'N1':<6} {'N2':<6}")
    print("-" * 70)
    
    for p1, p2, pct, n1, n2 in pair_results[:20]:
        print(f"{p1:<10} {p2:<10} {pct:<15.1f} {n1:<6} {n2:<6}")
    
    # Summary
    avg_pct = np.mean([x[2] for x in pair_results])
    print(f"\nAverage % significant features across all pairs: {avg_pct:.1f}%")
    
    if avg_pct < 5:
        print("\nDIAGNOSIS: Very few features differ between phonemes.")
        print("  -> Features may not capture phoneme-relevant information")
        print("  -> Consider: different frequency bands, different time windows, electrode selection")
    elif avg_pct < 15:
        print("\nDIAGNOSIS: Weak but present differences between phonemes.")
        print("  -> Some signal exists but may need feature engineering")
    else:
        print("\nDIAGNOSIS: Moderate feature differences exist.")
        print("  -> Classification should be possible with right approach")
    
    # Visualize the best-separable pair
    if pair_results:
        best_pair = pair_results[0]
        p1, p2 = best_pair[0], best_pair[1]
        
        features_p1 = phoneme_features_clean[p1]
        features_p2 = phoneme_features_clean[p2]
        min_dim = min(features_p1.shape[1], features_p2.shape[1])
        features_p1 = features_p1[:, :min_dim]
        features_p2 = features_p2[:, :min_dim]
        
        # Find the two most discriminative features
        t_stats = []
        for feat_idx in range(min_dim):
            stat, pval = ttest_ind(features_p1[:, feat_idx], features_p2[:, feat_idx])
            t_stats.append(abs(stat))
        
        best_feat_indices = np.argsort(t_stats)[-2:]
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.scatter(
            features_p1[:, best_feat_indices[0]], 
            features_p1[:, best_feat_indices[1]], 
            alpha=0.5, label=f"'{p1}' (n={len(features_p1)})"
        )
        ax.scatter(
            features_p2[:, best_feat_indices[0]], 
            features_p2[:, best_feat_indices[1]], 
            alpha=0.5, label=f"'{p2}' (n={len(features_p2)})"
        )
        
        ax.set_xlabel(f'Feature {best_feat_indices[0]}')
        ax.set_ylabel(f'Feature {best_feat_indices[1]}')
        ax.set_title(f'Best Separable Pair: {p1} vs {p2}\n({best_pair[2]:.1f}% features significantly different)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    return pair_results

pair_results = diagnose_feature_quality(pipeline, n_phonemes_to_check=15)

def train_and_evaluate_extended(pipeline, use_groups=False, method='markov'):
    """
    Train per patient with multiple approaches.
    
    Args:
        pipeline: Dutch30Pipeline with loaded data
        use_groups: Whether to use phoneme groups
        method: One of 'markov', 'gmm', 'soft_labels', 'gmm_informed'
    """
    import numpy as np
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import accuracy_score
    from scipy.special import softmax
    from markov_phoneme_model import MarkovPhonemeModel
    
    results = {}
    
    for pid in sorted(set(pipeline.train['phoneme_participant_ids'])):
        train_mask = [p == pid for p in pipeline.train['phoneme_participant_ids']]
        test_mask = [p == pid for p in pipeline.test['phoneme_participant_ids']]
        
        train_feat = [pipeline.train['features'][i] for i, m in enumerate(train_mask) if m]
        train_labels = [pipeline.train['phoneme_labels'][i] for i, m in enumerate(train_mask) if m]
        test_feat = [pipeline.test['features'][i] for i, m in enumerate(test_mask) if m]
        test_labels = [pipeline.test['phoneme_labels'][i] for i, m in enumerate(test_mask) if m]
        
        if len(train_feat) < 10 or len(test_feat) < 5:
            continue
        
        X_train = []
        for feat in train_feat:
            if feat.ndim > 1:
                X_train.append(np.mean(feat, axis=0))
            else:
                X_train.append(feat)
        X_train = np.array(X_train)
        
        X_test = []
        for feat in test_feat:
            if feat.ndim > 1:
                X_test.append(np.mean(feat, axis=0))
            else:
                X_test.append(feat)
        X_test = np.array(X_test)
        
        valid_train = ~(np.isnan(X_train).any(axis=1) | np.isinf(X_train).any(axis=1))
        valid_test = ~(np.isnan(X_test).any(axis=1) | np.isinf(X_test).any(axis=1))
        
        X_train = X_train[valid_train]
        y_train = [train_labels[i] for i in range(len(train_labels)) if valid_train[i]]
        X_test = X_test[valid_test]
        y_test = [test_labels[i] for i in range(len(test_labels)) if valid_test[i]]
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        le = LabelEncoder()
        y_train_encoded = le.fit_transform(y_train)
        n_classes = len(le.classes_)
        
        if method == 'markov':
            model = MarkovPhonemeModel(
                phonetic_dict=pipeline.detector.phonetic_dict,
                order=1,
                use_groups=use_groups
            )
            train_feat_valid = [train_feat[i] for i in range(len(train_feat)) if valid_train[i]]
            model.train(features=train_feat_valid, phoneme_labels=y_train)
            
            test_feat_valid = [test_feat[i] for i in range(len(test_feat)) if valid_test[i]]
            preds, _ = model.predict(test_feat_valid, use_viterbi=True)
            accuracy = sum(1 for p, t in zip(preds, y_test) if p == t) / len(y_test)
            
        elif method == 'gmm':
            gmm_per_class = {}
            for class_idx in range(n_classes):
                class_mask = y_train_encoded == class_idx
                if np.sum(class_mask) < 2:
                    continue
                
                X_class = X_train_scaled[class_mask]
                n_components = min(3, len(X_class) // 2)
                if n_components < 1:
                    n_components = 1
                
                gmm = GaussianMixture(
                    n_components=n_components,
                    covariance_type='diag',
                    max_iter=100,
                    random_state=42
                )
                gmm.fit(X_class)
                gmm_per_class[class_idx] = gmm
            
            preds_encoded = []
            for x in X_test_scaled:
                scores = []
                for class_idx in range(n_classes):
                    if class_idx in gmm_per_class:
                        score = gmm_per_class[class_idx].score_samples(x.reshape(1, -1))[0]
                    else:
                        score = -np.inf
                    scores.append(score)
                preds_encoded.append(np.argmax(scores))
            
            preds = le.inverse_transform(preds_encoded)
            accuracy = accuracy_score(y_test, preds)
            
        elif method == 'soft_labels':
            from sklearn.ensemble import RandomForestClassifier
            
            soft_targets = np.zeros((len(y_train_encoded), n_classes))
            smoothing = 0.1
            
            for i, label in enumerate(y_train_encoded):
                soft_targets[i, :] = smoothing / n_classes
                soft_targets[i, label] = 1.0 - smoothing + smoothing / n_classes
            
            rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
            rf.fit(X_train_scaled, y_train_encoded)
            
            preds_encoded = rf.predict(X_test_scaled)
            preds = le.inverse_transform(preds_encoded)
            accuracy = accuracy_score(y_test, preds)
            
        elif method == 'gmm_informed':
            n_components_total = min(n_classes * 2, 20)
            
            gmm_unsupervised = GaussianMixture(
                n_components=n_components_total,
                covariance_type='diag',
                max_iter=100,
                random_state=42
            )
            gmm_unsupervised.fit(X_train_scaled)
            
            cluster_probs = gmm_unsupervised.predict_proba(X_train_scaled)
            
            X_augmented_train = np.hstack([X_train_scaled, cluster_probs])
            X_augmented_test = np.hstack([
                X_test_scaled, 
                gmm_unsupervised.predict_proba(X_test_scaled)
            ])
            
            from sklearn.ensemble import RandomForestClassifier
            rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
            rf.fit(X_augmented_train, y_train_encoded)
            
            preds_encoded = rf.predict(X_augmented_test)
            preds = le.inverse_transform(preds_encoded)
            accuracy = accuracy_score(y_test, preds)
        
        results[pid] = {
            'accuracy': accuracy,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'n_classes': n_classes,
            'predictions': preds,
            'true_labels': y_test
        }
        
        print(f"  {pid}: Acc={accuracy:.3f} (method={method}, classes={n_classes}, train={len(X_train)})")
    
    accs = [r['accuracy'] for r in results.values()]
    print(f"\n  Mean: {np.mean(accs):.3f} +/- {np.std(accs):.3f}")
    
    return results


def compare_methods(pipeline, use_groups=False):
    """Compare all methods side by side."""
    import pandas as pd
    
    methods = ['markov', 'gmm', 'soft_labels', 'gmm_informed']
    all_results = {}
    
    for method in methods:
        print(f"\n{'='*60}")
        print(f"METHOD: {method}")
        print('='*60)
        all_results[method] = train_and_evaluate_extended(pipeline, use_groups, method)
    
    comparison = {}
    for method, results in all_results.items():
        for pid, res in results.items():
            if pid not in comparison:
                comparison[pid] = {}
            comparison[pid][method] = res['accuracy']
    
    df = pd.DataFrame(comparison).T
    df['Max'] = df.max(axis=1)
    df['Best'] = df.idxmax(axis=1)
    
    print("\n" + "="*80)
    print("COMPARISON TABLE")
    print("="*80)
    print(df.to_string())
    
    print("\n" + "="*80)
    print("MEAN ACCURACY BY METHOD")
    print("="*80)
    for method in methods:
        accs = [comparison[pid].get(method, np.nan) for pid in comparison]
        print(f"  {method:15s}: {np.nanmean(accs):.3f} +/- {np.nanstd(accs):.3f}")
    
    return all_results, df

# all_results, comparison_df = compare_methods(pipeline, use_groups=True)
