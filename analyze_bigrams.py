# Converted from analyze_bigrams.ipynb

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
    pipeline.dutch30_step6_resolve_unknowns()
    #pipeline.checkpoint_after_step6(sample_fraction=sample_fraction)
    pipeline.step7_filter_unknowns(unknown_keep_ratio=0.0025);
    

    print(f"  Train samples: {len(pipeline.train.get('features', []))}")
    print(f"  Test samples: {len(pipeline.test.get('features', []))}")   

"""Bigram analysis for Dutch30 phoneme pipeline.

Analyzes how many distinct bigrams exist in the phonetic dictionary
and how many samples per bigram are available per patient from the
accumulated pipeline data.

Usage:
    Run after pipeline step 6 (resolve unknowns) with train/test data available.

    Example:
        results = analyze_bigrams(pipeline)
"""

from collections import Counter, defaultdict
from phonetic_dictionary import PhoneticDictionary
from dataset_config import Dutch30Config


def get_dictionary_bigrams(phonetic_dict):
    """Extract all bigrams attested in the phonetic dictionary.

    Iterates over every word in the dictionary, extracts its phoneme
    sequence, and collects all adjacent phoneme pairs.

    Args:
        phonetic_dict: PhoneticDictionary instance with loaded dictionary.

    Returns:
        dict with keys:
            bigram_counts: Counter of (phoneme_a, phoneme_b) tuples.
            total_bigrams: int, total bigram tokens.
            unique_bigrams: int, distinct bigram types.
            words_processed: int, number of dictionary entries processed.
    """
    bigram_counts = Counter()
    words_processed = 0

    for word, transcription in phonetic_dict.dictionary.items():
        # Skip sentence-level entries
        if " " in word:
            continue

        phonemes = phonetic_dict.extract_phonemes(word)
        if phonemes is None or len(phonemes) < 2:
            continue

        words_processed += 1

        for i in range(len(phonemes) - 1):
            bigram = (phonemes[i], phonemes[i + 1])
            bigram_counts[bigram] += 1

    return {
        "bigram_counts": bigram_counts,
        "total_bigrams": sum(bigram_counts.values()),
        "unique_bigrams": len(bigram_counts),
        "words_processed": words_processed,
    }


def get_patient_bigrams(data, config=None):
    """Extract bigram counts from accumulated pipeline data, per patient.

    Pairs consecutive phonemes that belong to the same word and same
    patient. Does NOT pair across word boundaries.

    Args:
        data: dict from pipeline (train or test), must contain:
            phoneme_labels, phoneme_words, phoneme_participant_ids,
            phoneme_positions.
        config: Dutch30Config instance (unused currently, reserved).

    Returns:
        dict with keys:
            per_patient: dict mapping patient_id to Counter of bigram tuples.
            all_patients: Counter of bigram tuples pooled across patients.
            per_patient_totals: dict mapping patient_id to total bigram count.
    """
    labels = data["phoneme_labels"]
    words = data["phoneme_words"]
    pids = data["phoneme_participant_ids"]
    positions = data["phoneme_positions"]

    per_patient = defaultdict(Counter)
    all_patients = Counter()

    for i in range(len(labels) - 1):
        # Only pair phonemes within the same word AND same patient
        if words[i] != words[i + 1]:
            continue
        if pids[i] != pids[i + 1]:
            continue

        # Verify sequential position within word
        if positions[i + 1] != positions[i] + 1:
            continue

        # Skip unknown phonemes
        if labels[i] == "?" or labels[i + 1] == "?":
            continue

        bigram = (labels[i], labels[i + 1])
        per_patient[pids[i]][bigram] += 1
        all_patients[bigram] += 1

    per_patient_totals = {
        pid: sum(counts.values()) for pid, counts in per_patient.items()
    }

    return {
        "per_patient": dict(per_patient),
        "all_patients": all_patients,
        "per_patient_totals": per_patient_totals,
    }


def print_bigram_report(dict_results, patient_results, min_samples_threshold=10):
    """Print a summary report of bigram distribution.

    Args:
        dict_results: output from get_dictionary_bigrams().
        patient_results: output from get_patient_bigrams().
        min_samples_threshold: int, minimum samples per bigram to consider
            it trainable for a per-patient classifier.
    """
    print("=" * 70)
    print("BIGRAM ANALYSIS REPORT")
    print("=" * 70)

    # Dictionary-level stats
    print("\n--- Phonetic Dictionary ---")
    print(f"Words processed: {dict_results['words_processed']}")
    print(f"Unique bigrams in dictionary: {dict_results['unique_bigrams']}")
    print(f"Total bigram tokens: {dict_results['total_bigrams']}")

    dc = dict_results["bigram_counts"]
    print(f"\nTop 20 most common bigrams (dictionary):")
    for bigram, count in dc.most_common(20):
        print(f"  {bigram[0]}-{bigram[1]}: {count}")

    # Pooled patient stats
    print("\n--- Pooled Across All Patients (from neural data) ---")
    ac = patient_results["all_patients"]
    print(f"Unique bigrams observed: {len(ac)}")
    print(f"Total bigram tokens: {sum(ac.values())}")

    print(f"\nTop 30 most common bigrams (pooled neural data):")
    for bigram, count in ac.most_common(30):
        print(f"  {bigram[0]}-{bigram[1]}: {count}")

    # Distribution analysis
    counts_list = sorted(ac.values(), reverse=True)
    thresholds = [1, 2, 5, 10, 15, 20, 30, 50]
    print(f"\nBigram frequency distribution (pooled):")
    for t in thresholds:
        n_above = sum(1 for c in counts_list if c >= t)
        total_samples = sum(c for c in counts_list if c >= t)
        print(f"  >= {t:3d} samples: {n_above:4d} bigrams "
              f"({total_samples} total samples)")

    # Per-patient breakdown
    print("\n--- Per-Patient Breakdown ---")
    print(f"{'Patient':<10} {'Total':>8} {'Unique':>8} "
          f"{'>=5':>6} {'>=10':>6} {'>=15':>6} {'>=20':>6}")
    print("-" * 64)

    for pid in sorted(patient_results["per_patient"].keys()):
        pc = patient_results["per_patient"][pid]
        total = patient_results["per_patient_totals"][pid]
        unique = len(pc)
        ge5 = sum(1 for c in pc.values() if c >= 5)
        ge10 = sum(1 for c in pc.values() if c >= 10)
        ge15 = sum(1 for c in pc.values() if c >= 15)
        ge20 = sum(1 for c in pc.values() if c >= 20)
        print(f"{pid:<10} {total:>8} {unique:>8} "
              f"{ge5:>6} {ge10:>6} {ge15:>6} {ge20:>6}")

    # Feasibility assessment
    print("\n--- Feasibility for Per-Patient Bigram Classification ---")
    print(f"(Threshold: >= {min_samples_threshold} samples per bigram)")
    for pid in sorted(patient_results["per_patient"].keys()):
        pc = patient_results["per_patient"][pid]
        trainable = {bg: c for bg, c in pc.items()
                     if c >= min_samples_threshold}
        trainable_samples = sum(trainable.values())
        total = patient_results["per_patient_totals"][pid]
        coverage = trainable_samples / total * 100 if total > 0 else 0
        print(f"  {pid}: {len(trainable)} trainable bigrams, "
              f"{trainable_samples}/{total} samples "
              f"({coverage:.1f}% coverage)")


def analyze_bigrams(pipeline, min_samples_threshold=10):
    """Run the full bigram analysis on a pipeline after step 6.

    Args:
        pipeline: Dutch30Pipeline instance with train and test data loaded.
        min_samples_threshold: int, minimum samples per bigram for
            feasibility reporting.

    Returns:
        dict with keys: dict_results, train_results, test_results.
    """
    # Dictionary analysis
    dict_results = get_dictionary_bigrams(pipeline.phonetic_dict)

    # Train data analysis
    train_results = get_patient_bigrams(pipeline.train)

    # Test data analysis
    test_results = get_patient_bigrams(pipeline.test)

    print("\n### TRAIN SET ###")
    print_bigram_report(dict_results, train_results, min_samples_threshold)

    print("\n\n### TEST SET ###")
    print(f"Unique bigrams in test: {len(test_results['all_patients'])}")
    print(f"Total bigram tokens in test: "
          f"{sum(test_results['all_patients'].values())}")

    # Overlap analysis
    train_bigrams = set(train_results["all_patients"].keys())
    test_bigrams = set(test_results["all_patients"].keys())
    overlap = train_bigrams & test_bigrams
    test_only = test_bigrams - train_bigrams
    print(f"\nTrain-test overlap: {len(overlap)} bigrams in both sets")
    print(f"Test-only bigrams (unseen in train): {len(test_only)}")
    if test_only:
        print(f"  Examples: {list(test_only)[:10]}")

    return {
        "dict_results": dict_results,
        "train_results": train_results,
        "test_results": test_results,
    }

#from analyze_bigrams import analyze_bigrams
results = analyze_bigrams(pipeline)

"""Transition-type bigram analysis with vowels collapsed to wildcard.

Groups bigrams by collapsing all vowels to '*' while keeping consonants
as specific phonemes. This creates transition classes like '*-n', 'd-*',
's-t', '*-*' etc.

Usage:
    Run after pipeline step 6 with train/test data available.

    Example:
        from analyze_transition_types import analyze_transitions
        results = analyze_transitions(pipeline)
"""

from collections import Counter, defaultdict
from phonetic_dictionary import PhoneticDictionary
from dataset_config import Dutch30Config


# Vowels from PhoneticDictionary.phoneme_groups
VOWELS = {
    # back_vowels
    "u", "o", "ɔ", "a", "ɑ", "ɑu", "œy", "ə", "oː", "aː", "ɔː", "ɑː",
    # front_vowels
    "i", "ɪ", "e", "ɛ", "ɛi", "y", "ʏ", "eː", "iː", "ɪː", "yː", "øː",
}


def collapse_phoneme(phoneme):
    """Map a phoneme to its transition symbol.

    Vowels become '*', consonants stay as-is.

    Args:
        phoneme: str, IPA phoneme symbol.

    Returns:
        str, either '*' for vowels or the original phoneme for consonants.
    """
    if phoneme in VOWELS:
        return "*"
    return phoneme


def get_patient_transitions(data):
    """Extract transition-type counts from accumulated pipeline data, per patient.

    Pairs consecutive phonemes within the same word and patient,
    then collapses vowels to '*'.

    Args:
        data: dict from pipeline (train or test), must contain:
            phoneme_labels, phoneme_words, phoneme_participant_ids,
            phoneme_positions.

    Returns:
        dict with keys:
            per_patient: dict mapping patient_id to Counter of transition tuples.
            all_patients: Counter of transition tuples pooled across patients.
            per_patient_totals: dict mapping patient_id to total count.
            raw_to_transition: dict mapping (phoneme_a, phoneme_b) to
                (collapsed_a, collapsed_b) for reference.
    """
    labels = data["phoneme_labels"]
    words = data["phoneme_words"]
    pids = data["phoneme_participant_ids"]
    positions = data["phoneme_positions"]

    per_patient = defaultdict(Counter)
    all_patients = Counter()
    raw_to_transition = {}

    for i in range(len(labels) - 1):
        if words[i] != words[i + 1]:
            continue
        if pids[i] != pids[i + 1]:
            continue
        if positions[i + 1] != positions[i] + 1:
            continue
        if labels[i] == "?" or labels[i + 1] == "?":
            continue

        raw_bigram = (labels[i], labels[i + 1])
        transition = collapse_pair(labels[i], labels[i + 1])

        raw_to_transition[raw_bigram] = transition
        per_patient[pids[i]][transition] += 1
        all_patients[transition] += 1

    per_patient_totals = {
        pid: sum(counts.values()) for pid, counts in per_patient.items()
    }

    return {
        "per_patient": dict(per_patient),
        "all_patients": all_patients,
        "per_patient_totals": per_patient_totals,
        "raw_to_transition": raw_to_transition,
    }


def print_transition_report(results, min_samples_threshold=10):
    """Print summary report of transition-type distribution.

    Args:
        results: output from get_patient_transitions().
        min_samples_threshold: int, minimum samples for trainability.
    """
    ac = results["all_patients"]

    print("=" * 70)
    print("TRANSITION-TYPE BIGRAM ANALYSIS (vowels collapsed to *)")
    print("=" * 70)

    print(f"\nUnique transition types: {len(ac)}")
    print(f"Total transition tokens: {sum(ac.values())}")

    # Categorize transitions
    v_v = {t: c for t, c in ac.items() if t[0] in VOWELS and t[1] in VOWELS}
    v_c = {t: c for t, c in ac.items() if t[0] == "*" and t[1] not in VOWELS}
    c_v = {t: c for t, c in ac.items() if t[0] not in VOWELS and t[1] == "*"}
    c_c = {t: c for t, c in ac.items() if t[0] not in VOWELS and t[1] not in VOWELS
            and t[0] != "*" and t[1] != "*"}

    print(f"\nBy transition category:")
    print(f"  vowel-vowel (*-*):       {len(v_v):3d} types, "
          f"{sum(v_v.values()):5d} tokens")
    print(f"  vowel-consonant (*-C):   {len(v_c):3d} types, "
          f"{sum(v_c.values()):5d} tokens")
    print(f"  consonant-vowel (C-*):   {len(c_v):3d} types, "
          f"{sum(c_v.values()):5d} tokens")
    print(f"  consonant-consonant (C-C): {len(c_c):3d} types, "
          f"{sum(c_c.values()):5d} tokens")

    # All transitions ranked
    print(f"\nAll transition types ranked by frequency (pooled):")
    for rank, (trans, count) in enumerate(ac.most_common(), 1):
        print(f"  {rank:3d}. {trans[0]}-{trans[1]}: {count}")

    # Show which raw bigrams feed into each transition
    print(f"\nTop 15 transitions with contributing raw bigrams:")
    raw_map = results["raw_to_transition"]
    reverse_map = defaultdict(list)
    for raw, trans in raw_map.items():
        reverse_map[trans].append(raw)

    for trans, count in ac.most_common(15):
        raw_bigrams = reverse_map[trans]
        raw_examples = [f"{r[0]}-{r[1]}" for r in raw_bigrams[:8]]
        print(f"  {trans[0]}-{trans[1]} ({count}): "
              f"{', '.join(raw_examples)}"
              f"{'...' if len(raw_bigrams) > 8 else ''}")

    # Distribution thresholds
    counts_list = sorted(ac.values(), reverse=True)
    thresholds = [1, 5, 10, 15, 20, 30, 50, 100]
    print(f"\nTransition frequency distribution (pooled):")
    for t in thresholds:
        n_above = sum(1 for c in counts_list if c >= t)
        total_samples = sum(c for c in counts_list if c >= t)
        print(f"  >= {t:3d} samples: {n_above:4d} transition types "
              f"({total_samples} total samples)")

    # Per-patient breakdown
    print(f"\n{'Patient':<10} {'Total':>8} {'Types':>8} "
          f"{'>=5':>6} {'>=10':>6} {'>=15':>6} {'>=20':>6} {'>=30':>6}")
    print("-" * 72)

    for pid in sorted(results["per_patient"].keys()):
        pc = results["per_patient"][pid]
        total = results["per_patient_totals"][pid]
        unique = len(pc)
        ge5 = sum(1 for c in pc.values() if c >= 5)
        ge10 = sum(1 for c in pc.values() if c >= 10)
        ge15 = sum(1 for c in pc.values() if c >= 15)
        ge20 = sum(1 for c in pc.values() if c >= 20)
        ge30 = sum(1 for c in pc.values() if c >= 30)
        print(f"{pid:<10} {total:>8} {unique:>8} "
              f"{ge5:>6} {ge10:>6} {ge15:>6} {ge20:>6} {ge30:>6}")

    # Feasibility per patient
    print(f"\n--- Feasibility (threshold >= {min_samples_threshold}) ---")
    for pid in sorted(results["per_patient"].keys()):
        pc = results["per_patient"][pid]
        trainable = {t: c for t, c in pc.items()
                     if c >= min_samples_threshold}
        trainable_samples = sum(trainable.values())
        total = results["per_patient_totals"][pid]
        coverage = trainable_samples / total * 100 if total > 0 else 0
        classes_str = ", ".join(
            f"{t[0]}-{t[1]}({c})"
            for t, c in sorted(trainable.items(), key=lambda x: -x[1])
        )
        print(f"  {pid}: {len(trainable)} types, "
              f"{trainable_samples}/{total} samples "
              f"({coverage:.1f}% coverage)")
        if classes_str:
            print(f"       [{classes_str}]")


def analyze_transitions(pipeline, min_samples_threshold=10):
    """Run full transition-type analysis on a pipeline after step 6.

    Args:
        pipeline: Dutch30Pipeline instance with train and test data loaded.
        min_samples_threshold: int, minimum samples for feasibility.

    Returns:
        dict with keys: train_results, test_results.
    """
    train_results = get_patient_transitions(pipeline.train)
    test_results = get_patient_transitions(pipeline.test)

    print("\n### TRAIN SET ###")
    print_transition_report(train_results, min_samples_threshold)

    print("\n\n### TEST SET ###")
    test_ac = test_results["all_patients"]
    print(f"Unique transitions in test: {len(test_ac)}")
    print(f"Total transition tokens in test: {sum(test_ac.values())}")

    train_types = set(train_results["all_patients"].keys())
    test_types = set(test_results["all_patients"].keys())
    overlap = train_types & test_types
    test_only = test_types - train_types
    print(f"Train-test overlap: {len(overlap)} types in both")
    print(f"Test-only types (unseen in train): {len(test_only)}")
    if test_only:
        print(f"  {list(test_only)}")

    return {
        "train_results": train_results,
        "test_results": test_results,
    }
def collapse_pair(phoneme_a, phoneme_b):
    a_is_vowel = phoneme_a in VOWELS
    b_is_vowel = phoneme_b in VOWELS

    if a_is_vowel and b_is_vowel:
        return (phoneme_a, phoneme_b)
    if a_is_vowel:
        return ("*", phoneme_b)
    if b_is_vowel:
        return (phoneme_a, "*")
    return (phoneme_a, phoneme_b)

results = analyze_transitions(pipeline)

"""Check sentence counts per patient for train/test split feasibility.

Loads raw stimuli for each patient and counts unique sentences,
their repetition counts, and total word tokens per sentence.

Usage:
    from check_sentences import check_sentence_counts
    check_sentence_counts(pipeline.dutch30_extractor)
"""

import re
import numpy as np
from collections import Counter, defaultdict


# Patient groups from Dutch30 dataset
WORD_PATIENTS = ["P11", "P12", "P13", "P14", "P15", "P16", "P17", "P20"]
SENTENCE_PATIENTS = ["P21", "P22", "P23", "P24", "P25", "P26", "P27",
                     "P28", "P29", "P30"]
MIXED_PATIENTS = ["P01", "P02", "P03", "P04", "P06", "P07", "P08",
                  "P09", "P10"]


def check_sentence_counts(dutch30_extractor, patient_ids=None):
    """Check unique sentence counts and repetitions per patient.

    Args:
        dutch30_extractor: Dutch30FeatureExtractor instance.
        patient_ids: list of patient IDs to check. Defaults to
            sentence patients only.
    """
    if patient_ids is None:
        patient_ids = SENTENCE_PATIENTS

    print("=" * 70)
    print("SENTENCE COUNT ANALYSIS")
    print("=" * 70)

    for pid in patient_ids:
        try:
            raw_data = dutch30_extractor.load_patient_raw_data(pid)
        except Exception as e:
            print(f"\n{pid}: Failed to load - {e}")
            continue

        stimuli = raw_data["stimuli"]

        # Decode stimuli
        decoded = []
        for s in stimuli:
            text = s.decode() if isinstance(s, bytes) else str(s)
            decoded.append(text.strip())

        # Find unique non-empty stimuli
        unique_stimuli = set(s for s in decoded if s)

        # Categorize into sentences (multi-word) and single words
        sentences = []
        single_words = []
        for s in unique_stimuli:
            cleaned = re.sub(r'["""\']+', '', s).strip()
            if not cleaned:
                continue
            word_count = len(cleaned.split())
            if word_count > 1:
                sentences.append(cleaned)
            else:
                single_words.append(cleaned)

        # Count repetitions of each sentence in the stimulus stream
        # A sentence "repetition" = contiguous block of identical stimuli
        sentence_repetitions = Counter()
        current_stim = None
        for s in decoded:
            if s != current_stim:
                if s and len(s.split()) > 1:
                    cleaned = re.sub(r'["""\']+', '', s).strip()
                    sentence_repetitions[cleaned] += 1
                current_stim = s

        # Count words per sentence
        words_per_sentence = {
            s: len(s.split()) for s in sentences
        }

        # Report
        print(f"\n{'=' * 50}")
        print(f"{pid}")
        print(f"{'=' * 50}")
        print(f"  Unique sentences: {len(sentences)}")
        print(f"  Unique single words: {len(single_words)}")
        print(f"  Total unique stimuli: {len(unique_stimuli)}")

        if sentences:
            total_reps = sum(sentence_repetitions.values())
            print(f"\n  Total sentence presentations: {total_reps}")
            print(f"  Mean repetitions per sentence: "
                  f"{total_reps / len(sentences):.1f}")

            # Distribution of repetition counts
            rep_counts = list(sentence_repetitions.values())
            rep_dist = Counter(rep_counts)
            print(f"\n  Repetition distribution:")
            for n_reps in sorted(rep_dist.keys()):
                n_sentences = rep_dist[n_reps]
                print(f"    {n_reps}x repeated: {n_sentences} sentences")

            # Word count distribution
            wc_values = list(words_per_sentence.values())
            print(f"\n  Words per sentence: "
                  f"min={min(wc_values)}, max={max(wc_values)}, "
                  f"mean={np.mean(wc_values):.1f}")

            # Show sentences sorted by repetition count
            print(f"\n  Sentences (by repetitions):")
            for sent, reps in sorted(sentence_repetitions.items(),
                                      key=lambda x: -x[1])[:15]:
                n_words = len(sent.split())
                display = sent[:60] + "..." if len(sent) > 60 else sent
                print(f"    {reps:3d}x ({n_words:2d} words): {display}")
            if len(sentence_repetitions) > 15:
                print(f"    ... and {len(sentence_repetitions) - 15} more")

        # Feasibility for 70/30 split
        if sentences:
            n_train = int(len(sentences) * 0.7)
            n_test = len(sentences) - n_train
            print(f"\n  70/30 sentence-level split: "
                  f"{n_train} train / {n_test} test sentences")

            # Estimate transition counts from split
            train_transitions_est = sum(
                (words_per_sentence[s] - 1 + 1) * sentence_repetitions[s]
                for s in sorted(sentence_repetitions.keys())[:n_train]
            )
            test_transitions_est = sum(
                (words_per_sentence[s] - 1 + 1) * sentence_repetitions[s]
                for s in sorted(sentence_repetitions.keys())[n_train:]
            )
            print(f"  Estimated transitions: ~{train_transitions_est} train "
                  f"/ ~{test_transitions_est} test")

check_sentence_counts(pipeline.dutch30_extractor)
