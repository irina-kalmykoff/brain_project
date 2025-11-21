# Converted from parse_features_of_30_patients.ipynb

import os
import glob
import numpy as np
import pickle
import pandas as pd
#from IPython.display import Audio, display
from collections import Counter, defaultdict
from pynwb import NWBHDF5IO
from datetime import datetime
import scipy.signal

import matplotlib.pyplot as plt
import seaborn as sns

import h5py
import gc
import json

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, silhouette_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cosine, euclidean


from extract_features import extractHG, stackFeatures, downsampleLabels
from brain_audio_decoder import BrainAudioDecoder
from custom_decoder import CustomBrainAudioDecoder
from brain_audio_decoder_viz import BrainAudioDecoderViz
from acoustic_change_detector import AcousticChangeDetector
from phoneme_validator import PhonemeValidator
from phonetic_dictionary import PhoneticDictionary
from feature_vizualizer import PhonemeFeatureVisualizer
from markov_phoneme_model import MarkovPhonemeModel
from extract_features import extractHG, downsampleLabels, extractMelSpecs
from pipeline import UnifiedPhonemePipeline
from config import BIDS_PATH, OUTPUT_PATH, RESULTS_PATH, DUTCH_30_PATH, DUTCH_10_PATH, get_dataset_paths

from dutch_30_pipeline import Dutch30Pipeline
from dutch_30_feature_extractor import Dutch30FeatureExtractor
from phoneme_detection_diagnostic import Dutch30PhonemeDetectionDiagnostic 
from dataset_config import Dutch30Config

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
visualizer = PhonemeFeatureVisualizer(output_dir='./phoneme_visualizations')

# # Step 3: Test phoneme boundary detection
# print("\n[3/5] Testing phoneme boundary detection...")
# batch = None  # Initialize for error handling
# try:
#     if pipeline is None:
#         raise RuntimeError("Pipeline not initialized from previous step")
    
#     # FIXED: Call step4 BEFORE trying to access decoder
#     pipeline.step4_custom_detector()
    
#     # Verify decoder exists
#     if not hasattr(pipeline, 'detector'):
#         print("ERROR: Detector not created")
#         exit(1)
    
#     # Get a batch using the decoder's method
#     batch = pipeline.get_data_batch(
#         split_result=pipeline.split_result,
#         batch_type='train',
#         batch_size=3
#     )
    
#     print(f"  Batch has {len(batch.get('words', []))} samples")
    
#     if not batch.get('spectrogram_segments') or batch['spectrogram_segments'][0] is None:
#         print("ERROR: No spectrograms in batch")
#         exit(1)
    
#     # Check spectrogram is 2D
#     spec = batch['spectrogram_segments'][0]
#     print(f"  Spectrogram shape: {spec.shape}")
    
#     if len(spec.shape) != 2:
#         print("ERROR: Spectrogram should be 2D (time, frequency)")
#         exit(1)
    
#     # Test boundary detection on one word
#     word = batch['words'][0]
#     result = pipeline.detector.detect_boundaries(
#         spectrogram=spec,
#         word=word,
#         frameshift=0.01
#     )
    
#     print(f"  Word: '{word}'")
#     print(f"  Expected phonemes: {result.get('n_phonemes', '?')}")
#     print(f"  Detected segments: {len(result['segments'])}")
    
#     if result['segments']:
#         print(f"  Segment lengths: {[s.shape[0] for s in result['segments'][:5]]}")
        
#         # Check segments have different lengths
#         lengths = [s.shape[0] for s in result['segments']]
#         unique_lengths = len(set(lengths))
        
#         print(f"  Unique segment lengths: {unique_lengths}/{len(lengths)}")
        
#         if unique_lengths == 1:
#             print("WARNING: All segments same length - boundary detection may not be working")
#         else:
#             print("OK: Segments have varying lengths")
#     else:
#         print("ERROR: No segments detected")
#         exit(1)
    
# except Exception as e:
#     print(f"ERROR: {e}")
#     import traceback
#     traceback.print_exc()
#     if batch is None:
#         print("\n Cannot continue - batch creation failed")
#         exit(1)

# result = pipeline.detector.detect_boundaries(
#     spectrogram=spec,
#     word=word,
#     frameshift=0.01
# )

# # Step 4: Process batch and check phoneme-level data
# print("\n[4/5] Testing phoneme segmentation...")
# enhanced_batch = None  # Initialize for error handling
# try:
#     if batch is None:
#         raise RuntimeError("Batch not created from previous step")
    
#     enhanced_batch = pipeline.detector.process_batch(
#         batch,
#         apply_segmentation=True,
#         detect_phonemes=True
#     )
    
#     n_phonemes = len(enhanced_batch.get('phoneme_labels', []))
#     unique_phonemes = set(enhanced_batch.get('phoneme_labels', []))
#     unique_count = len(unique_phonemes)
    
#     print(f"  Total phoneme segments: {n_phonemes}")
#     print(f"  Unique phonemes: {unique_count}")
    
#     if n_phonemes == 0:
#         print("  ✗ ERROR: No phoneme segments extracted")
#         exit(1)
    
#     # Check for unknowns
#     unknown_count = enhanced_batch['phoneme_labels'].count('?')
#     unknown_pct = (unknown_count / n_phonemes) * 100 if n_phonemes > 0 else 0
    
#     print(f"  Unknown phonemes: {unknown_count} ({unknown_pct:.1f}%)")
    
#     # Show sample phonemes
#     sample_phonemes = [p for p in list(unique_phonemes)[:5] if p != '?']
#     if sample_phonemes:
#         print(f"  Sample phonemes: {sample_phonemes}")
    
#     if unknown_pct > 80:
#         print("  ⚠ WARNING: Too many unknown phonemes (>80%)")
#         print("     This suggests the phonetic dictionary may not cover these words")
#     elif unknown_pct > 50:
#         print("  ⚠ WARNING: Many unknown phonemes (>50%)")
#     else:
#         print("  ✓ OK: Most phonemes identified")
    
# except Exception as e:
#     print(f"  ✗ ERROR: {e}")
#     import traceback
#     traceback.print_exc()
#     if enhanced_batch is None:
#         print("\n⚠ Cannot continue - phoneme segmentation failed")
#         exit(1)

# result = pipeline.detector.detect_boundaries(
#     spectrogram=spec,
#     word=word,
#     frameshift=0.01
# )

# # DEBUG CODE - ADD THIS:
# print(f"\n=== DEBUG for '{word}' ===")
# print(f"Spectrogram shape: {spec.shape}")
# print(f"Spectrogram min/max: [{np.min(spec):.3f}, {np.max(spec):.3f}]")
# print(f"Spectrogram std: {np.std(spec):.3f}")

# if 'distances' in result:
#     dists = result['distances']
#     print(f"\nDistances shape: {dists.shape}")
#     print(f"Distances min/max: [{np.min(dists):.3f}, {np.max(dists):.3f}]")
#     print(f"Distances mean: {np.mean(dists):.3f}")
#     print(f"Distances std: {np.std(dists):.3f}")
    
#     # Show distribution
#     percentiles = [25, 50, 75, 90, 95, 99]
#     print(f"Percentiles: {[f'{p}%={np.percentile(dists, p):.3f}' for p in percentiles]}")
    
#     # Show enhanced distances if available
#     if 'enhanced_distances' in result:
#         enh = result['enhanced_distances']
#         print(f"\nEnhanced distances min/max: [{np.min(enh):.3f}, {np.max(enh):.3f}]")
#         print(f"Enhanced mean: {np.mean(enh):.3f}")
        
#     # Plot
#     plt.figure(figsize=(12, 4))
#     plt.subplot(1, 2, 1)
#     plt.imshow(spec.T, aspect='auto', origin='lower', cmap='viridis')
#     plt.title(f"Spectrogram: '{word}'")
#     plt.ylabel('Mel Bin')
#     plt.xlabel('Frame')
    
#     plt.subplot(1, 2, 2)
#     plt.plot(dists, label='distances')
#     if 'enhanced_distances' in result:
#         plt.plot(result['enhanced_distances'], label='enhanced', alpha=0.7)
#     plt.title('Frame-to-Frame Distances')
#     plt.xlabel('Frame')
#     plt.ylabel('Distance')
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(f'debug_{word}.png')
#     print(f"Saved plot to debug_{word}.png")
#     plt.close()

# print("="*40)

# # Step 5: Extract features and check variance
# print("\n[5/5] Testing feature extraction and variance...")
# try:
#     if enhanced_batch is None:
#         raise RuntimeError("Enhanced batch not created from previous step")
    
#     prepared = pipeline.detector.prepare_phoneme_training_data(
#         enhanced_batch,
#         feature_extraction_method='high_gamma'
#     )
    
#     features = prepared['features']
#     labels = prepared['phoneme_labels']
    
#     print(f"  Features extracted: {len(features)}")
#     if features:
#         print(f"  Sample feature shape: {features[0].shape}")
    
#     if not features:
#         print("  ERROR: No features extracted")
#         exit(1)
    
#     # Calculate variance for one phoneme
#     phoneme_features = defaultdict(list)
    
#     for feat, label in zip(features[:50], labels[:50]):
#         if label != '?':  # Skip unknowns
#             phoneme_features[label].append(feat)
    
#     # Check variance for most common phoneme
#     if phoneme_features:
#         most_common = max(phoneme_features.keys(), key=lambda k: len(phoneme_features[k]))
#         n_samples = len(phoneme_features[most_common])
        
#         print(f"  Most common phoneme: '{most_common}' ({n_samples} samples)")
        
#         if n_samples >= 2:
#             feats = phoneme_features[most_common][:min(5, n_samples)]
#             flat_feats = [f.flatten() for f in feats]
            
#             # Calculate pairwise distances
#             distances = []
#             for i in range(len(flat_feats)):
#                 for j in range(i+1, len(flat_feats)):
#                     min_len = min(len(flat_feats[i]), len(flat_feats[j]))
#                     if min_len > 0:
#                         dist = cosine(flat_feats[i][:min_len], flat_feats[j][:min_len])
#                         distances.append(dist)
            
#             if distances:
#                 avg_distance = np.mean(distances)
#                 min_distance = np.min(distances)
#                 max_distance = np.max(distances)
                
#                 print(f"  Pairwise distances: min={min_distance:.3f}, avg={avg_distance:.3f}, max={max_distance:.3f}")
                
#                 if avg_distance > 0.8:
#                     print("  WARNING: High variance (>0.8) - features may be inconsistent")
#                     print("     This suggests phoneme boundaries may not be working correctly")
#                 elif avg_distance < 0.5:
#                     print("  OK: Good consistency (<0.5)")
#                 else:
#                     print("  OK: Acceptable consistency (0.5-0.8)")
#             else:
#                 print(" Could not calculate distances")
#         else:
#             print(f"  Only {n_samples} sample(s) - cannot check variance")
#             print("     Try increasing batch_size or sample_fraction")
#     else:
#         print("  WARNING: No phoneme features to check variance")
#         print("     All phonemes may be unknown '?'")
    
# except Exception as e:
#     print(f"  ERROR: {e}")
#     import traceback
#     traceback.print_exc()
#     exit(1)

# print("="*70)
# print("\nYou can now run the full pipeline:")
# print("  pipeline.run_step1_to_step6(sample_fraction=0.01)")
# print("\nExpected improvements over previous version:")
# print("  ✓ Variance should be < 0.8 (was 0.95+)")
# print("  ✓ Unknown phonemes should be < 30% (was >80%)")
# print("  ✓ Features should vary by phoneme, not just word")

# Create config
config = Dutch30Config()

# Pass config to both extractor and pipeline
extractor = Dutch30FeatureExtractor(config=config)

pipeline_debug = Dutch30Pipeline(
    dutch30_extractor=extractor,
    config=config, 
    debug_mode=True,
    pca_components=100,
    feature_extraction_method='high_gamma',
    use_phoneme_groups=True
)

# Debug a specific patient
pipeline_debug.debug_sentence_parsing('sub-p21', max_samples=3)
print([attr for attr in dir(pipeline_debug) if 'detect' in attr.lower()])

diag = Dutch30PhonemeDetectionDiagnostic(pipeline_debug)

# Visualize word #5 from patient P21
diag.visualize_word_analysis('sub-p01', word_index=4, 
                             save_path='p21_word5.png')

# Quick check first 10 words
diag.batch_diagnostic('sub-p21', num_samples=10)

config = Dutch30Config()
extractor = Dutch30FeatureExtractor(config=config)

pipeline = Dutch30Pipeline(
    dutch30_extractor=extractor,
    config = config,
    pca_components = 100,
    feature_extraction_method= 'band_powers', #'high_gamma',
    debug_mode=False
)

# step_0
# pipeline.analyze_dutch30_channels()

# pipeline.step1_load_dutch30_data(sample_fraction=0.000001)
# pipeline.step2_3_use_existing_split()
# pipeline.step4_custom_detector()
# pipeline.step5_accumulate_data_dutch30()

# pipeline.dutch30_step6_resolve_unknowns();

pipeline.run_step1_to_step6(sample_fraction=0.0001, force_reprocess=True)
#pipeline.run_step1_to_step6(sample_fraction=0.0001)
# pipeline.step2_3_use_existing_split()
# pipeline.step4_custom_detector()
# pipeline.step5_accumulate_data()
# pipeline.step6_resolve_unknowns()
# pipeline.checkpoint_after_step6(sample_fraction=0.0005)
# pipeline.try_load_checkpoint(sample_fraction=0.1)
#pipeline.run_step1_to_step6(sample_fraction=0.01, force_reprocess=False)

# pipeline.checkpoint_after_step6(sample_fraction=0.0001)

# pipeline.try_load_checkpoint(sample_fraction=0.0001)

# pipeline.analyze_phoneme_lengths()

pipeline.step7_filter_unknowns(unknown_keep_ratio=0.1)
#pipeline.step8_convert_to_groups()

def analyze_phoneme_positions_and_patients(pipeline, long_threshold=0.5):
    """
    Analyze if extra-long phonemes correlate with:
    1. Specific patients
    2. Position in word (start/end vs middle)
    """
    from collections import defaultdict
    
    features = pipeline.train['features']
    labels = pipeline.train['phoneme_labels']
    words = pipeline.train.get('phoneme_words', ['unknown'] * len(labels))
    participants = pipeline.train.get('phoneme_participant_ids', ['unknown'] * len(labels))
    positions = pipeline.train.get('phoneme_positions', [0] * len(labels))
    
    # Collect data
    patient_stats = defaultdict(lambda: {'total': 0, 'long': 0, 'durations': []})
    position_stats = defaultdict(lambda: {'total': 0, 'long': 0, 'durations': []})
    
    for i, feat in enumerate(features):
        n_frames = feat.shape[0]
        duration = n_frames * 0.01
        
        patient = participants[i]
        position = positions[i]
        word = words[i]
        
        # Get word phoneme count to determine position category
        # Try to get expected phoneme count from word
        if hasattr(pipeline, 'detector') and hasattr(pipeline.detector, 'phonetic_dict'):
            try:
                expected_phonemes = pipeline.detector.phonetic_dict.extract_phonemes(word)
                n_phonemes = len(expected_phonemes)
            except:
                n_phonemes = None
        else:
            n_phonemes = None
        
        # Categorize position
        if n_phonemes and n_phonemes > 0:
            if position == 0:
                pos_category = 'first'
            elif position == n_phonemes - 1:
                pos_category = 'last'
            else:
                pos_category = 'middle'
        else:
            pos_category = 'unknown'
        
        # Track by patient
        patient_stats[patient]['total'] += 1
        patient_stats[patient]['durations'].append(duration)
        if duration > long_threshold:
            patient_stats[patient]['long'] += 1
        
        # Track by position
        position_stats[pos_category]['total'] += 1
        position_stats[pos_category]['durations'].append(duration)
        if duration > long_threshold:
            position_stats[pos_category]['long'] += 1
    
    # Print results
    print("\n" + "="*70)
    print(f"EXTRA-LONG PHONEME ANALYSIS (>{long_threshold}s)")
    print("="*70)
    
    # By patient
    print("\nPER-PATIENT STATISTICS:")
    print("-"*70)
    
    patient_data = []
    for patient in sorted(patient_stats.keys()):
        stats = patient_stats[patient]
        pct_long = (stats['long'] / stats['total'] * 100) if stats['total'] > 0 else 0
        mean_dur = np.mean(stats['durations'])
        patient_data.append({
            'patient': patient,
            'total': stats['total'],
            'long': stats['long'],
            'pct_long': pct_long,
            'mean_dur': mean_dur
        })
    
    # Sort by percentage of long phonemes
    patient_data.sort(key=lambda x: x['pct_long'], reverse=True)
    
    print(f"{'Patient':<15} {'Total':<10} {'Long':<10} {'% Long':<10} {'Mean Dur':<10}")
    print("-"*70)
    for p in patient_data:
        print(f"{p['patient']:<15} {p['total']:<10} {p['long']:<10} {p['pct_long']:>6.1f}%    {p['mean_dur']:>6.3f}s")
    
    # By position
    print("\n" + "="*70)
    print("BY POSITION IN WORD:")
    print("-"*70)
    print(f"{'Position':<15} {'Total':<10} {'Long':<10} {'% Long':<10} {'Mean Dur':<10}")
    print("-"*70)
    
    for pos in ['first', 'middle', 'last', 'unknown']:
        if pos in position_stats:
            stats = position_stats[pos]
            pct_long = (stats['long'] / stats['total'] * 100) if stats['total'] > 0 else 0
            mean_dur = np.mean(stats['durations'])
            print(f"{pos:<15} {stats['total']:<10} {stats['long']:<10} {pct_long:>6.1f}%    {mean_dur:>6.3f}s")
    
    # Visualizations
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Patients with most long phonemes
    top_patients = patient_data[:10]  # Top 10 worst
    patient_names = [p['patient'] for p in top_patients]
    pcts = [p['pct_long'] for p in top_patients]
    
    axes[0].barh(patient_names, pcts, color='coral')
    axes[0].set_xlabel('% Extra-Long Phonemes')
    axes[0].set_title('Top 10 Patients with Most Long Phonemes')
    axes[0].invert_yaxis()
    axes[0].grid(axis='x', alpha=0.3)
    
    # Plot 2: Position distribution
    positions_plot = ['first', 'middle', 'last']
    pcts_pos = [position_stats[p]['long'] / position_stats[p]['total'] * 100 
                if p in position_stats and position_stats[p]['total'] > 0 else 0
                for p in positions_plot]
    
    colors = ['lightblue', 'lightgreen', 'lightyellow']
    axes[1].bar(positions_plot, pcts_pos, color=colors, edgecolor='black')
    axes[1].set_ylabel('% Extra-Long Phonemes')
    axes[1].set_title('Long Phonemes by Position in Word')
    axes[1].grid(axis='y', alpha=0.3)
    
    # Plot 3: Duration distribution by position
    for pos, color in zip(['first', 'middle', 'last'], colors):
        if pos in position_stats:
            durations = position_stats[pos]['durations']
            axes[2].hist(durations, bins=50, alpha=0.5, label=pos, color=color, 
                        range=(0, 1.0), edgecolor='black')
    
    axes[2].axvline(long_threshold, color='red', linestyle='--', 
                   label=f'Long threshold ({long_threshold}s)')
    axes[2].set_xlabel('Duration (s)')
    axes[2].set_ylabel('Count')
    axes[2].set_title('Duration Distribution by Position')
    axes[2].legend()
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./phoneme_position_patient_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Statistical test: Are first/last significantly longer than middle?
    if 'first' in position_stats and 'middle' in position_stats and 'last' in position_stats:
        from scipy import stats
        
        first_durs = position_stats['first']['durations']
        middle_durs = position_stats['middle']['durations']
        last_durs = position_stats['last']['durations']
        
        print("\n" + "="*70)
        print("STATISTICAL COMPARISON:")
        print("-"*70)
        
        # Compare first vs middle
        t_stat, p_val = stats.ttest_ind(first_durs, middle_durs)
        print(f"First vs Middle: t={t_stat:.3f}, p={p_val:.4f}")
        if p_val < 0.001:
            print("  *** Highly significant difference!")
        elif p_val < 0.05:
            print("  ** Significant difference")
        else:
            print("  No significant difference")
        
        # Compare last vs middle
        t_stat, p_val = stats.ttest_ind(last_durs, middle_durs)
        print(f"Last vs Middle: t={t_stat:.3f}, p={p_val:.4f}")
        if p_val < 0.001:
            print("  *** Highly significant difference!")
        elif p_val < 0.05:
            print("  ** Significant difference")
        else:
            print("  No significant difference")
        
        print("\nMean durations:")
        print(f"  First:  {np.mean(first_durs):.3f}s")
        print(f"  Middle: {np.mean(middle_durs):.3f}s")
        print(f"  Last:   {np.mean(last_durs):.3f}s")
    
    return patient_stats, position_stats

# Run it after step 5
patient_stats, position_stats = analyze_phoneme_positions_and_patients(pipeline, long_threshold=0.4)

# # Now you can use the pipeline's training methods
train_data = pipeline.get_training_data(filtered=True)
test_data = pipeline.get_test_data()

def plot_detailed_confusion_matrix(model, test_features, true_labels, save_path=None):
    """Create detailed confusion matrix for phoneme predictions"""
    
    # Get predictions
    predictions, _ = model.predict(test_features, use_viterbi=True)
    
    # Get unique labels
    unique_labels = sorted(set(true_labels + predictions))
    
    # Filter to show only phonemes that actually appear
    label_to_idx = {label: i for i, label in enumerate(unique_labels)}
    n_classes = len(unique_labels)
    
    # Build confusion matrix
    conf_matrix = np.zeros((n_classes, n_classes))
    for true, pred in zip(true_labels, predictions):
        if true in label_to_idx and pred in label_to_idx:
            conf_matrix[label_to_idx[true], label_to_idx[pred]] += 1
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Raw counts
    im1 = ax1.imshow(conf_matrix, cmap='Blues', aspect='auto')
    ax1.set_xticks(range(n_classes))
    ax1.set_yticks(range(n_classes))
    ax1.set_xticklabels(unique_labels, rotation=90, ha='right')
    ax1.set_yticklabels(unique_labels)
    ax1.set_xlabel('Predicted Phoneme')
    ax1.set_ylabel('True Phoneme')
    ax1.set_title('Confusion Matrix (Counts)')
    
    # Add text annotations for non-zero values
    for i in range(n_classes):
        for j in range(n_classes):
            if conf_matrix[i, j] > 0:
                text = ax1.text(j, i, int(conf_matrix[i, j]),
                              ha="center", va="center", color="white" if conf_matrix[i, j] > conf_matrix.max()/2 else "black")
    
    plt.colorbar(im1, ax=ax1)
    
    # Normalized by row (recall per phoneme)
    conf_norm = conf_matrix / (conf_matrix.sum(axis=1, keepdims=True) + 1e-10)
    im2 = ax2.imshow(conf_norm, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
    ax2.set_xticks(range(n_classes))
    ax2.set_yticks(range(n_classes))
    ax2.set_xticklabels(unique_labels, rotation=90, ha='right')
    ax2.set_yticklabels(unique_labels)
    ax2.set_xlabel('Predicted Phoneme')
    ax2.set_ylabel('True Phoneme')
    ax2.set_title('Confusion Matrix (Normalized by True)')
    
    plt.colorbar(im2, ax=ax2)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print accuracy per phoneme
    print("\nPer-phoneme accuracy:")
    phoneme_correct = {}
    phoneme_total = {}
    
    for true, pred in zip(true_labels, predictions):
        if true not in phoneme_total:
            phoneme_total[true] = 0
            phoneme_correct[true] = 0
        phoneme_total[true] += 1
        if true == pred:
            phoneme_correct[true] += 1
    
    for phoneme in sorted(phoneme_total.keys()):
        acc = phoneme_correct[phoneme] / phoneme_total[phoneme]
        print(f"  {phoneme:5s}: {acc:.3f} ({phoneme_correct[phoneme]}/{phoneme_total[phoneme]})")
    
    return conf_matrix

def train_phoneme_model(pipeline, output_base='./results/dutch30', use_balanced=True,  use_filtered=True):
    
    if use_filtered and hasattr(pipeline, 'train_filtered'):
        train_data = pipeline.train_filtered
        print(f"Using FILTERED data with {len(train_data['features'])} train samples")
    else:
        train_data = pipeline.train
        print(f"Using ORIGINAL data with {len(train_data['features'])} train samples")
    
    # Choose which dataset to use
    if use_balanced and hasattr(pipeline, 'train_balanced'):
        train_data = pipeline.train_balanced
        print(f"Using BALANCED data with {len(train_data['features'])} train samples")
    else:
        train_data = pipeline.train
        print(f"Using ORIGINAL data with {len(train_data['features'])} train samples")
    
    # Count phonemes in training data
    train_counts = Counter(pipeline.train['phoneme_labels'])
    test_counts = Counter(pipeline.test['phoneme_labels'])
    
    print(f"\nTraining data summary, total samples: {len(pipeline.train['phoneme_labels'])}")
    print(f"  Unique phonemes: {len(train_counts)}, most common: {train_counts.most_common(5)}, least common: {train_counts.most_common()[-5:]}")
    
    # Initialize
    model_phonemes = MarkovPhonemeModel(
        phonetic_dict=pipeline.phonetic_dict,
        order=2,
        output_dir=os.path.join(output_base, 'markov_phonemes'),
        debug_mode=False, 
        use_groups=False
    )
    
    # Train
    results_phonemes = model_phonemes.train(
        features=pipeline.train['features'],
        phoneme_labels=pipeline.train['phoneme_labels'],
        words=pipeline.train.get('phoneme_words'),
        participant_ids=pipeline.train.get('phoneme_participant_ids')
    )
    
    # Evaluate
    eval_phonemes = model_phonemes.evaluate(
        features=pipeline.test['features'],
        true_labels=pipeline.test['phoneme_labels'],
        use_viterbi=True
    )
    
    print(f"\nOverall Accuracy: {eval_phonemes['accuracy']:.4f}")
    
    # Create histogram of samples per phoneme
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Training data distribution
    phonemes = list(train_counts.keys())
    counts = list(train_counts.values())
    
    axes[0, 0].bar(range(len(phonemes)), counts, color='cornflowerblue')
    axes[0, 0].set_xticks(range(len(phonemes)))
    axes[0, 0].set_xticklabels(phonemes, rotation=90)
    axes[0, 0].set_ylabel('Number of samples')
    axes[0, 0].set_title(f'Training Data Distribution ({len(pipeline.train["phoneme_labels"])} total samples)')
    axes[0, 0].axhline(y=np.mean(counts), color='red', linestyle='--', label=f'Mean: {np.mean(counts):.0f}')
    axes[0, 0].legend()
    
    # 2. Test data distribution
    test_phonemes = list(test_counts.keys())
    test_values = list(test_counts.values())
    
    axes[0, 1].bar(range(len(test_phonemes)), test_values, color='coral')
    axes[0, 1].set_xticks(range(len(test_phonemes)))
    axes[0, 1].set_xticklabels(test_phonemes, rotation=90)
    axes[0, 1].set_ylabel('Number of samples')
    axes[0, 1].set_title(f'Test Data Distribution ({len(pipeline.test["phoneme_labels"])} total samples)')
    axes[0, 1].axhline(y=np.mean(test_values), color='red', linestyle='--', label=f'Mean: {np.mean(test_values):.0f}')
    axes[0, 1].legend()
    
    # 3. Per-phoneme accuracy
    predictions, _ = model_phonemes.predict(pipeline.test['features'], use_viterbi=True)
    
    phoneme_correct = {}
    phoneme_total = {}
    
    for true, pred in zip(pipeline.test['phoneme_labels'], predictions):
        if true not in phoneme_total:
            phoneme_total[true] = 0
            phoneme_correct[true] = 0
        phoneme_total[true] += 1
        if true == pred:
            phoneme_correct[true] += 1
    
    accuracies = []
    phoneme_list = []
    for phoneme in sorted(phoneme_total.keys()):
        acc = phoneme_correct[phoneme] / phoneme_total[phoneme] if phoneme_total[phoneme] > 0 else 0
        accuracies.append(acc)
        phoneme_list.append(phoneme)
    
    axes[1, 0].bar(range(len(phoneme_list)), accuracies, color='mediumseagreen')
    axes[1, 0].set_xticks(range(len(phoneme_list)))
    axes[1, 0].set_xticklabels(phoneme_list, rotation=90)
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].set_title('Per-Phoneme Accuracy')
    axes[1, 0].set_ylim([0, 1])
    axes[1, 0].axhline(y=eval_phonemes['accuracy'], color='red', linestyle='--', 
                       label=f'Overall: {eval_phonemes["accuracy"]:.3f}')
    axes[1, 0].legend()
    
    # 4. Accuracy vs Sample Count
    sample_counts = [test_counts.get(p, 0) for p in phoneme_list]
    
    axes[1, 1].scatter(sample_counts, accuracies, alpha=0.6, s=50)
    axes[1, 1].set_xlabel('Number of test samples')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].set_title('Accuracy vs Sample Count')
    axes[1, 1].axhline(y=eval_phonemes['accuracy'], color='red', linestyle='--', alpha=0.5)
    axes[1, 1].axvline(x=10, color='orange', linestyle='--', alpha=0.5, label='10 samples')
    axes[1, 1].axvline(x=50, color='green', linestyle='--', alpha=0.5, label='50 samples')
    
    # Add phoneme labels for outliers
    for i, (x, y, p) in enumerate(zip(sample_counts, accuracies, phoneme_list)):
        if y > 0.1 or x > 200:  # Label high accuracy or high count phonemes
            axes[1, 1].annotate(p, (x, y), fontsize=8, alpha=0.7)
    
    axes[1, 1].legend()
    axes[1, 1].set_xlim([-5, max(sample_counts) + 10])
    axes[1, 1].set_ylim([-0.05, 1.05])
    
    plt.tight_layout()
    #plt.savefig(os.path.join(output_base, 'phoneme_analysis.png'), dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
    print(f"Overall accuracy: {eval_phonemes['accuracy']:.4f}")
    print(f"Average per-phoneme accuracy: {np.mean(accuracies):.4f}")
    print(f"Phonemes with >5% accuracy: {sum(1 for a in accuracies if a > 0.05)}/{len(accuracies)}")
    print(f"Phonemes with 0% accuracy: {sum(1 for a in accuracies if a == 0)}/{len(accuracies)}")
    
    # Identify problem areas
    print("\nPhonemes needing more data (< 50 training samples):")
    for phoneme, count in sorted(train_counts.items(), key=lambda x: x[1]):
        if count < 50:
            acc = phoneme_correct.get(phoneme, 0) / phoneme_total.get(phoneme, 1)
            print(f"  {phoneme:5s}: {count:3d} samples, {acc:.3f} accuracy")
    
    print("\nBest performing phonemes:")
    best_phonemes = [(p, phoneme_correct[p]/phoneme_total[p]) 
                     for p in phoneme_total if phoneme_total[p] > 0]
    best_phonemes.sort(key=lambda x: x[1], reverse=True)
    for phoneme, acc in best_phonemes[:5]:
        print(f"  {phoneme:5s}: {acc:.3f} ({phoneme_correct[phoneme]}/{phoneme_total[phoneme]})")
    
    return model_phonemes, eval_phonemes

def balance_phoneme_data(pipeline, strategy='quality_sample', target_samples=None, 
                        min_samples=50, outlier_threshold=2.0):
    """
    Balance phoneme data by selecting quality samples
    
    Parameters:
    -----------
    strategy : str
        'quality_sample' - pick samples closest to phoneme centroid
        'random_sample' - random sampling (old behavior)
    target_samples : int or None
        Number of samples per phoneme. If None, use median count
    min_samples : int
        Minimum samples required to keep a phoneme
    outlier_threshold : float
        Z-score threshold for outlier removal (default 2.0 = within 2 std devs)
    """
    from scipy.spatial.distance import cosine, euclidean
    
    # Count samples per phoneme
    train_counts = Counter(pipeline.train['phoneme_labels'])
    
    print(f"\n{'='*70}")
    print("BALANCING PHONEME DATA")
    print(f"{'='*70}")
    print(f"Original distribution:")
    print(f"  Total samples: {len(pipeline.train['phoneme_labels'])}")
    print(f"  Unique phonemes: {len(train_counts)}")
    print(f"  Max samples: {max(train_counts.values())}")
    print(f"  Min samples: {min(train_counts.values())}")
    print(f"  Mean samples: {np.mean(list(train_counts.values())):.0f}")
    print(f"  Median samples: {np.median(list(train_counts.values())):.0f}")
    
    # Set target samples
    if target_samples is None:
        target_samples = int(np.median(list(train_counts.values())))
    
    print(f"\nTarget samples per phoneme: {target_samples}")
    print(f"Outlier threshold: {outlier_threshold} std devs")
    
    # Filter out rare phonemes first
    common_phonemes = {p: c for p, c in train_counts.items() if c >= min_samples}
    print(f"\nRemoving {len(train_counts) - len(common_phonemes)} rare phonemes (< {min_samples} samples)")
    print(f"Removed: {set(train_counts.keys()) - set(common_phonemes.keys())}")
    
    if strategy == 'quality_sample':
        balanced_features = []
        balanced_labels = []
        balanced_words = []
        balanced_participant_ids = []
        
        outliers_removed_total = 0
        
        for phoneme in common_phonemes:
            # Get all samples for this phoneme
            indices = [i for i, label in enumerate(pipeline.train['phoneme_labels']) 
                      if label == phoneme]
            
            # Get features for this phoneme
            phoneme_features = [pipeline.train['features'][idx] for idx in indices]
            
            # Flatten features for distance calculation
            flattened = []
            for feat in phoneme_features:
                if feat.ndim > 1:
                    flattened.append(feat.flatten())
                else:
                    flattened.append(feat)
            
            # Pad to same length (use max length)
            max_len = max(len(f) for f in flattened)
            padded = []
            for f in flattened:
                if len(f) < max_len:
                    padded_f = np.zeros(max_len)
                    padded_f[:len(f)] = f
                    padded.append(padded_f)
                else:
                    padded.append(f)
            
            features_array = np.array(padded)
            
            # Calculate centroid (mean feature vector)
            centroid = np.mean(features_array, axis=0)
            
            # Calculate distance of each sample to centroid
            distances = []
            for feat in features_array:
                dist = euclidean(feat, centroid)
                distances.append(dist)
            
            distances = np.array(distances)
            
            # Remove outliers (samples too far from centroid)
            mean_dist = np.mean(distances)
            std_dist = np.std(distances)
            
            inlier_mask = distances < (mean_dist + outlier_threshold * std_dist)
            inlier_indices = [indices[i] for i in range(len(indices)) if inlier_mask[i]]
            
            outliers_removed = len(indices) - len(inlier_indices)
            outliers_removed_total += outliers_removed
            
            if len(inlier_indices) == 0:
                print(f"  WARNING: All samples removed as outliers for '{phoneme}'!")
                continue
            
            # Sample from inliers
            current_count = len(inlier_indices)
            
            if current_count >= target_samples:
                # We have enough inliers - pick the closest ones to centroid
                inlier_distances = distances[inlier_mask]
                sorted_inlier_idx = np.argsort(inlier_distances)[:target_samples]
                selected = [inlier_indices[i] for i in sorted_inlier_idx]
            else:
                # Not enough inliers - use all inliers and repeat closest ones
                print(f"  '{phoneme}': Only {current_count} inliers, upsampling to {target_samples}")
                inlier_distances = distances[inlier_mask]
                sorted_inlier_idx = np.argsort(inlier_distances)
                
                # Use all inliers
                selected = inlier_indices.copy()
                
                # Repeat closest samples to reach target
                n_repeats_needed = target_samples - current_count
                closest_indices = [inlier_indices[i] for i in sorted_inlier_idx[:n_repeats_needed]]
                selected.extend(closest_indices)
            
            # Add to balanced dataset
            for idx in selected:
                balanced_features.append(pipeline.train['features'][idx])
                balanced_labels.append(pipeline.train['phoneme_labels'][idx])
                if 'phoneme_words' in pipeline.train:
                    balanced_words.append(pipeline.train['phoneme_words'][idx])
                if 'phoneme_participant_ids' in pipeline.train:
                    balanced_participant_ids.append(pipeline.train['phoneme_participant_ids'][idx])
        
        print(f"\nTotal outliers removed: {outliers_removed_total}")
        
    elif strategy == 'random_sample':
        # Old random sampling behavior
        balanced_features = []
        balanced_labels = []
        balanced_words = []
        balanced_participant_ids = []
        
        for phoneme in common_phonemes:
            indices = [i for i, label in enumerate(pipeline.train['phoneme_labels']) 
                      if label == phoneme]
            
            current_count = len(indices)
            
            if current_count > target_samples:
                selected = np.random.choice(indices, target_samples, replace=False)
            else:
                selected = np.random.choice(indices, target_samples, replace=True)
            
            for idx in selected:
                balanced_features.append(pipeline.train['features'][idx])
                balanced_labels.append(pipeline.train['phoneme_labels'][idx])
                if 'phoneme_words' in pipeline.train:
                    balanced_words.append(pipeline.train['phoneme_words'][idx])
                if 'phoneme_participant_ids' in pipeline.train:
                    balanced_participant_ids.append(pipeline.train['phoneme_participant_ids'][idx])
    
    # Shuffle
    indices = np.random.permutation(len(balanced_features))
    
    pipeline.train_balanced = {
        'features': [balanced_features[i] for i in indices],
        'phoneme_labels': [balanced_labels[i] for i in indices],
        'phoneme_words': [balanced_words[i] for i in indices] if balanced_words else [],
        'phoneme_participant_ids': [balanced_participant_ids[i] for i in indices] if balanced_participant_ids else []
    }
    
    new_counts = Counter(pipeline.train_balanced['phoneme_labels'])
    print(f"\n{'='*70}")
    print("BALANCED DATASET CREATED")
    print(f"{'='*70}")
    print(f"  Total samples: {len(pipeline.train_balanced['phoneme_labels'])}")
    print(f"  Unique phonemes: {len(new_counts)}")
    print(f"  Samples per phoneme: {target_samples}")
    print(f"  Total samples: {len(new_counts) * target_samples}")
    
    return pipeline

# Balance data for training
pipeline = balance_phoneme_data(
    pipeline, 
    strategy='quality_sample',
    target_samples=100,      # 100 samples per phoneme
    min_samples=100,          # Remove phonemes with < x samples
    outlier_threshold=2.0    # Remove samples >2 std devs from centroid
)

print("First feature shape:", pipeline.train['features'][0].shape)
print("First feature sample:", pipeline.train['features'][0][0, :5])
print(pipeline.train['metadata'])

# Prepare data
X_train = np.array([f.flatten() for f in pipeline.train['features']])
y_train = np.array(pipeline.train['phoneme_labels'])

X_test = np.array([f.flatten() for f in pipeline.test['features']])
y_test = np.array(pipeline.test['phoneme_labels'])

print(f"Training: {X_train.shape}, Test: {X_test.shape}")
print(f"Unique phonemes: {len(set(y_train))}")

# 1. LDA
print("\n" + "="*70)
print("LINEAR DISCRIMINANT ANALYSIS")
print("="*70)

lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)

train_acc = lda.score(X_train, y_train)
test_acc = lda.score(X_test, y_test)

print(f"Train accuracy: {train_acc:.1%}")
print(f"Test accuracy:  {test_acc:.1%}")

y_pred_lda = lda.predict(X_test)
print("\nPer-phoneme performance:")
print(classification_report(y_test, y_pred_lda, zero_division=0))

# 2. SVM
print("\n" + "="*70)
print("SUPPORT VECTOR MACHINE (RBF)")
print("="*70)

svm = SVC(kernel='rbf', gamma='scale', C=1.0)
svm.fit(X_train, y_train)

train_acc = svm.score(X_train, y_train)
test_acc = svm.score(X_test, y_test)

print(f"Train accuracy: {train_acc:.1%}")
print(f"Test accuracy:  {test_acc:.1%}")

y_pred_svm = svm.predict(X_test)
print("\nPer-phoneme performance:")
print(classification_report(y_test, y_pred_svm, zero_division=0))

# Quick comparison
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"LDA:  {lda.score(X_test, y_test):.1%}")
print(f"SVM:  {svm.score(X_test, y_test):.1%}")

stats, distances, phonemes = comprehensive_phoneme_analysis(pipeline)

# train
model_phonemes, eval_phonemes = train_phoneme_model(pipeline, use_balanced=True)

# 1. Check feature shapes and consistency
print("\n1. FEATURE SHAPES:")
train_shapes = [f.shape for f in pipeline.train['features'][:10]]
print(f"   First 10 train feature shapes: {train_shapes}")
unique_shapes = set([f.shape[1] for f in pipeline.train['features']])
print(f"   Unique channel counts: {unique_shapes}")
print(f"   Expected: {133} channels")

if len(unique_shapes) > 1:
    print("   ERROR: Features have inconsistent shapes!")
elif 133 not in unique_shapes:
    print(f"   ERROR: Features have {unique_shapes} channels, expected 133!")

# 2. Check for actual data (not all zeros)
print("\n2. FEATURE VALUES:")
sample_feature = pipeline.train['features'][0]
print(f"   Sample feature stats: min={sample_feature.min():.4f}, max={sample_feature.max():.4f}, mean={sample_feature.mean():.4f}")

zero_features = sum(1 for f in pipeline.train['features'] if np.allclose(f, 0))
print(f"   Zero features: {zero_features}/{len(pipeline.train['features'])}")

if zero_features > len(pipeline.train['features']) * 0.1:
    print(f"   WARNING: {zero_features/len(pipeline.train['features'])*100:.1f}% features are all zeros!")

# 3. Check label distribution
print("\n3. LABEL DISTRIBUTION:")
train_counts = Counter(pipeline.train['phoneme_labels'])
print(f"   Unique labels: {len(train_counts)}")
print(f"   Unknown '?': {train_counts.get('?', 0)} ({train_counts.get('?', 0)/len(pipeline.train['phoneme_labels'])*100:.1f}%)")

if train_counts.get('?', 0) > len(pipeline.train['phoneme_labels']) * 0.3:
    print("   WARNING: >30% of labels are unknown!")

# 4. Check train-test alignment
print("\n4. TRAIN-TEST CONSISTENCY:")
print(f"   Train samples: {len(pipeline.train['features'])}")
print(f"   Train labels: {len(pipeline.train['phoneme_labels'])}")
print(f"   Test samples: {len(pipeline.test['features'])}")
print(f"   Test labels: {len(pipeline.test['phoneme_labels'])}")

if len(pipeline.train['features']) != len(pipeline.train['phoneme_labels']):
    print("   ERROR: Train features and labels misaligned!")

# 5. Visualize sample features
print("\n5. VISUALIZING SAMPLE FEATURES:")
fig, axes = plt.subplots(2, 3, figsize=(15, 8))

for i in range(6):
    ax = axes[i//3, i%3]
    feat = pipeline.train['features'][i]
    label = pipeline.train['phoneme_labels'][i]
    word = pipeline.train.get('phoneme_words', ['?'])[i]
    
    ax.imshow(feat.T, aspect='auto', cmap='viridis', origin='lower')
    ax.set_title(f"'{label}' in '{word}'\n{feat.shape}")
    ax.set_xlabel('Time frames')
    ax.set_ylabel('Channels')

plt.tight_layout()
plt.show()

# Use it
conf_matrix = plot_detailed_confusion_matrix(
    model_phonemes, 
    pipeline.test['features'], 
    pipeline.test['phoneme_labels'],
    save_path='./results/dutch30/phoneme_confusion_matrix.png'
)

predictions, _ = model_phonemes.predict(pipeline.test['features'], use_viterbi=True)

# Now create the confusion matrix analysis
print("\nMost confused phoneme pairs:")
conf_copy = conf_matrix.copy()
np.fill_diagonal(conf_copy, 0)

# Get the correct label list used for the confusion matrix
unique_labels = sorted(set(pipeline.test['phoneme_labels'] + predictions))

for _ in range(5):
    i, j = np.unravel_index(np.argmax(conf_copy), conf_copy.shape)
    if conf_copy[i, j] > 0 and i < len(unique_labels) and j < len(unique_labels):
        true_phoneme = unique_labels[i]
        pred_phoneme = unique_labels[j]
        print(f"  {true_phoneme} → {pred_phoneme}: {int(conf_copy[i, j])} times")
        conf_copy[i, j] = 0

def analyze_phoneme_confusions_with_context(model, features, true_labels, words, top_n=10):
    """Analyze confused phonemes with their word context"""
    
    # Get predictions
    predictions, _ = model.predict(features, use_viterbi=True)
    
    # Collect confusion instances with context
    confusions = []
    for i, (true, pred, word) in enumerate(zip(true_labels, predictions, words)):
        if true != pred and true != '?' and pred != '?':
            confusions.append({
                'true': true,
                'pred': pred,
                'word': word,
                'index': i
            })
    
    # Count confusion patterns
    confusion_counts = {}
    confusion_examples = {}
    
    for conf in confusions:
        pair = (conf['true'], conf['pred'])
        if pair not in confusion_counts:
            confusion_counts[pair] = 0
            confusion_examples[pair] = []
        confusion_counts[pair] += 1
        if len(confusion_examples[pair]) < 3:  # Keep up to 3 examples
            confusion_examples[pair].append(conf['word'])
    
    # Sort by frequency
    sorted_confusions = sorted(confusion_counts.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\nTop {min(top_n, len(sorted_confusions))} phoneme confusions with word context:")
    print("-" * 70)
    
    for (true_ph, pred_ph), count in sorted_confusions[:top_n]:
        example_words = confusion_examples[(true_ph, pred_ph)]
        unique_words = list(set(example_words))
        print(f"{true_ph:5s} → {pred_ph:5s}: {count:3d} times")
        print(f"       Words: {', '.join(unique_words[:5])}")
    
    return confusions, confusion_counts

# Check if model is using groups when it should use phonemes
def diagnose_model_labels(model, pipeline_data):
    """Diagnose what labels the model is actually using"""
    
    train_labels = pipeline_data['phoneme_labels'][:50]  # Sample
    
    # Check label types
    unique_labels = set(train_labels)
    print("Label diagnostic:")
    print(f"  Unique labels in data: {unique_labels}")
    print(f"  Number of unique labels: {len(unique_labels)}")
    
    # Check if they're phonemes or groups
    phoneme_groups = model.phonetic_dict.phoneme_groups if hasattr(model.phonetic_dict, 'phoneme_groups') else {}
    group_names = set(phoneme_groups.keys())
    
    if unique_labels.issubset(group_names):
        print("  WARNING: Labels are GROUPS, not phonemes!")
        return 'groups'
    else:
        # Check if they're actual phonemes
        all_phonemes = set()
        for group_phonemes in phoneme_groups.values():
            all_phonemes.update(group_phonemes)
        
        overlap = unique_labels.intersection(all_phonemes)
        print(f"  Labels matching known phonemes: {len(overlap)}/{len(unique_labels)}")
        
        if len(overlap) > len(unique_labels) * 0.5:
            print("  Labels appear to be PHONEMES (correct)")
            return 'phonemes'
        else:
            print("  WARNING: Labels don't match expected phonemes")
            unknown = unique_labels - all_phonemes
            print(f"  Unknown labels: {list(unknown)[:10]}")
            return 'mixed'

# Run diagnostics
label_type = diagnose_model_labels(model_phonemes, pipeline.train)

# Analyze confusions with word context
if hasattr(pipeline, 'test') and 'phoneme_words' in pipeline.test:
    confusions, confusion_counts = analyze_phoneme_confusions_with_context(
        model_phonemes,
        pipeline.test['features'],
        pipeline.test['phoneme_labels'],
        pipeline.test['phoneme_words'],
        top_n=10
    )

def analyze_phoneme_data_quality(pipeline, max_samples_per_phoneme=100):
   # Get data
    features = pipeline.train['features']
    labels = pipeline.train['phoneme_labels']
    words = pipeline.train.get('phoneme_words', ['unknown'] * len(labels))
    
    # Group by phoneme
    phoneme_data = defaultdict(list)
    phoneme_words = defaultdict(set)
    
    for feat, label, word in zip(features, labels, words):
        phoneme_data[label].append(feat)
        phoneme_words[label].add(word)
    
    # Calculate statistics for each phoneme
    phoneme_stats = {}
    
    for phoneme, feats in phoneme_data.items():
        if len(feats) == 0:
            continue
        
        # Limit samples for computational efficiency
        if len(feats) > max_samples_per_phoneme:
            feats = feats[:max_samples_per_phoneme]
        
        # Flatten features for distance calculation
        flat_feats = [f.flatten() for f in feats]
        
        # Calculate within-phoneme distances (similarity)
        within_distances = []
        if len(flat_feats) > 1:
            for i in range(min(len(flat_feats), 50)):  # Sample to avoid too many comparisons
                for j in range(i+1, min(len(flat_feats), 50)):
                    try:
                        dist = cosine(flat_feats[i], flat_feats[j])
                        if not np.isnan(dist):
                            within_distances.append(dist)
                    except:
                        pass
        
        phoneme_stats[phoneme] = {
            'count': len(phoneme_data[phoneme]),
            'words': phoneme_words[phoneme],
            'within_mean': np.mean(within_distances) if within_distances else 0,
            'within_std': np.std(within_distances) if within_distances else 0,
            'feature_mean': np.mean([np.mean(f) for f in feats]),
            'feature_std': np.std([np.mean(f) for f in feats])
        }
    
    # Sort by count
    sorted_phonemes = sorted(phoneme_stats.items(), key=lambda x: x[1]['count'], reverse=True)
    
    # Print summary
    print(f"\nTotal phonemes: {len(phoneme_stats)}")
    print(f"Total samples: {len(labels)}")
    print(f"Average samples per phoneme: {len(labels) / len(phoneme_stats):.1f}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Sample count per phoneme
    phonemes = [p for p, _ in sorted_phonemes]
    counts = [s['count'] for _, s in sorted_phonemes]
    
    axes[0, 0].bar(range(len(phonemes)), counts, color='steelblue')
    axes[0, 0].set_xticks(range(len(phonemes)))
    axes[0, 0].set_xticklabels(phonemes, rotation=90, fontsize=8)
    axes[0, 0].set_ylabel('Number of Samples', fontsize=12)
    axes[0, 0].set_title('Sample Count per Phoneme', fontsize=14, fontweight='bold')
    axes[0, 0].axhline(y=np.median(counts), color='red', linestyle='--', 
                       label=f'Median: {np.median(counts):.0f}')
    axes[0, 0].axhline(y=50, color='orange', linestyle=':', 
                       label='Min recommended: 50')
    axes[0, 0].legend()
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # 2. Within-phoneme similarity (lower = more similar)
    within_means = [s['within_mean'] for _, s in sorted_phonemes]
    
    colors = ['green' if w < 0.3 else 'orange' if w < 0.5 else 'red' 
              for w in within_means]
    
    axes[0, 1].bar(range(len(phonemes)), within_means, color=colors)
    axes[0, 1].set_xticks(range(len(phonemes)))
    axes[0, 1].set_xticklabels(phonemes, rotation=90, fontsize=8)
    axes[0, 1].set_ylabel('Average Cosine Distance', fontsize=12)
    axes[0, 1].set_title('Within-Phoneme Variance\n(Lower = More Consistent)', 
                         fontsize=14, fontweight='bold')
    axes[0, 1].axhline(y=0.3, color='green', linestyle='--', alpha=0.5, label='Good (<0.3)')
    axes[0, 1].axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='OK (<0.5)')
    axes[0, 1].legend()
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # 3. Quality scatter: Count vs Variance
    axes[1, 0].scatter(counts, within_means, alpha=0.6, s=50)
    axes[1, 0].set_xlabel('Number of Samples', fontsize=12)
    axes[1, 0].set_ylabel('Within-Phoneme Distance', fontsize=12)
    axes[1, 0].set_title('Data Quality: Count vs Consistency', fontsize=14, fontweight='bold')
    
    # Add quadrant lines
    axes[1, 0].axhline(y=0.4, color='gray', linestyle='--', alpha=0.3)
    axes[1, 0].axvline(x=50, color='gray', linestyle='--', alpha=0.3)
    
    # Label quadrants
    axes[1, 0].text(max(counts)*0.7, 0.6, 'Enough data\nBut inconsistent', 
                   ha='center', fontsize=9, color='red')
    axes[1, 0].text(max(counts)*0.7, 0.2, 'Good quality\nEnough data', 
                   ha='center', fontsize=9, color='green')
    axes[1, 0].text(25, 0.6, 'Not enough data\nAnd inconsistent', 
                   ha='center', fontsize=9, color='darkred')
    axes[1, 0].text(25, 0.2, 'Consistent\nBut need more data', 
                   ha='center', fontsize=9, color='orange')
    
    # Annotate problem phonemes
    for i, (phoneme, stats) in enumerate(sorted_phonemes):
        count = stats['count']
        variance = stats['within_mean']
        
        # Label if problematic
        if count < 20 or variance > 0.6:
            axes[1, 0].annotate(phoneme, (count, variance), 
                              fontsize=8, alpha=0.7)
    
    axes[1, 0].grid(alpha=0.3)
    
    # 4. Detailed table for worst phonemes
    axes[1, 1].axis('off')
    
    # Find problematic phonemes
    problems = []
    for phoneme, stats in sorted_phonemes:
        issues = []
        if stats['count'] < 20:
            issues.append(f"Low count ({stats['count']})")
        if stats['within_mean'] > 0.5:
            issues.append(f"High variance ({stats['within_mean']:.2f})")
        if len(stats['words']) == 1:
            issues.append("Only 1 word")
        
        if issues:
            problems.append((phoneme, stats, issues))
    
    # Create table
    table_data = [['Phoneme', 'Count', 'Variance', 'Words', 'Issues']]
    
    for phoneme, stats, issues in problems[:15]:  # Top 15 problems
        word_count = len(stats['words'])
        table_data.append([
            phoneme,
            str(stats['count']),
            f"{stats['within_mean']:.2f}",
            str(word_count),
            '\n'.join(issues[:2])
        ])
    
    table = axes[1, 1].table(cellText=table_data, loc='center', cellLoc='left')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 2)
    
    # Color header
    for i in range(5):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    axes[1, 1].set_title('Top Problem Phonemes', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('./phoneme_quality_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print detailed statistics
    print("\n" + "="*70)
    print("PROBLEM ANALYSIS")
    print("="*70)
    
    print(f"\nPhonemes with < 20 samples: {sum(1 for _, s in sorted_phonemes if s['count'] < 20)}")
    print(f"Phonemes with < 50 samples: {sum(1 for _, s in sorted_phonemes if s['count'] < 50)}")
    print(f"Phonemes with high variance (>0.5): {sum(1 for _, s in sorted_phonemes if s['within_mean'] > 0.5)}")
    print(f"Phonemes from only 1 word: {sum(1 for _, s in sorted_phonemes if len(s['words']) == 1)}")
    
    print("\nWorst 5 phonemes by sample count:")
    for phoneme, stats in sorted_phonemes[-5:]:
        words_list = list(stats['words'])[:3]
        print(f"  {phoneme:5s}: {stats['count']:3d} samples, variance={stats['within_mean']:.3f}, "
              f"words={words_list}")
    
    print("\nWorst 5 phonemes by variance:")
    by_variance = sorted(sorted_phonemes, key=lambda x: x[1]['within_mean'], reverse=True)
    for phoneme, stats in by_variance[:5]:
        words_list = list(stats['words'])[:3]
        print(f"  {phoneme:5s}: variance={stats['within_mean']:.3f}, {stats['count']:3d} samples, "
              f"words={words_list}")
    
    return phoneme_stats


def analyze_between_phoneme_separability(pipeline, sample_size=500):
    """
    Analyze how separable different phonemes are in feature space.
    """
    print("\n" + "="*70)
    print("BETWEEN-PHONEME SEPARABILITY ANALYSIS")
    print("="*70)
    
    features = pipeline.train['features']
    labels = pipeline.train['phoneme_labels']
    
    # Sample if needed
    if len(features) > sample_size:
        indices = np.random.choice(len(features), sample_size, replace=False)
        features = [features[i] for i in indices]
        labels = [labels[i] for i in indices]
    
    # CRITICAL FIX: Average across time dimension to get fixed-length representation
    # Each feature goes from (time, 133) → (133,)
    flat_features = np.array([np.mean(f, axis=0) for f in features])  # ← Changed this line!
    
    print(f"Feature shape after averaging: {flat_features.shape}")
    
    # Get unique phonemes
    unique_phonemes = sorted(set(labels))
    
    print(f"Analyzing {len(unique_phonemes)} phonemes with {len(features)} samples")
    
    # Calculate centroid for each phoneme
    centroids = {}
    for phoneme in unique_phonemes:
        phoneme_features = flat_features[[i for i, l in enumerate(labels) if l == phoneme]]
        if len(phoneme_features) > 0:
            centroids[phoneme] = np.mean(phoneme_features, axis=0)
    
    # Calculate pairwise distances between centroids
    n_phonemes = len(centroids)
    distance_matrix = np.zeros((n_phonemes, n_phonemes))
    
    phoneme_list = list(centroids.keys())
    for i, p1 in enumerate(phoneme_list):
        for j, p2 in enumerate(phoneme_list):
            if i != j:
                distance_matrix[i, j] = np.linalg.norm(centroids[p1] - centroids[p2])
    
    # Analyze results
    non_diag = distance_matrix[np.triu_indices(n_phonemes, k=1)]
    
    print(f"\nCentroid Distances:")
    print(f"  Mean:   {np.mean(non_diag):.3f}")
    print(f"  Median: {np.median(non_diag):.3f}")
    print(f"  Std:    {np.std(non_diag):.3f}")
    print(f"  Min:    {np.min(non_diag):.3f}")
    print(f"  Max:    {np.max(non_diag):.3f}")
    
    # Find most/least separable pairs
    flat_indices = np.argsort(non_diag)
    
    print(f"\nMost Similar Phonemes:")
    for idx in flat_indices[:5]:
        i, j = np.triu_indices(n_phonemes, k=1)
        p1, p2 = phoneme_list[i[idx]], phoneme_list[j[idx]]
        print(f"  {p1} - {p2}: {distance_matrix[i[idx], j[idx]]:.3f}")
    
    print(f"\nMost Different Phonemes:")
    for idx in flat_indices[-5:]:
        i, j = np.triu_indices(n_phonemes, k=1)
        p1, p2 = phoneme_list[i[idx]], phoneme_list[j[idx]]
        print(f"  {p1} - {p2}: {distance_matrix[i[idx], j[idx]]:.3f}")
    
    return distance_matrix, phoneme_list


def comprehensive_phoneme_analysis(pipeline):
    """
    Run all analyses
    """
    print("\n" + "="*70)
    print("COMPREHENSIVE PHONEME DATA ANALYSIS")
    print("="*70)
    
    # 1. Quality analysis
    phoneme_stats = analyze_phoneme_data_quality(pipeline)
    
    # 2. Separability analysis
    distance_matrix, phoneme_list = analyze_between_phoneme_separability(pipeline)
    
    # 3. Overall diagnosis
    print("\n" + "="*70)
    print("OVERALL DIAGNOSIS")
    print("="*70)
    
    # Count issues
    low_count = sum(1 for stats in phoneme_stats.values() if stats['count'] < 50)
    high_variance = sum(1 for stats in phoneme_stats.values() if stats['within_mean'] > 0.5)
    
    upper_triangle = distance_matrix[np.triu_indices(len(phoneme_list), k=1)]
    avg_separability = np.mean(upper_triangle)
    
    print(f"\nData Issues:")
    print(f"  Phonemes with < 50 samples: {low_count}/{len(phoneme_stats)} ({low_count/len(phoneme_stats)*100:.1f}%)")
    print(f"  Phonemes with high variance: {high_variance}/{len(phoneme_stats)} ({high_variance/len(phoneme_stats)*100:.1f}%)")
    print(f"  Average phoneme separability: {avg_separability:.3f}")
    
    print("\nLikely causes of poor accuracy:")
    if low_count > len(phoneme_stats) * 0.5:
        print("  - CRITICAL: Over 50% of phonemes have insufficient training data")
    if high_variance > len(phoneme_stats) * 0.3:
        print("  - PROBLEM: Many phonemes have inconsistent features")
    if avg_separability < 0.3:
        print("  - PROBLEM: Phonemes are not well-separated in feature space")
    
    print("\nRecommendations:")
    if low_count > len(phoneme_stats) * 0.5:
        print("  1. Collect more data OR reduce number of phoneme classes")
        print("  2. Consider grouping similar phonemes together")
    if high_variance > len(phoneme_stats) * 0.3:
        print("  1. Check if boundary detection is working correctly")
        print("  2. Verify feature extraction is consistent")
    if avg_separability < 0.3:
        print("  1. Try different feature extraction methods")
        print("  2. Consider using phoneme groups instead of individual phonemes")
    
    return phoneme_stats, distance_matrix, phoneme_list

def diagnose_high_variance(pipeline, phoneme='k', n_samples=10):
    # Get all samples for this phoneme
    features = pipeline.train['features']
    labels = pipeline.train['phoneme_labels']
    words = pipeline.train.get('phoneme_words', ['unknown'] * len(labels))
    
    # Find indices for this phoneme
    indices = [i for i, label in enumerate(labels) if label == phoneme]
    
    if not indices:
        print(f"No samples found for phoneme '{phoneme}'")
        return
    
    print(f"\n{'='*70}")
    print(f"ANALYZING PHONEME: '{phoneme}'")
    print(f"{'='*70}")
    print(f"Found {len(indices)} total samples for '{phoneme}'")
    
    # Sample a few for detailed analysis
    sample_indices = indices[:min(n_samples, len(indices))]
    
    # Print all words being analyzed
    print(f"\nWords being analyzed ({len(sample_indices)} samples):")
    analyzed_words = [words[idx] for idx in sample_indices]
    for i, word in enumerate(analyzed_words):
        print(f"  Sample {i}: '{word}'")
    
    # Collect info
    sample_info = []
    for idx in sample_indices:
        feat = features[idx]
        word = words[idx]
        
        sample_info.append({
            'index': idx,
            'word': word,
            'feature': feat,
            'shape': feat.shape,
            'n_frames': feat.shape[0],
            'duration_seconds': feat.shape[0] * 0.01,  # assuming 0.01s frameshift
            'mean': np.mean(feat),
            'std': np.std(feat),
            'min': np.min(feat),
            'max': np.max(feat)
        })
    
    # Analyze frame length consistency
    frame_lengths = [info['n_frames'] for info in sample_info]
    print(f"\n{'='*70}")
    print(f"FRAME LENGTH CONSISTENCY FOR PHONEME '{phoneme}'")
    print(f"{'='*70}")
    print(f"Frame lengths: {frame_lengths}")
    print(f"  Min frames: {min(frame_lengths)}")
    print(f"  Max frames: {max(frame_lengths)}")
    print(f"  Mean frames: {np.mean(frame_lengths):.1f}")
    print(f"  Std frames: {np.std(frame_lengths):.1f}")
    print(f"  Range: {max(frame_lengths) - min(frame_lengths)} frames")
    
    # Calculate coefficient of variation for length consistency
    cv = (np.std(frame_lengths) / np.mean(frame_lengths)) * 100
    print(f"  Coefficient of Variation: {cv:.1f}%")
    
    if cv > 50:
        print(f"  WARNING: High variation in length (>50%)!")
        print(f"           Same phoneme segments vary wildly in duration")
    elif cv > 30:
        print(f"  CAUTION: Moderate variation in length (30-50%)")
    else:
        print(f"  OK: Length is reasonably consistent (<30% variation)")
    
    # Print comparison
    print(f"\nDetailed comparison of {len(sample_info)} samples:")
    print("-"*70)
    
    for i, info in enumerate(sample_info):
        print(f"\nSample {i} (index {info['index']}):")
        print(f"  Word: '{info['word']}'")
        print(f"  Shape: {info['shape']}")
        print(f"  Length: {info['n_frames']} frames ({info['duration_seconds']:.3f}s)")
        print(f"  Mean: {info['mean']:.4f}, Std: {info['std']:.4f}")
        print(f"  Range: [{info['min']:.4f}, {info['max']:.4f}]")
    
    # Check if shapes are consistent
    shapes = [info['shape'] for info in sample_info]
    if len(set(shapes)) > 1:
        print(f"\nWARNING: Inconsistent shapes detected!")
        print(f"  Unique shapes: {set(shapes)}")
        print("  This suggests segments have different lengths")
    
    # Visualize the features
    n_plot = min(n_samples, 5)
    fig, axes = plt.subplots(n_plot, 2, figsize=(14, 3*n_plot))
    
    # Add main title with phoneme
    fig.suptitle(f"Feature Analysis for Phoneme '{phoneme}' (CV={cv:.1f}%)", 
                 fontsize=16, fontweight='bold', y=0.995)
    
    if n_plot == 1:
        axes = np.array([[axes[0], axes[1]]])
    
    for i, info in enumerate(sample_info[:5]):
        feat = info['feature']
        
        # Plot 1: Feature as heatmap
        axes[i, 0].imshow(feat.T, aspect='auto', cmap='viridis')
        axes[i, 0].set_title(f"Sample {i}: '{phoneme}' in '{info['word']}' "
                            f"({info['n_frames']} frames, {info['duration_seconds']:.3f}s)",
                            fontweight='bold')
        axes[i, 0].set_ylabel('Feature Dim')
        axes[i, 0].set_xlabel('Time Frames')
        
        # Plot 2: Feature statistics over time
        feat_mean = np.mean(feat, axis=1)
        feat_std = np.std(feat, axis=1)
        
        axes[i, 1].plot(feat_mean, label='Mean', color='blue', linewidth=2)
        axes[i, 1].fill_between(range(len(feat_mean)), 
                                feat_mean - feat_std, 
                                feat_mean + feat_std, 
                                alpha=0.3, color='blue')
        axes[i, 1].set_title(f"Evolution: '{phoneme}' in '{info['word']}' "
                           f"({info['n_frames']} frames)",
                           fontweight='bold')
        axes[i, 1].set_ylabel('Feature Value')
        axes[i, 1].set_xlabel('Time Frames')
        axes[i, 1].legend()
        axes[i, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'./variance_diagnosis_phoneme_{phoneme}.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Calculate pairwise distances
    print("\n" + "="*70)
    print(f"PAIRWISE DISTANCES FOR PHONEME '{phoneme}'")
    print("(should be < 0.5 for same phoneme)")
    print("="*70)
    
    from scipy.spatial.distance import cosine
    
    for i in range(len(sample_info)):
        for j in range(i+1, len(sample_info)):
            feat_i = sample_info[i]['feature'].flatten()
            feat_j = sample_info[j]['feature'].flatten()
            
            # Handle different lengths
            min_len = min(len(feat_i), len(feat_j))
            dist = cosine(feat_i[:min_len], feat_j[:min_len])
            
            word_i = sample_info[i]['word']
            word_j = sample_info[j]['word']
            len_i = sample_info[i]['n_frames']
            len_j = sample_info[j]['n_frames']
            
            status = "OK" if dist < 0.5 else "PROBLEM"
            
            print(f"  Sample {i} ('{phoneme}' in '{word_i}', {len_i}f) <-> "
                  f"Sample {j} ('{phoneme}' in '{word_j}', {len_j}f): "
                  f"{dist:.3f} [{status}]")
    
    return sample_info


def check_boundary_detection_quality(pipeline, n_words=5):
    print("\n" + "="*70)
    print("CHECKING BOUNDARY DETECTION QUALITY")
    print("="*70)
    
    # Get raw data (before phoneme segmentation)
    if not hasattr(pipeline, 'train_data'):
        print("No train_data attribute - can't check raw segments")
        return
    
    words = pipeline.train_data.get('phoneme_words', [])
    features = pipeline.train_data.get('features', [])
    
    # Find instances of the same word
    word_instances = defaultdict(list)
    for i, word in enumerate(words):
        word_instances[word].append(i)
    
    # Check words with multiple instances
    print(f"\nChecking {n_words} words with multiple instances:")
    
    checked = 0
    for word, indices in word_instances.items():
        if len(indices) < 2:
            continue
        
        if checked >= n_words:
            break
        
        print(f"\n  Word: '{word}' ({len(indices)} instances)")
        
        # Get features for each instance
        shapes = [features[i].shape for i in indices[:5]]
        print(f"    Shapes: {shapes}")
        
        if len(set(shapes)) > 1:
            print(f"    WARNING: Different shapes for same word!")
        else:
            print(f"    OK: All instances have same shape")
        
        checked += 1


def check_feature_extraction_consistency(pipeline):
    print("\n" + "="*70)
    print("CHECKING FEATURE EXTRACTION CONSISTENCY")
    print("="*70)
    
    features = pipeline.train['features']
    
    # Check shapes
    shapes = [f.shape for f in features[:100]]
    unique_shapes = set(shapes)
    
    print(f"\nChecked first 100 features:")
    print(f"  Unique shapes: {len(unique_shapes)}")
    
    if len(unique_shapes) > 10:
        print(f"  WARNING: Too many different shapes!")
        print(f"  Most common shapes: {sorted(unique_shapes)[:10]}")
    else:
        print(f"  All shapes: {unique_shapes}")
    
    # Check feature dimensions (second dimension should be consistent)
    if features and len(features[0].shape) > 1:
        feature_dims = [f.shape[1] for f in features[:100]]
        unique_dims = set(feature_dims)
        
        print(f"\n  Feature dimensions: {unique_dims}")
        
        if len(unique_dims) > 1:
            print(f"  WARNING: Inconsistent feature dimensions!")
            from collections import Counter
            dim_counts = Counter(feature_dims)
            print(f"  Distribution: {dict(dim_counts)}")
    
    # Check if any features are all zeros or NaN
    print(f"\nChecking for problematic features:")
    
    n_zeros = 0
    n_nans = 0
    n_constant = 0
    
    for feat in features[:100]:
        if np.all(feat == 0):
            n_zeros += 1
        if np.any(np.isnan(feat)):
            n_nans += 1
        if np.std(feat) < 1e-6:
            n_constant += 1
    
    print(f"  All-zero features: {n_zeros}/100")
    print(f"  Features with NaN: {n_nans}/100")
    print(f"  Constant features: {n_constant}/100")
    
    if n_zeros > 0 or n_nans > 0 or n_constant > 5:
        print(f"  WARNING: Found problematic features!")


def comprehensive_variance_diagnosis(pipeline):
    """
    Run all variance diagnostics
    """
    
    print("\n" + "="*70)
    print("COMPREHENSIVE VARIANCE DIAGNOSIS")
    print("="*70)
    
    # 1. Check feature extraction consistency
    check_feature_extraction_consistency(pipeline)
    
    # 2. Check boundary detection quality
    check_boundary_detection_quality(pipeline)
    
    # 3. Deep dive into a specific phoneme
    # Pick a common phoneme with high variance
    from collections import Counter
    label_counts = Counter(pipeline.train['phoneme_labels'])
    common_phoneme = label_counts.most_common(1)[0][0]
    
    print(f"\n\nDeep diving into most common phoneme: '{common_phoneme}'")
    diagnose_high_variance(pipeline, phoneme=common_phoneme, n_samples=5)
    
    # 4. Summary and recommendations
    print("\n" + "="*70)
    print("DIAGNOSIS SUMMARY")
    print("="*70)
    
    print("\nThe high variance (0.95+) suggests ONE of these issues:")
    print("  1. Boundary detection is finding random boundaries")
    print("     - Segments for same phoneme have different acoustic content")
    print("  2. Feature extraction is inconsistent")
    print("     - Same input produces different features")
    print("  3. Phoneme labels are wrong/misaligned")
    print("     - Different phonemes labeled as same")
    print("  4. Dutch30 data doesn't segment by phoneme properly")
    print("     - Features are averaged over entire words, not phonemes")
    
    print("\nNext steps:")
    print("  1. Check the visualizations above - do features look similar?")
    print("  2. If shapes vary wildly, boundary detection is broken")
    print("  3. If features look random, feature extraction is broken")
    print("  4. If features are consistent but labeled wrongly, labeling is broken")


comprehensive_variance_diagnosis(pipeline)

def visualize_length_variance_relationship(pipeline):
    """Show how length affects feature variance"""
    
    features = pipeline.train['features']
    labels = pipeline.train['phoneme_labels']
    
    # Collect length and variance for each sample
    lengths = []
    variances = []
    phoneme_list = []
    
    for i, feat in enumerate(features):
        n_frames = feat.shape[0]
        feat_var = np.var(feat)  # variance of features
        
        lengths.append(n_frames)
        variances.append(feat_var)
        phoneme_list.append(labels[i])
    
    # Plot
    plt.figure(figsize=(12, 5))
    
    # Subplot 1: Length vs Variance scatter
    plt.subplot(1, 2, 1)
    plt.scatter(lengths, variances, alpha=0.3, s=10)
    plt.xlabel('Number of Frames (Length)')
    plt.ylabel('Feature Variance')
    plt.title('Length vs Variance\n(Should be random if pooling is good)')
    plt.grid(alpha=0.3)
    
    # Add trend line
    z = np.polyfit(lengths, variances, 1)
    p = np.poly1d(z)
    plt.plot(sorted(lengths), p(sorted(lengths)), "r--", 
             label=f'Trend (slope={z[0]:.2f})')
    plt.legend()
    
    # Subplot 2: Length distribution
    plt.subplot(1, 2, 2)
    plt.hist(lengths, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Number of Frames')
    plt.ylabel('Count')
    plt.title('Distribution of Phoneme Lengths')
    plt.axvline(np.median(lengths), color='r', linestyle='--', 
                label=f'Median={np.median(lengths):.1f}')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./length_variance_relationship.png', dpi=150)
    plt.show()
    
    # Statistics
    print("\n" + "="*70)
    print("LENGTH-VARIANCE RELATIONSHIP")
    print("="*70)
    print(f"Length range: {min(lengths)} - {max(lengths)} frames")
    print(f"Length CV: {(np.std(lengths)/np.mean(lengths)*100):.1f}%")
    print(f"Correlation (length vs variance): {np.corrcoef(lengths, variances)[0,1]:.3f}")
    print("\nIf correlation > 0.3: Length is affecting variance!")
    print("If correlation ~ 0: Good - pooling strategy works")

# Run it
visualize_length_variance_relationship(pipeline)
