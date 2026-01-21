# Converted from parse_features_of_30_patients.ipynb

import os
import gc
import glob
import json
import h5py

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

# Create config
config = Dutch30Config()

# Pass config to both extractor and pipeline
extractor = Dutch30FeatureExtractor(config=config)

pipeline = Dutch30Pipeline(
    dutch30_extractor=extractor,
    config=config, 
    debug_mode=True,
    pca_components=100,
    feature_extraction_method='high_gamma',
    use_rms_boundaries=True,   
    use_multifeature=False
)

# # Debug a specific patient
# pipeline.debug_sentence_parsing('sub-p21', max_samples=3)
# print([attr for attr in dir(pipeline_debug) if 'detect' in attr.lower()])

pipeline.step1_load_dutch30_data(num_patients = 10)
#pipeline.step2_3_use_existing_split()
pipeline.step2_split_by_instances()

pid = 'P01'
word_segments = pipeline.split_result['word_segments_dict'][pid]
word = list(word_segments['words'].keys())[0]
instance = word_segments['words'][word]['instances'][0]

print(f"Audio available: {'audio_segment' in instance}")
print(f"Audio shape: {instance['audio_segment'].shape if 'audio_segment' in instance else 'N/A'}")

diag = Dutch30PhonemeDetectionDiagnostic(pipeline)
diag.visualize_word_analysis('P01', word_name = 'vogelkooitje', save_path='p21_word5.png')

#diag.visualize_multifeature_analysis('P01', word_index=50)
diag.visualize_rms_boundaries('P01',  word_name = 'vogelkooitje')

# See how many instances exist
pid = 'P01'
word = 'zevenduizend'
word_data = pipeline.split_result['word_segments_dict'][pid]['words'][word]
print(f"'{word}' has {len(word_data['instances'])} instances")

# What settings does training use?
print(f"use_multifeature: {pipeline.detector.use_multifeature}")
print(f"use_rms_boundaries: {pipeline.detector.use_rms_boundaries}")

# Diagnostic hardcodes use_multifeature=True

# # Quick check first 10 words
# diag.batch_diagnostic('sub-p21', num_samples=5)

config = Dutch30Config()
extractor = Dutch30FeatureExtractor(config=config)

pipeline = Dutch30Pipeline(
    dutch30_extractor=extractor,
    config=config, 
    debug_mode=False,
    pca_components=100,
    feature_extraction_method = 'high_gamma', # 'band_powers', # 
    use_rms_boundaries=True,   
    use_multifeature=False    
)

# step_0
# pipeline.analyze_dutch30_channels()

#pipeline.step1_load_dutch30_data(num_patients = 20)
pipeline.step1_load_dutch30_data(patient_range=(26,30))

pipeline.split_result = None
pipeline.step2_split_by_instances();
#pipeline.step2_3_use_existing_split()

# # Check how many sentences you actually have
# for pid in ['P20', 'P21', 'P22', 'P23']:
#     word_segments = pipeline.split_result['word_segments_dict'][pid]
#     print(f"{pid}:")
#     print(f"  Total word instances: {len(word_segments['words_list'])}")
#     print(f"  Unique words: {len(word_segments['words'])}")
    
#     # Check if sentences are being split properly
#     sentences = [w for w in word_segments['words_list'] if len(w.split()) > 3]
#     print(f"  Multi-word entries: {len(sentences)}")  # Should be 0!

pipeline.step4_custom_detector()
pipeline.step5_accumulate_data_dutch30()

pipeline.dutch30_step6_resolve_unknowns();

# After step4 or step5
print(pipeline.phonetic_dict.get_missing_words_summary())

# pipeline.run_step1_to_step6(sample_fraction=0.0001, force_reprocess=True)
#pipeline.run_step1_to_step6(sample_fraction=0.0001)
#pipeline.step1_load_dutch30_data(num_patients = 10)
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

pipeline.step7_filter_unknowns(unknown_keep_ratio=0.05);

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

# patient_stats, position_stats = analyze_phoneme_positions_and_patients(pipeline, long_threshold=0.4)

def train_per_patient(pipeline):
    """Train and evaluate separate model for each patient."""
    
    results = {}
    
    for pid in set(pipeline.train['phoneme_participant_ids']):
        # Filter data for this patient
        train_mask = [p == pid for p in pipeline.train['phoneme_participant_ids']]
        test_mask = [p == pid for p in pipeline.test['phoneme_participant_ids']]
        
        train_feat = [pipeline.train['features'][i] for i, m in enumerate(train_mask) if m]
        train_labels = [pipeline.train['phoneme_labels'][i] for i, m in enumerate(train_mask) if m]
        test_feat = [pipeline.test['features'][i] for i, m in enumerate(test_mask) if m]
        test_labels = [pipeline.test['phoneme_labels'][i] for i, m in enumerate(test_mask) if m]
        
        if len(train_feat) < 10 or len(test_feat) < 5:
            print(f"{pid}: Skipped (train={len(train_feat)}, test={len(test_feat)})")
            continue
        
        # Train Markov model
        model = MarkovPhonemeModel(
            phonetic_dict=pipeline.phonetic_dict,
            order=1,
            use_groups=False
        )
        
        model.train(features=train_feat, phoneme_labels=train_labels)
        eval_result = model.evaluate(features=test_feat, true_labels=test_labels, use_viterbi=True)
        
        results[pid] = {
            'model': model,
            'accuracy': eval_result['accuracy'],
            'train_size': len(train_feat),
            'test_size': len(test_feat)
        }
        
        print(f"{pid}: Acc={eval_result['accuracy']:.3f} (train={len(train_feat)}, test={len(test_feat)})")
    
    # Summary
    accs = [r['accuracy'] for r in results.values()]
    print(f"\nAverage: {np.mean(accs):.3f} ± {np.std(accs):.3f}")
    
    return results

def visualize_patient_model(pid, patient_results, pipeline):
    """Detailed analysis for one patient."""
    
    if pid not in patient_results:
        print(f"{pid} not found")
        return
    
    # Filter data
    train_mask = [p == pid for p in pipeline.train['phoneme_participant_ids']]
    test_mask = [p == pid for p in pipeline.test['phoneme_participant_ids']]
    
    train_labels = [pipeline.train['phoneme_labels'][i] for i, m in enumerate(train_mask) if m]
    train_feat = [pipeline.train['features'][i] for i, m in enumerate(train_mask) if m]
    test_feat = [pipeline.test['features'][i] for i, m in enumerate(test_mask) if m]
    test_labels = [pipeline.test['phoneme_labels'][i] for i, m in enumerate(test_mask) if m]
    
    # Get predictions
    model = patient_results[pid]['model']
    preds, _ = model.predict(test_feat, use_viterbi=True)
    
    # Setup figure - 3 plots in 1 row
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'{pid} - Accuracy: {patient_results[pid]["accuracy"]:.3f}', fontsize=14, fontweight='bold')
    
    # 1. Combined train/test distribution
    train_counts = Counter(train_labels)
    test_counts = Counter(test_labels)
    all_phonemes = sorted(set(list(train_counts.keys()) + list(test_counts.keys())))
    
    x = np.arange(len(all_phonemes))
    width = 0.35
    axes[0].bar(x - width/2, [train_counts.get(p, 0) for p in all_phonemes], width, label='Train', color='cornflowerblue')
    axes[0].bar(x + width/2, [test_counts.get(p, 0) for p in all_phonemes], width, label='Test', color='coral')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(all_phonemes, rotation=90)
    axes[0].set_title(f'Distribution (train={len(train_labels)}, test={len(test_labels)})')
    axes[0].set_ylabel('Count')
    axes[0].legend()
    
    # 2. Per-phoneme accuracy
    test_phonemes = sorted(test_counts.keys())
    phoneme_acc = {}
    for p in test_phonemes:
        mask = [l == p for l in test_labels]
        correct = sum(1 for i, m in enumerate(mask) if m and preds[i] == p)
        total = sum(mask)
        phoneme_acc[p] = correct / total if total > 0 else 0
    
    axes[1].bar(range(len(test_phonemes)), [phoneme_acc[p] for p in test_phonemes], color='green')
    axes[1].set_xticks(range(len(test_phonemes)))
    axes[1].set_xticklabels(test_phonemes, rotation=90)
    axes[1].set_title('Per-Phoneme Accuracy')
    axes[1].set_ylim([0, 1])
    axes[1].axhline(patient_results[pid]['accuracy'], color='red', linestyle='--', alpha=0.5)
    
    # 3. Confusion matrix
    from sklearn.metrics import confusion_matrix
    unique_labels = sorted(set(test_labels + preds))
    cm = confusion_matrix(test_labels, preds, labels=unique_labels)
    
    im = axes[2].imshow(cm, cmap='Blues')
    axes[2].set_xticks(range(len(unique_labels)))
    axes[2].set_yticks(range(len(unique_labels)))
    axes[2].set_xticklabels(unique_labels, rotation=90, fontsize=8)
    axes[2].set_yticklabels(unique_labels, fontsize=8)
    axes[2].set_xlabel('Predicted')
    axes[2].set_ylabel('True')
    axes[2].set_title('Confusion Matrix')
    plt.colorbar(im, ax=axes[2])
    
    plt.tight_layout()
    plt.show()
    
    # Print stats
    print(f"\n{'='*70}")
    print(f"{pid} - PER-PHONEME ACCURACY & DURATION")
    print(f"{'='*70}")
    print(f"{'Phoneme':<8} {'Acc':<6} {'Count':<6} {'Duration (ms)':<20}")
    print(f"{'':8} {'':6} {'':6} {'Min':>6} {'Mean':>6} {'Max':>6}")
    print('-'*70)
    
    frameshift = pipeline.config.frameshift * 1000
    
    for p in test_phonemes:
        phoneme_lengths = [test_feat[i].shape[0] for i, label in enumerate(test_labels) if label == p]
        durations_ms = [length * frameshift for length in phoneme_lengths]
        
        print(f"{p:<8} {phoneme_acc[p]:>5.2f} {test_counts[p]:>6} "
              f"{min(durations_ms):>6.0f} {np.mean(durations_ms):>6.0f} {max(durations_ms):>6.0f}")
    
    print(f"{'='*70}\n")

patient_results = train_per_patient(pipeline)

# Check actual sample counts after all fixes
print("="*70)
print("DATA SUMMARY")
print("="*70)

for pid in patient_results.keys():
    train_size = patient_results[pid]['train_size']
    test_size = patient_results[pid]['test_size']
    accuracy = patient_results[pid]['accuracy']
    
    patient_type = "SENTENCE" if int(pid[1:]) >= 20 else "WORD"
    
    print(f"{pid} ({patient_type}): {train_size} train, {test_size} test → Acc: {accuracy:.3f}")

for pid in sorted(patient_results.keys()):
    visualize_patient_model(pid, patient_results, pipeline)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

def train_simple_classifiers(pipeline):
    """Test multiple simple classifiers per patient."""
    
    classifiers = {
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'RandomForest_Simple': RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42),
        'RandomForest_Deep': RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42),
        'GaussianNB': GaussianNB(),
        'KNN': KNeighborsClassifier(n_neighbors=5)
    }
    
    results = {}
    
    for pid in set(pipeline.train['phoneme_participant_ids']):
        print(f"\n{pid}:")
        print("="*60)
        
        # Filter data
        train_mask = [p == pid for p in pipeline.train['phoneme_participant_ids']]
        test_mask = [p == pid for p in pipeline.test['phoneme_participant_ids']]
        
        train_feat = [pipeline.train['features'][i] for i, m in enumerate(train_mask) if m]
        train_labels = [pipeline.train['phoneme_labels'][i] for i, m in enumerate(train_mask) if m]
        test_feat = [pipeline.test['features'][i] for i, m in enumerate(test_mask) if m]
        test_labels = [pipeline.test['phoneme_labels'][i] for i, m in enumerate(test_mask) if m]
        
        if len(train_feat) < 10 or len(test_feat) < 5:
            print(f"Skipped (insufficient data)")
            continue
        
        # Flatten features
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
        
        # Check for NaN/Inf
        valid_train = ~(np.isnan(X_train).any(axis=1) | np.isinf(X_train).any(axis=1))
        valid_test = ~(np.isnan(X_test).any(axis=1) | np.isinf(X_test).any(axis=1))
        
        X_train = X_train[valid_train]
        y_train = [train_labels[i] for i in range(len(train_labels)) if valid_train[i]]
        X_test = X_test[valid_test]
        y_test = [test_labels[i] for i in range(len(test_labels)) if valid_test[i]]
        
        print(f"Samples: {len(X_train)} train, {len(X_test)} test")
        print(f"Features: {X_train.shape[1]}")
        print(f"Classes: {len(set(y_train))}")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Test each classifier
        results[pid] = {}
        
        for name, clf in classifiers.items():
            try:
                clf.fit(X_train_scaled, y_train)
                preds = clf.predict(X_test_scaled)
                acc = accuracy_score(y_test, preds)
                
                results[pid][name] = acc
                print(f"  {name:25s}: {acc:.3f}")
                
            except Exception as e:
                print(f"  {name:25s}: FAILED ({e})")
                results[pid][name] = 0.0
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY ACROSS ALL PATIENTS")
    print("="*70)
    
    for clf_name in classifiers.keys():
        accs = [results[pid][clf_name] for pid in results if clf_name in results[pid]]
        if accs:
            print(f"{clf_name:25s}: {np.mean(accs):.3f} ± {np.std(accs):.3f}")
    
    return results

simple_results = train_simple_classifiers(pipeline)

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

def visualize_sentence_segmentation(pipeline, participant_id, sentence_idx=0, play_audio=True):
    """Visualize sentence division into words with RMS-based phoneme boundaries"""
    from IPython.display import Audio, display
    from scipy.ndimage import gaussian_filter1d
    
    # Use existing segmented data
    if hasattr(pipeline, 'split_result') and participant_id in pipeline.split_result['word_segments_dict']:
        word_segments = pipeline.split_result['word_segments_dict'][participant_id]
    else:
        word_segments = pipeline.segment_data_by_words(participant_id)
    
    # Load raw data
    raw_data = pipeline.dutch30_extractor.load_patient_raw_data(participant_id)
    stimuli = raw_data['stimuli']
    eeg_sr = raw_data['eeg_sr']
    
    # Find sentences
    sentences = []
    current = None
    for i, stim in enumerate(stimuli):
        sent = stim.decode() if isinstance(stim, bytes) else str(stim)
        sent = sent.strip()
        if sent != current:
            if current: 
                sentences.append(current)
            current = sent
    if current:
        sentences.append(current)
    
    sentence = sentences[sentence_idx]
    print(f"Sentence: '{sentence}'")
    
    # Get words from this sentence
    import re
    cleaned = re.sub(r'["""„"''\r\n]+', '', sentence)
    sentence_words = [w for w in cleaned.split() if w]
    
    # Gather data
    word_specs = []
    word_durations = []
    phoneme_counts = []
    actual_words = []
    word_audios = []
    word_phoneme_boundaries = []
    word_expected_phonemes = []
    
    for word in sentence_words:
        if word in word_segments['words']:
            instance = word_segments['words'][word]['instances'][0]
            spec = instance['spectrogram_segment']
            audio_seg = instance.get('audio_segment', None)
            
            # Skip if too short
            if spec.shape[0] < 3 or audio_seg is None:
                continue
            
            word_specs.append(spec)
            word_durations.append(spec.shape[0] * pipeline.config.frameshift)
            word_audios.append(audio_seg)
            
            phonemes = pipeline.phonetic_dict.extract_phonemes(word)
            phoneme_counts.append(len(phonemes) if phonemes else 3)
            word_expected_phonemes.append(phonemes if phonemes else ['?'])
            actual_words.append(word)
            
            # Detect phoneme boundaries using RMS
            result = pipeline.detector.detect_boundaries(
                spec,
                word=word,
                use_multifeature=False,
                use_rms_boundaries=True,
                audio_segment=audio_seg,
                audio_sr=pipeline.config.audio_sr
            )
            
            word_phoneme_boundaries.append(result['boundaries'])
    
    if not word_specs:
        print(f"No words found for sentence {sentence_idx}")
        return
    
    # Play audio
    if play_audio and word_audios:
        print("\n" + "="*70)
        print("FULL SENTENCE AUDIO")
        print("="*70)
        full_audio = np.concatenate(word_audios)
        display(Audio(full_audio, rate=int(pipeline.config.audio_sr)))

        print("\n" + "="*70)
        print("INDIVIDUAL WORD AUDIO")
        print("="*70)
        for word, audio_seg in zip(actual_words, word_audios):
            print(f"\n'{word}':")
            display(Audio(audio_seg, rate=int(pipeline.config.audio_sr)))
    
    # Concatenate spectrograms and compute RMS
    full_spec = np.vstack(word_specs)
    full_audio_concat = np.concatenate(word_audios)
    
    # Compute RMS for full sentence
    sr = pipeline.config.audio_sr
    hop_length = int(0.005 * sr)
    frame_length = int(0.020 * sr)
    
    rms = []
    for i in range(0, len(full_audio_concat) - frame_length, hop_length):
        frame = full_audio_concat[i:i+frame_length]
        rms.append(np.sqrt(np.mean(frame**2)))
    rms = np.array(rms)
    rms_smoothed = gaussian_filter1d(rms, sigma=2)
    rms_change = np.abs(np.gradient(rms_smoothed))
    rms_change_smoothed = gaussian_filter1d(rms_change, sigma=1.5)
    
    # Calculate word boundaries in spectrogram frames
    word_boundaries_frames = [0]
    for spec in word_specs:
        word_boundaries_frames.append(word_boundaries_frames[-1] + spec.shape[0])
    
    # Calculate all phoneme boundaries (word-level to sentence-level)
    all_phoneme_boundaries_frames = []
    for i, boundaries in enumerate(word_phoneme_boundaries):
        offset = word_boundaries_frames[i]
        for b in boundaries[1:-1]:  # Skip word start/end
            all_phoneme_boundaries_frames.append(offset + b)
    
    # Visualization
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(6, 1, height_ratios=[1.5, 1, 1, 1, 0.8, 1.2])
    axes = [fig.add_subplot(gs[i]) for i in range(6)]
    
    # 1. Spectrogram with word AND phoneme boundaries
    axes[0].imshow(full_spec.T, aspect='auto', origin='lower', cmap='viridis')
    axes[0].set_title(f"Sentence: '{sentence}' | RMS-Based Phoneme Segmentation", 
                     fontweight='bold', fontsize=14)
    axes[0].set_ylabel('Mel Bin')
    
    # Word boundaries (red)
    for i, boundary in enumerate(word_boundaries_frames[:-1]):
        axes[0].axvline(boundary, color='red', linestyle='--', linewidth=3, alpha=0.8)
        mid = (word_boundaries_frames[i] + word_boundaries_frames[i+1]) / 2
        axes[0].text(mid, full_spec.shape[1] * 0.95, actual_words[i], 
                    ha='center', va='top', color='white', fontweight='bold', fontsize=11,
                    bbox=dict(boxstyle='round', facecolor='red', alpha=0.8))
    
    # Phoneme boundaries (yellow)
    for boundary in all_phoneme_boundaries_frames:
        axes[0].axvline(boundary, color='yellow', linestyle=':', linewidth=1.5, alpha=0.6)
    
    # 2. Energy contour
    energy = np.sum(full_spec ** 2, axis=1)
    axes[1].plot(energy, linewidth=2, color='blue')
    axes[1].set_ylabel('Energy')
    axes[1].set_title('Spectrogram Energy', fontweight='bold')
    axes[1].grid(alpha=0.3)
    
    for boundary in word_boundaries_frames[:-1]:
        axes[1].axvline(boundary, color='red', linestyle='--', linewidth=2, alpha=0.7)
    for boundary in all_phoneme_boundaries_frames:
        axes[1].axvline(boundary, color='yellow', linestyle=':', linewidth=1, alpha=0.5)
    
    # 3. RMS envelope
    rms_time_frames = np.linspace(0, len(full_spec), len(rms_smoothed))
    axes[2].plot(rms_time_frames, rms_smoothed, linewidth=2, color='darkblue')
    axes[2].fill_between(rms_time_frames, 0, rms_smoothed, alpha=0.3, color='blue')
    axes[2].set_ylabel('RMS Power')
    axes[2].set_title('RMS Envelope (Speech Energy)', fontweight='bold')
    axes[2].grid(alpha=0.3)
    
    for boundary in word_boundaries_frames[:-1]:
        axes[2].axvline(boundary, color='red', linestyle='--', linewidth=2, alpha=0.7)
    for boundary in all_phoneme_boundaries_frames:
        axes[2].axvline(boundary, color='yellow', linestyle=':', linewidth=1, alpha=0.5)
    
    # 4. RMS change (boundary detection signal)
    axes[3].plot(rms_time_frames, rms_change_smoothed, linewidth=2, color='darkred')
    axes[3].fill_between(rms_time_frames, 0, rms_change_smoothed, alpha=0.3, color='red')
    axes[3].set_ylabel('RMS Change')
    axes[3].set_title('RMS Change (Phoneme Boundary Detection Signal)', fontweight='bold')
    axes[3].grid(alpha=0.3)
    
    # Mark threshold
    median_val = np.median(rms_change_smoothed)
    mad = np.median(np.abs(rms_change_smoothed - median_val))
    threshold = median_val + 1.2 * mad
    axes[3].axhline(threshold, color='orange', linestyle=':', linewidth=2, label=f'Threshold: {threshold:.4f}')
    axes[3].legend()
    
    for boundary in word_boundaries_frames[:-1]:
        axes[3].axvline(boundary, color='red', linestyle='--', linewidth=2, alpha=0.7)
    for boundary in all_phoneme_boundaries_frames:
        axes[3].axvline(boundary, color='yellow', linestyle=':', linewidth=1, alpha=0.5)
    
    # 5. Word durations
    colors = plt.cm.viridis(np.linspace(0, 1, len(actual_words)))
    bars = axes[4].bar(range(len(actual_words)), word_durations, color=colors, 
                      edgecolor='black', linewidth=2)
    axes[4].set_xticks(range(len(actual_words)))
    axes[4].set_xticklabels(actual_words, rotation=45, ha='right')
    axes[4].set_ylabel('Duration (s)')
    axes[4].set_title('Word Durations', fontweight='bold')
    axes[4].grid(axis='y', alpha=0.3)
    
    for i, (bar, n_ph) in enumerate(zip(bars, phoneme_counts)):
        height = bar.get_height()
        axes[4].text(bar.get_x() + bar.get_width()/2, height + 0.01,
                    f'{n_ph}ph', ha='center', va='bottom', fontsize=9)
    
    # 6. Phoneme segmentation details per word
    axes[5].axis('off')
    axes[5].set_xlim(0, 1)
    axes[5].set_ylim(0, len(actual_words))
    axes[5].set_title('Detected Phoneme Boundaries per Word', fontweight='bold', fontsize=12)
    
    for i, (word, boundaries, expected) in enumerate(zip(actual_words, word_phoneme_boundaries, word_expected_phonemes)):
        y_pos = len(actual_words) - i - 0.5
        
        n_segments = len(boundaries) - 1
        n_expected = len(expected)
        match = "✓" if n_segments == n_expected else "✗"
        color = 'green' if n_segments == n_expected else 'red'
        
        text = f"{word:15s}: Expected {n_expected} ({expected}) → Detected {n_segments} {match}"
        axes[5].text(0.02, y_pos, text, fontsize=10, verticalalignment='center',
                    color=color, fontweight='bold' if color == 'red' else 'normal')
    
    plt.tight_layout()
    plt.savefig(f'rms_sentence_segmentation_{participant_id}_sent{sentence_idx}.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Summary
    print(f"\n{'='*70}")
    print("RMS-BASED PHONEME SEGMENTATION SUMMARY")
    print("="*70)
    for word, dur, boundaries, expected in zip(actual_words, word_durations, word_phoneme_boundaries, word_expected_phonemes):
        n_detected = len(boundaries) - 1
        n_expected = len(expected)
        match = "✓ MATCH" if n_detected == n_expected else "✗ MISMATCH"
        
        print(f"\n  {word:15s}: {dur:.2f}s")
        print(f"    Expected: {n_expected} phonemes {expected}")
        print(f"    Detected: {n_detected} segments {match}")
        
        if n_detected > 0:
            segment_durations = [(boundaries[i+1] - boundaries[i]) * pipeline.config.frameshift 
                               for i in range(len(boundaries)-1)]
            avg_duration = np.mean(segment_durations)
            print(f"    Avg segment: {avg_duration*1000:.0f}ms")

# Use it
visualize_sentence_segmentation(pipeline, 'P22', sentence_idx=2, play_audio=True)
