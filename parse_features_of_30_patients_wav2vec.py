# Converted from parse_features_of_30_patients_wav2vec.ipynb

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
    pipeline.step3_load_channel_exclusions('channel_exclusions.json')
    pipeline.step4_custom_detector()
    pipeline.step5_accumulate_data_dutch30()
    pipeline.dutch30_step6_resolve_unknowns()
    pipeline.checkpoint_after_step6(sample_fraction=sample_fraction)
    pipeline.step7_filter_unknowns(unknown_keep_ratio=0.0025);
    
    print(f"\nPipeline '{name}' created successfully!")
    print(f"  Train samples: {len(pipeline.train.get('features', []))}")
    print(f"  Test samples: {len(pipeline.test.get('features', []))}")   

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

# diag = Dutch30PhonemeDetectionDiagnostic(pipeline)
# diag.visualize_word_analysis('P01', word_name = 'vogelkooitje', save_path='p21_word5.png')

#diag.visualize_multifeature_analysis('P01', word_index=50)
# diag.visualize_rms_boundaries('P01',  word_name = 'vogelkooitje')

# Quick check first 10 words
# diag.batch_diagnostic('sub-p11', num_samples=5)

# # step_0
# # pipeline.analyze_dutch30_channels()
# #pipeline.step1_load_dutch30_data(num_patients = 20)
# pipeline.step1_load_dutch30_data(patient_range=(10,20))
# pipeline.split_result = None
# pipeline.step2_split_by_instances();
# #pipeline.step2_3_use_existing_split()
# pipeline.step4_custom_detector()
# pipeline.step5_accumulate_data_dutch30();
# pipeline.dutch30_step6_resolve_unknowns();
# # After step4 or step5
# # print(pipeline.phonetic_dict.get_missing_words_summary())
# # pipeline.checkpoint_after_step6(sample_fraction=0.0005)
# # pipeline.try_load_checkpoint(sample_fraction=0.1)
# pipeline.step7_filter_unknowns(unknown_keep_ratio=0.025);

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
        
        model.DEBUG_MODE = True  # Enable debug output

        model.train(features=train_feat, phoneme_labels=train_labels)
        eval_result = model.evaluate(features=test_feat, true_labels=test_labels, use_viterbi=True)
        
        print(f"Accuracy: {eval_result['accuracy']:.3f}")
        
        # ===== DEBUG: Check predictions directly =====
        preds, probs = model.predict(test_feat, use_viterbi=True)
        
        # Only print debug for first patient
        if len(results) == 0:
            print(f"\n{'='*60}")
            print(f"DEBUG FOR {pid}")
            print(f"{'='*60}")
            print(f"  test_feat length: {len(test_feat)}")
            print(f"  test_feat[0] shape: {test_feat[0].shape}")
            print(f"  test_labels length: {len(test_labels)}")
            print(f"  preds type: {type(preds)}")
            print(f"  preds length: {len(preds)}")
            
            if len(preds) > 0:
                print(f"  preds[0] type: {type(preds[0])}")
                print(f"  First 10 preds: {preds[:10]}")
                print(f"  First 10 test_labels: {test_labels[:10]}")
            
            # Manual accuracy check
            correct = sum(1 for p, t in zip(preds, test_labels) if p == t)
            print(f"  Manual accuracy: {correct}/{len(test_labels)} = {correct/len(test_labels):.3f}")
            
            # Check unique values
            print(f"  Unique preds: {sorted(set(preds))[:10]}...")
            print(f"  Unique test_labels: {sorted(set(test_labels))[:10]}...")
            
            # Check for format issues
            if len(preds) > 0 and len(test_labels) > 0:
                print(f"  preds[0] == test_labels[0]: {preds[0] == test_labels[0]}")
                print(f"  repr(preds[0]): {repr(preds[0])}")
                print(f"  repr(test_labels[0]): {repr(test_labels[0])}")
            print(f"{'='*60}\n")
        # ===== END DEBUG =====
        
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

def discover_data_driven_groups(pipeline, n_clusters_range=(2, 10), min_samples_per_phoneme=10, save_path=None):
    """
    Discover phoneme groupings based on iEEG feature similarity.
    
    Args:
        pipeline: Dutch30Pipeline with loaded data
        n_clusters_range: Range of cluster numbers to try
        min_samples_per_phoneme: Minimum samples to include a phoneme
        save_path: Directory to save figures (None = don't save)
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
    from scipy.spatial.distance import squareform
    import matplotlib.pyplot as plt
    import os
    
    # Get pipeline parameters for titles
    pca_components = getattr(pipeline, 'pca_components', None)
    feature_method = getattr(pipeline, 'feature_extraction_method', 'unknown')
    subtract_baseline = getattr(pipeline, 'subtract_baseline_flag', 'unknown')
    
    title_suffix = f"PCA={pca_components}, Features={feature_method}, Baseline={subtract_baseline}"
    
    phoneme_features = defaultdict(list)
    
    all_features = pipeline.train['features']
    all_labels = pipeline.train['phoneme_labels']
    
    for feat, label in zip(all_features, all_labels):
        if label == '?' or label == 'unknown':
            continue
        
        if feat.ndim > 1:
            feat_flat = np.mean(feat, axis=0)
        else:
            feat_flat = feat
        
        if not np.any(np.isnan(feat_flat)) and not np.any(np.isinf(feat_flat)):
            phoneme_features[label].append(feat_flat)
    
    # Find most common feature length
    all_lengths = []
    for features in phoneme_features.values():
        all_lengths.extend([len(f) for f in features])
    
    length_counts = Counter(all_lengths)
    common_len = length_counts.most_common(1)[0][0]
    print(f"Most common feature length: {common_len}")
    
    # Filter to consistent length and compute centroids
    phoneme_centroids = {}
    phoneme_counts = {}
    
    for phoneme, features in phoneme_features.items():
        filtered_features = [f for f in features if len(f) == common_len]
        
        if len(filtered_features) >= min_samples_per_phoneme:
            phoneme_centroids[phoneme] = np.mean(np.array(filtered_features), axis=0)
            phoneme_counts[phoneme] = len(filtered_features)
    
    print(f"Phonemes with sufficient samples: {len(phoneme_centroids)}")
    print(f"Phoneme counts: {dict(sorted(phoneme_counts.items(), key=lambda x: -x[1]))}")
    
    if len(phoneme_centroids) < 3:
        print("ERROR: Too few phonemes with sufficient samples")
        return None
    
    phoneme_names = list(phoneme_centroids.keys())
    X_centroids = np.array([phoneme_centroids[p] for p in phoneme_names])
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_centroids)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(f"Data-Driven Phoneme Clustering\n{title_suffix}", fontsize=12, fontweight='bold')
    
    # Dendrogram
    linkage_matrix = linkage(X_scaled, method='ward')
    
    dendrogram(
        linkage_matrix,
        labels=phoneme_names,
        leaf_rotation=90,
        leaf_font_size=8,
        ax=axes[0, 0]
    )
    axes[0, 0].set_title('Hierarchical Clustering of Phonemes (Ward)')
    axes[0, 0].set_ylabel('Distance')
    
    # Silhouette scores
    silhouette_scores = []
    k_range = range(n_clusters_range[0], min(n_clusters_range[1] + 1, len(phoneme_names)))
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        silhouette_scores.append(score)
    
    axes[0, 1].plot(list(k_range), silhouette_scores, 'bo-')
    axes[0, 1].set_xlabel('Number of Clusters')
    axes[0, 1].set_ylabel('Silhouette Score')
    axes[0, 1].set_title('Optimal Cluster Number (higher is better)')
    axes[0, 1].grid(True, alpha=0.3)
    
    best_k = list(k_range)[np.argmax(silhouette_scores)]
    axes[0, 1].axvline(x=best_k, color='r', linestyle='--', label=f'Best k={best_k}')
    axes[0, 1].legend()
    
    print(f"\nOptimal number of clusters: {best_k} (silhouette={max(silhouette_scores):.3f})")
    
    # Get final clusters
    kmeans_best = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    cluster_labels = kmeans_best.fit_predict(X_scaled)
    
    data_driven_groups = defaultdict(list)
    phoneme_to_group = {}
    
    for phoneme, cluster_id in zip(phoneme_names, cluster_labels):
        group_name = f'cluster_{cluster_id}'
        data_driven_groups[group_name].append(phoneme)
        phoneme_to_group[phoneme] = group_name
    
    print(f"\nData-driven groups:")
    print("=" * 60)
    for group_name, phonemes in sorted(data_driven_groups.items()):
        total_samples = sum(phoneme_counts[p] for p in phonemes)
        print(f"{group_name}: {phonemes}")
        print(f"    Total samples: {total_samples}")
    
    # PCA visualization
    from sklearn.decomposition import PCA
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    colors = plt.cm.tab10(np.linspace(0, 1, best_k))
    
    for cluster_id in range(best_k):
        mask = cluster_labels == cluster_id
        axes[1, 0].scatter(
            X_pca[mask, 0], 
            X_pca[mask, 1], 
            c=[colors[cluster_id]], 
            label=f'Cluster {cluster_id}',
            s=100
        )
    
    for i, phoneme in enumerate(phoneme_names):
        axes[1, 0].annotate(
            phoneme, 
            (X_pca[i, 0], X_pca[i, 1]),
            fontsize=8,
            ha='center',
            va='bottom'
        )
    
    axes[1, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)')
    axes[1, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)')
    axes[1, 0].set_title('Phoneme Clusters in PCA Space')
    axes[1, 0].legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Group sizes
    group_names = sorted(data_driven_groups.keys())
    group_sizes = [len(data_driven_groups[g]) for g in group_names]
    
    axes[1, 1].bar(group_names, group_sizes)
    axes[1, 1].set_xlabel('Group')
    axes[1, 1].set_ylabel('Number of Phonemes')
    axes[1, 1].set_title('Group Sizes')
    
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        filename = f"dendrogram_pca{pca_components}_{feature_method}_baseline{subtract_baseline}.png"
        filepath = os.path.join(save_path, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"\nFigure saved to: {filepath}")
    
    plt.show()
    
    results = {
        'phoneme_to_group': phoneme_to_group,
        'groups': dict(data_driven_groups),
        'n_clusters': best_k,
        'silhouette_score': max(silhouette_scores),
        'phoneme_centroids': phoneme_centroids,
        'phoneme_counts': phoneme_counts,
        'cluster_labels': dict(zip(phoneme_names, cluster_labels)),
        'pipeline_params': {
            'pca_components': pca_components,
            'feature_extraction_method': feature_method,
            'subtract_baseline': subtract_baseline
        }
    }
    
    return results
#clustering_results = discover_data_driven_groups(pipeline, n_clusters_range=(3, 12), min_samples_per_phoneme=3)
clustering_results = discover_data_driven_groups(
    pipeline, 
    n_clusters_range=(2, 30), 
    min_samples_per_phoneme=3,
    save_path='./results/clustering'
)

def train_and_evaluate(pipeline, use_groups=False):
    """Train per patient and return results."""
    from sklearn.preprocessing import StandardScaler
    
    results = {}
    for pid in sorted(set(pipeline.train['phoneme_participant_ids'])):
        # Filter data
        train_mask = [p == pid for p in pipeline.train['phoneme_participant_ids']]
        test_mask = [p == pid for p in pipeline.test['phoneme_participant_ids']]
        train_feat = [pipeline.train['features'][i] for i, m in enumerate(train_mask) if m]
        train_labels = [pipeline.train['phoneme_labels'][i] for i, m in enumerate(train_mask) if m]
        test_feat = [pipeline.test['features'][i] for i, m in enumerate(test_mask) if m]
        test_labels = [pipeline.test['phoneme_labels'][i] for i, m in enumerate(test_mask) if m]
        
        if len(train_feat) < 10 or len(test_feat) < 5:
            continue
        
        # Flatten features for scaler
        X_train = np.array([np.mean(f, axis=0) if f.ndim > 1 else f for f in train_feat])
        X_test = np.array([np.mean(f, axis=0) if f.ndim > 1 else f for f in test_feat])
        
        # Create and fit scaler
        scaler = StandardScaler()
        scaler.fit(X_train)
        
        # Train model
        model = MarkovPhonemeModel(
            phonetic_dict=pipeline.detector.phonetic_dict,
            order=1,
            use_groups=use_groups
        )
        model.train(features=train_feat, phoneme_labels=train_labels)
        
        # Predict and calculate accuracy
        preds, _ = model.predict(test_feat, use_viterbi=True)
        correct = sum(1 for p, t in zip(preds, test_labels) if p == t)
        accuracy = correct / len(test_labels)
        
        results[pid] = {
            'model': model,
            'scaler': scaler,  # Add scaler here
            'accuracy': accuracy,
            'train_size': len(train_feat),
            'test_size': len(test_feat),
            'n_classes': len(set(train_labels)),
            'predictions': preds,
            'true_labels': test_labels
        }
        print(f"  {pid}: Acc={accuracy:.3f} ({len(set(train_labels))} classes, train={len(train_feat)}, test={len(test_feat)})")
    
    # Summary
    accs = [r['accuracy'] for r in results.values()]
    print(f"\n  Mean: {np.mean(accs):.3f} +/- {np.std(accs):.3f}")
    return results

def visualize_patient_model(pid, patient_results, pipeline):
    """Detailed analysis for one patient."""
    
    from sklearn.metrics import confusion_matrix
    from matplotlib.patches import Rectangle

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

    # Use stored predictions if available
    if 'predictions' in patient_results[pid]:
        preds = patient_results[pid]['predictions']
        test_labels = patient_results[pid].get('true_labels', test_labels)
    else:
        model = patient_results[pid]['model']
        preds, _ = model.predict(test_feat, use_viterbi=True)

    # Flatten preds if it's nested (list of lists)
    if preds and isinstance(preds[0], list):
        preds = [p[0] if len(p) > 0 else '?' for p in preds]
    
    # Ensure preds is a list of strings
    preds = [str(p) if not isinstance(p, str) else p for p in preds]

    # Build confusion data
    confusion_data = {}
    for true_label, pred_label in zip(test_labels, preds):
        if true_label not in confusion_data:
            confusion_data[true_label] = Counter()
        confusion_data[true_label][pred_label] += 1

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
    unique_labels = sorted(set(list(test_labels) + list(preds)))
    cm = confusion_matrix(test_labels, preds, labels=unique_labels)

    im = axes[2].imshow(cm, cmap='Blues')
    for i in range(len(unique_labels)):
        axes[2].add_patch(Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, edgecolor='darkgrey', linewidth=0.5))
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
    print(f"{pid} - PER-PHONEME ACCURACY")
    print(f"{'='*70}")
    print(f"{'Phoneme':<8} {'Acc':<6} {'Count':<6} {'Top 3 Confusions'}")
    print('-'*70)

    for p in test_phonemes:
        if p in confusion_data:
            confusions = confusion_data[p].copy()
            confusions.pop(p, None)
            top_confusions = confusions.most_common(3)
            confusion_str = ', '.join([f"{pred}({cnt})" for pred, cnt in top_confusions]) if top_confusions else '-'
        else:
            confusion_str = '-'

        print(f"{p:<8} {phoneme_acc[p]:>5.2f} {test_counts[p]:>6}  {confusion_str}")

    print(f"{'='*70}\n")

# Train on raw phonemes
raw_results = train_and_evaluate(pipeline, use_groups=False)

# patient_results = train_per_patient(pipeline)

for pid in sorted(raw_results.keys()):
    visualize_patient_model(pid, raw_results, pipeline)

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

def plot_accuracy_vs_training_size(pipeline, use_groups=False, n_subsamples=5):
    """
    Plot accuracy vs number of training samples to diagnose learning curves.
    
    Args:
        pipeline: Dutch30Pipeline with loaded data
        use_groups: Whether to use phoneme groups
        n_subsamples: Number of subsample fractions to test
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from markov_phoneme_model import MarkovPhonemeModel
    
    fractions = [0.2, 0.4, 0.6, 0.8, 1.0]
    
    results_by_patient = {}
    
    for pid in sorted(set(pipeline.train['phoneme_participant_ids'])):
        train_mask = [p == pid for p in pipeline.train['phoneme_participant_ids']]
        test_mask = [p == pid for p in pipeline.test['phoneme_participant_ids']]
        
        train_feat = [pipeline.train['features'][i] for i, m in enumerate(train_mask) if m]
        train_labels = [pipeline.train['phoneme_labels'][i] for i, m in enumerate(train_mask) if m]
        test_feat = [pipeline.test['features'][i] for i, m in enumerate(test_mask) if m]
        test_labels = [pipeline.test['phoneme_labels'][i] for i, m in enumerate(test_mask) if m]
        
        if len(train_feat) < 50 or len(test_feat) < 10:
            continue
        
        patient_results = {'fractions': [], 'train_sizes': [], 'accuracies': []}
        
        for frac in fractions:
            n_samples = int(len(train_feat) * frac)
            if n_samples < 10:
                continue
            
            indices = np.random.choice(len(train_feat), n_samples, replace=False)
            subset_feat = [train_feat[i] for i in indices]
            subset_labels = [train_labels[i] for i in indices]
            
            model = MarkovPhonemeModel(
                phonetic_dict=pipeline.detector.phonetic_dict,
                order=1,
                use_groups=use_groups
            )
            
            model.train(features=subset_feat, phoneme_labels=subset_labels)
            preds, _ = model.predict(test_feat, use_viterbi=True)
            
            correct = sum(1 for p, t in zip(preds, test_labels) if p == t)
            accuracy = correct / len(test_labels)
            
            patient_results['fractions'].append(frac)
            patient_results['train_sizes'].append(n_samples)
            patient_results['accuracies'].append(accuracy)
        
        if patient_results['accuracies']:
            results_by_patient[pid] = patient_results
            print(f"{pid}: {patient_results['train_sizes']} -> {[f'{a:.3f}' for a in patient_results['accuracies']]}")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for pid, res in results_by_patient.items():
        axes[0].plot(res['train_sizes'], res['accuracies'], 'o-', label=pid, alpha=0.7)
    
    axes[0].set_xlabel('Number of Training Samples')
    axes[0].set_ylabel('Test Accuracy')
    axes[0].set_title('Learning Curves by Patient')
    axes[0].legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    axes[0].grid(True, alpha=0.3)
    
    all_sizes = []
    all_accs = []
    for res in results_by_patient.values():
        all_sizes.extend(res['train_sizes'])
        all_accs.extend(res['accuracies'])
    
    axes[1].scatter(all_sizes, all_accs, alpha=0.5)
    
    z = np.polyfit(all_sizes, all_accs, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(all_sizes), max(all_sizes), 100)
    axes[1].plot(x_line, p(x_line), 'r--', label=f'Trend (slope={z[0]:.6f})')
    
    axes[1].set_xlabel('Number of Training Samples')
    axes[1].set_ylabel('Test Accuracy')
    axes[1].set_title('Overall Trend: Accuracy vs Training Size')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    slope = z[0]
    if slope < 0.0001:
        print("\nDIAGNOSIS: Flat learning curve - features may be the bottleneck")
    elif slope > 0.001:
        print("\nDIAGNOSIS: Positive slope - model benefits from more data")
    else:
        print("\nDIAGNOSIS: Weak positive slope - moderate benefit from more data")
    
    return results_by_patient
plot_accuracy_vs_training_size(pipeline, use_groups=True)

def plot_accuracy_vs_training_size(pipeline, use_groups=False, n_repeats=3):
    """
    Plot accuracy vs number of training samples to diagnose learning curves.
    
    Args:
        pipeline: Dutch30Pipeline with loaded data
        use_groups: Whether to use phoneme groups (vowel/consonant)
        n_repeats: Number of repeats per fraction to reduce variance
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from collections import Counter
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score
    
    fractions = [0.2, 0.4, 0.6, 0.8, 1.0]
    
    vowels = {'a', 'aː', 'e', 'eː', 'i', 'iː', 'o', 'oː', 'u', 'uː', 
              'ə', 'ɑ', 'ɔ', 'ɛ', 'ɛi', 'ɪ', 'œ', 'øː', 'y', 'ʏ', 'ɑu', 'œy'}
    
    results_by_patient = {}
    
    for pid in sorted(set(pipeline.train['phoneme_participant_ids'])):
        train_mask = [p == pid for p in pipeline.train['phoneme_participant_ids']]
        test_mask = [p == pid for p in pipeline.test['phoneme_participant_ids']]
        
        train_feat = [pipeline.train['features'][i] for i, m in enumerate(train_mask) if m]
        train_labels = [pipeline.train['phoneme_labels'][i] for i, m in enumerate(train_mask) if m]
        test_feat = [pipeline.test['features'][i] for i, m in enumerate(test_mask) if m]
        test_labels = [pipeline.test['phoneme_labels'][i] for i, m in enumerate(test_mask) if m]
        
        if len(train_feat) < 50 or len(test_feat) < 10:
            continue
        
        # Flatten all features first
        X_train_all = []
        for feat in train_feat:
            if feat.ndim > 1:
                X_train_all.append(np.mean(feat, axis=0))
            else:
                X_train_all.append(feat)
        
        X_test_all = []
        for feat in test_feat:
            if feat.ndim > 1:
                X_test_all.append(np.mean(feat, axis=0))
            else:
                X_test_all.append(feat)
        
        # Filter to consistent length
        train_lens = Counter(len(x) for x in X_train_all)
        test_lens = Counter(len(x) for x in X_test_all)
        
        if not train_lens or not test_lens:
            continue
        
        common_len = train_lens.most_common(1)[0][0]
        
        valid_train_idx = [i for i, x in enumerate(X_train_all) if len(x) == common_len]
        valid_test_idx = [i for i, x in enumerate(X_test_all) if len(x) == common_len]
        
        if len(valid_train_idx) < 50 or len(valid_test_idx) < 10:
            continue
        
        X_train_all = np.array([X_train_all[i] for i in valid_train_idx])
        y_train_all = [train_labels[i] for i in valid_train_idx]
        X_test = np.array([X_test_all[i] for i in valid_test_idx])
        y_test = [test_labels[i] for i in valid_test_idx]
        
        # Remove NaN/Inf from training
        valid_train_mask = ~(np.isnan(X_train_all).any(axis=1) | np.isinf(X_train_all).any(axis=1))
        X_train_all = X_train_all[valid_train_mask]
        y_train_all = [y_train_all[i] for i in range(len(y_train_all)) if valid_train_mask[i]]
        
        # Remove NaN/Inf from test
        valid_test_mask = ~(np.isnan(X_test).any(axis=1) | np.isinf(X_test).any(axis=1))
        X_test = X_test[valid_test_mask]
        y_test = [y_test[i] for i in range(len(y_test)) if valid_test_mask[i]]
        
        if len(X_train_all) < 50 or len(X_test) < 10:
            continue
        
        # Apply grouping if requested
        if use_groups:
            y_train_all = ['vowel' if label in vowels else 'consonant' for label in y_train_all]
            y_test = ['vowel' if label in vowels else 'consonant' for label in y_test]
        
        patient_results = {'fractions': [], 'train_sizes': [], 'accuracies': [], 'std': []}
        
        for frac in fractions:
            n_samples = int(len(X_train_all) * frac)
            if n_samples < 10:
                continue
            
            repeat_accs = []
            
            for repeat in range(n_repeats):
                if frac == 1.0:
                    X_train = X_train_all
                    y_train = y_train_all
                else:
                    np.random.seed(repeat)
                    indices = np.random.choice(len(X_train_all), n_samples, replace=False)
                    X_train = X_train_all[indices]
                    y_train = [y_train_all[i] for i in indices]
                
                # Filter test labels not in training
                train_labels_set = set(y_train)
                valid_test_idx = [i for i, label in enumerate(y_test) if label in train_labels_set]
                
                if len(valid_test_idx) < 5:
                    continue
                
                X_test_filtered = X_test[valid_test_idx]
                y_test_filtered = [y_test[i] for i in valid_test_idx]
                
                # Scale
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test_filtered)
                
                # Train
                model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_leaf=2,
                    class_weight='balanced',
                    random_state=42,
                    n_jobs=-1
                )
                
                model.fit(X_train_scaled, y_train)
                preds = model.predict(X_test_scaled)
                
                accuracy = accuracy_score(y_test_filtered, preds)
                repeat_accs.append(accuracy)
            
            if repeat_accs:
                patient_results['fractions'].append(frac)
                patient_results['train_sizes'].append(n_samples)
                patient_results['accuracies'].append(np.mean(repeat_accs))
                patient_results['std'].append(np.std(repeat_accs))
        
        if patient_results['accuracies']:
            results_by_patient[pid] = patient_results
            accs_str = [f'{a:.3f}' for a in patient_results['accuracies']]
            print(f"{pid}: {patient_results['train_sizes']} -> {accs_str}")
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for pid, res in results_by_patient.items():
        axes[0].errorbar(
            res['train_sizes'], 
            res['accuracies'], 
            yerr=res['std'],
            fmt='o-', 
            label=pid, 
            alpha=0.7,
            capsize=3
        )
    
    axes[0].set_xlabel('Number of Training Samples')
    axes[0].set_ylabel('Test Accuracy')
    axes[0].set_title('Learning Curves by Patient')
    axes[0].legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    axes[0].grid(True, alpha=0.3)
    
    all_sizes = []
    all_accs = []
    for res in results_by_patient.values():
        all_sizes.extend(res['train_sizes'])
        all_accs.extend(res['accuracies'])
    
    axes[1].scatter(all_sizes, all_accs, alpha=0.5)
    
    if len(all_sizes) > 1:
        z = np.polyfit(all_sizes, all_accs, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(all_sizes), max(all_sizes), 100)
        axes[1].plot(x_line, p(x_line), 'r--', label=f'Trend (slope={z[0]:.6f})')
        axes[1].legend()
        
        slope = z[0]
        if slope < 0.0001:
            print("\nDIAGNOSIS: Flat learning curve - features may be the bottleneck")
        elif slope > 0.001:
            print("\nDIAGNOSIS: Positive slope - model benefits from more data")
        else:
            print("\nDIAGNOSIS: Weak positive slope - moderate benefit from more data")
    
    axes[1].set_xlabel('Number of Training Samples')
    axes[1].set_ylabel('Test Accuracy')
    axes[1].set_title('Overall Trend: Accuracy vs Training Size')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return results_by_patient

plot_accuracy_vs_training_size(pipeline, use_groups=False)

results_binary = plot_accuracy_vs_training_size(pipeline, use_groups=True, n_repeats=3)

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

all_results, comparison_df = compare_methods(pipeline, use_groups=True)
