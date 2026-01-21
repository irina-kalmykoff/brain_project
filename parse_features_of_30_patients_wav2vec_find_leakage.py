# Converted from parse_features_of_30_patients_wav2vec_find_leakage.ipynb

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
from feature_vizualizer import PhonemeFeatureVisualizer
from markov_phoneme_model import MarkovPhonemeModel
from extract_features import extractHG, downsampleLabels, extractMelSpecs
from pipeline import UnifiedPhonemePipeline
from config import BIDS_PATH, OUTPUT_PATH, RESULTS_PATH, DUTCH_30_PATH, DUTCH_10_PATH, get_dataset_paths

#from dutch_30_pipeline import Dutch30Pipeline
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
visualizer = PhonemeFeatureVisualizer(output_dir='./phoneme_visualizations')

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

config = Dutch30Config()
extractor = Dutch30FeatureExtractor(config=config)
from dutch_30_pipeline import Dutch30Pipeline

def test_patient_combinations_matrix_v2(
    test_patients=['P01', 'P02', 'P04', 'P07'],
    pipeline_params=None
):
    """
    Test accuracy using the simple pattern that works.
    """
    import numpy as np
    import pandas as pd
    from itertools import combinations
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score
    import gc
    
    # Default parameters
    default_params = {
        'pca_components': 100,
        'feature_extraction_method': 'high_gamma',
        'use_rms_boundaries': True,
        'use_multifeature': False
    }
    if pipeline_params:
        default_params.update(pipeline_params)
    params = default_params
    
    print("="*80)
    print("PATIENT COMBINATIONS TEST")
    print(f"Patients: {test_patients}")
    print(f"Params: {params}")
    print("="*80)
    
    results_matrix = []
    
    def get_accuracy(patient_ids, target_pid):
        """Run pipeline and get accuracy - using simple working pattern."""
        gc.collect()
        np.random.seed(42)
        
        # Create fresh pipeline
        pipeline = Dutch30Pipeline(
            dutch30_extractor=extractor,
            config=config,
            pca_components=params['pca_components'],
            feature_extraction_method=params['feature_extraction_method'],
            use_rms_boundaries=params['use_rms_boundaries'],
            use_multifeature=params['use_multifeature']
        )
        
        # Run all steps
        pipeline.step1_load_dutch30_data(patient_ids=patient_ids)
        pipeline.step2_split_by_instances()
        pipeline.step4_custom_detector()
        pipeline.step5_accumulate_data_dutch30()
        pipeline.dutch30_step6_resolve_unknowns()
        
        # Filter to target patient
        train_mask = [p == target_pid for p in pipeline.train['phoneme_participant_ids']]
        test_mask = [p == target_pid for p in pipeline.test['phoneme_participant_ids']]
        
        if not any(train_mask) or not any(test_mask):
            del pipeline
            gc.collect()
            return None
        
        train_feat = [pipeline.train['features'][i] for i, m in enumerate(train_mask) if m]
        train_labels = [pipeline.train['phoneme_labels'][i] for i, m in enumerate(train_mask) if m]
        test_feat = [pipeline.test['features'][i] for i, m in enumerate(test_mask) if m]
        test_labels = [pipeline.test['phoneme_labels'][i] for i, m in enumerate(test_mask) if m]
        
        if len(train_feat) < 10 or len(test_feat) < 5:
            del pipeline
            gc.collect()
            return None
        
        # Prepare features - EXACT same as simple_test_p02
        X_train = np.array([np.mean(f, axis=0) for f in train_feat])
        X_test = np.array([np.mean(f, axis=0) for f in test_feat])
        
        # Scale and train
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        clf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        clf.fit(X_train, train_labels)
        
        preds = clf.predict(X_test)
        acc = accuracy_score(test_labels, preds)
        
        del pipeline
        gc.collect()
        
        return acc
    
    # ===== STANDALONE =====
    print("\nStandalone patients:")
    standalone = {}
    for pid in test_patients:
        acc = get_accuracy([pid], pid)
        standalone[pid] = acc if acc else 0.0
        print(f"  {pid}: {standalone[pid]:.3f}")
        
        row = {'Combination': pid, 'N_Patients': 1}
        for p in test_patients:
            row[p] = standalone[pid] if p == pid else np.nan
        results_matrix.append(row)
    
    # ===== PAIRS =====
    print("\nPairs:")
    for pair in combinations(test_patients, 2):
        row = {'Combination': f"{pair[0]}+{pair[1]}", 'N_Patients': 2}
        print(f"  {pair}:")
        for pid in test_patients:
            if pid in pair:
                acc = get_accuracy(list(pair), pid)
                row[pid] = acc if acc else np.nan
                print(f"    {pid}: {acc:.3f}" if acc else f"    {pid}: FAILED")
            else:
                row[pid] = np.nan
        results_matrix.append(row)
    
    # ===== TRIPLETS =====
    print("\nTriplets:")
    for triplet in combinations(test_patients, 3):
        row = {'Combination': '+'.join(triplet), 'N_Patients': 3}
        print(f"  {triplet}:")
        for pid in test_patients:
            if pid in triplet:
                acc = get_accuracy(list(triplet), pid)
                row[pid] = acc if acc else np.nan
                print(f"    {pid}: {acc:.3f}" if acc else f"    {pid}: FAILED")
            else:
                row[pid] = np.nan
        results_matrix.append(row)
    
    # ===== ALL =====
    print("\nAll patients:")
    row = {'Combination': 'ALL', 'N_Patients': len(test_patients)}
    for pid in test_patients:
        acc = get_accuracy(test_patients, pid)
        row[pid] = acc if acc else np.nan
        print(f"  {pid}: {acc:.3f}" if acc else f"  {pid}: FAILED")
    results_matrix.append(row)
    
    # ===== RESULTS =====
    df = pd.DataFrame(results_matrix)
    
    print("\n" + "="*80)
    print("ACCURACY MATRIX")
    print("="*80)
    print(df.to_string(index=False))
    
    # ===== STABILITY =====
    print("\n" + "="*80)
    print("STABILITY ANALYSIS")
    print("="*80)
    
    for pid in test_patients:
        vals = df[pid].dropna().values
        if len(vals) > 0:
            base = standalone[pid]
            max_change = np.max(np.abs(vals - base))
            print(f"\n{pid}:")
            print(f"  Standalone: {base:.3f}")
            print(f"  Mean:       {vals.mean():.3f}")
            print(f"  Std:        {vals.std():.3f}")
            print(f"  Max change: {max_change:.3f}")
            if max_change < 0.02:
                print(f"  Status: STABLE")
            elif max_change < 0.05:
                print(f"  Status: MINOR VARIATION")
            else:
                print(f"  Status: UNSTABLE")
    
    return df

def train_single_patient(pipeline, pid):
    """
    Train and evaluate a classifier for a single patient.
    
    Args:
        pipeline: Dutch30Pipeline with loaded and processed data
        pid: Patient ID to train/evaluate
        
    Returns:
        Dict with 'accuracy', 'train_size', 'test_size' or None if failed
    """
    from sklearn.metrics import accuracy_score
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    
    # Filter data for this patient
    train_mask = [p == pid for p in pipeline.train['phoneme_participant_ids']]
    test_mask = [p == pid for p in pipeline.test['phoneme_participant_ids']]
    
    # Check if patient has any data
    if not any(train_mask) or not any(test_mask):
        return None
    
    train_feat = [pipeline.train['features'][i] for i, m in enumerate(train_mask) if m]
    train_labels = [pipeline.train['phoneme_labels'][i] for i, m in enumerate(train_mask) if m]
    test_feat = [pipeline.test['features'][i] for i, m in enumerate(test_mask) if m]
    test_labels = [pipeline.test['phoneme_labels'][i] for i, m in enumerate(test_mask) if m]
    
    # Check minimum data requirements
    if len(train_feat) < 10 or len(test_feat) < 5:
        return None
    
    # Flatten: average over time frames
    X_train = np.array([np.mean(f, axis=0) if f.ndim > 1 else f for f in train_feat])
    X_test = np.array([np.mean(f, axis=0) if f.ndim > 1 else f for f in test_feat])
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Train classifier
    clf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    clf.fit(X_train, train_labels)
    
    # Evaluate
    preds = clf.predict(X_test)
    acc = accuracy_score(test_labels, preds)
    
    return {
        'accuracy': acc,
        'train_size': len(train_feat),
        'test_size': len(test_feat)
    }

# Run it
gc.collect()
results_df = test_patient_combinations_matrix_v2(
    test_patients=['P01', 'P02', 'P04', 'P07']
)

# Default parameters (high_gamma with PCA)
# results_df = test_patient_combinations_matrix(
#     test_patients=['P01', 'P02', 'P04', 'P07']
# )

# Or with custom parameters:
results_df = test_patient_combinations_matrix(
    test_patients=['P01', 'P02', 'P04', 'P07'],
    pipeline_params={
        'pca_components': None, 
        'feature_extraction_method': 'high_gamma',
        'use_rms_boundaries': True,
        'use_multifeature': False,    
        'use_wav2vec': True 
    }
)

import numpy as np
import gc
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

def simple_test_p02():
    """Bare minimum test - no helper functions."""
    
    gc.collect()
    np.random.seed(42)
    
    print("="*60)
    print("SIMPLE P02 TEST")
    print("="*60)
    
    # Create pipeline
    pipeline = Dutch30Pipeline(
        dutch30_extractor=extractor,
        config=config,
        pca_components=100,
        feature_extraction_method='high_gamma',
        use_rms_boundaries=True,
        use_multifeature=False
    )
    
    # Run steps
    pipeline.step1_load_dutch30_data(patient_ids=['P02'])
    pipeline.step2_split_by_instances()
    pipeline.step4_custom_detector()
    pipeline.step5_accumulate_data_dutch30()
    pipeline.dutch30_step6_resolve_unknowns()
    
    # Check what's in the data
    print(f"\n1. Raw data check:")
    print(f"   Train features: {len(pipeline.train['features'])}")
    print(f"   Train labels: {len(pipeline.train['phoneme_labels'])}")
    print(f"   Train PIDs: {set(pipeline.train['phoneme_participant_ids'])}")
    print(f"   Test features: {len(pipeline.test['features'])}")
    
    # METHOD A: Use ALL data directly (like simple_comparison_p02)
    print(f"\n2. METHOD A - Use all data directly:")
    
    train_feat_a = pipeline.train['features']
    train_labels_a = pipeline.train['phoneme_labels']
    test_feat_a = pipeline.test['features']
    test_labels_a = pipeline.test['phoneme_labels']
    
    X_train_a = np.array([np.mean(f, axis=0) for f in train_feat_a])
    X_test_a = np.array([np.mean(f, axis=0) for f in test_feat_a])
    
    print(f"   X_train shape: {X_train_a.shape}")
    print(f"   X_test shape: {X_test_a.shape}")
    
    scaler_a = StandardScaler()
    X_train_a_scaled = scaler_a.fit_transform(X_train_a)
    X_test_a_scaled = scaler_a.transform(X_test_a)
    
    clf_a = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    clf_a.fit(X_train_a_scaled, train_labels_a)
    preds_a = clf_a.predict(X_test_a_scaled)
    acc_a = accuracy_score(test_labels_a, preds_a)
    
    print(f"   ACCURACY: {acc_a:.3f}")
    
    # METHOD B: Filter by PID (like train_single_patient)
    print(f"\n3. METHOD B - Filter by patient ID 'P02':")
    
    train_mask = [p == 'P02' for p in pipeline.train['phoneme_participant_ids']]
    test_mask = [p == 'P02' for p in pipeline.test['phoneme_participant_ids']]
    
    print(f"   Train mask sum: {sum(train_mask)}")
    print(f"   Test mask sum: {sum(test_mask)}")
    
    train_feat_b = [pipeline.train['features'][i] for i, m in enumerate(train_mask) if m]
    train_labels_b = [pipeline.train['phoneme_labels'][i] for i, m in enumerate(train_mask) if m]
    test_feat_b = [pipeline.test['features'][i] for i, m in enumerate(test_mask) if m]
    test_labels_b = [pipeline.test['phoneme_labels'][i] for i, m in enumerate(test_mask) if m]
    
    print(f"   Filtered train: {len(train_feat_b)}")
    print(f"   Filtered test: {len(test_feat_b)}")
    
    X_train_b = np.array([np.mean(f, axis=0) for f in train_feat_b])
    X_test_b = np.array([np.mean(f, axis=0) for f in test_feat_b])
    
    print(f"   X_train shape: {X_train_b.shape}")
    print(f"   X_test shape: {X_test_b.shape}")
    
    scaler_b = StandardScaler()
    X_train_b_scaled = scaler_b.fit_transform(X_train_b)
    X_test_b_scaled = scaler_b.transform(X_test_b)
    
    clf_b = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    clf_b.fit(X_train_b_scaled, train_labels_b)
    preds_b = clf_b.predict(X_test_b_scaled)
    acc_b = accuracy_score(test_labels_b, preds_b)
    
    print(f"   ACCURACY: {acc_b:.3f}")
    
    # Compare
    print(f"\n4. COMPARISON:")
    print(f"   Method A (all data):    {acc_a:.3f}")
    print(f"   Method B (filtered):    {acc_b:.3f}")
    print(f"   Same? {acc_a == acc_b}")
    
    # Check if data is identical
    print(f"\n5. DATA COMPARISON:")
    print(f"   Same train size? {len(train_feat_a) == len(train_feat_b)}")
    print(f"   Same test size? {len(test_feat_a) == len(test_feat_b)}")
    
    if len(train_feat_a) == len(train_feat_b):
        diff = np.abs(X_train_a - X_train_b).max()
        print(f"   Max feature diff: {diff:.6f}")
    
    del pipeline
    gc.collect()
    
    return acc_a, acc_b

# Run it
acc_a, acc_b = simple_test_p02()

# Run the matrix test, but ONLY for P02 combinations
# Watch if the numbers change as tests progress

import gc
gc.collect()

results = []

combos = [
    (['P02'], 'P02'),
    (['P02', 'P04'], 'P02'),
    (['P02', 'P07'], 'P02'),
    (['P02', 'P01'], 'P02'),
]

for patient_ids, target in combos:
    gc.collect()
    np.random.seed(42)
    
    pipeline = Dutch30Pipeline(
        dutch30_extractor=extractor,
        config=config,
        pca_components=100,
        feature_extraction_method='high_gamma',
        use_rms_boundaries=True,
        use_multifeature=False
    )
    
    pipeline.step1_load_dutch30_data(patient_ids=patient_ids)
    pipeline.step2_split_by_instances()
    pipeline.step4_custom_detector()
    pipeline.step5_accumulate_data_dutch30()
    pipeline.dutch30_step6_resolve_unknowns()
    
    p02_count = sum(1 for p in pipeline.train['phoneme_participant_ids'] if p == 'P02')
    p02_feat = [pipeline.train['features'][i] for i, m in enumerate([p == 'P02' for p in pipeline.train['phoneme_participant_ids']]) if m]
    
    if p02_feat:
        X = np.array([np.mean(f, axis=0) for f in p02_feat])
        first_5 = X[0][:5]
    else:
        first_5 = None
    
    print(f"{str(patient_ids):<25} P02 samples: {p02_count}, shape: {p02_feat[0].shape if p02_feat else 'N/A'}, first 5: {first_5}")
    
    results.append({
        'combo': patient_ids,
        'count': p02_count,
        'first_5': first_5
    })
    
    del pipeline
    gc.collect()

# Check consistency
counts = [r['count'] for r in results]
print(f"\nAll P02 counts same? {len(set(counts)) == 1}")
print(f"Counts: {counts}")
