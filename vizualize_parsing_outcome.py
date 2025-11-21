# Converted from vizualize_parsing_outcome.ipynb

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import Audio, display
from pynwb import NWBHDF5IO
from extract_features import extractMelSpecs
import scipy.signal

import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed, IntSlider, FloatSlider, Dropdown, Checkbox

from brain_audio_decoder import BrainAudioDecoder
from custom_decoder import CustomBrainAudioDecoder
from brain_audio_decoder_viz import BrainAudioDecoderViz
from acoustic_change_detector import AcousticChangeDetector
from phoneme_validator import PhonemeValidator
from phonetic_dictionary import PhoneticDictionary
from feature_vizualizer import PhonemeFeatureVisualizer
from pipeline import UnifiedPhonemePipeline
from config import BIDS_PATH, OUTPUT_PATH, RESULTS_PATH

# # Generate spectrograms for all participants
# participants = [f'sub-{i:02d}' for i in range(1, 11)]

# for participant in participants:
#     print(f"Processing {participant}...")
    
#     # Load audio
#     nwb_path = os.path.join(BIDS_PATH, participant, 'ieeg', 
#                             f'{participant}_task-wordProduction_ieeg.nwb')
#     io = NWBHDF5IO(nwb_path, 'r')
#     nwbfile = io.read()
#     audio = nwbfile.acquisition['Audio'].data[:]
#     io.close()
    
#     # Process audio
#     audio_sr = 48000
#     target_sr = 16000
#     audio = scipy.signal.decimate(audio, int(audio_sr / target_sr))
#     scaled = np.int16(audio / np.max(np.abs(audio)) * 32767)
    
#     # Extract spectrogram
#     spec = extractMelSpecs(scaled, target_sr, windowLength=0.05, frameshift=0.01)
    
#     # Save spectrogram
#     output_path = os.path.join(OUTPUT_PATH, f'{participant}_spec.npy')
#     np.save(output_path, spec)
#     print(f"  Saved spectrogram: {spec.shape}")

# Check we're using the right paths
print(f"BIDS path: {BIDS_PATH}")
print(f"Output path: {OUTPUT_PATH}")
print(f"Results path: {RESULTS_PATH}")
# Define paths
path_bids = BIDS_PATH # './SingleWordProductionDutch-iBIDS'  # Path to the BIDS dataset
path_output = OUTPUT_PATH #'./features'  # Path to save extracted features
path_results = RESULTS_PATH #'./results'  # Path to save results

visualizer = PhonemeFeatureVisualizer(output_dir='./phoneme_visualizations')

# Pipeline creation with PCA components management
use_augmentation = True
feature_extraction_method = 'high_gamma' #'high_gamma'  #
optimal_pca_components = 50  # Use your optimal value determined earlier

def modified_step5(pipeline):
    """Fixed version that properly aligns features and labels"""
    
    # Set up empty structures first
    pipeline.train = {'features': [], 'phoneme_labels': []}
    pipeline.test = {'features': [], 'phoneme_labels': []}
    
    # Try to get data batches directly
    if hasattr(pipeline, 'custom_decoder') and hasattr(pipeline, 'split_result'):
        # Get training batches
        for i in range(20):  # Get 20 batches
            try:
                batch = pipeline.custom_decoder.get_data_batch(
                    split_result=pipeline.split_result,
                    batch_type='train',
                    batch_size=32
                )
                
                # Extract features from EEG segments
                if 'eeg_segments' in batch and 'words' in batch:
                    # Process each segment individually
                    for j, eeg_seg in enumerate(batch['eeg_segments']):
                        try:
                            # Extract high gamma features
                            from extract_features import extractHG
                            feat = extractHG(eeg_seg, 1024)
                            
                            # Add each feature window with corresponding word label
                            for k in range(feat.shape[0]):
                                pipeline.train['features'].append(feat[k])
                                # Use the word for this segment
                                word = batch['words'][j] if j < len(batch['words']) else 'unknown'
                                pipeline.train['phoneme_labels'].append(word)
                        except Exception as e:
                            print(f"Failed to process segment {j}: {e}")
                            continue
            except Exception as e:
                print(f"Batch {i} failed: {e}")
                continue
    
    print(f"Features: {len(pipeline.train['features'])}, Labels: {len(pipeline.train['phoneme_labels'])}")
    
    # Ensure they match
    min_len = min(len(pipeline.train['features']), len(pipeline.train['phoneme_labels']))
    pipeline.train['features'] = pipeline.train['features'][:min_len]
    pipeline.train['phoneme_labels'] = pipeline.train['phoneme_labels'][:min_len]
    
    return pipeline.train, pipeline.test

# fix_and_pretrain_10patients.py
# Try to load existing pipeline, otherwise create new one
try:
    # Try loading existing pipeline
    pipeline = UnifiedPhonemePipeline.load_saved(path_results, method=feature_extraction_method)
    print(f"Loaded existing {feature_extraction_method} pipeline")
    
    # Check and update PCA components if needed
    current_pca = getattr(pipeline, 'pca_components', None)
    if current_pca != optimal_pca_components:
        print(f"Updating PCA components from {current_pca} to {optimal_pca_components}")
        pipeline.set_pca_components(optimal_pca_components)
        
        # Re-run data steps with new PCA components
        print("Re-processing data with updated PCA components...")
        pipeline.step4_initialize_detector()    
        
        pipeline.step5_accumulate_data()
        pipeline.step6_resolve_unknowns()
        pipeline.step7_filter_unknowns()
        
        # Save the updated pipeline
        pipeline.save()
        print(f"Updated and saved {feature_extraction_method} pipeline with {optimal_pca_components} PCA components")
    
except (FileNotFoundError, AttributeError, TypeError) as e:
    # No existing pipeline found, create new one
    print(f"No existing {feature_extraction_method} pipeline found. Creating new one...")
    
    pipeline = UnifiedPhonemePipeline(
        path_bids=path_bids,
        path_output=path_output,
        path_results=path_results,
        feature_extraction_method=feature_extraction_method,
        unknown_keep_ratio=0.1,
        channel_correlation_threshold=0.3,  
        prioritize_regions=True, 
        channel_selection='best_correlation',
        pca_components=optimal_pca_components, 
        debug_mode=True
    )
    
    # Run all steps
    print("Running pipeline steps...")
    pipeline.step1_initialize_decoder()
    pipeline.step2_stratify_participants()    
    pipeline.step3_create_split()
    pipeline.step4_initialize_detector() 
#     pipeline.step5_accumulate_data(
#             train_batches=20,  # Start with fewer batches
#             test_batches=1,   # Smaller test set
#             batch_size=32     # Smaller batch size
#         )
   
#     pipeline.step5_accumulate_data()
#     pipeline.step6_resolve_unknowns()
#     pipeline.step7_filter_unknowns()

    # Use modified step 5
    train_data, test_data = modified_step5(pipeline)
    pipeline.train = train_data
    pipeline.test = test_data

    # Now continue - but check we have data first
    if pipeline.train and len(pipeline.train['features']) > 0:
        pipeline.step6_resolve_unknowns()
        pipeline.step7_filter_unknowns()
        pipeline.save()
    else:
        print("No data accumulated - cannot continue")
    
    # Save the pipeline
    pipeline.save()
    print(f"Created and saved new {feature_extraction_method} pipeline with {optimal_pca_components} PCA components")

# The pipeline is now ready to use with optimal PCA components
print(f"Pipeline ready with {feature_extraction_method} features and {optimal_pca_components} PCA components")

import pickle
import numpy as np
import os
from collections import Counter

# Load the saved pipeline
saved_files = [f for f in os.listdir(RESULTS_PATH) if f.startswith('pipeline_') and f.endswith('.pkl')]
print(f"Found saved files: {saved_files}")

if saved_files:
    latest_file = sorted(saved_files)[-1]
    filepath = os.path.join(RESULTS_PATH, latest_file)
    
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    # Check structure
    print(f"\nLoaded {latest_file}")
    print(f"Keys in saved data: {data.keys()}")
    
    if 'train' in data and data['train']:
        train = data['train']
        print(f"\nTraining data:")
        print(f"  Features: {len(train.get('features', []))} samples")
        if train.get('features'):
            print(f"  Feature shape: {train['features'][0].shape}")
        print(f"  Labels: {len(train.get('phoneme_labels', []))} labels")
        
        # Check label distribution
        if 'phoneme_labels' in train:
            label_counts = Counter(train['phoneme_labels'])
            print(f"  Unique labels: {len(label_counts)}")
            print(f"  Top 5 labels: {label_counts.most_common(5)}")
    
    if 'test' in data and data['test']:
        test = data['test']
        print(f"\nTest data:")
        print(f"  Features: {len(test.get('features', []))}")
        print(f"  Labels: {len(test.get('phoneme_labels', []))}")

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import numpy as np

# Prepare data for sklearn
if train.get('features') and len(train['features']) > 0:
    # Flatten features to 2D array
    X_train = []
    for feat in train['features']:
        if len(feat.shape) == 1:
            X_train.append(feat)
        else:
            # If 2D, flatten or take mean
            X_train.append(feat.flatten())
    
    # Make all features same length (truncate or pad)
    max_len = max(len(f) for f in X_train)
    X_train_fixed = []
    for feat in X_train:
        if len(feat) < max_len:
            # Pad with zeros
            padded = np.zeros(max_len)
            padded[:len(feat)] = feat
            X_train_fixed.append(padded)
        else:
            X_train_fixed.append(feat[:max_len])
    
    X_train = np.array(X_train_fixed)
    
    # Encode labels
    le = LabelEncoder()
    y_train = le.fit_transform(train['phoneme_labels'])
    
    print(f"\nTraining data shape: {X_train.shape}")
    print(f"Number of classes: {len(le.classes_)}")
    
    # Train a simple model
    if len(X_train) > 10:  # Need at least some samples
        # Split for validation
        split_idx = int(0.8 * len(X_train))
        X_tr, X_val = X_train[:split_idx], X_train[split_idx:]
        y_tr, y_val = y_train[:split_idx], y_train[split_idx:]
        
        # Train model
        model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
        model.fit(X_tr, y_tr)
        
        # Evaluate
        train_pred = model.predict(X_tr)
        val_pred = model.predict(X_val)
        
        train_acc = accuracy_score(y_tr, train_pred)
        val_acc = accuracy_score(y_val, val_pred)
        
        print(f"\nModel Test Results:")
        print(f"  Training accuracy: {train_acc:.3f}")
        print(f"  Validation accuracy: {val_acc:.3f}")
        
        if train_acc > 0.8 and val_acc < 0.2:
            print("  WARNING: Severe overfitting - data might be problematic")
        elif train_acc < 0.2:
            print("  WARNING: Model not learning - check feature extraction")
        else:
            print("  Data appears reasonable for pre-training")

# Additional quality checks
if X_train is not None:
    # Check for NaN or infinite values
    has_nan = np.any(np.isnan(X_train))
    has_inf = np.any(np.isinf(X_train))
    
    print(f"\nData quality:")
    print(f"  Contains NaN: {has_nan}")
    print(f"  Contains Inf: {has_inf}")
    print(f"  Feature range: [{np.min(X_train):.3f}, {np.max(X_train):.3f}]")
    print(f"  Feature mean: {np.mean(X_train):.3f}")
    print(f"  Feature std: {np.std(X_train):.3f}")
    
    # Check if features are all zeros or constant
    if np.std(X_train) < 0.001:
        print("  WARNING: Features have very low variance")

phonetic_dict = PhoneticDictionary()
# Get FILTERED data (this is what you feed to the Markov model):
train_data = pipeline.get_training_data(filtered=True)  # or pipeline.train_filtered

visualizer.process_batches(train_data, method='high_gamma', band=(70, 150))

# 1. Check raw NWB files directly
io = NWBHDF5IO(os.path.join(path_bids, 'sub-04', 'ieeg', 'sub-04_task-wordProduction_ieeg.nwb'), 'r')
nwbfile = io.read()
raw_eeg = nwbfile.acquisition['iEEG'].data[:10240, :5]  # 10 seconds, 5 channels
io.close()

# 2. Analyze multiple channels
from scipy.signal import welch
import matplotlib.pyplot as plt

fig, axes = plt.subplots(5, 1, figsize=(10, 12))
for ch in range(5):
    freqs, psd = welch(raw_eeg[:, ch], fs=1024, nperseg=2048)
    axes[ch].semilogy(freqs, psd)
    axes[ch].axvline(200, color='r', linestyle='--', alpha=0.5)
    axes[ch].axvline(400, color='r', linestyle='--', alpha=0.5)
    axes[ch].set_title(f'Channel {ch}')
plt.tight_layout()
plt.show()

plt.figure(figsize=(15, 6))
# Check if consistent across all participants
for participant in ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-07', 'sub-08']:
#for participant in ['sub-01', 'sub-10']:
    io = NWBHDF5IO(f'{path_bids}/{participant}/ieeg/{participant}_task-wordProduction_ieeg.nwb', 'r')
    nwbfile = io.read()
    data = nwbfile.acquisition['iEEG'].data[:10240, 0]
    io.close()
    
    freqs, psd = welch(data, fs=1024, nperseg=2048)
    plt.semilogy(freqs, psd, label=participant)

plt.axvline(150, color='r', linestyle='--', alpha=0.3)
plt.axvline(450, color='r', linestyle='--', alpha=0.3)
plt.legend()
plt.show()

# # You could also visualize other frequency bands
#visualizer.process_batches(train_data, method='theta', band=(4, 8))
# visualizer.process_batches(train_data, method='alpha', band=(8, 13))
# visualizer.process_batches(train_data, method='beta', band=(13, 30))

# import os
# import re

# # Find all file references in Python files
# for root, dirs, files in os.walk('.'):
#     for file in files:
#         if file.endswith('.py'):
#             try:
#                 with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
#                     content = f.read()
#                     # Find potential file paths
#                     paths = re.findall(r'["\']([^"\']*\.[a-z]{3,4})["\']', content)
#                     if paths:
#                         print(f"\n{file} references:")
#                         for p in paths:
#                             print(f"  - {p}")
#             except Exception as e:
#                 print(f"Could not read {file}: {e}")

# fix_and_pretrain_10patients.py

# Pipeline creation with PCA components management
use_augmentation = True
feature_extraction_method = 'high_gamma' #'high_gamma'  #
optimal_pca_components = 49  # Use your optimal value determined earlier

# Try to load existing pipeline, otherwise create new one
try:
    # Try loading existing pipeline
    pipeline = UnifiedPhonemePipeline.load_saved(path_results, method=feature_extraction_method)
    print(f"Loaded existing {feature_extraction_method} pipeline")
    
#     # Check and update PCA components if needed
#     current_pca = getattr(pipeline, 'pca_components', None)
#     if current_pca != optimal_pca_components:
#         print(f"Updating PCA components from {current_pca} to {optimal_pca_components}")
#         pipeline.set_pca_components(optimal_pca_components)
        
#         # Re-run data steps with new PCA components
#         print("Re-processing data with updated PCA components...")
#         pipeline.step4_initialize_detector()    
        
#         pipeline.step5_accumulate_data()
#         pipeline.step6_resolve_unknowns()
#         pipeline.step7_filter_unknowns()
        
#         # Save the updated pipeline
#         pipeline.save()
#         print(f"Updated and saved {feature_extraction_method} pipeline with {optimal_pca_components} PCA components")
    
except (FileNotFoundError, AttributeError, TypeError) as e:
    # No existing pipeline found, create new one
    print(f"No existing {feature_extraction_method} pipeline found. Creating new one...")
    
#     pipeline = UnifiedPhonemePipeline(
#         path_bids=path_bids,
#         path_output=path_output,
#         path_results=path_results,
#         feature_extraction_method=feature_extraction_method,
#         unknown_keep_ratio=0.1,
#         channel_correlation_threshold=0.3,  # ADD THIS
#         prioritize_regions=True,  # ADD THIS
#         channel_selection='best_correlation',
#         pca_components=optimal_pca_components,  # Set optimal PCA components
#         debug_mode=True
#     )
    
#     # Run all steps
#     print("Running pipeline steps...")
#     pipeline.step1_initialize_decoder()
#     pipeline.step2_stratify_participants()    
#     pipeline.step3_create_split()
#     pipeline.step4_initialize_detector() 
#     pipeline.step5_accumulate_data(
#             train_batches=2,  # Start with fewer batches
#             test_batches=1,   # Smaller test set
#             batch_size=32     # Smaller batch size
#         )
   
#     #pipeline.step5_accumulate_data()
#     pipeline.step6_resolve_unknowns()
#     pipeline.step7_filter_unknowns()
    
#     # Save the pipeline
#     pipeline.save()
    print(f"Created and saved new {feature_extraction_method} pipeline with {optimal_pca_components} PCA components")

# The pipeline is now ready to use with optimal PCA components
print(f"Pipeline ready with {feature_extraction_method} features and {optimal_pca_components} PCA components")

pipeline_10 = UnifiedPhonemePipeline(
    path_bids=BIDS_PATH,
    path_output=OUTPUT_PATH,
    path_results=RESULTS_PATH,
    feature_extraction_method='high_gamma',
    unknown_keep_ratio=0.1,
    channel_correlation_threshold=0.3,
    prioritize_regions=False,  # Simplify for now
    channel_selection='all',     # Use all channels
    pca_components=50,
    use_phoneme_groups=True,
    debug_mode=True
)

# Run the pipeline
print("Running pipeline steps...")
pipeline_10.step1_initialize_decoder()
pipeline_10.step2_stratify_participants()
pipeline_10.step3_create_split()
pipeline_10.step4_initialize_detector()

# Try with more batches to get actual data
train_data, test_data = pipeline_10.step5_accumulate_data(
    train_batches=10,  # More batches
    test_batches=3,
    batch_size=32
)

if train_data and len(train_data.get('features', [])) > 0:
    print(f"Successfully accumulated {len(train_data['features'])} training samples")
    
    pipeline_10.step6_resolve_unknowns()
    pipeline_10.step7_filter_unknowns()
    pipeline_10.step8_convert_to_groups()
    
    # Save the pipeline AND the model
    pipeline_10.save()
    
    # Train a model and save it
    from diverse_models import SimplePhonemeModels
    
    models = SimplePhonemeModels(output_dir=RESULTS_PATH)
    
    # Get filtered training data
    train_filtered = pipeline_10.get_training_data(filtered=True)
    test = pipeline_10.get_test_data()
    
    # Train Markov model
    markov_model, markov_results = models.train_markov_model(
        train_data=train_filtered,
        test_data=test,
        use_groups=True
    )
    
    # Save the pre-trained model
    import pickle
    model_path = os.path.join(RESULTS_PATH, 'pretrained_markov_10patients.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': markov_model,
            'results': markov_results,
            'pca_models': pipeline_10.custom_decoder.pca_models if hasattr(pipeline_10.custom_decoder, 'pca_models') else None
        }, f)
    
    print(f"Saved pre-trained model to {model_path}")
else:
    print("Failed to accumulate training data")
