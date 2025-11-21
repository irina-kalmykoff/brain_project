# Converted from SimplifiedPhonemeModel.ipynb

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import Audio, display
from pynwb import NWBHDF5IO

import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed, IntSlider, FloatSlider, Dropdown, Checkbox

from brain_audio_decoder import BrainAudioDecoder
from custom_decoder import CustomBrainAudioDecoder
from brain_audio_decoder_viz import BrainAudioDecoderViz
from acoustic_change_detector import AcousticChangeDetector
from phoneme_validator import PhonemeValidator
from simplified_phoneme_model import SimplifiedPhonemeModel
from pipeline import UnifiedPhonemePipeline

# Define paths
path_bids = './SingleWordProductionDutch-iBIDS'  # Path to the BIDS dataset
path_output = './features'  # Path to save extracted features
path_results = './results'  # Path to save results

# Pipeline creation with PCA components management
use_augmentation = True
feature_extraction_method = 'multi_band' #'high_gamma'  #
optimal_pca_components = 30  # Use your optimal value determined earlier

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
        channel_correlation_threshold=0.3,  # ADD THIS
        prioritize_regions=True,  # ADD THIS
        channel_selection='best_correlation',
        pca_components=optimal_pca_components,  # Set optimal PCA components
        debug_mode=False
    )
    
    # Run all steps
    print("Running pipeline steps...")
    pipeline.step1_initialize_decoder()
    pipeline.step2_stratify_participants()    
    pipeline.step3_create_split()
    pipeline.step4_initialize_detector()    
    pipeline.step5_accumulate_data()
    pipeline.step6_resolve_unknowns()
    pipeline.step7_filter_unknowns()
    
    # Save the pipeline
    pipeline.save()
    print(f"Created and saved new {feature_extraction_method} pipeline with {optimal_pca_components} PCA components")

# The pipeline is now ready to use with optimal PCA components
print(f"Pipeline ready with {feature_extraction_method} features and {optimal_pca_components} PCA components")

from phonetic_dictionary import PhoneticDictionary
phonetic_dict = PhoneticDictionary()

# Get FILTERED data (this is what you feed to the Markov model):
train_data = pipeline.get_training_data(filtered=True)  # or pipeline.train_filtered

# Extract all components:
filtered_features = train_data['features']
filtered_labels = train_data['phoneme_labels']
filtered_words = train_data['phoneme_words']
filtered_participants = train_data['phoneme_participant_ids']

test_features = pipeline.test['features']
test_labels = pipeline.test['phoneme_labels']

# Create a minimal version that doesn't try to reinvent your pipeline
simplified_model = SimplifiedPhonemeModel(
        phonetic_dict=phonetic_dict,
        output_dir=os.path.join(path_results, 'simplified_phoneme_model'),
        debug_mode=True
    )

# Get FILTERED training data (this is important)
train_data = pipeline.get_training_data(filtered=True)
test_data = pipeline.get_test_data()
train_participant_ids = train_data.get('phoneme_participant_ids', None)
test_participant_ids = test_data.get('phoneme_participant_ids', None)

# First, check if the pipeline has completed the necessary steps
if not hasattr(pipeline, 'phonetic_dict'):
    # The phonetic dict might be in the detector
    if hasattr(pipeline, 'detector') and hasattr(pipeline.detector, 'phonetic_dict'):
        pipeline.phonetic_dict = pipeline.detector.phonetic_dict
    else:
        # Need to initialize it manually
        from phonetic_dictionary import PhoneticDictionary
        pipeline.phonetic_dict = PhoneticDictionary()

# Check if we have data after filtering
if not train_data or 'phoneme_labels' not in train_data or not train_data['phoneme_labels']:
    print("Warning: No data in filtered training set. Using unfiltered data instead.")
    # Fall back to unfiltered data
    train_data = pipeline.get_training_data(filtered=False)

# Map individual phonemes to phoneme groups
train_group_labels = []
for phoneme in train_data['phoneme_labels']:
    # Try to find the phoneme in the phoneme_to_group mapping
    if phoneme in simplified_model.phoneme_to_group:
        group = simplified_model.phoneme_to_group[phoneme]
    else:
        # If not directly mapped, search in the phoneme groups
        group = 'unknown'
        for group_name, phonemes in simplified_model.phoneme_groups.items():
            if phoneme in phonemes:
                group = group_name
                break
    train_group_labels.append(group)

# Do the same for test data
test_group_labels = []
for phoneme in test_data['phoneme_labels']:
    if phoneme in simplified_model.phoneme_to_group:
        group = simplified_model.phoneme_to_group[phoneme]
    else:
        group = 'unknown'
        for group_name, phonemes in simplified_model.phoneme_groups.items():
            if phoneme in phonemes:
                group = group_name
                break
    test_group_labels.append(group)

# Print the group distribution after mapping
from collections import Counter
group_counts = Counter(train_group_labels)
print("Phoneme group distribution after mapping:")
for group, count in group_counts.most_common():
    print(f"  {group}: {count}")

# Before training, check if we have enough data for each group
# If not, we might need to lower the min_occurrences
min_count = min(group_counts.values()) if group_counts else 0
recommended_min_occurrences = max(1, min(3, min_count))  # Between 1 and 3, but not more than min_count

print(f"Recommended min_occurrences: {recommended_min_occurrences}")

# 6. Train using your phoneme groups
results = simplified_model.train_with_grouped_data(
        train_features=train_data['features'],
        train_group_labels=train_group_labels,
        train_participant_ids=train_participant_ids,
        test_features=test_features,
        test_group_labels=test_group_labels,
        min_occurrences=recommended_min_occurrences,  # Use the recommended value
        test_participant_ids=test_participant_ids,
        epochs=50,
        batch_size=32,
        patience=10
    )

# # Initialize decoder
# custom_decoder = CustomBrainAudioDecoder(
#     path_bids=path_bids,
#     path_output=path_output,
#     path_results=path_results,
#     win_length=0.05,
#     frameshift=0.01,
#     model_order=4,
#     step_size=5,
#     n_components=50
# )
# # 7. Evaluate performance
# print(f"Simplified phoneme group model accuracy: {results['accuracy']:.4f}")
    
# # Compare to baseline
# baseline_results = custom_decoder.train_test_model(
#         participant_id='sub-08',
#         save_audio=False
#     )
# baseline_correlation = np.mean(baseline_results['correlations'])
# print(f"Baseline model correlation: {baseline_correlation:.4f}") 

evaluation_results = results
# First, check what your evaluation results contain
print("Evaluation results keys:", evaluation_results.keys())
print("Confusion matrix shape:", evaluation_results['confusion_matrix'].shape)

# Then try calling with explicit parameters
saved_files = pipeline.visualize_model_results(
    model=simplified_model,
    eval_results={
        'confusion_matrix': evaluation_results['confusion_matrix'],
        'accuracy': evaluation_results['accuracy'],
        'labels': evaluation_results['group_names']  # Make sure labels are included
    },
    title_prefix="Simplified Model",
    save_dir=path_results,
    show_plot=True
)

print("Saved files:", saved_files)

class_metrics = simplified_model.analyze_class_performance(
        results['true_groups'], 
        results['predicted_groups'],
        results['predictions']
    )   


# # Write out examples of predictions
# examples = simplified_model.analyze_examples(
#         test_data['features'],
#         results['true_groups'],
#         results['predicted_groups'],
#         test_data['phoneme_participant_ids'],
#         max_examples=3
#     )  


# Get FILTERED data (this is what you feed to the Markov model):
balance_strategy = 'weighted' #'undersample'#

train_data = pipeline.get_training_data(filtered=True)  # or pipeline.train_filtered

balanced_train_data = pipeline.balance_training_data(
                    train_data, 
                    balance_strategy = balance_strategy # 'weighted' #'undersample'#
                )

# Extract all components:
balanced_features = balanced_train_data['features']
balanced_labels = balanced_train_data['phoneme_labels']
balanced_words = balanced_train_data['phoneme_words']
balanced_participants = balanced_train_data['phoneme_participant_ids']

# test_features = pipeline.test['features']
# test_labels = pipeline.test['phoneme_labels']


balanced_train_parameters = {
    'features': filtered_features,
    'phoneme_labels': filtered_labels,
    'words': filtered_words,
    'participant_ids': filtered_participants
}

balanced_test_parameters = {
    'features': test_features,
    'true_labels': test_labels,  # Note: evaluate() uses 'true_labels' not 'phoneme_labels'
    'use_viterbi': True
}

# 6. Train using balanced groups
results = simplified_model.train_with_grouped_data(
        train_features = balanced_features,
        train_group_labels = balanced_labels,
        train_participant_ids = balanced_participants,
        test_features=test_features,
        test_group_labels=test_group_labels,
        min_occurrences=recommended_min_occurrences,  # Use the recommended value
        test_participant_ids=test_participant_ids,
        epochs=50,
        batch_size=32,
        patience=10
    )

evaluation_results = results
# First, check what your evaluation results contain
print("Evaluation results keys:", evaluation_results.keys())
print("Confusion matrix shape:", evaluation_results['confusion_matrix'].shape)

# Then try calling with explicit parameters
saved_files = pipeline.visualize_model_results(
    model=simplified_model,
    eval_results={
        'confusion_matrix': evaluation_results['confusion_matrix'],
        'accuracy': evaluation_results['accuracy'],
        'labels': evaluation_results['group_names']  # Make sure labels are included
    },
    title_prefix="Simplified Model",
    save_dir=path_results,
    show_plot=True
)

print("Saved files:", saved_files)
