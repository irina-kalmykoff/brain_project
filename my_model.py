# Converted from my_model.ipynb

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
from phoneme_model import PhonemeDecoderModel

# Define paths
path_bids = './SingleWordProductionDutch-iBIDS'  # Path to the BIDS dataset
path_output = './features'  # Path to save extracted features
path_results = './results'  # Path to save results

# This decoder extends the baseline BrainAudioDecoder with additional functionality
# for phoneme-level analysis and provides stratification capabilities for participants.
custom_decoder = CustomBrainAudioDecoder(
    path_bids=path_bids,      # Path to the BIDS-formatted dataset
    path_output=path_output,  # Path to save extracted features
    path_results=path_results # Path to save results and visualizations
)
custom_decoder.enable_debug() # Enable detailed logging for troubleshooting

# This loads metadata for all participants and extracts high gamma features
# from the EEG data, which will be used for model training.
custom_decoder.load_participants()
custom_decoder.extract_features_all_participants()

# This identifies which channels are most relevant for speech decoding
# and will be used to stratify participants by signal quality.
custom_decoder.analyze_channels_across_participants()

# Groups participants into high, medium, and low quality based on
# the strength of speech-related signals in their EEG data.
participant_strata = custom_decoder.stratify_participants_by_channel_quality(
    channel_correlation_threshold=0.1  # Minimum correlation for a channel to be considered relevant
)

# # Call segment_data_by_words
# # Choose a participant to segment (e.g., sub-01)
# participant_id = 'sub-01'
# word_segments = custom_decoder.segment_data_by_words(
#     participant_id='sub-01',
#     pre_onset_ms=200,
#     post_offset_ms=200,
#     handle_overlaps='adjust'  # Options: 'adjust', 'flag', 'skip', 'allow'
# )

# Ensures that the training and test sets have similar distributions of
# participant quality and word frequencies for better generalization.

# contains both training and test splits, structured as
# split_result = {
#     'train': {participant_id: {word: [instance_indices], ...}, ...},
#     'test': {participant_id: {word: [instance_indices], ...}, ...},
#     'statistics': {...},
#     'word_segments_dict': {...}
# }
split_result = custom_decoder.create_stratified_cross_word_split(
    participant_strata=participant_strata,  # Participant quality strata
    test_ratio=0.2,                         # Proportion of data for testing
    min_word_freq=1,                        # Minimum word frequency to include
    random_seed=42                          # For reproducibility
)

# # Get a batch of training data
train_batch = custom_decoder.get_data_batch(
    split_result=split_result,
    batch_type='train',
    participant_ids=None,  # Use all available participants
    max_instances_per_word=10,  # Limit instances per word for balance
    balanced_sampling=True,
    batch_size=32
)

# Prepare features for model training
model_data = custom_decoder.prepare_word_model_data(
    batch=train_batch,
    feature_extraction_method='high_gamma',
    temporal_context=True,
    standardize=True,
    pca_components=50
)

print(f"Prepared {len(model_data['features'])} feature sets for model training")
print(f"Feature dimensions: {model_data['features'][0].shape}")

# This component is responsible for segmenting continuous speech signals
# into individual phonemes by detecting acoustic boundaries.
detector = AcousticChangeDetector(
    min_segment_duration=0.03,    # Minimum phoneme duration in seconds
    max_segment_duration=0.3,     # Maximum phoneme duration in seconds
    distance_metric='cosine',     # Method to measure acoustic change
    smoothing_window=3,           # Window for smoothing distance curve
    peak_threshold=0.5,           # Threshold for boundary detection
    decoder=custom_decoder,       # Reference to the custom decoder
    debug_mode=True               # Enable detailed logging
)

# This allows the detector to access the train/test split when accumulating phoneme data
detector.split_result = split_result

# accumulated_data = detector.accumulate_phoneme_data(
#     num_batches=5,
#     batch_size=32,
#     feature_extraction_method='high_gamma'
# )

# print(f"Accumulated {accumulated_data['metadata']['n_phonemes']} phoneme segments")
# print(f"Found {accumulated_data['metadata']['unique_phonemes']} unique phonemes")

# converted_acc_data = {
#     'phoneme_labels': accumulated_data['phoneme_labels'],
#     'phoneme_spectrogram_segments': accumulated_data['spectrograms'] if accumulated_data['spectrograms'] is not None else [],
#     'phoneme_words': accumulated_data['phoneme_words'],
#     'phoneme_positions': [0] * len(accumulated_data['phoneme_labels']) if 'phoneme_positions' not in accumulated_data else accumulated_data['phoneme_positions'],
#     'phoneme_participant_ids': accumulated_data['phoneme_participant_ids'] if 'phoneme_participant_ids' in accumulated_data else ['unknown'] * len(accumulated_data['phoneme_labels'])
# }

# Processes multiple batches of data to build a comprehensive training set
# of phoneme segments with their corresponding EEG features
train = detector.accumulate_phoneme_data(
    num_batches=5,                         # Number of batches to process
    batch_size=32,                         # Instances per batch
    feature_extraction_method='high_gamma' # Method for extracting EEG features
)
print(f"Accumulated {train['metadata']['n_phonemes']} training phoneme segments")
print(f"Found {train['metadata']['unique_phonemes']} unique phonemes")

# Similar to training data accumulation but for the test set
test = detector.accumulate_phoneme_data(
    num_batches=3,                         # Fewer batches for test set
    batch_size=32,                         # Instances per batch
    feature_extraction_method='high_gamma' # Same feature extraction method
)
print(f"Accumulated {test['metadata']['n_phonemes']} test phoneme segments")

# 3. Initialize the validator with a reference to the detector
validator = PhonemeValidator(detector=detector)
validator.enable_debug()

# Extract phoneme segments
df  = train
converted_data = {
    'phoneme_labels': df['phoneme_labels'],
    'phoneme_spectrogram_segments': df['spectrograms'] if df['spectrograms'] is not None else [],
    'phoneme_words': df['phoneme_words'],
    'phoneme_positions': [0] * len(df['phoneme_labels']) if 'phoneme_positions' not in df else df['phoneme_positions'],
    'phoneme_participant_ids': df['phoneme_participant_ids'] if 'phoneme_participant_ids' in df else ['unknown'] * len(df['phoneme_labels'])
}


phoneme_segments = validator.extract_phoneme_segments_from_batch(converted_data)

# Check if we found any phonemes
if phoneme_segments:
    print(f"Found {len(phoneme_segments)} phonemes")
    print("Available phonemes:", list(phoneme_segments.keys()))
    
    # Find the phoneme with the most segments
    most_common_phoneme = max(phoneme_segments.keys(), key=lambda p: len(phoneme_segments[p]))
    print(f"Most common phoneme: '{most_common_phoneme}' with {len(phoneme_segments[most_common_phoneme])} segments")
    
    # Visualize the most common phoneme
    validator.visualize_phoneme_segments(
        phoneme_segments=phoneme_segments,
        phoneme=most_common_phoneme,
        max_examples=5
    )
else:
    print("No phonemes found")

# 2. Try to resolve unknown phonemes
resolved_batch = validator.resolve_unknown_phonemes(converted_data)

# 3. Extract segments from the resolved batch
resolved_segments = validator.extract_phoneme_segments_from_batch(resolved_batch)

# 4. Compare the number of resolved phonemes
print(f"Original segments: {len(phoneme_segments)} phonemes")
print(f"Resolved segments: {len(resolved_segments)} phonemes")

# 5. Visualize a phoneme with proper position display
if resolved_segments:
    top_phoneme = max(resolved_segments.keys(), key=lambda p: len(resolved_segments[p]))
    validator.visualize_phoneme_segments(
        phoneme_segments=resolved_segments,
        phoneme=top_phoneme,
        max_examples=5
    )

def visualize_phoneme_category(phoneme_batch, phoneme_label, max_examples=5):
    """Visualize examples of a specific phoneme"""
    import matplotlib.pyplot as plt
    
    # Find all segments for this phoneme
    indices = [i for i, label in enumerate(phoneme_batch['phoneme_labels']) 
               if label == phoneme_label]
    
    if not indices:
        print(f"No segments found for phoneme '{phoneme_label}'")
        return None
    
    # Limit to max_examples
    if len(indices) > max_examples:
        import random
        indices = random.sample(indices, max_examples)
    
    # Create figure
    fig, axs = plt.subplots(len(indices), 1, figsize=(10, 3*len(indices)))
    if len(indices) == 1:
        axs = [axs]
    
    # Plot each segment
    for i, idx in enumerate(indices):
        segment = phoneme_batch['phoneme_spectrogram_segments'][idx]
        word = phoneme_batch['phoneme_words'][idx]
        position = phoneme_batch['phoneme_positions'][idx]
        
        im = axs[i].imshow(segment.T, aspect='auto', origin='lower', cmap='viridis')
        axs[i].set_title(f"Phoneme '{phoneme_label}' in word '{word}' (position {position})")
        axs[i].set_ylabel('Frequency Bin')
        
        # Only add x-label to bottom plot
        if i == len(indices) - 1:
            axs[i].set_xlabel('Time Frame')
        
        # Add colorbar
        plt.colorbar(im, ax=axs[i])
    
    plt.tight_layout()
    plt.show()
    
    return fig

# Visualize a few interesting phonemes
for phoneme in ['m', 'ɛ', 'œy', 'n', 't']:
    print(f"\nVisualizing phoneme '{phoneme}':")
    visualize_phoneme_category(converted_data, phoneme)

# Structure the data correctly for validation
all_results = {}
for i, word in enumerate(converted_data['phoneme_words']):
    if word not in all_results:
        all_results[word] = []
    
    # Create a result entry for this instance
    result = {
        'segments': [converted_data['phoneme_spectrogram_segments'][i]],
        'boundaries': [0, converted_data['phoneme_spectrogram_segments'][i].shape[0]],
        'word': word,
        'participant_id': converted_data['phoneme_participant_ids'][i] if 'phoneme_participant_ids' in converted_data else 'unknown'
    }
    all_results[word].append(result)

# Now validate with properly structured data
validation_results = validator.validate_phoneme_consistency(
    all_results=all_results,
    min_occurrences=2
)
validation_results

# For training data
train_converted = {
    'phoneme_labels': train['phoneme_labels'],
    'phoneme_spectrogram_segments': train.get('spectrograms', []),
    'phoneme_words': train['phoneme_words'],
    'phoneme_positions': train.get('phoneme_positions', [0] * len(train['phoneme_labels'])),
    'phoneme_participant_ids': train.get('phoneme_participant_ids', ['unknown'] * len(train['phoneme_labels']))
}

print(f"Training data before resolution: {train_converted['phoneme_labels'].count('?')} unknown phonemes")
resolved_train = validator.resolve_unknown_phonemes(train_converted)
print(f"Training data after resolution: {resolved_train['phoneme_labels'].count('?')} unknown phonemes")

# Update the training data with resolved phonemes
train['phoneme_labels'] = resolved_train['phoneme_labels']

# For test data
test_converted = {
    'phoneme_labels': test['phoneme_labels'],
    'phoneme_spectrogram_segments': test.get('spectrograms', []),
    'phoneme_words': test['phoneme_words'],
    'phoneme_positions': test.get('phoneme_positions', [0] * len(test['phoneme_labels'])),
    'phoneme_participant_ids': test.get('phoneme_participant_ids', ['unknown'] * len(test['phoneme_labels']))
}

print(f"Test data before resolution: {test_converted['phoneme_labels'].count('?')} unknown phonemes")
resolved_test = validator.resolve_unknown_phonemes(test_converted)
print(f"Test data after resolution: {resolved_test['phoneme_labels'].count('?')} unknown phonemes")

# Update the test data with resolved phonemes
test['phoneme_labels'] = resolved_test['phoneme_labels']

# Prepare the input features (EEG segments) and target labels (phonemes)
# train_features = train['features']
# train_labels = train['phoneme_labels']
# test_features = test['features']
# test_labels = test['phoneme_labels']

# This neural network model will learn to map EEG signals to phonemes
phoneme_model = PhonemeDecoderModel(
    model_type='lstm_cnn',
    output_dir=os.path.join(path_results, 'phoneme_model'),
    debug_mode=True
)

# This handles all preprocessing, training, and evaluation in a single call
results = phoneme_model.train_with_accumulated_data(
    train_accumulated_data=train,  # Training data
    test_accumulated_data=test,    # Test data
    epochs=50,                                      # Maximum training epochs
    batch_size=32,                                  # Batch size for training
    patience=10,                                     # Early stopping patience
    handle_unseen_phonemes='filter',
    resolve_unknown=True
)

# Step 13: Evaluate model performance
evaluation = results['evaluation']
print(f"Phoneme model accuracy: {evaluation['accuracy']:.4f}")

# Step 14: Compare with baseline (optional)
# Evaluate the baseline BrainAudioDecoder for comparison
baseline_decoder = BrainAudioDecoder(
    path_bids=path_bids,
    path_output=path_output,
    path_results=path_results
)
baseline_results = baseline_decoder.train_test_model(
    participant_id='sub-08',  # Choose a representative participant
    save_audio=False          # Don't save audio reconstructions
)
baseline_accuracy = np.mean(baseline_results['correlations'])
print(f"Baseline model correlation: {baseline_accuracy:.4f}")

phoneme_model.plot_confusion_matrix(results['evaluation']['confusion_matrix'])

# Step 15: Generate visualizations (optional)
# Create plots comparing model performance, confusion matrices, etc.
phoneme_model.plot_training_history()
