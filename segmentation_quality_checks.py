# Converted from segmentation_quality_checks.ipynb

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
from phonetic_dictionary import PhoneticDictionary
from hybrid_phoneme_models import HybridPhonemeModels
from unified_phoneme_pipeline import UnifiedPhonemePipeline
from markov_phoneme_model import MarkovPhonemeModel

# Define paths
path_bids = './SingleWordProductionDutch-iBIDS'
path_output = './features'
path_results = './results'

# Create directories if they don't exist
os.makedirs(path_output, exist_ok=True)
os.makedirs(path_results, exist_ok=True)

def run_complete_pipeline(path_bids, path_output, path_results, 
                         feature_extraction_method='high_gamma',
                         use_augmentation=True):
    """
    Run the complete unified pipeline.
    """
    # Initialize pipeline
    pipeline = UnifiedPhonemePipeline(
        path_bids=path_bids,
        path_output=path_output,
        path_results=path_results,
        feature_extraction_method=feature_extraction_method,
        debug_mode=False
    )
    
    # Run all steps
    pipeline.setup(
        channel_correlation_threshold=0.1,
        test_ratio=0.2,
        min_word_freq=1,
        random_seed=42
    )
    
    pipeline.accumulate_data(
        train_batches=5,
        test_batches=3,
        batch_size=32
    )
    
    pipeline.resolve_phonemes()
    
    pipeline.map_to_groups()
    
    if use_augmentation:
        pipeline.apply_augmentation(
            augmentation_factor=3,
            min_samples_per_class=15,
            use_cross_participant=True
        )
    
    pipeline.validate_data()
    
    # Save state
    pipeline.save_pipeline_state()
    
    # Get processed data for model training
    train_data = pipeline.get_training_data(use_augmented=True, use_groups=True)
    test_data = pipeline.get_test_data(use_groups=True)
    
    # Print summary
    summary = pipeline.get_summary()
    print("\n" + "="*60)
    print("PIPELINE SUMMARY")
    print("="*60)
    print(f"Run ID: {summary['run_id']}")
    print(f"Participants: {summary['participants']}")
    print(f"Data samples: {summary['data']}")
    
    return pipeline, train_data, test_data


# Access all intermediate variables:
def access_pipeline_variables(pipeline):
    """
    Example of accessing all intermediate variables from the pipeline.
    """
    # Access decoder
    decoder = pipeline.custom_decoder
    
    # Access detector
    detector = pipeline.detector
    
    # Access validator
    validator = pipeline.validator
    
    # Access participant strata
    participant_strata = pipeline.pipeline_state['participant_strata']
    
    # Access split result
    split_result = pipeline.pipeline_state['split_result']
    
    # Access raw accumulated data
    train_raw = pipeline.pipeline_state['train_raw']
    test_raw = pipeline.pipeline_state['test_raw']
    
    # Access resolved data
    train_resolved = pipeline.pipeline_state['train_resolved']
    test_resolved = pipeline.pipeline_state['test_resolved']
    
    # Access augmented data
    train_augmented = pipeline.pipeline_state['train_augmented']
    
    # Access group labels
    train_group_labels = pipeline.pipeline_state['train_group_labels']
    test_group_labels = pipeline.pipeline_state['test_group_labels']
    
    # Access validation results
    validation_results = pipeline.pipeline_state['validation_results']
    
    return {
        'decoder': decoder,
        'detector': detector,
        'validator': validator,
        'participant_strata': participant_strata,
        'split_result': split_result,
        'train_raw': train_raw,
        'test_raw': test_raw,
        'train_resolved': train_resolved,
        'test_resolved': test_resolved,
        'train_augmented': train_augmented,
        'train_group_labels': train_group_labels,
        'test_group_labels': test_group_labels,
        'validation_results': validation_results
    }

pipeline, train_data, test_data = run_complete_pipeline(
    path_bids=path_bids,
    path_output=path_output,
    path_results=path_results,
    feature_extraction_method='multi_band',
    use_augmentation=True
)

train_data = pipeline.get_training_data(use_augmented=True, use_groups=False)
test_data = pipeline.get_test_data(use_groups=False)

vars = access_pipeline_variables(pipeline)
decoder = vars['decoder']
detector = vars['detector']
participant_strata = vars['participant_strata']
train_augmented = pipeline.pipeline_state['train_augmented']
split_result = pipeline.pipeline_state['split_result']

# Train your models with the processed data:
markov_model = MarkovPhonemeModel(
    phonetic_dict=pipeline.phonetic_dict,
    order=2,
    output_dir=os.path.join(path_results, 'markov_model'),
    debug_mode=True
)

# Filter unknowns just like in your working code
filtered_features = []
filtered_labels = []
filtered_words = []
filtered_participants = []

for i, label in enumerate(train_data['labels']):
    group = markov_model.phoneme_to_group.get(label, 'unknown')
    
    # Keep all non-unknown, but only keep 10% of unknown
    if group != 'unknown' or np.random.random() < 0.1:
        filtered_features.append(train_data['features'][i])
        filtered_labels.append(label)
        filtered_words.append(train_data['words'][i] if train_data['words'] else None)
        filtered_participants.append(train_data['participant_ids'][i] if train_data['participant_ids'] else 'unknown')


# Train the Markov model
markov_training_results = markov_model.train(
    features=filtered_features,
    phoneme_labels=filtered_labels,  # These are individual phonemes, not groups
    words=filtered_words,
    participant_ids=filtered_participants
)

# Evaluate
eval_results = markov_model.evaluate(
    features=test_data['features'],
    true_labels=test_data['labels'],  # Individual phonemes
    use_viterbi=True
)

print(f"Markov model accuracy: {eval_results['accuracy']:.4f}")

saved_path = pipeline.save_pipeline_state()  # Saves automatically to pipeline.run_dir
print(f"Pipeline saved to: {saved_path}")

def validate_phoneme_segmentation(detector, word_segments_dict, phonetic_dict):
    """
    Comprehensive validation of phoneme segmentation quality
    """
    validation_results = {
        'segmentation_accuracy': {},
        'boundary_consistency': {},
        'duration_analysis': {},
        'alignment_issues': []
    }
    
    # 1. Check if number of detected segments matches expected phonemes
    print("=" * 60)
    print("1. SEGMENTATION COUNT VALIDATION")
    print("=" * 60)
    
    correct_counts = 0
    total_words = 0
    
    for participant_id, segments in word_segments_dict.items():
        for word, word_info in segments.get('words', {}).items():
            if word in phonetic_dict:
                expected_phonemes = phonetic_dict.count_phonemes(word)
                
                for instance in word_info.get('instances', []):
                    if 'spectrogram_segment' in instance:
                        # Detect boundaries for this instance
                        result = detector.detect_boundaries(
                            instance['spectrogram_segment'],
                            word=word
                        )
                        
                        detected_segments = len(result['segments'])
                        total_words += 1
                        
                        if detected_segments == expected_phonemes:
                            correct_counts += 1
                        else:
                            validation_results['alignment_issues'].append({
                                'word': word,
                                'participant': participant_id,
                                'expected': expected_phonemes,
                                'detected': detected_segments
                            })
    
    accuracy = correct_counts / total_words if total_words > 0 else 0
    validation_results['segmentation_accuracy']['overall'] = accuracy
    print(f"Segmentation accuracy: {accuracy:.2%} ({correct_counts}/{total_words})")
    
    # 2. Check phoneme duration consistency
    print("\n" + "=" * 60)
    print("2. PHONEME DURATION ANALYSIS")
    print("=" * 60)
    
    phoneme_durations = {}
    
    for participant_id, segments in word_segments_dict.items():
        for word, word_info in segments.get('words', {}).items():
            if word in phonetic_dict:
                phonemes = phonetic_dict.extract_phonemes(word)
                
                for instance in word_info.get('instances', []):
                    if 'spectrogram_segment' in instance:
                        result = detector.detect_boundaries(
                            instance['spectrogram_segment'],
                            word=word
                        )
                        
                        if len(result['segments']) == len(phonemes):
                            for phoneme, segment in zip(phonemes, result['segments']):
                                if phoneme not in phoneme_durations:
                                    phoneme_durations[phoneme] = []
                                phoneme_durations[phoneme].append(segment.shape[0])
    
    # Analyze duration consistency
    for phoneme, durations in phoneme_durations.items():
        if len(durations) > 1:
            mean_dur = np.mean(durations)
            std_dur = np.std(durations)
            cv = std_dur / mean_dur if mean_dur > 0 else float('inf')
            
            validation_results['duration_analysis'][phoneme] = {
                'mean': mean_dur,
                'std': std_dur,
                'cv': cv,
                'n_samples': len(durations)
            }
            
            if cv > 0.5:  # High variability
                print(f"High variability for '{phoneme}': CV={cv:.2f}")
    
    # 3. Check boundary detection consistency across same words
    print("\n" + "=" * 60)
    print("3. BOUNDARY CONSISTENCY CHECK")
    print("=" * 60)
    
    word_boundaries = {}
    
    for participant_id, segments in word_segments_dict.items():
        for word, word_info in segments.get('words', {}).items():
            if word not in word_boundaries:
                word_boundaries[word] = []
            
            for instance in word_info.get('instances', []):
                if 'spectrogram_segment' in instance:
                    result = detector.detect_boundaries(
                        instance['spectrogram_segment'],
                        word=word
                    )
                    
                    # Normalize boundaries by total length
                    total_frames = instance['spectrogram_segment'].shape[0]
                    normalized = result['boundaries'] / total_frames
                    word_boundaries[word].append(normalized)
    
    # Analyze consistency
    for word, boundaries_list in word_boundaries.items():
        if len(boundaries_list) > 1:
            # Calculate variance in boundary positions
            boundaries_array = np.array([b for b in boundaries_list if len(b) == len(boundaries_list[0])])
            
            if len(boundaries_array) > 1:
                mean_boundaries = np.mean(boundaries_array, axis=0)
                std_boundaries = np.std(boundaries_array, axis=0)
                
                validation_results['boundary_consistency'][word] = {
                    'mean_positions': mean_boundaries.tolist(),
                    'std_positions': std_boundaries.tolist(),
                    'max_std': np.max(std_boundaries)
                }
                
                if np.max(std_boundaries) > 0.1:
                    print(f" Inconsistent boundaries for '{word}': max std={np.max(std_boundaries):.3f}")
    
    return validation_results


def check_data_balance(train_accumulated_data, test_accumulated_data):
    """
    Check class balance and data distribution
    """
    from collections import Counter
    
    print("=" * 60)
    print("DATA BALANCE ANALYSIS")
    print("=" * 60)
    
    # Analyze training data
    train_labels = train_accumulated_data['phoneme_labels']
    train_counter = Counter(train_labels)
    
    print("\nTraining data distribution:")
    for phoneme, count in train_counter.most_common(10):
        print(f"  {phoneme}: {count} ({count/len(train_labels)*100:.1f}%)")
    
    # Check for rare classes
    rare_threshold = 5
    rare_phonemes = [p for p, c in train_counter.items() if c < rare_threshold]
    if rare_phonemes:
        print(f"\n Warning: {len(rare_phonemes)} phonemes with < {rare_threshold} samples")
        print(f"   Rare phonemes: {rare_phonemes[:10]}")
    
    # Check test data
    test_labels = test_accumulated_data['phoneme_labels']
    test_counter = Counter(test_labels)
    
    # Check for unseen phonemes in test
    unseen = set(test_counter.keys()) - set(train_counter.keys())
    if unseen:
        print(f"\n Test set contains {len(unseen)} unseen phonemes: {list(unseen)[:10]}")
    
    # Calculate class imbalance ratio
    max_count = max(train_counter.values())
    min_count = min(train_counter.values())
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
    
    print(f"\nClass imbalance ratio: {imbalance_ratio:.2f}")
    if imbalance_ratio > 10:
        print(" Severe class imbalance detected!")
    
    return {
        'train_distribution': dict(train_counter),
        'test_distribution': dict(test_counter),
        'rare_phonemes': rare_phonemes,
        'unseen_phonemes': list(unseen),
        'imbalance_ratio': imbalance_ratio
    }


def validate_feature_extraction(train_accumulated_data):
    """
    Validate feature extraction quality
    """
    print("=" * 60)
    print("FEATURE EXTRACTION VALIDATION")
    print("=" * 60)
    
    features = train_accumulated_data['features']
    
    issues = []
    
    # 1. Check for NaN/Inf values
    nan_count = 0
    inf_count = 0
    zero_variance_count = 0
    
    for i, feat in enumerate(features):
        if np.any(np.isnan(feat)):
            nan_count += 1
            issues.append(f"Sample {i}: Contains NaN")
        
        if np.any(np.isinf(feat)):
            inf_count += 1
            issues.append(f"Sample {i}: Contains Inf")
        
        # Check for zero variance features
        if feat.ndim > 1:
            variances = np.var(feat, axis=0)
            if np.any(variances < 1e-10):
                zero_variance_count += 1
    
    print(f"NaN samples: {nan_count}/{len(features)}")
    print(f"Inf samples: {inf_count}/{len(features)}")
    print(f"Zero variance features: {zero_variance_count}/{len(features)}")
    
    # 2. Check feature dimensions
    shapes = [f.shape for f in features]
    unique_shapes = set(shapes)
    
    if len(unique_shapes) > 1:
        print(f"\n Inconsistent feature shapes detected:")
        for shape in unique_shapes:
            count = shapes.count(shape)
            print(f"  Shape {shape}: {count} samples")
    
    # 3. Check feature magnitudes
    magnitudes = []
    for feat in features[:100]:  # Sample first 100
        if feat.size > 0:
            magnitudes.append(np.mean(np.abs(feat)))
    
    mean_magnitude = np.mean(magnitudes)
    std_magnitude = np.std(magnitudes)
    
    print(f"\nFeature magnitudes:")
    print(f"  Mean: {mean_magnitude:.6f}")
    print(f"  Std: {std_magnitude:.6f}")
    
    if mean_magnitude < 1e-6:
        print("Features may be too small - consider scaling")
    elif mean_magnitude > 1e6:
        print(" Features may be too large - consider normalization")
    
    return {
        'nan_count': nan_count,
        'inf_count': inf_count,
        'zero_variance_count': zero_variance_count,
        'unique_shapes': list(unique_shapes),
        'magnitude_stats': {
            'mean': mean_magnitude,
            'std': std_magnitude
        }
    }


def validate_participant_split(split_result):
    """
    Validate train/test split quality
    """
    print("=" * 60)
    print("TRAIN/TEST SPLIT VALIDATION")
    print("=" * 60)
    
    train = split_result['train']
    test = split_result['test']
    
    # 1. Check participant overlap
    train_participants = set(train.keys())
    test_participants = set(test.keys())
    
    print(f"Train participants: {len(train_participants)}")
    print(f"Test participants: {len(test_participants)}")
    
    # 2. Check word distribution
    train_words = set()
    test_words = set()
    
    for participant_data in train.values():
        train_words.update(participant_data.keys())
    
    for participant_data in test.values():
        test_words.update(participant_data.keys())
    
    word_overlap = train_words & test_words
    unique_test_words = test_words - train_words
    
    print(f"\nWord statistics:")
    print(f"  Train words: {len(train_words)}")
    print(f"  Test words: {len(test_words)}")
    print(f"  Overlapping words: {len(word_overlap)}")
    print(f"  Unique test words: {len(unique_test_words)}")
    
    if len(unique_test_words) > 0:
        print(f" Test has {len(unique_test_words)} unseen words: {list(unique_test_words)[:5]}")
    
    # 3. Check instance counts
    train_instances = sum(
        len(instances) for participant_data in train.values() 
        for instances in participant_data.values()
    )
    
    test_instances = sum(
        len(instances) for participant_data in test.values() 
        for instances in participant_data.values()
    )
    
    split_ratio = train_instances / (train_instances + test_instances)
    
    print(f"\nInstance distribution:")
    print(f"  Train instances: {train_instances}")
    print(f"  Test instances: {test_instances}")
    print(f"  Train ratio: {split_ratio:.2%}")
    
    return {
        'train_participants': list(train_participants),
        'test_participants': list(test_participants),
        'train_words': list(train_words),
        'test_words': list(test_words),
        'unique_test_words': list(unique_test_words),
        'split_ratio': split_ratio
    }


# Usage example:
def run_all_validations(detector, split_result, train_data, test_data, word_segments_dict):
    """
    Run all validation checks
    """
    print("\n" + "="*80)
    print("COMPREHENSIVE DATA VALIDATION REPORT")
    print("="*80 + "\n")
    
    # 1. Validate segmentation
    segmentation_results = validate_phoneme_segmentation(
        detector, 
        word_segments_dict, 
        detector.phonetic_dict
    )
    
    # 2. Check data balance
    balance_results = check_data_balance(train_data, test_data)
    
    # 3. Validate features
    feature_results = validate_feature_extraction(train_data)
    
    # 4. Validate split
    split_results = validate_participant_split(split_result)
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY OF ISSUES")
    print("="*80)
    
    issues_found = []
    
    if segmentation_results['segmentation_accuracy']['overall'] < 0.7:
        issues_found.append(f"Low segmentation accuracy: {segmentation_results['segmentation_accuracy']['overall']:.2%}")
    
    if balance_results['imbalance_ratio'] > 10:
        issues_found.append(f"Severe class imbalance: {balance_results['imbalance_ratio']:.1f}x")
    
    if feature_results['nan_count'] > 0:
        issues_found.append(f"NaN values in {feature_results['nan_count']} samples")
    
    if len(balance_results['unseen_phonemes']) > 0:
        issues_found.append(f"{len(balance_results['unseen_phonemes'])} unseen phonemes in test")
    
    if issues_found:
        print("\nCritical issues detected:")
        for issue in issues_found:
            print(f"  • {issue}")
    else:
        print("\n✓ No critical issues detected")
    
    return {
        'segmentation': segmentation_results,
        'balance': balance_results,
        'features': feature_results,
        'split': split_results,
        'issues': issues_found
    }

word_segments_dict = split_result['word_segments_dict']

# All intermediate data you can access:
participant_strata = pipeline.pipeline_state['participant_strata']
split_result = pipeline.pipeline_state['split_result']
train_raw = pipeline.pipeline_state['train_raw']           # Before resolution
test_raw = pipeline.pipeline_state['test_raw']             # Before resolution
train_resolved = pipeline.pipeline_state['train_resolved'] # After resolution
test_resolved = pipeline.pipeline_state['test_resolved']   # After resolution
train_augmented = pipeline.pipeline_state['train_augmented'] # After augmentation (if applied)
train_group_labels = pipeline.pipeline_state['train_group_labels']
test_group_labels = pipeline.pipeline_state['test_group_labels']
validation_results = pipeline.pipeline_state['validation_results']

# Access components
decoder = pipeline.custom_decoder
detector = pipeline.detector
validator = pipeline.validator
phonetic_dict = pipeline.phonetic_dict

# Now you can use it for validation
validation_results = validate_phoneme_segmentation(
    detector=detector,
    word_segments_dict=word_segments_dict,
    phonetic_dict=detector.phonetic_dict
)

split_results = validate_participant_split(split_result)

data_balance = check_data_balance(train_augmented, test_resolved)

validate_feature_extraction(train_resolved)

validate_feature_extraction(train_augmented)
