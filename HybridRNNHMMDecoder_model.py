#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import matplotlib.pyplot as plt
from acoustic_change_detector import AcousticChangeDetector
from phoneme_validator import PhonemeValidator
from hybrid_rnn_hmm_decoder import HybridRNNHMMDecoder


# In[2]:


# Define paths
path_bids = './SingleWordProductionDutch-iBIDS'  # Path to the BIDS dataset
path_output = './features'  # Path to save extracted features
path_results = './results'  # Path to save results


# In[3]:


# 1. Initialize the custom decoder and detector
from custom_decoder import CustomBrainAudioDecoder
custom_decoder = CustomBrainAudioDecoder(
    path_bids=path_bids,
    path_output=path_output,
    path_results=path_results,
    debug_mode=True
)


# In[4]:


# 2. Load and prepare the data
print("Loading participants and extracting features...")
# This loads metadata for all participants and extracts high gamma features
# from the EEG data, which will be used for model training.
custom_decoder.load_participants()
custom_decoder.extract_features_all_participants()

# This identifies which channels are most relevant for speech decoding
# and will be used to stratify participants by signal quality.
custom_decoder.analyze_channels_across_participants()


# In[5]:


# 3. Stratify participants and create train/test split
print("Creating stratified cross-word split...")
participant_strata = custom_decoder.stratify_participants_by_channel_quality(
    channel_correlation_threshold=0.1 
)
split_result = custom_decoder.create_stratified_cross_word_split(
    participant_strata=participant_strata,
    test_ratio=0.2,
    min_word_freq=1,
    random_seed=42
)


# In[6]:


# 4. Initialize the acoustic change detector
print("Initializing acoustic change detector...")
detector = AcousticChangeDetector(
    min_segment_duration=0.03,
    max_segment_duration=0.3,
    distance_metric='cosine',
    smoothing_window=3,
    peak_threshold=0.5,
    decoder=custom_decoder,
    debug_mode=True 
)


# In[7]:


# 5. Initialize validator
validator = PhonemeValidator(detector=detector)
validator.enable_debug()


# In[8]:


# 6. Pass split_result to detector for data accumulation
detector.split_result = split_result


# In[9]:


# 7. Accumulate phoneme data
print("Accumulating training data...")
train_data = detector.accumulate_phoneme_data(
    num_batches=5,
    batch_size=32,
    feature_extraction_method='high_gamma',
    batch_type='train'
)


# In[10]:


print("Accumulating test data...")
test_data = detector.accumulate_phoneme_data(
    num_batches=3,
    batch_size=32,
    feature_extraction_method='high_gamma',
    batch_type='test'
)


# In[11]:


# 8. Resolve unknown phonemes for better data quality
# For training data
train_converted = {
    'phoneme_labels': train_data['phoneme_labels'],
    'phoneme_spectrogram_segments': train_data.get('spectrograms', []),
    'phoneme_words': train_data['phoneme_words'],
    'phoneme_positions': train_data.get('phoneme_positions', [0] * len(train_data['phoneme_labels'])),
    'phoneme_participant_ids': train_data.get('phoneme_participant_ids', ['unknown'] * len(train_data['phoneme_labels']))
}


# In[12]:


print(f"Training data before resolution: {train_converted['phoneme_labels'].count('?')} unknown phonemes")
resolved_train = validator.resolve_unknown_phonemes(train_converted)
print(f"Training data after resolution: {resolved_train['phoneme_labels'].count('?')} unknown phonemes")
# Update the training data with resolved phonemes
train_data['phoneme_labels'] = resolved_train['phoneme_labels']


# In[13]:


# For test data
test_converted = {
    'phoneme_labels': test_data['phoneme_labels'],
    'phoneme_spectrogram_segments': test_data.get('spectrograms', []),
    'phoneme_words': test_data['phoneme_words'],
    'phoneme_positions': test_data.get('phoneme_positions', [0] * len(test_data['phoneme_labels'])),
    'phoneme_participant_ids': test_data.get('phoneme_participant_ids', ['unknown'] * len(test_data['phoneme_labels']))}


# In[14]:


print(f"Test data before resolution: {test_converted['phoneme_labels'].count('?')} unknown phonemes")
resolved_test = validator.resolve_unknown_phonemes(test_converted)
print(f"Test data after resolution: {resolved_test['phoneme_labels'].count('?')} unknown phonemes")


# In[15]:


# Update the test data with resolved phonemes
test_data['phoneme_labels'] = resolved_test['phoneme_labels']


# In[16]:


# 9. Initialize the Hybrid RNN-HMM Decoder
print("Initializing HybridRNNHMMDecoder")
hybrid_model = HybridRNNHMMDecoder(
    output_dir=os.path.join(path_results, 'hybrid_model'),
    debug_mode=True,
    phonetic_dict=detector.phonetic_dict,
    hmm_states_per_phoneme=3
)


# In[17]:


# 10. Train the hybrid model
print("Training hybrid model...")
results = hybrid_model.train_with_accumulated_data(
    train_data=train_data,
    test_data=test_data,
    validation_split=0.2,
    nn_epochs=50,
    hmm_iter=50,
    patience=10
)


# In[18]:


# 11. Print results summary
print("\n===== Results Summary =====")
if results['eval_results'] is not None:
    nn_accuracy = results['eval_results']['nn_accuracy']
    hybrid_accuracy = results['eval_results']['hybrid_accuracy']
    print(f"Neural network accuracy: {nn_accuracy:.4f}")
    print(f"Hybrid model accuracy: {hybrid_accuracy:.4f}")
    
    # Improvement from the HMM component
    improvement = hybrid_accuracy - nn_accuracy
    print(f"Improvement from HMM: {improvement:.4f} ({improvement*100:.1f}%)")

print("\nModel training and evaluation complete!")


# In[19]:


# 12. Additional evaluation and analysis
if results['eval_results'] is not None:
    # Compare predicted phonemes with true phonemes for a sample
    true_phonemes = results['eval_results']['hybrid_phonemes'][:10]
    pred_phonemes = results['eval_results']['nn_phonemes'][:10]
    
    print("\nSample Predictions:")
    print("True vs Predicted Phonemes")
    print("-----------------------")
    for true, pred in zip(true_phonemes, pred_phonemes):
        print(f"{true:<5} -> {pred:<5} {'✓' if true == pred else '✗'}")
    
    # Print performance by phoneme category
    if results['eval_results']['classification_report'] is not None:
        report = results['eval_results']['classification_report']
        
        # Group phonemes into categories
        vowels = [p for p in hybrid_model.phoneme_list if p in 'aeiouəɑɛɪɔʊʌœyɵ' or p in ['ɛi', 'œy', 'ɑu']]
        consonants = [p for p in hybrid_model.phoneme_list if p not in vowels]
        
        print("\nPerformance by Phoneme Category:")
        print("Vowels:")
        vowel_f1 = 0
        vowel_count = 0
        for v in vowels:
            if v in report:
                vowel_f1 += report[v]['f1-score']
                vowel_count += 1
                print(f"  {v}: F1={report[v]['f1-score']:.4f}, Support={report[v]['support']}")
        
        if vowel_count > 0:
            print(f"  Average vowel F1: {vowel_f1/vowel_count:.4f}")
        
        print("\nConsonants:")
        consonant_f1 = 0
        consonant_count = 0
        for c in consonants:
            if c in report:
                consonant_f1 += report[c]['f1-score']
                consonant_count += 1
                print(f"  {c}: F1={report[c]['f1-score']:.4f}, Support={report[c]['support']}")
        
        if consonant_count > 0:
            print(f"  Average consonant F1: {consonant_f1/consonant_count:.4f}")


# In[ ]:


jupyter nbconvert --to python your_notebook.ipynb

