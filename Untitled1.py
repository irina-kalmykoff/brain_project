# Converted from Untitled1.ipynb

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import welch
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
from feature_vizualizer import PhonemeFeatureVisualizer
from pipeline import UnifiedPhonemePipeline
from config import BIDS_PATH, OUTPUT_PATH, RESULTS_PATH, get_dataset_paths

data_path = r'./Shared/Data/'  # Raw data location in their repo

paths = get_dataset_paths('dutch30')
data_dir = os.path.join(paths['data_path'], 'raw')

# Verify data first
print(f"Looking for data in: {data_dir}")
print(f"Directory exists: {os.path.exists(data_dir)}")

# Check what's there
if os.path.exists(data_dir):
    files = os.listdir(data_dir)
    print(f"Found {len(files)} files")
    
    # Check for P01
    pt_id = 'P01'
    eeg_file = os.path.join(data_dir, f'{pt_id}_sEEG.npy')
    
    if os.path.exists(eeg_file):
        print(f"Loading {pt_id} data...")
        
        # Load data
        raw_eeg = np.load(os.path.join(data_dir, f'{pt_id}_sEEG.npy'))
        channels = np.load(os.path.join(data_dir, f'{pt_id}_channels.npy'), allow_pickle=True)
        
        print(f"EEG shape: {raw_eeg.shape}")
        print(f"Channels: {len(channels)}")
        
        # Visualize first 5 channels
        fig, axes = plt.subplots(5, 1, figsize=(10, 12))
        for ch in range(min(5, raw_eeg.shape[1])):
            freqs, psd = welch(raw_eeg[:10240, ch], fs=1024, nperseg=2048)
            axes[ch].semilogy(freqs, psd)
            axes[ch].axvline(70, color='g', linestyle='--', alpha=0.5, label='High gamma start')
            axes[ch].axvline(170, color='g', linestyle='--', alpha=0.5, label='High gamma end')
            axes[ch].axvline(200, color='r', linestyle='--', alpha=0.5)
            axes[ch].axvline(400, color='r', linestyle='--', alpha=0.5)
            
            # Handle channel names properly
            if hasattr(channels[ch], '__iter__'):
                ch_name = channels[ch][0] if len(channels[ch]) > 0 else f'Ch{ch}'
            else:
                ch_name = str(channels[ch])
            
            axes[ch].set_title(f'Channel {ch_name}')
            if ch == 0:
                axes[ch].legend()
        
        plt.tight_layout()
        plt.show()
    else:
        print(f"File not found: {eeg_file}")
else:
    print(f"Directory not found: {data_dir}")
    print("Please make sure the data files are in Dutch_30patients/raw/")

import numpy as np
from scipy.signal import welch
import matplotlib.pyplot as plt
from config import DUTCH_30_PATH
import os

plt.figure(figsize=(15, 6))

# For 30-patient dataset (numpy format)
data_dir = os.path.join(DUTCH_30_PATH, 'raw')

# Select patients to visualize (adjust based on what you have)
patients = ['P01', 'P02', 'P03', 'P04', 'P05', 'P07', 'P08']  # or any subset

for pt_id in patients:
    eeg_file = os.path.join(data_dir, f'{pt_id}_sEEG.npy')
    
    if os.path.exists(eeg_file):
        # Load EEG data
        data = np.load(eeg_file)
        
        # Take first 10240 samples from first channel
        # (10 seconds at 1024 Hz)
        data_segment = data[:10240, 0]
        
        # Compute PSD
        freqs, psd = welch(data_segment, fs=1024, nperseg=2048)
        plt.semilogy(freqs, psd, label=pt_id)
    else:
        print(f"File not found: {eeg_file}")

# Add reference lines
plt.axvline(70, color='g', linestyle='--', alpha=0.3, label='High gamma start')
plt.axvline(170, color='g', linestyle='--', alpha=0.3, label='High gamma end')
plt.axvline(150, color='r', linestyle='--', alpha=0.3)
plt.axvline(450, color='r', linestyle='--', alpha=0.3)

plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Spectral Density')
plt.title('PSD Comparison Across Patients (30-patient dataset)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

import numpy as np
import os
from collections import Counter
from config import DUTCH_30_PATH

# Analyze word content for 30-patient dataset
data_dir = os.path.join(DUTCH_30_PATH, 'raw')

all_words = []
patient_word_counts = {}

print("Analyzing 30-patient dataset word content:\n")
print("="*60)

for i in range(1, 31):
    pt_id = f'P{i:02d}'
    stimuli_file = os.path.join(data_dir, f'{pt_id}_stimuli.npy')
    
    if os.path.exists(stimuli_file):
        # Load stimuli
        stimuli = np.load(stimuli_file, allow_pickle=True)
        
        # Get unique words for this patient
        unique_words = np.unique(stimuli)
        
        # Filter out empty strings
        unique_words = [w for w in unique_words if w and str(w).strip()]
        
        patient_word_counts[pt_id] = len(unique_words)
        all_words.extend(unique_words)
        
        print(f"\n{pt_id}:")
        print(f"  Total markers: {len(stimuli)}")
        print(f"  Unique words: {len(unique_words)}")
        print(f"  First 10 words: {unique_words[:10]}")
        
        # Count occurrences
        word_freq = Counter(stimuli)
        most_common = word_freq.most_common(5)
        print(f"  Most frequent: {most_common}")

print("\n" + "="*60)
print("OVERALL STATISTICS:")
print("="*60)

# Overall statistics
unique_words_overall = list(set(all_words))
print(f"\nTotal unique words across all patients: {len(unique_words_overall)}")
print(f"Average words per patient: {np.mean(list(patient_word_counts.values())):.1f}")

# Show sample of vocabulary
print(f"\nSample vocabulary (first 20 words):")
print(unique_words_overall[:20])

# Check if they're Dutch words
dutch_indicators = ['van', 'de', 'het', 'een', 'en', 'is', 'dat', 'niet']
contains_dutch = any(word in unique_words_overall for word in dutch_indicators)
print(f"\nAppears to be Dutch: {contains_dutch}")

# Look for numbers
numbers = [w for w in unique_words_overall if w.isdigit() or w in ['1','2','3','4','5','6','7','8','9','10','11','12']]
if numbers:
    print(f"\nNumbers found: {numbers}")

# Create a detailed summary
print("\n" + "="*60)
print("WORD DISTRIBUTION BY PATIENT:")
print("="*60)

for pt_id, count in sorted(patient_word_counts.items()):
    bar = '█' * (count // 2)  # Simple bar chart
    print(f"{pt_id}: {count:3d} words {bar}")

import numpy as np
import os
from config import DUTCH_30_PATH

# Get all unique words from 30-patient dataset
data_dir = os.path.join(DUTCH_30_PATH, 'raw')

all_words_set = set()

for i in range(1, 31):
    pt_id = f'P{i:02d}'
    stimuli_file = os.path.join(data_dir, f'{pt_id}_stimuli.npy')
    
    if os.path.exists(stimuli_file):
        stimuli = np.load(stimuli_file, allow_pickle=True)
        # Add unique words, filtering out empty strings
        unique_words = [str(w).strip() for w in np.unique(stimuli) if w and str(w).strip()]
        all_words_set.update(unique_words)

# Convert to sorted list
unique_words_list = sorted(list(all_words_set))

print(f"Total unique words: {len(unique_words_list)}\n")
print("Complete word list:")
print("-" * 40)
for word in unique_words_list:
    print(word)

# Save to file if needed
output_file = os.path.join(DUTCH_30_PATH, 'unique_words.txt')
with open(output_file, 'w', encoding='utf-8') as f:
    for word in unique_words_list:
        f.write(f"{word}\n")
print(f"\nSaved to: {output_file}")

# Also save as numpy array
np.save(os.path.join(DUTCH_30_PATH, 'unique_words.npy'), unique_words_list)

import numpy as np
import os
from config import DUTCH_30_PATH

data_dir = os.path.join(DUTCH_30_PATH, 'raw')

def classify_stimulus(text):
    """Classify if text is a word or sentence"""
    text = str(text).strip()
    if not text:
        return 'empty'
    # Check for sentence indicators
    if ' ' in text or len(text.split()) > 1:
        return 'sentence'
    elif len(text) > 20:  # Very long single "word" might be concatenated
        return 'possible_sentence'
    else:
        return 'word'

patient_analysis = {}

print("Patient Stimulus Type Analysis")
print("="*60)

for i in range(1, 31):
    pt_id = f'P{i:02d}'
    stimuli_file = os.path.join(data_dir, f'{pt_id}_stimuli.npy')
    
    if os.path.exists(stimuli_file):
        stimuli = np.load(stimuli_file, allow_pickle=True)
        
        # Classify each stimulus
        classifications = [classify_stimulus(s) for s in stimuli]
        
        # Count types
        n_words = classifications.count('word')
        n_sentences = classifications.count('sentence')
        n_possible = classifications.count('possible_sentence')
        n_empty = classifications.count('empty')
        
        # Get unique examples
        unique_stimuli = np.unique(stimuli)
        unique_words = [s for s in unique_stimuli if classify_stimulus(s) == 'word']
        unique_sentences = [s for s in unique_stimuli if classify_stimulus(s) == 'sentence']
        
        # Determine patient type
        if n_sentences > 0 and n_words == 0:
            patient_type = "SENTENCES ONLY"
        elif n_words > 0 and n_sentences == 0:
            patient_type = "WORDS ONLY"
        elif n_words > 0 and n_sentences > 0:
            patient_type = "MIXED"
        else:
            patient_type = "UNCLEAR"
        
        patient_analysis[pt_id] = {
            'type': patient_type,
            'n_words': n_words,
            'n_sentences': n_sentences,
            'n_unique_words': len(unique_words),
            'n_unique_sentences': len(unique_sentences),
            'example_words': unique_words[:5],
            'example_sentences': unique_sentences[:3]
        }
        
        print(f"\n{pt_id}: {patient_type}")
        print(f"  Words: {n_words} markers ({len(unique_words)} unique)")
        print(f"  Sentences: {n_sentences} markers ({len(unique_sentences)} unique)")
        if unique_words:
            print(f"  Example words: {unique_words[:5]}")
        if unique_sentences:
            print(f"  Example sentences: {unique_sentences[:2]}")

# Summary
print("\n" + "="*60)
print("SUMMARY BY PATIENT TYPE:")
print("="*60)

words_only = [p for p, info in patient_analysis.items() if info['type'] == 'WORDS ONLY']
sentences_only = [p for p, info in patient_analysis.items() if info['type'] == 'SENTENCES ONLY']
mixed = [p for p, info in patient_analysis.items() if info['type'] == 'MIXED']

print(f"\nWords Only Patients ({len(words_only)}): {words_only}")
print(f"Sentences Only Patients ({len(sentences_only)}): {sentences_only}")
print(f"Mixed Patients ({len(mixed)}): {mixed}")

# Check how markers change over time for a mixed patient
if mixed:
    print(f"\n" + "="*60)
    print(f"ANALYZING TEMPORAL PATTERN FOR {mixed[0]}:")
    print("="*60)
    
    stimuli_file = os.path.join(data_dir, f'{mixed[0]}_stimuli.npy')
    stimuli = np.load(stimuli_file, allow_pickle=True)
    
    # Look at transitions
    transitions = []
    for i in range(1, min(100, len(stimuli))):  # Look at first 100 transitions
        if stimuli[i] != stimuli[i-1]:
            type_from = classify_stimulus(stimuli[i-1])
            type_to = classify_stimulus(stimuli[i])
            transitions.append((i, stimuli[i-1], type_from, stimuli[i], type_to))
    
    print(f"First 10 stimulus changes:")
    for i, (idx, from_stim, from_type, to_stim, to_type) in enumerate(transitions[:10]):
        print(f"  {idx}: [{from_type}] '{from_stim}' -> [{to_type}] '{to_stim}'")
