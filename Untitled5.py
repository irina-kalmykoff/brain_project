# Converted from Untitled5.ipynb

import time

modules = [
    'custom_decoder', 'brain_audio_decoder_viz', 'acoustic_change_detector',
    'phoneme_validator', 'phonetic_dictionary', 'feature_vizualizer',
    'markov_phoneme_model', 'pipeline'
]

for mod in modules:
    start = time.time()
    __import__(mod)
    print(f"{mod}: {time.time() - start:.2f}s")
