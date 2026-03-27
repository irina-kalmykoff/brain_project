# Converted from Untitled3.ipynb

from transformers import AutoProcessor, AutoModel
import torch
import torchaudio

processor = AutoProcessor.from_pretrained("MahmoudAshraf/mms-300m-1130-forced-aligner")
model = AutoModel.from_pretrained("MahmoudAshraf/mms-300m-1130-forced-aligner")
model.eval()

import numpy as np

sample_rate = 16000
duration = 0.3  # seconds, roughly right for a short word like 'bak'
audio_numpy = np.sin(2 * np.pi * 440 * np.linspace(0, duration, int(sample_rate * duration))).astype(np.float32)

inputs = processor(
    audio=audio_numpy,
    sampling_rate=16000,
    return_tensors="pt"
)

transcript = "bak"  # the orthographic word

with torch.no_grad():
    outputs = model(**inputs, labels=processor.tokenizer(transcript, return_tensors="pt").input_ids)

alignments = processor.decode(
    outputs.logits[0].argmax(dim=-1),
    output_word_offsets=True
)

print(alignments)

from phonetic_dictionary import PhoneticDictionary
pd = PhoneticDictionary()
print(pd.extract_phonemes("bak"))

import torch
import numpy as np

sample_rate = 16000
duration = 0.3
audio_numpy = np.sin(2 * np.pi * 200 * np.linspace(0, duration, int(sample_rate * duration))).astype(np.float32)
audio_waveform = torch.tensor(audio_numpy).unsqueeze(0).to(alignment_model.device).to(alignment_model.dtype)

import torch
import numpy as np
import soundfile as sf
from ctc_forced_aligner import (
    load_audio,
    load_alignment_model,
    generate_emissions,
    preprocess_text,
    get_alignments,
    get_spans,
    postprocess_results,
)
from phonetic_dictionary import PhoneticDictionary

device = "cuda" if torch.cuda.is_available() else "cpu"

alignment_model, alignment_tokenizer = load_alignment_model(
    device,
    dtype=torch.float32,
)

# --- part 1: check what tokens the aligner knows for Dutch ---
print("aligner vocabulary (first 80 tokens):")
vocab = alignment_tokenizer.get_vocab()
tokens_sorted = sorted(vocab.items(), key=lambda x: x[1])
for token, idx in tokens_sorted[:80]:
    print(f"  {idx:4d}  {repr(token)}")

# --- part 2: align a real word and print the output spans ---
# replace this path with a real .wav file from your dataset
# the audio must be mono, 16kHz
audio_path = "path/to/your/word.wav"
audio_waveform = load_audio(audio_path, alignment_model.dtype, alignment_model.device)

# Dutch ISO 639-3 code is 'nld'
text = "bak"  # replace with whatever word the audio contains

emissions, stride = generate_emissions(alignment_model, audio_waveform, batch_size=1)

tokens_starred, text_starred = preprocess_text(
    text,
    romanize=True,
    language="nld",
)

print(f"\npreprocessed tokens for '{text}': {tokens_starred}")

segments, scores, blank_token = get_alignments(
    emissions,
    tokens_starred,
    alignment_tokenizer,
)

spans = get_spans(tokens_starred, segments, blank_token)
results = postprocess_results(text_starred, spans, stride, scores)

print(f"\nalignment results for '{text}':")
for r in results:
    print(f"  {r}")

# --- part 3: compare against your phonetic dictionary ---
pd = PhoneticDictionary()
print(f"\nyour dictionary gives for '{text}':")
print(f"  {pd.extract_phonemes(text)}")

from phonetic_dictionary import PhoneticDictionary

pd = PhoneticDictionary()

total = 0
matched = 0
mismatched_examples = []

for word in pd.dictionary:
    if ' ' in word:
        continue
    letters = list(word)
    phonemes = pd.extract_phonemes(word)
    total += 1
    if len(letters) == len(phonemes):
        matched += 1
    else:
        mismatched_examples.append((word, letters, phonemes))

print(f"total single words : {total}")
print(f"letter==phoneme    : {matched}  ({100*matched/total:.1f}%)")
print(f"mismatch           : {total - matched}  ({100*(total-matched)/total:.1f}%)")
print(f"\nfirst 20 mismatches:")
for word, letters, phonemes in mismatched_examples[:20]:
    print(f"  {word:<20} L={len(letters)} P={len(phonemes)}")

print(f"\ntorchaudio version: {torchaudio.__version__}")
import torch
print(f"torch version     : {torch.__version__}")

import numpy as np
import os

data_path = r'D:\Documents\UM DACS\bachelor\UM DACS\bachelor\mozg\code\SingleWordProductionDutch\Dutch_30patients\raw'

stimuli = np.load(os.path.join(data_path, 'P21_stimuli.npy'), allow_pickle=True)
print(f"shape: {stimuli.shape}")
print(f"dtype: {stimuli.dtype}")
print(f"first 10 entries:")
for i, s in enumerate(stimuli[:10]):
    print(f"  [{i}] {repr(s)}")
