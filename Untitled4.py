# Converted from Untitled4.ipynb

packages = [
    "torch",
    "transformers",
    "numpy",
    "scipy",
    ("sklearn", "sklearn"),  # skip this one
    "librosa",
    "mne",
    "h5py",
]

import sys, platform, importlib
print(f"python: {sys.version}")
print(f"platform: {platform.platform()}\n")

import torch
print(f"torch: {torch.__version__}")
print(f"cuda available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"cuda device: {torch.cuda.get_device_name(0)}")
    print(f"cuda version: {torch.version.cuda}")
print(f"device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

import transformers
print(f"\ntransformers: {importlib.metadata.version('transformers')}")

import numpy; print(f"numpy: {numpy.__version__}")
import scipy; print(f"scipy: {scipy.__version__}")
print(f"sklearn: {importlib.metadata.version('scikit-learn')}")


# ── 1. TORCH FIRST (before anything touches CUDA) ────────────────────────────
import torch
import torchaudio

# ── 2. TRANSFORMERS SECOND (before librosa loads via project imports) ─────────
from transformers import Wav2Vec2Model, Wav2Vec2Processor, Wav2Vec2FeatureExtractor

# ── 3. STANDARD LIBRARIES ───────────────────────────────────────────────────────
import os
import gc
import copy
import glob
import json
import pickle
import tempfile
from datetime import datetime
from collections import Counter, defaultdict
from itertools import combinations

# ── 4. THIRD-PARTY (no CUDA) ──────────────────────────────────────────────────
import numpy as np
import pandas as pd
import scipy.signal
import matplotlib.pyplot as plt
import seaborn as sns
from pynwb import NWBHDF5IO
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, silhouette_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cosine, euclidean
from scipy.signal import decimate

# ── 5. PROJECT IMPORTS ────────────────────────────────────────────────────────
from extract_features import extractHG, stackFeatures, downsampleLabels, extractMelSpecs
from brain_audio_decoder import BrainAudioDecoder
from custom_decoder import CustomBrainAudioDecoder
from brain_audio_decoder_viz import BrainAudioDecoderViz
from acoustic_change_detector import AcousticChangeDetector
from phoneme_validator import PhonemeValidator
from phonetic_dictionary import PhoneticDictionary
from markov_phoneme_model import MarkovPhonemeModel
from config import BIDS_PATH, OUTPUT_PATH, RESULTS_PATH, DUTCH_30_PATH, DUTCH_10_PATH, get_dataset_paths
from dutch_30_pipeline import Dutch30Pipeline
from dutch_30_feature_extractor import Dutch30FeatureExtractor
from phoneme_detection_diagnostic import Dutch30PhonemeDetectionDiagnostic
from dataset_config import Dutch30Config
from experiment_logger import ExperimentLogger

# ── 6. WHISPERX  ──────────────────────────────────────────────────────────
import whisperx

# feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-large-xlsr-53")
# model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-xlsr-53")
# print("Downloaded successfully, hidden size:", model.config.hidden_size)

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

# python run_pipeline.py                    # Path A (wav2vec)
# python run_pipeline.py --mfa             # Path B (MFA)
# python run_pipeline.py --mfa --sweep     # Path B + hyperparameter sweep
# python run_pipeline.py --mfa --analyze   # Path B + consecutive analysis
# python run_pipeline.py --export-mfa      # Export audio for MFA (one-time)
# python run_pipeline.py --diagnose-mfa    # Show MFA phoneme loss
# python run_pipeline.py --mfa-coverage    # Show alignment coverage
# python run_pipeline.py --patients 1-10   # Different patient range

# build pipeline
# Pipeline setup + load MFA-aligned features into pipeline.train / pipeline.test
from run_pipeline import DEFAULT_RUN_CONFIG, run_path_b
from dutch_30_pipeline import Dutch30Pipeline
from dutch_30_feature_extractor import Dutch30FeatureExtractor

run_config = dict(DEFAULT_RUN_CONFIG)
run_config['use_viterbi']        = True
run_config['stacking_order']     = 20
run_config['stacking_step_size'] = 1

# build pipeline
extractor = Dutch30FeatureExtractor()
pipeline = Dutch30Pipeline(
    dutch30_extractor=extractor, debug_mode=False,
    feature_extraction_method=run_config['feature_extraction_method'],
    use_wav2vec=False,
    subtract_baseline=run_config['subtract_baseline'],
    use_rms_boundaries=False,
    use_multifeature=False,
)

import os, glob, json, time
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch
import scipy.signal
from math import gcd

from config import DUTCH_30_PATH
from run_pipeline import load_mfa_alignments

# WhisperX (optional — only needed for cells that generate alignments)
try:
    import whisperx
    HAS_WHISPERX = True
except ImportError:
    HAS_WHISPERX = False
    print("WARNING: whisperx not installed; cells that generate alignments will skip.")

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

# Where we'll cache Whisper alignments per patient
WHISPER_CACHE_DIR = Path(DUTCH_30_PATH).parent / 'whisperx_alignments'
WHISPER_CACHE_DIR.mkdir(exist_ok=True)

def load_whisperx_models(language='nl'):
    """Loads two models: a small Whisper for transcription, plus the
    Dutch alignment model (wav2vec2-based) for phoneme-level timestamps.
    Caches in memory; ~30 sec on first call."""
    assert HAS_WHISPERX, "whisperx not installed"
    print("Loading WhisperX (this takes ~30s)...")
    # Small Whisper model for transcription — we already have transcripts so
    # we'll pass them directly into align(); the model just needs the audio.
    asr_model = whisperx.load_model("small", DEVICE, compute_type="float16",
                                     language=language)
    align_model, metadata = whisperx.load_align_model(
        language_code=language, device=DEVICE)
    print(f"  ASR + alignment model loaded for language={language}")
    return asr_model, align_model, metadata

def whisperx_align_one_sentence(audio_array, sr, transcript_text,
                                  align_model, metadata, device=DEVICE):
    """Force-align a single sentence audio + transcript via WhisperX.

    Returns:
      {
        'words': [{'word': 'ik', 'start': 0.12, 'end': 0.27, 'score': 0.93}, ...],
        'phones': [{'phone': 'ɪ', 'start': 0.12, 'end': 0.16}, ...],   # if alignment model returns char/phone
      }
    Uses transcript_text directly (no transcription step) — faster + more
    accurate since we know what was said.
    """
    # Resample to 16kHz if needed (Whisper requires this)
    if sr != 16000:
        g = gcd(int(sr), 16000)
        audio_16k = scipy.signal.resample_poly(audio_array, 16000 // g, sr // g)
    else:
        audio_16k = audio_array
    audio_16k = audio_16k.astype(np.float32)

    # Build a single "segment" describing the whole audio
    duration = len(audio_16k) / 16000.0
    segments = [{
        'start': 0.0,
        'end':   duration,
        'text':  transcript_text,
    }]

    # Force-align the transcript to the audio
    result = whisperx.align(
        segments, align_model, metadata, audio_16k,
        device=device, return_char_alignments=False,
    )

    word_segments = result.get('word_segments', [])
    return {
        'words':  [{'word': w.get('word'),
                     'start': w.get('start'),
                     'end':   w.get('end'),
                     'score': w.get('score', 0.0)}
                    for w in word_segments
                    if w.get('start') is not None and w.get('end') is not None],
        'duration_s': duration,
    }

def cache_whisper_alignments(pipeline, patient_ids, force=False):
    """For each sentence in each patient, save WhisperX word alignments
    to a per-patient JSON file. Idempotent (skips if file exists)."""
    if not HAS_WHISPERX:
        print("Skipping: WhisperX not available.")
        return
    asr_model, align_model, metadata = load_whisperx_models()
    audio_sr = int(pipeline.config.audio_sr)
    eeg_sr = pipeline.config.eeg_sr

    for pid in patient_ids:
        out_path = WHISPER_CACHE_DIR / f"{pid}.json"
        if out_path.exists() and not force:
            print(f"  {pid}: cached at {out_path} (skip; use force=True to overwrite)")
            continue

        wd = pipeline.split_result.get('word_segments_dict', {}).get(pid)
        if wd is None:
            print(f"  {pid}: not in word_segments_dict, skipping")
            continue

        raw_aud = np.load(os.path.join(DUTCH_30_PATH, 'raw', f'{pid}_audio.npy'))

        per_sentence = {}
        t_start = time.time()
        for sent_idx, sent_info in enumerate(wd['sentence_list']):
            text = sent_info['text'] if isinstance(sent_info, dict) else sent_info
            if not text or not text.strip():
                continue
            aud_lo = int(sent_info['stim_start_idx'] * audio_sr / eeg_sr)
            aud_hi = int(sent_info['stim_end_idx']   * audio_sr / eeg_sr)
            audio_clip = raw_aud[aud_lo:aud_hi].astype(np.float32)
            if audio_clip.size < int(audio_sr * 0.2):
                continue
            try:
                aln = whisperx_align_one_sentence(
                    audio_clip, audio_sr, text, align_model, metadata)
                per_sentence[str(sent_idx)] = aln
            except Exception as e:
                print(f"    {pid} sent {sent_idx}: align failed: {e}")
                continue

        with out_path.open('w', encoding='utf-8') as f:
            json.dump(per_sentence, f, ensure_ascii=False, indent=2)
        print(f"  {pid}: aligned {len(per_sentence)} sentences in "
              f"{time.time() - t_start:.1f}s → {out_path}")

from dutch_30_pipeline import Dutch30Pipeline
from dutch_30_feature_extractor import Dutch30FeatureExtractor

extractor = Dutch30FeatureExtractor()
pipeline = Dutch30Pipeline(
    dutch30_extractor=extractor, debug_mode=False,
    use_wav2vec=False, subtract_baseline=False,
    use_rms_boundaries=False, use_multifeature=False,
)

# Just two steps — populates pipeline.split_result['word_segments_dict']
pipeline.step1_load_dutch30_data(patient_range=(21, 30))
pipeline.step2_split_by_instances()

# That's all you need for Whisper:
from whisper_enhance_v6 import cache_whisper_alignments
cache_whisper_alignments(pipeline, [f'P{i:02d}' for i in range(21, 31)])

# 1. Load patient data + MFA-aligned features (the data pipeline)
from run_pipeline import DEFAULT_RUN_CONFIG, run_with_mfa_boundaries
run_config = dict(DEFAULT_RUN_CONFIG)
run_config['use_viterbi']        = True
run_config['stacking_order']     = 20
run_config['stacking_step_size'] = 1
cached_train, cached_test = run_with_mfa_boundaries(pipeline, run_config)

# 2. Train brain-only models (CRFs + v6 boundary detector)
from e2e_brain_decoder import train_brain_only_models
classifiers, v6_model, detector_test_ds = train_brain_only_models(pipeline)

# %% Listen + compare Whisper vs MFA for one sentence
import os
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Audio, display, HTML
import scipy.signal

from config import DUTCH_30_PATH
from run_pipeline import load_mfa_alignments
from whisper_enhance_v6 import load_whisper_alignments

def listen_and_compare(pipeline, pid, sent_idx, target_sr=16000):
    """Load the audio for one sentence, plot waveform + Whisper words +
    MFA phoneme onsets, and embed an audio player."""
    wd = pipeline.split_result['word_segments_dict'][pid]
    sent_info = wd['sentence_list'][sent_idx]
    text = sent_info['text'] if isinstance(sent_info, dict) else sent_info

    # Slice raw audio for this sentence
    audio_sr = int(pipeline.config.audio_sr)
    eeg_sr = pipeline.config.eeg_sr
    raw_aud = np.load(os.path.join(DUTCH_30_PATH, 'raw', f'{pid}_audio.npy'))
    aud_lo = int(sent_info['stim_start_idx'] * audio_sr / eeg_sr)
    aud_hi = int(sent_info['stim_end_idx']   * audio_sr / eeg_sr)
    audio_clip = raw_aud[aud_lo:aud_hi].astype(np.float32)

    # Normalize for playback
    peak = float(np.max(np.abs(audio_clip)))
    if peak > 0: audio_clip = audio_clip / peak

    # Resample for Audio widget (smaller payload)
    if audio_sr != target_sr:
        from math import gcd
        g = gcd(audio_sr, target_sr)
        playback = scipy.signal.resample_poly(audio_clip,
                                                target_sr // g, audio_sr // g)
        playback_sr = target_sr
    else:
        playback = audio_clip
        playback_sr = audio_sr

    duration_s = len(audio_clip) / audio_sr

    # Get both alignments
    whisper = load_whisper_alignments(pid).get(sent_idx)
    mfa     = load_mfa_alignments(pid).get(sent_idx, [])

    # Figure: top = waveform with overlays, bottom = audio player
    fig, ax = plt.subplots(1, 1, figsize=(14, 3))
    t = np.linspace(0, duration_s, len(audio_clip))
    ax.plot(t, audio_clip, color='gray', linewidth=0.5, alpha=0.7)
    ax.set_xlim(0, duration_s)
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlabel('Time (s)')

    # Whisper words — translucent blue spans + word labels at top
    if whisper:
        for w in whisper['words']:
            ax.axvspan(w['start'], w['end'], alpha=0.20, color='steelblue')
            ax.text((w['start'] + w['end']) / 2, 0.9, w['word'],
                    ha='center', va='top', fontsize=9, color='steelblue')

    # MFA phoneme onsets — vertical red lines + phoneme labels at bottom
    for ph in mfa:
        ax.axvline(ph['start_s'], color='red', alpha=0.35, linewidth=0.7)
        ax.text(ph['start_s'], -0.95, ph['phone'],
                ha='center', va='bottom', fontsize=8,
                color='darkred', alpha=0.7)
    if mfa:
        ax.axvline(mfa[-1]['end_s'], color='red', alpha=0.35,
                    linewidth=0.7, linestyle='--')

    ax.set_title(f"{pid} sent {sent_idx} — \"{text}\"\n"
                 f"Whisper words (blue) | MFA phonemes (red)",
                 fontsize=11)
    plt.tight_layout(); plt.show()

    # Print numbers for inspection
    print(f"Sentence duration: {duration_s:.2f}s")
    print(f"\n{'WHISPER WORDS':<20}{'MFA-DERIVED WORD ONSETS':<30}")
    print(f"{'='*20}{'='*30}")
    if whisper:
        whisper_words = [(w['word'], w['start'], w['end'])
                          for w in whisper['words']]
    else:
        whisper_words = []
    # Build MFA word onsets (start of first phoneme of each word)
    mfa_word_onsets = []
    prev_word = None
    for ph in mfa:
        wd_p = ph.get('word', '') or ''
        if wd_p != prev_word and wd_p:
            mfa_word_onsets.append((wd_p, ph['start_s']))
            prev_word = wd_p

    n = max(len(whisper_words), len(mfa_word_onsets))
    for i in range(n):
        if i < len(whisper_words):
            ww, ws, we = whisper_words[i]
            l = f"{ws:>5.2f}–{we:<5.2f}  {ww}"
        else:
            l = ""
        if i < len(mfa_word_onsets):
            mw, ms = mfa_word_onsets[i]
            r = f"{ms:>5.2f}        {mw}"
        else:
            r = ""
        print(f"  {l:<28}{r}")

    print()
    return Audio(playback, rate=playback_sr)


# Run it on the sentence you screenshotted
listen_and_compare(pipeline, 'P28', 20)

# Run on your screenshot example
listen_per_word(pipeline, 'P28', 20)

# %% Listen per-word: Whisper vs MFA
import os
import numpy as np
import scipy.signal
from math import gcd
from IPython.display import Audio, display, HTML

from config import DUTCH_30_PATH
from run_pipeline import load_mfa_alignments
from whisper_enhance_v6 import load_whisper_alignments


def _resample(audio, in_sr, out_sr=16000):
    if in_sr == out_sr:
        return audio
    g = gcd(in_sr, out_sr)
    return scipy.signal.resample_poly(audio, out_sr // g, in_sr // g)


def _audio_html(clip, sr, label, time_str):
    """Build a row of HTML containing an audio player + label + time range."""
    audio = Audio(clip, rate=sr)
    return f"""
    <tr>
      <td style="font-family:monospace; font-weight:bold; padding:4px 8px;
                  text-align:right; min-width:120px;">{label}</td>
      <td style="font-family:monospace; color:#888; font-size:11px;
                  padding:0 8px;">{time_str}</td>
      <td>{audio._repr_html_()}</td>
    </tr>"""


def listen_per_word(pipeline, pid, sent_idx, target_sr=16000):
    """Slice one sentence's audio per word according to BOTH alignments,
    and embed audio players side by side for A/B listening."""
    wd = pipeline.split_result['word_segments_dict'][pid]
    sent_info = wd['sentence_list'][sent_idx]
    text = sent_info['text'] if isinstance(sent_info, dict) else sent_info

    audio_sr = int(pipeline.config.audio_sr)
    eeg_sr = pipeline.config.eeg_sr
    raw_aud = np.load(os.path.join(DUTCH_30_PATH, 'raw', f'{pid}_audio.npy'))
    aud_lo = int(sent_info['stim_start_idx'] * audio_sr / eeg_sr)
    aud_hi = int(sent_info['stim_end_idx']   * audio_sr / eeg_sr)
    audio = raw_aud[aud_lo:aud_hi].astype(np.float32)

    # Normalize for playback
    peak = float(np.max(np.abs(audio)))
    if peak > 0: audio = audio / peak

    duration_s = len(audio) / audio_sr

    # Get both alignments
    whisper = load_whisper_alignments(pid).get(sent_idx)
    mfa     = load_mfa_alignments(pid).get(sent_idx, [])

    # Build MFA word groups: list of (word_text, start_s, end_s)
    mfa_words = []
    cur_word = None
    cur_start = None
    for ph in mfa:
        w = ph.get('word', '') or ''
        if w != cur_word:
            if cur_word is not None and cur_word.strip():
                mfa_words.append((cur_word, cur_start, prev_end))
            cur_word = w
            cur_start = ph['start_s']
        prev_end = ph['end_s']
    if cur_word is not None and cur_word.strip():
        mfa_words.append((cur_word, cur_start, prev_end))

    whisper_words = (whisper['words']
                     if whisper and 'words' in whisper else [])

    # Header
    print(f"\n  {pid}  sent {sent_idx}  (duration {duration_s:.2f}s)")
    print(f"  Transcript: \"{text}\"\n")

    # ── WHISPER WORDS ──
    print("─── WHISPER ───")
    if not whisper_words:
        print("  (no Whisper alignment cached)")
    else:
        rows = []
        for w in whisper_words:
            s, e = w['start'], w['end']
            lo = int(s * audio_sr); hi = int(e * audio_sr)
            clip = audio[max(0, lo):min(len(audio), hi)]
            if clip.size < int(audio_sr * 0.05):    # skip tiny < 50ms
                continue
            clip_play = _resample(clip, audio_sr, target_sr)
            time_str = f"{s:5.2f}–{e:5.2f}s"
            rows.append(_audio_html(clip_play, target_sr, w['word'], time_str))
        display(HTML(f'<table style="border-collapse:collapse">{"".join(rows)}</table>'))

    # ── MFA WORDS ──
    print("\n─── MFA ───")
    if not mfa_words:
        print("  (no MFA alignment)")
    else:
        rows = []
        for word, s, e in mfa_words:
            lo = int(s * audio_sr); hi = int(e * audio_sr)
            clip = audio[max(0, lo):min(len(audio), hi)]
            if clip.size < int(audio_sr * 0.05):
                continue
            clip_play = _resample(clip, audio_sr, target_sr)
            time_str = f"{s:5.2f}–{e:5.2f}s"
            rows.append(_audio_html(clip_play, target_sr, word, time_str))
        display(HTML(f'<table style="border-collapse:collapse">{"".join(rows)}</table>'))


# Run on your screenshot example
listen_per_word(pipeline, 'P24', 12)


# %% Cell 14 — End-to-end protocol summary
# 1. Run cache_whisper_alignments(pipeline, ALL_PIDS)              [Cell 5]
#    → produces whisperx_alignments/<pid>.json files (~30 min total).
# 2. Run compare_whisper_vs_mfa_word_onsets per patient            [Cell 8]
#    → see how often the two aligners agree. Diagnostic.
# 3. Build dataset with Whisper labels: build_dataset_with_whisper [Cell 10]
# 4. Build WhisperWordV6 model and train via your own loop based on
#    Cell 12's skeleton.
# 5. At test time, use ONLY the boundary head + count head → identical
#    inference to v6. Whisper was a training-time supervisor only.
# 6. Compare: PER of Whisper-supervised v6 vs original v6.

from e2e_brain_decoder import _features_to_dicts_one_seq

PID = 'P21'
def predict_sklearn_mfa(pid, sid, sent_eeg):
    phones = mfa_align.get(sid, [])
    if not phones: return []
    feats = [extract_phoneme_feature_pipeline_native(
                sent_eeg, p['start_s'], p['end_s'],
                eeg_sr, win_len, frameshift, stk_order)
             for p in phones]
    feats = [f for f in feats if f is not None]
    if not feats: return []
    sc  = classifiers[pid]['scaler']
    pca = classifiers[pid]['pca']
    crf = classifiers[pid]['crf']
    X = pca.transform(sc.transform(np.asarray(feats)))
    return list(crf.predict(_features_to_dicts_one_seq(X))[0])

for item in detector_test_ds:
    if item['pid'] != PID: continue
    sid = item['sentence_idx']
    sent_info = wd['sentence_list'][sid]
    sent_eeg = raw_eeg[int(sent_info['stim_start_idx']):int(sent_info['stim_end_idx'])]
    gold = get_mfa_oracle_labels(PID, sid, valid)
    sk = predict_sklearn_mfa(PID, sid, sent_eeg)
    pp, gg = color_matches(sk, gold)
    print(f"\n--- sent {sid} ---")
    print("sklearn MFA:", " ".join(pp))
    print("       gold:", " ".join(gg))

PID = 'P22'   # or 'P22'

# rebuild the per-patient bits
vocab = vocabs[PID]
as_str = (lambda i: vocab[int(i)]) if isinstance(vocab, list) \
         else (lambda i: {v:k for k,v in vocab.items()}[int(i)])
valid = set(vocab) if isinstance(vocab, list) else set(vocab.keys())

mask = get_pipeline_channel_mask(pipeline, PID)
raw  = np.load(os.path.join(DUTCH_30_PATH, 'raw', f'{PID}_sEEG.npy'))
raw_eeg = raw[:, mask] if mask is not None else raw
wd = pipeline.split_result['word_segments_dict'][PID]
mfa_align = load_mfa_alignments(PID)

for item in detector_test_ds:
    if item['pid'] != PID: continue
    sid = item['sentence_idx']
    sent_info = wd['sentence_list'][sid]
    sent_eeg = raw_eeg[int(sent_info['stim_start_idx']):int(sent_info['stim_end_idx'])]
    gold = get_mfa_oracle_labels(PID, sid, valid)
    sk = predict_sklearn_mfa(PID, sid, sent_eeg)
    pp, gg = color_matches(sk, gold)
    print(f"\n--- sent {sid} ---")
    print("sklearn MFA:", " ".join(pp))
    print("       gold:", " ".join(gg))

PID = 'P23'   # or whichever you want

# rebuild per-patient bits if needed
vocab = vocabs[PID]
as_str = (lambda i: vocab[int(i)]) if isinstance(vocab, list) \
         else (lambda i: {v:k for k,v in vocab.items()}[int(i)])
valid = set(vocab) if isinstance(vocab, list) else set(vocab.keys())
mask = get_pipeline_channel_mask(pipeline, PID)
raw  = np.load(os.path.join(DUTCH_30_PATH, 'raw', f'{PID}_sEEG.npy'))
raw_eeg = raw[:, mask] if mask is not None else raw
wd = pipeline.split_result['word_segments_dict'][PID]

hits = []
for item in detector_test_ds:
    if item['pid'] != PID: continue
    sid = item['sentence_idx']
    sent_info = wd['sentence_list'][sid]
    sent_eeg = raw_eeg[int(sent_info['stim_start_idx']):int(sent_info['stim_end_idx'])]
    gold = get_mfa_oracle_labels(PID, sid, valid)
    pred_nll = predict_v6(models_nll[PID], scalers[PID], sent_eeg, item['eeg'], item.get('n_phonemes'))
    pred_mrt = predict_v6(models_mrt[PID], scalers[PID], sent_eeg, item['eeg'], item.get('n_phonemes'))
    L_n = has_match(pred_nll, gold, 4)
    L_m = has_match(pred_mrt, gold, 4)
    if L_n >= 4 or L_m >= 4:
        hits.append((sid, L_n, L_m, pred_nll, pred_mrt, gold))

hits.sort(key=lambda x: -max(x[1], x[2]))
print(f"\n{len(hits)} sentences with length-≥4 matches (NLL or MRT):\n")
for sid, L_n, L_m, pn, pm, gold in hits:
    print(f"--- sent {sid}  (NLL longest={L_n}, MRT longest={L_m}) ---")
    pp_n, gg = color_matches(pn, gold)
    pp_m, _  = color_matches(pm, gold)
    print("v6 NLL :", " ".join(pp_n))
    print("v6 MRT :", " ".join(pp_m))
    print("  gold :", " ".join(gg))
    print()

import os, numpy as np, scipy.signal, pandas as pd, torch
import torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score
from collections import defaultdict, Counter

from e2e_brain_decoder import (
    DUTCH_30_PATH, get_pipeline_channel_mask, load_mfa_alignments,
    TRIM_TO_SPEECH, SPEECH_BUFFER_MS,
)

# global constants
EEG_SR        = 1024
DETECTOR_FPS  = 200
WINDOW_MS_EEG = 30.0
FRAMESHIFT_MS = 5.0

# manner mapping (5 classes)
MANNER = {
    'aː':'V','eː':'V','iː':'V','oː':'V','uː':'V','yː':'V','ɛ':'V','ɪ':'V','ɔ':'V',
    'ʏ':'V','œ':'V','ə':'V','ɑ':'V','y':'V','i':'V','e':'V','a':'V','o':'V','u':'V',
    'ɛj':'V','aʊ':'V','ɔɛ':'V','ɛɪ':'V','øː':'V',
    'p':'S','t':'S','k':'S','b':'S','d':'S','c':'S','ɟ':'S','ʔ':'S','g':'S',
    'f':'F','s':'F','ʃ':'F','x':'F','ɣ':'F','v':'F','z':'F','h':'F','ʒ':'F','χ':'F',
    'm':'N','n':'N','ŋ':'N','ɲ':'N',
    'l':'L','r':'L','j':'L','w':'L','ʋ':'L',
}
MANNER_CLASSES = ['V','S','F','N','L']

# train/test sentence-id sets per patient (from the detector dataset split)
# need train_ds + test_ds in scope; if not, rebuild them:
from e2e_brain_decoder import build_joint_dataset_fixed, split_by_sentence, fit_train_stats, apply_stats, ALL_PIDS
full_ds = build_joint_dataset_fixed(pipeline, ALL_PIDS)
train_ds, test_ds = split_by_sentence(full_ds)
eeg_stats  = fit_train_stats(train_ds, 'eeg')
mfcc_stats = fit_train_stats(train_ds, 'mfcc')
train_ds = apply_stats(train_ds, eeg_stats, mfcc_stats)
test_ds  = apply_stats(test_ds,  eeg_stats, mfcc_stats)

train_sids = defaultdict(set); test_sids = defaultdict(set)
for item in train_ds: train_sids[item['pid']].add(item['sentence_idx'])
for item in test_ds:  test_sids[item['pid']].add(item['sentence_idx'])

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def stack_frames(X, k):
    if k == 0: return X
    n, d = X.shape
    pad = np.zeros((k, d), dtype=X.dtype)
    Xp = np.concatenate([pad, X, pad], 0)
    return np.stack([Xp[i:i+2*k+1].reshape(-1) for i in range(n)], 0)

class FrameMLP(nn.Module):
    def __init__(self, n_in, n_out, h=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(h, h//2), nn.BatchNorm1d(h//2), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(h//2, n_out),
        )
    def forward(self, x): return self.net(x)

def train_eval(Xtr_s, ytr_i, Xte_s, yte_i, n_classes, seed=0, epochs=20):
    torch.manual_seed(seed)
    cls_counts = np.bincount(ytr_i, minlength=n_classes)
    weights = 1.0 / np.maximum(cls_counts[ytr_i], 1)
    sampler = WeightedRandomSampler(weights, num_samples=len(ytr_i), replacement=True)
    dl = DataLoader(TensorDataset(torch.as_tensor(Xtr_s, dtype=torch.float32),
                                   torch.as_tensor(ytr_i, dtype=torch.long)),
                    batch_size=512, sampler=sampler, drop_last=True)
    model = FrameMLP(Xtr_s.shape[1], n_classes).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    Xte_t = torch.as_tensor(Xte_s, dtype=torch.float32, device=DEVICE)
    for _ in range(epochs):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            F.cross_entropy(model(xb), yb).backward(); opt.step(); opt.zero_grad()
    model.eval()
    with torch.no_grad():
        pred = model(Xte_t).argmax(1).cpu().numpy()
    return balanced_accuracy_score(yte_i, pred) * 100

# === P22 manner-class: amplitude vs amplitude+amplitude vs amplitude+phase ===
import scipy.signal, numpy as np, os, torch, pandas as pd
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score

PID = 'P30'
K   = 20
EPOCHS = 20
N_NULL = 5
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

BAND_RANGES = {
    'theta':      (4,   8),
    'beta':       (13,  30),
    'low_gamma':  (30,  70),
    'high_gamma': (70, 170),
}

def get_analytic(eeg, low, high, sr=EEG_SR):
    """Bandpass + Hilbert → complex analytic signal (n_samples, n_channels)."""
    data = scipy.signal.detrend(eeg, axis=0)
    sos = scipy.signal.iirfilter(4, [low/(sr/2), high/(sr/2)],
                                  btype='bandpass', output='sos')
    data = scipy.signal.sosfiltfilt(sos, data, axis=0)
    if low <= 50 <= high:
        sos_n = scipy.signal.iirfilter(4, [48/(sr/2), 52/(sr/2)],
                                        btype='bandstop', output='sos')
        data = scipy.signal.sosfiltfilt(sos_n, data, axis=0)
    if low <= 150 <= high:
        sos_n = scipy.signal.iirfilter(4, [148/(sr/2), 152/(sr/2)],
                                        btype='bandstop', output='sos')
        data = scipy.signal.sosfiltfilt(sos_n, data, axis=0)
    return scipy.signal.hilbert(data, axis=0)   # complex

def frame_mean(arr, sr=EEG_SR, window_ms=30.0, frameshift_ms=5.0):
    """Window-mean of a (n_samples, ...) array."""
    win = window_ms / 1000.0; shift = frameshift_ms / 1000.0
    n_win = int(np.floor((arr.shape[0] - win*sr) / (shift*sr)))
    out = np.zeros((n_win,) + arr.shape[1:], dtype=arr.dtype)
    for w in range(n_win):
        s = int(np.floor(w * shift * sr))
        e = int(np.floor(s + win * sr))
        out[w] = arr[s:e].mean(axis=0)
    return out

def build_features_for_pid(pid, feature_kind):
    """feature_kind in {'A','B','C'}."""
    raw = np.load(os.path.join(DUTCH_30_PATH, 'raw', f'{pid}_sEEG.npy'))
    csv = pd.read_csv(os.path.join(DUTCH_30_PATH, 'raw', f'{pid}_electrode_locations.csv'))
    n_csv = len(csv)
    mask = get_pipeline_channel_mask(pipeline, pid)
    if mask is None:
        raw = raw[:, :n_csv]
    else:
        valid = (mask < raw.shape[1]) & (mask < n_csv)
        raw = raw[:, mask[valid]]
    wd  = pipeline.split_result['word_segments_dict'][pid]
    mfa = load_mfa_alignments(pid)

    Xtr, ytr_, Xte, yte_ = [], [], [], []
    for sid in sorted(set(train_sids[pid]) | set(test_sids[pid])):
        sent_info = wd['sentence_list'][sid]
        sent_eeg = raw[int(sent_info['stim_start_idx']):int(sent_info['stim_end_idx'])]
        phones = mfa.get(sid, [])
        if not phones or sent_eeg.shape[0] < int(0.2 * EEG_SR): continue
        if TRIM_TO_SPEECH:
            buf = SPEECH_BUFFER_MS / 1000.0
            sst = max(0.0, phones[0]['start_s'] - buf)
            sed = min(sent_eeg.shape[0]/EEG_SR, phones[-1]['end_s'] + buf)
            if sed <= sst: continue
            sent_eeg = sent_eeg[int(round(sst*EEG_SR)):int(round(sed*EEG_SR))]
            phones = [{**p, 'start_s': p['start_s']-sst,
                            'end_s':   p['end_s']  -sst} for p in phones]

        # Compute features per kind
        feats_parts = []
        if feature_kind in ('A','B','C'):
            an = get_analytic(sent_eeg, *BAND_RANGES['high_gamma'])
            feats_parts.append(frame_mean(np.abs(an)))   # HG amp
        if feature_kind == 'B':
            an = get_analytic(sent_eeg, *BAND_RANGES['beta'])
            feats_parts.append(frame_mean(np.abs(an)))
            an = get_analytic(sent_eeg, *BAND_RANGES['low_gamma'])
            feats_parts.append(frame_mean(np.abs(an)))
        if feature_kind == 'C':
            for band_name in ('beta', 'theta'):
                an = get_analytic(sent_eeg, *BAND_RANGES[band_name])
                ph = np.angle(an)
                feats_parts.append(frame_mean(np.cos(ph)))
                feats_parts.append(frame_mean(np.sin(ph)))
        feats = np.concatenate(feats_parts, axis=1).astype(np.float32)

        # MFA frame labels
        labs = []
        for i in range(feats.shape[0]):
            t = i / DETECTOR_FPS
            lab = next((p['phone'] for p in phones if p['start_s'] <= t < p['end_s']), None)
            labs.append(lab)
        keep = [l is not None for l in labs]
        if not any(keep): continue
        feats = feats[keep]; labs = [l for l, k in zip(labs, keep) if k]
        if sid in train_sids[pid]: Xtr.append(feats); ytr_.extend(labs)
        if sid in test_sids[pid]:  Xte.append(feats); yte_.extend(labs)
    return (np.concatenate(Xtr, 0), np.array(ytr_),
            np.concatenate(Xte, 0), np.array(yte_))

def stack_frames(X, k):
    if k == 0: return X
    n, d = X.shape
    pad = np.zeros((k, d), dtype=X.dtype)
    Xp = np.concatenate([pad, X, pad], 0)
    return np.stack([Xp[i:i+2*k+1].reshape(-1) for i in range(n)], 0)

class FrameMLP(nn.Module):
    def __init__(self, n_in, n_out, h=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(h, h//2), nn.BatchNorm1d(h//2), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(h//2, n_out),
        )
    def forward(self, x): return self.net(x)

def train_eval(Xtr_s, ytr_i, Xte_s, yte_i, n_classes, seed=0, epochs=EPOCHS):
    torch.manual_seed(seed)
    cls_counts = np.bincount(ytr_i, minlength=n_classes)
    weights = 1.0 / np.maximum(cls_counts[ytr_i], 1)
    sampler = WeightedRandomSampler(weights, num_samples=len(ytr_i), replacement=True)
    dl = DataLoader(TensorDataset(torch.as_tensor(Xtr_s, dtype=torch.float32),
                                   torch.as_tensor(ytr_i, dtype=torch.long)),
                    batch_size=512, sampler=sampler, drop_last=True)
    model = FrameMLP(Xtr_s.shape[1], n_classes).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    Xte_t = torch.as_tensor(Xte_s, dtype=torch.float32, device=DEVICE)
    for _ in range(epochs):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            F.cross_entropy(model(xb), yb).backward(); opt.step(); opt.zero_grad()
    model.eval()
    with torch.no_grad():
        pred = model(Xte_t).argmax(1).cpu().numpy()
    return balanced_accuracy_score(yte_i, pred) * 100

print(f"{'kind':<5} {'feat_dim':>9} {'real_bal':>9} {'null_µ':>7} {'null_σ':>7} {'gap':>5} {'z':>6}")
for kind, label in [('A','HG amp'),
                    ('B','HG+beta+lowγ amp'),
                    ('C','HG amp + beta phase + theta phase')]:
    Xb_tr, yb_tr, Xb_te, yb_te = build_features_for_pid(PID, kind)
    mu, sd = Xb_tr.mean(0, keepdims=True), Xb_tr.std(0, keepdims=True) + 1e-8
    Xb_tr = (Xb_tr - mu) / sd; Xb_te = (Xb_te - mu) / sd
    Xb_tr_k = stack_frames(Xb_tr, K); Xb_te_k = stack_frames(Xb_te, K)
    Xb_tr_s = Xb_tr_k.astype(np.float32)
    Xb_te_s = Xb_te_k.astype(np.float32)

    # filter to manner-mappable phonemes
    keep_tr = np.array([p in MANNER for p in yb_tr])
    keep_te = np.array([p in MANNER for p in yb_te])
    yman_tr = np.array([MANNER_CLASSES.index(MANNER[p]) for p in yb_tr[keep_tr]])
    yman_te = np.array([MANNER_CLASSES.index(MANNER[p]) for p in yb_te[keep_te]])
    Xman_tr = Xb_tr_s[keep_tr]; Xman_te = Xb_te_s[keep_te]
    n_man = len(MANNER_CLASSES)

    real = train_eval(Xman_tr, yman_tr, Xman_te, yman_te, n_man, seed=0)
    rng = np.random.default_rng(0)
    nulls = []
    for k in range(N_NULL):
        yp = rng.permutation(yman_tr)
        nulls.append(train_eval(Xman_tr, yp, Xman_te, yman_te, n_man, seed=100+k))
    nulls = np.array(nulls)
    gap = real - nulls.mean()
    z = gap / max(nulls.std(ddof=1), 1e-9)
    print(f"  {kind}   {Xman_tr.shape[1]:>8d}  {real:>7.2f}%  {nulls.mean():>5.2f}% {nulls.std(ddof=1):>5.2f}%  {gap:>+4.1f}  {z:>+5.1f}    [{label}]")

from sklearn.metrics import balanced_accuracy_score

print(f"{'pid':<5} {'pair':<10} {'n_tr':>6} {'n_te':>6} "
      f"{'real_bal':>9} {'null_µ':>7} {'null_σ':>7} {'z':>6}")
for pid in sorted(Xtr):
    if pid not in Xte: continue
    for a, b in PAIRS:
        m_tr = np.isin(ytr[pid], [a, b]); m_te = np.isin(yte[pid], [a, b])
        if m_tr.sum() < 100 or m_te.sum() < 50: continue
        Xa, ya = Xtr[pid][m_tr], (ytr[pid][m_tr] == a).astype(int)
        Xb, yb = Xte[pid][m_te], (yte[pid][m_te] == a).astype(int)
        sc = StandardScaler().fit(Xa)
        Xa_s, Xb_s = sc.transform(Xa), sc.transform(Xb)

        clf = LogisticRegression(max_iter=300, n_jobs=-1, class_weight='balanced').fit(Xa_s, ya)
        real = 100 * balanced_accuracy_score(yb, clf.predict(Xb_s))

        nulls = []
        for _ in range(K_NULL):
            yp = rng.permutation(ya)
            n_clf = LogisticRegression(max_iter=200, n_jobs=-1, class_weight='balanced').fit(Xa_s, yp)
            nulls.append(100 * balanced_accuracy_score(yb, n_clf.predict(Xb_s)))
        nulls = np.array(nulls)
        z = (real - nulls.mean()) / max(nulls.std(ddof=1), 1e-9)
        print(f"{pid:<5} {a+'/'+b:<10} {m_tr.sum():>6} {m_te.sum():>6} "
              f"{real:>8.1f}% {nulls.mean():>6.1f}% {nulls.std(ddof=1):>6.1f}% {z:>+6.1f}")

import numpy as np
rng = np.random.default_rng(0)
nulls_lgb = []
for s in range(5):
    yp = rng.permutation(yman_tr)
    clf = lgb.LGBMClassifier(n_estimators=400, num_leaves=64, learning_rate=0.05,
                             class_weight='balanced', n_jobs=-1, verbose=-1)
    clf.fit(Xtr_flat, yp)
    nulls_lgb.append(balanced_accuracy_score(yman_te, clf.predict(Xte_flat)) * 100)
nulls_lgb = np.array(nulls_lgb)
print(f"  REAL  {30.44:.2f}%")
print(f"  NULL  {nulls_lgb.mean():.2f}% ± {nulls_lgb.std(ddof=1):.2f}%")
print(f"  gap   +{30.44 - nulls_lgb.mean():.1f}    z = {(30.44 - nulls_lgb.mean()) / max(nulls_lgb.std(ddof=1), 1e-9):+.1f}")

import numpy as np
from sklearn.metrics import balanced_accuracy_score
import lightgbm as lgb

PID = 'P22'
# pull pipeline features for this patient
pids_tr = np.asarray(pipeline.train['phoneme_participant_ids'])
pids_te = np.asarray(pipeline.test['phoneme_participant_ids'])
m_tr = pids_tr == PID; m_te = pids_te == PID

X_tr = np.asarray([np.asarray(pipeline.train['features'][i]).flatten()
                   for i in np.where(m_tr)[0]])
X_te = np.asarray([np.asarray(pipeline.test['features'][i]).flatten()
                   for i in np.where(m_te)[0]])
y_tr = np.array([str(pipeline.train['phoneme_labels'][i]) for i in np.where(m_tr)[0]])
y_te = np.array([str(pipeline.test['phoneme_labels'][i]) for i in np.where(m_te)[0]])

# manner mapping
keep_tr = np.array([p in MANNER for p in y_tr])
keep_te = np.array([p in MANNER for p in y_te])
y_man_tr = np.array([MANNER_CLASSES.index(MANNER[p]) for p in y_tr[keep_tr]])
y_man_te = np.array([MANNER_CLASSES.index(MANNER[p]) for p in y_te[keep_te]])
X_man_tr = X_tr[keep_tr]; X_man_te = X_te[keep_te]
print(f"pipeline-features manner: train {X_man_tr.shape}, test {X_man_te.shape}")

clf = lgb.LGBMClassifier(n_estimators=400, num_leaves=64, learning_rate=0.05,
                         class_weight='balanced', n_jobs=-1, verbose=-1)
clf.fit(X_man_tr, y_man_tr)
print(f"  LightGBM on pipeline features: bal_acc = "
      f"{balanced_accuracy_score(y_man_te, clf.predict(X_man_te)) * 100:.2f}%")

# === Audio-supervised boundary + count model for P22 ===
import os, numpy as np, scipy.signal, scipy.fft, torch, pandas as pd
import torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score

PID = 'P22'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Assumes the following are still in scope from prior cells:
#   train_seqs, test_seqs (with normalized 112-ch features and per-frame audio targets)
#   N_CH, audio_teacher_multi, build_one_multi (or your prior multi-target build)
# If not, re-run the multi-target build cell first.

# ---------- model with count head ----------
class BoundaryCountNet(nn.Module):
    def __init__(self, n_in, n_targets=3, hidden=128):
        super().__init__()
        self.proj  = nn.Linear(n_in, hidden)
        self.lnorm = nn.LayerNorm(hidden)
        self.lstm  = nn.LSTM(hidden, hidden, num_layers=2, bidirectional=True,
                             batch_first=True, dropout=0.4)
        self.boundary_head = nn.Linear(hidden*2, n_targets)
        self.count_head    = nn.Sequential(
            nn.Linear(hidden*2, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 1),
        )
    def forward(self, x, in_lens=None):
        z = F.relu(self.lnorm(self.proj(x)))
        H, _ = self.lstm(z)                          # (B, T, 2h)
        b_pred = self.boundary_head(H)               # (B, T, n_targets)
        # sentence-level pooled feature for count head: mean over valid frames
        if in_lens is None:
            pooled = H.mean(dim=1)
        else:
            mask = (torch.arange(H.shape[1], device=H.device)[None, :] <
                    in_lens[:, None].to(H.device)).float().unsqueeze(-1)
            pooled = (H * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        log_k_pred = self.count_head(pooled).squeeze(-1)   # (B,)
        return b_pred, log_k_pred

# ---------- dataset including target log K ----------
class DSc(Dataset):
    def __init__(self, seqs):
        self.seqs = seqs
    def __len__(self): return len(self.seqs)
    def __getitem__(self, i):
        f, a, lab, sid = self.seqs[i]
        # K = number of phonemes in this sentence's MFA alignment, from per-frame label peaks
        # Easier: count from mfa_align directly via sid
        phones = mfa_align.get(sid, [])
        K = max(2, len(phones))
        return (torch.as_tensor(f, dtype=torch.float32),
                torch.as_tensor(a, dtype=torch.float32),
                torch.as_tensor(lab, dtype=torch.float32),
                torch.as_tensor(np.log(K), dtype=torch.float32),
                sid)

def collate_c(batch):
    feats, auds, labs, log_ks, sids = zip(*batch)
    in_lens = torch.as_tensor([f.shape[0] for f in feats], dtype=torch.long)
    F_pad = nn.utils.rnn.pad_sequence(feats, batch_first=True)
    A_pad = nn.utils.rnn.pad_sequence(auds,  batch_first=True)
    L_pad = nn.utils.rnn.pad_sequence(labs,  batch_first=True)
    log_K = torch.stack(log_ks)
    return F_pad, A_pad, L_pad, log_K, in_lens, sids

train_dl = DataLoader(DSc(train_seqs), batch_size=4, shuffle=True, collate_fn=collate_c)
test_dl  = DataLoader(DSc(test_seqs),  batch_size=4, shuffle=False, collate_fn=collate_c)

# ---------- train with joint loss ----------
torch.manual_seed(0)
model = BoundaryCountNet(N_CH, n_targets=3).to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
EPOCHS = 30
LAMBDA_COUNT = 0.3

best = {'auc': 0, 'k_mae': 999, 'ep': 0}

for ep in range(EPOCHS):
    model.train()
    ep_b_loss, ep_k_loss, n = 0.0, 0.0, 0
    for F_pad, A_pad, L_pad, log_K, in_lens, _ in train_dl:
        F_pad, A_pad, log_K = F_pad.to(DEVICE), A_pad.to(DEVICE), log_K.to(DEVICE)
        in_lens_d = in_lens.to(DEVICE)
        b_pred, log_k_pred = model(F_pad, in_lens_d)
        valid = torch.arange(F_pad.shape[1], device=DEVICE)[None, :] < in_lens_d[:, None]
        b_loss = ((b_pred - A_pad)**2 * valid.float().unsqueeze(-1)).sum() / valid.float().sum().clamp(min=1) / 3
        k_loss = F.smooth_l1_loss(log_k_pred, log_K)
        loss = b_loss + LAMBDA_COUNT * k_loss
        opt.zero_grad(); loss.backward(); opt.step()
        ep_b_loss += float(b_loss); ep_k_loss += float(k_loss); n += 1

    # eval
    model.eval()
    auc_preds, auc_labels = [], []
    k_errors = []
    with torch.no_grad():
        for F_pad, A_pad, L_pad, log_K, in_lens, _ in test_dl:
            F_pad = F_pad.to(DEVICE); in_lens_d = in_lens.to(DEVICE)
            b_pred, log_k_pred = model(F_pad, in_lens_d)
            score = b_pred.sum(dim=-1).cpu().numpy()
            for b in range(F_pad.shape[0]):
                T = int(in_lens[b])
                auc_preds.append(score[b, :T])
                auc_labels.append(L_pad[b, :T].numpy())
                K_pred = float(np.exp(log_k_pred[b].cpu().numpy()))
                K_true = float(np.exp(log_K[b].cpu().numpy()))
                k_errors.append(abs(K_pred - K_true) / K_true)
    flat_p = np.concatenate(auc_preds); flat_y = np.concatenate(auc_labels).astype(int)
    try:    auc = roc_auc_score(flat_y, flat_p)
    except: auc = float('nan')
    k_mae_pct = 100 * np.mean(k_errors)
    if auc > best['auc']:
        best.update(auc=auc, k_mae=k_mae_pct, ep=ep+1,
                    state={k: v.detach().clone() for k, v in model.state_dict().items()})
    print(f"  ep{ep+1:02d}  b_loss={ep_b_loss/n:.3f} k_loss={ep_k_loss/n:.3f}  "
          f"AUC={auc:.3f}  K_err={k_mae_pct:.1f}%")

print(f"\nBest: ep{best['ep']}, AUC={best['auc']:.3f}, K_err={best['k_mae']:.1f}%")
model.load_state_dict(best['state'])

# === Unified A/B evaluation: boundary source × K source × patient ===
import os, numpy as np, scipy.signal, torch, pandas as pd
import torch.nn as nn, torch.nn.functional as F
from collections import Counter, defaultdict
from e2e_brain_decoder import (
    DUTCH_30_PATH, get_pipeline_channel_mask, load_mfa_alignments,
    extract_phoneme_feature_pipeline_native, get_mfa_oracle_labels,
    _features_to_dicts_one_seq, edit_distance, train_per_patient_crfs,
    predict_segments, RUN_CONFIG,
)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EEG_SR = 1024; AUDIO_SR_RAW = 48000; AUDIO_SR_TARGET = 16000; TARGET_FPS = 200

# ─────────────────────────────────────────────────────────────────────
# Setup: assumes you have these in scope from earlier cells
#   - pipeline (with split_result, train, test)
#   - classifiers (from train_per_patient_crfs)
#   - v6_model (from train_brain_only_models)
#   - The trained audio-supervised `model` for ONE patient
#   - get_hg_envelope, audio_teacher, BoundaryNet (for re-creating audio-sup per patient)
# ─────────────────────────────────────────────────────────────────────

# --- Boundary source helpers ---
def boundaries_from_audio_sup(model_audio, eeg_features_norm_for_model, K):
    model_audio.eval()
    with torch.no_grad():
        x = torch.as_tensor(eeg_features_norm_for_model, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        in_lens_t = torch.as_tensor([eeg_features_norm_for_model.shape[0]], dtype=torch.long)
        out = model_audio(x, in_lens_t)
        if isinstance(out, tuple):
            b_pred = out[0]
        else:
            b_pred = out
        score = b_pred.sum(-1).squeeze(0).cpu().numpy() if b_pred.dim()==3 else b_pred.squeeze(0).cpu().numpy()
    peaks, _ = scipy.signal.find_peaks(score, distance=8)
    if len(peaks) < (K-1):
        return np.round(np.linspace(0, len(score)-1, K+1)[1:-1]).astype(int)
    top = peaks[np.argsort(-score[peaks])[:K-1]]
    return np.sort(top)

def boundaries_from_v6(v6_model, det_input, pid, n_phon=None):
    """Use predict_segments. Returns list of (start_s, end_s) tuples."""
    return predict_segments(v6_model, det_input, pid, oracle_n_phonemes=n_phon)

def boundaries_from_mfa(phones, T_frames, fps=TARGET_FPS):
    """Returns frame indices of K-1 internal boundaries (between consecutive phones)."""
    bndy_times = [p['start_s'] for p in phones[1:]]   # K-1 internal boundaries
    return np.array([int(round(t * fps)) for t in bndy_times])

# --- K source helpers ---
def predicted_k_audio_sup(score, train_score_per_phone):
    """Estimate K from total boundary mass, calibrated against train statistic."""
    total_mass = max(score.sum(), 1e-6)
    return max(2, int(round(total_mass / train_score_per_phone)))

def fit_audio_sup_k_calibrator(model_audio, train_seqs):
    return 1.0   # placeholder — not used when count head exists

# --- Segment builder from frame-level boundaries to (start_s, end_s) pairs ---
def frame_boundaries_to_segments(internal_boundaries_frames, T_frames, fps=TARGET_FPS):
    edges = [0] + list(internal_boundaries_frames) + [T_frames]
    return [(edges[i]/fps, edges[i+1]/fps) for i in range(len(edges)-1)]

# --- CRF inference for one sentence ---
def crf_decode(sent_eeg_full_channels, segs, classifiers_pid, cfg, stk):
    sc, pca, crf = (classifiers_pid[k] for k in ('scaler','pca','crf'))
    feats = []
    for s_s, e_s in segs:
        ff = extract_phoneme_feature_pipeline_native(
            sent_eeg_full_channels, s_s, e_s, cfg.eeg_sr, cfg.window_length, cfg.frameshift, stk)
        if ff is not None: feats.append(ff)
    if not feats: return []
    X = pca.transform(sc.transform(np.asarray(feats)))
    return list(crf.predict(_features_to_dicts_one_seq(X))[0])

# --- One-condition evaluation over all test sentences ---
def evaluate_one_condition(pid, source, k_source, model_audio, v6_model,
                            train_seqs, test_seqs, raw_eeg_full_for_crf,
                            wd_p, mfa_p, classifiers, det_test_ds, cfg, stk,
                            audio_k_calibrator=None):
    valid_classes = classifiers[pid]['valid_classes']
    out_pairs = []
    det_lookup = {(it['pid'], it['sentence_idx']): it for it in det_test_ds if it['pid'] == pid}

    def run_audio_model(f_arr):
        """Helper: forward pass through audio-sup model. Returns (per-frame-score, predicted_K)."""
        model_audio.eval()
        with torch.no_grad():
            x = torch.as_tensor(f_arr, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            in_lens_t = torch.as_tensor([f_arr.shape[0]], dtype=torch.long)
            out = model_audio(x, in_lens_t) if 'in_lens' in model_audio.forward.__code__.co_varnames else model_audio(x)
            if isinstance(out, tuple):
                b_pred, log_k_pred = out
                K_pred = max(2, int(round(float(np.exp(log_k_pred[0].cpu().numpy())))))
            else:
                b_pred = out
                K_pred = None
            sc = (b_pred.sum(-1).squeeze(0).cpu().numpy() if b_pred.dim() == 3
                  else b_pred.squeeze(0).cpu().numpy())
        return sc, K_pred

    for f, a, lab, sid in test_seqs:
        sent_info = wd_p['sentence_list'][sid]
        eeg_s, eeg_e = int(sent_info['stim_start_idx']), int(sent_info['stim_end_idx'])
        sent_eeg_full = raw_eeg_full_for_crf[eeg_s:eeg_e]
        phones = mfa_p.get(sid, [])
        if not phones: continue
        from e2e_brain_decoder import TRIM_TO_SPEECH, SPEECH_BUFFER_MS
        if TRIM_TO_SPEECH:
            buf = SPEECH_BUFFER_MS / 1000
            sst = max(0.0, phones[0]['start_s'] - buf)
            sed = min(sent_eeg_full.shape[0]/EEG_SR, phones[-1]['end_s'] + buf)
            if sed <= sst: continue
            sent_eeg_full = sent_eeg_full[int(round(sst*EEG_SR)):int(round(sed*EEG_SR))]
            phones_local = [{**p,'start_s':p['start_s']-sst,'end_s':p['end_s']-sst} for p in phones]
        else:
            phones_local = phones

        K_oracle = len(phones_local)
        if K_oracle < 2: continue

        if source == 'mfa':
            K = K_oracle
            internal = boundaries_from_mfa(phones_local, len(f))
            segs = frame_boundaries_to_segments(internal, len(f))
        elif source == 'audio_sup':
            sc, K_pred = run_audio_model(f)
            K = K_oracle if k_source == 'oracle' else (K_pred if K_pred is not None else K_oracle)
            K = min(max(K, 2), len(f) // 4)
            peaks, _ = scipy.signal.find_peaks(sc, distance=8)
            if len(peaks) < (K-1):
                internal = np.round(np.linspace(0, len(sc)-1, K+1)[1:-1]).astype(int)
            else:
                top = peaks[np.argsort(-sc[peaks])[:K-1]]
                internal = np.sort(top)
            segs = frame_boundaries_to_segments(internal, len(f))
        elif source == 'v6':
            it = det_lookup.get((pid, sid))
            if it is None: continue
            n_phon_arg = K_oracle if k_source == 'oracle' else None
            v6_segs = boundaries_from_v6(v6_model, it['eeg'], pid, n_phon=n_phon_arg)
            if not v6_segs: continue
            segs = v6_segs
        else:
            raise ValueError(source)

        y_pred = crf_decode(sent_eeg_full, segs, classifiers[pid], cfg, stk)
        if not y_pred: continue
        y_true = [p['phone'] for p in phones_local if p['phone'] in valid_classes]
        out_pairs.append((y_pred, y_true))
    return out_pairs

# --- Metrics ---
def longest_contig_exact(pred, gold):
    n, m, best = len(pred), len(gold), 0
    for i in range(n):
        for j in range(m):
            k = 0
            while i+k < n and j+k < m and pred[i+k] == gold[j+k]: k += 1
            if k > best: best = k
    return best

def compute_metrics(pairs, train_phonemes):
    if not pairs: return dict(n=0, max_run=0, per=float('nan'), z=float('nan'))
    max_run = max(longest_contig_exact(p, g) for p, g in pairs)
    per = np.mean([edit_distance(p, g) / max(len(g), 1) for p, g in pairs]) * 100
    prior = Counter(train_phonemes); total = sum(prior.values())
    log_p = {p: -np.log(c/total) for p, c in prior.items()}
    def m_surp(pr, gd, mn=3):
        n, m_, ts = len(pr), len(gd), 0.0
        up, ug = [False]*n, [False]*m_
        for L in range(min(n,m_), mn-1, -1):
            for i in range(n-L+1):
                if any(up[i:i+L]): continue
                for j in range(m_-L+1):
                    if any(ug[j:j+L]): continue
                    if pr[i:i+L] == gd[j:j+L]:
                        s = sum(log_p.get(t, np.log(total)) for t in pr[i:i+L])
                        ts += s
                        for k in range(L): up[i+k]=True; ug[j+k]=True
        return ts
    real = sum(m_surp(p, g) for p, g in pairs)
    rng = np.random.default_rng(0)
    flat = [t for p, _ in pairs for t in p]
    nulls = []
    for k in range(50):
        sh = list(rng.permutation(flat))
        sh_seqs, idx = [], 0
        for p, _ in pairs: sh_seqs.append(sh[idx:idx+len(p)]); idx += len(p)
        nulls.append(sum(m_surp(s, g) for s, (_, g) in zip(sh_seqs, pairs)))
    nulls = np.array(nulls)
    z = (real - nulls.mean()) / max(nulls.std(ddof=1), 1e-9)
    return dict(n=len(pairs), max_run=max_run, per=per, z=z)

def run_full_ab(pid, model_audio, train_seqs, test_seqs):
    # globals expected in scope: pipeline, classifiers, v6_model, detector_test_ds
    raw_for_crf = np.load(os.path.join(DUTCH_30_PATH, 'raw', f'{pid}_sEEG.npy'))
    mask = get_pipeline_channel_mask(pipeline, pid)
    if mask is not None: raw_for_crf = raw_for_crf[:, mask]
    wd_p = pipeline.split_result['word_segments_dict'][pid]
    mfa_p = load_mfa_alignments(pid)
    global mfa_align_global; mfa_align_global = mfa_p
    cfg = pipeline.config
    stk = RUN_CONFIG['stacking_order']
    train_phonemes = []
    for sid in sorted(train_sids[pid]):
        train_phonemes += [p['phone'] for p in mfa_p.get(sid, [])
                            if p['phone'] in classifiers[pid]['valid_classes']]
    audio_k_calib = fit_audio_sup_k_calibrator(model_audio, train_seqs)

    rows = []
    for source in ['mfa', 'v6', 'audio_sup']:
        for k_source in ['oracle', 'predicted']:
            if source == 'mfa' and k_source == 'predicted': continue   # nonsensical
            pairs = evaluate_one_condition(
                pid, source, k_source, model_audio, v6_model,
                train_seqs, test_seqs, raw_for_crf, wd_p, mfa_p,
                classifiers, detector_test_ds, cfg, stk,
                audio_k_calibrator=audio_k_calib)
            m = compute_metrics(pairs, train_phonemes)
            rows.append((source, k_source, m))
            print(f"  {pid:<5} {source:<10} {k_source:<10}  n={m['n']:>2}  "
                  f"max_run={m['max_run']:>2}  PER={m['per']:>5.1f}%  z={m['z']:+.1f}")
    return rows

# Run on P22 (assumes train_seqs, test_seqs from your audio-sup training cell are P22)
print("=" * 65)
results_p22 = run_full_ab('P22', model, train_seqs, test_seqs)

import os, numpy as np
from collections import Counter, defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import sklearn_crfsuite

from e2e_brain_decoder import (
    edit_distance, _features_to_dicts_one_seq,
    get_mfa_oracle_labels, extract_phoneme_feature_pipeline_native,
    get_pipeline_channel_mask, load_mfa_alignments,
    DUTCH_30_PATH, RUN_CONFIG,
)

OVERSAMPLE_GAMMA = 1.0   # how aggressively to oversample by surprise.
                          # 0 = no oversampling (equivalent to original CRF).
                          # 1 = factor proportional to avg surprise / mean surprise.
                          # 2 = squared — much heavier emphasis on rare-phoneme sequences.

def train_surprise_weighted_crfs(pipeline, gamma=OVERSAMPLE_GAMMA):
    """Same as train_per_patient_crfs but oversamples training sequences by
    their average phoneme surprise. Returns dict[pid] of scaler/pca/crf/valid."""
    classifiers = {}
    per_pid_feats = defaultdict(list)
    per_pid_labels = defaultdict(list)
    per_pid_words = defaultdict(list)

    for i, pid in enumerate(pipeline.train['phoneme_participant_ids']):
        per_pid_feats[pid].append(np.asarray(pipeline.train['features'][i]).flatten())
        per_pid_labels[pid].append(str(pipeline.train['phoneme_labels'][i]))
        per_pid_words[pid].append(pipeline.train['phoneme_words'][i])

    for pid in sorted(per_pid_feats):
        X_flat = np.asarray(per_pid_feats[pid])
        y_flat = per_pid_labels[pid]
        w_flat = per_pid_words[pid]

        # build train phoneme prior and surprise table
        counts = Counter(y_flat); total = sum(counts.values())
        log_p = {p: -np.log(c/total) for p, c in counts.items()}
        mean_surp = np.mean([log_p[p] for p in y_flat])

        scaler = StandardScaler().fit(X_flat)
        Xs = scaler.transform(X_flat)
        pca = PCA(n_components=50).fit(Xs)
        Xp = pca.transform(Xs)

        # group into word-level sequences
        X_seqs, y_seqs = [], []
        cur_X, cur_y, cur_w = [], [], None
        for x, lab, w in zip(Xp, y_flat, w_flat):
            if w != cur_w and cur_X:
                X_seqs.append(cur_X); y_seqs.append(cur_y)
                cur_X, cur_y = [], []
            cur_w = w
            cur_X.append(x); cur_y.append(lab)
        if cur_X:
            X_seqs.append(cur_X); y_seqs.append(cur_y)

        # oversample by average surprise of the sequence
        os_X, os_y = [], []
        for X_seq, y_seq in zip(X_seqs, y_seqs):
            avg = np.mean([log_p[p] for p in y_seq])
            n_copies = max(1, int(round((avg / mean_surp) ** gamma)))
            for _ in range(n_copies):
                os_X.append(X_seq); os_y.append(y_seq)
        print(f"  {pid}: {len(X_seqs)} → {len(os_X)} sequences after gamma={gamma} oversampling")

        # CRF training
        X_seqs_dicts = [_features_to_dicts_one_seq(np.asarray(seq))[0] for seq in os_X]
        crf = sklearn_crfsuite.CRF(algorithm='lbfgs',
                                    c1=0.1, c2=0.1,
                                    max_iterations=100,
                                    all_possible_transitions=True)
        crf.fit(X_seqs_dicts, os_y)

        classifiers[pid] = {
            'scaler': scaler, 'pca': pca, 'crf': crf,
            'valid_classes': sorted(set(y_flat)),
        }
    return classifiers

# train both, run inference, compare
print("Training baseline CRFs (gamma=2, no oversample):")
classifiers_base = train_surprise_weighted_crfs(pipeline, gamma=2)

print("\nTraining surprise-weighted CRFs (gamma=3):")
classifiers_sw   = train_surprise_weighted_crfs(pipeline, gamma=3)

import os, numpy as np
from collections import defaultdict, Counter
from e2e_brain_decoder import (
    extract_phoneme_feature_pipeline_native, get_pipeline_channel_mask,
    load_mfa_alignments, _features_to_dicts_one_seq, edit_distance,
    DUTCH_30_PATH, RUN_CONFIG,
)

stk_order  = RUN_CONFIG['stacking_order']
eeg_sr     = pipeline.config.eeg_sr
win_len    = pipeline.config.window_length
frameshift = pipeline.config.frameshift

def infer_all_patients(classifiers):
    """Run MFA-boundary CRF inference for every patient. Returns
    per_pid_pred / per_pid_true dicts keyed by pid."""
    per_pid_pred, per_pid_true = defaultdict(list), defaultdict(list)
    raw_cache = {}
    for pid in sorted(classifiers):
        if pid not in raw_cache:
            raw = np.load(os.path.join(DUTCH_30_PATH, 'raw', f'{pid}_sEEG.npy'))
            mask = get_pipeline_channel_mask(pipeline, pid)
            raw_cache[pid] = raw[:, mask] if mask is not None else raw
        wd = pipeline.split_result['word_segments_dict'][pid]
        mfa = load_mfa_alignments(pid)
        valid = classifiers[pid]['valid_classes']
        sc, pca, crf = (classifiers[pid][k] for k in ('scaler','pca','crf'))
        test_sids = sorted(set(mfa.keys()))
        for sid in test_sids:
            sent_info = wd['sentence_list'][sid]
            sent_eeg = raw_cache[pid][int(sent_info['stim_start_idx']):
                                      int(sent_info['stim_end_idx'])]
            phones = mfa.get(sid, [])
            if not phones: continue
            feats = []; y_true_sent = []
            for p in phones:
                if p['phone'] not in valid: continue
                f = extract_phoneme_feature_pipeline_native(
                    sent_eeg, p['start_s'], p['end_s'],
                    eeg_sr, win_len, frameshift, stk_order)
                if f is None: continue
                feats.append(f); y_true_sent.append(p['phone'])
            if not feats: continue
            X = pca.transform(sc.transform(np.asarray(feats)))
            y_pred = list(crf.predict(_features_to_dicts_one_seq(X))[0])
            per_pid_pred[pid].append(y_pred)
            per_pid_true[pid].append(y_true_sent)
    return per_pid_pred, per_pid_true

def longest_contig_exact(p, g):
    n, m, best = len(p), len(g), 0
    for i in range(n):
        for j in range(m):
            k = 0
            while i+k < n and j+k < m and p[i+k] == g[j+k]: k += 1
            if k > best: best = k
    return best

def surprise_z(per_pid_pred, per_pid_true, pid, train_phonemes):
    pairs = list(zip(per_pid_pred[pid], per_pid_true[pid]))
    if not pairs: return dict(max_run=0, per=float('nan'), z=float('nan'))
    max_run = max(longest_contig_exact(p, g) for p, g in pairs)
    per = np.mean([edit_distance(p, g) / max(len(g), 1) for p, g in pairs]) * 100
    prior = Counter(train_phonemes); total = sum(prior.values())
    log_p = {p_: -np.log(c/total) for p_, c in prior.items()}
    def m_surp(pr, gd, mn=3):
        n, m_, ts = len(pr), len(gd), 0.0
        up, ug = [False]*n, [False]*m_
        for L in range(min(n,m_), mn-1, -1):
            for i in range(n-L+1):
                if any(up[i:i+L]): continue
                for j in range(m_-L+1):
                    if any(ug[j:j+L]): continue
                    if pr[i:i+L] == gd[j:j+L]:
                        ts += sum(log_p.get(t, np.log(total)) for t in pr[i:i+L])
                        for k in range(L): up[i+k]=True; ug[j+k]=True
        return ts
    real = sum(m_surp(p, g) for p, g in pairs)
    rng = np.random.default_rng(0)
    flat = [t for p, _ in pairs for t in p]
    nulls = []
    for k in range(50):
        sh = list(rng.permutation(flat))
        sh_seqs, idx = [], 0
        for p, _ in pairs: sh_seqs.append(sh[idx:idx+len(p)]); idx += len(p)
        nulls.append(sum(m_surp(s, g) for s, (_, g) in zip(sh_seqs, pairs)))
    nulls = np.array(nulls)
    z = (real - nulls.mean()) / max(nulls.std(ddof=1), 1e-9)
    return dict(max_run=max_run, per=per, z=z)

# train phoneme labels per patient (for the prior)
train_phonemes_per_pid = {}
for i, pid in enumerate(pipeline.train['phoneme_participant_ids']):
    train_phonemes_per_pid.setdefault(pid, []).append(
        str(pipeline.train['phoneme_labels'][i]))

# run inference for both
print("Running baseline CRF inference ...")
pred_b, true_b = infer_all_patients(classifiers_base)
print("Running surprise-weighted CRF inference ...")
pred_s, true_s = infer_all_patients(classifiers_sw)

# compare per patient
print(f"\n{'pid':<5}  {'baseline':>30}  {'surprise-weighted γ=3':>30}")
print(f"{'':<5}  {'max_run':>8} {'PER%':>8} {'z':>8}   {'max_run':>8} {'PER%':>8} {'z':>8}")
print('-' * 80)
for pid in sorted(pred_b):
    mb = surprise_z(pred_b, true_b, pid, train_phonemes_per_pid[pid])
    ms = surprise_z(pred_s, true_s, pid, train_phonemes_per_pid[pid])
    print(f"{pid:<5}  {mb['max_run']:>8d} {mb['per']:>7.1f}% {mb['z']:>+8.1f}   "
          f"{ms['max_run']:>8d} {ms['per']:>7.1f}% {ms['z']:>+8.1f}")

import importlib, e2e_brain_decoder
importlib.reload(e2e_brain_decoder)

from e2e_brain_decoder import show_matched_sequences_with_times

from e2e_brain_decoder import (
    extract_phoneme_feature_pipeline_native, get_pipeline_channel_mask,
    load_mfa_alignments, _features_to_dicts_one_seq, edit_distance,
    DUTCH_30_PATH, RUN_CONFIG,
)

PID = 'P23'
classifiers = classifiers_sw            # the surprise-weighted dict from earlier
stk_order  = RUN_CONFIG['stacking_order']
eeg_sr     = pipeline.config.eeg_sr
win_len    = pipeline.config.window_length
frameshift = pipeline.config.frameshift

raw  = np.load(os.path.join(DUTCH_30_PATH, 'raw', f'{PID}_sEEG.npy'))
mask = get_pipeline_channel_mask(pipeline, PID)
raw  = raw[:, mask] if mask is not None else raw
wd   = pipeline.split_result['word_segments_dict'][PID]
mfa  = load_mfa_alignments(PID)
valid = classifiers[PID]['valid_classes']
sc, pca, crf = (classifiers[PID][k] for k in ('scaler','pca','crf'))

flat_pred, flat_true, flat_psids, flat_tsids = [], [], [], []
flat_psegs, flat_tsegs = [], []

for sid in sorted(mfa.keys()):
    sent_info = wd['sentence_list'][sid]
    sent_eeg = raw[int(sent_info['stim_start_idx']):int(sent_info['stim_end_idx'])]
    phones = mfa.get(sid, [])
    if not phones: continue
    feats, segs_sentence, y_true_sentence = [], [], []
    for p in phones:
        if p['phone'] not in valid: continue
        f = extract_phoneme_feature_pipeline_native(
            sent_eeg, p['start_s'], p['end_s'],
            eeg_sr, win_len, frameshift, stk_order)
        if f is None: continue
        feats.append(f)
        segs_sentence.append((p['start_s'], p['end_s']))
        y_true_sentence.append(p['phone'])
    if not feats: continue
    X = pca.transform(sc.transform(np.asarray(feats)))
    y_pred = list(crf.predict(_features_to_dicts_one_seq(X))[0])
    flat_pred.extend(y_pred);  flat_true.extend(y_true_sentence)
    flat_psids.extend([sid]*len(y_pred));  flat_tsids.extend([sid]*len(y_true_sentence))
    flat_psegs.extend(segs_sentence);      flat_tsegs.extend(segs_sentence)

if not hasattr(pipeline, 'patient_results') or pipeline.patient_results is None:
    pipeline.patient_results = {}
acc = sum(p == t for p, t in zip(flat_pred, flat_true)) / max(len(flat_true), 1)
ed  = edit_distance(flat_true, flat_pred)
pipeline.patient_results[PID] = {
    'true_labels':       flat_true,
    'predictions':       flat_pred,
    'true_sentence_ids': flat_tsids,
    'pred_sentence_ids': flat_psids,
    'true_segments':     flat_tsegs,
    'pred_segments':     flat_psegs,
    'accuracy':          acc,
    'edit_distance':     ed,
    'per':               ed / max(len(flat_true), 1),
    'n_test':            len(flat_true),
    'n_pred':            len(flat_pred),
}
print(f"P22 packed. acc={acc:.2%}  PER={ed/max(len(flat_true),1):.2%}  n={len(flat_true)}")

# now visualize
from e2e_brain_decoder import show_matched_sequences_with_times
show_matched_sequences_with_times(pipeline, 'P23', collapse_repeats=False)

import os, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score
from collections import Counter
from scipy.ndimage import uniform_filter1d
import numpy as np, pandas as pd
from e2e_brain_decoder import (
    DUTCH_30_PATH, get_pipeline_channel_mask, load_mfa_alignments,
    TRIM_TO_SPEECH, SPEECH_BUFFER_MS, edit_distance,
)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EEG_SR = 1024
TARGET_FPS = 200
K = 20
EPOCHS = 20
STRONG = ['P21','P22','P23','P24','P25','P26','P27','P28','P29','P30']

# Knobs
LAMBDA_MANNER = 0.3
VAL_FRACTION  = 0.10
ALPHA_BIGRAM  = 0.4    # bigram weight in Viterbi
SMOOTH_WIN    = 3      # frames; uniform smoothing on log-posteriors
MANNER_GATE   = 0.3    # 0 = no gating, 1 = strong manner constraint
MIN_CONSENSUS = 3      # min run length after Viterbi+manner

# Manner mapping
MANNER = {
    'aː':'V','eː':'V','iː':'V','oː':'V','uː':'V','yː':'V','ɛ':'V','ɪ':'V','ɔ':'V',
    'ʏ':'V','œ':'V','ə':'V','ɑ':'V','y':'V','i':'V','e':'V','a':'V','o':'V','u':'V',
    'ɛj':'V','aʊ':'V','ɔɛ':'V','ɛɪ':'V','øː':'V',
    'p':'S','t':'S','k':'S','b':'S','d':'S','c':'S','ɟ':'S','ʔ':'S','g':'S',
    'f':'F','s':'F','ʃ':'F','x':'F','ɣ':'F','v':'F','z':'F','h':'F','ʒ':'F','χ':'F',
    'm':'N','n':'N','ŋ':'N','ɲ':'N',
    'l':'L','r':'L','j':'L','w':'L','ʋ':'L',
}
MANNER_CLASSES = ['V', 'S', 'F', 'N', 'L']
MAN_TO_I = {m: i for i, m in enumerate(MANNER_CLASSES)}

def stack_frames(X, k):
    if k == 0: return X
    n, d = X.shape
    pad = np.zeros((k, d), dtype=X.dtype)
    Xp = np.concatenate([pad, X, pad], 0)
    return np.stack([Xp[i:i+2*k+1].reshape(-1) for i in range(n)], 0)

class DualMLP(nn.Module):
    def __init__(self, n_in, n_phon, n_manner=5, h=256):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(n_in, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(h, h//2), nn.BatchNorm1d(h//2), nn.ReLU(), nn.Dropout(0.3),
        )
        self.phon_head   = nn.Linear(h//2, n_phon)
        self.manner_head = nn.Linear(h//2, n_manner)
    def forward(self, x):
        z = self.backbone(x)
        return self.phon_head(z), self.manner_head(z)

def train_dual(Xtr, yphon_i, ymann_i, n_phon, seed=0, epochs=EPOCHS,
               lambda_manner=LAMBDA_MANNER):
    torch.manual_seed(seed)
    counts = np.bincount(yphon_i, minlength=n_phon)
    # weights = 1.0 / np.maximum(counts[yphon_i], 1)
    weights = 1.0 / np.maximum(np.sqrt(counts[yphon_i]), 1)
    sampler = WeightedRandomSampler(weights, num_samples=len(yphon_i), replacement=True)
    dl = DataLoader(TensorDataset(torch.as_tensor(Xtr, dtype=torch.float32),
                                   torch.as_tensor(yphon_i, dtype=torch.long),
                                   torch.as_tensor(ymann_i, dtype=torch.long)),
                    batch_size=256, sampler=sampler, drop_last=True)
    model = DualMLP(Xtr.shape[1], n_phon).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(epochs):
        model.train()
        for xb, yp, ym in dl:
            xb = xb.to(DEVICE); yp = yp.to(DEVICE); ym = ym.to(DEVICE)
            lp, lm = model(xb)
            loss = F.cross_entropy(lp, yp) + lambda_manner * F.cross_entropy(lm, ym)
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    return model

def fit_temperature(model, X_val, y_val, max_iter=50):
    Xt = torch.as_tensor(X_val, dtype=torch.float32, device=DEVICE)
    yt = torch.as_tensor(y_val, dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        logits, _ = model(Xt)
    log_T = nn.Parameter(torch.zeros(1, device=DEVICE))
    opt = torch.optim.LBFGS([log_T], lr=0.1, max_iter=max_iter)
    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / torch.exp(log_T), yt)
        loss.backward()
        return loss
    opt.step(closure)
    return float(torch.exp(log_T).item())

def build_bigram_logp(train_phone_seqs, V, i_to_cls):
    bigram = Counter()
    for seq in train_phone_seqs:
        for a, b in zip(seq[:-1], seq[1:]):
            bigram[(a, b)] += 1
    bg_lp = np.zeros((V, V), dtype=np.float32)
    for i in range(V):
        prev = i_to_cls[i]
        den = sum(bigram.get((prev, i_to_cls[j]), 0) for j in range(V)) + V
        for j in range(V):
            num = bigram.get((prev, i_to_cls[j]), 0) + 1
            bg_lp[i, j] = np.log(num / den)
    return bg_lp

def viterbi_decode(logp_seq, bg_lp, alpha=1.0):
    T_, V = logp_seq.shape
    dp = logp_seq[0].copy()
    backptr = np.zeros((T_, V), dtype=int)
    for t in range(1, T_):
        scores = dp[:, None] + alpha * bg_lp + logp_seq[t][None, :]
        backptr[t] = scores.argmax(0)
        dp = scores.max(0)
    out = [int(dp.argmax())]
    for t in range(T_-1, 0, -1):
        out.append(int(backptr[t, out[-1]]))
    return out[::-1]

print(f"{'pid':<5} {'cls':>4} {'chance%':>8} {'bal%':>6} {'top-1':>6} "
      f"{'top-3':>6} {'top-5':>6} {'T':>5}")
print('-' * 60)

if not hasattr(pipeline, 'patient_results') or pipeline.patient_results is None:
    pipeline.patient_results = {}
if 'saved_models' not in dir() or saved_models is None:
    saved_models = {}

for PID in STRONG:
    raw_p = np.load(os.path.join(DUTCH_30_PATH, 'raw', f'{PID}_sEEG.npy'))
    csv_p = pd.read_csv(os.path.join(DUTCH_30_PATH, 'raw', f'{PID}_electrode_locations.csv'))
    mask_p = get_pipeline_channel_mask(pipeline, PID)
    if mask_p is None: raw_p = raw_p[:, :len(csv_p)]
    else:
        valid = (mask_p < raw_p.shape[1]) & (mask_p < len(csv_p))
        raw_p = raw_p[:, mask_p[valid]]
    wd_p = pipeline.split_result['word_segments_dict'][PID]
    mfa_p = load_mfa_alignments(PID)

    def collect_flat(sids):
        Xs, ys = [], []
        for sid in sorted(sids):
            si = wd_p['sentence_list'][sid]
            sent_eeg = raw_p[int(si['stim_start_idx']):int(si['stim_end_idx'])]
            phones = mfa_p.get(sid, [])
            if not phones or sent_eeg.shape[0] < int(0.2*EEG_SR): continue
            if TRIM_TO_SPEECH:
                buf = SPEECH_BUFFER_MS/1000
                sst = max(0, phones[0]['start_s']-buf)
                sed = min(sent_eeg.shape[0]/EEG_SR, phones[-1]['end_s']+buf)
                if sed <= sst: continue
                sent_eeg = sent_eeg[int(round(sst*EEG_SR)):int(round(sed*EEG_SR))]
                phones = [{**p,'start_s':p['start_s']-sst,'end_s':p['end_s']-sst}
                          for p in phones]
            f = make_modalities(sent_eeg, notch=True)['hg_hilbert_meaned']
            n = f.shape[0]
            lab = np.full(n, '', dtype=object)
            for p in phones:
                s = max(0, int(round(p['start_s']*TARGET_FPS)))
                e = min(n, int(round(p['end_s']*TARGET_FPS)))
                if e > s: lab[s:e] = p['phone']
            keep = lab != ''
            if not keep.any(): continue
            Xs.append(f[keep]); ys.append(lab[keep])
        return np.concatenate(Xs, 0), np.concatenate(ys)

    def collect_per_sentence(sids):
        out = []
        for sid in sorted(sids):
            si = wd_p['sentence_list'][sid]
            sent_eeg = raw_p[int(si['stim_start_idx']):int(si['stim_end_idx'])]
            phones = mfa_p.get(sid, [])
            if not phones or sent_eeg.shape[0] < int(0.2*EEG_SR): continue
            if TRIM_TO_SPEECH:
                buf = SPEECH_BUFFER_MS/1000
                sst = max(0, phones[0]['start_s']-buf)
                sed = min(sent_eeg.shape[0]/EEG_SR, phones[-1]['end_s']+buf)
                if sed <= sst: continue
                sent_eeg = sent_eeg[int(round(sst*EEG_SR)):int(round(sed*EEG_SR))]
                phones = [{**p,'start_s':p['start_s']-sst,'end_s':p['end_s']-sst}
                          for p in phones]
            f = make_modalities(sent_eeg, notch=True)['hg_hilbert_meaned']
            out.append((f, phones, sid))
        return out

    # Train/val split
    train_sid_list = sorted(train_sids[PID])
    rng = np.random.default_rng(0)
    val_n = max(1, int(len(train_sid_list) * VAL_FRACTION))
    val_set = set(rng.choice(train_sid_list, val_n, replace=False))
    actual_train_sids = [s for s in train_sid_list if s not in val_set]
    val_sids_p = sorted(val_set)

    X_tr_raw, y_tr_raw   = collect_flat(actual_train_sids)
    X_val_raw, y_val_raw = collect_flat(val_sids_p)
    X_te_raw, y_te_raw   = collect_flat(test_sids[PID])

    classes = sorted(set(y_tr_raw))
    cls_to_i = {c: i for i, c in enumerate(classes)}
    i_to_cls = {i: c for c, i in cls_to_i.items()}
    n_classes = len(classes)

    def filter_keep(X, y):
        keep = np.array([c in cls_to_i for c in y])
        return X[keep], y[keep]
    X_val_raw, y_val_raw = filter_keep(X_val_raw, y_val_raw)
    X_te_raw, y_te_raw   = filter_keep(X_te_raw, y_te_raw)

    def make_manner(y):
        return np.array([MAN_TO_I.get(MANNER.get(c, '?'), -1) for c in y])
    y_tr_m = make_manner(y_tr_raw)
    train_keep = y_tr_m >= 0
    X_tr_raw, y_tr_raw, y_tr_m = X_tr_raw[train_keep], y_tr_raw[train_keep], y_tr_m[train_keep]

    mu, sd = X_tr_raw.mean(0), X_tr_raw.std(0) + 1e-8
    Xtr_n  = (X_tr_raw  - mu) / sd
    Xval_n = (X_val_raw - mu) / sd
    Xte_n  = (X_te_raw  - mu) / sd

    Xtr  = stack_frames(Xtr_n,  K).astype(np.float32)
    Xval = stack_frames(Xval_n, K).astype(np.float32)
    Xte  = stack_frames(Xte_n,  K).astype(np.float32)
    ytr_i  = np.array([cls_to_i[c] for c in y_tr_raw])
    yval_i = np.array([cls_to_i[c] for c in y_val_raw])
    yte_i  = np.array([cls_to_i[c] for c in y_te_raw])

    # Train + temperature
    model = train_dual(Xtr, ytr_i, y_tr_m, n_classes, seed=0)
    T = fit_temperature(model, Xval, yval_i)

    # Build bigram from train sentences
    train_phone_seqs = [
        [p['phone'] for p in mfa_p.get(sid, []) if p['phone'] in cls_to_i]
        for sid in actual_train_sids
    ]
    bg_lp = build_bigram_logp(train_phone_seqs, n_classes, i_to_cls)
    phone_to_manner = np.array([MAN_TO_I.get(MANNER.get(i_to_cls[i], '?'), -1)
                                  for i in range(n_classes)])

    # Flat-test top-k under temperature
    with torch.no_grad():
        plog_te, _ = model(torch.as_tensor(Xte, dtype=torch.float32, device=DEVICE))
        probs_te = F.softmax(plog_te / T, 1).cpu().numpy()
    pred_te = probs_te.argmax(1)
    bal_acc = balanced_accuracy_score(yte_i, pred_te) * 100
    topk = {}
    for kk in [1, 3, 5]:
        order = np.argsort(-probs_te, axis=1)[:, :kk]
        topk[kk] = np.mean([yte_i[i] in order[i] for i in range(len(yte_i))]) * 100
    chance = 100 / n_classes
    print(f"{PID:<5} {n_classes:>4d} {chance:>7.2f}% {bal_acc:>5.2f}% "
          f"{topk[1]:>5.2f}% {topk[3]:>5.2f}% {topk[5]:>5.2f}% {T:>4.2f}")

    # ---- Per-sentence: smooth → manner-gate → Viterbi → mask silence → consensus → pack ----
    test_per_sent = collect_per_sentence(test_sids[PID])
    flat_pred, flat_true = [], []
    flat_psid, flat_tsid = [], []
    flat_pseg, flat_tseg = [], []

    for f, phones, sid in test_per_sent:
        f_n = (f - mu) / sd
        f_stacked = stack_frames(f_n, K).astype(np.float32)
        with torch.no_grad():
            ft = torch.as_tensor(f_stacked, dtype=torch.float32, device=DEVICE)
            plog, mlog = model(ft)
            phon_logp = F.log_softmax(plog / T, 1).cpu().numpy()       # (T, V)
            manner_p  = F.softmax(mlog, 1).cpu().numpy()               # (T, 5)

        # Smooth log-posteriors
        phon_logp_s = uniform_filter1d(phon_logp, size=SMOOTH_WIN, axis=0)

        # Manner gating: shift each phoneme's log-prob by log(P(its manner))
        soft_mask = np.clip(manner_p[:, phone_to_manner], 1e-6, 1.0)
        gated_logp = phon_logp_s + MANNER_GATE * np.log(soft_mask)

        # Viterbi with bigram transitions
        viterbi_idx = np.array(viterbi_decode(gated_logp, bg_lp, alpha=ALPHA_BIGRAM))

        # Silence mask: only keep predictions inside MFA-defined regions
        n_frames = len(viterbi_idx)
        gold_mask = np.zeros(n_frames, dtype=bool)
        for p in phones:
            s = max(0, int(round(p['start_s']*TARGET_FPS)))
            e = min(n_frames, int(round(p['end_s']*TARGET_FPS)))
            if e > s and p['phone'] in cls_to_i:
                gold_mask[s:e] = True
        masked_idx = np.where(gold_mask, viterbi_idx, -1)

        # Consensus: emit a run only if MIN_CONSENSUS consecutive frames agree
        pred_runs = []
        i = 0
        while i < n_frames:
            cand = masked_idx[i]
            if cand == -1:
                i += 1; continue
            if i + MIN_CONSENSUS <= n_frames and all(masked_idx[i+k] == cand
                                                      for k in range(MIN_CONSENSUS)):
                j = i + MIN_CONSENSUS
                while j < n_frames and masked_idx[j] == cand: j += 1
                pred_runs.append((cand, i, j))
                i = j
            else:
                i += 1

        pred_labels_sent = [i_to_cls[idx] for idx, _, _ in pred_runs]
        pred_segs_sent   = [(s/TARGET_FPS, e/TARGET_FPS) for _, s, e in pred_runs]
        true_labels_sent = [p['phone'] for p in phones if p['phone'] in cls_to_i]
        true_segs_sent   = [(p['start_s'], p['end_s']) for p in phones if p['phone'] in cls_to_i]

        flat_pred.extend(pred_labels_sent); flat_true.extend(true_labels_sent)
        flat_psid.extend([sid]*len(pred_labels_sent))
        flat_tsid.extend([sid]*len(true_labels_sent))
        flat_pseg.extend(pred_segs_sent); flat_tseg.extend(true_segs_sent)

    common = min(len(flat_true), len(flat_pred))
    acc = sum(p == t for p, t in zip(flat_pred[:common], flat_true[:common])) / max(len(flat_true), 1)
    ed = edit_distance(flat_true, flat_pred)
    pipeline.patient_results[PID] = {
        'true_labels':       flat_true,
        'predictions':       flat_pred,
        'true_sentence_ids': flat_tsid,
        'pred_sentence_ids': flat_psid,
        'true_segments':     flat_tseg,
        'pred_segments':     flat_pseg,
        'accuracy':          acc,
        'edit_distance':     ed,
        'per':               ed / max(len(flat_true), 1),
        'n_test':            len(flat_true),
        'n_pred':            len(flat_pred),
    }

    saved_models[PID] = {
        'state':           {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        'mu':              mu.copy(), 'sd': sd.copy(),
        'cls_to_i':        dict(cls_to_i),
        'n_classes':       n_classes,
        'n_input':         Xtr.shape[1],
        'T':               T,
        'bg_lp':           bg_lp,
        'phone_to_manner': phone_to_manner.copy(),
    }

def load_patient_for_inspection(PID):
    """Restore everything needed to inspect a specific patient's dual-head MLP."""
    s = saved_models[PID]
    model = DualMLP(s['n_input'], s['n_classes']).to(DEVICE)
    model.load_state_dict(s['state'])
    model.eval()
    cls_to_i = s['cls_to_i']
    i_to_cls = {i: c for c, i in cls_to_i.items()}
    mu, sd = s['mu'], s['sd']
    T = s.get('T', 1.0)

    raw_p = np.load(os.path.join(DUTCH_30_PATH, 'raw', f'{PID}_sEEG.npy'))
    csv_p = pd.read_csv(os.path.join(DUTCH_30_PATH, 'raw', f'{PID}_electrode_locations.csv'))
    mask_p = get_pipeline_channel_mask(pipeline, PID)
    if mask_p is None: raw_p = raw_p[:, :len(csv_p)]
    else:
        valid = (mask_p < raw_p.shape[1]) & (mask_p < len(csv_p))
        raw_p = raw_p[:, mask_p[valid]]
    wd_p = pipeline.split_result['word_segments_dict'][PID]
    mfa_p = load_mfa_alignments(PID)
    return model, mu, sd, cls_to_i, i_to_cls, raw_p, wd_p, mfa_p, T

PID = 'P22'
model, mu, sd, cls_to_i, i_to_cls, raw_p, wd_p, mfa_p, T = load_patient_for_inspection(PID)
SID = sorted(test_sids[PID])[0]

# Rebuild features for this one sentence the same way as inference
si = wd_p['sentence_list'][SID]
sent_eeg = raw_p[int(si['stim_start_idx']):int(si['stim_end_idx'])]
phones = mfa_p.get(SID, [])
if TRIM_TO_SPEECH:
    buf = SPEECH_BUFFER_MS/1000
    sst = max(0, phones[0]['start_s']-buf)
    sed = min(sent_eeg.shape[0]/EEG_SR, phones[-1]['end_s']+buf)
    sent_eeg = sent_eeg[int(round(sst*EEG_SR)):int(round(sed*EEG_SR))]
    phones = [{**p,'start_s':p['start_s']-sst,'end_s':p['end_s']-sst} for p in phones]

f = make_modalities(sent_eeg, notch=True)['hg_hilbert_meaned']
f_n = (f - mu) / sd
f_stacked = stack_frames(f_n, K).astype(np.float32)

with torch.no_grad():
    ft = torch.as_tensor(f_stacked, dtype=torch.float32, device=DEVICE)
    plog, _ = model(ft)               # DualMLP returns (phoneme_logits, manner_logits)
    probs = F.softmax(plog / T, 1).cpu().numpy()   # apply temperature

n_frames = probs.shape[0]
print(f"Sentence {SID}: {n_frames} frames, {len(phones)} MFA phonemes\n")

# Build per-frame gold label
gold_per_frame = np.full(n_frames, '—', dtype=object)
for p in phones:
    s = max(0, int(round(p['start_s'] * TARGET_FPS)))
    e = min(n_frames, int(round(p['end_s'] * TARGET_FPS)))
    if e > s and p['phone'] in cls_to_i:
        gold_per_frame[s:e] = p['phone']

# ---- Table: top-3 per frame with probabilities ----
print(f"{'frame':>5} {'t (s)':>7} {'gold':>6}   {'top1 (p)':>14}   "
      f"{'top2 (p)':>14}   {'top3 (p)':>14}  {'gold rank':>10} {'gold p':>8}")
print('-' * 100)
# Print every 4th frame to keep it readable (you can change the step)
STEP = 4
for t in range(0, n_frames, STEP):
    sec = t / TARGET_FPS
    top3 = np.argsort(-probs[t])[:3]
    gold_lab = gold_per_frame[t]
    gold_idx = cls_to_i.get(gold_lab, -1)
    gold_rank = int(np.where(np.argsort(-probs[t]) == gold_idx)[0][0]) + 1 if gold_idx >= 0 else -1
    gold_p = probs[t, gold_idx] if gold_idx >= 0 else 0.0
    items = [f"{i_to_cls[k]:>3} ({probs[t,k]:.2f})" for k in top3]
    print(f"{t:>5} {sec:>6.2f} {gold_lab:>6}   {items[0]:>14}   "
          f"{items[1]:>14}   {items[2]:>14}  {gold_rank:>10} {gold_p:>7.2%}")

# Visualize one patient (change PID below to see others)
import importlib, e2e_brain_decoder
importlib.reload(e2e_brain_decoder)
from e2e_brain_decoder import show_matched_sequences_with_times
show_matched_sequences_with_times(pipeline, 'P21', max_per_line = 40,
                                    collapse_repeats=False,
                                    time_align_tol_s=0.02)

import os, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from collections import Counter
from scipy.ndimage import uniform_filter1d
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from e2e_brain_decoder import (
    DUTCH_30_PATH, get_pipeline_channel_mask, load_mfa_alignments,
    TRIM_TO_SPEECH, SPEECH_BUFFER_MS, edit_distance,
)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EEG_SR = 1024
TARGET_FPS = 200
K = 20
BALANCE_MODE = 'inv'   # locked in from prior sweep

# ---- Manner mapping ----
MANNER = {
    'aː':'V','eː':'V','iː':'V','oː':'V','uː':'V','yː':'V','ɛ':'V','ɪ':'V','ɔ':'V',
    'ʏ':'V','œ':'V','ə':'V','ɑ':'V','y':'V','i':'V','e':'V','a':'V','o':'V','u':'V',
    'ɛj':'V','aʊ':'V','ɔɛ':'V','ɛɪ':'V','øː':'V',
    'p':'S','t':'S','k':'S','b':'S','d':'S','c':'S','ɟ':'S','ʔ':'S','g':'S',
    'f':'F','s':'F','ʃ':'F','x':'F','ɣ':'F','v':'F','z':'F','h':'F','ʒ':'F','χ':'F',
    'm':'N','n':'N','ŋ':'N','ɲ':'N',
    'l':'L','r':'L','j':'L','w':'L','ʋ':'L',
}
MANNER_CLASSES = ['V','S','F','N','L']
MAN_TO_I = {m: i for i, m in enumerate(MANNER_CLASSES)}

# ---- Place mapping ----
PLACE = {
    'p':'LAB','b':'LAB','m':'LAB','f':'LAB','v':'LAB','w':'LAB','ʋ':'LAB',
    't':'ALV','d':'ALV','n':'ALV','s':'ALV','z':'ALV','l':'ALV','r':'ALV',
    'ʃ':'PAL','ʒ':'PAL','j':'PAL','c':'PAL','ɟ':'PAL','ɲ':'PAL',
    'k':'VEL','g':'VEL','ŋ':'VEL','x':'VEL','ɣ':'VEL','χ':'VEL',
    'h':'GLO','ʔ':'GLO',
    'iː':'FRT','i':'FRT','ɪ':'FRT','eː':'FRT','e':'FRT','ɛ':'FRT',
    'y':'FRT','yː':'FRT','ʏ':'FRT','øː':'FRT','ɛj':'FRT','ɛɪ':'FRT',
    'ə':'CEN','aː':'CEN','ɑ':'CEN','a':'CEN',
    'uː':'BCK','u':'BCK','oː':'BCK','o':'BCK','ɔ':'BCK','œ':'BCK',
    'aʊ':'BCK','ɔɛ':'BCK',
}
PLACE_CLASSES = ['LAB','ALV','PAL','VEL','GLO','FRT','CEN','BCK']
PLA_TO_I = {p: i for i, p in enumerate(PLACE_CLASSES)}


def stack_frames(X, k):
    if k == 0: return X
    n, d = X.shape
    pad = np.zeros((k, d), dtype=X.dtype)
    Xp = np.concatenate([pad, X, pad], 0)
    return np.stack([Xp[i:i+2*k+1].reshape(-1) for i in range(n)], 0)


def make_weights(yphon_i, n_phon, mode='inv'):
    counts = np.bincount(yphon_i, minlength=n_phon)
    counts_per = np.maximum(counts[yphon_i], 1).astype(np.float64)
    if mode == 'inv':   return 1.0 / counts_per
    if mode == 'sqrt':  return 1.0 / np.sqrt(counts_per)
    if mode == 'cbrt':  return 1.0 / np.cbrt(counts_per)
    if mode == 'log':   return 1.0 / np.log(counts_per + 2)
    if mode == 'none':  return np.ones_like(counts_per)
    raise ValueError(mode)


class TripleMLP(nn.Module):
    def __init__(self, n_in, n_phon, n_manner=5, n_place=8, h=256, dropout=0.3):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(n_in, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h, h//2), nn.BatchNorm1d(h//2), nn.ReLU(), nn.Dropout(dropout),
        )
        self.phon_head   = nn.Linear(h//2, n_phon)
        self.manner_head = nn.Linear(h//2, n_manner)
        self.place_head  = nn.Linear(h//2, n_place)
    def forward(self, x):
        z = self.backbone(x)
        return self.phon_head(z), self.manner_head(z), self.place_head(z)


def train_triple(Xtr, yphon_i, ymann_i, ypla_i, n_phon,
                 hidden=256, dropout=0.3, weight_decay=1e-4, lr=1e-3,
                 epochs=20, lambda_manner=0.3, lambda_place=0.3,
                 cosine=False, balance_mode=BALANCE_MODE, seed=0):
    torch.manual_seed(seed)
    weights = make_weights(yphon_i, n_phon, mode=balance_mode)
    sampler = WeightedRandomSampler(weights, num_samples=len(yphon_i), replacement=True)
    dl = DataLoader(TensorDataset(torch.as_tensor(Xtr, dtype=torch.float32),
                                   torch.as_tensor(yphon_i, dtype=torch.long),
                                   torch.as_tensor(ymann_i, dtype=torch.long),
                                   torch.as_tensor(ypla_i, dtype=torch.long)),
                    batch_size=256, sampler=sampler, drop_last=True)
    model = TripleMLP(Xtr.shape[1], n_phon, h=hidden, dropout=dropout).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
                 if cosine else None)
    for _ in range(epochs):
        model.train()
        for xb, yp, ym, ypl in dl:
            xb = xb.to(DEVICE); yp = yp.to(DEVICE); ym = ym.to(DEVICE); ypl = ypl.to(DEVICE)
            lp, lm, lpl = model(xb)
            loss = (F.cross_entropy(lp, yp)
                    + lambda_manner * F.cross_entropy(lm, ym)
                    + lambda_place  * F.cross_entropy(lpl, ypl))
            opt.zero_grad(); loss.backward(); opt.step()
        if scheduler is not None: scheduler.step()
    model.eval()
    return model


def fit_temperature(model, X_val, y_val, max_iter=50):
    Xt = torch.as_tensor(X_val, dtype=torch.float32, device=DEVICE)
    yt = torch.as_tensor(y_val, dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        logits, _, _ = model(Xt)
    log_T = nn.Parameter(torch.zeros(1, device=DEVICE))
    opt = torch.optim.LBFGS([log_T], lr=0.1, max_iter=max_iter)
    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / torch.exp(log_T), yt)
        loss.backward()
        return loss
    opt.step(closure)
    return float(torch.exp(log_T).item())


def eval_model(model, Xte, yte_i, T=1.0):
    with torch.no_grad():
        logits, _, _ = model(torch.as_tensor(Xte, dtype=torch.float32, device=DEVICE))
        probs = F.softmax(logits / T, 1).cpu().numpy()
    pred = probs.argmax(1)
    bal = balanced_accuracy_score(yte_i, pred) * 100
    topk = {}
    for kk in [1, 3, 5]:
        order = np.argsort(-probs, axis=1)[:, :kk]
        topk[kk] = np.mean([yte_i[i] in order[i] for i in range(len(yte_i))]) * 100
    return bal, topk, pred


def eval_model_manner_gated(model, Xte, yte_i, phone_to_manner, T=1.0):
    """Force phoneme prediction to match manner-head's predicted manner class."""
    with torch.no_grad():
        x = torch.as_tensor(Xte, dtype=torch.float32, device=DEVICE)
        plog, mlog, _ = model(x)
        phone_logp = F.log_softmax(plog / T, 1)
        manner_pred = mlog.argmax(1)
        phone_manner_t = torch.as_tensor(phone_to_manner, device=DEVICE)
        match = phone_manner_t[None, :] == manner_pred[:, None]
        gated = phone_logp + torch.where(match, torch.zeros_like(phone_logp),
                                          torch.full_like(phone_logp, -1e4))
        probs = F.softmax(gated, 1).cpu().numpy()
    pred = probs.argmax(1)
    bal = balanced_accuracy_score(yte_i, pred) * 100
    topk = {}
    for kk in [1, 3, 5]:
        order = np.argsort(-probs, axis=1)[:, :kk]
        topk[kk] = np.mean([yte_i[i] in order[i] for i in range(len(yte_i))]) * 100
    return bal, topk, pred


# ============================================================
# Build P22 data
# ============================================================
PID = 'P22'
raw_p = np.load(os.path.join(DUTCH_30_PATH, 'raw', f'{PID}_sEEG.npy'))
csv_p = pd.read_csv(os.path.join(DUTCH_30_PATH, 'raw', f'{PID}_electrode_locations.csv'))
mask_p = get_pipeline_channel_mask(pipeline, PID)
if mask_p is None: raw_p = raw_p[:, :len(csv_p)]
else:
    valid = (mask_p < raw_p.shape[1]) & (mask_p < len(csv_p))
    raw_p = raw_p[:, mask_p[valid]]
wd_p = pipeline.split_result['word_segments_dict'][PID]
mfa_p = load_mfa_alignments(PID)

def collect_flat(sids):
    Xs, ys = [], []
    for sid in sorted(sids):
        si = wd_p['sentence_list'][sid]
        sent_eeg = raw_p[int(si['stim_start_idx']):int(si['stim_end_idx'])]
        phones = mfa_p.get(sid, [])
        if not phones or sent_eeg.shape[0] < int(0.2*EEG_SR): continue
        if TRIM_TO_SPEECH:
            buf = SPEECH_BUFFER_MS/1000
            sst = max(0, phones[0]['start_s']-buf)
            sed = min(sent_eeg.shape[0]/EEG_SR, phones[-1]['end_s']+buf)
            if sed <= sst: continue
            sent_eeg = sent_eeg[int(round(sst*EEG_SR)):int(round(sed*EEG_SR))]
            phones = [{**p,'start_s':p['start_s']-sst,'end_s':p['end_s']-sst}
                      for p in phones]
        f = make_modalities(sent_eeg, notch=True)['hg_hilbert_meaned']
        n = f.shape[0]
        lab = np.full(n, '', dtype=object)
        for p in phones:
            s = max(0, int(round(p['start_s']*TARGET_FPS)))
            e = min(n, int(round(p['end_s']*TARGET_FPS)))
            if e > s: lab[s:e] = p['phone']
        keep = lab != ''
        if not keep.any(): continue
        Xs.append(f[keep]); ys.append(lab[keep])
    return np.concatenate(Xs, 0), np.concatenate(ys)

train_sid_list = sorted(train_sids[PID])
rng = np.random.default_rng(0)
val_n = max(1, int(len(train_sid_list) * 0.10))
val_set = set(rng.choice(train_sid_list, val_n, replace=False))
actual_train_sids = [s for s in train_sid_list if s not in val_set]
val_sids_p = sorted(val_set)

X_tr_raw, y_tr_raw   = collect_flat(actual_train_sids)
X_val_raw, y_val_raw = collect_flat(val_sids_p)
X_te_raw, y_te_raw   = collect_flat(test_sids[PID])

classes = sorted(set(y_tr_raw))
cls_to_i = {c: i for i, c in enumerate(classes)}
i_to_cls = {i: c for c, i in cls_to_i.items()}
n_classes = len(classes)

def filter_keep(X, y):
    keep = np.array([c in cls_to_i for c in y])
    return X[keep], y[keep]
X_val_raw, y_val_raw = filter_keep(X_val_raw, y_val_raw)
X_te_raw, y_te_raw   = filter_keep(X_te_raw, y_te_raw)

def make_manner(y): return np.array([MAN_TO_I.get(MANNER.get(c, '?'), -1) for c in y])
def make_place (y): return np.array([PLA_TO_I.get(PLACE.get(c,  '?'), -1) for c in y])
y_tr_m   = make_manner(y_tr_raw)
y_tr_pla = make_place(y_tr_raw)
train_keep = (y_tr_m >= 0) & (y_tr_pla >= 0)
X_tr_raw, y_tr_raw = X_tr_raw[train_keep], y_tr_raw[train_keep]
y_tr_m, y_tr_pla   = y_tr_m[train_keep], y_tr_pla[train_keep]

mu, sd = X_tr_raw.mean(0), X_tr_raw.std(0) + 1e-8
Xtr  = stack_frames((X_tr_raw  - mu)/sd, K).astype(np.float32)
Xval = stack_frames((X_val_raw - mu)/sd, K).astype(np.float32)
Xte  = stack_frames((X_te_raw  - mu)/sd, K).astype(np.float32)
ytr_i  = np.array([cls_to_i[c] for c in y_tr_raw])
yval_i = np.array([cls_to_i[c] for c in y_val_raw])
yte_i  = np.array([cls_to_i[c] for c in y_te_raw])
phone_to_manner = np.array([MAN_TO_I.get(MANNER.get(i_to_cls[i], '?'), -1)
                              for i in range(n_classes)])
print(f"P22: train {len(ytr_i)} frames, val {len(yval_i)}, test {len(yte_i)}, "
      f"{n_classes} phonemes; balance_mode={BALANCE_MODE}")

# ============================================================
# Architecture sweep with balance_mode fixed to 'inv'
# ============================================================
CONFIGS = []
for hidden in [128, 256, 512]:
    for dropout in [0.2, 0.4]:
        for wd in [1e-4, 1e-3]:
            for cosine in [False, True]:
                CONFIGS.append(dict(hidden=hidden, dropout=dropout,
                                     weight_decay=wd, cosine=cosine))
rng = np.random.default_rng(0)
CONFIGS = [CONFIGS[i] for i in rng.choice(len(CONFIGS), 10, replace=False)]

print(f"\nSweep ({len(CONFIGS)} configs):\n")
print(f"{'hidden':>6} {'drop':>5} {'wd':>7} {'cos':>4} {'T':>5} "
      f"{'bal%':>6} {'top-1%':>7} {'top-3%':>7} {'top-5%':>7}")
print('-' * 65)

results = []
for cfg in CONFIGS:
    model = train_triple(Xtr, ytr_i, y_tr_m, y_tr_pla, n_classes, epochs=20,
                          hidden=cfg['hidden'], dropout=cfg['dropout'],
                          weight_decay=cfg['weight_decay'], cosine=cfg['cosine'],
                          balance_mode=BALANCE_MODE, seed=0)
    T = fit_temperature(model, Xval, yval_i)
    bal, topk, pred = eval_model(model, Xte, yte_i, T=T)
    results.append(dict(cfg=cfg, T=T, bal=bal, topk=topk, model=model, pred=pred))
    print(f"{cfg['hidden']:>6} {cfg['dropout']:>5.2f} {cfg['weight_decay']:>7.0e} "
          f"{str(cfg['cosine']):>4} {T:>4.2f} {bal:>5.2f}% "
          f"{topk[1]:>6.2f}% {topk[3]:>6.2f}% {topk[5]:>6.2f}%")

best_i = int(np.argmax([r['topk'][1] for r in results]))
best = results[best_i]
print(f"\nBest config (by top-1): {best['cfg']}, top-1 = {best['topk'][1]:.2f}%")

# ============================================================
# Manner-gated evaluation on best model
# ============================================================
bal_g, topk_g, pred_g = eval_model_manner_gated(
    best['model'], Xte, yte_i, phone_to_manner, T=best['T'])
print(f"\nManner-gated evaluation on best model:")
print(f"             {'bal%':>7} {'top-1%':>8} {'top-3%':>8} {'top-5%':>8}")
print(f"  ungated:   {best['bal']:>6.2f}% {best['topk'][1]:>7.2f}% "
      f"{best['topk'][3]:>7.2f}% {best['topk'][5]:>7.2f}%")
print(f"  gated:     {bal_g:>6.2f}% {topk_g[1]:>7.2f}% {topk_g[3]:>7.2f}% {topk_g[5]:>7.2f}%")

# ============================================================
# Confusion matrices: ungated and gated, side by side
# ============================================================
cm_u   = confusion_matrix(yte_i, best['pred'], labels=list(range(n_classes)))
cm_g   = confusion_matrix(yte_i, pred_g, labels=list(range(n_classes)))
cm_u_n = cm_u / np.maximum(cm_u.sum(axis=1, keepdims=True), 1)
cm_g_n = cm_g / np.maximum(cm_g.sum(axis=1, keepdims=True), 1)

# Top-15 confused pairs (gated)
worst = []
for i in range(n_classes):
    row = cm_g_n[i].copy(); row[i] = 0
    for j in np.argsort(-row)[:3]:
        if cm_g[i, j] >= 3:
            worst.append((i_to_cls[i], i_to_cls[j], row[j], int(cm_g[i, j])))
worst.sort(key=lambda x: -x[2])

print(f"\nTop 15 confused pairs (manner-gated):")
print(f"{'true':<6} {'→ pred':<6} {'rate':>7} {'count':>6}")
for t, p, r, c in worst[:15]:
    print(f"  {t:<4} → {p:<4} {r*100:>6.1f}% {c:>5}")

# Per-class recall ungated vs gated
print(f"\nPer-class recall: ungated → gated (top 15 by support):")
rows = []
for i in range(n_classes):
    sup = cm_u[i].sum()
    if sup == 0: continue
    rows.append((i_to_cls[i],
                 cm_u[i, i] / sup * 100,
                 cm_g[i, i] / sup * 100,
                 int(sup)))
rows.sort(key=lambda x: -x[3])
print(f"{'phon':>4} {'support':>8} {'ungated%':>9} {'gated%':>8} {'Δ':>6}")
for p, r_u, r_g, s in rows[:15]:
    print(f"{p:>4} {s:>7} {r_u:>8.2f}% {r_g:>7.2f}% {r_g-r_u:>+5.2f}")

# Side-by-side heatmaps
fig, axes = plt.subplots(1, 2, figsize=(20, 9))
order = sorted(range(n_classes), key=lambda i: -cm_u[i].sum())
labels_ord = [i_to_cls[i] for i in order]
for ax, cm_n, title in zip(axes,
                            [cm_u_n[order][:, order], cm_g_n[order][:, order]],
                            ['ungated', f'manner-gated (top-1 {topk_g[1]:.1f}%)']):
    im = ax.imshow(cm_n, cmap='Blues', vmin=0, vmax=cm_n.max())
    ax.set_xticks(range(n_classes)); ax.set_yticks(range(n_classes))
    ax.set_xticklabels(labels_ord, rotation=90, fontsize=7)
    ax.set_yticklabels(labels_ord, fontsize=7)
    ax.set_xlabel('predicted'); ax.set_ylabel('true')
    ax.set_title(f'P22 confusion ({title})')
plt.tight_layout()
plt.show()

# ============================================================
# Path B — Cell 1: build frame-level (X, BIO, phon, manner, place)
# ============================================================
import os, numpy as np
from scipy.signal import butter, sosfiltfilt, iirnotch, tf2sos, hilbert
from collections import defaultdict
from config import DUTCH_30_PATH
from run_pipeline import load_mfa_alignments

EEG_SR     = 1024
WIN_MS     = 30
SHIFT_MS   = 5
HG_LOW     = 70
HG_HIGH    = 170
NOTCH_HZ   = [50, 150]
FRAME_HZ   = int(1000 / SHIFT_MS)            # 200 Hz
WIN_SAMP   = int(EEG_SR * WIN_MS / 1000)
SHIFT_SAMP = int(EEG_SR * SHIFT_MS / 1000)

# ---- HG extraction (single sentence slice) ----
def _design_filters():
    sos_bp = butter(4, [HG_LOW, HG_HIGH], btype='bandpass', fs=EEG_SR, output='sos')
    sos_notches = []
    for f0 in NOTCH_HZ:
        b, a = iirnotch(f0, 30, EEG_SR)
        sos_notches.append(tf2sos(b, a))
    return sos_bp, sos_notches

_SOS_BP, _SOS_NOTCH = _design_filters()

def extract_hg_frames(eeg_slice):
    """eeg_slice: (T_samples, n_ch). Returns (T_frames, n_ch) log-power HG."""
    x = eeg_slice.astype(np.float64)
    for sos in _SOS_NOTCH:
        x = sosfiltfilt(sos, x, axis=0)
    x = sosfiltfilt(_SOS_BP, x, axis=0)
    env = np.abs(hilbert(x, axis=0))               # (T, ch)
    T = env.shape[0]
    n_frames = max(0, (T - WIN_SAMP) // SHIFT_SAMP + 1)
    out = np.zeros((n_frames, env.shape[1]), dtype=np.float32)
    for k in range(n_frames):
        s = k * SHIFT_SAMP
        out[k] = env[s:s + WIN_SAMP].mean(axis=0)
    # log-compress for stability
    out = np.log1p(out)
    return out

def stack_context(X, K=5):
    """Sliding-window stack: out[t] = concat(X[t-K..t+K]). Edges zero-padded."""
    T, C = X.shape
    pad = np.zeros((K, C), dtype=X.dtype)
    Xp = np.vstack([pad, X, pad])
    cols = [Xp[k:k + T] for k in range(2 * K + 1)]
    return np.concatenate(cols, axis=1)            # (T, C*(2K+1))

# ---- BIO label builder ----
def build_bio_labels(mfa_phones, n_frames):
    """For each frame, return (bio_tag, phon_sym, in_speech) where bio_tag is
    'B-<phon>' or 'I-<phon>' or 'O'. Frame-center time in seconds:
        t = (k*SHIFT_SAMP + WIN_SAMP/2) / EEG_SR
    """
    bio = ['O'] * n_frames
    phon = [None] * n_frames
    for ph in mfa_phones:
        s_s, e_s = ph['start_s'], ph['end_s']
        sym = ph['phone']
        # find frames whose center falls inside [s_s, e_s)
        k_start = int(np.ceil((s_s * EEG_SR - WIN_SAMP / 2) / SHIFT_SAMP))
        k_end   = int(np.floor((e_s * EEG_SR - WIN_SAMP / 2) / SHIFT_SAMP))
        k_start = max(0, k_start)
        k_end   = min(n_frames - 1, k_end)
        if k_end < k_start:
            continue
        bio[k_start] = f'B-{sym}'
        phon[k_start] = sym
        for k in range(k_start + 1, k_end + 1):
            bio[k] = f'I-{sym}'
            phon[k] = sym
    return bio, phon

# ---- Phoneme → manner / place tables (your existing maps) ----
# Use the saved_models maps to stay consistent.
def get_manner_table(saved_models, pid):
    state = saved_models[pid]
    cls_to_i = state['cls_to_i']
    pm_arr = np.asarray(state['phone_to_manner']).astype(int)
    i_to_cls = {v: k for k, v in cls_to_i.items()}
    return {i_to_cls[i]: int(pm_arr[i]) for i in range(len(pm_arr))}

# place table: you defined PLACE earlier — paste/reuse the dict.
# Fallback: derive from phoneme symbol if PLACE is in scope.
try:
    PLACE_TABLE = PLACE
except NameError:
    PLACE_TABLE = {}   # set to your PLACE dict if not in scope

# ---- Main: build per-patient frame dataset ----
def build_frame_dataset(pid, pipeline, saved_models, channel_mask=None):
    """Returns dict with concatenated frame arrays + sentence-id vector."""
    raw_eeg = np.load(os.path.join(DUTCH_30_PATH, 'raw', f'{pid}_sEEG.npy'))
    # raw_eeg: (T_total, n_channels_raw)
    if raw_eeg.ndim == 2 and raw_eeg.shape[0] < raw_eeg.shape[1]:
        raw_eeg = raw_eeg.T   # ensure (samples, channels)

    word_data = pipeline.split_result['word_segments_dict'][pid]
    mfa = load_mfa_alignments(pid)

    # Channel mask: use the same as your TripleMLP training (114 ch then stacked)
    if channel_mask is not None:
        raw_eeg = raw_eeg[:, channel_mask]

    # train/test sentence sets — your split is at instance level; for BIO we
    # split at sentence level. Use the test set sentences from your saved
    # split_result['test'] where possible.
    test_sent_ids = set()
    if 'test' in pipeline.split_result:
        for inst in pipeline.split_result['test'].get(pid, []):
            if isinstance(inst, dict) and 'sentence_idx' in inst:
                test_sent_ids.add(inst['sentence_idx'])
    # fallback: every 6th sentence — REPLACE with your real split keying
    if not test_sent_ids:
        all_real = [i for i, s in enumerate(word_data['sentence_list'])
                    if isinstance(s, dict) and s.get('text')]
        test_sent_ids = set(all_real[::6])
        print(f"  [{pid}] WARNING: using fallback test split (every 6th sentence)")

    manner_map = get_manner_table(saved_models, pid)

    out = {'train': defaultdict(list), 'test': defaultdict(list)}
    n_used = n_skip = 0

    for sent_idx, sent in enumerate(word_data['sentence_list']):
        if not isinstance(sent, dict) or not sent.get('text'):
            continue
        if sent_idx not in mfa or not mfa[sent_idx]:
            n_skip += 1; continue

        s0 = sent['stim_start_idx']; s1 = sent['stim_end_idx']
        if s1 > raw_eeg.shape[0]:
            n_skip += 1; continue
        eeg_slice = raw_eeg[s0:s1]
        X = extract_hg_frames(eeg_slice)             # (T_frames, n_ch)
        if X.shape[0] < 11:
            n_skip += 1; continue

        bio, phon = build_bio_labels(mfa[sent_idx], X.shape[0])
        Xs = stack_context(X, K=5)                   # (T_frames, n_ch*11)

        split = 'test' if sent_idx in test_sent_ids else 'train'
        out[split]['X'].append(Xs)
        out[split]['bio'].append(np.array(bio))
        out[split]['phon'].append(np.array(phon))
        out[split]['manner'].append(np.array(
            [manner_map.get(p, -1) if p else -1 for p in phon]))
        out[split]['sent_idx'].append(np.full(X.shape[0], sent_idx, dtype=int))
        n_used += 1

    # concatenate
    result = {}
    for split in ('train', 'test'):
        if not out[split]['X']:
            result[split] = None; continue
        result[split] = {
            'X':        np.concatenate(out[split]['X'], axis=0),
            'bio':      np.concatenate(out[split]['bio']),
            'phon':     np.concatenate(out[split]['phon']),
            'manner':   np.concatenate(out[split]['manner']),
            'sent_idx': np.concatenate(out[split]['sent_idx']),
        }
    print(f"  [{pid}] used={n_used} skipped={n_skip}  "
          f"train_frames={result['train']['X'].shape[0] if result['train'] else 0}  "
          f"test_frames={result['test']['X'].shape[0] if result['test'] else 0}")
    return result

# ---- Run for target patients ----
TARGET_PIDS = ['P22', 'P23', 'P26', 'P29']
frame_datasets = {}
for pid in TARGET_PIDS:
    print(f"\nBuilding {pid}...")
    try:
        frame_datasets[pid] = build_frame_dataset(pid, pipeline, saved_models)
    except Exception as e:
        print(f"  [{pid}] FAILED: {type(e).__name__}: {e}")

# Quick sanity printout
for pid, d in frame_datasets.items():
    tr = d['train']; te = d['test']
    if tr is None: continue
    from collections import Counter
    bio_cnt = Counter(tr['bio'])
    print(f"\n{pid} train BIO sample: "
          f"B={sum(v for k,v in bio_cnt.items() if k.startswith('B-'))} "
          f"I={sum(v for k,v in bio_cnt.items() if k.startswith('I-'))} "
          f"O={bio_cnt.get('O', 0)}")
    print(f"  feature dim: {tr['X'].shape[1]}")

# ============================================================
# Path B — Cell 2: BIOMLP + linear-chain CRF
# ============================================================
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- Tag system ----
# Tag layout: index 0 = O, then for each phoneme p in cls_to_i order:
#   index 1 + 2*p     = B-<p>
#   index 1 + 2*p + 1 = I-<p>
def build_tag_index(cls_to_i):
    """Return (tag_to_idx, idx_to_tag, n_tags)."""
    tag_to_idx = {'O': 0}
    idx_to_tag = ['O']
    for ph, c in sorted(cls_to_i.items(), key=lambda kv: kv[1]):
        tag_to_idx[f'B-{ph}'] = len(idx_to_tag); idx_to_tag.append(f'B-{ph}')
        tag_to_idx[f'I-{ph}'] = len(idx_to_tag); idx_to_tag.append(f'I-{ph}')
    return tag_to_idx, idx_to_tag, len(idx_to_tag)

def build_transition_mask(idx_to_tag):
    """(n_tags, n_tags) tensor with 0 on allowed transitions, -inf on forbidden.
       mask[i, j] = score added to T[i->j]. We forbid I-X following anything
       other than B-X or I-X."""
    n = len(idx_to_tag)
    mask = torch.zeros(n, n)
    # parse tag info
    info = []  # (kind, phone)  kind in {'O','B','I'}
    for t in idx_to_tag:
        if t == 'O': info.append(('O', None))
        elif t.startswith('B-'): info.append(('B', t[2:]))
        elif t.startswith('I-'): info.append(('I', t[2:]))
    for j, (kj, pj) in enumerate(info):
        if kj == 'I':
            # only B-pj or I-pj can precede I-pj
            for i, (ki, pi) in enumerate(info):
                if not ((ki == 'B' and pi == pj) or (ki == 'I' and pi == pj)):
                    mask[i, j] = float('-inf')
    return mask

def init_transition_matrix(idx_to_tag, bg_lp=None, cls_to_i=None):
    """Initialize learnable transitions. Bigram log-probs go on B-X -> B-Y
       transitions, scaled. Others zero."""
    n = len(idx_to_tag)
    T = torch.zeros(n, n)
    if bg_lp is None or cls_to_i is None:
        return T
    bg = torch.from_numpy(bg_lp).float()             # (n_phon, n_phon)
    # B-X index for phoneme p with idx c: 1 + 2*c
    for ph_a, ca in cls_to_i.items():
        ba = 1 + 2 * ca
        for ph_b, cb in cls_to_i.items():
            bb = 1 + 2 * cb
            T[ba, bb] = bg[ca, cb] * 0.5             # gentle init
    return T


# ---- Linear-chain CRF ----
class LinearChainCRF(nn.Module):
    def __init__(self, n_tags, transition_mask, transition_init):
        super().__init__()
        self.n_tags = n_tags
        self.trans = nn.Parameter(transition_init.clone())
        self.start = nn.Parameter(torch.zeros(n_tags))
        self.end   = nn.Parameter(torch.zeros(n_tags))
        self.register_buffer('mask', transition_mask)

    def _masked_trans(self):
        return self.trans + self.mask                # forbidden = -inf

    def _forward_alg(self, emissions):
        # emissions: (T, n_tags)
        T_ = emissions.size(0)
        trans = self._masked_trans()                 # (K, K)
        alpha = self.start + emissions[0]            # (K,)
        for t in range(1, T_):
            # alpha_new[j] = logsumexp_i(alpha[i] + trans[i,j]) + emissions[t,j]
            alpha = torch.logsumexp(
                alpha.unsqueeze(1) + trans, dim=0
            ) + emissions[t]
        return torch.logsumexp(alpha + self.end, dim=0)

    def _score(self, emissions, tags):
        # tags: (T,) long
        T_ = emissions.size(0)
        trans = self._masked_trans()
        score = self.start[tags[0]] + emissions[0, tags[0]]
        for t in range(1, T_):
            score = score + trans[tags[t-1], tags[t]] + emissions[t, tags[t]]
        score = score + self.end[tags[-1]]
        return score

    def neg_log_likelihood(self, emissions, tags):
        return self._forward_alg(emissions) - self._score(emissions, tags)

    def viterbi(self, emissions):
        T_ = emissions.size(0)
        trans = self._masked_trans()
        bp = torch.zeros(T_, self.n_tags, dtype=torch.long, device=emissions.device)
        vit = self.start + emissions[0]
        for t in range(1, T_):
            scores = vit.unsqueeze(1) + trans        # (K_prev, K_curr)
            vit, bp[t] = scores.max(dim=0)
            vit = vit + emissions[t]
        vit = vit + self.end
        last = int(vit.argmax().item())
        path = [last]
        for t in range(T_ - 1, 0, -1):
            last = int(bp[t, last].item())
            path.append(last)
        return list(reversed(path))


# ---- BIOMLP backbone ----
class BIOMLP(nn.Module):
    def __init__(self, n_in, n_phon, n_manner=5, n_place=8, h=256, dropout=0.3,
                 transition_mask=None, transition_init=None, n_tags=None):
        super().__init__()
        self.n_phon = n_phon
        self.n_tags = n_tags
        self.backbone = nn.Sequential(
            nn.Linear(n_in, h), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(h, h),     nn.GELU(), nn.Dropout(dropout),
        )
        self.bio_head    = nn.Linear(h, n_tags)
        self.manner_head = nn.Linear(h, n_manner)
        self.place_head  = nn.Linear(h, n_place)
        self.crf = LinearChainCRF(n_tags, transition_mask, transition_init)

    def emissions(self, x):
        h = self.backbone(x)
        return self.bio_head(h), self.manner_head(h), self.place_head(h), h

    def loss(self, x, tags, manner, place,
             lam_manner=0.3, lam_place=0.1, b_boost=2.0):
        bio_em, mn_em, pl_em, _ = self.emissions(x)
        # add a constant boost to B-tag emissions during training to counter
        # B-rarity (~3-5% of frames)
        if b_boost != 0:
            # B-tag indices: 1, 3, 5, ...
            b_idx = torch.arange(1, self.n_tags, 2, device=bio_em.device)
            bio_em_boosted = bio_em.clone()
            bio_em_boosted[:, b_idx] = bio_em_boosted[:, b_idx] + b_boost
        else:
            bio_em_boosted = bio_em
        crf_nll = self.crf.neg_log_likelihood(bio_em_boosted, tags) / x.size(0)

        # manner/place CE on every frame (silence frames get manner=-1; mask)
        valid = manner >= 0
        if valid.sum() > 0:
            mn_loss = F.cross_entropy(mn_em[valid], manner[valid])
        else:
            mn_loss = torch.tensor(0.0, device=x.device)
        if place is not None and (place >= 0).sum() > 0:
            valid_p = place >= 0
            pl_loss = F.cross_entropy(pl_em[valid_p], place[valid_p])
        else:
            pl_loss = torch.tensor(0.0, device=x.device)

        return crf_nll + lam_manner * mn_loss + lam_place * pl_loss, {
            'crf': float(crf_nll.item()), 'mn': float(mn_loss.item()),
            'pl': float(pl_loss.item()),
        }

    @torch.no_grad()
    def decode(self, x):
        bio_em, _, _, _ = self.emissions(x)
        return self.crf.viterbi(bio_em)


# ---- Smoke test: instantiate for P22 ----
def make_model_for(pid, frame_datasets, saved_models, h=256, dropout=0.3):
    fd = frame_datasets[pid]
    state = saved_models[pid]
    cls_to_i = state['cls_to_i']
    bg_lp = state['bg_lp']
    pm_arr = np.asarray(state['phone_to_manner']).astype(int)

    n_in = fd['train']['X'].shape[1]
    n_phon = len(cls_to_i)
    n_manner = int(pm_arr.max()) + 1
    n_place = 8  # adjust if PLACE_TABLE has different size

    tag_to_idx, idx_to_tag, n_tags = build_tag_index(cls_to_i)
    trans_mask = build_transition_mask(idx_to_tag)
    trans_init = init_transition_matrix(idx_to_tag, bg_lp, cls_to_i)

    model = BIOMLP(n_in=n_in, n_phon=n_phon, n_manner=n_manner, n_place=n_place,
                   h=h, dropout=dropout,
                   transition_mask=trans_mask, transition_init=trans_init,
                   n_tags=n_tags)
    return model, tag_to_idx, idx_to_tag

# Quick instantiation test
for pid in TARGET_PIDS:
    if pid not in frame_datasets or frame_datasets[pid]['train'] is None:
        continue
    model, t2i, i2t = make_model_for(pid, frame_datasets, saved_models)
    n_tags = model.n_tags
    n_params = sum(p.numel() for p in model.parameters())
    n_forbidden = (model.crf.mask == float('-inf')).sum().item()
    print(f"{pid}: n_tags={n_tags}  params={n_params/1e6:.2f}M  "
          f"forbidden_transitions={n_forbidden}/{n_tags*n_tags}")

# ============================================================
# Path B — Cell 3: per-patient training (CRF NLL + aux heads)
# ============================================================
import math, time
import torch.optim as optim

EPOCHS         = 30
LR             = 3e-4
WEIGHT_DECAY   = 1e-3
LAM_MANNER     = 0.3
LAM_PLACE      = 0.1
B_BOOST        = 2.0
GRAD_CLIP      = 5.0
MIN_SENT_FRAMES = 30      # skip very short sentences
DEVICE         = 'cuda' if torch.cuda.is_available() else 'cpu'

def split_into_sentences(split_dict, tag_to_idx, place_table=None):
    """From {X, bio, phon, manner, sent_idx} concatenated arrays, build a
       list of per-sentence dicts with tensors ready for CRF."""
    sents = []
    sidx = split_dict['sent_idx']
    boundaries = np.where(np.diff(sidx, prepend=sidx[0]-1) != 0)[0].tolist() + [len(sidx)]
    for k in range(len(boundaries) - 1):
        s, e = boundaries[k], boundaries[k + 1]
        if e - s < MIN_SENT_FRAMES:
            continue
        bio_str = split_dict['bio'][s:e]
        tags = np.array([tag_to_idx.get(t, 0) for t in bio_str], dtype=np.int64)
        manner = split_dict['manner'][s:e].astype(np.int64)
        # Place: derive from phoneme if table given, else -1
        if place_table is not None:
            phon = split_dict['phon'][s:e]
            place = np.array([place_table.get(p, -1) if p else -1 for p in phon],
                             dtype=np.int64)
        else:
            place = np.full_like(manner, -1)
        sents.append({
            'X':      torch.from_numpy(split_dict['X'][s:e]).float(),
            'tags':   torch.from_numpy(tags),
            'manner': torch.from_numpy(manner),
            'place':  torch.from_numpy(place),
            'sent_idx': int(sidx[s]),
        })
    return sents


def standardize_inplace(sents, mu, sd):
    sd = np.where(sd < 1e-6, 1.0, sd)
    for s in sents:
        s['X'] = ((s['X'] - torch.from_numpy(mu).float()) /
                  torch.from_numpy(sd).float())


def fit_mu_sd(sents):
    Xall = torch.cat([s['X'] for s in sents], dim=0).numpy()
    return Xall.mean(0), Xall.std(0)


def evaluate_quick(model, sents_te, idx_to_tag):
    """Tag-level accuracy on test (a coarse sanity metric)."""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for s in sents_te:
            path = model.decode(s['X'].to(DEVICE))
            tg = s['tags'].tolist()
            for p, t in zip(path, tg):
                correct += (p == t)
                total += 1
    return correct / max(total, 1)


def train_one_patient(pid, frame_datasets, saved_models,
                      h=256, dropout=0.3, epochs=EPOCHS, place_table=None,
                      verbose=True):
    fd = frame_datasets[pid]
    if fd['train'] is None or fd['test'] is None:
        print(f"[{pid}] missing split, skipping"); return None

    model, tag_to_idx, idx_to_tag = make_model_for(
        pid, frame_datasets, saved_models, h=h, dropout=dropout)
    model = model.to(DEVICE)

    sents_tr = split_into_sentences(fd['train'], tag_to_idx, place_table)
    sents_te = split_into_sentences(fd['test'],  tag_to_idx, place_table)
    if not sents_tr:
        print(f"[{pid}] no training sentences"); return None

    mu, sd = fit_mu_sd(sents_tr)
    standardize_inplace(sents_tr, mu, sd)
    standardize_inplace(sents_te, mu, sd)

    # move sentences to device once (small enough)
    for s in sents_tr + sents_te:
        for k in ('X', 'tags', 'manner', 'place'):
            s[k] = s[k].to(DEVICE)

    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    history = []
    best_acc = 0.0; best_state = None
    rng = np.random.default_rng(0)

    for ep in range(epochs):
        model.train()
        perm = rng.permutation(len(sents_tr))
        losses = {'total': 0, 'crf': 0, 'mn': 0, 'pl': 0}
        for idx in perm:
            s = sents_tr[idx]
            opt.zero_grad()
            loss, parts = model.loss(
                s['X'], s['tags'], s['manner'], s['place'],
                lam_manner=LAM_MANNER, lam_place=LAM_PLACE, b_boost=B_BOOST)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            losses['total'] += float(loss.item())
            for k in ('crf', 'mn', 'pl'): losses[k] += parts[k]
        sched.step()

        n = len(sents_tr)
        avg = {k: v / n for k, v in losses.items()}
        if (ep + 1) % 5 == 0 or ep == 0:
            acc = evaluate_quick(model, sents_te, idx_to_tag)
            if acc > best_acc:
                best_acc = acc
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
            if verbose:
                print(f"  [{pid}] ep{ep+1:3d} "
                      f"loss={avg['total']:6.2f} crf={avg['crf']:6.2f} "
                      f"mn={avg['mn']:.3f} pl={avg['pl']:.3f}  "
                      f"test_tag_acc={acc:.3f} (best={best_acc:.3f})")
        history.append(avg)

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        'model': model.cpu(),
        'mu': mu, 'sd': sd,
        'tag_to_idx': tag_to_idx, 'idx_to_tag': idx_to_tag,
        'sents_tr': [{k: v.cpu() for k, v in s.items() if torch.is_tensor(v)}
                     | {'sent_idx': s['sent_idx']} for s in sents_tr],
        'sents_te': [{k: v.cpu() for k, v in s.items() if torch.is_tensor(v)}
                     | {'sent_idx': s['sent_idx']} for s in sents_te],
        'history': history, 'best_acc': best_acc,
    }


# ---- Train all four patients ----
bio_results = {}
for pid in TARGET_PIDS:
    if pid not in frame_datasets or frame_datasets[pid]['train'] is None:
        continue
    print(f"\n=== Training {pid} ===")
    t0 = time.time()
    bio_results[pid] = train_one_patient(
        pid, frame_datasets, saved_models, h=256, dropout=0.3, epochs=EPOCHS)
    print(f"  [{pid}] done in {time.time() - t0:.1f}s, best_tag_acc={bio_results[pid]['best_acc']:.3f}")

# ============================================================
# Path B — Cell 4: decode and score BIO-CRF predictions
# ============================================================
# Requires from earlier cells:
#   - longest_run_with_shift, collect_matches, surprise_score,
#     perm_null, score_run, collapse_consecutive
#   - N_PERM, SHIFT_MAX, MIN_MATCH

def decode_to_phoneme_sequence(tag_idx_path, idx_to_tag):
    """Collapse a Viterbi path of BIO tag indices into a phoneme sequence.
       Rules:
         B-X starts a new phoneme X.
         I-X continues current phoneme if it's also X (otherwise it's a no-op
           given the structural mask, but we still handle it defensively).
         O ends any current phoneme.
    """
    out = []
    cur = None
    for idx in tag_idx_path:
        tag = idx_to_tag[idx]
        if tag == 'O':
            cur = None
        elif tag.startswith('B-'):
            ph = tag[2:]
            out.append(ph)
            cur = ph
        elif tag.startswith('I-'):
            ph = tag[2:]
            if cur is None:
                # I-X without preceding B-X (shouldn't happen with mask)
                out.append(ph)
                cur = ph
    return out


def gold_phoneme_sequence(gold_bio_strs):
    """Same collapse rule on gold BIO labels."""
    out, cur = [], None
    for tag in gold_bio_strs:
        if tag == 'O':
            cur = None
        elif tag.startswith('B-'):
            ph = tag[2:]
            out.append(ph); cur = ph
        elif tag.startswith('I-'):
            ph = tag[2:]
            if cur is None:
                out.append(ph); cur = ph
    return out


def score_bio_patient(pid, bio_results, target_pids=None,
                      n_perm=2000, min_match=3, shift_max=3):
    if pid not in bio_results or bio_results[pid] is None:
        print(f"[{pid}] no trained model"); return None
    res = bio_results[pid]
    model = res['model'].eval()
    idx_to_tag = res['idx_to_tag']
    sents_te = res['sents_te']

    pred_sents, gold_sents = [], []
    for s in sents_te:
        with torch.no_grad():
            path = model.decode(s['X'])
        # Gold tags as strings — reverse the tag_to_idx lookup
        gold_str = [idx_to_tag[int(t)] for t in s['tags'].tolist()]
        pred_phon = decode_to_phoneme_sequence(path, idx_to_tag)
        gold_phon = gold_phoneme_sequence(gold_str)
        # phoneme sequences are already segment-level (collapse applied);
        # don't apply collapse_consecutive again, but guard against
        # accidental duplicates from I-X without B-X drift
        pred_sents.append(pred_phon)
        gold_sents.append(gold_phon)

    # Print sentence-level summary
    print(f"\n[{pid}] decoded {len(pred_sents)} test sentences")
    for i in range(min(5, len(pred_sents))):
        ps = pred_sents[i][:25]
        gs = gold_sents[i][:25]
        print(f"  sent {sents_te[i]['sent_idx']:3d}: ")
        print(f"    pred: {' '.join(ps)}")
        print(f"    gold: {' '.join(gs)}")

    # Score
    all_gold = [ph for s in gold_sents for ph in s]
    if not all_gold:
        print(f"[{pid}] empty gold"); return None
    cnt = Counter(all_gold); N = sum(cnt.values())
    gold_lp = {k: np.log(v / N) for k, v in cnt.items()}

    print()
    out = score_run(pred_sents, gold_sents, f"{pid} BIO-CRF", gold_lp)

    # Length stats
    pred_lens = [len(s) for s in pred_sents]
    gold_lens = [len(s) for s in gold_sents]
    print(f"  pred lens: mean={np.mean(pred_lens):.1f} min={min(pred_lens)} max={max(pred_lens)}")
    print(f"  gold lens: mean={np.mean(gold_lens):.1f} min={min(gold_lens)} max={max(gold_lens)}")
    return out


# ---- Score whatever is trained ----
for pid in TARGET_PIDS:
    if pid in bio_results and bio_results[pid] is not None:
        score_bio_patient(pid, bio_results)
