# Whisper-enhanced v6 experiment.
#
# Goal: train a v6 variant whose word-onset head is supervised by
# WhisperX-derived word boundaries (instead of/alongside MFA's word
# boundaries). Whisper is independent of MFA — it sees raw audio and a
# transcript, and aligns words via its own acoustic model. This catches
# sentences MFA failed on (e.g., compound words it didn't have in the
# lexicon) and provides a second supervision signal.
#
# Brain-only at test time. Whisper is a TRAINING-LABEL source, never an
# input to the deployed model.
#
# How to use: open in VSCode (cells visible) or convert with `jupytext --to
# ipynb whisper_enhance_v6.py` and step through the cells.
#
# Prerequisites:
#   - You've already done run_with_mfa_boundaries(pipeline, run_config).
#   - WhisperX installed (`pip install whisperx`). It's in your env per
#     borders.py. CUDA recommended.
#   - Raw audio files at DUTCH_30_PATH/raw/<pid>_audio.npy.

# %% Cell 1 — Imports
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


# %% Cell 2 — Helper: load WhisperX models once (slow, ~1GB)
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


# %% Cell 3 — Helper: align one sentence's audio with its transcript
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


# %% Cell 4 — Cache Whisper alignments per patient (~30 min for 10 patients)
# Run this once. Output: <pid>.json files in WHISPER_CACHE_DIR.
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


# %% Cell 5 — Run cache step (call once per session, then it's on disk)
# cache_whisper_alignments(pipeline, [f'P{i:02d}' for i in range(21, 31)])


# %% Cell 6 — Loader: read cached Whisper alignments
def load_whisper_alignments(pid):
    """Returns dict sent_idx (int) → {'words': [...], 'duration_s': ...}"""
    path = WHISPER_CACHE_DIR / f"{pid}.json"
    if not path.exists():
        return {}
    with path.open(encoding='utf-8') as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


# %% Cell 7 — Compare Whisper vs MFA word boundaries (sanity check)
# For sentences where both produced a result, see how often they agree on
# word onset times. Disagreement is informative — Whisper's onsets are
# acoustic; MFA's are forced-alignment from a lexicon.
def compare_whisper_vs_mfa_word_onsets(pid, tolerance_ms=50):
    """Return per-sentence stats: how many word onsets agree within ±tolerance."""
    whisper = load_whisper_alignments(pid)
    mfa = load_mfa_alignments(pid)

    rows = []
    for sent_idx, w_align in whisper.items():
        if sent_idx not in mfa or not mfa[sent_idx]:
            continue
        # MFA word onsets (start of each phoneme that begins a new word)
        mfa_word_onsets = []
        prev_word = None
        for ph in mfa[sent_idx]:
            wd = ph.get('word', '') or ''
            if wd != prev_word:
                mfa_word_onsets.append(ph['start_s'])
                prev_word = wd

        whisper_onsets = [w['start'] for w in w_align['words']]

        if not mfa_word_onsets or not whisper_onsets:
            continue

        # Greedy match: for each MFA onset, find closest Whisper onset
        tol_s = tolerance_ms / 1000.0
        n_match = sum(1 for t in mfa_word_onsets
                       if min(abs(t - w) for w in whisper_onsets) < tol_s)
        rows.append({
            'sent_idx':   sent_idx,
            'n_mfa':      len(mfa_word_onsets),
            'n_whisper':  len(whisper_onsets),
            'n_agreed':   n_match,
        })

    if not rows:
        return None

    n_mfa_total = sum(r['n_mfa'] for r in rows)
    n_agreed_total = sum(r['n_agreed'] for r in rows)
    print(f"  {pid}: {len(rows)} sentences compared, "
          f"{n_agreed_total}/{n_mfa_total} word onsets agree within ±{tolerance_ms}ms "
          f"({n_agreed_total/n_mfa_total:.1%})")
    return rows


# %% Cell 8 — Run the comparison for all patients
# for pid in [f'P{i:02d}' for i in range(21, 31)]:
#     compare_whisper_vs_mfa_word_onsets(pid)


# %% Cell 9 — Whisper-supervised word-onset labels for v6 training
# This produces per-sentence frame-level labels at 200 Hz, with Gaussian
# bumps centered on each Whisper word onset. Used as supervision for v6's
# word_head instead of (or alongside) MFA-derived word labels.
from boundary_detector_joint_audio import FRAME_HZ

def whisper_word_labels_from_alignment(whisper_align, n_frames,
                                         frame_hz=FRAME_HZ, sigma_ms=10.0):
    """Soft-Gaussian labels at 200 Hz, peaked at Whisper word onsets."""
    sigma_frames = sigma_ms * frame_hz / 1000.0
    half_window = max(1, int(np.ceil(3 * sigma_frames)))
    labels = np.zeros(n_frames, dtype=np.float32)

    for w in whisper_align['words']:
        c = int(round(w['start'] * frame_hz))
        for off in range(-half_window, half_window + 1):
            f = c + off
            if 0 <= f < n_frames:
                weight = float(np.exp(-(off ** 2) / (2 * sigma_frames ** 2)))
                if weight > labels[f]:
                    labels[f] = weight
    return labels


# %% Cell 10 — Build a Whisper-augmented training dataset
# Re-run build_joint_dataset_fixed but inject Whisper-derived word labels
# alongside the MFA-derived ones. The model will see BOTH supervisions.
import importlib
import e2e_brain_decoder; importlib.reload(e2e_brain_decoder)
from e2e_brain_decoder import build_joint_dataset_fixed

def build_dataset_with_whisper(pipeline, patient_ids):
    """Like build_joint_dataset_fixed but adds 'whisper_word_labels' field
    to each item (frame-level Gaussian labels from Whisper word onsets)."""
    dataset = build_joint_dataset_fixed(pipeline, patient_ids)
    n_with = 0
    n_without = 0
    for item in dataset:
        whisper = load_whisper_alignments(item['pid']).get(item['sentence_idx'])
        if whisper is None:
            item['whisper_word_labels'] = np.zeros(item['eeg'].shape[0],
                                                     dtype=np.float32)
            n_without += 1
        else:
            item['whisper_word_labels'] = whisper_word_labels_from_alignment(
                whisper, item['eeg'].shape[0])
            n_with += 1
    print(f"  Built {len(dataset)} items: {n_with} with Whisper labels, "
          f"{n_without} fell back to zero (no Whisper alignment cached)")
    return dataset


# %% Cell 11 — v6 variant: 4-headed model (boundary + count + word-MFA + word-Whisper)
import torch.nn as nn
import torch.nn.functional as F
from boundary_detector_joint_audio import (
    HIDDEN_DIM, N_LSTM_LAYERS, DROPOUT,
)

class WhisperWordV6(nn.Module):
    """v6 with an additional word-onset head supervised by Whisper.
    Otherwise identical architecture to CountWordAwareDetector."""
    def __init__(self, per_patient_eeg_n_ch, mfcc_dim,
                 hidden_dim=HIDDEN_DIM, n_layers=N_LSTM_LAYERS,
                 dropout=DROPOUT, proj_dim=64):
        super().__init__()
        self.eeg_proj = nn.ModuleDict({
            pid: nn.Linear(n_ch, proj_dim)
            for pid, n_ch in per_patient_eeg_n_ch.items()
        })
        self.mfcc_proj = nn.Linear(mfcc_dim, proj_dim)
        self.lstm = nn.LSTM(
            input_size=proj_dim, hidden_size=hidden_dim,
            num_layers=n_layers, batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        # Heads — same shapes as v6 plus one new
        self.boundary_head    = self._make_head(2 * hidden_dim, hidden_dim, 1, dropout)
        self.word_head        = self._make_head(2 * hidden_dim, hidden_dim, 1, dropout)
        self.whisper_word_head = self._make_head(2 * hidden_dim, hidden_dim, 1, dropout)
        self.count_head       = self._make_head(2 * hidden_dim, hidden_dim, 1, dropout)

    def _make_head(self, in_dim, mid_dim, out_dim, dropout):
        return nn.Sequential(
            nn.Linear(in_dim, mid_dim),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(mid_dim, out_dim),
        )

    def encode(self, eeg, mfcc, pid):
        h_eeg  = self.eeg_proj[pid](eeg)  if eeg  is not None else None
        h_mfcc = self.mfcc_proj(mfcc)     if mfcc is not None else None
        if   h_eeg is not None and h_mfcc is not None:
            h = (h_eeg + h_mfcc) / 2.0
        elif h_eeg is not None: h = h_eeg
        else:                    h = h_mfcc
        h, _ = self.lstm(h)
        return h

    def forward(self, eeg=None, mfcc=None, pid=None, mask=None):
        h = self.encode(eeg, mfcc, pid)
        b_logits = self.boundary_head(h).squeeze(-1)
        w_logits = self.word_head(h).squeeze(-1)
        ww_logits = self.whisper_word_head(h).squeeze(-1)
        if mask is None:
            pooled = h.mean(dim=1)
        else:
            m = mask.float().unsqueeze(-1)
            pooled = (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        count_pred = self.count_head(pooled).squeeze(-1)
        return b_logits, w_logits, ww_logits, count_pred


# %% Cell 12 — Training loop for the 4-headed model
# Skeleton — plug in the same DataLoader/batching from train_count_word_aware
# in e2e_brain_decoder, just add the whisper_word_loss to the total.
WHISPER_WORD_LOSS_WEIGHT = 0.1   # same weight class as the MFA word-onset loss

def train_whisper_v6(train_dataset, n_epochs=40, lr=5e-4, ckpt_prefix='v6_whisper_'):
    """Skeleton trainer. Borrows from `train_count_word_aware`. Each step:
       loss = bce_boundary + 0.5*bce_word_mfa + 0.1*bce_word_whisper +
              0.005*mse_count

    The Whisper-word head is supervised by item['whisper_word_labels'].
    Test-time inference uses ONLY the boundary + count heads (same as v6).
    """
    raise NotImplementedError(
        "Wire this up using train_count_word_aware as a template — copy its "
        "loop, add `ww_logits` and `bce(ww_logits, item['whisper_word_labels'])` "
        "to the loss. Keep boundary head as the primary objective.")


# %% Cell 13 — Quick sanity: load + plot one sentence's Whisper alignment
# import matplotlib.pyplot as plt
# pid = 'P21'
# sent_idx = 12
# whisper = load_whisper_alignments(pid).get(sent_idx)
# mfa     = load_mfa_alignments(pid).get(sent_idx, [])
# if whisper:
#     fig, ax = plt.subplots(1, 1, figsize=(12, 2))
#     for w in whisper['words']:
#         ax.axvspan(w['start'], w['end'], alpha=0.4, color='steelblue')
#         ax.text((w['start'] + w['end']) / 2, 0.7, w['word'],
#                 ha='center', va='center', fontsize=10)
#     for ph in mfa:
#         ax.axvline(ph['start_s'], color='red', alpha=0.3, linewidth=0.5)
#     ax.set_xlim(0, whisper['duration_s'])
#     ax.set_xlabel('Time (s)')
#     ax.set_title(f"{pid} sent {sent_idx}: Whisper words (blue) vs MFA phoneme onsets (red)")
#     plt.tight_layout(); plt.show()


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
