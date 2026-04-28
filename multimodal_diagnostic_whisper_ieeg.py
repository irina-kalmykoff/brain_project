# Multimodal diagnostic: Whisper acoustic features vs iEEG features.
#
# Purpose: determine whether decoding is information-limited (iEEG doesn't
# carry enough phoneme info) or model-limited (it does, but our classifier
# can't extract it).
#
# Three classifiers, same training procedure:
#   1. iEEG only             → current baseline (~3.5x lift)
#   2. Whisper only          → acoustic ceiling (expected very high)
#   3. iEEG + Whisper concat → diagnostic
#
# Usage:
#   - Run cells in order 1 → 5 to get the three numbers
#   - Cells 6-7 are extras for inspecting / saving results
#   - This file is NOT part of any deployment pipeline. It's a one-shot probe.

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 1 — Environment + imports
# ═══════════════════════════════════════════════════════════════════════════════

import sys, platform, importlib
print(f"python: {sys.version}")
print(f"platform: {platform.platform()}\n")

# ── 1. TORCH FIRST ────────────────────────────────────────────────────────────
import torch
print(f"torch: {torch.__version__}   cuda: {torch.cuda.is_available()}")
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ── 2. WHISPER ────────────────────────────────────────────────────────────────
import whisper
print(f"whisper: {whisper.__version__ if hasattr(whisper, '__version__') else 'installed'}")

# ── 3. STANDARD LIBRARIES ─────────────────────────────────────────────────────
import os
import gc
import copy
import pickle
from collections import Counter
from datetime import datetime

# ── 4. THIRD-PARTY ────────────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import resample_poly

# ── 5. PROJECT IMPORTS ────────────────────────────────────────────────────────
from config import DUTCH_30_PATH
from dataset_config import Dutch30Config
from dutch_30_feature_extractor import Dutch30FeatureExtractor
from dutch_30_pipeline import Dutch30Pipeline
from run_pipeline import (
    DEFAULT_RUN_CONFIG, run_path_b, run_from_config,
    _run_step5abc, _run_crf_experiment,
)


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 2 — Build (or reuse) pipeline through step 5
# ═══════════════════════════════════════════════════════════════════════════════
# This cell expects a pipeline up to step5_accumulate (2D features per phoneme).
# If you already have one in memory, skip the rebuild block.

REBUILD = False   # set True if pipeline isn't loaded yet

run_config = dict(DEFAULT_RUN_CONFIG)
run_config['use_viterbi']         = True
run_config['stacking_order']      = 7
run_config['stacking_step_size']  = 1
# run_config['patient_range']     = (21, 30)   # use whatever you had

if REBUILD:
    extractor = Dutch30FeatureExtractor()
    pipeline = Dutch30Pipeline(
        dutch30_extractor=extractor,
        debug_mode=False,
        feature_extraction_method=run_config['feature_extraction_method'],
        use_wav2vec=False,
        subtract_baseline=run_config['subtract_baseline'],
        use_rms_boundaries=False,
        use_multifeature=False,
    )
    cached_train, cached_test = run_path_b(pipeline, run_config)

# Sanity check: features should be 1D (after step5b/c stacked + collapsed)
f = pipeline.train['features'][0]
print(f"feature shape: {f.shape}   ndim: {f.ndim}")
print(f"n_train phonemes: {len(pipeline.train['features'])}")
print(f"n_test phonemes:  {len(pipeline.test['features'])}")


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 3 — Load Whisper encoder and define per-phoneme feature extraction
# ═══════════════════════════════════════════════════════════════════════════════
# Choose model size: "tiny" (fast, 39M params) → "large-v3" (slow, 1.5B params).
# For a diagnostic, "base" or "small" is plenty.

WHISPER_MODEL = "base"     # 74M params, fast on GPU
print(f"Loading Whisper '{WHISPER_MODEL}' on {DEVICE}…")
whisper_model = whisper.load_model(WHISPER_MODEL, device=DEVICE)
whisper_model.eval()
WHISPER_DIM = whisper_model.encoder.ln_post.normalized_shape[0]
print(f"  encoder hidden dim: {WHISPER_DIM}")

# Whisper encoder runs at 50 Hz on 16 kHz audio (10 ms hop, 25 ms window).
# Each encoder frame summarizes ~25 ms of audio.
WHISPER_FRAME_HZ = 50
WHISPER_SR       = 16000


def _resample_to_16k(audio, original_sr):
    """Resample mono audio to 16 kHz."""
    if original_sr == WHISPER_SR:
        return audio.astype(np.float32)
    g = np.gcd(int(original_sr), WHISPER_SR)
    return resample_poly(audio, WHISPER_SR // g, original_sr // g).astype(np.float32)


def whisper_encode_full_audio(audio, sr):
    """Run Whisper encoder once on a full clip; return (T_frames, hidden_dim)."""
    audio16 = _resample_to_16k(audio, sr)
    # Whisper expects exactly 30s; pad or chunk
    chunks = []
    chunk_samples = 30 * WHISPER_SR
    for s in range(0, len(audio16), chunk_samples):
        seg = audio16[s:s + chunk_samples]
        if len(seg) < chunk_samples:
            seg = np.pad(seg, (0, chunk_samples - len(seg)))
        mel = whisper.log_mel_spectrogram(torch.from_numpy(seg)).to(DEVICE)
        with torch.no_grad():
            emb = whisper_model.encoder(mel.unsqueeze(0))     # (1, 1500, D)
        chunks.append(emb.squeeze(0).cpu().numpy())            # (1500, D)
    return np.concatenate(chunks, axis=0)                     # (N_frames, D)


def whisper_feature_for_phoneme(encoder_emb, phoneme_start_s, phoneme_end_s):
    """Mean-pool Whisper frames within a phoneme's time window."""
    f_start = max(0, int(phoneme_start_s * WHISPER_FRAME_HZ))
    f_end   = min(encoder_emb.shape[0], int(phoneme_end_s * WHISPER_FRAME_HZ))
    if f_end <= f_start:
        f_end = f_start + 1
    return encoder_emb[f_start:f_end].mean(axis=0)


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 4 — Build per-phoneme Whisper features for each patient
# ═══════════════════════════════════════════════════════════════════════════════
# This iterates over phonemes in pipeline.train and pipeline.test, looks up the
# parent sentence's audio + phoneme timestamps from MFA TextGrids, and produces
# a Whisper feature vector per phoneme aligned with the iEEG features.
#
# Expects pipeline to have:
#   pipeline.train['phoneme_participant_ids']
#   pipeline.train['phoneme_words']
#   pipeline.split_result['word_segments_dict'][pid]['sentence_list']
#   And MFA TextGrid files in mfa_output/{pid}/{pid}_sent{NNN}.TextGrid
#
# Caches the per-patient encoder embeddings and per-phoneme features so reruns
# are fast.

WHISPER_CACHE_DIR = 'whisper_cache'
os.makedirs(WHISPER_CACHE_DIR, exist_ok=True)


def cache_path(pid):
    return os.path.join(WHISPER_CACHE_DIR, f'{pid}_whisper_{WHISPER_MODEL}_emb.pkl')


def build_whisper_features(pipeline):
    """Return dict {pid: {'train_features': [...], 'test_features': [...]}}.

    Each feature vector matches the corresponding entry in pipeline.train / .test.
    Uses MFA phoneme timestamps for alignment.
    """
    import tgt

    whisper_feats = {'train': [], 'test': []}

    pids = sorted(set(pipeline.train['phoneme_participant_ids']))
    for pid in pids:
        cf = cache_path(pid)
        if os.path.exists(cf):
            with open(cf, 'rb') as f:
                encoder_emb = pickle.load(f)
            print(f"  {pid}: encoder cache loaded ({encoder_emb.shape})")
        else:
            print(f"  {pid}: encoding audio with Whisper…")
            raw = pipeline.dutch30_extractor.load_patient_raw_data(pid)
            audio, sr_audio = raw['audio'], pipeline.config.audio_sr
            encoder_emb = whisper_encode_full_audio(audio, sr_audio)
            with open(cf, 'wb') as f:
                pickle.dump(encoder_emb, f)
            print(f"    saved → {cf} ({encoder_emb.shape})")

        # ── Compute per-phoneme Whisper features ─────────────────────────
        # We need the time of each phoneme in the original audio. The pipeline
        # stores phoneme_words but not absolute phoneme times. Pull from MFA.
        sent_list = pipeline.split_result['word_segments_dict'][pid]['sentence_list']
        # Build {(sent_idx) → list of (phoneme_text, abs_start, abs_end)}
        sentence_phones = _load_sentence_phone_times(pid, sent_list,
                                                     audio_sr=pipeline.config.audio_sr,
                                                     eeg_sr=pipeline.config.eeg_sr)
        # Match each phoneme in pipeline.train/test to a TextGrid phone
        # (this is the trickiest part — see helper below)
        for ds_name in ['train', 'test']:
            data = getattr(pipeline, ds_name)
            this_pid_idx = [i for i, p in enumerate(data['phoneme_participant_ids'])
                            if p == pid]
            for i in this_pid_idx:
                sent_idx = data['phoneme_sentence_indices'][i] if 'phoneme_sentence_indices' in data else None
                ph_text  = data['phoneme_labels'][i]
                ph_pos   = data['phoneme_positions'][i] if 'phoneme_positions' in data else 0
                start, end = _lookup_phoneme_time(sentence_phones, sent_idx,
                                                  ph_text, ph_pos)
                if start is None:
                    feat = np.zeros(WHISPER_DIM, dtype=np.float32)
                else:
                    feat = whisper_feature_for_phoneme(encoder_emb, start, end)
                whisper_feats[ds_name].append((i, feat))

    # Sort by index and unpack
    for ds_name in ['train', 'test']:
        whisper_feats[ds_name].sort(key=lambda x: x[0])
        whisper_feats[ds_name] = [x[1] for x in whisper_feats[ds_name]]

    return whisper_feats


def _load_sentence_phone_times(pid, sent_list, audio_sr, eeg_sr,
                               mfa_dir='mfa_output'):
    """Load MFA TextGrids per sentence; return {sent_idx: [(text, start, end)]}.

    Times are absolute seconds in the original audio."""
    import tgt
    out = {}
    for sent_idx, sent in enumerate(sent_list):
        if not isinstance(sent, dict):
            continue
        # sentence offset in audio (s) = stim_start_idx / eeg_sr
        sent_offset_s = sent['stim_start_idx'] / eeg_sr
        tg_path = os.path.join(mfa_dir, pid, f'{pid}_sent{sent_idx:03d}.TextGrid')
        if not os.path.exists(tg_path):
            continue
        try:
            tg = tgt.io.read_textgrid(tg_path)
            phone_tier = tg.get_tier_by_name('phones')
        except Exception:
            continue
        phones = []
        for ann in phone_tier.annotations:
            text = ann.text.strip()
            if text and text not in ('sp', 'sil', 'spn'):
                phones.append((text,
                               sent_offset_s + ann.start_time,
                               sent_offset_s + ann.end_time))
        out[sent_idx] = phones
    return out


def _lookup_phoneme_time(sentence_phones, sent_idx, ph_text, ph_pos):
    """Find a phoneme by sentence + position; fall back to text match."""
    if sent_idx is None or sent_idx not in sentence_phones:
        return None, None
    phones = sentence_phones[sent_idx]
    # First try by position (most reliable)
    if 0 <= ph_pos < len(phones):
        text, start, end = phones[ph_pos]
        if text == ph_text:
            return start, end
    # Fallback: first phone with matching text
    for text, start, end in phones:
        if text == ph_text:
            return start, end
    return None, None


# Run it (slow first time, fast on cache hit)
print("Building Whisper features per phoneme…")
whisper_feats = build_whisper_features(pipeline)
print(f"  train: {len(whisper_feats['train'])} features (expected {len(pipeline.train['features'])})")
print(f"  test:  {len(whisper_feats['test'])} features (expected {len(pipeline.test['features'])})")


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 5 — Run three CRF experiments: iEEG-only, Whisper-only, concat
# ═══════════════════════════════════════════════════════════════════════════════
# We reuse pipeline.train/test as the data source. To swap features, we just
# replace the 'features' list temporarily and call _run_crf_experiment.

def run_with_features(pipeline, features_train, features_test, label):
    """Replace features, run CRF, return per-patient results dict."""
    backup_train = pipeline.train['features']
    backup_test  = pipeline.test['features']
    try:
        pipeline.train['features'] = features_train
        pipeline.test['features']  = features_test
        results = _run_crf_experiment(pipeline, run_config)
    finally:
        pipeline.train['features'] = backup_train
        pipeline.test['features']  = backup_test
    return results


def summarize(results, label):
    accs, lifts = [], []
    for pid, r in results.items():
        n_cl = len(set(r['true_labels']))
        chance = 1.0 / n_cl if n_cl > 0 else 0
        lift = r['accuracy'] / chance if chance > 0 else 0
        accs.append(r['accuracy']); lifts.append(lift)
    print(f"\n  {label}: mean acc = {np.mean(accs):.3f}   lift = {np.mean(lifts):.2f}x")
    return accs, lifts


# ── Experiment 1: iEEG only (your baseline) ───────────────────────────────────
print("\n" + "═"*60)
print(" EXP 1: iEEG only (baseline)")
print("═"*60)
results_ieeg = run_with_features(pipeline,
                                  pipeline.train['features'],
                                  pipeline.test['features'],
                                  'iEEG')
accs_ieeg, lifts_ieeg = summarize(results_ieeg, 'iEEG only')


# ── Experiment 2: Whisper only ────────────────────────────────────────────────
print("\n" + "═"*60)
print(" EXP 2: Whisper only (acoustic ceiling)")
print("═"*60)
results_wsp = run_with_features(pipeline,
                                 whisper_feats['train'],
                                 whisper_feats['test'],
                                 'Whisper')
accs_wsp, lifts_wsp = summarize(results_wsp, 'Whisper only')


# ── Experiment 3: iEEG + Whisper concatenated ─────────────────────────────────
print("\n" + "═"*60)
print(" EXP 3: iEEG + Whisper concat (multimodal)")
print("═"*60)
concat_train = [np.concatenate([a, b])
                for a, b in zip(pipeline.train['features'], whisper_feats['train'])]
concat_test  = [np.concatenate([a, b])
                for a, b in zip(pipeline.test['features'], whisper_feats['test'])]
results_both = run_with_features(pipeline, concat_train, concat_test, 'concat')
accs_both, lifts_both = summarize(results_both, 'iEEG + Whisper')


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 6 — Side-by-side comparison + interpretation
# ═══════════════════════════════════════════════════════════════════════════════

pids = sorted(results_ieeg.keys())
print(f"\n  {'pid':<6} {'iEEG':>8} {'Whisper':>10} {'concat':>10}   {'Δ vs Whisper':>14}")
print("  " + "-" * 56)
for pid in pids:
    a_i = results_ieeg[pid]['accuracy']
    a_w = results_wsp[pid]['accuracy']
    a_b = results_both[pid]['accuracy']
    delta = a_b - a_w
    print(f"  {pid:<6} {a_i:>7.1%} {a_w:>9.1%} {a_b:>9.1%}   {delta:>+13.1%}")

print("  " + "-" * 56)
print(f"  {'mean':<6} {np.mean(accs_ieeg):>7.1%} {np.mean(accs_wsp):>9.1%} "
      f"{np.mean(accs_both):>9.1%}   {np.mean(accs_both) - np.mean(accs_wsp):>+13.1%}")

print("\nInterpretation cheat sheet:")
print("  Δ near zero  → iEEG didn't add anything Whisper didn't already have")
print("                 → iEEG is information-limited for this task")
print("  Δ > +5%      → iEEG carries unique signal beyond audio")
print("                 → invest in better iEEG models (BiLSTM, cross-patient)")
print("  Δ < 0        → adding iEEG hurt; classifier overfits to noisy features")


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 7 — Optional: bar chart + save results
# ═══════════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(pids))
width = 0.27
ax.bar(x - width, [results_ieeg[p]['accuracy'] for p in pids], width,
       color='steelblue', label='iEEG only')
ax.bar(x,         [results_wsp[p]['accuracy'] for p in pids], width,
       color='darkorange', label='Whisper only')
ax.bar(x + width, [results_both[p]['accuracy'] for p in pids], width,
       color='seagreen', label='iEEG + Whisper')
ax.set_xticks(x); ax.set_xticklabels(pids, fontsize=10)
ax.set_ylabel('Accuracy'); ax.set_title('Multimodal diagnostic — per-patient accuracy',
                                        fontsize=12, fontweight='bold')
ax.legend(); ax.grid(alpha=0.3, axis='y')
plt.tight_layout(); plt.show()

# Save raw results for later analysis
out = {
    'whisper_model':   WHISPER_MODEL,
    'run_config':      run_config,
    'results_ieeg':    {pid: {'accuracy': r['accuracy']} for pid, r in results_ieeg.items()},
    'results_whisper': {pid: {'accuracy': r['accuracy']} for pid, r in results_wsp.items()},
    'results_concat':  {pid: {'accuracy': r['accuracy']} for pid, r in results_both.items()},
    'timestamp':       datetime.now().isoformat(),
}
out_path = f'multimodal_diagnostic_{WHISPER_MODEL}_{datetime.now().strftime("%Y%m%d_%H%M")}.pkl'
with open(out_path, 'wb') as f:
    pickle.dump(out, f)
print(f"\nSaved results → {out_path}")
