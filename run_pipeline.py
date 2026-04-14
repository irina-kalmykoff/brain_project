#!/usr/bin/env python
"""
Brain-to-speech decoding pipeline — single-script runner.

Supports two phoneme segmentation paths:
  Path A: wav2vec / WhisperX acoustic boundary detection (step4 + step5_accumulate)
  Path B: Montreal Forced Aligner (MFA) pre-aligned TextGrids

Usage:
    python run_pipeline.py                     # Path A (wav2vec)
    python run_pipeline.py --mfa               # Path B (MFA)
    python run_pipeline.py --mfa --sweep       # Path B + hyperparameter sweep
    python run_pipeline.py --export-mfa        # Export audio for MFA alignment
    python run_pipeline.py --diagnose-mfa      # Show MFA phoneme loss breakdown

Requires: conda environment with torch, transformers, numpy, scipy, sklearn, tgt, etc.
See CLAUDE.md for full dependency list.
"""

# ═══════════════════════════════════════════════════════════════════════════════
#  IMPORTS
# ═══════════════════════════════════════════════════════════════════════════════

import argparse
import copy
import gc
import glob
import json
import os
import pickle
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from itertools import combinations, groupby

import numpy as np
import pandas as pd
import scipy.signal
import matplotlib
import matplotlib.pyplot as plt
from scipy.signal import decimate, resample_poly

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ── Project imports ───────────────────────────────────────────────────────────
from extract_features import extractHG, stackFeatures
from acoustic_change_detector import AcousticChangeDetector
from phoneme_validator import PhonemeValidator
from phonetic_dictionary import PhoneticDictionary
from markov_phoneme_model import MarkovPhonemeModel
from config import (
    BIDS_PATH, OUTPUT_PATH, RESULTS_PATH,
    DUTCH_30_PATH, DUTCH_10_PATH, get_dataset_paths,
    MFA_OUTPUT_PATH,
)
from dutch_30_pipeline import Dutch30Pipeline
from dutch_30_feature_extractor import Dutch30FeatureExtractor
from dataset_config import Dutch30Config
from experiment_logger import ExperimentLogger


# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_RUN_CONFIG = {
    # Patient selection
    'patient_range':             (21, 30),
    'sample_fraction':           1,
    # Pipeline settings
    'feature_extraction_method': 'high_gamma',
    'use_wav2vec':               False,
    'use_whisperx':              True,
    'subtract_baseline':         False,
    # Step 5a: frame filtering
    'min_frames':                0,
    'max_frames':                300,
    # Step 5b: stacking (set stacking_order=None to skip)
    'stacking_order':            7,
    'stacking_step_size':        2,
    'target_frames':             None,   # alternative: resampling
    # Classifier
    'classifier_type':           'logistic_regression',
    'class_weight':              'balanced',
    'markov_order':              1,
    'use_viterbi':               True,
    'random_state':              37,
    'scaler_type':               'standard',
    'feature_pooling_method':    'flatten',
    'min_class_samples':         0,
    # Unknown filtering
    'unknown_keep_ratio':        0.0025,
}


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def count(pipeline, label=""):
    """Print train/test sample counts."""
    tr = len(pipeline.train['features']) if pipeline.train else 0
    te = len(pipeline.test['features'])  if pipeline.test  else 0
    print(f"  {label:.<40s} train={tr:>6d}  test={te:>6d}")


def make_checkpoint_names(run_config):
    """Generate pickle checkpoint filenames from run_config."""
    pr = run_config['patient_range']
    so = run_config['stacking_order']
    ss = run_config['stacking_step_size']
    tf = run_config['target_frames']

    use_stacking   = so is not None
    use_resampling = tf is not None

    step3_ckpt = f'checkpoint_after_step3_P{pr[0]:02d}-P{pr[1]:02d}.pkl'
    stk_tag    = (f'stk{so}_s{ss}' if use_stacking
                  else (f'norm{tf}' if use_resampling else 'raw'))
    frame_cache = f'cache_frames_P{pr[0]:02d}-P{pr[1]:02d}.pkl'
    step5_cache = f'cache_step5_P{pr[0]:02d}-P{pr[1]:02d}_{stk_tag}.pkl'

    return step3_ckpt, frame_cache, step5_cache


# ═══════════════════════════════════════════════════════════════════════════════
#  MFA: TEXT CLEANING & AUDIO EXPORT
# ═══════════════════════════════════════════════════════════════════════════════

def clean_text_for_mfa(text):
    """Strip punctuation and quotes that MFA cannot handle."""
    text = text.lower().strip()
    text = re.sub(r'["""\'\u201c\u201d\u2018\u2019.,!?;:()\[\]{}<>]', ' ', text)
    text = re.sub(r'\b\d+\b', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def export_sentences_for_mfa(pid, pipeline, out_dir, eeg_sr=1024, audio_sr=48000):
    """Export per-sentence .wav + .lab files for MFA alignment.

    Creates: out_dir/{pid}/{pid}_sent{NNN}.wav and .lab
    Filenames use the original sentence_list index (even indices for real
    sentences, odd indices are rest intervals and are skipped).
    """
    import soundfile as sf

    raw_audio = np.load(os.path.join(DUTCH_30_PATH, 'raw', f'{pid}_audio.npy'))
    word_data = pipeline.split_result['word_segments_dict'][pid]
    pid_dir   = os.path.join(out_dir, pid)
    os.makedirs(pid_dir, exist_ok=True)

    exported = skipped = rest_skip = 0

    for i, sent in enumerate(word_data['sentence_list']):
        text      = sent['text'] if isinstance(sent, dict) else sent
        start_idx = sent['stim_start_idx']
        end_idx   = sent['stim_end_idx']

        cleaned_text = clean_text_for_mfa(text)
        if not cleaned_text:
            rest_skip += 1
            continue

        start_audio = int(start_idx * audio_sr / eeg_sr)
        end_audio   = int(end_idx   * audio_sr / eeg_sr)
        audio_slice = raw_audio[start_audio:end_audio].astype(np.float32)

        if len(audio_slice) < audio_sr * 0.1:
            skipped += 1
            print(f"  WARNING {pid} sent{i:03d}: audio too short "
                  f"({len(audio_slice)/audio_sr*1000:.0f}ms), skipping")
            continue

        peak = np.max(np.abs(audio_slice))
        if peak > 0:
            audio_slice /= peak

        audio_16k = resample_poly(audio_slice, 16000, audio_sr).astype(np.float32)
        sf.write(os.path.join(pid_dir, f'{pid}_sent{i:03d}.wav'), audio_16k, 16000)
        with open(os.path.join(pid_dir, f'{pid}_sent{i:03d}.lab'), 'w',
                  encoding='utf-8') as f:
            f.write(cleaned_text)
        exported += 1

    print(f'{pid}: exported={exported}  rest_intervals={rest_skip}  '
          f'bad_audio={skipped}')


# ═══════════════════════════════════════════════════════════════════════════════
#  MFA: LOAD TEXTGRID ALIGNMENTS
# ═══════════════════════════════════════════════════════════════════════════════

def load_mfa_alignments(pid, mfa_output_dir=None):
    """Load all MFA TextGrids for a patient.

    Returns:
        dict: sentence_list_index -> list of {phone, start_s, end_s, word}
    """
    import tgt

    if mfa_output_dir is None:
        mfa_output_dir = MFA_OUTPUT_PATH

    pid_dir = os.path.join(mfa_output_dir, pid)
    if not os.path.isdir(pid_dir):
        return {}

    alignments = {}
    for tg_file in sorted(os.listdir(pid_dir)):
        if not tg_file.endswith('.TextGrid'):
            continue
        try:
            sent_idx = int(tg_file.split('sent')[1].split('.')[0])
        except (IndexError, ValueError):
            continue

        tg = tgt.io.read_textgrid(os.path.join(pid_dir, tg_file))
        try:
            phone_tier = tg.get_tier_by_name('phones')
            word_tier  = tg.get_tier_by_name('words')
        except Exception:
            continue

        phones = []
        for ann in phone_tier.annotations:
            ph = ann.text.strip()
            if ph in ('', 'sp', 'sil', 'spn'):
                continue

            mid = (ann.start_time + ann.end_time) / 2
            word = ''
            for w_ann in word_tier.annotations:
                if w_ann.start_time <= mid <= w_ann.end_time:
                    word = w_ann.text.strip()
                    break

            phones.append({
                'phone':   ph,
                'start_s': ann.start_time,
                'end_s':   ann.end_time,
                'word':    word,
            })

        if phones:
            alignments[sent_idx] = phones

    return alignments


# ═══════════════════════════════════════════════════════════════════════════════
#  MFA: BUILD FEATURES (Path B — replaces step4 + step5_accumulate)
# ═══════════════════════════════════════════════════════════════════════════════

def build_mfa_features(pipeline, run_config):
    """Build phoneme-level HG features using MFA-aligned segments.

    For each patient:
      1. Load MFA TextGrids (precise per-phoneme timestamps).
      2. Slice raw EEG per phoneme, extract high-gamma envelope.
      3. Zero-pad short phonemes (< min_samples) instead of dropping.
      4. Split into train / test using pipeline.split_result word lookup.

    Returns:
        (train_dict, test_dict) ready for step5a/b/c.
    """
    config     = pipeline.config
    eeg_sr     = config.eeg_sr
    win_len    = config.window_length
    frameshift = config.frameshift
    mfa_output = MFA_OUTPUT_PATH
    min_samples = int(win_len * eeg_sr) + 1     # 31 samples ~ 30 ms

    accum = {
        split: {k: [] for k in (
            'features', 'phoneme_labels', 'phoneme_words',
            'phoneme_positions', 'phoneme_participant_ids',
            'phoneme_instance_ids', 'phoneme_durations_samples',
            'phone_sequences',
        )}
        for split in ('train', 'test')
    }

    split_result       = pipeline.split_result
    word_segments_dict = split_result['word_segments_dict']
    pr       = run_config['patient_range']
    patients = [f'P{i:02d}' for i in range(pr[0], pr[1] + 1)]

    global_iid   = 0
    total_padded = 0

    for pid in patients:
        if pid not in word_segments_dict:
            print(f"  {pid}: not in word_segments_dict, skipping")
            continue

        word_data = word_segments_dict[pid]

        raw_eeg_path = os.path.join(DUTCH_30_PATH, 'raw', f'{pid}_sEEG.npy')
        if not os.path.exists(raw_eeg_path):
            print(f"  {pid}: no sEEG file, skipping")
            continue
        raw_eeg = np.load(raw_eeg_path)

        # Apply channel exclusions
        if hasattr(pipeline, 'patient_data') and pid in pipeline.patient_data:
            pdata = pipeline.patient_data[pid]
            if 'channel_mask' in pdata:
                raw_eeg = raw_eeg[:, pdata['channel_mask']]
            elif 'included_channels' in pdata:
                raw_eeg = raw_eeg[:, pdata['included_channels']]

        alignments = load_mfa_alignments(pid, mfa_output)

        # Build sentence-level train/test lookup.
        # split_result stores {word: [instance_indices]}.  Each word instance
        # has a sentence_idx.  We collect which sentence indices are train
        # vs test so that ALL phonemes from a sentence go to the same split.
        train_sent_indices = set()
        test_sent_indices  = set()
        words_dict = word_data['words']
        for word_text_key, inst_indices in split_result.get('train', {}).get(pid, {}).items():
            if word_text_key in words_dict:
                for inst_idx in inst_indices:
                    instances = words_dict[word_text_key]['instances']
                    if inst_idx < len(instances):
                        train_sent_indices.add(instances[inst_idx]['sentence_idx'])
        for word_text_key, inst_indices in split_result.get('test', {}).get(pid, {}).items():
            if word_text_key in words_dict:
                for inst_idx in inst_indices:
                    instances = words_dict[word_text_key]['instances']
                    if inst_idx < len(instances):
                        test_sent_indices.add(instances[inst_idx]['sentence_idx'])

        sentence_list = word_data['sentence_list']
        pid_count  = 0
        pid_padded = 0

        for sent_list_idx, sent in enumerate(sentence_list):
            text = sent['text'] if isinstance(sent, dict) else sent
            if not text:
                continue

            start_eeg = sent['stim_start_idx']
            end_eeg   = sent['stim_end_idx']
            sentence_eeg = raw_eeg[start_eeg:end_eeg]

            if sent_list_idx not in alignments:
                continue

            # Determine train/test for this entire sentence
            if sent_list_idx in test_sent_indices:
                sent_split = 'test'
            elif sent_list_idx in train_sent_indices:
                sent_split = 'train'
            else:
                sent_split = 'train'    # fallback for unmatched sentences

            # Collect the full phone sequence for this sentence
            # (used to build Viterbi transition model from MFA phone set)
            sent_phones = [ph['phone'] for ph in alignments[sent_list_idx]]
            if sent_phones:
                accum[sent_split]['phone_sequences'].append(sent_phones)

            for ph in alignments[sent_list_idx]:
                phone_label = ph['phone']
                word_text   = ph['word'].lower() if ph['word'] else '?'

                ph_start = int(ph['start_s'] * eeg_sr)
                ph_end   = int(ph['end_s']   * eeg_sr)
                ph_start = max(0, min(ph_start, sentence_eeg.shape[0] - 1))
                ph_end   = max(ph_start + 1, min(ph_end, sentence_eeg.shape[0]))

                eeg_seg = sentence_eeg[ph_start:ph_end]
                n_samp  = eeg_seg.shape[0]

                # Zero-pad short phonemes instead of dropping
                if n_samp < min_samples:
                    eeg_seg = np.pad(eeg_seg,
                                     ((0, min_samples - n_samp), (0, 0)),
                                     mode='constant')
                    pid_padded += 1

                try:
                    feat = extractHG(eeg_seg, eeg_sr,
                                     windowLength=win_len,
                                     frameshift=frameshift)
                except Exception:
                    continue
                if feat is None or feat.shape[0] == 0:
                    continue

                split = sent_split

                accum[split]['features'].append(feat)
                accum[split]['phoneme_labels'].append(phone_label)
                accum[split]['phoneme_words'].append(word_text)
                accum[split]['phoneme_positions'].append(0)
                accum[split]['phoneme_participant_ids'].append(pid)
                accum[split]['phoneme_instance_ids'].append(global_iid)
                accum[split]['phoneme_durations_samples'].append(n_samp)

                global_iid += 1
                pid_count  += 1

        n_sent = sum(1 for s in sentence_list
                     if (s['text'] if isinstance(s, dict) else s))
        print(f"  {pid}: {pid_count} phonemes  "
              f"({len(alignments)}/{n_sent} sentences, "
              f"{pid_padded} zero-padded)")
        total_padded += pid_padded

    for split in ('train', 'test'):
        n  = len(accum[split]['features'])
        nu = len(set(accum[split]['phoneme_labels']))
        print(f"  {split}: {n} phoneme samples, {nu} unique phonemes")
    print(f"  Total zero-padded: {total_padded}")

    return accum['train'], accum['test']


# ═══════════════════════════════════════════════════════════════════════════════
#  MFA: DIAGNOSTIC — where are phonemes lost?
# ═══════════════════════════════════════════════════════════════════════════════

def diagnose_mfa_loss(pipeline, run_config):
    """Print per-patient breakdown of MFA phoneme loss."""
    config      = pipeline.config
    eeg_sr      = config.eeg_sr
    win_len     = config.window_length
    min_samples = int(win_len * eeg_sr) + 1
    mfa_output  = MFA_OUTPUT_PATH

    pr       = run_config['patient_range']
    patients = [f'P{i:02d}' for i in range(pr[0], pr[1] + 1)]

    print(f"min_samples threshold = {min_samples} ({min_samples/eeg_sr*1000:.1f}ms)")
    print(f"{'PID':<6} {'TextGrid':>10} {'No TG':>8} {'TooShort':>10} "
          f"{'Kept':>8} {'Loss%':>7}")
    print("-" * 55)

    total_tg = total_short = total_kept = 0

    for pid in patients:
        word_data = pipeline.split_result['word_segments_dict'].get(pid)
        if word_data is None:
            continue

        raw_eeg = np.load(os.path.join(DUTCH_30_PATH, 'raw', f'{pid}_sEEG.npy'))
        if hasattr(pipeline, 'patient_data') and pid in pipeline.patient_data:
            pdata = pipeline.patient_data[pid]
            if 'channel_mask' in pdata:
                raw_eeg = raw_eeg[:, pdata['channel_mask']]
            elif 'included_channels' in pdata:
                raw_eeg = raw_eeg[:, pdata['included_channels']]

        alignments    = load_mfa_alignments(pid, mfa_output)
        sentence_list = word_data['sentence_list']

        pid_too_short = pid_kept = 0

        for sent_list_idx, sent in enumerate(sentence_list):
            text = sent['text'] if isinstance(sent, dict) else sent
            if not text:
                continue
            sent_len = sent['stim_end_idx'] - sent['stim_start_idx']

            if sent_list_idx in alignments:
                for ph in alignments[sent_list_idx]:
                    ph_start = int(ph['start_s'] * eeg_sr)
                    ph_end   = int(ph['end_s']   * eeg_sr)
                    ph_start = max(0, min(ph_start, sent_len - 1))
                    ph_end   = max(ph_start + 1, min(ph_end, sent_len))
                    if (ph_end - ph_start) < min_samples:
                        pid_too_short += 1
                    else:
                        pid_kept += 1

        raw_count     = sum(len(ph) for ph in alignments.values())
        n_sent_text   = sum(1 for s in sentence_list
                            if (s['text'] if isinstance(s, dict) else s))
        n_sent_with_tg = len(alignments)
        loss_pct = (1 - pid_kept / raw_count) * 100 if raw_count > 0 else 0

        print(f"{pid:<6} {raw_count:>10} {n_sent_text - n_sent_with_tg:>8} sent  "
              f"{pid_too_short:>8} {pid_kept:>8} {loss_pct:>6.1f}%")

        total_tg    += raw_count
        total_short += pid_too_short
        total_kept  += pid_kept

    print("-" * 55)
    print(f"{'TOTAL':<6} {total_tg:>10} {'':>8}      "
          f"{total_short:>8} {total_kept:>8} "
          f"{(1 - total_kept / total_tg) * 100:>6.1f}%")
    print(f"\nPhonemes lost to min_samples: {total_short} "
          f"({total_short / total_tg * 100:.1f}%)")


# ═══════════════════════════════════════════════════════════════════════════════
#  MFA: COVERAGE SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

def mfa_coverage_summary(run_config):
    """Print per-patient MFA alignment coverage and phone counts."""
    import tgt

    mfa_output = MFA_OUTPUT_PATH
    mfa_input  = os.path.join(DUTCH_30_PATH, 'mfa_input')
    pr = run_config['patient_range']

    for pid in [f'P{i:02d}' for i in range(pr[0], pr[1] + 1)]:
        tg_dir  = os.path.join(mfa_output, pid)
        lab_dir = os.path.join(mfa_input, pid)

        n_lab = (len([f for f in os.listdir(lab_dir) if f.endswith('.lab')])
                 if os.path.isdir(lab_dir) else 0)
        n_tg  = (len([f for f in os.listdir(tg_dir)  if f.endswith('.TextGrid')])
                 if os.path.isdir(tg_dir) else 0)

        total_phones = 0
        if os.path.isdir(tg_dir):
            for f in os.listdir(tg_dir):
                if not f.endswith('.TextGrid'):
                    continue
                try:
                    tg_obj = tgt.io.read_textgrid(os.path.join(tg_dir, f))
                    tier   = tg_obj.get_tier_by_name('phones')
                    total_phones += sum(1 for a in tier.annotations
                                        if a.text not in ('', 'sp', 'sil', 'spn'))
                except Exception:
                    pass

        pct = n_tg / n_lab * 100 if n_lab else 0
        print(f"{pid}: {n_tg:>3}/{n_lab:>3} aligned ({pct:.0f}%)  "
              f"phones: {total_phones}")


# ═══════════════════════════════════════════════════════════════════════════════
#  WHISPERX ATTACHMENT
# ═══════════════════════════════════════════════════════════════════════════════

def attach_whisperx(pipeline, run_config):
    """Load and attach WhisperX alignment model to the pipeline."""
    if not run_config.get('use_whisperx'):
        return
    import whisperx
    print("Loading WhisperX alignment model...")
    model_a, metadata = whisperx.load_align_model(
        language_code="nl", device="cpu"
    )
    pipeline.whisperx_model    = model_a
    pipeline.whisperx_metadata = metadata
    print("WhisperX ready.")


# ═══════════════════════════════════════════════════════════════════════════════
#  RUN EXPERIMENT (classifier training + evaluation)
# ═══════════════════════════════════════════════════════════════════════════════

def run_experiment(pipeline, order=3, class_weight='balanced', use_groups=False,
                   classifier_type='random_forest', use_viterbi=False,
                   random_state=37, stacking_order=None, stacking_step_size=None,
                   scaler_type='standard', subtract_baseline=False,
                   max_frames=None, min_frames=None, target_frames=None,
                   min_class_samples=5):
    """Train and evaluate per-patient classifiers.

    Returns:
        (name, params, results) where results is a dict of per-patient metrics.
    """
    weight_str = str(class_weight) if class_weight else 'none'
    name = f"{classifier_type}_o{order}_w{weight_str}"
    if use_viterbi:           name += "_viterbi"
    if stacking_order:        name += f"_stack{stacking_order}x{stacking_step_size}"
    if target_frames:         name += f"_resamp{target_frames}"
    if max_frames:            name += f"_max{max_frames}"
    if min_frames is not None and min_frames > 0:
        name += f"_min{min_frames}"
    if scaler_type != 'standard':  name += f"_{scaler_type}"
    if subtract_baseline:     name += "_bsub"

    params = {
        'order': order, 'class_weight': str(class_weight),
        'use_groups': use_groups, 'classifier_type': classifier_type,
        'use_viterbi': use_viterbi, 'stacking_order': stacking_order,
        'stacking_step_size': stacking_step_size, 'max_frames': max_frames,
        'target_frames': target_frames, 'random_state': random_state,
        'scaler_type': scaler_type, 'subtract_baseline': subtract_baseline,
        'min_class_samples': min_class_samples,
    }
    print(f"\nRunning: {name}")

    results = {}
    for pid in sorted(set(pipeline.train['phoneme_participant_ids'])):
        tr_mask = [p == pid for p in pipeline.train['phoneme_participant_ids']]
        te_mask = [p == pid for p in pipeline.test['phoneme_participant_ids']]
        tr_feat = [pipeline.train['features'][i]
                   for i, m in enumerate(tr_mask) if m]
        tr_lbl  = [pipeline.train['phoneme_labels'][i]
                   for i, m in enumerate(tr_mask) if m]
        te_feat = [pipeline.test['features'][i]
                   for i, m in enumerate(te_mask) if m]
        te_lbl  = [pipeline.test['phoneme_labels'][i]
                   for i, m in enumerate(te_mask) if m]

        if len(tr_feat) < 10 or len(te_feat) < 5:
            continue

        valid = {c for c, n in Counter(tr_lbl).items() if n >= min_class_samples}
        tr_feat = [f for f, l in zip(tr_feat, tr_lbl) if l in valid]
        tr_lbl  = [l for l in tr_lbl if l in valid]
        te_feat = [f for f, l in zip(te_feat, te_lbl) if l in valid]
        te_lbl  = [l for l in te_lbl if l in valid]

        if len(tr_feat) < 10 or len(te_feat) < 5:
            continue

        model = MarkovPhonemeModel(
            phonetic_dict=pipeline.detector.phonetic_dict,
            order=order, use_groups=use_groups, class_weight=class_weight,
            classifier_type=classifier_type, random_state=random_state,
            scaler_type=scaler_type, feature_pooling_method='flatten',
        )

        # Pass MFA phone sequences for Viterbi transition model if available
        # (ensures transitions use dutch_cv phone set, not IPA dictionary)
        mfa_seqs = pipeline.train.get('phone_sequences') if use_viterbi else None

        model.train(features=tr_feat, phoneme_labels=tr_lbl,
                    phone_sequences=mfa_seqs)

        preds_nv, _ = model.predict(te_feat, use_viterbi=False)
        preds_nv = [str(p) for p in preds_nv]
        preds = ([str(p) for p in model.predict(te_feat, use_viterbi=True)[0]]
                 if use_viterbi else preds_nv)

        accuracy = sum(p == t for p, t in zip(preds, te_lbl)) / len(te_lbl)

        # Adjusted accuracy: penalize for not predicting all classes
        # adjusted = accuracy × (n_predicted_classes / n_true_classes)
        true_classes  = set(te_lbl)
        pred_classes  = set(preds)
        class_coverage = len(pred_classes & true_classes) / len(true_classes) if true_classes else 1
        adj_accuracy   = accuracy * class_coverage

        results[pid] = {
            'accuracy': accuracy,
            'adj_accuracy': adj_accuracy,
            'class_coverage': class_coverage,
            'n_classes_true': len(true_classes),
            'n_classes_pred': len(pred_classes & true_classes),
            'n_test': len(te_lbl),
            'n_train': len(tr_feat),
        }
        print(f"  {pid}: acc={accuracy:.3f}  adj={adj_accuracy:.3f}  "
              f"classes={len(pred_classes & true_classes)}/{len(true_classes)}  "
              f"(train={len(tr_feat)}, test={len(te_lbl)})")

    return name, params, results


def run_from_config(pipeline, run_config):
    """Convenience wrapper: call run_experiment with run_config dict."""
    name, params, results = run_experiment(
        pipeline,
        order=run_config.get('markov_order', 1),
        class_weight=run_config.get('class_weight', 'balanced'),
        use_groups=run_config.get('use_groups', False),
        classifier_type=run_config.get('classifier_type', 'random_forest'),
        use_viterbi=run_config.get('use_viterbi', False),
        stacking_order=run_config.get('stacking_order'),
        stacking_step_size=run_config.get('stacking_step_size'),
        max_frames=run_config.get('max_frames'),
        min_frames=run_config.get('min_frames'),
        target_frames=run_config.get('target_frames'),
        random_state=run_config.get('random_state', 37),
        scaler_type=run_config.get('scaler_type', 'standard'),
        subtract_baseline=run_config.get('subtract_baseline', False),
        min_class_samples=run_config.get('min_class_samples', 5),
    )
    params['patient_range']             = run_config.get('patient_range')
    params['feature_extraction_method'] = run_config.get('feature_extraction_method')
    params['subtract_baseline']         = run_config.get('subtract_baseline')
    params['sample_fraction']           = run_config.get('sample_fraction')
    return name, params, results


# ═══════════════════════════════════════════════════════════════════════════════
#  CONSECUTIVE-PREDICTION ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_consecutive_predictions(pipeline, run_config,
                                    min_run=2, n_example_words=12):
    """Per-patient analysis of consecutive correct phoneme predictions.

    TRAIN block: detection by word length and position within word.
    TEST  block: LR classifier, position accuracy, top runs, example words.
    """
    pr   = run_config['patient_range']
    pids = sorted(set(
        p for p in pipeline.train['phoneme_participant_ids']
        if p in [f'P{i:02d}' for i in range(pr[0], pr[1] + 1)]
    ))

    for pid in pids:
        # ── collect data ──────────────────────────────────────────────
        tr_idx = [i for i, p in enumerate(
            pipeline.train['phoneme_participant_ids']) if p == pid]
        X_tr = np.array([np.array(pipeline.train['features'][i]).flatten()
                         for i in tr_idx])
        y_tr = [pipeline.train['phoneme_labels'][i]  for i in tr_idx]
        w_tr = [pipeline.train['phoneme_words'][i]   for i in tr_idx]

        te_idx = [i for i, p in enumerate(
            pipeline.test['phoneme_participant_ids']) if p == pid]
        X_te = np.array([np.array(pipeline.test['features'][i]).flatten()
                         for i in te_idx])
        y_te = [pipeline.test['phoneme_labels'][i]  for i in te_idx]
        w_te = [pipeline.test['phoneme_words'][i]   for i in te_idx]

        if len(X_tr) == 0 or len(X_te) == 0:
            print(f"{pid}: no data, skipping")
            continue

        # ═══════ TRAIN: detection analysis ════════════════════════════
        print(f"\n{'='*70}")
        print(f"{pid}  train={len(y_tr)}  test={len(y_te)}")

        def word_groups_from(word_list):
            samples = list(zip(word_list, range(len(word_list))))
            groups  = []
            for word, grp in groupby(samples, key=lambda s: s[0]):
                grp      = list(grp)
                detected = len(grp)
                exp_ph   = pipeline.phonetic_dict.extract_phonemes(word)
                expected = len(exp_ph) if exp_ph else detected
                groups.append({'word': word, 'detected': detected,
                               'expected': expected})
            return groups

        train_groups = word_groups_from(w_tr)

        by_len = defaultdict(lambda: {'words': 0, 'ph_detected': 0,
                                       'ph_expected': 0, 'full_det': 0})
        for wg in train_groups:
            L = wg['expected']
            by_len[L]['words']       += 1
            by_len[L]['ph_expected'] += wg['expected']
            by_len[L]['ph_detected'] += min(wg['detected'], wg['expected'])
            by_len[L]['full_det']    += int(wg['detected'] >= wg['expected'])

        print(f"\n  TRAIN detection by word length:")
        print(f"  {'WrdLen':<8} {'#Words':>7} {'FullWord%':>11} {'PhDet%':>9}")
        for L in sorted(by_len):
            s  = by_len[L]
            fp = s['full_det'] / s['words'] * 100 if s['words'] else 0
            pp = s['ph_detected'] / s['ph_expected'] * 100 if s['ph_expected'] else 0
            print(f"  {L:<8} {s['words']:>7} {fp:>10.1f}%  {pp:>8.1f}%")

        # ═══════ TEST: classifier + position stats ════════════════════
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        le    = LabelEncoder()
        y_tr_e = le.fit_transform(y_tr)
        known = [l in le.classes_ for l in y_te]

        X_te_k = X_te_s[[i for i, m in enumerate(known) if m]]
        y_te_k = le.transform([l for l, m in zip(y_te, known) if m])
        w_te_k = [w for w, m in zip(w_te, known) if m]

        if len(X_te_k) == 0:
            print("  No test samples with known labels")
            continue

        clf = LogisticRegression(max_iter=1000, class_weight='balanced',
                                 C=1.0, random_state=37)
        clf.fit(X_tr_s, y_tr_e)
        y_pred  = clf.predict(X_te_k)
        correct = (y_pred == y_te_k)
        acc     = np.mean(correct)
        print(f"\n  TEST accuracy: {acc:.1%}")

        # consecutive runs
        test_samples = [
            {'true': le.classes_[y_te_k[i]],
             'pred': le.classes_[y_pred[i]],
             'correct': bool(correct[i]),
             'word': w_te_k[i]}
            for i in range(len(y_te_k))
        ]

        runs = []
        i = 0
        while i < len(test_samples):
            if test_samples[i]['correct']:
                j = i
                while j < len(test_samples) and test_samples[j]['correct']:
                    j += 1
                if j - i >= min_run:
                    runs.append({'len': j - i, 'samples': test_samples[i:j]})
                i = j
            else:
                i += 1

        if runs:
            runs.sort(key=lambda r: r['len'], reverse=True)
            print(f"\n  Top consecutive runs (>={min_run}):")
            for r in runs[:10]:
                phones = ' '.join(s['true'] for s in r['samples'])
                words  = ' '.join(dict.fromkeys(s['word'] for s in r['samples']))
                print(f"    {r['len']:>2} correct: [{phones}]  ({words})")


def plot_position_accuracy(pipeline, run_config, max_pos=12):
    """Bar plot of phoneme prediction accuracy by position within word.

    For each patient produces one figure with two subplots:
      Top:    TRAIN — detection rate per position (how many phonemes were
              found at each word position out of how many were expected).
      Bottom: TEST  — prediction accuracy per position (how many phonemes
              at each position were classified correctly).

    Bars show total count (light) with correct count overlaid (dark).
    Percentage labels are shown on each bar.

    Args:
        pipeline: Dutch30Pipeline with train/test populated.
        run_config: dict with 'patient_range'.
        max_pos: int, maximum position to show (higher positions are rare
            and clutter the plot).
    """
    import matplotlib.pyplot as plt
    from itertools import groupby
    from collections import defaultdict

    pr   = run_config['patient_range']
    pids = sorted(set(
        p for p in pipeline.train['phoneme_participant_ids']
        if p in [f'P{i:02d}' for i in range(pr[0], pr[1] + 1)]
    ))

    for pid in pids:
        # ── collect data ────────────────────────��─────────────────────
        tr_idx = [i for i, p in enumerate(
            pipeline.train['phoneme_participant_ids']) if p == pid]
        y_tr = [pipeline.train['phoneme_labels'][i] for i in tr_idx]
        w_tr = [pipeline.train['phoneme_words'][i]  for i in tr_idx]

        te_idx = [i for i, p in enumerate(
            pipeline.test['phoneme_participant_ids']) if p == pid]
        X_tr = np.array([np.array(pipeline.train['features'][i]).flatten()
                         for i in tr_idx])
        X_te = np.array([np.array(pipeline.test['features'][i]).flatten()
                         for i in te_idx])
        y_te = [pipeline.test['phoneme_labels'][i]  for i in te_idx]
        w_te = [pipeline.test['phoneme_words'][i]   for i in te_idx]

        if len(X_tr) == 0 or len(X_te) == 0:
            continue

        # ── TRAIN: detection per position ─────────────────────────────
        def word_groups_from(word_list):
            groups = []
            for word, grp in groupby(
                    zip(word_list, range(len(word_list))),
                    key=lambda s: s[0]):
                grp      = list(grp)
                detected = len(grp)
                exp_ph   = pipeline.phonetic_dict.extract_phonemes(word)
                expected = len(exp_ph) if exp_ph else detected
                groups.append({'word': word, 'detected': detected,
                               'expected': expected})
            return groups

        train_groups = word_groups_from(w_tr)

        tr_pos = defaultdict(lambda: {'detected': 0, 'expected': 0})
        for wg in train_groups:
            for pos in range(min(wg['expected'], max_pos)):
                tr_pos[pos]['expected'] += 1
                if pos < wg['detected']:
                    tr_pos[pos]['detected'] += 1

        # ── TEST: classify + accuracy per position ────────────────────
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        from sklearn.linear_model import LogisticRegression

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        le     = LabelEncoder()
        y_tr_e = le.fit_transform(y_tr)
        known  = [l in le.classes_ for l in y_te]

        X_te_k = X_te_s[[i for i, m in enumerate(known) if m]]
        y_te_k = le.transform([l for l, m in zip(y_te, known) if m])
        w_te_k = [w for w, m in zip(w_te, known) if m]

        if len(X_te_k) == 0:
            continue

        clf = LogisticRegression(max_iter=1000, class_weight='balanced',
                                 C=1.0, random_state=37)
        clf.fit(X_tr_s, y_tr_e)
        y_pred  = clf.predict(X_te_k)
        correct = (y_pred == y_te_k)

        # group test phonemes by word → assign position
        test_samples = [
            {'correct': bool(correct[i]), 'word': w_te_k[i]}
            for i in range(len(y_te_k))
        ]

        te_pos = defaultdict(lambda: {'correct': 0, 'total': 0})
        idx = 0
        for word, grp in groupby(test_samples, key=lambda s: s['word']):
            for pos, s in enumerate(grp):
                if pos >= max_pos:
                    break
                te_pos[pos]['total']   += 1
                te_pos[pos]['correct'] += int(s['correct'])
            idx += 1

        # ── PLOT ──────────────────────────────────────────────────────
        positions = sorted(set(list(tr_pos.keys()) + list(te_pos.keys())))
        positions = [p for p in positions if p < max_pos]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        fig.suptitle(f'{pid}  —  Phoneme accuracy by position in word',
                     fontsize=14, fontweight='bold')
        bar_w = 0.55

        # ── Top: TRAIN detection ──────────────────────────────────────
        tr_expected = [tr_pos[p]['expected'] for p in positions]
        tr_detected = [tr_pos[p]['detected'] for p in positions]

        ax1.bar(positions, tr_expected, width=bar_w,
                color='#c8daf0', edgecolor='#8fafd0', label='Expected')
        ax1.bar(positions, tr_detected, width=bar_w,
                color='#3b7dd8', edgecolor='#2a5a9e', label='Detected')

        for p in positions:
            exp = tr_pos[p]['expected']
            det = tr_pos[p]['detected']
            pct = det / exp * 100 if exp else 0
            ax1.text(p, max(exp, det) + max(tr_expected) * 0.02,
                     f'{pct:.0f}%', ha='center', va='bottom', fontsize=8,
                     fontweight='bold' if pct >= 100 else 'normal',
                     color='#1a5c1a' if pct >= 100 else '#333')

        ax1.set_ylabel('# phonemes')
        ax1.set_title('TRAIN — phoneme detection rate per position',
                       fontsize=11)
        ax1.legend(loc='upper right', fontsize=9)
        ax1.set_ylim(0, max(tr_expected) * 1.18)

        # ── Bottom: TEST prediction accuracy ──────────────────────────
        te_total   = [te_pos[p]['total']   for p in positions]
        te_correct = [te_pos[p]['correct'] for p in positions]

        ax2.bar(positions, te_total, width=bar_w,
                color='#f0d4c8', edgecolor='#d0a08f', label='Total test')
        ax2.bar(positions, te_correct, width=bar_w,
                color='#d84b3b', edgecolor='#9e3a2a', label='Correct')

        for p in positions:
            tot = te_pos[p]['total']
            cor = te_pos[p]['correct']
            pct = cor / tot * 100 if tot else 0
            ax2.text(p, max(tot, cor) + max(te_total) * 0.02,
                     f'{pct:.0f}%', ha='center', va='bottom', fontsize=8,
                     fontweight='bold' if pct >= 20 else 'normal',
                     color='#1a5c1a' if pct >= 20 else '#999')

        ax2.set_xlabel('Position within word')
        ax2.set_ylabel('# phonemes')
        ax2.set_title(f'TEST — prediction accuracy per position  '
                       f'(overall {np.mean(correct):.1%})', fontsize=11)
        ax2.legend(loc='upper right', fontsize=9)
        ax2.set_xticks(positions)
        ax2.set_ylim(0, max(te_total) * 1.18)

        plt.tight_layout()
        plt.show()


def _run_crf_experiment(pipeline, run_config):
    """Run CRF (Conditional Random Fields) per patient.

    CRF is a sequence model: it groups phonemes into word-level sequences
    and predicts the label sequence jointly, capturing transitions between
    phonemes within a word.  Features are reduced with PCA first (CRFsuite
    uses sparse dict features internally, so fewer dense dimensions = faster).

    Returns:
        dict: {pid: {'accuracy': float, 'n_test': int, 'n_train': int}}
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from itertools import groupby
    from collections import Counter

    try:
        from sklearn_crfsuite import CRF
    except ImportError:
        print("  >> crf: FAILED — pip install sklearn-crfsuite")
        return {}

    n_pca = 50  # reduce features for CRF speed
    min_class = run_config.get('min_class_samples', 5)

    results = {}
    for pid in sorted(set(pipeline.train['phoneme_participant_ids'])):
        tr_mask = [p == pid for p in pipeline.train['phoneme_participant_ids']]
        te_mask = [p == pid for p in pipeline.test['phoneme_participant_ids']]

        tr_feat = [pipeline.train['features'][i]
                   for i, m in enumerate(tr_mask) if m]
        tr_lbl  = [pipeline.train['phoneme_labels'][i]
                   for i, m in enumerate(tr_mask) if m]
        tr_wrd  = [pipeline.train['phoneme_words'][i]
                   for i, m in enumerate(tr_mask) if m]

        te_feat = [pipeline.test['features'][i]
                   for i, m in enumerate(te_mask) if m]
        te_lbl  = [pipeline.test['phoneme_labels'][i]
                   for i, m in enumerate(te_mask) if m]
        te_wrd  = [pipeline.test['phoneme_words'][i]
                   for i, m in enumerate(te_mask) if m]

        if len(tr_feat) < 10 or len(te_feat) < 5:
            continue

        # Filter rare classes
        valid = {c for c, n in Counter(tr_lbl).items() if n >= min_class}
        keep_tr = [i for i, l in enumerate(tr_lbl) if l in valid]
        keep_te = [i for i, l in enumerate(te_lbl) if l in valid]
        tr_feat = [tr_feat[i] for i in keep_tr]
        tr_lbl  = [tr_lbl[i]  for i in keep_tr]
        tr_wrd  = [tr_wrd[i]  for i in keep_tr]
        te_feat = [te_feat[i] for i in keep_te]
        te_lbl  = [te_lbl[i]  for i in keep_te]
        te_wrd  = [te_wrd[i]  for i in keep_te]

        if len(tr_feat) < 10 or len(te_feat) < 5:
            continue

        # Flatten + scale + PCA
        X_tr = np.array([np.array(f).flatten() for f in tr_feat])
        X_te = np.array([np.array(f).flatten() for f in te_feat])

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)

        n_comp = min(n_pca, X_tr.shape[1], X_tr.shape[0])
        pca = PCA(n_components=n_comp)
        X_tr = pca.fit_transform(X_tr)
        X_te = pca.transform(X_te)

        # Group into word-level sequences
        def to_sequences(X, labels, words):
            seqs, lbl_seqs = [], []
            cur_x, cur_y = [], []
            prev_w = None
            for i, (x, l, w) in enumerate(zip(X, labels, words)):
                if w != prev_w and prev_w is not None and cur_x:
                    seqs.append(cur_x)
                    lbl_seqs.append(cur_y)
                    cur_x, cur_y = [], []
                cur_x.append({f'f{j}': float(v) for j, v in enumerate(x)})
                cur_y.append(l)
                prev_w = w
            if cur_x:
                seqs.append(cur_x)
                lbl_seqs.append(cur_y)
            return seqs, lbl_seqs

        X_tr_seq, y_tr_seq = to_sequences(X_tr, tr_lbl, tr_wrd)
        X_te_seq, y_te_seq = to_sequences(X_te, te_lbl, te_wrd)

        crf = CRF(
            algorithm='lbfgs',
            c1=0.1,              # L1 regularization
            c2=0.1,              # L2 regularization
            max_iterations=100,
            all_possible_transitions=True,
        )
        crf.fit(X_tr_seq, y_tr_seq)

        y_pred_seq = crf.predict(X_te_seq)

        y_pred = [p for seq in y_pred_seq for p in seq]
        y_true = [l for seq in y_te_seq  for l in seq]

        accuracy = sum(p == t for p, t in zip(y_pred, y_true)) / len(y_true)

        true_classes  = set(y_true)
        pred_classes  = set(y_pred)
        class_coverage = len(pred_classes & true_classes) / len(true_classes) if true_classes else 1
        adj_accuracy   = accuracy * class_coverage

        results[pid] = {
            'accuracy': accuracy,
            'adj_accuracy': adj_accuracy,
            'class_coverage': class_coverage,
            'n_classes_true': len(true_classes),
            'n_classes_pred': len(pred_classes & true_classes),
            'n_test': len(y_true),
            'n_train': len(tr_lbl),
            'predictions': y_pred,
            'true_labels': y_true,
        }
        print(f"  {pid}: acc={accuracy:.3f}  adj={adj_accuracy:.3f}  "
              f"classes={len(pred_classes & true_classes)}/{len(true_classes)}  "
              f"(train={len(tr_lbl)}, test={len(te_lbl)})")

    return results


def compare_classifiers(pipeline, run_config, classifiers=None):
    """Test multiple classifiers and return a results dict for heatmap plotting.

    Args:
        pipeline: Dutch30Pipeline with train/test populated.
        run_config: base run_config dict.
        classifiers: list of classifier_type strings. If None, uses a
            sensible default list.  Include 'crf' for Conditional Random
            Fields (sequence model, needs sklearn-crfsuite installed).

    Returns:
        dict: {classifier_type: {pid: accuracy, ...}, ...}
        Call plot_classifier_heatmap(results) to visualize.
    """
    if classifiers is None:
        classifiers = [
            'logistic_regression',
            'random_forest',
            'extra_trees',
            'svm_linear',
            'knn',
            'lda',
            'gaussian_nb',
            'mlp',
            'crf',
        ]

    all_results = {}

    for ct in classifiers:
        print(f"\n{'='*60}")
        try:
            if ct == 'crf':
                # CRF is a sequence model — needs special handling
                print(f"Running: crf (sequence model, PCA→50d)...")
                results = _run_crf_experiment(pipeline, run_config)
            else:
                rc = dict(run_config)
                rc['classifier_type'] = ct
                rc['use_viterbi'] = False
                name, params, results = run_from_config(pipeline, rc)
        except Exception as e:
            print(f"  >> {ct}: FAILED — {e}")
            continue

        if results:
            accs = [r['accuracy'] for r in results.values()]
            adjs = [r['adj_accuracy'] for r in results.values()]
            covs = [r['class_coverage'] for r in results.values()]
            print(f"  >> {ct}: acc={np.mean(accs):.3f}  "
                  f"adj={np.mean(adjs):.3f}  "
                  f"coverage={np.mean(covs):.1%}")
            all_results[ct] = {
                pid: {
                    'accuracy': r['accuracy'],
                    'adj_accuracy': r['adj_accuracy'],
                    'class_coverage': r['class_coverage'],
                }
                for pid, r in results.items()
            }

    return all_results


def plot_classifier_heatmap(comparison_results, metric='adj_accuracy'):
    """Heatmap of classifier × patient accuracy from compare_classifiers().

    Args:
        comparison_results: dict from compare_classifiers().
        metric: 'adj_accuracy' (default, penalizes missing classes),
                'accuracy' (raw), or 'class_coverage'.
    """
    import matplotlib.pyplot as plt

    if not comparison_results:
        print("No results to plot.")
        return

    metric_labels = {
        'adj_accuracy':   'Adjusted accuracy (acc × class coverage)',
        'accuracy':       'Raw accuracy',
        'class_coverage': 'Class coverage (predicted / true classes)',
    }

    classifiers = list(comparison_results.keys())
    all_pids = sorted(set(
        pid for res in comparison_results.values() for pid in res
    ))

    # Build matrix: rows = classifiers, cols = patients
    # Handle both old format (float) and new format (dict with metrics)
    matrix = []
    for ct in classifiers:
        row = []
        for pid in all_pids:
            val = comparison_results[ct].get(pid, 0)
            if isinstance(val, dict):
                row.append(val.get(metric, val.get('accuracy', 0)))
            else:
                row.append(val)  # backward compat: plain float = accuracy
        matrix.append(row)
    matrix = np.array(matrix)

    # Compute means for sorting and display
    row_means = matrix.mean(axis=1)
    sort_idx = np.argsort(row_means)[::-1]
    matrix = matrix[sort_idx]
    classifiers = [classifiers[i] for i in sort_idx]
    row_means = row_means[sort_idx]

    fig, ax = plt.subplots(figsize=(max(10, len(all_pids) * 1.1),
                                     max(5, len(classifiers) * 0.6)))

    vmax = max(0.3, matrix.max())
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto',
                   vmin=0, vmax=vmax)

    # Labels
    ax.set_xticks(range(len(all_pids)))
    ax.set_xticklabels(all_pids, rotation=45, ha='right')
    ax.set_yticks(range(len(classifiers)))
    ax.set_yticklabels([f'{ct}  ({row_means[i]:.1%})'
                        for i, ct in enumerate(classifiers)])

    # Annotate cells
    for i in range(len(classifiers)):
        for j in range(len(all_pids)):
            val = matrix[i, j]
            color = 'white' if val < vmax * 0.35 else 'black'
            ax.text(j, i, f'{val:.1%}', ha='center', va='center',
                    fontsize=9, color=color, fontweight='bold')

    ax.set_title(metric_labels.get(metric, metric),
                 fontsize=13, fontweight='bold', pad=12)

    plt.colorbar(im, ax=ax, label=metric, shrink=0.8)
    plt.tight_layout()
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
#  HYPERPARAMETER SWEEP
# ═══════════════════════════════════════════════════════════════════════════════

def run_sweep(pipeline, run_config, cached_train, cached_test):
    """Run a grid of stacking / classifier / frame-filter experiments."""
    sweep_configs = [
        {'stacking_order': 5,  'stacking_step_size': 1, 'min_frames': 3,  'max_frames': 150, 'scaler_type': 'standard'},
        {'stacking_order': 5,  'stacking_step_size': 2, 'min_frames': 3,  'max_frames': 150, 'scaler_type': 'standard'},
        {'stacking_order': 7,  'stacking_step_size': 1, 'min_frames': 3,  'max_frames': 150, 'scaler_type': 'standard'},
        {'stacking_order': 7,  'stacking_step_size': 2, 'min_frames': 3,  'max_frames': 150, 'scaler_type': 'standard'},
        {'stacking_order': 9,  'stacking_step_size': 2, 'min_frames': 3,  'max_frames': 150, 'scaler_type': 'standard'},
        {'stacking_order': 5,  'stacking_step_size': 1, 'min_frames': 0,  'max_frames': 300, 'scaler_type': 'standard'},
        {'stacking_order': 7,  'stacking_step_size': 2, 'min_frames': 0,  'max_frames': 300, 'scaler_type': 'standard'},
        {'stacking_order': 9,  'stacking_step_size': 2, 'min_frames': 0,  'max_frames': 300, 'scaler_type': 'standard'},
    ]

    logger = ExperimentLogger('experiments_sweep.json')

    done_names = {e['name'] for e in logger.experiments}
    remaining  = [c for c in sweep_configs
                  if _sweep_name(c, run_config) not in done_names]
    print(f"Sweep: {len(sweep_configs)} total | "
          f"Done: {len(sweep_configs) - len(remaining)} | "
          f"Remaining: {len(remaining)}")

    for idx, sc in enumerate(remaining, 1):
        so = sc['stacking_order']
        ss = sc['stacking_step_size']
        mn = sc['min_frames']
        mx = sc['max_frames']
        scl = sc['scaler_type']
        bl  = run_config.get('subtract_baseline', False)

        print(f"\n--- [{idx}/{len(remaining)}] so={so} ss={ss} "
              f"fr={mn}-{mx} sc={scl} bl={bl} ---")

        pipeline.train = copy.deepcopy(cached_train)
        pipeline.test  = copy.deepcopy(cached_test)

        for data in [pipeline.train, pipeline.test]:
            data['phoneme_positions'] = [0] * len(data['phoneme_positions'])

        pipeline.step5a_filter_by_frame_count(min_frames=mn, max_frames=mx)
        pipeline.step5b_stack_features(model_order=so, step_size=ss)
        pipeline.step5c_collapse_to_phoneme_level()
        pipeline.dutch30_step6_resolve_unknowns()
        pipeline.step7_filter_unknowns(
            unknown_keep_ratio=run_config['unknown_keep_ratio'])

        rc = dict(run_config)
        rc.update(sc)
        name, params, results = run_from_config(pipeline, rc)

        if results:
            logger.log(name, params, results)

    logger.print_table()
    return logger


def _sweep_name(sc, run_config):
    """Generate experiment name for a sweep config."""
    ct = run_config.get('classifier_type', 'logistic_regression')
    o  = run_config.get('markov_order', 1)
    cw = run_config.get('class_weight', 'balanced')
    so = sc['stacking_order']
    ss = sc['stacking_step_size']
    return f"{ct}_o{o}_w{cw}_viterbi_stack{so}x{ss}"


# ═══════════════════════════════════════════════════════════════════════════════
#  PATH A: WAV2VEC / WHISPERX PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def run_path_a(pipeline, run_config):
    """Run the full wav2vec/WhisperX boundary detection pipeline.

    Uses 3-level pickle checkpoint system:
        STEP5_CACHE  -> FRAME_CACHE  -> STEP3_CHECKPOINT  -> scratch
    """
    pr = run_config['patient_range']
    so = run_config['stacking_order']
    ss = run_config['stacking_step_size']
    tf = run_config['target_frames']

    use_stacking   = so is not None
    use_resampling = tf is not None

    STEP3_CKPT, FRAME_CACHE, STEP5_CACHE = make_checkpoint_names(run_config)

    cached_train = cached_test = None

    # ── 1. Try post-step5c cache ─────────────────────────────────────────
    if os.path.exists(STEP5_CACHE):
        print(f"  Step-5 cache found: {STEP5_CACHE}")
        with open(STEP5_CACHE, 'rb') as f:
            state = pickle.load(f)
        pipeline.train        = state['train']
        pipeline.test         = state['test']
        pipeline.split_result = state['split_result']
        cached_train          = state['cached_train']
        cached_test           = state['cached_test']
        count(pipeline, "Loaded from step-5 cache")

    # ── 2. Try frame-level cache ─────────────────────────────────────────
    elif os.path.exists(FRAME_CACHE):
        print(f"  Frame cache found: {FRAME_CACHE}")
        with open(FRAME_CACHE, 'rb') as f:
            state = pickle.load(f)
        pipeline.train        = copy.deepcopy(state['cached_train'])
        pipeline.test         = copy.deepcopy(state['cached_test'])
        pipeline.split_result = state['split_result']
        cached_train          = state['cached_train']
        cached_test           = state['cached_test']
        count(pipeline, "Frame cache loaded")

        _run_step5abc(pipeline, run_config)

        print(f"Saving step-5 cache -> {STEP5_CACHE}")
        with open(STEP5_CACHE, 'wb') as f:
            pickle.dump({
                'train': pipeline.train, 'test': pipeline.test,
                'split_result': pipeline.split_result,
                'cached_train': cached_train, 'cached_test': cached_test,
            }, f)

    # ── 3. From scratch (or step-3 checkpoint) ───────────────────────────
    else:
        if os.path.exists(STEP3_CKPT):
            print(f"  Step-3 checkpoint found: {STEP3_CKPT}")
            with open(STEP3_CKPT, 'rb') as f:
                state = pickle.load(f)
            pipeline.split_result      = state['split_result']
            pipeline.patient_data      = state['patient_data']
            pipeline.patient_baselines = state['patient_baselines']
            print("  Continuing from step 4...")
        else:
            print("  No checkpoint -> running steps 1-3 from scratch...")
            pipeline.step1_load_dutch30_data(patient_range=pr)
            pipeline.step2_split_by_instances()
            pipeline.step3_load_channel_exclusions('channel_exclusions.json')
            pipeline.apply_channel_exclusions()
            pipeline.print_channel_counts()
            with open(STEP3_CKPT, 'wb') as f:
                pickle.dump({
                    'split_result':      pipeline.split_result,
                    'patient_data':      pipeline.patient_data,
                    'patient_baselines': getattr(pipeline, 'patient_baselines', None),
                }, f)
            print(f"  Step-3 checkpoint saved: {STEP3_CKPT}")

        # steps 4-5_accumulate
        pipeline.step4_custom_detector()
        pipeline.step5_accumulate_data_dutch30()
        count(pipeline, "After step5 accumulate")

        cached_train = copy.deepcopy(pipeline.train)
        cached_test  = copy.deepcopy(pipeline.test)
        print(f"Saving frame cache -> {FRAME_CACHE}")
        with open(FRAME_CACHE, 'wb') as f:
            pickle.dump({
                'cached_train': cached_train, 'cached_test': cached_test,
                'split_result': pipeline.split_result,
            }, f)

        _run_step5abc(pipeline, run_config)

        print(f"Saving step-5 cache -> {STEP5_CACHE}")
        with open(STEP5_CACHE, 'wb') as f:
            pickle.dump({
                'train': pipeline.train, 'test': pipeline.test,
                'split_result': pipeline.split_result,
                'cached_train': cached_train, 'cached_test': cached_test,
            }, f)

    # ── Steps 6-7 (always) ───────────────────────────────────────────────
    pipeline.dutch30_step6_resolve_unknowns()
    count(pipeline, "After step6")

    pipeline.step7_filter_unknowns(
        unknown_keep_ratio=run_config['unknown_keep_ratio'])
    count(pipeline, "After step7")

    return cached_train, cached_test


# ═══════════════════════════════════════════════════════════════════════════════
#  PATH B: MFA PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def run_path_b(pipeline, run_config):
    """Run the MFA-based pipeline (replaces step4 + step5_accumulate).

    Requires MFA TextGrids to already exist in Dutch_30patients/mfa_output/.
    """
    pr = run_config['patient_range']
    STEP3_CKPT = make_checkpoint_names(run_config)[0]

    # Ensure split_result is loaded
    if not hasattr(pipeline, 'split_result') or pipeline.split_result is None:
        if os.path.exists(STEP3_CKPT):
            print(f"Loading step-3 checkpoint: {STEP3_CKPT}")
            with open(STEP3_CKPT, 'rb') as f:
                state = pickle.load(f)
            pipeline.split_result      = state['split_result']
            pipeline.patient_data      = state['patient_data']
            pipeline.patient_baselines = state['patient_baselines']
        else:
            print("Running steps 1-3...")
            pipeline.step1_load_dutch30_data(patient_range=pr)
            pipeline.step2_split_by_instances()
            pipeline.step3_load_channel_exclusions('channel_exclusions.json')
            pipeline.apply_channel_exclusions()

    print("Building features from MFA alignments...")
    mfa_train, mfa_test = build_mfa_features(pipeline, run_config)

    pipeline.train = mfa_train
    pipeline.test  = mfa_test
    cached_train   = copy.deepcopy(mfa_train)
    cached_test    = copy.deepcopy(mfa_test)

    count(pipeline, "After MFA accumulation")

    # Step 5a/b/c
    _run_step5abc(pipeline, run_config)

    # Steps 6-7
    pipeline.dutch30_step6_resolve_unknowns()
    count(pipeline, "After step6")

    pipeline.step7_filter_unknowns(
        unknown_keep_ratio=run_config['unknown_keep_ratio'])
    count(pipeline, "After step7")

    return cached_train, cached_test


# ═══════════════════════════════════════════════════════════════════════════════
#  SHARED: STEP 5a/b/c
# ═══════════════════════════════════════════════════════════════════════════════

def _run_step5abc(pipeline, run_config):
    """Reset positions, then run steps 5a, 5b, 5c."""
    so = run_config['stacking_order']
    ss = run_config['stacking_step_size']
    tf = run_config['target_frames']
    use_stacking   = so is not None
    use_resampling = tf is not None

    for data in [pipeline.train, pipeline.test]:
        data['phoneme_positions'] = [0] * len(data['phoneme_positions'])

    pipeline.step5a_filter_by_frame_count(
        min_frames=run_config['min_frames'],
        max_frames=run_config['max_frames'])
    count(pipeline, "After 5a")

    if use_stacking:
        pipeline.step5b_stack_features(model_order=so, step_size=ss)
        count(pipeline, "After 5b (stacking)")
    elif use_resampling:
        pipeline.step5b_normalize_feature_lengths(target_frames=tf)
        count(pipeline, "After 5b (resampling)")

    pipeline.step5c_collapse_to_phoneme_level()
    count(pipeline, "After 5c")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--mfa', action='store_true',
                        help='Use MFA alignments (Path B) instead of wav2vec (Path A)')
    parser.add_argument('--export-mfa', action='store_true',
                        help='Export audio/text for MFA alignment, then exit')
    parser.add_argument('--diagnose-mfa', action='store_true',
                        help='Show MFA phoneme loss breakdown, then exit')
    parser.add_argument('--mfa-coverage', action='store_true',
                        help='Show MFA alignment coverage, then exit')
    parser.add_argument('--sweep', action='store_true',
                        help='Run hyperparameter sweep after pipeline')
    parser.add_argument('--analyze', action='store_true',
                        help='Run consecutive-prediction analysis')
    parser.add_argument('--patients', type=str, default='21-30',
                        help='Patient range, e.g. "21-30" (default: 21-30)')
    args = parser.parse_args()

    # Parse patient range
    p_start, p_end = [int(x) for x in args.patients.split('-')]
    run_config = dict(DEFAULT_RUN_CONFIG)
    run_config['patient_range'] = (p_start, p_end)

    # ── Create pipeline ──────────────────────────────────────────────────
    extractor = Dutch30FeatureExtractor()
    pipeline  = Dutch30Pipeline(
        dutch30_extractor=extractor,
        debug_mode=False,
        feature_extraction_method=run_config['feature_extraction_method'],
        use_wav2vec=run_config['use_wav2vec'],
        subtract_baseline=run_config['subtract_baseline'],
        use_rms_boundaries=False,
        use_multifeature=False,
    )

    # ── Diagnostic / export modes (exit early) ───────────────────────────
    if args.export_mfa:
        # Need step 1-3 for split_result
        STEP3_CKPT = make_checkpoint_names(run_config)[0]
        if os.path.exists(STEP3_CKPT):
            with open(STEP3_CKPT, 'rb') as f:
                state = pickle.load(f)
            pipeline.split_result      = state['split_result']
            pipeline.patient_data      = state['patient_data']
            pipeline.patient_baselines = state['patient_baselines']
        else:
            pipeline.step1_load_dutch30_data(
                patient_range=run_config['patient_range'])
            pipeline.step2_split_by_instances()

        out_dir = os.path.join(DUTCH_30_PATH, 'mfa_input')
        pr = run_config['patient_range']
        for pid in [f'P{i:02d}' for i in range(pr[0], pr[1] + 1)]:
            export_sentences_for_mfa(pid, pipeline, out_dir)

        print("\n--- MFA commands (run in conda 'aligner' environment) ---")
        print("# mfa model download acoustic dutch_cv")
        print("# mfa model download dictionary dutch_cv")
        inp = os.path.join(DUTCH_30_PATH, 'mfa_input').replace('/', '\\')
        out = MFA_OUTPUT_PATH.replace('/', '\\')
        print(f"# mfa validate {inp} dutch_cv dutch_cv")
        print(f"# mfa align {inp} dutch_cv dutch_cv {out} --clean")
        return

    if args.mfa_coverage:
        mfa_coverage_summary(run_config)
        return

    if args.diagnose_mfa:
        STEP3_CKPT = make_checkpoint_names(run_config)[0]
        if os.path.exists(STEP3_CKPT):
            with open(STEP3_CKPT, 'rb') as f:
                state = pickle.load(f)
            pipeline.split_result      = state['split_result']
            pipeline.patient_data      = state['patient_data']
            pipeline.patient_baselines = state['patient_baselines']
        else:
            pipeline.step1_load_dutch30_data(
                patient_range=run_config['patient_range'])
            pipeline.step2_split_by_instances()
        diagnose_mfa_loss(pipeline, run_config)
        return

    # ── Run pipeline ─────────────────────────────────────────────────────
    if not args.mfa:
        attach_whisperx(pipeline, run_config)

    if args.mfa:
        print("\n=== PATH B: MFA alignment ===")
        cached_train, cached_test = run_path_b(pipeline, run_config)
    else:
        print("\n=== PATH A: wav2vec / WhisperX ===")
        cached_train, cached_test = run_path_a(pipeline, run_config)

    # ── Classification ───────────────────────────────────────────────────
    name, params, results = run_from_config(pipeline, run_config)

    if results:
        accs = [r['accuracy'] for r in results.values()]
        print(f"\nMean accuracy: {np.mean(accs):.3f}  "
              f"({len(results)} patients)")

    # ── Optional: sweep ──────────────────────────────────────────────────
    if args.sweep and cached_train is not None:
        print("\n=== Hyperparameter sweep ===")
        run_sweep(pipeline, run_config, cached_train, cached_test)

    # ── Optional: consecutive analysis ───────────────────────────────────
    if args.analyze:
        print("\n=== Consecutive prediction analysis ===")
        analyze_consecutive_predictions(pipeline, run_config)


if __name__ == '__main__':
    main()
