# Hybrid decoder: v6 boundaries + CTC per-frame phoneme classification.
#
# The motivation: CTC alone tends to under-predict (len_ratio < 1.0×),
# while v6's boundary detector + count head produces well-calibrated
# segmentation. Take v6's segmentation as ground truth for boundaries,
# then within each segment use CTC's per-frame phoneme logits to vote on
# which phoneme that segment represents.
#
# Net effect:
#   - len_ratio = 1.0× by construction (one phoneme emitted per v6 segment)
#   - per-segment classification leverages CTC's frame-level supervision
#   - no retraining: both v6 (boundary_detector_v6_*.pt) and CTC
#     (frame_ctc_v3_*.pt) are loaded as-is

import os
import glob
import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from datetime import datetime

from config import DUTCH_30_PATH
from run_pipeline import load_mfa_alignments
from boundary_detector_joint_audio import (
    JointBoundaryDetector,
    HIDDEN_DIM as V6_HIDDEN, N_LSTM_LAYERS as V6_LAYERS, DROPOUT as V6_DROPOUT,
    split_by_sentence, fit_train_stats, apply_stats, ALL_PIDS,
)
from e2e_brain_decoder import (
    build_joint_dataset_fixed, edit_distance, rich_metrics,
    print_rich_metrics_table, CountWordAwareDetector,
)
from frame_level_decoder import (
    FrameCTCModel, build_ctc_dataset, attach_target_indices, Vocab,
    HIDDEN_DIM as CTC_HIDDEN, N_LSTM_LAYERS as CTC_LAYERS,
    DROPOUT as CTC_DROPOUT, PROJ_DIM as CTC_PROJ,
    DEVICE,
)


V6_CKPT_GLOB  = 'boundary_detector_v6_*.pt'
CTC_CKPT_GLOB = 'frame_ctc_v3_*.pt'

# How wide a window around the v6 segment to average CTC logits over.
# 0 = strictly within v6's segment boundaries.
# Positive frames = include +/- N frames of context (smoother, may hurt
# precision near boundaries).
LOGIT_AVG_CONTEXT_FRAMES = 0

# Whether to allow CTC to vote "blank" — if argmax is blank, fall back to
# second-best phoneme. False = always emit a real phoneme per v6 segment.
ALLOW_BLANK_FALLBACK = False


def _load_v6():
    ckpts = sorted(glob.glob(V6_CKPT_GLOB))
    assert ckpts, f"No v6 checkpoint matching {V6_CKPT_GLOB}"
    ckpt = torch.load(ckpts[-1], map_location=DEVICE)
    model = CountWordAwareDetector(
        per_patient_eeg_n_ch=ckpt['per_patient_eeg_n_ch'],
        mfcc_dim=ckpt['mfcc_dim'],
        hidden_dim=V6_HIDDEN, n_layers=V6_LAYERS, dropout=V6_DROPOUT,
    ).to(DEVICE)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f"  v6 loaded from {ckpts[-1]}")
    return model


def _load_ctc():
    ckpts = sorted(glob.glob(CTC_CKPT_GLOB))
    assert ckpts, f"No CTC checkpoint matching {CTC_CKPT_GLOB}"
    ckpt = torch.load(ckpts[-1], map_location=DEVICE)
    model = FrameCTCModel(
        per_patient_eeg_n_ch=ckpt['per_patient_eeg_n_ch'],
        vocab_size=ckpt['vocab_size'],
        hidden_dim=CTC_HIDDEN, n_layers=CTC_LAYERS,
        dropout=CTC_DROPOUT, proj_dim=CTC_PROJ,
    ).to(DEVICE)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    vocab = Vocab(set())
    vocab.itos = ckpt['vocab_itos']
    vocab.stoi = {p: i for i, p in enumerate(vocab.itos)}
    print(f"  CTC loaded from {ckpts[-1]}, vocab={len(vocab)}")
    return model, vocab


def _predict_v6_segments(v6_model, eeg_frames, pid,
                           peak_distance=4, frame_hz=200, min_seg_ms=20):
    """Use v6's boundary head + count head to produce a list of frame
    intervals (one per phoneme segment). Same logic as
    e2e_brain_decoder.predict_segments with ADAPTIVE_K_SOURCE='predicted'."""
    import scipy.signal
    with torch.no_grad():
        X = torch.from_numpy(eeg_frames.astype(np.float32)
                              ).unsqueeze(0).to(DEVICE)
        b_logits, _w_logits, count_pred = v6_model(eeg=X, mfcc=None, pid=pid)
        probs = torch.sigmoid(b_logits)[0].cpu().numpy()
        k = max(1, int(round(float(count_pred[0].item()))))

    n_frames = len(probs)
    peaks_all, _ = scipy.signal.find_peaks(probs, distance=peak_distance)
    n_boundaries = max(1, k - 1)
    if len(peaks_all) > n_boundaries:
        order = np.argsort(probs[peaks_all])[::-1][:n_boundaries]
        peaks = sorted(peaks_all[order].tolist())
    else:
        peaks = list(peaks_all)
    boundaries = sorted(set([0] + peaks + [n_frames]))
    min_seg_frames = int(min_seg_ms * frame_hz / 1000)
    segments_frames = [(boundaries[i], boundaries[i+1])
                       for i in range(len(boundaries) - 1)
                       if boundaries[i+1] - boundaries[i] >= min_seg_frames]
    return segments_frames


def _predict_ctc_per_frame(ctc_model, eeg_frames, pid):
    """Run CTC on the sentence and return (T, V) log-probabilities."""
    with torch.no_grad():
        X = torch.from_numpy(eeg_frames.astype(np.float32)
                              ).unsqueeze(0).to(DEVICE)
        logits = ctc_model(X, pid)[0]              # (T, V)
        log_probs = F.log_softmax(logits, dim=-1)
    return log_probs.cpu().numpy()


def hybrid_decode_sentence(v6_model, ctc_model, vocab, eeg_frames, pid,
                            ctx=LOGIT_AVG_CONTEXT_FRAMES,
                            allow_blank=ALLOW_BLANK_FALLBACK):
    """Hybrid decode of one sentence:
       1. v6 → segment boundary frame intervals
       2. CTC → per-frame log-probs
       3. For each segment, sum log-probs over its frames (+/- ctx context),
          take argmax (over non-blank if not allow_blank).
    Returns list of phoneme strings."""
    segments_frames = _predict_v6_segments(v6_model, eeg_frames, pid)
    if not segments_frames:
        return []
    log_probs = _predict_ctc_per_frame(ctc_model, eeg_frames, pid)
    n_frames, V = log_probs.shape

    pred_phonemes = []
    for (f_lo, f_hi) in segments_frames:
        s = max(0, f_lo - ctx)
        e = min(n_frames, f_hi + ctx)
        if e <= s:
            continue
        # Sum log-probs across the segment frames (equivalent to averaging
        # then argmax for log-softmax outputs)
        seg_logp = log_probs[s:e].sum(axis=0)
        if not allow_blank:
            seg_logp[0] = float('-inf')      # disable blank
        idx = int(seg_logp.argmax())
        if idx == 0:
            continue                          # blank fallback (shouldn't happen if !allow_blank)
        pred_phonemes.append(vocab.itos[idx])
    return pred_phonemes


# ═════════════════════════════════════════════════════════════════════════════
# Pipeline-step API
# ═════════════════════════════════════════════════════════════════════════════

def run_path_hybrid(pipeline):
    """Train-free pipeline: load v6 + CTC, evaluate on detector_test_ds.
    Stores results in pipeline.patient_results in the standard schema."""
    print("\n[1/3] Building joint dataset (corrected slicing)...")
    full_ds = build_ctc_dataset(pipeline, ALL_PIDS)
    train_ds, test_ds = split_by_sentence(full_ds)
    eeg_stats  = fit_train_stats(train_ds, 'eeg')
    mfcc_stats = fit_train_stats(train_ds, 'mfcc')
    apply_stats(train_ds, eeg_stats, mfcc_stats)
    apply_stats(test_ds,  eeg_stats, mfcc_stats)
    print(f"  test={len(test_ds)} sentences")

    print("\n[2/3] Loading v6 + CTC models...")
    v6_model = _load_v6()
    ctc_model, vocab = _load_ctc()
    attach_target_indices(test_ds, vocab)

    print("\n[3/3] Hybrid decoding (v6 boundaries + CTC frame logits)...")
    per_pid = defaultdict(lambda: {
        'true': [], 'pred': [], 'true_sids': [], 'pred_sids': [], 'n_sent': 0,
    })
    for item in test_ds:
        pid = item['pid']
        sent_idx = item['sentence_idx']
        true_seq = [vocab.itos[i] for i in item['target_idx']]
        pred_seq = hybrid_decode_sentence(
            v6_model, ctc_model, vocab, item['eeg'], pid)

        per_pid[pid]['true'].extend(true_seq)
        per_pid[pid]['pred'].extend(pred_seq)
        per_pid[pid]['true_sids'].extend([sent_idx] * len(true_seq))
        per_pid[pid]['pred_sids'].extend([sent_idx] * len(pred_seq))
        per_pid[pid]['n_sent'] += 1

    summary = {}
    for pid in sorted(per_pid):
        d = per_pid[pid]
        m = rich_metrics(d['true'], d['pred'])
        summary[pid] = {
            'true_labels':       list(d['true']),
            'predictions':       list(d['pred']),
            'true_sentence_ids': list(d['true_sids']),
            'pred_sentence_ids': list(d['pred_sids']),
            'n_sentences':       d['n_sent'],
            'accuracy':          m['correct_frac'],
            'edit_distance':     m['edit'],
            **m,
        }

    print_rich_metrics_table(summary, label='Hybrid (v6 boundaries + CTC frame logits)')
    pipeline.patient_results = summary
    return ('hybrid_v6_ctc',
            {'v6_ckpt': V6_CKPT_GLOB, 'ctc_ckpt': CTC_CKPT_GLOB,
             'context_frames': LOGIT_AVG_CONTEXT_FRAMES,
             'allow_blank_fallback': ALLOW_BLANK_FALLBACK},
            summary)
