# Converted from borders.ipynb

# ── 1. TORCH FIRST ────────────────────────────────────────────────────────────
import torch

# ── 2. STANDARD ───────────────────────────────────────────────────────────────
import os
import glob
import random
from collections import defaultdict
from datetime import datetime

# ── 3. THIRD-PARTY ────────────────────────────────────────────────────────────
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import sklearn_crfsuite

# ── 4. PROJECT ────────────────────────────────────────────────────────────────
from config import DUTCH_30_PATH
from extract_features import extractHG, stackFeatures
from dutch_30_pipeline import Dutch30Pipeline
from dutch_30_feature_extractor import Dutch30FeatureExtractor
from run_pipeline import (
    DEFAULT_RUN_CONFIG, run_path_b, load_mfa_alignments,
)
from boundary_detector_joint_audio import (
    build_joint_dataset, split_by_sentence,
    fit_train_stats, apply_stats,
    JointBoundaryDetector, train_joint,
    ALL_PIDS, HIDDEN_DIM, N_LSTM_LAYERS, DROPOUT,
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED   = 37
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

# Detector inference params
PEAK_HEIGHT     = 0.05      # lower than the F1-tuned 0.15 — we want recall
PEAK_DISTANCE   = 5         # frames; 5 × 5 ms = 25 ms minimum gap
MIN_SEGMENT_MS  = 20

# CRF
CRF_C1, CRF_C2, CRF_MAX_ITER = 0.1, 0.1, 100

# Pipeline run config (must match what created the CRF training data)
RUN_CONFIG = dict(DEFAULT_RUN_CONFIG)
RUN_CONFIG['use_viterbi']        = True
RUN_CONFIG['stacking_order']     = 20
RUN_CONFIG['stacking_step_size'] = 1

def build_pipeline():
    print("\n[1/6] Building pipeline + running Path B (MFA)...")
    extractor = Dutch30FeatureExtractor()
    pipeline = Dutch30Pipeline(
        dutch30_extractor=extractor,
        debug_mode=False,
        feature_extraction_method=RUN_CONFIG['feature_extraction_method'],
        use_wav2vec=False,
        subtract_baseline=RUN_CONFIG['subtract_baseline'],
        use_rms_boundaries=False,
        use_multifeature=False,
    )
    run_path_b(pipeline, RUN_CONFIG)
    print(f"  pipeline.train: {len(pipeline.train['features'])} phonemes, "
          f"feat dim={np.asarray(pipeline.train['features'][0]).shape}")
    print(f"  pipeline.test : {len(pipeline.test['features'])} phonemes")
    return pipeline

def features_to_crf_dict(feat):
    return {f'f{i}': float(v) for i, v in enumerate(feat)}


def train_per_patient_crfs(pipeline):
    print("\n[2/6] Training per-patient CRFs on pipeline.train...")
    by_pid = defaultdict(lambda: {'X': [], 'y': []})
    for i, p in enumerate(pipeline.train['phoneme_participant_ids']):
        by_pid[p]['X'].append(np.asarray(pipeline.train['features'][i]))
        by_pid[p]['y'].append(pipeline.train['phoneme_labels'][i])

    crfs = {}
    for pid, d in by_pid.items():
        X_seq = [[features_to_crf_dict(x) for x in d['X']]]
        y_seq = [list(d['y'])]
        crf = sklearn_crfsuite.CRF(
            algorithm='lbfgs', c1=CRF_C1, c2=CRF_C2,
            max_iterations=CRF_MAX_ITER, all_possible_transitions=True,
        )
        crf.fit(X_seq, y_seq)
        crfs[pid] = crf
    print(f"  Trained {len(crfs)} CRFs")
    return crfs

# ═════════════════════════════════════════════════════════════════════════════
# 3. MFA BASELINE — uses pipeline.test directly
# ═════════════════════════════════════════════════════════════════════════════

def evaluate_mfa_baseline(pipeline, crfs):
    print("\n[3/6] MFA baseline (pipeline.test → CRF)...")
    by_pid = defaultdict(lambda: {'X': [], 'y': []})
    for i, p in enumerate(pipeline.test['phoneme_participant_ids']):
        by_pid[p]['X'].append(np.asarray(pipeline.test['features'][i]))
        by_pid[p]['y'].append(pipeline.test['phoneme_labels'][i])

    summary = {}
    for pid, d in by_pid.items():
        if pid not in crfs: continue
        X_seq  = [[features_to_crf_dict(x) for x in d['X']]]
        y_true = list(d['y'])
        y_pred = crfs[pid].predict(X_seq)[0]
        ed = edit_distance(y_true, y_pred)
        summary[pid] = {
            'n_true':    len(y_true),
            'n_pred':    len(y_pred),
            'edit':      ed,
            'per':       ed / max(len(y_true), 1),
            'len_ratio': len(y_pred) / max(len(y_true), 1),
        }
    _print_table(summary, label='MFA baseline (pipeline.test, true upper bound)')
    return summary

# ═════════════════════════════════════════════════════════════════════════════
# 4. BOUNDARY DETECTOR — load or train
# ═════════════════════════════════════════════════════════════════════════════

def load_or_train_detector(pipeline):
    print("\n[4/6] Building joint dataset for detector...")
    full_ds = build_joint_dataset(pipeline, ALL_PIDS)
    train_ds, test_ds = split_by_sentence(full_ds)
    eeg_stats  = fit_train_stats(train_ds, 'eeg')
    mfcc_stats = fit_train_stats(train_ds, 'mfcc')
    train_ds   = apply_stats(train_ds, eeg_stats, mfcc_stats)
    test_ds    = apply_stats(test_ds,  eeg_stats, mfcc_stats)
    print(f"  detector dataset: train={len(train_ds)}  test={len(test_ds)}")

    ckpts = sorted(glob.glob('boundary_detector_joint_*.pt'))
    if ckpts:
        ckpt_path = ckpts[-1]
        print(f"  Loading {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        model = JointBoundaryDetector(
            per_patient_eeg_n_ch=ckpt['per_patient_eeg_n_ch'],
            mfcc_dim=ckpt['mfcc_dim'],
            hidden_dim=HIDDEN_DIM, n_layers=N_LSTM_LAYERS, dropout=DROPOUT,
        ).to(DEVICE)
        model.load_state_dict(ckpt['model_state'])
        model.eval()
        print(f"  ✓ loaded ({sum(p.numel() for p in model.parameters())/1e6:.2f}M params)")
    else:
        print("  No checkpoint — training from scratch (~5–10 min on GPU)")
        model = train_joint(train_ds)
        out_path = f'boundary_detector_joint_{datetime.now().strftime("%Y%m%d_%H%M")}.pt'
        torch.save({
            'model_state':         model.state_dict(),
            'per_patient_eeg_n_ch': {pid: model.eeg_proj[pid].in_features
                                      for pid in model.eeg_proj},
            'mfcc_dim':            model.mfcc_proj.in_features,
        }, out_path)
        print(f"  ✓ saved to {out_path}")

    return model, test_ds


# ═════════════════════════════════════════════════════════════════════════════
# 5. DETECTOR PATH — uses pipeline's extractHG + stackFeatures + channel mask
# ═════════════════════════════════════════════════════════════════════════════

def get_pipeline_channel_mask(pipeline, pid):
    """Return the channel index array the pipeline used for this patient
    (the SAME mask that produced pipeline.train['features'])."""
    if hasattr(pipeline, 'patient_data') and pid in pipeline.patient_data:
        pdata = pipeline.patient_data[pid]
        if 'channel_mask' in pdata:
            cm = pdata['channel_mask']
            return np.where(cm)[0] if cm.dtype == bool else cm
        if 'included_channels' in pdata:
            return np.asarray(pdata['included_channels'])
    return None  # no exclusion → use all channels


def extract_phoneme_feature_pipeline_native(sentence_eeg, start_s, end_s,
                                             eeg_sr, win_len, frameshift,
                                             stacking_order):
    """Mirror exactly what run_path_b + step5b/c do for ONE phoneme.

    Returns a single (n_features,) vector matching CRF input dim.
    """
    ph_start = int(start_s * eeg_sr)
    ph_end   = int(end_s   * eeg_sr)
    ph_start = max(0, min(ph_start, sentence_eeg.shape[0] - 1))
    ph_end   = max(ph_start + 1, min(ph_end, sentence_eeg.shape[0]))
    eeg_seg  = sentence_eeg[ph_start:ph_end]

    # Zero-pad short segments. extractHG's internal sosfiltfilt needs
    # ≥28 samples; we also want enough frames for the stacking step. Pad to
    # whichever is larger.
    min_samples = max(int(win_len * eeg_sr) + 1, 64)
    if len(eeg_seg) < min_samples:
        eeg_seg = np.pad(eeg_seg,
                         ((0, min_samples - len(eeg_seg)), (0, 0)),
                         mode='constant')

    try:
        feat = extractHG(eeg_seg, eeg_sr,
                         windowLength=win_len, frameshift=frameshift)
    except ValueError:
        return None
    if feat is None or feat.shape[0] == 0:
        return None

    # Step 5b: stackFeatures requires at least 2*so+1 frames; zero-pad if not
    n_needed = 2 * stacking_order + 1
    if feat.shape[0] < n_needed:
        feat = np.pad(feat,
                      ((0, n_needed - feat.shape[0]), (0, 0)),
                      mode='constant')

    stacked = stackFeatures(feat, modelOrder=stacking_order, stepSize=1)
    if stacked.shape[0] == 0:
        return None

    # Step 5c: collapse to one vector per phoneme via mean
    return stacked.mean(axis=0)


def predict_segments(model, eeg_frames_for_detector, pid):
    """Run detector → list of (start_s, end_s) phoneme intervals."""
    model.eval()
    with torch.no_grad():
        X = torch.from_numpy(eeg_frames_for_detector.astype(np.float32)
                              ).unsqueeze(0).to(DEVICE)
        logits = model(eeg=X, mfcc=None, pid=pid)
        probs  = torch.sigmoid(logits)[0].cpu().numpy()
    peaks, _ = scipy.signal.find_peaks(probs, height=PEAK_HEIGHT,
                                        distance=PEAK_DISTANCE)
    n_frames = len(probs)
    frame_hz = 200    # detector is at 5 ms shift → 200 Hz
    boundaries = sorted(set([0] + list(peaks) + [n_frames]))
    min_seg_frames = int(MIN_SEGMENT_MS * frame_hz / 1000)
    return [(boundaries[i] / frame_hz, boundaries[i+1] / frame_hz)
            for i in range(len(boundaries) - 1)
            if boundaries[i+1] - boundaries[i] >= min_seg_frames]


def evaluate_detector_path(pipeline, crfs, model, detector_test_ds):
    """For each test sentence: detect boundaries, extract features
    pipeline-native, classify, compare to MFA labels."""
    print("\n[5/6] Detector path (iEEG → boundaries → features → CRF)...")

    eeg_sr     = pipeline.config.eeg_sr
    win_len    = pipeline.config.window_length
    frameshift = pipeline.config.frameshift
    stk_order  = RUN_CONFIG['stacking_order']

    # Cache full-channel raw EEG per patient (slow to load)
    raw_eeg_cache  = {}
    chan_mask_cache = {pid: get_pipeline_channel_mask(pipeline, pid)
                       for pid in ALL_PIDS}

    per_pid = defaultdict(lambda: {'true': [], 'pred': [], 'n_sent': 0})
    n_skipped_segments = 0

    for item in detector_test_ds:
        pid      = item['pid']
        sent_idx = item['sentence_idx']
        if pid not in crfs:
            continue

        # Load raw EEG once per patient, apply pipeline channel mask
        if pid not in raw_eeg_cache:
            raw = np.load(os.path.join(DUTCH_30_PATH, 'raw', f'{pid}_sEEG.npy'))
            mask = chan_mask_cache[pid]
            raw_eeg_cache[pid] = raw[:, mask] if mask is not None else raw

        # Get sentence-level EEG slice (pipeline-native channels)
        wd        = pipeline.split_result['word_segments_dict'][pid]
        sent_info = wd['sentence_list'][sent_idx]
        eeg_start = int(sent_info['stim_start_idx'])
        eeg_end   = int(sent_info['stim_end_idx'])
        sentence_eeg = raw_eeg_cache[pid][eeg_start:eeg_end]

        # 5a. Detect boundaries from iEEG (uses detector-trained channels:
        #     item['eeg'] was already z-scored with detector channel mask)
        segments = predict_segments(model, item['eeg'], pid)
        if not segments:
            continue

        # 5b. Extract one feature vector per detected segment, pipeline-native
        seg_feats = []
        for (s_s, e_s) in segments:
            f = extract_phoneme_feature_pipeline_native(
                sentence_eeg, s_s, e_s, eeg_sr, win_len, frameshift, stk_order)
            if f is None:
                n_skipped_segments += 1
                continue
            seg_feats.append(f)
        if not seg_feats:
            continue

        # 5c. CRF inference
        X_seq = [[features_to_crf_dict(x) for x in seg_feats]]
        y_pred = crfs[pid].predict(X_seq)[0]

        # 5d. True labels = MFA phonemes for this sentence
        mfa = load_mfa_alignments(pid).get(sent_idx, [])
        y_true = [p['phone'] for p in mfa]

        per_pid[pid]['true'].extend(y_true)
        per_pid[pid]['pred'].extend(list(y_pred))
        per_pid[pid]['n_sent'] += 1

    summary = {}
    for pid, d in per_pid.items():
        ed = edit_distance(d['true'], d['pred'])
        summary[pid] = {
            'n_sentences': d['n_sent'],
            'n_true':      len(d['true']),
            'n_pred':      len(d['pred']),
            'edit':        ed,
            'per':         ed / max(len(d['true']), 1),
            'len_ratio':   len(d['pred']) / max(len(d['true']), 1),
        }
    print(f"  Skipped {n_skipped_segments} degenerate segments "
          f"(extractHG returned None)")
    _print_table(summary, label='DETECTED boundaries (iEEG-only path)')
    return summary



# ═════════════════════════════════════════════════════════════════════════════
# 6. COMPARISON
# ═════════════════════════════════════════════════════════════════════════════

def print_comparison(mfa, det):
    print("\n[6/6] " + "="*72)
    print("  Comparison: MFA boundaries (true upper bound) vs detected boundaries")
    print("="*78)
    print(f"  {'pid':<5} {'PER MFA':>9} {'PER det':>9} {'Δ PER':>8}  "
          f"{'len ratio':>10}")
    print("  " + "-" * 60)
    pids = sorted(set(mfa) & set(det))
    for pid in pids:
        m = mfa[pid]['per']; d = det[pid]['per']; lr = det[pid]['len_ratio']
        print(f"  {pid:<5} {m:>8.2%}  {d:>8.2%} {(d-m):>+7.2%}  {lr:>9.2f}×")
    print("  " + "-" * 60)
    print(f"  Mean MFA PER: {np.mean([mfa[p]['per'] for p in pids]):.2%}")
    print(f"  Mean det PER: {np.mean([det[p]['per'] for p in pids]):.2%}")
    print(f"  Mean Δ:       {np.mean([det[p]['per']-mfa[p]['per'] for p in pids]):+.2%}")
    print("="*78)

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 5 — Run boundary detector on a single test sentence
# ═══════════════════════════════════════════════════════════════════════════════

def predict_segments_for_sentence(detector_model, eeg_frames, pid,
                                   peak_height=PEAK_HEIGHT,
                                   peak_distance=PEAK_DISTANCE):
    """Returns segments_in_seconds (list of (start, end) tuples)."""
    detector_model.eval()
    with torch.no_grad():
        X = torch.from_numpy(eeg_frames.astype(np.float32)
                              ).unsqueeze(0).to(DEVICE)
        # Try iEEG-only path (joint detector or iEEG-only detector both work)
        try:
            logits = detector_model(eeg=X, mfcc=None, pid=pid)
        except TypeError:
            # iEEG-only detector signature
            logits = detector_model(X, [pid])
        probs = torch.sigmoid(logits)[0].cpu().numpy()
    peaks, _ = scipy.signal.find_peaks(probs, height=peak_height,
                                        distance=peak_distance)
    return peaks_to_segments(peaks, len(probs))


# ═════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ═════════════════════════════════════════════════════════════════════════════
def edit_distance(s1, s2):
    # Coerce to plain Python lists of hashable scalars
    s1 = [str(x) for x in s1]
    s2 = [str(x) for x in s2]
    if len(s1) < len(s2): return edit_distance(s2, s1)
    if len(s2) == 0: return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(prev[j+1]+1, curr[j]+1, prev[j]+(c1 != c2)))
        prev = curr
    return prev[-1]


def _print_table(summary, label=''):
    print(f"\n  {label}")
    print(f"  {'pid':<5} {'true_n':>7} {'pred_n':>7} {'edit':>6} "
          f"{'PER':>8} {'len_ratio':>10}")
    print("  " + "-" * 55)
    for pid in sorted(summary):
        s = summary[pid]
        print(f"  {pid:<5} {s['n_true']:>7} {s['n_pred']:>7} "
              f"{s['edit']:>6} {s['per']:>7.2%} {s['len_ratio']:>9.2f}×")
    print("  " + "-" * 55)
    print(f"  Mean PER: {np.mean([s['per'] for s in summary.values()]):.2%}")

# ═════════════════════════════════════════════════════════════════════════════
# MAIN — single linear flow
# ═════════════════════════════════════════════════════════════════════════════

def main():
    pipeline               = build_pipeline()
    crfs                   = train_per_patient_crfs(pipeline)
    mfa_summary            = evaluate_mfa_baseline(pipeline, crfs)
    model, detector_test_ds = load_or_train_detector(pipeline)
    det_summary            = evaluate_detector_path(pipeline, crfs, model,
                                                     detector_test_ds)
    print_comparison(mfa_summary, det_summary)
    return {
        'pipeline':    pipeline,
        'crfs':        crfs,
        'model':       model,
        'mfa_summary': mfa_summary,
        'det_summary': det_summary,
    }


if __name__ == '__main__':
    results = main()
