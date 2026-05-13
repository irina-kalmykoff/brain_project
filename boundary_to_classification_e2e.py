# End-to-end test: detected boundaries → CRF → phoneme labels.
#
# Stage 2 of the boundary-detection pipeline. Takes the trained boundary
# detector from boundary_detector_joint_audio.py (or boundary_detector_train.py),
# uses its boundary predictions on test sentences to extract per-phoneme
# features, runs them through a CRF trained on MFA-aligned features, and
# compares end-to-end accuracy to the MFA-baseline.
#
# Compares for each test patient on the SAME held-out sentences:
#   1. MFA-baseline: use real MFA boundaries → features → CRF → labels
#   2. Detector-based: use detected boundaries → features → CRF → labels
#
# Reports: phoneme accuracy, edit distance, PER (lengths can differ now).
# Pure deterministic logic.
#
# Requires:
#   - `pipeline` loaded with run_path_b done
#   - `model` (trained boundary detector) in scope, OR a saved .pt file
#   - `train_ds`, `test_ds` from boundary_detector_joint_audio.py
#   - All pipeline.split_result and MFA TextGrids accessible

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 1 — Imports + config
# ═══════════════════════════════════════════════════════════════════════════════

import os
import pickle
import random
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
from collections import defaultdict
from datetime import datetime
from math import gcd

import torch
import sklearn_crfsuite

from config import DUTCH_30_PATH
from run_pipeline import load_mfa_alignments

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"  Using device: {DEVICE}")

SEED = 37
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

# Production extractHG settings (must match what the CRF was trained with)
SR_EEG         = 1024
WINDOW_MS_EEG  = 15
FRAMESHIFT_MS  = 5
SMOOTHING_HZ   = 10.0
STK_ORDER      = 20             # ±100 ms context at 5 ms shift
FRAME_HZ       = 1000 // FRAMESHIFT_MS    # 200

# Boundary detection inference settings
PEAK_HEIGHT    = 0.15
PEAK_DISTANCE  = 8              # frames; 8 × 5 ms = 40 ms minimum phoneme

# Filter: drop segments shorter than this (likely false-positive boundaries)
MIN_SEGMENT_MS = 30


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 2 — Helper: production-matching per-frame extractor (pwr_lpf_10)
# ═══════════════════════════════════════════════════════════════════════════════

def extract_eeg_frames(eeg, sr=SR_EEG, window_ms=WINDOW_MS_EEG,
                       frameshift_ms=FRAMESHIFT_MS, smoothing_hz=SMOOTHING_HZ):
    """Same as production extractHG (pwr_lpf_10) — frame-level output."""
    win = window_ms / 1000.0
    shift = frameshift_ms / 1000.0
    data = scipy.signal.detrend(eeg, axis=0)
    sos_hg = scipy.signal.iirfilter(4, [70/(sr/2), 170/(sr/2)],
                                    btype='bandpass', output='sos')
    data = scipy.signal.sosfiltfilt(sos_hg, data, axis=0)
    for f_notch in (100.0, 150.0):
        sos_n = scipy.signal.iirfilter(4, [(f_notch-2)/(sr/2), (f_notch+2)/(sr/2)],
                                       btype='bandstop', output='sos')
        data = scipy.signal.sosfiltfilt(sos_n, data, axis=0)
    pwr = data ** 2
    sos_lp = scipy.signal.iirfilter(4, smoothing_hz/(sr/2),
                                    btype='lowpass', output='sos')
    smoothed = np.abs(scipy.signal.sosfiltfilt(sos_lp, pwr, axis=0))
    n_win = int(np.floor((data.shape[0] - win*sr) / (shift*sr)))
    feat = np.zeros((n_win, data.shape[1]), dtype=np.float32)
    for w in range(n_win):
        s = int(np.floor(w * shift * sr))
        e = int(np.floor(s + win * sr))
        feat[w, :] = smoothed[s:e, :].mean(axis=0)
    return np.sqrt(feat).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 3 — Stacking + segment-level collapse (matches production step5b/c)
# ═══════════════════════════════════════════════════════════════════════════════

def stack_with_context(frame_features, stk_order=STK_ORDER):
    """For each frame, concatenate ±stk_order neighbours along channel axis.

    Edge frames get zero-padded for missing context. Returns
    (n_frames, (2*stk_order+1) * n_channels).
    """
    n_frames, n_ch = frame_features.shape
    out = np.zeros((n_frames, (2 * stk_order + 1) * n_ch), dtype=np.float32)
    for k, offset in enumerate(range(-stk_order, stk_order + 1)):
        # Slice into out[:, k*n_ch:(k+1)*n_ch]
        if offset == 0:
            out[:, k*n_ch:(k+1)*n_ch] = frame_features
        elif offset > 0:
            out[:n_frames - offset, k*n_ch:(k+1)*n_ch] = frame_features[offset:]
        else:  # offset < 0
            out[-offset:, k*n_ch:(k+1)*n_ch] = frame_features[:n_frames + offset]
    return out


def collapse_to_segments(stacked_features, segments_seconds,
                          frame_hz=FRAME_HZ):
    """Average stacked frames per segment.

    segments_seconds: list of (start_s, end_s) phoneme intervals.
    Returns: (n_segments, n_features) per-phoneme feature matrix.
    """
    n_frames = stacked_features.shape[0]
    out = []
    for (start_s, end_s) in segments_seconds:
        s = max(0, int(round(start_s * frame_hz)))
        e = min(n_frames, int(round(end_s * frame_hz)))
        if e <= s:
            e = min(n_frames, s + 1)
        if s >= n_frames:
            # Degenerate segment past sentence end — pad with last frame
            out.append(stacked_features[-1])
            continue
        out.append(stacked_features[s:e].mean(axis=0))
    return np.array(out, dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 4 — Convert detected peak frames → phoneme segments
# ═══════════════════════════════════════════════════════════════════════════════
# A phoneme spans the interval BETWEEN two consecutive boundary peaks.
# We add implicit boundaries at frame 0 and at the last frame.

def peaks_to_segments(peak_frames, n_frames, frame_hz=FRAME_HZ,
                      min_segment_ms=MIN_SEGMENT_MS):
    """Returns list of (start_s, end_s) phoneme intervals."""
    boundaries = sorted(set([0] + list(peak_frames) + [n_frames]))
    # Filter out segments shorter than min_segment_ms (likely false positives)
    min_segment_frames = int(min_segment_ms * frame_hz / 1000)
    segments = []
    for i in range(len(boundaries) - 1):
        if boundaries[i+1] - boundaries[i] >= min_segment_frames:
            segments.append((boundaries[i] / frame_hz,
                              boundaries[i+1] / frame_hz))
    return segments


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


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 6 — Build per-sentence raw-data records (re-uses our earlier setup)
# ═══════════════════════════════════════════════════════════════════════════════
# Each record has the raw EEG slice, the MFA segments (true), and the
# extractHG frame features.

def build_e2e_test_records(pipeline, test_dataset_items):
    """For each item in test_dataset_items (from boundary_detector_joint_audio's
    test_ds), build a record with everything needed for end-to-end evaluation.

    Returns list of:
      {pid, sentence_idx,
       eeg_raw,            # raw EEG samples for the sentence
       eeg_frames,         # per-frame extractHG output
       mfa_segments,       # list of (start_s, end_s) from MFA
       mfa_labels,         # list of phoneme labels (one per MFA segment)
       boundary_times}     # for sanity check
    """
    records = []
    audio_sr_raw = 48000
    try:
        audio_sr_raw = int(pipeline.config.audio_sr)
    except (AttributeError, TypeError):
        pass

    # Cache raw EEG per patient (it's big)
    raw_eeg_cache = {}

    for item in test_dataset_items:
        pid = item['pid']
        sent_idx = item['sentence_idx']

        # Load raw EEG once per patient
        if pid not in raw_eeg_cache:
            raw_eeg_path = os.path.join(DUTCH_30_PATH, 'raw', f'{pid}_sEEG.npy')
            raw_eeg_cache[pid] = np.load(raw_eeg_path)
        raw_eeg = raw_eeg_cache[pid]

        # Apply same channel exclusion the detector training used
        wd = pipeline.split_result['word_segments_dict'][pid]
        sentence_list = wd['sentence_list']
        sent_info = sentence_list[sent_idx]
        eeg_start = int(sent_info['stim_start_idx'])
        eeg_end   = int(sent_info['stim_end_idx'])

        # Channel exclusion (if pipeline has channel_masks)
        if hasattr(pipeline, 'channel_masks') and pid in pipeline.channel_masks:
            cm = pipeline.channel_masks[pid]
            if isinstance(cm, dict) and 'keep_indices' in cm:
                keep_idx = np.asarray(cm['keep_indices'])
                eeg_for_sent = raw_eeg[eeg_start:eeg_end, keep_idx]
            elif isinstance(cm, np.ndarray) and cm.dtype == bool:
                keep_idx = np.where(cm)[0]
                eeg_for_sent = raw_eeg[eeg_start:eeg_end, keep_idx]
            else:
                eeg_for_sent = raw_eeg[eeg_start:eeg_end]
        else:
            eeg_for_sent = raw_eeg[eeg_start:eeg_end]

        # MFA segments + labels for this sentence
        mfa = load_mfa_alignments(pid).get(sent_idx, [])
        mfa_segments = [(p['start_s'], p['end_s']) for p in mfa]
        mfa_labels   = [p['phone'] for p in mfa]

        records.append({
            'pid':            pid,
            'sentence_idx':   sent_idx,
            'eeg_raw':        eeg_for_sent,
            'eeg_frames':     item['eeg'],   # already-extracted frames
            'mfa_segments':   mfa_segments,
            'mfa_labels':     mfa_labels,
            'boundary_times': item['boundary_times'],
        })

    print(f"  Built {len(records)} end-to-end test records "
          f"from {len(set(r['pid'] for r in records))} patients")
    return records


_ns = globals()
_have_pipeline = 'pipeline' in _ns

# Find the test dataset under common names
_test_ds = None
for _name in ('test_ds', 'test_dataset', 'test_records', 'val_ds'):
    if _name in _ns:
        _test_ds = _ns[_name]
        print(f"  Found test dataset under name: '{_name}'  (n={len(_test_ds)})")
        break

# Find the detector model — try common names
_detector = None
for _name in ('model', 'detector_model', 'joint_model', 'boundary_model'):
    if _name in _ns:
        _detector = _ns[_name]
        print(f"  Found detector model under name: '{_name}'")
        break

print(f"\n  Namespace check:")
print(f"    pipeline      : {'OK' if _have_pipeline else 'MISSING'}")
print(f"    test dataset  : {'OK' if _test_ds is not None else 'MISSING (looked for test_ds/test_dataset/test_records/val_ds)'}")
print(f"    detector model: {'OK' if _detector is not None else 'MISSING (looked for model/detector_model/joint_model/boundary_model)'}")

if _have_pipeline and _test_ds is not None:
    print("\n  Building end-to-end test records...")
    e2e_records = build_e2e_test_records(pipeline, _test_ds)
else:
    print("  >>> Skipping CELL 6 — need pipeline + test dataset in scope. <<<")
    print("      If your test list lives under another name, add it manually:")
    print("        e2e_records = build_e2e_test_records(pipeline, <your_test_list>)")


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 7 — Train a CRF on the production training set (MFA-aligned features)
# ═══════════════════════════════════════════════════════════════════════════════
# Uses pipeline.train['features'] and ['phoneme_labels'] directly, grouped by
# patient (one sequence per patient).

def features_to_crf_dict(feat):
    return {f'f{i}': float(v) for i, v in enumerate(feat)}


def train_per_patient_crf(pipeline, c1=0.1, c2=0.1, max_iter=100):
    """Train one CRF per patient on its MFA-aligned training features."""
    from collections import defaultdict
    train_by_pid = defaultdict(lambda: {'X': [], 'y': []})
    for i, p in enumerate(pipeline.train['phoneme_participant_ids']):
        train_by_pid[p]['X'].append(np.asarray(pipeline.train['features'][i]))
        train_by_pid[p]['y'].append(pipeline.train['phoneme_labels'][i])

    crfs = {}
    for pid, data in train_by_pid.items():
        X_seq = [[features_to_crf_dict(x) for x in data['X']]]
        y_seq = [list(data['y'])]
        crf = sklearn_crfsuite.CRF(
            algorithm='lbfgs', c1=c1, c2=c2, max_iterations=max_iter,
            all_possible_transitions=True,
        )
        crf.fit(X_seq, y_seq)
        crfs[pid] = crf
    return crfs


if 'pipeline' in dir():
    print("\n  Training per-patient CRFs on MFA-aligned features...")
    crfs = train_per_patient_crf(pipeline)
    print(f"  Trained {len(crfs)} per-patient CRFs")


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 8 — End-to-end pipeline: detected boundaries → CRF predictions
# ═══════════════════════════════════════════════════════════════════════════════

def run_e2e_one_sentence(detector_model, crf, record, use_detector=True):
    """Run the full pipeline on one test sentence.

    If use_detector=False, uses MFA segments instead (baseline comparison).

    Returns dict with detected segments, predicted labels, true labels.
    """
    pid          = record['pid']
    eeg_raw      = record['eeg_raw']
    eeg_frames   = record['eeg_frames']
    mfa_segments = record['mfa_segments']
    mfa_labels   = record['mfa_labels']

    # Choose segments
    if use_detector:
        segments = predict_segments_for_sentence(detector_model, eeg_frames, pid)
    else:
        segments = mfa_segments

    if not segments:
        return {
            'pid':          pid,
            'segments':     [],
            'predictions':  [],
            'true_labels':  mfa_labels,
        }

    # Extract per-segment features (production-matching: per-frame extractHG
    # over the full sentence, then stack with ±20 context, then collapse per
    # segment)
    full_frames = extract_eeg_frames(eeg_raw)
    stacked     = stack_with_context(full_frames)
    segment_feats = collapse_to_segments(stacked, segments)

    # CRF inference
    X_seq = [[features_to_crf_dict(x) for x in segment_feats]]
    predictions = crf.predict(X_seq)[0]

    return {
        'pid':          pid,
        'segments':     segments,
        'predictions':  list(predictions),
        'true_labels':  mfa_labels,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 9 — Evaluation metrics: edit distance + per-patient summary
# ═══════════════════════════════════════════════════════════════════════════════

def edit_distance(s1, s2):
    """Levenshtein."""
    if len(s1) < len(s2):
        return edit_distance(s2, s1)
    if not s2:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(
                prev[j + 1] + 1,
                curr[j] + 1,
                prev[j] + (c1 != c2),
            ))
        prev = curr
    return prev[-1]


def evaluate_e2e(detector_model, crfs, records, use_detector=True, label=''):
    """Run end-to-end on all test records, return per-patient metrics."""
    per_patient = defaultdict(lambda: {
        'true_seq':   [],
        'pred_seq':   [],
        'n_sentences': 0,
        'n_pred_segs': 0,
        'n_true_segs': 0,
    })

    for rec in records:
        pid = rec['pid']
        if pid not in crfs:
            continue
        out = run_e2e_one_sentence(detector_model, crfs[pid], rec,
                                    use_detector=use_detector)
        per_patient[pid]['true_seq'].extend(out['true_labels'])
        per_patient[pid]['pred_seq'].extend(out['predictions'])
        per_patient[pid]['n_sentences']  += 1
        per_patient[pid]['n_pred_segs']  += len(out['predictions'])
        per_patient[pid]['n_true_segs']  += len(out['true_labels'])

    # Compute metrics
    print(f"\n  {label}  ({'detector boundaries' if use_detector else 'MFA boundaries'}):")
    print(f"  {'pid':<5} {'n_sent':>7} {'true_n':>7} {'pred_n':>7} "
          f"{'edit':>6} {'PER':>8} {'len_ratio':>10}")
    print("  " + "-" * 60)
    summary = {}
    for pid in sorted(per_patient):
        d = per_patient[pid]
        true = d['true_seq']
        pred = d['pred_seq']
        ed = edit_distance(true, pred)
        per = ed / max(len(true), 1)
        len_ratio = len(pred) / max(len(true), 1)
        summary[pid] = {
            'n_sentences': d['n_sentences'],
            'n_true':      len(true),
            'n_pred':      len(pred),
            'edit':        ed,
            'per':         per,
            'len_ratio':   len_ratio,
        }
        print(f"  {pid:<5} {d['n_sentences']:>7} {len(true):>7} {len(pred):>7} "
              f"{ed:>6} {per:>7.2%} {len_ratio:>9.2f}×")

    print("  " + "-" * 60)
    mean_per = np.mean([s['per'] for s in summary.values()])
    mean_len_ratio = np.mean([s['len_ratio'] for s in summary.values()])
    total_edit = sum(s['edit'] for s in summary.values())
    total_true = sum(s['n_true']  for s in summary.values())
    overall_per = total_edit / max(total_true, 1)
    print(f"  Mean PER (per-patient avg): {mean_per:.2%}")
    print(f"  Overall PER (concatenated): {overall_per:.2%}")
    print(f"  Mean length ratio (pred/true): {mean_len_ratio:.2f}×")
    return summary


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 10 — Run the comparison: MFA baseline vs detector boundaries
# ═══════════════════════════════════════════════════════════════════════════════

# Re-resolve in case CELL 6 was skipped or names changed
_ns = globals()
_detector = None
for _name in ('model', 'detector_model', 'joint_model', 'boundary_model'):
    if _name in _ns:
        _detector = _ns[_name]; break
_have_records = 'e2e_records' in _ns
_have_crfs    = 'crfs' in _ns

print(f"\n  CELL 10 namespace check:")
print(f"    e2e_records   : {'OK (' + str(len(e2e_records)) + ' records)' if _have_records else 'MISSING'}")
print(f"    crfs          : {'OK (' + str(len(crfs)) + ' patients)' if _have_crfs else 'MISSING'}")
print(f"    detector      : {'OK' if _detector is not None else 'MISSING'}")

if _have_records and _have_crfs and _detector is not None:
    # 1. MFA baseline (use real boundaries) — sanity check
    mfa_summary = evaluate_e2e(_detector, crfs, e2e_records,
                                use_detector=False,
                                label='MFA-baseline (sanity check)')

    # 2. Detector-based (use predicted boundaries) — the real test
    det_summary = evaluate_e2e(_detector, crfs, e2e_records,
                                use_detector=True,
                                label='DETECTED boundaries (the real test)')

    # 3. Side-by-side comparison
    print("\n" + "="*78)
    print("  Comparison: MFA boundaries vs detector boundaries")
    print("="*78)
    print(f"  {'pid':<5} {'PER (MFA)':>10} {'PER (detected)':>16} {'Δ PER':>8} "
          f"{'len ratio (det)':>16}")
    print("  " + "-" * 60)
    for pid in sorted(mfa_summary):
        m_per = mfa_summary[pid]['per']
        d_per = det_summary[pid]['per']
        d_lr  = det_summary[pid]['len_ratio']
        print(f"  {pid:<5} {m_per:>9.2%}  {d_per:>15.2%} "
              f"{(d_per-m_per):>+7.2%} {d_lr:>15.2f}×")
    print("  " + "-" * 60)
    print(f"  Mean MFA PER:      {np.mean([s['per'] for s in mfa_summary.values()]):.2%}")
    print(f"  Mean detected PER: {np.mean([s['per'] for s in det_summary.values()]):.2%}")
    print(f"  Mean Δ:            {np.mean([det_summary[p]['per'] - mfa_summary[p]['per'] for p in mfa_summary]):+.2%}")
    print("="*78)
else:
    print("  >>> Skipping CELL 10 — missing one of e2e_records / crfs / detector. <<<")


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 11 — Visualize one sentence: detected vs MFA boundaries
# ═══════════════════════════════════════════════════════════════════════════════

def visualize_e2e_sentence(detector_model, crf, record):
    """Plot detected vs MFA boundaries with predicted/true labels."""
    out_det = run_e2e_one_sentence(detector_model, crf, record, use_detector=True)
    out_mfa = run_e2e_one_sentence(detector_model, crf, record, use_detector=False)

    fig, axes = plt.subplots(2, 1, figsize=(14, 5), sharex=True)

    # Top: detector path
    ax = axes[0]
    for (s, e), lbl in zip(out_det['segments'], out_det['predictions']):
        ax.axvspan(s, e, alpha=0.2, color='steelblue')
        ax.text((s+e)/2, 0.5, lbl, ha='center', va='center', fontsize=8)
    ax.set_title(f"DETECTED ({len(out_det['predictions'])} segments) — "
                 f"{record['pid']} sent {record['sentence_idx']}")
    ax.set_yticks([])

    # Bottom: MFA path
    ax = axes[1]
    for (s, e), lbl in zip(out_mfa['segments'], out_mfa['predictions']):
        ax.axvspan(s, e, alpha=0.2, color='seagreen')
        ax.text((s+e)/2, 0.5, lbl, ha='center', va='center', fontsize=8)
    ax.set_title(f"MFA ({len(out_mfa['predictions'])} segments)")
    ax.set_xlabel('Time (s)')
    ax.set_yticks([])

    plt.tight_layout(); plt.show()

    # Print true labels for reference
    print(f"\n  True labels: {' '.join(record['mfa_labels'])}")
    print(f"  Det pred:    {' '.join(out_det['predictions'])}")
    print(f"  MFA pred:    {' '.join(out_mfa['predictions'])}")


_ns = globals()
_detector = None
for _name in ('model', 'detector_model', 'joint_model', 'boundary_model'):
    if _name in _ns:
        _detector = _ns[_name]; break
if 'e2e_records' in _ns and 'crfs' in _ns and _detector is not None and e2e_records:
    visualize_e2e_sentence(_detector, crfs[e2e_records[0]['pid']], e2e_records[0])
else:
    print("  >>> Skipping CELL 11 viz — missing e2e_records / crfs / detector. <<<")


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 12 — Save results
# ═══════════════════════════════════════════════════════════════════════════════

if 'mfa_summary' in dir() and 'det_summary' in dir():
    out_path = f'e2e_classification_{datetime.now().strftime("%Y%m%d_%H%M")}.pkl'
    with open(out_path, 'wb') as f:
        pickle.dump({
            'mfa_summary': mfa_summary,
            'det_summary': det_summary,
            'config': {
                'peak_height':    PEAK_HEIGHT,
                'peak_distance':  PEAK_DISTANCE,
                'min_segment_ms': MIN_SEGMENT_MS,
                'stk_order':      STK_ORDER,
            },
        }, f)
    print(f"\n  Saved end-to-end results to {out_path}")
