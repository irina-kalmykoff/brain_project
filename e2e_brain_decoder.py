# End-to-end brain-only phoneme decoding.
#
# Single optimized flow:
#   1. Build pipeline + run_path_b (MFA-aligned features for train/test)
#   2. Train per-patient CRFs on pipeline.train (the production training data)
#   3. MFA baseline: feed pipeline.test directly to CRFs (true upper bound)
#   4. Load (or train+save) the joint boundary detector
#   5. Detector path: detect phoneme boundaries from iEEG, extract features
#      USING THE SAME PIPELINE FUNCTIONS, feed to CRFs
#   6. Side-by-side comparison
#
# Key invariants:
#   - CRF training inputs come from pipeline.train (no re-extraction)
#   - MFA baseline uses pipeline.test verbatim
#   - Detector-path features use pipeline's extractHG + stackFeatures with
#     the SAME channel mask, window, frameshift, and stacking_order. This
#     guarantees the only difference between MFA baseline and detector path
#     is the segmentation source.

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
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ── 4. PROJECT ────────────────────────────────────────────────────────────────
from config import DUTCH_30_PATH
from extract_features import extractHG, stackFeatures
from dutch_30_pipeline import Dutch30Pipeline
from dutch_30_feature_extractor import Dutch30FeatureExtractor
from run_pipeline import (
    DEFAULT_RUN_CONFIG, run_path_b, load_mfa_alignments,
)
from boundary_detector_joint_audio import (
    split_by_sentence,
    fit_train_stats, apply_stats,
    JointBoundaryDetector,
    ALL_PIDS, HIDDEN_DIM, N_LSTM_LAYERS, DROPOUT,
    SR_EEG, SR_AUDIO_TARGET, FRAME_HZ,
    extract_eeg_frames, compute_mfcc, add_delta_features,
    boundary_labels_from_mfa,
    MOD_DROPOUT, BATCH_SIZE, N_EPOCHS, LR, WEIGHT_DECAY, POS_WEIGHT,
    collate_padded_joint, sample_modality,
)
import torch.nn as nn
import torch.nn.functional as F


# ═════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═════════════════════════════════════════════════════════════════════════════

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED   = 37
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

# Detector inference params
PEAK_HEIGHT      = 0.05     # fixed-threshold mode (used only if adaptive=False)
PEAK_DISTANCE    = 4        # frames; 4 × 5 ms = 20 ms minimum gap
MIN_SEGMENT_MS   = 20

TARGET_PHONEME_RATE  = 11.0

# CRF (mirrors _run_crf_experiment in run_pipeline.py exactly)
N_PCA             = 50
CRF_C1, CRF_C2    = 0.1, 0.1
CRF_MAX_ITER      = 100
MIN_CLASS_SAMPLES = 5     # _run_crf_experiment default
RANDOM_STATE      = 37

# Pipeline run config (must match what created the CRF training data)
RUN_CONFIG = dict(DEFAULT_RUN_CONFIG)
RUN_CONFIG['use_viterbi']        = True
RUN_CONFIG['stacking_order']     = 20
RUN_CONFIG['stacking_step_size'] = 1


# ═════════════════════════════════════════════════════════════════════════════
# 1. PIPELINE SETUP
# ═════════════════════════════════════════════════════════════════════════════

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


# ═════════════════════════════════════════════════════════════════════════════
# 2. CRF TRAINING — mirrors _run_crf_experiment in run_pipeline.py
# ═════════════════════════════════════════════════════════════════════════════

def _to_word_sequences(X, labels, words):
    """Group consecutive phonemes by word into CRF sequences.
    Returns (list_of_seqs_of_dicts, list_of_label_seqs)."""
    seqs, lbl_seqs = [], []
    cur_x, cur_y, prev_w = [], [], None
    for x, l, w in zip(X, labels, words):
        if w != prev_w and prev_w is not None and cur_x:
            seqs.append(cur_x); lbl_seqs.append(cur_y)
            cur_x, cur_y = [], []
        cur_x.append({f'f{j}': float(v) for j, v in enumerate(x)})
        cur_y.append(str(l))
        prev_w = w
    if cur_x:
        seqs.append(cur_x); lbl_seqs.append(cur_y)
    return seqs, lbl_seqs


def _features_to_dicts_one_seq(X):
    """Treat the whole array as ONE sequence (used at detector inference,
    where we don't know word boundaries)."""
    return [[{f'f{j}': float(v) for j, v in enumerate(x)} for x in X]]


def train_per_patient_crfs(pipeline):
    """Train one CRF per patient. Reuses the exact recipe from
    _run_crf_experiment: StandardScaler → PCA(50) → word-level sequences →
    CRF(c1=0.1, c2=0.1, lbfgs, all_possible_transitions=True).

    Returns dict pid → {'crf', 'scaler', 'pca', 'valid_classes'}.
    """
    print("\n[2/6] Training per-patient CRFs on pipeline.train "
          f"(StandardScaler + PCA({N_PCA}) + word-level sequences)...")

    by_pid = defaultdict(lambda: {'X': [], 'y': [], 'w': []})
    for i, p in enumerate(pipeline.train['phoneme_participant_ids']):
        by_pid[p]['X'].append(np.asarray(pipeline.train['features'][i]).flatten())
        by_pid[p]['y'].append(str(pipeline.train['phoneme_labels'][i]))
        by_pid[p]['w'].append(pipeline.train['phoneme_words'][i])

    classifiers = {}
    for pid, d in by_pid.items():
        tr_feat, tr_lbl, tr_wrd = d['X'], d['y'], d['w']

        # Filter rare classes on TRAIN only (no leak)
        valid = {c for c, n in Counter(tr_lbl).items() if n >= MIN_CLASS_SAMPLES}
        keep = [i for i, l in enumerate(tr_lbl) if l in valid]
        tr_feat = [tr_feat[i] for i in keep]
        tr_lbl  = [tr_lbl[i]  for i in keep]
        tr_wrd  = [tr_wrd[i]  for i in keep]
        if len(tr_feat) < 10:
            continue

        X_tr = np.asarray(tr_feat)

        # Scale + PCA
        scaler = StandardScaler().fit(X_tr)
        X_tr = scaler.transform(X_tr)
        n_comp = min(N_PCA, X_tr.shape[1], X_tr.shape[0])
        pca = PCA(n_components=n_comp, random_state=RANDOM_STATE).fit(X_tr)
        X_tr = pca.transform(X_tr)

        # Word-level CRF sequences
        X_seq, y_seq = _to_word_sequences(X_tr, tr_lbl, tr_wrd)

        crf = sklearn_crfsuite.CRF(
            algorithm='lbfgs', c1=CRF_C1, c2=CRF_C2,
            max_iterations=CRF_MAX_ITER, all_possible_transitions=True,
        )
        crf.fit(X_seq, y_seq)

        classifiers[pid] = {
            'crf': crf, 'scaler': scaler, 'pca': pca,
            'valid_classes': valid,
        }
    print(f"  Trained {len(classifiers)} CRFs")
    return classifiers


def _transform(features, scaler, pca):
    X = np.asarray([np.asarray(f).flatten() for f in features])
    return pca.transform(scaler.transform(X))


# ═════════════════════════════════════════════════════════════════════════════
# 3. MFA BASELINE — uses pipeline.test directly
# ═════════════════════════════════════════════════════════════════════════════

def evaluate_mfa_baseline(pipeline, classifiers):
    """Feed pipeline.test directly to per-patient CRFs, with the SAME
    word-level sequencing the CRF was trained with. True upper bound."""
    print("\n[3/6] MFA baseline (pipeline.test -> CRF, word sequences)...")

    by_pid = defaultdict(lambda: {'X': [], 'y': [], 'w': []})
    for i, p in enumerate(pipeline.test['phoneme_participant_ids']):
        by_pid[p]['X'].append(np.asarray(pipeline.test['features'][i]).flatten())
        by_pid[p]['y'].append(str(pipeline.test['phoneme_labels'][i]))
        by_pid[p]['w'].append(pipeline.test['phoneme_words'][i])

    summary = {}
    for pid, d in by_pid.items():
        if pid not in classifiers:
            continue
        valid  = classifiers[pid]['valid_classes']
        scaler = classifiers[pid]['scaler']
        pca    = classifiers[pid]['pca']
        crf    = classifiers[pid]['crf']

        keep = [i for i, l in enumerate(d['y']) if l in valid]
        if len(keep) < 5:
            continue
        te_feat = [d['X'][i] for i in keep]
        te_lbl  = [d['y'][i] for i in keep]
        te_wrd  = [d['w'][i] for i in keep]

        X_te = pca.transform(scaler.transform(np.asarray(te_feat)))
        X_seq, y_seq = _to_word_sequences(X_te, te_lbl, te_wrd)
        y_pred_seq   = crf.predict(X_seq)

        y_pred = [p for s in y_pred_seq for p in s]
        y_true = [l for s in y_seq      for l in s]

        ed = edit_distance(y_true, y_pred)
        accuracy = sum(p == t for p, t in zip(y_pred, y_true)) / max(len(y_true), 1)
        summary[pid] = {
            'n_true':    len(y_true),
            'n_pred':    len(y_pred),
            'edit':      ed,
            'per':       ed / max(len(y_true), 1),
            'accuracy':  accuracy,
            'len_ratio': len(y_pred) / max(len(y_true), 1),
        }
    _print_table(summary, label='MFA baseline (pipeline.test, CRF upper bound)')
    return summary


# ═════════════════════════════════════════════════════════════════════════════
# 4. BOUNDARY DETECTOR — load or train
# ═════════════════════════════════════════════════════════════════════════════

# ── CORRECTED dataset builder ────────────────────────────────────────────────
# The version in boundary_detector_joint_audio.py reads `eeg_segments` from
# the pipeline, which contains short per-WORD slices, not full sentences.
# This version slices raw EEG by sentence using stim_start_idx/stim_end_idx,
# matching what run_path_b does for the CRF features.

# v6 = v5 with much lower word-onset loss weight (0.5 → 0.1)
CKPT_PREFIX = 'boundary_detector_v6_'

COUNT_LOSS_WEIGHT      = 0.005
WORD_ONSET_LOSS_WEIGHT = 0.1    # was 0.5 in v5 — lower to avoid hijacking encoder
WORD_ONSET_POS_WEIGHT  = 40.0
N_EPOCHS_OVERRIDE      = 40

# Inference: how to pick K (number of phoneme segments per sentence)
#   'predicted' — use the count-regressor head (BRAIN-ONLY, realistic)
#   'oracle'    — K = item['n_phonemes'] (best case, uses MFA at test)
#   'rate'      — TARGET_PHONEME_RATE × duration_s (single global rate)
#   'fixed'     — fixed PEAK_HEIGHT, no top-K
ADAPTIVE_K_SOURCE = 'predicted'


def word_onset_labels_from_mfa(phone_alignments, n_frames,
                                 frame_hz=FRAME_HZ, sigma_ms=10.0):
    """Soft Gaussian labels on word ONSETS (first phoneme of each word).

    Returns (labels, onset_times). Each unique consecutive `word` group is
    one onset event at the first phoneme's start_s.
    """
    sigma_frames = sigma_ms * frame_hz / 1000.0
    half_window = max(1, int(np.ceil(3 * sigma_frames)))
    labels = np.zeros(n_frames, dtype=np.float32)

    onset_times = []
    prev_word = None
    for ph in phone_alignments:
        w = ph.get('word', '') or ''
        if w != prev_word:
            onset_times.append(ph['start_s'])
            prev_word = w

    for t in onset_times:
        c = int(round(t * frame_hz))
        for off in range(-half_window, half_window + 1):
            f = c + off
            if 0 <= f < n_frames:
                weight = float(np.exp(-(off ** 2) / (2 * sigma_frames ** 2)))
                if weight > labels[f]:
                    labels[f] = weight
    return labels, onset_times


def _get_pipeline_chan_mask(pipeline, pid):
    """Same channel-keep logic as run_path_b uses."""
    if hasattr(pipeline, 'patient_data') and pid in pipeline.patient_data:
        pdata = pipeline.patient_data[pid]
        if 'channel_mask' in pdata:
            cm = pdata['channel_mask']
            return np.where(cm)[0] if cm.dtype == bool else np.asarray(cm)
        if 'included_channels' in pdata:
            return np.asarray(pdata['included_channels'])
    return None  # use all channels


def build_joint_dataset_fixed(pipeline, patient_ids):
    """Returns list of {pid, sent_idx, eeg, mfcc, labels, boundaries, ...}.

    Slices raw EEG and raw audio by sentence_list[sent_idx]['stim_start_idx']
    / ['stim_end_idx'] — mirrors run_path_b. Applies the same channel mask
    the pipeline applies for CRF features.
    """
    dataset = []
    audio_sr_raw = 48000
    try:
        audio_sr_raw = int(pipeline.config.audio_sr)
    except Exception:
        pass
    print(f"  [fixed] Audio raw sample rate: {audio_sr_raw} Hz, "
          f"target for MFCC: {SR_AUDIO_TARGET} Hz")
    eeg_sr = pipeline.config.eeg_sr

    raw_cache = {}    # pid -> (raw_eeg_kept_channels, raw_audio)

    for pid in patient_ids:
        wd = pipeline.split_result.get('word_segments_dict', {}).get(pid)
        if wd is None:
            print(f"  {pid}: not in word_segments_dict, skipping")
            continue

        try:
            mfa = load_mfa_alignments(pid)
        except Exception as e:
            print(f"  {pid}: MFA load failed ({e}), skipping")
            continue

        # Load and channel-mask raw EEG, raw audio (cached per pid)
        if pid not in raw_cache:
            raw_eeg = np.load(os.path.join(DUTCH_30_PATH, 'raw',
                                            f'{pid}_sEEG.npy'))
            raw_aud = np.load(os.path.join(DUTCH_30_PATH, 'raw',
                                            f'{pid}_audio.npy'))
            mask = _get_pipeline_chan_mask(pipeline, pid)
            if mask is not None:
                raw_eeg = raw_eeg[:, mask]
            raw_cache[pid] = (raw_eeg, raw_aud)
        raw_eeg, raw_aud = raw_cache[pid]

        sentence_list = wd['sentence_list']
        n_kept = 0
        for sent_idx, sent_info in enumerate(sentence_list):
            text = sent_info['text'] if isinstance(sent_info, dict) else sent_info
            if not text:
                continue
            if sent_idx not in mfa or not mfa[sent_idx]:
                continue

            eeg_start = int(sent_info['stim_start_idx'])
            eeg_end   = int(sent_info['stim_end_idx'])
            sent_eeg  = raw_eeg[eeg_start:eeg_end]
            if sent_eeg.shape[0] < int(0.2 * eeg_sr):  # < 200 ms — skip
                continue

            # Map EEG sample indices to audio sample indices
            aud_start = int(eeg_start * audio_sr_raw / eeg_sr)
            aud_end   = int(eeg_end   * audio_sr_raw / eeg_sr)
            sent_audio = raw_aud[aud_start:aud_end].astype(np.float32)
            if sent_audio.size == 0:
                continue

            # ── Trim to speech bounds ─────────────────────────────────────
            # If enabled, slice both sent_eeg and sent_audio to a snug window
            # around the actual speech (first MFA phone start → last MFA phone
            # end), with SPEECH_BUFFER_MS pre/post for motor-planning signal.
            # MFA phone alignments are also shifted to be relative to the new
            # trimmed start, so downstream label generators stay consistent.
            phone_alignments = mfa[sent_idx]
            if TRIM_TO_SPEECH and phone_alignments:
                buf_s = SPEECH_BUFFER_MS / 1000.0
                speech_start_s = max(0.0, phone_alignments[0]['start_s']  - buf_s)
                speech_end_s   = min(sent_eeg.shape[0] / eeg_sr,
                                      phone_alignments[-1]['end_s']      + buf_s)
                if speech_end_s > speech_start_s:
                    eeg_lo = int(round(speech_start_s * eeg_sr))
                    eeg_hi = int(round(speech_end_s   * eeg_sr))
                    aud_lo = int(round(speech_start_s * audio_sr_raw))
                    aud_hi = int(round(speech_end_s   * audio_sr_raw))
                    sent_eeg   = sent_eeg[eeg_lo:eeg_hi]
                    sent_audio = sent_audio[aud_lo:aud_hi]
                    # Shift phone alignments so t=0 is the new sentence start
                    phone_alignments = [
                        {**ph,
                         'start_s': ph['start_s'] - speech_start_s,
                         'end_s':   ph['end_s']   - speech_start_s}
                        for ph in phone_alignments
                    ]

            try:
                eeg_frames = extract_eeg_frames(sent_eeg)
                n_frames   = eeg_frames.shape[0]

                if audio_sr_raw != SR_AUDIO_TARGET:
                    g = gcd(int(audio_sr_raw), int(SR_AUDIO_TARGET))
                    sent_audio_rs = scipy.signal.resample_poly(
                        sent_audio,
                        int(SR_AUDIO_TARGET / g),
                        int(audio_sr_raw / g))
                else:
                    sent_audio_rs = sent_audio

                mfcc = compute_mfcc(sent_audio_rs, sr=SR_AUDIO_TARGET)
                mfcc = add_delta_features(mfcc)
                if mfcc.shape[0] < n_frames:
                    pad = np.zeros((n_frames - mfcc.shape[0], mfcc.shape[1]),
                                    dtype=np.float32)
                    mfcc = np.concatenate([mfcc, pad], axis=0)
                elif mfcc.shape[0] > n_frames:
                    mfcc = mfcc[:n_frames]

                labels, boundary_times = boundary_labels_from_mfa(
                    phone_alignments, n_frames)
                word_labels, word_onset_times = word_onset_labels_from_mfa(
                    phone_alignments, n_frames)
            except Exception as e:
                print(f"  {pid} sent {sent_idx}: feature extraction failed: {e}")
                continue

            dataset.append({
                'pid':              pid,
                'sentence_idx':     sent_idx,
                'eeg':              eeg_frames,
                'mfcc':             mfcc,
                'labels':           labels,
                'word_labels':      word_labels,
                'boundary_times':   boundary_times,
                'word_onset_times': word_onset_times,
                'n_phonemes':       len(mfa[sent_idx]),
                'n_words':          len(word_onset_times),
            })
            n_kept += 1

        # Per-patient frame-count diagnostic. The detector runs at 200 Hz
        # (5 ms/frame), so duration_s = mean_frames / 200. Useful sanity
        # checks:
        #   - mean ~4.0–5.0 s = normal Dutch sentence reading
        #   - mean >7 s usually means silence padding around speech
        #   - std/mean < ~0.05 (or min == max) ⇒ "uniform" distribution,
        #     which means stim_start_idx/stim_end_idx were set to a fixed
        #     window per sentence (not snug around actual speech). Affects:
        #     CTC has to emit lots of blanks, count regressor has to ignore
        #     duration as a phoneme-rate proxy.
        frames = [d['eeg'].shape[0] for d in dataset if d['pid'] == pid]
        if frames:
            arr = np.asarray(frames)
            mean_f = arr.mean()
            std_f  = arr.std()
            min_f  = arr.min()
            max_f  = arr.max()
            cv = std_f / max(mean_f, 1)            # coefficient of variation
            uniform_flag = ' [UNIFORM: likely silence-padded]' if cv < 0.05 else ''
            long_flag    = ' [LONG: >7s, silence padded]' if mean_f > 7 * 200 else ''
            print(f"  {pid}: kept {n_kept} sentences  "
                  f"frames mean={mean_f:.0f} (~{mean_f/200:.2f}s) "
                  f"std={std_f:.0f}  min={min_f}  max={max_f}  "
                  f"cv(=sd/mean)={cv:.2f}{uniform_flag}{long_flag}")

    return dataset


# Need gcd for audio resampling
from math import gcd

# ── Trim to speech bounds (silence-padding fix) ──────────────────────────────
# When stim_start_idx/stim_end_idx define a fixed window per sentence (e.g.
# always 4 sec), the sentence has lots of silence around the actual speech.
# Trimming to MFA-derived speech bounds + a small buffer cuts away that
# silence so the model spends its capacity on phonemes, not "predict blank".
#
# Buffer keeps a small pre-/post-speech window so motor-planning signal
# (~150–300 ms before phonation) is preserved.
TRIM_TO_SPEECH    = True
SPEECH_BUFFER_MS  = 200      # keep this much before first phone / after last phone


# ── COUNT-AWARE DETECTOR (v3) ────────────────────────────────────────────────
# Same encoder as JointBoundaryDetector, plus a second head that regresses
# per-sentence phoneme count from the pooled (mask-aware) encoder output.

class CountAwareJointDetector(nn.Module):
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
        self.boundary_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.count_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def encode(self, eeg, mfcc, pid):
        h_eeg  = self.eeg_proj[pid](eeg)  if eeg  is not None else None
        h_mfcc = self.mfcc_proj(mfcc)     if mfcc is not None else None
        if   h_eeg is not None and h_mfcc is not None:
            h = (h_eeg + h_mfcc) / 2.0
        elif h_eeg is not None: h = h_eeg
        else:                    h = h_mfcc
        h, _ = self.lstm(h)
        return h          # (B, T, 2*H)

    def forward(self, eeg=None, mfcc=None, pid=None, mask=None):
        """Returns (boundary_logits (B,T), count_pred (B,))."""
        h = self.encode(eeg, mfcc, pid)
        boundary_logits = self.boundary_head(h).squeeze(-1)   # (B, T)
        # Mask-aware mean pool for count head
        if mask is None:
            pooled = h.mean(dim=1)
        else:
            m = mask.float().unsqueeze(-1)            # (B, T, 1)
            pooled = (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        count_pred = self.count_head(pooled).squeeze(-1)      # (B,)
        return boundary_logits, count_pred


def collate_padded_with_count(batch):
    """Like collate_padded_joint, plus a count target."""
    X_eeg, X_mfcc, Y, mask = collate_padded_joint(batch)
    counts = torch.tensor([item['n_phonemes'] for item in batch],
                           dtype=torch.float32)
    return X_eeg, X_mfcc, Y, mask, counts


def train_count_aware(train_dataset, n_epochs=None, lr=LR):
    if n_epochs is None:
        n_epochs = N_EPOCHS_OVERRIDE
    per_patient_eeg_n_ch = {d['pid']: d['eeg'].shape[1] for d in train_dataset}
    mfcc_dim = train_dataset[0]['mfcc'].shape[1]

    model = CountAwareJointDetector(per_patient_eeg_n_ch, mfcc_dim).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                   weight_decay=WEIGHT_DECAY)
    pos_weight = torch.tensor([POS_WEIGHT], device=DEVICE)

    by_pid = defaultdict(list)
    for d in train_dataset:
        by_pid[d['pid']].append(d)
    pids = list(by_pid.keys())

    print(f"\n  Training count-aware detector — {n_epochs} epochs "
          f"(modality dropout: {MOD_DROPOUT}, count_loss_weight={COUNT_LOSS_WEIGHT})")

    for epoch in range(n_epochs):
        model.train()
        random.shuffle(pids)
        total_b_loss = 0.0; total_c_loss = 0.0
        total_frames = 0;   total_sents  = 0
        modality_counts = defaultdict(int)

        for pid in pids:
            items = by_pid[pid]
            random.shuffle(items)
            for i in range(0, len(items), BATCH_SIZE):
                batch = items[i:i + BATCH_SIZE]
                X_eeg, X_mfcc, Y, mask, counts = collate_padded_with_count(batch)
                X_eeg, X_mfcc = X_eeg.to(DEVICE), X_mfcc.to(DEVICE)
                Y, mask, counts = Y.to(DEVICE), mask.to(DEVICE), counts.to(DEVICE)

                modality = sample_modality()
                modality_counts[modality] += 1
                optimizer.zero_grad()

                if   modality == 'ieeg_only':
                    logits, count_pred = model(eeg=X_eeg, mfcc=None,    pid=pid, mask=mask)
                elif modality == 'audio_only':
                    logits, count_pred = model(eeg=None,  mfcc=X_mfcc,  pid=pid, mask=mask)
                else:
                    logits, count_pred = model(eeg=X_eeg, mfcc=X_mfcc,  pid=pid, mask=mask)

                # Boundary BCE on valid frames
                b_loss = F.binary_cross_entropy_with_logits(
                    logits, Y, pos_weight=pos_weight, reduction='none')
                b_loss = (b_loss * mask.float()).sum() / mask.float().sum().clamp(min=1)

                # Count MSE on every sentence
                c_loss = F.mse_loss(count_pred, counts)

                loss = b_loss + COUNT_LOSS_WEIGHT * c_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                total_b_loss += b_loss.item() * mask.float().sum().item()
                total_c_loss += c_loss.item() * len(batch)
                total_frames += mask.float().sum().item()
                total_sents  += len(batch)

        if (epoch + 1) % 5 == 0 or epoch == n_epochs - 1:
            print(f"    epoch {epoch+1:2d}/{n_epochs}  "
                  f"bce={total_b_loss/max(total_frames,1):.4f}  "
                  f"count_mse={total_c_loss/max(total_sents,1):.2f}  "
                  f"sqrt_mse≈{(total_c_loss/max(total_sents,1))**0.5:.2f} phonemes  "
                  f"modalities: {dict(modality_counts)}")
    return model


# ── COUNT + WORD-ONSET AWARE DETECTOR (v5) ───────────────────────────────────
# Three heads:
#   - boundary_head: per-frame phoneme boundary BCE (primary signal, top-K peaks)
#   - count_head:    per-sentence phoneme count regression (chooses K)
#   - word_head:     per-frame word-onset BCE (auxiliary, regularizes encoder)
# At inference we only consume boundary + count. The word head is training-only
# regularization to encourage the encoder to learn word-level structure.

class CountWordAwareDetector(nn.Module):
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
        self.boundary_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.word_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.count_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
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
        boundary_logits = self.boundary_head(h).squeeze(-1)
        word_logits     = self.word_head(h).squeeze(-1)
        if mask is None:
            pooled = h.mean(dim=1)
        else:
            m = mask.float().unsqueeze(-1)
            pooled = (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        count_pred = self.count_head(pooled).squeeze(-1)
        return boundary_logits, word_logits, count_pred


def collate_padded_with_count_and_words(batch):
    """Like collate_padded_with_count, plus a word-onset label tensor."""
    max_len = max(item['eeg'].shape[0] for item in batch)
    n_ch = batch[0]['eeg'].shape[1]
    n_mfcc = batch[0]['mfcc'].shape[1]

    X_eeg  = torch.zeros(len(batch), max_len, n_ch)
    X_mfcc = torch.zeros(len(batch), max_len, n_mfcc)
    Y_b    = torch.zeros(len(batch), max_len)
    Y_w    = torch.zeros(len(batch), max_len)
    mask   = torch.zeros(len(batch), max_len, dtype=torch.bool)

    for i, item in enumerate(batch):
        n = item['eeg'].shape[0]
        X_eeg[i, :n]  = torch.from_numpy(item['eeg'])
        X_mfcc[i, :n] = torch.from_numpy(item['mfcc'])
        Y_b[i, :n]    = torch.from_numpy(item['labels'])
        Y_w[i, :n]    = torch.from_numpy(item['word_labels'])
        mask[i, :n]   = True
    counts = torch.tensor([item['n_phonemes'] for item in batch],
                           dtype=torch.float32)
    return X_eeg, X_mfcc, Y_b, Y_w, mask, counts


def train_count_word_aware(train_dataset, n_epochs=None, lr=LR):
    if n_epochs is None:
        n_epochs = N_EPOCHS_OVERRIDE
    per_patient_eeg_n_ch = {d['pid']: d['eeg'].shape[1] for d in train_dataset}
    mfcc_dim = train_dataset[0]['mfcc'].shape[1]

    model = CountWordAwareDetector(per_patient_eeg_n_ch, mfcc_dim).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                   weight_decay=WEIGHT_DECAY)
    pos_weight_b = torch.tensor([POS_WEIGHT],            device=DEVICE)
    pos_weight_w = torch.tensor([WORD_ONSET_POS_WEIGHT], device=DEVICE)

    by_pid = defaultdict(list)
    for d in train_dataset:
        by_pid[d['pid']].append(d)
    pids = list(by_pid.keys())

    print(f"\n  Training count+word-aware detector — {n_epochs} epochs "
          f"(modality dropout: {MOD_DROPOUT}, "
          f"count_w={COUNT_LOSS_WEIGHT}, word_w={WORD_ONSET_LOSS_WEIGHT})")

    for epoch in range(n_epochs):
        model.train()
        random.shuffle(pids)
        tot_b = tot_w = tot_c = 0.0
        tot_frames = tot_sents = 0
        modality_counts = defaultdict(int)

        for pid in pids:
            items = by_pid[pid]
            random.shuffle(items)
            for i in range(0, len(items), BATCH_SIZE):
                batch = items[i:i + BATCH_SIZE]
                X_eeg, X_mfcc, Y_b, Y_w, mask, counts = collate_padded_with_count_and_words(batch)
                X_eeg, X_mfcc = X_eeg.to(DEVICE), X_mfcc.to(DEVICE)
                Y_b, Y_w, mask, counts = Y_b.to(DEVICE), Y_w.to(DEVICE), mask.to(DEVICE), counts.to(DEVICE)

                modality = sample_modality()
                modality_counts[modality] += 1
                optimizer.zero_grad()

                if   modality == 'ieeg_only':
                    b_logits, w_logits, count_pred = model(eeg=X_eeg, mfcc=None,    pid=pid, mask=mask)
                elif modality == 'audio_only':
                    b_logits, w_logits, count_pred = model(eeg=None,  mfcc=X_mfcc,  pid=pid, mask=mask)
                else:
                    b_logits, w_logits, count_pred = model(eeg=X_eeg, mfcc=X_mfcc,  pid=pid, mask=mask)

                m_f = mask.float()
                m_sum = m_f.sum().clamp(min=1)

                b_loss = F.binary_cross_entropy_with_logits(
                    b_logits, Y_b, pos_weight=pos_weight_b, reduction='none')
                b_loss = (b_loss * m_f).sum() / m_sum

                w_loss = F.binary_cross_entropy_with_logits(
                    w_logits, Y_w, pos_weight=pos_weight_w, reduction='none')
                w_loss = (w_loss * m_f).sum() / m_sum

                c_loss = F.mse_loss(count_pred, counts)

                loss = b_loss + WORD_ONSET_LOSS_WEIGHT * w_loss + COUNT_LOSS_WEIGHT * c_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                tot_b += b_loss.item() * m_sum.item()
                tot_w += w_loss.item() * m_sum.item()
                tot_c += c_loss.item() * len(batch)
                tot_frames += m_sum.item()
                tot_sents  += len(batch)

        if (epoch + 1) % 5 == 0 or epoch == n_epochs - 1:
            print(f"    epoch {epoch+1:2d}/{n_epochs}  "
                  f"bce_b={tot_b/max(tot_frames,1):.4f}  "
                  f"bce_w={tot_w/max(tot_frames,1):.4f}  "
                  f"count_mse={tot_c/max(tot_sents,1):.2f}  "
                  f"sqrt≈{(tot_c/max(tot_sents,1))**0.5:.2f}p  "
                  f"mods: {dict(modality_counts)}")
    return model


def load_or_train_detector(pipeline):
    print("\n[4/6] Building joint dataset for detector (corrected slicing)...")
    full_ds = build_joint_dataset_fixed(pipeline, ALL_PIDS)
    train_ds, test_ds = split_by_sentence(full_ds)
    eeg_stats  = fit_train_stats(train_ds, 'eeg')
    mfcc_stats = fit_train_stats(train_ds, 'mfcc')
    train_ds   = apply_stats(train_ds, eeg_stats, mfcc_stats)
    test_ds    = apply_stats(test_ds,  eeg_stats, mfcc_stats)
    print(f"  detector dataset: train={len(train_ds)}  test={len(test_ds)}")

    # Use a v2 prefix so we don't accidentally load the stale .pt that was
    # trained on broken (per-word, truncated) EEG slices.
    ckpts = sorted(glob.glob(f'{CKPT_PREFIX}*.pt'))
    if ckpts:
        ckpt_path = ckpts[-1]
        print(f"  Loading {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        model = CountWordAwareDetector(
            per_patient_eeg_n_ch=ckpt['per_patient_eeg_n_ch'],
            mfcc_dim=ckpt['mfcc_dim'],
            hidden_dim=HIDDEN_DIM, n_layers=N_LSTM_LAYERS, dropout=DROPOUT,
        ).to(DEVICE)
        model.load_state_dict(ckpt['model_state'])
        model.eval()
        print(f"  loaded ({sum(p.numel() for p in model.parameters())/1e6:.2f}M params)")
    else:
        print(f"  No {CKPT_PREFIX} checkpoint -- training count+word-aware "
              "detector from scratch (~5-10 min on GPU)")
        model = train_count_word_aware(train_ds)
        out_path = f'{CKPT_PREFIX}{datetime.now().strftime("%Y%m%d_%H%M")}.pt'
        torch.save({
            'model_state':         model.state_dict(),
            'per_patient_eeg_n_ch': {pid: model.eeg_proj[pid].in_features
                                      for pid in model.eeg_proj},
            'mfcc_dim':            model.mfcc_proj.in_features,
        }, out_path)
        print(f"  saved to {out_path}")

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


def predict_segments(model, eeg_frames_for_detector, pid, oracle_n_phonemes=None):
    """Run detector → list of (start_s, end_s) phoneme intervals.

    K (target segment count) source depends on ADAPTIVE_K_SOURCE.
    """
    model.eval()
    # Check by class NAME so importlib.reload of this module doesn't break
    # isinstance checks against pre-reload model instances.
    cls_name = model.__class__.__name__

    with torch.no_grad():
        X = torch.from_numpy(eeg_frames_for_detector.astype(np.float32)
                              ).unsqueeze(0).to(DEVICE)
        out = model(eeg=X, mfcc=None, pid=pid)

        # Disambiguate by inspecting return type:
        #   - CountWordAwareDetector → (boundary_logits, word_logits, count_pred)
        #   - CountAwareJointDetector → (boundary_logits, count_pred)
        #   - JointBoundaryDetector → boundary_logits
        if isinstance(out, tuple):
            if len(out) == 3:
                logits, _w_logits, count_pred = out
                count_pred_val = float(count_pred[0].item())
            elif len(out) == 2:
                logits, count_pred = out
                count_pred_val = float(count_pred[0].item())
            else:
                raise ValueError(
                    f"Unexpected forward output shape: tuple of length {len(out)} "
                    f"(model class: {cls_name})")
        else:
            logits = out
            count_pred_val = None
        probs  = torch.sigmoid(logits)[0].cpu().numpy()

    n_frames   = len(probs)
    frame_hz   = 200
    duration_s = n_frames / frame_hz

    # Choose K
    if ADAPTIVE_K_SOURCE == 'oracle':
        if oracle_n_phonemes is None:
            raise ValueError("oracle K-mode requires oracle_n_phonemes")
        k = int(oracle_n_phonemes)
    elif ADAPTIVE_K_SOURCE == 'predicted':
        if count_pred_val is None:
            raise ValueError("predicted K-mode requires CountAwareJointDetector")
        k = max(1, int(round(count_pred_val)))
    elif ADAPTIVE_K_SOURCE == 'rate':
        k = int(round(TARGET_PHONEME_RATE * duration_s))
    elif ADAPTIVE_K_SOURCE == 'fixed':
        peaks_arr, _ = scipy.signal.find_peaks(probs, height=PEAK_HEIGHT,
                                                distance=PEAK_DISTANCE)
        peaks = list(peaks_arr)
        boundaries = sorted(set([0] + peaks + [n_frames]))
        min_seg_frames = int(MIN_SEGMENT_MS * frame_hz / 1000)
        return [(boundaries[i] / frame_hz, boundaries[i+1] / frame_hz)
                for i in range(len(boundaries) - 1)
                if boundaries[i+1] - boundaries[i] >= min_seg_frames]
    else:
        raise ValueError(f"Unknown ADAPTIVE_K_SOURCE: {ADAPTIVE_K_SOURCE}")

    # Top-K peak selection
    peaks_all, _ = scipy.signal.find_peaks(probs, distance=PEAK_DISTANCE)
    n_boundaries_needed = max(1, k - 1)
    predict_segments.last_n_candidates = len(peaks_all)
    predict_segments.last_n_wanted     = n_boundaries_needed
    predict_segments.last_prob_max     = float(probs.max())
    predict_segments.last_prob_p95     = float(np.percentile(probs, 95))
    predict_segments.last_count_pred   = count_pred_val
    predict_segments.last_k_used       = k
    if len(peaks_all) > n_boundaries_needed:
        order = np.argsort(probs[peaks_all])[::-1][:n_boundaries_needed]
        peaks = sorted(peaks_all[order].tolist())
    else:
        peaks = list(peaks_all)

    boundaries = sorted(set([0] + peaks + [n_frames]))
    min_seg_frames = int(MIN_SEGMENT_MS * frame_hz / 1000)
    return [(boundaries[i] / frame_hz, boundaries[i+1] / frame_hz)
            for i in range(len(boundaries) - 1)
            if boundaries[i+1] - boundaries[i] >= min_seg_frames]


def evaluate_detector_path(pipeline, classifiers,
                            model, detector_test_ds):
    """For each test sentence: detect boundaries, extract features
    pipeline-native, classify, compare to MFA labels."""
    print("\n[5/6] Detector path (iEEG -> boundaries -> features -> CRF)...")

    # Shape sanity check — first 5 items from detector_test_ds
    print("  Detector test_ds shape sanity check (first 5):")
    for i, item in enumerate(detector_test_ds[:5]):
        print(f"    {item['pid']}  sent {item['sentence_idx']:3d}  "
              f"eeg.shape={item['eeg'].shape}  "
              f"labels.shape={item['labels'].shape}  "
              f"n_phonemes={item['n_phonemes']}  "
              f"duration={item['eeg'].shape[0]/200:.2f}s")

    eeg_sr     = pipeline.config.eeg_sr
    win_len    = pipeline.config.window_length
    frameshift = pipeline.config.frameshift
    stk_order  = RUN_CONFIG['stacking_order']

    # Cache full-channel raw EEG per patient (slow to load)
    raw_eeg_cache  = {}
    chan_mask_cache = {pid: get_pipeline_channel_mask(pipeline, pid)
                       for pid in ALL_PIDS}

    per_pid = defaultdict(lambda: {'true': [], 'pred': [], 'n_sent': 0,
                                     'n_cand': [], 'n_wanted': [],
                                     'prob_max': [], 'prob_p95': [],
                                     'k_pred': [], 'k_true': []})
    n_skipped_segments = 0

    for item in detector_test_ds:
        pid      = item['pid']
        sent_idx = item['sentence_idx']
        if pid not in classifiers:
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
        segments = predict_segments(
            model, item['eeg'], pid,
            oracle_n_phonemes=item.get('n_phonemes'))
        # capture diagnostics from the last call
        per_pid[pid]['n_cand'].append(getattr(predict_segments, 'last_n_candidates', 0))
        per_pid[pid]['n_wanted'].append(getattr(predict_segments, 'last_n_wanted', 0))
        per_pid[pid]['prob_max'].append(getattr(predict_segments, 'last_prob_max', 0.0))
        per_pid[pid]['prob_p95'].append(getattr(predict_segments, 'last_prob_p95', 0.0))
        cp = getattr(predict_segments, 'last_count_pred', None)
        if cp is not None:
            per_pid[pid]['k_pred'].append(cp)
            per_pid[pid]['k_true'].append(item.get('n_phonemes', 0))
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

        # 5c. CRF inference: scaler+PCA from train, then one sequence per
        #     sentence (we don't have word boundaries from iEEG)
        scaler = classifiers[pid]['scaler']
        pca    = classifiers[pid]['pca']
        crf    = classifiers[pid]['crf']
        valid  = classifiers[pid]['valid_classes']

        X = pca.transform(scaler.transform(np.asarray(seg_feats)))
        X_seq = _features_to_dicts_one_seq(X)
        y_pred = crf.predict(X_seq)[0]

        # 5d. True labels = MFA phonemes for this sentence, filtered to the
        #     valid class set used in training (matches MFA baseline filter)
        mfa = load_mfa_alignments(pid).get(sent_idx, [])
        y_true = [p['phone'] for p in mfa if p['phone'] in valid]

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

    # Detector diagnostic
    print("\n  Detector diagnostic:")
    has_count = any(per_pid[p]['k_pred'] for p in per_pid)
    if has_count:
        print(f"  {'pid':<5} {'k_pred_mean':>11} {'k_true_mean':>11} "
              f"{'count_mae':>9} {'p95':>6} {'pmax':>6}")
        print("  " + "-" * 60)
        for pid in sorted(per_pid):
            d = per_pid[pid]
            if not d['k_pred']: continue
            kp = np.array(d['k_pred']); kt = np.array(d['k_true'])
            mae = np.mean(np.abs(kp - kt))
            print(f"  {pid:<5} {kp.mean():>11.1f} {kt.mean():>11.1f} "
                  f"{mae:>9.2f} {np.mean(d['prob_p95']):>6.3f} "
                  f"{np.mean(d['prob_max']):>6.3f}")
    else:
        print(f"  {'pid':<5} {'n_cand_mean':>11} {'n_wanted_mean':>13} "
              f"{'p95':>6} {'pmax':>6}")
        print("  " + "-" * 50)
        for pid in sorted(per_pid):
            d = per_pid[pid]
            if not d['n_cand']: continue
            print(f"  {pid:<5} {np.mean(d['n_cand']):>11.1f} "
                  f"{np.mean(d['n_wanted']):>13.1f} "
                  f"{np.mean(d['prob_p95']):>6.3f} {np.mean(d['prob_max']):>6.3f}")

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


def edit_ops_breakdown(true_seq, pred_seq):
    """Return (n_correct, n_sub, n_ins, n_del) from optimal Levenshtein
    alignment. Insertions = pred has extra; deletions = pred missing.
    Sub + ins + del + correct does not equal sum of lengths because each
    position can only be one op."""
    s1 = [str(x) for x in true_seq]
    s2 = [str(x) for x in pred_seq]
    n1, n2 = len(s1), len(s2)
    # DP for minimum cost path
    INF = float('inf')
    cost = [[0] * (n2 + 1) for _ in range(n1 + 1)]
    for i in range(n1 + 1): cost[i][0] = i
    for j in range(n2 + 1): cost[0][j] = j
    for i in range(1, n1 + 1):
        for j in range(1, n2 + 1):
            if s1[i-1] == s2[j-1]:
                cost[i][j] = cost[i-1][j-1]
            else:
                cost[i][j] = 1 + min(cost[i-1][j-1], cost[i-1][j], cost[i][j-1])
    # Backtrace
    n_correct = n_sub = n_ins = n_del = 0
    i, j = n1, n2
    while i > 0 and j > 0:
        if s1[i-1] == s2[j-1]:
            n_correct += 1; i -= 1; j -= 1
        elif cost[i][j] == cost[i-1][j-1] + 1:
            n_sub += 1; i -= 1; j -= 1
        elif cost[i][j] == cost[i-1][j] + 1:
            n_del += 1; i -= 1     # true had a phoneme pred didn't emit
        else:
            n_ins += 1; j -= 1     # pred had an extra phoneme
    n_del += i
    n_ins += j
    return n_correct, n_sub, n_ins, n_del


def ngram_coverage(true_seq, pred_seq, min_n=3, shift_by_len=None):
    """Greedy longest-first n-gram coverage. For each contiguous n-gram
    (n >= min_n) in true_seq that also appears in pred_seq (within the
    shift budget per length), claim those phonemes as covered. Returns
    fraction of true phonemes covered.

    Used as the reward signal in MRT training. Shift-tolerant matching
    (same as `find_color_matches`) so that sequences with the right
    phonemes in the right relative order get credit even if positionally
    offset. Each phoneme can only be claimed once."""
    if not true_seq or not pred_seq:
        return 0.0
    if shift_by_len is None:
        shift_by_len = DEFAULT_SHIFT_BY_LEN
    matches = find_color_matches(true_seq, pred_seq,
                                  shift_by_len=shift_by_len,
                                  max_ngram_len=15)
    covered = sum(L for m in matches
                   for (_, _, L, _, _) in [m if len(m) == 5 else m + (None,)]
                   if L >= min_n)
    return covered / len(true_seq)


def lcs_length(s1, s2):
    """Length of longest common subsequence (in-order, not contiguous).
    Equivalent to: longest run of phonemes pred got in the right order."""
    s1 = [str(x) for x in s1]
    s2 = [str(x) for x in s2]
    n1, n2 = len(s1), len(s2)
    if n1 == 0 or n2 == 0: return 0
    prev = [0] * (n2 + 1)
    for i in range(1, n1 + 1):
        curr = [0] * (n2 + 1)
        for j in range(1, n2 + 1):
            if s1[i-1] == s2[j-1]:
                curr[j] = prev[j-1] + 1
            else:
                curr[j] = max(prev[j], curr[j-1])
        prev = curr
    return prev[n2]


def ngram_recall(true_seq, pred_seq, n):
    """Fraction of true n-grams (with repetition) that also appear in pred.
    Multiset overlap, not unique. Chance level for random sequences over
    V phonemes is (1/V)^(n-1) for true bigrams already in pred sequence."""
    if len(true_seq) < n or len(pred_seq) < n:
        return 0.0
    true_ngrams = Counter(tuple(true_seq[i:i+n])
                           for i in range(len(true_seq) - n + 1))
    pred_ngrams = Counter(tuple(pred_seq[i:i+n])
                           for i in range(len(pred_seq) - n + 1))
    matched = sum((true_ngrams & pred_ngrams).values())
    total = sum(true_ngrams.values())
    return matched / max(total, 1)


def rich_metrics(true_seq, pred_seq):
    """Collect a richer panel of metrics than just PER."""
    n_true = len(true_seq); n_pred = len(pred_seq)
    ed = edit_distance(true_seq, pred_seq)
    n_correct, n_sub, n_ins, n_del = edit_ops_breakdown(true_seq, pred_seq)
    lcs = lcs_length(true_seq, pred_seq)
    return {
        'n_true':       n_true,
        'n_pred':       n_pred,
        'edit':         ed,
        'per':          ed / max(n_true, 1),
        'len_ratio':    n_pred / max(n_true, 1),
        'sub_rate':     n_sub  / max(n_true, 1),
        'del_rate':     n_del  / max(n_true, 1),
        'ins_rate':     n_ins  / max(n_true, 1),
        'correct':      n_correct,
        'correct_frac': n_correct / max(n_true, 1),
        'lcs':          lcs,
        'lcs_frac':     lcs / max(n_true, 1),
        '2gram_recall': ngram_recall(true_seq, pred_seq, 2),
        '3gram_recall': ngram_recall(true_seq, pred_seq, 3),
    }


def print_rich_metrics_table(per_pid_metrics, label=''):
    """per_pid_metrics: dict pid -> rich_metrics dict."""
    print(f"\n  {label}")
    print(f"  {'pid':<5} {'PER':>7} {'sub%':>6} {'del%':>6} {'ins%':>6} "
          f"{'corr%':>6} {'LCS%':>6} {'2g_r%':>6} {'3g_r%':>6} {'len':>5}")
    print("  " + "-" * 75)
    for pid in sorted(per_pid_metrics):
        m = per_pid_metrics[pid]
        print(f"  {pid:<5} "
              f"{m['per']:>6.1%} "
              f"{m['sub_rate']:>5.1%} "
              f"{m['del_rate']:>5.1%} "
              f"{m['ins_rate']:>5.1%} "
              f"{m['correct_frac']:>5.1%} "
              f"{m['lcs_frac']:>5.1%} "
              f"{m['2gram_recall']:>5.1%} "
              f"{m['3gram_recall']:>5.1%} "
              f"{m['len_ratio']:>4.2f}×")
    print("  " + "-" * 75)
    keys = ['per', 'sub_rate', 'del_rate', 'ins_rate', 'correct_frac',
            'lcs_frac', '2gram_recall', '3gram_recall', 'len_ratio']
    means = {k: np.mean([m[k] for m in per_pid_metrics.values()])
             for k in keys}
    print(f"  {'mean':<5} "
          f"{means['per']:>6.1%} "
          f"{means['sub_rate']:>5.1%} "
          f"{means['del_rate']:>5.1%} "
          f"{means['ins_rate']:>5.1%} "
          f"{means['correct_frac']:>5.1%} "
          f"{means['lcs_frac']:>5.1%} "
          f"{means['2gram_recall']:>5.1%} "
          f"{means['3gram_recall']:>5.1%} "
          f"{means['len_ratio']:>4.2f}×")


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
    pipeline                = build_pipeline()
    classifiers             = train_per_patient_crfs(pipeline)
    mfa_summary             = evaluate_mfa_baseline(pipeline, classifiers)
    model, detector_test_ds = load_or_train_detector(pipeline)
    det_summary             = evaluate_detector_path(
        pipeline, classifiers, model, detector_test_ds)
    print_comparison(mfa_summary, det_summary)
    return {
        'pipeline':    pipeline,
        'classifiers': classifiers,
        'model':       model,
        'mfa_summary': mfa_summary,
        'det_summary': det_summary,
    }


# ═════════════════════════════════════════════════════════════════════════════
# PIPELINE-STEP API
# ═════════════════════════════════════════════════════════════════════════════
# Drop-in replacement for run_from_config that uses the iEEG boundary
# detector for test-time segmentation instead of MFA. Populates
# pipeline.patient_results[pid] = {'true_labels', 'predictions',
# 'accuracy', ...} so the colored-n-gram viz keeps working unchanged.

def train_brain_only_models(pipeline):
    """TRAIN-TIME ONLY. Trains all components needed for brain-only inference,
    using MFA-derived data as the supervision signal.

    What gets trained:
      - One CRF per patient on `pipeline.train['features']` (per-phoneme HG
        features computed by run_with_mfa_boundaries; MFA timestamps were
        used to slice phonemes — supervision only, not test input).
      - The v6 boundary detector (BiLSTM + count head + word-onset head),
        trained on per-sentence iEEG with MFA timestamps as boundary labels.

    No test-time inference happens here. No per-sentence prediction occurs.
    MFA's role at this stage is purely as a TRAINING-LABEL source.

    Returns:
        classifiers:        dict pid → {'crf', 'scaler', 'pca', 'valid_classes'}
        v6_model:           trained CountWordAwareDetector
        detector_test_ds:   held-out sentences for the brain-only inference
                            phase. Each item has 'eeg' (z-scored frame
                            features for v6's input), 'pid', 'sentence_idx',
                            and (for diagnostics) 'n_phonemes'.
    """
    classifiers = train_per_patient_crfs(pipeline)
    v6_model, detector_test_ds = load_or_train_detector(pipeline)
    return classifiers, v6_model, detector_test_ds


def predict_one_sentence_brain_only(v6_model, classifiers, sentence_eeg,
                                     detector_input_frames, pid,
                                     oracle_n_phonemes=None,
                                     stk_order=None,
                                     pipeline_config=None):
    """TEST-TIME, BRAIN-ONLY. Predict the phoneme sequence for one sentence
    using ONLY iEEG. No MFA, no audio.

    Steps:
      1. v6 detector reads detector_input_frames (z-scored 200 Hz iEEG
         features) → predicts phoneme boundaries + count.
      2. Slice raw iEEG (`sentence_eeg`) at those predicted boundaries →
         pipeline-native HG features per segment.
      3. Apply patient's StandardScaler + PCA + CRF → predicted phoneme labels.

    Args:
        v6_model:                trained CountWordAwareDetector
        classifiers:             dict from train_brain_only_models
        sentence_eeg:            np.ndarray (n_samples, n_channels) — raw iEEG
                                 for this sentence (channel-masked)
        detector_input_frames:   np.ndarray (T, n_channels) — z-scored 200 Hz
                                 features for v6 (typically item['eeg'] from
                                 detector_test_ds)
        pid:                     patient id
        oracle_n_phonemes:       only used if v6 is configured to use 'oracle'
                                 K-source; ignored under default 'predicted'
        stk_order:               stacking order for feature extraction
        pipeline_config:         pipeline.config (for sample rate, etc.)

    Returns:
        (predicted_phonemes, segments_seconds): list[str], list[(start_s, end_s)]
        Empty lists if no boundaries detected.
    """
    if pid not in classifiers:
        return [], []

    eeg_sr     = pipeline_config.eeg_sr      if pipeline_config else 1024
    win_len    = pipeline_config.window_length if pipeline_config else 0.015
    frameshift = pipeline_config.frameshift  if pipeline_config else 0.005
    if stk_order is None:
        stk_order = RUN_CONFIG['stacking_order']

    # Step 1 — boundaries from iEEG (no MFA, no audio)
    segments = predict_segments(v6_model, detector_input_frames, pid,
                                  oracle_n_phonemes=oracle_n_phonemes)
    if not segments:
        return [], []

    # Step 2 — pipeline-native features per detected segment (raw iEEG only)
    seg_feats = []
    for (s_s, e_s) in segments:
        f = extract_phoneme_feature_pipeline_native(
            sentence_eeg, s_s, e_s, eeg_sr, win_len, frameshift, stk_order)
        if f is not None:
            seg_feats.append(f)
    if not seg_feats:
        return [], segments

    # Step 3 — CRF inference (StandardScaler + PCA + CRF, all from train-side)
    scaler = classifiers[pid]['scaler']
    pca    = classifiers[pid]['pca']
    crf    = classifiers[pid]['crf']

    X = pca.transform(scaler.transform(np.asarray(seg_feats)))
    y_pred = crf.predict(_features_to_dicts_one_seq(X))[0]
    return list(y_pred), segments


def get_mfa_oracle_labels(pid, sent_idx, valid_classes):
    """SCORING-TIME ONLY. Look up MFA's ground-truth phoneme sequence for a
    sentence, filtered to the classes the patient's CRF was trained on.

    MFA appears here as the ORACLE REFERENCE for computing PER. It is never
    fed to the predictor — this function exists separately to make that
    explicit.

    Returns: list of phoneme strings (the true sequence).
    """
    mfa = load_mfa_alignments(pid).get(sent_idx, [])
    return [p['phone'] for p in mfa if p['phone'] in valid_classes]


def run_with_ieeg_boundaries(pipeline, run_config=None):
    """Brain-only inference orchestration. Three explicit phases:

       1. TRAIN  (uses MFA labels as supervision):
          train_brain_only_models(pipeline)
       2. INFER  (iEEG only, no MFA, no audio):
          predict_one_sentence_brain_only(...) per held-out sentence
       3. SCORE  (compare to MFA oracle for PER computation):
          get_mfa_oracle_labels(...) per held-out sentence

    Stores per-patient results in pipeline.patient_results.
    Returns (name, params, results) — same shape as run_from_config.
    """
    # ── Phase 1: TRAIN ─────────────────────────────────────────────────────
    classifiers, v6_model, detector_test_ds = train_brain_only_models(pipeline)

    # (Diagnostic table — not part of brain-only inference, just printed)
    _ = evaluate_detector_path(pipeline, classifiers, v6_model,
                                detector_test_ds)

    # Per-sentence storage
    per_pid_pred = defaultdict(list)
    per_pid_true = defaultdict(list)
    per_pid_true_sids = defaultdict(list)
    per_pid_pred_sids = defaultdict(list)
    per_pid_pred_segs = defaultdict(list)   # (start_s, end_s) per pred phoneme
    per_pid_true_segs = defaultdict(list)   # (start_s, end_s) per true phoneme

    # Cache raw EEG per patient (with the same channel mask the CRF saw)
    raw_eeg_cache  = {}
    chan_mask_cache = {pid: get_pipeline_channel_mask(pipeline, pid)
                       for pid in ALL_PIDS}

    stk_order = (run_config or RUN_CONFIG)['stacking_order']

    for item in detector_test_ds:
        pid      = item['pid']
        sent_idx = item['sentence_idx']

        if pid not in raw_eeg_cache:
            raw  = np.load(os.path.join(DUTCH_30_PATH, 'raw',
                                         f'{pid}_sEEG.npy'))
            mask = chan_mask_cache[pid]
            raw_eeg_cache[pid] = raw[:, mask] if mask is not None else raw

        wd        = pipeline.split_result['word_segments_dict'][pid]
        sent_info = wd['sentence_list'][sent_idx]
        sentence_eeg = raw_eeg_cache[pid][int(sent_info['stim_start_idx']):
                                           int(sent_info['stim_end_idx'])]

        # ── Phase 2: INFER (brain-only) ──────────────────────────────────
        y_pred, _segments = predict_one_sentence_brain_only(
            v6_model, classifiers,
            sentence_eeg=sentence_eeg,
            detector_input_frames=item['eeg'],
            pid=pid,
            oracle_n_phonemes=item.get('n_phonemes'),
            stk_order=stk_order,
            pipeline_config=pipeline.config,
        )
        if not y_pred:
            continue

        # ── Phase 3: SCORE (compare to MFA oracle) ────────────────────────
        valid_classes = classifiers[pid]['valid_classes']
        y_true = get_mfa_oracle_labels(pid, sent_idx, valid_classes)

        # MFA-derived true-phoneme times (same filter the oracle labels use)
        mfa_phones = load_mfa_alignments(pid).get(sent_idx, [])
        true_segs  = [(p['start_s'], p['end_s'])
                       for p in mfa_phones if p['phone'] in valid_classes]
        # Predicted segment times — pad/truncate to len(y_pred) defensively.
        pred_segs = list(_segments[:len(y_pred)])
        while len(pred_segs) < len(y_pred):
            pred_segs.append((float('nan'), float('nan')))

        per_pid_pred[pid].extend(y_pred)
        per_pid_true[pid].extend(y_true)
        per_pid_pred_sids[pid].extend([sent_idx] * len(y_pred))
        per_pid_true_sids[pid].extend([sent_idx] * len(y_true))
        per_pid_pred_segs[pid].extend(pred_segs)
        per_pid_true_segs[pid].extend(true_segs[:len(y_true)])

    # Pack per-patient summary into pipeline.patient_results
    pipeline.patient_results = {}
    for pid in sorted(per_pid_pred):
        true = per_pid_true[pid]
        pred = per_pid_pred[pid]
        if not true:
            continue
        common = min(len(true), len(pred))
        accuracy = (sum(t == p for t, p in zip(true[:common], pred[:common]))
                    / max(len(true), 1))
        ed = edit_distance(true, pred)
        pipeline.patient_results[pid] = {
            'true_labels':       list(true),
            'predictions':       list(pred),
            'true_sentence_ids': list(per_pid_true_sids[pid]),
            'pred_sentence_ids': list(per_pid_pred_sids[pid]),
            'true_segments':     list(per_pid_true_segs[pid]),   # (start_s,end_s) per phoneme
            'pred_segments':     list(per_pid_pred_segs[pid]),
            'accuracy':          accuracy,
            'edit_distance':     ed,
            'per':               ed / max(len(true), 1),
            'n_test':            len(true),
            'n_pred':            len(pred),
        }

    return ('detector_path',
            {'ckpt_prefix': CKPT_PREFIX, 'k_source': ADAPTIVE_K_SOURCE},
            pipeline.patient_results)


# Backward-compat alias — `run_path_detector` is the older name kept so
# previous notebook code keeps working. New code should use
# `run_with_ieeg_boundaries`.
run_path_detector = run_with_ieeg_boundaries


# ═════════════════════════════════════════════════════════════════════════════
# COLORED N-GRAM VISUALIZATION  (variable shift per n-gram length)
# ═════════════════════════════════════════════════════════════════════════════

# Per-length max shift. Keys are n-gram lengths; the largest length ≤ key
# wins for any actual length. {2:5, 3:10, 4:20} means 2-grams allow ±5,
# 3-grams ±10, 4-grams (and longer) ±20.
DEFAULT_SHIFT_BY_LEN = {2: 5, 3: 10, 4: 20}


# Weak phonetic equivalence classes — phonemes within the same set are
# considered "almost the same". Used only at viz time to extend strong
# n-gram matches by one position on either side, rendered with a dashed
# border in the parent match's color.
DEFAULT_WEAK_EQUIVALENCE = [
    # Vowel length pairs (Dutch CGN inventory uses long-vowel ː markers)
    {'eː', 'e'}, {'aː', 'a'}, {'iː', 'i'}, {'oː', 'o'},
    {'uː', 'u'}, {'yː', 'y'}, {'øː', 'ø'},
    # Vowel quality (close pairs)
    {'e', 'ɛ'}, {'o', 'ɔ'}, {'i', 'ɪ'}, {'u', 'ʊ'},
    {'eː', 'ɛ'}, {'oː', 'ɔ'},
    {'ɪ', 'ɛ'},      # high-lax vs mid-lax front vowels (neighbors on vowel chart)
    {'aː', 'eː'},    # long low central vs long mid front (looser pair, but
                      # the model in this dataset frequently confuses them)
    # Voicing pairs (stops + fricatives)
    {'t', 'd'}, {'p', 'b'},
    {'k', 'ɡ', 'g'},                    # IPA ɡ (U+0261) and ASCII g often interchangeable
    {'s', 'z'}, {'f', 'v'}, {'ʃ', 'ʒ'},
    {'ɣ', 'x'},                         # voiced/voiceless velar fricatives
    {'ɣ', 'ɦ'},                         # both voiced; cross-dialect overlap
    # /r/ realization variants (Dutch has alveolar trill, uvular trill, uvular fric.)
    {'r', 'ʀ', 'ʁ'},
    # Liquids — alveolar /r/ and /l/ share place + sonorance, often confusable
    {'r', 'l'},
    # Schwa-adjacent
    {'ə', 'ɛ'}, {'ə', 'ɪ'},
    # Glide / labial-velar approximant vs near-back rounded vowel — perceptually
    # close in Dutch (ʋ is the labio-dental approximant 'w'-like sound)
    {'ʊ', 'ʋ'},
    # Velar stop vs glottal/voiceless-velar fricative — close in place
    {'k', 'h'},
]


def _build_weak_equiv_map(equiv_classes):
    """phoneme -> set of weakly-equivalent phonemes (excluding self)."""
    from collections import defaultdict as _dd
    weak = _dd(set)
    for cls in equiv_classes:
        for p in cls:
            for q in cls:
                if p != q:
                    weak[p].add(q)
    return dict(weak)


def _is_weak_match(a, b, weak_map):
    if a == b:
        return False    # exact match, not weak
    return b in weak_map.get(a, ())


def find_weak_extensions(matches, true_seq, pred_seq,
                          weak_map=None,
                          true_sentence_ids=None,
                          pred_sentence_ids=None):
    """Try to extend each strong n-gram by one position on the left and one
    on the right using weak phonetic equivalence. Returns a list of
    (ts, ps, color_idx) triples — single-cell weak extensions, each tagged
    with the color_idx of the parent match.

    Skips an extension if either side's neighbor is already claimed by a
    strong match. Respects sentence boundaries when sentence ids provided.
    """
    if weak_map is None:
        weak_map = _build_weak_equiv_map(DEFAULT_WEAK_EQUIVALENCE)
    n_t, n_p = len(true_seq), len(pred_seq)

    used_true = [False] * n_t
    used_pred = [False] * n_p
    for (ts, ps, L, _) in matches:
        for k in range(L):
            if ts + k < n_t: used_true[ts + k] = True
            if ps + k < n_p: used_pred[ps + k] = True

    use_sid = true_sentence_ids is not None and pred_sentence_ids is not None

    def same_sentence(t_idx, p_idx, ref_sid):
        if not use_sid:
            return True
        return (true_sentence_ids[t_idx] == ref_sid
                and pred_sentence_ids[p_idx] == ref_sid)

    extensions = []
    for (ts, ps, L, color_idx) in matches:
        ref_sid = (true_sentence_ids[ts]
                   if use_sid and ts < len(true_sentence_ids) else None)

        # Left extension
        lt, lp = ts - 1, ps - 1
        if (0 <= lt and 0 <= lp
                and not used_true[lt] and not used_pred[lp]
                and same_sentence(lt, lp, ref_sid)
                and _is_weak_match(true_seq[lt], pred_seq[lp], weak_map)):
            extensions.append((lt, lp, color_idx))
            used_true[lt] = True
            used_pred[lp] = True

        # Right extension
        rt, rp = ts + L, ps + L
        if (rt < n_t and rp < n_p
                and not used_true[rt] and not used_pred[rp]
                and same_sentence(rt, rp, ref_sid)
                and _is_weak_match(true_seq[rt], pred_seq[rp], weak_map)):
            extensions.append((rt, rp, color_idx))
            used_true[rt] = True
            used_pred[rp] = True

    return extensions

PALETTE = [
    '#FFB3BA', '#BAFFC9', '#BAE1FF', '#FFFFBA', '#FFD9B3',
    '#E1BAFF', '#B3FFE4', '#FFC8DD', '#C8DDFF', '#DDFFC8',
    '#FFE4C8', '#C8FFE4', '#E4C8FF', '#FFC8C8', '#C8FFFF',
    '#F0B3FF', '#FFB3F0', '#B3F0FF', '#F0FFB3', '#FFF0B3',
]


def _shift_for_length(length, shift_by_len):
    """Look up max shift for a given n-gram length using the largest-key-≤-length rule."""
    keys = sorted(k for k in shift_by_len if k <= length)
    return shift_by_len[keys[-1]] if keys else 0


def find_color_matches(true_seq, pred_seq,
                        shift_by_len=None, max_ngram_len=15,
                        true_sentence_ids=None, pred_sentence_ids=None):
    """Find non-overlapping matched n-grams between true_seq and pred_seq.

    shift_by_len: dict mapping n-gram length → max allowed shift. The largest
        key ≤ candidate length wins. 1-grams always require exact position.

    true_sentence_ids / pred_sentence_ids: optional per-position sentence
        labels. When provided, a candidate match is rejected if (a) the
        true range crosses a sentence boundary, (b) the pred range crosses
        a sentence boundary, or (c) the true and pred halves come from
        different sentence ids. This prevents spurious cross-sentence
        n-grams in concatenated patient-level sequences.

    Phase 1a: exact-position n-grams (shift=0, length ≥ 2), GLOBALLY
        longest first.
    Phase 1b: shifted n-grams (length ≥ 2), globally longest first.
    Phase 2:  exact-position 1-grams.

    Returns: list of (true_start, pred_start, length, color_idx).
    """
    if shift_by_len is None:
        shift_by_len = DEFAULT_SHIFT_BY_LEN

    n_t, n_p = len(true_seq), len(pred_seq)
    true_seq, pred_seq = list(true_seq), list(pred_seq)
    matches = []
    color_idx_holder = [0]
    weak_map = _build_weak_equiv_map(DEFAULT_WEAK_EQUIVALENCE)

    def cell_kind(a, b):
        """Per-position classifier: 'exact', 'weak', or None."""
        if a == b: return 'exact'
        if b in weak_map.get(a, ()): return 'weak'
        return None

    def _match_one_slice(t_slice, p_slice, t_offset, p_offset):
        """Run the full matching algorithm on a single (sentence) slice.
        Each match is a 5-tuple: (ts_global, ps_global, L, color_idx,
        cell_types) where cell_types is a tuple of 'exact'/'weak' per
        position. Pure-exact matches have cell_types = ('exact',) * L."""
        n_t_l, n_p_l = len(t_slice), len(p_slice)
        used_true = [False] * n_t_l
        used_pred = [False] * n_p_l

        def claim(ts_l, ps_l, L, cell_types=None):
            if cell_types is None:
                cell_types = ('exact',) * L
            c = color_idx_holder[0]; color_idx_holder[0] += 1
            matches.append((t_offset + ts_l, p_offset + ps_l, L, c, cell_types))
            for k in range(L):
                used_true[ts_l + k] = True
                used_pred[ps_l + k] = True

        # Phase 1 — UNIFIED length-descending pass.
        # For each length L (longest first), for each true_start (L→R), find
        # the best candidate match: any shift in shift_by_len[L] budget,
        # cells either exact or weak. Among candidates, pick the one with
        # fewest weak cells (pure-exact preferred); break shift ties by
        # smallest |shift|. Claims a length-L mixed match in preference to
        # a shorter pure-exact one — longer matches always win.
        for length in range(max_ngram_len, 1, -1):
            max_shift = _shift_for_length(length, shift_by_len)
            shifts = [0] + [s for k in range(1, max_shift + 1) for s in (-k, +k)]
            for true_start in range(n_t_l - length + 1):
                if any(used_true[true_start + k] for k in range(length)):
                    continue
                best = None    # (weak_count, |shift|, shift, types)
                for shift in shifts:
                    pred_start = true_start + shift
                    if pred_start < 0 or pred_start + length > n_p_l:
                        continue
                    if any(used_pred[pred_start + k] for k in range(length)):
                        continue
                    types = []
                    ok = True
                    weak_count = 0
                    for k in range(length):
                        kind = cell_kind(t_slice[true_start + k],
                                          p_slice[pred_start + k])
                        if kind is None:
                            ok = False
                            break
                        if kind == 'weak':
                            weak_count += 1
                        types.append(kind)
                    if not ok:
                        continue
                    cand = (weak_count, abs(shift), shift, tuple(types))
                    if best is None or cand < best:
                        best = cand
                    if weak_count == 0 and shift == 0:
                        break    # can't beat pure-exact at exact position
                if best is not None:
                    weak_count, _absshift, shift, types = best
                    claim(true_start, true_start + shift, length, types)

        # Phase 2 — exact within-slice position 1-grams (no weak 1-grams,
        # those would dilute the visualization)
        for i in range(min(n_t_l, n_p_l)):
            if used_true[i] or used_pred[i]:
                continue
            if t_slice[i] == p_slice[i]:
                claim(i, i, 1)

    if true_sentence_ids is not None and pred_sentence_ids is not None:
        # Walk both sequences sentence-by-sentence in lockstep
        i_t = 0
        i_p = 0
        while i_t < n_t or i_p < n_p:
            sid_t = true_sentence_ids[i_t] if i_t < n_t else None
            sid_p = pred_sentence_ids[i_p] if i_p < n_p else None
            if sid_t is None:   sid = sid_p
            elif sid_p is None: sid = sid_t
            else:               sid = sid_t if sid_t == sid_p else min(sid_t, sid_p)

            t_lo = i_t
            while i_t < n_t and true_sentence_ids[i_t] == sid:
                i_t += 1
            t_hi = i_t

            p_lo = i_p
            while i_p < n_p and pred_sentence_ids[i_p] == sid:
                i_p += 1
            p_hi = i_p

            if t_hi > t_lo or p_hi > p_lo:
                _match_one_slice(true_seq[t_lo:t_hi], pred_seq[p_lo:p_hi],
                                  t_lo, p_lo)
    else:
        _match_one_slice(true_seq, pred_seq, 0, 0)

    # Post-process: merge adjacent same-shift matches with small intron gaps
    matches = _merge_adjacent_with_introns(matches, true_seq, pred_seq, weak_map)

    return matches


# Maximum gap (in cells) between two adjacent matches that we'll merge with
# an intron. Symmetric: applies to both true side and pred side. Set to 0
# to disable intron-based merging.
INTRON_MAX_GAP = 2


def _merge_adjacent_with_introns(matches, true_seq, pred_seq, weak_map,
                                   max_gap=None):
    """Post-process step. If two matches share the same shift and have a
    small symmetric gap between them on both sides, merge them into one
    longer match. Cells in the gap that happen to align at the shared
    shift get classified normally (exact or weak); those that don't get
    tagged as 'intron' (rendered with a slim dotted border + faint fill).

    Concretely, for adjacent matches A=(ts_a, ps_a, L_a, ...) and
    B=(ts_b, ps_b, L_b, ...) at the same shift:
        true_gap = ts_b - (ts_a + L_a)
        pred_gap = ps_b - (ps_a + L_a)
    They merge if true_gap == pred_gap and 0 < true_gap <= max_gap.
    """
    if max_gap is None:
        max_gap = INTRON_MAX_GAP
    if max_gap <= 0 or not matches:
        return matches

    def cell_kind_local(a, b):
        if a == b: return 'exact'
        if b in weak_map.get(a, ()): return 'weak'
        return None

    # Sort by true position
    sorted_m = sorted(matches, key=lambda m: m[0])
    out = []
    cur = sorted_m[0]
    for nxt in sorted_m[1:]:
        cur_ts, cur_ps, cur_L, cur_c, cur_types = cur
        n_ts, n_ps, n_L, n_c, n_types = nxt

        if (n_ps - n_ts) == (cur_ps - cur_ts):
            true_gap = n_ts - (cur_ts + cur_L)
            pred_gap = n_ps - (cur_ps + cur_L)
            if (true_gap == pred_gap
                    and 0 < true_gap <= max_gap):
                # Merge: extend cur to absorb the gap + nxt, classify each
                # gap cell as exact / weak / intron.
                gap_types = []
                for k in range(true_gap):
                    t_idx = cur_ts + cur_L + k
                    p_idx = cur_ps + cur_L + k
                    if t_idx >= len(true_seq) or p_idx >= len(pred_seq):
                        gap_types.append('intron')
                        continue
                    kind = cell_kind_local(true_seq[t_idx], pred_seq[p_idx])
                    gap_types.append(kind if kind is not None else 'intron')

                merged_types = tuple(cur_types) + tuple(gap_types) + tuple(n_types)
                cur = (cur_ts, cur_ps, cur_L + true_gap + n_L,
                       cur_c, merged_types)
                continue

        out.append(cur)
        cur = nxt
    out.append(cur)
    return out


def _cell_html(content, color=None, is_pos=False, is_label=False,
                cell_kind='exact'):
    """cell_kind: 'exact' = solid background fill; 'weak' = dashed border
    in the match color, light fill, italic text. Only relevant when color
    is set."""
    base = ("text-align:center; padding:3px 8px; border:1px solid #eee; "
            "font-family:monospace;")
    if is_label:
        return (f'<td style="font-family:monospace; font-weight:bold; '
                f'padding-right:10px; text-align:right;">{content}</td>')
    if is_pos:
        return f'<td style="{base} font-size:10px; color:#888;">{content}</td>'
    if color is not None:
        if cell_kind == 'weak':
            # Dashed border in match color, very light fill, italic
            return (f'<td style="text-align:center; padding:3px 8px; '
                    f'border:2px dashed {color}; '
                    f'background:{color}30; '            # 30 = ~19% opacity
                    f'font-family:monospace; font-style:italic;">{content}</td>')
        if cell_kind == 'intron':
            # Slim dotted border in match color, very faint fill, gray italic.
            # Says "this cell is part of the match group but didn't directly
            # align — it's a skipped/intronic cell."
            return (f'<td style="text-align:center; padding:3px 8px; '
                    f'border:1px dotted {color}; '
                    f'background:{color}15; '            # 15 = ~8% opacity
                    f'font-family:monospace; font-style:italic; '
                    f'color:#888;">{content}</td>')
        return (f'<td style="{base} background:{color}; '
                f'font-weight:bold;">{content}</td>')
    return f'<td style="{base} color:#bbb;">{content}</td>'


def display_matched_sequences(true_seq, pred_seq, matches,
                                max_per_line=20,
                                true_sentence_ids=None,
                                pred_sentence_ids=None):
    """Render true_seq vs pred_seq as HTML with matches highlighted.

    If sentence ids are provided, render ONE table per sentence: each
    sentence's true & pred slices are padded independently to the longer of
    the two, then chunked into rows of `max_per_line`. This avoids the
    misleading "column = global flat-list position" layout that drifts when
    pred and true have different per-sentence lengths.

    Without sentence ids, falls back to flat-list rendering (legacy behavior).
    """
    from IPython.display import HTML, display

    n_t, n_p = len(true_seq), len(pred_seq)
    true_color = [None] * n_t
    pred_color = [None] * n_p
    true_kind  = ['exact'] * n_t
    pred_kind  = ['exact'] * n_p
    for m in matches:
        # Backward compat: 4-tuple (no cell_types) → all 'exact'
        if len(m) == 5:
            ts, ps, L, color_idx, cell_types = m
        else:
            ts, ps, L, color_idx = m
            cell_types = ('exact',) * L
        c = PALETTE[color_idx % len(PALETTE)]
        for k in range(L):
            if ts + k < n_t:
                true_color[ts + k] = c
                true_kind[ts + k]  = cell_types[k]
            if ps + k < n_p:
                pred_color[ps + k] = c
                pred_kind[ps + k]  = cell_types[k]

    chunks = []

    def render_sentence_block(t_lo, t_hi, p_lo, p_hi, sid=None):
        """Render one sentence's true/pred slice as table(s)."""
        len_t = t_hi - t_lo
        len_p = p_hi - p_lo
        L = max(len_t, len_p)
        if sid is not None:
            chunks.append(
                f'<div style="margin:8px 0 2px 0; font-family:monospace; '
                f'font-size:11px; color:#555;">'
                f'sentence {sid}  (true={len_t}, pred={len_p})'
                f'</div>'
            )
        for chunk_start in range(0, L, max_per_line):
            chunk_end = min(chunk_start + max_per_line, L)
            pos_row, true_row, pred_row = '', '', ''
            for k in range(chunk_start, chunk_end):
                pos_row += _cell_html(str(k), is_pos=True)
                if k < len_t:
                    gi = t_lo + k
                    true_row += _cell_html(true_seq[gi], true_color[gi],
                                            cell_kind=true_kind[gi])
                else:
                    true_row += _cell_html('', None)
                if k < len_p:
                    gi = p_lo + k
                    pred_row += _cell_html(pred_seq[gi], pred_color[gi],
                                            cell_kind=pred_kind[gi])
                else:
                    pred_row += _cell_html('', None)
            chunks.append(f"""
            <table style="border-collapse:collapse; margin-bottom:4px">
              <tr>{_cell_html("pos",  is_label=True)}{pos_row}</tr>
              <tr>{_cell_html("true", is_label=True)}{true_row}</tr>
              <tr>{_cell_html("pred", is_label=True)}{pred_row}</tr>
            </table>
            """)

    if true_sentence_ids is not None and pred_sentence_ids is not None:
        # Walk both sequences sentence-by-sentence in lockstep
        i_t = 0
        i_p = 0
        while i_t < n_t or i_p < n_p:
            sid_t = true_sentence_ids[i_t] if i_t < n_t else None
            sid_p = pred_sentence_ids[i_p] if i_p < n_p else None

            # Pick which sentence to render next: in our pipeline they're
            # always added in matching order, so when both exist they should
            # equal each other. If they don't (shouldn't happen), advance the
            # smaller one to resync.
            if sid_t is None:
                sid = sid_p
            elif sid_p is None:
                sid = sid_t
            else:
                sid = sid_t if sid_t == sid_p else min(sid_t, sid_p)

            t_lo = i_t
            while i_t < n_t and true_sentence_ids[i_t] == sid:
                i_t += 1
            t_hi = i_t

            p_lo = i_p
            while i_p < n_p and pred_sentence_ids[i_p] == sid:
                i_p += 1
            p_hi = i_p

            render_sentence_block(t_lo, t_hi, p_lo, p_hi, sid=sid)
    else:
        # Legacy flat-list rendering
        n = max(n_t, n_p)
        for chunk_start in range(0, n, max_per_line):
            chunk_end = min(chunk_start + max_per_line, n)
            idxs = range(chunk_start, chunk_end)
            pos_row  = ''.join(_cell_html(str(i), is_pos=True) for i in idxs)
            true_row = ''.join(_cell_html(
                                    true_seq[i] if i < n_t else '',
                                    true_color[i] if i < n_t else None,
                                    cell_kind=(true_kind[i] if i < n_t else 'exact'))
                                for i in idxs)
            pred_row = ''.join(_cell_html(
                                    pred_seq[i] if i < n_p else '',
                                    pred_color[i] if i < n_p else None,
                                    cell_kind=(pred_kind[i] if i < n_p else 'exact'))
                                for i in idxs)
            chunks.append(f"""
            <table style="border-collapse:collapse; margin-bottom:6px">
              <tr>{_cell_html("pos",  is_label=True)}{pos_row}</tr>
              <tr>{_cell_html("true", is_label=True)}{true_row}</tr>
              <tr>{_cell_html("pred", is_label=True)}{pred_row}</tr>
            </table>
            """)
    display(HTML(''.join(chunks)))


def _collapse_consecutive_repeats(seq, sids=None):
    """Collapse consecutive duplicates: ['n','n','ɛ','n']  → ['n','ɛ','n'].
    Returns (collapsed_seq, collapsed_sids). If sids is None, returns
    (collapsed_seq, None)."""
    out = []
    out_sids = [] if sids is not None else None
    prev = object()    # sentinel — never matches
    for i, x in enumerate(seq):
        if x != prev:
            out.append(x)
            if sids is not None:
                out_sids.append(sids[i])
        prev = x
    return out, out_sids


def show_matched_sequences(pipeline, pid,
                            shift_by_len=None,
                            max_per_line=50, n_phonemes=None,
                            collapse_repeats=True):
    """Show one patient's true vs pred phoneme sequence with colored
    n-gram matches (variable shift per length).

    collapse_repeats: if True (default), consecutive repeated phonemes in
    BOTH true and pred are collapsed to one before matching/rendering.
    `n n n` becomes `n`. Sentence ids are preserved (the first sid of each
    run wins). Affects only the visualization — `pipeline.patient_results`
    is unchanged.
    """
    res  = pipeline.patient_results[pid]
    true = list(res['true_labels'])
    pred = list(res['predictions'])
    true_sids = res.get('true_sentence_ids')
    pred_sids = res.get('pred_sentence_ids')

    if n_phonemes is not None:
        true = true[:n_phonemes]; pred = pred[:n_phonemes]
        if true_sids is not None: true_sids = true_sids[:n_phonemes]
        if pred_sids is not None: pred_sids = pred_sids[:n_phonemes]

    if collapse_repeats:
        true, true_sids = _collapse_consecutive_repeats(true, true_sids)
        pred, pred_sids = _collapse_consecutive_repeats(pred, pred_sids)

    matches = find_color_matches(
        true, pred, shift_by_len=shift_by_len,
        true_sentence_ids=true_sids, pred_sentence_ids=pred_sids,
    )

    n_2plus  = sum(1 for m in matches if m[2] >= 2)
    n_1grams = sum(1 for m in matches if m[2] == 1)
    n_covered = sum(m[2] for m in matches)

    n_classes = len(set(true))
    chance = 1.0 / n_classes if n_classes > 0 else 0
    acc = res.get('accuracy', float('nan'))
    lift = acc / chance if chance > 0 else 0
    ed   = res.get('edit_distance', edit_distance(true, pred))
    per  = res.get('per', ed / max(len(true), 1))

    sb = shift_by_len or DEFAULT_SHIFT_BY_LEN
    shift_descr = ', '.join(f'{k}-gram±{v}' for k, v in sorted(sb.items()))

    print(f"\n  {pid}  acc={acc:.2%}  lift={lift:.2f}×  "
          f"({n_classes} classes)   edit={ed}  PER={per:.2%}")
    print(f"  {len(matches)} matched n-grams: "
          f"{n_2plus} of length ≥2  +  {n_1grams} exact 1-grams  "
          f"(shift tolerance: {shift_descr})")
    print(f"  Coverage: {n_covered}/{len(true)} phonemes "
          f"({100*n_covered/max(len(true),1):.1f}% in matched n-grams)")
    print()

    display_matched_sequences(true, pred, matches, max_per_line=max_per_line,
                               true_sentence_ids=true_sids,
                               pred_sentence_ids=pred_sids)


def _time_merge_columns(true_segs, pred_segs, tol_s=0.04):
    """Two-pointer merge by start_s. Returns list of (t_idx_or_None, p_idx_or_None).
    Events with |Δstart_s| ≤ tol_s share a column."""
    def t_of(seg):
        if seg is None: return None
        s = seg[0]
        if s is None or (isinstance(s, float) and np.isnan(s)): return None
        return s
    n_t, n_p = len(true_segs), len(pred_segs)
    cols, i, j = [], 0, 0
    while i < n_t or j < n_p:
        t = t_of(true_segs[i]) if i < n_t else None
        p = t_of(pred_segs[j]) if j < n_p else None
        if t is None and p is None:
            # both have nan/None time — pair them by position
            cols.append((i if i < n_t else None, j if j < n_p else None))
            if i < n_t: i += 1
            if j < n_p: j += 1
        elif p is None or (t is not None and t + tol_s < p):
            cols.append((i, None)); i += 1
        elif t is None or (p is not None and p + tol_s < t):
            cols.append((None, j)); j += 1
        else:
            cols.append((i, j)); i += 1; j += 1
    return cols


def display_matched_sequences_with_times(true_seq, pred_seq, matches,
                                          true_segments=None, pred_segments=None,
                                          max_per_line=20,
                                          true_sentence_ids=None,
                                          pred_sentence_ids=None,
                                          time_align_tol_s=None,
                                          sentence_texts=None,
                                          patient_id=None):
    """Same as display_matched_sequences but adds time rows.

    true_segments / pred_segments: lists of (start_s, end_s) tuples,
    one per phoneme in true_seq / pred_seq. If a value is None or NaN,
    the cell is left blank.
    """
    from IPython.display import HTML, display

    n_t, n_p = len(true_seq), len(pred_seq)
    true_color = [None] * n_t
    pred_color = [None] * n_p
    true_kind  = ['exact'] * n_t
    pred_kind  = ['exact'] * n_p
    for m in matches:
        if len(m) == 5:
            ts, ps, L, color_idx, cell_types = m
        else:
            ts, ps, L, color_idx = m
            cell_types = ('exact',) * L
        c = PALETTE[color_idx % len(PALETTE)]
        for k in range(L):
            if ts + k < n_t:
                true_color[ts + k] = c
                true_kind[ts + k]  = cell_types[k]
            if ps + k < n_p:
                pred_color[ps + k] = c
                pred_kind[ps + k]  = cell_types[k]

    def fmt_time(seg):
        if seg is None: return ''
        s, e = seg
        if s is None or (isinstance(s, float) and np.isnan(s)): return ''
        return f'{s:.2f}'

    chunks = []

    def render_sentence_block(t_lo, t_hi, p_lo, p_hi, sid=None):
        len_t = t_hi - t_lo
        len_p = p_hi - p_lo

        if time_align_tol_s is not None and true_segments is not None and pred_segments is not None:
            # time-merged column layout — each column is a time slot
            cols = _time_merge_columns(
                true_segments[t_lo:t_hi], pred_segments[p_lo:p_hi],
                tol_s=time_align_tol_s)
        else:
            # legacy: column k pairs true[t_lo+k] with pred[p_lo+k]
            L = max(len_t, len_p)
            cols = [((k if k < len_t else None),
                     (k if k < len_p else None)) for k in range(L)]

        L = len(cols)
        if sid is not None:
            mode = " (time-aligned)" if time_align_tol_s is not None else ""
            text = sentence_texts.get(sid) if sentence_texts else None
            text_html = (f'<div style="font-family:sans-serif; font-size:13px; '
                          f'color:#222; margin-top:4px;">{text}</div>'
                          if text else '')
            chunks.append(
                f'<div style="margin:8px 0 2px 0; font-family:monospace; '
                f'font-size:11px; color:#555;">'
                f'{patient_id + "  " if patient_id else ""}'
                f'sentence {sid}  (true={len_t}, pred={len_p}){mode}'
                f'{text_html}'
                f'</div>'
            )
        for chunk_start in range(0, L, max_per_line):
            chunk_end = min(chunk_start + max_per_line, L)
            pos_row, true_row, pred_row = '', '', ''
            t_time_row, p_time_row = '', ''
            for k in range(chunk_start, chunk_end):
                ti, pi = cols[k]
                pos_row += _cell_html(str(k), is_pos=True)
                if ti is not None:
                    gi = t_lo + ti
                    t_time_row += _cell_html(
                        fmt_time(true_segments[gi]) if true_segments else '',
                        is_pos=True)
                    true_row += _cell_html(true_seq[gi], true_color[gi],
                                            cell_kind=true_kind[gi])
                else:
                    t_time_row += _cell_html('', None)
                    true_row += _cell_html('', None)
                if pi is not None:
                    gi = p_lo + pi
                    p_time_row += _cell_html(
                        fmt_time(pred_segments[gi]) if pred_segments else '',
                        is_pos=True)
                    pred_row += _cell_html(pred_seq[gi], pred_color[gi],
                                            cell_kind=pred_kind[gi])
                else:
                    p_time_row += _cell_html('', None)
                    pred_row += _cell_html('', None)
            rows = [
                f'<tr>{_cell_html("pos",  is_label=True)}{pos_row}</tr>',
            ]
            if true_segments is not None:
                rows.append(f'<tr>{_cell_html("t_true (s)", is_label=True)}{t_time_row}</tr>')
            rows.append(f'<tr>{_cell_html("true", is_label=True)}{true_row}</tr>')
            if pred_segments is not None:
                rows.append(f'<tr>{_cell_html("t_pred (s)", is_label=True)}{p_time_row}</tr>')
            rows.append(f'<tr>{_cell_html("pred", is_label=True)}{pred_row}</tr>')
            chunks.append(
                '<table style="border-collapse:collapse; margin-bottom:4px">' +
                ''.join(rows) + '</table>'
            )

    if true_sentence_ids is not None and pred_sentence_ids is not None:
        i_t = 0; i_p = 0
        while i_t < n_t or i_p < n_p:
            sid_t = true_sentence_ids[i_t] if i_t < n_t else None
            sid_p = pred_sentence_ids[i_p] if i_p < n_p else None
            if sid_t is None: sid = sid_p
            elif sid_p is None: sid = sid_t
            elif sid_t == sid_p: sid = sid_t
            else: sid = min(sid_t, sid_p)
            t_lo = i_t
            while i_t < n_t and true_sentence_ids[i_t] == sid: i_t += 1
            p_lo = i_p
            while i_p < n_p and pred_sentence_ids[i_p] == sid: i_p += 1
            render_sentence_block(t_lo, i_t, p_lo, i_p, sid=sid)
    else:
        render_sentence_block(0, n_t, 0, n_p)
    display(HTML(''.join(chunks)))


def show_matched_sequences_with_times(pipeline, pid,
                                       shift_by_len=None,
                                       max_per_line=30,
                                       n_phonemes=None,
                                       collapse_repeats=True,
                                       time_align_tol_s=None):
    """Same as show_matched_sequences but renders time rows for true and pred
    phonemes. Uses 'true_segments' and 'pred_segments' fields from
    pipeline.patient_results[pid] if present.
    """
    res = pipeline.patient_results[pid]
    true       = list(res['true_labels'])
    pred       = list(res['predictions'])
    true_sids  = res.get('true_sentence_ids')
    pred_sids  = res.get('pred_sentence_ids')
    true_segs  = res.get('true_segments')
    pred_segs  = res.get('pred_segments')

    if n_phonemes is not None:
        true = true[:n_phonemes]; pred = pred[:n_phonemes]
        if true_sids is not None: true_sids = true_sids[:n_phonemes]
        if pred_sids is not None: pred_sids = pred_sids[:n_phonemes]
        if true_segs is not None: true_segs = true_segs[:n_phonemes]
        if pred_segs is not None: pred_segs = pred_segs[:n_phonemes]

    if collapse_repeats:
        # collapse phoneme sequences AND keep the first time of each run
        true, true_sids, true_segs = _collapse_consecutive_repeats_with_segs(
            true, true_sids, true_segs)
        pred, pred_sids, pred_segs = _collapse_consecutive_repeats_with_segs(
            pred, pred_sids, pred_segs)

    matches = find_color_matches(
        true, pred, shift_by_len=shift_by_len,
        true_sentence_ids=true_sids, pred_sentence_ids=pred_sids,
    )

    n_2plus  = sum(1 for m in matches if m[2] >= 2)
    n_1grams = sum(1 for m in matches if m[2] == 1)
    n_covered = sum(m[2] for m in matches)

    n_classes = len(set(true))
    chance = 1.0 / n_classes if n_classes > 0 else 0
    acc = res.get('accuracy', float('nan'))
    lift = acc / chance if chance > 0 else 0
    ed   = res.get('edit_distance', edit_distance(true, pred))
    per  = res.get('per', ed / max(len(true), 1))

    sb = shift_by_len or DEFAULT_SHIFT_BY_LEN
    shift_descr = ', '.join(f'{k}-gram±{v}' for k, v in sorted(sb.items()))

    print(f"\n  {pid}  acc={acc:.2%}  lift={lift:.2f}×  "
          f"({n_classes} classes)   edit={ed}  PER={per:.2%}")
    print(f"  {len(matches)} matched n-grams: "
          f"{n_2plus} of length ≥2  +  {n_1grams} exact 1-grams  "
          f"(shift tolerance: {shift_descr})")
    print(f"  Coverage: {n_covered}/{len(true)} phonemes "
          f"({100*n_covered/max(len(true),1):.1f}% in matched n-grams)")
    print()

    # Build {sid: text} from pipeline.split_result for this patient
    sentence_texts = {}
    try:
        sl = pipeline.split_result['word_segments_dict'][pid]['sentence_list']
        for i, entry in enumerate(sl):
            if isinstance(entry, dict):
                sentence_texts[i] = entry.get('text', '')
            else:
                sentence_texts[i] = str(entry)
    except Exception:
        pass

    display_matched_sequences_with_times(
        true, pred, matches,
        true_segments=true_segs, pred_segments=pred_segs,
        max_per_line=max_per_line,
        true_sentence_ids=true_sids,
        pred_sentence_ids=pred_sids,
        time_align_tol_s=time_align_tol_s,
        sentence_texts=sentence_texts,
        patient_id=pid,
    )


def _collapse_consecutive_repeats_with_segs(seq, sids, segs):
    """Like _collapse_consecutive_repeats but also collapses parallel segs list.
    Keeps the start_s of the first occurrence and end_s of the last.
    If segs is shorter than seq, missing entries are filled with (nan, nan)."""
    if not seq:
        return seq, sids, segs
    def _get_seg(i):
        if segs is None: return None
        if i < len(segs): return list(segs[i])
        return [float('nan'), float('nan')]
    out_seq = [seq[0]]
    out_sids = [sids[0]] if sids is not None else None
    out_segs = [_get_seg(0)] if segs is not None else None
    for i in range(1, len(seq)):
        same = (seq[i] == out_seq[-1] and
                (sids is None or sids[i] == out_sids[-1]))
        if same:
            if segs is not None:
                end = _get_seg(i)[1]
                out_segs[-1][1] = end
        else:
            out_seq.append(seq[i])
            if sids is not None: out_sids.append(sids[i])
            if segs is not None: out_segs.append(_get_seg(i))
    if segs is not None: out_segs = [tuple(s) for s in out_segs]
    return out_seq, out_sids, out_segs


def show_all_patients_with_times(pipeline, shift_by_len=None, max_per_line=30,
                                  n_phonemes=None, collapse_repeats=True,
                                  time_align_tol_s=None):
    """Time-aware version of show_all_patients."""
    pids = sorted(pipeline.patient_results.keys())
    for pid in pids:
        show_matched_sequences_with_times(pipeline, pid,
                                           shift_by_len=shift_by_len,
                                           max_per_line=max_per_line,
                                           n_phonemes=n_phonemes,
                                           collapse_repeats=collapse_repeats,
                                           time_align_tol_s=time_align_tol_s)


def show_all_patients(pipeline, shift_by_len=None, max_per_line=50,
                       n_phonemes=None, collapse_repeats=True):
    """Loop all patients in pipeline.patient_results and print per-patient
    edit-distance summary at the end."""
    pids = sorted(pipeline.patient_results.keys())
    for pid in pids:
        show_matched_sequences(pipeline, pid,
                                shift_by_len=shift_by_len,
                                max_per_line=max_per_line,
                                n_phonemes=n_phonemes,
                                collapse_repeats=collapse_repeats)

    # Per-patient edit-distance / PER summary
    print("\n  " + "=" * 60)
    print(f"  Per-patient summary")
    print("  " + "=" * 60)
    print(f"  {'pid':<5} {'n_true':>7} {'n_pred':>7} {'edit':>6} "
          f"{'PER':>8} {'acc':>7}")
    print("  " + "-" * 50)
    eds, pers, accs = [], [], []
    for pid in pids:
        r = pipeline.patient_results[pid]
        ed  = r.get('edit_distance', edit_distance(r['true_labels'],
                                                    r['predictions']))
        per = r.get('per', ed / max(len(r['true_labels']), 1))
        acc = r.get('accuracy', float('nan'))
        eds.append(ed); pers.append(per); accs.append(acc)
        print(f"  {pid:<5} {len(r['true_labels']):>7} "
              f"{len(r['predictions']):>7} {ed:>6} {per:>7.2%} {acc:>6.2%}")
    print("  " + "-" * 50)
    print(f"  {'mean':<5} {'':>7} {'':>7} {np.mean(eds):>6.1f} "
          f"{np.mean(pers):>7.2%} {np.mean(accs):>6.2%}")


if __name__ == '__main__':
    results = main()
