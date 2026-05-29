# Converted from MFA_CRF.ipynb

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

from run_pipeline import run_path_b, _run_crf_experiment

cached_train, cached_test = run_path_b(pipeline, run_config)

pipeline.patient_results = _run_crf_experiment(pipeline, run_config)

# Visualize MFA-CRF matched sequences
# ============================================================
# Sentence ids are now emitted directly by _run_crf_experiment
# (true_sentence_ids / pred_sentence_ids), carried from build_mfa_features
# through step5. No post-hoc reconstruction sidecar is needed — the old
# word-occurrence sidecar assigned scrambled, near-per-word ids that
# fragmented matches down to the word level.
n_with_sids = sum('true_sentence_ids' in r
                  for r in pipeline.patient_results.values())
print(f"sentence ids present for "
      f"{n_with_sids}/{len(pipeline.patient_results)} patients")

from e2e_brain_decoder import edit_distance, show_matched_sequences_with_times

for pid in sorted(pipeline.patient_results):
    show_matched_sequences_with_times(pipeline, pid,
                                      max_per_line=50,
                                      collapse_repeats=True,
                                      time_align_tol_s=0.10)

# %% Cell A (fast) — extract sentence HG ONCE, shift HG frames per permutation
import os, io, contextlib
import numpy as np
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn_crfsuite import CRF
from config import DUTCH_30_PATH
from run_pipeline import load_mfa_alignments
from extract_features import extractHG


def _crf_to_seqs(X, labels, words):
    seqs, ys, cx, cy, prev = [], [], [], [], None
    for x, l, w in zip(X, labels, words):
        if w != prev and prev is not None and cx:
            seqs.append(cx); ys.append(cy); cx, cy = [], []
        cx.append({f'f{j}': float(v) for j, v in enumerate(x)}); cy.append(l); prev = w
    if cx: seqs.append(cx); ys.append(cy)
    return seqs, ys


def pvalue_crf_shift_features(pid, pipeline, run_config, n_perm=500, seed=0,
                              n_pca=50, buffer_frac=0.10, verbose=False):
    rng = np.random.RandomState(seed)
    cfg = pipeline.config
    eeg_sr, win_s, shift_s = cfg.eeg_sr, cfg.window_length, cfg.frameshift
    win = int(round(win_s * eeg_sr)); hop = int(round(shift_s * eeg_sr))
    model_order = run_config['stacking_order']; step_size = run_config['stacking_step_size']
    min_class = run_config.get('min_class_samples', 5)

    # ── frozen model on REAL train features ──────────────────────────────────
    trm = [p == pid for p in pipeline.train['phoneme_participant_ids']]
    tr_feat = [pipeline.train['features'][i]       for i, m in enumerate(trm) if m]
    tr_lbl  = [pipeline.train['phoneme_labels'][i] for i, m in enumerate(trm) if m]
    tr_wrd  = [pipeline.train['phoneme_words'][i]  for i, m in enumerate(trm) if m]
    valid = {c for c, n in Counter(tr_lbl).items() if n >= min_class}
    ktr = [i for i, l in enumerate(tr_lbl) if l in valid]
    tr_feat = [tr_feat[i] for i in ktr]; tr_lbl = [tr_lbl[i] for i in ktr]; tr_wrd = [tr_wrd[i] for i in ktr]
    if len(tr_feat) < 10:
        return {'error': f'too few train ({len(tr_feat)})'}
    Xtr = np.array([np.asarray(f).flatten() for f in tr_feat])
    scaler = StandardScaler().fit(Xtr); Xtr = scaler.transform(Xtr)
    pca = PCA(n_components=min(n_pca, Xtr.shape[1], Xtr.shape[0])).fit(Xtr); Xtr = pca.transform(Xtr)
    crf = CRF(algorithm='lbfgs', c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=True)
    crf.fit(*_crf_to_seqs(Xtr, tr_lbl, tr_wrd)); classes = set(crf.classes_)

    def ce_of(feats, labels, words):
        k = [i for i, l in enumerate(labels) if l in valid]
        if len(k) < 5:
            return None
        feats = [feats[i] for i in k]; labels = [labels[i] for i in k]; words = [words[i] for i in k]
        X = pca.transform(scaler.transform(np.array([np.asarray(f).flatten() for f in feats])))
        Xs, ys = _crf_to_seqs(X, labels, words)
        marg = crf.predict_marginals(Xs); s, n = 0.0, 0
        for ms, gs in zip(marg, ys):
            for tm, g in zip(ms, gs):
                if g in classes:
                    s += -np.log(max(tm.get(g, 1e-12), 1e-12)); n += 1
        return s / max(n, 1)

    # ── load raw ONCE; extract each test sentence's HG ONCE; precompute frame ranges
    wd = pipeline.split_result['word_segments_dict'][pid]
    words_dict, sentence_list = wd['words'], wd['sentence_list']
    def sent_ids(sp):
        s = set()
        for w, idxs in sp.items():
            for i in idxs:
                s.add(words_dict[w]['instances'][i]['sentence_idx'])
        return s
    mfa = load_mfa_alignments(pid)
    tr_ids = sent_ids(pipeline.split_result['train'][pid])
    te_ids = sent_ids(pipeline.split_result['test'][pid])
    overlap = tr_ids & te_ids
    if overlap:
        return {'error': f'train/test sentence overlap: {len(overlap)} '
                         f'sentence(s), e.g. {sorted(overlap)[:5]}'}
    test_sids = sorted(te_ids & set(mfa))
    raw = np.load(os.path.join(DUTCH_30_PATH, 'raw', f'{pid}_sEEG.npy'))
    if hasattr(pipeline, 'patient_data') and pid in pipeline.patient_data:
        pdat = pipeline.patient_data[pid]
        if 'channel_mask' in pdat:        raw = raw[:, pdat['channel_mask']]
        elif 'included_channels' in pdat: raw = raw[:, pdat['included_channels']]

    sent_HG, ph_ranges = {}, {}
    for sid in test_sids:
        s = sentence_list[sid]
        if not (isinstance(s, dict) and s['stim_end_idx'] <= raw.shape[0]):
            continue
        hg = extractHG(raw[s['stim_start_idx']:s['stim_end_idx']], eeg_sr,
                       windowLength=win_s, frameshift=shift_s).astype(np.float32)
        T = hg.shape[0]
        if T == 0:                      # sentence too short to yield any HG frame
            continue
        rngs = []
        for ph in mfa[sid]:
            k_s = int(round((ph['start_s'] * eeg_sr - win / 2) / hop))
            k_e = int(round((ph['end_s']   * eeg_sr - win / 2) / hop))
            k_s = min(max(0, k_s), T - 1)          # 0 <= k_s <= T-1
            k_e = min(max(k_s, k_e), T - 1)        # k_s <= k_e <= T-1  -> >=1 frame
            rngs.append((ph['phone'], ph['word'].lower() if ph['word'] else '?', k_s, k_e))
        sent_HG[sid] = hg; ph_ranges[sid] = rngs
    test_sids = [s for s in test_sids if s in sent_HG]
    del raw

    # ── build test feats from (optionally frame-shifted) sentence HG, real step5
    def build_and_step5(shift):
        d = {k: [] for k in ('features', 'phoneme_labels', 'phoneme_words', 'phoneme_positions',
                             'phoneme_participant_ids', 'phoneme_sentence_indices')}
        for sid in test_sids:
            hg = sent_HG[sid]
            if shift:
                T = hg.shape[0]; buf = max(1, int(np.floor(buffer_frac * T)))
                if T - 2 * buf >= 1:
                    hg = np.roll(hg, int(rng.randint(buf, T - buf + 1)), axis=0)
            for phone, word, k_s, k_e in ph_ranges[sid]:
                feat = hg[k_s:k_e + 1]
                if feat.shape[0] == 0:          # safety; should not happen after clamping
                    continue
                d['features'].append(feat)
                d['phoneme_labels'].append(phone); d['phoneme_words'].append(word)
                d['phoneme_positions'].append(0); d['phoneme_participant_ids'].append(pid)
                d['phoneme_sentence_indices'].append(sid)
        saved_tr, saved_te = pipeline.train, pipeline.test
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                pipeline.train = None; pipeline.test = d
                pipeline.step5a_filter_by_frame_count(min_frames=run_config['min_frames'],
                                                      max_frames=run_config['max_frames'])
                pipeline.step5b_stack_features(model_order=model_order, step_size=step_size)
                pipeline.step5c_collapse_to_phoneme_level()
                out = (list(pipeline.test['features']), list(pipeline.test['phoneme_labels']),
                       list(pipeline.test['phoneme_words']))
        finally:
            pipeline.train, pipeline.test = saved_tr, saved_te
        return out

    # ── observed + null ──────────────────────────────────────────────────────
    ce_obs = ce_of(*build_and_step5(shift=False))
    if ce_obs is None:
        return {'error': 'too few test phonemes'}
    nulls = np.empty(n_perm)
    for b in range(n_perm):
        nulls[b] = ce_of(*build_and_step5(shift=True))
        if verbose and ((b + 1) % 100 == 0 or b == 0):
            print(f"    perm {b+1}/{n_perm}  null CE={nulls[b]:.4f}", flush=True)

    z = (nulls.mean() - ce_obs) / (nulls.std(ddof=1) + 1e-9)
    p = (np.sum(nulls <= ce_obs) + 1) / (n_perm + 1)
    return {'pid': pid, 'ce_obs': float(ce_obs), 'null_mean': float(nulls.mean()),
            'null_std': float(nulls.std(ddof=1)), 'z': float(z), 'p_one_sided': float(p),
            'n_perm': n_perm}

# Permutations test ~40 min
import numpy as np, scipy.stats as ss

crf_shift_results = {}
print(f"{'pid':<5} {'CE_obs':>8} {'null_mu':>8} {'null_sd':>8} {'z':>7} {'p':>9}")
print('-' * 55)
for pid in sorted(pipeline.patient_results):
    r = pvalue_crf_shift_features(pid, pipeline, run_config, n_perm=2000, seed=0)
    if 'error' in r:
        print(f"{pid:<5} SKIP — {r['error']}"); continue
    crf_shift_results[pid] = r
    print(f"{pid:<5} {r['ce_obs']:8.3f} {r['null_mean']:8.3f} {r['null_std']:8.4f} "
          f"{r['z']:+7.2f} {r['p_one_sided']:9.4f}")

pv   = np.clip([r['p_one_sided'] for r in crf_shift_results.values()], 1e-300, 1.0)
q_bh = ss.false_discovery_control(pv, method='bh')
chi2 = -2 * np.log(pv).sum(); df = 2 * len(pv)
print('-' * 55)
print(f"BH-FDR significant: {(q_bh < 0.05).sum()}/{len(pv)}")
print(f"Fisher combined p:  {1 - ss.chi2.cdf(chi2, df):.2e}")
