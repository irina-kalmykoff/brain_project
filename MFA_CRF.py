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
run_config['stacking_order']     = 7
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

# from run_pipeline import run_path_b, _run_crf_experiment

# cached_train, cached_test = run_path_b(pipeline, run_config)

# pipeline.patient_results = _run_crf_experiment(pipeline, run_config)

from run_pipeline import run_path_b
run_path_b(pipeline, run_config)          # builds pipeline.train/test + stacks at order=7

# then your CRF block as-is:
from run_pipeline import _run_crf_experiment
pipeline.patient_results = {}
crf_results = _run_crf_experiment(pipeline, run_config)

# verify no overlap between train and test sentences
for pid in sorted(pipeline.split_result['test']):
    wd = pipeline.split_result['word_segments_dict'][pid]
    words_dict, sl = wd['words'], wd['sentence_list']
    def side_texts(side):
        s = set()
        for w, idxs in pipeline.split_result[side].get(pid, {}).items():
            for i in idxs:
                sid = words_dict[w]['instances'][i]['sentence_idx']
                s.add(sl[sid]['text'] if isinstance(sl[sid], dict) else sl[sid])
        return s
    te = side_texts('test')
    print(f"{pid}: {len(side_texts('train') & te)} of {len(te)} test texts in train")

# Permutation test, features shifted
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
                              n_pca=50, buffer_frac=0.10, statistic='ce', verbose=False):
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
    pca = PCA(n_components=min(n_pca, Xtr.shape[1], Xtr.shape[0]), random_state=seed).fit(Xtr); Xtr = pca.transform(Xtr)
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


    def acc_of(feats, labels, words):
        k = [i for i, l in enumerate(labels) if l in valid]
        if len(k) < 5:
            return None
        feats = [feats[i] for i in k]; labels = [labels[i] for i in k]; words = [words[i] for i in k]
        X = pca.transform(scaler.transform(np.array([np.asarray(f).flatten() for f in feats])))
        Xs, ys = _crf_to_seqs(X, labels, words)
        yp = [p for seq in crf.predict(Xs) for p in seq]
        yt = [l for seq in ys           for l in seq]
        return float(np.mean([a == b for a, b in zip(yp, yt)]))
        

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
    score_fn         = acc_of if statistic == 'accuracy' else ce_of
    higher_is_better = (statistic == 'accuracy')

    obs = score_fn(*build_and_step5(shift=False))
    if obs is None or not np.isfinite(obs):
        return {'error': 'observed statistic undefined (too few test phonemes)'}

    null_list, n_bad = [], 0
    for b in range(n_perm):
        v = score_fn(*build_and_step5(shift=True))
        if v is None or not np.isfinite(v):     # shifted build fell below the 5-phoneme floor
            n_bad += 1
            continue
        null_list.append(v)
        if verbose and ((b + 1) % 100 == 0 or b == 0):
            print(f"    perm {b+1}/{n_perm}  null {statistic}={v:.4f}", flush=True)

    nulls = np.asarray(null_list, dtype=float)
    if nulls.size < max(20, n_perm // 2):       # bail if most perms were invalid
        return {'error': f'too many invalid null perms ({n_bad}/{n_perm})'}

    n_eff = nulls.size
    if higher_is_better:                                   # accuracy: obs ABOVE null = good
        z = (obs - nulls.mean()) / (nulls.std(ddof=1) + 1e-9)
        p = (np.sum(nulls >= obs) + 1) / (n_eff + 1)
    else:                                                  # CE: obs BELOW null = good
        z = (nulls.mean() - obs) / (nulls.std(ddof=1) + 1e-9)
        p = (np.sum(nulls <= obs) + 1) / (n_eff + 1)

    return {'pid': pid, 'statistic': statistic, 'ce_obs': float(obs),   # 'ce_obs' = observed stat
            'null_mean': float(nulls.mean()), 'null_std': float(nulls.std(ddof=1)),
            'z': float(z), 'p_one_sided': float(p),
            'n_perm': int(n_eff), 'n_bad': int(n_bad)}

# Permutations test ~40 min
import numpy as np, scipy.stats as ss

pipeline.patient_results = crf_results
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

crf_acc = {}
print(f"{'pid':<5} {'acc_obs':>8} {'null_mu':>8} {'null_sd':>8} {'z':>7} {'p':>9}")
print('-' * 56)
for pid in sorted(pipeline.patient_results):
    r = pvalue_crf_shift_features(pid, pipeline, run_config,
                                  n_perm=2000, seed=0, statistic='accuracy')
    if 'error' in r:
        print(f"{pid:<5} SKIP — {r['error']}"); continue
    crf_acc[pid] = r
    print(f"{pid:<5} {r['ce_obs']:8.3f} {r['null_mean']:8.3f} {r['null_std']:8.4f} "
          f"{r['z']:+7.2f} {r['p_one_sided']:9.4f}")

pv   = np.clip([r['p_one_sided'] for r in crf_acc.values()], 1e-300, 1.0)
q_bh = ss.false_discovery_control(pv, method='bh')
chi2 = -2 * np.log(pv).sum(); df = 2 * len(pv)
print('-' * 56)
print(f"BH-FDR significant: {(q_bh < 0.05).sum()}/{len(pv)}")
print(f"Fisher combined p:  {1 - ss.chi2.cdf(chi2, df):.2e}")

# write out perm test results for the report (CRF kernel)
import numpy as np, scipy.stats as ss, pickle, os

crf_shift_ce = crf_shift_results          # reuse  CE run
n_expected   = len(crf_shift_ce)

# reuse accuracy else run fresh
if 'crf_acc' in globals() and len(crf_acc) == n_expected:
    print(f"Reusing complete crf_acc ({len(crf_acc)} patients)")
else:
    print("Running accuracy shift test (~40 min)...")
    crf_acc = {}
    print(f"{'pid':<5} {'acc_obs':>8} {'null_mu':>8} {'null_sd':>8} {'z':>7} {'p':>9}\n" + '-'*55)
    for pid in sorted(pipeline.patient_results):
        r = pvalue_crf_shift_features(pid, pipeline, run_config, n_perm=2000, seed=0,
                                      statistic='accuracy')
        if 'error' in r:
            print(f"{pid:<5} SKIP — {r['error']}"); continue
        crf_acc[pid] = r
        print(f"{pid:<5} {r['ce_obs']:8.3f} {r['null_mean']:8.3f} {r['null_std']:8.4f} "
              f"{r['z']:+7.2f} {r['p_one_sided']:9.4f}")

assert len(crf_acc) == n_expected, \
    f"accuracy={len(crf_acc)} vs CE={n_expected} — incomplete run, NOT saving!"

os.makedirs('results', exist_ok=True)
pickle.dump({'acc_pho': crf_acc, 'ce_pho': crf_shift_ce},
            open('results/crf_shift_perm.pkl', 'wb'))
print('saved results/crf_shift_perm.pkl  (acc patients:', len(crf_acc), ')')

# %% CRF consonant confusion matrix — paste after _run_crf_experiment ===========
import numpy as np, matplotlib.pyplot as plt
from collections import Counter
import importlib, phon_helpers; importlib.reload(phon_helpers)
from phon_helpers import manner, place, is_cons

# pull predictions from whichever is populated
src = crf_results if 'crf_results' in dir() else pipeline.patient_results

# pooled consonant->consonant pairs (CRF is 1:1, so just zip)
pairs = Counter()
for pid, r in src.items():
    for g, p in zip(r['true_labels'], r['predictions']):
        if is_cons(g) and is_cons(p):
            pairs[(g, p)] += 1

mord = {'plosive': 0, 'fricative': 1, 'nasal': 2, 'approx': 3}
cons = sorted({c for gp in pairs for c in gp},
              key=lambda c: (mord.get(manner(c), 9), str(place(c)), c))
idx = {c: i for i, c in enumerate(cons)}; n = len(cons)
M = np.zeros((n, n))
for (g, p), v in pairs.items(): M[idx[g], idx[p]] += v
row = M.sum(1, keepdims=True); Mn = np.divide(M, row, where=row > 0)

fig, ax = plt.subplots(figsize=(8.5, 7.5))
im = ax.imshow(Mn, cmap='Blues', vmin=0, vmax=1)
ax.set_xticks(range(n)); ax.set_xticklabels(cons, fontsize=7, rotation=90)
ax.set_yticks(range(n)); ax.set_yticklabels(cons, fontsize=7)
ax.set_xlabel('predicted'); ax.set_ylabel('gold')
ax.set_title('CRF (order 7) consonant confusions, P(pred|gold)\n'
             'ordered by manner/place — diagonal = correct')
for i in range(n):
    for j in range(n):
        if Mn[i, j] >= 0.10:
            ax.text(j, i, f'{Mn[i, j]:.2f}'.lstrip('0'), ha='center', va='center',
                    fontsize=6, color='white' if Mn[i, j] > 0.55 else '#222')
ax.set_xticks(np.arange(-.5, n, 1), minor=True)
ax.set_yticks(np.arange(-.5, n, 1), minor=True)
ax.grid(which='minor', color='0.85', lw=0.5); ax.tick_params(which='minor', length=0)
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig('report/fig_crf_confusion_order7.png', dpi=150, bbox_inches='tight')
plt.show()

# ---- collapse diagnostics (the numbers that actually answer the question) ----
diag = np.diag(Mn)
print(f"mean diagonal (per-class accuracy): {np.nanmean(diag):.3f}")
allpred = Counter(p for r in src.values()
                  for g, p in zip(r['true_labels'], r['predictions']) if is_cons(p))
tot = sum(allpred.values()); top3 = sum(v for _, v in allpred.most_common(3))
print(f"top-3 predicted consonants = {top3/tot:.0%} of all consonant predictions:",
      [f'{k}({v})' for k, v in allpred.most_common(5)])
errs = sorted(((v, g, p) for (g, p), v in pairs.items() if g != p), reverse=True)
print("top off-diagonal confusions (gold→pred):",
      ', '.join(f'{g}→{p}({v})' for v, g, p in errs[:10]))

# %% CRF confusion — recall + precision side by side, with per-class F1 =========
import numpy as np, matplotlib.pyplot as plt
from collections import Counter
import importlib, phon_helpers; importlib.reload(phon_helpers)
from phon_helpers import manner, place, is_cons

src = crf_results if 'crf_results' in dir() else pipeline.patient_results
CONS_ONLY = True

pairs = Counter()
for r in src.values():
    for g, p in zip(r['true_labels'], r['predictions']):
        if (not CONS_ONLY) or (is_cons(g) and is_cons(p)):
            pairs[(g, p)] += 1

mord = {'plosive':0,'fricative':1,'nasal':2,'approx':3,'vowel':4}
syms = sorted({c for gp in pairs for c in gp},
              key=lambda c:(mord.get(manner(c),9), str(place(c)), c))
idx = {c:i for i,c in enumerate(syms)}; n=len(syms)
M = np.zeros((n,n))
for (g,p),v in pairs.items(): M[idx[g],idx[p]] += v

rsum = M.sum(1, keepdims=True)
csum = M.sum(0, keepdims=True)
recall = np.divide(M, rsum, out=np.zeros_like(M), where=rsum > 0)
prec   = np.divide(M, csum, out=np.zeros_like(M), where=csum > 0)

fig, axes = plt.subplots(1, 2, figsize=(17, 7.5))
for ax, Mx, ttl in [(axes[0], recall, 'Recall  P(pred|gold)'),
                    (axes[1], prec,   'Precision  P(gold|pred)')]:
    im = ax.imshow(Mx, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(n)); ax.set_xticklabels(syms, fontsize=7, rotation=90)
    ax.set_yticks(range(n)); ax.set_yticklabels(syms, fontsize=7)
    ax.set_xlabel('predicted'); ax.set_ylabel('gold'); ax.set_title(ttl)
    for i in range(n):
        for j in range(n):
            if Mx[i,j] >= 0.10:
                ax.text(j,i,f'{Mx[i,j]:.2f}'.lstrip('0'),ha='center',va='center',
                        fontsize=6,color='white' if Mx[i,j]>0.55 else '#222')
    ax.set_xticks(np.arange(-.5,n,1),minor=True); ax.set_yticks(np.arange(-.5,n,1),minor=True)
    ax.grid(which='minor',color='0.85',lw=0.5); ax.tick_params(which='minor',length=0)
    fig.colorbar(im,ax=ax,fraction=0.046,pad=0.04)
plt.tight_layout(); plt.savefig('report/fig_crf_confusion_recall_prec.png',dpi=150,bbox_inches='tight'); plt.show()

print(f"{'sym':<4}{'recall':>7}{'prec':>7}{'F1':>7}{'support':>8}")
for c in syms:
    i=idx[c]; rec=recall[i,i]; pr=prec[i,i]
    f1=2*rec*pr/(rec+pr) if (rec+pr)>0 else 0
    print(f"{c:<4}{rec:>7.2f}{pr:>7.2f}{f1:>7.2f}{int(M[i].sum()):>8}")
print(f"\nmacro-F1: {np.mean([2*recall[i,i]*prec[i,i]/(recall[i,i]+prec[i,i]+1e-9) for i in range(n)]):.3f}")

for pid, r in crf_results.items():
    true_labels = r['true_labels']
    acc = r['accuracy']
    from collections import Counter
    # label_counts = Counter(true_labels)
    n_classes = len(set(true_labels))
    chance = 1.0 / n_classes if n_classes > 0 else 0
    lift = acc / chance if chance > 0 else 0

    pipeline.patient_results[pid] = {
        'accuracy': acc,
        'lift': lift,
        'chance': chance,
        'predictions': r['predictions'],
        'true_labels': true_labels,
        'train_size': r['n_train'],
        'test_size': r['n_test'],
        'n_classes': len(set(true_labels)),
    }

pr = run_config['patient_range']
for pid in [f'P{i:02d}' for i in range(pr[0], pr[1] + 1)]:
    if pid in pipeline.patient_results:
        pipeline.step10_visualize_patient(pid, show_table=False)

pipeline.step10_visualize_group()

# Visualize MFA-CRF matched sequences
# ============================================================
# Sentence ids are emitted directly by _run_crf_experiment
# (true_sentence_ids / pred_sentence_ids), carried from build_mfa_features
# through step5. 
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

# contribution of MFA word borders & priors
def contribution_word_borders(pid, pipeline, run_config, n_perm=1000, seed=0, n_pca=50):
    """Frozen CRF (real borders). Null = scramble test word-segment lengths
    WITHIN each sentence. Measures the decode-time value of correct borders."""
    d = _prepare_crf_data(pid, pipeline, run_config, n_pca)
    if d is None: return {'error': 'too few samples'}
    if d['tes'] is None: return {'error': 'no phoneme_sentence_indices — rebuild via run_path_b'}
    rng = np.random.RandomState(seed)
    crf = _fit_crf(*_seqs(d['Xtr'], d['trl'], _runs(d['trw'])))

    # contiguous per-sentence blocks of the test stream
    sids = d['tes']; blocks, i = [], 0
    while i < len(sids):
        j = i
        while j < len(sids) and sids[j] == sids[i]: j += 1
        blocks.append((i, j)); i = j

    def acc(group_lengths):
        seqs, _ = _seqs(d['Xte'], d['tel'], group_lengths)
        yp = [p for s in crf.predict(seqs) for p in s]
        return np.mean([a == b for a, b in zip(yp, d['tel'])])

    real_gl = [L for a, b in blocks for L in _runs(d['tew'][a:b])]
    obs = acc(real_gl)
    nulls = np.empty(n_perm)
    for k in range(n_perm):
        gl = []
        for a, b in blocks:
            r = _runs(d['tew'][a:b]); rng.shuffle(r); gl += r
        nulls[k] = acc(gl)
    z = (obs - nulls.mean()) / (nulls.std(ddof=1) + 1e-9)
    p = (np.sum(nulls >= obs) + 1) / (n_perm + 1)   # borders help if obs > null
    return dict(pid=pid, obs_acc=float(obs), null_mean=float(nulls.mean()),
                null_std=float(nulls.std(ddof=1)), z=float(z), p=float(p),
                border_contribution=float(obs - nulls.mean()), n_perm=n_perm)


def contribution_priors(pid, pipeline, run_config, n_perm=200, seed=0, n_pca=50):
    """Null = shuffle TRAIN labels (keeps class prior, destroys feature->label),
    retrain CRF, decode test. Separates prior-only acc from feature decoding."""
    d = _prepare_crf_data(pid, pipeline, run_config, n_pca)
    if d is None: return {'error': 'too few samples'}
    rng = np.random.default_rng(seed)
    tr_gl, te_gl = _runs(d['trw']), _runs(d['tew'])
    te_seqs, _ = _seqs(d['Xte'], d['tel'], te_gl)

    def fit_predict(train_labels):
        crf = _fit_crf(*_seqs(d['Xtr'], train_labels, tr_gl))
        yp = [p for s in crf.predict(te_seqs) for p in s]
        return np.mean([a == b for a, b in zip(yp, d['tel'])])

    obs = fit_predict(d['trl'])
    chance = 1.0 / len(set(d['tel']))
    nulls = np.empty(n_perm)
    for k in range(n_perm):
        shuf = list(d['trl']); rng.shuffle(shuf)
        nulls[k] = fit_predict(shuf)
    z = (obs - nulls.mean()) / (nulls.std(ddof=1) + 1e-9)
    p = (np.sum(nulls >= obs) + 1) / (n_perm + 1)   # features add beyond prior if obs > null
    return dict(pid=pid, obs_acc=float(obs), prior_only_mean=float(nulls.mean()),
                prior_only_std=float(nulls.std(ddof=1)), chance=float(chance),
                z=float(z), p=float(p),
                prior_contribution=float(nulls.mean() - chance),
                feature_contribution=float(obs - nulls.mean()), n_perm=n_perm)

# contribution ablations: shared helpers  (run me first)
import numpy as np
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn_crfsuite import CRF

def _runs(seq):
    """Run-length list of consecutive-equal items (== to_sequences word grouping)."""
    out, prev, c = [], object(), 0
    for x in seq:
        if x == prev: c += 1
        else:
            if c: out.append(c)
            prev, c = x, 1
    if c: out.append(c)
    return out

def _seqs(X, labels, group_lengths):
    seqs, ys, i = [], [], 0
    for L in group_lengths:
        seqs.append([{f'f{j}': float(v) for j, v in enumerate(X[k])} for k in range(i, i + L)])
        ys.append(list(labels[i:i + L])); i += L
    return seqs, ys

def _fit_crf(Xseq, yseq):
    crf = CRF(algorithm='lbfgs', c1=0.1, c2=0.1, max_iterations=100,
              all_possible_transitions=True)
    crf.fit(Xseq, yseq); return crf

def _prepare_crf_data(pid, pipeline, run_config, n_pca=50):
    """Filter to pid, drop rare classes (train counts), fit StandardScaler+PCA on
    TRAIN, transform test. Returns projected X + labels/words/sentence-ids."""
    mc = run_config.get('min_class_samples', 5)
    def col(split, key):
        m = [p == pid for p in getattr(pipeline, split)['phoneme_participant_ids']]
        return [getattr(pipeline, split)[key][i] for i, x in enumerate(m) if x]
    trf, trl, trw = col('train','features'), col('train','phoneme_labels'), col('train','phoneme_words')
    tef, tel, tew = col('test','features'),  col('test','phoneme_labels'),  col('test','phoneme_words')
    tes = col('test','phoneme_sentence_indices') if 'phoneme_sentence_indices' in pipeline.test else None
    valid = {c for c, n in Counter(trl).items() if n >= mc}
    ktr = [i for i, l in enumerate(trl) if l in valid]
    kte = [i for i, l in enumerate(tel) if l in valid]
    trf=[trf[i] for i in ktr]; trl=[trl[i] for i in ktr]; trw=[trw[i] for i in ktr]
    tef=[tef[i] for i in kte]; tel=[tel[i] for i in kte]; tew=[tew[i] for i in kte]
    if tes is not None: tes=[tes[i] for i in kte]
    if len(trf) < 10 or len(tef) < 5:
        return None
    Xtr = np.array([np.asarray(f).flatten() for f in trf])
    Xte = np.array([np.asarray(f).flatten() for f in tef])
    sc = StandardScaler().fit(Xtr); Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    pca = PCA(n_components=min(n_pca, Xtr.shape[1], Xtr.shape[0])).fit(Xtr)
    return dict(Xtr=pca.transform(Xtr), trl=trl, trw=trw,
                Xte=pca.transform(Xte), tel=tel, tew=tew, tes=tes)

# word border and priors contribution
import numpy as np, scipy.stats as ss

border, prior = {}, {}

print("dprior = lbl_null - chance      dfeat = real - lbl_null      dbord = real - bord_null")
print("(all values are per-phoneme accuracy; lbl_null = label-shuffle null, "
      "bord_null = scrambled-border null)\n")

hdr = (f"{'pid':<5} {'chance':>7} {'lbl_null':>9} {'real':>7} "
       f"{'dprior':>8} {'dfeat':>8} {'p_feat':>8} "
       f"{'bord_null':>10} {'dbord':>8} {'p_bord':>8}")
print(hdr)
print('-' * len(hdr))

for pid in sorted(pipeline.patient_results):
    rb = contribution_word_borders(pid, pipeline, run_config, n_perm=1000)
    rp = contribution_priors(pid, pipeline, run_config, n_perm=200)
    if 'error' in rb or 'error' in rp:
        print(f"{pid:<5} SKIP — {rb.get('error', rp.get('error'))}"); continue
    border[pid], prior[pid] = rb, rp
    print(f"{pid:<5} "
          f"{rp['chance']:7.3f} "
          f"{rp['prior_only_mean']:9.3f} "          # label-shuffle null
          f"{rp['obs_acc']:7.3f} "                  # real
          f"{rp['prior_contribution']:+8.3f} "      # dprior = lbl_null - chance
          f"{rp['feature_contribution']:+8.3f} "    # dfeat  = real - lbl_null
          f"{rp['p']:8.4f} "                         # p_feat
          f"{rb['null_mean']:10.3f} "               # border null
          f"{rb['border_contribution']:+8.3f} "     # dbord  = real - bord_null
          f"{rb['p']:8.4f}")                          # p_bord

def _agg(d, name):
    pv = np.clip([r['p'] for r in d.values()], 1e-300, 1.0)
    q = ss.false_discovery_control(pv, method='bh')
    chi2 = -2 * np.log(pv).sum(); df = 2 * len(pv)
    print(f"  {name:<22} BH-FDR sig {(q < 0.05).sum()}/{len(pv)}   "
          f"Fisher p = {1 - ss.chi2.cdf(chi2, df):.2e}")

print('-' * len(hdr))
_agg(prior,  "features-beyond-prior")
_agg(border, "word-border")

def contribution_phoneme_borders(pid, pipeline, run_config, n_perm=1000, seed=0,
                                 n_pca=50, statistic='accuracy', verbose=False):
    """Frozen CRF. Null = scramble PHONEME segmentation within each sentence
    (permute phoneme durations, re-tile, keep labels/order), RE-EXTRACT features,
    re-decode. Measures the value of correct MFA phoneme boundaries for the FEATURES.
    (contribution_word_borders instead scrambles the CRF sequence grouping.)"""
    rng = np.random.RandomState(seed)
    cfg = pipeline.config
    eeg_sr, win_s, shift_s = cfg.eeg_sr, cfg.window_length, cfg.frameshift
    win = int(round(win_s*eeg_sr)); hop = int(round(shift_s*eeg_sr))
    model_order = run_config['stacking_order']; step_size = run_config['stacking_step_size']
    min_class = run_config.get('min_class_samples', 5)

    trm = [p == pid for p in pipeline.train['phoneme_participant_ids']]
    tr_feat = [pipeline.train['features'][i]       for i, m in enumerate(trm) if m]
    tr_lbl  = [pipeline.train['phoneme_labels'][i] for i, m in enumerate(trm) if m]
    tr_wrd  = [pipeline.train['phoneme_words'][i]  for i, m in enumerate(trm) if m]
    valid = {c for c, n in Counter(tr_lbl).items() if n >= min_class}
    ktr = [i for i, l in enumerate(tr_lbl) if l in valid]
    tr_feat=[tr_feat[i] for i in ktr]; tr_lbl=[tr_lbl[i] for i in ktr]; tr_wrd=[tr_wrd[i] for i in ktr]
    if len(tr_feat) < 10: return {'error': f'too few train ({len(tr_feat)})'}
    Xtr = np.array([np.asarray(f).flatten() for f in tr_feat])
    scaler = StandardScaler().fit(Xtr); Xtr = scaler.transform(Xtr)
    pca = PCA(n_components=min(n_pca, Xtr.shape[1], Xtr.shape[0])).fit(Xtr); Xtr = pca.transform(Xtr)
    crf = CRF(algorithm='lbfgs', c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=True)
    crf.fit(*_crf_to_seqs(Xtr, tr_lbl, tr_wrd)); classes = set(crf.classes_)

    def acc_of(feats, labels, words):
        k = [i for i, l in enumerate(labels) if l in valid]
        if len(k) < 5: return None
        feats=[feats[i] for i in k]; labels=[labels[i] for i in k]; words=[words[i] for i in k]
        X = pca.transform(scaler.transform(np.array([np.asarray(f).flatten() for f in feats])))
        Xs, ys = _crf_to_seqs(X, labels, words)
        yp = [p for s in crf.predict(Xs) for p in s]; yt = [l for s in ys for l in s]
        return float(np.mean([a == b for a, b in zip(yp, yt)]))
    def ce_of(feats, labels, words):
        k = [i for i, l in enumerate(labels) if l in valid]
        if len(k) < 5: return None
        feats=[feats[i] for i in k]; labels=[labels[i] for i in k]; words=[words[i] for i in k]
        X = pca.transform(scaler.transform(np.array([np.asarray(f).flatten() for f in feats])))
        Xs, ys = _crf_to_seqs(X, labels, words); marg = crf.predict_marginals(Xs); s, n = 0.0, 0
        for ms, gs in zip(marg, ys):
            for tm, g in zip(ms, gs):
                if g in classes: s += -np.log(max(tm.get(g, 1e-12), 1e-12)); n += 1
        return s / max(n, 1)
    score_fn = acc_of if statistic == 'accuracy' else ce_of
    higher_is_better = (statistic == 'accuracy')

    wd = pipeline.split_result['word_segments_dict'][pid]
    words_dict, sentence_list = wd['words'], wd['sentence_list']
    def sent_ids(sp):
        s = set()
        for w, idxs in sp.items():
            for i in idxs: s.add(words_dict[w]['instances'][i]['sentence_idx'])
        return s
    mfa = load_mfa_alignments(pid)
    if sent_ids(pipeline.split_result['train'][pid]) & sent_ids(pipeline.split_result['test'][pid]):
        return {'error': 'train/test sentence overlap'}
    test_sids = sorted(sent_ids(pipeline.split_result['test'][pid]) & set(mfa))

    raw = np.load(os.path.join(DUTCH_30_PATH, 'raw', f'{pid}_sEEG.npy'))
    if hasattr(pipeline, 'patient_data') and pid in pipeline.patient_data:
        pdat = pipeline.patient_data[pid]
        if 'channel_mask' in pdat:        raw = raw[:, pdat['channel_mask']]
        elif 'included_channels' in pdat: raw = raw[:, pdat['included_channels']]
    sent_HG, ph_ranges = {}, {}
    for sid in test_sids:
        s = sentence_list[sid]
        if not (isinstance(s, dict) and s['stim_end_idx'] <= raw.shape[0]): continue
        hg = extractHG(raw[s['stim_start_idx']:s['stim_end_idx']], eeg_sr,
                       windowLength=win_s, frameshift=shift_s).astype(np.float32)
        T = hg.shape[0]
        if T == 0: continue
        rngs = []
        for ph in mfa[sid]:
            ks = min(max(0, int(round((ph['start_s']*eeg_sr - win/2)/hop))), T-1)
            ke = min(max(ks, int(round((ph['end_s']*eeg_sr - win/2)/hop))), T-1)
            rngs.append((ph['phone'], ph['word'].lower() if ph['word'] else '?', ks, ke))
        sent_HG[sid] = hg; ph_ranges[sid] = rngs
    test_sids = [s for s in test_sids if s in sent_HG]
    del raw

    def build_and_step5(scramble):
        d = {k: [] for k in ('features','phoneme_labels','phoneme_words','phoneme_positions',
                             'phoneme_participant_ids','phoneme_sentence_indices')}
        for sid in test_sids:
            hg = sent_HG[sid]; T = hg.shape[0]; rngs = ph_ranges[sid]
            if scramble and len(rngs) > 1:
                durs = [ke - ks + 1 for (_, _, ks, ke) in rngs]
                perm = rng.permutation(len(durs))           # shuffle durations -> move cut points
                cur = rngs[0][2]; new = []
                for (ph, w, _, _), idx in zip(rngs, perm):
                    ks = min(cur, T - 1); ke = min(cur + durs[idx] - 1, T - 1)
                    new.append((ph, w, ks, ke)); cur = min(ke + 1, T - 1)
                rngs = new
            for phone, word, ks, ke in rngs:
                feat = hg[ks:ke + 1]
                if feat.shape[0] == 0: continue
                d['features'].append(feat); d['phoneme_labels'].append(phone)
                d['phoneme_words'].append(word); d['phoneme_positions'].append(0)
                d['phoneme_participant_ids'].append(pid); d['phoneme_sentence_indices'].append(sid)
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

    obs = score_fn(*build_and_step5(False))
    if obs is None or not np.isfinite(obs): return {'error': 'observed undefined'}
    null_list, n_bad = [], 0
    for b in range(n_perm):
        v = score_fn(*build_and_step5(True))
        if v is None or not np.isfinite(v): n_bad += 1; continue
        null_list.append(v)
    nulls = np.asarray(null_list, float)
    if nulls.size < max(20, n_perm // 2): return {'error': f'too many invalid perms ({n_bad}/{n_perm})'}
    if higher_is_better:
        z = (obs - nulls.mean()) / (nulls.std(ddof=1) + 1e-9); p = (np.sum(nulls >= obs) + 1) / (nulls.size + 1)
        contrib = obs - nulls.mean()
    else:
        z = (nulls.mean() - obs) / (nulls.std(ddof=1) + 1e-9); p = (np.sum(nulls <= obs) + 1) / (nulls.size + 1)
        contrib = nulls.mean() - obs
    return {'pid': pid, 'statistic': statistic, 'obs_acc': float(obs), 'null_mean': float(nulls.mean()),
            'null_std': float(nulls.std(ddof=1)), 'z': float(z), 'p': float(p),
            'border_contribution': float(contrib), 'n_perm': int(nulls.size), 'n_bad': int(n_bad)}

phon = {}
print(f"{'pid':<5} {'real':>7} {'ph_null':>8} {'dphon':>8} {'p_phon':>8}")
print('-' * 40)
for pid in sorted(pipeline.patient_results):
    r = contribution_phoneme_borders(pid, pipeline, run_config, n_perm=1000)
    if 'error' in r: print(f"{pid:<5} SKIP — {r['error']}"); continue
    phon[pid] = r
    print(f"{pid:<5} {r['obs_acc']:7.3f} {r['null_mean']:8.3f} "
          f"{r['border_contribution']:+8.3f} {r['p']:8.4f}")

# %% Build per-phoneme arrays for the crf_feats export (run AFTER _run_step5abc) ===
import numpy as np

# sanity: confirm the key names in your pipeline splits
print("train keys:", list(pipeline.train.keys()))

vecs, labs, sids, ppid = [], [], [], []
for split in (pipeline.train, pipeline.test):          # pool train+test; probe CVs by sentence
    feats = split['features']
    vecs += [np.asarray(f, np.float32).ravel() for f in feats]   # one flat vector per phoneme
    labs += list(split['phoneme_labels'])
    sids += list(split['phoneme_sentence_indices'])
    ppid += list(split['phoneme_participant_ids'])

print(f"{len(vecs)} phonemes | example dim {vecs[0].shape} | patients {sorted(set(ppid))}")
# at order 7 the dim should be ~n_channels*15 (e.g. 110*15=1650), NOT 4510

import pickle, numpy as np
labs = np.asarray(labs); sids = np.asarray(sids); ppid = np.asarray(ppid)
crf_feats = {pid: {'X': np.array([vecs[i] for i in np.where(ppid == pid)[0]], np.float16),
                   'y': labs[ppid == pid].astype(str), 'sid': sids[ppid == pid].astype(int)}
             for pid in sorted(set(ppid))}
pickle.dump(crf_feats,  open('results/crf_feats.pkl',  'wb'))
pickle.dump(crf_export, open('results/crf_export.pkl', 'wb'))    # if not already saved
print('saved', {k: v['X'].shape for k, v in crf_feats.items()})

# per-patient per-class AUC dict (uses phoneme_auc_perclass + vecs/labs/ppid/sids)
import numpy as np, pickle
crf_pp = {}
for pid in sorted(set(ppid)):
    idx = np.where(ppid == pid)[0]
    Xp = np.array([vecs[i] for i in idx])
    yp = np.array([labs[i]  for i in idx])
    gp = np.asarray(sids)[idx]
    crf_pp[pid] = phoneme_auc_perclass(Xp, yp, gp, n_pca=50)
    print(f"  {pid} done")
pickle.dump(crf_pp, open('results/crf_perpatient_perclass.pkl', 'wb'))
print("saved CRF")

# %% pdat + pair_auc (reconstructed) — run after building vecs/labs/sids/ppid ====
import numpy as np, itertools
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

labs = np.asarray(labs); sids = np.asarray(sids); ppid = np.asarray(ppid)
LONG = {v for v in set(labs.tolist()) if v.endswith('ː')}   # long vowels, data-driven
print("LONG vowels:", sorted(LONG))

def pdat(pid):
    idx = np.where(ppid == pid)[0]
    return np.array([vecs[i] for i in idx], np.float32), labs[idx], sids[idx]

def pair_auc(X, y, sid, a, b, n_pca=50, n_splits=5):
    m = (y == a) | (y == b)
    Xb, yb, gb = X[m], (y[m] == a).astype(int), sid[m]
    na, nb = int(yb.sum()), int(len(yb) - yb.sum())
    if na < 5 or nb < 5:                       # need both classes present
        return np.nan
    k = min(n_splits, len(set(gb)), na, nb)    # can't have more folds than groups/minority
    if k < 2:
        return np.nan
    n_comp = min(n_pca, Xb.shape[1], int(0.7 * len(Xb)) - 1)
    if n_comp < 2:
        return np.nan
    pipe = make_pipeline(StandardScaler(),
                         PCA(n_components=n_comp),
                         LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'))
    try:
        score = cross_val_predict(pipe, Xb, yb, groups=gb,
                                  cv=GroupKFold(n_splits=k), method='decision_function')
    except Exception:
        return np.nan
    return roc_auc_score(yb, score)

import itertools, numpy as np
LONG = ['aː', 'eː', 'iː', 'oː', 'uː', 'yː', 'øː']
def pdat(pid):
    idx = np.where(ppid == pid)[0]
    return (np.array([vecs[i] for i in idx]), np.array([labs[i] for i in idx]), np.asarray(sids)[idx])
rows = []
for a, b in itertools.combinations(LONG, 2):
    vals = [pair_auc(*pdat(pid), a, b) for pid in sorted(set(ppid))]
    vals = [v for v in vals if np.isfinite(v)]
    if len(vals) >= 5: rows.append((f"{a}-{b}", float(np.mean(vals)), len(vals)))
rows.sort(key=lambda r: -r[1])
for nm, a_, n in rows: print(f"{nm:10} CRF AUC={a_:.3f}  (n_pat={n})")
print(f"\nCRF mean long-vowel pairwise AUC = {np.mean([r[1] for r in rows]):.3f}")

import pickle
pickle.dump({nm: a for nm, a, _ in rows}, open('results/crf_vowelpairs.pkl', 'wb'))
print("saved CRF vowel pairs:", len(rows))

import os, pickle
os.makedirs('results', exist_ok=True)
crf_export = {pid: {'true_labels': list(r['true_labels']),
                    'predictions': list(r['predictions']),
                    'true_sentence_ids': list(r['true_sentence_ids']),
                    'pred_sentence_ids': list(r['pred_sentence_ids'])}
              for pid, r in pipeline.patient_results.items() if 'true_sentence_ids' in r}
with open('results/crf_patient_results.pkl', 'wb') as f:
    pickle.dump(crf_export, f)
print("saved", list(crf_export))

import importlib, phon_helpers; importlib.reload(phon_helpers)
from phon_helpers import cv

import pickle; pickle.dump(crf_export,  open('results/crf_export.pkl',  'wb'))

# verify the decoded outputs are actually there
print("crf_results:", len(crf_results), "patients →", sorted(crf_results)[:4])
assert len(crf_results) > 0, "crf_results is empty — _run_crf_experiment produced nothing (see below)"
s = crf_results[sorted(crf_results)[0]]
print("per-pid keys:", list(s.keys()), "| n_true:", len(s['true_labels']))

import pickle
pickle.dump(crf_results, open('results/crf_export.pkl', 'wb'))   # save the POPULATED dict
print("saved crf_export.pkl with", len(crf_results), "patients")

import pickle
crf_feats = pickle.load(open('results/crf_feats.pkl', 'rb'))
print(len(crf_feats), 'patients loaded')

# Balanced vs unbalanced classifier on CRF features — does balancing exploit separability?
import numpy as np, warnings
warnings.filterwarnings('ignore')
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score

def run(d, balanced, min_count=25, n_pca=50):
    X = np.asarray(d['X'], np.float32); y = np.asarray(d['y']); g = np.asarray(d['sid'])
    keep = [c for c, n in Counter(y).items() if n >= min_count]
    m = np.isin(y, keep); X, y, g = X[m], y[m], g[m]
    oof = np.empty(len(y), dtype=object)
    for tr, te in GroupKFold(5).split(X, y, g):
        sc = StandardScaler().fit(X[tr]); Xtr = sc.transform(X[tr]); Xte = sc.transform(X[te])
        pca = PCA(min(n_pca, Xtr.shape[1], Xtr.shape[0]), random_state=0).fit(Xtr)
        Xtr, Xte = pca.transform(Xtr), pca.transform(Xte)
        clf = LogisticRegression(max_iter=300, class_weight=('balanced' if balanced else None))
        clf.fit(Xtr, y[tr]); oof[te] = clf.predict(Xte)
    return (accuracy_score(y, oof), balanced_accuracy_score(y, oof),
            len(set(oof)), len(set(y)))   # acc, balanced-acc, #classes predicted, #classes present

pids = sorted(crf_feats)
print(f"{'pid':>5} | {'UNBAL  acc/bal/#pred':>22} | {'BAL    acc/bal/#pred':>22}")
ua, ub, ba, bb = [], [], [], []
for pid in pids:
    a0, b0, p0, nt = run(crf_feats[pid], False)
    a1, b1, p1, _  = run(crf_feats[pid], True)
    ua.append(a0); ub.append(b0); ba.append(a1); bb.append(b1)
    print(f"{pid:>5} | {a0:.3f}/{b0:.3f}/{p0:>2}of{nt:<2}        | {a1:.3f}/{b1:.3f}/{p1:>2}of{nt:<2}")
print(f"\nCOHORT  UNBAL: acc={np.mean(ua):.3f}  bal-acc={np.mean(ub):.3f}")
print(f"        BAL:   acc={np.mean(ba):.3f}  bal-acc={np.mean(bb):.3f}")
print(f"  -> balancing changes overall acc {np.mean(ba)-np.mean(ua):+.3f}, "
      f"balanced-acc {np.mean(bb)-np.mean(ub):+.3f}")
