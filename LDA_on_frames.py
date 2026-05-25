# Converted from LDA_on_frames.ipynb

# build pipeline
import os, numpy as np, warnings
from collections import defaultdict, Counter
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

from dutch_30_pipeline import Dutch30Pipeline
from dutch_30_feature_extractor import Dutch30FeatureExtractor
from dataset_config import Dutch30Config
from config import DUTCH_30_PATH
from run_pipeline import load_mfa_alignments
from extract_features import extractHG, stackFeatures

config = Dutch30Config()
extractor = Dutch30FeatureExtractor(config=config)
pipeline = Dutch30Pipeline(extractor, config=config, use_wav2vec=False)
pipeline.step1_load_dutch30_data(patient_range=(21, 30))
pipeline.step2_split_by_instances(train_fraction=0.8)
pipeline.step3_load_channel_exclusions('channel_exclusions.json')
pipeline.apply_channel_exclusions()

# notebook level constants
# ============================================================
# Dataset + split
# ============================================================
TARGET_PIDS     = ['P21', 'P22', 'P23', 'P24', 'P25',
                   'P26', 'P27', 'P28', 'P29', 'P30']
TEST_OFFSET     = 0          # 0..5
VAL_FRAC        = 0.15       # fraction of TRAIN held out for bonus tuning (leak-safe)

# ============================================================
# Sampling and frame geometry
# ============================================================
EEG_SR          = 1024
WIN_S, SHIFT_S  = 0.015, 0.005
WIN_SAMP        = int(EEG_SR * WIN_S)
SHIFT_SAMP      = int(EEG_SR * SHIFT_S)

# ============================================================
# Feature stacking
# ============================================================
MO, SS          = 11, 1
LDA_MARGIN      = MO * SS
SD_MARGIN       = 5 * 1      # speech detector was trained with MO=5, SS=1

# ============================================================
# Phoneme label filter (used only on the TRAIN side)
# ============================================================
MN_FRAMES       = 0          # drop training phonemes shorter than this
MX_FRAMES       = 300        # drop training phonemes longer than this

# ============================================================
# LDA feature extraction — knobs for extract_features_multiband
# ============================================================
HG_BAND         = (70, 170)
LG_BAND         = (30, 70)
THETA_BAND      = (4, 8)
NOTCH_HZ        = (100, 150) # line-noise harmonics (MFA-CRF style)

HG_LP_HZ        = 10.0       # HG envelope LP
LG_LP_HZ        = 10.0       # LG envelope LP
PHASE_LP_HZ     = 20.0       # theta cos/sin LP
PAC_LP_HZ       = 10.0       # theta-HG PAC LP
LG_PAC_LP_HZ    = 10.0       # theta-LG PAC LP

# Default feature spec — pass this to run_for_patient* unless you're sweeping
DEFAULT_FEATURE_SPEC = {
    'hg_amp':       True,
    'hg_lp_hz':     HG_LP_HZ,
}

# ============================================================
# Post-LDA decoding
# ============================================================
SMOOTH_LOGP_W   = 31         # moving-avg window on per-class log-posteriors
SELF_LOOP_BONUS = None       # None = auto-tune on val; else a float
TARGET_RATIO    = 1.0        # pred/gold count ratio target (1.0 = exact)
MIN_PRED_FRAMES = 3          # drop predicted runs shorter than this
SMOOTH_W        = 1          # legacy feature-row smoothing; usually keep at 1

# ============================================================
# Speech gating
# ----- LOCKED at training time: don't change SD_* unless you retrain ----
# ============================================================
SD_LP_HZ        = 12.0       # what the speech detector was trained on
SD_NOTCH_HZ     = (50, 150)  # same; speech detector legacy
SD_BAND         = (70, 170)  # HG band the detector saw

USE_SPEECH_GATE   = True     # default for run_for_patient_sd
SPEECH_THRESHOLD  = 0.5      # cutoff on the detector's softmax(speech)
SPEECH_FRAC_MIN   = 0.5      # fraction of a segment that must be in speech

# ============================================================
# Helpers
# ============================================================

def stk_frame_to_time_s(i):
    return ((i + LDA_MARGIN) * SHIFT_SAMP + WIN_SAMP / 2) / EEG_SR

# sweep helpers
import re

ANSI_RE = re.compile(r'\x1b\[[0-9;]*m')

def _visible_len(s):
    return len(ANSI_RE.sub('', s))

def _pad(s, width):
    """Right-justify a string that may contain ANSI escapes."""
    return ' ' * max(0, width - _visible_len(s)) + s

def z_str(z, w=6):
    s = f"{z:+{w}.2f}"
    return f"\033[91m{s}\033[0m" if z > 2 else s

def per_str(per, w=4, decimals=0):
    fmt = f"{{:>{w}.{decimals}%}}"
    s = fmt.format(per)
    return f"\033[91m{s}\033[0m" if per < 0.80 else s

RARE_TOP_N = 5

def count_rare_in_matches(matches, gold_sents, top_n=RARE_TOP_N):
    all_gold = [ph for s in gold_sents for ph in s]
    common = set(p for p, _ in Counter(all_gold).most_common(top_n))
    return sum(1 for m in matches for ph in m if ph not in common)

# helper functions
import time
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from e2e_brain_decoder import edit_distance

def smooth_cols(logp, w):
    """Centered moving-average along time (axis=0), per class column."""
    if w <= 1: return logp
    T, K = logp.shape
    pad_left  = (w - 1) // 2
    pad_right = w - 1 - pad_left
    padded = np.pad(logp, ((pad_left, pad_right), (0, 0)), mode='edge')
    csum = np.concatenate([np.zeros((1, K)), np.cumsum(padded, axis=0)])
    return (csum[w:] - csum[:-w]) / w

def viterbi_decode(logp, self_bonus):
    """Viterbi where staying in the same class earns +self_bonus per frame.
       Switching pays 0. Higher bonus → longer runs."""
    T, K = logp.shape
    if T == 0: return np.zeros(0, dtype=np.int32)
    delta = np.empty((T, K)); bptr = np.empty((T, K), dtype=np.int32)
    delta[0] = logp[0]
    all_k = np.arange(K)
    for t in range(1, T):
        prev = delta[t - 1]
        order = np.argsort(prev)
        idx1, idx2 = order[-1], order[-2]
        # For each k, the best "switch from somewhere else" score:
        best_switch        = np.full(K, prev[idx1])
        best_switch[idx1]  = prev[idx2]
        bptr_switch        = np.full(K, idx1)
        bptr_switch[idx1]  = idx2
        stay = prev + self_bonus
        choose_stay = stay >= best_switch
        delta[t] = logp[t] + np.where(choose_stay, stay, best_switch)
        bptr[t]  = np.where(choose_stay, all_k, bptr_switch)
    path = np.empty(T, dtype=np.int32)
    path[-1] = delta[-1].argmax()
    for t in range(T - 2, -1, -1):
        path[t] = bptr[t + 1, path[t + 1]]
    return path

def count_runs(path, min_len):
    n = i = 0; T = len(path)
    while i < T:
        j = i + 1
        while j < T and path[j] == path[i]: j += 1
        if (j - i) >= min_len: n += 1
        i = j
    return n

def auto_tune_bonus(logp_list, target_count, min_pred_frames,
                    lo=0.0, hi=50.0, n_iter=18):
    """Binary search: smallest bonus that brings total segment count ≤ target."""
    for _ in range(n_iter):
        mid = (lo + hi) / 2
        cnt = sum(count_runs(viterbi_decode(lp, mid), min_pred_frames)
                  for lp in logp_list)
        if cnt > target_count: lo = mid
        else:                  hi = mid
    return (lo + hi) / 2

def smooth_cols(logp, w):
    if w <= 1: return logp
    T, K = logp.shape
    pl = (w - 1) // 2; pr = w - 1 - pl
    padded = np.pad(logp, ((pl, pr), (0, 0)), mode='edge')
    csum = np.concatenate([np.zeros((1, K)), np.cumsum(padded, axis=0)])
    return (csum[w:] - csum[:-w]) / w

def run_for_patient(pid, test_offset=TEST_OFFSET, smoothing_hz=10.0):
    wd = pipeline.split_result['word_segments_dict'].get(pid)
    if wd is None: return None, "no word_segments_dict"
    try: mfa = load_mfa_alignments(pid)
    except Exception as e: return None, f"no MFA ({type(e).__name__})"
    if not mfa: return None, "empty MFA"

    all_real = [i for i, s in enumerate(wd['sentence_list'])
                if isinstance(s, dict) and s.get('text')]
    test_sent_ids      = set(all_real[test_offset::6])
    train_sent_ids_all = sorted(set(all_real) - test_sent_ids)

    # Hold a val slice OUT of train: every (1/VAL_FRAC)th sentence
    val_step     = max(2, int(round(1 / VAL_FRAC)))
    val_sent_ids = set(train_sent_ids_all[::val_step])
    fit_sent_ids = set(train_sent_ids_all) - val_sent_ids

    raw_path = os.path.join(DUTCH_30_PATH, 'raw', f'{pid}_sEEG.npy')
    if not os.path.exists(raw_path): return None, "no sEEG file"
    raw_eeg = np.load(raw_path)
    if raw_eeg.ndim == 2 and raw_eeg.shape[0] < raw_eeg.shape[1]:
        raw_eeg = raw_eeg.T

    per_sentence_stk = {}
    for sent_idx in all_real:
        if sent_idx not in mfa or not mfa[sent_idx]: continue
        s = wd['sentence_list'][sent_idx]
        s0, s1 = s['stim_start_idx'], s['stim_end_idx']
        if s1 > raw_eeg.shape[0]: continue
        ext = extractHG(raw_eeg[s0:s1], EEG_SR,
                        windowLength=WIN_S, frameshift=SHIFT_S,
                        smoothing_hz=smoothing_hz).astype(np.float32)
        if ext.shape[0] < 2 * LDA_MARGIN + 1: continue
        per_sentence_stk[sent_idx] = stackFeatures(ext, modelOrder=MO, stepSize=SS)
    if not per_sentence_stk: return None, "no usable sentences"

    def build_train_set(sent_ids):
        X, y = [], []
        for sent_idx in sent_ids:
            if sent_idx not in per_sentence_stk: continue
            stk = per_sentence_stk[sent_idx]; T_stk = stk.shape[0]
            for ph in mfa[sent_idx]:
                k_s = int(np.ceil ((ph['start_s']*EEG_SR - WIN_SAMP/2) / SHIFT_SAMP))
                k_e = int(np.floor((ph['end_s']  *EEG_SR - WIN_SAMP/2) / SHIFT_SAMP))
                n_fr = k_e - k_s + 1
                if n_fr < MN_FRAMES or n_fr > MX_FRAMES: continue
                ks = max(0, k_s - LDA_MARGIN); ke = min(T_stk - 1, k_e - LDA_MARGIN)
                if ke < ks: continue
                X.append(stk[ks:ke+1].mean(axis=0))
                y.append(ph['phone'])
        return np.array(X), np.array(y)

    # ── Step 1: Fit LDA on the 85% — for bonus tuning only ──
    X_fit, y_fit = build_train_set(fit_sent_ids)
    if len(X_fit) < 50: return None, f"too few fit samples ({len(X_fit)})"
    fit_classes = set(y_fit)
    sc_fit = StandardScaler().fit(X_fit)
    clf_fit = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    clf_fit.fit(sc_fit.transform(X_fit), y_fit)

    # ── Step 2: Compute val log-probs and val target gold-count ──
    val_logps, val_target = [], 0
    for sent_idx in val_sent_ids:
        if sent_idx not in per_sentence_stk: continue
        stk = per_sentence_stk[sent_idx]
        logp = clf_fit.predict_log_proba(sc_fit.transform(stk))
        logp = smooth_cols(logp, SMOOTH_LOGP_W)
        val_logps.append(logp)
        val_target += sum(1 for ph in mfa[sent_idx] if ph['phone'] in fit_classes)
    if not val_logps: return None, "no val sentences"
    val_target = int(val_target * TARGET_RATIO)

    # ── Step 3: Auto-tune bonus on val log-probs only ──
    if SELF_LOOP_BONUS is None:
        bonus = auto_tune_bonus(val_logps, val_target, MIN_PRED_FRAMES)
    else:
        bonus = float(SELF_LOOP_BONUS)

    # ── Step 4: Refit LDA on the full train (fit + val) ──
    X_train, y_train = build_train_set(set(all_real) - test_sent_ids)
    if len(X_train) < 50: return None, f"too few train samples ({len(X_train)})"
    train_classes = set(y_train)
    scaler = StandardScaler().fit(X_train)
    clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    clf.fit(scaler.transform(X_train), y_train)
    class_labels = list(clf.classes_)

    # ── Step 5: Apply final LDA + tuned bonus to test (no leak) ──
    predictions, pred_sentence_ids, pred_segments = [], [], []
    true_labels, true_sentence_ids, true_segments = [], [], []
    for sent_idx in test_sent_ids:
        if sent_idx not in per_sentence_stk: continue
        stk = per_sentence_stk[sent_idx]; T_stk = stk.shape[0]
        logp = clf.predict_log_proba(scaler.transform(stk))
        logp = smooth_cols(logp, SMOOTH_LOGP_W)
        path = viterbi_decode(logp, bonus)
        i = 0
        while i < T_stk:
            ci = path[i]; j = i + 1
            while j < T_stk and path[j] == ci: j += 1
            if (j - i) >= MIN_PRED_FRAMES:
                predictions.append(class_labels[ci])
                pred_sentence_ids.append(sent_idx)
                pred_segments.append((stk_frame_to_time_s(i),
                                      stk_frame_to_time_s(j - 1)))
            i = j
        for ph in mfa[sent_idx]:
            if ph['phone'] not in train_classes: continue
            true_labels.append(ph['phone'])
            true_sentence_ids.append(sent_idx)
            true_segments.append((ph['start_s'], ph['end_s']))

    if not true_labels: return None, "no test gold labels"
    true_arr = np.array(true_labels); pred_arr = np.array(predictions)
    ed = edit_distance(list(true_arr), list(pred_arr))
    per = ed / max(len(true_arr), 1)
    return {
        'true_labels':       true_arr,
        'predictions':       pred_arr,
        'true_sentence_ids': np.array(true_sentence_ids),
        'pred_sentence_ids': np.array(pred_sentence_ids),
        'true_segments':     true_segments,
        'pred_segments':     pred_segments,
        'accuracy':          float('nan'),
        'edit_distance':     ed,
        'per':               per,
        'n_test':            len(true_arr),
        'n_pred':            len(pred_arr),
        'n_train':           len(X_train),
        'n_val_sents':       len(val_logps),
        'bonus':             bonus,
        'val_target':        val_target,
    }, None

def longest_run_with_shift(pred, gold, shift_max=3):
    best, P, G = 0, len(pred), len(gold)
    for i in range(P):
        for j in range(max(0, i - shift_max), min(G, i + shift_max + 1)):
            k = 0
            while i + k < P and j + k < G and pred[i + k] == gold[j + k]: k += 1
            if k > best: best = k
    return best

def collect_matches(pred_sents, gold_sents, min_match=3, shift_max=3):
    matches = []
    for p, g in zip(pred_sents, gold_sents):
        L, span, P, G = 0, None, len(p), len(g)
        for i in range(P):
            for j in range(max(0, i - shift_max), min(G, i + shift_max + 1)):
                k = 0
                while i + k < P and j + k < G and p[i + k] == g[j + k]: k += 1
                if k > L: L, span = k, (i, j, k)
        if L >= min_match and span is not None:
            i, j, k = span; matches.append(tuple(p[i:i + k]))
    return matches

def surprise_score(matches, marginal_logp):
    fallback = -np.log(1e-6)
    return sum(-marginal_logp.get(ph, fallback) for m in matches for ph in m)

def perm_null(pred_sents, gold_sents, marginal_logp, n_perm=2000, seed=0):
    rng = np.random.default_rng(seed); nulls = np.zeros(n_perm)
    for b in range(n_perm):
        shuf = []
        for p in pred_sents:
            if len(p) == 0: shuf.append(p); continue
            idx = rng.permutation(len(p))
            shuf.append([p[k] for k in idx])
        nulls[b] = surprise_score(collect_matches(shuf, gold_sents), marginal_logp)
    return nulls

import numpy as np
from collections import Counter

def by_sentence(arr_labels, arr_sids):
    out = {}
    for lbl, sid in zip(arr_labels, arr_sids):
        out.setdefault(int(sid), []).append(lbl)
    return out

def longest_run_with_shift(pred, gold, shift_max=3):
    best = 0
    for i in range(len(pred)):
        for j in range(max(0, i - shift_max), min(len(gold), i + shift_max + 1)):
            k = 0
            while i + k < len(pred) and j + k < len(gold) and pred[i + k] == gold[j + k]:
                k += 1
            best = max(best, k)
    return best

def count_ngrams_at_least(pred, gold, min_len=4, shift_max=3):
    n, used = 0, set()
    for i in range(len(pred) - min_len + 1):
        if any(x in used for x in range(i, i+min_len)): continue
        for j in range(max(0, i - shift_max), min(len(gold) - min_len + 1, i + shift_max + 1)):
            k = 0
            while i + k < len(pred) and j + k < len(gold) and pred[i + k] == gold[j + k]:
                k += 1
            if k >= min_len:
                n += 1
                for x in range(i, i+k): used.add(x)
                break
    return n

def perm_z(pred_sents, gold_sents, n_perm=500, seed=0):
    rng = np.random.default_rng(seed)
    pred_all = [ph for s in pred_sents for ph in s]
    def stat(pp, gg):
        return sum(longest_run_with_shift(p, g) for p, g in zip(pp, gg))
    obs = stat(pred_sents, gold_sents)
    nulls = np.zeros(n_perm)
    for n in range(n_perm):
        shuf = list(pred_all); rng.shuffle(shuf)
        out, cur = [], 0
        for s in pred_sents:
            out.append(shuf[cur:cur+len(s)]); cur += len(s)
        nulls[n] = stat(out, gold_sents)
    mu, sd = nulls.mean(), nulls.std() + 1e-9
    return (obs - mu) / sd, obs, mu, sd

# # extract_features_multiband
# from scipy.signal import butter, sosfiltfilt, hilbert, iirfilter
# import scipy.signal

# def extract_features_multiband(eeg_slice, sr=EEG_SR, win_s=WIN_S, shift_s=SHIFT_S,
#                                 hg_amp=False, lg_amp=False,
#                                 theta_phase=False,
#                                 theta_hg_pac=False, lg_theta_pac=False,
#                                 hg_x_lg=False,
#                                 hg_lp_hz=12.0, lg_lp_hz=10.0,
#                                 phase_lp_hz=20.0,
#                                 pac_lp_hz=10.0, lg_pac_lp_hz=10.0):
#     """Cross-band feature extractor.

#     Amplitude features (one block of n_channels each):
#         hg_amp           — HG envelope (70–170 Hz)
#         lg_amp           — LG envelope (30–70 Hz)
#         hg_x_lg          — HG_env × LG_env (cross-band co-activation)

#     Phase / coupling features (two blocks each, cos + sin):
#         theta_phase      — cos/sin of theta phase (4–8 Hz)
#         theta_hg_pac     — HG_env × cos(θ),  HG_env × sin(θ)
#         lg_theta_pac     — LG_env × cos(θ),  LG_env × sin(θ)
#     """
#     x = scipy.signal.detrend(eeg_slice, axis=0)
#     for f0 in [100, 150]:
#         sos = iirfilter(4, [(f0-2)/(sr/2), (f0+2)/(sr/2)],
#                         btype='bandstop', output='sos')
#         x = sosfiltfilt(sos, x, axis=0)

#     hg_env = lg_env = theta_ph = None

#     need_hg    = hg_amp or theta_hg_pac or hg_x_lg
#     need_lg    = lg_amp or lg_theta_pac or hg_x_lg
#     need_theta = theta_phase or theta_hg_pac or lg_theta_pac

#     if need_hg:
#         sos_hg = butter(4, [70, 170], btype='bandpass', fs=sr, output='sos')
#         x_hg   = sosfiltfilt(sos_hg, x, axis=0)
#         lp     = butter(4, hg_lp_hz, btype='lowpass', fs=sr, output='sos')
#         hg_env = np.sqrt(np.abs(sosfiltfilt(lp, x_hg ** 2, axis=0)))
#     if need_lg:
#         sos_lg = butter(4, [30, 70], btype='bandpass', fs=sr, output='sos')
#         x_lg   = sosfiltfilt(sos_lg, x, axis=0)
#         lp     = butter(4, lg_lp_hz, btype='lowpass', fs=sr, output='sos')
#         lg_env = np.sqrt(np.abs(sosfiltfilt(lp, x_lg ** 2, axis=0)))
#     if need_theta:
#         sos_th  = butter(4, [4, 8], btype='bandpass', fs=sr, output='sos')
#         x_th    = sosfiltfilt(sos_th, x, axis=0)
#         theta_ph = np.angle(hilbert(x_th, axis=0))

#     win_n, shift_n = int(sr * win_s), int(sr * shift_s)
#     n_w = int(np.floor((eeg_slice.shape[0] - win_n) / shift_n))

#     def wm(arr):
#         out = np.zeros((n_w, arr.shape[1]))
#         for w in range(n_w):
#             s = w * shift_n
#             out[w] = arr[s:s + win_n].mean(axis=0)
#         return out

#     def lp_smooth(arr, hz):
#         sos = butter(4, hz, btype='lowpass', fs=sr, output='sos')
#         return sosfiltfilt(sos, arr, axis=0)

#     blocks = []

#     # ── amplitude blocks ─────────────────────────────────────────────
#     if hg_amp:      blocks.append(wm(hg_env))
#     if lg_amp:      blocks.append(wm(lg_env))
#     if hg_x_lg:     blocks.append(wm(hg_env * lg_env))

#     # ── phase blocks ─────────────────────────────────────────────────
#     if theta_phase:
#         cos_p = lp_smooth(np.cos(theta_ph), phase_lp_hz)
#         sin_p = lp_smooth(np.sin(theta_ph), phase_lp_hz)
#         blocks.append(wm(cos_p)); blocks.append(wm(sin_p))

#     # ── PAC blocks ───────────────────────────────────────────────────
#     if theta_hg_pac:
#         pac_c = lp_smooth(hg_env * np.cos(theta_ph), pac_lp_hz)
#         pac_s = lp_smooth(hg_env * np.sin(theta_ph), pac_lp_hz)
#         blocks.append(wm(pac_c)); blocks.append(wm(pac_s))
#     if lg_theta_pac:
#         lpac_c = lp_smooth(lg_env * np.cos(theta_ph), lg_pac_lp_hz)
#         lpac_s = lp_smooth(lg_env * np.sin(theta_ph), lg_pac_lp_hz)
#         blocks.append(wm(lpac_c)); blocks.append(wm(lpac_s))

#     return np.concatenate(blocks, axis=1).astype(np.float32)

# extract_features_multiband
from scipy.signal import sosfilt

def extract_features_multiband(eeg_slice, sr=EEG_SR, win_s=WIN_S, shift_s=SHIFT_S,
                                hg_amp=False, lg_amp=False,
                                theta_phase=False,
                                theta_hg_pac=False, lg_theta_pac=False,
                                hg_x_lg=False,
                                hg_lp_hz=12.0, lg_lp_hz=10.0,
                                phase_lp_hz=20.0,
                                pac_lp_hz=10.0, lg_pac_lp_hz=10.0,
                                causal=False):                  # ← NEW
    filter_fn = sosfilt if causal else sosfiltfilt              # ← NEW

    x = scipy.signal.detrend(eeg_slice, axis=0)
    for f0 in [100, 150]:
        sos = iirfilter(4, [(f0-2)/(sr/2), (f0+2)/(sr/2)],
                        btype='bandstop', output='sos')
        x = filter_fn(sos, x, axis=0)                           # ← swap

    hg_env = lg_env = theta_ph = None
    need_hg    = hg_amp or theta_hg_pac or hg_x_lg
    need_lg    = lg_amp or lg_theta_pac or hg_x_lg
    need_theta = theta_phase or theta_hg_pac or lg_theta_pac

    if need_hg:
        sos_hg = butter(4, [70, 170], btype='bandpass', fs=sr, output='sos')
        x_hg   = filter_fn(sos_hg, x, axis=0)                   # ← swap
        lp     = butter(4, hg_lp_hz, btype='lowpass', fs=sr, output='sos')
        hg_env = np.sqrt(np.abs(filter_fn(lp, x_hg ** 2, axis=0)))   # ← swap
    if need_lg:
        sos_lg = butter(4, [30, 70], btype='bandpass', fs=sr, output='sos')
        x_lg   = filter_fn(sos_lg, x, axis=0)                   # ← swap
        lp     = butter(4, lg_lp_hz, btype='lowpass', fs=sr, output='sos')
        lg_env = np.sqrt(np.abs(filter_fn(lp, x_lg ** 2, axis=0)))   # ← swap
    if need_theta:
        # NOTE: hilbert is FFT-based — inherently non-causal regardless of `causal` flag.
        # Theta phase / PAC features can't be made fully causal here. Skip for this test.
        if causal:
            raise NotImplementedError("Causal mode doesn't support theta_phase/PAC; "
                                       "those need a redesigned envelope path.")
        sos_th = butter(4, [4, 8], btype='bandpass', fs=sr, output='sos')
        x_th   = sosfiltfilt(sos_th, x, axis=0)
        theta_ph = np.angle(hilbert(x_th, axis=0))

    win_n, shift_n = int(sr * win_s), int(sr * shift_s)
    n_w = int(np.floor((eeg_slice.shape[0] - win_n) / shift_n))

    def wm(arr):
        out = np.zeros((n_w, arr.shape[1]))
        for w in range(n_w):
            s = w * shift_n
            out[w] = arr[s:s + win_n].mean(axis=0)
        return out

    def lp_smooth(arr, hz):
        sos = butter(4, hz, btype='lowpass', fs=sr, output='sos')
        return filter_fn(sos, arr, axis=0)                       # ← swap

    blocks = []
    if hg_amp:      blocks.append(wm(hg_env))
    if lg_amp:      blocks.append(wm(lg_env))
    if hg_x_lg:     blocks.append(wm(hg_env * lg_env))
    if theta_phase:
        cos_p = lp_smooth(np.cos(theta_ph), phase_lp_hz)
        sin_p = lp_smooth(np.sin(theta_ph), phase_lp_hz)
        blocks.append(wm(cos_p)); blocks.append(wm(sin_p))
    if theta_hg_pac:
        pac_c = lp_smooth(hg_env * np.cos(theta_ph), pac_lp_hz)
        pac_s = lp_smooth(hg_env * np.sin(theta_ph), pac_lp_hz)
        blocks.append(wm(pac_c)); blocks.append(wm(pac_s))
    if lg_theta_pac:
        lpac_c = lp_smooth(lg_env * np.cos(theta_ph), lg_pac_lp_hz)
        lpac_s = lp_smooth(lg_env * np.sin(theta_ph), lg_pac_lp_hz)
        blocks.append(wm(lpac_c)); blocks.append(wm(lpac_s))

    return np.concatenate(blocks, axis=1).astype(np.float32)

# Speech detector integration
#   Uses the LOCKED SD_* constants from the config block:
#     SD_BAND, SD_NOTCH_HZ, SD_LP_HZ — match training-time signal proc
#     USE_SPEECH_GATE, SPEECH_THRESHOLD, SPEECH_FRAC_MIN — gating knobs
# ============================================================
import torch
import torch.nn as nn
from scipy.signal import butter, sosfiltfilt, iirnotch, tf2sos, hilbert

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Speech detector architecture ───────────────────────────────────
class CrossPatientSpeechDetector(nn.Module):
    def __init__(self, n_in_per_pid, embed_dim=128,
                 lstm_hidden=128, lstm_layers=2, dropout=0.2):
        super().__init__()
        self.projs = nn.ModuleDict({
            pid: nn.Sequential(nn.Linear(n_in, embed_dim), nn.GELU(),
                               nn.Dropout(dropout))
            for pid, n_in in n_in_per_pid.items()
        })
        self.lstm = nn.LSTM(embed_dim, lstm_hidden, num_layers=lstm_layers,
                            dropout=dropout if lstm_layers > 1 else 0.0,
                            bidirectional=True, batch_first=False)
        self.head = nn.Linear(lstm_hidden * 2, 2)

    def forward(self, x, pid):
        h = self.projs[pid](x).unsqueeze(1)
        h, _ = self.lstm(h)
        return self.head(h.squeeze(1))

# ── Load checkpoint ────────────────────────────────────────────────
ckpt = torch.load('bio_models/speech_detector_cross_patient.pt',
                  map_location=DEVICE, weights_only=False)
sd_model = CrossPatientSpeechDetector(
    n_in_per_pid=ckpt['n_in_per_pid'],
    **ckpt['arch']).to(DEVICE)
sd_model.load_state_dict(ckpt['state_dict'])
sd_model.eval()
mu_sd_speech = ckpt['mu_sd']
print(f"Loaded speech detector for {sorted(sd_model.projs.keys())}")

# ── Signal-processing chain — locked to training-time values ──────
_sd_sos_bp = butter(4, list(SD_BAND), btype='bandpass',
                    fs=EEG_SR, output='sos')
_sd_sos_lp = butter(4, SD_LP_HZ, btype='lowpass', fs=EEG_SR, output='sos')
_sd_sos_notch = []
for f0 in SD_NOTCH_HZ:
    b, a = iirnotch(f0, 30, EEG_SR)
    _sd_sos_notch.append(tf2sos(b, a))

def extract_hg_frames(eeg_slice):
    """Hilbert HG envelope at 200 fps — matches the speech detector's training input."""
    x = eeg_slice.astype(np.float64)
    for sos in _sd_sos_notch:
        x = sosfiltfilt(sos, x, axis=0)
    x = sosfiltfilt(_sd_sos_bp, x, axis=0)
    env = np.abs(hilbert(x, axis=0))
    env = sosfiltfilt(_sd_sos_lp, env, axis=0)
    env = np.maximum(env, 0)
    return np.log1p(env[::SHIFT_SAMP].astype(np.float32))

@torch.no_grad()
def predict_speech_prob(raw_eeg_slice, pid):
    """Per-stacked-frame speech probability. Uses MO=5, SS=1 stacking
       to match how the detector was trained."""
    hg = extract_hg_frames(raw_eeg_slice)
    hg_stk = stackFeatures(hg, modelOrder=5, stepSize=1)
    mu, sd = mu_sd_speech[pid]
    sd_safe = np.where(sd < 1e-6, 1.0, sd)
    x_t = torch.from_numpy((hg_stk - mu) / sd_safe).float().to(DEVICE)
    return torch.softmax(sd_model(x_t, pid), dim=-1)[:, 1].cpu().numpy()


# ============================================================
# Mask-aware bonus tuning
# ============================================================
def auto_tune_bonus_masked(logp_list, mask_list, target_count, min_pred_frames,
                            speech_frac=None, lo=0.0, hi=50.0, n_iter=18):
    """Tune bonus so #(speech-overlapping runs) ≈ target_count."""
    if speech_frac is None:
        speech_frac = SPEECH_FRAC_MIN
    def count(bonus):
        n = 0
        for logp, mask in zip(logp_list, mask_list):
            path = viterbi_decode(logp, bonus)
            i = 0; T = len(path)
            while i < T:
                j = i + 1
                while j < T and path[j] == path[i]: j += 1
                if (j - i) >= min_pred_frames and mask[i:j].mean() >= speech_frac:
                    n += 1
                i = j
        return n
    for _ in range(n_iter):
        mid = (lo + hi) / 2
        if count(mid) > target_count: lo = mid
        else:                         hi = mid
    return (lo + hi) / 2


# ============================================================
# run_for_patient + speech gating
# ============================================================
def run_for_patient_sd(pid, test_offset=TEST_OFFSET, feature_spec=None,
                       use_speech_gate=USE_SPEECH_GATE,
                       speech_thresh=SPEECH_THRESHOLD,
                       speech_frac_min=SPEECH_FRAC_MIN):
    """run_for_patient_mb with optional speech-gated decoding.
       Set use_speech_gate=False to recover the un-gated baseline."""
    feature_spec = feature_spec or DEFAULT_FEATURE_SPEC
    wd = pipeline.split_result['word_segments_dict'].get(pid)
    if wd is None: return None, "no word_segments_dict"
    try: mfa = load_mfa_alignments(pid)
    except Exception as e: return None, f"no MFA ({type(e).__name__})"
    if not mfa: return None, "empty MFA"
    if use_speech_gate and pid not in mu_sd_speech:
        return None, f"no speech-detector stats for {pid}"

    all_real = [i for i, s in enumerate(wd['sentence_list'])
                if isinstance(s, dict) and s.get('text')]
    test_sent_ids      = set(all_real[test_offset::6])
    train_sent_ids_all = sorted(set(all_real) - test_sent_ids)
    val_step     = max(2, int(round(1 / VAL_FRAC)))
    val_sent_ids = set(train_sent_ids_all[::val_step])
    fit_sent_ids = set(train_sent_ids_all) - val_sent_ids

    raw_path = os.path.join(DUTCH_30_PATH, 'raw', f'{pid}_sEEG.npy')
    raw_eeg = np.load(raw_path)
    if raw_eeg.ndim == 2 and raw_eeg.shape[0] < raw_eeg.shape[1]:
        raw_eeg = raw_eeg.T

    per_sentence_stk  = {}
    per_sentence_mask = {}    # speech mask aligned to LDA stk space
    for sent_idx in all_real:
        if sent_idx not in mfa or not mfa[sent_idx]: continue
        s = wd['sentence_list'][sent_idx]
        s0, s1 = s['stim_start_idx'], s['stim_end_idx']
        if s1 > raw_eeg.shape[0]: continue
        ext = extract_features_multiband(raw_eeg[s0:s1], **feature_spec)
        if ext.shape[0] < 2 * LDA_MARGIN + 1: continue
        per_sentence_stk[sent_idx] = stackFeatures(ext, modelOrder=MO, stepSize=SS)
        if use_speech_gate:
            sp = predict_speech_prob(raw_eeg[s0:s1], pid)
            T_stk = per_sentence_stk[sent_idx].shape[0]
            if len(sp) >= T_stk:
                mask = sp[:T_stk] > speech_thresh
            else:
                mask = np.concatenate([sp > speech_thresh,
                                        np.zeros(T_stk - len(sp), dtype=bool)])
            per_sentence_mask[sent_idx] = mask
    if not per_sentence_stk: return None, "no usable sentences"

    GROUP_DELAY_FRAMES = 10   # = 0.050 s for LP=10 Hz Butterworth-4 (rough)

    def build_train_set(sent_ids):
        causal_mode = feature_spec.get('causal', False)
        shift = GROUP_DELAY_FRAMES if causal_mode else 0
        X, y = [], []
        for sent_idx in sent_ids:
            if sent_idx not in per_sentence_stk: continue
            stk = per_sentence_stk[sent_idx]; T_stk = stk.shape[0]
            for ph in mfa[sent_idx]:
                k_s = int(np.ceil ((ph['start_s']*EEG_SR - WIN_SAMP/2) / SHIFT_SAMP))
                k_e = int(np.floor((ph['end_s']  *EEG_SR - WIN_SAMP/2) / SHIFT_SAMP))
                if (k_e - k_s + 1) < MN_FRAMES or (k_e - k_s + 1) > MX_FRAMES: continue
                # ★ Shift window LATER in causal mode to compensate for group delay
                ks = max(0,         k_s - LDA_MARGIN + shift)
                ke = min(T_stk - 1, k_e - LDA_MARGIN + shift)
                if ke < ks: continue
                X.append(stk[ks:ke+1].mean(axis=0))
                y.append(ph['phone'])
        return np.array(X), np.array(y)

    # ── Step 1: fit LDA on the 85% (for bonus tuning only) ────────
    X_fit, y_fit = build_train_set(fit_sent_ids)
    if len(X_fit) < 50: return None, f"too few fit samples ({len(X_fit)})"
    fit_classes = set(y_fit)
    sc_fit  = StandardScaler().fit(X_fit)
    clf_fit = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    clf_fit.fit(sc_fit.transform(X_fit), y_fit)

    # ── Step 2: val log-probs + val target gold-count ─────────────
    val_logps, val_masks, val_target = [], [], 0
    for sent_idx in val_sent_ids:
        if sent_idx not in per_sentence_stk: continue
        stk = per_sentence_stk[sent_idx]
        logp = clf_fit.predict_log_proba(sc_fit.transform(stk))
        logp = smooth_cols(logp, SMOOTH_LOGP_W)
        val_logps.append(logp)
        if use_speech_gate:
            val_masks.append(per_sentence_mask[sent_idx])
        val_target += sum(1 for ph in mfa[sent_idx] if ph['phone'] in fit_classes)
    if not val_logps: return None, "no val sentences"
    val_target = int(val_target * TARGET_RATIO)

    # ── Step 3: tune bonus (mask-aware if gating) ─────────────────
    if SELF_LOOP_BONUS is None:
        if use_speech_gate:
            bonus = auto_tune_bonus_masked(val_logps, val_masks,
                                            val_target, MIN_PRED_FRAMES,
                                            speech_frac=speech_frac_min)
        else:
            bonus = auto_tune_bonus(val_logps, val_target, MIN_PRED_FRAMES)
    else:
        bonus = float(SELF_LOOP_BONUS)

    # ── Step 4: refit LDA on full train ───────────────────────────
    X_train, y_train = build_train_set(set(all_real) - test_sent_ids)
    train_classes = set(y_train)
    scaler = StandardScaler().fit(X_train)
    clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    clf.fit(scaler.transform(X_train), y_train)
    class_labels = list(clf.classes_)

    # ── Step 5: decode test with optional speech gating ───────────
    predictions, pred_sentence_ids, pred_segments = [], [], []
    true_labels, true_sentence_ids, true_segments = [], [], []
    n_dropped_silence = 0
    for sent_idx in test_sent_ids:
        if sent_idx not in per_sentence_stk: continue
        stk = per_sentence_stk[sent_idx]; T_stk = stk.shape[0]
        mask = per_sentence_mask.get(sent_idx, np.ones(T_stk, dtype=bool))
        logp = clf.predict_log_proba(scaler.transform(stk))
        logp = smooth_cols(logp, SMOOTH_LOGP_W)
        path = viterbi_decode(logp, bonus)
        i = 0
        while i < T_stk:
            ci = path[i]; j = i + 1
            while j < T_stk and path[j] == ci: j += 1
            if (j - i) >= MIN_PRED_FRAMES:
                if mask[i:j].mean() >= speech_frac_min:
                    predictions.append(class_labels[ci])
                    pred_sentence_ids.append(sent_idx)
                    pred_segments.append((stk_frame_to_time_s(i),
                                          stk_frame_to_time_s(j - 1)))
                else:
                    n_dropped_silence += 1
            i = j
        for ph in mfa[sent_idx]:
            if ph['phone'] not in train_classes: continue
            true_labels.append(ph['phone'])
            true_sentence_ids.append(sent_idx)
            true_segments.append((ph['start_s'], ph['end_s']))

    if not true_labels: return None, "no test gold labels"
    true_arr = np.array(true_labels); pred_arr = np.array(predictions)
    ed = edit_distance(list(true_arr), list(pred_arr))
    per = ed / max(len(true_arr), 1)
    return {
        'true_labels': true_arr, 'predictions': pred_arr,
        'true_sentence_ids': np.array(true_sentence_ids),
        'pred_sentence_ids': np.array(pred_sentence_ids),
        'true_segments': true_segments, 'pred_segments': pred_segments,
        'edit_distance': ed, 'per': per,
        'n_test': len(true_arr), 'n_pred': len(pred_arr),
        'n_train': len(X_train), 'n_val_sents': len(val_logps),
        'bonus': bonus, 'val_target': val_target,
        'n_dropped_silence': n_dropped_silence,
    }, None

# Sanity check if speech detedctor works
import scipy
from scipy.signal import iirfilter
for use_gate in [False, True]:
    res, _ = run_for_patient_sd('P22', feature_spec={'hg_amp': True},
                                 use_speech_gate=use_gate)
    print(f"gate={use_gate}  z=?  n_pred={res['n_pred']}/{res['n_test']}  "
          f"bonus={res['bonus']:.2f}  dropped_silence={res['n_dropped_silence']}")

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier

def _make_classifier(kind, n_estimators=200, seed=0):
    if kind == 'rf':
        return RandomForestClassifier(
            n_estimators=n_estimators, n_jobs=-1,
            random_state=seed,
            max_features='sqrt')           # ★ no class_weight
    elif kind == 'lda':
        return LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')

def _predict_log_proba_safe(clf, X, temperature=1.0,
                             log_prior=None, alpha_prior=0.0):
    from scipy.special import logsumexp
    p = clf.predict_proba(X)
    log_p = np.log(np.clip(p, 1e-10, 1.0))
    if temperature != 1.0:
        log_p = log_p / temperature
    if alpha_prior > 0 and log_prior is not None:
        log_p = log_p - alpha_prior * log_prior
    log_p -= logsumexp(log_p, axis=1, keepdims=True)
    return log_p

def run_for_patient_sd_clf(pid, classifier_type='rf',
                            n_estimators=200, n_pca=100,
                            proba_temperature=1.0, 
                            alpha_prior=0.0,  
                            test_offset=TEST_OFFSET, feature_spec=None,
                            use_speech_gate=USE_SPEECH_GATE,
                            speech_thresh=SPEECH_THRESHOLD,
                            speech_frac_min=SPEECH_FRAC_MIN):
    """run_for_patient_sd with swappable classifier (rf/lda) and PCA reduction."""
    feature_spec = feature_spec or DEFAULT_FEATURE_SPEC
    wd = pipeline.split_result['word_segments_dict'].get(pid)
    if wd is None: return None, "no word_segments_dict"
    try: mfa = load_mfa_alignments(pid)
    except Exception as e: return None, f"no MFA ({type(e).__name__})"
    if not mfa: return None, "empty MFA"
    if use_speech_gate and pid not in mu_sd_speech:
        return None, f"no speech-detector stats for {pid}"

    all_real = [i for i, s in enumerate(wd['sentence_list'])
                if isinstance(s, dict) and s.get('text')]
    test_sent_ids = set(all_real[test_offset::6])
    train_sent_ids_all = sorted(set(all_real) - test_sent_ids)
    val_step = max(2, int(round(1 / VAL_FRAC)))
    val_sent_ids = set(train_sent_ids_all[::val_step])
    fit_sent_ids = set(train_sent_ids_all) - val_sent_ids

    raw_path = os.path.join(DUTCH_30_PATH, 'raw', f'{pid}_sEEG.npy')
    raw_eeg = np.load(raw_path)
    if raw_eeg.ndim == 2 and raw_eeg.shape[0] < raw_eeg.shape[1]:
        raw_eeg = raw_eeg.T

    per_sentence_stk, per_sentence_mask = {}, {}
    for sent_idx in all_real:
        if sent_idx not in mfa or not mfa[sent_idx]: continue
        s = wd['sentence_list'][sent_idx]
        s0, s1 = s['stim_start_idx'], s['stim_end_idx']
        if s1 > raw_eeg.shape[0]: continue
        ext = extract_features_multiband(raw_eeg[s0:s1], **feature_spec)
        if ext.shape[0] < 2 * LDA_MARGIN + 1: continue
        per_sentence_stk[sent_idx] = stackFeatures(ext, modelOrder=MO, stepSize=SS)
        if use_speech_gate:
            sp = predict_speech_prob(raw_eeg[s0:s1], pid)
            T_stk = per_sentence_stk[sent_idx].shape[0]
            mask = (sp[:T_stk] > speech_thresh) if len(sp) >= T_stk else \
                   np.concatenate([sp > speech_thresh, np.zeros(T_stk - len(sp), dtype=bool)])
            per_sentence_mask[sent_idx] = mask
    if not per_sentence_stk: return None, "no usable sentences"

    GROUP_DELAY_FRAMES = 10
    def build_train_set(sent_ids):
        causal_mode = feature_spec.get('causal', False)
        shift = GROUP_DELAY_FRAMES if causal_mode else 0
        X, y = [], []
        for sent_idx in sent_ids:
            if sent_idx not in per_sentence_stk: continue
            stk = per_sentence_stk[sent_idx]; T_stk = stk.shape[0]
            for ph in mfa[sent_idx]:
                k_s = int(np.ceil ((ph['start_s']*EEG_SR - WIN_SAMP/2) / SHIFT_SAMP))
                k_e = int(np.floor((ph['end_s']  *EEG_SR - WIN_SAMP/2) / SHIFT_SAMP))
                if (k_e - k_s + 1) < MN_FRAMES or (k_e - k_s + 1) > MX_FRAMES: continue
                ks = max(0, k_s - LDA_MARGIN + shift); ke = min(T_stk-1, k_e - LDA_MARGIN + shift)
                if ke < ks: continue
                X.append(stk[ks:ke+1].mean(axis=0)); y.append(ph['phone'])
        return np.array(X, dtype=np.float32), np.array(y)

    # ── Step 1: fit on 85% for bonus tuning ──
    X_fit, y_fit = build_train_set(fit_sent_ids)
    if len(X_fit) < 50: return None, f"too few fit samples ({len(X_fit)})"
    fit_classes = set(y_fit)
    sc_fit = StandardScaler().fit(X_fit)
    n_comp_fit = min(n_pca, X_fit.shape[1], X_fit.shape[0] - 1)
    pca_fit = PCA(n_components=n_comp_fit, svd_solver='randomized', random_state=0)
    Xp_fit  = pca_fit.fit_transform(sc_fit.transform(X_fit))
    clf_fit = _make_classifier(classifier_type, n_estimators=n_estimators)
    clf_fit.fit(Xp_fit, y_fit)
    fit_priors = np.array([np.mean(y_fit == c) for c in clf_fit.classes_])
    log_fit_prior = np.log(fit_priors + 1e-12)

    # ── Step 2: val log-probs ──
    val_logps, val_masks, val_target = [], [], 0
    for sent_idx in val_sent_ids:
        if sent_idx not in per_sentence_stk: continue
        stk = per_sentence_stk[sent_idx].astype(np.float32)
        Xs = pca_fit.transform(sc_fit.transform(stk))
        logp = _predict_log_proba_safe(clf_fit, Xs,
                                 temperature=proba_temperature,
                                 log_prior=log_fit_prior,
                                 alpha_prior=alpha_prior)

        logp = smooth_cols(logp, SMOOTH_LOGP_W)
        val_logps.append(logp)
        if use_speech_gate: val_masks.append(per_sentence_mask[sent_idx])
        val_target += sum(1 for ph in mfa[sent_idx] if ph['phone'] in fit_classes)
    if not val_logps: return None, "no val sentences"
    val_target = int(val_target * TARGET_RATIO)

    # ── Step 3: tune bonus ──
    if SELF_LOOP_BONUS is None:
        bonus = (auto_tune_bonus_masked(val_logps, val_masks, val_target,
                                         MIN_PRED_FRAMES, speech_frac=speech_frac_min)
                 if use_speech_gate
                 else auto_tune_bonus(val_logps, val_target, MIN_PRED_FRAMES))
    else:
        bonus = float(SELF_LOOP_BONUS)

    # ── Step 4: refit on full train ──
    X_train, y_train = build_train_set(set(all_real) - test_sent_ids)
    train_classes = set(y_train)
    scaler = StandardScaler().fit(X_train)
    n_comp = min(n_pca, X_train.shape[1], X_train.shape[0] - 1)
    pca = PCA(n_components=n_comp, svd_solver='randomized', random_state=0)
    Xp_train = pca.fit_transform(scaler.transform(X_train))
    clf = _make_classifier(classifier_type, n_estimators=n_estimators)
    clf.fit(Xp_train, y_train)
    class_labels = list(clf.classes_)
    train_priors = np.array([np.mean(y_train == c) for c in clf.classes_])
    log_train_prior = np.log(train_priors + 1e-12)

    # ── Step 5: decode test ──
    predictions, pred_sentence_ids, pred_segments = [], [], []
    true_labels, true_sentence_ids, true_segments = [], [], []
    n_dropped_silence = 0
    for sent_idx in test_sent_ids:
        if sent_idx not in per_sentence_stk: continue
        stk = per_sentence_stk[sent_idx].astype(np.float32); T_stk = stk.shape[0]
        mask = per_sentence_mask.get(sent_idx, np.ones(T_stk, dtype=bool))
        Xs = pca.transform(scaler.transform(stk))
        logp = _predict_log_proba_safe(clf, Xs,
                                 temperature=proba_temperature,
                                 log_prior=log_train_prior,
                                 alpha_prior=alpha_prior)
        logp = smooth_cols(logp, SMOOTH_LOGP_W)
        path = viterbi_decode(logp, bonus)
        i = 0
        while i < T_stk:
            ci = path[i]; j = i + 1
            while j < T_stk and path[j] == ci: j += 1
            if (j - i) >= MIN_PRED_FRAMES:
                if mask[i:j].mean() >= speech_frac_min:
                    predictions.append(class_labels[ci])
                    pred_sentence_ids.append(sent_idx)
                    pred_segments.append((stk_frame_to_time_s(i), stk_frame_to_time_s(j - 1)))
                else:
                    n_dropped_silence += 1
            i = j
        for ph in mfa[sent_idx]:
            if ph['phone'] not in train_classes: continue
            true_labels.append(ph['phone'])
            true_sentence_ids.append(sent_idx)
            true_segments.append((ph['start_s'], ph['end_s']))

    if not true_labels: return None, "no test gold labels"
    true_arr = np.array(true_labels); pred_arr = np.array(predictions)
    ed = edit_distance(list(true_arr), list(pred_arr))
    per = ed / max(len(true_arr), 1)
    return {
        'true_labels': true_arr, 'predictions': pred_arr,
        'true_sentence_ids': np.array(true_sentence_ids),
        'pred_sentence_ids': np.array(pred_sentence_ids),
        'true_segments': true_segments, 'pred_segments': pred_segments,
        'edit_distance': ed, 'per': per,
        'n_test': len(true_arr), 'n_pred': len(pred_arr),
        'n_train': len(X_train),                # ★ add
        'n_val_sents': len(val_logps),          # ★ add
        'bonus': bonus, 'val_target': val_target,
        'n_dropped_silence': n_dropped_silence,
        'classifier_type': classifier_type,
    }, None

import time, warnings
from sklearn.exceptions import ConvergenceWarning, DataConversionWarning
warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings('ignore', category=DataConversionWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*divide by zero.*')
warnings.filterwarnings('ignore', message='.*invalid value.*')

if not hasattr(pipeline, 'patient_results') or pipeline.patient_results is None:
    pipeline.patient_results = {}

CLASSIFIER = 'rf'   # ← 'rf' or 'lda' — your headline choice
ALPHA_PRIOR = 0.7

print(f"=== Running gated ({CLASSIFIER.upper()}) for {len(TARGET_PIDS)} patients "
      f"(offset={TEST_OFFSET}) ===\n", flush=True)
t_start = time.time()

for pid in TARGET_PIDS:
    t0 = time.time()
    print(f"[{pid}]", end=' ', flush=True)
    try:
        result, reason = run_for_patient_sd_clf(
            pid,
            classifier_type=CLASSIFIER,
            proba_temperature=1.0,
            alpha_prior=ALPHA_PRIOR,
            feature_spec=DEFAULT_FEATURE_SPEC,
            use_speech_gate=True,
        )
        if result is None:
            print(f"SKIPPED — {reason}", flush=True); continue
        pipeline.patient_results[pid] = result
        print(f"n_train={result['n_train']:>5}  n_test={result['n_test']:>4}  "
              f"n_pred={result['n_pred']:>4}  bonus={result['bonus']:>5.2f}  "
              f"dropped={result['n_dropped_silence']:>4}  "
              f"(val_target={result['val_target']}, val_sents={result['n_val_sents']})  "
              f"PER={100*result['per']:5.1f}%  "
              f"({time.time()-t0:.0f}s)", flush=True)
    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}", flush=True)

print(f"\nTotal: {time.time()-t_start:.0f}s  "
      f"({len(pipeline.patient_results)} patients done)", flush=True)

from e2e_brain_decoder import show_matched_sequences_with_times

for pid in TARGET_PIDS:
    if pid not in pipeline.patient_results: continue
    # print(f"\n{'='*70}\n{pid}\n{'='*70}")
    show_matched_sequences_with_times(
        pipeline, pid,
        max_per_line=45,
        collapse_repeats=False,
        # time_align_tol_s=0.1 
    )

# #### Sweep through features

# sweep through feature set
import time
import re

HG_LP_HZ_FIXED    = 10.0   # held constant across feature sweep
SMOOTH_LOGP_FIXED = 31     # held constant across feature sweep

FEATURE_SPECS = {
    'HG_amp (baseline)':       {'hg_amp': True},
    'LG_amp only':             {'lg_amp': True},
    'HG_amp + LG_amp':         {'hg_amp': True, 'lg_amp': True},
    'HG_amp + theta_phase':    {'hg_amp': True, 'theta_phase': True},
    'HG_amp + PAC':            {'hg_amp': True, 'theta_hg_pac': True},
    'HG_amp + LG + PAC':       {'hg_amp': True, 'lg_amp': True, 'theta_hg_pac': True},
    'HG + LG-theta PAC':       {'hg_amp': True, 'lg_theta_pac': True},
    'HG + LG + LG-theta PAC':  {'hg_amp': True, 'lg_amp': True, 'lg_theta_pac': True},
    'HG × LG':                 {'hg_x_lg': True},
    'HG + HG×LG':              {'hg_amp': True, 'hg_x_lg': True},
    'HG + LG + HG×LG':         {'hg_amp': True, 'lg_amp': True, 'hg_x_lg': True},
}

SWEEP_PIDS     = [f'P{i:02d}' for i in range(21, 31)]
RARE_TOP_N     = 5
BASELINE_LABEL = 'HG_amp (baseline)'

# ---- helpers ----
ANSI_RE = re.compile(r'\x1b\[[0-9;]*m')

def _visible_len(s):
    return len(ANSI_RE.sub('', s))

def _pad(s, width):
    return ' ' * max(0, width - _visible_len(s)) + s

def z_str(z, w=6):
    s = f"{z:+{w}.2f}"
    return f"\033[91m{s}\033[0m" if z > 2 else s

def per_str(per, w=4, decimals=0):
    fmt = f"{{:>{w}.{decimals}%}}"
    s = fmt.format(per)
    return f"\033[91m{s}\033[0m" if per < 0.80 else s

def delta_per_str(per, baseline_per, w=6):
    if baseline_per is None:
        return f"{'—':>{w}}"
    d = per - baseline_per
    s = f"{d:+{w-1}.1%}"
    if d <= -0.02: return f"\033[92m{s}\033[0m"
    if d >= +0.02: return f"\033[91m{s}\033[0m"
    return s

def delta_z_str(z, baseline_z, w=6):
    if baseline_z is None:
        return f"{'—':>{w}}"
    d = z - baseline_z
    s = f"{d:+{w-1}.2f}"
    if d >=  0.5: return f"\033[92m{s}\033[0m"
    if d <= -0.5: return f"\033[91m{s}\033[0m"
    return s

def count_rare_in_matches(matches, gold_sents, top_n=RARE_TOP_N):
    all_gold = [ph for s in gold_sents for ph in s]
    common = set(p for p, _ in Counter(all_gold).most_common(top_n))
    return sum(1 for m in matches for ph in m if ph not in common)

# ---- sweep ----
sweep_results = {}
print(f"=== Feature sweep: {len(FEATURE_SPECS)} configs × {len(SWEEP_PIDS)} patients ===")
print(f"   baseline = '{BASELINE_LABEL}'")
print(f"   hg_lp_hz = {HG_LP_HZ_FIXED}, SMOOTH_LOGP_W = {SMOOTH_LOGP_FIXED}")
print(f"   (rare = phonemes outside the top-{RARE_TOP_N} most common gold tokens)\n")

t_start = time.time()

for label, spec in FEATURE_SPECS.items():
    print(f"\n--- {label}  spec={spec} ---")
    SMOOTH_LOGP_W = SMOOTH_LOGP_FIXED
    spec = {**spec, 'hg_lp_hz': HG_LP_HZ_FIXED}

    for pid in SWEEP_PIDS:
        t0 = time.time()
        try:
            result, reason = run_for_patient_sd_clf(
                pid,
                classifier_type='rf',
                proba_temperature=1.0,
                alpha_prior=0.7,
                feature_spec=spec,
                use_speech_gate=True)
            if result is None:
                print(f"  {pid}: SKIPPED — {reason}"); continue
            pr = result
            pred_sents, gold_sents = [], []
            for sid in np.unique(pr['true_sentence_ids']):
                pred_sents.append(list(pr['predictions'][pr['pred_sentence_ids'] == sid]))
                gold_sents.append(list(pr['true_labels'][pr['true_sentence_ids'] == sid]))
            all_gold = [ph for s in gold_sents for ph in s]
            cnt = Counter(all_gold); N = sum(cnt.values())
            gold_lp = {k: np.log(v / N) for k, v in cnt.items()}
            max_run = max((longest_run_with_shift(p, g)
                           for p, g in zip(pred_sents, gold_sents)), default=0)
            matches  = collect_matches(pred_sents, gold_sents)
            n_rare   = count_rare_in_matches(matches, gold_sents)
            obs      = surprise_score(matches, gold_lp)
            nulls    = perm_null(pred_sents, gold_sents, gold_lp)
            z        = float((obs - nulls.mean()) / (nulls.std() + 1e-9))

            sweep_results[(label, pid)] = {
                'z': z, 'longest': max_run, 'ngrams': len(matches),
                'rare': n_rare, 'per': pr['per'],
                'n_pred': pr['n_pred'], 'n_test': pr['n_test'],
                'bonus': pr['bonus'], 'runtime': time.time() - t0,
                'dropped': pr.get('n_dropped_silence', 0),
            }

            base = sweep_results.get((BASELINE_LABEL, pid))
            base_per = base['per'] if base else None
            base_z   = base['z']   if base else None
            d_per_txt = "" if label == BASELINE_LABEL else f"  ΔPER={delta_per_str(pr['per'], base_per)}"
            d_z_txt   = "" if label == BASELINE_LABEL else f"  Δz={delta_z_str(z, base_z)}"

            print(f"  {pid}: z={z_str(z)}  longest={max_run}  "
                  f"ngrams={len(matches)}  rare={n_rare:>2}  "
                  f"PER={pr['per']:.0%}  dropped={pr.get('n_dropped_silence', 0):>3}"
                  f"{d_z_txt}{d_per_txt}  ({time.time()-t0:.0f}s)")
        except Exception as e:
            print(f"  {pid}: FAILED {type(e).__name__}: {e}")

    # ── per-config mini-summary across patients ────────────────────
    rs_cell = [sweep_results[(label, pid)] for pid in SWEEP_PIDS
               if (label, pid) in sweep_results]
    if rs_cell:
        tot_ngrams  = sum(r['ngrams']  for r in rs_cell)
        max_longest = max(r['longest'] for r in rs_cell)
        n_ge4       = sum(1 for r in rs_cell if r['longest'] >= 4)
        mean_z      = float(np.mean([r['z'] for r in rs_cell]))
        n_z2        = sum(1 for r in rs_cell if r['z'] > 2)
        mean_drop   = float(np.mean([r['dropped'] for r in rs_cell]))
        print(f"  → cell summary: max longest={max_longest}  "
              f"#≥4-gram={n_ge4}  tot ngrams={tot_ngrams}  "
              f"mean z={z_str(mean_z)}  #z>+2={n_z2}  "
              f"mean drop={mean_drop:.0f}")

print(f"\nTotal: {(time.time()-t_start)/60:.1f} min\n")

# ---- Summary 1: z grid (configs × patients) ----
print("=" * 110)
print("Summary 1: z values (red = z > +2)\n")
print(f"{'config':<27} " + "  ".join(f"{p:>8}" for p in SWEEP_PIDS) + "  best z")
print('-' * (27 + 10 * len(SWEEP_PIDS) + 10))
for label in FEATURE_SPECS:
    cells, zs = [], []
    for pid in SWEEP_PIDS:
        r = sweep_results.get((label, pid))
        if r is None: cells.append(_pad("--", 8))
        else:
            cells.append(_pad(z_str(r['z']), 8))
            zs.append(r['z'])
    best = z_str(max(zs)) if zs else "--"
    print(f"{label:<27} " + "  ".join(cells) + f"  {_pad(best, 6)}")

# ---- Summary 2: detail (longest/ngrams/rare/PER  ΔPER  bonus) ----
print()
print("=" * 130)
print("Summary 2: longest / ngrams / rare / PER  ΔPER  bonus    "
      "(red PER<80%, green ΔPER≤−2pp, red ΔPER≥+2pp)\n")
print(f"{'config':<27} " + "  ".join(f"{p:>22}" for p in SWEEP_PIDS))
print('-' * (27 + 24 * len(SWEEP_PIDS)))
for label in FEATURE_SPECS:
    cells = []
    for pid in SWEEP_PIDS:
        r = sweep_results.get((label, pid))
        if r is None:
            cells.append(f"{'--':>22}")
        else:
            base = sweep_results.get((BASELINE_LABEL, pid))
            base_per = base['per'] if base else None
            head = f"{r['longest']}/{r['ngrams']}/{r['rare']}/{per_str(r['per'])}"
            if label == BASELINE_LABEL:
                txt = f"{head}  b:{r['bonus']:4.1f}"
            else:
                txt = f"{head} Δ{delta_per_str(r['per'], base_per, w=5)} b:{r['bonus']:4.1f}"
            cells.append(_pad(txt, 22))
    print(f"{label:<27} " + "  ".join(cells))

# ---- Per-patient best ----
print()
print("=" * 110)
print("Per-patient best config (by z):\n")
print(f"{'PID':>5}  {'best config':<27}  {'z':>7}  {'Δz':>6}  {'longest':>7}  "
      f"{'ngrams':>6}  {'rare':>5}  {'PER':>6}  {'ΔPER':>6}  {'bonus':>6}")
print('-' * 110)
for pid in SWEEP_PIDS:
    best_label, best_r = None, None
    for label in FEATURE_SPECS:
        r = sweep_results.get((label, pid))
        if r is None: continue
        if best_r is None or r['z'] > best_r['z']:
            best_label, best_r = label, r
    if best_r is None:
        print(f"{pid:>5}  no results")
    else:
        base = sweep_results.get((BASELINE_LABEL, pid))
        base_per = base['per'] if base else None
        base_z   = base['z']   if base else None
        dper = "—" if best_label == BASELINE_LABEL else delta_per_str(best_r['per'], base_per, w=5)
        dz   = "—" if best_label == BASELINE_LABEL else delta_z_str(best_r['z'], base_z, w=5)
        print(f"{pid:>5}  {best_label:<27}  {_pad(z_str(best_r['z']), 7)}  "
              f"{_pad(dz, 6)}  {best_r['longest']:>7}  {best_r['ngrams']:>6}  "
              f"{best_r['rare']:>5}  {_pad(per_str(best_r['per']), 6)}  "
              f"{_pad(dper, 6)}  {best_r['bonus']:>6.2f}")

# ---- Aggregate per config across patients ----
print()
print("=" * 130)
print("Aggregate per config across patients:\n")
print(f"{'config':<27}  {'mean z':>7}  {'Δmean z':>8}  {'#z>+2':>6}  "
      f"{'mean PER':>9}  {'ΔmeanPER':>9}  {'#PER<80%':>9}  {'mean bonus':>11}  "
      f"{'#pat≥4':>7}  {'#pat≥3':>7}  {'#pat≥2':>7}")
print('-' * 130)
base_mean_z   = None
base_mean_per = None
if all((BASELINE_LABEL, pid) in sweep_results for pid in SWEEP_PIDS):
    base_rs = [sweep_results[(BASELINE_LABEL, pid)] for pid in SWEEP_PIDS]
    base_mean_z   = float(np.mean([r['z']   for r in base_rs]))
    base_mean_per = float(np.mean([r['per'] for r in base_rs]))

for label in FEATURE_SPECS:
    rs = [sweep_results[(label, pid)] for pid in SWEEP_PIDS
          if (label, pid) in sweep_results]
    if not rs:
        print(f"{label:<27}  no results"); continue
    mean_z   = float(np.mean([r['z'] for r in rs]))
    mean_per = float(np.mean([r['per'] for r in rs]))
    mean_bon = float(np.mean([r['bonus'] for r in rs]))
    n_z2     = sum(1 for r in rs if r['z'] > 2)
    n_per    = sum(1 for r in rs if r['per'] < 0.80)
    pat_ge4  = sum(1 for r in rs if r['longest'] >= 4)
    pat_ge3  = sum(1 for r in rs if r['longest'] >= 3)
    pat_ge2  = sum(1 for r in rs if r['longest'] >= 2)

    if label == BASELINE_LABEL:
        dz_txt   = _pad('—', 8)
        dper_txt = _pad('—', 9)
    else:
        dz_txt   = _pad(delta_z_str(mean_z, base_mean_z, w=7), 8)
        dper_txt = _pad(delta_per_str(mean_per, base_mean_per, w=8), 9)
    print(f"{label:<27}  {_pad(z_str(mean_z), 7)}  {dz_txt}  {n_z2:>6}  "
          f"{_pad(per_str(mean_per), 9)}  {dper_txt}  {n_per:>9}  "
          f"{mean_bon:>11.2f}  {pat_ge4:>7}  {pat_ge3:>7}  {pat_ge2:>7}")

# Side-by-side examples with different feature sets 
# ============================================================
from IPython.utils.capture import capture_output
from IPython.display import HTML, display

# --- pick patient and the two configs to compare --------------------
PID = 'P24'
SPEC_A = ('HG_amp baseline',            {'hg_amp': True})
# SPEC_B = ('HG × LG',                   {'hg_x_lg': True})
# SPEC_B = ('LG_amp',                     {'hg_amp': False, 'lg_amp': True})
SPEC_B = ('HG + LG-theta PAC',         {'hg_amp': True, 'lg_theta_pac': True})
# SPEC_B = (     'HG_amp + LG_amp',     {'hg_amp': True, 'lg_amp': True})
# SPEC_B = (    'HG_amp + LG + PAC',   {'hg_amp': True, 'lg_amp': True, 'theta_hg_pac': True})
# SPEC_B = ( 'HG_amp + PAC',              {'hg_amp': True, 'theta_hg_pac': True})

# SPEC_B = ('HG_amp + theta_phase',       {'hg_amp': True, 'theta_phase': True})
# --------------------------------------------------------------------

if not hasattr(pipeline, 'patient_results') or pipeline.patient_results is None:
    pipeline.patient_results = {}

def render_to_html(pid, spec):
    """Run run_for_patient + score + visualization, capture as one HTML blob."""
    result, reason = run_for_patient_sd_clf(
                pid,
                classifier_type='rf',
                proba_temperature=1.0,
                alpha_prior=0.7,
                feature_spec=spec,
                use_speech_gate=True)
    if result is None:
        return f"<p style='color:#a00'>SKIPPED — {reason}</p>", None
    pipeline.patient_results[pid] = result

    # Compute summary scores
    pr = result
    pred_sents, gold_sents = [], []
    for sid in np.unique(pr['true_sentence_ids']):
        pred_sents.append(list(pr['predictions'][pr['pred_sentence_ids'] == sid]))
        gold_sents.append(list(pr['true_labels'][pr['true_sentence_ids'] == sid]))
    all_gold = [ph for s in gold_sents for ph in s]
    cnt = Counter(all_gold); N = sum(cnt.values())
    gold_lp = {k: np.log(v / N) for k, v in cnt.items()}
    max_run = max((longest_run_with_shift(p, g)
                   for p, g in zip(pred_sents, gold_sents)), default=0)
    matches = collect_matches(pred_sents, gold_sents)
    obs   = surprise_score(matches, gold_lp)
    nulls = perm_null(pred_sents, gold_sents, gold_lp)
    z     = float((obs - nulls.mean()) / (nulls.std() + 1e-9))
    summary = {'z': z, 'longest': max_run, 'ngrams': len(matches),
               'n_pred': pr['n_pred'], 'n_test': pr['n_test'],
               'bonus': pr['bonus'], 'per': pr['per']}

    # Capture the textual + HTML output of the standard visualizer
    with capture_output() as cap:
        show_matched_sequences_with_times(
            pipeline, pid,
            max_per_line=20,
            collapse_repeats=False,
            # time_align_tol_s=0.1,
        )

    html_parts = []
    if cap.stdout:
        # The print-based summary header → wrap in <pre> for monospace
        html_parts.append(
            f"<pre style='font-size:11px; margin:4px 0; "
            f"white-space:pre-wrap;'>{cap.stdout}</pre>"
        )
    for output in cap.outputs:
        if hasattr(output, 'data') and 'text/html' in output.data:
            html_parts.append(output.data['text/html'])
    return '\n'.join(html_parts), summary

print(f"=== Comparing {PID}:  '{SPEC_A[0]}'  vs  '{SPEC_B[0]}'  ===")

html_a, sum_a = render_to_html(PID, SPEC_A[1])
html_b, sum_b = render_to_html(PID, SPEC_B[1])

# Headline numbers
def _color_z(z):
    return f"<span style='color:#c00; font-weight:bold;'>{z:+.2f}</span>" if z > 2 \
        else f"<span style='font-weight:bold;'>{z:+.2f}</span>"

def _header_block(label, spec, s):
    if s is None: return f"<h3>{label}</h3><p>no result</p>"
    return (
        f"<h3 style='margin:4px 0;'>{label}</h3>"
        f"<div style='font-family:monospace; font-size:12px; "
        f"padding:6px; background:#f4f4f4; border-radius:4px; margin-bottom:8px;'>"
        f"spec={spec}<br>"
        f"z={_color_z(s['z'])} &nbsp; longest n-gram={s['longest']} &nbsp; "
        f"n-grams matched={s['ngrams']} &nbsp; PER={s['per']:.1%}<br>"
        f"bonus={s['bonus']:.2f} &nbsp; "
        f"n_pred={s['n_pred']}/{s['n_test']}"
        f"</div>"
    )

combined = f"""
<div style="display:flex; gap:20px; align-items:flex-start;">
  <div style="flex:1; min-width:0; overflow-x:auto;">
    {_header_block(SPEC_A[0], SPEC_A[1], sum_a)}
    {html_a}
  </div>
  <div style="flex:1; min-width:0; overflow-x:auto;">
    {_header_block(SPEC_B[0], SPEC_B[1], sum_b)}
    {html_b}
  </div>
</div>
"""
display(HTML(combined))

# sweep through signal processing and smoothing constants
HG_LP_VALUES       = [8.0, 9.0, 10.0, 11.0, 12.0]
SMOOTH_LOGP_VALUES = [31]
SWEEP_PIDS         = [f'P{i:02d}' for i in range(21, 31)]
# 5 × 4 × 10 = 200 runs

import time
sweep_results = {}   # keyed by (lp, w, pid)

print(f"=== 2D sweep: hg_lp_hz ∈ {HG_LP_VALUES}  ×  "
      f"SMOOTH_LOGP_W ∈ {SMOOTH_LOGP_VALUES}  ×  {len(SWEEP_PIDS)} patients ===")
print(f"   total = {len(HG_LP_VALUES)*len(SMOOTH_LOGP_VALUES)*len(SWEEP_PIDS)} runs\n")
t_start = time.time()
done = 0
total = len(HG_LP_VALUES) * len(SMOOTH_LOGP_VALUES) * len(SWEEP_PIDS)

for lp in HG_LP_VALUES:
    for w in SMOOTH_LOGP_VALUES:
        SMOOTH_LOGP_W = w   # overwrite module-level constant
        spec = {'hg_amp': True, 'hg_lp_hz': lp}
        print(f"\n--- hg_lp={lp:>4.1f} Hz  SMOOTH_LOGP_W={w:>3} ({w*5} ms) ---")
        for pid in SWEEP_PIDS:
            t0 = time.time(); done += 1
            try:
                result, reason = run_for_patient_sd_clf(
                    pid,
                    classifier_type=CLASSIFIER,
                    proba_temperature=1.0,
                    alpha_prior=ALPHA_PRIOR,
                    feature_spec=DEFAULT_FEATURE_SPEC,
                    use_speech_gate=True,
                )
                if result is None:
                    print(f"  {pid}: SKIPPED — {reason}"); continue
                pr = result
                pred_sents, gold_sents = [], []
                for sid in np.unique(pr['true_sentence_ids']):
                    pred_sents.append(list(pr['predictions'][pr['pred_sentence_ids'] == sid]))
                    gold_sents.append(list(pr['true_labels'][pr['true_sentence_ids'] == sid]))
                all_gold = [ph for s in gold_sents for ph in s]
                cnt = Counter(all_gold); N = sum(cnt.values())
                gold_lp = {k: np.log(v / N) for k, v in cnt.items()}
                max_run = max((longest_run_with_shift(p, g)
                               for p, g in zip(pred_sents, gold_sents)), default=0)
                matches = collect_matches(pred_sents, gold_sents)
                obs   = surprise_score(matches, gold_lp)
                nulls = perm_null(pred_sents, gold_sents, gold_lp)
                z = float((obs - nulls.mean()) / (nulls.std() + 1e-9))

                matches_disp = collect_matches(pred_sents, gold_sents,
                                                min_match=2, shift_max=3)
                n_rare = count_rare_in_matches(matches_disp, gold_sents)
                longest_content = (' '.join(max(matches_disp, key=len))
                                    if matches_disp else '—')

                sweep_results[(lp, w, pid)] = {
                    'z': z, 'longest': max_run, 'ngrams': len(matches),
                    'rare': n_rare, 'per': pr['per'], 'bonus': pr['bonus'],
                    'longest_seq': longest_content,
                    'dropped': pr.get('n_dropped_silence', 0),
                }
                eta = (time.time() - t_start) / done * (total - done) / 60
                print(f"  {pid}: z={z_str(z)}  longest={max_run} [{longest_content}]  "
                      f"ngrams={len(matches):>2}  rare={n_rare:>2}  "
                      f"PER={per_str(pr['per'])}  bonus={pr['bonus']:5.2f}  "
                      f"dropped={pr.get('n_dropped_silence', 0):>3}  "
                      f"({time.time()-t0:.0f}s, ETA {eta:.0f}m)")
            except Exception as e:
                print(f"  {pid}: FAILED {type(e).__name__}: {e}")

        # ── per-(LP, W) mini-summary across patients ───────────────
        rs_cell = [sweep_results[(lp, w, pid)] for pid in SWEEP_PIDS
                   if (lp, w, pid) in sweep_results]
        if rs_cell:
            tot_ngrams  = sum(r['ngrams']  for r in rs_cell)
            max_longest = max(r['longest'] for r in rs_cell)
            n_ge4       = sum(1 for r in rs_cell if r['longest'] >= 4)
            mean_z      = float(np.mean([r['z'] for r in rs_cell]))
            n_z2        = sum(1 for r in rs_cell if r['z'] > 2)
            print(f"  → cell summary: max longest={max_longest}  "
                  f"#≥4-gram={n_ge4}  tot ngrams={tot_ngrams}  "
                  f"mean z={z_str(mean_z)}  #z>+2={n_z2}")

print(f"\nTotal: {(time.time()-t_start)/60:.1f} min\n")

# ============================================================
# Aggregate 1: LP × W heat-grid of mean z (across patients)
# ============================================================
print("=" * 110)
print("Mean z across patients  (rows=hg_lp_hz, cols=SMOOTH_LOGP_W)\n")
hdr = f"{'LP \\ W':>8}  " + "  ".join(f"{w:>7}" for w in SMOOTH_LOGP_VALUES)
print(hdr); print('-' * len(hdr))
for lp in HG_LP_VALUES:
    row = []
    for w in SMOOTH_LOGP_VALUES:
        rs = [sweep_results[(lp, w, pid)] for pid in SWEEP_PIDS
              if (lp, w, pid) in sweep_results]
        if not rs:
            row.append(_pad('--', 7))
        else:
            row.append(_pad(z_str(np.mean([r['z'] for r in rs]), w=7), 7))
    print(f"{lp:>8.1f}  " + "  ".join(row))

# ============================================================
# Aggregate 2: full table per (LP, W)
# ============================================================
print()
print("=" * 110)
print("Per-(LP, W) summary\n")
print(f"{'LP':>5}  {'W':>3}  {'mean z':>7}  {'#z>+2':>6}  {'mean bon':>9}  "
      f"{'mean PER':>9}  {'max longest':>11}  {'#≥4-gram':>9}  "
      f"{'tot ngrams':>10}  {'mean drop':>10}")
print('-' * 110)
for lp in HG_LP_VALUES:
    for w in SMOOTH_LOGP_VALUES:
        rs = [sweep_results[(lp, w, pid)] for pid in SWEEP_PIDS
              if (lp, w, pid) in sweep_results]
        if not rs: continue
        mean_z       = np.mean([r['z']     for r in rs])
        n_z2         = sum(1 for r in rs if r['z'] > 2)
        mean_bon     = np.mean([r['bonus'] for r in rs])
        mean_per     = np.mean([r['per']   for r in rs])
        max_longest  = max(r['longest']    for r in rs)
        n_ge4        = sum(1 for r in rs if r['longest'] >= 4)
        tot_ngrams   = sum(r['ngrams']     for r in rs)
        mean_dropped = np.mean([r['dropped'] for r in rs])
        print(f"{lp:>5.1f}  {w:>3}  {_pad(z_str(mean_z), 7)}  {n_z2:>6}  "
              f"{mean_bon:>9.2f}  {_pad(per_str(mean_per), 9)}  "
              f"{max_longest:>11}  {n_ge4:>9}  "
              f"{tot_ngrams:>10}  {mean_dropped:>10.0f}")

# ============================================================
# Aggregate 3: best (LP, W) per patient
# ============================================================
print()
print("=" * 110)
print("Per-patient best (LP, W) by z:\n")
print(f"{'PID':>5}  {'LP':>5}  {'W':>3}  {'z':>7}  {'longest':>7}  "
      f"{'longest seq':<20}  {'ngrams':>6}  {'rare':>5}  {'PER':>6}  {'bonus':>6}")
print('-' * 110)
for pid in SWEEP_PIDS:
    best_key, best_r = None, None
    for lp in HG_LP_VALUES:
        for w in SMOOTH_LOGP_VALUES:
            r = sweep_results.get((lp, w, pid))
            if r is None: continue
            if best_r is None or r['z'] > best_r['z']:
                best_key, best_r = (lp, w), r
    if best_r is None:
        print(f"{pid:>5}  no results"); continue
    lp, w = best_key
    print(f"{pid:>5}  {lp:>5.1f}  {w:>3}  {_pad(z_str(best_r['z']), 7)}  "
          f"{best_r['longest']:>7}  {best_r['longest_seq']:<20}  "
          f"{best_r['ngrams']:>6}  {best_r['rare']:>5}  "
          f"{_pad(per_str(best_r['per']), 6)}  {best_r['bonus']:>6.2f}")

# ============================================================
# Aggregate 4: marginal means per LP and per W
# ============================================================
print()
print("=" * 80)
print("Marginal means (averaged across the other axis):\n")
print("Per LP (averaged over all W and patients):")
print(f"{'LP':>5}  {'mean z':>7}  {'mean bon':>9}  {'mean PER':>9}  {'#z>+2':>6}")
print('-' * 50)
for lp in HG_LP_VALUES:
    rs = [sweep_results[(lp, w, pid)]
          for w in SMOOTH_LOGP_VALUES for pid in SWEEP_PIDS
          if (lp, w, pid) in sweep_results]
    if not rs: continue
    print(f"{lp:>5.1f}  {_pad(z_str(np.mean([r['z'] for r in rs])), 7)}  "
          f"{np.mean([r['bonus'] for r in rs]):>9.2f}  "
          f"{_pad(per_str(np.mean([r['per'] for r in rs])), 9)}  "
          f"{sum(1 for r in rs if r['z'] > 2):>6}")

print(f"\nPer W (averaged over all LP and patients):")
print(f"{'W':>3}  {'mean z':>7}  {'mean bon':>9}  {'mean PER':>9}  {'#z>+2':>6}")
print('-' * 50)
for w in SMOOTH_LOGP_VALUES:
    rs = [sweep_results[(lp, w, pid)]
          for lp in HG_LP_VALUES for pid in SWEEP_PIDS
          if (lp, w, pid) in sweep_results]
    if not rs: continue
    print(f"{w:>3}  {_pad(z_str(np.mean([r['z'] for r in rs])), 7)}  "
          f"{np.mean([r['bonus'] for r in rs]):>9.2f}  "
          f"{_pad(per_str(np.mean([r['per'] for r in rs])), 9)}  "
          f"{sum(1 for r in rs if r['z'] > 2):>6}")

from IPython.utils.capture import capture_output
from IPython.display import HTML, display

# Edit this list — each tuple is (PID, LP, W)
EXAMPLES = [
    ('P28', 8.0,  25),    # z=+3.26, PER=86% — real signal
    ('P28', 10.0, 31),    # z=+2.86, PER=74% — low PER but smeared
]

if not hasattr(pipeline, 'patient_results') or pipeline.patient_results is None:
    pipeline.patient_results = {}

def render_one(pid, lp, w):
    global SMOOTH_LOGP_W
    SMOOTH_LOGP_W = w
    spec = {'hg_amp': True, 'hg_lp_hz': lp}
    result, reason = run_for_patient_sd_clf(
            pid,
            classifier_type=CLASSIFIER,
            proba_temperature=1.0,
            alpha_prior=ALPHA_PRIOR,
            feature_spec=DEFAULT_FEATURE_SPEC,
            use_speech_gate=True,
        )
    if result is None:
        return f"<p style='color:#a00'>SKIPPED — {reason}</p>", None
    pipeline.patient_results[pid] = result

    pr = result
    pred_sents, gold_sents = [], []
    for sid in np.unique(pr['true_sentence_ids']):
        pred_sents.append(list(pr['predictions'][pr['pred_sentence_ids'] == sid]))
        gold_sents.append(list(pr['true_labels'][pr['true_sentence_ids'] == sid]))
    all_gold = [ph for s in gold_sents for ph in s]
    cnt = Counter(all_gold); N = sum(cnt.values())
    gold_lp = {k: np.log(v / N) for k, v in cnt.items()}
    max_run = max((longest_run_with_shift(p, g)
                   for p, g in zip(pred_sents, gold_sents)), default=0)
    matches = collect_matches(pred_sents, gold_sents)
    obs   = surprise_score(matches, gold_lp)
    nulls = perm_null(pred_sents, gold_sents, gold_lp)
    z     = float((obs - nulls.mean()) / (nulls.std() + 1e-9))

    matches_disp = collect_matches(pred_sents, gold_sents, min_match=2, shift_max=3)
    longest_seq = ' '.join(max(matches_disp, key=len)) if matches_disp else '—'

    summary = {'z': z, 'longest': max_run, 'ngrams': len(matches),
           'longest_seq': longest_seq, 'per': pr['per'],
           'bonus': pr['bonus'], 'n_pred': pr['n_pred'], 'n_test': pr['n_test'],
           'dropped': pr.get('n_dropped_silence', 0)}     # NEW

    with capture_output() as cap:
        show_matched_sequences_with_times(
            pipeline, pid,
            max_per_line=20,
            collapse_repeats=False,
            time_align_tol_s=0.1,
        )
    html_parts = []
    if cap.stdout:
        html_parts.append(f"<pre style='font-size:11px; margin:4px 0; "
                          f"white-space:pre-wrap;'>{cap.stdout}</pre>")
    for output in cap.outputs:
        if hasattr(output, 'data') and 'text/html' in output.data:
            html_parts.append(output.data['text/html'])
    return '\n'.join(html_parts), summary

def _zh(z):
    return (f"<span style='color:#c00; font-weight:bold;'>{z:+.2f}</span>"
            if z > 2 else f"<span style='font-weight:bold;'>{z:+.2f}</span>")
def _perh(p):
    return (f"<span style='color:#c00; font-weight:bold;'>{p:.1%}</span>"
            if p < 0.80 else f"{p:.1%}")

def header(pid, lp, w, s):
    if s is None: return f"<h3>{pid}  (LP={lp}, W={w})</h3><p>no result</p>"
    return (
        f"<h3 style='margin:4px 0;'>{pid}  (LP={lp}, W={w})</h3>"
        f"<div style='font-family:monospace; font-size:12px; padding:6px; "
        f"background:#f4f4f4; border-radius:4px; margin-bottom:8px;'>"
        f"z={_zh(s['z'])} &nbsp; longest={s['longest']} [{s['longest_seq']}] &nbsp; "
        f"ngrams={s['ngrams']} &nbsp; PER={_perh(s['per'])}<br>"
        f"bonus={s['bonus']:.2f} &nbsp; n_pred={s['n_pred']}/{s['n_test']}"
        f"dropped={s['dropped']}"
        f"</div>"
    )

columns = []
for pid, lp, w in EXAMPLES:
    html_blob, summary = render_one(pid, lp, w)
    columns.append(
        f"<div style='flex:1; min-width:0; overflow-x:auto;'>"
        f"{header(pid, lp, w, summary)}{html_blob}</div>"
    )
display(HTML(
    f"<div style='display:flex; gap:20px; align-items:flex-start;'>"
    f"{''.join(columns)}</div>"
))

# Run + score + visualize all patients with a chosen feature spec
# ============================================================
import time
from e2e_brain_decoder import show_matched_sequences_with_times

# --- Pick the feature config ----------------------------------------
FEATURE_SPEC = {'hg_amp': True, 'theta_phase': True}             # add theta phase

# {'hg_amp': True, 'theta_hg_pac': True}
# Other things to try:
#   {'hg_amp': True}                                  # baseline
#   {'hg_amp': True, 'lg_amp': True}                  # add low gamma

#   {'hg_amp': True, 'theta_hg_pac': True}            # add PAC
#   {'hg_amp': True, 'lg_amp': True, 'theta_hg_pac': True}
# --------------------------------------------------------------------

if not hasattr(pipeline, 'patient_results') or pipeline.patient_results is None:
    pipeline.patient_results = {}

print(f"=== {FEATURE_SPEC}  (offset={TEST_OFFSET}) ===\n")
t_start = time.time()
for pid in TARGET_PIDS:
    t0 = time.time()
    print(f"[{pid}]", end=' ', flush=True)
    try:
        result, reason = run_for_patient(pid, feature_spec=FEATURE_SPEC)
        if result is None:
            print(f"SKIPPED — {reason}"); continue
        pipeline.patient_results[pid] = result
        print(f"n_train={result['n_train']:>5}  n_test={result['n_test']:>4}  "
              f"n_pred={result['n_pred']:>4}  bonus={result['bonus']:.2f}  "
              f"({time.time()-t0:.0f}s)")
    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}")
print(f"\nTotal: {time.time()-t_start:.0f}s")

# ── Surprise-z table ───────────────────────────────────────────────
print(f"\n=== Surprise-z scoring ({FEATURE_SPEC}) ===\n")
print(f"{'PID':>5}  {'n_test':>6}  {'n_pred':>6}  {'bonus':>6}  "
      f"{'max_run':>7}  {'matches':>7}  {'z':>7}")
print('-' * 60)

scores = {}
for pid in TARGET_PIDS:
    if pid not in pipeline.patient_results: continue
    pr = pipeline.patient_results[pid]
    pred_sents, gold_sents = [], []
    for sid in np.unique(pr['true_sentence_ids']):
        pred_sents.append(list(pr['predictions'][pr['pred_sentence_ids'] == sid]))
        gold_sents.append(list(pr['true_labels'][pr['true_sentence_ids'] == sid]))
    all_gold = [ph for s in gold_sents for ph in s]
    if not all_gold: continue
    cnt = Counter(all_gold); N = sum(cnt.values())
    gold_lp = {k: np.log(v / N) for k, v in cnt.items()}
    max_run = max((longest_run_with_shift(p, g)
                   for p, g in zip(pred_sents, gold_sents)), default=0)
    matches = collect_matches(pred_sents, gold_sents)
    obs   = surprise_score(matches, gold_lp)
    nulls = perm_null(pred_sents, gold_sents, gold_lp)
    z     = float((obs - nulls.mean()) / (nulls.std() + 1e-9))
    scores[pid] = {'z': z, 'max_run': max_run, 'matches': len(matches)}
    print(f"{pid:>5}  {pr['n_test']:>6}  {pr['n_pred']:>6}  "
          f"{pr['bonus']:>6.2f}  {max_run:>7}  {len(matches):>7}  {z:+7.2f}")

# Summary across patients
zs = [s['z'] for s in scores.values()]
mrs = [s['max_run'] for s in scores.values()]
print(f"\nAcross {len(scores)} patients:")
print(f"  mean z   = {np.mean(zs):+.2f}  std = {np.std(zs):.2f}  "
      f"max = {max(zs):+.2f}")
print(f"  mean mr  = {np.mean(mrs):.1f}  "
      f"patients with z > +2: {sum(1 for z in zs if z > 2)}")

# ── Visualize each patient ─────────────────────────────────────────
import importlib, e2e_brain_decoder
importlib.reload(e2e_brain_decoder)
from e2e_brain_decoder import show_matched_sequences_with_times

for pid in TARGET_PIDS:
    if pid not in pipeline.patient_results: continue
    print(f"\n{'='*70}\n{pid}\n{'='*70}")
    show_matched_sequences_with_times(
        pipeline, pid,
        max_per_line=45,
        collapse_repeats=False,
        time_align_tol_s=0.1,    # align true ↔ pred columns by time
    )

# lop ctuoff sweep
import time, re

BASE_SPEC  = {'hg_amp': True}     # base spec; LP value will be added per iteration
LP_KEY     = 'smoothing_hz'       # change to 'hg_lp_hz' if you're on the per-band variant
LP_VALUES  = [8.0, 9.0, 10.0, 11.0, 12.0]
SWEEP_PIDS = [f'P{i:02d}' for i in range(21, 31)]
RARE_TOP_N = 5
LP_KEY = 'hg_lp_hz'   # instead of 'smoothing_hz'

ANSI_RE = re.compile(r'\x1b\[[0-9;]*m')
def _vlen(s): return len(ANSI_RE.sub('', s))
def _pad(s, width): return ' ' * max(0, width - _vlen(s)) + s
def z_str(z, w=6):
    s = f"{z:+{w}.2f}"
    return f"\033[91m{s}\033[0m" if z > 2 else s

def z_str(z, w=6):
    s = f"{z:+{w}.2f}"
    return f"\033[91m{s}\033[0m" if z > 2 else s

def per_str(per, w=4, decimals=0):
    fmt = f"{{:>{w}.{decimals}%}}"
    s = fmt.format(per)
    return f"\033[91m{s}\033[0m" if per < 0.80 else s
    
def length_breakdown(pred_sents, gold_sents):
    """Returns (n_2, n_3, n_4plus, longest, all_matches_min2)."""
    m_all = collect_matches(pred_sents, gold_sents, min_match=2, shift_max=3)
    n_2 = sum(1 for m in m_all if len(m) == 2)
    n_3 = sum(1 for m in m_all if len(m) == 3)
    n_4 = sum(1 for m in m_all if len(m) >= 4)
    longest = max((len(m) for m in m_all), default=0)
    return n_2, n_3, n_4, longest, m_all

def count_rare(matches, gold_sents, top_n=RARE_TOP_N):
    all_gold = [ph for s in gold_sents for ph in s]
    common = set(p for p, _ in Counter(all_gold).most_common(top_n))
    return sum(1 for m in matches for ph in m if ph not in common)

sweep_results = {}
print(f"=== LP sweep: {LP_KEY} ∈ {LP_VALUES}  ×  {len(SWEEP_PIDS)} patients ===")
print(f"   base spec = {BASE_SPEC}")
print(f"   columns: longest / 4+ / 3-gram / 2-gram / rare / PER\n")

t_start = time.time()
for lp in LP_VALUES:
    spec = {**BASE_SPEC, LP_KEY: lp}
    print(f"\n--- {LP_KEY}={lp} Hz ---")
    for pid in SWEEP_PIDS:
        t0 = time.time()
        try:
            result, reason = run_for_patient_sd(pid, feature_spec=spec, use_speech_gate=True)
            if result is None:
                print(f"  {pid}: SKIPPED — {reason}"); continue
            pr = result
            pred_sents, gold_sents = [], []
            for sid in np.unique(pr['true_sentence_ids']):
                pred_sents.append(list(pr['predictions'][pr['pred_sentence_ids'] == sid]))
                gold_sents.append(list(pr['true_labels'][pr['true_sentence_ids'] == sid]))
            all_gold = [ph for s in gold_sents for ph in s]
            cnt = Counter(all_gold); N = sum(cnt.values())
            gold_lp_dist = {k: np.log(v / N) for k, v in cnt.items()}

            n_2, n_3, n_4, longest, m_all = length_breakdown(pred_sents, gold_sents)
            n_rare = count_rare(m_all, gold_sents)
            m_z = collect_matches(pred_sents, gold_sents)        # min_match=3 for z
            obs   = surprise_score(m_z, gold_lp_dist)
            nulls = perm_null(pred_sents, gold_sents, gold_lp_dist)
            z     = float((obs - nulls.mean()) / (nulls.std() + 1e-9))

            sweep_results[(lp, pid)] = {
                'z': z, 'longest': longest,
                'n_4plus': n_4, 'n_3': n_3, 'n_2': n_2,
                'rare': n_rare, 'per': pr['per'],
                'n_pred': pr['n_pred'], 'n_test': pr['n_test'],
                'bonus': pr['bonus'], 'runtime': time.time() - t0,
                'dropped': pr.get('n_dropped_silence', 0),     # NEW
            }
            print(f"  {pid}: z={z_str(z)}  longest={longest}  "
              f"4+/3/2={n_4}/{n_3}/{n_2}  rare={n_rare:>2}  "
              f"PER={pr['per']:.0%}  dropped={pr.get('n_dropped_silence', 0):>3}  "
              f"({time.time()-t0:.0f}s)")
        except Exception as e:
            print(f"  {pid}: FAILED {type(e).__name__}: {e}")

print(f"\nTotal: {(time.time()-t_start)/60:.1f} min\n")

# ── Summary 1: z grid ──────────────────────────────────────────────
print("=" * 100)
print("Summary: z values  (red = z > +2)\n")
print(f"{'LP (Hz)':>8}  " + "  ".join(f"{p:>8}" for p in SWEEP_PIDS) + "  best z")
print('-' * (10 + 10 * len(SWEEP_PIDS) + 10))
for lp in LP_VALUES:
    cells, zs = [], []
    for pid in SWEEP_PIDS:
        r = sweep_results.get((lp, pid))
        if r is None: cells.append(_pad('--', 8))
        else:
            cells.append(_pad(z_str(r['z']), 8))
            zs.append(r['z'])
    best = z_str(max(zs)) if zs else '--'
    print(f"{lp:>8.1f}  " + "  ".join(cells) + f"  {_pad(best, 6)}")

# ── Summary 2: detail grid (longest / 4+ / 3-gram / 2-gram / rare / PER) ──
print()
print("=" * 100)
print("Summary: longest / 4+ / 3-gram / 2-gram / rare / PER\n")
print(f"{'LP':>5}  " + "  ".join(f"{p:>16}" for p in SWEEP_PIDS))
print('-' * (7 + 18 * len(SWEEP_PIDS)))
for lp in LP_VALUES:
    cells = []
    for pid in SWEEP_PIDS:
        r = sweep_results.get((lp, pid))
        if r is None: cells.append(f"{'--':>16}")
        else:
            txt = (f"{r['longest']}/{r['n_4plus']}/{r['n_3']}/"
                   f"{r['n_2']}/{r['rare']}/{r['per']:.0%}")
            cells.append(f"{txt:>16}")
    print(f"{lp:>5.1f}  " + "  ".join(cells))

# ── Per-patient best LP ────────────────────────────────────────────
print()
print("=" * 100)
print("Per-patient best LP (by z):\n")
print(f"{'PID':>5}  {'LP':>6}  {'z':>7}  {'longest':>7}  "
      f"{'4+/3/2':>10}  {'rare':>5}  {'PER':>6}")
print('-' * 60)
for pid in SWEEP_PIDS:
    best_lp, best_r = None, None
    for lp in LP_VALUES:
        r = sweep_results.get((lp, pid))
        if r is None: continue
        if best_r is None or r['z'] > best_r['z']:
            best_lp, best_r = lp, r
    if best_r is None:
        print(f"{pid:>5}  no results")
    else:
        ngrams = f"{best_r['n_4plus']}/{best_r['n_3']}/{best_r['n_2']}"
        print(f"{pid:>5}  {best_lp:>6.1f}  {_pad(z_str(best_r['z']), 7)}  "
              f"{best_r['longest']:>7}  {ngrams:>10}  "
              f"{best_r['rare']:>5}  {best_r['per']:>5.0%}")

# ── Summary 3: aggregate across all patients per LP ────────────────
print()
print("=" * 100)
print("Aggregate across patients (per LP):\n")
print(f"{'LP':>5}  {'tot 4+':>7}  {'tot 3':>6}  {'tot 2':>6}  {'tot rare':>9}  "
      f"{'mean z':>7}  {'max z':>7}  {'#z>+2':>6}  "
      f"{'mean PER':>9}  {'#PER<80%':>9}  {'mean bonus':>11}")
print('-' * 100)
for lp in LP_VALUES:
    rs = [sweep_results[(lp, pid)]
          for pid in SWEEP_PIDS if (lp, pid) in sweep_results]
    if not rs:
        print(f"{lp:>5.1f}  no results"); continue
    tot_4   = sum(r['n_4plus'] for r in rs)
    tot_3   = sum(r['n_3']     for r in rs)
    tot_2   = sum(r['n_2']     for r in rs)
    tot_rare= sum(r['rare']    for r in rs)
    mean_z  = float(np.mean([r['z']   for r in rs]))
    max_z   = float(np.max ([r['z']   for r in rs]))
    n_z2    = sum(1 for r in rs if r['z'] > 2)
    mean_per= float(np.mean([r['per'] for r in rs]))
    n_per   = sum(1 for r in rs if r['per'] < 0.80)
    mean_bon= float(np.mean([r['bonus'] for r in rs]))
    mean_drop = float(np.mean([r['dropped'] for r in rs]))

    print(f"{lp:>5.1f}  {tot_4:>7}  {tot_3:>6}  {tot_2:>6}  {tot_rare:>9}  "
          f"{_pad(z_str(mean_z), 7)}  {_pad(z_str(max_z), 7)}  {n_z2:>6}  "
          f"{_pad(per_str(mean_per), 9)}  {n_per:>9}  {mean_bon:>11.2f}  "
          f"{mean_drop:>9.0f}")

# side by side for lp cutoff
from IPython.utils.capture import capture_output
from IPython.display import HTML, display

PID                = 'P21'
BASE_SPEC          = {'hg_amp': True}
LP_KEY             = 'smoothing_hz'
LP_VALUES_COMPARE  = [8.0, 12.0]    # 2-4 values look readable side by side
LP_KEY = 'hg_lp_hz'   # not 'smoothing_hz'

if not hasattr(pipeline, 'patient_results') or pipeline.patient_results is None:
    pipeline.patient_results = {}

def render_for_lp(pid, lp):
    spec = {**BASE_SPEC, LP_KEY: lp}
    result, reason = run_for_patient(pid, feature_spec=spec)
    if result is None:
        return f"<p style='color:#a00'>SKIPPED — {reason}</p>", None
    pipeline.patient_results[pid] = result

    pr = result
    pred_sents, gold_sents = [], []
    for sid in np.unique(pr['true_sentence_ids']):
        pred_sents.append(list(pr['predictions'][pr['pred_sentence_ids'] == sid]))
        gold_sents.append(list(pr['true_labels'][pr['true_sentence_ids'] == sid]))
    all_gold = [ph for s in gold_sents for ph in s]
    cnt = Counter(all_gold); N = sum(cnt.values())
    gold_lp_dist = {k: np.log(v / N) for k, v in cnt.items()}

    n_2, n_3, n_4, longest, m_all = length_breakdown(pred_sents, gold_sents)
    n_rare = count_rare(m_all, gold_sents)
    m_z = collect_matches(pred_sents, gold_sents)
    obs   = surprise_score(m_z, gold_lp_dist)
    nulls = perm_null(pred_sents, gold_sents, gold_lp_dist)
    z     = float((obs - nulls.mean()) / (nulls.std() + 1e-9))

    summary = {'z': z, 'longest': longest, 'n_4plus': n_4, 'n_3': n_3, 'n_2': n_2,
               'rare': n_rare, 'per': pr['per'],
               'n_pred': pr['n_pred'], 'n_test': pr['n_test'], 'bonus': pr['bonus']}

    with capture_output() as cap:
        show_matched_sequences_with_times(
            pipeline, pid,
            max_per_line=20,
            collapse_repeats=False,
            # time_align_tol_s=0.1,
        )
    html_parts = []
    if cap.stdout:
        html_parts.append(
            f"<pre style='font-size:11px; margin:4px 0; white-space:pre-wrap;'>"
            f"{cap.stdout}</pre>")
    for output in cap.outputs:
        if hasattr(output, 'data') and 'text/html' in output.data:
            html_parts.append(output.data['text/html'])
    return '\n'.join(html_parts), summary

def _z_html(z):
    if z > 2:
        return f"<span style='color:#c00; font-weight:bold;'>{z:+.2f}</span>"
    return f"<span style='font-weight:bold;'>{z:+.2f}</span>"

def _header(lp, s):
    if s is None: return f"<h3>{LP_KEY}={lp}</h3><p>no result</p>"
    return (
        f"<h3 style='margin:4px 0;'>{LP_KEY}={lp} Hz</h3>"
        f"<div style='font-family:monospace; font-size:12px; padding:6px; "
        f"background:#f4f4f4; border-radius:4px; margin-bottom:8px;'>"
        f"z={_z_html(s['z'])} &nbsp; longest={s['longest']} &nbsp; "
        f"4+/3/2={s['n_4plus']}/{s['n_3']}/{s['n_2']} &nbsp; "
        f"rare={s['rare']} &nbsp; PER={s['per']:.1%}<br>"
        f"bonus={s['bonus']:.2f} &nbsp; n_pred={s['n_pred']}/{s['n_test']}"
        f"</div>"
    )

print(f"=== {PID}  across {LP_KEY} ∈ {LP_VALUES_COMPARE} ===")
columns = []
for lp in LP_VALUES_COMPARE:
    html_blob, summary = render_for_lp(PID, lp)
    columns.append(
        f"<div style='flex:1; min-width:0; overflow-x:auto;'>"
        f"{_header(lp, summary)}{html_blob}</div>"
    )

display(HTML(
    f"<div style='display:flex; gap:20px; align-items:flex-start;'>"
    f"{''.join(columns)}</div>"
))

# ### manner fitting

# MO sweep — wider temporal context
# ============================================================
import numpy as np

# Save originals so we can restore at the end
ORIG_MO, ORIG_LDA_MARGIN = MO, LDA_MARGIN

PATIENTS_SW = ["P21","P22","P23","P24","P25","P26","P27","P28","P29","P30"]
MO_VALUES = [5, 7, 9, 11, 13]

results_by_mo = {}

for mo_val in MO_VALUES:
    # Update globals (will be picked up by run_for_patient_sd_dec at call time)
    MO = mo_val
    LDA_MARGIN = mo_val * SS
    print(f"\n=== MO = {mo_val}  (n_lags = {2*mo_val+1}, LDA_MARGIN = {LDA_MARGIN}) ===", flush=True)

    zs, n4s, longests, pers = [], [], [], []
    for pt in PATIENTS_SW:
        try:
            out, err = run_for_patient_sd_dec(pt, decoder="viterbi")
            if out is None:
                print(f"  {pt}  SKIP: {err}", flush=True); continue
            preds_by = by_sentence(out['predictions'], out['pred_sentence_ids'])
            gold_by  = by_sentence(out['true_labels'], out['true_sentence_ids'])
            common = sorted(set(preds_by) | set(gold_by))
            pred_sents = [preds_by.get(s, []) for s in common]
            gold_sents = [gold_by.get(s, [])  for s in common]
            longest = max((longest_run_with_shift(p, g) for p, g in zip(pred_sents, gold_sents)), default=0)
            n4 = sum(count_ngrams_at_least(p, g, 4) for p, g in zip(pred_sents, gold_sents))
            z, *_ = perm_z(pred_sents, gold_sents, n_perm=500)
            zs.append(z); n4s.append(n4); longests.append(longest); pers.append(out['per'])
            print(f"  {pt}  z={z:+5.2f}  L={longest}  n4={n4}  per={100*out['per']:5.1f}%", flush=True)
        except Exception as e:
            import traceback
            print(f"  {pt}  EXCEPTION: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()

    results_by_mo[mo_val] = dict(z=np.mean(zs) if zs else np.nan,
                                  n4=np.mean(n4s) if n4s else np.nan,
                                  L=np.mean(longests) if longests else np.nan,
                                  per=np.mean(pers) if pers else np.nan)
    print(f"  MEAN  z̄={results_by_mo[mo_val]['z']:+5.2f}  n4̄={results_by_mo[mo_val]['n4']:.2f}  "
          f"L̄={results_by_mo[mo_val]['L']:.1f}  per̄={100*results_by_mo[mo_val]['per']:5.1f}%")

# Restore originals
MO, LDA_MARGIN = ORIG_MO, ORIG_LDA_MARGIN
print(f"\n>>> Restored MO={MO}, LDA_MARGIN={LDA_MARGIN}")

# Summary table
print("\n" + "="*60)
print(f"{'MO':<5} {'n_lags':>7} {'z̄':>8} {'n4̄':>7} {'L̄':>6} {'per̄':>8}")
print("="*60)
for mo_val in MO_VALUES:
    r = results_by_mo[mo_val]
    print(f"{mo_val:<5} {2*mo_val+1:>7} {r['z']:>+7.2f} {r['n4']:>7.2f} {r['L']:>6.1f} {100*r['per']:>7.1f}%")

# ============================================================
# F-test based per-patient electrode pruning
# ============================================================
import numpy as np
from sklearn.feature_selection import f_classif

def select_top_electrodes(X, y, n_lags, keep_k, agg='mean'):
    """Univariate F-test per stacked column, aggregated to per-electrode.
    
    X: (n_samples, n_electrodes * n_lags) stacked features. stackFeatures
       lays columns out as [lag0_ch0..chE-1, lag1_ch0..chE-1, ...].
    y: (n_samples,) phoneme labels.
    n_lags: 2*MO+1.
    keep_k: number of electrodes to retain.
    agg: 'mean' or 'max' across lags when aggregating to electrode score.
    
    Returns: (col_mask [n_electrodes*n_lags], electrode_scores [n_electrodes],
              kept_electrode_indices, F_per_column).
    """
    n_features = X.shape[1]
    n_electrodes = n_features // n_lags
    F, _ = f_classif(X, y)
    F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
    # Reshape: col index k → lag = k // n_electrodes, electrode = k % n_electrodes
    F_resh = F.reshape(n_lags, n_electrodes)
    if agg == 'max':
        electrode_scores = F_resh.max(axis=0)
    else:
        electrode_scores = F_resh.mean(axis=0)
    kept = np.argsort(-electrode_scores)[:keep_k]
    col_mask = np.zeros(n_features, dtype=bool)
    for l in range(n_lags):
        col_mask[l * n_electrodes + kept] = True
    return col_mask, electrode_scores, kept, F

def run_for_patient_sd_dec(pid, test_offset=TEST_OFFSET, feature_spec=None,
                           use_speech_gate=USE_SPEECH_GATE,
                           speech_thresh=SPEECH_THRESHOLD,
                           speech_frac_min=SPEECH_FRAC_MIN,
                           decoder="viterbi", use_bigram=False,
                           lambda_bigram=0.0, beam_width=1,
                           dur_temp=1.0, alpha_manner=0.0):
    feature_spec = feature_spec or DEFAULT_FEATURE_SPEC
    wd = pipeline.split_result['word_segments_dict'].get(pid)
    if wd is None: return None, "no word_segments_dict"
    try: mfa = load_mfa_alignments(pid)
    except Exception as e: return None, f"no MFA ({type(e).__name__})"
    if not mfa: return None, "empty MFA"
    if use_speech_gate and pid not in mu_sd_speech:
        return None, f"no speech-detector stats for {pid}"

    all_real = [i for i, s in enumerate(wd['sentence_list'])
                if isinstance(s, dict) and s.get('text')]
    test_sent_ids      = set(all_real[test_offset::6])
    train_sent_ids_all = sorted(set(all_real) - test_sent_ids)
    val_step     = max(2, int(round(1 / VAL_FRAC)))
    val_sent_ids = set(train_sent_ids_all[::val_step])
    fit_sent_ids = set(train_sent_ids_all) - val_sent_ids

    raw_path = os.path.join(DUTCH_30_PATH, 'raw', f'{pid}_sEEG.npy')
    raw_eeg = np.load(raw_path)
    if raw_eeg.ndim == 2 and raw_eeg.shape[0] < raw_eeg.shape[1]:
        raw_eeg = raw_eeg.T

    per_sentence_stk, per_sentence_mask = {}, {}
    for sent_idx in all_real:
        if sent_idx not in mfa or not mfa[sent_idx]: continue
        s = wd['sentence_list'][sent_idx]
        s0, s1 = s['stim_start_idx'], s['stim_end_idx']
        if s1 > raw_eeg.shape[0]: continue
        ext = extract_features_multiband(raw_eeg[s0:s1], **feature_spec)
        if ext.shape[0] < 2 * LDA_MARGIN + 1: continue
        per_sentence_stk[sent_idx] = stackFeatures(ext, modelOrder=MO, stepSize=SS)
        if use_speech_gate:
            sp = predict_speech_prob(raw_eeg[s0:s1], pid)
            T_stk = per_sentence_stk[sent_idx].shape[0]
            mask = (sp[:T_stk] > speech_thresh) if len(sp) >= T_stk else \
                   np.concatenate([sp > speech_thresh,
                                   np.zeros(T_stk - len(sp), dtype=bool)])
            per_sentence_mask[sent_idx] = mask
    if not per_sentence_stk: return None, "no usable sentences"

    GROUP_DELAY_FRAMES = 10
    def build_train_set(sent_ids):
        causal_mode = feature_spec.get('causal', False)
        shift = GROUP_DELAY_FRAMES if causal_mode else 0
        X, y = [], []
        for sent_idx in sent_ids:
            if sent_idx not in per_sentence_stk: continue
            stk = per_sentence_stk[sent_idx]; T_stk = stk.shape[0]
            for ph in mfa[sent_idx]:
                k_s = int(np.ceil ((ph['start_s']*EEG_SR - WIN_SAMP/2) / SHIFT_SAMP))
                k_e = int(np.floor((ph['end_s']  *EEG_SR - WIN_SAMP/2) / SHIFT_SAMP))
                if (k_e - k_s + 1) < MN_FRAMES or (k_e - k_s + 1) > MX_FRAMES: continue
                ks = max(0,         k_s - LDA_MARGIN + shift)
                ke = min(T_stk - 1, k_e - LDA_MARGIN + shift)
                if ke < ks: continue
                X.append(stk[ks:ke+1].mean(axis=0))
                y.append(ph['phone'])
        return np.array(X), np.array(y)

    # ── Step 1: fit on 85% for bonus tuning ──
    X_fit, y_fit = build_train_set(fit_sent_ids)
    if len(X_fit) < 50: return None, f"too few fit samples ({len(X_fit)})"
    fit_classes = set(y_fit)
    sc_fit  = StandardScaler().fit(X_fit)
    clf_fit = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    clf_fit.fit(sc_fit.transform(X_fit), y_fit)

    # ── Step 2: val log-probs + target gold-count ──
    val_logps, val_masks, val_target = [], [], 0
    for sent_idx in val_sent_ids:
        if sent_idx not in per_sentence_stk: continue
        stk = per_sentence_stk[sent_idx]
        logp = clf_fit.predict_log_proba(sc_fit.transform(stk))
        logp = smooth_cols(logp, SMOOTH_LOGP_W)
        val_logps.append(logp)
        if use_speech_gate:
            val_masks.append(per_sentence_mask[sent_idx])
        val_target += sum(1 for ph in mfa[sent_idx] if ph['phone'] in fit_classes)
    if not val_logps: return None, "no val sentences"
    val_target = int(val_target * TARGET_RATIO)

    # ── Step 3: tune bonus ──
    if SELF_LOOP_BONUS is None:
        bonus = (auto_tune_bonus_masked(val_logps, val_masks, val_target,
                                         MIN_PRED_FRAMES, speech_frac=speech_frac_min)
                 if use_speech_gate
                 else auto_tune_bonus(val_logps, val_target, MIN_PRED_FRAMES))
    else:
        bonus = float(SELF_LOOP_BONUS)

    # ── Step 4: refit main LDA on full train ──
    X_train, y_train = build_train_set(set(all_real) - test_sent_ids)
    train_classes = set(y_train)
    scaler = StandardScaler().fit(X_train)
    clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    clf.fit(scaler.transform(X_train), y_train)
    class_labels = list(clf.classes_)

    # ── ★ Step 4b: train manner factor classifier (if requested) ──
    manner_scaler, manner_clf, manner_classes = (None, None, None)
    if alpha_manner > 0:
        manner_scaler, manner_clf, manner_classes = train_factor_classifier(
            X_train, y_train, manner_of)

    # ── Step 4c: bigram LP (optional) ──
    bigram_lp = None
    if use_bigram and lambda_bigram > 0:
        bigram_lp = build_bigram_lp(mfa, set(all_real) - test_sent_ids, class_labels)

    # ── Step 4d: duration model (HSMM only) ──
    log_dur_pmf, D_max = (None, None)
    if decoder == "hsmm":
        log_dur_pmf, D_max = fit_phoneme_durations(
            mfa, set(all_real) - test_sent_ids, class_labels, dur_temp=dur_temp)

    # ── Step 5: decode test with chosen decoder + optional manner mixing ──
    predictions, pred_sentence_ids, pred_segments = [], [], []
    true_labels, true_sentence_ids, true_segments = [], [], []
    n_dropped_silence = 0
    for sent_idx in test_sent_ids:
        if sent_idx not in per_sentence_stk: continue
        stk = per_sentence_stk[sent_idx]; T_stk = stk.shape[0]
        mask = per_sentence_mask.get(sent_idx, np.ones(T_stk, dtype=bool))

        logp = clf.predict_log_proba(scaler.transform(stk))
        # ★ manner-factor mixing
        if manner_clf is not None and alpha_manner > 0:
            logp_manner_factor = manner_clf.predict_log_proba(manner_scaler.transform(stk))
            logp_manner_lifted = lift_factor_logp_to_phoneme(
                logp_manner_factor, manner_classes, class_labels, manner_of)
            logp = (1 - alpha_manner) * logp + alpha_manner * logp_manner_lifted
        logp = smooth_cols(logp, SMOOTH_LOGP_W)

        path = decode_with_choice(logp, bonus, decoder=decoder,
                                   use_bigram=use_bigram,
                                   lambda_bigram=lambda_bigram,
                                   beam_width=beam_width,
                                   bigram_lp=bigram_lp,
                                   log_dur_pmf=log_dur_pmf, D_max=D_max)
        i = 0
        while i < T_stk:
            ci = path[i]; j = i + 1
            while j < T_stk and path[j] == ci: j += 1
            if (j - i) >= MIN_PRED_FRAMES:
                if mask[i:j].mean() >= speech_frac_min:
                    predictions.append(class_labels[ci])
                    pred_sentence_ids.append(sent_idx)
                    pred_segments.append((stk_frame_to_time_s(i),
                                          stk_frame_to_time_s(j - 1)))
                else:
                    n_dropped_silence += 1
            i = j
        for ph in mfa[sent_idx]:
            if ph['phone'] not in train_classes: continue
            true_labels.append(ph['phone'])
            true_sentence_ids.append(sent_idx)
            true_segments.append((ph['start_s'], ph['end_s']))

    if not true_labels: return None, "no test gold labels"
    true_arr = np.array(true_labels); pred_arr = np.array(predictions)
    ed = edit_distance(list(true_arr), list(pred_arr))
    per = ed / max(len(true_arr), 1)
    return {
        'true_labels': true_arr, 'predictions': pred_arr,
        'true_sentence_ids': np.array(true_sentence_ids),
        'pred_sentence_ids': np.array(pred_sentence_ids),
        'true_segments': true_segments, 'pred_segments': pred_segments,
        'edit_distance': ed, 'per': per,
        'n_test': len(true_arr), 'n_pred': len(pred_arr),
        'bonus': bonus, 'val_target': val_target,
        'n_dropped_silence': n_dropped_silence,
    }, None

def run_for_patient_sd_dec(pid, test_offset=TEST_OFFSET, feature_spec=None,
                           use_speech_gate=USE_SPEECH_GATE,
                           speech_thresh=SPEECH_THRESHOLD,
                           speech_frac_min=SPEECH_FRAC_MIN,
                           decoder="viterbi", use_bigram=False,
                           lambda_bigram=0.0, beam_width=1,
                           dur_temp=1.0, alpha_manner=0.0,
                           keep_top_electrodes=None, keep_top_fraction=None):
    feature_spec = feature_spec or DEFAULT_FEATURE_SPEC
    wd = pipeline.split_result['word_segments_dict'].get(pid)
    if wd is None: return None, "no word_segments_dict"
    try: mfa = load_mfa_alignments(pid)
    except Exception as e: return None, f"no MFA ({type(e).__name__})"
    if not mfa: return None, "empty MFA"
    if use_speech_gate and pid not in mu_sd_speech:
        return None, f"no speech-detector stats for {pid}"

    all_real = [i for i, s in enumerate(wd['sentence_list'])
                if isinstance(s, dict) and s.get('text')]
    test_sent_ids      = set(all_real[test_offset::6])
    train_sent_ids_all = sorted(set(all_real) - test_sent_ids)
    val_step     = max(2, int(round(1 / VAL_FRAC)))
    val_sent_ids = set(train_sent_ids_all[::val_step])
    fit_sent_ids = set(train_sent_ids_all) - val_sent_ids

    raw_path = os.path.join(DUTCH_30_PATH, 'raw', f'{pid}_sEEG.npy')
    raw_eeg = np.load(raw_path)
    if raw_eeg.ndim == 2 and raw_eeg.shape[0] < raw_eeg.shape[1]:
        raw_eeg = raw_eeg.T

    per_sentence_stk, per_sentence_mask = {}, {}
    for sent_idx in all_real:
        if sent_idx not in mfa or not mfa[sent_idx]: continue
        s = wd['sentence_list'][sent_idx]
        s0, s1 = s['stim_start_idx'], s['stim_end_idx']
        if s1 > raw_eeg.shape[0]: continue
        ext = extract_features_multiband(raw_eeg[s0:s1], **feature_spec)
        if ext.shape[0] < 2 * LDA_MARGIN + 1: continue
        per_sentence_stk[sent_idx] = stackFeatures(ext, modelOrder=MO, stepSize=SS)
        if use_speech_gate:
            sp = predict_speech_prob(raw_eeg[s0:s1], pid)
            T_stk = per_sentence_stk[sent_idx].shape[0]
            mask = (sp[:T_stk] > speech_thresh) if len(sp) >= T_stk else \
                   np.concatenate([sp > speech_thresh,
                                   np.zeros(T_stk - len(sp), dtype=bool)])
            per_sentence_mask[sent_idx] = mask
    if not per_sentence_stk: return None, "no usable sentences"

    GROUP_DELAY_FRAMES = 10
    def build_train_set(sent_ids):
        causal_mode = feature_spec.get('causal', False)
        shift = GROUP_DELAY_FRAMES if causal_mode else 0
        X, y = [], []
        for sent_idx in sent_ids:
            if sent_idx not in per_sentence_stk: continue
            stk = per_sentence_stk[sent_idx]; T_stk = stk.shape[0]
            for ph in mfa[sent_idx]:
                k_s = int(np.ceil ((ph['start_s']*EEG_SR - WIN_SAMP/2) / SHIFT_SAMP))
                k_e = int(np.floor((ph['end_s']  *EEG_SR - WIN_SAMP/2) / SHIFT_SAMP))
                if (k_e - k_s + 1) < MN_FRAMES or (k_e - k_s + 1) > MX_FRAMES: continue
                ks = max(0,         k_s - LDA_MARGIN + shift)
                ke = min(T_stk - 1, k_e - LDA_MARGIN + shift)
                if ke < ks: continue
                X.append(stk[ks:ke+1].mean(axis=0))
                y.append(ph['phone'])
        return np.array(X), np.array(y)

    X_fit, y_fit = build_train_set(fit_sent_ids)
    if len(X_fit) < 50: return None, f"too few fit samples ({len(X_fit)})"
    fit_classes = set(y_fit)

    # ── Electrode pruning ──
    n_lags = 2 * MO + 1
    n_total_electrodes = X_fit.shape[1] // n_lags
    K_eff = keep_top_electrodes
    if keep_top_fraction is not None:
        K_eff = max(1, int(round(keep_top_fraction * n_total_electrodes)))
    if K_eff is not None:
        col_mask, _, kept_idx, _ = select_top_electrodes(
            X_fit, y_fit, n_lags, K_eff)
    else:
        col_mask = np.ones(X_fit.shape[1], dtype=bool)
        kept_idx = np.arange(n_total_electrodes)
    def slice_stk(stk): return stk[:, col_mask]

    X_fit_m = X_fit[:, col_mask]
    sc_fit  = StandardScaler().fit(X_fit_m)
    clf_fit = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    clf_fit.fit(sc_fit.transform(X_fit_m), y_fit)

    val_logps, val_masks, val_target = [], [], 0
    for sent_idx in val_sent_ids:
        if sent_idx not in per_sentence_stk: continue
        stk_m = slice_stk(per_sentence_stk[sent_idx])
        logp = clf_fit.predict_log_proba(sc_fit.transform(stk_m))
        logp = smooth_cols(logp, SMOOTH_LOGP_W)
        val_logps.append(logp)
        if use_speech_gate:
            val_masks.append(per_sentence_mask[sent_idx])
        val_target += sum(1 for ph in mfa[sent_idx] if ph['phone'] in fit_classes)
    if not val_logps: return None, "no val sentences"
    val_target = int(val_target * TARGET_RATIO)

    if SELF_LOOP_BONUS is None:
        bonus = (auto_tune_bonus_masked(val_logps, val_masks, val_target,
                                         MIN_PRED_FRAMES, speech_frac=speech_frac_min)
                 if use_speech_gate
                 else auto_tune_bonus(val_logps, val_target, MIN_PRED_FRAMES))
    else:
        bonus = float(SELF_LOOP_BONUS)

    # ── ★ Val-set decode → computes BOTH val_per (PER metric) AND val_z (perm z) ──
    _fit_class_labels = list(clf_fit.classes_)
    _val_preds_flat, _val_truths_flat = [], []
    _val_preds_by, _val_gold_by = {}, {}
    for _sid in val_sent_ids:
        if _sid not in per_sentence_stk: continue
        _stk_m = slice_stk(per_sentence_stk[_sid]); _T = _stk_m.shape[0]
        _mask_v = per_sentence_mask.get(_sid, np.ones(_T, dtype=bool))
        _logp = clf_fit.predict_log_proba(sc_fit.transform(_stk_m))
        _logp = smooth_cols(_logp, SMOOTH_LOGP_W)
        _path = viterbi_decode(_logp, bonus)
        _val_preds_by[_sid] = []
        _i = 0
        while _i < _T:
            _ci = _path[_i]; _j = _i + 1
            while _j < _T and _path[_j] == _ci: _j += 1
            if (_j - _i) >= MIN_PRED_FRAMES and _mask_v[_i:_j].mean() >= speech_frac_min:
                _lbl = _fit_class_labels[_ci]
                _val_preds_flat.append(_lbl)
                _val_preds_by[_sid].append(_lbl)
            _i = _j
        _val_gold_by[_sid] = []
        for _ph in mfa[_sid]:
            if _ph['phone'] in fit_classes:
                _val_truths_flat.append(_ph['phone'])
                _val_gold_by[_sid].append(_ph['phone'])
    val_per = (edit_distance(_val_truths_flat, _val_preds_flat) / max(len(_val_truths_flat), 1)
               if _val_truths_flat else float('nan'))
    _common_v = sorted(set(_val_preds_by) | set(_val_gold_by))
    _val_pred_sents = [_val_preds_by.get(s, []) for s in _common_v]
    _val_gold_sents = [_val_gold_by.get(s, [])  for s in _common_v]
    if _val_gold_sents and any(_val_gold_sents):
        val_z, *_ = perm_z(_val_pred_sents, _val_gold_sents, n_perm=200)
    else:
        val_z = float('nan')

    # ── Refit main LDA on full train (fit+val) ──
    X_train, y_train = build_train_set(set(all_real) - test_sent_ids)
    train_classes = set(y_train)
    X_train_m = X_train[:, col_mask]
    scaler = StandardScaler().fit(X_train_m)
    clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    clf.fit(scaler.transform(X_train_m), y_train)
    class_labels = list(clf.classes_)

    manner_scaler, manner_clf, manner_classes = (None, None, None)
    if alpha_manner > 0:
        manner_scaler, manner_clf, manner_classes = train_factor_classifier(
            X_train_m, y_train, manner_of)

    bigram_lp = None
    if use_bigram and lambda_bigram > 0:
        bigram_lp = build_bigram_lp(mfa, set(all_real) - test_sent_ids, class_labels)

    log_dur_pmf, D_max = (None, None)
    if decoder == "hsmm":
        log_dur_pmf, D_max = fit_phoneme_durations(
            mfa, set(all_real) - test_sent_ids, class_labels, dur_temp=dur_temp)

    predictions, pred_sentence_ids, pred_segments = [], [], []
    true_labels, true_sentence_ids, true_segments = [], [], []
    n_dropped_silence = 0
    for sent_idx in test_sent_ids:
        if sent_idx not in per_sentence_stk: continue
        stk = per_sentence_stk[sent_idx]; T_stk = stk.shape[0]
        mask = per_sentence_mask.get(sent_idx, np.ones(T_stk, dtype=bool))
        stk_m = slice_stk(stk)
        logp = clf.predict_log_proba(scaler.transform(stk_m))
        if manner_clf is not None and alpha_manner > 0:
            logp_manner_factor = manner_clf.predict_log_proba(manner_scaler.transform(stk_m))
            logp_manner_lifted = lift_factor_logp_to_phoneme(
                logp_manner_factor, manner_classes, class_labels, manner_of)
            logp = (1 - alpha_manner) * logp + alpha_manner * logp_manner_lifted
        logp = smooth_cols(logp, SMOOTH_LOGP_W)
        path = decode_with_choice(logp, bonus, decoder=decoder,
                                   use_bigram=use_bigram,
                                   lambda_bigram=lambda_bigram,
                                   beam_width=beam_width,
                                   bigram_lp=bigram_lp,
                                   log_dur_pmf=log_dur_pmf, D_max=D_max)
        i = 0
        while i < T_stk:
            ci = path[i]; j = i + 1
            while j < T_stk and path[j] == ci: j += 1
            if (j - i) >= MIN_PRED_FRAMES:
                if mask[i:j].mean() >= speech_frac_min:
                    predictions.append(class_labels[ci])
                    pred_sentence_ids.append(sent_idx)
                    pred_segments.append((stk_frame_to_time_s(i),
                                          stk_frame_to_time_s(j - 1)))
                else:
                    n_dropped_silence += 1
            i = j
        for ph in mfa[sent_idx]:
            if ph['phone'] not in train_classes: continue
            true_labels.append(ph['phone'])
            true_sentence_ids.append(sent_idx)
            true_segments.append((ph['start_s'], ph['end_s']))

    if not true_labels: return None, "no test gold labels"
    true_arr = np.array(true_labels); pred_arr = np.array(predictions)
    ed = edit_distance(list(true_arr), list(pred_arr))
    per = ed / max(len(true_arr), 1)
    return {
        'true_labels': true_arr, 'predictions': pred_arr,
        'true_sentence_ids': np.array(true_sentence_ids),
        'pred_sentence_ids': np.array(pred_sentence_ids),
        'true_segments': true_segments, 'pred_segments': pred_segments,
        'edit_distance': ed, 'per': per,
        'n_test': len(true_arr), 'n_pred': len(pred_arr),
        'bonus': bonus, 'val_target': val_target,
        'n_dropped_silence': n_dropped_silence,
        'n_electrodes_kept': len(kept_idx),
        'val_per': val_per,
        'val_z':   val_z,
    }, None

PATIENTS_SW = ["P21","P22","P23","P24","P25","P26","P27","P28","P29","P30"]
FRACTIONS = [None, 0.95, 0.90, 0.85, 0.75, 0.60]
SWITCH_MARGIN = 0.015   # require val_per improvement of 1.5% absolute to switch from baseline

# ── PASS 1: collect (val_per, test metrics) at every (patient, fraction) ──
all_runs = {pt: {} for pt in PATIENTS_SW}
for frac in FRACTIONS:
    print(f"\n=== fraction = {frac} ===", flush=True)
    for pt in PATIENTS_SW:
        try:
            out, err = run_for_patient_sd_dec(pt, decoder="viterbi",
                                               keep_top_fraction=frac)
            if out is None:
                print(f"  {pt}  SKIP: {err}", flush=True); continue
            preds_by = by_sentence(out['predictions'], out['pred_sentence_ids'])
            gold_by  = by_sentence(out['true_labels'], out['true_sentence_ids'])
            common = sorted(set(preds_by) | set(gold_by))
            pred_sents = [preds_by.get(s, []) for s in common]
            gold_sents = [gold_by.get(s, [])  for s in common]
            longest = max((longest_run_with_shift(p, g) for p, g in zip(pred_sents, gold_sents)), default=0)
            n4 = sum(count_ngrams_at_least(p, g, 4) for p, g in zip(pred_sents, gold_sents))
            z, *_ = perm_z(pred_sents, gold_sents, n_perm=500)
            val_per = out.get('val_per', float('nan'))
            all_runs[pt][frac] = dict(val_per=val_per, test_z=z,
                                       test_L=longest, test_n4=n4,
                                       test_per=out['per'])
            print(f"  {pt}  val_per={100*val_per:5.1f}%  test_z={z:+5.2f}  L={longest}  n4={n4}", flush=True)
        except Exception as e:
            import traceback
            print(f"  {pt}  EXCEPTION: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()

# ── PASS 2: conservative per-patient selection ──
# Default to baseline (None). Switch to a pruned fraction only if its val_per
# beats baseline's by at least SWITCH_MARGIN (absolute, in PER units).
print("\n" + "="*70)
print(f"{'patient':<8} {'best_frac':<10} {'val_per':>8} {'baseline_val':>13} {'test_z':>8} {'test_L':>6} {'test_n4':>7}")
print("="*70)
zs, n4s, longests = [], [], []
for pt in PATIENTS_SW:
    if not all_runs[pt]:
        print(f"{pt:<8}  no runs", flush=True); continue
    base = all_runs[pt].get(None)
    if base is None or np.isnan(base['val_per']):
        print(f"{pt:<8}  no baseline run", flush=True); continue
    # candidates: pruned fractions with val_per at least SWITCH_MARGIN below baseline
    candidates = {k: v for k, v in all_runs[pt].items()
                  if k is not None and not np.isnan(v['val_per'])
                  and (base['val_per'] - v['val_per']) >= SWITCH_MARGIN}
    if candidates:
        best_frac, best = min(candidates.items(), key=lambda kv: kv[1]['val_per'])
    else:
        best_frac, best = None, base
    zs.append(best['test_z']); n4s.append(best['test_n4']); longests.append(best['test_L'])
    print(f"{pt:<8} {str(best_frac):<10} {100*best['val_per']:>7.1f}%  {100*base['val_per']:>12.1f}%  {best['test_z']:>+6.2f}  {best['test_L']:>6}  {best['test_n4']:>7}", flush=True)
print(f"\n  MEAN  z̄={np.mean(zs):+5.2f}  n4̄={np.mean(n4s):.2f}  L̄={np.mean(longests):.1f}")

# ============================================================
# MO=11 × fraction sweep
# ============================================================
import numpy as np

ORIG_MO, ORIG_LDA_MARGIN = MO, LDA_MARGIN
MO = 11
LDA_MARGIN = 11 * SS

PATIENTS_SW = ["P21","P22","P23","P24","P25","P26","P27","P28","P29","P30"]
FRACTIONS = [None, 0.95, 0.90, 0.85, 0.75, 0.60]

all_runs_mo11 = {pt: {} for pt in PATIENTS_SW}
for frac in FRACTIONS:
    print(f"\n=== MO=11  fraction = {frac} ===", flush=True)
    for pt in PATIENTS_SW:
        try:
            out, err = run_for_patient_sd_dec(pt, decoder="viterbi",
                                               keep_top_fraction=frac)
            if out is None:
                print(f"  {pt}  SKIP: {err}", flush=True); continue
            preds_by = by_sentence(out['predictions'], out['pred_sentence_ids'])
            gold_by  = by_sentence(out['true_labels'], out['true_sentence_ids'])
            common = sorted(set(preds_by) | set(gold_by))
            pred_sents = [preds_by.get(s, []) for s in common]
            gold_sents = [gold_by.get(s, [])  for s in common]
            longest = max((longest_run_with_shift(p, g) for p, g in zip(pred_sents, gold_sents)), default=0)
            n4 = sum(count_ngrams_at_least(p, g, 4) for p, g in zip(pred_sents, gold_sents))
            z, *_ = perm_z(pred_sents, gold_sents, n_perm=500)
            val_per = out.get('val_per', float('nan'))
            val_z   = out.get('val_z',   float('nan'))
            all_runs_mo11[pt][frac] = dict(val_per=val_per, val_z=val_z,
                                            test_z=z, test_L=longest, test_n4=n4,
                                            test_per=out['per'])
            print(f"  {pt}  val_per={100*val_per:5.1f}%  val_z={val_z:+5.2f}  "
                  f"test_z={z:+5.2f}  L={longest}  n4={n4}", flush=True)
        except Exception as e:
            import traceback
            print(f"  {pt}  EXCEPTION: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()

        zs  = [all_runs_mo11[p][frac]['test_z']  for p in PATIENTS_SW if frac in all_runs_mo11[p]]
        n4s = [all_runs_mo11[p][frac]['test_n4'] for p in PATIENTS_SW if frac in all_runs_mo11[p]]
        Ls  = [all_runs_mo11[p][frac]['test_L']  for p in PATIENTS_SW if frac in all_runs_mo11[p]]
        if zs:
            print(f"  MEAN  z̄={np.mean(zs):+5.2f}  n4̄={np.mean(n4s):.2f}  L̄={np.mean(Ls):.1f}", flush=True)

# Restore MO
MO, LDA_MARGIN = ORIG_MO, ORIG_LDA_MARGIN
print(f"\n>>> Restored MO={MO}, LDA_MARGIN={LDA_MARGIN}")

# Compact summary table
print("\n" + "="*72)
print(f"MO=11 × fraction grid")
print("="*72)
hdr = f"{'patient':<8} | " + " | ".join(f"frac={str(f):<6}" for f in FRACTIONS)
print(hdr); print("-"*len(hdr))
for pt in PATIENTS_SW:
    cells = []
    for f in FRACTIONS:
        r = all_runs_mo11[pt].get(f)
        if r is None:
            cells.append("       --       ")
        else:
            cells.append(f"z={r['test_z']:+5.2f} L={r['test_L']} n4={r['test_n4']}")
    print(f"{pt:<8} | " + " | ".join(cells))
print("-"*len(hdr))
means = []
for f in FRACTIONS:
    vals = [all_runs_mo11[p][f] for p in PATIENTS_SW if f in all_runs_mo11[p]]
    if not vals: means.append("       --       "); continue
    means.append(f"z̄={np.mean([v['test_z'] for v in vals]):+5.2f} n4̄={np.mean([v['test_n4'] for v in vals]):.2f}")
print(f"{'MEAN':<8} | " + " | ".join(means))

# ============================================================
# Full LDA-frame vs MFA-CRF comparison
# Includes the CRF training run if needed
# ============================================================
import numpy as np
from collections import Counter, defaultdict

# ── Step 1: Make sure CRF results are populated ────────────────
def ensure_crf_results():
    """Run CRF training if pipeline.patient_results is empty or LDA-shaped."""
    needs_crf = True
    if hasattr(pipeline, 'patient_results') and pipeline.patient_results:
        sample = next(iter(pipeline.patient_results.values()))
        # CRF dicts have 'accuracy' but no 'bonus'/'val_target'
        if 'bonus' not in sample and 'predictions' in sample:
            needs_crf = False
            print(f"  pipeline.patient_results already has CRF-shaped data "
                  f"({len(pipeline.patient_results)} patients) — keeping it", flush=True)
    if needs_crf:
        print(f"  pipeline.patient_results missing or wrong shape. Running CRF...", flush=True)
        from run_pipeline import run_path_b, _run_crf_experiment, DEFAULT_RUN_CONFIG
        run_config = dict(DEFAULT_RUN_CONFIG)
        run_config['stacking_order'] = 7        # what your saved CRF models use
        run_config['stacking_step_size'] = 1
        run_config['use_viterbi'] = True
        run_path_b(pipeline, run_config)
        crf_results = _run_crf_experiment(pipeline, run_config)
        pipeline.patient_results = {pid: {
            'predictions': r['predictions'],
            'true_labels': r['true_labels'],
            'accuracy':    r['accuracy'],
            'n_test':      r['n_test'],
            'n_train':     r['n_train'],
        } for pid, r in crf_results.items()}
        print(f"  CRF done: {len(pipeline.patient_results)} patients", flush=True)

ensure_crf_results()

# ── Step 2: Alignment helpers ──────────────────────────────────
def crf_pairs(crf_result):
    """CRF predictions are position-aligned 1-to-1 with gold. Trivial."""
    return list(zip(crf_result['true_labels'], crf_result['predictions']))

def lda_pairs(lda_result):
    """LDA predictions are time-segment based. Align each gold phoneme to the
    predicted phoneme whose time-interval overlaps it most."""
    pairs = []
    pred_labels   = list(lda_result['predictions'])
    pred_segments = lda_result['pred_segments']
    gold_labels   = list(lda_result['true_labels'])
    gold_segments = lda_result['true_segments']
    for g_label, (g_s, g_e) in zip(gold_labels, gold_segments):
        best_pred, best_overlap = None, 0.0
        for p_label, (p_s, p_e) in zip(pred_labels, pred_segments):
            overlap = max(0.0, min(g_e, p_e) - max(g_s, p_s))
            if overlap > best_overlap:
                best_overlap = overlap; best_pred = p_label
        pairs.append((g_label, best_pred))
    return pairs

def per_phoneme_stats(pairs, min_count=3):
    gold_count = Counter(g for g, _ in pairs)
    pred_count = Counter(p for _, p in pairs if p is not None)
    correct    = Counter(g for g, p in pairs if g == p)
    confusions = defaultdict(Counter)
    for g, p in pairs:
        if p is not None and g != p:
            confusions[g][p] += 1
    rows = []
    for ph, n_gold in gold_count.most_common():
        if n_gold < min_count: continue
        n_pred = pred_count.get(ph, 0)
        n_corr = correct.get(ph, 0)
        recall    = n_corr / max(n_gold, 1)
        precision = n_corr / max(n_pred, 1) if n_pred else 0.0
        top_conf  = confusions[ph].most_common(1)[0] if confusions[ph] else (None, 0)
        rows.append((ph, n_gold, n_pred, n_corr, recall, precision, top_conf))
    return rows

# ── Step 3: Run comparison ─────────────────────────────────────
def compare_one(pid):
    crf_out = pipeline.patient_results.get(pid)
    if crf_out is None or 'predictions' not in crf_out:
        return None
    lda_out, err = run_for_patient_sd(pid)
    if lda_out is None:
        print(f"  {pid}: LDA failed: {err}"); return None

    crf_p = crf_pairs(crf_out)
    lda_p = lda_pairs(lda_out)
    crf_rows = per_phoneme_stats(crf_p)
    lda_rows = per_phoneme_stats(lda_p)
    crf_idx = {r[0]: r for r in crf_rows}
    lda_idx = {r[0]: r for r in lda_rows}
    all_ph  = sorted(set(crf_idx) | set(lda_idx),
                     key=lambda p: -(crf_idx.get(p, [0,0])[1] + lda_idx.get(p, [0,0])[1]))

    print(f"\n=== {pid} ===  (CRF: {len(crf_p)} gold, LDA: {len(lda_p)} gold)", flush=True)
    print(f"{'phoneme':<8} {'n':>4} | {'LDA_rec':>7} {'LDA_pre':>7} {'LDA_top':<12} | "
          f"{'CRF_rec':>7} {'CRF_pre':>7} {'CRF_top':<12}")
    print("-"*90)
    for ph in all_ph:
        lr = lda_idx.get(ph, (ph, 0, 0, 0, 0.0, 0.0, (None, 0)))
        cr = crf_idx.get(ph, (ph, 0, 0, 0, 0.0, 0.0, (None, 0)))
        n = max(lr[1], cr[1])
        lda_c = f"{lr[6][0] or '-'}({lr[6][1]})"
        crf_c = f"{cr[6][0] or '-'}({cr[6][1]})"
        print(f"{ph:<8} {n:>4} | "
              f"{lr[4]*100:>6.0f}% {lr[5]*100:>6.0f}% {lda_c:<12} | "
              f"{cr[4]*100:>6.0f}% {cr[5]*100:>6.0f}% {crf_c:<12}", flush=True)

    def macro(rows):
        return (np.mean([r[4] for r in rows]) if rows else 0,
                np.mean([r[5] for r in rows]) if rows else 0)
    lda_mr, lda_mp = macro(lda_rows)
    crf_mr, crf_mp = macro(crf_rows)
    print(f"  LDA-frame  macro-rec={100*lda_mr:5.1f}%  macro-pre={100*lda_mp:5.1f}%")
    print(f"  MFA-CRF    macro-rec={100*crf_mr:5.1f}%  macro-pre={100*crf_mp:5.1f}%", flush=True)
    return dict(pid=pid, lda_mr=lda_mr, lda_mp=lda_mp, crf_mr=crf_mr, crf_mp=crf_mp)

# Patients with cached CRF .pkl models
for pid in ["P22", "P23", "P26", "P29"]:
    compare_one(pid)

# ============================================================
# Word-onset phoneme classification: LDA vs RF vs LR
# Also compares vs all-phoneme classification on the same data.
# ============================================================
import numpy as np
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

def build_phoneme_set(per_sentence_stk, mfa, sent_ids, onset_only=False):
    """Standard phoneme-averaged feature extraction. If onset_only=True,
    keeps only the first phoneme of each word."""
    Xs, ys = [], []
    for sid in sent_ids:
        if sid not in per_sentence_stk: continue
        stk = per_sentence_stk[sid]; T_stk = stk.shape[0]
        prev_word = None
        for ph in mfa[sid]:
            cur_word = ph.get('word')
            is_onset = (cur_word != prev_word)
            k_s = int(np.ceil ((ph['start_s']*EEG_SR - WIN_SAMP/2) / SHIFT_SAMP))
            k_e = int(np.floor((ph['end_s']  *EEG_SR - WIN_SAMP/2) / SHIFT_SAMP))
            if (k_e - k_s + 1) >= MN_FRAMES and (k_e - k_s + 1) <= MX_FRAMES:
                ks = max(0, k_s - LDA_MARGIN); ke = min(T_stk-1, k_e - LDA_MARGIN)
                if ke >= ks and (not onset_only or is_onset):
                    Xs.append(stk[ks:ke+1].mean(axis=0))
                    ys.append(ph['phone'])
            prev_word = cur_word
    return np.array(Xs), np.array(ys)


from sklearn.decomposition import PCA

def evaluate_classifiers(X, y, top_k_values=(1, 3, 5), n_splits=5, n_pca=100):
    classifiers = {
        'LDA': lambda: LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'),
        'LR':  lambda: LogisticRegression(max_iter=1000, C=1.0,
                                            class_weight='balanced', n_jobs=-1),
        'RF':  lambda: RandomForestClassifier(n_estimators=200, n_jobs=-1,
                                                random_state=0,
                                                class_weight='balanced'),
    }
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    results = {name: {k: [] for k in top_k_values} for name in classifiers}
    for tr, te in kf.split(X):
        scaler = StandardScaler().fit(X[tr])
        Xtr_s = scaler.transform(X[tr]).astype(np.float32)
        Xte_s = scaler.transform(X[te]).astype(np.float32)
        # ★ PCA reduction (fit on train only)
        n_comp = min(n_pca, Xtr_s.shape[1], Xtr_s.shape[0] - 1)
        pca = PCA(n_components=n_comp, svd_solver='randomized', random_state=0)
        Xtr_p = pca.fit_transform(Xtr_s)
        Xte_p = pca.transform(Xte_s)
        for name, mk in classifiers.items():
            clf = mk()
            clf.fit(Xtr_p, y[tr])
            proba = clf.predict_proba(Xte_p)
            classes = clf.classes_
            y_te = y[te]
            ranking = np.argsort(-proba, axis=1)
            for k in top_k_values:
                correct = 0
                for i, true_lbl in enumerate(y_te):
                    if true_lbl in classes[ranking[i, :k]]:
                        correct += 1
                results[name][k].append(correct / len(y_te))
    return {name: {k: np.mean(v) for k, v in d.items()} for name, d in results.items()}


def compare_onset_vs_full(pid):
    """Per patient: compare word-onset-only classification vs all-phoneme,
    across LDA / LR / RF."""
    wd = pipeline.split_result['word_segments_dict'].get(pid)
    if wd is None: return None, "no word_segments_dict"
    mfa = load_mfa_alignments(pid)
    if not mfa: return None, "no MFA"

    all_real = [i for i, s in enumerate(wd['sentence_list'])
                if isinstance(s, dict) and s.get('text')]
    test_ids = set(all_real[TEST_OFFSET::6])
    train_ids = sorted(set(all_real) - test_ids)

    raw = np.load(os.path.join(DUTCH_30_PATH, 'raw', f'{pid}_sEEG.npy'))
    if raw.ndim == 2 and raw.shape[0] < raw.shape[1]: raw = raw.T
    per_sent_stk = {}
    for sid in all_real:
        if sid not in mfa or not mfa[sid]: continue
        s = wd['sentence_list'][sid]
        s0, s1 = s['stim_start_idx'], s['stim_end_idx']
        if s1 > raw.shape[0]: continue
        ext = extract_features_multiband(raw[s0:s1], **DEFAULT_FEATURE_SPEC)
        if ext.shape[0] < 2 * LDA_MARGIN + 1: continue
        per_sent_stk[sid] = stackFeatures(ext, modelOrder=MO, stepSize=SS)

    X_full, y_full = build_phoneme_set(per_sent_stk, mfa, train_ids, onset_only=False)
    X_ons,  y_ons  = build_phoneme_set(per_sent_stk, mfa, train_ids, onset_only=True)

    print(f"  All phonemes: {len(X_full)} samples, {len(set(y_full))} classes  "
          f"(chance ≈ {100/len(set(y_full)):.1f}%)")
    print(f"  Word onsets : {len(X_ons)}  samples, {len(set(y_ons))} classes  "
          f"(chance ≈ {100/len(set(y_ons)):.1f}%)")

    print(f"\n  ALL_PHONEMES task:")
    full_res = evaluate_classifiers(X_full, y_full)
    for name, d in full_res.items():
        print(f"    {name:<4} top-1={100*d[1]:5.1f}%  top-3={100*d[3]:5.1f}%  top-5={100*d[5]:5.1f}%")

    print(f"  WORD_ONSET task:")
    ons_res = evaluate_classifiers(X_ons, y_ons)
    for name, d in ons_res.items():
        print(f"    {name:<4} top-1={100*d[1]:5.1f}%  top-3={100*d[3]:5.1f}%  top-5={100*d[5]:5.1f}%")
    return dict(full=full_res, onset=ons_res), None


for pid in ["P22", "P23", "P26", "P29"]:
    print(f"\n=== {pid} ===", flush=True)
    res, err = compare_onset_vs_full(pid)
    if res is None: print(f"  SKIP: {err}", flush=True)

import numpy as np
from collections import Counter, defaultdict

def train_phoneme_lm_from_pipeline(pipeline, all_pids, test_offset=TEST_OFFSET,
                                     order=3, alpha=1.0):
    """Train phoneme N-gram LM with Laplace smoothing on MFA training data."""
    ngram_counts = defaultdict(Counter)
    context_counts = Counter()
    vocab = set()
    n_sentences = 0; n_tokens = 0

    for pid in all_pids:
        wd = pipeline.split_result['word_segments_dict'].get(pid)
        if wd is None: continue
        try: mfa = load_mfa_alignments(pid)
        except Exception: continue
        if not mfa: continue
        all_real = [i for i, s in enumerate(wd['sentence_list'])
                    if isinstance(s, dict) and s.get('text')]
        test_sent_ids = set(all_real[test_offset::6])
        train_ids = sorted(set(all_real) - test_sent_ids)
        for sid in train_ids:
            if sid not in mfa: continue
            phones = [ph['phone'] for ph in mfa[sid]]
            if not phones: continue
            seq = ['<s>'] * (order - 1) + phones + ['</s>']
            vocab.update(phones); n_sentences += 1
            for i in range(order - 1, len(seq)):
                ctx = tuple(seq[i-(order-1):i]); tok = seq[i]
                ngram_counts[ctx][tok] += 1
                context_counts[ctx] += 1; n_tokens += 1

    V = len(vocab) + 2
    cache = {}
    def log_prob_token(ctx, tok):
        key = (ctx, tok)
        if key in cache: return cache[key]
        num = ngram_counts[ctx][tok] + alpha
        den = context_counts[ctx] + alpha * V
        lp = float(np.log(num / max(den, 1)))
        cache[key] = lp
        return lp
    def log_prob_sequence(seq):
        full = ['<s>'] * (order - 1) + list(seq) + ['</s>']
        lp = 0.0
        for i in range(order - 1, len(full)):
            lp += log_prob_token(tuple(full[i-(order-1):i]), full[i])
        return lp
    return dict(log_prob_token=log_prob_token,
                log_prob_sequence=log_prob_sequence,
                order=order, alpha=alpha, vocab=vocab,
                V=V, n_sentences=n_sentences, n_tokens=n_tokens)


ALL_PIDS = [f"P{i:02d}" for i in range(21, 31)]
LM = train_phoneme_lm_from_pipeline(pipeline, ALL_PIDS, order=3, alpha=1.0)
print(f"LM trained:  order={LM['order']}  vocab={len(LM['vocab'])}  V={LM['V']}")
print(f"             n_sentences={LM['n_sentences']}  n_tokens={LM['n_tokens']}")
# Quick sanity check
test_seqs = [['ɛ','n','t','ɛ'], ['p','a','p','a'], ['x','b','g','f']]
for s in test_seqs:
    print(f"  log_p_seq({s}) = {LM['log_prob_sequence'](s):.2f}  (more negative = less Dutch-like)")

def lm_guided_beam(logp, class_labels, bonus, lm, lm_weight=1.0,
                    beam_width=20, top_k_per_step=12):
    """Beam search combining acoustic per-frame logp and LM phoneme prior.
    
    State: (frame_states_tuple, lm_ctx, acoustic_score, weighted_lm_score)
    LM contribution added only at phoneme transitions (not self-loops).
    """
    T, K = logp.shape
    order = lm['order']
    log_prob_token = lm['log_prob_token']
    START = '<s>'
    init_ctx = (START,) * (order - 1)

    # initial beams
    init_top_k = np.argsort(-logp[0])[:top_k_per_step]
    beams = []
    for k in init_top_k:
        tok = class_labels[k]
        lm_s = lm_weight * log_prob_token(init_ctx, tok)
        new_ctx = (init_ctx + (tok,))[-(order-1):]
        beams.append(((int(k),), new_ctx, float(logp[0, k]), lm_s))
    beams.sort(key=lambda b: -(b[2] + b[3]))
    beams = beams[:beam_width]

    for t in range(1, T):
        # Top-K phonemes for this frame (by acoustic) — global pruning
        top_k_t = np.argsort(-logp[t])[:top_k_per_step]
        cands = []
        for frames, ctx, ac, lm_acc in beams:
            last_k = frames[-1]
            # Always allow self-loop (preserves the persistence mechanism)
            new_ac = ac + logp[t, last_k] + bonus
            cands.append((frames + (last_k,), ctx, new_ac, lm_acc))
            # Try transitions to top-K
            for k_int in top_k_t:
                k = int(k_int)
                if k == last_k: continue
                tok = class_labels[k]
                new_ctx = (ctx + (tok,))[-(order-1):]
                new_lm = lm_acc + lm_weight * log_prob_token(ctx, tok) + insertion_bonus
                new_ac = ac + logp[t, k]
                cands.append((frames + (k,), new_ctx, new_ac, new_lm))
        cands.sort(key=lambda b: -(b[2] + b[3]))
        beams = cands[:beam_width]

    return np.array(beams[0][0], dtype=np.int32)

def rescore_with_lm(beam_paths, beam_acoustic_scores, class_labels,
                     lm, lm_weight=1.0):
    """beam_paths: list of (T,) state-index arrays
       beam_acoustic_scores: list of float acoustic scores
       Returns the index of the best beam after LM rescoring."""
    best_idx, best_combined = 0, -np.inf
    for i, (path, ac) in enumerate(zip(beam_paths, beam_acoustic_scores)):
        # Collapse repeats into a phoneme sequence
        seq = []
        prev = -1
        for s in path:
            if s != prev:
                seq.append(class_labels[s])
                prev = s
        lm_score = lm['log_prob_sequence'](seq)
        combined = ac + lm_weight * lm_score
        if combined > best_combined:
            best_combined, best_idx = combined, i
    return best_idx

def beam_decode_n_best(logp, bonus, beam_width):
    """Plain beam search (no LM). Returns ALL beams as list of
    (acoustic_score, frame_state_path_tuple), sorted best-first."""
    T, K = logp.shape
    beams = [(float(logp[0, k]), (int(k),)) for k in range(K)]
    beams.sort(reverse=True, key=lambda b: b[0]); beams = beams[:beam_width]
    for t in range(1, T):
        candidates = []
        n_keep = min(beam_width, K)
        for score, path in beams:
            last = path[-1]
            base = logp[t]
            bonus_vec = np.zeros(K); bonus_vec[last] = bonus
            new_scores = score + base + bonus_vec
            if n_keep < K:
                top_k = np.argpartition(new_scores, -n_keep)[-n_keep:]
            else:
                top_k = np.arange(K)
            for k in top_k:
                candidates.append((float(new_scores[k]), path + (int(k),)))
        candidates.sort(reverse=True, key=lambda x: x[0])
        beams = candidates[:beam_width]
    return beams


def rescore_n_best(beams, class_labels, lm, lm_weight=1.0):
    """Pick the beam with the best combined acoustic + LM score.
    Returns the index of the best beam in the input list."""
    if lm is None or lm_weight <= 0:
        return 0
    best_idx, best_combined = 0, -np.inf
    for i, (ac_score, path) in enumerate(beams):
        seq = []
        prev = -1
        for s in path:
            if s != prev:
                seq.append(class_labels[s])
                prev = s
        lm_score = lm['log_prob_sequence'](seq)
        combined = ac_score + lm_weight * lm_score
        if combined > best_combined:
            best_combined, best_idx = combined, i
    return best_idx

def run_for_patient_sd_lm(pid, lm,
                           classifier_type='rf',
                           lm_weight=1.0, beam_width=20, top_k_per_step=12,
                           alpha_prior=0.7, proba_temperature=1.0,
                           n_estimators=200, n_pca=100,
                           test_offset=TEST_OFFSET, feature_spec=None,
                           use_speech_gate=USE_SPEECH_GATE,
                           speech_thresh=SPEECH_THRESHOLD,
                           speech_frac_min=SPEECH_FRAC_MIN):
    """RF + α=0.7 pipeline with N-best beam decoding + LM rescoring at test."""
    feature_spec = feature_spec or DEFAULT_FEATURE_SPEC
    wd = pipeline.split_result['word_segments_dict'].get(pid)
    if wd is None: return None, "no word_segments_dict"
    try: mfa = load_mfa_alignments(pid)
    except Exception as e: return None, f"no MFA ({type(e).__name__})"
    if not mfa: return None, "empty MFA"
    if use_speech_gate and pid not in mu_sd_speech:
        return None, f"no speech-detector stats for {pid}"

    all_real = [i for i, s in enumerate(wd['sentence_list'])
                if isinstance(s, dict) and s.get('text')]
    test_sent_ids = set(all_real[test_offset::6])
    train_sent_ids_all = sorted(set(all_real) - test_sent_ids)
    val_step = max(2, int(round(1 / VAL_FRAC)))
    val_sent_ids = set(train_sent_ids_all[::val_step])
    fit_sent_ids = set(train_sent_ids_all) - val_sent_ids

    raw_path = os.path.join(DUTCH_30_PATH, 'raw', f'{pid}_sEEG.npy')
    raw_eeg = np.load(raw_path)
    if raw_eeg.ndim == 2 and raw_eeg.shape[0] < raw_eeg.shape[1]:
        raw_eeg = raw_eeg.T

    per_sentence_stk, per_sentence_mask = {}, {}
    for sent_idx in all_real:
        if sent_idx not in mfa or not mfa[sent_idx]: continue
        s = wd['sentence_list'][sent_idx]
        s0, s1 = s['stim_start_idx'], s['stim_end_idx']
        if s1 > raw_eeg.shape[0]: continue
        ext = extract_features_multiband(raw_eeg[s0:s1], **feature_spec)
        if ext.shape[0] < 2 * LDA_MARGIN + 1: continue
        per_sentence_stk[sent_idx] = stackFeatures(ext, modelOrder=MO, stepSize=SS)
        if use_speech_gate:
            sp = predict_speech_prob(raw_eeg[s0:s1], pid)
            T_stk = per_sentence_stk[sent_idx].shape[0]
            mask = (sp[:T_stk] > speech_thresh) if len(sp) >= T_stk else \
                   np.concatenate([sp > speech_thresh, np.zeros(T_stk - len(sp), dtype=bool)])
            per_sentence_mask[sent_idx] = mask
    if not per_sentence_stk: return None, "no usable sentences"

    GROUP_DELAY_FRAMES = 10
    def build_train_set(sent_ids):
        causal_mode = feature_spec.get('causal', False)
        shift = GROUP_DELAY_FRAMES if causal_mode else 0
        X, y = [], []
        for sent_idx in sent_ids:
            if sent_idx not in per_sentence_stk: continue
            stk = per_sentence_stk[sent_idx]; T_stk = stk.shape[0]
            for ph in mfa[sent_idx]:
                k_s = int(np.ceil ((ph['start_s']*EEG_SR - WIN_SAMP/2) / SHIFT_SAMP))
                k_e = int(np.floor((ph['end_s']  *EEG_SR - WIN_SAMP/2) / SHIFT_SAMP))
                if (k_e - k_s + 1) < MN_FRAMES or (k_e - k_s + 1) > MX_FRAMES: continue
                ks = max(0, k_s - LDA_MARGIN + shift); ke = min(T_stk-1, k_e - LDA_MARGIN + shift)
                if ke < ks: continue
                X.append(stk[ks:ke+1].mean(axis=0)); y.append(ph['phone'])
        return np.array(X, dtype=np.float32), np.array(y)

    X_fit, y_fit = build_train_set(fit_sent_ids)
    if len(X_fit) < 50: return None, f"too few fit samples ({len(X_fit)})"
    fit_classes = set(y_fit)
    sc_fit = StandardScaler().fit(X_fit)
    n_comp_fit = min(n_pca, X_fit.shape[1], X_fit.shape[0] - 1)
    pca_fit = PCA(n_components=n_comp_fit, svd_solver='randomized', random_state=0)
    Xp_fit  = pca_fit.fit_transform(sc_fit.transform(X_fit))
    clf_fit = _make_classifier(classifier_type, n_estimators=n_estimators)
    clf_fit.fit(Xp_fit, y_fit)
    fit_priors = np.array([np.mean(y_fit == c) for c in clf_fit.classes_])
    log_fit_prior = np.log(fit_priors + 1e-12)

    val_logps, val_masks, val_target = [], [], 0
    for sent_idx in val_sent_ids:
        if sent_idx not in per_sentence_stk: continue
        stk = per_sentence_stk[sent_idx].astype(np.float32)
        Xs = pca_fit.transform(sc_fit.transform(stk))
        logp = _predict_log_proba_safe(clf_fit, Xs, temperature=proba_temperature,
                                         log_prior=log_fit_prior, alpha_prior=alpha_prior)
        logp = smooth_cols(logp, SMOOTH_LOGP_W)
        val_logps.append(logp)
        if use_speech_gate: val_masks.append(per_sentence_mask[sent_idx])
        val_target += sum(1 for ph in mfa[sent_idx] if ph['phone'] in fit_classes)
    if not val_logps: return None, "no val sentences"
    val_target = int(val_target * TARGET_RATIO)

    if SELF_LOOP_BONUS is None:
        bonus = (auto_tune_bonus_masked(val_logps, val_masks, val_target,
                                         MIN_PRED_FRAMES, speech_frac=speech_frac_min)
                 if use_speech_gate
                 else auto_tune_bonus(val_logps, val_target, MIN_PRED_FRAMES))
    else:
        bonus = float(SELF_LOOP_BONUS)

    X_train, y_train = build_train_set(set(all_real) - test_sent_ids)
    train_classes = set(y_train)
    scaler = StandardScaler().fit(X_train)
    n_comp = min(n_pca, X_train.shape[1], X_train.shape[0] - 1)
    pca = PCA(n_components=n_comp, svd_solver='randomized', random_state=0)
    Xp_train = pca.fit_transform(scaler.transform(X_train))
    clf = _make_classifier(classifier_type, n_estimators=n_estimators)
    clf.fit(Xp_train, y_train)
    class_labels = list(clf.classes_)
    train_priors = np.array([np.mean(y_train == c) for c in class_labels])
    log_train_prior = np.log(train_priors + 1e-12)

    predictions, pred_sentence_ids, pred_segments = [], [], []
    true_labels, true_sentence_ids, true_segments = [], [], []
    n_dropped_silence = 0
    # Track which beam rank gets picked per sentence (for diagnostics)
    chosen_ranks = []

    for sent_idx in test_sent_ids:
        if sent_idx not in per_sentence_stk: continue
        stk = per_sentence_stk[sent_idx].astype(np.float32); T_stk = stk.shape[0]
        mask = per_sentence_mask.get(sent_idx, np.ones(T_stk, dtype=bool))
        Xs = pca.transform(scaler.transform(stk))
        logp = _predict_log_proba_safe(clf, Xs, temperature=proba_temperature,
                                         log_prior=log_train_prior, alpha_prior=alpha_prior)
        logp = smooth_cols(logp, SMOOTH_LOGP_W)

        # ★ N-best beam search + LM rescoring
        if lm is not None and lm_weight > 0:
            beams = beam_decode_n_best(logp, bonus, beam_width)
            best_rank = rescore_n_best(beams, class_labels, lm, lm_weight=lm_weight)
            path = np.array(beams[best_rank][1], dtype=np.int32)
            chosen_ranks.append(best_rank)
        else:
            path = viterbi_decode(logp, bonus)

        i = 0
        while i < T_stk:
            ci = path[i]; j = i + 1
            while j < T_stk and path[j] == ci: j += 1
            if (j - i) >= MIN_PRED_FRAMES:
                if mask[i:j].mean() >= speech_frac_min:
                    predictions.append(class_labels[ci])
                    pred_sentence_ids.append(sent_idx)
                    pred_segments.append((stk_frame_to_time_s(i), stk_frame_to_time_s(j - 1)))
                else:
                    n_dropped_silence += 1
            i = j
        for ph in mfa[sent_idx]:
            if ph['phone'] not in train_classes: continue
            true_labels.append(ph['phone'])
            true_sentence_ids.append(sent_idx)
            true_segments.append((ph['start_s'], ph['end_s']))

    if not true_labels: return None, "no test gold labels"
    true_arr = np.array(true_labels); pred_arr = np.array(predictions)
    ed = edit_distance(list(true_arr), list(pred_arr))
    per = ed / max(len(true_arr), 1)
    return {
        'true_labels': true_arr, 'predictions': pred_arr,
        'true_sentence_ids': np.array(true_sentence_ids),
        'pred_sentence_ids': np.array(pred_sentence_ids),
        'true_segments': true_segments, 'pred_segments': pred_segments,
        'edit_distance': ed, 'per': per,
        'n_test': len(true_arr), 'n_pred': len(pred_arr),
        'n_train': len(X_train), 'n_val_sents': len(val_logps),
        'bonus': bonus, 'val_target': val_target,
        'n_dropped_silence': n_dropped_silence,
        'lm_weight': lm_weight, 'beam_width': beam_width,
        'chosen_ranks': chosen_ranks,   # diagnostic
    }, None

from collections import Counter
import time

for lmw in [0.0, 0.5, 1.0, 2.0, 5.0]:
    t0 = time.time()
    out, err = run_for_patient_sd_lm("P22", LM,
                                       classifier_type='rf',
                                       lm_weight=lmw,
                                       alpha_prior=0.7,
                                       beam_width=100, top_k_per_step=12)
    if out is None:
        print(f"  lm_weight={lmw}  SKIP: {err}"); continue
    preds_by = by_sentence(out['predictions'], out['pred_sentence_ids'])
    gold_by  = by_sentence(out['true_labels'], out['true_sentence_ids'])
    common = sorted(set(preds_by) | set(gold_by))
    pred_sents = [preds_by.get(s, []) for s in common]
    gold_sents = [gold_by.get(s, [])  for s in common]
    longest = max((longest_run_with_shift(p, g) for p, g in zip(pred_sents, gold_sents)), default=0)
    n3 = sum(count_ngrams_at_least(p, g, 3) for p, g in zip(pred_sents, gold_sents))
    n4 = sum(count_ngrams_at_least(p, g, 4) for p, g in zip(pred_sents, gold_sents))
    n5 = sum(count_ngrams_at_least(p, g, 5) for p, g in zip(pred_sents, gold_sents))
    z, *_ = perm_z(pred_sents, gold_sents, n_perm=500)
    match_score = sum(max(0, longest_run_with_shift(p,g)-2)
                       for p, g in zip(pred_sents, gold_sents))
    pc = Counter(out['predictions'])
    print(f"  lm_weight={lmw:.1f}  per={100*out['per']:5.1f}%  z={z:+5.2f}  "
          f"L={longest}  n3={n3:>3}  n4={n4:>2}  n5={n5:>2}  "
          f"score={match_score:>3}  n_dist={len(pc):>2}  "
          f"({time.time()-t0:.0f}s)", flush=True)

import numpy as np
from collections import Counter, defaultdict

def needleman_wunsch(gold, pred, match_score=1, mismatch=-1, gap=-1):
    """Global sequence alignment. Returns list of (gold_token, pred_token) pairs
    where None indicates a gap (insertion/deletion)."""
    n, m = len(gold), len(pred)
    if n == 0 or m == 0:
        return [(g, None) for g in gold] + [(None, p) for p in pred]
    S = np.zeros((n+1, m+1), dtype=np.float32)
    S[:, 0] = np.arange(n+1) * gap
    S[0, :] = np.arange(m+1) * gap
    BT = np.zeros((n+1, m+1), dtype=np.int8)   # 0=diag, 1=up(del), 2=left(ins)
    BT[:, 0] = 1; BT[0, :] = 2; BT[0, 0] = 0
    for i in range(1, n+1):
        gi = gold[i-1]
        for j in range(1, m+1):
            diag = S[i-1, j-1] + (match_score if gi == pred[j-1] else mismatch)
            up   = S[i-1, j] + gap
            left = S[i, j-1] + gap
            if diag >= up and diag >= left: S[i, j] = diag; BT[i, j] = 0
            elif up >= left:                S[i, j] = up;   BT[i, j] = 1
            else:                           S[i, j] = left; BT[i, j] = 2
    # Backtrack
    aligned, i, j = [], n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and BT[i, j] == 0:
            aligned.append((gold[i-1], pred[j-1])); i -= 1; j -= 1
        elif i > 0 and BT[i, j] == 1:
            aligned.append((gold[i-1], None)); i -= 1
        else:
            aligned.append((None, pred[j-1])); j -= 1
    return list(reversed(aligned))


def analyze_sequence_aligned(out, label="model"):
    """Run NW per-sentence, aggregate substitution / deletion / insertion stats."""
    # Group by sentence
    gold_per = defaultdict(list); pred_per = defaultdict(list)
    for lbl, sid in zip(out['true_labels'], out['true_sentence_ids']):
        gold_per[int(sid)].append(lbl)
    for lbl, sid in zip(out['predictions'], out['pred_sentence_ids']):
        pred_per[int(sid)].append(lbl)

    all_aligned = []
    for sid in sorted(set(gold_per) | set(pred_per)):
        gold = gold_per.get(sid, [])
        pred = pred_per.get(sid, [])
        all_aligned.extend(needleman_wunsch(gold, pred))

    n_match  = sum(1 for g, p in all_aligned if g is not None and p is not None and g == p)
    n_subst  = sum(1 for g, p in all_aligned if g is not None and p is not None and g != p)
    n_del    = sum(1 for g, p in all_aligned if g is not None and p is None)
    n_ins    = sum(1 for g, p in all_aligned if g is None and p is not None)
    n_gold   = sum(1 for g, p in all_aligned if g is not None)
    
    print(f"\n=== {label}: SEQUENCE-LEVEL alignment (Needleman-Wunsch) ===")
    print(f"  Gold tokens:    {n_gold}")
    print(f"  Match:          {n_match:>4}  ({100*n_match/max(n_gold,1):5.1f}%)")
    print(f"  Substitution:   {n_subst:>4}  ({100*n_subst/max(n_gold,1):5.1f}%)")
    print(f"  Deletion:       {n_del:>4}  ({100*n_del/max(n_gold,1):5.1f}%)")
    print(f"  Insertion:      {n_ins:>4}")
    
    # Manner-preserved on substitutions
    manner_eq = sum(1 for g, p in all_aligned
                    if g is not None and p is not None and g != p
                    and manner_of(g) != '?' and manner_of(p) != '?'
                    and manner_of(g) == manner_of(p))
    manner_total = sum(1 for g, p in all_aligned
                        if g is not None and p is not None and g != p
                        and manner_of(g) != '?' and manner_of(p) != '?')
    print(f"  Manner-preserved on subst: {manner_eq}/{manner_total} "
          f"({100*manner_eq/max(manner_total,1):.1f}%)")

    # Per-gold-phoneme recall (sequence-level)
    gold_count = Counter(g for g, _ in all_aligned if g is not None)
    correct    = Counter(g for g, p in all_aligned if g is not None and p is not None and g == p)
    top_pred   = defaultdict(Counter)
    for g, p in all_aligned:
        if g is not None and p is not None and g != p:
            top_pred[g][p] += 1
    print(f"\n  Per-gold-phoneme recall (≥5 occurrences):")
    print(f"  {'gold':<6} {'man':<4} {'n':>4} {'recall':>7}   top mistakes")
    for ph, n in gold_count.most_common():
        if n < 5: continue
        rec = correct.get(ph, 0)
        top_subs = ', '.join(f"{p}({c})" for p, c in top_pred[ph].most_common(3))
        print(f"  {ph:<6} {manner_of(ph):<4} {n:>4} {100*rec/n:>6.0f}%   {top_subs}")

    return all_aligned


# Compare against your earlier RF+α=0.7 result
print("="*60)
print("Compare time-strict vs sequence-only alignment for P22")
print("="*60)
out, _ = run_for_patient_sd_clf("P22", classifier_type='rf',
                                  proba_temperature=1.0, alpha_prior=0.7)
all_aligned = analyze_sequence_aligned(out, label="P22 (RF + α=0.7)")

import numpy as np
from collections import defaultdict
from IPython.display import display, HTML

def needleman_wunsch(gold, pred, match=1, mismatch=-1, gap=-1):
    """Global sequence alignment. Returns list of (g, p) pairs;
    None on either side = gap (insertion/deletion)."""
    n, m = len(gold), len(pred)
    if n == 0: return [(None, p) for p in pred]
    if m == 0: return [(g, None) for g in gold]
    S = np.zeros((n+1, m+1), dtype=np.float32)
    S[:, 0] = np.arange(n+1) * gap; S[0, :] = np.arange(m+1) * gap
    BT = np.zeros((n+1, m+1), dtype=np.int8)
    BT[:, 0] = 1; BT[0, :] = 2; BT[0, 0] = 0
    for i in range(1, n+1):
        gi = gold[i-1]
        for j in range(1, m+1):
            d = S[i-1, j-1] + (match if gi == pred[j-1] else mismatch)
            u = S[i-1, j] + gap; l = S[i, j-1] + gap
            if d >= u and d >= l: S[i, j] = d; BT[i, j] = 0
            elif u >= l:          S[i, j] = u; BT[i, j] = 1
            else:                 S[i, j] = l; BT[i, j] = 2
    aligned, i, j = [], n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and BT[i, j] == 0:
            aligned.append((gold[i-1], pred[j-1])); i -= 1; j -= 1
        elif i > 0 and BT[i, j] == 1:
            aligned.append((gold[i-1], None)); i -= 1
        else:
            aligned.append((None, pred[j-1])); j -= 1
    return list(reversed(aligned))


def gather_sequences(out):
    """Group flat predictions/gold by sentence_id, preserving sequence order."""
    gold_per = defaultdict(list)
    for lbl, sid in zip(out['true_labels'], out['true_sentence_ids']):
        gold_per[int(sid)].append(lbl)
    pred_per = defaultdict(list)
    for lbl, sid in zip(out['predictions'], out['pred_sentence_ids']):
        pred_per[int(sid)].append(lbl)
    return gold_per, pred_per


# Color scheme — same colors used for gold cell, pred cell, and op marker
COL_MATCH = '#a6e3a1'    # green
COL_SUB   = '#f5c2c0'    # red
COL_INS   = '#ffd966'    # yellow
COL_DEL   = '#dddddd'    # gray

def render_aligned_pair(aligned):
    """Render one NW alignment as 2 stacked HTML rows (gold + pred)
       with consistent coloring per edit operation."""
    gold_cells, pred_cells = [], []
    cell_style = "padding:2px 5px;margin-right:1px;display:inline-block;min-width:14px;text-align:center;border-radius:3px;"
    for g, p in aligned:
        if g is not None and p is not None and g == p:
            c = COL_MATCH
            gold_cells.append(f"<span style='{cell_style}background:{c};font-weight:bold;'>{g}</span>")
            pred_cells.append(f"<span style='{cell_style}background:{c};font-weight:bold;'>{p}</span>")
        elif g is not None and p is not None:
            c = COL_SUB
            gold_cells.append(f"<span style='{cell_style}background:{c};font-weight:bold;'>{g}</span>")
            pred_cells.append(f"<span style='{cell_style}background:{c};'>{p}</span>")
        elif g is not None:
            c = COL_DEL
            gold_cells.append(f"<span style='{cell_style}background:{c};font-weight:bold;'>{g}</span>")
            pred_cells.append(f"<span style='{cell_style}background:{c};color:#888;'>·</span>")
        else:
            c = COL_INS
            gold_cells.append(f"<span style='{cell_style}background:{c};color:#888;'>·</span>")
            pred_cells.append(f"<span style='{cell_style}background:{c};'>{p}</span>")
    return ''.join(gold_cells), ''.join(pred_cells)


def compare_predictions_html(out_a, out_b, label_a="A", label_b="B",
                              max_sentences=20):
    """Sequence-level (Needleman-Wunsch) comparison of two models against gold,
    with consistent coloring: green=match, red=sub, yellow=insertion, gray=deletion."""
    gold_a_per, pred_a_per = gather_sequences(out_a)
    gold_b_per, pred_b_per = gather_sequences(out_b)
    common = sorted(set(gold_a_per) | set(gold_b_per))

    rows = []
    rows.append("<style>"
                ".pcomp td { padding:4px 8px; font-family:monospace; font-size:13px; }"
                ".pcomp tr.header td { background:#444; color:#fff; font-weight:bold; }"
                ".pcomp tr.sentheader td { background:#e0e0e0; font-weight:bold; padding-top:8px; }"
                "</style>")
    rows.append("<div style='margin-bottom:8px;font-family:sans-serif;font-size:13px;'>"
                f"<span style='background:{COL_MATCH};padding:3px 8px;margin-right:6px;border-radius:3px;'>match</span>"
                f"<span style='background:{COL_SUB};padding:3px 8px;margin-right:6px;border-radius:3px;'>substitution</span>"
                f"<span style='background:{COL_INS};padding:3px 8px;margin-right:6px;border-radius:3px;'>insertion (predicted, no gold)</span>"
                f"<span style='background:{COL_DEL};padding:3px 8px;margin-right:6px;border-radius:3px;'>deletion (gold, no pred)</span>"
                "</div>")
    rows.append("<table class='pcomp' style='border-collapse:collapse;'>")
    for sid in common[:max_sentences]:
        gold_a = gold_a_per.get(sid, [])
        gold_b = gold_b_per.get(sid, [])
        pred_a = pred_a_per.get(sid, [])
        pred_b = pred_b_per.get(sid, [])
        if not gold_a and not gold_b: continue
        gold = gold_a if gold_a else gold_b

        align_a = needleman_wunsch(gold, pred_a)
        align_b = needleman_wunsch(gold, pred_b)
        g_a, p_a = render_aligned_pair(align_a)
        g_b, p_b = render_aligned_pair(align_b)

        rows.append(f"<tr class='sentheader'><td colspan='2'>Sentence {sid}</td></tr>")
        rows.append(f"<tr><td>{label_a} gold</td><td>{g_a}</td></tr>")
        rows.append(f"<tr><td>{label_a} pred</td><td>{p_a}</td></tr>")
        rows.append(f"<tr><td>{label_b} gold</td><td>{g_b}</td></tr>")
        rows.append(f"<tr><td>{label_b} pred</td><td>{p_b}</td></tr>")
    rows.append("</table>")
    return ''.join(rows)


# ── Run ──
PID = "P23"
print("Running model A (RF + α=0.7)...", flush=True)
out_a, _ = run_for_patient_sd_clf(PID, classifier_type='rf',
                                    proba_temperature=1.0, alpha_prior=0.7)
print("Running model B (LDA + α=0)...", flush=True)
out_b, _ = run_for_patient_sd_clf(PID, classifier_type='lda',
                                    proba_temperature=1.0, alpha_prior=0.0)

# Quick stats using NW alignment
def stats_nw(out, label):
    gold_per, pred_per = gather_sequences(out)
    total_align = []
    for sid in set(gold_per) | set(pred_per):
        total_align.extend(needleman_wunsch(gold_per.get(sid, []), pred_per.get(sid, [])))
    n_match = sum(1 for g, p in total_align if g is not None and p is not None and g == p)
    n_sub   = sum(1 for g, p in total_align if g is not None and p is not None and g != p)
    n_del   = sum(1 for g, p in total_align if g is not None and p is None)
    n_ins   = sum(1 for g, p in total_align if g is None and p is not None)
    n_gold  = sum(1 for g, p in total_align if g is not None)
    n_pred  = sum(1 for g, p in total_align if p is not None)
    print(f"\n  {label}:  gold={n_gold}  pred={n_pred}")
    print(f"    match={n_match} ({100*n_match/max(n_gold,1):.1f}%)  "
          f"sub={n_sub} ({100*n_sub/max(n_gold,1):.1f}%)  "
          f"del={n_del} ({100*n_del/max(n_gold,1):.1f}%)  "
          f"ins={n_ins}")

stats_nw(out_a, "A (RF + α=0.7)")
stats_nw(out_b, "B (LDA + α=0)")

display(HTML(compare_predictions_html(out_a, out_b,
                                        label_a="RF+α0.7",
                                        label_b="LDA",
                                        max_sentences=20)))

import numpy as np
from collections import Counter, defaultdict

def nw_metrics(out, manner_fn=None, n_perm=500, seed=0):
    """Compute a suite of NW-alignment metrics for one patient's result.
    
    Returns a dict with:
      - match_rate (S-Acc): proportion of gold tokens recovered by NW match
      - per_nw: sequence-level PER = (sub+del+ins)/|gold|
      - manner_acc: match + 0.5*(manner-preserved substitutions) per gold
      - chance_baseline: best you'd do by always predicting the modal phoneme
      - freq_baseline: NW match rate if predictions = randomly drawn from gold freq distribution
      - z_match: permutation z for the match rate (null = shuffled predictions)
      - n2, n3, n4: NW-aligned consecutive match counts
    """
    gold_per = defaultdict(list); pred_per = defaultdict(list)
    for lbl, sid in zip(out['true_labels'], out['true_sentence_ids']):
        gold_per[int(sid)].append(lbl)
    for lbl, sid in zip(out['predictions'], out['pred_sentence_ids']):
        pred_per[int(sid)].append(lbl)

    # ── per-sentence NW alignment ──
    sent_alignments = {}
    all_gold, all_pred = [], []
    for sid in set(gold_per) | set(pred_per):
        g, p = gold_per.get(sid, []), pred_per.get(sid, [])
        sent_alignments[sid] = needleman_wunsch(g, p)
        all_gold += g; all_pred += p

    # Aggregate edit-op counts
    all_aligned = [pair for a in sent_alignments.values() for pair in a]
    n_match = sum(1 for g, p in all_aligned if g is not None and p is not None and g == p)
    n_sub   = sum(1 for g, p in all_aligned if g is not None and p is not None and g != p)
    n_del   = sum(1 for g, p in all_aligned if g is not None and p is None)
    n_ins   = sum(1 for g, p in all_aligned if g is None and p is not None)
    n_gold  = sum(1 for g, p in all_aligned if g is not None)

    # ── Match rate (NW accuracy) ──
    match_rate = n_match / max(n_gold, 1)
    per_nw     = (n_sub + n_del + n_ins) / max(n_gold, 1)

    # ── Manner-weighted accuracy ──
    if manner_fn is None:
        manner_fn = lambda x: '?'
    manner_partial = sum(
        0.5 for g, p in all_aligned
        if g is not None and p is not None and g != p
        and manner_fn(g) != '?' and manner_fn(p) != '?'
        and manner_fn(g) == manner_fn(p)
    )
    manner_acc = (n_match + manner_partial) / max(n_gold, 1)

    # ── Baselines ──
    gold_dist = Counter(all_gold); N = sum(gold_dist.values())
    chance_baseline = max(gold_dist.values()) / N if N > 0 else 0
    freq_baseline   = sum((c / N) ** 2 for c in gold_dist.values()) if N > 0 else 0

    # ── Permutation z for match rate ──
    rng = np.random.default_rng(seed)
    null_match_rates = []
    pred_pool = all_pred[:]
    for _ in range(n_perm):
        rng.shuffle(pred_pool)
        cur = 0; nm = 0
        for sid in sent_alignments:
            g = gold_per.get(sid, [])
            p_len = len(pred_per.get(sid, []))
            p_shuf = pred_pool[cur:cur + p_len]; cur += p_len
            a = needleman_wunsch(g, p_shuf)
            nm += sum(1 for x, y in a if x is not None and y is not None and x == y)
        null_match_rates.append(nm / max(n_gold, 1))
    null_mean = np.mean(null_match_rates); null_std = np.std(null_match_rates) + 1e-9
    z_match = (match_rate - null_mean) / null_std

    # ── NW-aligned consecutive matches ──
    # Walk through alignment; count runs of consecutive matches (gold side)
    def count_runs(alignment, min_len):
        runs = 0
        cur_run = 0
        for g, p in alignment:
            if g is not None and p is not None and g == p:
                cur_run += 1
            else:
                if cur_run >= min_len: runs += 1
                cur_run = 0
        if cur_run >= min_len: runs += 1
        return runs
    n2 = sum(count_runs(a, 2) for a in sent_alignments.values())
    n3 = sum(count_runs(a, 3) for a in sent_alignments.values())
    n4 = sum(count_runs(a, 4) for a in sent_alignments.values())

    return dict(
        n_gold=n_gold, n_match=n_match, n_sub=n_sub, n_del=n_del, n_ins=n_ins,
        match_rate=match_rate, per_nw=per_nw, manner_acc=manner_acc,
        chance_baseline=chance_baseline, freq_baseline=freq_baseline,
        z_match=z_match, null_mean=null_mean, null_std=null_std,
        n2=n2, n3=n3, n4=n4,
    )


def print_nw_metrics(m, label=""):
    print(f"\n=== NW metrics: {label} ===")
    print(f"  gold tokens:        {m['n_gold']}")
    print(f"  matches:            {m['n_match']} ({100*m['match_rate']:.1f}%)")
    print(f"  substitutions:      {m['n_sub']}")
    print(f"  deletions:          {m['n_del']}")
    print(f"  insertions:         {m['n_ins']}")
    print(f"  sequence PER:       {100*m['per_nw']:.1f}%")
    print(f"  manner accuracy:    {100*m['manner_acc']:.1f}% (match + 0.5×same-manner subs)")
    print(f"\n  baselines:")
    print(f"    majority class:   {100*m['chance_baseline']:.1f}% (predict most common phoneme always)")
    print(f"    freq-matched:     {100*m['freq_baseline']:.1f}% (predict ∝ gold distribution)")
    print(f"    lift vs majority: {m['match_rate'] / max(m['chance_baseline'], 1e-9):.2f}×")
    print(f"\n  permutation z (NW match rate): {m['z_match']:+5.2f}")
    print(f"    null mean = {100*m['null_mean']:.1f}%   null std = {100*m['null_std']:.2f}pp")
    print(f"\n  NW-aligned consecutive runs:  n2={m['n2']}  n3={m['n3']}  n4={m['n4']}")


# Run on whatever pipeline output you want to evaluate
m_a = nw_metrics(out_a, manner_fn=manner_of)
print_nw_metrics(m_a, "RF + α=0.7  (P22)")

m_b = nw_metrics(out_b, manner_fn=manner_of)
print_nw_metrics(m_b, "LDA + α=0  (P22)")

PATIENTS_SW = ["P21","P22","P23","P24","P25","P26","P27","P28","P29","P30"]
LM_WEIGHTS  = [0.0, 0.5, 1.0, 2.0, 5.0]

all_runs_lm = {pt: {} for pt in PATIENTS_SW}

for pt in PATIENTS_SW:
    print(f"\n=== {pt} ===", flush=True)
    for lmw in LM_WEIGHTS:
        t0 = time.time()
        try:
            out, err = run_for_patient_sd_lm(pt, LM,
                                               classifier_type='rf',
                                               lm_weight=lmw,
                                               alpha_prior=0.7,
                                               beam_width=20, top_k_per_step=12)
            if out is None:
                print(f"  lm_w={lmw:.1f}  SKIP: {err}", flush=True); continue
            preds_by = by_sentence(out['predictions'], out['pred_sentence_ids'])
            gold_by  = by_sentence(out['true_labels'], out['true_sentence_ids'])
            common = sorted(set(preds_by) | set(gold_by))
            pred_sents = [preds_by.get(s, []) for s in common]
            gold_sents = [gold_by.get(s, [])  for s in common]
            longest = max((longest_run_with_shift(p, g) for p, g in zip(pred_sents, gold_sents)), default=0)
            n3 = sum(count_ngrams_at_least(p, g, 3) for p, g in zip(pred_sents, gold_sents))
            n4 = sum(count_ngrams_at_least(p, g, 4) for p, g in zip(pred_sents, gold_sents))
            n5 = sum(count_ngrams_at_least(p, g, 5) for p, g in zip(pred_sents, gold_sents))
            z, *_ = perm_z(pred_sents, gold_sents, n_perm=500)
            match_score = sum(max(0, longest_run_with_shift(p,g)-2)
                               for p, g in zip(pred_sents, gold_sents))
            all_runs_lm[pt][lmw] = dict(per=out['per'], z=z, longest=longest,
                                        n3=n3, n4=n4, n5=n5,
                                        match_score=match_score)
            print(f"  lm_w={lmw:.1f}  per={100*out['per']:5.1f}%  z={z:+5.2f}  "
                  f"L={longest}  n3={n3:>3}  n4={n4:>2}  n5={n5:>2}  "
                  f"score={match_score:>3}  ({time.time()-t0:.0f}s)", flush=True)
        except Exception as e:
            import traceback
            print(f"  lm_w={lmw:.1f}  EXCEPTION: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()

# Cohort summary
print(f"\n{'='*80}")
print(f"Cohort summary @ RF + α=0.7 + LM-guided beam:")
print(f"{'lm_w':<6} {'z̄':>7} {'per̄':>7} {'L̄':>5} {'Σn3':>5} {'Σn4':>4} {'Σn5':>4} {'Σscore':>8}")
print("="*80)
for lmw in LM_WEIGHTS:
    vals = [all_runs_lm[pt][lmw] for pt in PATIENTS_SW if lmw in all_runs_lm[pt]]
    if not vals: continue
    print(f"{lmw:<6.1f} {np.mean([v['z'] for v in vals]):>+6.2f} "
          f"{100*np.mean([v['per'] for v in vals]):>6.1f}% "
          f"{np.mean([v['longest'] for v in vals]):>5.1f} "
          f"{sum(v['n3'] for v in vals):>5} "
          f"{sum(v['n4'] for v in vals):>4} "
          f"{sum(v['n5'] for v in vals):>4} "
          f"{sum(v['match_score'] for v in vals):>8}")
