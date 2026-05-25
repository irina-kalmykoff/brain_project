# LDA-on-frames pipeline — clean version (LDA only, no RF / decoder / LM branches).
# Kept: original run_for_patient + speech-gated run_for_patient_sd, all NW analysis,
#       one visualization, permutation null, cohort sweep skeleton.

# ============================================================
# 1. Setup
# ============================================================
import os, time, re
import warnings
import numpy as np
import scipy
from scipy.signal import (butter, sosfiltfilt, sosfilt, hilbert,
                          iirfilter, iirnotch, tf2sos)
from collections import Counter, defaultdict

import torch
import torch.nn as nn

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

from dutch_30_pipeline import Dutch30Pipeline
from dutch_30_feature_extractor import Dutch30FeatureExtractor
from dataset_config import Dutch30Config
from config import DUTCH_30_PATH
from run_pipeline import load_mfa_alignments
from extract_features import extractHG, stackFeatures
from e2e_brain_decoder import edit_distance

# ── pipeline init ───────────────────────────────────────────
config = Dutch30Config()
extractor = Dutch30FeatureExtractor(config=config)
pipeline = Dutch30Pipeline(extractor, config=config, use_wav2vec=False)
pipeline.step1_load_dutch30_data(patient_range=(21, 30))
pipeline.step2_split_by_instances(train_fraction=0.8)
pipeline.step3_load_channel_exclusions('channel_exclusions.json')
pipeline.apply_channel_exclusions()

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================
# 2. Constants
# ============================================================
# ── dataset + split ─────────────────────────────────────────
TARGET_PIDS  = ['P21', 'P22', 'P23', 'P24', 'P25',
                'P26', 'P27', 'P28', 'P29', 'P30']
TEST_OFFSET  = 0
VAL_FRAC     = 0.15

# ── sampling and frame geometry ─────────────────────────────
EEG_SR          = 1024
WIN_S, SHIFT_S  = 0.015, 0.005
WIN_SAMP        = int(EEG_SR * WIN_S)
SHIFT_SAMP      = int(EEG_SR * SHIFT_S)

# ── feature stacking ────────────────────────────────────────
MO, SS          = 11, 1
LDA_MARGIN      = MO * SS

# ── phoneme label filter (train side only) ──────────────────
MN_FRAMES       = 0
MX_FRAMES       = 300

# ── multiband feature extractor knobs ───────────────────────
HG_BAND       = (70, 170)
LG_BAND       = (30, 70)
THETA_BAND    = (4, 8)
NOTCH_HZ      = (100, 150)

HG_LP_HZ      = 10.0
LG_LP_HZ      = 10.0
PHASE_LP_HZ   = 20.0
PAC_LP_HZ     = 10.0
LG_PAC_LP_HZ  = 10.0

DEFAULT_FEATURE_SPEC = {
    'hg_amp':   True,
    'hg_lp_hz': HG_LP_HZ,
}

# ── post-LDA decoding ───────────────────────────────────────
SMOOTH_LOGP_W   = 31
SELF_LOOP_BONUS = None        # None ⇒ auto-tune on val
TARGET_RATIO    = 1.0
MIN_PRED_FRAMES = 3

# ── speech gating (LOCKED — do not change without retraining the detector) ──
SD_LP_HZ           = 12.0
SD_NOTCH_HZ        = (50, 150)
SD_BAND            = (70, 170)
USE_SPEECH_GATE    = True
SPEECH_THRESHOLD   = 0.5
SPEECH_FRAC_MIN    = 0.5

# ── misc ────────────────────────────────────────────────────
RARE_TOP_N = 5

# ============================================================
# 3. Formatting & lightweight helpers
# ============================================================
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

def count_rare_in_matches(matches, gold_sents, top_n=RARE_TOP_N):
    all_gold = [ph for s in gold_sents for ph in s]
    common = set(p for p, _ in Counter(all_gold).most_common(top_n))
    return sum(1 for m in matches for ph in m if ph not in common)

def stk_frame_to_time_s(i):
    """Map a stacked-frame index back to a wall-clock time (s) in the sentence."""
    return ((i + LDA_MARGIN) * SHIFT_SAMP + WIN_SAMP / 2) / EEG_SR

def by_sentence(arr_labels, arr_sids):
    out = {}
    for lbl, sid in zip(arr_labels, arr_sids):
        out.setdefault(int(sid), []).append(lbl)
    return out

# ============================================================
# 4. Smoothing, Viterbi, bonus tuning
# ============================================================
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
    """Viterbi where staying in the same class earns +self_bonus per frame;
       switching pays 0. Higher bonus → longer runs."""
    T, K = logp.shape
    if T == 0: return np.zeros(0, dtype=np.int32)
    delta = np.empty((T, K))
    bptr  = np.empty((T, K), dtype=np.int32)
    delta[0] = logp[0]
    all_k = np.arange(K)
    for t in range(1, T):
        prev = delta[t - 1]
        order = np.argsort(prev)
        idx1, idx2 = order[-1], order[-2]
        best_switch       = np.full(K, prev[idx1])
        best_switch[idx1] = prev[idx2]
        bptr_switch       = np.full(K, idx1)
        bptr_switch[idx1] = idx2
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
        else:                          hi = mid
    return (lo + hi) / 2

# ============================================================
# 5. Sequence-level analysis helpers (z stats, n-grams, permutation null)
# ============================================================
def longest_run_with_shift(pred, gold, shift_max=3):
    best, P, G = 0, len(pred), len(gold)
    for i in range(P):
        for j in range(max(0, i - shift_max), min(G, i + shift_max + 1)):
            k = 0
            while i + k < P and j + k < G and pred[i + k] == gold[j + k]:
                k += 1
            if k > best: best = k
    return best

def count_ngrams_at_least(pred, gold, min_len=4, shift_max=3):
    n, used = 0, set()
    for i in range(len(pred) - min_len + 1):
        if any(x in used for x in range(i, i + min_len)): continue
        for j in range(max(0, i - shift_max),
                       min(len(gold) - min_len + 1, i + shift_max + 1)):
            k = 0
            while (i + k < len(pred) and j + k < len(gold)
                   and pred[i + k] == gold[j + k]): k += 1
            if k >= min_len:
                n += 1
                for x in range(i, i + k): used.add(x)
                break
    return n

def collect_matches(pred_sents, gold_sents, min_match=3, shift_max=3):
    """For each sentence pair, return the longest matching substring (with small shift)."""
    matches = []
    for p, g in zip(pred_sents, gold_sents):
        L, span, P, G = 0, None, len(p), len(g)
        for i in range(P):
            for j in range(max(0, i - shift_max), min(G, i + shift_max + 1)):
                k = 0
                while i + k < P and j + k < G and p[i + k] == g[j + k]: k += 1
                if k > L: L, span = k, (i, j, k)
        if L >= min_match and span is not None:
            i, j, k = span
            matches.append(tuple(p[i:i + k]))
    return matches

def surprise_score(matches, marginal_logp):
    """Sum of -log P(ph) across matched phonemes (rare matches are more surprising)."""
    fallback = -np.log(1e-6)
    return sum(-marginal_logp.get(ph, fallback) for m in matches for ph in m)

def perm_null(pred_sents, gold_sents, marginal_logp, n_perm=2000, seed=0):
    """Within-sentence permutation null for the surprise score."""
    rng = np.random.default_rng(seed)
    nulls = np.zeros(n_perm)
    for b in range(n_perm):
        shuf = []
        for p in pred_sents:
            if len(p) == 0: shuf.append(p); continue
            idx = rng.permutation(len(p))
            shuf.append([p[k] for k in idx])
        nulls[b] = surprise_score(collect_matches(shuf, gold_sents), marginal_logp)
    return nulls

def perm_z(pred_sents, gold_sents, n_perm=500, seed=0):
    """Cross-sentence permutation z for the sum of per-sentence longest runs."""
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
            out.append(shuf[cur:cur + len(s)]); cur += len(s)
        nulls[n] = stat(out, gold_sents)
    mu, sd = nulls.mean(), nulls.std() + 1e-9
    return (obs - mu) / sd, obs, mu, sd

# ============================================================
# 6. Feature extraction: extract_features_multiband
# ============================================================
def extract_features_multiband(eeg_slice, sr=EEG_SR, win_s=WIN_S, shift_s=SHIFT_S,
                               hg_amp=False, lg_amp=False,
                               theta_phase=False,
                               theta_hg_pac=False, lg_theta_pac=False,
                               hg_x_lg=False,
                               hg_lp_hz=12.0, lg_lp_hz=10.0,
                               phase_lp_hz=20.0,
                               pac_lp_hz=10.0, lg_pac_lp_hz=10.0,
                               causal=False):
    """Cross-band feature extractor.

    Amplitude features (one block of n_channels each):
        hg_amp           — HG envelope (70–170 Hz)
        lg_amp           — LG envelope (30–70 Hz)
        hg_x_lg          — HG_env × LG_env (cross-band co-activation)

    Phase / coupling features (two blocks each, cos + sin):
        theta_phase      — cos/sin of theta phase (4–8 Hz)
        theta_hg_pac     — HG_env × cos(θ),  HG_env × sin(θ)
        lg_theta_pac     — LG_env × cos(θ),  LG_env × sin(θ)
    """
    filter_fn = sosfilt if causal else sosfiltfilt

    x = scipy.signal.detrend(eeg_slice, axis=0)
    for f0 in [100, 150]:
        sos = iirfilter(4, [(f0 - 2) / (sr / 2), (f0 + 2) / (sr / 2)],
                        btype='bandstop', output='sos')
        x = filter_fn(sos, x, axis=0)

    hg_env = lg_env = theta_ph = None
    need_hg    = hg_amp or theta_hg_pac or hg_x_lg
    need_lg    = lg_amp or lg_theta_pac or hg_x_lg
    need_theta = theta_phase or theta_hg_pac or lg_theta_pac

    if need_hg:
        sos_hg = butter(4, [70, 170], btype='bandpass', fs=sr, output='sos')
        x_hg   = filter_fn(sos_hg, x, axis=0)
        lp     = butter(4, hg_lp_hz, btype='lowpass', fs=sr, output='sos')
        hg_env = np.sqrt(np.abs(filter_fn(lp, x_hg ** 2, axis=0)))
    if need_lg:
        sos_lg = butter(4, [30, 70], btype='bandpass', fs=sr, output='sos')
        x_lg   = filter_fn(sos_lg, x, axis=0)
        lp     = butter(4, lg_lp_hz, btype='lowpass', fs=sr, output='sos')
        lg_env = np.sqrt(np.abs(filter_fn(lp, x_lg ** 2, axis=0)))
    if need_theta:
        # hilbert is FFT-based — inherently non-causal.
        if causal:
            raise NotImplementedError("Causal mode doesn't support theta_phase/PAC.")
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
        return filter_fn(sos, arr, axis=0)

    blocks = []
    if hg_amp:  blocks.append(wm(hg_env))
    if lg_amp:  blocks.append(wm(lg_env))
    if hg_x_lg: blocks.append(wm(hg_env * lg_env))
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

# ============================================================
# 7. Speech detector (LOCKED at training time — do not modify the SD_* signal-proc chain)
# ============================================================
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

# Load checkpoint at import time (matches notebook behavior)
ckpt = torch.load('bio_models/speech_detector_cross_patient.pt',
                  map_location=DEVICE, weights_only=False)
sd_model = CrossPatientSpeechDetector(
    n_in_per_pid=ckpt['n_in_per_pid'],
    **ckpt['arch']).to(DEVICE)
sd_model.load_state_dict(ckpt['state_dict'])
sd_model.eval()
mu_sd_speech = ckpt['mu_sd']
print(f"Loaded speech detector for {sorted(sd_model.projs.keys())}")

# Signal-processing chain — locked to training-time values
_sd_sos_bp = butter(4, list(SD_BAND), btype='bandpass', fs=EEG_SR, output='sos')
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
    """Per-stacked-frame speech probability. Uses MO=5, SS=1 stacking to match
       how the detector was trained."""
    hg = extract_hg_frames(raw_eeg_slice)
    hg_stk = stackFeatures(hg, modelOrder=5, stepSize=1)
    mu, sd = mu_sd_speech[pid]
    sd_safe = np.where(sd < 1e-6, 1.0, sd)
    x_t = torch.from_numpy((hg_stk - mu) / sd_safe).float().to(DEVICE)
    return torch.softmax(sd_model(x_t, pid), dim=-1)[:, 1].cpu().numpy()

# ============================================================
# 8. Per-patient pipelines
# ============================================================
def run_for_patient(pid, test_offset=TEST_OFFSET, smoothing_hz=10.0):
    """Original LDA pipeline: extractHG → stack → LDA → smooth → Viterbi.
       Auto-tunes the self-loop bonus on a held-out val slice."""
    wd = pipeline.split_result['word_segments_dict'].get(pid)
    if wd is None: return None, "no word_segments_dict"
    try: mfa = load_mfa_alignments(pid)
    except Exception as e: return None, f"no MFA ({type(e).__name__})"
    if not mfa: return None, "empty MFA"

    all_real = [i for i, s in enumerate(wd['sentence_list'])
                if isinstance(s, dict) and s.get('text')]
    test_sent_ids      = set(all_real[test_offset::6])
    train_sent_ids_all = sorted(set(all_real) - test_sent_ids)
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
                k_s = int(np.ceil ((ph['start_s'] * EEG_SR - WIN_SAMP / 2) / SHIFT_SAMP))
                k_e = int(np.floor((ph['end_s']   * EEG_SR - WIN_SAMP / 2) / SHIFT_SAMP))
                n_fr = k_e - k_s + 1
                if n_fr < MN_FRAMES or n_fr > MX_FRAMES: continue
                ks = max(0, k_s - LDA_MARGIN); ke = min(T_stk - 1, k_e - LDA_MARGIN)
                if ke < ks: continue
                X.append(stk[ks:ke + 1].mean(axis=0))
                y.append(ph['phone'])
        return np.array(X), np.array(y)

    # Step 1: fit LDA on the 85% for bonus tuning only
    X_fit, y_fit = build_train_set(fit_sent_ids)
    if len(X_fit) < 50: return None, f"too few fit samples ({len(X_fit)})"
    fit_classes = set(y_fit)
    sc_fit  = StandardScaler().fit(X_fit)
    clf_fit = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    clf_fit.fit(sc_fit.transform(X_fit), y_fit)

    # Step 2: val log-probs + val target gold-count
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

    # Step 3: auto-tune bonus on val log-probs only
    if SELF_LOOP_BONUS is None:
        bonus = auto_tune_bonus(val_logps, val_target, MIN_PRED_FRAMES)
    else:
        bonus = float(SELF_LOOP_BONUS)

    # Step 4: refit LDA on the full train (fit + val)
    X_train, y_train = build_train_set(set(all_real) - test_sent_ids)
    if len(X_train) < 50: return None, f"too few train samples ({len(X_train)})"
    train_classes = set(y_train)
    scaler = StandardScaler().fit(X_train)
    clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    clf.fit(scaler.transform(X_train), y_train)
    class_labels = list(clf.classes_)

    # Step 5: apply final LDA + tuned bonus to test
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


def run_for_patient_sd(pid, test_offset=TEST_OFFSET, feature_spec=None,
                       use_speech_gate=USE_SPEECH_GATE,
                       speech_thresh=SPEECH_THRESHOLD,
                       speech_frac_min=SPEECH_FRAC_MIN):
    """run_for_patient with multiband features + optional speech-gated decoding.
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
    per_sentence_mask = {}
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

    GROUP_DELAY_FRAMES = 10   # ≈ 50 ms for LP=10 Hz Butterworth-4

    def build_train_set(sent_ids):
        causal_mode = feature_spec.get('causal', False)
        shift = GROUP_DELAY_FRAMES if causal_mode else 0
        X, y = [], []
        for sent_idx in sent_ids:
            if sent_idx not in per_sentence_stk: continue
            stk = per_sentence_stk[sent_idx]; T_stk = stk.shape[0]
            for ph in mfa[sent_idx]:
                k_s = int(np.ceil ((ph['start_s'] * EEG_SR - WIN_SAMP / 2) / SHIFT_SAMP))
                k_e = int(np.floor((ph['end_s']   * EEG_SR - WIN_SAMP / 2) / SHIFT_SAMP))
                if (k_e - k_s + 1) < MN_FRAMES or (k_e - k_s + 1) > MX_FRAMES: continue
                ks = max(0,         k_s - LDA_MARGIN + shift)
                ke = min(T_stk - 1, k_e - LDA_MARGIN + shift)
                if ke < ks: continue
                X.append(stk[ks:ke + 1].mean(axis=0))
                y.append(ph['phone'])
        return np.array(X), np.array(y)

    # Step 1
    X_fit, y_fit = build_train_set(fit_sent_ids)
    if len(X_fit) < 50: return None, f"too few fit samples ({len(X_fit)})"
    fit_classes = set(y_fit)
    sc_fit  = StandardScaler().fit(X_fit)
    clf_fit = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    clf_fit.fit(sc_fit.transform(X_fit), y_fit)

    # Step 2
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

    # Step 3
    if SELF_LOOP_BONUS is None:
        if use_speech_gate:
            bonus = auto_tune_bonus_masked(val_logps, val_masks,
                                            val_target, MIN_PRED_FRAMES,
                                            speech_frac=speech_frac_min)
        else:
            bonus = auto_tune_bonus(val_logps, val_target, MIN_PRED_FRAMES)
    else:
        bonus = float(SELF_LOOP_BONUS)

    # Step 4
    X_train, y_train = build_train_set(set(all_real) - test_sent_ids)
    train_classes = set(y_train)
    scaler = StandardScaler().fit(X_train)
    clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    clf.fit(scaler.transform(X_train), y_train)
    class_labels = list(clf.classes_)

    # Step 5
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

# ============================================================
# 9. Phoneme manner (minimal stub — extend or replace as needed)
# ============================================================
# Dutch IPA manner groups; '?' means unknown phoneme.
_DUTCH_MANNER = {
    # plosives
    'p': 'P', 'b': 'P', 't': 'P', 'd': 'P', 'k': 'P', 'g': 'P', 'c': 'P', 'ʔ': 'P',
    # fricatives
    'f': 'F', 'v': 'F', 's': 'F', 'z': 'F', 'ʃ': 'F', 'ʒ': 'F',
    'x': 'F', 'ɣ': 'F', 'h': 'F',
    # nasals
    'm': 'N', 'n': 'N', 'ŋ': 'N', 'ɲ': 'N',
    # approximants
    'l': 'L', 'r': 'L', 'j': 'L', 'w': 'L', 'ʋ': 'L',
    # vowels
    'a': 'V', 'aː': 'V', 'ɑ': 'V', 'ɛ': 'V', 'eː': 'V', 'ɛi': 'V',
    'i': 'V', 'iː': 'V', 'ɪ': 'V',
    'o': 'V', 'oː': 'V', 'ɔ': 'V', 'ɔu': 'V',
    'u': 'V', 'uː': 'V',
    'y': 'V', 'yː': 'V', 'øː': 'V', 'œ': 'V', 'œy': 'V',
    'ə': 'V', 'ɑu': 'V', 'ɛy': 'V',
}
def manner_of(ph):
    return _DUTCH_MANNER.get(ph, '?')

# ============================================================
# 10. NW alignment + sequence metrics
# ============================================================
def needleman_wunsch(gold, pred, match=1, mismatch=-1, gap=-1):
    """Global sequence alignment. Returns list of (g, p) pairs;
       None on either side = gap (insertion/deletion)."""
    n, m = len(gold), len(pred)
    if n == 0: return [(None, p) for p in pred]
    if m == 0: return [(g, None) for g in gold]
    S  = np.zeros((n + 1, m + 1), dtype=np.float32)
    S[:, 0] = np.arange(n + 1) * gap
    S[0, :] = np.arange(m + 1) * gap
    BT = np.zeros((n + 1, m + 1), dtype=np.int8)   # 0=diag, 1=up(del), 2=left(ins)
    BT[:, 0] = 1; BT[0, :] = 2; BT[0, 0] = 0
    for i in range(1, n + 1):
        gi = gold[i - 1]
        for j in range(1, m + 1):
            d = S[i - 1, j - 1] + (match if gi == pred[j - 1] else mismatch)
            u = S[i - 1, j] + gap
            l = S[i,     j - 1] + gap
            if d >= u and d >= l: S[i, j] = d; BT[i, j] = 0
            elif u >= l:          S[i, j] = u; BT[i, j] = 1
            else:                 S[i, j] = l; BT[i, j] = 2
    aligned, i, j = [], n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and BT[i, j] == 0:
            aligned.append((gold[i - 1], pred[j - 1])); i -= 1; j -= 1
        elif i > 0 and BT[i, j] == 1:
            aligned.append((gold[i - 1], None)); i -= 1
        else:
            aligned.append((None, pred[j - 1])); j -= 1
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

def analyze_sequence_aligned(out, label="model"):
    """Per-sentence NW; aggregate substitution/deletion/insertion stats with manner stats."""
    gold_per, pred_per = gather_sequences(out)
    all_aligned = []
    for sid in sorted(set(gold_per) | set(pred_per)):
        all_aligned.extend(needleman_wunsch(gold_per.get(sid, []),
                                            pred_per.get(sid, [])))

    n_match = sum(1 for g, p in all_aligned if g is not None and p is not None and g == p)
    n_subst = sum(1 for g, p in all_aligned if g is not None and p is not None and g != p)
    n_del   = sum(1 for g, p in all_aligned if g is not None and p is None)
    n_ins   = sum(1 for g, p in all_aligned if g is None and p is not None)
    n_gold  = sum(1 for g, p in all_aligned if g is not None)

    print(f"\n=== {label}: SEQUENCE-LEVEL alignment (Needleman-Wunsch) ===")
    print(f"  Gold tokens:    {n_gold}")
    print(f"  Match:          {n_match:>4}  ({100 * n_match / max(n_gold, 1):5.1f}%)")
    print(f"  Substitution:   {n_subst:>4}  ({100 * n_subst / max(n_gold, 1):5.1f}%)")
    print(f"  Deletion:       {n_del:>4}  ({100 * n_del   / max(n_gold, 1):5.1f}%)")
    print(f"  Insertion:      {n_ins:>4}")

    manner_eq = sum(1 for g, p in all_aligned
                    if g is not None and p is not None and g != p
                    and manner_of(g) != '?' and manner_of(p) != '?'
                    and manner_of(g) == manner_of(p))
    manner_total = sum(1 for g, p in all_aligned
                       if g is not None and p is not None and g != p
                       and manner_of(g) != '?' and manner_of(p) != '?')
    print(f"  Manner-preserved on subst: {manner_eq}/{manner_total} "
          f"({100 * manner_eq / max(manner_total, 1):.1f}%)")

    gold_count = Counter(g for g, _ in all_aligned if g is not None)
    correct    = Counter(g for g, p in all_aligned
                         if g is not None and p is not None and g == p)
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
        print(f"  {ph:<6} {manner_of(ph):<4} {n:>4} {100 * rec / n:>6.0f}%   {top_subs}")

    return all_aligned

def nw_metrics(out, manner_fn=None, n_perm=500, seed=0):
    """Suite of NW-alignment metrics for one patient's result.

    Returns a dict with:
      - match_rate (S-Acc), per_nw (sequence-level PER)
      - manner_acc (match + 0.5 * manner-preserved substitutions)
      - chance_baseline, freq_baseline (sanity baselines)
      - z_match (permutation z for the match rate)
      - n2, n3, n4 (NW-aligned consecutive match counts)
    """
    gold_per, pred_per = gather_sequences(out)

    sent_alignments = {}
    all_gold, all_pred = [], []
    for sid in set(gold_per) | set(pred_per):
        g, p = gold_per.get(sid, []), pred_per.get(sid, [])
        sent_alignments[sid] = needleman_wunsch(g, p)
        all_gold += g; all_pred += p

    all_aligned = [pair for a in sent_alignments.values() for pair in a]
    n_match = sum(1 for g, p in all_aligned if g is not None and p is not None and g == p)
    n_sub   = sum(1 for g, p in all_aligned if g is not None and p is not None and g != p)
    n_del   = sum(1 for g, p in all_aligned if g is not None and p is None)
    n_ins   = sum(1 for g, p in all_aligned if g is None and p is not None)
    n_gold  = sum(1 for g, p in all_aligned if g is not None)

    match_rate = n_match / max(n_gold, 1)
    per_nw     = (n_sub + n_del + n_ins) / max(n_gold, 1)

    if manner_fn is None: manner_fn = lambda x: '?'
    manner_partial = sum(
        0.5 for g, p in all_aligned
        if g is not None and p is not None and g != p
        and manner_fn(g) != '?' and manner_fn(p) != '?'
        and manner_fn(g) == manner_fn(p)
    )
    manner_acc = (n_match + manner_partial) / max(n_gold, 1)

    gold_dist = Counter(all_gold); N = sum(gold_dist.values())
    chance_baseline = max(gold_dist.values()) / N if N > 0 else 0
    freq_baseline   = sum((c / N) ** 2 for c in gold_dist.values()) if N > 0 else 0

    # Permutation z for match rate
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

    # NW-aligned consecutive run counts
    def count_runs_in_alignment(alignment, min_len):
        runs = 0; cur_run = 0
        for g, p in alignment:
            if g is not None and p is not None and g == p:
                cur_run += 1
            else:
                if cur_run >= min_len: runs += 1
                cur_run = 0
        if cur_run >= min_len: runs += 1
        return runs
    n2 = sum(count_runs_in_alignment(a, 2) for a in sent_alignments.values())
    n3 = sum(count_runs_in_alignment(a, 3) for a in sent_alignments.values())
    n4 = sum(count_runs_in_alignment(a, 4) for a in sent_alignments.values())

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
    print(f"  matches:            {m['n_match']} ({100 * m['match_rate']:.1f}%)")
    print(f"  substitutions:      {m['n_sub']}")
    print(f"  deletions:          {m['n_del']}")
    print(f"  insertions:         {m['n_ins']}")
    print(f"  sequence PER:       {100 * m['per_nw']:.1f}%")
    print(f"  manner accuracy:    {100 * m['manner_acc']:.1f}% (match + 0.5×same-manner subs)")
    print(f"\n  baselines:")
    print(f"    majority class:   {100 * m['chance_baseline']:.1f}% (predict most common phoneme always)")
    print(f"    freq-matched:     {100 * m['freq_baseline']:.1f}% (predict ∝ gold distribution)")
    print(f"    lift vs majority: {m['match_rate'] / max(m['chance_baseline'], 1e-9):.2f}×")
    print(f"\n  permutation z (NW match rate): {m['z_match']:+5.2f}")
    print(f"    null mean = {100 * m['null_mean']:.1f}%   null std = {100 * m['null_std']:.2f}pp")
    print(f"\n  NW-aligned consecutive runs:  n2={m['n2']}  n3={m['n3']}  n4={m['n4']}")

# ============================================================
# 11. Visualization (HTML — works in Jupyter)
# ============================================================
COL_MATCH = '#a6e3a1'    # green
COL_SUB   = '#f5c2c0'    # red
COL_INS   = '#ffd966'    # yellow
COL_DEL   = '#dddddd'    # gray

def render_aligned_pair(aligned):
    """Render one NW alignment as 2 HTML rows (gold + pred) with consistent coloring."""
    gold_cells, pred_cells = [], []
    cell_style = ("padding:2px 5px;margin-right:1px;display:inline-block;"
                  "min-width:14px;text-align:center;border-radius:3px;")
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

def compare_predictions_html(out_a, out_b=None, label_a="A", label_b="B",
                             max_sentences=20):
    """Sequence-level (Needleman-Wunsch) visualization of model predictions vs gold.
       Pass out_b=None to render just out_a."""
    gold_a_per, pred_a_per = gather_sequences(out_a)
    if out_b is not None:
        gold_b_per, pred_b_per = gather_sequences(out_b)
        common = sorted(set(gold_a_per) | set(gold_b_per))
    else:
        gold_b_per = pred_b_per = None
        common = sorted(set(gold_a_per))

    rows = []
    rows.append("<style>"
                ".pcomp td { padding:4px 8px; font-family:monospace; font-size:13px; }"
                ".pcomp tr.header td { background:#444; color:#fff; font-weight:bold; }"
                ".pcomp tr.sentheader td { background:#e0e0e0; font-weight:bold; padding-top:8px; }"
                "</style>")
    rows.append("<div style='margin-bottom:8px;font-family:sans-serif;font-size:13px;'>"
                f"<span style='background:{COL_MATCH};padding:3px 8px;margin-right:6px;border-radius:3px;'>match</span>"
                f"<span style='background:{COL_SUB};padding:3px 8px;margin-right:6px;border-radius:3px;'>substitution</span>"
                f"<span style='background:{COL_INS};padding:3px 8px;margin-right:6px;border-radius:3px;'>insertion</span>"
                f"<span style='background:{COL_DEL};padding:3px 8px;margin-right:6px;border-radius:3px;'>deletion</span>"
                "</div>")
    rows.append("<table class='pcomp' style='border-collapse:collapse;'>")
    for sid in common[:max_sentences]:
        gold_a = gold_a_per.get(sid, [])
        pred_a = pred_a_per.get(sid, [])
        if out_b is not None:
            gold_b = gold_b_per.get(sid, [])
            pred_b = pred_b_per.get(sid, [])
            gold = gold_a if gold_a else gold_b
        else:
            gold = gold_a

        if not gold: continue

        align_a = needleman_wunsch(gold, pred_a)
        g_a, p_a = render_aligned_pair(align_a)
        rows.append(f"<tr class='sentheader'><td colspan='2'>Sentence {sid}</td></tr>")
        rows.append(f"<tr><td>{label_a} gold</td><td>{g_a}</td></tr>")
        rows.append(f"<tr><td>{label_a} pred</td><td>{p_a}</td></tr>")
        if out_b is not None:
            align_b = needleman_wunsch(gold, pred_b)
            g_b, p_b = render_aligned_pair(align_b)
            rows.append(f"<tr><td>{label_b} gold</td><td>{g_b}</td></tr>")
            rows.append(f"<tr><td>{label_b} pred</td><td>{p_b}</td></tr>")
    rows.append("</table>")
    return ''.join(rows)

def stats_nw(out, label):
    """Quick one-line summary of NW match/sub/del/ins counts and rates."""
    gold_per, pred_per = gather_sequences(out)
    total_align = []
    for sid in set(gold_per) | set(pred_per):
        total_align.extend(needleman_wunsch(gold_per.get(sid, []),
                                            pred_per.get(sid, [])))
    n_match = sum(1 for g, p in total_align if g is not None and p is not None and g == p)
    n_sub   = sum(1 for g, p in total_align if g is not None and p is not None and g != p)
    n_del   = sum(1 for g, p in total_align if g is not None and p is None)
    n_ins   = sum(1 for g, p in total_align if g is None and p is not None)
    n_gold  = sum(1 for g, p in total_align if g is not None)
    n_pred  = sum(1 for g, p in total_align if p is not None)
    print(f"\n  {label}:  gold={n_gold}  pred={n_pred}")
    print(f"    match={n_match} ({100 * n_match / max(n_gold, 1):.1f}%)  "
          f"sub={n_sub} ({100 * n_sub / max(n_gold, 1):.1f}%)  "
          f"del={n_del} ({100 * n_del / max(n_gold, 1):.1f}%)  "
          f"ins={n_ins}")

# ============================================================
# 12. Example usage / cohort sweep
# ============================================================
# Single patient — original LDA pipeline
# ---------------------------------------
# out, err = run_for_patient('P22', smoothing_hz=10.0)
# if out is None:
#     print(f"P22 skipped: {err}")
# else:
#     print(f"P22: PER={100*out['per']:.1f}%  n_pred={out['n_pred']}/{out['n_test']}  "
#           f"bonus={out['bonus']:.2f}")
#     m = nw_metrics(out, manner_fn=manner_of)
#     print_nw_metrics(m, label="P22 (LDA)")

# Single patient — LDA + multiband features + speech gating
# ---------------------------------------------------------
# out, err = run_for_patient_sd('P22', feature_spec={'hg_amp': True},
#                                use_speech_gate=True)

# Cohort sweep
# -------------
# results = {}
# for pid in TARGET_PIDS:
#     t0 = time.time()
#     out, err = run_for_patient(pid)
#     if out is None:
#         print(f"  {pid}: SKIP — {err}", flush=True)
#         continue
#     m = nw_metrics(out, manner_fn=manner_of)
#     results[pid] = (out, m)
#     print(f"  {pid}: PER={100*out['per']:5.1f}%  match={100*m['match_rate']:5.1f}%  "
#           f"z={m['z_match']:+5.2f}  bonus={out['bonus']:.2f}  "
#           f"({time.time()-t0:.0f}s)", flush=True)

# Cohort summary
# ---------------
# print(f"\n{'pid':<5} {'PER':>7} {'match':>7} {'z':>6} {'n_gold':>7}")
# for pid, (out, m) in results.items():
#     print(f"{pid:<5} {100*out['per']:>6.1f}% {100*m['match_rate']:>6.1f}% "
#           f"{m['z_match']:>+6.2f} {m['n_gold']:>7}")
