# Converted from Untitled6.ipynb

# ============================================================
# Dutch phoneme → MANNER / PLACE tables (paste before Cell 1)
# ============================================================

MANNER = {
    # ---- Vowels (manner=0) ----
    'ɪ': 0, 'ɛ': 0, 'ɑ': 0, 'ɔ': 0, 'ʏ': 0, 'ə': 0, 'a': 0,
    'i':  0, 'iː': 0, 'eː': 0, 'aː': 0, 'oː': 0,
    'uː': 0, 'yː': 0, 'øː': 0, 'u': 0, 'y': 0, 'e': 0, 'o': 0,
    'ɔ̈': 0, 'ɛ̈': 0,
    'ɛi': 0, 'ɛj': 0, 'œy': 0, 'ɔu': 0, 'ɑu': 0, 'ɑi': 0, 'au': 0,
    'ui': 0, 'oi': 0, 'œ': 0,
    # ---- Stops / plosives (manner=1) ----
    'p': 1, 'b': 1, 't': 1, 'd': 1, 'k': 1, 'g': 1, 'ɡ': 1,
    'ʔ': 1, 'c': 1,
    # ---- Fricatives (manner=2) ----
    'f': 2, 'v': 2, 's': 2, 'z': 2, 'x': 2, 'ɣ': 2, 'h': 2,
    'ʃ': 2, 'ʒ': 2, 'ç': 2, 'χ': 2,
    # ---- Nasals (manner=3) ----
    'm': 3, 'n': 3, 'ŋ': 3, 'ɲ': 3,
    # ---- Approximants / liquids (manner=4) ----
    'l': 4, 'r': 4, 'ʁ': 4, 'ɹ': 4, 'j': 4, 'ʋ': 4, 'w': 4, 'ɥ': 4,
    # ---- Affricates (treated as stops) ----
    'ts': 1, 'tʃ': 1, 'dʒ': 1,
}

PLACE = {
    # ---- All vowels = 0 ----
    'ɪ': 0, 'ɛ': 0, 'ɑ': 0, 'ɔ': 0, 'ʏ': 0, 'ə': 0, 'a': 0,
    'i':  0, 'iː': 0, 'eː': 0, 'aː': 0, 'oː': 0,
    'uː': 0, 'yː': 0, 'øː': 0, 'u': 0, 'y': 0, 'e': 0, 'o': 0,
    'ɛi': 0, 'ɛj': 0, 'œy': 0, 'ɔu': 0, 'ɑu': 0, 'ɑi': 0, 'au': 0,
    'ui': 0, 'oi': 0, 'œ': 0, 'ɔ̈': 0, 'ɛ̈': 0,
    # ---- Bilabial (1) ----
    'p': 1, 'b': 1, 'm': 1,
    # ---- Labiodental (2) ----
    'f': 2, 'v': 2, 'ʋ': 2, 'w': 2,
    # ---- Alveolar (3) ----
    't': 3, 'd': 3, 's': 3, 'z': 3, 'n': 3, 'l': 3,
    'r': 3, 'ɹ': 3, 'ts': 3,
    # ---- Postalveolar (4) ----
    'ʃ': 4, 'ʒ': 4, 'tʃ': 4, 'dʒ': 4,
    # ---- Velar (5) ----
    'k': 5, 'g': 5, 'ɡ': 5, 'x': 5, 'ɣ': 5, 'ŋ': 5,
    'χ': 5, 'ʁ': 5,
    # ---- Glottal (6) ----
    'h': 6, 'ʔ': 6,
    # ---- Palatal (7) ----
    'j': 7, 'ɲ': 7, 'ç': 7, 'c': 7, 'ɥ': 7,
}

PLACE_TABLE = PLACE   # alias used by some data builders

print(f"MANNER: {len(MANNER)} phonemes, {len(set(MANNER.values()))} classes")
print(f"PLACE:  {len(PLACE)} phonemes, {len(set(PLACE.values()))} classes")

# ============================================================
# Signal-processing constants + filter design
# ============================================================
import numpy as np
import torch
from scipy.signal import butter, sosfiltfilt, iirnotch, tf2sos, hilbert

# Core constants
EEG_SR     = 1024
HG_LOW     = 70
HG_HIGH    = 170
NOTCH_HZ   = [50, 150]
LP_CUT_HZ  = 12.0          # Butterworth lowpass on envelope
N_BUTTER   = 4
WIN_MS     = 30
SHIFT_MS   = 5
WIN_SAMP   = int(EEG_SR * WIN_MS / 1000)
SHIFT_SAMP = int(EEG_SR * SHIFT_MS / 1000)
FRAME_HZ   = int(1000 / SHIFT_MS)
MIN_SENT_FRAMES = 30

# Mixup augmentation defaults (used by training)
P_AUG          = 0.5
AUG_FRAC       = 0.2
MIX_RATIO      = 0.8
MIN_PHON_LEN_FOR_AUG = 6

# Training defaults
LR             = 3e-4
WEIGHT_DECAY   = 1e-3
LAM_MANNER     = 0.3
LAM_PLACE      = 0.1
LAM_BIO_CE     = 0.5
GRAD_CLIP      = 5.0

# Device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"DEVICE = {DEVICE}")
if DEVICE == 'cuda':
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

# Filter design
def _design_filters():
    sos_bp = butter(N_BUTTER, [HG_LOW, HG_HIGH], btype='bandpass',
                    fs=EEG_SR, output='sos')
    sos_lp = butter(N_BUTTER, LP_CUT_HZ, btype='lowpass',
                    fs=EEG_SR, output='sos')
    sos_notches = []
    for f0 in NOTCH_HZ:
        b, a = iirnotch(f0, 30, EEG_SR)
        sos_notches.append(tf2sos(b, a))
    return sos_bp, sos_lp, sos_notches

_SOS_BP, _SOS_LP, _SOS_NOTCH = _design_filters()

# Core feature extraction
def extract_hg_frames(eeg_slice):
    """Raw EEG → 200 Hz log-amplitude HG envelope.
       Butterworth-LP smoothing + decimation."""
    x = eeg_slice.astype(np.float64)
    for sos in _SOS_NOTCH:
        x = sosfiltfilt(sos, x, axis=0)
    x = sosfiltfilt(_SOS_BP, x, axis=0)
    env = np.abs(hilbert(x, axis=0))
    env = sosfiltfilt(_SOS_LP, env, axis=0)
    env = np.maximum(env, 0)
    out = env[::SHIFT_SAMP].astype(np.float32)
    return np.log1p(out)

def stack_context(X, K=5):
    T, C = X.shape
    pad = np.zeros((K, C), dtype=X.dtype)
    Xp = np.vstack([pad, X, pad])
    cols = [Xp[k:k + T] for k in range(2 * K + 1)]
    return np.concatenate(cols, axis=1)

print(f"Signal processing ready: LP_CUT_HZ={LP_CUT_HZ}, N_BUTTER={N_BUTTER}")

required = ['MANNER', 'PLACE', 'PLACE_TABLE', 'EEG_SR', 'N_BUTTER',
            'LP_CUT_HZ', 'DEVICE', 'MIN_SENT_FRAMES',
            'build_frame_dataset_step1_car', 'train_v2_aggressive',
            'score_result', 'collapse_bio_to_segments',
            'longest_run_with_shift', 'collect_matches',
            'surprise_score', 'perm_null', 'extract_hg_frames',
            'stack_context', 'build_speech_only_labels', 'get_manner_table',
            'make_model_v2', 'LinearChainCRF', 'BiLSTM_BIO_CRF',
            'build_tag_index_noO', 'build_transition_mask_noO',
            'init_transition_matrix_noO',
            'split_into_sentences_v2', 'fit_mu_sd', 'standardize',
            'to_device', 'make_ce_weights', 'augment_sentence',
            'evaluate_for_selection']
missing = [n for n in required if n not in dir()]
if missing:
    print("MISSING:", missing)
else:
    print("All required names are in scope.")

# Cell 1: Load P21–P30 + build per-patient inventory + bigram + manner table
# ============================================================
from dutch_30_pipeline import Dutch30Pipeline
from dutch_30_feature_extractor import Dutch30FeatureExtractor
from dataset_config import Dutch30Config
from run_pipeline import load_mfa_alignments
from collections import Counter
import numpy as np

ALL_PIDS = [f'P{i:02d}' for i in range(21, 31)]

# Build pipeline only if not already loaded with all 10 patients
need_rebuild = (
    'pipeline' not in dir()
    or not all(pid in pipeline.split_result['word_segments_dict'] for pid in ALL_PIDS)
)

if need_rebuild:
    print("Building pipeline for P21–P30 ...")
    config    = Dutch30Config()
    extractor = Dutch30FeatureExtractor(config=config)
    pipeline  = Dutch30Pipeline(extractor, config=config, use_wav2vec=False)
    pipeline.step1_load_dutch30_data(patient_range=(21, 30))
    pipeline.step2_split_by_instances(train_fraction=0.8)
    pipeline.step3_load_channel_exclusions('channel_exclusions.json')
    pipeline.apply_channel_exclusions()
else:
    print("Using existing pipeline (all 10 patients already loaded).")


def build_saved_state(pid, pipeline):
    """Build cls_to_i / bg_lp / phone_to_manner from train MFA only."""
    wd = pipeline.split_result['word_segments_dict'][pid]
    mfa = load_mfa_alignments(pid)
    train_sent_ids = set()
    if 'train' in pipeline.split_result:
        for inst in pipeline.split_result['train'].get(pid, []):
            if isinstance(inst, dict) and 'sentence_idx' in inst:
                train_sent_ids.add(inst['sentence_idx'])
    if not train_sent_ids:
        all_real = [i for i, s in enumerate(wd['sentence_list'])
                    if isinstance(s, dict) and s.get('text')]
        train_sent_ids = set(i for i in all_real if i not in all_real[::6])

    phons = []
    for sidx in train_sent_ids:
        if sidx in mfa:
            phons.extend(p['phone'] for p in mfa[sidx])
    inv = sorted(set(phons))
    cls_to_i = {ph: i for i, ph in enumerate(inv)}

    n = len(inv)
    bg = np.ones((n, n), dtype=np.float32)
    for sidx in train_sent_ids:
        if sidx not in mfa: continue
        seq = [cls_to_i[p['phone']] for p in mfa[sidx] if p['phone'] in cls_to_i]
        for a, b in zip(seq[:-1], seq[1:]):
            bg[a, b] += 1
    bg_lp = np.log(bg / bg.sum(axis=1, keepdims=True))
    pm_arr = np.array([MANNER.get(ph, 0) for ph in inv], dtype=int)
    return {'cls_to_i': cls_to_i, 'bg_lp': bg_lp, 'phone_to_manner': pm_arr}

saved_models = {pid: build_saved_state(pid, pipeline) for pid in ALL_PIDS}
print(f"Built saved_models for {len(saved_models)} patients")

# Coverage check (MANNER and PLACE need to cover all phonemes in the data)
missing_m = set(); missing_p = set()
for st in saved_models.values():
    for ph in st['cls_to_i'].keys():
        if ph not in MANNER: missing_m.add(ph)
        if ph not in PLACE:  missing_p.add(ph)
if missing_m: print(f"MISSING from MANNER: {sorted(missing_m)}")
if missing_p: print(f"MISSING from PLACE:  {sorted(missing_p)}")
if not missing_m and not missing_p: print("All phonemes covered.")

# ALL-IN-ONE bundle: signal processing + model + training + scoring
# Paste once, then run Cell 2 (training)
# ============================================================
import os, time, torch, numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict, Counter
from scipy.signal import butter, sosfiltfilt, iirnotch, tf2sos, hilbert
from config import DUTCH_30_PATH
from run_pipeline import load_mfa_alignments

# ---------- Constants ----------
EEG_SR     = 1024
HG_LOW     = 70
HG_HIGH    = 170
NOTCH_HZ   = [50, 150]
LP_CUT_HZ  = 12.0
N_BUTTER   = 4
WIN_MS     = 30
SHIFT_MS   = 5
WIN_SAMP   = int(EEG_SR * WIN_MS / 1000)
SHIFT_SAMP = int(EEG_SR * SHIFT_MS / 1000)
FRAME_HZ   = int(1000 / SHIFT_MS)
MIN_SENT_FRAMES = 30

LR             = 3e-4
WEIGHT_DECAY   = 1e-3
LAM_MANNER     = 0.3
LAM_PLACE      = 0.1
LAM_BIO_CE     = 0.5
GRAD_CLIP      = 5.0
P_AUG          = 0.5
AUG_FRAC       = 0.2
MIX_RATIO      = 0.8
MIN_PHON_LEN_FOR_AUG = 6

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"DEVICE = {DEVICE}")
PLACE_TABLE = PLACE  # alias (assumes PLACE already defined)

# ---------- Filters ----------
def _design_filters():
    sos_bp = butter(N_BUTTER, [HG_LOW, HG_HIGH], btype='bandpass',
                    fs=EEG_SR, output='sos')
    sos_lp = butter(N_BUTTER, LP_CUT_HZ, btype='lowpass',
                    fs=EEG_SR, output='sos')
    sos_notches = []
    for f0 in NOTCH_HZ:
        b, a = iirnotch(f0, 30, EEG_SR)
        sos_notches.append(tf2sos(b, a))
    return sos_bp, sos_lp, sos_notches

_SOS_BP, _SOS_LP, _SOS_NOTCH = _design_filters()

def extract_hg_frames(eeg_slice):
    x = eeg_slice.astype(np.float64)
    for sos in _SOS_NOTCH:
        x = sosfiltfilt(sos, x, axis=0)
    x = sosfiltfilt(_SOS_BP, x, axis=0)
    env = np.abs(hilbert(x, axis=0))
    env = sosfiltfilt(_SOS_LP, env, axis=0)
    env = np.maximum(env, 0)
    out = env[::SHIFT_SAMP].astype(np.float32)
    return np.log1p(out)

def stack_context(X, K=5):
    T, C = X.shape
    pad = np.zeros((K, C), dtype=X.dtype)
    Xp = np.vstack([pad, X, pad])
    cols = [Xp[k:k + T] for k in range(2 * K + 1)]
    return np.concatenate(cols, axis=1)

def build_speech_only_labels(mfa_phones, n_frames):
    bio = [None] * n_frames; phon = [None] * n_frames
    keep = np.zeros(n_frames, dtype=bool)
    for ph in mfa_phones:
        s_s, e_s = ph['start_s'], ph['end_s']; sym = ph['phone']
        k_start = int(np.ceil((s_s * EEG_SR - WIN_SAMP / 2) / SHIFT_SAMP))
        k_end   = int(np.floor((e_s * EEG_SR - WIN_SAMP / 2) / SHIFT_SAMP))
        k_start = max(0, k_start); k_end = min(n_frames - 1, k_end)
        if k_end < k_start: continue
        bio[k_start] = f'B-{sym}'; phon[k_start] = sym; keep[k_start] = True
        for k in range(k_start + 1, k_end + 1):
            bio[k] = f'I-{sym}'; phon[k] = sym; keep[k] = True
    return bio, phon, keep

def get_manner_table(saved_models, pid):
    state = saved_models[pid]; cls_to_i = state['cls_to_i']
    pm_arr = np.asarray(state['phone_to_manner']).astype(int)
    i_to_cls = {v: k for k, v in cls_to_i.items()}
    return {i_to_cls[i]: int(pm_arr[i]) for i in range(len(pm_arr))}

# ---------- Data builder ----------
def build_frame_dataset_step1_car(pid, pipeline, saved_models, K=10,
                                    channel_mask=None, apply_car=True):
    raw_eeg = np.load(os.path.join(DUTCH_30_PATH, 'raw', f'{pid}_sEEG.npy'))
    if raw_eeg.ndim == 2 and raw_eeg.shape[0] < raw_eeg.shape[1]:
        raw_eeg = raw_eeg.T
    if channel_mask is not None:
        raw_eeg = raw_eeg[:, channel_mask]
    if apply_car:
        raw_eeg = raw_eeg - raw_eeg.mean(axis=1, keepdims=True)

    wd = pipeline.split_result['word_segments_dict'][pid]
    mfa = load_mfa_alignments(pid)
    all_real = [i for i, s in enumerate(wd['sentence_list'])
                if isinstance(s, dict) and s.get('text')]
    test_sent_ids = set(all_real[::6])
    manner_map = get_manner_table(saved_models, pid)

    out = {'train': defaultdict(list), 'test': defaultdict(list)}
    n_used = n_skip = 0
    for sent_idx in all_real:
        if sent_idx not in mfa or not mfa[sent_idx]: n_skip += 1; continue
        s = wd['sentence_list'][sent_idx]
        s0, s1 = s['stim_start_idx'], s['stim_end_idx']
        if s1 > raw_eeg.shape[0]: n_skip += 1; continue
        X = extract_hg_frames(raw_eeg[s0:s1])
        if X.shape[0] < 11: n_skip += 1; continue
        bio, phon, keep = build_speech_only_labels(mfa[sent_idx], X.shape[0])
        if keep.sum() < 11: n_skip += 1; continue
        Xs = stack_context(X, K=K)
        bio_arr = np.array([b for b, k in zip(bio, keep) if k])
        phon_arr = np.array([p for p, k in zip(phon, keep) if k])
        manner_arr = np.array([manner_map.get(p, -1) for p in phon_arr], dtype=np.int64)
        place_arr  = np.array([PLACE_TABLE.get(p, -1) for p in phon_arr], dtype=np.int64)
        sidx_arr = np.full(int(keep.sum()), sent_idx, dtype=int)
        split = 'test' if sent_idx in test_sent_ids else 'train'
        out[split]['X'].append(Xs[keep])
        out[split]['bio'].append(bio_arr)
        out[split]['phon'].append(phon_arr)
        out[split]['manner'].append(manner_arr)
        out[split]['place'].append(place_arr)
        out[split]['sent_idx'].append(sidx_arr)
        n_used += 1
    result = {}
    for split in ('train', 'test'):
        if not out[split]['X']:
            result[split] = None; continue
        result[split] = {k: (np.concatenate(v, axis=0) if isinstance(v[0], np.ndarray)
                             else v) for k, v in out[split].items()}
    print(f"  [{pid}] CAR={apply_car} K={K}  used={n_used}  "
          f"train_fr={result['train']['X'].shape[0] if result['train'] else 0}  "
          f"test_fr={result['test']['X'].shape[0] if result['test'] else 0}")
    return result

# ---------- CRF + Model ----------
def build_tag_index_noO(cls_to_i):
    tag_to_idx = {}; idx_to_tag = []
    for ph, c in sorted(cls_to_i.items(), key=lambda kv: kv[1]):
        tag_to_idx[f'B-{ph}'] = len(idx_to_tag); idx_to_tag.append(f'B-{ph}')
        tag_to_idx[f'I-{ph}'] = len(idx_to_tag); idx_to_tag.append(f'I-{ph}')
    return tag_to_idx, idx_to_tag, len(idx_to_tag)

def build_transition_mask_noO(idx_to_tag):
    n = len(idx_to_tag); mask = torch.zeros(n, n)
    info = [('B', t[2:]) if t.startswith('B-') else ('I', t[2:]) for t in idx_to_tag]
    for j, (kj, pj) in enumerate(info):
        if kj == 'I':
            for i, (ki, pi) in enumerate(info):
                if not ((ki == 'B' and pi == pj) or (ki == 'I' and pi == pj)):
                    mask[i, j] = float('-inf')
    return mask

def init_transition_matrix_noO(idx_to_tag, bg_lp, cls_to_i):
    n = len(idx_to_tag); T = torch.zeros(n, n)
    if bg_lp is None: return T
    bg = torch.from_numpy(bg_lp).float()
    for _, ca in cls_to_i.items():
        ba = 2 * ca
        for _, cb in cls_to_i.items():
            T[ba, 2 * cb] = bg[ca, cb] * 0.5
    return T

class LinearChainCRF(nn.Module):
    def __init__(self, n_tags, transition_mask, transition_init):
        super().__init__(); self.n_tags = n_tags
        self.trans = nn.Parameter(transition_init.clone())
        self.start = nn.Parameter(torch.zeros(n_tags))
        self.end   = nn.Parameter(torch.zeros(n_tags))
        self.register_buffer('mask', transition_mask)
    def _T(self): return self.trans + self.mask
    def _forward_alg(self, emissions):
        alpha = self.start + emissions[0]
        for t in range(1, emissions.size(0)):
            alpha = torch.logsumexp(alpha[:, None] + self._T(), dim=0) + emissions[t]
        return torch.logsumexp(alpha + self.end, dim=0)
    def _score(self, emissions, tags):
        s = self.start[tags[0]] + emissions[0, tags[0]]
        for t in range(1, emissions.size(0)):
            s = s + self._T()[tags[t-1], tags[t]] + emissions[t, tags[t]]
        return s + self.end[tags[-1]]
    def neg_log_likelihood(self, emissions, tags):
        return self._forward_alg(emissions) - self._score(emissions, tags)
    def viterbi(self, emissions):
        T_, K = emissions.shape
        bp = torch.zeros(T_, K, dtype=torch.long, device=emissions.device)
        v = self.start + emissions[0]
        for t in range(1, T_):
            scores = v.unsqueeze(1) + self._T()
            v, bp[t] = scores.max(dim=0); v = v + emissions[t]
        last = int((v + self.end).argmax().item()); path = [last]
        for t in range(T_ - 1, 0, -1):
            last = int(bp[t, last].item()); path.append(last)
        return list(reversed(path))

class BiLSTM_BIO_CRF(nn.Module):
    def __init__(self, n_in, n_phon, n_manner=5, n_place=8,
                 lstm_hidden=128, lstm_layers=2,
                 lstm_dropout=0.3, head_dropout=0.3,
                 transition_mask=None, transition_init=None, n_tags=None):
        super().__init__(); self.n_tags = n_tags
        self.proj = nn.Sequential(
            nn.Linear(n_in, lstm_hidden * 2), nn.GELU(), nn.Dropout(head_dropout))
        self.lstm = nn.LSTM(lstm_hidden * 2, lstm_hidden,
                            num_layers=lstm_layers,
                            dropout=lstm_dropout if lstm_layers > 1 else 0.0,
                            bidirectional=True, batch_first=False)
        self.drop = nn.Dropout(head_dropout)
        self.bio_head     = nn.Linear(lstm_hidden * 2, n_tags)
        self.bio_aux_head = nn.Linear(lstm_hidden * 2, n_tags)
        self.manner_head  = nn.Linear(lstm_hidden * 2, n_manner)
        self.place_head   = nn.Linear(lstm_hidden * 2, n_place)
        self.crf = LinearChainCRF(n_tags, transition_mask, transition_init)
    def encode(self, x):
        h, _ = self.lstm(self.proj(x).unsqueeze(1))
        return self.drop(h.squeeze(1))
    def loss(self, x, tags, manner, place,
             lam_manner=0.3, lam_place=0.1, lam_bio_ce=0.5,
             b_boost=0.0, ce_weights=None):
        h = self.encode(x)
        bio_em = self.bio_head(h)
        bio_ce = F.cross_entropy(self.bio_aux_head(h), tags, weight=ce_weights)
        crf_nll = self.crf.neg_log_likelihood(bio_em, tags) / x.size(0)
        valid_m = manner >= 0
        mn_loss = (F.cross_entropy(self.manner_head(h)[valid_m], manner[valid_m])
                   if valid_m.sum() > 0 else torch.tensor(0., device=x.device))
        valid_p = place >= 0
        pl_loss = (F.cross_entropy(self.place_head(h)[valid_p], place[valid_p])
                   if valid_p.sum() > 0 else torch.tensor(0., device=x.device))
        total = crf_nll + lam_bio_ce*bio_ce + lam_manner*mn_loss + lam_place*pl_loss
        return total, {'crf': float(crf_nll.item()), 'bio_ce': float(bio_ce.item()),
                       'mn': float(mn_loss.item()), 'pl': float(pl_loss.item())}
    @torch.no_grad()
    def decode(self, x):
        return self.crf.viterbi(self.bio_head(self.encode(x)))

def make_model_v2(pid, frame_datasets_v2, saved_models, **arch):
    fd = frame_datasets_v2[pid]; state = saved_models[pid]
    cls_to_i = state['cls_to_i']; bg_lp = state['bg_lp']
    pm_arr = np.asarray(state['phone_to_manner']).astype(int)
    n_in = fd['train']['X'].shape[1]; n_phon = len(cls_to_i)
    n_manner = int(pm_arr.max()) + 1
    tag_to_idx, idx_to_tag, n_tags = build_tag_index_noO(cls_to_i)
    trans_mask = build_transition_mask_noO(idx_to_tag)
    trans_init = init_transition_matrix_noO(idx_to_tag, bg_lp, cls_to_i)
    model = BiLSTM_BIO_CRF(
        n_in=n_in, n_phon=n_phon, n_manner=n_manner, n_place=8,
        transition_mask=trans_mask, transition_init=trans_init, n_tags=n_tags,
        **arch)
    return model, tag_to_idx, idx_to_tag

# ---------- Training helpers ----------
def split_into_sentences_v2(split_dict, tag_to_idx):
    sents = []
    sidx = split_dict['sent_idx']
    boundaries = np.where(np.diff(sidx, prepend=sidx[0]-1) != 0)[0].tolist() + [len(sidx)]
    n_oov = 0
    for k in range(len(boundaries) - 1):
        s, e = boundaries[k], boundaries[k + 1]
        if e - s < MIN_SENT_FRAMES: continue
        bio_str_full = split_dict['bio'][s:e]
        keep_local = np.array([t in tag_to_idx for t in bio_str_full])
        n_oov += int((~keep_local).sum())
        if keep_local.sum() < MIN_SENT_FRAMES: continue
        bio_str = bio_str_full[keep_local]
        tags = np.array([tag_to_idx[t] for t in bio_str], dtype=np.int64)
        mid_mask = np.zeros(len(tags), dtype=bool)
        i = 0
        while i < len(bio_str):
            if bio_str[i].startswith('B-'):
                ph = bio_str[i][2:]; j = i + 1
                while j < len(bio_str) and bio_str[j] == f'I-{ph}': j += 1
                if (j - i) >= MIN_PHON_LEN_FOR_AUG:
                    mid_mask[i + 2: j - 2] = True
                i = j
            else: i += 1
        sents.append({
            'X':      torch.from_numpy(split_dict['X'][s:e][keep_local]).float(),
            'tags':   torch.from_numpy(tags),
            'manner': torch.from_numpy(split_dict['manner'][s:e][keep_local]),
            'place':  torch.from_numpy(split_dict['place'][s:e][keep_local]),
            'mid':    torch.from_numpy(mid_mask),
            'sent_idx': int(sidx[s]),
        })
    if n_oov: print(f"  dropped {n_oov} OOV-tag frames")
    return sents

def fit_mu_sd(sents):
    Xall = torch.cat([s['X'].cpu() for s in sents], dim=0).numpy()
    return Xall.mean(0), Xall.std(0)

def standardize(sents, mu, sd):
    sd_safe = np.where(sd < 1e-6, 1.0, sd)
    for s in sents:
        device = s['X'].device
        mu_t = torch.from_numpy(mu).float().to(device)
        sd_t = torch.from_numpy(sd_safe).float().to(device)
        s['X'] = (s['X'] - mu_t) / sd_t

def to_device(sents, device):
    for s in sents:
        for k in ('X', 'tags', 'manner', 'place', 'mid'):
            s[k] = s[k].to(device)

def make_ce_weights(sents_tr, n_tags, device):
    cnt = torch.zeros(n_tags)
    for s in sents_tr:
        for t in s['tags'].cpu().tolist(): cnt[t] += 1
    cnt = cnt.clamp(min=1.0)
    return (cnt.sum() / (n_tags * cnt)).clamp(min=0.2, max=5.0).to(device)

def augment_sentence(s_anchor, partner_pool, p_aug, aug_frac, mix_ratio, rng):
    if rng.random() > p_aug or s_anchor['mid'].sum() == 0:
        return s_anchor['X']
    mid_idx = s_anchor['mid'].nonzero(as_tuple=False).flatten()
    n_perturb = max(1, int(len(mid_idx) * aug_frac))
    perm = mid_idx[torch.randperm(len(mid_idx), device=mid_idx.device)[:n_perturb]]
    partner = partner_pool[rng.integers(0, len(partner_pool))]
    if partner['X'].size(0) == 0: return s_anchor['X']
    p_idx = torch.randint(0, partner['X'].size(0), (n_perturb,),
                          device=partner['X'].device)
    X = s_anchor['X'].clone()
    X[perm] = mix_ratio * X[perm] + (1.0 - mix_ratio) * partner['X'][p_idx]
    return X

def evaluate_for_selection(model, sents_te, idx_to_tag):
    model.eval()
    correct = total = 0; pred_symbols = set(); n_pred = 0
    with torch.no_grad():
        for s in sents_te:
            path = model.decode(s['X'])
            for p, t in zip(path, s['tags'].tolist()):
                correct += (p == t); total += 1
            for p in path:
                tag = idx_to_tag[int(p)]
                if tag.startswith('B-'):
                    pred_symbols.add(tag[2:]); n_pred += 1
    tag_acc = correct / max(total, 1)
    diversity = min(1.0, len(pred_symbols) / 3.0)
    coverage = min(1.0, n_pred / max(1, total / 34))
    return tag_acc * diversity * coverage, tag_acc, len(pred_symbols), n_pred


def train_v2_aggressive(pid, frame_datasets, saved_models, epochs=40,
                         patience=4, warmup_epochs=5, **arch):
    fd = frame_datasets[pid]
    if fd['train'] is None or fd['test'] is None: return None
    model, tag_to_idx, idx_to_tag = make_model_v2(
        pid, frame_datasets, saved_models, **arch)
    model = model.to(DEVICE)
    sents_tr = split_into_sentences_v2(fd['train'], tag_to_idx)
    sents_te = split_into_sentences_v2(fd['test'],  tag_to_idx)
    if not sents_tr: return None
    mu, sd = fit_mu_sd(sents_tr)
    standardize(sents_tr, mu, sd); standardize(sents_te, mu, sd)
    to_device(sents_tr, DEVICE); to_device(sents_te, DEVICE)
    ce_weights = make_ce_weights(sents_tr, model.n_tags, DEVICE)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    rng = np.random.default_rng(0)
    best_score, best_state, no_improve = 0.0, None, 0

    for ep in range(epochs):
        model.train()
        perm = rng.permutation(len(sents_tr))
        losses = {'total': 0, 'crf': 0, 'bio_ce': 0, 'mn': 0, 'pl': 0}
        for idx in perm:
            s = sents_tr[idx]
            X_aug = augment_sentence(s, sents_tr, P_AUG, AUG_FRAC, MIX_RATIO, rng)
            opt.zero_grad()
            loss, parts = model.loss(
                X_aug, s['tags'], s['manner'], s['place'],
                lam_manner=LAM_MANNER, lam_place=LAM_PLACE,
                lam_bio_ce=LAM_BIO_CE, ce_weights=ce_weights, b_boost = 2.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            losses['total'] += float(loss.item())
            for k in ('crf', 'bio_ce', 'mn', 'pl'): losses[k] += parts[k]
        sched.step()

        score, tag_acc, n_uniq, n_pred = evaluate_for_selection(
            model, sents_te, idx_to_tag)
        if score > best_score and ep >= warmup_epochs:
            best_score = score
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            no_improve = 0
        elif ep >= warmup_epochs:
            no_improve += 1
        if (ep + 1) % 5 == 0 or ep == 0:
            n = len(sents_tr)
            print(f"  [{pid}] ep{ep+1:3d} loss={losses['total']/n:.2f} "
                  f"crf={losses['crf']/n:.2f} bio_ce={losses['bio_ce']/n:.2f} "
                  f"score={score:.3f} tag={tag_acc:.3f} n_uniq={n_uniq} "
                  f"(best={best_score:.3f})")
        if no_improve >= patience:
            print(f"  [{pid}] early stop at ep {ep+1}")
            break

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return {'model': model, 'mu': mu, 'sd': sd,
            'tag_to_idx': tag_to_idx, 'idx_to_tag': idx_to_tag,
            'sents_tr': sents_tr, 'sents_te': sents_te,
            'best_acc': best_score}


# ---------- Scoring ----------
N_PERM = 1500
SHIFT_MAX = 3
MIN_MATCH = 3

def longest_run_with_shift(pred, gold, shift_max=SHIFT_MAX):
    best, best_span = 0, None
    P, G = len(pred), len(gold)
    if P == 0 or G == 0: return 0, None
    for i in range(P):
        for j in range(max(0, i-shift_max), min(G, i+shift_max+1)):
            k = 0
            while i+k < P and j+k < G and pred[i+k] == gold[j+k]: k += 1
            if k > best: best, best_span = k, (i, j, k)
    return best, best_span

def collect_matches(pred_sents, gold_sents, min_match=MIN_MATCH):
    matches = []
    for p, g in zip(pred_sents, gold_sents):
        L, span = longest_run_with_shift(p, g)
        if L >= min_match and span is not None:
            i, j, k = span; matches.append(tuple(p[i:i+k]))
    return matches

def surprise_score(matches, marginal_logp):
    fallback = -np.log(1e-6)
    return sum(-marginal_logp.get(ph, fallback) for m in matches for ph in m)

def perm_null(pred_sents, gold_sents, marginal_logp, n_perm=N_PERM, seed=0):
    rng = np.random.default_rng(seed); nulls = np.zeros(n_perm)
    for b in range(n_perm):
        shuf = []
        for p in pred_sents:
            if len(p) == 0: shuf.append(p); continue
            idx = rng.permutation(len(p))
            shuf.append([p[k] for k in idx])
        nulls[b] = surprise_score(collect_matches(shuf, gold_sents), marginal_logp)
    return nulls

def collapse_bio_to_segments(bio_tag_strs):
    phons, starts, ends = [], [], []
    i = 0
    while i < len(bio_tag_strs):
        t = bio_tag_strs[i]
        if t.startswith('B-') or t.startswith('I-'):
            sym = t[2:]; j = i + 1
            while j < len(bio_tag_strs) and bio_tag_strs[j] == f'I-{sym}': j += 1
            phons.append(sym); starts.append(i); ends.append(j); i = j
        else: i += 1
    return phons, starts, ends

def score_result(res):
    if res is None: return {'z': 0.0, 'max_run': 0, 'n_matches': 0}
    pred_sents, gold_sents = [], []
    model = res['model']; model.eval()
    idx_to_tag = res['idx_to_tag']
    with torch.no_grad():
        for s in res['sents_te']:
            path = model.decode(s['X'])
            pp, _, _ = collapse_bio_to_segments(
                [idx_to_tag[int(t)] for t in path])
            gp, _, _ = collapse_bio_to_segments(
                [idx_to_tag[int(t)] for t in s['tags'].tolist()])
            pred_sents.append(pp); gold_sents.append(gp)
    all_gold = [ph for s in gold_sents for ph in s]
    if not all_gold: return {'z': 0.0, 'max_run': 0, 'n_matches': 0}
    cnt = Counter(all_gold); N = sum(cnt.values())
    gold_lp = {k: np.log(v / N) for k, v in cnt.items()}
    runs = [longest_run_with_shift(p, g)[0]
            for p, g in zip(pred_sents, gold_sents)]
    max_run = max(runs) if runs else 0
    matches = collect_matches(pred_sents, gold_sents)
    obs = surprise_score(matches, gold_lp)
    nulls = perm_null(pred_sents, gold_sents, gold_lp, n_perm=1500)
    mu, sd = nulls.mean(), nulls.std() + 1e-9
    return {'z': float((obs - mu) / sd), 'max_run': max_run,
            'n_matches': len(matches)}


print("\nAll functions defined. Ready for Cell 2.")

# Cell 2: Single config (v2 + CAR + Hilbert LP=12 + K=10) on all 10 patients
# ============================================================
import time
from scipy.signal import butter

# Lock in the config
CONFIG = {
    'description': 'v2 + Hilbert + LP=12 + K=10 + CAR',
    'K': 10,
    'apply_car': True,
    'smoothing_hz': 12.0,
    'epochs': 40,
    'lstm_hidden': 128,
    'lstm_layers': 2,
    'lstm_dropout': 0.3,
    'head_dropout': 0.3,
    'patience': 4,
    'warmup_epochs': 5,
    'seed': 0,        # single seed for the deliverable; change if you want a different one
}

# Reset LP filter
_SOS_LP = butter(N_BUTTER, CONFIG['smoothing_hz'], btype='lowpass',
                 fs=EEG_SR, output='sos')
globals()['_SOS_LP'] = _SOS_LP

print(f"Config: {CONFIG['description']}")
print(f"Training {len(ALL_PIDS)} patients with seed={CONFIG['seed']}\n")

all_patient_results = {}
t_start = time.time()

for i, pid in enumerate(ALL_PIDS):
    print(f"\n=========== [{i+1}/{len(ALL_PIDS)}]  {pid} ===========")
    torch.manual_seed(CONFIG['seed']); np.random.seed(CONFIG['seed'])
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(CONFIG['seed'])

    t0 = time.time()
    try:
        ds = build_frame_dataset_step1_car(
            pid, pipeline, saved_models,
            K=CONFIG['K'], apply_car=CONFIG['apply_car'])
        if ds['train'] is None or ds['test'] is None:
            print(f"  [{pid}] no train/test data, skipping"); continue
        res = train_v2_aggressive(
            pid, {pid: ds}, saved_models,
            epochs=CONFIG['epochs'],
            patience=CONFIG['patience'],
            warmup_epochs=CONFIG['warmup_epochs'],
            lstm_hidden=CONFIG['lstm_hidden'],
            lstm_layers=CONFIG['lstm_layers'],
            lstm_dropout=CONFIG['lstm_dropout'],
            head_dropout=CONFIG['head_dropout'])
        if res is None: continue
        sc = score_result(res)
        sc['res'] = res
        sc['runtime'] = time.time() - t0
        all_patient_results[pid] = sc
        print(f"  [{pid}] z={sc['z']:+.2f}  max_run={sc['max_run']}  "
              f"matches={sc['n_matches']}  ({sc['runtime']:.0f}s)")
    except Exception as e:
        print(f"  [{pid}] FAILED: {type(e).__name__}: {e}")

# Summary table
print(f"\n\n========== ALL PATIENTS SUMMARY ({(time.time()-t_start)/60:.0f} min) ==========\n")
print(f"{'patient':>8}  {'z':>7}  {'max_run':>8}  {'matches':>8}  {'runtime':>9}")
print("-" * 50)
for pid in ALL_PIDS:
    if pid not in all_patient_results:
        print(f"{pid:>8}  {'—':>7}  {'—':>8}  {'—':>8}  {'—':>9}")
        continue
    r = all_patient_results[pid]
    print(f"{pid:>8}  {r['z']:+7.2f}  {r['max_run']:>8d}  {r['n_matches']:>8d}  "
          f"{r['runtime']:>7.0f}s")

if all_patient_results:
    zs = [r['z'] for r in all_patient_results.values()]
    print(f"\nMean z across patients: {np.mean(zs):+.2f} ± {np.std(zs):.2f}")
    print(f"Patients with z > +2: "
          f"{sum(1 for z in zs if z > 2)}/{len(zs)}")
    print(f"Patients with z > +3: "
          f"{sum(1 for z in zs if z > 3)}/{len(zs)}")

# Sanity check: how many phonemes did we predict per patient vs gold?
print("\nPrediction coverage per patient:")
for pid in ALL_PIDS:
    if pid not in all_patient_results: continue
    pr = pipeline.patient_results.get(pid)
    if pr is None: continue
    print(f"  {pid}: gold={pr['n_test']}  pred={pr['n_pred']}  "
          f"coverage={pr['n_pred']/max(pr['n_test'],1):.1%}")

# ============================================================
# Cell 3: Adapt trained results into pipeline.patient_results format
# ============================================================
FRAME_SHIFT_S = 5e-3

if not hasattr(pipeline, 'patient_results') or pipeline.patient_results is None:
    pipeline.patient_results = {}

def build_patient_results(pid, res):
    model = res['model']; model.eval()
    idx_to_tag = res['idx_to_tag']
    sents_te = res['sents_te']

    true_labels, predictions = [], []
    true_sentence_ids, pred_sentence_ids = [], []
    true_segments, pred_segments = [], []

    with torch.no_grad():
        for s in sents_te:
            path = model.decode(s['X'])
            pred_tag_strs = [idx_to_tag[int(t)] for t in path]
            gold_tag_strs = [idx_to_tag[int(t)] for t in s['tags'].tolist()]
            pp, ps, pe = collapse_bio_to_segments(pred_tag_strs)
            gp, gs, ge = collapse_bio_to_segments(gold_tag_strs)
            sid = s['sent_idx']
            for ph, fa, fb in zip(pp, ps, pe):
                predictions.append(ph)
                pred_sentence_ids.append(sid)
                pred_segments.append((fa * FRAME_SHIFT_S, fb * FRAME_SHIFT_S))
            for ph, fa, fb in zip(gp, gs, ge):
                true_labels.append(ph)
                true_sentence_ids.append(sid)
                true_segments.append((fa * FRAME_SHIFT_S, fb * FRAME_SHIFT_S))

    from e2e_brain_decoder import edit_distance
    true_arr = np.array(true_labels); pred_arr = np.array(predictions)
    ed = edit_distance(list(true_arr), list(pred_arr))
    per = ed / max(len(true_arr), 1)
    acc = float((true_arr[:min(len(true_arr), len(pred_arr))]
                 == pred_arr[:min(len(true_arr), len(pred_arr))]).mean()) \
          if len(true_arr) and len(pred_arr) else 0.0
    return {
        'true_labels': true_arr, 'predictions': pred_arr,
        'true_sentence_ids': np.array(true_sentence_ids),
        'pred_sentence_ids': np.array(pred_sentence_ids),
        'true_segments': true_segments, 'pred_segments': pred_segments,
        'accuracy': acc, 'edit_distance': ed, 'per': per,
        'n_test': len(true_labels), 'n_pred': len(predictions),
    }

for pid, r in all_patient_results.items():
    pipeline.patient_results[pid] = build_patient_results(pid, r['res'])
    pr = pipeline.patient_results[pid]
    print(f"{pid}: gold={pr['n_test']}  pred={pr['n_pred']}  PER={pr['per']:.2%}")

# ============================================================
# Cell 4: Visualize all patients
# ============================================================
import importlib, e2e_brain_decoder
importlib.reload(e2e_brain_decoder)
from e2e_brain_decoder import show_matched_sequences_with_times

for pid in ALL_PIDS:
    if pid not in pipeline.patient_results: continue
    r = all_patient_results.get(pid)
    print(f"\n{'='*70}")
    z_str = f"z={r['z']:+.2f}  max_run={r['max_run']}  matches={r['n_matches']}" if r else "—"
    print(f"{pid}  ({CONFIG['description']})   {z_str}")
    print(f"{'='*70}")
    show_matched_sequences_with_times(
        pipeline, pid,
        max_per_line=45,
        time_align_tol_s=0.05,
        collapse_repeats=False,
    )
