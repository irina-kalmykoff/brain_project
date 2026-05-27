# Converted from Untitled8.ipynb

import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import os, numpy as np, warnings

from collections import defaultdict, Counter
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

import scipy
import random
import scipy.signal
from scipy.signal import iirfilter, sosfilt, sosfiltfilt
from scipy.signal import butter, sosfiltfilt, hilbert, iirfilter
from scipy.signal import stft

import numpy as np
from collections import defaultdict
from IPython.display import display, HTML

# build pipeline

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

SMOOTH_W   = 21
MIN_FRAMES = 4

def stk_frame_to_time_s(i):
    return ((i + LDA_MARGIN) * SHIFT_SAMP + WIN_SAMP / 2) / EEG_SR

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

class PhonemeDatasetMTMel(torch.utils.data.Dataset):
    def __init__(self, sents, phone_to_idx,
                 scaler=None, mel_scaler=None,
                 manner_map=None, voicing_map=None, place_map=None,   # ★ str->id maps
                 noise_std=0.0, augment=False,
                 n_time_masks=0, time_mask_max=0,
                 n_feat_masks=0, feat_mask_max_frac=0.0,
                 max_time_shift=0,
                 include_frame_labels=True):
        self.sents = sents
        self.p2i = phone_to_idx
        self.scaler = scaler
        self.mel_scaler = mel_scaler
        self.manner_map  = manner_map  or {}
        self.voicing_map = voicing_map or {}
        self.place_map   = place_map   or {}
        self.sil_id = phone_to_idx.get('sil', phone_to_idx.get('SIL', 0))
        self.noise_std = noise_std
        self.augment = augment
        self.n_time_masks = n_time_masks
        self.time_mask_max = time_mask_max
        self.n_feat_masks = n_feat_masks
        self.feat_mask_max_frac = feat_mask_max_frac
        self.max_time_shift = max_time_shift
        self.include_frame_labels = include_frame_labels

    def __len__(self):
        return len(self.sents)

    def _phone_to_aux(self, ph_str_per_frame_ids):
        """Given (T,) phoneme ids, return (T,) manner/voicing/place ids."""
        idx_to_ph = {v: k for k, v in self.p2i.items()}
        T = ph_str_per_frame_ids.shape[0]
        mn = np.zeros(T, dtype=np.int64)
        vo = np.zeros(T, dtype=np.int64)
        pl = np.zeros(T, dtype=np.int64)
        for t, pid in enumerate(ph_str_per_frame_ids):
            ph = idx_to_ph.get(int(pid), 'sil')
            mn[t] = self.manner_map.get(ph,  0)
            vo[t] = self.voicing_map.get(ph, 0)
            pl[t] = self.place_map.get(ph,   0)
        return mn, vo, pl

    def __getitem__(self, i):
        s = self.sents[i]
        x   = s['features'].astype(np.float32)     # (T, F)
        mel = s['mel'].astype(np.float32)          # (T, M)

        T = min(x.shape[0], mel.shape[0])
        x, mel = x[:T], mel[:T]

        if self.scaler is not None:
            x = self.scaler.transform(x).astype(np.float32)
        if self.mel_scaler is not None:
            mel = self.mel_scaler.transform(mel).astype(np.float32)

        # ----- per-frame phoneme ids from mfa_intervals -----
        ph = intervals_to_frames(s['mfa_intervals'], T, self.p2i, default_id=self.sil_id)
        mn, vo, pl = self._phone_to_aux(ph)

        # ----- augmentation -----
        if self.augment:
            if self.max_time_shift > 0:
                k = np.random.randint(-self.max_time_shift, self.max_time_shift + 1)
                if k != 0:
                    x = np.roll(x, k, axis=0)
            if self.noise_std > 0:
                x = x + np.random.randn(*x.shape).astype(np.float32) * self.noise_std
            for _ in range(self.n_time_masks):
                w = np.random.randint(0, self.time_mask_max + 1)
                if w > 0 and T - w > 0:
                    t0 = np.random.randint(0, T - w)
                    x[t0:t0+w, :] = 0.0
            F = x.shape[1]
            fmax = max(1, int(self.feat_mask_max_frac * F))
            for _ in range(self.n_feat_masks):
                w = np.random.randint(0, fmax + 1)
                if w > 0 and F - w > 0:
                    f0 = np.random.randint(0, F - w)
                    x[:, f0:f0+w] = 0.0

        # ----- CTC targets straight from s['target'] (list of phoneme strings) -----
        # drop SIL, collapse consecutive duplicates
        seq = []
        prev = None
        for p in s['target']:
            if p == 'sil' or p == 'SIL':
                prev = p; continue
            if p != prev:
                seq.append(self.p2i.get(p, self.sil_id))
            prev = p
        targets = np.asarray(seq, dtype=np.int64)

        return (torch.from_numpy(x),
                torch.from_numpy(targets),
                torch.from_numpy(ph),
                torch.from_numpy(mn),
                torch.from_numpy(vo),
                torch.from_numpy(pl),
                torch.from_numpy(mel))

#  Mel-spectrogram helpers (librosa-free)
# ---- mel scale ----
def _hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + hz / 700.0)

def _mel_to_hz(mel):
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

def mel_filterbank(sr, n_fft, n_mels, fmin=0.0, fmax=None):
    """Triangular mel filterbank: shape (n_mels, n_fft//2 + 1)."""
    fmax = fmax if fmax is not None else sr / 2.0
    mel_min, mel_max = _hz_to_mel(fmin), _hz_to_mel(fmax)
    mel_pts = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_pts  = _mel_to_hz(mel_pts)
    bin_pts = np.floor((n_fft + 1) * hz_pts / sr).astype(int)
    bin_pts = np.clip(bin_pts, 0, n_fft // 2)

    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(n_mels):
        l, c, r = bin_pts[m], bin_pts[m + 1], bin_pts[m + 2]
        if c > l:
            fb[m, l:c] = (np.arange(l, c) - l) / max(1, c - l)
        if r > c:
            fb[m, c:r] = (r - np.arange(c, r)) / max(1, r - c)
    return fb


# ---- log-mel spectrogram from a waveform segment ----
def compute_mel_spectrogram(audio_seg, sr, n_mels=40, n_fft=1024, hop=256):
    """
    audio_seg: 1-D float array
    Returns log-mel of shape (T_mel, n_mels).
    """
    audio_seg = np.asarray(audio_seg, dtype=np.float32)
    if audio_seg.size < n_fft:
        # too short: just pad
        audio_seg = np.pad(audio_seg, (0, n_fft - audio_seg.size))

    _, _, Zxx = stft(audio_seg, fs=sr,
                     nperseg=n_fft, noverlap=n_fft - hop,
                     boundary=None, padded=False)
    power = (np.abs(Zxx) ** 2).astype(np.float32)         # (n_fft/2 + 1, T_mel)
    fb    = mel_filterbank(sr, n_fft, n_mels)             # (n_mels, n_fft/2 + 1)
    mel   = fb @ power                                    # (n_mels, T_mel)
    log_mel = np.log(mel + 1e-6).astype(np.float32)
    return log_mel.T                                      # (T_mel, n_mels)


# ---- align mel to a target number of frames (T from iEEG features) ----
def build_mel_for_sentence(audio, sr, audio_start, audio_end,
                           n_mels=40, target_n_frames=None,
                           n_fft=1024, hop=256):
    """
    audio          : full waveform (1-D)
    audio_start/end: sample indices into `audio`
    target_n_frames: pass T_stacked so mel matches iEEG feature length

    Returns (target_n_frames, n_mels) float32, log-mel.
    """
    seg = audio[audio_start:audio_end]
    if seg.size == 0:
        return np.zeros((target_n_frames or 1, n_mels), dtype=np.float32)

    mel = compute_mel_spectrogram(seg, sr, n_mels=n_mels, n_fft=n_fft, hop=hop)
    T_src = mel.shape[0]
    T_dst = target_n_frames if target_n_frames is not None else T_src
    if T_src == T_dst or T_dst <= 0:
        return mel.astype(np.float32)

    if T_src < 2:
        return np.broadcast_to(mel[0:1], (T_dst, n_mels)).astype(np.float32).copy()

    # linear interpolation along time
    src_idx = np.linspace(0.0, T_src - 1, T_dst)
    lo      = np.floor(src_idx).astype(int)
    hi      = np.clip(lo + 1, 0, T_src - 1)
    frac    = (src_idx - lo).astype(np.float32)[:, None]
    out     = (1.0 - frac) * mel[lo] + frac * mel[hi]
    return out.astype(np.float32)

# smoke test
sr = 48000
# fake 5s audio: a 200 Hz tone
t = np.arange(int(5*sr)) / sr
audio = (0.3 * np.sin(2*np.pi*200*t)).astype(np.float32)

mel = build_mel_for_sentence(audio, sr, 0, len(audio),
                             n_mels=40, target_n_frames=999)
print('mel shape:', mel.shape)        # → (999, 40)
print('mel range:', mel.min(), mel.max())
print('finite?   ', np.isfinite(mel).all())

# Per-patient dataset builder with mel
def build_patient_dataset_with_mel(pid, feature_spec=None, test_offset=None, n_mels=40):
    """
    Returns splits = {'train': [...], 'val': [...], 'test': [...]}
    Each sentence is a dict with:
        features       (T, F)  float32  — stacked HG features
        mel            (T, M)  float32  — log-mel aligned to T
        target         list[str]        — phoneme strings (MFA order, no sil)
        mfa_intervals  list[(t0, t1, phone)]
        duration_sec   float            — (s1 - s0) / EEG_SR
        feature_fps    float            — T / duration_sec
        sent_idx       int
    """
    feature_spec = feature_spec or DEFAULT_FEATURE_SPEC
    test_offset  = TEST_OFFSET if test_offset is None else test_offset

    # 1) per-patient sentence metadata
    wd = pipeline.split_result['word_segments_dict'].get(pid)
    if wd is None:
        return None
    try:
        mfa = load_mfa_alignments(pid)
    except Exception:
        return None
    if not mfa:
        return None

    # 2) split ids: test = every 6th sentence (offset), val = subset of the rest
    all_real = [i for i, s in enumerate(wd['sentence_list'])
                if isinstance(s, dict) and s.get('text')]
    test_sent_ids      = set(all_real[test_offset::6])
    train_sent_ids_all = sorted(set(all_real) - test_sent_ids)
    val_step           = max(2, int(round(1 / VAL_FRAC)))
    val_sent_ids       = set(train_sent_ids_all[::val_step])

    # 3) raw EEG + raw audio for this patient
    raw_eeg = np.load(os.path.join(DUTCH_30_PATH, 'raw', f'{pid}_sEEG.npy'))
    if raw_eeg.ndim == 2 and raw_eeg.shape[0] < raw_eeg.shape[1]:
        raw_eeg = raw_eeg.T

    audio    = np.load(os.path.join(DUTCH_30_PATH, 'raw', f'{pid}_audio.npy'))
    audio_sr = 48000
    ratio    = audio_sr / EEG_SR     # 48000 / 1024 = 46.875

    # 4) build splits
    splits = dict(train=[], val=[], test=[])
    for sent_idx in all_real:
        if sent_idx not in mfa or not mfa[sent_idx]:
            continue
        sd       = wd['sentence_list'][sent_idx]
        s0, s1   = sd['stim_start_idx'], sd['stim_end_idx']
        if s1 > raw_eeg.shape[0]:
            continue

        # neural features (stacked HG)
        ext = extract_features_multiband(raw_eeg[s0:s1], **feature_spec)
        if ext.shape[0] < 2 * LDA_MARGIN + 1:
            continue
        stk = stackFeatures(ext, modelOrder=MO, stepSize=SS).astype(np.float32)
        T   = stk.shape[0]

        # MFA targets / intervals
        target        = [ph['phone'] for ph in mfa[sent_idx]]
        if not target:
            continue
        mfa_intervals = [(ph['start_s'], ph['end_s'], ph['phone']) for ph in mfa[sent_idx]]

        # mel aligned to T iEEG frames
        audio_start, audio_end = int(s0 * ratio), int(s1 * ratio)
        if audio_end > len(audio):
            continue
        mel = build_mel_for_sentence(audio, audio_sr,
                                     audio_start, audio_end,
                                     n_mels=n_mels,
                                     target_n_frames=T)

        # framerate bookkeeping  ── used later for MFA→frame alignment
        duration_sec = (s1 - s0) / EEG_SR
        feature_fps  = T / duration_sec

        split = ('test' if sent_idx in test_sent_ids else
                 'val'  if sent_idx in val_sent_ids  else 'train')
        splits[split].append(dict(
            features      = stk,
            mel           = mel,
            target        = target,
            mfa_intervals = mfa_intervals,
            duration_sec  = duration_sec,
            feature_fps   = feature_fps,
            sent_idx      = sent_idx,
        ))

    return splits

# Articulatory maps
# ============================================================
# Cell 4 — Articulatory feature maps (Dutch phonemes, integer ids)
# ============================================================
# manner: 0=sil, 1=vowel, 2=plosive, 3=fricative, 4=nasal, 5=approximant
N_MANNER = 6
MANNER_ID = {'sil':0, 'vowel':1, 'plosive':2, 'fricative':3, 'nasal':4, 'approx':5}

# voicing: 0=sil/n.a., 1=voiced, 2=voiceless
N_VOICING = 3
VOICING_ID = {'sil':0, 'voiced':1, 'voiceless':2}

# place: 0=sil, 1=labial, 2=alveolar, 3=postalveolar/palatal,
#        4=velar, 5=glottal, 6=vowel-front, 7=vowel-central, 8=vowel-back
N_PLACE = 9
PLACE_ID = {'sil':0, 'labial':1, 'alveolar':2, 'postalv':3, 'velar':4,
            'glottal':5, 'v_front':6, 'v_central':7, 'v_back':8}

# ---- per-phoneme assignments ----
# (covers the 41 Dutch phonemes typical for this dataset; unknowns fall back to sil)
_MANNER = {
    # silence
    'sil': 'sil',
    # vowels
    'a':'vowel','aː':'vowel','ɑ':'vowel','ɛ':'vowel','eː':'vowel',
    'i':'vowel','iː':'vowel','ɪ':'vowel','o':'vowel','oː':'vowel',
    'ɔ':'vowel','u':'vowel','uː':'vowel','y':'vowel','yː':'vowel',
    'øː':'vowel','œ':'vowel','ə':'vowel',
    # diphthongs (treat as vowel)
    'ɛi':'vowel','ɔu':'vowel','œy':'vowel','ɑu':'vowel','ɛy':'vowel',
    # plosives
    'p':'plosive','b':'plosive','t':'plosive','d':'plosive',
    'k':'plosive','g':'plosive','c':'plosive','ʔ':'plosive',
    # fricatives
    'f':'fricative','v':'fricative','s':'fricative','z':'fricative',
    'ʃ':'fricative','ʒ':'fricative','x':'fricative','ɣ':'fricative',
    'h':'fricative',
    # nasals
    'm':'nasal','n':'nasal','ŋ':'nasal','ɲ':'nasal',
    # approximants
    'l':'approx','r':'approx','j':'approx','w':'approx','ʋ':'approx',
}

_VOICING = {
    'sil':'sil',
    # voiceless obstruents
    'p':'voiceless','t':'voiceless','k':'voiceless','c':'voiceless','ʔ':'voiceless',
    'f':'voiceless','s':'voiceless','ʃ':'voiceless','x':'voiceless','h':'voiceless',
    # voiced obstruents
    'b':'voiced','d':'voiced','g':'voiced',
    'v':'voiced','z':'voiced','ʒ':'voiced','ɣ':'voiced',
    # sonorants (all voiced)
    'm':'voiced','n':'voiced','ŋ':'voiced','ɲ':'voiced',
    'l':'voiced','r':'voiced','j':'voiced','w':'voiced','ʋ':'voiced',
}
# every vowel is voiced
for _v in ['a','aː','ɑ','ɛ','eː','i','iː','ɪ','o','oː','ɔ',
          'u','uː','y','yː','øː','œ','ə','ɛi','ɔu','œy','ɑu','ɛy']:
    _VOICING[_v] = 'voiced'

_PLACE = {
    'sil':'sil',
    # consonants
    'p':'labial','b':'labial','m':'labial','f':'labial','v':'labial',
    'w':'labial','ʋ':'labial',
    't':'alveolar','d':'alveolar','n':'alveolar','s':'alveolar',
    'z':'alveolar','l':'alveolar','r':'alveolar',
    'ʃ':'postalv','ʒ':'postalv','j':'postalv','c':'postalv','ɲ':'postalv',
    'k':'velar','g':'velar','x':'velar','ɣ':'velar','ŋ':'velar',
    'h':'glottal','ʔ':'glottal',
    # vowels (front / central / back)
    'i':'v_front','iː':'v_front','ɪ':'v_front',
    'eː':'v_front','ɛ':'v_front','ɛi':'v_front','ɛy':'v_front',
    'y':'v_front','yː':'v_front','øː':'v_front','œ':'v_front','œy':'v_front',
    'ə':'v_central','a':'v_central','aː':'v_central','ɑ':'v_central','ɑu':'v_central',
    'o':'v_back','oː':'v_back','ɔ':'v_back','ɔu':'v_back','u':'v_back','uː':'v_back',
}

# ---- compose to int-valued dicts (what the dataset/model uses) ----
DUTCH_MANNER  = {p: MANNER_ID[v]  for p, v in _MANNER.items()}
DUTCH_VOICING = {p: VOICING_ID[v] for p, v in _VOICING.items()}
DUTCH_PLACE   = {p: PLACE_ID[v]   for p, v in _PLACE.items()}

# quick sanity
print('|MANNER|=', N_MANNER, '|VOICING|=', N_VOICING, '|PLACE|=', N_PLACE)
print('sample (a):',
      'manner=',  DUTCH_MANNER.get('a'),
      'voicing=', DUTCH_VOICING.get('a'),
      'place=',   DUTCH_PLACE.get('a'))

# Vocab + interval helpers
def _unpack_interval(it):
    """Return (phone_str, t0, t1) regardless of tuple order.
       Handles tgt-style (start, end, text) and (text, start, end)."""
    str_elems = [(i, x) for i, x in enumerate(it) if isinstance(x, str)]
    if not str_elems:
        raise ValueError(f"no string label in interval {it}")
    s_idx, ph = str_elems[0]
    nums = [float(x) for i, x in enumerate(it) if i != s_idx]
    t0, t1 = sorted(nums)
    return ph, t0, t1


def build_phone_vocab(sents, sil_token='sil'):
    """Collect phonemes from both s['target'] and s['mfa_intervals'].
       Inserts `sil_token` at index 0 (MFA usually doesn't emit silence)."""
    phones = set()
    for s in sents:
        phones.update(s['target'])
        for it in s['mfa_intervals']:
            ph, _, _ = _unpack_interval(it)
            phones.add(ph)
    others  = sorted(p for p in phones if p != sil_token)
    ordered = [sil_token] + others
    p2i = {p: i for i, p in enumerate(ordered)}
    i2p = {i: p for p, i in p2i.items()}
    return p2i, i2p


def intervals_to_frames_fps(mfa_intervals, T, label_to_id, feature_fps,
                            default_id=0):
    """MFA intervals (seconds) → per-frame phoneme ids of length T."""
    out = np.full(T, default_id, dtype=np.int64)
    for it in mfa_intervals:
        ph, t0, t1 = _unpack_interval(it)
        f0 = max(0, int(round(t0 * feature_fps)))
        f1 = min(T, int(round(t1 * feature_fps)))
        if f1 > f0:
            out[f0:f1] = label_to_id.get(ph, default_id)
    return out

# scalers
from sklearn.preprocessing import StandardScaler
import numpy as np
pid = "P22"

# --- scaler helpers (using the real keys: 'features' and 'mel') ---
def fit_scaler(train_sents):
    sc = StandardScaler()
    for s in train_sents:
        sc.partial_fit(s['features'].astype(np.float32))
    return sc

def fit_mel_scaler(train_sents):
    sc = StandardScaler()
    for s in train_sents:
        sc.partial_fit(s['mel'].astype(np.float32))
    return sc

# 1) splits with mel  (skip if you already ran it and `splits` exists with mel)
splits = build_patient_dataset_with_mel(pid, n_mels=40)

# 2) fit BOTH scalers — this defines `scaler` and `mel_scaler`
scaler     = fit_scaler(splits['train'])
mel_scaler = fit_mel_scaler(splits['train'])

# 3) phoneme vocab
phone_to_idx, idx_to_phone = build_phone_vocab(splits['train'])

# sanity
s0 = splits['train'][0]
print('F:', s0['features'].shape[1],   'scaler.mean_:', scaler.mean_.shape)
print('M:', s0['mel'].shape[1],        'mel_scaler.mean_:', mel_scaler.mean_.shape)
print('|phones|:', len(phone_to_idx))

#  Collate (mel-aware, multi-task)
# ============================================================
import torch

def collate_mt_mel(batch):
    """
    Each item from PhonemeDatasetMTMel is:
        (feats, ctc_targets, ph, mn, vo, pl, mel)
    where shapes are:
        feats       (T, F)   float32
        ctc_targets (L,)     int64       — not used by CE-only training; kept for future CTC
        ph, mn, vo, pl  (T,) int64
        mel         (T, M)   float32

    Returns a dict with everything padded to T_max along the time axis.
    Padded label frames use -100 so CrossEntropyLoss(ignore_index=-100) skips them.
    """
    feats, ctc_targets, ph, mn, vo, pl, mel = zip(*batch)

    B    = len(feats)
    Tmax = max(f.shape[0] for f in feats)
    F    = feats[0].shape[1]
    M    = mel[0].shape[1]

    feats_pad = torch.zeros(B, Tmax, F, dtype=torch.float32)
    mel_pad   = torch.zeros(B, Tmax, M, dtype=torch.float32)
    ph_pad    = torch.full((B, Tmax), -100, dtype=torch.long)
    mn_pad    = torch.full((B, Tmax), -100, dtype=torch.long)
    vo_pad    = torch.full((B, Tmax), -100, dtype=torch.long)
    pl_pad    = torch.full((B, Tmax), -100, dtype=torch.long)
    feat_mask = torch.zeros(B, Tmax, dtype=torch.bool)   # True = real frame
    mel_mask  = torch.zeros(B, Tmax, dtype=torch.bool)
    in_lens   = torch.zeros(B,         dtype=torch.long)

    for i, (f, p, m, v, pp, me) in enumerate(zip(feats, ph, mn, vo, pl, mel)):
        T = f.shape[0]
        feats_pad[i, :T] = f
        mel_pad[i,   :T] = me
        ph_pad[i,    :T] = p
        mn_pad[i,    :T] = m
        vo_pad[i,    :T] = v
        pl_pad[i,    :T] = pp
        feat_mask[i, :T] = True
        mel_mask[i,  :T] = True
        in_lens[i]       = T

    # CTC label bookkeeping (unused for CE-only training, but cheap to keep)
    tgt_lens = torch.tensor([t.numel() for t in ctc_targets], dtype=torch.long)
    tgt_cat  = torch.cat(ctc_targets) if tgt_lens.sum() > 0 \
               else torch.zeros(0, dtype=torch.long)

    return {
        'feats'    : feats_pad,   # (B, T, F)
        'feat_mask': feat_mask,   # (B, T) bool
        'in_lens'  : in_lens,     # (B,)
        'mel'      : mel_pad,     # (B, T, M)
        'mel_mask' : mel_mask,    # (B, T) bool
        'ph'       : ph_pad,      # (B, T) long, -100 padding
        'mn'       : mn_pad,      # (B, T) long, -100 padding
        'vo'       : vo_pad,      # (B, T) long, -100 padding
        'pl'       : pl_pad,      # (B, T) long, -100 padding
        'ctc_tgt'  : tgt_cat,     # (sum L_i,)  — only used if you re-enable CTC later
        'ctc_lens' : tgt_lens,    # (B,)
    }

import numpy as np
import torch

class PhonemeDatasetMTMel(torch.utils.data.Dataset):
    def __init__(self, sents, phone_to_idx, *,
                 sil_token='sil',
                 scaler=None, mel_scaler=None,
                 manner_map=None, voicing_map=None, place_map=None,
                 noise_std=0.0, augment=False,
                 n_time_masks=0, time_mask_max=0,
                 n_feat_masks=0, feat_mask_max_frac=0.0,
                 max_time_shift=0,
                 include_frame_labels=True):
        self.sents       = sents
        self.p2i         = phone_to_idx
        self.sil_token   = sil_token
        self.sil_id      = phone_to_idx[sil_token]
        self.scaler      = scaler
        self.mel_scaler  = mel_scaler
        self.manner_map  = manner_map  or {}
        self.voicing_map = voicing_map or {}
        self.place_map   = place_map   or {}
        self.noise_std   = noise_std
        self.augment     = augment
        self.n_time_masks       = n_time_masks
        self.time_mask_max      = time_mask_max
        self.n_feat_masks       = n_feat_masks
        self.feat_mask_max_frac = feat_mask_max_frac
        self.max_time_shift     = max_time_shift
        self.include_frame_labels = include_frame_labels
        # cache idx→ph for aux mapping
        self._idx_to_ph = {v: k for k, v in phone_to_idx.items()}

    def __len__(self):
        return len(self.sents)

    def _phone_to_aux(self, ph_ids):
        T = ph_ids.shape[0]
        mn = np.zeros(T, dtype=np.int64)
        vo = np.zeros(T, dtype=np.int64)
        pl = np.zeros(T, dtype=np.int64)
        for t, pid in enumerate(ph_ids):
            ph = self._idx_to_ph.get(int(pid), self.sil_token)
            mn[t] = self.manner_map.get(ph,  0)
            vo[t] = self.voicing_map.get(ph, 0)
            pl[t] = self.place_map.get(ph,   0)
        return mn, vo, pl

    def __getitem__(self, i):
        s   = self.sents[i]
        x   = s['features'].astype(np.float32)
        mel = s['mel'].astype(np.float32)
        T   = min(x.shape[0], mel.shape[0])
        x, mel = x[:T], mel[:T]

        if self.scaler     is not None: x   = self.scaler.transform(x).astype(np.float32)
        if self.mel_scaler is not None: mel = self.mel_scaler.transform(mel).astype(np.float32)

        ph = intervals_to_frames_fps(
            s['mfa_intervals'], T, self.p2i,
            feature_fps=s['feature_fps'],
            default_id=self.sil_id,
        )
        mn, vo, pl = self._phone_to_aux(ph)

        if self.augment:
            if self.max_time_shift > 0:
                k = np.random.randint(-self.max_time_shift, self.max_time_shift + 1)
                if k != 0: x = np.roll(x, k, axis=0)
            if self.noise_std > 0:
                x = x + np.random.randn(*x.shape).astype(np.float32) * self.noise_std
            for _ in range(self.n_time_masks):
                w = np.random.randint(0, self.time_mask_max + 1)
                if w > 0 and T - w > 0:
                    t0 = np.random.randint(0, T - w)
                    x[t0:t0+w, :] = 0.0
            F = x.shape[1]
            fmax = max(1, int(self.feat_mask_max_frac * F))
            for _ in range(self.n_feat_masks):
                w = np.random.randint(0, fmax + 1)
                if w > 0 and F - w > 0:
                    f0 = np.random.randint(0, F - w)
                    x[:, f0:f0+w] = 0.0

        # CTC targets (unused by CE-only train, but cheap)
        seq, prev = [], None
        for p in s['target']:
            if p == self.sil_token:
                prev = p; continue
            if p != prev:
                seq.append(self.p2i.get(p, self.sil_id))
            prev = p
        targets = np.asarray(seq, dtype=np.int64)

        return (torch.from_numpy(x),
                torch.from_numpy(targets),
                torch.from_numpy(ph),
                torch.from_numpy(mn),
                torch.from_numpy(vo),
                torch.from_numpy(pl),
                torch.from_numpy(mel))

import torch
import torch.nn as nn

class PhonemeMTMelBiLSTM(nn.Module):
    def __init__(self, n_features, n_phoneme_classes,
                 n_manner=N_MANNER, n_voicing=N_VOICING, n_place=N_PLACE,
                 n_mels=40, hidden=64, n_layers=1, dropout=0.3):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(
            input_size=hidden, hidden_size=hidden,
            num_layers=n_layers, batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        H = 2 * hidden
        self.drop    = nn.Dropout(dropout)
        self.head_ph = nn.Linear(H, n_phoneme_classes)
        self.head_mn = nn.Linear(H, n_manner)
        self.head_vo = nn.Linear(H, n_voicing)
        self.head_pl = nn.Linear(H, n_place)
        self.head_mel= nn.Linear(H, n_mels)

    def forward(self, x, mask=None):
        h = self.in_proj(x)
        h, _ = self.lstm(h)
        h = self.drop(h)
        return {
            'ph_logits': self.head_ph(h),
            'mn_logits': self.head_mn(h),
            'vo_logits': self.head_vo(h),
            'pl_logits': self.head_pl(h),
            'mel_pred' : self.head_mel(h),
        }

n_features = splits['train'][0]['features'].shape[1]
model = PhonemeMTMelBiLSTM(
    n_features=n_features,
    n_phoneme_classes=len(phone_to_idx),
    n_mels=40,
    hidden=64, n_layers=1, dropout=0.3,
)
x = torch.randn(2, 50, n_features)
out = model(x)
for k, v in out.items():
    print(k, tuple(v.shape))

# ============================================================
# Cell 10 — Training loop (multi-task: mel + ph + mn + vo + pl)
# ============================================================
import copy
import torch
import torch.nn as nn

def train_mt_mel(model, train_loader, val_loader,
                 n_epochs, lrs,
                 weights=None,
                 ph_weights=None,
                 weight_decay=1e-4,
                 grad_clip=1.0,
                 device=None,
                 verbose=True):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    # ---- loss weights ----
    w = dict(mel=1.0, ph=1.0, mn=0.3, vo=0.3, pl=0.3)
    if weights: w.update(weights)

    # ---- optimizer / scheduler ----
    if isinstance(lrs, (int, float)):
        opt = torch.optim.AdamW(model.parameters(), lr=float(lrs), weight_decay=weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
        manual_lrs = False
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=lrs[0], weight_decay=weight_decay)
        sched = None
        manual_lrs = True

    # ---- losses (weighted CE only for the phoneme head) ----
    if ph_weights is not None:
        ce_ph_obj = nn.CrossEntropyLoss(ignore_index=-100,
                                        weight=ph_weights.to(device))
    else:
        ce_ph_obj = nn.CrossEntropyLoss(ignore_index=-100)
    ce_obj = nn.CrossEntropyLoss(ignore_index=-100)       # for mn / vo / pl

    # ---- helpers ----
    def _masked_mel_mse(pred, gt, mask):
        per_frame = ((pred - gt) ** 2).mean(dim=-1)
        return (per_frame * mask).sum() / mask.sum().clamp(min=1)

    def _ce(logits, gt):
        B, T, C = logits.shape
        return ce_obj(logits.reshape(B*T, C), gt.reshape(-1))

    def _ce_ph(logits, gt):
        B, T, C = logits.shape
        return ce_ph_obj(logits.reshape(B*T, C), gt.reshape(-1))

    def _step(batch, train):
        feats   = batch['feats'].to(device)
        mel_gt  = batch['mel'].to(device)
        mel_msk = batch['mel_mask'].to(device).float()
        ph_gt   = batch['ph'].to(device)
        mn_gt   = batch['mn'].to(device)
        vo_gt   = batch['vo'].to(device)
        pl_gt   = batch['pl'].to(device)

        out = model(feats)
        l_mel = _masked_mel_mse(out['mel_pred'], mel_gt, mel_msk)
        l_ph  = _ce_ph(out['ph_logits'], ph_gt)
        l_mn  = _ce(out['mn_logits'], mn_gt)
        l_vo  = _ce(out['vo_logits'], vo_gt)
        l_pl  = _ce(out['pl_logits'], pl_gt)

        loss = (w['mel']*l_mel + w['ph']*l_ph
                + w['mn']*l_mn + w['vo']*l_vo + w['pl']*l_pl)

        if train:
            opt.zero_grad()
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

        with torch.no_grad():
            pred_ph = out['ph_logits'].argmax(dim=-1)
            valid   = (ph_gt != -100)
            n_ok    = ((pred_ph == ph_gt) & valid).sum().item()
            n_tot   = valid.sum().item()

        return dict(loss=loss.item(), mel=l_mel.item(), ph=l_ph.item(),
                    mn=l_mn.item(), vo=l_vo.item(), pl=l_pl.item(),
                    ph_correct=n_ok, ph_total=n_tot)

    # ---- training loop ----
    best_val   = float('inf')
    best_model = None
    history    = {k: [] for k in
                  ['train_total','val_total','train_mel','val_mel',
                   'train_ph','val_ph','val_ph_acc','lr']}

    for epoch in range(n_epochs):
        if manual_lrs:
            for pg in opt.param_groups:
                pg['lr'] = lrs[min(epoch, len(lrs)-1)]

        model.train()
        agg = {'loss':0.0, 'mel':0.0, 'ph':0.0}; nb = 0
        for batch in train_loader:
            s = _step(batch, train=True)
            for k in agg: agg[k] += s[k]
            nb += 1

        model.eval()
        vagg = {'loss':0.0, 'mel':0.0, 'ph':0.0}; v_nb = 0; v_ok = v_tot = 0
        with torch.no_grad():
            for batch in val_loader:
                s = _step(batch, train=False)
                for k in vagg: vagg[k] += s[k]
                v_ok += s['ph_correct']; v_tot += s['ph_total']; v_nb += 1

        if sched is not None: sched.step()

        avg_tr  = agg['loss']  / max(nb, 1)
        avg_val = vagg['loss'] / max(v_nb, 1)
        ph_acc  = v_ok / max(v_tot, 1)
        lr_now  = opt.param_groups[0]['lr']

        history['train_total'].append(avg_tr)
        history['val_total'].append(avg_val)
        history['train_mel'].append(agg['mel']/max(nb,1))
        history['val_mel'].append(vagg['mel']/max(v_nb,1))
        history['train_ph'].append(agg['ph']/max(nb,1))
        history['val_ph'].append(vagg['ph']/max(v_nb,1))
        history['val_ph_acc'].append(ph_acc)
        history['lr'].append(lr_now)

        improved = avg_val < best_val
        if improved:
            best_val = avg_val
            best_model = copy.deepcopy(model).cpu()

        if verbose:
            star = ' *' if improved else ''
            print(f'ep {epoch+1:3d}  lr={lr_now:.5f}  '
                  f'train={avg_tr:.4f}  val={avg_val:.4f}  '
                  f'mel={vagg["mel"]/max(v_nb,1):.4f}  '
                  f'ph_ce={vagg["ph"]/max(v_nb,1):.4f}  '
                  f'ph_acc={ph_acc*100:5.2f}%{star}')

    if best_model is not None:
        best_model = best_model.to(device)
    return best_model, history

# Decoding & evaluation
# ============================================================
import numpy as np

# ------------------------------------------------------------
# 1. Viterbi with self-loop bonus  (frame-level decode)
# ------------------------------------------------------------
def viterbi_with_self_loop(logp, bonus=0.0):
    """
    logp  : (T, K) log-probabilities (or logits — only relative scale matters)
    bonus : extra score for taking the self-transition (state stays the same).
            Positive bonus → smoother, longer runs.  Try 0.5 - 3.0.

    Returns the optimal state sequence, shape (T,) int64.

    State model: every state can transition to every other state (no LM),
    plus a `bonus` reward when it stays in the same state.
    O(T·K) thanks to the "top-1 / top-2 of previous frame" trick.
    """
    logp = np.asarray(logp, dtype=np.float64)
    T, K = logp.shape
    score = np.full((T, K), -np.inf, dtype=np.float64)
    back  = np.zeros((T, K), dtype=np.int64)

    score[0] = logp[0]

    arange_K = np.arange(K)
    for t in range(1, T):
        prev = score[t - 1]                                # (K,)

        # top-1 and top-2 across all previous states  -----------------
        top1_j = int(np.argmax(prev))
        top1_v = prev[top1_j]
        if K >= 2:
            tmp = prev.copy(); tmp[top1_j] = -np.inf
            top2_j = int(np.argmax(tmp))
            top2_v = prev[top2_j]
        else:
            top2_j, top2_v = top1_j, -np.inf

        # for each target state k, best NON-self predecessor
        non_self_arg = np.where(arange_K == top1_j, top2_j, top1_j)
        non_self_val = np.where(arange_K == top1_j, top2_v, top1_v)

        # self-loop option
        self_val = prev + bonus
        use_self = self_val >= non_self_val
        best_val = np.where(use_self, self_val, non_self_val)
        best_arg = np.where(use_self, arange_K,  non_self_arg)

        score[t] = best_val + logp[t]
        back[t]  = best_arg

    # backtrace
    out = np.zeros(T, dtype=np.int64)
    out[-1] = int(np.argmax(score[-1]))
    for t in range(T - 2, -1, -1):
        out[t] = back[t + 1, out[t + 1]]
    return out


# ------------------------------------------------------------
# 2. Collapse consecutive duplicates (CTC-style), drop silence
# ------------------------------------------------------------
def collapse_repeats(seq, sil_id=None):
    """
    seq    : iterable of ints (frame-level state IDs from Viterbi)
    sil_id : if given, frames in this state are dropped entirely

    Returns list[int] — the collapsed phoneme sequence.
    """
    out, prev = [], None
    for x in seq:
        x = int(x)
        if sil_id is not None and x == sil_id:
            prev = x
            continue
        if x != prev:
            out.append(x)
        prev = x
    return out


# ------------------------------------------------------------
# 3. Needleman-Wunsch match rate
# ------------------------------------------------------------
def needleman_wunsch_match_rate(pred, gold,
                                match=1, mismatch=-1, gap=-1,
                                normalize='max'):
    """
    Globally aligns `pred` and `gold` with NW, returns match rate.

    pred, gold : list[int] (or list[str]) phoneme sequences
    normalize  : 'max'  -> matches / max(len(pred), len(gold))   ← default
                 'gold' -> matches / len(gold)
                 'pred' -> matches / len(pred)

    Symmetric for 'max'; 'gold' is the WER-style "recall".
    """
    n, m = len(pred), len(gold)
    if n == 0 and m == 0: return 1.0
    if n == 0 or m == 0:  return 0.0

    dp = np.zeros((n + 1, m + 1), dtype=np.int32)
    dp[:, 0] = np.arange(n + 1) * gap
    dp[0, :] = np.arange(m + 1) * gap

    for i in range(1, n + 1):
        pi = pred[i - 1]
        for j in range(1, m + 1):
            s = match if pi == gold[j - 1] else mismatch
            dp[i, j] = max(dp[i - 1, j - 1] + s,
                           dp[i - 1, j]     + gap,
                           dp[i,     j - 1] + gap)

    # backtrace, counting exact matches
    i, j, matches = n, m, 0
    while i > 0 and j > 0:
        s = match if pred[i - 1] == gold[j - 1] else mismatch
        if dp[i, j] == dp[i - 1, j - 1] + s:
            if pred[i - 1] == gold[j - 1]:
                matches += 1
            i -= 1; j -= 1
        elif dp[i, j] == dp[i - 1, j] + gap:
            i -= 1
        else:
            j -= 1

    if normalize == 'gold':  denom = m
    elif normalize == 'pred': denom = n
    else:                     denom = max(n, m)
    return matches / max(denom, 1)


# ------------------------------------------------------------
# Convenience: end-to-end "decode a model on a loader" helper
# ------------------------------------------------------------
@torch.no_grad()
def decode_loader(model, loader, sil_id, bonus=1.0, device=None):
    """
    Runs the model over a loader, decodes per-sentence phoneme sequences,
    and returns lists of (pred_ids, gold_ids) for each sentence.
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    all_pred, all_gold = [], []
    for batch in loader:
        feats = batch['feats'].to(device)
        out   = model(feats)
        logp  = torch.log_softmax(out['ph_logits'], dim=-1).cpu().numpy()  # (B, T, K)
        ph_gt = batch['ph'].cpu().numpy()                                  # (B, T)
        in_len = batch['in_lens'].cpu().numpy()

        for b in range(logp.shape[0]):
            T = int(in_len[b])
            states = viterbi_with_self_loop(logp[b, :T], bonus=bonus)
            pred   = collapse_repeats(states, sil_id=sil_id)
            # gold: collapse the frame-level gold too, ignoring padding (-100) and sil
            gold_frames = ph_gt[b, :T]
            gold_frames = gold_frames[gold_frames != -100]
            gold        = collapse_repeats(gold_frames, sil_id=sil_id)
            all_pred.append(pred)
            all_gold.append(gold)
    return all_pred, all_gold

# 1) viterbi sanity: a noisy 3-state signal
np.random.seed(0)
T, K = 30, 4
logp = np.full((T, K), -2.0)
true = [0]*10 + [1]*10 + [2]*10
for t, s in enumerate(true):
    logp[t, s] = 0.0
logp += np.random.randn(T, K) * 0.5
print('greedy:', logp.argmax(axis=1).tolist())
print('viterb:', viterbi_with_self_loop(logp, bonus=1.0).tolist())

# 2) collapse
print(collapse_repeats([0,0,1,1,1,0,2,2,3], sil_id=0))   # → [1, 2, 3]

# 3) NW
pred = [1,2,3,4,5]; gold = [1,2,9,4,5]
print(needleman_wunsch_match_rate(pred, gold))           # → 4/5 = 0.8

# Cell 12 — Per-patient runner
# ============================================================
import gc, random
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

SIL = 'sil'   # invented silence token (MFA didn't emit one)

def _set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _sentence_weights(sents, phone_to_idx, sil_token=SIL):
    """Up-weight sentences whose rarest phoneme is rare in the train set."""
    counts = np.zeros(len(phone_to_idx), dtype=np.float64)
    for s in sents:
        for p in s['target']:
            counts[phone_to_idx.get(p, phone_to_idx[sil_token])] += 1
    inv = 1.0 / (counts + 1e-3)
    inv[phone_to_idx[sil_token]] = 0.0           # don't reward silence
    weights = []
    for s in sents:
        ids = [phone_to_idx.get(p, phone_to_idx[sil_token]) for p in s['target']]
        weights.append(float(inv[ids].mean()) if ids else 1.0)
    return np.array(weights, dtype=np.float64)

def _mel_pearson(model, loader, mel_scaler, device):
    """Average Pearson correlation between predicted and gold mel over all valid frames."""
    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for batch in loader:
            feats   = batch['feats'].to(device)
            mel_gt  = batch['mel'].cpu().numpy()
            mel_msk = batch['mel_mask'].cpu().numpy().astype(bool)
            out     = model(feats)
            mel_pr  = out['mel_pred'].cpu().numpy()
            for b in range(mel_pr.shape[0]):
                T = mel_msk[b].sum()
                preds.append(mel_pr[b, :T])
                gts.append(mel_gt[b, :T])
    pr = np.concatenate(preds, axis=0)            # (Nframes, M)
    gt = np.concatenate(gts,   axis=0)
    # Pearson per mel-bin, averaged
    pr0 = pr - pr.mean(0, keepdims=True)
    gt0 = gt - gt.mean(0, keepdims=True)
    num = (pr0 * gt0).sum(0)
    den = np.sqrt((pr0**2).sum(0) * (gt0**2).sum(0)) + 1e-9
    r_per_bin = num / den                          # (M,)
    return float(r_per_bin.mean()), r_per_bin

def run_mt_mel_for_patient(pid, seed=0,
                           # data
                           n_mels=40,
                           # augmentation
                           noise_std=0.75,
                           n_time_masks=2, time_mask_max=10,
                           n_feat_masks=2, feat_mask_max_frac=0.10,
                           max_time_shift=3,
                           # model
                           model_kind='inception',
                           hidden=64, n_layers=1, dropout=0.3,
                           # training
                           n_epochs=60, lr=1e-3, batch_size=8,
                           weights=None,
                           # decoding
                           viterbi_bonus=1.0,
                           # logistics
                           use_sampler=True,
                           oversample=2.0,  # 
                           verbose=True):
    """
    Returns dict:
        nw_match_rate         float   mean across test sentences
        nw_match_rates        list    per-sentence
        val_ph_acc            float   best-val phoneme frame accuracy
        mel_pearson_test      float   averaged across mel bins
        mel_pearson_per_bin   ndarray
        history               dict    from train_mt_mel
        predictions           list of (sent_idx, pred_phones, gold_phones)
        model                 trained best-val model (on CPU)
        scalers               (scaler, mel_scaler)
        vocab                 (phone_to_idx, idx_to_phone)
    """
    _set_seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ---- 1. data ----
    splits = build_patient_dataset_with_mel(pid, n_mels=n_mels)
    if splits is None or len(splits['train']) == 0:
        return None
    if verbose:
        print(f'[{pid}] train/val/test = '
              f'{len(splits["train"])}/{len(splits["val"])}/{len(splits["test"])}')

    # ---- 2. scalers + vocab ----
    scaler     = fit_scaler(splits['train'])
    mel_scaler = fit_mel_scaler(splits['train'])
    phone_to_idx, idx_to_phone = build_phone_vocab(splits['train'], sil_token=SIL)
    sil_id = phone_to_idx[SIL]
    
    from collections import Counter
    ph_counts = Counter()
    for s in splits['train']:
        for it in s['mfa_intervals']:
            ph, t0, t1 = _unpack_interval(it)
            if ph not in phone_to_idx:
                continue
            n_frames = int(round((t1 - t0) * s['feature_fps']))
            ph_counts[phone_to_idx[ph]] += n_frames
    
    n_classes = len(phone_to_idx)
    # ph_weights = torch.ones(n_classes, dtype=torch.float32)
    # total = sum(ph_counts.values())
    # for k in range(n_classes):
    #     c = ph_counts.get(k, 1)
    #     ph_weights[k] = total / (n_classes * c)        # inverse-freq
    # ph_weights = ph_weights.clamp(0.3, 3.0)           # avoid extreme values
    # ph_weights = None
    # ph_weights[phone_to_idx['sil']] = 0.1              # downweight silence

    # inside run_mt_mel_for_patient, replace the entire ph_weights block with:

    # inside run_mt_mel_for_patient — NO imports inside the function

    USE_CLASS_BALANCING = False   # toggle per experiment
    
    if USE_CLASS_BALANCING:
        ph_counts = Counter()
        for s in splits['train']:
            for it in s['mfa_intervals']:
                ph, t0, t1 = _unpack_interval(it)
                if ph not in phone_to_idx:
                    continue
                n_frames = int(round((t1 - t0) * s['feature_fps']))
                ph_counts[phone_to_idx[ph]] += n_frames
    
        n_classes = len(phone_to_idx)
        total = sum(ph_counts.values()) or 1
        ph_weights = torch.ones(n_classes, dtype=torch.float32)
        for k in range(n_classes):
            c = ph_counts.get(k, 1)
            ph_weights[k] = total / (n_classes * c)
        ph_weights = ph_weights.clamp(0.7, 1.5)
        ph_weights[sil_id] = 0.3
    else:
        ph_weights = None

    # ---- 3. datasets ----
    common = dict(scaler=scaler, mel_scaler=mel_scaler,
                  manner_map=DUTCH_MANNER, voicing_map=DUTCH_VOICING,
                  place_map=DUTCH_PLACE, sil_token=SIL)

    train_ds = PhonemeDatasetMTMel(
        splits['train'], phone_to_idx, **common,
        noise_std=noise_std, augment=True,
        n_time_masks=n_time_masks, time_mask_max=time_mask_max,
        n_feat_masks=n_feat_masks, feat_mask_max_frac=feat_mask_max_frac,
        max_time_shift=max_time_shift,
        include_frame_labels=True,
    )
    val_ds = PhonemeDatasetMTMel(
        splits['val'], phone_to_idx, **common,
        noise_std=0.0, augment=False,
        n_time_masks=0, time_mask_max=0,
        n_feat_masks=0, feat_mask_max_frac=0.0, max_time_shift=0,
        include_frame_labels=True,
    )
    test_ds = PhonemeDatasetMTMel(
        splits['test'], phone_to_idx, **common,
        noise_std=0.0, augment=False,
        n_time_masks=0, time_mask_max=0,
        n_feat_masks=0, feat_mask_max_frac=0.0, max_time_shift=0,
        include_frame_labels=True,
    )

    # ---- 4. loaders ----
    if use_sampler:
        sw = _sentence_weights(splits['train'], phone_to_idx, sil_token=SIL)
        n_samples = int(round(len(sw) * oversample))
        sampler = WeightedRandomSampler(weights=sw, num_samples=n_samples, replacement=True)
        train_loader = DataLoader(train_ds, batch_size=batch_size,
                                  sampler=sampler, collate_fn=collate_mt_mel,
                                  num_workers=0, pin_memory=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size,
                                  shuffle=True, collate_fn=collate_mt_mel,
                                  num_workers=0, pin_memory=True)

    val_loader  = DataLoader(val_ds,  batch_size=batch_size, shuffle=False,
                             collate_fn=collate_mt_mel, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             collate_fn=collate_mt_mel, num_workers=0, pin_memory=True)

    # ---- 5. model ----
    n_features = splits['train'][0]['features'].shape[1]
    if model_kind == 'inception':
        model = PhonemeMTMelInception(
            n_features=n_features,
            n_phoneme_classes=len(phone_to_idx),
            n_mels=n_mels,
            conv_dim=32, branch_ch=16, n_inception=2,
            hidden=hidden, n_layers=n_layers, dropout=dropout,
        )
    else:
        model = PhonemeMTMelBiLSTM(
            n_features=n_features,
            n_phoneme_classes=len(phone_to_idx),
            n_mels=n_mels,
            hidden=hidden, n_layers=n_layers, dropout=dropout,
        )

    # ---- 6. train ----
    best_model, history = train_mt_mel(
        model, train_loader, val_loader,
        n_epochs=n_epochs, lrs=lr,
        weights=weights,
        ph_weights=ph_weights,                          # ← pass it in
        device=device, verbose=verbose,
    )

    # ---- 7. evaluate on TEST ----
    all_pred, all_gold = decode_loader(
        best_model, test_loader, sil_id=sil_id, bonus=viterbi_bonus, device=device
    )
    # convert ids → phoneme strings for return (easier to read)
    pred_str = [[idx_to_phone[i] for i in p] for p in all_pred]
    gold_str = [[idx_to_phone[i] for i in g] for g in all_gold]
    rates = [needleman_wunsch_match_rate(p, g) for p, g in zip(all_pred, all_gold)]
    nw_match = float(np.mean(rates)) if rates else 0.0

    # mel Pearson on test
    mel_r, mel_r_bins = _mel_pearson(best_model, test_loader, mel_scaler, device)

    # collect predictions with sent_idx for traceability
    predictions = [
        {'sent_idx': splits['test'][i]['sent_idx'],
         'pred': pred_str[i], 'gold': gold_str[i],
         'nw_rate': rates[i]}
        for i in range(len(rates))
    ]

    # ---- 8. tidy + return ----
    best_model = best_model.cpu()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if verbose:
        print(f'[{pid}] TEST  NW match = {nw_match*100:5.2f}%   '
              f'mel Pearson = {mel_r:.3f}   '
              f'best val ph_acc = {max(history["val_ph_acc"])*100:5.2f}%')

    return dict(
        pid                = pid,
        seed               = seed,
        nw_match_rate      = nw_match,
        nw_match_rates     = rates,
        val_ph_acc         = float(max(history['val_ph_acc'])),
        mel_pearson_test   = mel_r,
        mel_pearson_per_bin= mel_r_bins,
        history            = history,
        predictions        = predictions,
        model              = best_model,
        scalers            = (scaler, mel_scaler),
        vocab              = (phone_to_idx, idx_to_phone),
    )

# device = 'cuda' if torch.cuda.is_available() else 'cpu'

def ctc_greedy_decode(log_probs, idx_to_phone):
    """Argmax per frame, collapse repeats, remove blanks."""
    path = log_probs.argmax(axis=-1)
    decoded, prev = [], 0
    for p in path:
        if p != prev and p != 0:
            decoded.append(idx_to_phone[p])
        prev = p
    return decoded


def evaluate(model, samples, scaler, idx_to_phone):
    model.eval(); results = []
    with torch.no_grad():
        for s in samples:
            feats = scaler.transform(s['features']).astype(np.float32)
            x = torch.from_numpy(feats).unsqueeze(0).to(DEVICE)
            log_p = model(x).squeeze(0).cpu().numpy()
            pred = ctc_greedy_decode(log_p, idx_to_phone)
            results.append(dict(sent_idx=s['sent_idx'], gold=s['target'],
                                 pred=pred, log_probs=log_p))
    return results

def ctc_results_to_out(results):
    """Wrap CTC results list into an LDA-style 'out' dict for use with the
    existing visualization and metric functions."""
    true_labels, true_sentence_ids = [], []
    predictions, pred_sentence_ids = [], []
    for r in results:
        for ph in r['gold']:
            true_labels.append(ph); true_sentence_ids.append(r['sent_idx'])
        for ph in r['pred']:
            predictions.append(ph); pred_sentence_ids.append(r['sent_idx'])
    return dict(
        true_labels=np.array(true_labels),
        true_sentence_ids=np.array(true_sentence_ids),
        predictions=np.array(predictions),
        pred_sentence_ids=np.array(pred_sentence_ids),
    )

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

def show_predictions_html(out, label="model", max_sentences=20):
    """Single-model NW visualization (gold + pred, no comparison)."""
    gold_per, pred_per = gather_sequences(out)
    common = sorted(gold_per.keys())
    rows = []
    rows.append("<style>"
                ".pcomp td { padding:4px 8px; font-family:monospace; font-size:13px; }"
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
        gold = gold_per.get(sid, [])
        pred = pred_per.get(sid, [])
        if not gold: continue
        aligned = needleman_wunsch(gold, pred)
        g_html, p_html = render_aligned_pair(aligned)
        rows.append(f"<tr class='sentheader'><td colspan='2'>Sentence {sid}</td></tr>")
        rows.append(f"<tr><td>{label} gold</td><td>{g_html}</td></tr>")
        rows.append(f"<tr><td>{label} pred</td><td>{p_html}</td></tr>")
    rows.append("</table>")
    return ''.join(rows)

def needleman_wunsch(gold, pred, match=1, mismatch=-1, gap=-1):
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


def nw_summary(results):
    """results: list of dicts with 'gold' and 'pred' keys (sentence-level)."""
    all_aligned = []
    for r in results:
        all_aligned.extend(needleman_wunsch(r['gold'], r['pred']))
    n_match = sum(1 for g, p in all_aligned if g is not None and p is not None and g == p)
    n_sub   = sum(1 for g, p in all_aligned if g is not None and p is not None and g != p)
    n_del   = sum(1 for g, p in all_aligned if g is not None and p is None)
    n_ins   = sum(1 for g, p in all_aligned if g is None and p is not None)
    n_gold  = sum(1 for g, p in all_aligned if g is not None)
    per     = (n_sub + n_del + n_ins) / max(n_gold, 1)
    return dict(n_gold=n_gold, n_match=n_match, n_sub=n_sub, n_del=n_del, n_ins=n_ins,
                match_rate=n_match/max(n_gold,1), per=per)

res = run_mt_mel_for_patient(
    'P22', seed=0,
    n_epochs=2, lr=1e-3, batch_size=8,
    weights={'mel':1.0,'ph':1.0,'mn':0.3,'vo':0.3,'pl':0.3},
    noise_std=0.75, viterbi_bonus=1.0,
)

res = run_mt_mel_for_patient(
    'P22', seed=0,
    n_epochs=60, lr=1e-3, batch_size=8,
    weights={'mel': 1.0, 'ph': 1.0, 'mn': 0.3, 'vo': 0.3, 'pl': 0.3},
    noise_std=0.75,
    viterbi_bonus=1.0,
)
print(f'NW match: {res["nw_match_rate"]*100:.2f}%')
print(f'Mel Pearson: {res["mel_pearson_test"]:.3f}')
# peek at one prediction
print('pred:', res['predictions'][0]['pred'][:20])
print('gold:', res['predictions'][0]['gold'][:20])

# Cell 15 — Result visualization
# ============================================================
import matplotlib.pyplot as plt
import numpy as np

def visualize_results(res, splits, n_test_examples=2):
    h = res['history']
    pid = res['pid']

    fig = plt.figure(figsize=(14, 10))

    # ─── 1. Training curves: total loss ───
    ax = plt.subplot(3, 2, 1)
    ax.plot(h['train_total'], label='train')
    ax.plot(h['val_total'],   label='val')
    ax.set_title(f'[{pid}] Total loss');  ax.set_xlabel('epoch'); ax.legend()

    # ─── 2. Per-task val losses ───
    ax = plt.subplot(3, 2, 2)
    ax.plot(h['val_mel'], label='mel MSE (val)')
    ax.plot(h['val_ph'],  label='ph CE (val)')
    ax.set_title('Val losses per task'); ax.set_xlabel('epoch'); ax.legend()

    # ─── 3. Phoneme frame accuracy on val ───
    ax = plt.subplot(3, 2, 3)
    ax.plot(np.array(h['val_ph_acc']) * 100)
    ax.axhline(100 / len(res['vocab'][0]), color='gray', ls='--',
               label=f'uniform chance ({100/len(res["vocab"][0]):.1f}%)')
    ax.set_title('Val phoneme frame accuracy')
    ax.set_xlabel('epoch'); ax.set_ylabel('%'); ax.legend()

    # ─── 4. NW match rate per test sentence ───
    ax = plt.subplot(3, 2, 4)
    rates = np.array(res['nw_match_rates']) * 100
    ax.bar(range(len(rates)), rates)
    ax.axhline(rates.mean(), color='red', ls='--',
               label=f'mean = {rates.mean():.1f}%')
    ax.set_title('NW match rate per test sentence')
    ax.set_xlabel('test sent idx'); ax.set_ylabel('%'); ax.legend()

    # ─── 5. Mel Pearson per bin ───
    ax = plt.subplot(3, 2, 5)
    r = res['mel_pearson_per_bin']
    ax.bar(range(len(r)), r)
    ax.axhline(r.mean(), color='red', ls='--',
               label=f'mean r = {r.mean():.3f}')
    ax.set_title('Mel Pearson per bin')
    ax.set_xlabel('mel bin (low → high freq)')
    ax.set_ylabel('Pearson r'); ax.legend()

    # ─── 6. NW match-rate histogram ───
    ax = plt.subplot(3, 2, 6)
    ax.hist(rates, bins=10, edgecolor='black')
    ax.set_title('NW match-rate distribution')
    ax.set_xlabel('%'); ax.set_ylabel('# sentences')

    plt.tight_layout(); plt.show()

    # ─── 7. Mel predicted vs gold for a few test sentences ───
    model = res['model'].to('cpu').eval()
    scaler, mel_scaler = res['scalers']
    phone_to_idx, idx_to_phone = res['vocab']

    test_sents = splits['test'][:n_test_examples]
    fig, axes = plt.subplots(n_test_examples, 2,
                              figsize=(14, 3.5 * n_test_examples),
                              squeeze=False)
    for row, s in enumerate(test_sents):
        x   = scaler.transform(s['features']).astype(np.float32)
        mel_gold = s['mel']
        with torch.no_grad():
            out = model(torch.from_numpy(x).unsqueeze(0))
        mel_pred = out['mel_pred'].squeeze(0).numpy()
        # un-scale predictions back into the gold's space for comparison
        mel_pred_unscaled = mel_pred * np.sqrt(mel_scaler.var_) + mel_scaler.mean_

        vmin = min(mel_gold.min(), mel_pred_unscaled.min())
        vmax = max(mel_gold.max(), mel_pred_unscaled.max())
        axes[row,0].imshow(mel_gold.T, aspect='auto', origin='lower',
                           vmin=vmin, vmax=vmax)
        axes[row,0].set_title(f'sent {s["sent_idx"]}  GOLD mel')
        axes[row,0].set_xlabel('frame'); axes[row,0].set_ylabel('mel bin')

        axes[row,1].imshow(mel_pred_unscaled.T, aspect='auto', origin='lower',
                           vmin=vmin, vmax=vmax)
        axes[row,1].set_title(f'sent {s["sent_idx"]}  PREDICTED mel')
        axes[row,1].set_xlabel('frame')
    plt.tight_layout(); plt.show()

    # ─── 8. Phoneme alignment for one test sentence ───
    p = res['predictions'][0]
    print(f'\nSentence {p["sent_idx"]} (NW match = {p["nw_rate"]*100:.1f}%)')
    print('GOLD :', ' '.join(p['gold']))
    print('PRED :', ' '.join(p['pred']))

    # ─── headline numbers ───
    print(f'\n[{pid}] ─── headline ─────────────────────────')
    print(f'  NW match rate (test)  : {res["nw_match_rate"]*100:5.2f}%')
    print(f'  Mel Pearson   (test)  : {res["mel_pearson_test"]:.3f}')
    print(f'  Best val ph_acc       : {res["val_ph_acc"]*100:5.2f}%')

# call it
splits = build_patient_dataset_with_mel(res['pid'])     # re-build if not in scope
visualize_results(res, splits, n_test_examples=2)

import matplotlib.pyplot as plt

audio_sr = 48000
audio = np.load(os.path.join(DUTCH_30_PATH, 'raw', 'P22_audio.npy'))
ratio = audio_sr / EEG_SR

sent_idx = 4
sample = next(x for x in splits['train'] if x['sent_idx'] == sent_idx)
s = wd['sentence_list'][sent_idx]

audio_start = int(s['stim_start_idx'] * ratio)
audio_end   = int(s['stim_end_idx']   * ratio)
T_stk = sample['features'].shape[0]

# Compute mel just for this sentence
mel = build_mel_for_sentence(audio, audio_sr, audio_start, audio_end,
                               n_mels=40, target_n_frames=T_stk)

sent_dur = (s['stim_end_idx'] - s['stim_start_idx']) / EEG_SR

fig, ax = plt.subplots(figsize=(16, 5))
ax.imshow(mel.T, aspect='auto', origin='lower', cmap='magma',
           extent=[0, sent_dur, 0, mel.shape[1]])

# Phoneme boundaries
for start_s, end_s, phone in sample['mfa_intervals']:
    ax.axvline(start_s, color='cyan', alpha=0.6, linewidth=0.8)
    ax.text((start_s + end_s) / 2, mel.shape[1] + 0.8, phone,
             ha='center', va='bottom', fontsize=11, color='cyan', weight='bold')
if sample['mfa_intervals']:
    ax.axvline(sample['mfa_intervals'][-1][1], color='cyan', alpha=0.6, linewidth=0.8)

# Word boundaries
prev_word = None
mfa = load_mfa_alignments('P22')[sent_idx]
for ph in mfa:
    cur_word = ph.get('word')
    if prev_word is not None and cur_word != prev_word:
        ax.axvline(ph['start_s'], color='yellow', alpha=0.8, linewidth=1.5, linestyle='--')
    prev_word = cur_word

ax.set_xlabel('time (s)')
ax.set_ylabel('mel bin')
ax.set_ylim(0, mel.shape[1] + 3)
ax.set_title(f"Sentence {sent_idx}: '{s.get('text', '?')}'  "
             f"(cyan = phoneme bounds, yellow dashed = word bounds)")
plt.tight_layout(); plt.show()

# Cell 9b — NeuroIncept-style CNN+BiLSTM (multi-scale temporal)
# ============================================================
import torch
import torch.nn as nn

class InceptionBlock1D(nn.Module):
    """
    Parallel 1×1 / 3×3 / 5×5 1-D temporal convolutions + max-pool branch.
    Operates over (B, C, T). Concatenates four branches → out-channels = 4 * branch_ch.
    """
    def __init__(self, in_ch, branch_ch, dropout=0.1):
        super().__init__()
        def block(k, pad):
            layers = [nn.Conv1d(in_ch, branch_ch, 1)]
            if k > 1:
                layers.append(nn.Conv1d(branch_ch, branch_ch, k, padding=pad))
            layers += [nn.BatchNorm1d(branch_ch), nn.GELU()]
            return nn.Sequential(*layers)

        self.b1 = block(1, 0)
        self.b3 = block(3, 1)
        self.b5 = block(5, 2)
        self.bp = nn.Sequential(
            nn.MaxPool1d(3, stride=1, padding=1),
            nn.Conv1d(in_ch, branch_ch, 1),
            nn.BatchNorm1d(branch_ch), nn.GELU(),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(torch.cat(
            [self.b1(x), self.b3(x), self.b5(x), self.bp(x)], dim=1))


class PhonemeMTMelInception(nn.Module):
    """
    CNN front-end (inception blocks) → BiLSTM → 5 heads.
    Same dict-style output as PhonemeMTMelBiLSTM (drop-in compatible).
    """
    def __init__(self, n_features, n_phoneme_classes,
                 n_manner=N_MANNER, n_voicing=N_VOICING, n_place=N_PLACE,
                 n_mels=40,
                 conv_dim=32, branch_ch=16, n_inception=2,
                 hidden=64, n_layers=1, dropout=0.3):
        super().__init__()
        # F → conv_dim (1×1 conv used as a learnable per-frame linear proj)
        self.in_proj = nn.Sequential(
            nn.Conv1d(n_features, conv_dim, 1),
            nn.BatchNorm1d(conv_dim), nn.GELU(),
            nn.Dropout(dropout),
        )

        # stacked inception blocks, each followed by a 1×1 bottleneck back to conv_dim
        blocks = []
        in_ch = conv_dim
        for _ in range(n_inception):
            blocks.append(InceptionBlock1D(in_ch, branch_ch=branch_ch, dropout=dropout))
            blocks.append(nn.Conv1d(4 * branch_ch, conv_dim, 1))   # bottleneck
            blocks.append(nn.BatchNorm1d(conv_dim))
            blocks.append(nn.GELU())
            in_ch = conv_dim
        self.cnn = nn.Sequential(*blocks)

        # temporal recurrence
        self.lstm = nn.LSTM(
            input_size=conv_dim, hidden_size=hidden,
            num_layers=n_layers, batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        H = 2 * hidden
        self.drop = nn.Dropout(dropout)

        # heads
        self.head_ph  = nn.Linear(H, n_phoneme_classes)
        self.head_mn  = nn.Linear(H, n_manner)
        self.head_vo  = nn.Linear(H, n_voicing)
        self.head_pl  = nn.Linear(H, n_place)
        self.head_mel = nn.Linear(H, n_mels)

    def forward(self, x, mask=None):
        # x: (B, T, F)  -> (B, F, T) for Conv1d
        h = x.transpose(1, 2)
        h = self.in_proj(h)
        h = self.cnn(h)
        h = h.transpose(1, 2)              # (B, T, conv_dim)
        h, _ = self.lstm(h)
        h = self.drop(h)
        return {
            'ph_logits': self.head_ph(h),
            'mn_logits': self.head_mn(h),
            'vo_logits': self.head_vo(h),
            'pl_logits': self.head_pl(h),
            'mel_pred' : self.head_mel(h),
        }

res = run_mt_mel_for_patient(
    'P22', seed=0,
    model_kind='inception',
    n_epochs=40,                # ↓ since each epoch is now 2× the work
    lr=5e-4,
    batch_size=16,
    weights={'mel': 0.3, 'ph': 2.0, 'mn': 0.2, 'vo': 0.2, 'pl': 0.2},
    # --- gentle aug ---
    noise_std=0.1,
    n_time_masks=0, time_mask_max=0,
    n_feat_masks=0, feat_mask_max_frac=0.0,
    max_time_shift=0,
    oversample=2.0,             # ← see each sent ~2× per epoch
    viterbi_bonus=0.3,
)
print(f'NW match: {res["nw_match_rate"]*100:.2f}%')
print(f'Mel Pearson: {res["mel_pearson_test"]:.3f}')
# peek at one prediction
print('pred:', res['predictions'][0]['pred'][:20])
print('gold:', res['predictions'][0]['gold'][:20])

from IPython.display import display, HTML
PID = res['pid']

out_viz = ctc_results_to_out(res['predictions'])
display(HTML(
    f"<h3>{PID} — Inception+BiLSTM "
    f"(NW={res['nw_match_rate']*100:.1f}%, "
    f"mel r={res['mel_pearson_test']:.3f})</h3>"
))
display(HTML(show_predictions_html(out_viz, label="MT-mel",
                                   max_sentences=15)))

from torch.utils.data import DataLoader
device = 'cuda' if torch.cuda.is_available() else 'cpu'

test_ds = PhonemeDatasetMTMel(
    splits['test'], phone_to_idx,
    sil_token='sil',
    scaler=res['scalers'][0], mel_scaler=res['scalers'][1],
    manner_map=DUTCH_MANNER, voicing_map=DUTCH_VOICING, place_map=DUTCH_PLACE,
    noise_std=0.0, augment=False,
)
test_loader = DataLoader(test_ds, batch_size=8, shuffle=False,
                         collate_fn=collate_mt_mel, num_workers=0)

# try several bonuses
for b in [1.0, 0.5, 0.0, -0.5, -1.0]:
    preds, golds = decode_loader(
        res['model'].to(device).eval(), test_loader,
        sil_id=phone_to_idx['sil'], bonus=b, device=device,
    )
    rates = [needleman_wunsch_match_rate(p, g) for p, g in zip(preds, golds)]
    avg_len = np.mean([len(p) for p in preds])
    print(f'bonus={b:+.1f}  NW={np.mean(rates)*100:5.2f}%  '
          f'mean pred len={avg_len:.1f}')

import torch
from collections import Counter

best_model = res['model'].to('cpu').eval()
phone_to_idx, idx_to_phone = res['vocab']
scaler, mel_scaler = res['scalers']
splits = build_patient_dataset_with_mel(res['pid'], n_mels=40)

# pick a test sentence with low NW match
s = splits['test'][0]
x = torch.from_numpy(scaler.transform(s['features']).astype('float32')).unsqueeze(0)
with torch.no_grad():
    out = best_model(x)
logits = out['ph_logits'].squeeze(0)              # (T, K)

raw_argmax = logits.argmax(dim=-1).numpy()        # per-frame greedy
print('raw  per-frame distribution :',
      Counter(idx_to_phone[i] for i in raw_argmax).most_common(8))
# top-5 per-frame margins
top2 = logits.topk(2, dim=-1)
margins = (top2.values[:, 0] - top2.values[:, 1]).numpy()
print('mean top1-top2 margin :', margins.mean(), '(low = unconfident)')

res_A = run_mt_mel_for_patient(
    'P22', seed=0,
    model_kind='bilstm',
    n_epochs=60, lr=1e-3, batch_size=8,
    weights={'mel': 0.0, 'ph': 1.0, 'mn': 0.3, 'vo': 0.3, 'pl': 0.3},  # mel OFF
    noise_std=0.75, n_time_masks=2, time_mask_max=10,
    n_feat_masks=2, feat_mask_max_frac=0.10, max_time_shift=3,
    oversample=1.0,
    viterbi_bonus=1.0,
)
# Also: disable class-balanced CE for this run.
# Quick way: in run_mt_mel_for_patient, set `ph_weights = None` right after the
# ph_weights computation. Or comment out the entire ph_counts/ph_weights block.

from IPython.display import display, HTML
PID = res['pid']

out_viz = ctc_results_to_out(res['predictions'])
display(HTML(
    f"<h3>{PID} — Inception+BiLSTM "
    f"(NW={res['nw_match_rate']*100:.1f}%, "
    f"mel r={res['mel_pearson_test']:.3f})</h3>"
))
display(HTML(show_predictions_html(out_viz, label="MT-mel",
                                   max_sentences=15)))

res_B = run_mt_mel_for_patient(
    'P22', seed=0,
    model_kind='bilstm',
    n_epochs=60, lr=1e-3, batch_size=8,
    weights={'mel': 0.5, 'ph': 1.0, 'mn': 0.3, 'vo': 0.3, 'pl': 0.3},  # mel ON
    noise_std=0.75, n_time_masks=2, time_mask_max=10,
    n_feat_masks=2, feat_mask_max_frac=0.10, max_time_shift=3,
    oversample=1.0,
    viterbi_bonus=1.0,
)
# Keep class-balanced CE OFF (ph_weights = None) for this run too.

res_C = run_mt_mel_for_patient(
    'P22', seed=0,
    model_kind='bilstm',
    n_epochs=60, lr=1e-3, batch_size=8,
    weights={'mel': 0.0, 'ph': 1.0, 'mn': 0.3, 'vo': 0.3, 'pl': 0.3},  # mel OFF
    noise_std=0.75, n_time_masks=2, time_mask_max=10,
    n_feat_masks=2, feat_mask_max_frac=0.10, max_time_shift=3,
    oversample=1.0,
    viterbi_bonus=1.0,
)
# Class-balanced CE ON with mild clamping:
#   ph_weights.clamp(0.5, 2.0) and ph_weights[sil_id] = 0.3

from IPython.display import display, HTML

for name, r in [('A: baseline',       res_A),
                ('B: +mel',           res_B),
                ('C: +class-balance', res_C)]:
    out_viz = ctc_results_to_out(r['predictions'])
    display(HTML(
        f"<h3>{name} — NW={r['nw_match_rate']*100:.1f}%, "
        f"mel r={r['mel_pearson_test']:.3f}</h3>"
    ))
    display(HTML(show_predictions_html(out_viz, label=name,
                                       max_sentences=5)))     # 5 sentences each for compactness

# Cell 16 — Stage-1 model: BiLSTM encoder that predicts mel
# ============================================================
import torch
import torch.nn as nn

class MelEncoder(nn.Module):
    """
    Stage 1: brain features -> 40-dim log-mel via BiLSTM trunk.
    Exposes encode(x) so Stage 2 can read the LSTM hidden states directly.
    """
    def __init__(self, n_features, hidden=64, n_layers=2, dropout=0.3, n_mels=40):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(
            input_size=hidden, hidden_size=hidden,
            num_layers=n_layers, batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.H = 2 * hidden
        self.drop     = nn.Dropout(dropout)
        self.head_mel = nn.Linear(self.H, n_mels)

    def encode(self, x):
        """Return per-frame LSTM hidden states (B, T, 2*hidden)."""
        h = self.in_proj(x)
        h, _ = self.lstm(h)
        return h

    def forward(self, x):
        h = self.encode(x)
        h = self.drop(h)
        return {'mel_pred': self.head_mel(h), 'hidden': h}

# Cell 17 — Stage-1 trainer: maximize mel Pearson correlation
# ============================================================
import copy
import torch
import torch.nn as nn

def neg_pearson_mel_loss(pred, gt, mask):
    """
    Per-mel-bin Pearson correlation over all valid frames in the batch.
    Loss = -mean(Pearson). Range: [-1, +1] before negation.
    """
    valid = mask.bool()
    pv = pred[valid]                 # (Nvalid, M)
    gv = gt[valid]
    if pv.shape[0] < 2:
        return ((pred - gt) ** 2).mean()
    pc = pv - pv.mean(dim=0, keepdim=True)
    gc = gv - gv.mean(dim=0, keepdim=True)
    num = (pc * gc).sum(dim=0)
    den = torch.sqrt((pc**2).sum(dim=0).clamp(min=1e-9) *
                     (gc**2).sum(dim=0).clamp(min=1e-9))
    return -(num / den).mean()


def train_mel_encoder(encoder, train_loader, val_loader,
                      n_epochs, lr=1e-3, weight_decay=1e-4,
                      device=None, verbose=True):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    encoder = encoder.to(device)
    # opt   = torch.optim.AdamW(encoder.parameters(), lr=lr, weight_decay=weight_decay)
    opt = torch.optim.AdamW(encoder.parameters(), lr=lr, weight_decay=5e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)

    best_val, best_encoder = float('inf'), None
    history = {'train_loss': [], 'val_loss': [], 'val_pearson': []}

    for epoch in range(n_epochs):
        encoder.train()
        tr, nb = 0.0, 0
        for batch in train_loader:
            feats   = batch['feats'].to(device)
            mel_gt  = batch['mel'].to(device)
            mel_msk = batch['mel_mask'].to(device).float()
            out = encoder(feats)
            loss = neg_pearson_mel_loss(out['mel_pred'], mel_gt, mel_msk)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            opt.step()
            tr += loss.item(); nb += 1

        encoder.eval()
        vl, vnb = 0.0, 0
        preds, gts = [], []
        with torch.no_grad():
            for batch in val_loader:
                feats   = batch['feats'].to(device)
                mel_gt  = batch['mel'].to(device)
                mel_msk = batch['mel_mask'].to(device).bool()
                out = encoder(feats)
                vl += neg_pearson_mel_loss(out['mel_pred'], mel_gt, mel_msk.float()).item()
                vnb += 1
                preds.append(out['mel_pred'][mel_msk].cpu())
                gts.append(mel_gt[mel_msk].cpu())
        pp = torch.cat(preds, 0); gg = torch.cat(gts, 0)
        pc = pp - pp.mean(0, keepdim=True); gc = gg - gg.mean(0, keepdim=True)
        num = (pc * gc).sum(0)
        den = torch.sqrt((pc**2).sum(0).clamp(min=1e-9) *
                         (gc**2).sum(0).clamp(min=1e-9))
        val_pearson = (num / den).mean().item()

        sched.step()
        avg_tr, avg_val = tr / max(nb, 1), vl / max(vnb, 1)
        history['train_loss'].append(avg_tr)
        history['val_loss'].append(avg_val)
        history['val_pearson'].append(val_pearson)

        improved = avg_val < best_val
        if improved:
            best_val = avg_val
            best_encoder = copy.deepcopy(encoder).cpu()
        if verbose:
            star = ' *' if improved else ''
            print(f'[mel] ep {epoch+1:3d}  '
                  f'train={avg_tr:+.4f}  val={avg_val:+.4f}  '
                  f'val_r={val_pearson*100:5.2f}%{star}')

    if best_encoder is not None:
        best_encoder = best_encoder.to(device)
    return best_encoder, history

import torch.nn.functional as F

class FocalCE(nn.Module):
    def __init__(self, gamma=2.0, weight=None, ignore_index=-100):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, logits, target):
        # logits: (N, C), target: (N,)
        logp  = F.log_softmax(logits, dim=-1)
        ce    = F.nll_loss(logp, target, weight=self.weight,
                           ignore_index=self.ignore_index, reduction='none')
        # ce is -log(p_true); recover p_true
        p_true = torch.exp(-ce)
        focal  = ((1 - p_true) ** self.gamma) * ce
        # mask out ignore_index (NLL already returns 0 there in 'none' mode? Verify)
        mask = (target != self.ignore_index).float()
        return (focal * mask).sum() / mask.sum().clamp(min=1)

# Cell 18 — Stage-2 model: frozen encoder + phoneme classifier
# ============================================================
import torch
import torch.nn as nn

class PhonemeClassifierOnFrozen(nn.Module):
    """
    Wraps a frozen MelEncoder. Reads the LSTM hidden states (NOT the decoded mel,
    per the paper) and runs a small 1D-conv classifier over them.
    """
    def __init__(self, encoder, n_phoneme_classes,
                 hidden_dim=128, dropout=0.3):
        super().__init__()
        self.encoder = encoder
        # for p in self.encoder.parameters():
        #     p.requires_grad = False
        H = encoder.H
        self.classifier = nn.Sequential(
            nn.Conv1d(H, hidden_dim, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, n_phoneme_classes, kernel_size=1),
        )

    def forward(self, x):
        self.encoder.eval()
        with torch.no_grad():
            h = self.encoder.encode(x)            # (B, T, H)
        h = h.transpose(1, 2)                     # (B, H, T) for Conv1d
        logits = self.classifier(h).transpose(1, 2)   # (B, T, K)
        return {'ph_logits': logits}


def train_phoneme_classifier(model, train_loader, val_loader,
                             n_epochs, lr=1e-3, weight_decay=1e-4,
                             ph_weights=None,
                             device=None, verbose=True):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    params = list(model.classifier.parameters())
    # opt = torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-4)
    opt   = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)

    if ph_weights is not None:
        # ce = nn.CrossEntropyLoss(ignore_index=-100, weight=ph_weights.to(device))
        ce = FocalCE(gamma=2.0, weight=ph_weights.to(device) if ph_weights is not None else None,
             ignore_index=-100)
    else:
        ce = nn.CrossEntropyLoss(ignore_index=-100)

    best_val, best_model = float('inf'), None
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(n_epochs):
        model.train()
        model.encoder.eval()                    # keep encoder frozen + in eval
        tr, nb = 0.0, 0
        for batch in train_loader:
            feats = batch['feats'].to(device)
            ph_gt = batch['ph'].to(device)
            out = model(feats)
            B, T, C = out['ph_logits'].shape
            loss = ce(out['ph_logits'].reshape(B*T, C), ph_gt.reshape(-1))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            tr += loss.item(); nb += 1

        model.eval()
        vl, vnb, ok, tot = 0.0, 0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                feats = batch['feats'].to(device)
                ph_gt = batch['ph'].to(device)
                out = model(feats)
                B, T, C = out['ph_logits'].shape
                vl += ce(out['ph_logits'].reshape(B*T, C), ph_gt.reshape(-1)).item()
                vnb += 1
                pred = out['ph_logits'].argmax(dim=-1)
                valid = ph_gt != -100
                ok += ((pred == ph_gt) & valid).sum().item()
                tot += valid.sum().item()
        sched.step()

        avg_tr, avg_val = tr / max(nb,1), vl / max(vnb,1)
        acc = ok / max(tot, 1)
        history['train_loss'].append(avg_tr)
        history['val_loss'].append(avg_val)
        history['val_acc'].append(acc)

        improved = avg_val < best_val
        if improved:
            best_val = avg_val
            best_model = copy.deepcopy(model).cpu()
        if verbose:
            star = ' *' if improved else ''
            print(f'[cls] ep {epoch+1:3d}  '
                  f'train={avg_tr:.4f}  val={avg_val:.4f}  '
                  f'ph_acc={acc*100:5.2f}%{star}')

    if best_model is not None:
        best_model = best_model.to(device)
    return best_model, history

# Cell 19 — Two-stage runner per patient
# ============================================================
import gc, random, numpy as np, torch
from torch.utils.data import DataLoader

SIL = 'sil'

def run_two_stage_for_patient(pid, seed=0,
                              # data
                              n_mels=40, batch_size=8,
                              # stage 1
                              mel_epochs=60, mel_lr=1e-3,
                              hidden=64, n_layers=2, dropout=0.3,
                              # stage 2
                              cls_epochs=40, cls_lr=1e-3,
                              cls_hidden_dim=128,
                              # decoding
                              viterbi_bonus=0.3,
                              # augment (only used in stage 1 to help mel learn invariances)
                              noise_std=0.2,
                              verbose=True):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ---- data ----
    splits = build_patient_dataset_with_mel(pid, n_mels=n_mels)
    if splits is None or len(splits['train']) == 0:
        return None
    if verbose:
        print(f'[{pid}] train/val/test = '
              f'{len(splits["train"])}/{len(splits["val"])}/{len(splits["test"])}')

    scaler     = fit_scaler(splits['train'])
    mel_scaler = fit_mel_scaler(splits['train'])
    phone_to_idx, idx_to_phone = build_phone_vocab(splits['train'], sil_token=SIL)
    sil_id = phone_to_idx[SIL]

    # ────────────── build class-balanced phoneme weights ──────────────
    from collections import Counter
    ph_counts = Counter()
    for s in splits['train']:
        for it in s['mfa_intervals']:
            ph, t0, t1 = _unpack_interval(it)
            if ph not in phone_to_idx:
                continue
            n_frames = int(round((t1 - t0) * s['feature_fps']))
            ph_counts[phone_to_idx[ph]] += n_frames

    n_classes = len(phone_to_idx)
    total = sum(ph_counts.values()) or 1
    ph_weights = torch.ones(n_classes, dtype=torch.float32)
    for k in range(n_classes):
        c = ph_counts.get(k, 1)
        ph_weights[k] = total / (n_classes * c)
    ph_weights = ph_weights.clamp(0.5, 2.0)
    ph_weights[sil_id] = 0.2
    if verbose:
        print(f'[{pid}] sil weight = {ph_weights[sil_id]:.2f}, '
              f'max ph weight = {ph_weights.max():.2f}, '
              f'min ph weight = {ph_weights.min():.2f}')
    # ────────────────────────────────────────────────────────────────────────

    common = dict(scaler=scaler, mel_scaler=mel_scaler, sil_token=SIL,
                  manner_map=DUTCH_MANNER, voicing_map=DUTCH_VOICING,
                  place_map=DUTCH_PLACE)

    # ---- stage 1 datasets: light aug (helps the trunk generalize) ----
    train_ds_s1 = PhonemeDatasetMTMel(
        splits['train'], phone_to_idx, **common,
        noise_std=noise_std, augment=True,
        n_time_masks=0, time_mask_max=0,
        n_feat_masks=0, feat_mask_max_frac=0.0,
        max_time_shift=0, include_frame_labels=True,
    )
    val_ds = PhonemeDatasetMTMel(
        splits['val'], phone_to_idx, **common,
        noise_std=0.0, augment=False,
        n_time_masks=0, time_mask_max=0,
        n_feat_masks=0, feat_mask_max_frac=0.0,
        max_time_shift=0, include_frame_labels=True,
    )
    test_ds = PhonemeDatasetMTMel(
        splits['test'], phone_to_idx, **common,
        noise_std=0.0, augment=False,
        n_time_masks=0, time_mask_max=0,
        n_feat_masks=0, feat_mask_max_frac=0.0,
        max_time_shift=0, include_frame_labels=True,
    )

    train_loader_s1 = DataLoader(train_ds_s1, batch_size=batch_size, shuffle=True,
                                 collate_fn=collate_mt_mel, num_workers=0, pin_memory=True)
    val_loader      = DataLoader(val_ds,  batch_size=batch_size, shuffle=False,
                                 collate_fn=collate_mt_mel, num_workers=0, pin_memory=True)
    test_loader     = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                                 collate_fn=collate_mt_mel, num_workers=0, pin_memory=True)

    n_features = splits['train'][0]['features'].shape[1]

    # ---- STAGE 1: train mel encoder ----
    if verbose: print(f'\n=== [{pid}] STAGE 1: mel encoder ===')
    encoder = MelEncoder(n_features=n_features,
                         hidden=hidden, n_layers=n_layers,
                         dropout=dropout, n_mels=n_mels)
    encoder, hist_s1 = train_mel_encoder(
        encoder, train_loader_s1, val_loader,
        n_epochs=mel_epochs, lr=mel_lr,
        device=device, verbose=verbose,
    )
    best_mel_pearson = max(hist_s1['val_pearson'])

    # ---- STAGE 2: train phoneme classifier on FROZEN encoder ----
    if verbose: print(f'\n=== [{pid}] STAGE 2: phoneme classifier ===')
    # for stage 2 we use NO augmentation (the encoder is fixed, so noisy inputs
    # would just inject noise into the frozen features)
    train_ds_s2 = PhonemeDatasetMTMel(
        splits['train'], phone_to_idx, **common,
        noise_std=0.0, augment=False,
        n_time_masks=0, n_feat_masks=0, max_time_shift=0,
        include_frame_labels=True,
    )
    train_loader_s2 = DataLoader(train_ds_s2, batch_size=batch_size, shuffle=True,
                                 collate_fn=collate_mt_mel, num_workers=0, pin_memory=True)

    classifier = PhonemeClassifierOnFrozen(
        encoder=encoder,
        n_phoneme_classes=len(phone_to_idx),
        hidden_dim=cls_hidden_dim,
        dropout=dropout,
    )
    classifier, hist_s2 = train_phoneme_classifier(
        classifier, train_loader_s2, val_loader,
        n_epochs=cls_epochs, lr=cls_lr,
        ph_weights=None,                          # start unweighted; add if needed
        device=device, verbose=verbose,
    )

    # ---- evaluate on TEST ----
    preds, golds = decode_loader(
        classifier, test_loader,
        sil_id=sil_id, bonus=viterbi_bonus, device=device,
    )
    pred_str = [[idx_to_phone[i] for i in p] for p in preds]
    gold_str = [[idx_to_phone[i] for i in g] for g in golds]
    rates = [needleman_wunsch_match_rate(p, g) for p, g in zip(preds, golds)]
    nw    = float(np.mean(rates)) if rates else 0.0

    predictions = [
        {'sent_idx': splits['test'][i]['sent_idx'],
         'pred':     pred_str[i],
         'gold':     gold_str[i],
         'nw_rate':  rates[i]}
        for i in range(len(rates))
    ]

    if verbose:
        print(f'\n[{pid}] ─── headline ─────────────────────────')
        print(f'  Stage 1 best mel Pearson : {best_mel_pearson*100:5.2f}%')
        print(f'  Stage 2 best val ph_acc  : {max(hist_s2["val_acc"])*100:5.2f}%')
        print(f'  TEST NW match rate       : {nw*100:5.2f}%')

    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    return dict(
        pid                = pid,
        seed               = seed,
        encoder            = encoder.cpu(),
        classifier         = classifier.cpu(),
        scalers            = (scaler, mel_scaler),
        vocab              = (phone_to_idx, idx_to_phone),
        mel_pearson_best   = best_mel_pearson,
        val_ph_acc         = float(max(hist_s2['val_acc'])),
        nw_match_rate      = nw,
        nw_match_rates     = rates,
        predictions        = predictions,
        history_stage1     = hist_s1,
        history_stage2     = hist_s2,
    )

device = 'cuda' if torch.cuda.is_available() else 'cpu'
classifier = res['classifier'].to(device).eval()
phone_to_idx, idx_to_phone = res['vocab']
sil_id = phone_to_idx['sil']

# rebuild test loader (must match what was used during training)
test_ds = PhonemeDatasetMTMel(
    splits['test'], phone_to_idx, sil_token='sil',
    scaler=res['scalers'][0], mel_scaler=res['scalers'][1],
    manner_map=DUTCH_MANNER, voicing_map=DUTCH_VOICING, place_map=DUTCH_PLACE,
    noise_std=0.0, augment=False,
)
from torch.utils.data import DataLoader
test_loader = DataLoader(test_ds, batch_size=8, shuffle=False,
                         collate_fn=collate_mt_mel, num_workers=0)

for b in [0.3, 0.1, 0.0, -0.1, -0.3, -0.5]:
    preds, golds = decode_loader(classifier, test_loader,
                                 sil_id=sil_id, bonus=b, device=device)
    rates  = [needleman_wunsch_match_rate(p, g) for p, g in zip(preds, golds)]
    avg_l  = np.mean([len(p) for p in preds])
    avg_g  = np.mean([len(g) for g in golds])
    print(f'bonus={b:+.2f}  NW={np.mean(rates)*100:5.2f}%  '
          f'pred_len={avg_l:5.1f}  gold_len={avg_g:5.1f}')

res = run_two_stage_for_patient(
    'P22', seed=0,
    n_mels=20, batch_size=8,
    # ── stage 1: smaller + more regularized ──
    mel_epochs=10,                  
    mel_lr=1e-3,
    hidden=32,                      # ↓ from 64 — half the trunk capacity
    n_layers=1,                     # ↓ from 2
    dropout=0.5,                    # ↑ from 0.3
    # ── stage 2: unchanged ──
    cls_epochs=40, cls_lr=1e-3, cls_hidden_dim=128,
    viterbi_bonus=0.0,
    noise_std=0.3,                  # ↑ slightly from 0.2
)

try:
    out_viz = ctc_results_to_out(res['predictions'])
    display(HTML(
            f"<h3>{pid} — two-stage "
            f"(NW={res['nw_match_rate']*100:.1f}%, "
            f"mel r={res['mel_pearson_best']*100:.1f}%)</h3>"
        ))
    display(HTML(show_predictions_html(out_viz, label='two-stage',
                                           max_sentences=10)))
except NameError:
    # fall back to plain print if the ctc helpers aren't loaded in this notebook
    for p in res['predictions'][:5]:
        print(f"\nSentence {p['sent_idx']} (NW = {p['nw_rate']*100:.1f}%)")
        print('GOLD :', ' '.join(p['gold']))
        print('PRED :', ' '.join(p['pred']))

# ============================================================
# Cell 27 — MAE on pre-extracted features (per-patient)
# ============================================================
import torch
import torch.nn as nn

class FeatureMAE(nn.Module):
    """
    Patch-based MAE on (T, F) features.
    - Patchify time into chunks of patch_len frames.
    - Mask 50% of patches; predict masked patches from visible ones.
    - Encoder = small Transformer; decoder = linear back to patch features.
    """
    def __init__(self, n_features, patch_len=10,
                 d_model=64, n_layers=2, n_heads=4, dropout=0.3,
                 max_patches=200):
        super().__init__()
        self.n_features  = n_features
        self.patch_len   = patch_len
        self.d_model     = d_model
        self.patch_dim   = n_features * patch_len

        # input/output projections (in: flat patch -> d_model; out: d_model -> flat patch)
        self.encoder_proj = nn.Linear(self.patch_dim, d_model)
        self.decoder_proj = nn.Linear(d_model, self.patch_dim)

        # learned position embedding + mask token
        self.pos_emb    = nn.Parameter(torch.zeros(1, max_patches, d_model))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.pos_emb,    std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 2,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)

    # ── patchify / unpatchify helpers ──
    def patchify(self, x):
        """x: (B, T, F) -> (B, N, patch_len*F)"""
        B, T, F = x.shape
        N = T // self.patch_len
        x = x[:, :N * self.patch_len]
        return x.reshape(B, N, self.patch_len, F).reshape(B, N, self.patch_len * F)

    # ── training forward: mask + reconstruct ──
    def forward(self, x, mask_ratio=0.5):
        patches = self.patchify(x)                          # (B, N, patch_dim)
        B, N, _ = patches.shape
        h = self.encoder_proj(patches) + self.pos_emb[:, :N]

        # random mask
        rand = torch.rand(B, N, device=h.device)
        n_mask = max(1, int(mask_ratio * N))
        mask_idx = rand.argsort(-1)[:, :n_mask]
        mask = torch.zeros(B, N, dtype=torch.bool, device=h.device)
        mask.scatter_(1, mask_idx, True)
        h_masked = torch.where(mask.unsqueeze(-1),
                               self.mask_token.expand(B, N, -1), h)

        encoded = self.transformer(h_masked)
        pred    = self.decoder_proj(encoded)                # (B, N, patch_dim)

        # MSE only on masked patches
        diff = ((pred - patches) ** 2).mean(-1)             # (B, N)
        loss = (diff * mask.float()).sum() / mask.float().sum().clamp(min=1)
        return loss, encoded, mask

    # ── inference: encode without masking, return per-frame embeddings ──
    def encode_per_frame(self, x):
        """Returns (B, T_used, d_model) where T_used = (T // patch_len) * patch_len.
        Per-patch embedding is broadcast across its patch_len frames."""
        patches = self.patchify(x)
        B, N, _ = patches.shape
        h = self.encoder_proj(patches) + self.pos_emb[:, :N]
        encoded = self.transformer(h)                        # (B, N, d_model)
        # repeat each patch's embedding patch_len times
        return encoded.unsqueeze(2).expand(-1, -1, self.patch_len, -1) \
                      .reshape(B, N * self.patch_len, self.d_model)

    @property
    def H(self):
        return self.d_model

# ============================================================
# Cell 28 — MAE pretraining (per-patient, on train sentences)
# ============================================================
import copy

def pretrain_mae(mae, train_loader, val_loader,
                 n_epochs=80, lr=1e-3, weight_decay=1e-4,
                 mask_ratio=0.5, device=None, verbose=True):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mae = mae.to(device)
    opt   = torch.optim.AdamW(mae.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)

    best_val, best_mae = float('inf'), None
    history = {'train_loss': [], 'val_loss': []}

    for ep in range(n_epochs):
        mae.train()
        tr, nb = 0.0, 0
        for batch in train_loader:
            feats = batch['feats'].to(device)
            loss, _, _ = mae(feats, mask_ratio=mask_ratio)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(mae.parameters(), 1.0)
            opt.step()
            tr += loss.item(); nb += 1

        mae.eval()
        vl, vnb = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                feats = batch['feats'].to(device)
                loss, _, _ = mae(feats, mask_ratio=mask_ratio)
                vl += loss.item(); vnb += 1
        sched.step()

        avg_tr, avg_val = tr / max(nb, 1), vl / max(vnb, 1)
        history['train_loss'].append(avg_tr)
        history['val_loss'].append(avg_val)

        improved = avg_val < best_val
        if improved:
            best_val = avg_val
            best_mae = copy.deepcopy(mae).cpu()
        if verbose:
            star = ' *' if improved else ''
            print(f'[mae] ep {ep+1:3d}  train={avg_tr:.4f}  val={avg_val:.4f}{star}')

    if best_mae is not None:
        best_mae = best_mae.to(device)
    return best_mae, history

# ============================================================
# Cell 29 — Phoneme classifier on MAE-pretrained encoder
# ============================================================
class PhonemeClassifierOnMAE(nn.Module):
    """Reads per-frame MAE embeddings, predicts per-frame phoneme logits."""
    def __init__(self, mae, n_phoneme_classes,
                 hidden_dim=128, dropout=0.3, freeze_encoder=True):
        super().__init__()
        self.mae = mae
        self.freeze_encoder = freeze_encoder
        if freeze_encoder:
            for p in self.mae.parameters():
                p.requires_grad = False
        H = mae.H
        self.classifier = nn.Sequential(
            nn.Conv1d(H, hidden_dim, kernel_size=5, padding=2),
            nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, n_phoneme_classes, kernel_size=1),
        )

    def forward(self, x):
        if self.freeze_encoder:
            self.mae.eval()
            with torch.no_grad():
                h = self.mae.encode_per_frame(x)
        else:
            h = self.mae.encode_per_frame(x)
        # h: (B, T_used, d_model) where T_used = (T // patch_len) * patch_len
        h = h.transpose(1, 2)                  # (B, d_model, T_used)
        logits = self.classifier(h).transpose(1, 2)   # (B, T_used, K)
        return {'ph_logits': logits, 'T_used': h.shape[-1]}

# ============================================================
# Cell 30 — Full MAE → Phoneme runner per patient
# ============================================================
from torch.utils.data import DataLoader
import random, gc, numpy as np
from collections import Counter

def run_mae_phoneme_for_patient(pid, seed=0,
                                # MAE pretrain
                                mae_epochs=80, mae_lr=1e-3,
                                patch_len=10, d_model=64, n_layers=2,
                                n_heads=4, dropout=0.3,
                                mask_ratio=0.5,
                                # downstream
                                cls_epochs=40, cls_lr=1e-3,
                                cls_hidden_dim=128,
                                freeze_encoder=True,
                                # general
                                batch_size=8, n_mels=40,
                                noise_std=0.2,
                                viterbi_bonus=0.3,
                                verbose=True):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    splits = build_patient_dataset_with_words(pid, n_mels=n_mels)
    if splits is None or len(splits['train']) == 0:
        return None
    if verbose:
        print(f'[{pid}] train/val/test = '
              f'{len(splits["train"])}/{len(splits["val"])}/{len(splits["test"])}')

    scaler     = fit_scaler(splits['train'])
    mel_scaler = fit_mel_scaler(splits['train'])
    phone_to_idx, idx_to_phone = build_phone_vocab(splits['train'], sil_token='sil')
    sil_id = phone_to_idx['sil']

    # ── phoneme weights ──
    pc = Counter()
    for s in splits['train']:
        for it in s['mfa_intervals']:
            ph, t0, t1 = _unpack_interval(it)
            if ph not in phone_to_idx: continue
            pc[phone_to_idx[ph]] += int(round((t1 - t0) * s['feature_fps']))
    nC = len(phone_to_idx); tot = sum(pc.values()) or 1
    ph_weights = torch.ones(nC)
    for k in range(nC): ph_weights[k] = tot / (nC * max(pc.get(k, 1), 1))
    ph_weights = ph_weights.clamp(0.5, 2.0); ph_weights[sil_id] = 0.2

    common = dict(scaler=scaler, mel_scaler=mel_scaler, sil_token='sil',
                  manner_map=DUTCH_MANNER, voicing_map=DUTCH_VOICING,
                  place_map=DUTCH_PLACE)

    # ── STAGE 1: MAE pretrain ──
    # Use existing dataset/collate (only `feats` is needed for MAE).
    train_ds = PhonemeDatasetMTMel(splits['train'], phone_to_idx, **common,
                                   noise_std=noise_std, augment=True,
                                   include_frame_labels=True)
    val_ds   = PhonemeDatasetMTMel(splits['val'],   phone_to_idx, **common,
                                   noise_std=0.0, augment=False,
                                   include_frame_labels=True)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_mt_mel, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              collate_fn=collate_mt_mel, num_workers=0)

    n_features = splits['train'][0]['features'].shape[1]
    if verbose: print(f'\n=== [{pid}] STAGE 1: MAE pretrain ===')
    mae = FeatureMAE(n_features=n_features, patch_len=patch_len,
                     d_model=d_model, n_layers=n_layers, n_heads=n_heads,
                     dropout=dropout)
    mae, hist_mae = pretrain_mae(mae, train_loader, val_loader,
                                 n_epochs=mae_epochs, lr=mae_lr,
                                 mask_ratio=mask_ratio,
                                 device=device, verbose=verbose)

    # ── STAGE 2: phoneme classifier ──
    if verbose: print(f'\n=== [{pid}] STAGE 2: phoneme classifier (frozen={freeze_encoder}) ===')
    test_ds = PhonemeDatasetMTMel(splits['test'], phone_to_idx, **common,
                                  noise_std=0.0, augment=False,
                                  include_frame_labels=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             collate_fn=collate_mt_mel, num_workers=0)

    classifier = PhonemeClassifierOnMAE(mae=mae, n_phoneme_classes=nC,
                                        hidden_dim=cls_hidden_dim,
                                        dropout=dropout,
                                        freeze_encoder=freeze_encoder).to(device)

    if freeze_encoder:
        params = list(classifier.classifier.parameters())
    else:
        params = list(classifier.parameters())
    opt   = torch.optim.AdamW(params, lr=cls_lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cls_epochs)
    ce = nn.CrossEntropyLoss(ignore_index=-100, weight=ph_weights.to(device))

    best_val, best_classifier = float('inf'), None
    hist_cls = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    for ep in range(cls_epochs):
        classifier.train()
        if freeze_encoder: classifier.mae.eval()
        tr, nb = 0.0, 0
        for batch in train_loader:
            feats = batch['feats'].to(device)
            ph_gt = batch['ph'].to(device)
            out = classifier(feats)
            B, T_used, C = out['ph_logits'].shape
            # trim phoneme labels to T_used (MAE patchifies, may drop a few frames)
            ph_trim = ph_gt[:, :T_used]
            loss = ce(out['ph_logits'].reshape(B*T_used, C), ph_trim.reshape(-1))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            tr += loss.item(); nb += 1

        classifier.eval()
        vl, vnb, ok, totv = 0.0, 0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                feats = batch['feats'].to(device)
                ph_gt = batch['ph'].to(device)
                out = classifier(feats)
                B, T_used, C = out['ph_logits'].shape
                ph_trim = ph_gt[:, :T_used]
                vl += ce(out['ph_logits'].reshape(B*T_used, C), ph_trim.reshape(-1)).item()
                vnb += 1
                pred  = out['ph_logits'].argmax(-1)
                valid = ph_trim != -100
                ok   += ((pred == ph_trim) & valid).sum().item()
                totv += valid.sum().item()
        sched.step()

        avg_tr, avg_val = tr/max(nb,1), vl/max(vnb,1)
        acc = ok/max(totv,1)
        hist_cls['train_loss'].append(avg_tr)
        hist_cls['val_loss'].append(avg_val)
        hist_cls['val_acc'].append(acc)

        improved = avg_val < best_val
        if improved:
            best_val = avg_val
            best_classifier = copy.deepcopy(classifier).cpu()
        if verbose:
            star = ' *' if improved else ''
            print(f'[cls] ep {ep+1:3d}  train={avg_tr:.4f}  val={avg_val:.4f}  '
                  f'ph_acc={acc*100:5.2f}%{star}')

    if best_classifier is not None:
        classifier = best_classifier.to(device)

    # ── decode test ──
    classifier.eval()
    all_pred, all_gold = [], []
    with torch.no_grad():
        for batch in test_loader:
            feats   = batch['feats'].to(device)
            out     = classifier(feats)
            T_used  = out['T_used']
            logp    = torch.log_softmax(out['ph_logits'], dim=-1).cpu().numpy()
            ph_gt   = batch['ph'].cpu().numpy()
            in_lens = batch['in_lens'].cpu().numpy()
            for b in range(logp.shape[0]):
                T = min(int(in_lens[b]), T_used)
                states = viterbi_with_self_loop(logp[b, :T], bonus=viterbi_bonus)
                pred   = collapse_repeats(states, sil_id=sil_id)
                gf     = ph_gt[b, :T]; gf = gf[gf != -100]
                gold   = collapse_repeats(gf, sil_id=sil_id)
                all_pred.append(pred); all_gold.append(gold)

    pred_str = [[idx_to_phone[i] for i in p] for p in all_pred]
    gold_str = [[idx_to_phone[i] for i in g] for g in all_gold]
    rates    = [needleman_wunsch_match_rate(p, g) for p, g in zip(all_pred, all_gold)]
    nw       = float(np.mean(rates)) if rates else 0.0
    flat = sum(all_pred, [])
    ctr  = Counter(idx_to_phone[i] for i in flat)

    if verbose:
        print(f'\n[{pid}] MAE → PHONEME ─── headline ───')
        print(f'  MAE best val recon loss : {min(hist_mae["val_loss"]):.4f}')
        print(f'  Stage 2 best val ph_acc : {max(hist_cls["val_acc"])*100:5.2f}%')
        print(f'  TEST NW match rate      : {nw*100:5.2f}%')
        print(f'  Distinct phonemes used  : {len(ctr)} / {nC}')
        print(f'  Top 5: {ctr.most_common(5)}')

    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    return dict(
        pid=pid, seed=seed,
        mae=mae.cpu(), classifier=classifier.cpu(),
        scalers=(scaler, mel_scaler), vocab=(phone_to_idx, idx_to_phone),
        nw_match_rate=nw, nw_match_rates=rates,
        predictions=[{'sent_idx': splits['test'][i]['sent_idx'],
                      'pred': pred_str[i], 'gold': gold_str[i],
                      'nw_rate': rates[i]} for i in range(len(rates))],
        history_mae=hist_mae, history_cls=hist_cls,
    )
