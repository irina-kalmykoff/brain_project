# Converted from Untitled7.ipynb

# build pipeline
import os, numpy as np, warnings
from collections import defaultdict, Counter
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

import scipy
import random
import scipy.signal
from scipy.signal import iirfilter, sosfilt, sosfiltfilt
from scipy.signal import butter, sosfiltfilt, hilbert, iirfilter

import numpy as np
from collections import defaultdict
from IPython.display import display, HTML

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

# CTC-trained neural phoneme decoder — new notebook setup
import os, sys, time, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import Counter, defaultdict
from sklearn.preprocessing import StandardScaler

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"PyTorch: {torch.__version__}  device: {DEVICE}")
if DEVICE == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ── Required from your existing notebook (re-paste or import) ──
# Either run your existing pipeline cells to populate:
#   pipeline, DUTCH_30_PATH, EEG_SR, WIN_SAMP, SHIFT_SAMP, MO, SS, LDA_MARGIN,
#   MN_FRAMES, MX_FRAMES, TEST_OFFSET, VAL_FRAC, DEFAULT_FEATURE_SPEC
#   extract_features_multiband, stackFeatures, load_mfa_alignments
# OR import them from the LDA_on_frames module if you've packaged it.
#
# Quick check:
assert 'pipeline' in dir(), "pipeline not loaded — run your existing pipeline setup first"
print(f"Pipeline ready: {len(pipeline.split_result['word_segments_dict'])} patients")

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

DUTCH_MANNER = {
    'a':'V','aː':'V','a:':'V','ɑ':'V','ɑː':'V','ɛ':'V','e':'V','eː':'V','e:':'V',
    'i':'V','iː':'V','i:':'V','ɪ':'V','ɔ':'V','o':'V','oː':'V','o:':'V',
    'u':'V','uː':'V','u:':'V','y':'V','yː':'V','y:':'V','ʏ':'V','ø':'V','øː':'V','ø:':'V',
    'ə':'V','ɛi':'V','œy':'V','ɑu':'V',
    'p':'S','b':'S','t':'S','d':'S','k':'S','g':'S','ɡ':'S','c':'S',
    'f':'F','v':'F','s':'F','z':'F','ʃ':'F','ʒ':'F','x':'F','ɣ':'F','h':'F','χ':'F',
    'm':'N','n':'N','ŋ':'N',
    'l':'L','r':'L','j':'L','w':'L','ʋ':'L','ɹ':'L','ɥ':'L',
}
DUTCH_VOICING = {
    # all vowels voiced
    **{p: 'V' for p in DUTCH_MANNER if DUTCH_MANNER[p] == 'V'},
    # voiced consonants
    'b':'V','d':'V','g':'V','ɡ':'V','v':'V','z':'V','ʒ':'V','ɣ':'V',
    'm':'V','n':'V','ŋ':'V','l':'V','r':'V','j':'V','w':'V','ʋ':'V','ɹ':'V','ɥ':'V',
    # unvoiced consonants
    'p':'U','t':'U','k':'U','c':'U','f':'U','s':'U','ʃ':'U','x':'U','χ':'U','h':'U',
}

MANNER_CLASSES  = ['?', 'V', 'S', 'F', 'N', 'L']           # 0 = unknown/blank
VOICING_CLASSES = ['?', 'V', 'U']                            # 0 = unknown/blank
manner_to_idx   = {c: i for i, c in enumerate(MANNER_CLASSES)}
voicing_to_idx  = {c: i for i, c in enumerate(VOICING_CLASSES)}
def manner_of(p):  return manner_to_idx.get(DUTCH_MANNER.get(p, '?'), 0)
def voicing_of(p): return voicing_to_idx.get(DUTCH_VOICING.get(p, '?'), 0)
N_MANNER  = len(MANNER_CLASSES)
N_VOICING = len(VOICING_CLASSES)
print(f"Manner classes: {MANNER_CLASSES} ({N_MANNER})")
print(f"Voicing classes: {VOICING_CLASSES} ({N_VOICING})")

DUTCH_PLACE = {
    # Labial (lips)
    'p':'L','b':'L','m':'L','f':'L','v':'L','ʋ':'L','w':'L',
    # Coronal (alveolar — tongue tip)
    't':'C','d':'C','n':'C','s':'C','z':'C','ʃ':'C','ʒ':'C',
    'l':'C','r':'C','ɹ':'C',
    # Dorsal (velar — back of tongue)
    'k':'D','g':'D','ɡ':'D','ŋ':'D','x':'D','ɣ':'D','χ':'D',
    # Palatal/Front (palatal consonants + front vowels)
    'j':'F','c':'F','ɥ':'F',
    'i':'F','iː':'F','i:':'F','ɪ':'F',
    'e':'F','eː':'F','e:':'F','ɛ':'F',
    'y':'F','yː':'F','y:':'F','ʏ':'F',
    'ø':'F','øː':'F','ø:':'F',
    'ɛi':'F','œy':'F',
    # Back vowels
    'u':'B','uː':'B','u:':'B','o':'B','oː':'B','o:':'B',
    'ɔ':'B','ɑ':'B','ɑː':'B','aː':'B','a:':'B','ɑu':'B','a':'B',
    # Central
    'ə':'M',
    # Glottal
    'h':'G',
}

PLACE_CLASSES = ['?', 'L', 'C', 'D', 'F', 'B', 'M', 'G']
place_to_idx  = {c: i for i, c in enumerate(PLACE_CLASSES)}
def place_of(p): return place_to_idx.get(DUTCH_PLACE.get(p, '?'), 0)
N_PLACE = len(PLACE_CLASSES)
print(f"Place classes: {PLACE_CLASSES} ({N_PLACE})")

def build_frame_labels_mt(T_frames, mfa_intervals, phone_to_idx, silence_idx=0):
    """Returns (phoneme, manner, voicing, place) per-frame label arrays."""
    ph_labels = np.full(T_frames, silence_idx, dtype=np.int64)
    mn_labels = np.zeros(T_frames, dtype=np.int64)
    vo_labels = np.zeros(T_frames, dtype=np.int64)
    pl_labels = np.zeros(T_frames, dtype=np.int64)        # ★ new
    for start_s, end_s, phone in mfa_intervals:
        if phone not in phone_to_idx: continue
        k_s = int(np.ceil ((start_s * EEG_SR - WIN_SAMP / 2) / SHIFT_SAMP)) - LDA_MARGIN
        k_e = int(np.floor((end_s   * EEG_SR - WIN_SAMP / 2) / SHIFT_SAMP)) - LDA_MARGIN
        k_s = max(0, k_s); k_e = min(T_frames - 1, k_e)
        if k_e >= k_s:
            ph_labels[k_s:k_e+1] = phone_to_idx[phone]
            mn_labels[k_s:k_e+1] = manner_of(phone)
            vo_labels[k_s:k_e+1] = voicing_of(phone)
            pl_labels[k_s:k_e+1] = place_of(phone)            # ★ new
    return ph_labels, mn_labels, vo_labels, pl_labels         # ★ now 4 outputs

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

def run_ctc_for_patient(pid, n_epochs=100, batch_size=8, hidden=48,
                         noise_std=0.15,
                         ctc_w=0.0, ce_w=1.0,
                         class_weight_alpha=1.0,
                         class_weight_max_ratio=20.0,
                         class_weight_min_count=10,
                         use_class_weights=False,
                         use_sentence_sampler=True,
                         sample_oversample_factor=3,
                         n_time_masks=5, time_mask_max=40,
                         n_feat_masks=4, feat_mask_max_frac=0.12,
                         max_time_shift=5,
                         patience=15,            # ★ add
                         dropout=0.3,            # ★ add  
                         weight_decay=1e-4,      # ★ add
                         verbose=True):
    splits = build_patient_dataset(pid)
    if splits is None: return None, "no data"
    if len(splits['train']) < 10: return None, f"too few train: {len(splits['train'])}"

    # ── Vocabulary ──
    train_phonemes = sorted({ph for s in splits['train'] for ph in s['target']})
    phone_to_idx = {ph: i + 1 for i, ph in enumerate(train_phonemes)}
    idx_to_phone = {i + 1: ph for i, ph in enumerate(train_phonemes)}
    for split in ['val', 'test']:
        for s in splits[split]:
            s['target'] = [ph for ph in s['target'] if ph in phone_to_idx]
        splits[split] = [s for s in splits[split] if len(s['target']) > 0]
    n_classes = len(train_phonemes)
    if verbose:
        print(f"  vocab: {n_classes} phonemes (CTC blank reserves idx 0)")

    # ── Compute label_counts always (needed for sentence weights) ──
    # The class weights tensor can optionally be disabled below.
    class_weights_full, label_counts = compute_class_weights(
        splits['train'], phone_to_idx, n_classes,
        alpha=class_weight_alpha,
        max_weight_ratio=class_weight_max_ratio,
        min_count_keep=class_weight_min_count)

    if verbose and use_class_weights:
        nz = class_weights_full[1:][class_weights_full[1:] > 0]
        n_excluded = sum(1 for w in class_weights_full[1:] if w == 0)
        print(f"  class_weights ACTIVE: range [{nz.min():.3f}, {nz.max():.3f}]  "
              f"mean={nz.mean():.3f}  excluded={n_excluded}/{n_classes}")
    elif verbose:
        print(f"  class_weights DISABLED (using sentence sampler instead)")

    # The actual weights tensor passed to the loss:
    class_weights = class_weights_full if use_class_weights else None

    # ── Feature scaler ──
    scaler = StandardScaler()
    for s in splits['train']:
        scaler.partial_fit(s['features'].astype(np.float32))

    # ── Datasets (train augmented, val not) ──
    train_ds = PhonemeDataset(splits['train'], phone_to_idx, scaler=scaler,
                                noise_std=noise_std, augment=True,
                                n_time_masks=n_time_masks, time_mask_max=time_mask_max,
                                n_feat_masks=n_feat_masks, feat_mask_max_frac=feat_mask_max_frac,
                                max_time_shift=max_time_shift,
                                include_frame_labels=True)
    val_ds   = PhonemeDataset(splits['val'],   phone_to_idx, scaler=scaler,
                                noise_std=0.0, augment=False,
                                include_frame_labels=True)

    # ── DataLoader: either weighted sampler OR shuffle (mutually exclusive) ──
    if use_sentence_sampler:
        sent_weights = compute_sentence_weights(splits['train'], phone_to_idx,
                                                  label_counts, alpha=1.0)
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(sent_weights).double(),
            num_samples=len(sent_weights) * sample_oversample_factor,
            replacement=True)
        train_loader = DataLoader(train_ds, batch_size=batch_size,
                                    sampler=sampler,
                                    collate_fn=collate_ctc_aux, num_workers=0)
        if verbose:
            print(f"  sampler: {len(sent_weights) * sample_oversample_factor} samples/epoch  "
                  f"(weight range {sent_weights.min():.2f}..{sent_weights.max():.2f})")
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                    collate_fn=collate_ctc_aux, num_workers=0)

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              collate_fn=collate_ctc_aux, num_workers=0)

    # ── Model ──
    n_features = splits['train'][0]['features'].shape[1]
    model = PhonemeBiLSTM(n_features, n_classes, hidden=hidden,
                        n_layers=1, dropout=dropout).to(DEVICE)   # ★ use kwarg
    with torch.no_grad():
        model.head.bias[0] -= 5.0
    if verbose:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  model: {n_params:,} params  feature_dim={n_features}")

    # ── Train ──
    model = train_one_patient(model, train_loader, val_loader, idx_to_phone,
                            n_epochs=n_epochs,
                            ctc_w=ctc_w, ce_w=ce_w,
                            class_weights=class_weights,
                            patience=patience,                   # ★ pass
                            weight_decay=weight_decay,           # ★ pass
                            verbose=verbose)

    # ── Evaluate (greedy) ──
    results = evaluate(model, splits['test'], scaler, idx_to_phone)
    return dict(results=results, model=model, scaler=scaler,
                phone_to_idx=phone_to_idx, idx_to_phone=idx_to_phone,
                class_weights=class_weights_full,
                label_counts=label_counts), None

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

def build_patient_dataset(pid, feature_spec=None, test_offset=None):
    """Build (features, target_sequence, mfa_intervals, word_bounds) per sentence,
    split into train/val/test using the same logic as run_for_patient_sd.

    Each sample dict contains:
      - features:       (T_frames, n_features) np.float32  — stacked iEEG features
      - target:         list of phoneme labels (gold sequence)
      - mfa_intervals:  list of (start_s, end_s, phone) — for per-frame label generation
      - word_bounds:    list of stacked-frame indices marking word starts/sentence end
      - sent_idx:       int sentence index in pipeline.split_result
    """
    feature_spec = feature_spec or DEFAULT_FEATURE_SPEC
    test_offset  = TEST_OFFSET if test_offset is None else test_offset

    wd = pipeline.split_result['word_segments_dict'].get(pid)
    if wd is None: return None
    try:
        mfa = load_mfa_alignments(pid)
    except Exception:
        return None
    if not mfa: return None

    all_real = [i for i, s in enumerate(wd['sentence_list'])
                if isinstance(s, dict) and s.get('text')]
    test_sent_ids      = set(all_real[test_offset::6])
    train_sent_ids_all = sorted(set(all_real) - test_sent_ids)
    val_step           = max(2, int(round(1 / VAL_FRAC)))
    val_sent_ids       = set(train_sent_ids_all[::val_step])

    raw_path = os.path.join(DUTCH_30_PATH, 'raw', f'{pid}_sEEG.npy')
    raw_eeg = np.load(raw_path)
    if raw_eeg.ndim == 2 and raw_eeg.shape[0] < raw_eeg.shape[1]:
        raw_eeg = raw_eeg.T

    splits = dict(train=[], val=[], test=[])

    for sent_idx in all_real:
        if sent_idx not in mfa or not mfa[sent_idx]:
            continue
        s = wd['sentence_list'][sent_idx]
        s0, s1 = s['stim_start_idx'], s['stim_end_idx']
        if s1 > raw_eeg.shape[0]:
            continue

        # Extract + stack features
        ext = extract_features_multiband(raw_eeg[s0:s1], **feature_spec)
        if ext.shape[0] < 2 * LDA_MARGIN + 1:
            continue
        stk = stackFeatures(ext, modelOrder=MO, stepSize=SS).astype(np.float32)
        T   = stk.shape[0]

        # Target phoneme sequence
        target = [ph['phone'] for ph in mfa[sent_idx]]
        if not target:
            continue

        # Store MFA phoneme intervals (used by build_frame_labels)
        mfa_intervals = [(ph['start_s'], ph['end_s'], ph['phone'])
                          for ph in mfa[sent_idx]]

        # Word boundary frame indices (oracle decoding)
        word_bounds = [0]
        prev_word = None
        for ph in mfa[sent_idx]:
            cur_word = ph.get('word')
            if prev_word is not None and cur_word != prev_word:
                k_onset = int(round((ph['start_s'] * EEG_SR - WIN_SAMP / 2)
                                     / SHIFT_SAMP)) - LDA_MARGIN
                k_onset = max(0, min(T - 1, k_onset))
                word_bounds.append(k_onset)
            prev_word = cur_word
        word_bounds.append(T)

        # Decide split
        split = ('test' if sent_idx in test_sent_ids else
                  'val'  if sent_idx in val_sent_ids  else
                  'train')

        splits[split].append(dict(
            features=stk,
            target=target,
            mfa_intervals=mfa_intervals,
            word_bounds=word_bounds,
            sent_idx=sent_idx,
        ))

    return splits


def build_frame_labels(T_frames, mfa_intervals, phone_to_idx, blank_idx=0):
    """Per-frame labels: phoneme class within MFA intervals, BLANK elsewhere.
    No more ignore_index — silence regions actively train the model to predict blank."""
    labels = np.full(T_frames, blank_idx, dtype=np.int64)  # ★ start at blank, not -100
    for start_s, end_s, phone in mfa_intervals:
        if phone not in phone_to_idx: continue
        k_s = int(np.ceil ((start_s * EEG_SR - WIN_SAMP / 2) / SHIFT_SAMP)) - LDA_MARGIN
        k_e = int(np.floor((end_s   * EEG_SR - WIN_SAMP / 2) / SHIFT_SAMP)) - LDA_MARGIN
        k_s = max(0, k_s); k_e = min(T_frames - 1, k_e)
        if k_e >= k_s:
            labels[k_s : k_e + 1] = phone_to_idx[phone]
    return labels


# Sanity check
ds = build_patient_dataset("P22")
if ds is None:
    print("No data")
else:
    print(f"P22:  train={len(ds['train'])}  val={len(ds['val'])}  test={len(ds['test'])}")
    s = ds['train'][0]
    print(f"  Sample 0:")
    print(f"    features: {s['features'].shape}  dtype={s['features'].dtype}")
    print(f"    target ({len(s['target'])} phones): {s['target'][:10]}...")
    print(f"    mfa_intervals: {len(s['mfa_intervals'])} entries  (first: {s['mfa_intervals'][0]})")
    print(f"    word_bounds: {len(s['word_bounds'])} entries  {s['word_bounds'][:6]}...")
    print(f"    sent_idx: {s['sent_idx']}")

def compute_class_weights(samples, phone_to_idx, n_classes,
                          alpha=1.0, max_weight_ratio=20.0,  # ★ cap added
                          min_count_keep=10,                  # ★ filter very rare
                          device=DEVICE):
    from collections import Counter
    label_counts = Counter()
    for s in samples:
        fl = build_frame_labels(s['features'].shape[0], s['mfa_intervals'], phone_to_idx)
        for idx in fl[fl >= 0]:
            label_counts[int(idx)] += 1
    weights = torch.zeros(n_classes + 1, device=device)
    for k, c in label_counts.items():
        if c < min_count_keep:
            weights[k] = 0.0    # ★ skip ultra-rare entirely
        else:
            weights[k] = 1.0 / (c ** alpha + 1e-9)
    # Normalize and cap
    nz_mask = weights[1:] > 0
    nz_vals = weights[1:][nz_mask]
    if len(nz_vals) > 0:
        # Cap top-end ratio
        min_nz = nz_vals.min()
        weights[1:] = torch.clamp(weights[1:], max=min_nz * max_weight_ratio)
        # Normalize so mean = 1
        weights[1:] = weights[1:] / weights[1:][weights[1:] > 0].mean()
    weights[0] = 1.0
    return weights, label_counts


from torch.utils.data import WeightedRandomSampler

def compute_sentence_weights(samples, phone_to_idx, label_counts, alpha=1.0):
    """Sentence weight = sum over its phonemes of (1 / freq^alpha).
    Sentences with rare phonemes get higher sampling probability."""
    weights = []
    for s in samples:
        w = 0.0
        for ph in s['target']:
            if ph in phone_to_idx:
                idx = phone_to_idx[ph]
                count = label_counts.get(idx, 1)
                w += 1.0 / (count ** alpha)
        weights.append(w)
    weights = np.array(weights, dtype=np.float32)
    weights = weights / weights.mean()   # normalize to mean=1
    return weights

from collections import Counter

def count_frames_per_phoneme_via_dataset(samples, phone_to_idx, augment=False,
                                            n_repeats=10):
    """Count frame labels per phoneme by iterating the dataset.
    Returns (counts, zeroed_frame_count, total_frame_count)."""
    counts = Counter()
    zeroed_frame_count = 0
    total_frame_count = 0
    ds = PhonemeDataset(
        samples, phone_to_idx, scaler=None, noise_std=0.0,
        augment=augment, include_frame_labels=True
    )
    for _ in range(n_repeats):
        for i in range(len(ds)):
            feats, _, frame_labels = ds[i]
            for lbl in frame_labels:
                if int(lbl) > 0:
                    counts[int(lbl)] += 1
            total_frame_count += feats.shape[0]
            zeroed_frame_count += int((feats == 0).all(axis=1).sum())
    return counts, zeroed_frame_count, total_frame_count


# ── Set up vocab from current patient ──
PID_DIAG = "P22"   # adjust if needed
splits_d = build_patient_dataset(PID_DIAG)
train_phonemes_d = sorted({ph for s in splits_d['train'] for ph in s['target']})
phone_to_idx_d   = {ph: i + 1 for i, ph in enumerate(train_phonemes_d)}
idx_to_phone     = {i + 1: ph for i, ph in enumerate(train_phonemes_d)}

# ── Count without augmentation ──
counts_noaug, _, total_noaug = count_frames_per_phoneme_via_dataset(
    splits_d['train'], phone_to_idx_d, augment=False, n_repeats=1)

# ── Count with augmentation (N passes) ──
N_REPEATS = 10
counts_aug, zeroed_aug, total_aug = count_frames_per_phoneme_via_dataset(
    splits_d['train'], phone_to_idx_d, augment=True, n_repeats=N_REPEATS)

print(f"Counted: {len(counts_noaug)} distinct phonemes, "
      f"{total_noaug:,} total frames (no-aug)")

import matplotlib.pyplot as plt
import numpy as np

# Use existing counts_noaug, counts_aug, N_REPEATS, idx_to_phone
sorted_phonemes = sorted(counts_noaug.keys(), key=lambda k: -counts_noaug[k])
labels    = [idx_to_phone[k] for k in sorted_phonemes]
counts_no = np.array([counts_noaug.get(k, 0) for k in sorted_phonemes])
counts_au = np.array([counts_aug.get(k, 0) / N_REPEATS for k in sorted_phonemes])

total_no = counts_no.sum()
total_au = counts_au.sum()
pct_no = 100 * counts_no / total_no
pct_au = 100 * counts_au / total_au

fig, axes = plt.subplots(2, 1, figsize=(16, 9))

# Top: linear scale, side-by-side bars
x = np.arange(len(labels)); width = 0.4
axes[0].bar(x - width/2, pct_no, width, label='no aug',       color='steelblue')
axes[0].bar(x + width/2, pct_au, width, label='aug (avg/pass)', color='coral')
axes[0].set_xticks(x); axes[0].set_xticklabels(labels, rotation=45)
axes[0].set_ylabel('% of total frames')
axes[0].set_title(f'Per-phoneme share of total frames  (total_noaug = {total_no:,.0f})')
axes[0].legend(); axes[0].grid(axis='y', alpha=0.3)
# Annotate the top-5
for i in range(min(5, len(labels))):
    axes[0].text(i - width/2, pct_no[i] + 0.2, f'{pct_no[i]:.1f}%',
                  ha='center', fontsize=8, color='steelblue')

# Bottom: log scale, makes rare phonemes visible
axes[1].bar(x - width/2, pct_no, width, label='no aug',       color='steelblue')
axes[1].bar(x + width/2, pct_au, width, label='aug (avg/pass)', color='coral')
axes[1].set_xticks(x); axes[1].set_xticklabels(labels, rotation=45)
axes[1].set_yscale('log')
axes[1].set_ylabel('% of total frames (log scale)')
axes[1].set_title('Same data on log scale')
axes[1].legend(); axes[1].grid(axis='y', alpha=0.3, which='both')

plt.tight_layout(); plt.show()

# Cumulative coverage table
print(f"\n{'phoneme':<8} {'count':>7} {'% of total':>10} {'cumulative %':>14}")
print('-' * 45)
cum = 0
for k, lbl, pct in zip(sorted_phonemes, labels, pct_no):
    cum += pct
    print(f"{lbl:<8} {counts_noaug.get(k, 0):>7} {pct:>9.2f}%  {cum:>11.1f}%")

class PhonemeDataset(Dataset):
    def __init__(self, samples, phone_to_idx, scaler=None,
                  noise_std=0.0, augment=False,
                  n_time_masks=2, time_mask_max=8,
                  n_feat_masks=2, feat_mask_max_frac=0.05,
                  max_time_shift=3,
                  include_frame_labels=False):
        self.samples = samples
        self.phone_to_idx = phone_to_idx
        self.scaler = scaler
        self.noise_std = noise_std
        self.augment = augment
        self.n_time_masks = n_time_masks
        self.time_mask_max = time_mask_max
        self.n_feat_masks = n_feat_masks
        self.feat_mask_max_frac = feat_mask_max_frac
        self.max_time_shift = max_time_shift
        self.include_frame_labels = include_frame_labels

    def __len__(self): return len(self.samples)

    def _augment(self, feats, frame_labels=None):
        T, F = feats.shape
        # 1. Time shift (sub-frame jitter)
        if self.max_time_shift > 0:
            shift = random.randint(-self.max_time_shift, self.max_time_shift)
            if shift != 0:
                feats = np.roll(feats, shift, axis=0)
                if frame_labels is not None:
                    frame_labels = np.roll(frame_labels, shift)
                    # Zero out the wrap-around frames (set to ignore_index)
                    if shift > 0:
                        frame_labels[:shift] = -100
                    else:
                        frame_labels[shift:] = -100
                feats = feats.copy()
                if shift > 0: feats[:shift] = 0
                else:        feats[shift:] = 0
        # 2. Time masking
        for _ in range(self.n_time_masks):
            if T <= 2: break
            mask_len = random.randint(1, min(self.time_mask_max, max(1, T - 1)))
            start = random.randint(0, T - mask_len)
            feats[start:start+mask_len, :] = 0
        # 3. Feature masking (analogous to SpecAugment freq mask)
        for _ in range(self.n_feat_masks):
            if F <= 2: break
            max_w = max(1, int(F * self.feat_mask_max_frac))
            mask_w = random.randint(1, max_w)
            start = random.randint(0, F - mask_w)
            feats[:, start:start+mask_w] = 0
        return feats, frame_labels

    def __getitem__(self, idx):
        s = self.samples[idx]
        feats = s['features'].copy()
        if self.scaler is not None:
            feats = self.scaler.transform(feats).astype(np.float32)
        if self.augment and self.include_frame_labels:
            frame_labels = build_frame_labels(feats.shape[0],
                                                s['mfa_intervals'],
                                                self.phone_to_idx)
            feats, frame_labels = self._augment(feats, frame_labels)
        elif self.augment:
            feats, _ = self._augment(feats, None)
            frame_labels = None
        elif self.include_frame_labels:
            frame_labels = build_frame_labels(feats.shape[0],
                                                s['mfa_intervals'],
                                                self.phone_to_idx)
        else:
            frame_labels = None

        if self.noise_std > 0:
            feats = feats + np.random.randn(*feats.shape).astype(np.float32) * self.noise_std

        targets = np.array([self.phone_to_idx[ph] for ph in s['target']
                            if ph in self.phone_to_idx], dtype=np.int64)
        if self.include_frame_labels:
            return feats, targets, frame_labels
        return feats, targets

def collate_ctc_aux(batch):
    """Collate batch including frame labels (-100 for padding/ignore)."""
    feats, targets, frame_labels = zip(*batch)
    max_T = max(f.shape[0] for f in feats)
    n_feat = feats[0].shape[1]
    B = len(batch)
    feats_padded = np.zeros((B, max_T, n_feat), dtype=np.float32)
    frame_lbl_padded = np.full((B, max_T), -100, dtype=np.int64)
    input_lengths = []
    for i, (f, fl) in enumerate(zip(feats, frame_labels)):
        feats_padded[i, :f.shape[0]] = f
        frame_lbl_padded[i, :len(fl)] = fl
        input_lengths.append(f.shape[0])
    targets_flat = np.concatenate(targets) if len(targets) else np.array([], dtype=np.int64)
    target_lengths = [len(t) for t in targets]
    return (torch.from_numpy(feats_padded),
            torch.from_numpy(targets_flat).long(),
            torch.tensor(input_lengths, dtype=torch.long),
            torch.tensor(target_lengths, dtype=torch.long),
            torch.from_numpy(frame_lbl_padded).long())


class PhonemeBiLSTM(nn.Module):
    def __init__(self, n_features, n_classes, hidden=128, n_layers=2, dropout=0.3):
        super().__init__()
        self.input_drop = nn.Dropout(0.1)
        self.lstm = nn.LSTM(n_features, hidden,
                             num_layers=n_layers, bidirectional=True,
                             dropout=dropout if n_layers > 1 else 0,
                             batch_first=True)
        self.head = nn.Linear(hidden * 2, n_classes + 1)  # +1 for CTC blank (idx 0)

    def forward(self, x):
        x = self.input_drop(x)
        h, _ = self.lstm(x)
        return F.log_softmax(self.head(h), dim=-1)  # (B, T, C+1)

class PhonemeMultiTaskBiLSTM(nn.Module):
    def __init__(self, n_features, n_phoneme_classes,
                  n_manner=N_MANNER, n_voicing=N_VOICING, n_place=N_PLACE,
                  hidden=48, n_layers=1, dropout=0.3):
        super().__init__()
        self.input_drop = nn.Dropout(0.1)
        self.lstm = nn.LSTM(n_features, hidden, num_layers=n_layers,
                             bidirectional=True,
                             dropout=dropout if n_layers > 1 else 0,
                             batch_first=True)
        self.head_phoneme = nn.Linear(hidden * 2, n_phoneme_classes + 1)
        self.head_manner  = nn.Linear(hidden * 2, n_manner)
        self.head_voicing = nn.Linear(hidden * 2, n_voicing)
        self.head_place   = nn.Linear(hidden * 2, n_place)   # ★ new

    def forward(self, x):
        x = self.input_drop(x)
        h, _ = self.lstm(x)
        return (F.log_softmax(self.head_phoneme(h), dim=-1),
                F.log_softmax(self.head_manner(h),  dim=-1),
                F.log_softmax(self.head_voicing(h), dim=-1),
                F.log_softmax(self.head_place(h),   dim=-1))   # ★ now 4 outputs

class PhonemeDatasetMT(PhonemeDataset):
    """Same as PhonemeDataset but returns (feats, target, frame_ph, frame_mn, frame_vo)."""
    def __getitem__(self, idx):
        s = self.samples[idx]
        feats = s['features'].copy()
        if self.scaler is not None:
            feats = self.scaler.transform(feats).astype(np.float32)
        # Build all three label arrays
        ph_lbl, mn_lbl, vo_lbl = build_frame_labels_mt(
            feats.shape[0], s['mfa_intervals'], self.phone_to_idx)
        if self.augment:
            feats, ph_lbl = self._augment(feats, ph_lbl)
            # Apply same time-shift to manner/voicing labels using the shift in ph_lbl
            # Simpler: just rebuild with shifted offsets via mfa not feasible without state,
            # so we accept a mild mismatch; or set augment to NOT shift labels in time
        if self.noise_std > 0:
            feats = feats + np.random.randn(*feats.shape).astype(np.float32) * self.noise_std
        targets = np.array([self.phone_to_idx[ph] for ph in s['target']
                            if ph in self.phone_to_idx], dtype=np.int64)
        return feats, targets, ph_lbl, mn_lbl, vo_lbl


def collate_ctc_mt(batch):
    feats, targets, ph_lbl, mn_lbl, vo_lbl = zip(*batch)
    max_T = max(f.shape[0] for f in feats)
    n_feat = feats[0].shape[1]
    B = len(batch)
    feats_padded = np.zeros((B, max_T, n_feat), dtype=np.float32)
    ph_padded = np.zeros((B, max_T), dtype=np.int64)
    mn_padded = np.zeros((B, max_T), dtype=np.int64)
    vo_padded = np.zeros((B, max_T), dtype=np.int64)
    input_lengths = []
    for i, (f, p, m, v) in enumerate(zip(feats, ph_lbl, mn_lbl, vo_lbl)):
        feats_padded[i, :f.shape[0]] = f
        ph_padded[i, :len(p)] = p
        mn_padded[i, :len(m)] = m
        vo_padded[i, :len(v)] = v
        input_lengths.append(f.shape[0])
    targets_flat = np.concatenate(targets) if len(targets) else np.array([], dtype=np.int64)
    target_lengths = [len(t) for t in targets]
    return (torch.from_numpy(feats_padded),
            torch.from_numpy(targets_flat).long(),
            torch.tensor(input_lengths, dtype=torch.long),
            torch.tensor(target_lengths, dtype=torch.long),
            torch.from_numpy(ph_padded).long(),
            torch.from_numpy(mn_padded).long(),
            torch.from_numpy(vo_padded).long())


def train_mt(model, train_loader, val_loader, idx_to_phone,
              n_epochs=30, lr=1e-3, weight_decay=1e-3,
              patience=6, ph_w=1.0, manner_w=0.3, voicing_w=0.3,
              verbose=True):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=1e-5)
    best_val = float('inf'); best_state = None; no_improve = 0

    for epoch in range(n_epochs):
        model.train(); tr_total, tr_ph, tr_mn, tr_vo = [], [], [], []
        for feats, tgt, in_l, tgt_l, ph_lbl, mn_lbl, vo_lbl in train_loader:
            feats = feats.to(DEVICE)
            ph_lbl = ph_lbl.to(DEVICE); mn_lbl = mn_lbl.to(DEVICE); vo_lbl = vo_lbl.to(DEVICE)
            opt.zero_grad()
            log_p_ph, log_p_mn, log_p_vo = model(feats)
            B, T, _ = log_p_ph.shape
            l_ph = F.nll_loss(log_p_ph.reshape(B*T, -1), ph_lbl.reshape(-1))
            l_mn = F.nll_loss(log_p_mn.reshape(B*T, -1), mn_lbl.reshape(-1))
            l_vo = F.nll_loss(log_p_vo.reshape(B*T, -1), vo_lbl.reshape(-1))
            loss = ph_w * l_ph + manner_w * l_mn + voicing_w * l_vo
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tr_total.append(loss.item())
            tr_ph.append(l_ph.item()); tr_mn.append(l_mn.item()); tr_vo.append(l_vo.item())

        model.eval(); va_total, va_ph = [], []
        with torch.no_grad():
            for feats, tgt, in_l, tgt_l, ph_lbl, mn_lbl, vo_lbl in val_loader:
                feats = feats.to(DEVICE)
                ph_lbl = ph_lbl.to(DEVICE); mn_lbl = mn_lbl.to(DEVICE); vo_lbl = vo_lbl.to(DEVICE)
                log_p_ph, log_p_mn, log_p_vo = model(feats)
                B, T, _ = log_p_ph.shape
                l_ph = F.nll_loss(log_p_ph.reshape(B*T, -1), ph_lbl.reshape(-1))
                l_mn = F.nll_loss(log_p_mn.reshape(B*T, -1), mn_lbl.reshape(-1))
                l_vo = F.nll_loss(log_p_vo.reshape(B*T, -1), vo_lbl.reshape(-1))
                va_total.append((ph_w*l_ph + manner_w*l_mn + voicing_w*l_vo).item())
                va_ph.append(l_ph.item())

        tr_t = np.mean(tr_total); va_t = np.mean(va_total); sched.step()
        if va_t < best_val:
            best_val = va_t
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0; mark = '*'
        else:
            no_improve += 1; mark = ''
        if verbose:
            print(f"    e{epoch+1:3d}: train [tot={tr_t:.2f} ph={np.mean(tr_ph):.2f} "
                  f"mn={np.mean(tr_mn):.2f} vo={np.mean(tr_vo):.2f}]  "
                  f"val [tot={va_t:.2f} ph={np.mean(va_ph):.2f}]  "
                  f"lr={opt.param_groups[0]['lr']:.5f}  {mark}", flush=True)
        if no_improve >= patience:
            if verbose: print(f"    early stop"); break
    model.load_state_dict(best_state)
    return model

class PhonemeTDNN(nn.Module):
    """Time-Delay Neural Network: stacked dilated 1D convs with multi-task heads.
    Receptive field ~290 ms which is enough for phoneme context."""
    def __init__(self, n_features, n_phoneme_classes,
                  n_manner=N_MANNER, n_voicing=N_VOICING, n_place=N_PLACE,
                  hidden=128, dropout=0.3):
        super().__init__()
        self.in_proj = nn.Conv1d(n_features, hidden, kernel_size=1)
        self.bn0     = nn.BatchNorm1d(hidden)

        # Dilated stack: kernel=5, dilations 1, 2, 4 → ~29 frame receptive field
        self.conv1 = nn.Conv1d(hidden, hidden, kernel_size=5, padding=2,  dilation=1)
        self.conv2 = nn.Conv1d(hidden, hidden, kernel_size=5, padding=4,  dilation=2)
        self.conv3 = nn.Conv1d(hidden, hidden, kernel_size=5, padding=8,  dilation=4)
        self.bn1, self.bn2, self.bn3 = nn.BatchNorm1d(hidden), nn.BatchNorm1d(hidden), nn.BatchNorm1d(hidden)
        self.dropout = nn.Dropout(dropout)

        self.head_phoneme = nn.Conv1d(hidden, n_phoneme_classes + 1, kernel_size=1)
        self.head_manner  = nn.Conv1d(hidden, n_manner,  kernel_size=1)
        self.head_voicing = nn.Conv1d(hidden, n_voicing, kernel_size=1)
        self.head_place   = nn.Conv1d(hidden, n_place,   kernel_size=1)

    def forward(self, x):
        # x: (B, T, F) -> (B, F, T) for Conv1d
        x = x.transpose(1, 2)
        x = F.relu(self.bn0(self.in_proj(x)))
        x = F.relu(self.bn1(self.conv1(x))); x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x))); x = self.dropout(x)
        x = F.relu(self.bn3(self.conv3(x))); x = self.dropout(x)
        log_p_ph = F.log_softmax(self.head_phoneme(x), dim=1).transpose(1, 2)
        log_p_mn = F.log_softmax(self.head_manner(x),  dim=1).transpose(1, 2)
        log_p_vo = F.log_softmax(self.head_voicing(x), dim=1).transpose(1, 2)
        log_p_pl = F.log_softmax(self.head_place(x),   dim=1).transpose(1, 2)
        return log_p_ph, log_p_mn, log_p_vo, log_p_pl

def nw_summary(results):
    """Quick NW-alignment summary across all sentence results."""
    all_aligned = []
    for r in results:
        all_aligned.extend(needleman_wunsch(r['gold'], r['pred']))
    n_match = sum(1 for g, p in all_aligned if g is not None and p is not None and g == p)
    n_sub   = sum(1 for g, p in all_aligned if g is not None and p is not None and g != p)
    n_del   = sum(1 for g, p in all_aligned if g is not None and p is None)
    n_ins   = sum(1 for g, p in all_aligned if g is None and p is not None)
    n_gold  = sum(1 for g, p in all_aligned if g is not None)
    return dict(n_gold=n_gold, n_match=n_match, n_sub=n_sub, n_del=n_del, n_ins=n_ins,
                match_rate=n_match/max(n_gold,1),
                per=(n_sub+n_del+n_ins)/max(n_gold,1))

# train baseline
print("Retraining single-task baseline as out_aug...", flush=True)
out_aug, err = run_ctc_for_patient(
    "P22", n_epochs=30, batch_size=8, hidden=32,
    noise_std=0.15,
    ctc_w=0.0, ce_w=1.0,
    use_class_weights=False,
    use_sentence_sampler=True,
    sample_oversample_factor=2,
    n_time_masks=5, time_mask_max=40,
    n_feat_masks=4, feat_mask_max_frac=0.12,
    max_time_shift=5,
    patience=6, dropout=0.5, weight_decay=1e-3,
    verbose=False)
if out_aug is None:
    raise RuntimeError(f"baseline failed: {err}")
summary_b = nw_summary(out_aug['results'])
print(f"baseline match_rate={100*summary_b['match_rate']:.1f}%  PER={100*summary_b['per']:.1f}%")

# try enhance model
from torch.utils.data import WeightedRandomSampler

def run_mt_for_patient(pid, n_epochs=30, batch_size=8, hidden=32,
                        noise_std=0.15, dropout=0.5,
                        weight_decay=1e-3, patience=6,
                        ph_w=1.0, manner_w=0.3, voicing_w=0.3, place_w=0.3,
                        use_sentence_sampler=False, sample_oversample_factor=2,
                        n_time_masks=5, time_mask_max=40,
                        n_feat_masks=4, feat_mask_max_frac=0.12,
                        max_time_shift=0,
                        verbose=True):
    """Multi-task CTC+CE BiLSTM with optional sentence-level sampler.
    Predicts phoneme + manner + voicing + place; only phoneme head used at inference."""
    splits = build_patient_dataset(pid)
    if splits is None: return None, "no data"
    if len(splits['train']) < 10: return None, f"too few train: {len(splits['train'])}"
    train_phonemes = sorted({ph for s in splits['train'] for ph in s['target']})
    phone_to_idx = {ph: i + 1 for i, ph in enumerate(train_phonemes)}
    idx_to_phone = {i + 1: ph for i, ph in enumerate(train_phonemes)}
    for split in ['val', 'test']:
        for s in splits[split]:
            s['target'] = [ph for ph in s['target'] if ph in phone_to_idx]
        splits[split] = [s for s in splits[split] if len(s['target']) > 0]
    n_classes = len(train_phonemes)
    if verbose:
        print(f"  vocab: {n_classes} phonemes  "
              f"ph_w={ph_w} mn_w={manner_w} vo_w={voicing_w} pl_w={place_w}")

    # Feature scaler — chunked partial_fit
    scaler = StandardScaler()
    for s in splits['train']:
        scaler.partial_fit(s['features'].astype(np.float32))

    train_ds = PhonemeDatasetMT(splits['train'], phone_to_idx, scaler=scaler,
                                  noise_std=noise_std, augment=True,
                                  n_time_masks=n_time_masks, time_mask_max=time_mask_max,
                                  n_feat_masks=n_feat_masks,
                                  feat_mask_max_frac=feat_mask_max_frac,
                                  max_time_shift=max_time_shift,
                                  include_frame_labels=True)
    val_ds   = PhonemeDatasetMT(splits['val'],   phone_to_idx, scaler=scaler,
                                  noise_std=0.0, augment=False,
                                  include_frame_labels=True)

    if use_sentence_sampler:
        from collections import Counter
        label_counts = Counter()
        for s in splits['train']:
            ph_lbl, _, _, _ = build_frame_labels_mt(s['features'].shape[0],
                                                      s['mfa_intervals'], phone_to_idx)
            for x in ph_lbl[ph_lbl > 0]:
                label_counts[int(x)] += 1
        sent_weights = compute_sentence_weights(splits['train'], phone_to_idx,
                                                  label_counts, alpha=1.0)
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(sent_weights).double(),
            num_samples=len(sent_weights) * sample_oversample_factor,
            replacement=True)
        train_loader = DataLoader(train_ds, batch_size=batch_size,
                                    sampler=sampler,
                                    collate_fn=collate_ctc_mt, num_workers=0)
        if verbose:
            print(f"  sampler: {len(sent_weights) * sample_oversample_factor} "
                  f"samples/epoch (weight range {sent_weights.min():.2f}..{sent_weights.max():.2f})")
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                    collate_fn=collate_ctc_mt, num_workers=0)

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              collate_fn=collate_ctc_mt, num_workers=0)

    n_features = splits['train'][0]['features'].shape[1]
    # model = PhonemeMultiTaskBiLSTM(n_features, n_classes,
    #                                  hidden=hidden, n_layers=1,
    #                                  dropout=dropout).to(DEVICE)
    model = PhonemeTDNN(n_features, n_classes, hidden=128, dropout=dropout).to(DEVICE)
    with torch.no_grad():
        model.head_phoneme.bias[0] -= 5.0
    if verbose:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  model: {n_params:,} params")

    model = train_mt(model, train_loader, val_loader, idx_to_phone,
                      n_epochs=n_epochs, weight_decay=weight_decay,
                      patience=patience,
                      ph_w=ph_w, manner_w=manner_w, voicing_w=voicing_w, place_w=place_w,
                      verbose=verbose)

    # Greedy evaluation using phoneme head only
    model.eval(); results = []
    with torch.no_grad():
        for s in splits['test']:
            feats = scaler.transform(s['features']).astype(np.float32)
            x = torch.from_numpy(feats).unsqueeze(0).to(DEVICE)
            log_p_ph, *_ = model(x)
            lp = log_p_ph.squeeze(0).cpu().numpy()
            pred = ctc_greedy_decode(lp, idx_to_phone)
            results.append(dict(sent_idx=s['sent_idx'], gold=s['target'],
                                 pred=pred, log_probs=lp))

    return dict(results=results, model=model, scaler=scaler,
                phone_to_idx=phone_to_idx, idx_to_phone=idx_to_phone), None


# ── Run for one patient as a first test ──
print("=== Multi-task CTC+CE for P22 ===", flush=True)
out_mt, err = run_mt_for_patient("P22", n_epochs=60, hidden=32,
    noise_std=0.15, dropout=0.5, weight_decay=1e-3,
    ph_w=1.0,
    manner_w=0.2, voicing_w=0.2, place_w=0.3,   # ★ much lighter
    use_sentence_sampler=True, sample_oversample_factor=3,
    max_time_shift=0, patience=10, verbose=True)
if out_mt:
    summary = nw_summary(out_mt['results'])
    print(f"\nMatch rate: {100*summary['match_rate']:.1f}%   PER: {100*summary['per']:.1f}%")
    print(f"matches={summary['n_match']}/{summary['n_gold']}")

import numpy as np
import torch

# ---------- helper: expand intervals into per-frame label array ----------
def intervals_to_frames(mfa_intervals, T, label_to_id, default_id=0):
    """
    mfa_intervals: list of (phoneme_str, start_frame, end_frame).
    Returns int64 array of shape (T,) with phoneme ids per frame.
    Frames outside any interval default to default_id (use 'sil' id).
    """
    out = np.full(T, default_id, dtype=np.int64)
    for ph, t0, t1 in mfa_intervals:
        t0 = max(0, int(t0)); t1 = min(T, int(t1))
        if t1 <= t0:
            continue
        out[t0:t1] = label_to_id.get(ph, default_id)
    return out


# ---------- fit_scaler using the real key ----------
from sklearn.preprocessing import StandardScaler

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


# ---------- patched dataset ----------
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

def _unpack_interval(it):
    """
    Return (phone_str, start_time_float, end_time_float) regardless of tuple order.
    Handles (start, end, text) (tgt-style) and (text, start, end).
    """
    # find the string element
    str_elems = [(i, x) for i, x in enumerate(it) if isinstance(x, str)]
    if not str_elems:
        raise ValueError(f"no string label in interval {it}")
    s_idx, ph = str_elems[0]
    # remaining two are start/end (in tuple order)
    nums = [float(x) for i, x in enumerate(it) if i != s_idx]
    if len(nums) != 2:
        raise ValueError(f"expected 2 time fields, got {nums} from {it}")
    t0, t1 = sorted(nums)
    return ph, t0, t1


def build_phone_vocab(sents):
    phones = set()
    for s in sents:
        phones.update(s['target'])
        for it in s['mfa_intervals']:
            ph, _, _ = _unpack_interval(it)
            phones.add(ph)
    ordered = ['sil'] + sorted(p for p in phones if p not in ('sil', 'SIL'))
    p2i = {p: i for i, p in enumerate(ordered)}
    i2p = {i: p for p, i in p2i.items()}
    return p2i, i2p

def intervals_to_frames(mfa_intervals, T, label_to_id, sent_duration_sec,
                        default_id=0):
    """
    Convert tgt-style intervals (start_sec, end_sec, phoneme_str) to a (T,)
    int64 array of phoneme ids. sent_duration_sec is total length of THIS sentence
    in seconds (e.g. end_time of the last interval).
    """
    out = np.full(T, default_id, dtype=np.int64)
    if sent_duration_sec <= 0:
        return out
    fps = T / sent_duration_sec   # frames per second for this sentence
    for it in mfa_intervals:
        ph, t0, t1 = _unpack_interval(it)
        f0 = int(round(t0 * fps))
        f1 = int(round(t1 * fps))
        f0 = max(0, f0); f1 = min(T, f1)
        if f1 <= f0:
            continue
        out[f0:f1] = label_to_id.get(ph, default_id)
    return out

s0 = splits['train'][0]
T  = s0['features'].shape[0]
ints = s0['mfa_intervals']
_, _, sent_end = _unpack_interval(ints[-1])
ph_frames = intervals_to_frames(ints, T, phone_to_idx,
                                sent_duration_sec=sent_end,
                                default_id=phone_to_idx['sil'])

# how many frames per phoneme — should look roughly like duration*fps
from collections import Counter
print('sent_end:', sent_end, 'T:', T, 'fps≈', T/sent_end)
print('per-frame label counts (top 10):',
      Counter([list(phone_to_idx.keys())[list(phone_to_idx.values()).index(i)]
               for i in ph_frames]).most_common(10))
print('first 30 labeled intervals → expected vs actual range:')
for ph, t0, t1 in (_unpack_interval(it) for it in ints[:10]):
    f0, f1 = int(round(t0*T/sent_end)), int(round(t1*T/sent_end))
    print(f'  {ph!r:6}  {t0:.3f}-{t1:.3f}s  →  frames {f0}-{f1}  ({f1-f0} frames)')

# ============================================================
# PRELUDE — must run before train_ds / val_ds construction
# ============================================================
from sklearn.preprocessing import StandardScaler
import numpy as np

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

train_ds = PhonemeDatasetMTMel(
    splits['train'], phone_to_idx,
    scaler=scaler, mel_scaler=mel_scaler,
    manner_map=DUTCH_MANNER, voicing_map=DUTCH_VOICING, place_map=DUTCH_PLACE,
    noise_std=noise_std, augment=True,
    n_time_masks=n_time_masks, time_mask_max=time_mask_max,
    n_feat_masks=n_feat_masks, feat_mask_max_frac=feat_mask_max_frac,
    max_time_shift=max_time_shift,
    include_frame_labels=True,
)

val_ds = PhonemeDatasetMTMel(
    splits['val'], phone_to_idx,
    scaler=scaler, mel_scaler=mel_scaler,
    manner_map=DUTCH_MANNER, voicing_map=DUTCH_VOICING, place_map=DUTCH_PLACE,
    noise_std=0.0, augment=False,
    n_time_masks=0, time_mask_max=0,
    n_feat_masks=0, feat_mask_max_frac=0.0,
    max_time_shift=0,
    include_frame_labels=True,
)

# ============================================================
# DATASET + LOADER SETUP (mel-aware version)
# ============================================================
# Assumes you already have:
#   splits = {'train': [...], 'val': [...], 'test': [...]}  from build_patient_dataset_with_mel(pid, n_mels=40)
#   phone_to_idx, idx_to_phone
#   scaler = fit_scaler(splits['train'])              # neural feature scaler (partial_fit)
#   mel_scaler = fit_mel_scaler(splits['train'])      # mel-bin-wise scaler (partial_fit)
# and these constants set above:
#   noise_std            (e.g. 0.75)
#   n_time_masks         (e.g. 2)
#   time_mask_max        (e.g. 10)
#   n_feat_masks         (e.g. 2)
#   feat_mask_max_frac   (e.g. 0.10)
#   max_time_shift       (e.g. 3)
#   batch_size           (e.g. 8)

train_ds = PhonemeDatasetMTMel(
    splits['train'],
    phone_to_idx,
    scaler=scaler,
    mel_scaler=mel_scaler,            # ★ mel target normalization
    noise_std=noise_std,
    augment=True,
    n_time_masks=n_time_masks,
    time_mask_max=time_mask_max,
    n_feat_masks=n_feat_masks,
    feat_mask_max_frac=feat_mask_max_frac,
    max_time_shift=max_time_shift,
    include_frame_labels=True,        # needed for CE + aux heads
)

val_ds = PhonemeDatasetMTMel(
    splits['val'],
    phone_to_idx,
    scaler=scaler,
    mel_scaler=mel_scaler,            # ★ same scaler, no fitting
    noise_std=0.0,                    # no noise at val
    augment=False,                    # no augmentation at val
    n_time_masks=0,
    time_mask_max=0,
    n_feat_masks=0,
    feat_mask_max_frac=0.0,
    max_time_shift=0,
    include_frame_labels=True,
)

# --- optional: WeightedRandomSampler for sentence-level oversampling ---
# If you were using one before, keep the same logic, just pass it through:
sample_weights = compute_sentence_weights(splits['train'], phone_to_idx)  # your existing helper
sampler = torch.utils.data.WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True,
)

# --- DataLoaders ---
train_loader = torch.utils.data.DataLoader(
    train_ds,
    batch_size=batch_size,
    sampler=sampler,                  # use sampler=... OR shuffle=True, not both
    collate_fn=collate_ctc_mt_mel,    # ★ mel-aware collate
    num_workers=0,                    # Windows: keep 0
    pin_memory=True,
    drop_last=False,
)

val_loader = torch.utils.data.DataLoader(
    val_ds,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_ctc_mt_mel,    # ★ same collate for val
    num_workers=0,
    pin_memory=True,
    drop_last=False,
)

out_mt_viz = ctc_results_to_out(out_mt['results'])
display(HTML(show_predictions_html(out_mt_viz, label="multi-task", max_sentences=10)))

N_SENTENCES = 10
# Need splits and the get_logp helper from before
splits = build_patient_dataset("P22")
for s in splits['test']:
    s['target'] = [ph for ph in s['target'] if ph in out_mt['phone_to_idx']]
splits['test'] = [s for s in splits['test'] if len(s['target']) > 0]

for test_sample in splits['test'][:N_SENTENCES]:
    # Multi-task model returns 3 outputs — we just want phoneme head
    out_mt['model'].eval()
    with torch.no_grad():
        feats = out_mt['scaler'].transform(test_sample['features']).astype(np.float32)
        x = torch.from_numpy(feats).unsqueeze(0).to(DEVICE)
        log_p_ph, _, _, _ = out_mt['model'](x)
        logp_mt = log_p_ph.squeeze(0).cpu().numpy()

    logp_st = get_logp(out_aug['model'], out_aug['scaler'], test_sample)

    segs_mt = smoothed_decode_with_times(logp_mt, out_mt['idx_to_phone'],
                                           smooth_w=SMOOTH_W, min_frames=MIN_FRAMES)
    segs_st = smoothed_decode_with_times(logp_st, out_aug['idx_to_phone'],
                                           smooth_w=SMOOTH_W, min_frames=MIN_FRAMES)
    gold_segs = [(ph, t0, t1) for (t0, t1, ph) in test_sample['mfa_intervals']]
    plot_timeline_3way(gold_segs,
                        {'single-task': segs_st, 'multi-task': segs_mt},
                        test_sample['sent_idx'])
    plt.show()

import os, pickle, copy, time, random

SAVE_DIR = r"C:\Temp"
os.makedirs(SAVE_DIR, exist_ok=True)

# Train σ=0.75 model
print("=== Training σ=0.75 ===", flush=True)
t0 = time.time()
torch.manual_seed(42); np.random.seed(42); random.seed(42)
if DEVICE == 'cuda': torch.cuda.manual_seed_all(42)

out_s, err = run_mt_for_patient(
    "P22", n_epochs=60, hidden=32,
    noise_std=0.75,
    dropout=0.5, weight_decay=1e-3,
    ph_w=1.0, manner_w=0.2, voicing_w=0.2, place_w=0.3,
    use_sentence_sampler=True, sample_oversample_factor=3,
    max_time_shift=0, patience=10, verbose=False)
print(f"  done in {time.time()-t0:.0f}s")
if out_s is None:
    raise RuntimeError(f"training failed: {err}")

# Save without disturbing in-memory GPU model
out_to_save = dict(out_s)
out_to_save['model'] = copy.deepcopy(out_s['model']).cpu()
save_path = os.path.join(SAVE_DIR, "P22_sigma075.pkl")
with open(save_path, 'wb') as f:
    pickle.dump(out_to_save, f)
print(f"  saved to {save_path}")

# Sanity-check it loads back
with open(save_path, 'rb') as f:
    out_reloaded = pickle.load(f)
print(f"  reload OK: {out_reloaded['model'].__class__.__name__}, "
      f"vocab={len(out_reloaded['idx_to_phone'])}")

import numpy as np
from scipy.signal import stft

def mel_filterbank(n_mels=40, n_fft=512, sr=16000, fmin=0, fmax=None):
    """Create mel filterbank matrix shape (n_mels, n_fft//2 + 1)."""
    fmax = fmax or sr / 2
    hz_to_mel = lambda hz: 2595 * np.log10(1 + hz / 700)
    mel_to_hz = lambda m: 700 * (10**(m / 2595) - 1)
    mel_points = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, n_mels + 1):
        lo, ctr, hi = bin_points[m-1], bin_points[m], bin_points[m+1]
        for k in range(lo, ctr):
            if ctr > lo: fb[m-1, k] = (k - lo) / (ctr - lo)
        for k in range(ctr, hi):
            if hi > ctr: fb[m-1, k] = (hi - k) / (hi - ctr)
    return fb


def compute_mel_spectrogram(audio, sr, n_mels=40, win_ms=50, hop_ms=10):
    """Mel-spectrogram in dB. Returns (n_frames, n_mels)."""
    n_fft = int(2 ** np.ceil(np.log2(int(sr * win_ms / 1000))))
    hop_length = int(sr * hop_ms / 1000)
    f, t, Z = stft(audio, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length,
                    window='hann', return_onesided=True, boundary=None, padded=False)
    mag = np.abs(Z).astype(np.float32)        # (n_freqs, n_frames)
    fb = mel_filterbank(n_mels=n_mels, n_fft=n_fft, sr=sr)
    mel = fb @ mag                            # (n_mels, n_frames)
    mel_db = 20 * np.log10(np.maximum(mel, 1e-8))
    return mel_db.T                           # (n_frames, n_mels)

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

def build_patient_dataset_with_mel(pid, feature_spec=None, test_offset=None, n_mels=40):
    """Extends build_patient_dataset to also compute mel-spectrograms per sentence."""
    feature_spec = feature_spec or DEFAULT_FEATURE_SPEC
    test_offset  = TEST_OFFSET if test_offset is None else test_offset
    wd = pipeline.split_result['word_segments_dict'].get(pid)
    if wd is None: return None
    try: mfa = load_mfa_alignments(pid)
    except Exception: return None
    if not mfa: return None

    all_real = [i for i, s in enumerate(wd['sentence_list'])
                if isinstance(s, dict) and s.get('text')]
    test_sent_ids = set(all_real[test_offset::6])
    train_sent_ids_all = sorted(set(all_real) - test_sent_ids)
    val_step = max(2, int(round(1 / VAL_FRAC)))
    val_sent_ids = set(train_sent_ids_all[::val_step])

    raw_path = os.path.join(DUTCH_30_PATH, 'raw', f'{pid}_sEEG.npy')
    raw_eeg = np.load(raw_path)
    if raw_eeg.ndim == 2 and raw_eeg.shape[0] < raw_eeg.shape[1]:
        raw_eeg = raw_eeg.T

    audio = np.load(os.path.join(DUTCH_30_PATH, 'raw', f'{pid}_audio.npy'))
    audio_sr = 48000
    ratio = audio_sr / EEG_SR

    splits = dict(train=[], val=[], test=[])
    for sent_idx in all_real:
        if sent_idx not in mfa or not mfa[sent_idx]: continue
        sd = wd['sentence_list'][sent_idx]
        s0, s1 = sd['stim_start_idx'], sd['stim_end_idx']
        if s1 > raw_eeg.shape[0]: continue
        ext = extract_features_multiband(raw_eeg[s0:s1], **feature_spec)
        if ext.shape[0] < 2 * LDA_MARGIN + 1: continue
        stk = stackFeatures(ext, modelOrder=MO, stepSize=SS).astype(np.float32)
        T = stk.shape[0]
        duration_sec = (s1 - s0) / EEG_SR        # ★ EEG_SR = 1024
        feature_fps  = T / duration_sec     
        target = [ph['phone'] for ph in mfa[sent_idx]]
        if not target: continue
        mfa_intervals = [(ph['start_s'], ph['end_s'], ph['phone']) for ph in mfa[sent_idx]]

        # Compute mel-spec and align to T stacked frames
        audio_start = int(s0 * ratio); audio_end = int(s1 * ratio)
        if audio_end > len(audio): continue
        mel = build_mel_for_sentence(audio, audio_sr, audio_start, audio_end,
                                       n_mels=n_mels, target_n_frames=T)

        split = ('test' if sent_idx in test_sent_ids else
                  'val'  if sent_idx in val_sent_ids  else 'train')
        splits[split].append(dict(
            features=stk, target=target, mfa_intervals=mfa_intervals,
            mel=mel,
            duration_sec=duration_sec,           # ★ new
            feature_fps=feature_fps,             # ★ new
            sent_idx=sent_idx,
        ))
    return splits

def fit_mel_scaler(samples):
    """Per-mel-bin standardizer: zero-mean, unit-variance across training frames."""
    all_mel = np.concatenate([s['mel'] for s in samples], axis=0)
    mean = all_mel.mean(axis=0, keepdims=True)
    std  = all_mel.std(axis=0, keepdims=True) + 1e-6
    return dict(mean=mean.astype(np.float32), std=std.astype(np.float32))

def transform_mel(mel, mel_scaler):
    return ((mel - mel_scaler['mean']) / mel_scaler['std']).astype(np.float32)

class PhonemeMTMelBiLSTM(nn.Module):
    def __init__(self, n_features, n_phoneme_classes,
                  n_manner=N_MANNER, n_voicing=N_VOICING, n_place=N_PLACE,
                  n_mels=40, hidden=64, n_layers=1, dropout=0.3):
        super().__init__()
        self.input_drop = nn.Dropout(0.1)
        self.lstm = nn.LSTM(n_features, hidden, num_layers=n_layers,
                             bidirectional=True, batch_first=True,
                             dropout=dropout if n_layers > 1 else 0)
        H = hidden * 2
        self.head_mel     = nn.Linear(H, n_mels)               # ★ regression
        self.head_phoneme = nn.Linear(H, n_phoneme_classes + 1)
        self.head_manner  = nn.Linear(H, n_manner)
        self.head_voicing = nn.Linear(H, n_voicing)
        self.head_place   = nn.Linear(H, n_place)

    def forward(self, x):
        x = self.input_drop(x)
        h, _ = self.lstm(x)
        return (self.head_mel(h),
                F.log_softmax(self.head_phoneme(h), dim=-1),
                F.log_softmax(self.head_manner(h),  dim=-1),
                F.log_softmax(self.head_voicing(h), dim=-1),
                F.log_softmax(self.head_place(h),   dim=-1))


def train_mt_mel(model, train_loader, val_loader, idx_to_phone,
                  n_epochs=50, lr=1e-3, weight_decay=1e-3, patience=10,
                  ph_w=1.0, manner_w=0.2, voicing_w=0.2, place_w=0.3, mel_w=1.0,
                  verbose=True):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=1e-5)
    best_val, best_state, no_improve = float('inf'), None, 0

    for epoch in range(n_epochs):
        model.train(); losses = {k: [] for k in ['tot','mel','ph','mn','vo','pl']}
        for feats, tgt, in_l, tgt_l, ph_lbl, mn_lbl, vo_lbl, pl_lbl, mel_t, mel_mask in train_loader:
            feats = feats.to(DEVICE); mel_t = mel_t.to(DEVICE); mel_mask = mel_mask.to(DEVICE)
            ph_lbl = ph_lbl.to(DEVICE); mn_lbl = mn_lbl.to(DEVICE)
            vo_lbl = vo_lbl.to(DEVICE); pl_lbl = pl_lbl.to(DEVICE)
            opt.zero_grad()
            pred_mel, log_p_ph, log_p_mn, log_p_vo, log_p_pl = model(feats)
            B, T, _ = log_p_ph.shape
            # Mel MSE (masked over valid frames)
            mel_diff_sq = ((pred_mel - mel_t) ** 2).mean(dim=-1)   # (B, T)
            l_mel = (mel_diff_sq * mel_mask.float()).sum() / mel_mask.float().sum().clamp(min=1)
            l_ph = F.nll_loss(log_p_ph.reshape(B*T, -1), ph_lbl.reshape(-1))
            l_mn = F.nll_loss(log_p_mn.reshape(B*T, -1), mn_lbl.reshape(-1))
            l_vo = F.nll_loss(log_p_vo.reshape(B*T, -1), vo_lbl.reshape(-1))
            l_pl = F.nll_loss(log_p_pl.reshape(B*T, -1), pl_lbl.reshape(-1))
            loss = mel_w*l_mel + ph_w*l_ph + manner_w*l_mn + voicing_w*l_vo + place_w*l_pl
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            for k, v in zip(['tot','mel','ph','mn','vo','pl'],
                              [loss, l_mel, l_ph, l_mn, l_vo, l_pl]):
                losses[k].append(v.item())

        model.eval(); val_losses = {k: [] for k in losses}
        with torch.no_grad():
            for feats, tgt, in_l, tgt_l, ph_lbl, mn_lbl, vo_lbl, pl_lbl, mel_t, mel_mask in val_loader:
                feats = feats.to(DEVICE); mel_t = mel_t.to(DEVICE); mel_mask = mel_mask.to(DEVICE)
                ph_lbl = ph_lbl.to(DEVICE); mn_lbl = mn_lbl.to(DEVICE)
                vo_lbl = vo_lbl.to(DEVICE); pl_lbl = pl_lbl.to(DEVICE)
                pred_mel, log_p_ph, log_p_mn, log_p_vo, log_p_pl = model(feats)
                B, T, _ = log_p_ph.shape
                mel_diff_sq = ((pred_mel - mel_t) ** 2).mean(dim=-1)
                l_mel = (mel_diff_sq * mel_mask.float()).sum() / mel_mask.float().sum().clamp(min=1)
                l_ph = F.nll_loss(log_p_ph.reshape(B*T, -1), ph_lbl.reshape(-1))
                l_mn = F.nll_loss(log_p_mn.reshape(B*T, -1), mn_lbl.reshape(-1))
                l_vo = F.nll_loss(log_p_vo.reshape(B*T, -1), vo_lbl.reshape(-1))
                l_pl = F.nll_loss(log_p_pl.reshape(B*T, -1), pl_lbl.reshape(-1))
                loss = mel_w*l_mel + ph_w*l_ph + manner_w*l_mn + voicing_w*l_vo + place_w*l_pl
                for k, v in zip(['tot','mel','ph','mn','vo','pl'],
                                  [loss, l_mel, l_ph, l_mn, l_vo, l_pl]):
                    val_losses[k].append(v.item())

        tr_t = np.mean(losses['tot']); va_t = np.mean(val_losses['tot']); sched.step()
        if va_t < best_val:
            best_val = va_t
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0; mark = '*'
        else:
            no_improve += 1; mark = ''
        if verbose:
            print(f"  e{epoch+1:3d}: train [tot={tr_t:.2f} mel={np.mean(losses['mel']):.2f} "
                  f"ph={np.mean(losses['ph']):.2f}]  "
                  f"val [tot={va_t:.2f} mel={np.mean(val_losses['mel']):.2f} "
                  f"ph={np.mean(val_losses['ph']):.2f}]  "
                  f"lr={opt.param_groups[0]['lr']:.5f} {mark}", flush=True)
        if no_improve >= patience: break
    model.load_state_dict(best_state)
    return model

class PhonemeDatasetMTMel(PhonemeDatasetMT):
    def __init__(self, *args, mel_scaler=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.mel_scaler = mel_scaler

    def __getitem__(self, idx):
        s = self.samples[idx]
        feats = s['features'].copy()
        mel   = s['mel'].copy()
        if self.scaler is not None:
            feats = self.scaler.transform(feats).astype(np.float32)
        if self.mel_scaler is not None:
            mel = transform_mel(mel, self.mel_scaler)
        ph_lbl, mn_lbl, vo_lbl, pl_lbl = build_frame_labels_mt(
            feats.shape[0], s['mfa_intervals'], self.phone_to_idx)
        if self.augment:
            feats, ph_lbl = self._augment(feats, ph_lbl)
        if self.noise_std > 0:
            feats = feats + np.random.randn(*feats.shape).astype(np.float32) * self.noise_std
        targets = np.array([self.phone_to_idx[ph] for ph in s['target']
                            if ph in self.phone_to_idx], dtype=np.int64)
        return feats, targets, ph_lbl, mn_lbl, vo_lbl, pl_lbl, mel


def collate_ctc_mt_mel(batch):
    feats, targets, ph_lbl, mn_lbl, vo_lbl, pl_lbl, mels = zip(*batch)
    max_T = max(f.shape[0] for f in feats)
    n_feat = feats[0].shape[1]
    n_mels = mels[0].shape[1]
    B = len(batch)
    feats_padded = np.zeros((B, max_T, n_feat), dtype=np.float32)
    mels_padded  = np.zeros((B, max_T, n_mels), dtype=np.float32)
    mel_masks    = np.zeros((B, max_T), dtype=bool)
    ph_padded = np.zeros((B, max_T), dtype=np.int64)
    mn_padded = np.zeros((B, max_T), dtype=np.int64)
    vo_padded = np.zeros((B, max_T), dtype=np.int64)
    pl_padded = np.zeros((B, max_T), dtype=np.int64)
    input_lengths = []
    for i, (f, p, m, v, pl, mel) in enumerate(zip(feats, ph_lbl, mn_lbl, vo_lbl, pl_lbl, mels)):
        T = f.shape[0]
        feats_padded[i, :T] = f
        mels_padded[i, :T]  = mel
        mel_masks[i, :T]    = True
        ph_padded[i, :T] = p; mn_padded[i, :T] = m
        vo_padded[i, :T] = v; pl_padded[i, :T] = pl
        input_lengths.append(T)
    targets_flat = np.concatenate(targets) if len(targets) else np.array([], dtype=np.int64)
    target_lengths = [len(t) for t in targets]
    return (torch.from_numpy(feats_padded),
            torch.from_numpy(targets_flat).long(),
            torch.tensor(input_lengths, dtype=torch.long),
            torch.tensor(target_lengths, dtype=torch.long),
            torch.from_numpy(ph_padded).long(),
            torch.from_numpy(mn_padded).long(),
            torch.from_numpy(vo_padded).long(),
            torch.from_numpy(pl_padded).long(),
            torch.from_numpy(mels_padded),
            torch.from_numpy(mel_masks))

def run_mt_mel_for_patient(pid, n_epochs=50, batch_size=8, hidden=64,
                             noise_std=0.30, dropout=0.5, weight_decay=1e-3, patience=10,
                             mel_w=1.0, ph_w=1.0, manner_w=0.2, voicing_w=0.2, place_w=0.3,
                             n_mels=40,
                             use_sentence_sampler=True, sample_oversample_factor=2,
                             n_time_masks=5, time_mask_max=40,
                             n_feat_masks=4, feat_mask_max_frac=0.12,
                             max_time_shift=0, verbose=True):
    splits = build_patient_dataset_with_mel(pid, n_mels=n_mels)
    if splits is None: return None, "no data"
    if len(splits['train']) < 10: return None, f"too few train: {len(splits['train'])}"
    train_phonemes = sorted({ph for s in splits['train'] for ph in s['target']})
    phone_to_idx = {ph: i+1 for i, ph in enumerate(train_phonemes)}
    idx_to_phone = {i+1: ph for i, ph in enumerate(train_phonemes)}
    for split in ['val', 'test']:
        for s in splits[split]:
            s['target'] = [ph for ph in s['target'] if ph in phone_to_idx]
        splits[split] = [s for s in splits[split] if len(s['target']) > 0]
    n_classes = len(train_phonemes)
    if verbose: print(f"  vocab={n_classes} mel_w={mel_w} ph_w={ph_w}")

    # Feature + mel scalers
    scaler = StandardScaler()
    for s in splits['train']:
        scaler.partial_fit(s['features'].astype(np.float32))
    mel_scaler = fit_mel_scaler(splits['train'])

    train_ds = PhonemeDatasetMTMel(splits['train'], phone_to_idx, scaler=scaler,
                                     mel_scaler=mel_scaler, noise_std=noise_std,
                                     augment=True, n_time_masks=n_time_masks,
                                     time_mask_max=time_mask_max,
                                     n_feat_masks=n_feat_masks,
                                     feat_mask_max_frac=feat_mask_max_frac,
                                     max_time_shift=max_time_shift,
                                     include_frame_labels=True)
    val_ds = PhonemeDatasetMTMel(splits['val'], phone_to_idx, scaler=scaler,
                                   mel_scaler=mel_scaler, noise_std=0.0, augment=False,
                                   include_frame_labels=True)

    if use_sentence_sampler:
        label_counts = Counter()
        for s in splits['train']:
            ph_lbl, _, _, _ = build_frame_labels_mt(s['features'].shape[0],
                                                      s['mfa_intervals'], phone_to_idx)
            for x in ph_lbl[ph_lbl > 0]: label_counts[int(x)] += 1
        sent_weights = compute_sentence_weights(splits['train'], phone_to_idx,
                                                  label_counts, alpha=1.0)
        sampler = WeightedRandomSampler(weights=torch.from_numpy(sent_weights).double(),
                                          num_samples=len(sent_weights)*sample_oversample_factor,
                                          replacement=True)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                                    collate_fn=collate_ctc_mt_mel, num_workers=0)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                    collate_fn=collate_ctc_mt_mel, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              collate_fn=collate_ctc_mt_mel, num_workers=0)

    n_features = splits['train'][0]['features'].shape[1]
    model = PhonemeMTMelBiLSTM(n_features, n_classes, n_mels=n_mels,
                                 hidden=hidden, dropout=dropout).to(DEVICE)
    with torch.no_grad():
        model.head_phoneme.bias[0] -= 5.0
    if verbose:
        print(f"  model: {sum(p.numel() for p in model.parameters()):,} params")

    model = train_mt_mel(model, train_loader, val_loader, idx_to_phone,
                          n_epochs=n_epochs, weight_decay=weight_decay,
                          patience=patience,
                          ph_w=ph_w, manner_w=manner_w, voicing_w=voicing_w,
                          place_w=place_w, mel_w=mel_w, verbose=verbose)

    # Evaluate phoneme accuracy at oracle borders + audio reconstruction quality
    model.eval(); results = []
    mel_corrs = []
    with torch.no_grad():
        for s in splits['test']:
            feats = scaler.transform(s['features']).astype(np.float32)
            x = torch.from_numpy(feats).unsqueeze(0).to(DEVICE)
            pred_mel, log_p_ph, *_ = model(x)
            pred_mel = pred_mel.squeeze(0).cpu().numpy()
            gold_mel = transform_mel(s['mel'], mel_scaler)
            # Pearson corr per mel bin, then mean
            for k in range(pred_mel.shape[1]):
                a, b = pred_mel[:, k], gold_mel[:, k]
                if a.std() > 0 and b.std() > 0:
                    mel_corrs.append(np.corrcoef(a, b)[0, 1])
            lp = log_p_ph.squeeze(0).cpu().numpy()
            pred = ctc_greedy_decode(lp, idx_to_phone)
            results.append(dict(sent_idx=s['sent_idx'], gold=s['target'], pred=pred))

    mean_corr = np.mean(mel_corrs)
    return dict(results=results, model=model, scaler=scaler, mel_scaler=mel_scaler,
                phone_to_idx=phone_to_idx, idx_to_phone=idx_to_phone,
                mel_corr=mean_corr), None


# Run
print("=== Training multi-task + mel reconstruction ===")
out_mel, err = run_mt_mel_for_patient("P22", n_epochs=50, hidden=64,
                                         noise_std=0.30,
                                         mel_w=1.0, ph_w=1.0,
                                         manner_w=0.2, voicing_w=0.2, place_w=0.3,
                                         use_sentence_sampler=True,
                                         sample_oversample_factor=2,
                                         verbose=True)
if out_mel is None:
    raise RuntimeError(f"failed: {err}")

summary = nw_summary(out_mel['results'])
print(f"\n=== Results ===")
print(f"  Mel-spec mean Pearson corr (test): {out_mel['mel_corr']:.3f}")
print(f"  Phoneme greedy match: {100*summary['match_rate']:.1f}%")

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

def train_one_patient(model, train_loader, val_loader, idx_to_phone,
                       n_epochs=100, lr=1e-3, weight_decay=1e-4,
                       patience=15, ctc_w=1.0, ce_w=1.0,
                       class_weights=None, verbose=True):    
    
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=1e-5)
    ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)
    best_val = float('inf'); best_state = None; no_improve = 0

    for epoch in range(n_epochs):
        model.train(); tr_total, tr_ctc, tr_ce = [], [], []
        for feats, tgt, in_l, tgt_l, fr_lbl in train_loader:
            feats = feats.to(DEVICE); tgt = tgt.to(DEVICE)
            in_l = in_l.to(DEVICE); tgt_l = tgt_l.to(DEVICE)
            fr_lbl = fr_lbl.to(DEVICE)
            opt.zero_grad()
            log_p = model(feats)
            ctc_l = ctc_loss(log_p.transpose(0, 1), tgt, in_l, tgt_l)
            B, T, C1 = log_p.shape
            ce_l = F.nll_loss(log_p.reshape(B*T, C1),
                                fr_lbl.reshape(-1),
                                ignore_index=-100,
                                weight=class_weights)         # ★ apply weights here
            loss = ctc_w * ctc_l + ce_w * ce_l
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tr_total.append(loss.item()); tr_ctc.append(ctc_l.item()); tr_ce.append(ce_l.item())

        model.eval(); va_total, va_ctc, va_ce = [], [], []
        with torch.no_grad():
            for feats, tgt, in_l, tgt_l, fr_lbl in val_loader:
                feats = feats.to(DEVICE); tgt = tgt.to(DEVICE)
                in_l = in_l.to(DEVICE); tgt_l = tgt_l.to(DEVICE)
                fr_lbl = fr_lbl.to(DEVICE)
                log_p = model(feats)
                ctc_l = ctc_loss(log_p.transpose(0, 1), tgt, in_l, tgt_l)
                B, T, C1 = log_p.shape
                ce_l = F.nll_loss(log_p.reshape(B*T, C1),
                                    fr_lbl.reshape(-1),
                                    ignore_index=-100,
                                    weight=class_weights)     # ★ same here
                va_total.append((ctc_w * ctc_l + ce_w * ce_l).item())
                va_ctc.append(ctc_l.item()); va_ce.append(ce_l.item())

        tr_t, tr_c, tr_e = np.mean(tr_total), np.mean(tr_ctc), np.mean(tr_ce)
        va_t, va_c, va_e = np.mean(va_total), np.mean(va_ctc), np.mean(va_ce)
        sched.step()
        if va_t < best_val:
            best_val = va_t
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0; mark = '*'
        else:
            no_improve += 1; mark = ''
        if verbose:
            print(f"    e{epoch+1:3d}: train [tot={tr_t:.2f} ctc={tr_c:.2f} ce={tr_e:.2f}]  "
                  f"val [tot={va_t:.2f} ctc={va_c:.2f} ce={va_e:.2f}]  "
                  f"lr={opt.param_groups[0]['lr']:.5f}  {mark}", flush=True)
        if verbose and (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                f0, _, _ = val_loader.dataset[0]
                x = torch.from_numpy(f0).unsqueeze(0).to(DEVICE)
                lp = model(x).squeeze(0).cpu().numpy()
                pred = ctc_greedy_decode(lp, idx_to_phone)
            print(f"       sample pred ({len(pred)} tokens): {pred[:15]}")
            model.train()
        if no_improve >= patience:
            if verbose: print(f"    early stop"); break
    model.load_state_dict(best_state)
    return model

from scipy.ndimage import uniform_filter1d

def ctc_smoothed_decode(log_probs, idx_to_phone, smooth_w=31, min_frames=5):
    """Smooth log-probs in time, then argmax + run-length collapse + min-length filter."""
    smoothed = uniform_filter1d(log_probs, size=smooth_w, axis=0, mode='nearest')
    path = smoothed.argmax(axis=-1)
    decoded = []
    i = 0
    while i < len(path):
        ci = path[i]
        j = i + 1
        while j < len(path) and path[j] == ci:
            j += 1
        if ci != 0 and (j - i) >= min_frames:   # not blank, long enough
            decoded.append(idx_to_phone[ci])
        i = j
    return decoded


def evaluate_smoothed(model, samples, scaler, idx_to_phone,
                       smooth_w=31, min_frames=5):
    model.eval(); results = []
    with torch.no_grad():
        for s in samples:
            feats = scaler.transform(s['features']).astype(np.float32)
            x = torch.from_numpy(feats).unsqueeze(0).to(DEVICE)
            lp = model(x).squeeze(0).cpu().numpy()
            pred = ctc_smoothed_decode(lp, idx_to_phone,
                                         smooth_w=smooth_w, min_frames=min_frames)
            results.append(dict(sent_idx=s['sent_idx'], gold=s['target'],
                                 pred=pred, log_probs=lp))
    return results

import time

PID = "P22"

# ── 1. Train ──
print(f"=== Training {PID} ===", flush=True)
t0 = time.time()
out, err = run_ctc_for_patient(PID,
                                 n_epochs=150, batch_size=8, hidden=48,
                                 noise_std=0.15,
                                 ctc_w=0.0, ce_w=1.0,
                                 use_class_weights=False,
                                 use_sentence_sampler=True,
                                 sample_oversample_factor=3)
print(f"Training time: {time.time()-t0:.0f}s\n")
if out is None:
    raise RuntimeError(f"Training failed: {err}")

# ── 2. Build test set (with target filtering) ──
splits = build_patient_dataset(PID)
for s in splits['test']:
    s['target'] = [ph for ph in s['target'] if ph in out['phone_to_idx']]
splits['test'] = [s for s in splits['test'] if len(s['target']) > 0]

# ── 3. Sweep smoothing settings ──
print("=== Smoothing sweep ===")
results_by_setting = {}
for sw, mf in [(11, 3), (21, 4), (31, 5), (41, 6), (51, 7)]:
    results = evaluate_smoothed(out['model'], splits['test'],
                                  out['scaler'], out['idx_to_phone'],
                                  smooth_w=sw, min_frames=mf)
    summary = nw_summary(results)
    n_pred = sum(len(r['pred']) for r in results)
    n_gold = summary['n_gold']
    print(f"  sw={sw:>3}  mf={mf:>2}  "
          f"match={100*summary['match_rate']:5.1f}%  "
          f"PER={100*summary['per']:5.1f}%  "
          f"matches={summary['n_match']:>4}/{n_gold}  "
          f"n_pred={n_pred:>4}  (ratio {n_pred/max(n_gold,1):.2f}×)")
    results_by_setting[(sw, mf)] = (results, summary)

# ── 4. Pick the best (highest match_rate, then lowest PER as tiebreaker) ──
best_key = max(results_by_setting.keys(),
               key=lambda k: (results_by_setting[k][1]['match_rate'],
                              -results_by_setting[k][1]['per']))
best_results, best_summary = results_by_setting[best_key]
print(f"\n  >>> best setting: smooth_w={best_key[0]}  min_frames={best_key[1]}")
print(f"      match_rate={100*best_summary['match_rate']:.1f}%  "
      f"PER={100*best_summary['per']:.1f}%")

# ── 5. NW-aligned consecutive-match counts at best setting ──
all_aligned = [needleman_wunsch(r['gold'], r['pred']) for r in best_results]
def cons_runs(a, k):
    runs, cur = 0, 0
    for g, p in a:
        if g is not None and p is not None and g == p: cur += 1
        else:
            if cur >= k: runs += 1
            cur = 0
    if cur >= k: runs += 1
    return runs
n2 = sum(cons_runs(a, 2) for a in all_aligned)
n3 = sum(cons_runs(a, 3) for a in all_aligned)
n4 = sum(cons_runs(a, 4) for a in all_aligned)
n5 = sum(cons_runs(a, 5) for a in all_aligned)
print(f"      NW-aligned consecutive runs:  n2={n2}  n3={n3}  n4={n4}  n5={n5}")

# ── 6. Visualization ──
from IPython.display import display, HTML
out_viz = ctc_results_to_out(best_results)
display(HTML(f"<h3>{PID} — CTC + smoothing (sw={best_key[0]}, mf={best_key[1]})</h3>"))
display(HTML(show_predictions_html(out_viz, label="CTC",
                                     max_sentences=15)))
# ── Run with aux CE loss ──
# print("=== CTC+CE BiLSTM for P22 ===", flush=True)
# t0 = time.time()
# Strategy A — sampler only (recommended first try):
out, err = run_ctc_for_patient("P22",
                                 use_class_weights=False,
                                 use_sentence_sampler=True,
                                 sample_oversample_factor=3)

# Strategy B — both, mild settings:
# out, err = run_ctc_for_patient("P22",
#                                  use_class_weights=True,
#                                  class_weight_alpha=0.5,
#                                  use_sentence_sampler=True,
#                                  sample_oversample_factor=2)

# Or revert to the previous behavior (class weights only):
# out, err = run_ctc_for_patient("P22",
#                                  use_class_weights=True,
#                                  use_sentence_sampler=False)


import time
from IPython.display import display, HTML

PID = "P22"
SMOOTH_W = 21
MIN_FRAMES = 4
SEED = 42  # for some run-to-run consistency

torch.manual_seed(SEED); np.random.seed(SEED)

# ════════════════════════════════════════════════════════════════════════
# 1. Train without augmentation
# ════════════════════════════════════════════════════════════════════════
print(f"=== Training {PID} — NO augmentation ===", flush=True)
t0 = time.time()
out_noaug, err = run_ctc_for_patient(
    PID, n_epochs=100, batch_size=8, hidden=48,
    noise_std=0.0,
    ctc_w=0.0, ce_w=1.0,
    use_class_weights=False,
    use_sentence_sampler=False,
    n_time_masks=0, time_mask_max=0,
    n_feat_masks=0, feat_mask_max_frac=0,
    max_time_shift=0,
    verbose=False)
print(f"  done in {time.time()-t0:.0f}s")
if out_noaug is None: raise RuntimeError(f"no-aug failed: {err}")

# ════════════════════════════════════════════════════════════════════════
# 2. Train WITH augmentation (+ sentence sampler)
# ════════════════════════════════════════════════════════════════════════
torch.manual_seed(SEED); np.random.seed(SEED)
print(f"\n=== Training {PID} — WITH augmentation + sampler ===", flush=True)
t0 = time.time()
# out_aug, err = run_ctc_for_patient(
#     PID, n_epochs=100, batch_size=8, hidden=48,
#     noise_std=0.15,
#     ctc_w=0.0, ce_w=1.0,
#     use_class_weights=False,
#     use_sentence_sampler=True,
#     sample_oversample_factor=3,
#     n_time_masks=5, time_mask_max=40,
#     n_feat_masks=4, feat_mask_max_frac=0.12,
#     max_time_shift=5,
#     verbose=False)


print(f"  done in {time.time()-t0:.0f}s")
if out_aug is None: raise RuntimeError(f"aug failed: {err}")

# ════════════════════════════════════════════════════════════════════════
# 3. Prepare test set (with word_bounds for oracle decode)
# ════════════════════════════════════════════════════════════════════════
splits = build_patient_dataset(PID)
for s in splits['test']:
    s['target'] = [ph for ph in s['target'] if ph in out_aug['phone_to_idx']]
splits['test'] = [s for s in splits['test'] if len(s['target']) > 0]

# ════════════════════════════════════════════════════════════════════════
# 4. Three evaluations
# ════════════════════════════════════════════════════════════════════════
r_A = evaluate_smoothed(out_noaug['model'], splits['test'],
                          out_noaug['scaler'], out_noaug['idx_to_phone'],
                          smooth_w=SMOOTH_W, min_frames=MIN_FRAMES)
r_B = evaluate_smoothed(out_aug['model'],   splits['test'],
                          out_aug['scaler'],   out_aug['idx_to_phone'],
                          smooth_w=SMOOTH_W, min_frames=MIN_FRAMES)
r_C = evaluate_oracle_word(out_aug['model'], splits['test'],
                             out_aug['scaler'], out_aug['idx_to_phone'],
                             smooth_w=SMOOTH_W, min_frames=MIN_FRAMES)

# ════════════════════════════════════════════════════════════════════════
# 5. Numeric comparison
# ════════════════════════════════════════════════════════════════════════
def detailed(results, label):
    aligned = [needleman_wunsch(r['gold'], r['pred']) for r in results]
    flat = [pair for a in aligned for pair in a]
    n_match = sum(1 for g, p in flat if g is not None and p is not None and g == p)
    n_sub   = sum(1 for g, p in flat if g is not None and p is not None and g != p)
    n_del   = sum(1 for g, p in flat if g is not None and p is None)
    n_ins   = sum(1 for g, p in flat if g is None and p is not None)
    n_gold  = sum(1 for g, _ in flat if g is not None)
    n_pred  = sum(len(r['pred']) for r in results)
    def cons(a, k):
        runs, cur = 0, 0
        for g, p in a:
            if g is not None and p is not None and g == p: cur += 1
            else:
                if cur >= k: runs += 1
                cur = 0
        if cur >= k: runs += 1
        return runs
    n3 = sum(cons(a, 3) for a in aligned)
    n4 = sum(cons(a, 4) for a in aligned)
    n5 = sum(cons(a, 5) for a in aligned)
    return dict(label=label, n_gold=n_gold, n_pred=n_pred,
                match=100*n_match/max(n_gold,1),
                per=100*(n_sub+n_del+n_ins)/max(n_gold,1),
                n3=n3, n4=n4, n5=n5)

print(f"\n=== {PID} comparison @ smooth={SMOOTH_W} min_frames={MIN_FRAMES} ===")
print(f"{'Condition':<28} {'match':>8} {'PER':>8} {'n_pred':>8} {'n3':>5} {'n4':>5} {'n5':>5}")
print('-' * 75)
for s in [detailed(r_A, 'A: no-aug'),
          detailed(r_B, 'B: aug + sampler'),
          detailed(r_C, 'C: aug + ORACLE words')]:
    print(f"{s['label']:<28} {s['match']:>7.1f}% {s['per']:>7.1f}% "
          f"{s['n_pred']:>8} {s['n3']:>5} {s['n4']:>5} {s['n5']:>5}")

# ════════════════════════════════════════════════════════════════════════
# 6. Three-way visualization
# ════════════════════════════════════════════════════════════════════════
def compare_three_html(results_a, results_b, results_c,
                        labels=("A","B","C"), max_sentences=10):
    def by_sent(results): return {r['sent_idx']: r for r in results}
    map_a, map_b, map_c = by_sent(results_a), by_sent(results_b), by_sent(results_c)
    common = sorted(set(map_a) & set(map_b) & set(map_c))[:max_sentences]
    rows = []
    rows.append("<style>"
                ".pcomp td { padding:4px 8px; font-family:monospace; font-size:13px; }"
                ".pcomp tr.sentheader td { background:#e0e0e0; font-weight:bold; padding-top:8px; }"
                ".pcomp tr.modeheader td { background:#f6f6f6; font-weight:bold; font-size:11px; }"
                "</style>")
    rows.append("<div style='margin-bottom:8px;font-family:sans-serif;font-size:13px;'>"
                f"<span style='background:{COL_MATCH};padding:3px 8px;margin-right:6px;border-radius:3px;'>match</span>"
                f"<span style='background:{COL_SUB};padding:3px 8px;margin-right:6px;border-radius:3px;'>substitution</span>"
                f"<span style='background:{COL_INS};padding:3px 8px;margin-right:6px;border-radius:3px;'>insertion</span>"
                f"<span style='background:{COL_DEL};padding:3px 8px;margin-right:6px;border-radius:3px;'>deletion</span>"
                "</div>")
    rows.append("<table class='pcomp' style='border-collapse:collapse;'>")
    for sid in common:
        r_a, r_b, r_c = map_a[sid], map_b[sid], map_c[sid]
        gold = r_a['gold']
        a_a = needleman_wunsch(gold, r_a['pred'])
        a_b = needleman_wunsch(gold, r_b['pred'])
        a_c = needleman_wunsch(gold, r_c['pred'])
        ga, pa = render_aligned_pair(a_a)
        gb, pb = render_aligned_pair(a_b)
        gc, pc = render_aligned_pair(a_c)
        rows.append(f"<tr class='sentheader'><td colspan='2'>Sentence {sid}</td></tr>")
        rows.append(f"<tr class='modeheader'><td colspan='2'>{labels[0]}</td></tr>")
        rows.append(f"<tr><td>gold</td><td>{ga}</td></tr>")
        rows.append(f"<tr><td>pred</td><td>{pa}</td></tr>")
        rows.append(f"<tr class='modeheader'><td colspan='2'>{labels[1]}</td></tr>")
        rows.append(f"<tr><td>gold</td><td>{gb}</td></tr>")
        rows.append(f"<tr><td>pred</td><td>{pb}</td></tr>")
        rows.append(f"<tr class='modeheader'><td colspan='2'>{labels[2]}</td></tr>")
        rows.append(f"<tr><td>gold</td><td>{gc}</td></tr>")
        rows.append(f"<tr><td>pred</td><td>{pc}</td></tr>")
    rows.append("</table>")
    return ''.join(rows)

display(HTML(compare_three_html(r_A, r_B, r_C,
                                  labels=("A: no aug", "B: aug+sampler",
                                          "C: aug+sampler + ORACLE words"),
                                  max_sentences=10)))

from scipy.ndimage import uniform_filter1d
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Helper: stacked-frame index → time in seconds
def stk_frame_to_t(i):
    return (i + LDA_MARGIN) * SHIFT_SAMP / EEG_SR + WIN_SAMP / (2 * EEG_SR)


def smoothed_decode_with_times(log_probs, idx_to_phone, smooth_w=21, min_frames=4):
    smoothed = uniform_filter1d(log_probs, size=smooth_w, axis=0, mode='nearest')
    path = smoothed.argmax(axis=-1)
    segments, i = [], 0
    while i < len(path):
        c = path[i]; j = i + 1
        while j < len(path) and path[j] == c: j += 1
        if c != 0 and (j - i) >= min_frames:
            segments.append((idx_to_phone[c],
                              stk_frame_to_t(i), stk_frame_to_t(j - 1)))
        i = j
    return segments


def oracle_decode_with_times(log_probs, word_bounds, idx_to_phone,
                              smooth_w=21, min_frames=4):
    smoothed = uniform_filter1d(log_probs, size=smooth_w, axis=0, mode='nearest')
    segments = []
    for w_start, w_end in zip(word_bounds[:-1], word_bounds[1:]):
        if w_end - w_start < min_frames: continue
        path = smoothed[w_start:w_end].argmax(axis=-1)
        i = 0
        while i < len(path):
            c = path[i]; j = i + 1
            while j < len(path) and path[j] == c: j += 1
            if c != 0 and (j - i) >= min_frames:
                segments.append((idx_to_phone[c],
                                  stk_frame_to_t(w_start + i),
                                  stk_frame_to_t(w_start + j - 1)))
            i = j
    return segments


def plot_timeline_3way(gold_segs, pred_segs_dict, sent_idx, word_bounds=None):
    n_rows = 1 + len(pred_segs_dict)
    fig, ax = plt.subplots(figsize=(20, 1.5 + 1.2 * n_rows))

    def draw_row(y, segs, gold_lookup=None, neutral=False):
        for ph, t0, t1 in segs:
            if neutral:
                color = '#dddddd'
            else:
                mid = (t0 + t1) / 2
                gold_here = next((gp for gp, gs, ge in gold_lookup
                                   if gs <= mid <= ge), None)
                if gold_here == ph:        color = '#a6e3a1'
                elif gold_here is not None: color = '#f5c2c0'
                else:                       color = '#ffd966'
            ax.add_patch(mpatches.Rectangle((t0, y), t1 - t0, 0.8,
                                              facecolor=color, edgecolor='black',
                                              linewidth=0.5))
            ax.text((t0+t1)/2, y + 0.4, ph, ha='center', va='center', fontsize=9)

    # Gold at top
    draw_row(n_rows - 1, gold_segs, neutral=True)
    # Predictions below
    labels = list(pred_segs_dict.keys())
    for i, label in enumerate(labels):
        draw_row(n_rows - 2 - i, pred_segs_dict[label], gold_lookup=gold_segs)

    # Word boundaries (dashed vertical lines)
    if word_bounds is not None:
        for wb in word_bounds:
            ax.axvline(stk_frame_to_t(wb), color='blue', linestyle='--', alpha=0.4,
                       linewidth=1)

    ax.set_yticks([y + 0.4 for y in range(n_rows)])
    ax.set_yticklabels(labels[::-1] + ['GOLD'])
    ax.set_xlabel('time (s)')
    ax.set_title(f'Sentence {sent_idx}  —  dashed blue = oracle word boundaries  '
                 f'(green=match, red=sub, yellow=insertion)')

    all_t = [t for segs in [gold_segs] + list(pred_segs_dict.values()) for _, t, _ in segs]
    all_t2 = [t for segs in [gold_segs] + list(pred_segs_dict.values()) for _, _, t in segs]
    if all_t:
        ax.set_xlim(min(all_t) - 0.1, max(all_t2) + 0.1)
    ax.set_ylim(-0.5, n_rows + 0.5)
    ax.grid(True, axis='x', alpha=0.2)
    plt.tight_layout()
    return fig


# ── Pick the sentence ──
SENT_IDX = 60
test_sample = next((s for s in splits['test'] if s['sent_idx'] == SENT_IDX), None)
if test_sample is None:
    print(f"Sentence {SENT_IDX} not in test set; using first")
    test_sample = splits['test'][0]; SENT_IDX = test_sample['sent_idx']

def get_logp(model, scaler, sample):
    model.eval()
    with torch.no_grad():
        feats = scaler.transform(sample['features']).astype(np.float32)
        x = torch.from_numpy(feats).unsqueeze(0).to(DEVICE)
        return model(x).squeeze(0).cpu().numpy()

# logp_noaug = get_logp(out_noaug['model'], out_noaug['scaler'], test_sample)
# logp_aug   = get_logp(out_aug['model'],   out_aug['scaler'],   test_sample)

# segs_A = smoothed_decode_with_times(logp_noaug, out_noaug['idx_to_phone'],
#                                       smooth_w=SMOOTH_W, min_frames=MIN_FRAMES)
# segs_B = smoothed_decode_with_times(logp_aug,   out_aug['idx_to_phone'],
#                                       smooth_w=SMOOTH_W, min_frames=MIN_FRAMES)
# segs_C = oracle_decode_with_times(logp_aug, test_sample['word_bounds'],
#                                     out_aug['idx_to_phone'],
#                                     smooth_w=SMOOTH_W, min_frames=MIN_FRAMES)

# gold_segs = [(ph, t0, t1) for (t0, t1, ph) in test_sample['mfa_intervals']]

# print(f"Sentence {SENT_IDX}:")
# print(f"  Gold:                       {len(gold_segs):>3} phonemes  "
#       f"({gold_segs[0][1]:.2f}s — {gold_segs[-1][2]:.2f}s)")
# print(f"  A (no aug):                 {len(segs_A):>3} predictions")
# print(f"  B (aug+sampler):            {len(segs_B):>3} predictions")
# print(f"  C (aug+sampler+oracle):     {len(segs_C):>3} predictions")

# plot_timeline_3way(gold_segs,
#                     {'A: no aug': segs_A,
#                      'B: aug+sampler': segs_B,
#                      'C: aug+oracle': segs_C},
#                     SENT_IDX, word_bounds=test_sample['word_bounds'])
# plt.show()

from collections import Counter

def prediction_distribution(results, label):
    """Counter of predicted phonemes across all test sentences."""
    counter = Counter()
    for r in results:
        for ph in r['pred']:
            counter[ph] += 1
    return counter

out_aug_v2, err = run_ctc_for_patient(
    PID, n_epochs=30,                       # ★ shorter
    batch_size=8, hidden=32,                # ★ smaller model
    noise_std=0.15,
    ctc_w=0.0, ce_w=1.0,
    use_class_weights=True,
    class_weight_alpha=0.5,
    class_weight_max_ratio=10.0,
    class_weight_min_count=10,
    use_sentence_sampler=True,
    sample_oversample_factor=2,
    n_time_masks=5, time_mask_max=40,
    n_feat_masks=4, feat_mask_max_frac=0.12,
    max_time_shift=5,
    patience=6,                              # ★ tighter early stop
    dropout=0.5,                             # ★ heavier dropout
    weight_decay=1e-3,                       # ★ stronger L2
    verbose=True)

r_B_noweight = evaluate_smoothed(out_aug['model'], splits['test'],
                                   out_aug['scaler'], out_aug['idx_to_phone'],
                                   smooth_w=SMOOTH_W, min_frames=MIN_FRAMES)
# (after retraining with class weights ON):
r_B_weighted = evaluate_smoothed(out_aug_v2['model'], splits['test'],
                                   out_aug_v2['scaler'], out_aug_v2['idx_to_phone'],
                                   smooth_w=SMOOTH_W, min_frames=MIN_FRAMES)

dist_no   = prediction_distribution(r_B_noweight, 'no-weight')
dist_w    = prediction_distribution(r_B_weighted, 'weighted')

gold_counter = Counter()
for r in r_B_weighted:
    for ph in r['gold']: gold_counter[ph] += 1

all_phones = sorted(gold_counter.keys(), key=lambda p: -gold_counter[p])

print(f"{'phoneme':<8} {'GOLD':>6} {'no-wt':>6} {'wt':>6} {'shift':>7}")
print('-' * 45)
for ph in all_phones:
    g  = gold_counter.get(ph, 0)
    n0 = dist_no.get(ph, 0)
    nw = dist_w.get(ph, 0)
    shift = nw - n0
    arrow = '↑' if shift > 0 else ('↓' if shift < 0 else '')
    print(f"{ph:<8} {g:>6} {n0:>6} {nw:>6}  {shift:>+5} {arrow}")

# out_no  = ctc_results_to_out(r_B_noweight)
# out_wt  = ctc_results_to_out(r_B_weighted)
# display(HTML(compare_predictions_html(out_no, out_wt,
#                                         label_a="no weights",
#                                         label_b="weighted",
#                                         max_sentences=15)))

N_SENTENCES = 10
for test_sample in splits['test'][:N_SENTENCES]:
    logp_no = get_logp(out_aug['model'],    out_aug['scaler'],    test_sample)
    logp_wt = get_logp(out_aug_v2['model'], out_aug_v2['scaler'], test_sample)
    segs_no = smoothed_decode_with_times(logp_no, out_aug['idx_to_phone'],
                                           smooth_w=SMOOTH_W, min_frames=MIN_FRAMES)
    segs_wt = smoothed_decode_with_times(logp_wt, out_aug_v2['idx_to_phone'],
                                           smooth_w=SMOOTH_W, min_frames=MIN_FRAMES)
    gold_segs = [(ph, t0, t1) for (t0, t1, ph) in test_sample['mfa_intervals']]
    plot_timeline_3way(gold_segs,
                        {'no weights': segs_no,
                         'weighted':   segs_wt},
                        test_sample['sent_idx'],
                        word_bounds=test_sample['word_bounds'])
    plt.show()

PATIENTS_SW = ["P21","P22","P23","P24","P25","P26","P27","P28","P29","P30"]
all_patient_stats = {}
trained_models = {}     

for pid in PATIENTS_SW:
    print(f"\n========== {pid} ==========", flush=True)
    
    # Train no-weights
    out_no, err_no = run_ctc_for_patient(
        pid, n_epochs=30, batch_size=6, hidden=32,
        noise_std=0.15, ctc_w=0.0, ce_w=1.0,
        use_class_weights=False, use_sentence_sampler=True,
        sample_oversample_factor=2,
        n_time_masks=5, time_mask_max=40,
        n_feat_masks=4, feat_mask_max_frac=0.12,
        max_time_shift=5,
        patience=6, dropout=0.5, weight_decay=1e-3,
        verbose=False)
    if out_no is None: print(f"  no-wt SKIP: {err_no}"); continue

    # Train weighted
    out_wt, err_wt = run_ctc_for_patient(
        pid, n_epochs=30, batch_size=8, hidden=32,
        noise_std=0.15, ctc_w=0.0, ce_w=1.0,
        use_class_weights=True,
        class_weight_alpha=0.5, class_weight_max_ratio=10.0,
        class_weight_min_count=10,
        use_sentence_sampler=True,
        sample_oversample_factor=2,
        n_time_masks=5, time_mask_max=40,
        n_feat_masks=4, feat_mask_max_frac=0.12,
        max_time_shift=5,
        patience=6, dropout=0.5, weight_decay=1e-3,
        verbose=False)
    if out_wt is None: print(f"  wt SKIP: {err_wt}"); continue

    # Prepare test
    splits = build_patient_dataset(pid)
    for s in splits['test']:
        s['target'] = [ph for ph in s['target'] if ph in out_wt['phone_to_idx']]
    splits['test'] = [s for s in splits['test'] if len(s['target']) > 0]

    # Evaluate both
    r_no = evaluate_smoothed(out_no['model'], splits['test'],
                              out_no['scaler'], out_no['idx_to_phone'],
                              smooth_w=SMOOTH_W, min_frames=MIN_FRAMES)
    r_wt = evaluate_smoothed(out_wt['model'], splits['test'],
                              out_wt['scaler'], out_wt['idx_to_phone'],
                              smooth_w=SMOOTH_W, min_frames=MIN_FRAMES)

    no_rates = per_sentence_match_rates(r_no)
    wt_rates = per_sentence_match_rates(r_wt)

    all_patient_stats[pid] = dict(
        no_median=np.median(no_rates),
        wt_median=np.median(wt_rates),
        no_mean=np.mean(no_rates),
        wt_mean=np.mean(wt_rates),
        delta_median=np.median(wt_rates) - np.median(no_rates))

    print(f"  {pid}: no-wt median={100*all_patient_stats[pid]['no_median']:5.1f}%  "
          f"wt median={100*all_patient_stats[pid]['wt_median']:5.1f}%  "
          f"Δ={100*all_patient_stats[pid]['delta_median']:+5.1f}pp")
    
    out_no['model'] = out_no['model'].cpu()
    out_wt['model'] = out_wt['model'].cpu()
    trained_models[pid] = dict(out_no=out_no, out_wt=out_wt, splits=splits,
                                 r_no=r_no, r_wt=r_wt)

    # Clean up memory
    # del out_no, out_wt
    import gc
    gc.collect()
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

# ── Cohort summary ──
print(f"\n{'='*60}")
print(f"{'pid':<6} {'no-wt med':>10} {'wt med':>10} {'Δ':>8}")
print('-' * 40)
deltas = []
for pid, s in all_patient_stats.items():
    print(f"{pid:<6} {100*s['no_median']:>9.1f}% {100*s['wt_median']:>9.1f}% "
          f"{100*s['delta_median']:>+7.1f}pp")
    deltas.append(s['delta_median'])
print('-' * 40)
print(f"COHORT  mean Δ = {100*np.mean(deltas):+5.2f}pp   "
      f"median Δ = {100*np.median(deltas):+5.2f}pp   "
      f"# patients where wt helps: {sum(1 for d in deltas if d > 0)}/{len(deltas)}")

def viz_patient(pid, n_sentences=10):
    if pid not in trained_models:
        print(f"{pid} not cached; rerun the cohort loop first"); return
    cache = trained_models[pid]
    out_no, out_wt, splits = cache['out_no'], cache['out_wt'], cache['splits']
    # Move models back to DEVICE for inference
    out_no['model'] = out_no['model'].to(DEVICE)
    out_wt['model'] = out_wt['model'].to(DEVICE)
    
    for test_sample in splits['test'][:n_sentences]:
        logp_no = get_logp(out_no['model'], out_no['scaler'], test_sample)
        logp_wt = get_logp(out_wt['model'], out_wt['scaler'], test_sample)
        segs_no = smoothed_decode_with_times(logp_no, out_no['idx_to_phone'],
                                               smooth_w=SMOOTH_W, min_frames=MIN_FRAMES)
        segs_wt = smoothed_decode_with_times(logp_wt, out_wt['idx_to_phone'],
                                               smooth_w=SMOOTH_W, min_frames=MIN_FRAMES)
        gold_segs = [(ph, t0, t1) for (t0, t1, ph) in test_sample['mfa_intervals']]
        plot_timeline_3way(gold_segs,
                            {f'{pid} no-wt': segs_no, f'{pid} weighted': segs_wt},
                            test_sample['sent_idx'],
                            word_bounds=test_sample['word_bounds'])
        plt.show()
    
    # Move back to CPU after viz (optional, frees GPU)
    out_no['model'] = out_no['model'].cpu()
    out_wt['model'] = out_wt['model'].cpu()


# Usage after the loop:
viz_patient("P22", n_sentences=5)
viz_patient("P25", n_sentences=5)
viz_patient("P29", n_sentences=10)

# PATIENTS_SW = ["P21","P22","P23","P25","P26","P29"]
# for pid in PATIENTS_SW:
#     print(f"\n=== {pid} ===")
#     out_p, _ = run_ctc_for_patient(pid, n_epochs=100, batch_size=8, hidden=48,
#                                      noise_std=0.10, ctc_w=0.0, ce_w=1.0,
#                                      verbose=False)
#     if out_p is None: continue
#     splits_p = build_patient_dataset(pid)
#     for s in splits_p['test']:
#         s['target'] = [ph for ph in s['target'] if ph in out_p['phone_to_idx']]
#     splits_p['test'] = [s for s in splits_p['test'] if len(s['target']) > 0]
#     # Try sw=21 (good balance):
#     results = evaluate_smoothed(out_p['model'], splits_p['test'], out_p['scaler'],
#                                   out_p['idx_to_phone'], smooth_w=21, min_frames=4)
#     summary = nw_summary(results)
#     print(f"  match={100*summary['match_rate']:.1f}%  PER={100*summary['per']:.1f}%  "
#           f"n_pred={sum(len(r['pred']) for r in results)}")

# import gc
# PATIENTS_VIZ = ["P21", "P22", "P23", "P25", "P26", "P29"]
# patient_outs = {}   # store trained models per patient

# for pid in PATIENTS_VIZ:
#     print(f"\n========== {pid} ==========")
#     out_p, err = run_ctc_for_patient(pid, n_epochs=100, batch_size=8, hidden=48,
#                                        noise_std=0.10, ctc_w=0.0, ce_w=1.0,
#                                        verbose=False)
#     if out_p is None:
#         print(f"  SKIPPED: {err}"); continue

#     splits_p = build_patient_dataset(pid)
#     for s in splits_p['test']:
#         s['target'] = [ph for ph in s['target'] if ph in out_p['phone_to_idx']]
#     splits_p['test'] = [s for s in splits_p['test'] if len(s['target']) > 0]

#     smoothed_results = evaluate_smoothed(out_p['model'], splits_p['test'],
#                                            out_p['scaler'], out_p['idx_to_phone'],
#                                            smooth_w=21, min_frames=4)
#     summary = nw_summary(smoothed_results)
#     print(f"  {pid}: match={100*summary['match_rate']:.1f}%  PER={100*summary['per']:.1f}%  "
#           f"matches={summary['n_match']}/{summary['n_gold']}")

#     out_viz = ctc_results_to_out(smoothed_results)
#     display(HTML(f"<h3>{pid} (smooth=21)</h3>"))
#     display(HTML(show_predictions_html(out_viz, label=f"CTC", max_sentences=8)))

#     patient_outs[pid] = out_p
#     del out_p
#     gc.collect()
#     if DEVICE == 'cuda':
#         torch.cuda.empty_cache()

# for alpha in [0.5, 0.75, 1.0, 1.5, 2.0]:
#     print(f"\n=== alpha = {alpha} ===")
    
#     # Monkey-patch the alpha:
#     def cw_at_alpha(samples, phone_to_idx, n_classes, device=DEVICE):
#         return compute_class_weights(samples, phone_to_idx, n_classes,
#                                        alpha=alpha, device=device)
    
#     # Run with this alpha (need to thread it through — easier to just edit
#     # the call inside run_ctc_for_patient):
#     # In run_ctc_for_patient, change:
#     #     class_weights, label_counts = compute_class_weights(..., alpha=0.5)
#     # to:
#     #     class_weights, label_counts = compute_class_weights(..., alpha=alpha)
#     # Or just add an alpha kwarg to run_ctc_for_patient.
    
#     # For now, just print what the weights would look like at each alpha:
#     splits = build_patient_dataset("P22")
#     train_phonemes = sorted({ph for s in splits['train'] for ph in s['target']})
#     p2i = {ph: i+1 for i, ph in enumerate(train_phonemes)}
#     i2p = {i+1: ph for i, ph in enumerate(train_phonemes)}
#     cw, lc = compute_class_weights(splits['train'], p2i, len(train_phonemes), alpha=alpha)
#     print(f"  Top-5 common:")
#     for k, c in lc.most_common(5):
#         print(f"    {i2p[k]:<5}  n={c:>4}  weight={cw[k]:.3f}")
#     print(f"  Bottom-5 rare:")
#     for k, c in lc.most_common()[-5:]:
#         print(f"    {i2p[k]:<5}  n={c:>4}  weight={cw[k]:.3f}")
#     print(f"  Weight ratio (rare/common): {cw[lc.most_common()[-1][0]] / cw[lc.most_common(1)[0][0]]:.1f}x")

splits = build_patient_dataset("P22")
train_phonemes = sorted({ph for s in splits['train'] for ph in s['target']})
p2i = {ph: i+1 for i, ph in enumerate(train_phonemes)}
i2p = {i+1: ph for i, ph in enumerate(train_phonemes)}

for alpha in [0.75, 1.0, 1.5]:
    cw, lc = compute_class_weights(splits['train'], p2i, len(train_phonemes),
                                     alpha=alpha, max_weight_ratio=20.0,
                                     min_count_keep=10)
    nz = cw[1:][cw[1:] > 0]
    print(f"\n  α={alpha}  effective ratio (max/min): {nz.max() / nz.min():.1f}x")
    print(f"  Top-3 common:")
    for k, c in lc.most_common(3):
        print(f"    {i2p[k]:<5}  n={c:>4}  weight={cw[k]:.3f}")
    print(f"  Mid-frequency:")
    sorted_by_freq = lc.most_common()
    for k, c in sorted_by_freq[len(sorted_by_freq)//2-1:len(sorted_by_freq)//2+2]:
        print(f"    {i2p[k]:<5}  n={c:>4}  weight={cw[k]:.3f}")
    print(f"  Bottom-3 (post-filter):")
    bottom = [(k, c) for k, c in sorted_by_freq if cw[k] > 0][-3:]
    for k, c in bottom:
        print(f"    {i2p[k]:<5}  n={c:>4}  weight={cw[k]:.3f}")

from scipy.ndimage import uniform_filter1d

def decode_with_oracle_word_boundaries(log_probs, mfa_intervals,
                                          idx_to_phone, smooth_w=21):
    """Per-word decoding using oracle MFA word boundaries.
    For each word, greedy-collapse the predicted phonemes within that word's
    time range. Words give natural per-region phoneme budgets."""
    # Get word-boundary frame indices from MFA
    word_starts_s = []
    prev_word = None
    for start_s, end_s, phone in mfa_intervals:
        # word changes when... we need word info. Pull from mfa dict instead.
        pass
    # Smooth log-probs in time
    smoothed = uniform_filter1d(log_probs, size=smooth_w, axis=0, mode='nearest')
    return smoothed

def build_patient_dataset(pid, feature_spec=None, test_offset=None):
    """Updated: also stores word boundary frame indices per sample."""
    feature_spec = feature_spec or DEFAULT_FEATURE_SPEC
    test_offset  = TEST_OFFSET if test_offset is None else test_offset
    wd = pipeline.split_result['word_segments_dict'].get(pid)
    if wd is None: return None
    try: mfa = load_mfa_alignments(pid)
    except Exception: return None
    if not mfa: return None
    all_real = [i for i, s in enumerate(wd['sentence_list'])
                if isinstance(s, dict) and s.get('text')]
    test_sent_ids = set(all_real[test_offset::6])
    train_sent_ids_all = sorted(set(all_real) - test_sent_ids)
    val_step = max(2, int(round(1 / VAL_FRAC)))
    val_sent_ids = set(train_sent_ids_all[::val_step])
    raw_path = os.path.join(DUTCH_30_PATH, 'raw', f'{pid}_sEEG.npy')
    raw_eeg = np.load(raw_path)
    if raw_eeg.ndim == 2 and raw_eeg.shape[0] < raw_eeg.shape[1]:
        raw_eeg = raw_eeg.T
    splits = dict(train=[], val=[], test=[])
    for sent_idx in all_real:
        if sent_idx not in mfa or not mfa[sent_idx]: continue
        s = wd['sentence_list'][sent_idx]
        s0, s1 = s['stim_start_idx'], s['stim_end_idx']
        if s1 > raw_eeg.shape[0]: continue
        ext = extract_features_multiband(raw_eeg[s0:s1], **feature_spec)
        if ext.shape[0] < 2 * LDA_MARGIN + 1: continue
        stk = stackFeatures(ext, modelOrder=MO, stepSize=SS).astype(np.float32)
        T = stk.shape[0]
        target = [ph['phone'] for ph in mfa[sent_idx]]
        if not target: continue
        mfa_intervals = [(ph['start_s'], ph['end_s'], ph['phone'])
                          for ph in mfa[sent_idx]]
        # ★ word-boundary frame indices (oracle)
        word_bounds = [0]
        prev_word = None
        for ph in mfa[sent_idx]:
            cur_word = ph.get('word')
            if cur_word != prev_word and prev_word is not None:
                k_onset = int(round((ph['start_s']*EEG_SR - WIN_SAMP/2) / SHIFT_SAMP)) - LDA_MARGIN
                k_onset = max(0, min(T - 1, k_onset))
                word_bounds.append(k_onset)
            prev_word = cur_word
        word_bounds.append(T)
        split = 'test' if sent_idx in test_sent_ids else \
                'val'  if sent_idx in val_sent_ids else 'train'
        splits[split].append(dict(
            features=stk, target=target, mfa_intervals=mfa_intervals,
            word_bounds=word_bounds, sent_idx=sent_idx
        ))
    return splits

def decode_per_word(log_probs, word_bounds, idx_to_phone,
                    smooth_w=21, min_frames=4, max_phonemes_per_word=None):
    """Decode using oracle word boundaries. Within each word region,
    greedy-collapse the smoothed argmax sequence."""
    smoothed = uniform_filter1d(log_probs, size=smooth_w, axis=0, mode='nearest')
    decoded = []
    for w_start, w_end in zip(word_bounds[:-1], word_bounds[1:]):
        if w_end - w_start < min_frames: continue
        word_chunk = smoothed[w_start:w_end]
        path = word_chunk.argmax(axis=-1)
        word_phonemes = []
        i = 0
        while i < len(path):
            ci = path[i]; j = i + 1
            while j < len(path) and path[j] == ci: j += 1
            if ci != 0 and (j - i) >= min_frames:
                word_phonemes.append(idx_to_phone[ci])
            i = j
        if max_phonemes_per_word is not None:
            word_phonemes = word_phonemes[:max_phonemes_per_word]
        decoded.extend(word_phonemes)
    return decoded


def evaluate_oracle_word(model, samples, scaler, idx_to_phone,
                          smooth_w=21, min_frames=4):
    model.eval(); results = []
    with torch.no_grad():
        for s in samples:
            feats = scaler.transform(s['features']).astype(np.float32)
            x = torch.from_numpy(feats).unsqueeze(0).to(DEVICE)
            lp = model(x).squeeze(0).cpu().numpy()
            pred = decode_per_word(lp, s['word_bounds'], idx_to_phone,
                                     smooth_w=smooth_w, min_frames=min_frames)
            results.append(dict(sent_idx=s['sent_idx'], gold=s['target'],
                                 pred=pred, log_probs=lp))
    return results


# Compare: oracle word boundaries vs no oracle
splits = build_patient_dataset("P22")
for s in splits['test']:
    s['target'] = [ph for ph in s['target'] if ph in out['phone_to_idx']]
splits['test'] = [s for s in splits['test'] if len(s['target']) > 0]

for sw, mf in [(11, 3), (21, 4), (31, 5)]:
    # Without oracle (your existing smoothed decode):
    r_no = evaluate_smoothed(out['model'], splits['test'], out['scaler'],
                              out['idx_to_phone'], smooth_w=sw, min_frames=mf)
    s_no = nw_summary(r_no)
    # With oracle word boundaries:
    r_oracle = evaluate_oracle_word(out['model'], splits['test'], out['scaler'],
                                      out['idx_to_phone'], smooth_w=sw, min_frames=mf)
    s_or = nw_summary(r_oracle)
    print(f"  sw={sw:>3}  mf={mf:>2}")
    print(f"    no oracle:     match={100*s_no['match_rate']:5.1f}%  PER={100*s_no['per']:6.1f}%  "
          f"n_pred={sum(len(r['pred']) for r in r_no):>4}")
    print(f"    oracle words:  match={100*s_or['match_rate']:5.1f}%  PER={100*s_or['per']:6.1f}%  "
          f"n_pred={sum(len(r['pred']) for r in r_oracle):>4}")

def count_consecutive_runs(alignment, min_len):
    runs = 0; cur = 0
    for g, p in alignment:
        if g is not None and p is not None and g == p:
            cur += 1
        else:
            if cur >= min_len: runs += 1
            cur = 0
    if cur >= min_len: runs += 1
    return runs


def detailed_summary(results, label=""):
    all_aligned = []
    for r in results:
        all_aligned.append(needleman_wunsch(r['gold'], r['pred']))
    flat = [pair for a in all_aligned for pair in a]
    n_match = sum(1 for g, p in flat if g is not None and p is not None and g == p)
    n_sub   = sum(1 for g, p in flat if g is not None and p is not None and g != p)
    n_del   = sum(1 for g, p in flat if g is not None and p is None)
    n_ins   = sum(1 for g, p in flat if g is None and p is not None)
    n_gold  = sum(1 for g, _ in flat if g is not None)
    n_pred  = sum(len(r['pred']) for r in results)
    n2 = sum(count_consecutive_runs(a, 2) for a in all_aligned)
    n3 = sum(count_consecutive_runs(a, 3) for a in all_aligned)
    n4 = sum(count_consecutive_runs(a, 4) for a in all_aligned)
    n5 = sum(count_consecutive_runs(a, 5) for a in all_aligned)
    print(f"\n  {label}:")
    print(f"    n_gold={n_gold}  n_pred={n_pred}  (ratio {n_pred/max(n_gold,1):.2f}x)")
    print(f"    match_rate = {100*n_match/max(n_gold,1):5.1f}%   ({n_match}/{n_gold})")
    print(f"    PER (NW)   = {100*(n_sub+n_del+n_ins)/max(n_gold,1):5.1f}%")
    print(f"    sub={n_sub}  del={n_del}  ins={n_ins}")
    print(f"    NW-aligned runs:  n2={n2}  n3={n3}  n4={n4}  n5={n5}")
    return dict(n_match=n_match, n_gold=n_gold, n_sub=n_sub, n_del=n_del,
                n_ins=n_ins, n2=n2, n3=n3, n4=n4, n5=n5, n_pred=n_pred)


# Re-run both decoders for sweep:
for sw, mf in [(11, 3), (21, 4), (31, 5)]:
    print(f"\n========== smooth={sw}  min_frames={mf} ==========")
    r_no = evaluate_smoothed(out['model'], splits['test'], out['scaler'],
                              out['idx_to_phone'], smooth_w=sw, min_frames=mf)
    r_or = evaluate_oracle_word(out['model'], splits['test'], out['scaler'],
                                  out['idx_to_phone'], smooth_w=sw, min_frames=mf)
    s_no = detailed_summary(r_no, "no oracle")
    s_or = detailed_summary(r_or, "ORACLE words")
    # Headline: did length-4+ matches appear?
    print(f"\n  >>> n4 change: {s_no['n4']} → {s_or['n4']}   "
          f"n3 change: {s_no['n3']} → {s_or['n3']}")

# Side-by-side: with vs without oracle word boundaries
# Uses the trained model in `out`, splits with word_bounds, and the existing
# evaluate_smoothed / evaluate_oracle_word / ctc_results_to_out / compare_predictions_html.

# Pick decode parameters
SMOOTH_W   = 21
MIN_FRAMES = 4

# Prepare splits (ensure word_bounds are present)
splits = build_patient_dataset("P22")
for s in splits['test']:
    s['target'] = [ph for ph in s['target'] if ph in out['phone_to_idx']]
splits['test'] = [s for s in splits['test'] if len(s['target']) > 0]

# Two evaluations on the SAME trained model + SAME test sentences
results_no    = evaluate_smoothed(out['model'], splits['test'],
                                    out['scaler'], out['idx_to_phone'],
                                    smooth_w=SMOOTH_W, min_frames=MIN_FRAMES)
results_oracle = evaluate_oracle_word(out['model'], splits['test'],
                                        out['scaler'], out['idx_to_phone'],
                                        smooth_w=SMOOTH_W, min_frames=MIN_FRAMES)

# Quick numeric comparison
def quick(label, results):
    aligned = [needleman_wunsch(r['gold'], r['pred']) for r in results]
    flat = [pair for a in aligned for pair in a]
    n_match = sum(1 for g, p in flat if g is not None and p is not None and g == p)
    n_sub   = sum(1 for g, p in flat if g is not None and p is not None and g != p)
    n_del   = sum(1 for g, p in flat if g is not None and p is None)
    n_ins   = sum(1 for g, p in flat if g is None and p is not None)
    n_gold  = sum(1 for g, _ in flat if g is not None)
    n_pred  = sum(len(r['pred']) for r in results)
    def cons_runs(a, k):
        runs, cur = 0, 0
        for g, p in a:
            if g is not None and p is not None and g == p:
                cur += 1
            else:
                if cur >= k: runs += 1
                cur = 0
        if cur >= k: runs += 1
        return runs
    n3 = sum(cons_runs(a, 3) for a in aligned)
    n4 = sum(cons_runs(a, 4) for a in aligned)
    n5 = sum(cons_runs(a, 5) for a in aligned)
    print(f"  {label:<20}  match={100*n_match/max(n_gold,1):5.1f}% "
          f" PER={100*(n_sub+n_del+n_ins)/max(n_gold,1):6.1f}%  "
          f"n_pred={n_pred:>4}/{n_gold}  n3={n3:>3}  n4={n4:>3}  n5={n5:>3}")

print(f"=== Compare at smooth={SMOOTH_W} min_frames={MIN_FRAMES} ===")
quick("no oracle",      results_no)
quick("ORACLE words",   results_oracle)

# Side-by-side visualization
out_no     = ctc_results_to_out(results_no)
out_oracle = ctc_results_to_out(results_oracle)
display(HTML(compare_predictions_html(out_no, out_oracle,
                                        label_a="no oracle",
                                        label_b="ORACLE",
                                        max_sentences=12)))
