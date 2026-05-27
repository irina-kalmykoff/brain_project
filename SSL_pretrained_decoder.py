# Converted from SSL_pretrained_decoder.ipynb

# %% Cell 1 — imports, config, pipeline init
# ============================================================
import os, time, math, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

from dutch_30_pipeline import Dutch30Pipeline
from dutch_30_feature_extractor import Dutch30FeatureExtractor
from dataset_config import Dutch30Config
from config import DUTCH_30_PATH
from run_pipeline import load_mfa_alignments
from extract_features import extractHG
from e2e_brain_decoder import edit_distance, show_matched_sequences_with_times

# reuse the user's existing LDA helpers (smooth/viterbi/tune/NW)
from LDA_on_frames_clean import (
    smooth_cols, viterbi_decode, auto_tune_bonus,
    nw_metrics, print_nw_metrics, gather_sequences, needleman_wunsch,
    SELF_LOOP_BONUS, 
    TARGET_RATIO, MIN_PRED_FRAMES, SMOOTH_LOGP_W, 
    VAL_FRAC, TEST_OFFSET, EEG_SR, WIN_S, SHIFT_S, WIN_SAMP, SHIFT_SAMP,
    predict_speech_prob,         # cross-patient speech detector (gate)
    USE_SPEECH_GATE, SPEECH_THRESHOLD, SPEECH_FRAC_MIN,
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"DEVICE = {DEVICE}")

TARGET_PIDS = ['P21', 'P22', 'P23', 'P24', 'P25',
               'P26', 'P27', 'P28', 'P29', 'P30']

# SSL hyperparameters
HIDDEN_DIM      = 128
TCN_KERNEL      = 5
TCN_DILATIONS   = (1, 2, 4, 8)      # past receptive field ~150 ms
DROPOUT         = 0.1

SSL_EPOCHS      = 80
SSL_LR          = 3e-4
SSL_WD          = 1e-3
SSL_BATCH       = 4                  # sentences per batch
SSL_MASK_FRAC   = 0.15               # fraction of frames to mask
SSL_MASK_SPAN   = 10                 # span length in frames (= 50 ms)

# Aux finetune hyperparameters
AUX_EPOCHS      = 30
AUX_LR_ENC      = 3e-5               # small LR on encoder (slow finetune)
AUX_LR_HEAD     = 3e-4
AUX_WD          = 1e-3
WORD_ONSET_TOL  = 2                  # frames ±2 around onset = positive
SYL_ONSET_TOL   = 2

# Phoneme stage geometry (matches LDA_on_frames_clean)
MN_FRAMES       = 0
MX_FRAMES       = 300

MODEL_DIR = 'bio_models'
os.makedirs(MODEL_DIR, exist_ok=True)

# ── feature stacking ────────────────────────────────────────
MO, SS          = 11, 1
LDA_MARGIN      = MO * SS

# ── misc ────────────────────────────────────────────────────
RARE_TOP_N = 5

# ── pipeline init (skip if already done in another notebook) ──
try:
    pipeline
    print("Reusing existing `pipeline` object.")
except NameError:
    config = Dutch30Config()
    extractor = Dutch30FeatureExtractor(config=config)
    pipeline = Dutch30Pipeline(extractor, config=config, use_wav2vec=False)
    pipeline.step1_load_dutch30_data(patient_range=(21, 30))
    pipeline.step2_split_by_instances(train_fraction=0.8)
    pipeline.step3_load_channel_exclusions('channel_exclusions.json')
    pipeline.apply_channel_exclusions()

# %% Cell 2 — build per-patient sentence dataset with extractHG
# ============================================================
def _channel_mask(pid):
    """Return the keep_indices array (or None) for this patient."""
    cm = getattr(pipeline, 'channel_masks', {}).get(pid, None)
    if cm is None: return None
    return np.asarray(cm['keep_indices'], dtype=np.int64)


def build_sentence_dataset(pid):
    """Returns dict {'train':[...], 'test':[...]} of per-sentence dicts:
        {'X': (T, n_ch) float32, 'mfa': [...], 'sent_idx': int}
    Uses extractHG (the MFA-CRF recipe) — same features as LDA_on_frames_clean.
    """
    raw_eeg = np.load(os.path.join(DUTCH_30_PATH, 'raw', f'{pid}_sEEG.npy'))
    if raw_eeg.ndim == 2 and raw_eeg.shape[0] < raw_eeg.shape[1]:
        raw_eeg = raw_eeg.T

    keep = _channel_mask(pid)
    if keep is not None:
        raw_eeg = raw_eeg[:, keep]

    wd  = pipeline.split_result['word_segments_dict'][pid]
    mfa = load_mfa_alignments(pid)

    all_real = [i for i, s in enumerate(wd['sentence_list'])
                if isinstance(s, dict) and s.get('text')]
    test_sent_ids = set(all_real[TEST_OFFSET::6])

    out = {'train': [], 'test': []}
    n_used = n_skip = 0
    for sent_idx in all_real:
        if sent_idx not in mfa or not mfa[sent_idx]:
            n_skip += 1; continue
        s = wd['sentence_list'][sent_idx]
        s0, s1 = s['stim_start_idx'], s['stim_end_idx']
        if s1 > raw_eeg.shape[0]:
            n_skip += 1; continue
        X = extractHG(raw_eeg[s0:s1], sr=EEG_SR,
                      windowLength=WIN_S, frameshift=SHIFT_S,
                      smoothing_hz=10.0).astype(np.float32)
        if X.shape[0] < 30:
            n_skip += 1; continue
        split = 'test' if sent_idx in test_sent_ids else 'train'
        out[split].append({
            'X':        torch.from_numpy(X),
            'mfa':      mfa[sent_idx],
            'sent_idx': sent_idx,
            'sent_t0':  s0 / EEG_SR,
        })
        n_used += 1

    n_in = out['train'][0]['X'].shape[1] if out['train'] else 0
    print(f"  [{pid}] used={n_used} skipped={n_skip}  n_in={n_in}  "
          f"train={len(out['train'])}  test={len(out['test'])}")
    return out


datasets = {}
for pid in TARGET_PIDS:
    print(f"\nBuilding {pid}...")
    datasets[pid] = build_sentence_dataset(pid)

# %% Cell 3 — causal TCN encoder + masked-frame head
# ============================================================
class CausalConv1d(nn.Conv1d):
    """Conv1d that pads only on the left -> output[t] sees only input[<=t]."""
    def __init__(self, in_ch, out_ch, kernel_size, dilation=1):
        super().__init__(in_ch, out_ch, kernel_size,
                         dilation=dilation, padding=0)
        self.left_pad = (kernel_size - 1) * dilation
    def forward(self, x):
        return super().forward(F.pad(x, (self.left_pad, 0)))


class TCNBlock(nn.Module):
    def __init__(self, dim, kernel_size, dilation, dropout=0.1):
        super().__init__()
        self.conv1 = CausalConv1d(dim, dim, kernel_size, dilation)
        self.norm1 = nn.GroupNorm(8, dim)
        self.conv2 = CausalConv1d(dim, dim, kernel_size, dilation)
        self.norm2 = nn.GroupNorm(8, dim)
        self.drop  = nn.Dropout(dropout)
    def forward(self, x):                       # (B, dim, T)
        h = F.gelu(self.norm1(self.conv1(x)))
        h = self.drop(F.gelu(self.norm2(self.conv2(h))))
        return h + x                            # residual


class CausalTCNEncoder(nn.Module):
    """Per-patient causal TCN.  Input (B, T, n_in) -> output (B, T, hidden).
    A learnable mask token can be substituted into the latent at masked
    positions for SSL pretraining (`forward(x, mask=...)`)."""
    def __init__(self, n_in, hidden=HIDDEN_DIM, kernel=TCN_KERNEL,
                 dilations=TCN_DILATIONS, dropout=DROPOUT):
        super().__init__()
        self.proj_in    = nn.Conv1d(n_in, hidden, kernel_size=1)
        self.blocks     = nn.ModuleList(
            [TCNBlock(hidden, kernel, d, dropout) for d in dilations])
        self.mask_token = nn.Parameter(torch.zeros(hidden))
        nn.init.normal_(self.mask_token, std=0.02)
    def forward(self, x, mask=None):
        # x: (B, T, n_in) ; mask: (B, T) bool, True = masked
        h = self.proj_in(x.transpose(1, 2))     # (B, hidden, T)
        if mask is not None:
            h = torch.where(mask.unsqueeze(1),
                            self.mask_token.view(1, -1, 1), h)
        for blk in self.blocks: h = blk(h)
        return h.transpose(1, 2)                # (B, T, hidden)


class SSLHead(nn.Module):
    """Predicts the original HG values at masked positions."""
    def __init__(self, hidden, n_out):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, n_out))
    def forward(self, h): return self.fc(h)


def make_span_mask(T, frac=SSL_MASK_FRAC, span=SSL_MASK_SPAN, rng=None):
    rng = rng or np.random
    n_masked = int(frac * T)
    n_starts = max(1, n_masked // span)
    mask = np.zeros(T, dtype=bool)
    for _ in range(n_starts):
        s = rng.randint(0, max(1, T - span))
        mask[s:s + span] = True
    return mask

# %% Cell 4 — SSL pretraining loop (one encoder per patient)
# ============================================================
def fit_mu_sd(sents):
    X = torch.cat([s['X'] for s in sents], dim=0).numpy()
    mu = X.mean(0); sd = X.std(0)
    sd = np.where(sd < 1e-6, 1.0, sd)
    return mu.astype(np.float32), sd.astype(np.float32)


def standardize_inplace(sents, mu, sd):
    mu_t = torch.from_numpy(mu); sd_t = torch.from_numpy(sd)
    for s in sents:
        s['X'] = (s['X'] - mu_t) / sd_t


def ssl_pretrain_one(pid, ds, epochs=SSL_EPOCHS, seed=0):
    rng = np.random.RandomState(seed)
    mu, sd = fit_mu_sd(ds['train'])
    standardize_inplace(ds['train'], mu, sd)
    standardize_inplace(ds['test'],  mu, sd)
    n_in = ds['train'][0]['X'].shape[1]

    enc  = CausalTCNEncoder(n_in).to(DEVICE)
    head = SSLHead(HIDDEN_DIM, n_in).to(DEVICE)
    opt  = torch.optim.AdamW(list(enc.parameters()) + list(head.parameters()),
                             lr=SSL_LR, weight_decay=SSL_WD)
    sch  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    train_sents = ds['train']
    n = len(train_sents)
    print(f"  [{pid}] SSL pretrain: n_in={n_in} n_train={n}")
    t0 = time.time()
    for ep in range(epochs):
        enc.train(); head.train()
        rng.shuffle(train_sents)
        total = 0.0; nb = 0
        for i in range(0, n, SSL_BATCH):
            batch = train_sents[i:i + SSL_BATCH]
            # pad to max length in batch
            Tmax = max(s['X'].shape[0] for s in batch)
            X = torch.zeros(len(batch), Tmax, n_in)
            valid = torch.zeros(len(batch), Tmax, dtype=torch.bool)
            mask  = torch.zeros(len(batch), Tmax, dtype=torch.bool)
            for b, s in enumerate(batch):
                T = s['X'].shape[0]
                X[b, :T] = s['X']
                valid[b, :T] = True
                mb = make_span_mask(T, rng=rng)
                mask[b, :T] = torch.from_numpy(mb)
            X = X.to(DEVICE); mask = mask.to(DEVICE); valid = valid.to(DEVICE)
            h    = enc(X, mask=mask)
            pred = head(h)
            tgt  = X                                # reconstruct original
            sel  = mask & valid
            if sel.sum() == 0: continue
            loss = F.mse_loss(pred[sel], tgt[sel])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(enc.parameters(), 1.0)
            opt.step()
            total += loss.item(); nb += 1
        sch.step()
        if ep == 0 or (ep + 1) % 10 == 0:
            print(f"    ep {ep+1:3d}/{epochs}  mse={total/max(nb,1):.4f}"
                  f"  lr={opt.param_groups[0]['lr']:.2e}"
                  f"  ({time.time()-t0:.1f}s)")
    return enc, mu, sd


encoders, mus, sds = {}, {}, {}
for pid in TARGET_PIDS:
    enc, mu, sd = ssl_pretrain_one(pid, datasets[pid])
    encoders[pid] = enc; mus[pid] = mu; sds[pid] = sd
    torch.save({'enc': enc.state_dict(), 'mu': mu, 'sd': sd,
                'n_in': enc.proj_in.in_channels},
               os.path.join(MODEL_DIR, f'{pid}_ssl_encoder.pt'))
    print(f"  [{pid}] saved encoder")


# # %% Cell 4b — mask-fraction sweep
# # ============================================================
# # Vary SSL_MASK_FRAC and (optionally) SSL_MASK_SPAN.
# # Trains a fresh encoder per (pid, frac, span), reuses Cell 7/8 inference.
# # Restores the original encoder afterwards so downstream cells still work.

# from copy import deepcopy

# SWEEP_PIDS  = ['P22', 'P30']
# SWEEP_FRACS = [0.10, 0.15, 0.25, 0.40, 0.60]
# SWEEP_SPANS = [10]                  # add 5, 20 if you also want to vary span
# SWEEP_EPOCHS = 50                   # shorter than full SSL to save time

# # stash the current encoders so the sweep doesn't clobber them
# _saved_enc = {pid: encoders[pid] for pid in SWEEP_PIDS}
# _saved_mu  = {pid: mus[pid]      for pid in SWEEP_PIDS}
# _saved_sd  = {pid: sds[pid]      for pid in SWEEP_PIDS}
# _saved_emb = {pid: embeddings[pid] for pid in SWEEP_PIDS}

# # patch the global mask constants used by ssl_pretrain_one via make_span_mask
# _orig_frac, _orig_span = SSL_MASK_FRAC, SSL_MASK_SPAN

# def ssl_pretrain_with_mask(pid, ds_clean, frac, span, epochs):
#     """Train a fresh encoder with a specific mask config.  Returns enc, mu, sd
#     Re-standardises a copy of the dataset to avoid contaminating ds in place."""
#     global SSL_MASK_FRAC, SSL_MASK_SPAN
#     SSL_MASK_FRAC, SSL_MASK_SPAN = frac, span
#     ds = {'train': deepcopy(ds_clean['train']),
#           'test':  deepcopy(ds_clean['test'])}
#     # need un-standardised X for ssl_pretrain_one's fit_mu_sd to work cleanly
#     # (it standardises in place); easiest: reload raw HG for these two pids
#     fresh = build_sentence_dataset(pid)
#     return ssl_pretrain_one(pid, fresh, epochs=epochs)

# sweep_results = {}
# for pid in SWEEP_PIDS:
#     for span in SWEEP_SPANS:
#         for frac in SWEEP_FRACS:
#             tag = f"{pid}_f{int(frac*100):02d}_s{span:02d}"
#             print(f"\n=== {tag}: training (frac={frac}, span={span}) ===")
#             enc, mu, sd = ssl_pretrain_with_mask(pid, datasets[pid],
#                                                   frac, span, SWEEP_EPOCHS)
#             encoders[pid] = enc; mus[pid] = mu; sds[pid] = sd
#             # re-standardise + re-extract embeddings for this pid
#             ds = build_sentence_dataset(pid)
#             standardize_inplace(ds['train'], mu, sd)
#             standardize_inplace(ds['test'],  mu, sd)
#             datasets[pid] = ds
#             emb_tr = extract_embeddings(pid, ds['train'])
#             emb_te = extract_embeddings(pid, ds['test'])
#             embeddings[pid] = {**emb_tr, **emb_te}
#             # downstream LDA + scalar-bonus Viterbi
#             out, err = run_for_patient_ssl(pid)
#             if err:
#                 print(f"   SKIP — {err}"); continue
#             m = nw_metrics(out)
#             sweep_results[tag] = {
#                 'pid': pid, 'frac': frac, 'span': span,
#                 'match': m['match_rate'], 'z': m['z_match'],
#                 'n2': m['n2'], 'n3': m['n3'], 'n4': m['n4'],
#                 'n_pred': out['n_pred'], 'n_test': out['n_test'],
#             }
#             print(f"   match={100*m['match_rate']:5.1f}%  z={m['z_match']:+5.2f}  "
#                   f"n2={m['n2']} n3={m['n3']} n4={m['n4']}")

# # restore
# SSL_MASK_FRAC, SSL_MASK_SPAN = _orig_frac, _orig_span
# for pid in SWEEP_PIDS:
#     encoders[pid] = _saved_enc[pid]
#     mus[pid] = _saved_mu[pid]; sds[pid] = _saved_sd[pid]
#     embeddings[pid] = _saved_emb[pid]

# # summary
# print(f"\n{'tag':<18} {'frac':>5} {'span':>5} {'match':>7} {'z':>6} {'n3':>3} {'n4':>3}")
# for tag, r in sweep_results.items():
#     print(f"{tag:<18} {r['frac']:>5.2f} {r['span']:>5d} "
#           f"{100*r['match']:6.1f}% {r['z']:+5.2f} {r['n3']:>3} {r['n4']:>3}")

# %% Cell 5 — auxiliary labels: silence/speech, word-onset, syllable-onset
# ============================================================
# Dutch MFA phoneme set — vowels (syllable nuclei)
DUTCH_VOWELS = {
    # short
    'ɑ', 'ɛ', 'ɪ', 'ɔ', 'ʏ', 'ə',
    # long / tense
    'a', 'aː', 'eː', 'iː', 'oː', 'uː', 'yː', 'øː', 'i', 'o', 'u', 'y',
    # diphthongs (MFA often writes these as 2-char tokens)
    'ɛi', 'œy', 'ʌu', 'ɔi', 'ai', 'au', 'ɛɪ', 'œʏ', 'ɑu',
}

FRAME_HZ = int(round(1.0 / SHIFT_S))   # 200 Hz

def time_to_frame(t_s):
    return int(round((t_s * EEG_SR - WIN_SAMP / 2) / SHIFT_SAMP))

def is_vowel(ph):
    return ph in DUTCH_VOWELS or any(ph.startswith(v) for v in DUTCH_VOWELS)

def speech_labels(mfa, T, pre_onset_ms=200):
    """1 if frame is inside any phoneme or up to pre_onset_ms before it."""
    pre = int(pre_onset_ms / 1000.0 * FRAME_HZ)
    y = np.zeros(T, dtype=np.int64)
    for ph in mfa:
        ks = max(0, time_to_frame(ph['start_s']) - pre)
        ke = min(T - 1, time_to_frame(ph['end_s']))
        if ke >= ks: y[ks:ke + 1] = 1
    return y

def word_onset_labels(mfa, T, tol=WORD_ONSET_TOL):
    """1 at start of each new word ±tol frames."""
    y = np.zeros(T, dtype=np.int64)
    prev_word = None
    for ph in mfa:
        w = ph.get('word', '')
        if w and w != prev_word:
            k = time_to_frame(ph['start_s'])
            ks = max(0, k - tol); ke = min(T - 1, k + tol)
            y[ks:ke + 1] = 1
            prev_word = w
    return y

def syllable_onset_labels(mfa, T, tol=SYL_ONSET_TOL):
    """A syllable starts at the first consonant of the onset cluster preceding
    each vowel (or at the vowel itself if no consonants since last vowel).
    Onsets are derived within each word — we don't cross word boundaries."""
    y = np.zeros(T, dtype=np.int64)
    if not mfa: return y
    # group phones by word
    groups, cur, cur_w = [], [], None
    for ph in mfa:
        w = ph.get('word', '')
        if w != cur_w and cur:
            groups.append(cur); cur = []
        cur.append(ph); cur_w = w
    if cur: groups.append(cur)

    for g in groups:
        last_consonant_start_idx = None
        last_was_vowel = True   # reset at word start: 1st cluster is onset
        for i, ph in enumerate(g):
            if is_vowel(ph['phone']):
                # syllable onset = start of preceding consonant cluster
                if last_consonant_start_idx is None or last_was_vowel:
                    # no preceding consonant since last vowel
                    onset_phone = ph
                else:
                    onset_phone = g[last_consonant_start_idx]
                k = time_to_frame(onset_phone['start_s'])
                ks = max(0, k - tol); ke = min(T - 1, k + tol)
                y[ks:ke + 1] = 1
                last_consonant_start_idx = None
                last_was_vowel = True
            else:
                if last_consonant_start_idx is None or last_was_vowel:
                    last_consonant_start_idx = i
                last_was_vowel = False
    return y


def build_aux_labels_for_sent(sent):
    T = sent['X'].shape[0]
    return {
        'speech': torch.from_numpy(speech_labels(sent['mfa'], T)),
        'word':   torch.from_numpy(word_onset_labels(sent['mfa'], T)),
        'syl':    torch.from_numpy(syllable_onset_labels(sent['mfa'], T)),
    }


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

# %% Cell 6 — multi-task aux finetune (speech + word-onset + syllable-onset)
# ============================================================
class AuxHeads(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.speech = nn.Linear(hidden, 2)
        self.word   = nn.Linear(hidden, 2)
        self.syl    = nn.Linear(hidden, 2)
    def forward(self, h):
        return self.speech(h), self.word(h), self.syl(h)


def pos_weight(labels_list, key):
    n_pos = sum(int(l[key].sum()) for l in labels_list)
    n_tot = sum(int(l[key].numel()) for l in labels_list)
    n_neg = n_tot - n_pos
    w_pos = n_neg / max(n_pos, 1)
    # weights for [neg, pos] in CrossEntropyLoss(weight=...) — balance to ~1:1
    return torch.tensor([1.0, w_pos], dtype=torch.float32)


def aux_finetune_one(pid, ds, enc, epochs=AUX_EPOCHS, seed=0):
    rng = np.random.RandomState(seed)
    labels_tr = [build_aux_labels_for_sent(s) for s in ds['train']]
    cw_sp = pos_weight(labels_tr, 'speech').to(DEVICE)
    cw_wd = pos_weight(labels_tr, 'word'  ).to(DEVICE)
    cw_sy = pos_weight(labels_tr, 'syl'   ).to(DEVICE)
    print(f"  [{pid}] aux class weights:"
          f"  speech={cw_sp[1]:.2f}  word={cw_wd[1]:.2f}  syl={cw_sy[1]:.2f}")

    heads = AuxHeads(HIDDEN_DIM).to(DEVICE)
    opt = torch.optim.AdamW([
        {'params': enc.parameters(),   'lr': AUX_LR_ENC},
        {'params': heads.parameters(), 'lr': AUX_LR_HEAD},
    ], weight_decay=AUX_WD)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    n = len(ds['train'])
    indices = list(range(n))
    t0 = time.time()
    for ep in range(epochs):
        enc.train(); heads.train()
        rng.shuffle(indices)
        totals = np.zeros(3); nb = 0
        for i in indices:
            X = ds['train'][i]['X'].unsqueeze(0).to(DEVICE)    # (1, T, n_in)
            lbl = labels_tr[i]
            h = enc(X)                                          # (1, T, H)
            o_sp, o_wd, o_sy = heads(h.squeeze(0))             # (T, 2) each
            l_sp = F.cross_entropy(o_sp, lbl['speech'].to(DEVICE), weight=cw_sp)
            l_wd = F.cross_entropy(o_wd, lbl['word'  ].to(DEVICE), weight=cw_wd)
            l_sy = F.cross_entropy(o_sy, lbl['syl'   ].to(DEVICE), weight=cw_sy)
            loss = l_sp + l_wd + l_sy
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(enc.parameters(), 1.0)
            opt.step()
            totals += [l_sp.item(), l_wd.item(), l_sy.item()]; nb += 1
        sch.step()
        if ep == 0 or (ep + 1) % 5 == 0:
            t = totals / max(nb, 1)
            print(f"    ep {ep+1:3d}/{epochs}  sp={t[0]:.3f} wd={t[1]:.3f} "
                  f"sy={t[2]:.3f}  ({time.time()-t0:.1f}s)")
    return enc, heads


aux_heads = {}
for pid in TARGET_PIDS:
    enc, heads = aux_finetune_one(pid, datasets[pid], encoders[pid])
    encoders[pid] = enc
    aux_heads[pid] = heads
    torch.save({'enc': enc.state_dict(),
                'heads': heads.state_dict(),
                'mu': mus[pid], 'sd': sds[pid]},
               os.path.join(MODEL_DIR, f'{pid}_ssl_encoder_aux.pt'))

# %% Cell 7 — frozen encoder -> 128-d per-sentence embeddings
# ============================================================
@torch.no_grad()
def extract_embeddings(pid, sents):
    """Return {sent_idx: (T, HIDDEN_DIM) float32 ndarray} for each sentence."""
    enc = encoders[pid]; enc.eval()
    out = {}
    for s in sents:
        X = s['X'].unsqueeze(0).to(DEVICE)
        h = enc(X).squeeze(0).cpu().numpy().astype(np.float32)
        out[s['sent_idx']] = h
    return out


# build train + test embeddings (test sentences are *also* in ds['test'])
embeddings = {}
for pid in TARGET_PIDS:
    ds = datasets[pid]
    emb_tr = extract_embeddings(pid, ds['train'])
    emb_te = extract_embeddings(pid, ds['test'])
    embeddings[pid] = {**emb_tr, **emb_te}
    print(f"  [{pid}] embeddings: train={len(emb_tr)} test={len(emb_te)}  "
          f"shape e.g. {next(iter(emb_tr.values())).shape}")


# %% Cell — re-evaluate with SSL-only encoders (no aux finetune)
# ============================================================
# Loads bio_models/{pid}_ssl_encoder.pt (NOT the _aux variant) and
# runs the full inference pipeline.  Self-contained: no dependency on
# Cell 7b or prior in-memory state.

# 1. rebuild datasets (only takes a few minutes total)
datasets = {}
for pid in TARGET_PIDS:
    print(f"Building {pid}...")
    datasets[pid] = build_sentence_dataset(pid)

# 2. load SSL-only encoders + saved standardisation stats
encoders, mus, sds = {}, {}, {}
for pid in TARGET_PIDS:
    ckpt_path = os.path.join(MODEL_DIR, f'{pid}_ssl_encoder.pt')
    if not os.path.exists(ckpt_path):
        print(f"  [{pid}] MISSING {ckpt_path} — skipping")
        continue
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    n_in = datasets[pid]['train'][0]['X'].shape[1]
    enc = CausalTCNEncoder(n_in).to(DEVICE)
    enc.load_state_dict(ckpt['enc'])
    encoders[pid] = enc
    mus[pid] = ckpt['mu']; sds[pid] = ckpt['sd']
    standardize_inplace(datasets[pid]['train'], mus[pid], sds[pid])
    standardize_inplace(datasets[pid]['test'],  mus[pid], sds[pid])
    print(f"  [{pid}] loaded SSL-only encoder")

# 3. extract embeddings
embeddings = {}
for pid in TARGET_PIDS:
    if pid not in encoders: continue
    ds = datasets[pid]
    emb_tr = extract_embeddings(pid, ds['train'])
    emb_te = extract_embeddings(pid, ds['test'])
    embeddings[pid] = {**emb_tr, **emb_te}

# 4. run LDA + scalar-bonus Viterbi
ssl_only_results = {}
for pid in TARGET_PIDS:
    if pid not in embeddings: continue
    out, err = run_for_patient_ssl(pid)
    if err:
        print(f"  {pid}: SKIP — {err}"); continue
    ssl_only_results[pid] = out
    print(f"  {pid}: PER={100*out['per']:5.1f}%  "
          f"n_pred={out['n_pred']}/{out['n_test']}  bonus={out['bonus']:.2f}")

# 5. NW summary
print(f"\n{'pid':<5} {'match':>7} {'z':>6} {'n2':>4} {'n3':>4} {'n4':>4}  pred/gold")
for pid in TARGET_PIDS:
    if pid not in ssl_only_results: continue
    out = ssl_only_results[pid]
    m = nw_metrics(out)
    print(f"{pid:<5} {100*m['match_rate']:6.1f}% {m['z_match']:+5.2f} "
          f"{m['n2']:>4} {m['n3']:>4} {m['n4']:>4}  "
          f"{out['n_pred']:>3}/{out['n_test']:>3}")

# %% Cell — P22 seed-1 sanity check
# ============================================================
import random
SEED = 1
PID  = 'P22'

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if DEVICE.type == 'cuda':
    torch.cuda.manual_seed_all(SEED)

# fresh dataset (un-standardised) so ssl_pretrain_one can fit mu/sd from scratch
ds = build_sentence_dataset(PID)

# train
enc, mu, sd = ssl_pretrain_one(PID, ds, epochs=SSL_EPOCHS, seed=SEED)

# save separately
torch.save({'enc': enc.state_dict(), 'mu': mu, 'sd': sd,
            'n_in': enc.proj_in.in_channels},
           os.path.join(MODEL_DIR, f'{PID}_ssl_encoder_seed{SEED}.pt'))

# swap in for inference
encoders[PID] = enc; mus[PID] = mu; sds[PID] = sd
datasets[PID] = ds   # already standardised in place by ssl_pretrain_one
emb_tr = extract_embeddings(PID, ds['train'])
emb_te = extract_embeddings(PID, ds['test'])
embeddings[PID] = {**emb_tr, **emb_te}

# run + report
out, err = run_for_patient_ssl(PID)
if err:
    print(f"  {PID}: SKIP — {err}")
else:
    m = nw_metrics(out)
    print(f"\n  {PID} seed={SEED}: match={100*m['match_rate']:5.1f}%  "
          f"z={m['z_match']:+5.2f}  n2={m['n2']} n3={m['n3']} n4={m['n4']}  "
          f"pred/gold={out['n_pred']}/{out['n_test']}  bonus={out['bonus']:.2f}")
    print(f"  for reference, seed=0: match=26.0%  z=+0.75  n2=18 n3=3 n4=0")

# %% Cell 8 — LDA + Viterbi on SSL embeddings (mirrors LDA_on_frames_clean)
#scalar self-loop bonus
# ============================================================
def ssl_frame_to_time_s(i):
    """SSL embeddings are at 200 Hz with no stacking margin."""
    return (i * SHIFT_SAMP + WIN_SAMP / 2) / EEG_SR


def run_for_patient_ssl(pid, use_speech_gate=USE_SPEECH_GATE,
                        speech_thresh=SPEECH_THRESHOLD,
                        speech_frac_min=SPEECH_FRAC_MIN):
    """Drop-in replacement for run_for_patient_sd, but uses SSL embeddings
    in place of stacked HG.  Returns the same dict that show_matched_*
    expects."""
    ds  = datasets[pid]
    mfa = {s['sent_idx']: s['mfa'] for s in ds['train'] + ds['test']}
    per_sent = embeddings[pid]

    all_real = sorted(per_sent.keys())
    test_sent_ids = set(s['sent_idx'] for s in ds['test'])
    train_sent_ids = [i for i in all_real if i not in test_sent_ids]
    # held-out val split for bonus tuning
    rng = np.random.RandomState(0)
    rng.shuffle(train_sent_ids)
    n_val = max(1, int(len(train_sent_ids) * VAL_FRAC))
    val_sent_ids = set(train_sent_ids[:n_val])
    fit_sent_ids = set(train_sent_ids[n_val:])

    def build_set(sent_id_set):
        X, y = [], []
        for sid in sent_id_set:
            if sid not in per_sent: continue
            emb = per_sent[sid]; T = emb.shape[0]
            for ph in mfa[sid]:
                k_s = max(0, time_to_frame(ph['start_s']))
                k_e = min(T - 1, time_to_frame(ph['end_s']))
                n_fr = k_e - k_s + 1
                if n_fr < max(MN_FRAMES, 1) or n_fr > MX_FRAMES: continue
                X.append(emb[k_s:k_e + 1].mean(axis=0))
                y.append(ph['phone'])
        return np.array(X), np.array(y)

    X_fit, y_fit = build_set(fit_sent_ids)
    if len(X_fit) < 50: return None, f"too few fit samples ({len(X_fit)})"

    sc_fit  = StandardScaler().fit(X_fit)
    clf_fit = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    clf_fit.fit(sc_fit.transform(X_fit), y_fit)
    fit_classes = set(y_fit)

    # auto-tune bonus on val
    val_logps, val_target = [], 0
    for sid in val_sent_ids:
        if sid not in per_sent: continue
        logp = clf_fit.predict_log_proba(sc_fit.transform(per_sent[sid]))
        logp = smooth_cols(logp, SMOOTH_LOGP_W)
        val_logps.append(logp)
        val_target += sum(1 for ph in mfa[sid] if ph['phone'] in fit_classes)
    if not val_logps: return None, "no val sentences"
    val_target = int(val_target * TARGET_RATIO)
    bonus = (auto_tune_bonus(val_logps, val_target, MIN_PRED_FRAMES)
             if SELF_LOOP_BONUS is None else float(SELF_LOOP_BONUS))

    # refit on all train
    X_tr, y_tr = build_set(set(all_real) - test_sent_ids)
    scaler = StandardScaler().fit(X_tr)
    clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    clf.fit(scaler.transform(X_tr), y_tr)
    train_classes = set(y_tr); class_labels = list(clf.classes_)

    # raw eeg cache for the speech gate
    raw_eeg = np.load(os.path.join(DUTCH_30_PATH, 'raw', f'{pid}_sEEG.npy'))
    if raw_eeg.ndim == 2 and raw_eeg.shape[0] < raw_eeg.shape[1]:
        raw_eeg = raw_eeg.T
    keep = _channel_mask(pid)
    if keep is not None: raw_eeg = raw_eeg[:, keep]
    wd = pipeline.split_result['word_segments_dict'][pid]

    predictions, pred_sentence_ids, pred_segments = [], [], []
    true_labels, true_sentence_ids, true_segments = [], [], []
    n_dropped_silence = 0
    for sid in test_sent_ids:
        if sid not in per_sent: continue
        emb = per_sent[sid]; T = emb.shape[0]
        logp = clf.predict_log_proba(scaler.transform(emb))
        logp = smooth_cols(logp, SMOOTH_LOGP_W)
        path = viterbi_decode(logp, bonus)

        # speech-gate mask at the same time grid as the embeddings
        if use_speech_gate:
            s = wd['sentence_list'][sid]
            s0, s1 = s['stim_start_idx'], s['stim_end_idx']
            try:
                p_speech = predict_speech_prob(raw_eeg[s0:s1], pid)
            except Exception:
                p_speech = None
            if p_speech is not None and len(p_speech) >= T:
                speech_mask = (p_speech[:T] >= speech_thresh)
            else:
                speech_mask = np.ones(T, dtype=bool)
        else:
            speech_mask = np.ones(T, dtype=bool)

        i = 0
        while i < T:
            ci = path[i]; j = i + 1
            while j < T and path[j] == ci: j += 1
            if (j - i) >= MIN_PRED_FRAMES:
                if speech_mask[i:j].mean() < speech_frac_min:
                    n_dropped_silence += 1
                else:
                    predictions.append(class_labels[ci])
                    pred_sentence_ids.append(sid)
                    pred_segments.append((ssl_frame_to_time_s(i),
                                          ssl_frame_to_time_s(j - 1)))
            i = j
        for ph in mfa[sid]:
            if ph['phone'] not in train_classes: continue
            true_labels.append(ph['phone'])
            true_sentence_ids.append(sid)
            true_segments.append((ph['start_s'], ph['end_s']))

    if not true_labels: return None, "no test gold labels"
    true_arr = np.array(true_labels); pred_arr = np.array(predictions)
    ed  = edit_distance(list(true_arr), list(pred_arr))
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
        'n_train':           len(X_tr),
        'bonus':             bonus,
        'n_dropped_silence': n_dropped_silence,
    }, None


# run on all patients and stash into pipeline.patient_results
if not hasattr(pipeline, 'patient_results'):
    pipeline.patient_results = {}
ssl_results = {}
for pid in TARGET_PIDS:
    out, err = run_for_patient_ssl(pid)
    if err is not None:
        print(f"  {pid}: SKIP — {err}"); continue
    pipeline.patient_results[pid] = out
    ssl_results[pid] = out
    print(f"  {pid}: PER={100*out['per']:5.1f}%  "
          f"n_pred={out['n_pred']}/{out['n_test']}  bonus={out['bonus']:.2f}  "
          f"dropped_silence={out['n_dropped_silence']}")


# # %% Cell 8.5b — frame-level bigram replacement (paste after Cell 8.5)
# # ============================================================
# def fit_frame_bigram_logprob(mfa_by_sid, sent_ids, classes, alpha=1.0):
#     """log P(class_{t+1} | class_t) over the per-frame label stream.
#     Self-transitions dominate (~15 frames per phoneme) — what frame-level
#     Viterbi actually needs, unlike the token bigram."""
#     K = len(classes); idx = {c: i for i, c in enumerate(classes)}
#     counts = np.full((K, K), alpha, dtype=np.float64)
#     for sid in sent_ids:
#         if sid not in mfa_by_sid: continue
#         frames = []
#         for p in mfa_by_sid[sid]:
#             if p['phone'] not in idx: continue
#             k_s = max(0, time_to_frame(p['start_s']))
#             k_e = max(k_s, time_to_frame(p['end_s']))
#             frames.extend([idx[p['phone']]] * (k_e - k_s + 1))
#         for a, b in zip(frames[:-1], frames[1:]):
#             counts[a, b] += 1
#     return np.log(counts / counts.sum(axis=1, keepdims=True))


# def run_for_patient_ssl_bigram(pid, use_speech_gate=USE_SPEECH_GATE,
#                                 speech_thresh=SPEECH_THRESHOLD,
#                                 speech_frac_min=SPEECH_FRAC_MIN,
#                                 lm_grid=(0.0, 0.05, 0.1, 0.2, 0.4,
#                                          0.7, 1.0, 1.5, 2.5)):
#     ds  = datasets[pid]
#     mfa_by_sid = {s['sent_idx']: s['mfa'] for s in ds['train'] + ds['test']}
#     per_sent   = embeddings[pid]

#     all_real      = sorted(per_sent.keys())
#     test_sent_ids = set(s['sent_idx'] for s in ds['test'])
#     train_sent_ids = [i for i in all_real if i not in test_sent_ids]
#     rng = np.random.RandomState(0); rng.shuffle(train_sent_ids)
#     n_val = max(1, int(len(train_sent_ids) * VAL_FRAC))
#     val_sent_ids = set(train_sent_ids[:n_val])
#     fit_sent_ids = set(train_sent_ids[n_val:])

#     def build_set(sent_id_set):
#         X, y = [], []
#         for sid in sent_id_set:
#             if sid not in per_sent: continue
#             emb = per_sent[sid]; T = emb.shape[0]
#             for ph in mfa_by_sid[sid]:
#                 k_s = max(0, time_to_frame(ph['start_s']))
#                 k_e = min(T - 1, time_to_frame(ph['end_s']))
#                 n_fr = k_e - k_s + 1
#                 if n_fr < max(MN_FRAMES, 1) or n_fr > MX_FRAMES: continue
#                 X.append(emb[k_s:k_e + 1].mean(axis=0))
#                 y.append(ph['phone'])
#         return np.array(X), np.array(y)

#     X_fit, y_fit = build_set(fit_sent_ids)
#     if len(X_fit) < 50: return None, f"too few fit samples ({len(X_fit)})"

#     sc_fit  = StandardScaler().fit(X_fit)
#     clf_fit = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
#     clf_fit.fit(sc_fit.transform(X_fit), y_fit)
#     fit_classes = list(clf_fit.classes_)

#     log_trans_fit = fit_frame_bigram_logprob(mfa_by_sid, fit_sent_ids,
#                                              fit_classes, alpha=1.0)

#     val_logps, val_target = [], 0
#     for sid in val_sent_ids:
#         if sid not in per_sent: continue
#         logp = clf_fit.predict_log_proba(sc_fit.transform(per_sent[sid]))
#         logp = smooth_cols(logp, SMOOTH_LOGP_W)
#         val_logps.append(logp)
#         val_target += sum(1 for ph in mfa_by_sid[sid]
#                           if ph['phone'] in set(fit_classes))
#     if not val_logps: return None, "no val sentences"
#     val_target = int(val_target * TARGET_RATIO)
#     lam = auto_tune_lm_weight(val_logps, log_trans_fit,
#                               val_target, MIN_PRED_FRAMES, grid=lm_grid)

#     X_tr, y_tr = build_set(set(all_real) - test_sent_ids)
#     scaler = StandardScaler().fit(X_tr)
#     clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
#     clf.fit(scaler.transform(X_tr), y_tr)
#     train_classes = list(clf.classes_); train_classes_set = set(train_classes)
#     log_trans = fit_frame_bigram_logprob(mfa_by_sid,
#                                           set(all_real) - test_sent_ids,
#                                           train_classes, alpha=1.0)
#     scaled_trans = lam * log_trans

#     raw_eeg = np.load(os.path.join(DUTCH_30_PATH, 'raw', f'{pid}_sEEG.npy'))
#     if raw_eeg.ndim == 2 and raw_eeg.shape[0] < raw_eeg.shape[1]:
#         raw_eeg = raw_eeg.T
#     keep = _channel_mask(pid)
#     if keep is not None: raw_eeg = raw_eeg[:, keep]
#     wd = pipeline.split_result['word_segments_dict'][pid]

#     predictions, pred_sentence_ids, pred_segments = [], [], []
#     true_labels, true_sentence_ids, true_segments = [], [], []
#     n_dropped_silence = 0
#     for sid in test_sent_ids:
#         if sid not in per_sent: continue
#         emb = per_sent[sid]; T = emb.shape[0]
#         logp = clf.predict_log_proba(scaler.transform(emb))
#         logp = smooth_cols(logp, SMOOTH_LOGP_W)
#         path = viterbi_decode_bigram(logp, scaled_trans)

#         if use_speech_gate:
#             s = wd['sentence_list'][sid]
#             s0, s1 = s['stim_start_idx'], s['stim_end_idx']
#             try:
#                 p_speech = predict_speech_prob(raw_eeg[s0:s1], pid)
#             except Exception:
#                 p_speech = None
#             if p_speech is not None and len(p_speech) >= T:
#                 speech_mask = (p_speech[:T] >= speech_thresh)
#             else:
#                 speech_mask = np.ones(T, dtype=bool)
#         else:
#             speech_mask = np.ones(T, dtype=bool)

#         i = 0
#         while i < T:
#             ci = path[i]; j = i + 1
#             while j < T and path[j] == ci: j += 1
#             if (j - i) >= MIN_PRED_FRAMES:
#                 if speech_mask[i:j].mean() < speech_frac_min:
#                     n_dropped_silence += 1
#                 else:
#                     predictions.append(train_classes[ci])
#                     pred_sentence_ids.append(sid)
#                     pred_segments.append((ssl_frame_to_time_s(i),
#                                           ssl_frame_to_time_s(j - 1)))
#             i = j
#         for ph in mfa_by_sid[sid]:
#             if ph['phone'] not in train_classes_set: continue
#             true_labels.append(ph['phone'])
#             true_sentence_ids.append(sid)
#             true_segments.append((ph['start_s'], ph['end_s']))

#     if not true_labels: return None, "no test gold labels"
#     true_arr = np.array(true_labels); pred_arr = np.array(predictions)
#     ed  = edit_distance(list(true_arr), list(pred_arr))
#     per = ed / max(len(true_arr), 1)
#     return {
#         'true_labels':       true_arr,
#         'predictions':       pred_arr,
#         'true_sentence_ids': np.array(true_sentence_ids),
#         'pred_sentence_ids': np.array(pred_sentence_ids),
#         'true_segments':     true_segments,
#         'pred_segments':     pred_segments,
#         'accuracy':          float('nan'),
#         'edit_distance':     ed,
#         'per':               per,
#         'n_test':            len(true_arr),
#         'n_pred':            len(pred_arr),
#         'n_train':           len(X_tr),
#         'lm_weight':         lam,
#         'n_dropped_silence': n_dropped_silence,
#     }, None


# ssl_bigram_results = {}
# for pid in TARGET_PIDS:
#     out, err = run_for_patient_ssl_bigram(pid)
#     if err:
#         print(f"  {pid}: SKIP — {err}"); continue
#     ssl_bigram_results[pid] = out
#     print(f"  {pid}: PER={100*out['per']:5.1f}%  "
#           f"n_pred={out['n_pred']}/{out['n_test']}  λ={out['lm_weight']:.2f}")

# print(f"\n{'pid':<5} {'match':>7} {'z':>6} {'n2':>4} {'n3':>4} {'n4':>4}  pred/gold  λ")
# for pid in TARGET_PIDS:
#     if pid not in ssl_bigram_results: continue
#     out = ssl_bigram_results[pid]
#     m = nw_metrics(out)
#     print(f"{pid:<5} {100*m['match_rate']:6.1f}% {m['z_match']:+5.2f} "
#           f"{m['n2']:>4} {m['n3']:>4} {m['n4']:>4}  "
#           f"{out['n_pred']:>3}/{out['n_test']:>3}  {out['lm_weight']:.2f}")

import scipy
from scipy.signal import sosfiltfilt, iirfilter, butter
DEFAULT_FEATURE_SPEC = {'hg_amp': True, 'hg_lp_hz': 10.0}

baseline_results = {}
for pid in TARGET_PIDS:
    out, err = run_for_patient_sd(pid,
                                  feature_spec=DEFAULT_FEATURE_SPEC,
                                  use_speech_gate=False)
    if err: print(f"  {pid}: SKIP — {err}"); continue
    baseline_results[pid] = out
    m = nw_metrics(out)
    print(f"{pid}: match={100*m['match_rate']:5.1f}%  z={m['z_match']:+5.2f}  "
          f"n2={m['n2']:>3} n3={m['n3']:>3} n4={m['n4']:>3}  "
          f"pred/gold={out['n_pred']}/{out['n_test']}")

# helper functions for Needleman Wunsch viz
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


# Needlman Wunsch viz
# ============================================================
import re
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

import numpy as np
from IPython.display import display, HTML

def ctc_results_to_out(predictions_list):
    """Convert a list of per-sentence dicts
       [{'sent_idx': int, 'pred': [...], 'gold': [...], ...}, ...]
    into the flat `out` format used by compare_predictions_html, nw_metrics, etc.
    """
    true_labels, true_sids = [], []
    preds, pred_sids = [], []
    for p in predictions_list:
        sid = p['sent_idx']
        for ph in p['gold']:
            true_labels.append(ph); true_sids.append(sid)
        for ph in p['pred']:
            preds.append(ph); pred_sids.append(sid)
    return {
        'true_labels':       np.array(true_labels),
        'predictions':       np.array(preds),
        'true_sentence_ids': np.array(true_sids),
        'pred_sentence_ids': np.array(pred_sids),
    }


def show_predictions_html(out, label='model', max_sentences=10):
    """Single-model wrapper around compare_predictions_html."""
    return compare_predictions_html(out, out_b=None,
                                    label_a=label,
                                    max_sentences=max_sentences)

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

# %% Cell 9 — evaluation + visualization (SSL-only encoder run)
# ============================================================
from IPython.display import display, HTML

# point the visualization at the SSL-only results (the latest cohort run)
pipeline.patient_results = dict(ssl_only_results)

for pid in TARGET_PIDS:
    if pid not in pipeline.patient_results: continue
    print(f"\n{'='*60}\n{pid} — SSL-only (15% mask) + LDA + scalar Viterbi\n{'='*60}")

    show_matched_sequences_with_times(pipeline, pid,
                                       max_per_line=45,
                                       collapse_repeats=True,
                                       time_align_tol_s=0.10)

    m = nw_metrics(pipeline.patient_results[pid])
    print_nw_metrics(m, label=pid)

PID = 'P23'

out = ssl_only_results[PID]      # latest SSL-only encoder + LDA + scalar Viterbi
m   = nw_metrics(out)

display(HTML(
    f"<h3>{PID} — SSL-only encoder + LDA "
    f"(match={100*m['match_rate']:.1f}%  z={m['z_match']:+.2f}  "
    f"PER={100*out['per']:.1f}%)</h3>"
))
display(HTML(show_predictions_html(out, label='SSL-only', max_sentences=20)))

# %% Cell — train per-patient word-onset detector
# ============================================================
class WordOnsetHead(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden, 2))
    def forward(self, h): return self.fc(h)

ONSET_TOL_FRAMES = 2     # ± frames around true onset = positive
ONSET_EPOCHS     = 30
ONSET_LR         = 3e-4

def build_onset_labels(sents):
    return [torch.from_numpy(
                word_onset_labels(s['mfa'], s['X'].shape[0],
                                  tol=ONSET_TOL_FRAMES))
            for s in sents]

@torch.no_grad()
def encoder_features(enc, sents):
    enc.eval()
    return [enc(s['X'].unsqueeze(0).to(DEVICE)).squeeze(0).cpu()
            for s in sents]

onset_heads = {}
for pid in TARGET_PIDS:
    if pid not in encoders: continue
    print(f"\n[{pid}] training word-onset head")
    ds = datasets[pid]
    H_tr  = encoder_features(encoders[pid], ds['train'])
    Y_tr  = build_onset_labels(ds['train'])
    n_pos = sum(int(y.sum())   for y in Y_tr)
    n_tot = sum(int(y.numel()) for y in Y_tr)
    cw    = torch.tensor([1.0, (n_tot - n_pos) / max(n_pos, 1)],
                          dtype=torch.float32, device=DEVICE)
    print(f"   pos frac={n_pos/n_tot:.3%}  pos weight={cw[1].item():.2f}")

    head = WordOnsetHead(HIDDEN_DIM).to(DEVICE)
    opt  = torch.optim.AdamW(head.parameters(), lr=ONSET_LR, weight_decay=1e-3)
    sch  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=ONSET_EPOCHS)
    idx  = list(range(len(H_tr)))
    rng  = np.random.RandomState(0)
    for ep in range(ONSET_EPOCHS):
        head.train(); rng.shuffle(idx); tot = 0.0
        for i in idx:
            h = H_tr[i].to(DEVICE); y = Y_tr[i].to(DEVICE)
            o = head(h)
            loss = F.cross_entropy(o, y, weight=cw)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        sch.step()
        if ep == 0 or (ep + 1) % 10 == 0:
            print(f"   ep {ep+1:2d}/{ONSET_EPOCHS}  loss={tot/len(idx):.3f}")
    onset_heads[pid] = head
    torch.save(head.state_dict(),
               os.path.join(MODEL_DIR, f'{pid}_word_onset_head.pt'))

# %% Cell — predict word-onset probability per frame
# ============================================================
@torch.no_grad()
def predict_onset_probs(pid, sents):
    enc = encoders[pid]; head = onset_heads[pid]
    enc.eval(); head.eval()
    out = {}
    for s in sents:
        h = enc(s['X'].unsqueeze(0).to(DEVICE))     # (1, T, H)
        logits = head(h.squeeze(0))                  # (T, 2)
        p = F.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        out[s['sent_idx']] = p.astype(np.float32)
    return out

onset_probs = {}
for pid in TARGET_PIDS:
    if pid not in onset_heads: continue
    ds = datasets[pid]
    p_tr = predict_onset_probs(pid, ds['train'])
    p_te = predict_onset_probs(pid, ds['test'])
    onset_probs[pid] = {**p_tr, **p_te}
    # quick sanity: mean predicted onset rate on train sentences
    rates = [p.mean() for p in p_tr.values()]
    print(f"  [{pid}] mean P(onset) on train = {np.mean(rates):.3%}")

# %% Cell — Viterbi with word-onset-modulated self-loop bonus
# ============================================================
def viterbi_decode_onset(logp, self_bonus, onset_prob, onset_weight=2.0):
    """At frame t, the effective self-loop bonus is:
         self_bonus * (1 - onset_weight * onset_prob[t])
       At a confident boundary (onset_prob≈1), bonus shrinks or flips
       negative, encouraging a state change.  At low-prob frames,
       behaviour matches the standard scalar-bonus Viterbi."""
    T, K = logp.shape
    dp = logp[0].copy()
    bp = np.zeros((T, K), dtype=np.int32)
    eff_bonus = self_bonus * (1.0 - onset_weight * onset_prob)  # (T,)
    for t in range(1, T):
        am = int(dp.argmax())
        # best "switch into j": max over i!=j of dp[i]
        if K > 1:
            max1 = dp[am]
            # second-best
            tmp = dp.copy(); tmp[am] = -np.inf
            am2 = int(tmp.argmax()); max2 = dp[am2]
            best_other = np.full(K, max1, dtype=dp.dtype)
            best_other[am] = max2
            from_idx = np.full(K, am, dtype=np.int32)
            from_idx[am] = am2
        else:
            best_other = np.full(K, -np.inf, dtype=dp.dtype)
            from_idx   = np.zeros(K, dtype=np.int32)

        from_self = dp + eff_bonus[t]                # (K,)
        take_self = from_self >= best_other          # (K,)
        bp[t]     = np.where(take_self, np.arange(K), from_idx)
        dp        = np.where(take_self, from_self, best_other) + logp[t]

    path = np.empty(T, dtype=np.int32); path[-1] = int(dp.argmax())
    for t in range(T - 1, 0, -1):
        path[t - 1] = bp[t, path[t]]
    return path

# %% Cell — run cohort with onset-constrained Viterbi
# ============================================================
ONSET_WEIGHT = 2.0      # try also 1.0, 3.0, 5.0 — tune on val later

def run_for_patient_ssl_onset(pid, onset_weight=ONSET_WEIGHT,
                                use_speech_gate=False):
    ds  = datasets[pid]
    mfa_by_sid = {s['sent_idx']: s['mfa'] for s in ds['train'] + ds['test']}
    per_sent   = embeddings[pid]
    onset_sent = onset_probs[pid]

    all_real      = sorted(per_sent.keys())
    test_sent_ids = set(s['sent_idx'] for s in ds['test'])
    train_sent_ids = [i for i in all_real if i not in test_sent_ids]
    rng = np.random.RandomState(0); rng.shuffle(train_sent_ids)
    n_val = max(1, int(len(train_sent_ids) * VAL_FRAC))
    val_sent_ids = set(train_sent_ids[:n_val])
    fit_sent_ids = set(train_sent_ids[n_val:])

    def build_set(sent_id_set):
        X, y = [], []
        for sid in sent_id_set:
            if sid not in per_sent: continue
            emb = per_sent[sid]; T = emb.shape[0]
            for ph in mfa_by_sid[sid]:
                k_s = max(0, time_to_frame(ph['start_s']))
                k_e = min(T - 1, time_to_frame(ph['end_s']))
                n_fr = k_e - k_s + 1
                if n_fr < max(MN_FRAMES, 1) or n_fr > MX_FRAMES: continue
                X.append(emb[k_s:k_e + 1].mean(axis=0))
                y.append(ph['phone'])
        return np.array(X), np.array(y)

    X_fit, y_fit = build_set(fit_sent_ids)
    if len(X_fit) < 50: return None, f"too few fit samples ({len(X_fit)})"
    sc_fit  = StandardScaler().fit(X_fit)
    clf_fit = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    clf_fit.fit(sc_fit.transform(X_fit), y_fit)
    fit_classes = set(y_fit)

    # tune bonus on val with onset-constrained Viterbi
    val_logps, val_onsets, val_target = [], [], 0
    for sid in val_sent_ids:
        if sid not in per_sent: continue
        logp = clf_fit.predict_log_proba(sc_fit.transform(per_sent[sid]))
        logp = smooth_cols(logp, SMOOTH_LOGP_W)
        val_logps.append(logp)
        val_onsets.append(onset_sent[sid][:logp.shape[0]])
        val_target += sum(1 for ph in mfa_by_sid[sid]
                          if ph['phone'] in fit_classes)
    val_target = int(val_target * TARGET_RATIO)

    def runs_at(bonus):
        n = 0
        for lp, on in zip(val_logps, val_onsets):
            path = viterbi_decode_onset(lp, bonus, on, onset_weight)
            i = 0
            while i < len(path):
                j = i + 1
                while j < len(path) and path[j] == path[i]: j += 1
                if (j - i) >= MIN_PRED_FRAMES: n += 1
                i = j
        return n
    # simple search like auto_tune_bonus
    grid = np.linspace(0.0, 8.0, 33)
    best_b, best_err = grid[0], float('inf')
    for b in grid:
        err = abs(runs_at(b) - val_target)
        if err < best_err: best_b, best_err = b, err
    bonus = float(best_b)

    # refit on all train, decode test
    X_tr, y_tr = build_set(set(all_real) - test_sent_ids)
    scaler = StandardScaler().fit(X_tr)
    clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    clf.fit(scaler.transform(X_tr), y_tr)
    train_classes = set(y_tr); class_labels = list(clf.classes_)

    predictions, pred_sentence_ids, pred_segments = [], [], []
    true_labels, true_sentence_ids, true_segments = [], [], []
    for sid in test_sent_ids:
        if sid not in per_sent: continue
        emb = per_sent[sid]; T = emb.shape[0]
        logp = smooth_cols(clf.predict_log_proba(scaler.transform(emb)),
                           SMOOTH_LOGP_W)
        on   = onset_sent[sid][:T]
        path = viterbi_decode_onset(logp, bonus, on, onset_weight)
        i = 0
        while i < T:
            ci = path[i]; j = i + 1
            while j < T and path[j] == ci: j += 1
            if (j - i) >= MIN_PRED_FRAMES:
                predictions.append(class_labels[ci])
                pred_sentence_ids.append(sid)
                pred_segments.append((ssl_frame_to_time_s(i),
                                      ssl_frame_to_time_s(j - 1)))
            i = j
        for ph in mfa_by_sid[sid]:
            if ph['phone'] not in train_classes: continue
            true_labels.append(ph['phone'])
            true_sentence_ids.append(sid)
            true_segments.append((ph['start_s'], ph['end_s']))

    if not true_labels: return None, "no test gold labels"
    true_arr = np.array(true_labels); pred_arr = np.array(predictions)
    ed  = edit_distance(list(true_arr), list(pred_arr))
    per = ed / max(len(true_arr), 1)
    return {
        'true_labels': true_arr, 'predictions': pred_arr,
        'true_sentence_ids': np.array(true_sentence_ids),
        'pred_sentence_ids': np.array(pred_sentence_ids),
        'true_segments': true_segments, 'pred_segments': pred_segments,
        'accuracy': float('nan'), 'edit_distance': ed, 'per': per,
        'n_test': len(true_arr), 'n_pred': len(pred_arr),
        'n_train': len(X_tr), 'bonus': bonus,
        'onset_weight': onset_weight,
    }, None


ssl_onset_results = {}
for pid in TARGET_PIDS:
    if pid not in onset_heads: continue
    out, err = run_for_patient_ssl_onset(pid, onset_weight=ONSET_WEIGHT)
    if err: print(f"  {pid}: SKIP — {err}"); continue
    ssl_onset_results[pid] = out
    m = nw_metrics(out)
    print(f"{pid}: match={100*m['match_rate']:5.1f}%  z={m['z_match']:+5.2f}  "
          f"n2={m['n2']:>3} n3={m['n3']:>3} n4={m['n4']:>3}  "
          f"pred/gold={out['n_pred']}/{out['n_test']}  "
          f"bonus={out['bonus']:.2f}")

am = np.mean([nw_metrics(o)['match_rate'] for o in ssl_onset_results.values()])
az = np.mean([nw_metrics(o)['z_match']    for o in ssl_onset_results.values()])
print(f"\nCohort: match={100*am:.1f}%  z={az:+.2f}")
print(f"For reference, SSL-only without onsets:  match=29.0%  z=+2.18")

# %% Cell 9 — evaluation + visualization (SSL-only encoder run)
# ============================================================
# point the visualization at the SSL-only results (the latest cohort run)
pipeline.patient_results = dict(ssl_onset_results)

for pid in TARGET_PIDS:
    if pid not in pipeline.patient_results: continue
    print(f"\n{'='*60}\n{pid} — SSL-only (15% mask) + LDA + scalar Viterbi\n{'='*60}")

    show_matched_sequences_with_times(pipeline, pid,
                                       max_per_line=45,
                                       collapse_repeats=True,
                                       time_align_tol_s=0.10)

    m = nw_metrics(pipeline.patient_results[pid])
    print_nw_metrics(m, label=pid)

# %% Cell — extract matched n-grams + diversity stats
# ============================================================
def extract_match_ngrams(out):
    """Walk NW alignments per sentence; collect runs of consecutive matches.
    Returns lists of (sid, gram) tuples for n2/n3/n4+, and a flat list of
    matched single phonemes (with repetition)."""
    gold_per, pred_per = gather_sequences(out)
    all_match_phones = []
    n2g, n3g, n4plus = [], [], []
    for sid in sorted(set(gold_per) | set(pred_per)):
        gold = gold_per.get(sid, []); pred = pred_per.get(sid, [])
        aligned = needleman_wunsch(gold, pred)
        run = []
        for g, p in aligned:
            if g is not None and p is not None and g == p:
                run.append(g)
            else:
                if run:
                    all_match_phones.extend(run)
                    if   len(run) == 2: n2g.append((sid, tuple(run)))
                    elif len(run) == 3: n3g.append((sid, tuple(run)))
                    elif len(run) >= 4: n4plus.append((sid, tuple(run)))
                run = []
        if run:
            all_match_phones.extend(run)
            if   len(run) == 2: n2g.append((sid, tuple(run)))
            elif len(run) == 3: n3g.append((sid, tuple(run)))
            elif len(run) >= 4: n4plus.append((sid, tuple(run)))
    return all_match_phones, n2g, n3g, n4plus


def diversity_stats(out, n_inventory=None):
    """Returns dict of diversity stats.  n_inventory = gold-side total
    unique phoneme count (so we can express 'coverage' as a fraction)."""
    phones, n2g, n3g, n4g = extract_match_ngrams(out)

    # gold inventory: how many distinct phonemes exist in the test set
    if n_inventory is None:
        gold_per, _ = gather_sequences(out)
        all_gold = [p for seq in gold_per.values() for p in seq]
        n_inventory = len(set(all_gold))

    uniq_ph = set(phones)
    uniq_n2 = set(g for _, g in n2g)
    uniq_n3 = set(g for _, g in n3g)
    uniq_n4 = set(g for _, g in n4g)
    return {
        'n_match':     len(phones),
        'uniq_phones': len(uniq_ph),
        'inv_size':    n_inventory,
        'phone_cov':   len(uniq_ph) / max(n_inventory, 1),  # fraction of gold inv hit
        'n2_total':    len(n2g),  'uniq_n2': len(uniq_n2),
        'n3_total':    len(n3g),  'uniq_n3': len(uniq_n3),
        'n4_total':    len(n4g),  'uniq_n4': len(uniq_n4),
        'n3_diversity': len(uniq_n3) / max(len(n3g), 1),    # 1.0 = all unique
        'top_n3':      Counter(g for _, g in n3g).most_common(3),
        'top_n4':      Counter(g for _, g in n4g).most_common(3),
    }

# %% Cell — cohort TR sweep with diversity reporting
# ============================================================
from collections import Counter

TR_GRID = [0.8, 0.9, 1.0, 1.1, 1.2,  1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]    # extend further if you want
ssl_tr_div_sweep = {}

for tr in TR_GRID:
    TARGET_RATIO = tr
    ssl_tr_div_sweep[tr] = {}
    for pid in TARGET_PIDS:
        if pid not in embeddings: continue
        out, err = run_for_patient_ssl(pid)
        if err: continue
        ssl_tr_div_sweep[tr][pid] = out

# header
print(f"\n{'TR':>4} {'match':>7} {'z':>6}  "
      f"{'uniq_ph':>8} {'cov':>6}  "
      f"{'uniq_n2':>8} {'uniq_n3':>8} {'uniq_n4':>8}  "
      f"{'n3_div':>7}  top-n3")
print('-' * 100)

for tr in TR_GRID:
    rows = [(pid, ssl_tr_div_sweep[tr][pid])
            for pid in TARGET_PIDS if pid in ssl_tr_div_sweep[tr]]
    if not rows: continue

    # cohort aggregates
    cohort_phones, cohort_n2, cohort_n3, cohort_n4 = [], [], [], []
    matches, zs = [], []
    for pid, out in rows:
        m = nw_metrics(out); matches.append(m['match_rate']); zs.append(m['z_match'])
        ph, n2g, n3g, n4g = extract_match_ngrams(out)
        cohort_phones.extend(ph)
        cohort_n2.extend(g for _, g in n2g)
        cohort_n3.extend(g for _, g in n3g)
        cohort_n4.extend(g for _, g in n4g)

    inv = len(set(p for pid, out in rows
                  for seq in gather_sequences(out)[0].values()
                  for p in seq))
    u_ph = set(cohort_phones)
    u_n2, u_n3, u_n4 = set(cohort_n2), set(cohort_n3), set(cohort_n4)
    n3_div = len(u_n3) / max(len(cohort_n3), 1)
    top3 = Counter(cohort_n3).most_common(3)
    top3_str = '  '.join(f"{''.join(g)}×{c}" for g, c in top3) or '—'

    print(f"{tr:>4.1f} {100*np.mean(matches):6.1f}% {np.mean(zs):+5.2f}  "
          f"{len(u_ph):>4}/{inv:<3} {len(u_ph)/inv:5.0%}  "
          f"{len(u_n2):>4}/{len(cohort_n2):<3} "
          f"{len(u_n3):>4}/{len(cohort_n3):<3} "
          f"{len(u_n4):>4}/{len(cohort_n4):<3}  "
          f"{n3_div:>6.1%}  {top3_str}")

# per-patient unique n3 / top n3, if you want to spot patient-specific repetition
print("\nPer-patient diversity at best TR:")
BEST_TR = max(TR_GRID, key=lambda t: np.mean(
    [nw_metrics(o)['match_rate'] for o in ssl_tr_div_sweep[t].values()]))
print(f"  (using TR={BEST_TR})\n")
for pid in TARGET_PIDS:
    if pid not in ssl_tr_div_sweep[BEST_TR]: continue
    out = ssl_tr_div_sweep[BEST_TR][pid]
    d = diversity_stats(out)
    top = '  '.join(f"{''.join(g)}×{c}" for g, c in d['top_n3'])
    print(f"  {pid}: uniq_phones={d['uniq_phones']:>2}/{d['inv_size']:>2} ({d['phone_cov']:>4.0%})  "
          f"n3 {d['uniq_n3']}/{d['n3_total']} unique  ({d['n3_diversity']:>4.0%}) "
          f"  top: {top or '—'}")

# %% Cell — full-cohort HG_only vs HG+low_beta, fair A/B
# ============================================================
import random

CONFIGS = [
    ('HG_only',     [(70, 170)],         80),
    ('HG+low_beta', [(70, 170), (13, 20)], 120),
]

ab_results = {}   # (spec, pid, tr) -> out
for spec_name, bands, epochs in CONFIGS:
    print(f"\n{'='*60}\n  {spec_name}  bands={bands}  epochs={epochs}\n{'='*60}")
    for pid in TARGET_PIDS:
        print(f"\n[{pid}] {spec_name}")
        torch.manual_seed(0); np.random.seed(0); random.seed(0)
        if DEVICE.type == 'cuda': torch.cuda.manual_seed_all(0)
        ds = build_sentence_dataset_multi(pid, bands)
        enc, _, _ = ssl_pretrain_one(pid, ds, epochs=epochs, seed=0)
        enc.eval()
        emb = {}
        with torch.no_grad():
            for s in ds['train'] + ds['test']:
                h = enc(s['X'].unsqueeze(0).to(DEVICE)).squeeze(0).cpu().numpy()
                emb[s['sent_idx']] = h.astype(np.float32)
        datasets[pid] = ds; encoders[pid] = enc; embeddings[pid] = emb
        for tr in [1.5, 1.7, 1.8, 2.0]:
            TARGET_RATIO = tr
            out, err = run_for_patient_ssl(pid)
            if err: continue
            ab_results[(spec_name, pid, tr)] = out

# summary per (spec, tr)
print(f"\n{'spec':<14} {'TR':>4} {'avg_match':>10} {'avg_z':>7} "
      f"{'Σn3':>5} {'Σn4':>5} {'Σuniq_n3':>9} {'n4_pids':>9}")
for spec_name, _, _ in CONFIGS:
    for tr in [1.5, 1.7, 1.8, 2.0]:
        outs = [ab_results.get((spec_name, pid, tr)) for pid in TARGET_PIDS]
        outs = [o for o in outs if o is not None]
        if not outs: continue
        ms = [nw_metrics(o) for o in outs]
        ds_ = [diversity_stats(o) for o in outs]
        am  = np.mean([m['match_rate'] for m in ms])
        az  = np.mean([m['z_match']    for m in ms])
        sn3 = sum(m['n3'] for m in ms); sn4 = sum(m['n4'] for m in ms)
        su3 = sum(d['uniq_n3'] for d in ds_)
        n4p = sum(1 for m in ms if m['n4'] > 0)
        print(f"{spec_name:<14} {tr:>4.1f} {100*am:9.1f}% {az:+6.2f} "
              f"{sn3:>5} {sn4:>5} {su3:>9} {n4p:>3}/{len(outs)}")

# %% Cell — lock HG+low_beta v1
# ============================================================
import pickle, json, os
os.makedirs('results/v1', exist_ok=True)

V1_SPEC = 'HG+low_beta'
V1_TR   = 1.7

# pull the runs into a single dict
v1_results = {pid: ab_results[(V1_SPEC, pid, V1_TR)]
              for pid in TARGET_PIDS
              if (V1_SPEC, pid, V1_TR) in ab_results}

with open('results/v1/hg_lowbeta_tr17_cohort.pkl', 'wb') as f:
    pickle.dump(v1_results, f)

summary = {'spec': V1_SPEC, 'bands': [(70, 170), (13, 20)],
           'SSL_EPOCHS': 120, 'TARGET_RATIO': V1_TR,
           'MIN_PRED_FRAMES': 3, 'SMOOTH_LOGP_W': 31,
           'cohort': {}}
for pid in TARGET_PIDS:
    if pid not in v1_results: continue
    out = v1_results[pid]
    m = nw_metrics(out); d = diversity_stats(out)
    summary['cohort'][pid] = {
        'match_rate': float(m['match_rate']),
        'z_match':    float(m['z_match']),
        'n2': int(m['n2']), 'n3': int(m['n3']), 'n4': int(m['n4']),
        'uniq_n3': int(d['uniq_n3']), 'n3_total': int(d['n3_total']),
        'n_pred': int(out['n_pred']), 'n_test': int(out['n_test']),
        'bonus': float(out['bonus']),
    }
with open('results/v1/hg_lowbeta_tr17_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f"v1 locked: {V1_SPEC} at TR={V1_TR}")
print(f"  cohort match  = {100*np.mean([v['match_rate'] for v in summary['cohort'].values()]):.1f}%")
print(f"  cohort z      = {np.mean([v['z_match']     for v in summary['cohort'].values()]):+.2f}")
print(f"  cohort Σ(n≥3) = {sum(v['n3']+v['n4'] for v in summary['cohort'].values())}")
print(f"  n_pat_n4      = {sum(1 for v in summary['cohort'].values() if v['n4']>0)}/{len(summary['cohort'])}")

# the encoders are still in memory from the sweep — save them under the new tag
for pid in TARGET_PIDS:
    if pid not in encoders: continue
    enc = encoders[pid]
    torch.save({'enc': enc.state_dict(),
                'n_in': enc.proj_in.in_channels,
                'bands': [(70, 170), (13, 20)],
                'epochs': 120},
               os.path.join(MODEL_DIR, f'{pid}_ssl_encoder_low_beta.pt'))
print("Encoder checkpoints saved with `_low_beta` suffix.")

# %% Cell — multiband sweep at TR=1.8 with diversity + chain criteria
# ============================================================
SWEEP_SPECS = {
    'HG_only'        : [(70, 170)],
    'HG_plus_LG'     : [(70, 170), (30, 70)],
    'HG_LG_theta'    : [(70, 170), (30, 70), (4, 8)],
    'HG_beta'        : [(70, 170), (13, 30)],
}
SWEEP_PIDS = ['P22', 'P28', 'P30']
TARGET_RATIO = 1.8
MIN_PRED_FRAMES = 3
SMOOTH_LOGP_W   = 31

def train_spec_ssl_only(pid, bands, seed=0, ssl_epochs=80):
    """SSL pretrain + LDA inference, no onset head, scalar-bonus Viterbi."""
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    if DEVICE.type == 'cuda': torch.cuda.manual_seed_all(seed)
    ds = build_sentence_dataset_multi(pid, bands)
    enc, mu, sd = ssl_pretrain_one(pid, ds, epochs=ssl_epochs, seed=seed)
    enc.eval()
    emb = {}
    with torch.no_grad():
        for s in ds['train'] + ds['test']:
            h = enc(s['X'].unsqueeze(0).to(DEVICE)).squeeze(0).cpu().numpy()
            emb[s['sent_idx']] = h.astype(np.float32)
    global datasets, encoders, embeddings
    datasets[pid] = ds; encoders[pid] = enc; embeddings[pid] = emb
    return run_for_patient_ssl(pid)


# save state so we can restore after sweep
_saved = {pid: (datasets.get(pid), encoders.get(pid), embeddings.get(pid))
          for pid in SWEEP_PIDS}

mb_tr18_results = {}
for pid in SWEEP_PIDS:
    for name, bands in SWEEP_SPECS.items():
        tag = f"{pid}_{name}"
        print(f"\n=== {tag}  bands={bands} ===")
        out, err = train_spec_ssl_only(pid, bands)
        if err: print(f"  SKIP — {err}"); continue
        m = nw_metrics(out)
        d = diversity_stats(out)
        mb_tr18_results[tag] = {
            'pid':pid, 'spec':name, 'out':out, 'm':m, 'd':d,
        }
        top3 = '  '.join(f"{''.join(g)}×{c}" for g, c in d['top_n3'])
        print(f"  match={100*m['match_rate']:5.1f}%  z={m['z_match']:+5.2f}  "
              f"n3={m['n3']} n4={m['n4']}  uniq_n3={d['uniq_n3']}/{d['n3_total']}  "
              f"top: {top3 or '—'}")

# restore
for pid in SWEEP_PIDS:
    ds_, enc_, emb_ = _saved[pid]
    if ds_  is not None: datasets[pid]   = ds_
    if enc_ is not None: encoders[pid]   = enc_
    if emb_ is not None: embeddings[pid] = emb_

# group by spec, sum over patients
print(f"\n{'spec':<18} {'avg_match':>10} {'avg_z':>7} {'Σn3':>5} {'Σn4':>5} "
      f"{'Σuniq_n3':>9} {'Σuniq_n4':>9}  n4_pids")
by_spec = {}
for tag, r in mb_tr18_results.items():
    by_spec.setdefault(r['spec'], []).append(r)
for spec, rows in by_spec.items():
    am = np.mean([r['m']['match_rate'] for r in rows])
    az = np.mean([r['m']['z_match']    for r in rows])
    sn3 = sum(r['m']['n3'] for r in rows)
    sn4 = sum(r['m']['n4'] for r in rows)
    su3 = sum(r['d']['uniq_n3'] for r in rows)
    su4 = sum(r['d']['uniq_n4'] for r in rows)
    n4_pids = [r['pid'] for r in rows if r['m']['n4'] > 0]
    print(f"{spec:<18} {100*am:9.1f}% {az:+6.2f} {sn3:>5} {sn4:>5} "
          f"{su3:>9} {su4:>9}  {n4_pids}")

# %% Cell — side-by-side HG_only vs HG_beta at TR=1.8 (HTML flex layout)
# ============================================================
from IPython.utils.capture import capture_output
from IPython.display import HTML, display
from collections import Counter

# same metric helpers used in your existing template
from LDA_on_frames_clean import (
    longest_run_with_shift, collect_matches,
    surprise_score, perm_null,
)

# --- pick patient and the two configs to compare ---------------------
PID    = 'P22'                              # one of SWEEP_PIDS: P22, P28, P30
SPEC_A = ('HG_only',  f'{PID}_HG_only')
SPEC_B = ('HG_beta',  f'{PID}_HG_beta')
# --------------------------------------------------------------------

if not hasattr(pipeline, 'patient_results') or pipeline.patient_results is None:
    pipeline.patient_results = {}


def _summary(pr):
    """Recompute the same headline numbers your template uses
    (surprise-based z, longest n-gram, n-grams matched, PER, etc.)."""
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
    return {'z': z, 'longest': max_run, 'ngrams': len(matches),
            'n_pred': pr['n_pred'], 'n_test': pr['n_test'],
            'bonus':  pr.get('bonus', float('nan')),
            'per':    pr['per']}


def render_to_html(pid, tag):
    """Pull `out` from mb_tr18_results, install into pipeline.patient_results,
    run show_matched_sequences_with_times, capture as one HTML blob."""
    if tag not in mb_tr18_results:
        return f"<p style='color:#a00'>SKIPPED — missing {tag}</p>", None
    out = mb_tr18_results[tag]['out']
    pipeline.patient_results[pid] = out
    summary = _summary(out)
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
            f"<pre style='font-size:11px; margin:4px 0; "
            f"white-space:pre-wrap;'>{cap.stdout}</pre>")
    for output in cap.outputs:
        if hasattr(output, 'data') and 'text/html' in output.data:
            html_parts.append(output.data['text/html'])
    return '\n'.join(html_parts), summary


print(f"=== Comparing {PID}:  '{SPEC_A[0]}'  vs  '{SPEC_B[0]}'  (TR=1.8) ===")
html_a, sum_a = render_to_html(PID, SPEC_A[1])
html_b, sum_b = render_to_html(PID, SPEC_B[1])


def _color_z(z):
    return (f"<span style='color:#c00; font-weight:bold;'>{z:+.2f}</span>"
            if z > 2 else f"<span style='font-weight:bold;'>{z:+.2f}</span>")


def _header_block(label, s):
    if s is None: return f"<h3>{label}</h3><p>no result</p>"
    return (
        f"<h3 style='margin:4px 0;'>{label}</h3>"
        f"<div style='font-family:monospace; font-size:12px; "
        f"padding:6px; background:#f4f4f4; border-radius:4px; margin-bottom:8px;'>"
        f"z={_color_z(s['z'])} &nbsp; longest n-gram={s['longest']} &nbsp; "
        f"n-grams matched={s['ngrams']} &nbsp; PER={s['per']:.1%}<br>"
        f"bonus={s['bonus']:.2f} &nbsp; "
        f"n_pred={s['n_pred']}/{s['n_test']}"
        f"</div>"
    )


combined = f"""
<div style="display:flex; gap:20px; align-items:flex-start;">
  <div style="flex:1; min-width:0; overflow-x:auto;">
    {_header_block(SPEC_A[0], sum_a)}
    {html_a}
  </div>
  <div style="flex:1; min-width:0; overflow-x:auto;">
    {_header_block(SPEC_B[0], sum_b)}
    {html_b}
  </div>
</div>
"""
display(HTML(combined))

# %% Cell — beta-processing variants sweep at TR=1.8
# ============================================================
# Extended extractor allowing per-band LP cutoff + log-power option.

def _extract_band_amp_cfg(data, sr, low, high, lp_hz, log_power=False,
                           win_s=0.015, shift_s=0.005):
    x = sps.detrend(data, axis=0)
    sos = sps.iirfilter(4, [low/(sr/2), high/(sr/2)], btype='bandpass',
                        output='sos')
    x = sps.sosfiltfilt(sos, x, axis=0)
    # only notch if the band includes line-noise frequencies
    if high > 95:
        for f0 in (100, 150):
            sos_n = sps.iirfilter(4, [(f0-2)/(sr/2), (f0+2)/(sr/2)],
                                  btype='bandstop', output='sos')
            x = sps.sosfiltfilt(sos_n, x, axis=0)
    x = x ** 2
    sos_lp = sps.iirfilter(4, lp_hz/(sr/2), btype='lowpass', output='sos')
    x = np.abs(sps.sosfiltfilt(sos_lp, x, axis=0))
    win = int(win_s * sr); hop = int(shift_s * sr)
    n_win = int(np.floor((x.shape[0] - win) / hop))
    feat = np.zeros((n_win, x.shape[1]))
    for i in range(n_win):
        feat[i] = x[i*hop : i*hop + win].mean(axis=0)
    return np.log1p(feat) if log_power else np.sqrt(feat)


def extract_multiband_cfg(data, sr, band_configs):
    """band_configs: list of dicts {low, high, lp_hz, log_power}"""
    feats = [_extract_band_amp_cfg(data, sr, **cfg) for cfg in band_configs]
    n_min = min(f.shape[0] for f in feats)
    return np.concatenate([f[:n_min] for f in feats], axis=1).astype(np.float32)


def build_sentence_dataset_cfg(pid, band_configs):
    """Same as build_sentence_dataset_multi but takes band_configs (list of dicts)."""
    raw_eeg = np.load(os.path.join(DUTCH_30_PATH, 'raw', f'{pid}_sEEG.npy'))
    if raw_eeg.ndim == 2 and raw_eeg.shape[0] < raw_eeg.shape[1]:
        raw_eeg = raw_eeg.T
    keep = _channel_mask(pid)
    if keep is not None: raw_eeg = raw_eeg[:, keep]
    wd  = pipeline.split_result['word_segments_dict'][pid]
    mfa = load_mfa_alignments(pid)
    all_real = [i for i, s in enumerate(wd['sentence_list'])
                if isinstance(s, dict) and s.get('text')]
    test_sent_ids = set(all_real[TEST_OFFSET::6])
    out = {'train': [], 'test': []}
    for sent_idx in all_real:
        if sent_idx not in mfa or not mfa[sent_idx]: continue
        s = wd['sentence_list'][sent_idx]
        s0, s1 = s['stim_start_idx'], s['stim_end_idx']
        if s1 > raw_eeg.shape[0]: continue
        X = extract_multiband_cfg(raw_eeg[s0:s1], EEG_SR, band_configs)
        if X.shape[0] < 30: continue
        split = 'test' if sent_idx in test_sent_ids else 'train'
        out[split].append({'X': torch.from_numpy(X),
                           'mfa': mfa[sent_idx], 'sent_idx': sent_idx})
    return out


def train_cfg_ssl_only(pid, band_configs, seed=0, ssl_epochs=80):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    if DEVICE.type == 'cuda': torch.cuda.manual_seed_all(seed)
    ds = build_sentence_dataset_cfg(pid, band_configs)
    enc, mu, sd = ssl_pretrain_one(pid, ds, epochs=ssl_epochs, seed=seed)
    enc.eval()
    emb = {}
    with torch.no_grad():
        for s in ds['train'] + ds['test']:
            h = enc(s['X'].unsqueeze(0).to(DEVICE)).squeeze(0).cpu().numpy()
            emb[s['sent_idx']] = h.astype(np.float32)
    global datasets, encoders, embeddings
    datasets[pid] = ds; encoders[pid] = enc; embeddings[pid] = emb
    return run_for_patient_ssl(pid)


# ── sweep configurations ────────────────────────────────────────────
# at top of the cell, replace TARGET_RATIO
TARGET_RATIO    = 1.7        # was 1.8 — match v1
MIN_PRED_FRAMES = 3
SMOOTH_LOGP_W   = 31

# spec values are now (band_configs, ssl_epochs) so each spec sets its own ep
HG_cfg = {'low':70, 'high':170, 'lp_hz':10.0, 'log_power':False}
BETA_SPECS = {
    'HG_only':                    ([HG_cfg],                                                            80),
    'HG+low_beta_v1_10Hz':        ([HG_cfg, {'low':13, 'high':20, 'lp_hz':10.0, 'log_power':False}],  120),  # ← v1
    'HG+low_beta_slowLP_5Hz':     ([HG_cfg, {'low':13, 'high':20, 'lp_hz':5.0,  'log_power':False}],  120),  # ← what you ran
    'HG+beta_default':            ([HG_cfg, {'low':13, 'high':30, 'lp_hz':10.0, 'log_power':False}],  120),
    'HG+beta_slowLP':             ([HG_cfg, {'low':13, 'high':30, 'lp_hz':5.0,  'log_power':False}],  120),
    'HG+beta_log':                ([HG_cfg, {'low':13, 'high':30, 'lp_hz':10.0, 'log_power':True}],   120),
    'HG+high_beta':               ([HG_cfg, {'low':20, 'high':30, 'lp_hz':5.0,  'log_power':False}],  120),
}

# the inner loop now unpacks the tuple and passes ssl_epochs explicitly
for pid in SWEEP_PIDS:
    for name, (cfgs, ep) in BETA_SPECS.items():
        tag = f"{pid}_{name}"
        print(f"\n=== {tag}  ep={ep} ===")
        out, err = train_cfg_ssl_only(pid, cfgs, ssl_epochs=ep)
        if err: print(f"  SKIP — {err}"); continue
        m = nw_metrics(out); d = diversity_stats(out)
        beta_sweep[tag] = {'pid':pid, 'spec':name, 'epochs':ep,
                           'out':out, 'm':m, 'd':d}
        ...

# in the summary, include the epochs column too
print(f"\n{'spec':<20} {'ep':>4} {'avg_match':>10} {'avg_z':>7} {'Σn3':>5} {'Σn4':>5} "
      f"{'Σuniq_n3':>9}  n4_pids")

by_spec = {}
for tag, r in beta_sweep.items():
    by_spec.setdefault(r['spec'], []).append(r)
for spec in BETA_SPECS:
    if spec not in by_spec: continue
    rows = by_spec[spec]
    print(f"{spec:<20} {100*np.mean([r['m']['match_rate'] for r in rows]):9.1f}% "
          f"{np.mean([r['m']['z_match'] for r in rows]):+6.2f} "
          f"{sum(r['m']['n3'] for r in rows):>5} "
          f"{sum(r['m']['n4'] for r in rows):>5} "
          f"{sum(r['d']['uniq_n3'] for r in rows):>9}  "
          f"{[r['pid'] for r in rows if r['m']['n4'] > 0]}")

# %% Cell — phase-band sweep on top of v1
# ============================================================
import scipy.signal as sps
import random

def _extract_phase_pair(data, sr, low, high, win_s=0.015, shift_s=0.005):
    """Per-electrode Hilbert phase as [sin(φ), cos(φ)] channels, sampled
    at frame grid centres.  Returns (T, n_ch * 2)."""
    x = sps.detrend(data, axis=0)
    sos = sps.iirfilter(4, [low/(sr/2), high/(sr/2)],
                        btype='bandpass', output='sos')
    x = sps.sosfiltfilt(sos, x, axis=0)
    phi = np.angle(sps.hilbert(x, axis=0))                    # (samples, ch)
    win = int(win_s * sr); hop = int(shift_s * sr)
    n_win = int(np.floor((x.shape[0] - win) / hop))
    sins = np.zeros((n_win, phi.shape[1]), dtype=np.float32)
    coss = np.zeros((n_win, phi.shape[1]), dtype=np.float32)
    for i in range(n_win):
        c = i * hop + win // 2
        sins[i] = np.sin(phi[c])
        coss[i] = np.cos(phi[c])
    return np.concatenate([sins, coss], axis=1)


def build_sentence_dataset_with_phase(pid, amp_bands, phase_bands):
    """amp_bands: list of (low, high) for amplitude features (extractHG recipe)
       phase_bands: list of (low, high) for phase features (sin/cos encoded)"""
    raw_eeg = np.load(os.path.join(DUTCH_30_PATH, 'raw', f'{pid}_sEEG.npy'))
    if raw_eeg.ndim == 2 and raw_eeg.shape[0] < raw_eeg.shape[1]:
        raw_eeg = raw_eeg.T
    keep = _channel_mask(pid)
    if keep is not None: raw_eeg = raw_eeg[:, keep]

    wd  = pipeline.split_result['word_segments_dict'][pid]
    mfa = load_mfa_alignments(pid)
    all_real = [i for i, s in enumerate(wd['sentence_list'])
                if isinstance(s, dict) and s.get('text')]
    test_sent_ids = set(all_real[TEST_OFFSET::6])

    out = {'train': [], 'test': []}
    for sent_idx in all_real:
        if sent_idx not in mfa or not mfa[sent_idx]: continue
        s = wd['sentence_list'][sent_idx]
        s0, s1 = s['stim_start_idx'], s['stim_end_idx']
        if s1 > raw_eeg.shape[0]: continue
        sl = raw_eeg[s0:s1]
        amp_feats   = [_extract_band_amp(sl, EEG_SR, lo, hi)
                       for (lo, hi) in amp_bands]
        phase_feats = [_extract_phase_pair(sl, EEG_SR, lo, hi)
                       for (lo, hi) in phase_bands]
        feats = amp_feats + phase_feats
        n_min = min(f.shape[0] for f in feats)
        X = np.concatenate([f[:n_min] for f in feats], axis=1).astype(np.float32)
        if X.shape[0] < 30: continue
        split = 'test' if sent_idx in test_sent_ids else 'train'
        out[split].append({'X': torch.from_numpy(X),
                           'mfa': mfa[sent_idx], 'sent_idx': sent_idx})
    print(f"  [{pid}] amp={amp_bands} phase={phase_bands}  "
          f"n_in={out['train'][0]['X'].shape[1] if out['train'] else 0}  "
          f"train={len(out['train'])}  test={len(out['test'])}")
    return out


def train_phase_cfg(pid, amp_bands, phase_bands, epochs=180, seed=0):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    if DEVICE.type == 'cuda': torch.cuda.manual_seed_all(seed)
    ds = build_sentence_dataset_with_phase(pid, amp_bands, phase_bands)
    enc, _, _ = ssl_pretrain_one(pid, ds, epochs=epochs, seed=seed)
    enc.eval()
    emb = {}
    with torch.no_grad():
        for s in ds['train'] + ds['test']:
            h = enc(s['X'].unsqueeze(0).to(DEVICE)).squeeze(0).cpu().numpy()
            emb[s['sent_idx']] = h.astype(np.float32)
    global datasets, encoders, embeddings
    datasets[pid] = ds; encoders[pid] = enc; embeddings[pid] = emb
    return run_for_patient_ssl(pid)


# ── sweep configs ───────────────────────────────────────────────────
PHASE_SPECS = {
    'v1_HG+low_beta_amp':              ([(70,170), (13,20)], [],          120),  # control
    'v1 + low_beta_phase':             ([(70,170), (13,20)], [(13,20)],   180),
    'v1 + theta_phase':                ([(70,170), (13,20)], [(4,8)],     180),
}
SWEEP_PIDS   = ['P22', 'P28', 'P30']
TARGET_RATIO = 1.7
MIN_PRED_FRAMES = 3
SMOOTH_LOGP_W   = 31

_saved = {pid: (datasets.get(pid), encoders.get(pid), embeddings.get(pid))
          for pid in SWEEP_PIDS}

phase_sweep = {}
for pid in SWEEP_PIDS:
    for name, (amp, phase, ep) in PHASE_SPECS.items():
        tag = f"{pid}_{name}"
        print(f"\n=== {tag}  ep={ep} ===")
        out, err = train_phase_cfg(pid, amp, phase, epochs=ep)
        if err: print(f"  SKIP — {err}"); continue
        m = nw_metrics(out); d = diversity_stats(out)
        phase_sweep[tag] = {'pid':pid, 'spec':name, 'epochs':ep,
                            'out':out, 'm':m, 'd':d}
        top3 = '  '.join(f"{''.join(g)}×{c}" for g, c in d['top_n3'][:3])
        print(f"  match={100*m['match_rate']:5.1f}%  z={m['z_match']:+5.2f}  "
              f"n3={m['n3']} n4={m['n4']}  uniq_n3={d['uniq_n3']}/{d['n3_total']}  "
              f"top: {top3 or '—'}")

for pid in SWEEP_PIDS:
    ds_, enc_, emb_ = _saved[pid]
    if ds_  is not None: datasets[pid]   = ds_
    if enc_ is not None: encoders[pid]   = enc_
    if emb_ is not None: embeddings[pid] = emb_

print(f"\n{'spec':<28} {'ep':>4} {'avg_match':>10} {'avg_z':>7} "
      f"{'Σn3':>5} {'Σn4':>5} {'Σuniq_n3':>9}  n4_pids")
by_spec = {}
for tag, r in phase_sweep.items():
    by_spec.setdefault((r['spec'], r['epochs']), []).append(r)
for (spec, ep), rows in by_spec.items():
    am = np.mean([r['m']['match_rate'] for r in rows])
    az = np.mean([r['m']['z_match']    for r in rows])
    print(f"{spec:<28} {ep:>4} {100*am:9.1f}% {az:+6.2f} "
          f"{sum(r['m']['n3'] for r in rows):>5} "
          f"{sum(r['m']['n4'] for r in rows):>5} "
          f"{sum(r['d']['uniq_n3'] for r in rows):>9}  "
          f"{[r['pid'] for r in rows if r['m']['n4'] > 0]}")

# %% Cell — full-cohort v1 vs v1+low_beta_phase
# ============================================================
CONFIGS = [
    ('v1_HG+low_beta_amp',  [(70,170), (13,20)], [],         120),
    ('v1+low_beta_phase',   [(70,170), (13,20)], [(13,20)],  180),
]

phase_ab = {}
for spec_name, amp, phase, epochs in CONFIGS:
    print(f"\n{'='*60}\n  {spec_name}  ep={epochs}\n{'='*60}")
    for pid in TARGET_PIDS:
        out, err = train_phase_cfg(pid, amp, phase, epochs=epochs)
        if err: continue
        phase_ab[(spec_name, pid)] = out

print(f"\n{'spec':<24} {'avg_match':>10} {'avg_z':>7} {'Σn4':>5} {'n_pat_n4':>9}")
for spec_name, _, _, _ in CONFIGS:
    outs = [phase_ab.get((spec_name, pid)) for pid in TARGET_PIDS]
    outs = [o for o in outs if o is not None]
    ms = [nw_metrics(o) for o in outs]
    am = np.mean([m['match_rate'] for m in ms])
    az = np.mean([m['z_match']    for m in ms])
    sn4 = sum(m['n4'] for m in ms)
    n4p = sum(1 for m in ms if m['n4'] > 0)
    print(f"{spec_name:<24} {100*am:9.1f}% {az:+6.2f} {sn4:>5} {n4p:>3}/{len(outs)}")
