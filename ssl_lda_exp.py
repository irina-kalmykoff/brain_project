# Converted from ssl_lda_exp.ipynb

# %% Cell 1 — setup
# ============================================================
# Imports + pipeline init.  Reuses helpers from arrive_at_v1.py if you already
# ran it; otherwise loads the minimum needed for a standalone walkthrough.

import os, time, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.signal as sps
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from collections import Counter

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

from dutch_30_pipeline import Dutch30Pipeline
from dutch_30_feature_extractor import Dutch30FeatureExtractor
from dataset_config import Dutch30Config
from config import DUTCH_30_PATH
from run_pipeline import load_mfa_alignments
from LDA_on_frames_clean import (
    smooth_cols, viterbi_decode, auto_tune_bonus,
    nw_metrics, gather_sequences, needleman_wunsch,
    EEG_SR, WIN_S, SHIFT_S, WIN_SAMP, SHIFT_SAMP, VAL_FRAC, TEST_OFFSET,
    SELF_LOOP_BONUS,
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
plt.rcParams['figure.dpi'] = 100
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

# v1 constants (single source of truth — change here, propagates everywhere)
V1_BANDS        = [(70, 170), (13, 20)]   # HG amp + low_beta amp
HIDDEN_DIM      = 128
TCN_KERNEL      = 5
TCN_DILATIONS   = (1, 2, 4, 8)
TARGET_RATIO    = 1.7
MIN_PRED_FRAMES = 3
SMOOTH_LOGP_W   = 31

DEMO_PID  = 'P22'
DEMO_SENT = None   # set automatically below — first non-trivial train sentence

# pipeline init (idempotent — skip if already loaded)
try:
    pipeline
    print("Reusing existing `pipeline`.")
except NameError:
    cfg = Dutch30Config()
    extractor = Dutch30FeatureExtractor(config=cfg)
    pipeline = Dutch30Pipeline(extractor, config=cfg, use_wav2vec=False)
    pipeline.step1_load_dutch30_data(patient_range=(21, 30))
    pipeline.step2_split_by_instances(train_fraction=0.8)
    pipeline.step3_load_channel_exclusions('channel_exclusions.json')
    pipeline.apply_channel_exclusions()

print(f"DEVICE = {DEVICE}   running demo on patient {DEMO_PID}")

# %% Cell 2 — what does the raw EEG actually look like?
# ============================================================
# sEEG is 1024 Hz, multi-channel intracranial recording.  Each channel is a
# voltage time-series from one electrode contact.  This cell shows a 1-sec
# slice during one spoken sentence — you can see the high-frequency noise +
# slow drift that any speech-decoding pipeline has to work with.

raw_eeg = np.load(os.path.join(DUTCH_30_PATH, 'raw', f'{DEMO_PID}_sEEG.npy'))
if raw_eeg.ndim == 2 and raw_eeg.shape[0] < raw_eeg.shape[1]:
    raw_eeg = raw_eeg.T
keep = pipeline.channel_masks[DEMO_PID]['keep_indices']
raw_eeg = raw_eeg[:, np.asarray(keep)]
print(f"  {DEMO_PID} raw EEG: shape={raw_eeg.shape}  "
      f"({raw_eeg.shape[0]/EEG_SR:.0f} seconds, {raw_eeg.shape[1]} channels)")

# pick a train sentence to use throughout
wd = pipeline.split_result['word_segments_dict'][DEMO_PID]
mfa_all = load_mfa_alignments(DEMO_PID)
all_real = [i for i, s in enumerate(wd['sentence_list'])
            if isinstance(s, dict) and s.get('text')]
test_sent_ids = set(all_real[TEST_OFFSET::6])
for sid in all_real:
    if sid in test_sent_ids: continue
    if sid not in mfa_all or not mfa_all[sid]: continue
    if len(mfa_all[sid]) >= 8:   # reasonably long sentence
        DEMO_SENT = sid
        break
print(f"  using demo sentence sid={DEMO_SENT}  text='{wd['sentence_list'][DEMO_SENT]['text']}'")
print(f"  {len(mfa_all[DEMO_SENT])} phonemes in this sentence")

s = wd['sentence_list'][DEMO_SENT]
s0, s1 = s['stim_start_idx'], s['stim_end_idx']
eeg_slice = raw_eeg[s0:s1]
print(f"  sentence EEG slice: shape={eeg_slice.shape}  ({(s1-s0)/EEG_SR:.2f} s)")

# plot 6 random channels for the first second
np.random.seed(0)
chs = np.random.choice(eeg_slice.shape[1], 6, replace=False)
fig, axes = plt.subplots(6, 1, figsize=(12, 7), sharex=True)
t = np.arange(eeg_slice.shape[0]) / EEG_SR
for ax, ch in zip(axes, chs):
    ax.plot(t, eeg_slice[:, ch], color='C0', linewidth=0.6)
    ax.set_ylabel(f"ch {ch}", rotation=0, ha='right', va='center')
    ax.grid(alpha=0.2)
axes[-1].set_xlabel('time (s)')
fig.suptitle(f"{DEMO_PID} sentence {DEMO_SENT} — raw EEG (6 channels)",
             fontsize=11)
plt.tight_layout(); plt.show()

# what to notice:
# - voltages span ±hundreds of μV; lots of high-frequency content
# - channels look qualitatively different (different brain locations)
# - hard to see speech-related structure directly in raw signal

# %% Cell 3 — turning raw EEG into ONE band's amplitude envelope
# ============================================================
# The HG (70–170 Hz) band carries phoneme-rate information in sensorimotor
# cortex.  We extract its amplitude envelope via:
#   1. bandpass to that frequency range
#   2. square the signal (positive-valued "instantaneous power")
#   3. 10 Hz lowpass (smooth out the carrier; keep phoneme-rate fluctuations)
#   4. sqrt back to amplitude units
# This cell shows each step on one channel.

ch_demo = 30  # arbitrary informative channel; pick one near speech motor cortex
x0 = eeg_slice[:, ch_demo].astype(np.float64)
x0 = sps.detrend(x0)

# step 1: bandpass 70-170 Hz
sos_bp = sps.iirfilter(4, [70/(EEG_SR/2), 170/(EEG_SR/2)],
                      btype='bandpass', output='sos')
x1 = sps.sosfiltfilt(sos_bp, x0)

# step 1.5: notch 100, 150 Hz
x_notched = x1.copy()
for f0 in (100, 150):
    sos_n = sps.iirfilter(4, [(f0-2)/(EEG_SR/2), (f0+2)/(EEG_SR/2)],
                          btype='bandstop', output='sos')
    x_notched = sps.sosfiltfilt(sos_n, x_notched)

# step 2: instantaneous power
x2 = x_notched ** 2

# step 3: 10 Hz lowpass
sos_lp = sps.iirfilter(4, 10/(EEG_SR/2), btype='lowpass', output='sos')
x3 = np.abs(sps.sosfiltfilt(sos_lp, x2))

# step 4: sqrt
x4 = np.sqrt(x3)

t = np.arange(len(x0)) / EEG_SR
fig, axes = plt.subplots(5, 1, figsize=(12, 8), sharex=True)
axes[0].plot(t, x0,        linewidth=0.5, color='gray');    axes[0].set_title("0. raw EEG (detrended)")
axes[1].plot(t, x_notched, linewidth=0.5, color='C0');      axes[1].set_title("1. 70-170 Hz bandpass + notches (100, 150 Hz)")
axes[2].plot(t, x2,        linewidth=0.5, color='C1');      axes[2].set_title("2. squared (instantaneous power)")
axes[3].plot(t, x3,        linewidth=1.2, color='C2');      axes[3].set_title("3. 10 Hz lowpass (envelope smoother)")
axes[4].plot(t, x4,        linewidth=1.2, color='C3');      axes[4].set_title("4. sqrt → amplitude")
for ax in axes: ax.grid(alpha=0.2)
axes[-1].set_xlabel('time (s)')
fig.suptitle(f"{DEMO_PID} ch{ch_demo}  sentence {DEMO_SENT} — building HG amplitude",
             fontsize=11, y=1.01)
plt.tight_layout(); plt.show()

# what to notice:
# - raw signal is wide-band noise; can't see speech
# - bandpass leaves only the HG component (still oscillating)
# - squaring + LP gives a positive smooth envelope (phoneme-rate fluctuations
#   become visible as bumps lasting ~50-100 ms)

# %% Cell 4 — multi-band features: HG amplitude + low_beta amplitude
# ============================================================
# v1 stacks TWO bands per electrode:
#   - HG (70-170 Hz) — phoneme-level articulation
#   - low_beta (13-20 Hz) — motor planning / preparation signal
# Same envelope recipe applied to each band, then concatenated along the
# channel axis.  So if a patient has 111 electrodes, the feature width
# goes from 111 (HG-only) to 222 (HG + low_beta) — TWO numbers per
# electrode per frame.

def _extract_band_amp(data, sr, low, high, lp_hz=10.0,
                      win_s=0.015, shift_s=0.005):
    x = sps.detrend(data, axis=0)
    sos = sps.iirfilter(4, [low/(sr/2), high/(sr/2)],
                        btype='bandpass', output='sos')
    x = sps.sosfiltfilt(sos, x, axis=0)
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
    return np.sqrt(feat).astype(np.float32)


def extract_multiband(data, sr, bands):
    feats = [_extract_band_amp(data, sr, lo, hi) for (lo, hi) in bands]
    n_min = min(f.shape[0] for f in feats)
    return np.concatenate([f[:n_min] for f in feats], axis=1).astype(np.float32)


# build v1 feature matrix for our demo sentence
X_multi = extract_multiband(eeg_slice, EEG_SR, V1_BANDS)
n_ch    = eeg_slice.shape[1]
print(f"  v1 features for sentence {DEMO_SENT}:  shape={X_multi.shape}  "
      f"= ({X_multi.shape[0]} frames × {X_multi.shape[1]} channels)")
print(f"  channels: [0..{n_ch-1}] HG amp,  [{n_ch}..{2*n_ch-1}] low_beta amp")

# visualize the two bands as heatmaps stacked
fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
t_frames = np.arange(X_multi.shape[0]) * SHIFT_S

vmax_hg = np.percentile(X_multi[:, :n_ch], 99)
im0 = axes[0].imshow(X_multi[:, :n_ch].T, aspect='auto', origin='lower',
                    extent=[t_frames[0], t_frames[-1], 0, n_ch],
                    cmap='viridis', vmax=vmax_hg)
axes[0].set_ylabel('electrode')
axes[0].set_title(f"HG amplitude (70-170 Hz)  — {n_ch} channels")
plt.colorbar(im0, ax=axes[0], fraction=0.02)

vmax_lb = np.percentile(X_multi[:, n_ch:], 99)
im1 = axes[1].imshow(X_multi[:, n_ch:].T, aspect='auto', origin='lower',
                    extent=[t_frames[0], t_frames[-1], 0, n_ch],
                    cmap='magma', vmax=vmax_lb)
axes[1].set_ylabel('electrode')
axes[1].set_xlabel('time (s)')
axes[1].set_title(f"low_beta amplitude (13-20 Hz)  — {n_ch} channels")
plt.colorbar(im1, ax=axes[1], fraction=0.02)

# overlay phoneme boundaries from MFA
for ax in axes:
    for ph in mfa_all[DEMO_SENT]:
        ax.axvline(ph['start_s'], color='white', linewidth=0.3, alpha=0.5)
    # label every other phoneme along the top
    for i, ph in enumerate(mfa_all[DEMO_SENT][::2]):
        mid = (ph['start_s'] + ph['end_s']) / 2
        ax.text(mid, n_ch * 1.02, ph['phone'], fontsize=8, ha='center',
                color='white' if ax is axes[0] else 'black')

fig.suptitle(f"{DEMO_PID} sentence {DEMO_SENT} — v1 features (two bands)", fontsize=11)
plt.tight_layout(); plt.show()

# what to notice:
# - both bands are time × electrodes, sampled at 200 Hz
# - HG and low_beta look qualitatively different (different magnitudes,
#   different time-courses) — they carry complementary information
# - white vertical lines = MFA phoneme boundaries; you can sometimes
#   see signal changes synchronised with them


# %% Cell 5 — standardisation: equalising the scale of HG and low_beta
# ============================================================
# low_beta amplitude is much larger than HG amplitude (lower frequencies
# carry more power on the 1/f spectrum).  Without standardisation, the
# encoder would only learn from low_beta (it has more variance to model).
# We z-score each channel across all train frames using the train-set
# mean and std.  After this, both bands sit on a comparable scale and
# the encoder can use both.

X = X_multi.copy()
mu = X.mean(0); sd = X.std(0); sd = np.where(sd < 1e-6, 1.0, sd)
X_std = (X - mu) / sd

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
# distribution before
axes[0].hist(X[:, :n_ch].ravel(),  bins=80, alpha=0.6, label='HG amp', color='C0')
axes[0].hist(X[:, n_ch:].ravel(),  bins=80, alpha=0.6, label='low_beta amp', color='C3')
axes[0].set_title("before standardisation — different scales")
axes[0].set_xlabel('amplitude'); axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].hist(X_std[:, :n_ch].ravel(),  bins=80, alpha=0.6, label='HG (std)', color='C0')
axes[1].hist(X_std[:, n_ch:].ravel(),  bins=80, alpha=0.6, label='low_beta (std)', color='C3')
axes[1].set_title("after standardisation — both centred, unit variance")
axes[1].set_xlabel('standardised amplitude'); axes[1].legend(); axes[1].grid(alpha=0.3)
plt.tight_layout(); plt.show()

print(f"  HG       mean={X[:, :n_ch].mean():.3f}  std={X[:, :n_ch].std():.3f}")
print(f"  low_beta mean={X[:, n_ch:].mean():.3f}  std={X[:, n_ch:].std():.3f}")
print(f"  → after standardise:  both mean ≈ 0, std ≈ 1")

# %% Cell 6 — the causal TCN encoder, anatomically
# ============================================================
# A Temporal Convolutional Network (TCN) is a stack of 1D convolutions over
# the time axis.  Two key properties for us:
#
#   1. CAUSAL — output at time t depends only on inputs at time ≤ t.
#      We achieve this by zero-padding on the LEFT only (not centred).
#      This is required for real-time decoding: the model can never peek
#      into the future.
#
#   2. DILATED — each successive layer skips more samples between taps,
#      so the receptive field grows exponentially with depth.
#      Dilations (1, 2, 4, 8) with kernel 5 → receptive field of
#      1 + (5-1)*(1+2+4+8) ≈ 60 frames ≈ 300 ms of past context.
#
# The encoder has 4 such TCN blocks + residual connections + GroupNorm.

class CausalConv1d(nn.Conv1d):
    """Conv1d that pads only on the left."""
    def __init__(self, in_ch, out_ch, kernel_size, dilation=1):
        super().__init__(in_ch, out_ch, kernel_size, dilation=dilation, padding=0)
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
    def forward(self, x):
        h = F.gelu(self.norm1(self.conv1(x)))
        h = self.drop(F.gelu(self.norm2(self.conv2(h))))
        return h + x   # residual


class CausalTCNEncoder(nn.Module):
    def __init__(self, n_in, hidden=HIDDEN_DIM, kernel=TCN_KERNEL,
                 dilations=TCN_DILATIONS, dropout=0.1):
        super().__init__()
        self.proj_in    = nn.Conv1d(n_in, hidden, kernel_size=1)
        self.blocks     = nn.ModuleList(
            [TCNBlock(hidden, kernel, d, dropout) for d in dilations])
        self.mask_token = nn.Parameter(torch.zeros(hidden))
    def forward(self, x, mask=None):
        h = self.proj_in(x.transpose(1, 2))
        if mask is not None:
            h = torch.where(mask.unsqueeze(1),
                            self.mask_token.view(1, -1, 1), h)
        for blk in self.blocks: h = blk(h)
        return h.transpose(1, 2)

# visualise the receptive field — which input frames does the output at
# frame t depend on, after all 4 dilated layers?
fig, ax = plt.subplots(figsize=(11, 4))
t_out = 60
for d in TCN_DILATIONS:
    # each layer's contribution
    deltas = np.arange(0, TCN_KERNEL) * d
    for delta in deltas:
        ax.scatter(t_out - delta, d, s=80, alpha=0.7,
                   color=plt.cm.viridis(np.log(d+1)/np.log(9)))
ax.set_xlim(-5, t_out + 3)
ax.set_yticks(TCN_DILATIONS)
ax.set_yticklabels([f"layer {i+1} (dilation {d})" for i, d in enumerate(TCN_DILATIONS)])
ax.set_xlabel(f"input frame index (output is at frame {t_out})")
ax.axvline(t_out, color='red', linewidth=2, alpha=0.5, label='output time')
ax.set_title(f"Causal TCN receptive field — output at frame {t_out} sees input back to frame "
             f"{t_out - (TCN_KERNEL-1)*sum(TCN_DILATIONS)}\n"
             f"(span = {(TCN_KERNEL-1)*sum(TCN_DILATIONS)} frames = "
             f"{(TCN_KERNEL-1)*sum(TCN_DILATIONS) * SHIFT_S * 1000:.0f} ms of past)")
ax.legend(loc='upper right'); ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()

# what to notice:
# - never any dot to the RIGHT of t_out: the encoder cannot see the future
# - deeper layers reach further back (dilations 1 → 8)
# - total span ≈ 60 frames = 300 ms — enough context to recognise a phoneme



# %% Cell 7 — load or briefly train an encoder so we can inspect it
# ============================================================
# We need a TRAINED encoder to look at meaningful embeddings.  Two options:
#   A) load the saved v1 encoder from bio_models/ (instant)
#   B) brief in-line SSL pretrain on the demo patient (~5 min)
# We try A first, fall back to B.

ENC_PATH = os.path.join('bio_models', f'{DEMO_PID}_ssl_encoder_low_beta.pt')

# build full demo dataset so we have train + test sentences in memory
def build_demo_dataset(pid, bands):
    raw = np.load(os.path.join(DUTCH_30_PATH, 'raw', f'{pid}_sEEG.npy'))
    if raw.ndim == 2 and raw.shape[0] < raw.shape[1]: raw = raw.T
    keep = np.asarray(pipeline.channel_masks[pid]['keep_indices'])
    raw = raw[:, keep]
    wd  = pipeline.split_result['word_segments_dict'][pid]
    mfa = load_mfa_alignments(pid)
    all_real = [i for i, s in enumerate(wd['sentence_list'])
                if isinstance(s, dict) and s.get('text')]
    test_ids = set(all_real[TEST_OFFSET::6])
    out = {'train': [], 'test': []}
    for sid in all_real:
        if sid not in mfa or not mfa[sid]: continue
        s = wd['sentence_list'][sid]
        s0, s1 = s['stim_start_idx'], s['stim_end_idx']
        if s1 > raw.shape[0]: continue
        X = extract_multiband(raw[s0:s1], EEG_SR, bands)
        if X.shape[0] < 30: continue
        split = 'test' if sid in test_ids else 'train'
        out[split].append({'X': torch.from_numpy(X), 'mfa': mfa[sid], 'sent_idx': sid})
    return out

print(f"Building demo dataset for {DEMO_PID}...")
ds_demo = build_demo_dataset(DEMO_PID, V1_BANDS)
n_in = ds_demo['train'][0]['X'].shape[1]
print(f"  n_in={n_in}  train={len(ds_demo['train'])}  test={len(ds_demo['test'])}")

if os.path.exists(ENC_PATH):
    print(f"\nLoading trained v1 encoder from {ENC_PATH}")
    ckpt = torch.load(ENC_PATH, map_location=DEVICE, weights_only=False)
    enc = CausalTCNEncoder(n_in).to(DEVICE)
    enc.load_state_dict(ckpt['enc'])
    if 'mu' in ckpt and 'sd' in ckpt:
        mu, sd = ckpt['mu'], ckpt['sd']
    else:
        Xall = torch.cat([s['X'] for s in ds_demo['train']], 0).numpy()
        mu, sd = Xall.mean(0), Xall.std(0); sd = np.where(sd < 1e-6, 1.0, sd)
    print(f"  ✓ loaded (n_in={ckpt['n_in']}, bands={ckpt.get('bands', '?')})")
else:
    print(f"\n{ENC_PATH} not found — brief in-line SSL training (~5 min)")
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    if DEVICE.type == 'cuda': torch.cuda.manual_seed_all(0)
    Xall = torch.cat([s['X'] for s in ds_demo['train']], 0).numpy()
    mu, sd = Xall.mean(0), Xall.std(0); sd = np.where(sd < 1e-6, 1.0, sd)
    enc = CausalTCNEncoder(n_in).to(DEVICE)
    head = nn.Sequential(nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.GELU(),
                         nn.Linear(HIDDEN_DIM, n_in)).to(DEVICE)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(head.parameters()),
                            lr=3e-4, weight_decay=1e-3)
    EPOCHS_DEMO = 30
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS_DEMO)
    mu_t = torch.from_numpy(mu.astype(np.float32))
    sd_t = torch.from_numpy(sd.astype(np.float32))
    losses = []
    rng = np.random.RandomState(0)
    for ep in range(EPOCHS_DEMO):
        enc.train(); head.train(); tot = 0; nb = 0
        for s in rng.permutation(ds_demo['train']):
            X = ((s['X'] - mu_t) / sd_t).unsqueeze(0).to(DEVICE)
            T = X.shape[1]
            mask = torch.zeros(1, T, dtype=torch.bool, device=DEVICE)
            for _ in range(max(1, int(0.15 * T) // 10)):
                start = rng.randint(0, max(1, T - 10))
                mask[0, start:start+10] = True
            h = enc(X, mask=mask)
            pred = head(h)
            if mask.sum() == 0: continue
            loss = F.mse_loss(pred[mask.unsqueeze(-1).expand_as(pred)],
                              X[mask.unsqueeze(-1).expand_as(X)])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        sch.step(); losses.append(tot/max(nb, 1))
        if ep == 0 or (ep + 1) % 5 == 0:
            print(f"    ep {ep+1:2d}/{EPOCHS_DEMO}  mse={losses[-1]:.4f}")

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(range(1, len(losses)+1), losses, marker='o', markersize=4)
    ax.set_xlabel('epoch'); ax.set_ylabel('masked-frame MSE')
    ax.set_title(f"in-line SSL training on {DEMO_PID} (brief, ~{EPOCHS_DEMO} ep)")
    ax.grid(alpha=0.3); plt.tight_layout(); plt.show()

# standardise the full demo dataset using mu/sd
mu_t = torch.from_numpy(mu.astype(np.float32))
sd_t = torch.from_numpy(sd.astype(np.float32))
for split in ('train', 'test'):
    for s in ds_demo[split]:
        s['X'] = (s['X'] - mu_t) / sd_t

# %% Cell 8 — what does the encoder output look like?
# ============================================================
# Run the demo sentence through the encoder.  Input: (T, 222) standardised
# features.  Output: (T, 128) embedding sequence — a 128-d vector at each
# 5-ms frame, encoding the past ~300 ms of HG + low_beta activity.

enc.eval()
# find our demo sentence in the standardised dataset
demo_X = None
for s in ds_demo['train']:
    if s['sent_idx'] == DEMO_SENT:
        demo_X = s['X']; break
with torch.no_grad():
    h = enc(demo_X.unsqueeze(0).to(DEVICE)).squeeze(0).cpu().numpy()
T, D = h.shape
print(f"  encoder output for sentence {DEMO_SENT}:  shape=(T={T}, D={D})")
print(f"  one 128-d embedding per 5 ms frame")

# heatmap of the embedding stream
fig, ax = plt.subplots(figsize=(12, 5))
vmax = np.percentile(np.abs(h), 99)
im = ax.imshow(h.T, aspect='auto', origin='lower',
               extent=[0, T*SHIFT_S, 0, D], cmap='RdBu_r', vmin=-vmax, vmax=vmax)
ax.set_xlabel('time (s)'); ax.set_ylabel('embedding dim (0-127)')
ax.set_title(f"{DEMO_PID} sentence {DEMO_SENT} — 128-d encoder embeddings over time")
# overlay phoneme boundaries + labels
for ph in mfa_all[DEMO_SENT]:
    ax.axvline(ph['start_s'], color='black', linewidth=0.5, alpha=0.4)
for ph in mfa_all[DEMO_SENT]:
    mid = (ph['start_s'] + ph['end_s']) / 2
    ax.text(mid, D * 1.02, ph['phone'], fontsize=9, ha='center')
plt.colorbar(im, ax=ax, fraction=0.02, label='activation')
plt.tight_layout(); plt.show()

# what to notice:
# - each row is a hidden unit's response over time
# - horizontal stripes = units that respond consistently across the sentence
# - patches/blobs = units that respond at specific moments
# - look for similarity between same-phoneme repeats (e.g. multiple /ɛ/'s
#   should produce similar embedding patterns)



# %% Cell 9 — per-phoneme averaging: collapsing time into phoneme-tokens
# ============================================================
# The classifier doesn't see frame-by-frame embeddings.  It sees ONE 128-d
# vector per phoneme, computed by averaging the encoder outputs over the
# frames that fall inside each phoneme's MFA interval.  This is the key
# step that turns a frame-level encoder output into a phoneme-level
# feature vector.

def time_to_frame(t_s):
    return int(round((t_s * EEG_SR - WIN_SAMP / 2) / SHIFT_SAMP))

phoneme_vecs = []
phoneme_labels = []
phoneme_segs = []
for ph in mfa_all[DEMO_SENT]:
    k_s = max(0, time_to_frame(ph['start_s']))
    k_e = min(T - 1, time_to_frame(ph['end_s']))
    if k_e < k_s: continue
    vec = h[k_s:k_e+1].mean(axis=0)
    phoneme_vecs.append(vec)
    phoneme_labels.append(ph['phone'])
    phoneme_segs.append((ph['start_s'], ph['end_s'], k_s, k_e))

phoneme_vecs = np.array(phoneme_vecs)
print(f"  collapsed {T} frames → {len(phoneme_vecs)} per-phoneme vectors")
print(f"  shape: {phoneme_vecs.shape}  (one 128-d vector per phoneme)")

# visualise the collapse: frame-level embeddings (top) → per-phoneme averages (bottom)
fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
vmax = np.percentile(np.abs(h), 99)
axes[0].imshow(h.T, aspect='auto', origin='lower',
               extent=[0, T*SHIFT_S, 0, D], cmap='RdBu_r', vmin=-vmax, vmax=vmax)
axes[0].set_ylabel('emb dim'); axes[0].set_title('frame-level embeddings (T, 128)')
for k_s_, k_e_ in [(seg[2], seg[3]) for seg in phoneme_segs]:
    axes[0].axvspan(k_s_*SHIFT_S, k_e_*SHIFT_S, color='black', alpha=0.05)

# bottom: per-phoneme averages tiled to fill their interval
ph_heat = np.zeros((D, T))
for i, (start_s, end_s, k_s, k_e) in enumerate(phoneme_segs):
    ph_heat[:, k_s:k_e+1] = phoneme_vecs[i][:, None]
axes[1].imshow(ph_heat, aspect='auto', origin='lower',
               extent=[0, T*SHIFT_S, 0, D], cmap='RdBu_r', vmin=-vmax, vmax=vmax)
axes[1].set_ylabel('emb dim'); axes[1].set_xlabel('time (s)')
axes[1].set_title('per-phoneme means (tiled to their MFA intervals)')
for ph in mfa_all[DEMO_SENT]:
    mid = (ph['start_s'] + ph['end_s']) / 2
    axes[1].text(mid, D + 5, ph['phone'], fontsize=9, ha='center')
    axes[1].axvline(ph['start_s'], color='black', linewidth=0.5, alpha=0.4)
plt.tight_layout(); plt.show()

# what to notice:
# - top: continuous time-varying activations
# - bottom: piecewise-constant blocks (one block per phoneme); same-phoneme
#   blocks should look similar across the sentence

# %% Cell 10 — what do the phoneme embeddings cluster like?  (PCA)
# ============================================================
# Collect per-phoneme means across ALL train sentences and project to 2D
# with PCA.  Colour by phoneme.  We expect to see at least some structure:
# vowels grouping with vowels, fricatives with fricatives, etc.  Encoder
# wasn't trained on phoneme labels (SSL is unsupervised) so any clustering
# is emergent.

def collect_phoneme_vectors(pid, ds, encoder):
    encoder.eval()
    vecs, labels = [], []
    with torch.no_grad():
        for s in ds['train']:
            X = s['X'].unsqueeze(0).to(DEVICE)
            h = encoder(X).squeeze(0).cpu().numpy()
            T = h.shape[0]
            for ph in s['mfa']:
                k_s = max(0, time_to_frame(ph['start_s']))
                k_e = min(T - 1, time_to_frame(ph['end_s']))
                if k_e < k_s: continue
                vecs.append(h[k_s:k_e+1].mean(axis=0))
                labels.append(ph['phone'])
    return np.array(vecs), np.array(labels)

print("Collecting per-phoneme embeddings across train sentences...")
V, L = collect_phoneme_vectors(DEMO_PID, ds_demo, enc)
print(f"  {len(V)} phoneme tokens, {len(set(L))} unique phonemes")

# keep the top-N most frequent phonemes for a readable plot
top_n = 12
counts = Counter(L)
top_labels = [ph for ph, _ in counts.most_common(top_n)]
mask = np.isin(L, top_labels)
V_top = V[mask]; L_top = L[mask]
print(f"  top-{top_n} phonemes:  {top_labels}")
print(f"  showing {len(V_top)} tokens")

# PCA to 2D
pca = PCA(n_components=2).fit(V)
V2 = pca.transform(V_top)
print(f"  PCA explained variance ratio:  PC1={pca.explained_variance_ratio_[0]:.2%}  "
      f"PC2={pca.explained_variance_ratio_[1]:.2%}")

fig, ax = plt.subplots(figsize=(10, 8))
# colour by manner-of-articulation for interpretability
manner = {
    **{v: 'V' for v in ['a','aː','ɑ','ɛ','eː','ɛi','i','iː','ɪ','o','oː',
                        'ɔ','ɔu','u','uː','y','yː','øː','œ','œy','ə','ɑu','ɛy']},
    **{c: 'P' for c in ['p','b','t','d','k','g']},
    **{c: 'F' for c in ['f','v','s','z','ʃ','ʒ','x','ɣ','h']},
    **{c: 'N' for c in ['m','n','ŋ','ɲ']},
    **{c: 'L' for c in ['l','r','j','w','ʋ']},
}
manner_color = {'V': 'C0', 'P': 'C3', 'F': 'C2', 'N': 'C1', 'L': 'C4'}
for ph in top_labels:
    sel = (L_top == ph)
    m   = manner.get(ph, '?')
    ax.scatter(V2[sel, 0], V2[sel, 1], s=30, alpha=0.5,
               color=manner_color.get(m, 'gray'), label=f"{ph} ({m})")
    # label cluster centroid
    if sel.sum() > 3:
        cx, cy = V2[sel, 0].mean(), V2[sel, 1].mean()
        ax.annotate(ph, (cx, cy), fontsize=14, fontweight='bold',
                    ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8))
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
ax.set_title(f"{DEMO_PID} per-phoneme encoder embeddings — PCA 2D\n"
             f"colour = manner (V=vowel, P=plosive, F=fricative, N=nasal, L=approximant)")
ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
ax.grid(alpha=0.3); plt.tight_layout(); plt.show()

# what to notice:
# - clusters of same-phoneme tokens (not perfect but visible structure)
# - vowels often cluster together; consonants in other regions
# - this structure was learned by SSL — the encoder never saw labels

# %% Cell 11 — fitting LDA and visualising decision regions
# ============================================================
# Linear Discriminant Analysis projects 128-d embeddings into a lower-dim
# space where classes are maximally separable.  For K=N phonemes, LDA can
# project to up to K-1 dimensions.  We project to 2D for visualisation.
# The shaded regions are LDA's decision regions in that 2D space.

scaler = StandardScaler().fit(V)
V_std  = scaler.transform(V)

lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
lda.fit(V_std, L)
print(f"  LDA fitted: {len(lda.classes_)} classes, "
      f"input dim {V_std.shape[1]} → decision space {len(lda.classes_)-1}-d")

# for visualisation we need a 2D projection; use a fresh LDA with svd solver
# (it exposes the discriminant directions explicitly)
lda_vis = LinearDiscriminantAnalysis(solver='svd', n_components=2)
lda_vis.fit(V_std, L)
V2_lda = lda_vis.transform(V_std)
mask = np.isin(L, top_labels)

fig, ax = plt.subplots(figsize=(10, 8))

# decision regions: classify a grid of points in 2D LDA space
# (this is approximate — true LDA decisions are in 128-d; we visualise
# the projection)
x_min, x_max = V2_lda[mask, 0].min() - 1, V2_lda[mask, 0].max() + 1
y_min, y_max = V2_lda[mask, 1].min() - 1, V2_lda[mask, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                     np.linspace(y_min, y_max, 300))
# train a quick 2D LDA just for visualising regions
lda2d = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
lda2d.fit(V2_lda[mask], L[mask])
Z = lda2d.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
# map predicted labels to ints for imshow
top_labels_arr = np.array(top_labels)
Z_int = np.searchsorted(top_labels_arr, Z)
Z_int[~np.isin(Z, top_labels_arr)] = -1
cmap = plt.cm.tab20
ax.imshow(Z_int, extent=[x_min, x_max, y_min, y_max],
          origin='lower', aspect='auto', cmap=cmap, alpha=0.18)

# scatter the actual phoneme tokens
for ph in top_labels:
    sel = (L[mask] == ph)
    m   = manner.get(ph, '?')
    ax.scatter(V2_lda[mask][sel, 0], V2_lda[mask][sel, 1], s=25, alpha=0.7,
               color=manner_color.get(m, 'gray'), edgecolors='white', linewidth=0.4)
    if sel.sum() > 3:
        cx, cy = V2_lda[mask][sel, 0].mean(), V2_lda[mask][sel, 1].mean()
        ax.annotate(ph, (cx, cy), fontsize=13, fontweight='bold', ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.85))

ax.set_xlabel('LDA dim 1'); ax.set_ylabel('LDA dim 2')
ax.set_title(f"{DEMO_PID} — LDA decision regions in 2D discriminant space\n"
             f"(shaded = classifier's choice region; dots = train phoneme tokens)")
ax.grid(alpha=0.3); plt.tight_layout(); plt.show()

# what to notice:
# - classes occupy distinct regions in LDA space
# - boundaries are linear in this projection (LDA is a linear classifier
#   in the embedding space, hence linear in any projection)
# - overlap = phonemes that are intrinsically hard to discriminate from
#   the encoder's 128-d output


# %% Cell 12 — per-frame log-probabilities for the demo sentence
# ============================================================
# At test time, we don't have phoneme boundaries.  We feed every frame's
# 128-d embedding through the (full-128-d) LDA and get K log-probabilities
# at every frame — a (T, K) matrix of "how likely is each phoneme right
# now?".  This is the input to Viterbi.

logp_raw = lda.predict_log_proba(scaler.transform(h))     # (T, K)
class_labels = list(lda.classes_)
K = len(class_labels)
print(f"  log-prob matrix:  shape=(T={T}, K={K} phonemes)")

# heatmap of log-probs over time
fig, ax = plt.subplots(figsize=(12, 7))
vmax = logp_raw.max(); vmin = max(logp_raw.min(), vmax - 10)
im = ax.imshow(logp_raw.T, aspect='auto', origin='lower',
               extent=[0, T*SHIFT_S, 0, K], cmap='inferno', vmin=vmin, vmax=vmax)
# label each class on the y-axis
ax.set_yticks(np.arange(K) + 0.5)
ax.set_yticklabels(class_labels, fontsize=7)
ax.set_xlabel('time (s)'); ax.set_ylabel('phoneme class')
ax.set_title(f"{DEMO_PID} sentence {DEMO_SENT} — per-frame LDA log P(phone | embedding)")
# overlay phoneme boundaries + true labels
for ph in mfa_all[DEMO_SENT]:
    ax.axvline(ph['start_s'], color='cyan', linewidth=0.5, alpha=0.6)
for ph in mfa_all[DEMO_SENT]:
    mid = (ph['start_s'] + ph['end_s']) / 2
    if ph['phone'] in class_labels:
        y = class_labels.index(ph['phone']) + 0.5
        ax.scatter(mid, y, marker='*', s=60, color='cyan',
                   edgecolors='black', linewidth=0.4, zorder=5)
plt.colorbar(im, ax=ax, fraction=0.02, label='log P')
plt.tight_layout(); plt.show()

# what to notice:
# - bright = high probability for that class at that time
# - cyan stars mark TRUE phoneme classes at their MFA midpoints —
#   when the bright region is near a star, the classifier is "looking
#   at the right answer" at that time

# %% Cell 13 — smoothing the log-probabilities
# ============================================================
# Per-frame log-probs are noisy.  We smooth each class's log-prob track
# along time with a 31-frame moving average (~155 ms).  This stabilises
# Viterbi's path choices and is critical to avoiding rapid spurious
# transitions.

logp_smooth = smooth_cols(logp_raw, SMOOTH_LOGP_W)

# show a single class's track before vs after smoothing
focus_class = 't'   # plosive
if focus_class not in class_labels:
    focus_class = class_labels[0]
ci = class_labels.index(focus_class)

fig, ax = plt.subplots(figsize=(12, 4))
t_frames = np.arange(T) * SHIFT_S
ax.plot(t_frames, logp_raw[:, ci],    color='gray',   alpha=0.6, label='raw log P', linewidth=0.7)
ax.plot(t_frames, logp_smooth[:, ci], color='C3',     linewidth=2,   label=f'31-frame smoothed')
# mark all occurrences of this phoneme in the true sentence
for ph in mfa_all[DEMO_SENT]:
    if ph['phone'] == focus_class:
        ax.axvspan(ph['start_s'], ph['end_s'], color='yellow', alpha=0.3,
                   label=f"gold /{focus_class}/" if 'gold' not in [l.get_label()
                                                                    for l in ax.get_lines()] else None)
ax.set_xlabel('time (s)'); ax.set_ylabel('log P')
ax.set_title(f"smoothing example — class /{focus_class}/")
ax.legend(loc='lower right'); ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()


# %% Cell 14 — Viterbi decoding: from log-probs to a phoneme sequence
# ============================================================
# Viterbi finds the most-probable phoneme sequence given the per-frame
# log-probs PLUS a "self-loop bonus" that rewards staying in the same
# phoneme.  Without a self-loop bonus, the decoder rapidly switches
# classes every few frames (each frame's most-likely class might differ).
# With it, the decoder produces piecewise-constant runs that better match
# how phonemes actually appear (each one lasts dozens of frames).

# auto-tune the self-loop bonus to predict TARGET_RATIO × val_target phonemes
val_target = int(len(mfa_all[DEMO_SENT]) * TARGET_RATIO)
bonus = auto_tune_bonus([logp_smooth], val_target, MIN_PRED_FRAMES)
print(f"  auto-tuned self-loop bonus = {bonus:.2f}")

# decode
path_argmax = np.argmax(logp_smooth, axis=1)              # naive per-frame
path_vit    = viterbi_decode(logp_smooth, bonus)          # with Viterbi smoothing

# visualise both paths over the log-prob heatmap
fig, ax = plt.subplots(figsize=(12, 7))
vmax = logp_smooth.max(); vmin = max(logp_smooth.min(), vmax - 10)
ax.imshow(logp_smooth.T, aspect='auto', origin='lower',
          extent=[0, T*SHIFT_S, 0, K], cmap='inferno', vmin=vmin, vmax=vmax,
          alpha=0.7)
ax.plot(t_frames, path_argmax + 0.5, color='cyan',   linewidth=0.7, alpha=0.7,
        label='per-frame argmax (no smoothing)')
ax.plot(t_frames, path_vit + 0.5,    color='lime',   linewidth=2,
        label=f'Viterbi path (bonus={bonus:.2f})')
ax.set_yticks(np.arange(K) + 0.5)
ax.set_yticklabels(class_labels, fontsize=7)
ax.set_xlabel('time (s)'); ax.set_ylabel('phoneme class')
ax.set_title(f"argmax (cyan) vs Viterbi (green)")
ax.legend(loc='upper right')
plt.tight_layout(); plt.show()

# %% Cell 15 — extracting predicted phoneme sequence + comparing to gold
# ============================================================
# Final step: collapse the Viterbi path into runs (consecutive same-class
# spans), keep runs of length ≥ MIN_PRED_FRAMES.  Each run becomes one
# predicted phoneme.  Then align predicted sequence to the gold sequence
# with Needleman-Wunsch and colour-code the alignment.

predictions = []
i = 0
while i < T:
    ci = path_vit[i]; j = i + 1
    while j < T and path_vit[j] == ci: j += 1
    if (j - i) >= MIN_PRED_FRAMES:
        predictions.append(class_labels[ci])
    i = j

gold = [ph['phone'] for ph in mfa_all[DEMO_SENT] if ph['phone'] in class_labels]
print(f"  gold sequence ({len(gold)} phonemes):  {' '.join(gold)}")
print(f"  pred sequence ({len(predictions)} phonemes):  {' '.join(predictions)}")

aligned = needleman_wunsch(gold, predictions)

# colour-coded alignment plot
fig, ax = plt.subplots(figsize=(14, 3))
ax.set_xlim(0, len(aligned)); ax.set_ylim(0, 2)
ax.set_yticks([0.5, 1.5]); ax.set_yticklabels(['pred', 'gold'])
ax.set_xticks([]); ax.spines['left'].set_visible(False); ax.spines['bottom'].set_visible(False)
for i, (g, p) in enumerate(aligned):
    if g is not None and p is not None and g == p:
        c = '#a6e3a1'    # green — match
    elif g is not None and p is not None:
        c = '#f5c2c0'    # red — sub
    elif g is not None:
        c = '#dddddd'    # gray — del
    else:
        c = '#ffd966'    # yellow — ins
    ax.add_patch(plt.Rectangle((i, 1), 1, 1, color=c, ec='white'))
    ax.add_patch(plt.Rectangle((i, 0), 1, 1, color=c, ec='white'))
    if g is not None:
        ax.text(i + 0.5, 1.5, g, ha='center', va='center', fontsize=11, fontweight='bold')
    if p is not None:
        ax.text(i + 0.5, 0.5, p, ha='center', va='center', fontsize=11)
# legend
from matplotlib.patches import Patch
ax.legend(handles=[
    Patch(facecolor='#a6e3a1', label='match'),
    Patch(facecolor='#f5c2c0', label='substitution'),
    Patch(facecolor='#dddddd', label='deletion (gold w/o pred)'),
    Patch(facecolor='#ffd966', label='insertion (pred w/o gold)'),
], loc='upper right', ncol=4, fontsize=9)
ax.set_title(f"{DEMO_PID} sentence {DEMO_SENT} — NW-aligned gold vs prediction")
plt.tight_layout(); plt.show()

n_match = sum(1 for g, p in aligned if g is not None and p is not None and g == p)
n_sub   = sum(1 for g, p in aligned if g is not None and p is not None and g != p)
n_del   = sum(1 for g, p in aligned if g is not None and p is None)
n_ins   = sum(1 for g, p in aligned if g is None and p is not None)
print(f"\n  matches:       {n_match} ({100*n_match/max(len(gold),1):.0f}% of gold)")
print(f"  substitutions: {n_sub}")
print(f"  deletions:     {n_del}")
print(f"  insertions:    {n_ins}")

# %% Cell 16 — recap: the full pipeline in one diagram
# ============================================================
# Stitch everything together: every transformation visualised vertically
# for the one demo sentence.

fig, axes = plt.subplots(5, 1, figsize=(12, 12), sharex=True,
                          gridspec_kw={'height_ratios': [1, 1, 2, 1.5, 0.5]})

# 1. raw EEG (one channel)
axes[0].plot(np.arange(eeg_slice.shape[0])/EEG_SR, eeg_slice[:, ch_demo],
             color='gray', linewidth=0.5)
axes[0].set_ylabel(f'raw EEG\nch {ch_demo}')
axes[0].set_title(f"v1 pipeline trace — {DEMO_PID} sentence {DEMO_SENT}", fontsize=11)
axes[0].grid(alpha=0.3)

# 2. multiband amplitude features (HG + low_beta) — heat
n_ch = X_multi.shape[1] // 2
im2 = axes[1].imshow(X_multi.T, aspect='auto', origin='lower',
                     extent=[0, X_multi.shape[0]*SHIFT_S, 0, X_multi.shape[1]],
                     cmap='viridis')
axes[1].axhline(n_ch, color='red', linewidth=1, alpha=0.7)
axes[1].set_ylabel('multi-band\nfeatures')
axes[1].text(0.005, n_ch*0.5,  'HG amp',       ha='left', va='center', color='white', fontsize=8)
axes[1].text(0.005, n_ch*1.5,  'low_beta amp', ha='left', va='center', color='white', fontsize=8)

# 3. encoder embeddings — heat
vmax = np.percentile(np.abs(h), 99)
axes[2].imshow(h.T, aspect='auto', origin='lower',
               extent=[0, T*SHIFT_S, 0, D], cmap='RdBu_r', vmin=-vmax, vmax=vmax)
axes[2].set_ylabel('encoder\nembeddings\n(128-d)')

# 4. LDA log-probs — heat
vmax = logp_smooth.max(); vmin = max(logp_smooth.min(), vmax - 10)
axes[3].imshow(logp_smooth.T, aspect='auto', origin='lower',
               extent=[0, T*SHIFT_S, 0, K], cmap='inferno', vmin=vmin, vmax=vmax)
axes[3].plot(np.arange(T)*SHIFT_S, path_vit + 0.5,
             color='lime', linewidth=1.5, label='Viterbi')
axes[3].set_ylabel('LDA log P\n+ Viterbi path')
axes[3].legend(loc='upper right', fontsize=8)

# 5. predicted vs gold spans
axes[4].set_ylim(0, 2); axes[4].set_yticks([0.5, 1.5])
axes[4].set_yticklabels(['pred', 'gold'])
# gold
for ph in mfa_all[DEMO_SENT]:
    axes[4].add_patch(plt.Rectangle((ph['start_s'], 1), ph['end_s']-ph['start_s'], 1,
                                    facecolor='#cccccc', edgecolor='white'))
    mid = (ph['start_s']+ph['end_s'])/2
    axes[4].text(mid, 1.5, ph['phone'], ha='center', va='center', fontsize=8)
# pred: walk through Viterbi path again with timing
i = 0
while i < T:
    ci = path_vit[i]; j = i + 1
    while j < T and path_vit[j] == ci: j += 1
    if (j - i) >= MIN_PRED_FRAMES:
        start = i * SHIFT_S; end = (j - 1) * SHIFT_S
        ph_pred = class_labels[ci]
        # colour match vs miss
        mid_t = (start + end) / 2
        gold_at_t = next((g['phone'] for g in mfa_all[DEMO_SENT]
                          if g['start_s'] <= mid_t <= g['end_s']), None)
        color = '#a6e3a1' if gold_at_t == ph_pred else '#f5c2c0'
        axes[4].add_patch(plt.Rectangle((start, 0), end - start, 1,
                                        facecolor=color, edgecolor='white'))
        axes[4].text((start+end)/2, 0.5, ph_pred, ha='center', va='center', fontsize=8)
    i = j
axes[4].set_xlabel('time (s)')
axes[4].set_xlim(0, T*SHIFT_S)

plt.tight_layout(); plt.show()

print("\nPipeline trace complete.  Top-to-bottom:")
print("  1) raw EEG: 1024 Hz multi-channel voltages")
print("  2) features: per-band amplitude envelopes at 200 Hz, two bands concatenated")
print("  3) encoder: causal TCN, outputs 128-d frame embeddings (300 ms past context)")
print("  4) LDA + Viterbi: classifies + smooths into a phoneme sequence")
print("  5) compared to gold (grey = gold; green = match; red = wrong prediction)")

# %% Cell — visualise what SSL training did
# ============================================================
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

# pick a patient with a trained v1 encoder loaded
pid = 'P22'
enc_trained = encoders[pid]   # from your v1 training

# build a random-init encoder of the same shape — just initialise fresh weights
import torch
n_in = enc_trained.proj_in.in_channels
enc_random = CausalTCNEncoder(n_in).to(DEVICE)
enc_random.eval()
enc_trained.eval()

# extract embeddings from each on the same train data
ds = datasets[pid]
def extract(enc):
    vecs, labels = [], []
    with torch.no_grad():
        for s in ds['train']:
            h = enc(s['X'].unsqueeze(0).to(DEVICE)).squeeze(0).cpu().numpy()
            for ph in s['mfa']:
                k_s = max(0, time_to_frame(ph['start_s']))
                k_e = min(h.shape[0]-1, time_to_frame(ph['end_s']))
                if k_e >= k_s:
                    vecs.append(h[k_s:k_e+1].mean(axis=0))
                    labels.append(ph['phone'])
    return np.array(vecs), np.array(labels)

V_random,  L_random  = extract(enc_random)
V_trained, L_trained = extract(enc_trained)

# project both to 2D PCA for visual comparison (use the trained PCA basis
# for both, so projections are comparable)
pca = PCA(n_components=2).fit(V_trained)
V2_random  = pca.transform(V_random)
V2_trained = pca.transform(V_trained)

# pick 6 most frequent phonemes to colour
from collections import Counter
top6 = [ph for ph, _ in Counter(L_trained).most_common(6)]
cmap = plt.cm.tab10

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, V2, L, title in [
    (axes[0], V2_random,  L_random,  "random-init encoder (no SSL)"),
    (axes[1], V2_trained, L_trained, "v1 encoder (SSL-pretrained)"),
]:
    for i, ph in enumerate(top6):
        sel = L == ph
        ax.scatter(V2[sel, 0], V2[sel, 1], s=20, alpha=0.5,
                   color=cmap(i), label=ph)
    ax.set_title(title); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_xlabel('PC1'); ax.set_ylabel('PC2')

fig.suptitle(f"{pid} — per-phoneme embeddings projected to 2D PCA")
plt.tight_layout(); plt.show()
