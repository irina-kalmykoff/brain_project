# Converted from silence_vs_speech_pretraining.ipynb

# ============================================================
# Speech-vs-non-speech detector — pretraining notebook
# Two modes: per-patient + cross-patient (per-pid projections,
# shared BiLSTM + head). Save best model to bio_models/.
# ============================================================
import os, pickle, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import butter, sosfiltfilt, iirnotch, tf2sos, hilbert
from collections import defaultdict

from dutch_30_pipeline import Dutch30Pipeline
from dutch_30_feature_extractor import Dutch30FeatureExtractor
from dataset_config import Dutch30Config
from config import DUTCH_30_PATH
from run_pipeline import load_mfa_alignments

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"DEVICE = {DEVICE}")
if DEVICE == 'cuda':
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  free: {torch.cuda.mem_get_info()[0]/1e9:.1f} GB")

# Signal processing constants (Butterworth path, no boxcar)
EEG_SR     = 1024
HG_LOW     = 70
HG_HIGH    = 170
NOTCH_HZ   = [50, 150]
LP_CUT_HZ  = 10.0     # envelope smoother
N_BUTTER   = 4
SHIFT_MS   = 5
SHIFT_SAMP = int(EEG_SR * SHIFT_MS / 1000)
FRAME_HZ   = int(1000 / SHIFT_MS)   # 200 Hz

# Speech-label extension: include up to 200 ms before phoneme onset
PRE_ONSET_MS = 200
PRE_ONSET_FRAMES = int(PRE_ONSET_MS / SHIFT_MS)

TARGET_PIDS = [f'P{i:02d}' for i in range(21, 31)]
# → ['P21', 'P22', 'P23', 'P24', 'P25', 'P26', 'P27', 'P28', 'P29', 'P30']

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
    env = np.maximum(env, 0)                       # ← clamp filter ringing
    out = env[::SHIFT_SAMP].astype(np.float32)
    return np.log1p(out)


def stack_context(X, K=5):
    T, C = X.shape
    pad = np.zeros((K, C), dtype=X.dtype)
    Xp = np.vstack([pad, X, pad])
    cols = [Xp[k:k + T] for k in range(2 * K + 1)]
    return np.concatenate(cols, axis=1)

def build_speech_labels(mfa_phones, n_frames, pre_onset_frames=PRE_ONSET_FRAMES):
    """For each frame, return 1 if it lies inside any phoneme interval OR
       within `pre_onset_frames` before the start of one. Else 0."""
    label = np.zeros(n_frames, dtype=np.int64)
    for ph in mfa_phones:
        k_start = int(np.ceil(ph['start_s'] * FRAME_HZ))
        k_end   = int(np.floor(ph['end_s']  * FRAME_HZ))
        k_pre   = max(0, k_start - pre_onset_frames)
        k_start = max(0, k_start); k_end = min(n_frames - 1, k_end)
        if k_end < k_pre: continue
        label[k_pre:k_end + 1] = 1
    return label


def build_speech_dataset(pid, pipeline, channel_mask=None):
    """Returns dict with per-sentence frame features + speech labels,
       split into train/test (every-6th-sentence fallback)."""
    raw_eeg = np.load(os.path.join(DUTCH_30_PATH, 'raw', f'{pid}_sEEG.npy'))
    if raw_eeg.ndim == 2 and raw_eeg.shape[0] < raw_eeg.shape[1]:
        raw_eeg = raw_eeg.T
    if channel_mask is not None:
        raw_eeg = raw_eeg[:, channel_mask]

    wd = pipeline.split_result['word_segments_dict'][pid]
    mfa = load_mfa_alignments(pid)

    all_real = [i for i, s in enumerate(wd['sentence_list'])
                if isinstance(s, dict) and s.get('text')]
    test_sent_ids = set(all_real[::6])

    sents = {'train': [], 'test': []}
    n_used = n_skip = 0
    for sent_idx in all_real:
        if sent_idx not in mfa or not mfa[sent_idx]:
            n_skip += 1; continue
        s = wd['sentence_list'][sent_idx]
        s0, s1 = s['stim_start_idx'], s['stim_end_idx']
        if s1 > raw_eeg.shape[0]:
            n_skip += 1; continue
        X = extract_hg_frames(raw_eeg[s0:s1])
        if X.shape[0] < 30:
            n_skip += 1; continue
        labels = build_speech_labels(mfa[sent_idx], X.shape[0])
        Xs = stack_context(X, K=5)
        split = 'test' if sent_idx in test_sent_ids else 'train'
        sents[split].append({
            'X':     torch.from_numpy(Xs).float(),
            'y':     torch.from_numpy(labels),
            'pid':   pid,
            'sent_idx': sent_idx,
        })
        n_used += 1

    n_in = sents['train'][0]['X'].shape[1] if sents['train'] else 0
    n_speech = sum(int(s['y'].sum()) for s in sents['train'])
    n_total  = sum(int(s['y'].numel()) for s in sents['train'])
    print(f"  [{pid}] used={n_used} skipped={n_skip}  n_in={n_in}  "
          f"train: {len(sents['train'])} sents, "
          f"speech_frac={n_speech/max(n_total,1):.2%}")
    return sents


# Build pipeline + datasets
config = Dutch30Config()
extractor = Dutch30FeatureExtractor(config=config)
pipeline = Dutch30Pipeline(extractor, config=config, use_wav2vec=False)
pipeline.step1_load_dutch30_data(patient_ids=TARGET_PIDS)
pipeline.step2_split_by_instances(train_fraction=0.8)

datasets = {}
for pid in TARGET_PIDS:
    print(f"\nBuilding {pid}...")
    datasets[pid] = build_speech_dataset(pid, pipeline)

class SpeechDetector(nn.Module):
    """Per-patient model: simple BiLSTM + binary head."""
    def __init__(self, n_in, lstm_hidden=64, lstm_layers=2, dropout=0.3):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(n_in, lstm_hidden * 2), nn.GELU(), nn.Dropout(dropout))
        self.lstm = nn.LSTM(lstm_hidden * 2, lstm_hidden,
                            num_layers=lstm_layers,
                            dropout=dropout if lstm_layers > 1 else 0.0,
                            bidirectional=True, batch_first=False)
        self.head = nn.Linear(lstm_hidden * 2, 2)

    def forward(self, x):
        h = self.proj(x).unsqueeze(1)
        h, _ = self.lstm(h)
        return self.head(h.squeeze(1))


class CrossPatientSpeechDetector(nn.Module):
    """Per-patient input projection feeds into a SHARED BiLSTM + head."""
    def __init__(self, n_in_per_pid, embed_dim=128,
                 lstm_hidden=64, lstm_layers=2, dropout=0.3):
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

# Run this once before Cell 6
print("Rebuilding datasets...")
datasets = {}
for pid in TARGET_PIDS:
    datasets[pid] = build_speech_dataset(pid, pipeline)

for pid in TARGET_PIDS:
    for split in ('train', 'test'):
        for s in datasets[pid][split]:
            assert not torch.isnan(s['X']).any(), f"NaN in {pid} {split} sent {s['sent_idx']}"
            assert not torch.isinf(s['X']).any(), f"Inf in {pid} {split} sent {s['sent_idx']}"
print("No NaN/Inf in any frame.")

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
        s['X'] = s['X'].to(device); s['y'] = s['y'].to(device)


def eval_speech(model, sents_te, pid=None):
    model.eval()
    correct = total = 0
    n_speech_correct = n_speech = 0
    with torch.no_grad():
        for s in sents_te:
            logits = model(s['X'], s['pid']) if pid is None else model(s['X'])
            pred = logits.argmax(-1)
            y = s['y']
            correct += (pred == y).sum().item(); total += y.numel()
            n_speech += (y == 1).sum().item()
            n_speech_correct += ((pred == 1) & (y == 1)).sum().item()
    acc = correct / max(total, 1)
    recall = n_speech_correct / max(n_speech, 1)
    return acc, recall

def train_per_patient(pid, sents_tr, sents_te, epochs=30, lr=1e-3, wd=1e-3):
    n_in = sents_tr[0]['X'].shape[1]
    model = SpeechDetector(n_in, lstm_hidden=128, dropout=0.2).to(DEVICE)
    mu, sd = fit_mu_sd(sents_tr)
    standardize(sents_tr, mu, sd); standardize(sents_te, mu, sd)
    to_device(sents_tr, DEVICE); to_device(sents_te, DEVICE)

    # Class weights for inverse-frequency balanced CE
    all_y = torch.cat([s['y'] for s in sents_tr])
    n0 = (all_y == 0).sum().item(); n1 = (all_y == 1).sum().item()
    cw = torch.tensor([n1 / (n0 + n1), n0 / (n0 + n1)],
                      dtype=torch.float32, device=DEVICE)
    print(f"  [{pid}] class weights: non-speech={cw[0]:.3f}  speech={cw[1]:.3f}  "
          f"(n0={n0}  n1={n1})")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    rng = np.random.default_rng(0)
    best_acc, best_state = 0.0, None

    for ep in range(epochs):
        model.train()
        perm = rng.permutation(len(sents_tr))
        total_loss = 0.0
        for i in perm:
            s = sents_tr[i]
            opt.zero_grad()
            logits = model(s['X'])
            loss = F.cross_entropy(logits, s['y'], weight=cw)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); total_loss += float(loss.item())
        sched.step()
        if (ep + 1) % 5 == 0 or ep == 0:
            acc, recall = eval_speech(model, sents_te, pid=pid)
            if acc > best_acc:
                best_acc = acc
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
            print(f"  [{pid}] ep{ep+1:3d}  loss={total_loss/len(sents_tr):.3f}  "
                  f"test_acc={acc:.3f}  speech_recall={recall:.3f}  best={best_acc:.3f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return {'model': model, 'mu': mu, 'sd': sd, 'best_acc': best_acc,
            'n_in': n_in}

def train_cross_patient(datasets, epochs=30, lr=1e-3, wd=1e-3, embed_dim=128):
    n_in_per_pid = {pid: datasets[pid]['train'][0]['X'].shape[1]
                    for pid in datasets if datasets[pid]['train']}
    model = CrossPatientSpeechDetector(n_in_per_pid, embed_dim=embed_dim,
                                       lstm_hidden=128, dropout=0.2).to(DEVICE)

    mu_sd = {}
    sents_tr_all, sents_te_all = [], []
    for pid in n_in_per_pid:
        mu, sd = fit_mu_sd(datasets[pid]['train'])
        mu_sd[pid] = (mu, sd)
        standardize(datasets[pid]['train'], mu, sd)
        standardize(datasets[pid]['test'],  mu, sd)
        to_device(datasets[pid]['train'], DEVICE)
        to_device(datasets[pid]['test'],  DEVICE)
        sents_tr_all.extend(datasets[pid]['train'])
        sents_te_all.extend(datasets[pid]['test'])

    # Pooled class weights across all patients
    all_y = torch.cat([s['y'] for s in sents_tr_all])
    n0 = (all_y == 0).sum().item(); n1 = (all_y == 1).sum().item()
    cw = torch.tensor([n1 / (n0 + n1), n0 / (n0 + n1)],
                      dtype=torch.float32, device=DEVICE)
    print(f"  [CROSS] class weights: non-speech={cw[0]:.3f}  speech={cw[1]:.3f}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    rng = np.random.default_rng(0)
    best_acc, best_state = 0.0, None

    for ep in range(epochs):
        model.train()
        perm = rng.permutation(len(sents_tr_all))
        total_loss = 0.0
        for i in perm:
            s = sents_tr_all[i]
            opt.zero_grad()
            logits = model(s['X'], s['pid'])
            loss = F.cross_entropy(logits, s['y'], weight=cw)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); total_loss += float(loss.item())
        sched.step()
        if (ep + 1) % 5 == 0 or ep == 0:
            acc, recall = eval_speech(model, sents_te_all, pid=None)
            if acc > best_acc:
                best_acc = acc
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
            print(f"  [CROSS] ep{ep+1:3d}  loss={total_loss/len(sents_tr_all):.3f}  "
                  f"test_acc={acc:.3f}  speech_recall={recall:.3f}  best={best_acc:.3f}")
            for pid in n_in_per_pid:
                a, r = eval_speech(model, datasets[pid]['test'], pid=None)
                print(f"      {pid}  acc={a:.3f}  recall={r:.3f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return {'model': model, 'mu_sd': mu_sd, 'best_acc': best_acc,
            'n_in_per_pid': n_in_per_pid}


def train_cross_patient(datasets, epochs=30, lr=3e-4, wd=1e-3, embed_dim=128):
    # Build per-pid mu/sd and concat train pool
    n_in_per_pid = {pid: datasets[pid]['train'][0]['X'].shape[1]
                    for pid in datasets if datasets[pid]['train']}
    model = CrossPatientSpeechDetector(n_in_per_pid, embed_dim=embed_dim).to(DEVICE)

    mu_sd = {}
    sents_tr_all, sents_te_all = [], []
    for pid in n_in_per_pid:
        mu, sd = fit_mu_sd(datasets[pid]['train'])
        mu_sd[pid] = (mu, sd)
        standardize(datasets[pid]['train'], mu, sd)
        standardize(datasets[pid]['test'],  mu, sd)
        to_device(datasets[pid]['train'], DEVICE)
        to_device(datasets[pid]['test'],  DEVICE)
        sents_tr_all.extend(datasets[pid]['train'])
        sents_te_all.extend(datasets[pid]['test'])

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    rng = np.random.default_rng(0)
    best_acc, best_state = 0.0, None

    for ep in range(epochs):
        model.train()
        perm = rng.permutation(len(sents_tr_all))
        total_loss = 0.0
        for i in perm:
            s = sents_tr_all[i]
            opt.zero_grad()
            logits = model(s['X'], s['pid'])
            loss = F.cross_entropy(logits, s['y'])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); total_loss += float(loss.item())
        sched.step()
        if (ep + 1) % 5 == 0 or ep == 0:
            acc, recall = eval_speech(model, sents_te_all, pid=None)
            if acc > best_acc:
                best_acc = acc
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
            print(f"  [CROSS] ep{ep+1:3d}  loss={total_loss/len(sents_tr_all):.3f}  "
                  f"test_acc={acc:.3f}  speech_recall={recall:.3f}  best={best_acc:.3f}")
            # also report per-patient breakdown
            for pid in n_in_per_pid:
                a, r = eval_speech(model, datasets[pid]['test'], pid=None)
                print(f"      {pid}  acc={a:.3f}  recall={r:.3f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return {'model': model, 'mu_sd': mu_sd, 'best_acc': best_acc,
            'n_in_per_pid': n_in_per_pid}

# --- Per-patient ---
per_patient = {}
print("\n========== PER-PATIENT TRAINING ==========")
for pid in TARGET_PIDS:
    if not datasets[pid]['train']:
        print(f"  [{pid}] skip (no train data)"); continue
    print(f"\n--- {pid} ---")
    t0 = time.time()
    per_patient[pid] = train_per_patient(
        pid, datasets[pid]['train'], datasets[pid]['test'], epochs=30)
    print(f"  done in {time.time()-t0:.1f}s")

# Re-build datasets for cross-patient (the per-patient pass standardized them in place)
print("\nRebuilding datasets for cross-patient...")
datasets = {}
for pid in TARGET_PIDS:
    datasets[pid] = build_speech_dataset(pid, pipeline)

# --- Cross-patient ---
print("\n========== CROSS-PATIENT TRAINING ==========")
t0 = time.time()
cross = train_cross_patient(datasets, epochs=30)
print(f"  done in {time.time()-t0:.1f}s")

# --- Save ---
os.makedirs('bio_models', exist_ok=True)

for pid, res in per_patient.items():
    path = f'bio_models/{pid}_speech_detector.pt'
    torch.save({
        'kind': 'per_patient',
        'pid': pid,
        'state_dict': res['model'].state_dict(),
        'mu': res['mu'], 'sd': res['sd'],
        'n_in': res['n_in'],
        'best_acc': res['best_acc'],
        'arch': dict(lstm_hidden=64, lstm_layers=2, dropout=0.3),
        'pre_onset_ms': PRE_ONSET_MS,
        'signal_processing': dict(eeg_sr=EEG_SR, hg_low=HG_LOW, hg_high=HG_HIGH,
                                  notch_hz=NOTCH_HZ, lp_cut_hz=LP_CUT_HZ,
                                  shift_ms=SHIFT_MS, n_butter=N_BUTTER),
    }, path)
    print(f"saved {path}")

torch.save({
    'kind': 'cross_patient',
    'state_dict': cross['model'].state_dict(),
    'mu_sd': cross['mu_sd'],
    'n_in_per_pid': cross['n_in_per_pid'],
    'best_acc': cross['best_acc'],
    'arch': dict(embed_dim=128, lstm_hidden=64, lstm_layers=2, dropout=0.3),
    'pre_onset_ms': PRE_ONSET_MS,
    'signal_processing': dict(eeg_sr=EEG_SR, hg_low=HG_LOW, hg_high=HG_HIGH,
                              notch_hz=NOTCH_HZ, lp_cut_hz=LP_CUT_HZ,
                              shift_ms=SHIFT_MS, n_butter=N_BUTTER),
}, 'bio_models/speech_detector_cross_patient.pt')
print("saved bio_models/speech_detector_cross_patient.pt")

# Summary
print("\n========== SUMMARY ==========")
print(f"Cross-patient best test_acc: {cross['best_acc']:.3f}")
for pid in TARGET_PIDS:
    if pid in per_patient:
        print(f"{pid} per-patient best test_acc: {per_patient[pid]['best_acc']:.3f}")
