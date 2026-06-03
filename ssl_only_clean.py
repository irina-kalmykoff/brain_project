# Standalone clean SSL pipeline — HG-only features
# =====================================================================
# This is the trimmed, single-path version of ssl_lda_frames_clean.py:
# high-gamma-only (70-170 Hz) SSL encoder -> LDA on per-phoneme embeddings
# -> scalar self-loop Viterbi -> NW metrics + shift-input permutation test.
#
# It LOADS the canonical per-patient HG-only encoders from
#   bio_models/{pid}_ssl_encoder.pt
# and only trains an encoder if its checkpoint is missing.
# Cells are marked with "# %%" — paste into Jupyter top-to-bottom.

# %% Cell 1 — imports, hyperparameters, pipeline init
# ============================================================
import os, time, math, json, random, pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.signal as sps
import scipy.stats as ss

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from collections import Counter, defaultdict

from dutch_30_pipeline import Dutch30Pipeline
from dutch_30_feature_extractor import Dutch30FeatureExtractor
from dataset_config import Dutch30Config
from config import DUTCH_30_PATH
from run_pipeline import load_mfa_alignments
from e2e_brain_decoder import edit_distance, show_matched_sequences_with_times

# ── vendored from LDA_on_frames_clean (kept inline so this file is import-safe;
#    LDA_on_frames_clean.py runs a top-level driver on import and isn't importable) ──
TEST_OFFSET, VAL_FRAC = 0, 0.15
EEG_SR          = 1024
WIN_S, SHIFT_S  = 0.015, 0.005
WIN_SAMP        = int(EEG_SR * WIN_S)
SHIFT_SAMP      = int(EEG_SR * SHIFT_S)
SELF_LOOP_BONUS = None        # None => auto-tune on val

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
    """Viterbi where staying in the same class earns +self_bonus per frame."""
    T, K = logp.shape
    if T == 0: return np.zeros(0, dtype=np.int32)
    delta = np.empty((T, K)); bptr = np.empty((T, K), dtype=np.int32)
    delta[0] = logp[0]; all_k = np.arange(K)
    for t in range(1, T):
        prev = delta[t - 1]
        order = np.argsort(prev); idx1, idx2 = order[-1], order[-2]
        best_switch = np.full(K, prev[idx1]); best_switch[idx1] = prev[idx2]
        bptr_switch = np.full(K, idx1);       bptr_switch[idx1] = idx2
        stay = prev + self_bonus
        choose_stay = stay >= best_switch
        delta[t] = logp[t] + np.where(choose_stay, stay, best_switch)
        bptr[t]  = np.where(choose_stay, all_k, bptr_switch)
    path = np.empty(T, dtype=np.int32); path[-1] = delta[-1].argmax()
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
    """Binary search: smallest bonus that brings total segment count <= target."""
    for _ in range(n_iter):
        mid = (lo + hi) / 2
        cnt = sum(count_runs(viterbi_decode(lp, mid), min_pred_frames) for lp in logp_list)
        if cnt > target_count: lo = mid
        else:                  hi = mid
    return (lo + hi) / 2

def needleman_wunsch(gold, pred, match=1, mismatch=-1, gap=-1):
    """Global sequence alignment. Returns (g, p) pairs; None = gap."""
    n, m = len(gold), len(pred)
    if n == 0: return [(None, p) for p in pred]
    if m == 0: return [(g, None) for g in gold]
    S = np.zeros((n + 1, m + 1), dtype=np.float32)
    S[:, 0] = np.arange(n + 1) * gap; S[0, :] = np.arange(m + 1) * gap
    BT = np.zeros((n + 1, m + 1), dtype=np.int8)
    BT[:, 0] = 1; BT[0, :] = 2; BT[0, 0] = 0
    for i in range(1, n + 1):
        gi = gold[i - 1]
        for j in range(1, m + 1):
            d = S[i - 1, j - 1] + (match if gi == pred[j - 1] else mismatch)
            u = S[i - 1, j] + gap; l = S[i, j - 1] + gap
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
    """Group flat predictions/gold by sentence_id, preserving order."""
    gold_per = defaultdict(list)
    for lbl, sid in zip(out['true_labels'], out['true_sentence_ids']):
        gold_per[int(sid)].append(lbl)
    pred_per = defaultdict(list)
    for lbl, sid in zip(out['predictions'], out['pred_sentence_ids']):
        pred_per[int(sid)].append(lbl)
    return gold_per, pred_per

def nw_metrics(out, manner_fn=None, n_perm=500, seed=0):
    """NW-alignment metrics: match_rate, z_match (permutation), n2/n3/n4 runs."""
    gold_per, pred_per = gather_sequences(out)
    sent_alignments = {}; all_gold, all_pred = [], []
    for sid in set(gold_per) | set(pred_per):
        g, p = gold_per.get(sid, []), pred_per.get(sid, [])
        sent_alignments[sid] = needleman_wunsch(g, p); all_gold += g; all_pred += p
    all_aligned = [pair for a in sent_alignments.values() for pair in a]
    n_match = sum(1 for g, p in all_aligned if g is not None and p is not None and g == p)
    n_sub   = sum(1 for g, p in all_aligned if g is not None and p is not None and g != p)
    n_del   = sum(1 for g, p in all_aligned if g is not None and p is None)
    n_ins   = sum(1 for g, p in all_aligned if g is None and p is not None)
    n_gold  = sum(1 for g, p in all_aligned if g is not None)
    match_rate = n_match / max(n_gold, 1)
    per_nw     = (n_sub + n_del + n_ins) / max(n_gold, 1)
    rng = np.random.default_rng(seed); null_match_rates = []; pred_pool = all_pred[:]
    for _ in range(n_perm):
        rng.shuffle(pred_pool); cur = 0; nm = 0
        for sid in sent_alignments:
            g = gold_per.get(sid, []); p_len = len(pred_per.get(sid, []))
            p_shuf = pred_pool[cur:cur + p_len]; cur += p_len
            a = needleman_wunsch(g, p_shuf)
            nm += sum(1 for x, y in a if x is not None and y is not None and x == y)
        null_match_rates.append(nm / max(n_gold, 1))
    null_mean = np.mean(null_match_rates); null_std = np.std(null_match_rates) + 1e-9
    z_match = (match_rate - null_mean) / null_std
    def runs_in(al, min_len):
        runs = 0; cur_run = 0
        for g, p in al:
            if g is not None and p is not None and g == p: cur_run += 1
            else:
                if cur_run >= min_len: runs += 1
                cur_run = 0
        if cur_run >= min_len: runs += 1
        return runs
    n2 = sum(runs_in(a, 2) for a in sent_alignments.values())
    n3 = sum(runs_in(a, 3) for a in sent_alignments.values())
    n4 = sum(runs_in(a, 4) for a in sent_alignments.values())
    return dict(n_gold=n_gold, n_match=n_match, n_sub=n_sub, n_del=n_del, n_ins=n_ins,
                match_rate=match_rate, per_nw=per_nw,
                z_match=z_match, null_mean=null_mean, null_std=null_std,
                n2=n2, n3=n3, n4=n4)

# ── HG-only hyperparameters ──────────────────────────────────────────
BANDS           = [(70, 170)]   # high-gamma only
SSL_EPOCHS      = 80            # HG-only cohort optimum (see CLAUDE.md)
TARGET_RATIO    = 1.7
MIN_PRED_FRAMES = 3
SMOOTH_LOGP_W   = 31
MN_FRAMES       = 0
MX_FRAMES       = 300

# encoder + SSL knobs
HIDDEN_DIM    = 128
TCN_KERNEL    = 5
TCN_DILATIONS = (1, 2, 4, 8)
DROPOUT       = 0.1
SSL_LR        = 3e-4
SSL_WD        = 1e-3
SSL_BATCH     = 4
SSL_MASK_FRAC = 0.15
SSL_MASK_SPAN = 10

TARGET_PIDS = ['P21', 'P22', 'P23', 'P24', 'P25',
               'P26', 'P27', 'P28', 'P29', 'P30']
MODEL_DIR   = 'bio_models'
ENCODER_NAME = lambda pid: os.path.join(MODEL_DIR, f'{pid}_ssl_encoder.pt')
os.makedirs(MODEL_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"DEVICE = {DEVICE}")

# pipeline init (skips if already loaded)
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

# %% Cell 2 — HG feature extractor
# ============================================================
def _extract_band_amp(data, sr, low, high, lp_hz=10.0,
                      win_s=0.015, shift_s=0.005):
    """extractHG recipe for one band: power -> 10 Hz LP -> 15 ms window -> sqrt."""
    x = sps.detrend(data, axis=0)
    sos = sps.iirfilter(4, [low/(sr/2), high/(sr/2)],
                        btype='bandpass', output='sos')
    x = sps.sosfiltfilt(sos, x, axis=0)
    if high > 95:                        # notch only when band touches line-noise
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


def extract_multiband(data, sr, bands, lp_hz=10.0):
    """For each (low, high) band, extract amplitude envelope; concat on channels.
    bands=[(70,170)] reproduces single-band extractHG."""
    feats = [_extract_band_amp(data, sr, lo, hi, lp_hz=lp_hz)
             for (lo, hi) in bands]
    n_min = min(f.shape[0] for f in feats)
    return np.concatenate([f[:n_min] for f in feats], axis=1).astype(np.float32)


def _channel_mask(pid):
    cm = getattr(pipeline, 'channel_masks', {}).get(pid, None)
    if cm is None: return None
    return np.asarray(cm['keep_indices'], dtype=np.int64)


def build_sentence_dataset(pid, bands=BANDS):
    """Returns {'train': [...], 'test': [...]} of per-sentence dicts:
        {'X': (T, n_ch * n_bands) float32, 'mfa': [...], 'sent_idx': int}
    Test split = every 6th real sentence (TEST_OFFSET::6) — the SSL convention."""
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
        X = extract_multiband(raw_eeg[s0:s1], EEG_SR, bands)
        if X.shape[0] < 30: continue
        split = 'test' if sent_idx in test_sent_ids else 'train'
        out[split].append({'X':        torch.from_numpy(X),
                           'mfa':      mfa[sent_idx],
                           'sent_idx': sent_idx})
    n_in = out['train'][0]['X'].shape[1] if out['train'] else 0
    print(f"  [{pid}] bands={bands}  n_in={n_in}  "
          f"train={len(out['train'])}  test={len(out['test'])}")
    return out

# %% Cell 3 — causal TCN encoder + SSL masking head
# ============================================================
class CausalConv1d(nn.Conv1d):
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
        return h + x


class CausalTCNEncoder(nn.Module):
    def __init__(self, n_in, hidden=HIDDEN_DIM, kernel=TCN_KERNEL,
                 dilations=TCN_DILATIONS, dropout=DROPOUT):
        super().__init__()
        self.proj_in    = nn.Conv1d(n_in, hidden, kernel_size=1)
        self.blocks     = nn.ModuleList(
            [TCNBlock(hidden, kernel, d, dropout) for d in dilations])
        self.mask_token = nn.Parameter(torch.zeros(hidden))
        nn.init.normal_(self.mask_token, std=0.02)
    def forward(self, x, mask=None):
        h = self.proj_in(x.transpose(1, 2))     # (B, hidden, T)
        if mask is not None:
            h = torch.where(mask.unsqueeze(1),
                            self.mask_token.view(1, -1, 1), h)
        for blk in self.blocks: h = blk(h)
        return h.transpose(1, 2)                 # (B, T, hidden)


class SSLHead(nn.Module):
    def __init__(self, hidden, n_out):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(),
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

# %% Cell 4 — SSL pretraining loop (used only if a checkpoint is missing)
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
    """Train per-patient encoder on masked-frame MSE. Standardizes ds in place.
    Returns enc, mu, sd."""
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    if DEVICE.type == 'cuda': torch.cuda.manual_seed_all(seed)

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

    train_sents = ds['train']; n = len(train_sents)
    print(f"  [{pid}] SSL pretrain: n_in={n_in} n_train={n} epochs={epochs}")
    t0 = time.time()
    for ep in range(epochs):
        enc.train(); head.train()
        rng.shuffle(train_sents)
        total = 0.0; nb = 0
        for i in range(0, n, SSL_BATCH):
            batch = train_sents[i:i + SSL_BATCH]
            Tmax = max(s['X'].shape[0] for s in batch)
            X = torch.zeros(len(batch), Tmax, n_in)
            valid = torch.zeros(len(batch), Tmax, dtype=torch.bool)
            mask  = torch.zeros(len(batch), Tmax, dtype=torch.bool)
            for b, s in enumerate(batch):
                T = s['X'].shape[0]
                X[b, :T] = s['X']; valid[b, :T] = True
                mask[b, :T] = torch.from_numpy(make_span_mask(T, rng=rng))
            X = X.to(DEVICE); mask = mask.to(DEVICE); valid = valid.to(DEVICE)
            h = enc(X, mask=mask); pred = head(h)
            sel = mask & valid
            if sel.sum() == 0: continue
            loss = F.mse_loss(pred[sel], X[sel])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(enc.parameters(), 1.0)
            opt.step(); total += loss.item(); nb += 1
        sch.step()
        if ep == 0 or (ep + 1) % 20 == 0:
            print(f"    ep {ep+1:3d}/{epochs}  mse={total/max(nb,1):.4f}  "
                  f"lr={opt.param_groups[0]['lr']:.2e}  ({time.time()-t0:.1f}s)")
    return enc, mu, sd

# %% Cell 5 — MFA helpers + embedding extraction + diversity metrics
# ============================================================
def time_to_frame(t_s):
    return int(round((t_s * EEG_SR - WIN_SAMP / 2) / SHIFT_SAMP))


def ssl_frame_to_time_s(i):
    return (i * SHIFT_SAMP + WIN_SAMP / 2) / EEG_SR


@torch.no_grad()
def extract_embeddings(pid, sents, encoder):
    encoder.eval()
    out = {}
    for s in sents:
        X = s['X'].unsqueeze(0).to(DEVICE)
        out[s['sent_idx']] = encoder(X).squeeze(0).cpu().numpy().astype(np.float32)
    return out


def extract_match_ngrams(out):
    gold_per, pred_per = gather_sequences(out)
    all_match_phones, n2g, n3g, n4plus = [], [], [], []
    for sid in sorted(set(gold_per) | set(pred_per)):
        gold = gold_per.get(sid, []); pred = pred_per.get(sid, [])
        run = []
        for g, p in needleman_wunsch(gold, pred):
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


def diversity_stats(out):
    phones, n2g, n3g, n4g = extract_match_ngrams(out)
    gold_per, _ = gather_sequences(out)
    inv_size = len(set(p for seq in gold_per.values() for p in seq))
    return {
        'n_match': len(phones), 'uniq_phones': len(set(phones)), 'inv_size': inv_size,
        'n2_total': len(n2g), 'uniq_n2': len(set(g for _, g in n2g)),
        'n3_total': len(n3g), 'uniq_n3': len(set(g for _, g in n3g)),
        'n4_total': len(n4g), 'uniq_n4': len(set(g for _, g in n4g)),
        'top_n3': Counter(g for _, g in n3g).most_common(3),
        'top_n4': Counter(g for _, g in n4g).most_common(3),
    }

# %% Cell 6 — inference: LDA on embeddings + scalar-bonus Viterbi
# ============================================================
def run_for_patient_ssl(pid, datasets, embeddings):
    """LDA on per-phoneme embedding means, decode test with scalar self-loop
    Viterbi. Returns the standard `out` dict for nw_metrics + the viz."""
    ds  = datasets[pid]
    mfa_by_sid = {s['sent_idx']: s['mfa'] for s in ds['train'] + ds['test']}
    per_sent   = embeddings[pid]

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
                X.append(emb[k_s:k_e + 1].mean(axis=0)); y.append(ph['phone'])
        return np.array(X), np.array(y)

    X_fit, y_fit = build_set(fit_sent_ids)
    if len(X_fit) < 50: return None, f"too few fit samples ({len(X_fit)})"

    sc_fit  = StandardScaler().fit(X_fit)
    clf_fit = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    clf_fit.fit(sc_fit.transform(X_fit), y_fit)
    fit_classes = set(y_fit)

    val_logps, val_target = [], 0
    for sid in val_sent_ids:
        if sid not in per_sent: continue
        logp = smooth_cols(clf_fit.predict_log_proba(sc_fit.transform(per_sent[sid])),
                           SMOOTH_LOGP_W)
        val_logps.append(logp)
        val_target += sum(1 for ph in mfa_by_sid[sid] if ph['phone'] in fit_classes)
    if not val_logps: return None, "no val sentences"
    val_target = int(val_target * TARGET_RATIO)
    bonus = (auto_tune_bonus(val_logps, val_target, MIN_PRED_FRAMES)
             if SELF_LOOP_BONUS is None else float(SELF_LOOP_BONUS))

    X_tr, y_tr = build_set(set(all_real) - test_sent_ids)
    scaler = StandardScaler().fit(X_tr)
    clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    clf.fit(scaler.transform(X_tr), y_tr)
    train_classes = set(y_tr); class_labels = list(clf.classes_)

    predictions, pred_sentence_ids, pred_segments = [], [], []
    true_labels, true_sentence_ids, true_segments = [], [], []
    for sid in sorted(test_sent_ids):
        if sid not in per_sent: continue
        emb = per_sent[sid]; T = emb.shape[0]
        logp = smooth_cols(clf.predict_log_proba(scaler.transform(emb)), SMOOTH_LOGP_W)
        path = viterbi_decode(logp, bonus)
        i = 0
        while i < T:
            ci = path[i]; j = i + 1
            while j < T and path[j] == ci: j += 1
            if (j - i) >= MIN_PRED_FRAMES:
                predictions.append(class_labels[ci]); pred_sentence_ids.append(sid)
                pred_segments.append((ssl_frame_to_time_s(i), ssl_frame_to_time_s(j - 1)))
            i = j
        for ph in mfa_by_sid[sid]:
            if ph['phone'] not in train_classes: continue
            true_labels.append(ph['phone']); true_sentence_ids.append(sid)
            true_segments.append((ph['start_s'], ph['end_s']))

    if not true_labels: return None, "no test gold labels"
    true_arr = np.array(true_labels); pred_arr = np.array(predictions)
    ed = edit_distance(list(true_arr), list(pred_arr)); per = ed / max(len(true_arr), 1)
    return {
        'true_labels': true_arr, 'predictions': pred_arr,
        'true_sentence_ids': np.array(true_sentence_ids),
        'pred_sentence_ids': np.array(pred_sentence_ids),
        'true_segments': true_segments, 'pred_segments': pred_segments,
        'accuracy': float('nan'), 'edit_distance': ed, 'per': per,
        'n_test': len(true_arr), 'n_pred': len(pred_arr),
        'n_train': len(X_tr), 'bonus': bonus,
    }, None

# %% Cell 7 — build/load HG-only encoders + embeddings for all patients
# ============================================================
def build_or_load(pid, bands=BANDS, epochs=SSL_EPOCHS, seed=0, train_if_missing=True):
    """Build the HG dataset, then load the encoder checkpoint if present
    (standardizing with its saved mu/sd) or SSL-pretrain one if missing."""
    ds = build_sentence_dataset(pid, bands)              # raw (un-standardized) X
    path = ENCODER_NAME(pid)
    if os.path.exists(path):
        ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
        enc = CausalTCNEncoder(ds['train'][0]['X'].shape[1]).to(DEVICE)
        enc.load_state_dict(ckpt['enc'])
        mu, sd = (ckpt['mu'], ckpt['sd']) if 'mu' in ckpt else fit_mu_sd(ds['train'])
        standardize_inplace(ds['train'], mu, sd)
        standardize_inplace(ds['test'],  mu, sd)
        print(f"  [{pid}] loaded encoder: {path}")
    elif train_if_missing:
        enc, mu, sd = ssl_pretrain_one(pid, ds, epochs=epochs, seed=seed)  # standardizes in place
        torch.save({'enc': enc.state_dict(), 'n_in': enc.proj_in.in_channels,
                    'bands': bands, 'epochs': epochs, 'mu': mu, 'sd': sd}, path)
        print(f"  [{pid}] trained + saved encoder: {path}")
    else:
        return None, None, None, None
    return ds, enc, mu, sd


datasets, encoders, embeddings = {}, {}, {}
print("Building HG-only state (load checkpoint, else train)...")
for pid in TARGET_PIDS:
    ds, enc, mu, sd = build_or_load(pid)
    if ds is None: continue
    datasets[pid] = ds
    encoders[pid] = enc
    embeddings[pid] = {**extract_embeddings(pid, ds['train'], enc),
                       **extract_embeddings(pid, ds['test'],  enc)}
print("HG-only state ready.")

# %% Cell 8 — cohort inference + NW metrics + save
# ============================================================
ssl_results = {}
print(f"{'pid':<5} {'match':>7} {'z':>6} {'n2':>4} {'n3':>4} {'n4':>4}  pred/gold  bonus")
print('-' * 65)
for pid in TARGET_PIDS:
    if pid not in embeddings: continue
    out, err = run_for_patient_ssl(pid, datasets, embeddings)
    if err: print(f"  {pid}: SKIP — {err}"); continue
    ssl_results[pid] = out
    m = nw_metrics(out)
    print(f"{pid:<5} {100*m['match_rate']:6.1f}% {m['z_match']:+5.2f} "
          f"{m['n2']:>4} {m['n3']:>4} {m['n4']:>4}  "
          f"{out['n_pred']:>3}/{out['n_test']:>3}  {out['bonus']:.2f}")

if ssl_results:
    ams = [nw_metrics(o)['match_rate'] for o in ssl_results.values()]
    azs = [nw_metrics(o)['z_match']    for o in ssl_results.values()]
    print('-' * 65)
    print(f"AVG   {100*np.mean(ams):6.1f}% {np.mean(azs):+5.2f}")

    os.makedirs('results/ssl_only', exist_ok=True)
    with open('results/ssl_only/hg_only_cohort.pkl', 'wb') as f:
        pickle.dump(ssl_results, f)
    summary = {'spec': 'HG_only', 'bands': BANDS, 'SSL_EPOCHS': SSL_EPOCHS,
               'TARGET_RATIO': TARGET_RATIO, 'MIN_PRED_FRAMES': MIN_PRED_FRAMES,
               'SMOOTH_LOGP_W': SMOOTH_LOGP_W, 'cohort': {}}
    for pid, out in ssl_results.items():
        m = nw_metrics(out); d = diversity_stats(out)
        summary['cohort'][pid] = {
            'match_rate': float(m['match_rate']), 'z_match': float(m['z_match']),
            'n2': int(m['n2']), 'n3': int(m['n3']), 'n4': int(m['n4']),
            'uniq_n3': int(d['uniq_n3']), 'n3_total': int(d['n3_total']),
            'n_pred': int(out['n_pred']), 'n_test': int(out['n_test']),
            'bonus': float(out['bonus']),
        }
    with open('results/ssl_only/hg_only_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print("saved -> results/ssl_only/hg_only_cohort.pkl + hg_only_summary.json")

# %% Cell 9 — visualize matched sequences
# ============================================================
from IPython.display import display, HTML

pipeline.patient_results = dict(ssl_results)   # so the viz can find them
for pid in sorted(ssl_results):
    out = ssl_results[pid]; m = nw_metrics(out); d = diversity_stats(out)
    top3 = '  '.join(f"{''.join(g)}x{c}" for g, c in d['top_n3'][:3]) or '-'
    display(HTML(
        f"<hr><h3>{pid} — HG-only (TR={TARGET_RATIO})</h3>"
        f"<div style='font-family:monospace;font-size:12px;background:#f4f4f4;"
        f"padding:6px;border-radius:4px;margin-bottom:8px;'>"
        f"match={100*m['match_rate']:.1f}% &nbsp; z={m['z_match']:+.2f} &nbsp; "
        f"n2={m['n2']} n3={m['n3']} n4={m['n4']} &nbsp; "
        f"uniq_n3={d['uniq_n3']}/{d['n3_total']} &nbsp; "
        f"pred/gold={out['n_pred']}/{out['n_test']} &nbsp; bonus={out['bonus']:.2f}"
        f"<br>top n3: {top3}</div>"))
    show_matched_sequences_with_times(pipeline, pid, max_per_line=25,
                                      collapse_repeats=True, time_align_tol_s=0.10)

# %% Cell 10 — shift-input permutation test (CE + accuracy)
# ============================================================
def pvalue_shift_input_ssl(pid, datasets_d, encoders_d, statistic='ce',
                            n_perm=2000, edge_frac=0.1, seed=0, verbose=False):
    """Shift-input null: per test sentence, circular-shift the input frames,
    re-encode (frozen encoder), re-LDA (fit on train, frozen), score CE or
    per-frame accuracy against the ORIGINAL gold. statistic in {'ce','accuracy'}."""
    ds  = datasets_d[pid]
    enc = encoders_d[pid]; enc.eval()

    # fit LDA on train per-phoneme embedding means (fixed for all perms)
    train_X, train_y = [], []
    with torch.no_grad():
        for s in ds['train']:
            h = enc(s['X'].unsqueeze(0).to(DEVICE)).squeeze(0).cpu().numpy(); T = h.shape[0]
            for ph in s['mfa']:
                k_s = max(0, time_to_frame(ph['start_s']))
                k_e = min(T - 1, time_to_frame(ph['end_s']))
                if k_e < k_s: continue
                train_X.append(h[k_s:k_e+1].mean(axis=0)); train_y.append(ph['phone'])
    train_X = np.array(train_X); train_y = np.array(train_y)
    scaler = StandardScaler().fit(train_X)
    lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    lda.fit(scaler.transform(train_X), train_y)
    class_idx = {c: i for i, c in enumerate(lda.classes_)}

    def encode_and_score(shift_fn):
        all_logp, all_gold = [], []
        with torch.no_grad():
            for s in ds['test']:
                X = s['X'] if shift_fn is None else shift_fn(s['X'])
                h = enc(X.unsqueeze(0).to(DEVICE)).squeeze(0).cpu().numpy(); T = h.shape[0]
                logp = lda.predict_log_proba(scaler.transform(h))
                gold = np.full(T, -1, dtype=int)
                for ph in s['mfa']:
                    if ph['phone'] not in class_idx: continue
                    k_s = max(0, time_to_frame(ph['start_s']))
                    k_e = min(T - 1, time_to_frame(ph['end_s']))
                    if k_e >= k_s: gold[k_s:k_e+1] = class_idx[ph['phone']]
                all_logp.append(logp); all_gold.append(gold)
        lp = np.concatenate(all_logp); gd = np.concatenate(all_gold); v = gd >= 0
        if v.sum() == 0: return np.nan
        if statistic == 'accuracy':
            return (lp[v].argmax(axis=1) == gd[v]).mean()
        return -lp[v][np.arange(v.sum()), gd[v]].mean()

    obs = encode_and_score(None)
    if not np.isfinite(obs): return {'error': 'observed undefined'}

    rng = np.random.RandomState(seed)
    null_list, n_bad = [], 0
    for b in range(n_perm):
        shifts = {}
        for s in ds['test']:
            T = s['X'].shape[0]
            if T < 20: shifts[s['sent_idx']] = 0; continue
            lo = max(1, int(edge_frac * T)); hi = T - lo
            shifts[s['sent_idx']] = rng.randint(lo, hi + 1) if hi > lo else rng.randint(1, T)
        # per-sentence circular shift, re-encode, score
        def make_shifted_score():
            all_logp, all_gold = [], []
            with torch.no_grad():
                for s in ds['test']:
                    X = torch.roll(s['X'], shifts=int(shifts[s['sent_idx']]), dims=0)
                    h = enc(X.unsqueeze(0).to(DEVICE)).squeeze(0).cpu().numpy(); T = h.shape[0]
                    logp = lda.predict_log_proba(scaler.transform(h))
                    gold = np.full(T, -1, dtype=int)
                    for ph in s['mfa']:
                        if ph['phone'] not in class_idx: continue
                        k_s = max(0, time_to_frame(ph['start_s']))
                        k_e = min(T - 1, time_to_frame(ph['end_s']))
                        if k_e >= k_s: gold[k_s:k_e+1] = class_idx[ph['phone']]
                    all_logp.append(logp); all_gold.append(gold)
            lp = np.concatenate(all_logp); gd = np.concatenate(all_gold); vmask = gd >= 0
            if vmask.sum() == 0: return np.nan
            if statistic == 'accuracy':
                return (lp[vmask].argmax(axis=1) == gd[vmask]).mean()
            return -lp[vmask][np.arange(vmask.sum()), gd[vmask]].mean()
        v = make_shifted_score()
        if not np.isfinite(v): n_bad += 1; continue
        null_list.append(v)
        if verbose and ((b+1) % 100 == 0 or b == 0):
            print(f"    perm {b+1}/{n_perm}  null {statistic}={v:.4f}")

    nulls = np.asarray(null_list, float)
    if nulls.size < max(20, n_perm // 2):
        return {'error': f'too many invalid perms ({n_bad}/{n_perm})'}
    if statistic == 'accuracy':                    # higher is better
        z = (obs - nulls.mean()) / (nulls.std(ddof=1) + 1e-9)
        p = (np.sum(nulls >= obs) + 1) / (nulls.size + 1)
    else:                                          # CE: lower is better
        z = (nulls.mean() - obs) / (nulls.std(ddof=1) + 1e-9)
        p = (np.sum(nulls <= obs) + 1) / (nulls.size + 1)
    return {'pid': pid, 'statistic': statistic, 'obs': float(obs),
            'null_mean': float(nulls.mean()), 'null_std': float(nulls.std(ddof=1)),
            'z': float(z), 'p_one_sided': float(p), 'n_perm': int(nulls.size), 'n_bad': n_bad}


def run_shift_test_table(label, statistic='ce', n_perm=2000, edge_frac=0.1, seed=0):
    stat_col = 'acc_obs' if statistic == 'accuracy' else 'CE_obs'
    print(f"\n{label}: {statistic} based")
    print(f"{'pid':<5} {stat_col:>8} {'null_mu':>8} {'null_sigma':>9} {'z':>6} {'p':>9}")
    print('-' * 60)
    results = {}
    for pid in TARGET_PIDS:
        if pid not in encoders: continue
        r = pvalue_shift_input_ssl(pid, datasets, encoders, statistic=statistic,
                                   n_perm=n_perm, edge_frac=edge_frac, seed=seed)
        if 'error' in r:
            print(f"{pid:<5} SKIP — {r['error']}"); continue
        results[pid] = r
        print(f"{pid:<5} {r['obs']:8.3f} {r['null_mean']:8.3f} "
              f"{r['null_std']:9.4f} {r['z']:+6.2f} {r['p_one_sided']:9.4f}")
    if results:
        pv = np.clip([r['p_one_sided'] for r in results.values()], 1e-300, 1.0)
        q = ss.false_discovery_control(pv, method='bh')
        chi2 = -2 * np.log(pv).sum(); df = 2 * len(pv)
        print('-' * 60)
        print(f"BH-FDR significant: {(q < 0.05).sum()}/{len(pv)}")
        print(f"Fisher combined p:  {1 - ss.chi2.cdf(chi2, df):.2e}")
    return results


ssl_ce  = run_shift_test_table('ssl_only', statistic='ce')
ssl_acc = run_shift_test_table('ssl_only', statistic='accuracy')
