# ============================================================
# SSL pretraining for per-patient causal sEEG encoder
#
# Stage 1: masked-frame MSE on `extractHG` features (no labels)
# Stage 2: multi-task aux finetune
#          (silence/speech + word-onset + syllable-onset)
# Stage 3: frozen encoder -> 128-d frame embeddings ->
#          drop into existing LDA + Viterbi pipeline
#
# Mirrors silence_vs_speech_pretraining.py structure for review.
# Run cells top-to-bottom.  Each `# %%` boundary is a notebook cell.
# ============================================================


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
    SMOOTH_LOGP_W, SELF_LOOP_BONUS, TARGET_RATIO, MIN_PRED_FRAMES,
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


# %% Cell 8 — LDA + Viterbi on SSL embeddings (mirrors LDA_on_frames_clean)
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


# %% Cell 9 — evaluation + visualization
# ============================================================
# Paste the user's HTML visualization cell (render_aligned_pair,
# compare_predictions_html, stats_nw, ctc_results_to_out,
# show_predictions_html) into a cell ABOVE this one if not already loaded.

from IPython.display import display, HTML

for pid in TARGET_PIDS:
    if pid not in pipeline.patient_results: continue
    print(f"\n{'='*60}\n{pid} — SSL encoder + LDA + Viterbi\n{'='*60}")

    # 1) time-aware matched-sequences view (your pipeline function)
    show_matched_sequences_with_times(pipeline, pid,
                                       max_per_line=30,
                                       collapse_repeats=True,
                                       time_align_tol_s=0.10)

    # 2) NW metrics + permutation z
    m = nw_metrics(pipeline.patient_results[pid])
    print_nw_metrics(m, label=pid)

    # 3) HTML alignment view (uses the helpers you pasted)
    out = pipeline.patient_results[pid]
    stats_nw(out, label=pid)
    display(HTML(show_predictions_html(out, label=pid, max_sentences=10)))


# %% Cell 10 — (optional) A/B SSL vs baseline stacked-HG LDA
# ============================================================
# If you ran the original `LDA_on_frames_clean.run_for_patient_sd` and
# stashed its outputs in `baseline_results[pid]`, you can do a side-by-side
# comparison cell:
#
#     for pid in TARGET_PIDS:
#         if pid not in baseline_results or pid not in ssl_results: continue
#         print(f"\n=== {pid}: baseline (stacked HG) vs SSL encoder ===")
#         stats_nw(baseline_results[pid], label='baseline')
#         stats_nw(ssl_results[pid],      label='SSL')
#         display(HTML(compare_predictions_html(
#             baseline_results[pid], ssl_results[pid],
#             label_a='baseline', label_b='SSL', max_sentences=10)))
