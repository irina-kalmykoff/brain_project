# ============================================================
# arrive_at_v1.py — minimal cell-by-cell path to the v1 recipe
#
# v1 = HG amplitude (70-170 Hz) + low_beta amplitude (13-20 Hz)
#      -> per-patient causal TCN encoder, SSL pretrained for 120 epochs
#      -> LDA on per-phoneme-averaged 128-d embeddings
#      -> scalar self-loop bonus Viterbi at TARGET_RATIO = 1.7
#
# Copy each `# %%` block into a Jupyter cell.  Run top-to-bottom on a fresh
# kernel.  Cells 1-8 are setup (fast).  Cell 9 is the training loop (~3 hrs
# on 10 patients).  Cells 10-11 save artefacts and visualise.
# ============================================================


# %% Cell 1 — imports, hyperparameters, pipeline init
# ============================================================
import os, time, math, json, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.signal as sps

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from collections import Counter

from dutch_30_pipeline import Dutch30Pipeline
from dutch_30_feature_extractor import Dutch30FeatureExtractor
from dataset_config import Dutch30Config
from config import DUTCH_30_PATH
from run_pipeline import load_mfa_alignments
from e2e_brain_decoder import edit_distance, show_matched_sequences_with_times

# reuse the helpers from the LDA library (smoothing, Viterbi, NW metrics)
from LDA_on_frames_clean import (
    smooth_cols, viterbi_decode, auto_tune_bonus,
    nw_metrics, print_nw_metrics, gather_sequences, needleman_wunsch,
    SELF_LOOP_BONUS, VAL_FRAC, TEST_OFFSET,
    EEG_SR, WIN_S, SHIFT_S, WIN_SAMP, SHIFT_SAMP,
)

# ── v1 hyperparameters ───────────────────────────────────────────────
V1_BANDS        = [(70, 170), (13, 20)]   # HG + low_beta amplitude
SSL_EPOCHS_V1   = 120                      # scales with n_in (wider than HG-only)
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


# %% Cell 2 — multiband feature extractor (THIS is where beta is added)
# ============================================================
def _extract_band_amp(data, sr, low, high, lp_hz=10.0,
                      win_s=0.015, shift_s=0.005):
    """extractHG recipe applied to one band: power -> 10 Hz LP -> sqrt at 200 Hz."""
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
    """For each (low, high) band, extract amplitude envelope.  Concatenate
    along channels.  bands=[(70,170)] reproduces single-band extractHG;
    bands=[(70,170), (13,20)] is the v1 feature stack."""
    feats = [_extract_band_amp(data, sr, lo, hi, lp_hz=lp_hz)
             for (lo, hi) in bands]
    n_min = min(f.shape[0] for f in feats)
    return np.concatenate([f[:n_min] for f in feats], axis=1).astype(np.float32)


def _channel_mask(pid):
    cm = getattr(pipeline, 'channel_masks', {}).get(pid, None)
    if cm is None: return None
    return np.asarray(cm['keep_indices'], dtype=np.int64)


def build_sentence_dataset_multi(pid, bands):
    """Returns {'train': [...], 'test': [...]} of per-sentence dicts:
        {'X': (T, n_ch * n_bands) float32, 'mfa': [...], 'sent_idx': int}
    For v1 call with bands = V1_BANDS = [(70, 170), (13, 20)]."""
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


# %% Cell 4 — SSL pretraining loop
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


def ssl_pretrain_one(pid, ds, epochs, seed=0):
    """Train per-patient encoder on masked-frame MSE.  Returns enc, mu, sd."""
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

    train_sents = ds['train']
    n = len(train_sents)
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
                mb = make_span_mask(T, rng=rng)
                mask[b, :T] = torch.from_numpy(mb)
            X = X.to(DEVICE); mask = mask.to(DEVICE); valid = valid.to(DEVICE)
            h    = enc(X, mask=mask); pred = head(h)
            sel  = mask & valid
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
        h = encoder(X).squeeze(0).cpu().numpy().astype(np.float32)
        out[s['sent_idx']] = h
    return out


def extract_match_ngrams(out):
    """Walk NW alignments; collect runs of consecutive matches."""
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


def diversity_stats(out):
    phones, n2g, n3g, n4g = extract_match_ngrams(out)
    gold_per, _ = gather_sequences(out)
    inv_size = len(set(p for seq in gold_per.values() for p in seq))
    return {
        'n_match':      len(phones),
        'uniq_phones':  len(set(phones)),
        'inv_size':     inv_size,
        'n2_total':     len(n2g),  'uniq_n2': len(set(g for _, g in n2g)),
        'n3_total':     len(n3g),  'uniq_n3': len(set(g for _, g in n3g)),
        'n4_total':     len(n4g),  'uniq_n4': len(set(g for _, g in n4g)),
        'top_n3':       Counter(g for _, g in n3g).most_common(3),
        'top_n4':       Counter(g for _, g in n4g).most_common(3),
    }


# %% Cell 6 — inference: LDA on embeddings + scalar-bonus Viterbi
# ============================================================
def run_for_patient_ssl(pid, datasets, embeddings):
    """Drop-in inference: take v1 embeddings, train LDA on per-phoneme
    averages, decode test with scalar self-loop bonus Viterbi.  Returns the
    standard `out` dict used by nw_metrics + show_matched_sequences_with_times."""
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
                X.append(emb[k_s:k_e + 1].mean(axis=0))
                y.append(ph['phone'])
        return np.array(X), np.array(y)

    X_fit, y_fit = build_set(fit_sent_ids)
    if len(X_fit) < 50: return None, f"too few fit samples ({len(X_fit)})"

    sc_fit  = StandardScaler().fit(X_fit)
    clf_fit = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    clf_fit.fit(sc_fit.transform(X_fit), y_fit)
    fit_classes = set(y_fit)

    # auto-tune Viterbi self-loop bonus on val
    val_logps, val_target = [], 0
    for sid in val_sent_ids:
        if sid not in per_sent: continue
        logp = clf_fit.predict_log_proba(sc_fit.transform(per_sent[sid]))
        logp = smooth_cols(logp, SMOOTH_LOGP_W)
        val_logps.append(logp)
        val_target += sum(1 for ph in mfa_by_sid[sid] if ph['phone'] in fit_classes)
    if not val_logps: return None, "no val sentences"
    val_target = int(val_target * TARGET_RATIO)
    bonus = (auto_tune_bonus(val_logps, val_target, MIN_PRED_FRAMES)
             if SELF_LOOP_BONUS is None else float(SELF_LOOP_BONUS))

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
        path = viterbi_decode(logp, bonus)
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
        'true_labels':       true_arr,
        'predictions':       pred_arr,
        'true_sentence_ids': np.array(true_sentence_ids),
        'pred_sentence_ids': np.array(pred_sentence_ids),
        'true_segments':     true_segments,
        'pred_segments':     pred_segments,
        'accuracy':          float('nan'),
        'edit_distance':     ed, 'per': per,
        'n_test':            len(true_arr),
        'n_pred':            len(pred_arr),
        'n_train':           len(X_tr),
        'bonus':             bonus,
    }, None


# %% Cell 7 — HTML visualisation helpers
# ============================================================
COL_MATCH = '#a6e3a1'
COL_SUB   = '#f5c2c0'
COL_INS   = '#ffd966'
COL_DEL   = '#dddddd'

def render_aligned_pair(aligned):
    gold_cells, pred_cells = [], []
    style = ("padding:2px 5px;margin-right:1px;display:inline-block;"
             "min-width:14px;text-align:center;border-radius:3px;")
    for g, p in aligned:
        if g is not None and p is not None and g == p:
            gold_cells.append(f"<span style='{style}background:{COL_MATCH};font-weight:bold;'>{g}</span>")
            pred_cells.append(f"<span style='{style}background:{COL_MATCH};font-weight:bold;'>{p}</span>")
        elif g is not None and p is not None:
            gold_cells.append(f"<span style='{style}background:{COL_SUB};font-weight:bold;'>{g}</span>")
            pred_cells.append(f"<span style='{style}background:{COL_SUB};'>{p}</span>")
        elif g is not None:
            gold_cells.append(f"<span style='{style}background:{COL_DEL};font-weight:bold;'>{g}</span>")
            pred_cells.append(f"<span style='{style}background:{COL_DEL};color:#888;'>·</span>")
        else:
            gold_cells.append(f"<span style='{style}background:{COL_INS};color:#888;'>·</span>")
            pred_cells.append(f"<span style='{style}background:{COL_INS};'>{p}</span>")
    return ''.join(gold_cells), ''.join(pred_cells)


def compare_predictions_html(out_a, out_b=None, label_a="A", label_b="B",
                              max_sentences=20):
    gold_a_per, pred_a_per = gather_sequences(out_a)
    if out_b is not None:
        gold_b_per, pred_b_per = gather_sequences(out_b)
        common = sorted(set(gold_a_per) | set(gold_b_per))
    else:
        gold_b_per = pred_b_per = None
        common = sorted(set(gold_a_per))
    rows = ["<style>.pcomp td{padding:4px 8px;font-family:monospace;font-size:13px;}"
            ".pcomp tr.sentheader td{background:#e0e0e0;font-weight:bold;padding-top:8px;}</style>"]
    rows.append("<div style='margin-bottom:8px;font-family:sans-serif;font-size:13px;'>"
                f"<span style='background:{COL_MATCH};padding:3px 8px;margin-right:6px;border-radius:3px;'>match</span>"
                f"<span style='background:{COL_SUB};padding:3px 8px;margin-right:6px;border-radius:3px;'>substitution</span>"
                f"<span style='background:{COL_INS};padding:3px 8px;margin-right:6px;border-radius:3px;'>insertion</span>"
                f"<span style='background:{COL_DEL};padding:3px 8px;margin-right:6px;border-radius:3px;'>deletion</span></div>")
    rows.append("<table class='pcomp' style='border-collapse:collapse;'>")
    for sid in common[:max_sentences]:
        gold_a = gold_a_per.get(sid, [])
        pred_a = pred_a_per.get(sid, [])
        gold = gold_a if gold_a else (gold_b_per.get(sid, []) if out_b else [])
        if not gold: continue
        g_a, p_a = render_aligned_pair(needleman_wunsch(gold, pred_a))
        rows.append(f"<tr class='sentheader'><td colspan='2'>Sentence {sid}</td></tr>")
        rows.append(f"<tr><td>{label_a} gold</td><td>{g_a}</td></tr>")
        rows.append(f"<tr><td>{label_a} pred</td><td>{p_a}</td></tr>")
        if out_b is not None:
            pred_b = pred_b_per.get(sid, [])
            g_b, p_b = render_aligned_pair(needleman_wunsch(gold, pred_b))
            rows.append(f"<tr><td>{label_b} gold</td><td>{g_b}</td></tr>")
            rows.append(f"<tr><td>{label_b} pred</td><td>{p_b}</td></tr>")
    rows.append("</table>")
    return ''.join(rows)


def show_predictions_html(out, label='model', max_sentences=10):
    return compare_predictions_html(out, out_b=None,
                                    label_a=label, max_sentences=max_sentences)


# %% Cell 8 — sanity check: confirm beta is in the v1 feature matrix
# ============================================================
# Cheap check before kicking off the 3-hour training: builds a v1 dataset
# for one patient and verifies the feature width is double the HG-only width.
print("Verifying v1 features include low_beta...")
ds_check = build_sentence_dataset_multi('P22', V1_BANDS)
n_in = ds_check['train'][0]['X'].shape[1]

X = ds_check['train'][0]['X'].numpy()
n_ch = n_in // 2
print(f"\n  P22 v1 feature matrix:")
print(f"    n_in              = {n_in}  (= 2 × {n_ch} electrodes — HG + low_beta)")
print(f"    HG block stats    = mean={X[:, :n_ch].mean():.3f}  var={X[:, :n_ch].var():.3f}")
print(f"    low_beta block    = mean={X[:, n_ch:].mean():.3f}  var={X[:, n_ch:].var():.3f}")
print(f"\n  If n_in is twice the HG-only width and the two blocks have")
print(f"  different statistics, v1 features are correctly set up.\n")


# %% Cell 9 — TRAIN v1 ENCODERS FOR ALL 10 PATIENTS  (~2-3 hours)
# ============================================================
# This is the long-running cell.  Builds v1 features, SSL-pretrains the
# encoder for each patient (120 epochs on n_in ≈ 220 inputs), and stashes
# encoder + embeddings in memory.  Saves checkpoints to bio_models/.

datasets, encoders, mus, sds, embeddings = {}, {}, {}, {}, {}

for pid in TARGET_PIDS:
    print(f"\n[{pid}] training v1 encoder")
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    if DEVICE.type == 'cuda': torch.cuda.manual_seed_all(0)

    ds = build_sentence_dataset_multi(pid, V1_BANDS)
    enc, mu, sd = ssl_pretrain_one(pid, ds, epochs=SSL_EPOCHS_V1, seed=0)
    datasets[pid]  = ds
    encoders[pid]  = enc
    mus[pid]       = mu
    sds[pid]       = sd

    # save encoder + standardisation stats together
    torch.save({'enc':    enc.state_dict(),
                'n_in':   enc.proj_in.in_channels,
                'bands':  V1_BANDS,
                'epochs': SSL_EPOCHS_V1,
                'mu':     mu,
                'sd':     sd},
               os.path.join(MODEL_DIR, f'{pid}_ssl_encoder_low_beta.pt'))

    # extract per-sentence embeddings (frozen encoder)
    emb_tr = extract_embeddings(pid, ds['train'], enc)
    emb_te = extract_embeddings(pid, ds['test'],  enc)
    embeddings[pid] = {**emb_tr, **emb_te}

print("\nAll v1 encoders trained and saved.")


# %% Cell 10 — run v1 inference for all patients + lock results
# ============================================================
v1_results = {}
print(f"{'pid':<5} {'match':>7} {'z':>6} {'n2':>4} {'n3':>4} {'n4':>4}  pred/gold  bonus")
print('-' * 65)
for pid in TARGET_PIDS:
    if pid not in embeddings: continue
    out, err = run_for_patient_ssl(pid, datasets, embeddings)
    if err: print(f"  {pid}: SKIP — {err}"); continue
    v1_results[pid] = out
    m = nw_metrics(out)
    print(f"{pid:<5} {100*m['match_rate']:6.1f}% {m['z_match']:+5.2f} "
          f"{m['n2']:>4} {m['n3']:>4} {m['n4']:>4}  "
          f"{out['n_pred']:>3}/{out['n_test']:>3}  {out['bonus']:.2f}")

ams = [nw_metrics(o)['match_rate'] for o in v1_results.values()]
azs = [nw_metrics(o)['z_match']    for o in v1_results.values()]
print('-' * 65)
print(f"AVG   {100*np.mean(ams):6.1f}% {np.mean(azs):+5.2f}")

# save pickle + JSON summary
import pickle
os.makedirs('results/v1', exist_ok=True)
with open('results/v1/hg_lowbeta_tr17_cohort.pkl', 'wb') as f:
    pickle.dump(v1_results, f)

summary = {'spec': 'HG+low_beta', 'bands': V1_BANDS,
           'SSL_EPOCHS': SSL_EPOCHS_V1, 'TARGET_RATIO': TARGET_RATIO,
           'MIN_PRED_FRAMES': MIN_PRED_FRAMES,
           'SMOOTH_LOGP_W': SMOOTH_LOGP_W,
           'cohort': {}}
for pid in TARGET_PIDS:
    if pid not in v1_results: continue
    out = v1_results[pid]; m = nw_metrics(out); d = diversity_stats(out)
    summary['cohort'][pid] = {
        'match_rate': float(m['match_rate']),
        'z_match':    float(m['z_match']),
        'n2': int(m['n2']), 'n3': int(m['n3']), 'n4': int(m['n4']),
        'uniq_n3': int(d['uniq_n3']), 'n3_total': int(d['n3_total']),
        'n_pred':  int(out['n_pred']), 'n_test':  int(out['n_test']),
        'bonus':   float(out['bonus']),
    }
with open('results/v1/hg_lowbeta_tr17_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\nv1 saved to:")
print(f"  results/v1/hg_lowbeta_tr17_cohort.pkl")
print(f"  results/v1/hg_lowbeta_tr17_summary.json")
print(f"  bio_models/{{P21..P30}}_ssl_encoder_low_beta.pt")


# %% Cell 11 — visualise v1 results
# ============================================================
from IPython.display import display, HTML

# install into pipeline so show_matched_sequences_with_times can find them
pipeline.patient_results = dict(v1_results)

for pid in sorted(v1_results):
    out = v1_results[pid]; m = nw_metrics(out); d = diversity_stats(out)
    top3 = '  '.join(f"{''.join(g)}×{c}" for g, c in d['top_n3'][:3]) or '—'
    display(HTML(
        f"<hr><h3>{pid} — v1 (HG + low_beta amp, TR={TARGET_RATIO})</h3>"
        f"<div style='font-family:monospace;font-size:12px;"
        f"background:#f4f4f4;padding:6px;border-radius:4px;margin-bottom:8px;'>"
        f"match={100*m['match_rate']:.1f}% &nbsp; z={m['z_match']:+.2f} &nbsp; "
        f"n2={m['n2']} n3={m['n3']} n4={m['n4']} &nbsp; "
        f"uniq_n3={d['uniq_n3']}/{d['n3_total']} &nbsp; "
        f"pred/gold={out['n_pred']}/{out['n_test']} &nbsp; "
        f"bonus={out['bonus']:.2f}<br>top n3: {top3}</div>"
    ))
    show_matched_sequences_with_times(pipeline, pid,
                                       max_per_line=45,
                                       collapse_repeats=True,
                                       time_align_tol_s=0.10)
    display(HTML(show_predictions_html(out, label=f'{pid} v1',
                                        max_sentences=8)))
