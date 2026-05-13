"""
Path B v2 — brain-to-phoneme decoding with BIO-CRF on a BiLSTM backbone.

A simple, end-to-end walkthrough of what we built. The numbered steps below
match the conceptual pipeline; each is self-contained and explains *why*
before *how*.

Pipeline at a glance:

    raw EEG  ──►  high-gamma envelope  ──►  frame-level features
                                                    │
                                                    ▼
                       MFA phoneme borders  ──►  BIO labels per frame
                                                    │
                                                    ▼
                                BiLSTM emissions  +  linear-chain CRF
                                                    │
                                                    ▼
                          Viterbi decode  ──►  collapse BIO → phonemes
                                                    │
                                                    ▼
                                  max_run  +  surprise z   ← real metric
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import butter, sosfiltfilt, iirnotch, tf2sos, hilbert


# ═════════════════════════════════════════════════════════════════════════════
# STEP 1 — Configuration
# ═════════════════════════════════════════════════════════════════════════════
# These constants describe the signal processing: 1024 Hz sEEG, 30 ms analysis
# window stepped every 5 ms (= 200 Hz frame rate). The high-gamma band is the
# canonical iEEG speech-production correlate.

EEG_SR   = 1024                  # sEEG sampling rate (Hz)
HG_LOW   = 70                    # high-gamma lower bound (Hz)
HG_HIGH  = 170                   # high-gamma upper bound (Hz)
NOTCH_HZ = [50, 150]             # power-line noise notches
WIN_MS   = 30                    # analysis window
SHIFT_MS = 5                     # frame stride → 200 Hz output rate

WIN_SAMP   = int(EEG_SR * WIN_MS   / 1000)
SHIFT_SAMP = int(EEG_SR * SHIFT_MS / 1000)
DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'


# ═════════════════════════════════════════════════════════════════════════════
# STEP 2 — High-gamma envelope from raw EEG
# ═════════════════════════════════════════════════════════════════════════════
# The 70-170 Hz band itself is a fast-oscillating broadband signal; what we
# actually want is its *amplitude envelope*, which tracks underlying neural
# firing rate. The recipe is the iEEG/ECoG standard:
#   1. notch out 50/150 Hz power-line noise
#   2. bandpass filter to 70-170 Hz
#   3. Hilbert transform → analytic signal; |analytic| is the envelope
#   4. average within 30 ms windows stepped every 5 ms → frame-level features

def _design_filters():
    sos_bp = butter(4, [HG_LOW, HG_HIGH], btype='bandpass',
                    fs=EEG_SR, output='sos')
    sos_notches = []
    for f0 in NOTCH_HZ:
        b, a = iirnotch(f0, 30, EEG_SR)
        sos_notches.append(tf2sos(b, a))
    return sos_bp, sos_notches


_SOS_BP, _SOS_NOTCH = _design_filters()


def extract_hg_frames(eeg_slice):
    """Raw EEG → frame-level high-gamma envelope.

    Input:  (T_samples, n_channels) at 1024 Hz
    Output: (T_frames,  n_channels) at 200 Hz, log-compressed
    """
    x = eeg_slice.astype(np.float64)
    for sos in _SOS_NOTCH:
        x = sosfiltfilt(sos, x, axis=0)
    x = sosfiltfilt(_SOS_BP, x, axis=0)
    env = np.abs(hilbert(x, axis=0))                 # amplitude envelope

    n_frames = max(0, (env.shape[0] - WIN_SAMP) // SHIFT_SAMP + 1)
    out = np.zeros((n_frames, env.shape[1]), dtype=np.float32)
    for k in range(n_frames):
        s = k * SHIFT_SAMP
        out[k] = env[s:s + WIN_SAMP].mean(axis=0)
    return np.log1p(out)                             # tame extreme values


def stack_context(X, K=5):
    """Sliding-window stack so each frame "sees" ±K neighbors.

    Output: (T, n_channels * (2K+1)). For K=5 that's 11 frames of context
    (±25 ms), giving the per-frame MLP access to local temporal patterns
    even before the BiLSTM adds its own longer-range context.
    """
    T, C = X.shape
    pad  = np.zeros((K, C), dtype=X.dtype)
    Xp   = np.vstack([pad, X, pad])
    cols = [Xp[k:k + T] for k in range(2 * K + 1)]
    return np.concatenate(cols, axis=1)


# ═════════════════════════════════════════════════════════════════════════════
# STEP 3 — BIO labels per frame from MFA borders
# ═════════════════════════════════════════════════════════════════════════════
# BIO tagging tells the model not just "what phoneme" but "are we starting
# this phoneme or continuing it." For each MFA-aligned phoneme interval, the
# first frame is B-X, every following frame is I-X. Frames outside any
# phoneme interval (silence) are dropped entirely in v2 — dropping them
# kills the "predict silence everywhere" collapse mode.

def build_speech_only_labels(mfa_phones, n_frames):
    """For each frame, return (bio_tag, phoneme_symbol, keep_mask).
       keep_mask[i] = True if frame i lies inside any phoneme interval.
    """
    bio  = [None] * n_frames
    phon = [None] * n_frames
    keep = np.zeros(n_frames, dtype=bool)
    for ph in mfa_phones:
        start_f = int(np.ceil((ph['start_s'] * EEG_SR - WIN_SAMP / 2) / SHIFT_SAMP))
        end_f   = int(np.floor((ph['end_s']   * EEG_SR - WIN_SAMP / 2) / SHIFT_SAMP))
        start_f = max(0, start_f)
        end_f   = min(n_frames - 1, end_f)
        if end_f < start_f:
            continue
        sym = ph['phone']
        bio[start_f]  = f'B-{sym}'; phon[start_f] = sym; keep[start_f] = True
        for k in range(start_f + 1, end_f + 1):
            bio[k] = f'I-{sym}'; phon[k] = sym; keep[k] = True
    return bio, phon, keep


# ═════════════════════════════════════════════════════════════════════════════
# STEP 4 — Linear-chain CRF
# ═════════════════════════════════════════════════════════════════════════════
# A CRF on top of per-frame logits enforces sequence-level structure. Two
# pieces matter here:
#
# (a) Structural mask: I-X may *only* follow B-X or I-X (of the same X).
#     This bakes the BIO grammar in as hard constraints. Without it the
#     model is free to emit nonsense like "B-a I-t" (start /a/, continue
#     /t/) and learns slowly.
#
# (b) Bigram-initialized transition matrix: B-X → B-Y entries seeded from
#     phoneme-pair frequencies. Gives the optimizer a head start.

class LinearChainCRF(nn.Module):
    def __init__(self, n_tags, transition_mask, transition_init):
        super().__init__()
        self.n_tags = n_tags
        self.trans  = nn.Parameter(transition_init.clone())   # learnable
        self.start  = nn.Parameter(torch.zeros(n_tags))
        self.end    = nn.Parameter(torch.zeros(n_tags))
        self.register_buffer('mask', transition_mask)         # static -inf

    def _T(self):
        return self.trans + self.mask                         # add -inf to forbidden

    def neg_log_likelihood(self, emissions, tags):
        return self._forward_alg(emissions) - self._score(emissions, tags)

    # forward algorithm: partition function via dynamic programming
    def _forward_alg(self, emissions):
        T_, K = emissions.shape
        alpha = self.start + emissions[0]
        for t in range(1, T_):
            alpha = torch.logsumexp(alpha[:, None] + self._T(), dim=0) + emissions[t]
        return torch.logsumexp(alpha + self.end, dim=0)

    # score the gold tag sequence
    def _score(self, emissions, tags):
        s = self.start[tags[0]] + emissions[0, tags[0]]
        for t in range(1, emissions.size(0)):
            s = s + self._T()[tags[t - 1], tags[t]] + emissions[t, tags[t]]
        return s + self.end[tags[-1]]

    # Viterbi: highest-scoring tag path
    def viterbi(self, emissions):
        T_, K = emissions.shape
        bp = torch.zeros(T_, K, dtype=torch.long, device=emissions.device)
        v  = self.start + emissions[0]
        for t in range(1, T_):
            scores = v.unsqueeze(1) + self._T()
            v, bp[t] = scores.max(dim=0)
            v = v + emissions[t]
        last = int((v + self.end).argmax().item())
        path = [last]
        for t in range(T_ - 1, 0, -1):
            last = int(bp[t, last].item())
            path.append(last)
        return list(reversed(path))


# ═════════════════════════════════════════════════════════════════════════════
# STEP 5 — BiLSTM emission backbone + multi-head architecture
# ═════════════════════════════════════════════════════════════════════════════
# The BiLSTM is what makes per-frame predictions temporally coherent. Each
# emission sees a hidden state aggregated over the whole sentence (forward
# *and* backward), so the model can integrate evidence across the ~17 frames
# of a single phoneme before deciding any tag.
#
# Auxiliary heads (manner, place of articulation) target slowly-varying
# articulatory features. These pin the shared backbone representation: a
# frame inside /b/ is "stop + bilabial" regardless of which /b/ it is, and
# the LSTM learns that smoothness quickly.

class BiLSTM_BIO_CRF(nn.Module):
    def __init__(self, n_in, n_tags, n_manner, n_place,
                 lstm_hidden=128, lstm_layers=2, dropout=0.3,
                 transition_mask=None, transition_init=None):
        super().__init__()
        self.n_tags = n_tags

        # Project then BiLSTM
        self.proj = nn.Sequential(
            nn.Linear(n_in, lstm_hidden * 2), nn.GELU(), nn.Dropout(dropout))
        self.lstm = nn.LSTM(lstm_hidden * 2, lstm_hidden,
                            num_layers=lstm_layers, dropout=dropout,
                            bidirectional=True, batch_first=False)

        # Heads on the BiLSTM output (size 2 * lstm_hidden)
        self.bio_head     = nn.Linear(lstm_hidden * 2, n_tags)      # for CRF
        self.bio_aux_head = nn.Linear(lstm_hidden * 2, n_tags)      # for direct CE
        self.manner_head  = nn.Linear(lstm_hidden * 2, n_manner)
        self.place_head   = nn.Linear(lstm_hidden * 2, n_place)
        self.drop         = nn.Dropout(dropout)
        self.crf          = LinearChainCRF(n_tags, transition_mask, transition_init)

    def encode(self, x):
        h, _ = self.lstm(self.proj(x).unsqueeze(1))                 # (T, 1, 2H)
        return self.drop(h.squeeze(1))

    def loss(self, x, tags, manner, place,
             lam_manner=0.3, lam_place=0.1, lam_bio_ce=0.5, ce_weights=None):
        h = self.encode(x)
        bio_em  = self.bio_head(h)
        crf_nll = self.crf.neg_log_likelihood(bio_em, tags) / x.size(0)

        # Direct CE on BIO tags. CRF NLL gradients are weak when one class
        # dominates the partition function; direct CE gives the emission
        # head an unambiguous per-frame signal.
        bio_ce  = F.cross_entropy(self.bio_aux_head(h), tags, weight=ce_weights)

        mn_loss = F.cross_entropy(self.manner_head(h), manner)
        pl_loss = F.cross_entropy(self.place_head(h),  place)

        return crf_nll + lam_bio_ce * bio_ce + lam_manner * mn_loss + lam_place * pl_loss

    @torch.no_grad()
    def decode(self, x):
        return self.crf.viterbi(self.bio_head(self.encode(x)))


# ═════════════════════════════════════════════════════════════════════════════
# STEP 6 — Training (mixup augmentation + composite model-selection)
# ═════════════════════════════════════════════════════════════════════════════
# Augmentation: for each training step, ~20% of the *mid-phoneme* frames
# (centers of phonemes ≥6 frames long, where the label is least ambiguous)
# get mixed 80/20 with a random frame from a different sentence. Label
# stays as the anchor's. This is label-preserving mixup, cheap regularization
# that helps the BiLSTM not overfit ~80 training sentences.

def augment_sentence(anchor, partner_pool, p_aug, aug_frac, mix_ratio, rng):
    if rng.random() > p_aug or anchor['mid'].sum() == 0:
        return anchor['X']
    mid_idx   = anchor['mid'].nonzero(as_tuple=False).flatten()
    n_perturb = max(1, int(len(mid_idx) * aug_frac))
    perm      = mid_idx[torch.randperm(len(mid_idx))[:n_perturb]]
    partner   = partner_pool[rng.integers(0, len(partner_pool))]
    p_idx     = torch.randint(0, partner['X'].size(0), (n_perturb,))
    X = anchor['X'].clone()
    X[perm] = mix_ratio * X[perm] + (1.0 - mix_ratio) * partner['X'][p_idx]
    return X


# Composite model-selection score. Raw frame-tag accuracy is a poor target
# because a model that always emits the most-frequent phoneme can score well
# while being useless. We multiply tag-acc by two factors:
#   - diversity:    how many unique phonemes did the model emit?
#   - coverage:     how many phoneme onsets relative to expected count?
# Together they reject collapsed models even at high tag-acc.

def evaluate_for_selection(model, sents_te, idx_to_tag):
    model.eval()
    correct = total = 0
    pred_symbols = set(); n_pred = 0
    with torch.no_grad():
        for s in sents_te:
            path = model.decode(s['X'])
            tg = s['tags'].tolist()
            for p, t in zip(path, tg):
                correct += (p == t); total += 1
            for p in path:
                tag = idx_to_tag[int(p)]
                if tag.startswith('B-'):
                    pred_symbols.add(tag[2:]); n_pred += 1
    tag_acc        = correct / max(total, 1)
    diversity      = min(1.0, len(pred_symbols) / 3.0)
    coverage_ratio = min(1.0, n_pred / max(1, total / 34))
    return tag_acc * diversity * coverage_ratio


# ═════════════════════════════════════════════════════════════════════════════
# STEP 7 — Decode + collapse BIO sequences to phoneme sequences
# ═════════════════════════════════════════════════════════════════════════════
# After Viterbi gives us a tag path, we collapse consecutive B-X / I-X tags
# of the same phoneme into a single symbol. A run of frames labeled
# `B-aː I-aː I-aː I-aː I-aː` collapses to one /aː/ in the phoneme sequence.

def collapse_bio_to_segments(bio_tag_strs):
    """Returns (phoneme_symbols, start_frames, end_frames_exclusive)."""
    phons, starts, ends = [], [], []
    i = 0
    while i < len(bio_tag_strs):
        t = bio_tag_strs[i]
        if t.startswith('B-') or t.startswith('I-'):
            sym = t[2:]
            j = i + 1
            while j < len(bio_tag_strs) and bio_tag_strs[j] == f'I-{sym}':
                j += 1
            phons.append(sym); starts.append(i); ends.append(j)
            i = j
        else:
            i += 1
    return phons, starts, ends


# ═════════════════════════════════════════════════════════════════════════════
# STEP 8 — Score against gold with the real metric
# ═════════════════════════════════════════════════════════════════════════════
# The headline metric is *longest contiguous exact match per sentence*
# (max_run), validated against a permutation null (surprise z). Frame-level
# tag accuracy is necessary but not sufficient: a model with high frame-acc
# from prior-collapse can still score z ≈ 0 because the matches it produces
# are typical of what shuffled predictions with the same marginal would
# produce. A genuine win is max_run ≥ 4 with z ≥ +3.

def longest_run_with_shift(pred, gold, shift_max=3):
    best = 0
    for i in range(len(pred)):
        for j in range(max(0, i - shift_max), min(len(gold), i + shift_max + 1)):
            k = 0
            while i + k < len(pred) and j + k < len(gold) and pred[i + k] == gold[j + k]:
                k += 1
            best = max(best, k)
    return best


def surprise_z(pred_sents, gold_sents, n_perm=2000, min_match=3, seed=0):
    """Pred & gold are lists of per-sentence phoneme sequences. Returns
       (max_run, observed_surprise, null_mean, null_std, z).
    """
    from collections import Counter
    rng  = np.random.default_rng(seed)
    gold_all = [ph for s in gold_sents for ph in s]
    cnt  = Counter(gold_all); N = sum(cnt.values())
    logp = {k: np.log(v / N) for k, v in cnt.items()}

    def score(p_sents):
        s = 0.0
        for p, g in zip(p_sents, gold_sents):
            L = longest_run_with_shift(p, g)
            if L >= min_match:
                # find which match this run is; sum -log P over its symbols
                for i in range(len(p) - L + 1):
                    for j in range(len(g) - L + 1):
                        if p[i:i + L] == g[j:j + L]:
                            s += -sum(logp.get(ph, -np.log(1e-6))
                                      for ph in p[i:i + L])
                            break
        return s

    obs   = score(pred_sents)
    nulls = np.zeros(n_perm)
    for b in range(n_perm):
        shuf = [[p[k] for k in rng.permutation(len(p))] for p in pred_sents]
        nulls[b] = score(shuf)
    mu, sd = nulls.mean(), nulls.std() + 1e-9
    max_run = max(longest_run_with_shift(p, g)
                  for p, g in zip(pred_sents, gold_sents))
    return max_run, obs, mu, sd, (obs - mu) / sd


# ═════════════════════════════════════════════════════════════════════════════
# What this file leaves out
# ═════════════════════════════════════════════════════════════════════════════
# This walkthrough shows the conceptual core. The notebook code wraps it
# with:
#   - per-patient data builders that pull raw EEG from disk and run steps 2-4
#   - the manner/place phoneme tables (Dutch IPA → 5 manner, 8 place classes)
#   - the full training loop with AdamW + cosine annealing + early stopping
#   - GPU placement / standardization / batching by sentence
#   - the adapter that wires results into `pipeline.patient_results[pid]`
#     so the visualizer (`show_matched_sequences_with_times`) can render them
#
# The story to remember:
#
#   1. raw EEG  →  Hilbert envelope of 70-170 Hz band  →  log-compressed frames
#   2. MFA borders + frames  →  BIO labels (no O; silence frames dropped)
#   3. BiLSTM emissions  →  CRF with structural mask  →  Viterbi decode
#   4. Collapse B-X I-X I-X ...  →  one phoneme per run
#   5. Score with max_run + surprise z, never per-frame accuracy alone
