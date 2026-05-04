# Frame-level brain-only phoneme decoding via CTC.
#
# Per-frame BiLSTM emits logits over (phoneme vocabulary ∪ blank).
# Trained with CTC loss against the unsegmented MFA phoneme sequence
# per sentence — no boundaries needed at training, no segmentation at
# inference. Greedy decode (argmax per frame, collapse repeats, drop
# blanks) gives the predicted phoneme sequence directly.
#
# Comparison target: e2e_brain_decoder.py (v6) achieved +0.5 PER pts vs
# MFA oracle. CTC bypasses the boundary-detection bottleneck entirely.

# ── 1. TORCH FIRST ────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── 2. STANDARD ───────────────────────────────────────────────────────────────
import os
import glob
import random
from collections import Counter, defaultdict
from datetime import datetime

# ── 3. THIRD-PARTY ────────────────────────────────────────────────────────────
import numpy as np

# ── 4. PROJECT ────────────────────────────────────────────────────────────────
from config import DUTCH_30_PATH
from run_pipeline import load_mfa_alignments
from e2e_brain_decoder import (
    build_joint_dataset_fixed, edit_distance,
)
from boundary_detector_joint_audio import (
    split_by_sentence,
    fit_train_stats, apply_stats,
    ALL_PIDS, FRAME_HZ,
    BATCH_SIZE,
)


# ═════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═════════════════════════════════════════════════════════════════════════════

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED   = 37
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

CKPT_PREFIX = 'frame_ctc_v3_'           # CTC pretraining with regularization
CKPT_PREFIX_MRT = 'frame_ctc_v4_mrt_'   # MRT fine-tuning of v3

# Validation + checkpointing
VAL_FRACTION         = 0.10  # held-out val split inside train_ds
CHECKPOINT_EVERY     = 20    # save a periodic checkpoint every N epochs
CHECKPOINT_START_EPOCH = 0   # don't save periodics before this epoch (saves disk on early useless ckpts)
SAVE_BEST_VAL        = True  # also save best-val checkpoint (suffix _best)
EARLY_STOP_PATIENCE  = None  # set to int N to stop after N epochs without val improvement; None = never

# ── MRT (Minimum Risk Training) — sequence-discriminative fine-tuning ────────
# At each step, sample N alignments from the model's per-frame distribution,
# collapse each to a phoneme sequence, score with ngram_coverage(min_n=3),
# and use the rewards as policy-gradient signal to push the model toward
# higher-scoring sequences. Mixed with CTC loss for stability.
MRT_MODE             = False    # set True to switch run_path_ctc to MRT fine-tuning
MRT_N_SAMPLES        = 5
MRT_TEMPERATURE      = 1.0      # sampling temperature; 1.0 = use model dist as-is
MRT_CTC_WEIGHT       = 0.7      # λ in: total = λ*CTC + (1-λ)*MRT
MRT_MIN_NGRAM        = 3
MRT_LR               = 1e-4     # half of pretraining LR (5e-4)
MRT_N_EPOCHS         = 40       # fine-tune is short

# Model
HIDDEN_DIM    = 256
N_LSTM_LAYERS = 2
DROPOUT       = 0.25    # eased slightly from v2 (was 0.3)
PROJ_DIM      = 96
HEAD_HIDDEN   = 256

# Training
N_EPOCHS_TOTAL = 120
LR             = 5e-4
WEIGHT_DECAY   = 5e-4   # eased slightly from v2 (was 1e-3)
GRAD_CLIP      = 1.0

# Per-step channel masking from v2 — turned off in v3 (replaced by the
# per-epoch single-channel dropout below). Set to 0 to disable.
CHANNEL_MASK_PROB = 0.00
CHANNEL_MASK_MAX  = 0.20

# Per-EPOCH single-channel dropout: at the start of each epoch, pick one
# random channel per patient and zero it out for the whole epoch's worth of
# training batches. Forces the model to develop redundant representations
# (no single channel is irreplaceable). At eval time no channel is dropped.
PER_EPOCH_CHANNEL_DROPOUT = True

# Decoding
DECODE_MODE = 'beam'        # 'greedy' (argmax + collapse) or 'beam' (CTC beam search)
DECODE_LENGTH_BONUS = 0.0   # subtracted from blank logit; positive -> more emissions
BEAM_SIZE           = 8     # only used when DECODE_MODE='beam'

# Warm-start: if True and a v6 boundary detector checkpoint exists, copy
# its eeg_proj + LSTM weights into the CTC model before any CTC training.
# Skipped automatically if dims don't match.
WARM_START_FROM_V6 = False
V6_CKPT_GLOB       = 'boundary_detector_v6_*.pt'

# Class filter (mirrors run_path_b / _run_crf_experiment behavior)
MIN_CLASS_SAMPLES = 1


# ═════════════════════════════════════════════════════════════════════════════
# VOCAB — built from training-side phonemes only (leak-safe)
# ═════════════════════════════════════════════════════════════════════════════

class Vocab:
    """Phoneme ↔ integer index. Index 0 is reserved for CTC blank."""
    BLANK = '<blank>'

    def __init__(self, phonemes):
        self.itos = [self.BLANK] + sorted(phonemes)
        self.stoi = {p: i for i, p in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def encode(self, seq):
        return [self.stoi[p] for p in seq if p in self.stoi]

    def decode(self, idx_seq):
        return [self.itos[i] for i in idx_seq if i != 0]


def build_vocab_from_train(train_ds, min_count=MIN_CLASS_SAMPLES):
    """Collect phonemes seen at least `min_count` times across train MFA
    sequences. Filtering is done train-side only — no test peeking."""
    counts = Counter()
    for item in train_ds:
        for ph in item['phoneme_sequence']:
            counts[ph] += 1
    phonemes = {p for p, n in counts.items() if n >= min_count}
    return Vocab(phonemes)


# ═════════════════════════════════════════════════════════════════════════════
# DATASET — reuse build_joint_dataset_fixed, add target phoneme sequence
# ═════════════════════════════════════════════════════════════════════════════

def build_ctc_dataset(pipeline, patient_ids):
    """Reuse the corrected per-sentence EEG slicer from e2e_brain_decoder,
    then attach the MFA phoneme sequence as a CTC target."""
    full_ds = build_joint_dataset_fixed(pipeline, patient_ids)
    for item in full_ds:
        mfa = load_mfa_alignments(item['pid']).get(item['sentence_idx'], [])
        item['phoneme_sequence'] = [p['phone'] for p in mfa]
    return full_ds


def attach_target_indices(dataset, vocab):
    """Encode each item's phoneme_sequence to integer indices using vocab.
    Items whose entire sequence is filtered out (no in-vocab phonemes)
    are flagged so the dataloader can skip them."""
    for item in dataset:
        item['target_idx'] = vocab.encode(item['phoneme_sequence'])
    return dataset


# ═════════════════════════════════════════════════════════════════════════════
# MODEL — per-patient input proj + shared BiLSTM + linear over (V + blank)
# ═════════════════════════════════════════════════════════════════════════════

class FrameCTCModel(nn.Module):
    def __init__(self, per_patient_eeg_n_ch, vocab_size,
                 hidden_dim=HIDDEN_DIM, n_layers=N_LSTM_LAYERS,
                 dropout=DROPOUT, proj_dim=PROJ_DIM):
        super().__init__()
        self.eeg_proj = nn.ModuleDict({
            pid: nn.Linear(n_ch, proj_dim)
            for pid, n_ch in per_patient_eeg_n_ch.items()
        })
        self.lstm = nn.LSTM(
            input_size=proj_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        # MLP head: (2*hidden_dim) → HEAD_HIDDEN → vocab, with ReLU + dropout
        self.out = nn.Sequential(
            nn.Linear(2 * hidden_dim, HEAD_HIDDEN),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(HEAD_HIDDEN, vocab_size),
        )

    def forward(self, eeg, pid):
        h = self.eeg_proj[pid](eeg)
        h, _ = self.lstm(h)
        return self.out(h)   # (B, T, V); apply log_softmax outside if needed


# ═════════════════════════════════════════════════════════════════════════════
# TRAINING
# ═════════════════════════════════════════════════════════════════════════════

def collate_ctc(batch):
    """Pad EEG to longest in batch. Targets concatenated flat with lengths.

    Returns:
        X_eeg          (B, T_max, n_ch)
        input_lengths  (B,) actual T per item
        targets_flat   (sum_S,)  concatenated target indices
        target_lengths (B,)      target length per item
    """
    T_max = max(item['eeg'].shape[0] for item in batch)
    n_ch  = batch[0]['eeg'].shape[1]
    X_eeg = torch.zeros(len(batch), T_max, n_ch)
    input_lengths = torch.zeros(len(batch), dtype=torch.long)
    target_lengths = torch.zeros(len(batch), dtype=torch.long)
    targets_flat = []
    for i, item in enumerate(batch):
        T = item['eeg'].shape[0]
        X_eeg[i, :T] = torch.from_numpy(item['eeg'])
        input_lengths[i] = T
        target_lengths[i] = len(item['target_idx'])
        targets_flat.extend(item['target_idx'])
    targets_flat = torch.tensor(targets_flat, dtype=torch.long)
    return X_eeg, input_lengths, targets_flat, target_lengths


def warm_start_from_v6(model):
    """Copy eeg_proj + LSTM weights from v6 boundary detector. Skips
    silently if no v6 ckpt exists or any tensor shape disagrees."""
    ckpts = sorted(glob.glob(V6_CKPT_GLOB))
    if not ckpts:
        print("  [warm-start] no v6 checkpoint found, skipping")
        return False
    src = torch.load(ckpts[-1], map_location='cpu')['model_state']
    dst = model.state_dict()
    copied, skipped = 0, 0
    for k_dst in dst:
        # Map: model.eeg_proj.<pid>.X → eeg_proj.<pid>.X (same name in v6)
        # Map: model.lstm.X → lstm.X (same name in v6)
        if k_dst.startswith('eeg_proj.') or k_dst.startswith('lstm.'):
            if k_dst in src and src[k_dst].shape == dst[k_dst].shape:
                dst[k_dst] = src[k_dst]
                copied += 1
            else:
                skipped += 1
    model.load_state_dict(dst)
    print(f"  [warm-start] copied {copied} tensors from {ckpts[-1]}, "
          f"skipped {skipped} (shape mismatch)")
    return True


@torch.no_grad()
def _evaluate_ctc_loss(model, dataset, vocab, ctc_module=None):
    """Compute mean CTC loss + mean sampled-greedy n-gram coverage on a
    held-out dataset. Used for validation tracking."""
    if not dataset:
        return float('nan'), float('nan')
    if ctc_module is None:
        ctc_module = nn.CTCLoss(blank=0, zero_infinity=True)
    model.eval()
    by_pid = defaultdict(list)
    for d in dataset:
        if d['target_idx']:
            by_pid[d['pid']].append(d)
    sum_loss = 0.0; sum_seqs = 0
    sum_reward = 0.0
    from e2e_brain_decoder import ngram_coverage as _ngc
    for pid, items in by_pid.items():
        for i in range(0, len(items), BATCH_SIZE):
            batch = items[i:i + BATCH_SIZE]
            X_eeg, in_len, tgt, tgt_len = collate_ctc(batch)
            X_eeg = X_eeg.to(DEVICE); in_len = in_len.to(DEVICE)
            tgt = tgt.to(DEVICE); tgt_len = tgt_len.to(DEVICE)
            logits = model(X_eeg, pid)
            log_probs = F.log_softmax(logits, dim=-1)
            loss = ctc_module(log_probs.transpose(0, 1), tgt, in_len, tgt_len)
            if torch.isfinite(loss):
                sum_loss += loss.item() * len(batch)
                sum_seqs += len(batch)
            # Greedy decoded reward
            tgt_offset = 0
            for b_idx in range(len(batch)):
                T = int(in_len[b_idx].item())
                S = int(tgt_len[b_idx].item())
                true_idx = tgt[tgt_offset:tgt_offset + S].cpu().tolist()
                tgt_offset += S
                pred_idx = greedy_decode(log_probs[b_idx, :T])
                true_seq = [vocab.itos[i] for i in true_idx]
                pred_seq = [vocab.itos[i] for i in pred_idx]
                sum_reward += _ngc(true_seq, pred_seq, min_n=MRT_MIN_NGRAM)
    avg_loss = sum_loss / max(sum_seqs, 1)
    avg_r = sum_reward / max(sum_seqs, 1)
    return avg_loss, avg_r


def _save_ctc_checkpoint(model, vocab, epochs_done, out_path):
    torch.save({
        'model_state':          model.state_dict(),
        'per_patient_eeg_n_ch': {pid: model.eeg_proj[pid].in_features
                                  for pid in model.eeg_proj},
        'vocab_size':           len(vocab),
        'vocab_itos':           vocab.itos,
        'epochs_done':          epochs_done,
    }, out_path)


def train_ctc(train_dataset, vocab,
               n_epochs_total=N_EPOCHS_TOTAL, lr=LR,
               resume_model=None, resume_epoch=0,
               val_dataset=None,
               checkpoint_every=CHECKPOINT_EVERY,
               save_best=SAVE_BEST_VAL,
               early_stop_patience=EARLY_STOP_PATIENCE):
    """Train (or continue training). If resume_model is provided, picks up
    from resume_epoch and trains until n_epochs_total is reached.

    Periodic checkpoints are saved every `checkpoint_every` epochs as
    `frame_ctc_v3_e{epoch}.pt`. The best-val checkpoint (lowest val CTC
    loss) is saved as `frame_ctc_v3_best.pt`.

    If `early_stop_patience` is an int, stop after that many consecutive
    epochs with no val-loss improvement. None disables early stop.
    """
    per_patient_eeg_n_ch = {d['pid']: d['eeg'].shape[1] for d in train_dataset}
    if resume_model is None:
        model = FrameCTCModel(per_patient_eeg_n_ch, len(vocab)).to(DEVICE)
        if WARM_START_FROM_V6:
            warm_start_from_v6(model)
    else:
        model = resume_model
    optim = torch.optim.AdamW(model.parameters(), lr=lr,
                               weight_decay=WEIGHT_DECAY)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)

    # Group by patient — single-patient batches because eeg_proj is per-pid
    by_pid = defaultdict(list)
    for d in train_dataset:
        if d['target_idx']:    # skip empty targets
            by_pid[d['pid']].append(d)
    pids = list(by_pid.keys())

    if resume_epoch >= n_epochs_total:
        print(f"\n  Already at {resume_epoch} epochs ≥ target {n_epochs_total}; "
              f"skipping training.")
        return model, resume_epoch

    print(f"\n  Training frame-level CTC — epochs {resume_epoch+1}..{n_epochs_total}, "
          f"vocab={len(vocab)}, hidden={HIDDEN_DIM}, layers={N_LSTM_LAYERS}, "
          f"val_size={len(val_dataset) if val_dataset else 0}")

    best_val_loss = float('inf')
    best_path = f'{CKPT_PREFIX}best.pt'
    epochs_since_improve = 0

    for epoch in range(resume_epoch, n_epochs_total):
        model.train()
        random.shuffle(pids)
        total_loss = 0.0; total_seqs = 0

        # Per-epoch single-channel dropout: one random channel zeroed for
        # all batches this epoch (different per patient because channel
        # counts differ; different channel each epoch).
        if PER_EPOCH_CHANNEL_DROPOUT:
            epoch_drop_ch = {
                pid: random.randint(0, by_pid[pid][0]['eeg'].shape[1] - 1)
                for pid in pids
            }
        else:
            epoch_drop_ch = {}

        for pid in pids:
            items = by_pid[pid]
            random.shuffle(items)
            for i in range(0, len(items), BATCH_SIZE):
                batch = items[i:i + BATCH_SIZE]
                X_eeg, in_len, tgt, tgt_len = collate_ctc(batch)
                X_eeg = X_eeg.to(DEVICE)
                in_len = in_len.to(DEVICE)
                tgt = tgt.to(DEVICE)
                tgt_len = tgt_len.to(DEVICE)

                # CTC requires T must be ≥ 2*S+1 in the worst case; if any
                # item has T < 2*S+1 the loss may be inf. With blank=0 and
                # zero_infinity=True we just skip those.

                # Per-epoch single-channel dropout (one channel for the whole epoch)
                if PER_EPOCH_CHANNEL_DROPOUT and pid in epoch_drop_ch:
                    X_eeg = X_eeg.clone()
                    X_eeg[..., epoch_drop_ch[pid]] = 0.0

                # Per-channel masking: independently zero out random channels
                # for this batch (data augmentation). Capped at CHANNEL_MASK_MAX
                # of total channels.
                if CHANNEL_MASK_PROB > 0:
                    n_ch = X_eeg.shape[-1]
                    drop_mask = torch.rand(n_ch, device=X_eeg.device) < CHANNEL_MASK_PROB
                    if drop_mask.float().mean().item() > CHANNEL_MASK_MAX:
                        # Too many masked; randomly keep enough to satisfy cap
                        n_to_keep = int(n_ch * (1 - CHANNEL_MASK_MAX))
                        drop_idx = drop_mask.nonzero(as_tuple=True)[0]
                        keep = drop_idx[torch.randperm(len(drop_idx))[:len(drop_idx) - (n_ch - n_to_keep)]]
                        drop_mask[:] = False
                        drop_mask[keep] = True
                    if drop_mask.any():
                        X_eeg = X_eeg.clone()
                        X_eeg[..., drop_mask] = 0.0

                optim.zero_grad()
                logits = model(X_eeg, pid)               # (B, T, V)
                log_probs = F.log_softmax(logits, dim=-1)
                # CTC expects (T, B, V)
                log_probs = log_probs.transpose(0, 1)
                loss = ctc(log_probs, tgt, in_len, tgt_len)
                if not torch.isfinite(loss):
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optim.step()
                total_loss += loss.item() * len(batch)
                total_seqs += len(batch)

        # ── Per-epoch validation + checkpointing ─────────────────────────
        train_avg = total_loss / max(total_seqs, 1)
        val_loss = float('nan'); val_reward = float('nan')
        if val_dataset:
            val_loss, val_reward = _evaluate_ctc_loss(model, val_dataset, vocab, ctc)

        # Print every 5 epochs (or on last)
        if (epoch + 1) % 5 == 0 or epoch == n_epochs_total - 1:
            if val_dataset:
                print(f"    epoch {epoch+1:3d}/{n_epochs_total}  "
                      f"train={train_avg:.4f}  val={val_loss:.4f}  "
                      f"val_reward={val_reward:.3f}")
            else:
                print(f"    epoch {epoch+1:3d}/{n_epochs_total}  ctc_loss={train_avg:.4f}")

        # Save best-val checkpoint
        if save_best and val_dataset and val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_since_improve = 0
            _save_ctc_checkpoint(model, vocab, epoch + 1, best_path)
        else:
            epochs_since_improve += 1

        # Periodic checkpoint (only after CHECKPOINT_START_EPOCH)
        if (checkpoint_every
                and (epoch + 1) >= CHECKPOINT_START_EPOCH
                and (epoch + 1) % checkpoint_every == 0):
            periodic_path = f'{CKPT_PREFIX}e{epoch+1:03d}.pt'
            _save_ctc_checkpoint(model, vocab, epoch + 1, periodic_path)
            print(f"      saved periodic checkpoint: {periodic_path}")

        # Early stop
        if (early_stop_patience is not None
                and epochs_since_improve >= early_stop_patience):
            print(f"    early stop at epoch {epoch+1} "
                  f"(no val improvement for {early_stop_patience} epochs)")
            return model, epoch + 1

    return model, n_epochs_total


# ═════════════════════════════════════════════════════════════════════════════
# DECODE
# ═════════════════════════════════════════════════════════════════════════════

import math
from collections import defaultdict as _defaultdict


def _logsumexp(a, b):
    """Numerically stable log(exp(a) + exp(b)) for two scalars."""
    if a == float('-inf'): return b
    if b == float('-inf'): return a
    if a > b:
        return a + math.log1p(math.exp(b - a))
    return b + math.log1p(math.exp(a - b))


def beam_search_decode(log_probs, beam_size=None, length_bonus=None):
    """CTC beam search decoder (no language model).

    log_probs: torch.Tensor of shape (T, V), log-softmax outputs.
    beam_size: number of hypotheses kept at each step.
    length_bonus: subtracted from blank log-prob (index 0) at every frame —
        positive values bias toward more phoneme emissions.

    Returns list of vocab indices (collapsed sequence, no blanks).

    Algorithm: standard prefix beam search (Hannun et al. 2014). For each
    candidate prefix we maintain two log-probabilities — one for paths
    ending in blank (lpb), one for paths ending in the prefix's last
    non-blank symbol (lpn). At each frame we extend prefixes by either
    blank, the last non-blank symbol again (which collapses), or a new
    non-blank symbol. Top-K by total log-prob = logsumexp(lpb, lpn) is
    kept after each frame.
    """
    if beam_size is None:
        beam_size = BEAM_SIZE
    if length_bonus is None:
        length_bonus = DECODE_LENGTH_BONUS

    log_probs_np = log_probs.cpu().numpy().astype(np.float64)
    if length_bonus != 0.0:
        log_probs_np = log_probs_np.copy()
        log_probs_np[:, 0] -= length_bonus

    T, V = log_probs_np.shape
    NEG_INF = float('-inf')

    # beams: dict[prefix_tuple -> (log_pb, log_pn)]
    beams = {(): (0.0, NEG_INF)}

    for t in range(T):
        log_p_t = log_probs_np[t]
        new_beams = _defaultdict(lambda: (NEG_INF, NEG_INF))

        for prefix, (lpb, lpn) in beams.items():
            lp_total = _logsumexp(lpb, lpn)

            # Path: emit blank — prefix unchanged
            cur_pb, cur_pn = new_beams[prefix]
            new_beams[prefix] = (
                _logsumexp(cur_pb, lp_total + log_p_t[0]),
                cur_pn,
            )

            for c in range(1, V):
                lpc = log_p_t[c]
                if prefix and prefix[-1] == c:
                    # Same as last symbol of prefix — two cases:
                    # (1) extend non-blank (collapses with previous): prefix unchanged
                    cur_pb_p, cur_pn_p = new_beams[prefix]
                    new_beams[prefix] = (
                        cur_pb_p,
                        _logsumexp(cur_pn_p, lpn + lpc),
                    )
                    # (2) extend through blank → new emission, prefix grows
                    new_prefix = prefix + (c,)
                    cur_pb_n, cur_pn_n = new_beams[new_prefix]
                    new_beams[new_prefix] = (
                        cur_pb_n,
                        _logsumexp(cur_pn_n, lpb + lpc),
                    )
                else:
                    # Different symbol — prefix always grows
                    new_prefix = prefix + (c,)
                    cur_pb_n, cur_pn_n = new_beams[new_prefix]
                    new_beams[new_prefix] = (
                        cur_pb_n,
                        _logsumexp(cur_pn_n, lp_total + lpc),
                    )

        # Prune to top beam_size by total log-prob
        scored = [(p, pb_pn, _logsumexp(pb_pn[0], pb_pn[1]))
                  for p, pb_pn in new_beams.items()]
        scored.sort(key=lambda x: x[2], reverse=True)
        beams = {p: pb_pn for p, pb_pn, _ in scored[:beam_size]}

    best_prefix = max(beams, key=lambda p: _logsumexp(*beams[p]))
    return list(best_prefix)


def decode(log_probs, length_bonus=None):
    """Dispatch to greedy or beam based on DECODE_MODE config."""
    if DECODE_MODE == 'beam':
        return beam_search_decode(log_probs, length_bonus=length_bonus)
    return greedy_decode(log_probs, length_bonus=length_bonus)


def greedy_decode(log_probs, length_bonus=None):
    """log_probs: (T, V) for one item. Returns list of vocab indices.
    Greedy CTC decode: argmax per frame, collapse consecutive duplicates,
    drop blanks (index 0).

    length_bonus (defaults to DECODE_LENGTH_BONUS): positive value
    subtracts from the blank logit, biasing the argmax toward non-blank
    emissions and increasing predicted sequence length. Negative does
    the opposite. 0 = pure argmax.
    """
    if length_bonus is None:
        length_bonus = DECODE_LENGTH_BONUS
    if length_bonus != 0.0:
        log_probs = log_probs.clone()
        log_probs[..., 0] = log_probs[..., 0] - length_bonus
    pred = log_probs.argmax(dim=-1).cpu().numpy()
    decoded = []
    prev = -1
    for x in pred:
        x = int(x)
        if x != prev and x != 0:
            decoded.append(x)
        prev = x
    return decoded


# ═════════════════════════════════════════════════════════════════════════════
# MRT — Minimum Risk Training (sequence-discriminative fine-tuning)
# ═════════════════════════════════════════════════════════════════════════════

from e2e_brain_decoder import ngram_coverage as _ngram_coverage


def _sample_ctc_alignment(log_probs, temperature=1.0):
    """Draw one alignment from a model's per-frame log-prob distribution.

    Args:
        log_probs: torch.Tensor (T, V), log_softmax outputs.
        temperature: scaling factor; >1 = more random, <1 = sharper.

    Returns:
        tokens:   torch.LongTensor (T,) — the sampled token at each frame.
        log_p:    torch.Tensor (scalar) — log P(this alignment | model).
                  Differentiable w.r.t. model params.
    """
    if temperature != 1.0:
        log_probs = log_probs / temperature
        log_probs = F.log_softmax(log_probs, dim=-1)
    probs = log_probs.exp()
    # Per-frame categorical sample
    tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)   # (T,)
    # log P of the sampled alignment = sum of per-frame log P at chosen tokens
    log_p = log_probs.gather(1, tokens.unsqueeze(-1)).squeeze(-1).sum()
    return tokens, log_p


def _collapse_alignment(tokens):
    """Collapse a per-frame alignment to a phoneme index sequence.
    Standard CTC collapse: merge consecutive duplicates, drop blanks."""
    decoded = []
    prev = -1
    for x in tokens.cpu().numpy().tolist():
        x = int(x)
        if x != prev and x != 0:
            decoded.append(x)
        prev = x
    return decoded


def train_ctc_mrt(model, train_dataset, vocab,
                    n_epochs=None, lr=None,
                    n_samples=MRT_N_SAMPLES,
                    temperature=MRT_TEMPERATURE,
                    ctc_weight=MRT_CTC_WEIGHT):
    """Fine-tune `model` with mixed CTC + MRT loss.

    For each training sentence:
      1. Forward pass → log_probs (T, V)
      2. Sample N alignments; collapse each to a phoneme sequence
      3. Reward = ngram_coverage(sample, truth, min_n=MRT_MIN_NGRAM)
      4. mrt_loss   = -Σ (advantage_i × log_p_i)            # policy gradient
         total_loss = λ × ctc_loss + (1-λ) × mrt_loss
    """
    if n_epochs is None: n_epochs = MRT_N_EPOCHS
    if lr is None:       lr = MRT_LR
    optim = torch.optim.AdamW(model.parameters(), lr=lr,
                               weight_decay=WEIGHT_DECAY)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)

    by_pid = defaultdict(list)
    for d in train_dataset:
        if d['target_idx']:
            by_pid[d['pid']].append(d)
    pids = list(by_pid.keys())

    print(f"\n  MRT fine-tuning — {n_epochs} epochs, "
          f"N_samples={n_samples}, T={temperature}, λ={ctc_weight}, "
          f"min_ngram={MRT_MIN_NGRAM}")

    for epoch in range(n_epochs):
        model.train()
        random.shuffle(pids)
        sum_ctc = 0.0; sum_mrt = 0.0; sum_reward = 0.0
        n_seqs = 0

        for pid in pids:
            items = by_pid[pid]
            random.shuffle(items)
            for i in range(0, len(items), BATCH_SIZE):
                batch = items[i:i + BATCH_SIZE]
                X_eeg, in_len, tgt, tgt_len = collate_ctc(batch)
                X_eeg = X_eeg.to(DEVICE)
                in_len = in_len.to(DEVICE)
                tgt = tgt.to(DEVICE)
                tgt_len = tgt_len.to(DEVICE)

                optim.zero_grad()
                logits = model(X_eeg, pid)                 # (B, T, V)
                log_probs = F.log_softmax(logits, dim=-1)

                # ── CTC component (anchor) ─────────────────────────────
                ctc_input = log_probs.transpose(0, 1)      # (T, B, V)
                ctc_loss = ctc(ctc_input, tgt, in_len, tgt_len)
                if not torch.isfinite(ctc_loss):
                    continue

                # ── MRT component ──────────────────────────────────────
                # For each sequence in batch, sample n_samples alignments,
                # compute rewards, accumulate policy-gradient loss.
                mrt_loss_batch = 0.0
                tgt_offset = 0
                batch_rewards = []
                for b_idx in range(len(batch)):
                    item = batch[b_idx]
                    T = int(in_len[b_idx].item())
                    S = int(tgt_len[b_idx].item())
                    true_idx_seq = tgt[tgt_offset:tgt_offset + S].cpu().tolist()
                    tgt_offset += S
                    true_seq = [vocab.itos[i] for i in true_idx_seq]
                    seq_log_probs = log_probs[b_idx, :T]   # (T, V), differentiable

                    # Sample n hypotheses, collect rewards and log_ps
                    rewards = []
                    log_ps  = []
                    for _ in range(n_samples):
                        tokens, log_p = _sample_ctc_alignment(
                            seq_log_probs, temperature=temperature)
                        decoded = _collapse_alignment(tokens)
                        pred_seq = [vocab.itos[i] for i in decoded]
                        r = _ngram_coverage(true_seq, pred_seq,
                                             min_n=MRT_MIN_NGRAM)
                        rewards.append(r)
                        log_ps.append(log_p)

                    # Variance-reduced advantages (subtract mean reward)
                    mean_r = sum(rewards) / max(n_samples, 1)
                    seq_mrt = 0.0
                    for r, lp in zip(rewards, log_ps):
                        adv = r - mean_r
                        seq_mrt = seq_mrt - adv * lp
                    seq_mrt = seq_mrt / n_samples
                    mrt_loss_batch = mrt_loss_batch + seq_mrt
                    batch_rewards.append(mean_r)

                mrt_loss_batch = mrt_loss_batch / len(batch)

                total = ctc_weight * ctc_loss + (1 - ctc_weight) * mrt_loss_batch
                if not torch.isfinite(total):
                    continue
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optim.step()

                sum_ctc += ctc_loss.item() * len(batch)
                sum_mrt += float(mrt_loss_batch.item()) * len(batch)
                sum_reward += sum(batch_rewards)
                n_seqs += len(batch)

        if (epoch + 1) % 5 == 0 or epoch == n_epochs - 1:
            avg_ctc = sum_ctc / max(n_seqs, 1)
            avg_mrt = sum_mrt / max(n_seqs, 1)
            avg_r   = sum_reward / max(n_seqs, 1)
            print(f"    epoch {epoch+1:3d}/{n_epochs}  "
                  f"ctc={avg_ctc:.4f}  mrt={avg_mrt:+.4f}  "
                  f"mean_reward={avg_r:.3f}")
    return model


# ═════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ═════════════════════════════════════════════════════════════════════════════

def evaluate_ctc(model, vocab, test_ds):
    """For each test item: decode → predicted phoneme sequence. Score per-
    sentence vs MFA target. Aggregate per patient."""
    model.eval()
    per_pid = defaultdict(lambda: {
        'true': [], 'pred': [], 'true_sids': [], 'pred_sids': [], 'n_sent': 0,
        'sent_eds': [], 'sent_lens': [],
    })

    with torch.no_grad():
        for item in test_ds:
            pid = item['pid']
            sent_idx = item['sentence_idx']
            X = torch.from_numpy(item['eeg'].astype(np.float32)
                                  ).unsqueeze(0).to(DEVICE)
            logits = model(X, pid)[0]
            log_probs = F.log_softmax(logits, dim=-1)
            pred_idx = decode(log_probs)
            pred_seq = vocab.decode(pred_idx)
            true_seq = [vocab.itos[i] for i in item['target_idx']]

            per_pid[pid]['true'].extend(true_seq)
            per_pid[pid]['pred'].extend(pred_seq)
            per_pid[pid]['true_sids'].extend([sent_idx] * len(true_seq))
            per_pid[pid]['pred_sids'].extend([sent_idx] * len(pred_seq))
            per_pid[pid]['n_sent'] += 1
            per_pid[pid]['sent_eds'].append(edit_distance(true_seq, pred_seq))
            per_pid[pid]['sent_lens'].append(len(true_seq))

    summary = {}
    print(f"\n  CTC evaluation:")
    print(f"  {'pid':<5} {'n_sent':>7} {'true_n':>7} {'pred_n':>7} "
          f"{'edit':>6} {'PER':>8} {'len_ratio':>10}")
    print("  " + "-" * 60)
    for pid in sorted(per_pid):
        d = per_pid[pid]
        true = d['true']; pred = d['pred']
        ed = edit_distance(true, pred)
        per = ed / max(len(true), 1)
        len_ratio = len(pred) / max(len(true), 1)
        accuracy = (sum(t == p for t, p in zip(true[:min(len(true), len(pred))],
                                                 pred[:min(len(true), len(pred))]))
                    / max(len(true), 1))
        summary[pid] = {
            'true_labels':       list(true),
            'predictions':       list(pred),
            'true_sentence_ids': list(d['true_sids']),
            'pred_sentence_ids': list(d['pred_sids']),
            'n_sentences':       d['n_sent'],
            'n_test':            len(true),
            'n_pred':            len(pred),
            'edit_distance':     ed,
            'per':               per,
            'accuracy':          accuracy,
            'len_ratio':         len_ratio,
        }
        print(f"  {pid:<5} {d['n_sent']:>7} {len(true):>7} {len(pred):>7} "
              f"{ed:>6} {per:>7.2%} {len_ratio:>9.2f}×")
    print("  " + "-" * 60)
    print(f"  Mean PER: {np.mean([s['per'] for s in summary.values()]):.2%}")
    print(f"  Mean len_ratio: {np.mean([s['len_ratio'] for s in summary.values()]):.2f}×")
    return summary


# ═════════════════════════════════════════════════════════════════════════════
# PIPELINE-STEP API — same shape as run_path_detector
# ═════════════════════════════════════════════════════════════════════════════

def run_path_ctc(pipeline, run_config=None):
    """Train (or load) the CTC model, evaluate on held-out sentences,
    populate pipeline.patient_results. Schema is compatible with
    e2e_brain_decoder.show_all_patients / show_matched_sequences."""
    print("\n[1/4] Building per-sentence dataset (corrected slicing)...")
    full_ds = build_ctc_dataset(pipeline, ALL_PIDS)
    train_ds, test_ds = split_by_sentence(full_ds)

    # Z-score EEG per-patient using train stats only (leak-safe).
    # apply_stats also normalizes MFCC; CTC doesn't use it, but compute the
    # stats so the function call doesn't crash.
    eeg_stats  = fit_train_stats(train_ds, 'eeg')
    mfcc_stats = fit_train_stats(train_ds, 'mfcc')
    apply_stats(train_ds, eeg_stats, mfcc_stats)
    apply_stats(test_ds,  eeg_stats, mfcc_stats)

    print(f"  train={len(train_ds)}  test={len(test_ds)}")

    print("\n[2/4] Building vocab from train MFA sequences...")
    vocab = build_vocab_from_train(train_ds, min_count=MIN_CLASS_SAMPLES)
    print(f"  vocab size: {len(vocab)} (incl blank)")
    print(f"  example phonemes: {vocab.itos[1:11]}")

    attach_target_indices(train_ds, vocab)
    attach_target_indices(test_ds,  vocab)

    print("\n[3/4] Loading or training CTC model...")
    # Prefer best-val checkpoint over final or periodic ones if present
    best_ckpt = f'{CKPT_PREFIX}best.pt'
    if os.path.exists(best_ckpt):
        ckpts = [best_ckpt]
    else:
        ckpts = sorted(glob.glob(f'{CKPT_PREFIX}*.pt'))
    resume_model = None
    epochs_done  = 0
    if ckpts:
        ckpt_path = ckpts[-1]
        print(f"  Found existing checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        epochs_done = int(ckpt.get('epochs_done', 0))
        # Vocab in ckpt is authoritative — reload it so we don't add/drop
        # phonemes mid-training
        vocab.itos = ckpt['vocab_itos']
        vocab.stoi = {p: i for i, p in enumerate(vocab.itos)}
        attach_target_indices(train_ds, vocab)
        attach_target_indices(test_ds,  vocab)
        resume_model = FrameCTCModel(
            per_patient_eeg_n_ch=ckpt['per_patient_eeg_n_ch'],
            vocab_size=ckpt['vocab_size'],
            hidden_dim=HIDDEN_DIM, n_layers=N_LSTM_LAYERS,
            dropout=DROPOUT, proj_dim=PROJ_DIM,
        ).to(DEVICE)
        resume_model.load_state_dict(ckpt['model_state'])
        print(f"  loaded ({sum(p.numel() for p in resume_model.parameters())/1e6:.2f}M params), "
              f"trained for {epochs_done} epochs")

    if epochs_done < N_EPOCHS_TOTAL:
        if resume_model is None:
            print(f"  No {CKPT_PREFIX} checkpoint — training from scratch")
        else:
            print(f"  Continuing training from epoch {epochs_done} -> {N_EPOCHS_TOTAL}")

        # Carve out a deterministic 10% val split from train_ds
        val_size = int(len(train_ds) * VAL_FRACTION)
        if val_size > 0:
            rng = random.Random(13)
            shuffled = list(train_ds)
            rng.shuffle(shuffled)
            val_split = shuffled[:val_size]
            train_split = shuffled[val_size:]
            print(f"  train/val split: {len(train_split)} / {len(val_split)}")
        else:
            val_split = []
            train_split = train_ds

        model, epochs_done = train_ctc(
            train_split, vocab, n_epochs_total=N_EPOCHS_TOTAL,
            resume_model=resume_model, resume_epoch=epochs_done,
            val_dataset=val_split,
        )
        out_path = f'{CKPT_PREFIX}{datetime.now().strftime("%Y%m%d_%H%M")}.pt'
        torch.save({
            'model_state':          model.state_dict(),
            'per_patient_eeg_n_ch': {pid: model.eeg_proj[pid].in_features
                                      for pid in model.eeg_proj},
            'vocab_size':           len(vocab),
            'vocab_itos':           vocab.itos,
            'epochs_done':          epochs_done,
        }, out_path)
        print(f"  saved to {out_path} (epochs_done={epochs_done})")
    else:
        model = resume_model
        print(f"  Already at {epochs_done} ≥ target {N_EPOCHS_TOTAL} epochs; skipping training.")

    # ── Optional MRT fine-tuning ───────────────────────────────────────────
    # When MRT_MODE=True, after CTC pretraining is done, fine-tune with
    # mixed CTC + n-gram-coverage policy gradient. Saves to a separate
    # CKPT_PREFIX_MRT so the original CTC v3 weights are preserved.
    if MRT_MODE:
        mrt_ckpts = sorted(glob.glob(f'{CKPT_PREFIX_MRT}*.pt'))
        if mrt_ckpts:
            ckpt = torch.load(mrt_ckpts[-1], map_location=DEVICE)
            model.load_state_dict(ckpt['model_state'])
            print(f"  Loaded existing MRT checkpoint: {mrt_ckpts[-1]}")
        else:
            print(f"\n[3.5/4] MRT fine-tuning starting from CTC v3 weights...")
            model = train_ctc_mrt(model, train_ds, vocab)
            out_path = f'{CKPT_PREFIX_MRT}{datetime.now().strftime("%Y%m%d_%H%M")}.pt'
            torch.save({
                'model_state':          model.state_dict(),
                'per_patient_eeg_n_ch': {pid: model.eeg_proj[pid].in_features
                                          for pid in model.eeg_proj},
                'vocab_size':           len(vocab),
                'vocab_itos':           vocab.itos,
                'mrt_n_samples':        MRT_N_SAMPLES,
                'mrt_ctc_weight':       MRT_CTC_WEIGHT,
                'mrt_min_ngram':        MRT_MIN_NGRAM,
            }, out_path)
            print(f"  saved MRT-finetuned model to {out_path}")
    model.eval()

    print("\n[4/4] Evaluating CTC on held-out sentences...")
    summary = evaluate_ctc(model, vocab, test_ds)

    pipeline.patient_results = summary
    return ('frame_ctc',
            {'ckpt_prefix': CKPT_PREFIX, 'vocab_size': len(vocab),
             'hidden_dim': HIDDEN_DIM, 'n_layers': N_LSTM_LAYERS},
            summary)


# ═════════════════════════════════════════════════════════════════════════════
# LENGTH-BONUS SWEEP — re-evaluate without retraining
# ═════════════════════════════════════════════════════════════════════════════

def sweep_length_bonus(pipeline, bonuses=(0.0, 0.5, 1.0, 1.5, 2.0)):
    """Reload latest checkpoint, rebuild test_ds, and evaluate at several
    DECODE_LENGTH_BONUS values. Prints a summary row per bonus.
    No retraining."""
    global DECODE_LENGTH_BONUS

    # Rebuild test_ds + vocab from latest ckpt
    full_ds = build_ctc_dataset(pipeline, ALL_PIDS)
    train_ds, test_ds = split_by_sentence(full_ds)
    eeg_stats  = fit_train_stats(train_ds, 'eeg')
    mfcc_stats = fit_train_stats(train_ds, 'mfcc')
    apply_stats(train_ds, eeg_stats, mfcc_stats)
    apply_stats(test_ds,  eeg_stats, mfcc_stats)

    ckpts = sorted(glob.glob(f'{CKPT_PREFIX}*.pt'))
    assert ckpts, f"No {CKPT_PREFIX} checkpoint found — train first via run_path_ctc"
    ckpt = torch.load(ckpts[-1], map_location=DEVICE)
    print(f"  Loaded {ckpts[-1]} (epochs_done={ckpt.get('epochs_done', '?')})")

    vocab = Vocab(set())
    vocab.itos = ckpt['vocab_itos']
    vocab.stoi = {p: i for i, p in enumerate(vocab.itos)}
    attach_target_indices(test_ds, vocab)

    model = FrameCTCModel(
        per_patient_eeg_n_ch=ckpt['per_patient_eeg_n_ch'],
        vocab_size=ckpt['vocab_size'],
        hidden_dim=HIDDEN_DIM, n_layers=N_LSTM_LAYERS,
        dropout=DROPOUT, proj_dim=PROJ_DIM,
    ).to(DEVICE)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    results = {}
    print(f"\n  Length-bonus sweep over {list(bonuses)}:")
    print(f"  {'bonus':>6}  {'mean PER':>9}  {'mean len_ratio':>14}")
    print("  " + "-" * 38)
    saved_bonus = DECODE_LENGTH_BONUS
    for b in bonuses:
        DECODE_LENGTH_BONUS = b
        summary = evaluate_ctc(model, vocab, test_ds)
        mean_per = np.mean([s['per']       for s in summary.values()])
        mean_lr  = np.mean([s['len_ratio'] for s in summary.values()])
        results[b] = {'per': mean_per, 'len_ratio': mean_lr,
                      'summary': summary}
        print(f"  {b:>6.2f}  {mean_per:>8.2%}  {mean_lr:>13.2f}×")
    DECODE_LENGTH_BONUS = saved_bonus    # restore
    print("  " + "-" * 38)
    return results


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    # Standalone mode: build pipeline + run CTC end-to-end.
    from dutch_30_pipeline import Dutch30Pipeline
    from dutch_30_feature_extractor import Dutch30FeatureExtractor
    from run_pipeline import DEFAULT_RUN_CONFIG, run_path_b

    run_config = dict(DEFAULT_RUN_CONFIG)
    run_config['use_viterbi']        = True
    run_config['stacking_order']     = 20
    run_config['stacking_step_size'] = 1

    extractor = Dutch30FeatureExtractor()
    pipeline = Dutch30Pipeline(
        dutch30_extractor=extractor, debug_mode=False,
        feature_extraction_method=run_config['feature_extraction_method'],
        use_wav2vec=False,
        subtract_baseline=run_config['subtract_baseline'],
        use_rms_boundaries=False, use_multifeature=False,
    )
    run_path_b(pipeline, run_config)

    name, params, results = run_path_ctc(pipeline, run_config)
