# Cross-patient pretraining for phoneme decoding.
#
# Idea: each patient's CRF currently sees only their own ~2000 phonemes. By
# pretraining a SHARED encoder on combined data from all train patients, then
# fine-tuning per patient, every classifier can benefit from ~24× more data.
#
# Architecture:
#   per-patient input projection (handles variable channel count)
#       ↓
#   shared encoder MLP
#       ↓
#   shared classifier head (predicts phoneme labels)
#
# Training phases:
#   PHASE 1 — Pretrain: train all components jointly on COMBINED data from all
#             24 train patients. Per-patient projection lets each patient have
#             its own input layer; everything else is shared.
#
#   PHASE 2 — Fine-tune per test patient: freeze the shared encoder, retrain
#             only that patient's projection + classifier head on their own data.
#             (Or: unfreeze everything with a small learning rate.)
#
# This is a per-phoneme classifier (no sequence/CRF transitions). The lift
# benchmark is a per-patient MLP without pretraining, NOT the full CRF baseline,
# since CRF has Viterbi decoding that this MLP doesn't.
#
# Once this works, the next step is wrapping with a BiLSTM-CRF for sequence
# structure on top of the pretrained features.

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 1 — Imports + setup
# ═══════════════════════════════════════════════════════════════════════════════

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from dutch_30_pipeline import Dutch30Pipeline
from dutch_30_feature_extractor import Dutch30FeatureExtractor
from dataset_config import Dutch30Config
from run_pipeline import DEFAULT_RUN_CONFIG, run_path_b

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"  Using device: {DEVICE}")

torch.manual_seed(37)
np.random.seed(37)


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 2 — Build pipeline (uses cached features) + per-patient feature dict
# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline produces phoneme features at the production w15/s5/stk20 settings.
# Note: features have DIFFERENT dimensions per patient because each has a
# different number of usable channels after exclusion.

config = Dutch30Config()
extractor = Dutch30FeatureExtractor(config=config)
pipeline = Dutch30Pipeline(
    dutch30_extractor=extractor, debug_mode=False,
    feature_extraction_method='high_gamma',
    use_wav2vec=False, subtract_baseline=False,
    use_rms_boundaries=False, use_multifeature=False,
)

run_config = dict(DEFAULT_RUN_CONFIG)
run_config['use_viterbi'] = True
print("  Building pipeline (uses cached features if present)...")
run_path_b(pipeline, run_config)
print(f"  ✓ Train phonemes: {len(pipeline.train['features'])}")
print(f"  ✓ Test phonemes:  {len(pipeline.test['features'])}")

# Group features by patient
def collect_per_patient(split):
    out = defaultdict(lambda: {'X': [], 'y': []})
    for i, pid in enumerate(split['phoneme_participant_ids']):
        out[pid]['X'].append(np.asarray(split['features'][i]))
        out[pid]['y'].append(split['phoneme_labels'][i])
    for pid in out:
        out[pid]['X'] = np.stack(out[pid]['X'])
    return dict(out)

train_per_patient = collect_per_patient(pipeline.train)
test_per_patient  = collect_per_patient(pipeline.test)

PIDS = sorted(train_per_patient.keys())

# Build a single phoneme label vocabulary across ALL patients
all_labels = set()
for pid in PIDS:
    all_labels.update(train_per_patient[pid]['y'])
    if pid in test_per_patient:
        all_labels.update(test_per_patient[pid]['y'])
LABEL_TO_IDX = {l: i for i, l in enumerate(sorted(all_labels))}
IDX_TO_LABEL = {i: l for l, i in LABEL_TO_IDX.items()}
N_CLASSES = len(LABEL_TO_IDX)

print(f"\n  Patients: {PIDS}")
print(f"  Total phoneme classes (shared vocab): {N_CLASSES}")
print(f"\n  Per-patient feature dimensions (vary because of channel exclusions):")
for pid in PIDS:
    n_train = len(train_per_patient[pid]['y'])
    n_test  = len(test_per_patient.get(pid, {'y': []})['y'])
    feat_dim = train_per_patient[pid]['X'].shape[1]
    print(f"    {pid}: feat_dim={feat_dim}, train={n_train}, test={n_test}")


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 3 — PyTorch dataset + cross-patient model
# ═══════════════════════════════════════════════════════════════════════════════

class PhonemeDataset(Dataset):
    """Holds per-patient features + labels with patient ID for routing."""
    def __init__(self, X, y, pid):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.tensor([LABEL_TO_IDX[l] for l in y], dtype=torch.long)
        self.pid = pid

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i], self.pid


def make_combined_loader(per_patient_data, batch_size=128, shuffle=True):
    """Create one loader that yields (features, label, patient_id) from all
    train patients pooled together. The patient_id is used to route through
    the right input projection at forward time."""
    datasets = [PhonemeDataset(d['X'], d['y'], pid)
                for pid, d in per_patient_data.items()]
    combined = torch.utils.data.ConcatDataset(datasets)
    return DataLoader(combined, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=_collate)


def _collate(batch):
    """Group batch items by patient_id so we can apply the right projection."""
    by_pid = defaultdict(lambda: {'X': [], 'y': []})
    for X, y, pid in batch:
        by_pid[pid]['X'].append(X)
        by_pid[pid]['y'].append(y)
    return {pid: {'X': torch.stack(d['X']), 'y': torch.stack(d['y'])}
            for pid, d in by_pid.items()}


class CrossPatientModel(nn.Module):
    """Per-patient input projection → shared encoder → shared classifier."""
    def __init__(self, per_patient_input_dims, hidden_dim=256, n_classes=36,
                 n_layers=3, dropout=0.3):
        super().__init__()
        self.per_patient_input_dims = per_patient_input_dims

        # One input projection per patient
        self.projections = nn.ModuleDict({
            pid: nn.Linear(input_dim, hidden_dim)
            for pid, input_dim in per_patient_input_dims.items()
        })

        # Shared encoder: stack of MLP layers
        layers = []
        for _ in range(n_layers):
            layers += [nn.Linear(hidden_dim, hidden_dim),
                       nn.LayerNorm(hidden_dim),
                       nn.ReLU(),
                       nn.Dropout(dropout)]
        self.encoder = nn.Sequential(*layers)

        # Shared classifier head
        self.classifier = nn.Linear(hidden_dim, n_classes)

    def forward(self, X, pid):
        h = self.projections[pid](X)
        h = self.encoder(h)
        return self.classifier(h)


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 4 — Phase 1: Pretrain on combined data
# ═══════════════════════════════════════════════════════════════════════════════

def pretrain(model, train_loader, n_epochs=20, lr=1e-3, weight_decay=1e-4):
    """Train all components (per-patient projections + shared encoder + classifier)
    jointly on the combined data."""
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                   weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    print(f"\n  Pretraining on combined data — {n_epochs} epochs...")
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        for batch in train_loader:
            optimizer.zero_grad()
            batch_loss = 0.0
            n_in_batch = 0
            for pid, d in batch.items():
                X = d['X'].to(DEVICE)
                y = d['y'].to(DEVICE)
                logits = model(X, pid)
                loss = F.cross_entropy(logits, y)
                batch_loss += loss * len(y)
                total_correct += (logits.argmax(dim=1) == y).sum().item()
                n_in_batch += len(y)
                total_samples += len(y)
            batch_loss = batch_loss / max(n_in_batch, 1)
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += batch_loss.item() * n_in_batch
        scheduler.step()
        avg_loss = total_loss / max(total_samples, 1)
        train_acc = total_correct / max(total_samples, 1)
        print(f"    epoch {epoch+1:2}/{n_epochs}  loss={avg_loss:.4f}  "
              f"train_acc={train_acc:.3f}")
    return model


# Build per-patient input dim mapping
input_dims = {pid: train_per_patient[pid]['X'].shape[1] for pid in PIDS}

model = CrossPatientModel(per_patient_input_dims=input_dims,
                          hidden_dim=256, n_classes=N_CLASSES,
                          n_layers=3, dropout=0.3)

# Per-patient z-score normalization (using train statistics)
patient_stats = {}
for pid in PIDS:
    mu = train_per_patient[pid]['X'].mean(axis=0, keepdims=True)
    sd = train_per_patient[pid]['X'].std(axis=0, keepdims=True) + 1e-6
    patient_stats[pid] = {'mu': mu, 'sd': sd}
    train_per_patient[pid]['X'] = (train_per_patient[pid]['X'] - mu) / sd
    if pid in test_per_patient:
        test_per_patient[pid]['X']  = (test_per_patient[pid]['X']  - mu) / sd

train_loader = make_combined_loader(train_per_patient, batch_size=128)
model = pretrain(model, train_loader, n_epochs=20, lr=1e-3)


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 5 — Phase 2: per-patient fine-tune + evaluate
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate(model, X, y, pid):
    model.eval()
    with torch.no_grad():
        X = torch.from_numpy(X.astype(np.float32)).to(DEVICE)
        y_idx = torch.tensor([LABEL_TO_IDX[l] for l in y], dtype=torch.long).to(DEVICE)
        logits = model(X, pid)
        preds = logits.argmax(dim=1)
        acc = (preds == y_idx).float().mean().item()
    return acc, preds.cpu().numpy()


def finetune(model, train_X, train_y, pid, n_epochs=10, lr=1e-4):
    """Fine-tune the shared encoder + classifier on one patient's data
    with a small learning rate. Per-patient projection also updates."""
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    X = torch.from_numpy(train_X.astype(np.float32)).to(DEVICE)
    y_idx = torch.tensor([LABEL_TO_IDX[l] for l in train_y], dtype=torch.long).to(DEVICE)

    for epoch in range(n_epochs):
        model.train()
        # Shuffle and use mini-batches
        perm = torch.randperm(len(y_idx))
        batch_size = 128
        for i in range(0, len(perm), batch_size):
            idx = perm[i:i+batch_size]
            optimizer.zero_grad()
            logits = model(X[idx], pid)
            loss = F.cross_entropy(logits, y_idx[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
    return model


print("\n  Phase 2: per-patient fine-tune + evaluate")
print("="*70)

# Save pretrained state so we can fine-tune from the same start each time
import copy
pretrained_state = copy.deepcopy(model.state_dict())

results_pretrain_only  = {}
results_with_finetune  = {}

for pid in PIDS:
    if pid not in test_per_patient:
        print(f"  {pid}: no test data, skipping")
        continue

    test_X = test_per_patient[pid]['X']
    test_y = test_per_patient[pid]['y']

    # ── Evaluation 1: pretrain-only (no fine-tune) ──
    model.load_state_dict(pretrained_state)
    acc_pre, _ = evaluate(model, test_X, test_y, pid)

    # ── Evaluation 2: fine-tune then evaluate ──
    model.load_state_dict(pretrained_state)
    finetune(model,
             train_per_patient[pid]['X'],
             train_per_patient[pid]['y'],
             pid, n_epochs=10, lr=1e-4)
    acc_ft, _ = evaluate(model, test_X, test_y, pid)

    n_classes_pid = len(set(test_y))
    chance = 1.0 / n_classes_pid if n_classes_pid > 0 else 0
    lift_pre = acc_pre / chance if chance > 0 else 0
    lift_ft  = acc_ft  / chance if chance > 0 else 0

    results_pretrain_only[pid] = {'accuracy': acc_pre, 'lift': lift_pre,
                                   'n_classes': n_classes_pid}
    results_with_finetune[pid] = {'accuracy': acc_ft,  'lift': lift_ft,
                                   'n_classes': n_classes_pid}
    print(f"  {pid}: pretrain-only acc={acc_pre:.3f} lift={lift_pre:.2f}×    "
          f"+fine-tune acc={acc_ft:.3f} lift={lift_ft:.2f}×")


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 6 — Comparison vs CRF baseline
# ═══════════════════════════════════════════════════════════════════════════════
# CRF baseline lifts from production sweep (window_frameshift_sweep.py).
# Note: CRF has sequence-structure benefit (Viterbi); our MLP doesn't, so a
# fair comparison would be against a per-patient MLP without pretraining.
# We provide the CRF reference for context but the bigger story is whether
# pretraining beats single-patient training in this same MLP architecture.

CRF_BASELINE = {
    'P21': 4.78, 'P22': 5.57, 'P23': 5.37, 'P24': 4.30, 'P25': 5.36,
    'P26': 5.27, 'P27': 5.67, 'P28': 5.60, 'P29': 5.78, 'P30': 6.01,
}

print("\n" + "="*84)
print(f"  Cross-patient pretraining results")
print("="*84)
print(f"  {'pid':<5} {'CRF baseline':>13} "
      f"{'pretrain only':>15} {'+fine-tune':>13} {'Δ vs CRF':>10}")
print("-"*84)
all_lifts_pre, all_lifts_ft = [], []
for pid in PIDS:
    if pid not in results_with_finetune:
        continue
    crf = CRF_BASELINE.get(pid, float('nan'))
    pre = results_pretrain_only[pid]['lift']
    ft  = results_with_finetune[pid]['lift']
    delta = ft - crf if not np.isnan(crf) else float('nan')
    all_lifts_pre.append(pre)
    all_lifts_ft.append(ft)
    print(f"  {pid:<5} {crf:>12.2f}× {pre:>14.2f}× {ft:>12.2f}× {delta:>+9.2f}×")
print("-"*84)
mean_crf = np.mean(list(CRF_BASELINE.values()))
print(f"  {'mean':<5} {mean_crf:>12.2f}× "
      f"{np.mean(all_lifts_pre):>14.2f}× {np.mean(all_lifts_ft):>12.2f}× "
      f"{np.mean(all_lifts_ft) - mean_crf:>+9.2f}×")
print("="*84)


# Bar chart
fig, ax = plt.subplots(figsize=(11, 5))
x = np.arange(len(PIDS))
width = 0.27
crfs = [CRF_BASELINE.get(pid, 0) for pid in PIDS]
pres = [results_pretrain_only[pid]['lift'] if pid in results_pretrain_only else 0
        for pid in PIDS]
fts  = [results_with_finetune[pid]['lift']  if pid in results_with_finetune  else 0
        for pid in PIDS]
ax.bar(x - width, crfs, width, label='CRF baseline (production)',
       color='gray', edgecolor='black')
ax.bar(x,         pres, width, label='Cross-patient pretrain only',
       color='steelblue', edgecolor='black')
ax.bar(x + width, fts,  width, label='Pretrain + fine-tune',
       color='crimson',   edgecolor='black')
ax.axhline(1.0, color='red', ls=':', label='chance')
ax.set_xticks(x); ax.set_xticklabels(PIDS)
ax.set_ylabel('lift over chance')
ax.set_title('Cross-patient pretraining — per-patient lift comparison',
             fontsize=12, fontweight='bold')
ax.legend(); ax.grid(alpha=0.3, axis='y')
plt.tight_layout(); plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 7 — Save results
# ═══════════════════════════════════════════════════════════════════════════════

out_path = f'cross_patient_pretrain_{datetime.now().strftime("%Y%m%d_%H%M")}.pkl'
to_save = {
    'pretrain_only': results_pretrain_only,
    'with_finetune': results_with_finetune,
    'crf_baseline':  CRF_BASELINE,
    'config': {
        'hidden_dim': 256, 'n_layers': 3, 'dropout': 0.3,
        'pretrain_epochs': 20, 'finetune_epochs': 10,
        'pretrain_lr': 1e-3, 'finetune_lr': 1e-4,
    }
}
with open(out_path, 'wb') as f:
    pickle.dump(to_save, f)
print(f"\n  Saved {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Summary at end
# ═══════════════════════════════════════════════════════════════════════════════
# What success looks like:
#
#   Pretrain only > chance: shared representation captures something
#   transferable across patients. Even without fine-tuning, the per-patient
#   projection + shared encoder + shared classifier produces meaningful
#   predictions on held-out test phonemes.
#
#   +Fine-tune > Pretrain only: per-patient adaptation helps. Expected.
#
#   +Fine-tune > CRF baseline: cross-patient pretraining + MLP beats
#   single-patient CRF. THIS is the headline result. If achieved, it
#   means the cross-patient signal is real and large enough to
#   compensate for the loss of CRF's sequence structure.
#
#   +Fine-tune <  CRF baseline: pretraining helps relative to single-patient
#   MLP, but not enough to beat CRF's Viterbi. Next step would be to wrap
#   the pretrained encoder with a CRF or BiLSTM-CRF layer.
#
# Likely outcome based on the data sizes (~2000 phonemes/patient × 24
# patients): pretrain-only ~3-4×, fine-tune ~5-6×. That would put it on par
# with CRF, and adding sequence structure on top would push past it.
#
# If the result is much worse than CRF (e.g., ~3× even after fine-tune),
# the most likely culprits are:
#   - Insufficient pretraining (try more epochs, larger encoder)
#   - Domain shift between patients too severe (per-patient projection not
#     enough; might need patient-conditioned everything)
#   - Hyperparameters off (try lr 5e-4 instead of 1e-3, reduce dropout)
