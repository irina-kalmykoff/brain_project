# Converted from normalization_approaches.ipynb

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 1 — Imports + pipeline build (uses cached features if available)
# ═══════════════════════════════════════════════════════════════════════════════

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import sklearn_crfsuite

from dutch_30_pipeline import Dutch30Pipeline
from dutch_30_feature_extractor import Dutch30FeatureExtractor
from dataset_config import Dutch30Config
from run_pipeline import DEFAULT_RUN_CONFIG, run_path_b

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 2 — Build pipeline (uses defaults from your latest commit: pwr_lpf,
#            window=15ms, shift=5ms, stk=20)
# ═══════════════════════════════════════════════════════════════════════════════

config = Dutch30Config()
extractor = Dutch30FeatureExtractor(config=config)
pipeline = Dutch30Pipeline(
    dutch30_extractor=extractor,
    debug_mode=False,
    feature_extraction_method='high_gamma',
    use_wav2vec=False,
    subtract_baseline=False,
    use_rms_boundaries=False,
    use_multifeature=False,
)

run_config = dict(DEFAULT_RUN_CONFIG)
run_config['use_viterbi'] = True
# stacking_order=20, step_size=1 are now the new defaults; no override needed

print("Building pipeline (uses cached features if present)...")
run_path_b(pipeline, run_config)
print(f"  train phonemes: {len(pipeline.train['features'])}")
print(f"  test phonemes:  {len(pipeline.test['features'])}")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 3 — Helper: run CRF with a chosen scaling strategy
# ═══════════════════════════════════════════════════════════════════════════════

def features_to_crf_dict(features):
    """Convert a feature vector (1D array) to a dict format for sklearn-crfsuite."""
    return {f'f{i}': float(v) for i, v in enumerate(features)}


def run_with_scaling(pipeline, scaling_kind):
    """Run CRF per patient using a specific scaling strategy.

    scaling_kind options:
      'standard'  — z-score (subtract mean, divide by std)
      'mean_only' — subtract mean only (no std division)
      'none'      — no scaling at all
    """
    print(f"\n  Running with scaling = '{scaling_kind}'")
    pids = sorted(set(pipeline.train['phoneme_participant_ids']))
    results = {}

    for pid in pids:
        train_mask = [i for i, p in enumerate(pipeline.train['phoneme_participant_ids']) if p == pid]
        test_mask  = [i for i, p in enumerate(pipeline.test ['phoneme_participant_ids']) if p == pid]
        if not train_mask or not test_mask:
            continue

        X_train = np.array([pipeline.train['features'][i] for i in train_mask])
        X_test  = np.array([pipeline.test ['features'][i] for i in test_mask ])
        y_train = [pipeline.train['phoneme_labels'][i] for i in train_mask]
        y_test  = [pipeline.test ['phoneme_labels'][i] for i in test_mask ]

        # ── Apply chosen scaling ────────────────────────────────────────────
        if scaling_kind == 'standard':
            scaler = StandardScaler(with_mean=True, with_std=True).fit(X_train)
            X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)
        elif scaling_kind == 'mean_only':
            scaler = StandardScaler(with_mean=True, with_std=False).fit(X_train)
            X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)
        elif scaling_kind == 'none':
            pass
        else:
            raise ValueError(f"unknown scaling_kind: {scaling_kind}")

        # ── Convert to CRF format (one sequence per patient) ────────────────
        X_train_crf = [[features_to_crf_dict(x) for x in X_train]]
        y_train_crf = [list(y_train)]
        X_test_crf  = [[features_to_crf_dict(x) for x in X_test]]

        # ── Train + predict ────────────────────────────────────────────────
        crf = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            c1=0.1, c2=0.1,
            max_iterations=100,
            all_possible_transitions=True,
        )
        crf.fit(X_train_crf, y_train_crf)
        y_pred = crf.predict(X_test_crf)[0]

        # ── Metrics ────────────────────────────────────────────────────────
        n_correct = sum(p == t for p, t in zip(y_pred, y_test))
        accuracy = n_correct / len(y_test) if y_test else 0.0
        n_classes = len(set(y_test))
        chance = 1.0 / n_classes if n_classes > 0 else 0.0
        lift = accuracy / chance if chance > 0 else 0.0

        results[pid] = {
            'accuracy':    accuracy,
            'lift':        lift,
            'n_classes':   n_classes,
            'predictions': list(y_pred),
            'true_labels': list(y_test),
        }
        print(f"    {pid}: acc={accuracy:.3f}  lift={lift:.2f}×  ({n_classes} classes)")

    accs  = [r['accuracy'] for r in results.values()]
    lifts = [r['lift']     for r in results.values()]
    print(f"  MEAN: acc={np.mean(accs):.3f}  lift={np.mean(lifts):.2f}×")
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 4 — Run all three variants
# ═══════════════════════════════════════════════════════════════════════════════

results = {}
results['standard']  = run_with_scaling(pipeline, 'standard')
results['mean_only'] = run_with_scaling(pipeline, 'mean_only')
results['none']      = run_with_scaling(pipeline, 'none')

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 5 — Per-patient comparison table
# ═══════════════════════════════════════════════════════════════════════════════

variants = ['standard', 'mean_only', 'none']
pids = sorted(next(iter(results.values())).keys())

print("\n" + "="*70)
print("  Per-patient CRF lift — scaling strategy comparison")
print("="*70)
print(f"  {'pid':<6} " + "".join(f"{v:>14}" for v in variants))
print("  " + "-" * 62)
for pid in pids:
    line = f"  {pid:<6} "
    for v in variants:
        lift = results[v].get(pid, {}).get('lift', float('nan'))
        line += f"{lift:>13.2f}×"
    print(line)
print("  " + "-" * 62)
mean_line = f"  {'mean':<6} "
for v in variants:
    mean_lift = np.mean([results[v][pid]['lift'] for pid in pids if pid in results[v]])
    mean_line += f"{mean_lift:>13.2f}×"
print(mean_line)

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 6 — Bar chart per patient + save
# ═══════════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(11, 5))
x = np.arange(len(pids))
width = 0.25
colors = {'standard': 'steelblue', 'mean_only': 'crimson', 'none': 'gray'}

for k, v in enumerate(variants):
    lifts = [results[v].get(pid, {}).get('lift', 0) for pid in pids]
    offset = (k - 1) * width
    ax.bar(x + offset, lifts, width, color=colors[v],
           label=v, edgecolor='black', linewidth=0.5)

ax.axhline(1.0, color='red', ls=':', label='chance')
ax.set_xticks(x); ax.set_xticklabels(pids)
ax.set_ylabel('lift over chance')
ax.set_title('Scaling strategy comparison — does removing std help with pwr_lpf?',
             fontsize=12, fontweight='bold')
ax.grid(alpha=0.3, axis='y')
ax.legend()
plt.tight_layout(); plt.show()

# Save raw results for later inspection (also used by the edit-distance cell)
out_path = f'std_normalization_results_{datetime.now().strftime("%Y%m%d_%H%M")}.pkl'
with open(out_path, 'wb') as f:
    pickle.dump(results, f)
print(f"\n  Saved results to {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 7 — Edit distance between true and predicted phoneme sequences
# ═══════════════════════════════════════════════════════════════════════════════
# A pure-Python Levenshtein implementation, no extra packages needed.
# Edit distance counts the minimum (insertion + deletion + substitution)
# operations to transform one sequence into another.
#
# Phoneme Error Rate (PER) = edit_distance / len(true_sequence)
#   PER = 0 means perfect match
#   PER = 1 means worst case (every phoneme wrong)
#   PER > 1 is possible if predicted sequence has many extra insertions

def edit_distance(s1, s2):
    """Levenshtein edit distance between two sequences."""
    if len(s1) < len(s2):
        return edit_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions    = previous_row[j + 1] + 1
            deletions     = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


# Compute edit distance and PER for each patient under each scaling variant
print("\n" + "="*78)
print("  Edit distance between predicted and true phoneme sequences")
print("="*78)
header = f"  {'pid':<6} " + "".join(f"{v:>22}" for v in variants)
print(header)
sub  = f"  {'':<6} " + "".join(f"{'edit (PER)':>22}" for _ in variants)
print(sub)
print("  " + "-" * (len(header)-2))

for pid in pids:
    line = f"  {pid:<6} "
    for v in variants:
        if pid not in results[v]:
            line += f"{'—':>22}"
            continue
        true = results[v][pid]['true_labels']
        pred = results[v][pid]['predictions']
        ed = edit_distance(true, pred)
        per = ed / len(true) if true else float('nan')
        line += f"   {ed:>4} ({per:.2%})    "
    print(line)

# Aggregate stats
print("  " + "-" * (len(header)-2))
mean_line = f"  {'mean':<6} "
for v in variants:
    eds = []
    pers = []
    for pid in pids:
        if pid in results[v]:
            true = results[v][pid]['true_labels']
            pred = results[v][pid]['predictions']
            eds.append(edit_distance(true, pred))
            pers.append(edit_distance(true, pred) / len(true) if true else 0)
    mean_line += f"   {np.mean(eds):>4.0f} ({np.mean(pers):.2%})    "
print(mean_line)



# ═══════════════════════════════════════════════════════════════════════════════
# Why is P22 unusual? Compare feature distributions across patients
# ═══════════════════════════════════════════════════════════════════════════════
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

PID_OF_INTEREST = 'P22'

pids = sorted(set(pipeline.train['phoneme_participant_ids']))


# ── 1. Per-channel feature std distribution per patient ────────────────────
print("="*72)
print(f"  Per-patient feature std distribution (across all phonemes)")
print("="*72)
print(f"  {'pid':<6} {'n_dim':>6} {'min std':>10} {'median std':>12} "
      f"{'max std':>10} {'max/median':>12}")
print("-"*72)

per_patient_std_stats = {}
for pid in pids:
    idx = [i for i, p in enumerate(pipeline.train['phoneme_participant_ids']) if p == pid]
    X = np.array([pipeline.train['features'][i] for i in idx])
    stds = X.std(axis=0)   # std along axis 0 → one value per feature dim
    per_patient_std_stats[pid] = stds
    star = ' ←★' if pid == PID_OF_INTEREST else ''
    print(f"  {pid:<6} {X.shape[1]:>6} {stds.min():>10.3f} "
          f"{np.median(stds):>12.3f} {stds.max():>10.3f} "
          f"{stds.max()/max(np.median(stds), 1e-9):>11.1f}×{star}")


# ── 2. Visualise std distribution: sorted std per dim, all patients overlaid
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
ax = axes[0]
for pid in pids:
    stds = np.sort(per_patient_std_stats[pid])[::-1]
    color = 'crimson' if pid == PID_OF_INTEREST else 'steelblue'
    alpha = 1.0 if pid == PID_OF_INTEREST else 0.35
    lw = 2.0 if pid == PID_OF_INTEREST else 1.0
    ax.plot(np.arange(len(stds)) / len(stds), stds, color=color,
            alpha=alpha, lw=lw, label=pid if pid == PID_OF_INTEREST else None)
ax.set_xlabel('feature dim (rank, normalised)')
ax.set_ylabel('std across phonemes')
ax.set_title(f'Sorted feature stds — {PID_OF_INTEREST} vs others')
ax.set_yscale('log'); ax.grid(alpha=0.3, which='both'); ax.legend()

# Same on linear y to see absolute spread
ax = axes[1]
for pid in pids:
    stds = np.sort(per_patient_std_stats[pid])[::-1]
    color = 'crimson' if pid == PID_OF_INTEREST else 'steelblue'
    alpha = 1.0 if pid == PID_OF_INTEREST else 0.35
    lw = 2.0 if pid == PID_OF_INTEREST else 1.0
    ax.plot(np.arange(len(stds)) / len(stds), stds, color=color,
            alpha=alpha, lw=lw)
ax.set_xlabel('feature dim (rank, normalised)')
ax.set_ylabel('std across phonemes')
ax.set_title('Same on linear y')
ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()


# ── 3. Per-class accuracy: which phonemes does P22 do well/badly on?
def per_class_accuracy(true, pred):
    """Return {class: (accuracy, support)} dict."""
    classes = sorted(set(true))
    out = {}
    for c in classes:
        mask = [t == c for t in true]
        if not any(mask):
            continue
        n = sum(mask)
        correct = sum(p == t for p, t, m in zip(pred, true, mask) if m)
        out[c] = (correct / n, n)
    return out

# Need the predictions/labels from your three scaling variants
# Assumes you saved 'results' dict from std_normalization_test.py
print("\n"+"="*72)
print(f"  {PID_OF_INTEREST} per-class accuracy across scaling variants")
print("="*72)

if 'results' in dir():    # if std test results are still in scope
    if PID_OF_INTEREST in results['standard']:
        true_p22 = results['standard'][PID_OF_INTEREST]['true_labels']
        all_classes = sorted(set(true_p22))
        print(f"  {'class':<8} {'support':>8} {'standard':>10} "
              f"{'mean_only':>10} {'none':>10}")
        print("-"*60)
        for cls in all_classes:
            row = f"  {cls:<8}"
            n = sum(t == cls for t in true_p22)
            row += f" {n:>8}"
            for variant in ['standard', 'mean_only', 'none']:
                pca = per_class_accuracy(
                    results[variant][PID_OF_INTEREST]['true_labels'],
                    results[variant][PID_OF_INTEREST]['predictions'])
                acc = pca.get(cls, (0, 0))[0]
                row += f" {acc:>9.0%}"
            print(row)


# ── 4. Class distribution: is P22 unusually skewed?
print(f"\n  {PID_OF_INTEREST} class distribution:")
true_p22 = pipeline.patient_results[PID_OF_INTEREST]['true_labels']
counts = Counter(true_p22)
total = len(true_p22)
print(f"  {'class':<8} {'count':>6} {'fraction':>10}")
for cls, n in counts.most_common():
    print(f"  {cls:<8} {n:>6} {n/total:>9.1%}")
print(f"  TOTAL    {total:>6}  ({len(counts)} unique classes)")

from collections import Counter
def confusion(true, pred, true_class):
    return Counter(p for t, p in zip(true, pred) if t == true_class).most_common()

print("What does the model predict when the truth is /z/?")
for variant in ['standard', 'mean_only', 'none']:
    r = results[variant]['P22']
    c = confusion(r['true_labels'], r['predictions'], 'z')
    print(f"  {variant:10}: {c[:5]}")

from collections import Counter

VOICING_PAIRS = [
    ('z', 's'), ('v', 'f'), ('b', 'p'), ('d', 't'), ('ɣ', 'x'),
]

def confusion(true, pred, true_class):
    return Counter(p for t, p in zip(true, pred) if t == true_class).most_common(5)

for voiced, unvoiced in VOICING_PAIRS:
    print(f"\n  /{voiced}/ vs /{unvoiced}/  (voicing pair):")
    for variant in ['standard', 'mean_only', 'none']:
        r = results[variant]['P22']
        true = r['true_labels']
        pred = r['predictions']
        n_voiced = sum(t == voiced for t in true)
        if n_voiced == 0:
            continue
        # Top confusions for the voiced phoneme
        conf = confusion(true, pred, voiced)
        # Cross-confusion: how often is voiced predicted as unvoiced?
        as_unvoiced = sum(1 for t, p in zip(true, pred) if t == voiced and p == unvoiced)
        as_correct = sum(1 for t, p in zip(true, pred) if t == voiced and p == voiced)
        print(f"    {variant:10}: support={n_voiced:>3}  "
              f"correct={as_correct:>3}  predicted-as-unvoiced={as_unvoiced:>3}  "
              f"top-5: {conf}")
