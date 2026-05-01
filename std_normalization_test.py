# Test whether removing the std-division step (z-score → mean-only centering)
# improves or degrades CRF performance after the pwr_lpf envelope change.
#
# We test three feature scaling strategies, all on the SAME features built
# with the new pipeline:
#   1. 'standard'  — full z-score (subtract mean, divide by std)
#   2. 'mean_only' — subtract mean only (no std division)
#   3. 'none'      — no scaling (raw features)
#
# The previous test (before the envelope change) showed std normalization
# helped slightly. With the cleaner pwr_lpf envelope, this might have
# changed: features are more uniform across phonemes now, so per-channel
# std rescaling might be redundant or even harmful.

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

print("""
  Reading the table:
    edit  = number of substitutions/insertions/deletions to transform the
            predicted sequence into the true sequence
    PER   = phoneme error rate = edit / len(true);
            0% is perfect, 100% means every phoneme wrong (and no extras)

  Note that edit distance can be larger than the number of phonemes if the
  Viterbi-decoded prediction has insertion/deletion errors (different
  number of predicted vs true phonemes). With the current pipeline, the
  predicted sequence has the same length as the true sequence (one
  prediction per phoneme), so PER ≈ 1 - accuracy in most cases.
""")
