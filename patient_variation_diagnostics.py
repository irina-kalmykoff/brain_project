# Three-part diagnostic to understand why scaling sensitivity varies across
# patients (as observed in std_normalization_test.py).
#
#   PART A — Recording-quality / noise metrics per patient
#   PART B — Electrode-location coverage comparison
#   PART C — Phoneme-distribution comparison across training sets
#
# Goal: explain why P22 (+1.76× from removing scaling), P30 (+1.01×), P24
# (+0.91×) benefit so much, while P21 (-0.78×) actually regresses.
#
# Hypotheses to test:
#   • Recording quality differs (P21 noisy → scaling helps; P22 clean → not needed)
#   • Electrode coverage differs (P22 has more laryngeal/speech-motor coverage)
#   • Class distribution differs (P22 has more amplitude-distinctive phonemes)
#
# Run cells in order. Reuses pipeline.train from your live session for PART C.


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 1 — Imports + patient grouping
# ═══════════════════════════════════════════════════════════════════════════════

import os
import numpy as np
import pandas as pd
import scipy.signal
import scipy.stats
import matplotlib.pyplot as plt
from collections import Counter

from config import DUTCH_30_PATH

PIDS = [f'P{i:02d}' for i in range(21, 31)]
SR = 1024

# Group patients by their scaling-sensitivity from std_normalization_test.py
GROUPS = {
    'huge gain':   ['P22', 'P30'],          # Δ > +1.0×
    'big gain':    ['P24'],                 # Δ ≈ +0.9×
    'modest gain': ['P23', 'P28', 'P26'],   # +0.3–0.7×
    'flat':        ['P25', 'P27', 'P29'],   # |Δ| ≤ 0.3×
    'regression':  ['P21'],                 # Δ < 0
}
GROUP_OF = {pid: g for g, pids in GROUPS.items() for pid in pids}
GROUP_COLOR = {
    'huge gain':   'darkgreen',
    'big gain':    'seagreen',
    'modest gain': 'goldenrod',
    'flat':        'gray',
    'regression':  'crimson',
}


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 2 — PART A: Compute recording-quality metrics per patient
# ═══════════════════════════════════════════════════════════════════════════════
# Six metrics, designed to capture different kinds of "noisiness":
#
#   median_std         — typical amplitude scale (hardware-related)
#   max/median std     — channel imbalance (extreme channels)
#   n_outlier_chans    — channels with std > 5× median (likely artifact)
#   median_kurtosis    — tail heaviness (high = spiky transients)
#   line_noise_50      — residual energy at 50 Hz vs quiet band
#   drift              — std variation over time chunks (electrode movement)
#   mean_abs_corr      — cross-channel correlation (shared noise from CAR-needed)
#   hf_floor           — energy at 200-300 Hz (broadband electronic noise)

def compute_noise_metrics(pid, n_chunk_for_psd=4096, n_seconds_for_corr=30):
    """Load raw EEG and compute noise/quality metrics."""
    raw = np.load(os.path.join(DUTCH_30_PATH, 'raw', f'{pid}_sEEG.npy'))
    n_samples, n_channels = raw.shape

    # -- Channel statistics --
    channel_stds = raw.std(axis=0)
    median_std = float(np.median(channel_stds))
    max_std    = float(channel_stds.max())

    channel_kurt = scipy.stats.kurtosis(raw, axis=0)
    median_kurt  = float(np.median(channel_kurt))

    n_outlier  = int(np.sum(channel_stds > 5 * median_std))

    # -- Spectral analysis on a representative central channel --
    # Pick the channel with median std (avoid extreme channels)
    mid_chan = int(np.argsort(channel_stds)[n_channels // 2])
    f, Pxx = scipy.signal.welch(raw[:n_seconds_for_corr*SR, mid_chan],
                                fs=SR, nperseg=n_chunk_for_psd)

    def power_at(freq):
        return float(Pxx[np.argmin(np.abs(f - freq))])

    quiet_mask = ((f >= 80) & (f <= 99)) | ((f >= 110) & (f <= 130))
    ref_power  = float(Pxx[quiet_mask].mean())

    line_50  = power_at(50)  / ref_power if ref_power > 0 else float('nan')
    line_100 = power_at(100) / ref_power if ref_power > 0 else float('nan')
    line_150 = power_at(150) / ref_power if ref_power > 0 else float('nan')

    hf_mask = (f >= 200) & (f <= 300)
    hf_floor = float(Pxx[hf_mask].mean()) if hf_mask.sum() > 0 else float('nan')

    # -- Drift over time (10 equal chunks) --
    n_chunks = 10
    cs = n_samples // n_chunks
    chunk_stds = np.array([raw[i*cs:(i+1)*cs].std() for i in range(n_chunks)])
    drift = float(chunk_stds.std() / chunk_stds.mean()) if chunk_stds.mean() > 0 else 0

    # -- Cross-channel correlation (shared noise) --
    sample = raw[:n_seconds_for_corr*SR, :min(30, n_channels)]
    corr = np.corrcoef(sample.T)
    np.fill_diagonal(corr, 0)
    mean_abs_corr = float(np.abs(corr).mean())

    return {
        'n_channels':     n_channels,
        'median_std':     median_std,
        'max_std':        max_std,
        'std_ratio':      max_std / median_std,
        'n_outlier':      n_outlier,
        'median_kurt':    median_kurt,
        'line_50':        line_50,
        'line_100':       line_100,
        'line_150':       line_150,
        'drift':          drift,
        'mean_abs_corr':  mean_abs_corr,
        'hf_floor':       hf_floor,
    }


print("\n  Computing noise metrics for P21–P30 (this takes ~30 s)...")
noise = {}
for pid in PIDS:
    print(f"    {pid}", end=' ', flush=True)
    noise[pid] = compute_noise_metrics(pid)
print('\n  done.\n')


# Print a table organised by group
print("="*112)
print(f"  {'group':<13} {'pid':<5} {'n_ch':>5} {'med_std':>8} "
      f"{'std_ratio':>10} {'n_outl':>7} {'kurt':>6} "
      f"{'line50×':>8} {'drift':>7} {'corr':>6} {'hf':>10}")
print("-"*112)

for group_name in ['huge gain', 'big gain', 'modest gain', 'flat', 'regression']:
    for pid in GROUPS[group_name]:
        n = noise[pid]
        print(f"  {group_name:<13} {pid:<5} {n['n_channels']:>5} "
              f"{n['median_std']:>8.1f} {n['std_ratio']:>10.1f} "
              f"{n['n_outlier']:>7} {n['median_kurt']:>6.1f} "
              f"{n['line_50']:>8.1f} {n['drift']:>7.3f} "
              f"{n['mean_abs_corr']:>6.3f} {n['hf_floor']:>10.2e}")
print("="*112)


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 3 — PART A: Visualise noise metrics — group-coloured scatter
# ═══════════════════════════════════════════════════════════════════════════════
# Plot pairs of metrics with patients coloured by their scaling-sensitivity
# group, looking for separators between high-gain and regression groups.

metric_pairs = [
    ('std_ratio',     'line_50',       'log',    'log'),
    ('drift',         'mean_abs_corr', 'linear', 'linear'),
    ('median_kurt',   'std_ratio',     'log',    'log'),
    ('line_50',       'hf_floor',      'log',    'log'),
]

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for ax, (xkey, ykey, xscale, yscale) in zip(axes.flat, metric_pairs):
    for pid in PIDS:
        x = noise[pid][xkey]
        y = noise[pid][ykey]
        g = GROUP_OF[pid]
        ax.scatter(x, y, s=140, color=GROUP_COLOR[g], alpha=0.85,
                   edgecolor='black', linewidth=0.8,
                   label=g if pid == GROUPS[g][0] else None)
        ax.annotate(pid, (x, y), fontsize=9, xytext=(5, 5),
                    textcoords='offset points')
    ax.set_xscale(xscale); ax.set_yscale(yscale)
    ax.set_xlabel(xkey); ax.set_ylabel(ykey)
    ax.grid(alpha=0.3, which='both')

axes[0, 0].legend(loc='best', fontsize=9)
plt.suptitle('Noise metrics by scaling-sensitivity group (look for clusters)',
             fontsize=12, fontweight='bold')
plt.tight_layout(); plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 4 — PART B: Electrode location comparison
# ═══════════════════════════════════════════════════════════════════════════════
# Each patient has a {pid}_electrode_locations.csv — load all and compare
# coverage across patients. The exact columns depend on your CSV format,
# so the script first inspects, then runs region-aware analysis if a
# 'region' or 'label' column exists.

def load_locations(pid):
    path = os.path.join(DUTCH_30_PATH, 'raw', f'{pid}_electrode_locations.csv')
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


# Inspect format of one CSV
sample = load_locations('P22')
if sample is not None:
    print("\n  Electrode CSV columns:", list(sample.columns))
    print("  First 3 rows:")
    print(sample.head(3).to_string(index=False))
    print()
else:
    print("\n  No electrode_locations.csv found.")


# Try to identify a label/region column automatically
loc_dfs = {pid: load_locations(pid) for pid in PIDS}
loc_dfs = {pid: df for pid, df in loc_dfs.items() if df is not None}

if loc_dfs:
    # Find a column that looks like an anatomical label
    sample_cols = list(next(iter(loc_dfs.values())).columns)
    label_col = None
    for candidate in ['region', 'label', 'anatomy', 'location', 'name',
                      'AAL_label', 'Label', 'Region', 'Anat']:
        if candidate in sample_cols:
            label_col = candidate
            break
    print(f"  Using anatomical label column: {label_col!r}")

    # Per-patient region counts
    print("\n  Per-patient electrode count by region (top 10 regions per patient):")
    for pid in PIDS:
        if pid not in loc_dfs:
            continue
        df = loc_dfs[pid]
        if label_col is None:
            print(f"  {pid:<5}: {len(df)} electrodes (no region info)")
            continue
        counts = df[label_col].value_counts().head(10)
        print(f"\n  {pid:<5} ({GROUP_OF[pid]:>11}): {len(df)} electrodes total")
        for region, n in counts.items():
            print(f"          {region:<40} {n}")

    # Speech-motor cortex coverage check (look for keywords)
    if label_col is not None:
        print("\n" + "="*72)
        print("  Coverage of speech-relevant regions per patient")
        print("="*72)
        SPEECH_KEYWORDS = [
            'precentral', 'central', 'motor', 'sensori', 'roland',
            'temporal', 'super', 'auditory', 'STG', 'MTG',
            'broca', 'opercul', 'inferior frontal',
            'laryng', 'larynx',
        ]
        print(f"  {'pid':<5} {'group':<13} {'speech-relevant electrodes':>30}")
        print("-"*72)
        for pid in PIDS:
            if pid not in loc_dfs:
                continue
            df = loc_dfs[pid]
            labels = df[label_col].astype(str).str.lower()
            speech_mask = labels.apply(
                lambda lbl: any(k in lbl for k in SPEECH_KEYWORDS))
            n_speech = int(speech_mask.sum())
            pct = 100 * n_speech / len(df) if len(df) else 0
            print(f"  {pid:<5} {GROUP_OF[pid]:<13} "
                  f"{n_speech:>4}/{len(df):<4} ({pct:>4.0f}%)")


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 5 — PART B: Spatial scatter of electrodes (if x/y/z columns exist)
# ═══════════════════════════════════════════════════════════════════════════════

if loc_dfs:
    sample = next(iter(loc_dfs.values()))
    coord_cols = None
    for x, y, z in [('x', 'y', 'z'), ('X', 'Y', 'Z'),
                    ('mni_x', 'mni_y', 'mni_z'),
                    ('MNI_x', 'MNI_y', 'MNI_z')]:
        if all(c in sample.columns for c in [x, y, z]):
            coord_cols = (x, y, z)
            break

    if coord_cols:
        x_col, y_col, z_col = coord_cols
        # 3-view layout: sagittal (x,z), axial (x,y), coronal (y,z)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        views = [(x_col, z_col, 'sagittal'),
                 (x_col, y_col, 'axial'),
                 (y_col, z_col, 'coronal')]
        for ax, (a, b, title) in zip(axes, views):
            for pid in PIDS:
                if pid not in loc_dfs:
                    continue
                df = loc_dfs[pid]
                g = GROUP_OF[pid]
                ax.scatter(df[a], df[b], s=18, alpha=0.55,
                           color=GROUP_COLOR[g],
                           label=g if pid == GROUPS[g][0] else None)
            ax.set_title(f'{title} view ({a}, {b})', fontsize=11)
            ax.set_xlabel(a); ax.set_ylabel(b)
            ax.set_aspect('equal'); ax.grid(alpha=0.3)
        axes[0].legend(loc='best', fontsize=9)
        plt.suptitle('Electrode spatial coverage by scaling-sensitivity group',
                     fontsize=12, fontweight='bold')
        plt.tight_layout(); plt.show()
    else:
        print("\n  No 3D coord columns found in the CSVs — skipping spatial scatter.")


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 6 — PART C: Phoneme-distribution comparison across training sets
# ═══════════════════════════════════════════════════════════════════════════════
# Compare each patient's training-set phoneme distribution.  Looking for:
#   • Imbalance (some patients dominated by a few classes?)
#   • Class diversity (some patients with very few unique phonemes?)
#   • Fricative/voicing share (which group has more amplitude-distinctive
#     phonemes — predicting the scaling-removal benefit?)

# Phonemes that are "amplitude-coded" per the voicing-pair analysis
AMPLITUDE_CODED = ['z', 's', 'v', 'f', 'ɣ', 'x', 'ʒ', 'ʃ',  # fricatives
                   'd', 't', 'b', 'p', 'g', 'k']               # plosives
# Vowels are reference (mostly spatial-pattern-coded)
VOWELS = ['a', 'aː', 'e', 'eː', 'i', 'iː', 'o', 'oː', 'u', 'uː',
          'ɑ', 'ɛ', 'ɪ', 'ɔ', 'ʊ', 'ʏ', 'œ', 'øː', 'yː']

per_patient_dist = {}
for pid in PIDS:
    train_idx = [i for i, p in enumerate(pipeline.train['phoneme_participant_ids'])
                 if p == pid]
    labels = [pipeline.train['phoneme_labels'][i] for i in train_idx]
    counts = Counter(labels)
    per_patient_dist[pid] = counts


# Summary table
print("\n" + "="*84)
print("  Per-patient training set phoneme distribution")
print("="*84)
print(f"  {'pid':<5} {'group':<13} {'total':>7} {'classes':>9} "
      f"{'top-3':>20} {'%fric+plos':>12} {'%vowels':>9}")
print("-"*84)

for pid in PIDS:
    counts = per_patient_dist[pid]
    total = sum(counts.values())
    n_classes = len(counts)
    top3 = ' '.join(f"{c}({n})" for c, n in counts.most_common(3))
    n_amp = sum(counts.get(p, 0) for p in AMPLITUDE_CODED)
    n_vowel = sum(counts.get(p, 0) for p in VOWELS)
    pct_amp = 100 * n_amp / total if total else 0
    pct_vow = 100 * n_vowel / total if total else 0
    print(f"  {pid:<5} {GROUP_OF[pid]:<13} {total:>7} {n_classes:>9} "
          f"{top3:>20} {pct_amp:>11.1f}% {pct_vow:>8.1f}%")
print("="*84)


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 7 — Phoneme distribution heatmap (rows = patients, cols = phonemes)
# ═══════════════════════════════════════════════════════════════════════════════
# Visualises which phonemes are over-/under-represented per patient.
# Sort patients by group; sort phonemes by total cohort frequency.

all_phonemes = sorted(set().union(*(c.keys() for c in per_patient_dist.values())))
total_counts = Counter()
for c in per_patient_dist.values():
    total_counts.update(c)
phonemes_by_freq = [p for p, _ in total_counts.most_common()]
patients_by_group = (GROUPS['huge gain'] + GROUPS['big gain'] +
                     GROUPS['modest gain'] + GROUPS['flat'] +
                     GROUPS['regression'])

# Build matrix: each row is a patient's phoneme distribution (as proportions)
matrix = np.zeros((len(patients_by_group), len(phonemes_by_freq)))
for i, pid in enumerate(patients_by_group):
    counts = per_patient_dist[pid]
    total = sum(counts.values())
    for j, p in enumerate(phonemes_by_freq):
        matrix[i, j] = counts.get(p, 0) / total if total else 0

fig, ax = plt.subplots(figsize=(14, 5))
im = ax.imshow(matrix, aspect='auto', cmap='viridis')
ax.set_xticks(range(len(phonemes_by_freq)))
ax.set_xticklabels(phonemes_by_freq, rotation=90, fontsize=8)
ax.set_yticks(range(len(patients_by_group)))
ax.set_yticklabels([f'{pid} ({GROUP_OF[pid]})' for pid in patients_by_group])
ax.set_title('Phoneme proportion by patient (rows grouped by scaling sensitivity)',
             fontsize=11, fontweight='bold')

# Highlight rows by group with coloured y-tick labels
for tick_label, pid in zip(ax.get_yticklabels(), patients_by_group):
    tick_label.set_color(GROUP_COLOR[GROUP_OF[pid]])

plt.colorbar(im, ax=ax, label='proportion of training set')
plt.tight_layout(); plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 8 — Final readout: which dimension correlates with scaling sensitivity?
# ═══════════════════════════════════════════════════════════════════════════════
# Pearson correlation between each metric and the observed scaling-sensitivity
# delta (none − standard).  Whichever metric correlates most strongly is
# the most likely cause.

# These are the deltas you observed
DELTAS = {
    'P21': -0.78, 'P22': +1.76, 'P23': +0.69, 'P24': +0.91, 'P25': +0.13,
    'P26': +0.27, 'P27': +0.27, 'P28': +0.37, 'P29': -0.09, 'P30': +1.01,
}

# Build feature matrix
feature_names = ['median_std', 'std_ratio', 'n_outlier', 'median_kurt',
                 'line_50', 'drift', 'mean_abs_corr', 'hf_floor']
deltas = np.array([DELTAS[pid] for pid in PIDS])

print("\n" + "="*72)
print("  Pearson correlation of each metric with scaling-sensitivity Δ")
print("="*72)
print(f"  {'metric':<20} {'r':>10} {'sign':<8} {'interpretation':<35}")
print("-"*72)

for name in feature_names:
    values = np.array([noise[pid][name] for pid in PIDS])
    if np.std(values) == 0:
        continue
    r, p = scipy.stats.pearsonr(values, deltas)
    sign = '+' if r > 0 else '−'
    if abs(r) > 0.6:
        interp = '★ strong correlation'
    elif abs(r) > 0.4:
        interp = '  moderate correlation'
    elif abs(r) > 0.2:
        interp = '  weak correlation'
    else:
        interp = '  no correlation'
    print(f"  {name:<20} {r:>+10.3f}  ({sign}) {interp}")

# Phoneme-distribution correlations
print(f"\n  {'metric':<20} {'r':>10} {'sign':<8} {'interpretation':<35}")
print("-"*72)
for metric_name, key in [('%amplitude_coded', AMPLITUDE_CODED),
                          ('%vowels',          VOWELS),
                          ('total_phonemes',   None),
                          ('n_classes',        None)]:
    values = []
    for pid in PIDS:
        c = per_patient_dist[pid]
        total = sum(c.values())
        if metric_name == 'total_phonemes':
            v = total
        elif metric_name == 'n_classes':
            v = len(c)
        else:
            v = sum(c.get(p, 0) for p in key) / max(total, 1)
        values.append(v)
    values = np.array(values)
    if np.std(values) == 0:
        continue
    r, p = scipy.stats.pearsonr(values, deltas)
    sign = '+' if r > 0 else '−'
    if abs(r) > 0.6:
        interp = '★ strong correlation'
    elif abs(r) > 0.4:
        interp = '  moderate correlation'
    elif abs(r) > 0.2:
        interp = '  weak correlation'
    else:
        interp = '  no correlation'
    print(f"  {metric_name:<20} {r:>+10.3f}  ({sign}) {interp}")

print("""
  How to read this:
    • r > 0  : higher metric value → more positive scaling-sensitivity Δ
    • r < 0  : higher metric value → more negative Δ (i.e. scaling helps more)
    • |r| > 0.6 with n=10 patients is borderline significant (p ~ 0.05);
      treat 0.4–0.6 as suggestive, < 0.4 as no evidence.

  Strong NEGATIVE correlation with a noise metric (line_50, drift, hf_floor)
    → noisy patients benefit from scaling (it normalises noise floors).
      Removing scaling hurts them. ← this explains P21's regression.

  Strong POSITIVE correlation with %amplitude_coded
    → patients with more fricatives/plosives benefit from removing scaling
      because the amplitude info matters most for those phonemes. ← this
      explains P22's outlier gain.

  Strong POSITIVE correlation with electrode-coverage metrics (computed in
  CELL 4) would be the anatomical explanation.
""")
