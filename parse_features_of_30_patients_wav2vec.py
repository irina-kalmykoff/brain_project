# Converted from parse_features_of_30_patients_wav2vec.ipynb

packages = [
    "torch",
    "transformers",
    "numpy",
    "scipy",
    ("sklearn", "sklearn"),  # skip this one
    "librosa",
    "mne",
    "h5py",
]

import sys, platform, importlib
print(f"python: {sys.version}")
print(f"platform: {platform.platform()}\n")

import torch
print(f"torch: {torch.__version__}")
print(f"cuda available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"cuda device: {torch.cuda.get_device_name(0)}")
    print(f"cuda version: {torch.version.cuda}")
print(f"device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

import transformers
print(f"\ntransformers: {importlib.metadata.version('transformers')}")

import numpy; print(f"numpy: {numpy.__version__}")
import scipy; print(f"scipy: {scipy.__version__}")
print(f"sklearn: {importlib.metadata.version('scikit-learn')}")


# ── 1. TORCH FIRST (before anything touches CUDA) ────────────────────────────
import torch
import torchaudio

# ── 2. TRANSFORMERS SECOND (before librosa loads via project imports) ─────────
from transformers import Wav2Vec2Model, Wav2Vec2Processor, Wav2Vec2FeatureExtractor

# ── 3. STANDARD LIBRARIES ───────────────────────────────────────────────────────
import os
import gc
import copy
import glob
import json
import pickle
import tempfile
from datetime import datetime
from collections import Counter, defaultdict
from itertools import combinations

# ── 4. THIRD-PARTY (no CUDA) ──────────────────────────────────────────────────
import numpy as np
import pandas as pd
import scipy.signal
import matplotlib.pyplot as plt
import seaborn as sns
from pynwb import NWBHDF5IO
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, silhouette_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cosine, euclidean
from scipy.signal import decimate

# ── 5. PROJECT IMPORTS ────────────────────────────────────────────────────────
from extract_features import extractHG, stackFeatures, downsampleLabels, extractMelSpecs
from brain_audio_decoder import BrainAudioDecoder
from custom_decoder import CustomBrainAudioDecoder
from brain_audio_decoder_viz import BrainAudioDecoderViz
from acoustic_change_detector import AcousticChangeDetector
from phoneme_validator import PhonemeValidator
from phonetic_dictionary import PhoneticDictionary
from markov_phoneme_model import MarkovPhonemeModel
from config import BIDS_PATH, OUTPUT_PATH, RESULTS_PATH, DUTCH_30_PATH, DUTCH_10_PATH, get_dataset_paths
from dutch_30_pipeline import Dutch30Pipeline
from dutch_30_feature_extractor import Dutch30FeatureExtractor
from phoneme_detection_diagnostic import Dutch30PhonemeDetectionDiagnostic
from dataset_config import Dutch30Config
from experiment_logger import ExperimentLogger

# ── 6. WHISPERX  ──────────────────────────────────────────────────────────
import whisperx

# feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-large-xlsr-53")
# model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-xlsr-53")
# print("Downloaded successfully, hidden size:", model.config.hidden_size)

dutch30_dir = DUTCH_30_PATH
# List all .npy files for one patient
patient_files = glob.glob(os.path.join(dutch30_dir, 'P01*.npy'))
# Check we're using the right paths
print(f"BIDS path: {BIDS_PATH}")
print(f"Output path: {OUTPUT_PATH}")
print(f"Results path: {RESULTS_PATH}")
# Define paths
path_bids = BIDS_PATH # './SingleWordProductionDutch-iBIDS'  # Path to the BIDS dataset
path_output = OUTPUT_PATH #'./features'  # Path to save extracted features
path_results = RESULTS_PATH #'./results'  # Path to save results
paths_30 = get_dataset_paths('dutch30')

    # def run_from_config(pipeline, run_config):
    #     """Run experiment using a unified config dict.
    
    #     Passes all parameters to run_experiment and includes
    #     the full pipeline config in the logged params.
    
    #     Args:
    #         pipeline: Pipeline with train/test data.
    #         run_config: dict with all pipeline and classifier settings.
    
    #     Returns:
    #         Tuple of (name, params, results).
    #     """
    #     name, params, results = run_experiment(
    #         pipeline,
    #         order=run_config.get('markov_order', 1),
    #         class_weight=run_config.get('class_weight', 'balanced'),
    #         use_groups=run_config.get('use_groups', False),
    #         classifier_type=run_config.get('classifier_type', 'random_forest'),
    #         use_viterbi=run_config.get('use_viterbi', False),
    #         stacking_order=run_config.get('stacking_order'),
    #         stacking_step_size=run_config.get('stacking_step_size'),
    #         max_frames=run_config.get('max_frames'),
    #         min_frames=run_config.get('min_frames'),
    #         target_frames=run_config.get('target_frames'),
    #         random_state=run_config.get('random_state', 37),
    #         scaler_type=run_config.get('scaler_type', 'standard'),
    #         subtract_baseline=run_config.get('subtract_baseline', False),
    #         min_class_samples = run_config.get('min_class_samples', 5)
    #     )
    
    #     # Add pipeline config to params so it's all in one place
    #     params['patient_range'] = run_config.get('patient_range')
    #     params['feature_extraction_method'] = run_config.get('feature_extraction_method')
    #     params['subtract_baseline'] = run_config.get('subtract_baseline')
    #     params['sample_fraction'] = run_config.get('sample_fraction')
    
    #     return name, params, results

# python run_pipeline.py                    # Path A (wav2vec)
# python run_pipeline.py --mfa             # Path B (MFA)
# python run_pipeline.py --mfa --sweep     # Path B + hyperparameter sweep
# python run_pipeline.py --mfa --analyze   # Path B + consecutive analysis
# python run_pipeline.py --export-mfa      # Export audio for MFA (one-time)
# python run_pipeline.py --diagnose-mfa    # Show MFA phoneme loss
# python run_pipeline.py --mfa-coverage    # Show alignment coverage
# python run_pipeline.py --patients 1-10   # Different patient range

from run_pipeline import (
    # ── Configuration ─────────────────────────────────────────────────────
    DEFAULT_RUN_CONFIG,          # dict with all default hyperparameters

    # ── Pipeline paths (choose one) ───────────────────────────────────────
    # run_path_a,                # Path A: wav2vec/WhisperX boundary detection
    #                            #   - detects phoneme boundaries from audio in real time
    #                            #   - uses step4_custom_detector + step5_accumulate
    #                            #   - requires WhisperX model loaded (slow, ~1GB RAM)
    #                            #   - 3-level checkpoint system (step5 → frame → step3)
    run_path_b,                  # Path B: MFA pre-aligned TextGrids
                                 #   - reads phoneme timestamps from MFA TextGrid files
                                 #   - bypasses step4 + step5_accumulate entirely
                                 #   - requires mfa_output/ TextGrids to exist already
                                 #   - only needs step3 checkpoint (for train/test split)

    # ── Classification ────────────────────────────────────────────────────
    run_from_config,             # train + evaluate per-patient classifiers (uses run_config)
    # run_experiment,            # same but with explicit keyword args instead of dict

    # ── Analysis & diagnostics ────────────────────────────────────────────
    count,                       # print train/test sample counts
    analyze_consecutive_predictions,  # per-patient consecutive-correct runs + position stats
    # diagnose_mfa_loss,         # show where MFA phonemes are lost (min_samples, missing TG)
    # mfa_coverage_summary,      # per-patient MFA alignment coverage (sentences, phones)

    # ── MFA setup (one-time, already done for P21-P30) ────────────────────
    # export_sentences_for_mfa,  # export .wav + .lab per sentence for MFA input
    # clean_text_for_mfa,        # strip punctuation from transcripts
    # load_mfa_alignments,       # read TextGrid files into dict

    # ── Sweep ─────────────────────────────────────────────────────────────
    run_sweep,                 # grid search over stacking_order, step_size, frames, etc.

    # ── Helpers ───────────────────────────────────────────────────────────
    # attach_whisperx,           # load WhisperX model (only needed for Path A)
    # make_checkpoint_names,     # generate pickle filenames from run_config
)

run_config = dict(DEFAULT_RUN_CONFIG)
# Override if needed:
run_config['use_viterbi'] = True
# run_config['patient_range'] = (21, 30)
run_config['stacking_order'] = 7
run_config['stacking_step_size'] = 1

# ---- Pipeline setup ----
extractor = Dutch30FeatureExtractor()
pipeline = Dutch30Pipeline(
    dutch30_extractor=extractor,
    debug_mode=False,
    feature_extraction_method=run_config['feature_extraction_method'],
    use_wav2vec=False,
    subtract_baseline=run_config['subtract_baseline'],
    use_rms_boundaries=False,
    use_multifeature=False,
)

# ── Run Path B (MFA) ─────────────────────────────────────────────────────────
cached_train, cached_test = run_path_b(pipeline, run_config)

# ── Classification ────────────────────────────────────────────────────────────
# name, params, results = run_from_config(pipeline, run_config)

count(pipeline)

from run_pipeline import _run_crf_experiment
pipeline.patient_results = {}
# Run CRF per patient
crf_results = _run_crf_experiment(pipeline, run_config)

for pid, r in crf_results.items():
    true_labels = r['true_labels']
    acc = r['accuracy']
    from collections import Counter
    # label_counts = Counter(true_labels)
    n_classes = len(set(true_labels))
    chance = 1.0 / n_classes if n_classes > 0 else 0
    lift = acc / chance if chance > 0 else 0

    pipeline.patient_results[pid] = {
        'accuracy': acc,
        'lift': lift,
        'chance': chance,
        'predictions': r['predictions'],
        'true_labels': true_labels,
        'train_size': r['n_train'],
        'test_size': r['n_test'],
        'n_classes': len(set(true_labels)),
    }

pr = run_config['patient_range']
for pid in [f'P{i:02d}' for i in range(pr[0], pr[1] + 1)]:
    if pid in pipeline.patient_results:
        pipeline.step10_visualize_patient(pid, show_table=False)

pipeline.step10_visualize_group()

# ── Step 9: train & evaluate (populates pipeline.patient_results for step 10) ─
results = pipeline.step9_train_and_evaluate(
    model_factory=MarkovPhonemeModel,
    model_params={
        'phonetic_dict': pipeline.detector.phonetic_dict,
        'order':         run_config['markov_order'],
        'use_groups':    False,
        'class_weight':  run_config['class_weight'],
        'classifier_type': run_config['classifier_type'],
        'random_state':  run_config['random_state'],
        'scaler_type':   run_config['scaler_type'],
        'feature_pooling_method': 'flatten',
        'classifier_type': 'random_forest',
    },
    use_viterbi=run_config['use_viterbi'],
)

# ── Step 10: per-patient visualization ────────────────────────────────────────
pr = run_config['patient_range']
for pid in [f'P{i:02d}' for i in range(pr[0], pr[1] + 1)]:
    if pid in pipeline.patient_results:
        pipeline.step10_visualize_patient(pid, show_table=False)

# ── Step 10: group summary ────────────────────────────────────────────────────
pipeline.step10_visualize_group()

from run_pipeline import plot_position_accuracy
import importlib, run_pipeline
plot_position_accuracy(pipeline, run_config)

# Check per-patient counts on this machine
for pid in sorted(set(pipeline.train['phoneme_participant_ids'])):
    train_count = sum(1 for p in pipeline.train['phoneme_participant_ids'] if p == pid)
    test_count = sum(1 for p in pipeline.test['phoneme_participant_ids'] if p == pid)
    print(f"  {pid}: train={train_count}, test={test_count}, total={train_count + test_count}")

# ── Run Path B (MFA) ─────────────────────────────────────────────────────────
cached_train, cached_test = run_path_b(pipeline, run_config)

# ── Sweep over stacking/frame configs ─────────────────────────────────────────
logger = run_sweep(pipeline, run_config, cached_train, cached_test)

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def analyze_sample_size_vs_performance(pipeline, run_config):
    """Analyze and visualize how training sample count affects per-phoneme performance.
    
    Produces 3 figures:
      1. Scatter: train samples vs recall (all patients pooled, one dot per phoneme×patient)
      2. Binned bar chart: rare/medium/common phonemes → mean recall
      3. Per-patient curves: recall vs training count, with trend line
    """
    pr = run_config['patient_range']
    pids = [f'P{i:02d}' for i in range(pr[0], pr[1] + 1)]

    # ── Collect per-phoneme data across all patients ──────────────────
    rows = []  # (pid, phoneme, train_count, test_count, recall, precision, f1)
    for pid in pids:
        if pid not in pipeline.patient_results:
            continue
        res = pipeline.patient_results[pid]
        preds = res['predictions']
        true_labels = res['true_labels']

        tr_mask = [p == pid for p in pipeline.train['phoneme_participant_ids']]
        train_labels = [pipeline.train['phoneme_labels'][i]
                        for i, m in enumerate(tr_mask) if m]
        train_counts = Counter(train_labels)
        test_counts = Counter(true_labels)

        for phoneme in sorted(set(true_labels)):
            true_mask = [l == phoneme for l in true_labels]
            correct = sum(1 for i, m in enumerate(true_mask)
                         if m and preds[i] == phoneme)
            total_true = sum(true_mask)
            recall = correct / total_true if total_true > 0 else 0

            pred_mask = [p == phoneme for p in preds]
            total_pred = sum(pred_mask)
            precision = correct / total_pred if total_pred > 0 else 0
            f1 = (2 * precision * recall / (precision + recall)
                  if (precision + recall) > 0 else 0)

            rows.append({
                'pid': pid,
                'phoneme': phoneme,
                'train_count': train_counts.get(phoneme, 0),
                'test_count': total_true,
                'recall': recall,
                'precision': precision,
                'f1': f1,
            })

    train_counts_arr = np.array([r['train_count'] for r in rows])
    recalls = np.array([r['recall'] for r in rows])
    precisions = np.array([r['precision'] for r in rows])
    f1s = np.array([r['f1'] for r in rows])
    n_classes = len(set(r['phoneme'] for r in rows))
    chance = 1.0 / n_classes if n_classes > 0 else 0

    # ══════════════════════════════════════════════════════════════════
    #  FIGURE 1: Scatter — train count vs recall/precision
    # ══════════════════════════════════════════════════════════════════
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.scatter(train_counts_arr, recalls, alpha=0.4, s=20, c='steelblue',
                edgecolors='none', label='Per phoneme × patient')
    # Trend line
    z = np.polyfit(train_counts_arr, recalls, 1)
    x_fit = np.linspace(0, train_counts_arr.max(), 100)
    ax1.plot(x_fit, np.polyval(z, x_fit), 'k-', lw=2, label='Trend')
    ax1.axhline(chance, color='red', ls=':', lw=1.5,
                label=f'Uniform chance ({chance:.3f})')
    ax1.set_xlabel('Training samples per phoneme', fontsize=11)
    ax1.set_ylabel('Recall', fontsize=11)
    ax1.set_title('Recall vs Training Sample Count', fontsize=12, fontweight='bold')
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend(fontsize=9)

    ax2.scatter(train_counts_arr, precisions, alpha=0.4, s=20, c='darkorange',
                edgecolors='none', label='Per phoneme × patient')
    z2 = np.polyfit(train_counts_arr, precisions, 1)
    ax2.plot(x_fit, np.polyval(z2, x_fit), 'k-', lw=2, label='Trend')
    ax2.axhline(chance, color='red', ls=':', lw=1.5,
                label=f'Uniform chance ({chance:.3f})')
    ax2.set_xlabel('Training samples per phoneme', fontsize=11)
    ax2.set_ylabel('Precision', fontsize=11)
    ax2.set_title('Precision vs Training Sample Count', fontsize=12, fontweight='bold')
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(fontsize=9)

    plt.tight_layout()
    plt.show()

    # ══════════════════════════════════════════════════════════════════
    #  FIGURE 2: Binned bar chart — rare / medium / common
    # ══════════════════════════════════════════════════════════════════
    # Define bins by percentiles of training count
    p33 = np.percentile(train_counts_arr, 33)
    p66 = np.percentile(train_counts_arr, 66)

    bins = {
        f'Rare (≤{p33:.0f})': train_counts_arr <= p33,
        f'Medium ({p33:.0f}–{p66:.0f})': (train_counts_arr > p33) & (train_counts_arr <= p66),
        f'Common (>{p66:.0f})': train_counts_arr > p66,
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    x_pos = np.arange(len(bins))
    width = 0.25
    recall_means, prec_means, f1_means = [], [], []
    recall_stds, prec_stds, f1_stds = [], [], []

    for mask in bins.values():
        recall_means.append(recalls[mask].mean())
        recall_stds.append(recalls[mask].std())
        prec_means.append(precisions[mask].mean())
        prec_stds.append(precisions[mask].std())
        f1_means.append(f1s[mask].mean())
        f1_stds.append(f1s[mask].std())

    ax.bar(x_pos - width, recall_means, width, yerr=recall_stds,
           label='Recall', color='steelblue', capsize=4, alpha=0.85)
    ax.bar(x_pos, prec_means, width, yerr=prec_stds,
           label='Precision', color='darkorange', capsize=4, alpha=0.85)
    ax.bar(x_pos + width, f1_means, width, yerr=f1_stds,
           label='F1', color='seagreen', capsize=4, alpha=0.85)

    ax.axhline(chance, color='red', ls=':', lw=1.5,
               label=f'Uniform chance ({chance:.3f})')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(list(bins.keys()), fontsize=10)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title('Performance by Phoneme Frequency Bin\n(pooled across patients)',
                 fontsize=12, fontweight='bold')
    ax.set_ylim(0, max(max(recall_means), max(prec_means)) * 1.5)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.show()

    # Print bin statistics
    for name, mask in bins.items():
        n = mask.sum()
        print(f"  {name}: n={n}  recall={recalls[mask].mean():.3f}±{recalls[mask].std():.3f}  "
              f"prec={precisions[mask].mean():.3f}±{precisions[mask].std():.3f}  "
              f"f1={f1s[mask].mean():.3f}±{f1s[mask].std():.3f}")

    # ══════════════════════════════════════════════════════════════════
    #  FIGURE 3: Aggregated phoneme view — each phoneme as a dot,
    #            averaged across patients, with phoneme labels
    # ══════════════════════════════════════════════════════════════════
    from collections import defaultdict
    phoneme_agg = defaultdict(lambda: {'train': [], 'recall': [], 'precision': [], 'f1': []})
    for r in rows:
        phoneme_agg[r['phoneme']]['train'].append(r['train_count'])
        phoneme_agg[r['phoneme']]['recall'].append(r['recall'])
        phoneme_agg[r['phoneme']]['precision'].append(r['precision'])
        phoneme_agg[r['phoneme']]['f1'].append(r['f1'])

    fig, ax = plt.subplots(figsize=(10, 6))
    for ph, data in phoneme_agg.items():
        mean_train = np.mean(data['train'])
        mean_recall = np.mean(data['recall'])
        mean_f1 = np.mean(data['f1'])
        ax.scatter(mean_train, mean_recall, s=60, c='steelblue',
                   edgecolors='black', linewidth=0.5, zorder=3)
        ax.annotate(ph, (mean_train, mean_recall),
                    textcoords='offset points', xytext=(5, 5),
                    fontsize=9, fontweight='bold', color='#333')

    ax.axhline(chance, color='red', ls=':', lw=1.5,
               label=f'Uniform chance ({chance:.3f})')

    # Trend line on phoneme averages
    ph_trains = [np.mean(d['train']) for d in phoneme_agg.values()]
    ph_recalls = [np.mean(d['recall']) for d in phoneme_agg.values()]
    z3 = np.polyfit(ph_trains, ph_recalls, 1)
    x_fit3 = np.linspace(0, max(ph_trains), 100)
    ax.plot(x_fit3, np.polyval(z3, x_fit3), 'k--', lw=1.5, alpha=0.6,
            label=f'Linear trend (slope={z3[0]:.4f})')

    ax.set_xlabel('Mean training samples (across patients)', fontsize=11)
    ax.set_ylabel('Mean recall (across patients)', fontsize=11)
    ax.set_title('Per-Phoneme: Training Samples vs Recall\n(each dot = one phoneme, averaged across patients)',
                 fontsize=12, fontweight='bold')
    ax.set_ylim(-0.05, max(ph_recalls) * 1.3)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.show()

    # Correlation
    from scipy.stats import spearmanr, pearsonr
    r_sp, p_sp = spearmanr(train_counts_arr, recalls)
    r_pe, p_pe = pearsonr(train_counts_arr, recalls)
    print(f"\nCorrelation (all phoneme×patient pairs):")
    print(f"  Spearman: r={r_sp:.3f}, p={p_sp:.2e}")
    print(f"  Pearson:  r={r_pe:.3f}, p={p_pe:.2e}")

    r_sp2, p_sp2 = spearmanr(ph_trains, ph_recalls)
    print(f"\nCorrelation (per-phoneme averages):")
    print(f"  Spearman: r={r_sp2:.3f}, p={p_sp2:.2e}")

    return rows

rows = analyze_sample_size_vs_performance(pipeline, run_config)


def plot_patient_metrics_heatmap(pipeline, run_config):
    """Heatmap: metrics (rows) × patients (columns), with row means."""
    import numpy as np
    import matplotlib.pyplot as plt
    from collections import Counter
    from matplotlib.colors import LinearSegmentedColormap

    pr = run_config['patient_range']
    pids = [f'P{i:02d}' for i in range(pr[0], pr[1] + 1)
            if f'P{i:02d}' in pipeline.patient_results]

    metrics_names = ['Accuracy', 'Adj. Accuracy', 'Macro Precision',
                     'Macro Recall', 'Class Coverage']
    matrix = np.zeros((len(metrics_names), len(pids)))

    for col, pid in enumerate(pids):
        res = pipeline.patient_results[pid]
        preds = res['predictions']
        true_labels = res['true_labels']
        acc = res['accuracy']

        true_classes = set(true_labels)
        pred_classes = set(preds)
        n_true = len(true_classes)
        coverage = len(pred_classes & true_classes) / n_true if n_true > 0 else 0
        adj_acc = acc * coverage

        precisions, recalls = [], []
        for ph in sorted(true_classes):
            true_mask = [l == ph for l in true_labels]
            correct = sum(1 for i, m in enumerate(true_mask)
                         if m and preds[i] == ph)
            total_true = sum(true_mask)
            recall = correct / total_true if total_true > 0 else 0

            pred_mask = [p == ph for p in preds]
            total_pred = sum(pred_mask)
            precision = correct / total_pred if total_pred > 0 else 0

            recalls.append(recall)
            precisions.append(precision)

        matrix[0, col] = acc
        matrix[1, col] = adj_acc
        matrix[2, col] = np.mean(precisions)
        matrix[3, col] = np.mean(recalls)
        matrix[4, col] = coverage

    # Add mean column
    means = matrix.mean(axis=1)
    matrix_ext = np.column_stack([matrix, means])
    col_labels = pids + ['Mean']

    # ── Custom colormap: white → blue (avoids dark green problem) ─────
    cmap = LinearSegmentedColormap.from_list(
        'white_blue', ['#ffffff', '#c6dbef', '#6baed6', '#2171b5', '#08306b'])

    fig, ax = plt.subplots(figsize=(len(pids) * 1.1 + 3, len(metrics_names) * 0.8 + 2))

    im = ax.imshow(matrix_ext, cmap=cmap, vmin=0, vmax=1.0, aspect='auto')

    # Annotate cells
    for i in range(len(metrics_names)):
        for j in range(len(col_labels)):
            val = matrix_ext[i, j]
            color = 'white' if val > 0.6 else 'black'
            weight = 'bold' if j == len(pids) else 'normal'  # bold for Mean col
            ax.text(j, i, f'{val:.1%}', ha='center', va='center',
                    fontsize=10, color=color, fontweight=weight)

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=10)
    ax.set_yticks(range(len(metrics_names)))
    ax.set_yticklabels(metrics_names, fontsize=11, fontweight='bold')

    # Separator line before Mean column
    ax.axvline(len(pids) - 0.5, color='black', lw=2)

    n_classes = pipeline.patient_results[pids[0]].get('n_classes', 35)
    chance = 1.0 / n_classes

    ax.set_title(
        f'Patient-Level Metrics Overview\n'
        f'Uniform chance = {chance:.1%} · '
        f'Adj. accuracy = accuracy × class_coverage',
        fontsize=12, pad=12)

    plt.colorbar(im, ax=ax, label='Score', fraction=0.025, pad=0.02)
    plt.tight_layout()
    plt.show()

plot_patient_metrics_heatmap(pipeline, run_config)


# logger.print_table()
logger.best_experiment()

def show_sentences(pipeline, run_config, n_sentences=3):
    """Show example sentences with correct/wrong phoneme predictions per patient.
    
    Uses predictions already stored in pipeline.patient_results (from CRF,
    random_forest, or whatever classifier was last run via step9).
    
    Correct = phoneme symbol, Wrong = *
    Example: de[d-*] kat[k-ɑ-*] zit[*-ɪ-t]
    """
    import numpy as np

    pr   = run_config['patient_range']
    pids = [f'P{i:02d}' for i in range(pr[0], pr[1] + 1)]

    for pid in pids:
        if pid not in pipeline.patient_results:
            continue

        res = pipeline.patient_results[pid]
        preds       = res['predictions']
        true_labels = res['true_labels']
        acc         = res['accuracy']

        # Get word labels + true labels for test set (unfiltered)
        te_idx = [i for i, p in enumerate(pipeline.test['phoneme_participant_ids']) if p == pid]
        w_te_all = [pipeline.test['phoneme_words'][i] for i in te_idx]
        y_te_all = [pipeline.test['phoneme_labels'][i] for i in te_idx]

        # Match filtered predictions back to words:
        # patient_results may have fewer samples (e.g. CRF drops rare classes).
        # Align by matching the true_labels sequence against the full label list.
        if len(w_te_all) != len(preds):
            # Rebuild word list keeping only samples whose label is in true_labels set
            # Walk both lists in order to align
            w_te = []
            j = 0  # pointer into true_labels
            for i in range(len(y_te_all)):
                if j < len(true_labels) and y_te_all[i] == true_labels[j]:
                    w_te.append(w_te_all[i])
                    j += 1
            if len(w_te) != len(preds):
                print(f"{pid}: could not align words ({len(w_te)}) to predictions ({len(preds)}), skipping")
                continue
        else:
            w_te = w_te_all

        def render(words, true, pred):
            """Group phonemes by word, show symbol if correct else *."""
            result = []
            prev_word = None
            buf = []
            for word, t, p in zip(words, true, pred):
                symbol = t if p == t else '*'
                if word != prev_word:
                    if prev_word is not None:
                        result.append(f"{prev_word}[{'-'.join(buf)}]")
                    buf = [symbol]
                    prev_word = word
                else:
                    buf.append(symbol)
            if prev_word is not None:
                result.append(f"{prev_word}[{'-'.join(buf)}]")
            return result

        rendered = render(w_te, true_labels, preds)

        def chunk_words(rendered, n_sent, words_per_sent=8):
            out = []
            for i in range(0, min(len(rendered), n_sent * words_per_sent), words_per_sent):
                out.append(' '.join(rendered[i:i + words_per_sent]))
                if len(out) >= n_sent:
                    break
            return out

        n_classes = res.get('n_classes', len(set(true_labels)))
        chance = 1.0 / n_classes if n_classes > 0 else 0
        lift = acc / chance if chance > 0 else 0

        print(f"\n{'='*70}")
        print(f"{pid}  acc={acc:.1%}  lift={lift:.1f}x  ({n_classes} classes)")
        print(f"  TEST predictions (phoneme=correct, *=wrong):")
        for s in chunk_words(rendered, n_sentences):
            print(f"    {s}")

show_sentences(pipeline, run_config, n_sentences=10)

# more extensive version with train set predictions
# def show_sentences(pipeline, run_config, n_sentences=3):
#     """Show example sentences with correct/wrong phoneme predictions per patient.
    
#     Uses predictions from pipeline.patient_results for TEST.
#     Re-predicts on train data using stored model (step9) or a quick LogReg fallback.
    
#     Correct = phoneme symbol, Wrong = *
#     Example: de[d-*] kat[k-ɑ-*] zit[*-ɪ-t]
#     """
#     import numpy as np
#     from sklearn.preprocessing import StandardScaler, LabelEncoder
#     from sklearn.linear_model import LogisticRegression

#     pr   = run_config['patient_range']
#     pids = [f'P{i:02d}' for i in range(pr[0], pr[1] + 1)]

#     for pid in pids:
#         if pid not in pipeline.patient_results:
#             continue

#         res = pipeline.patient_results[pid]
#         te_preds    = res['predictions']
#         te_true     = res['true_labels']
#         te_acc      = res['accuracy']

#         # ── Gather raw data for this patient ──────────────────────────
#         tr_idx = [i for i, p in enumerate(pipeline.train['phoneme_participant_ids']) if p == pid]
#         te_idx = [i for i, p in enumerate(pipeline.test['phoneme_participant_ids']) if p == pid]
#         if not tr_idx or not te_idx:
#             continue

#         tr_lbl_all = [pipeline.train['phoneme_labels'][i] for i in tr_idx]
#         tr_wrd_all = [pipeline.train['phoneme_words'][i] for i in tr_idx]

#         te_lbl_all = [pipeline.test['phoneme_labels'][i] for i in te_idx]
#         te_wrd_all = [pipeline.test['phoneme_words'][i] for i in te_idx]

#         # ── Align test words to (possibly filtered) predictions ───────
#         if len(te_wrd_all) != len(te_preds):
#             w_te, j = [], 0
#             for i in range(len(te_lbl_all)):
#                 if j < len(te_true) and te_lbl_all[i] == te_true[j]:
#                     w_te.append(te_wrd_all[i])
#                     j += 1
#             if len(w_te) != len(te_preds):
#                 print(f"{pid}: could not align words to predictions, skipping")
#                 continue
#         else:
#             w_te = te_wrd_all

#         # ── Get train predictions ─────────────────────────────────────
#         # Try using stored model; fall back to quick LogReg
#         model = res.get('model')
#         tr_feat_all = [pipeline.train['features'][i] for i in tr_idx]

#         if model is not None and hasattr(model, 'predict'):
#             try:
#                 tr_preds_raw, _ = model.predict(tr_feat_all, use_viterbi=False)
#                 tr_preds = [str(p) for p in tr_preds_raw]
#                 tr_lbl = tr_lbl_all
#                 tr_wrd = tr_wrd_all
#             except Exception:
#                 model = None  # fall through to LogReg

#         if model is None:
#             # Quick LogReg on same label set as test predictions
#             valid = set(te_true)
#             keep_tr = [i for i, l in enumerate(tr_lbl_all) if l in valid]
#             tr_feat = [tr_feat_all[i] for i in keep_tr]
#             tr_lbl  = [tr_lbl_all[i] for i in keep_tr]
#             tr_wrd  = [tr_wrd_all[i] for i in keep_tr]

#             X_tr = np.array([np.array(f).flatten() for f in tr_feat])
#             scaler = StandardScaler()
#             X_tr_s = scaler.fit_transform(X_tr)
#             le = LabelEncoder()
#             y_tr_e = le.fit_transform(tr_lbl)

#             clf = LogisticRegression(max_iter=1000, class_weight='balanced',
#                                      C=1.0, random_state=37)
#             clf.fit(X_tr_s, y_tr_e)
#             tr_preds = [le.classes_[i] for i in clf.predict(X_tr_s)]

#         tr_acc = sum(p == t for p, t in zip(tr_preds, tr_lbl)) / len(tr_lbl)

#         # ── Render ────────────────────────────────────────────────────
#         def render(words, true, pred):
#             result, prev_word, buf = [], None, []
#             for word, t, p in zip(words, true, pred):
#                 symbol = t if p == t else '*'
#                 if word != prev_word:
#                     if prev_word is not None:
#                         result.append(f"{prev_word}[{'-'.join(buf)}]")
#                     buf = [symbol]
#                     prev_word = word
#                 else:
#                     buf.append(symbol)
#             if prev_word is not None:
#                 result.append(f"{prev_word}[{'-'.join(buf)}]")
#             return result

#         tr_rendered = render(tr_wrd, tr_lbl, tr_preds)
#         te_rendered = render(w_te, te_true, te_preds)

#         def chunk_words(rendered, n_sent, words_per_sent=8):
#             out = []
#             for i in range(0, min(len(rendered), n_sent * words_per_sent), words_per_sent):
#                 out.append(' '.join(rendered[i:i + words_per_sent]))
#                 if len(out) >= n_sent:
#                     break
#             return out

#         n_classes = res.get('n_classes', len(set(te_true)))
#         chance = 1.0 / n_classes if n_classes > 0 else 0
#         lift = te_acc / chance if chance > 0 else 0

#         print(f"\n{'='*70}")
#         print(f"{pid}  train_acc={tr_acc:.1%}  test_acc={te_acc:.1%}  "
#               f"lift={lift:.1f}x  ({n_classes} classes)")
#         print(f"  TRAIN (phoneme=correct, *=wrong):")
#         for s in chunk_words(tr_rendered, n_sentences):
#             print(f"    {s}")
#         print(f"  TEST:")
#         for s in chunk_words(te_rendered, n_sentences):
#             print(f"    {s}")

# show_sentences(pipeline, run_config, n_sentences=5)

import importlib, run_pipeline, markov_phoneme_model
importlib.reload(markov_phoneme_model)
importlib.reload(run_pipeline)
from run_pipeline import compare_classifiers, plot_classifier_heatmap

comparison = compare_classifiers(pipeline, run_config)

plot_classifier_heatmap(comparison)                          # adjusted accuracy (default)
# plot_classifier_heatmap(comparison, metric='accuracy')     # raw accuracy
# plot_classifier_heatmap(comparison, metric='class_coverage')  # class coverage only

# Quick summary of alignment coverage across all patients
import os
from config import DUTCH_30_PATH

mfa_output = os.path.join(DUTCH_30_PATH, 'mfa_output')
mfa_input  = os.path.join(DUTCH_30_PATH, 'mfa_input')

pr = (21, 30)
for pid in [f'P{i:02d}' for i in range(pr[0], pr[1]+1)]:
    tg_dir  = os.path.join(mfa_output, pid)
    lab_dir = os.path.join(mfa_input, pid)
    
    n_lab = len([f for f in os.listdir(lab_dir) if f.endswith('.lab')]) if os.path.isdir(lab_dir) else 0
    n_tg  = len([f for f in os.listdir(tg_dir) if f.endswith('.TextGrid')]) if os.path.isdir(tg_dir) else 0
    
    # Count total phones across all TextGrids
    total_phones = 0
    if os.path.isdir(tg_dir):
        import tgt
        for f in os.listdir(tg_dir):
            if not f.endswith('.TextGrid'): continue
            try:
                tg = tgt.io.read_textgrid(os.path.join(tg_dir, f))
                tier = tg.get_tier_by_name('phones')
                total_phones += sum(1 for a in tier.annotations 
                                    if a.text not in ('', 'sp', 'sil', 'spn'))
            except:
                pass
    
    pct = n_tg / n_lab * 100 if n_lab else 0
    print(f"{pid}: {n_tg:>3}/{n_lab:>3} sentences aligned ({pct:.0f}%)  "
          f"total phones: {total_phones}")


import copy
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

FRAMESHIFT_MS = 5  # ms per HG frame

def make_feature_matrix(features, labels, participant_ids, expected_size=None):
    """Flatten features and drop any with wrong size."""
    flat = [np.array(f).flatten() for f in features]
    if expected_size is None:
        sizes = Counter(f.shape[0] for f in flat)
        expected_size = sizes.most_common(1)[0][0]
    keep = [f.shape[0] == expected_size for f in flat]
    X   = np.array([flat[i]          for i in range(len(flat))           if keep[i]])
    y   = [labels[i]                  for i in range(len(labels))          if keep[i]]
    ids = [participant_ids[i]          for i in range(len(participant_ids)) if keep[i]]
    n_dropped = sum(1 for k in keep if not k)
    if n_dropped:
        print(f"  dropped {n_dropped} malformed samples (expected size={expected_size})")
    return X, y, ids, expected_size

def sweep_temporal_offset(cached_train, cached_test, pipeline, run_config,
                           offsets_frames=[-30, -26, -20, -16, -10, -8, -6, 0, 6, 10]):
    patients = sorted(set(cached_train['phoneme_participant_ids']))
    results  = {offset: {} for offset in offsets_frames}

    for offset in offsets_frames:
        ms        = offset * FRAMESHIFT_MS
        direction = "pre-onset"  if offset < 0 else ("post-onset" if offset > 0 else "no shift")
        print(f"\n=== offset={offset:+d} frames ({abs(ms)}ms {direction}) ===")

        train = copy.deepcopy(cached_train)
        test  = copy.deepcopy(cached_test)

        for data in [train, test]:
            new_feats = []
            for feat in data['features']:
                arr = np.array(feat)
                if arr.ndim != 2 or arr.shape[0] < 2:
                    new_feats.append(arr)
                    continue
                if offset > 0:
                    new_feats.append(arr[offset:] if arr.shape[0] > offset else arr)
                elif offset < 0:
                    end = arr.shape[0] + offset
                    new_feats.append(arr[:end] if end > 0 else arr)
                else:
                    new_feats.append(arr)
            data['features'] = new_feats

        pipeline.train = train
        pipeline.test  = test

        for d in [pipeline.train, pipeline.test]:
            d['phoneme_positions'] = [0] * len(d['phoneme_positions'])

        pipeline.step5a_filter_by_frame_count(
            min_frames=run_config['min_frames'],
            max_frames=run_config['max_frames'],
        )
        pipeline.step5b_stack_features(
            model_order=run_config['stacking_order'],
            step_size=run_config['stacking_step_size'],
        )
        pipeline.step5c_collapse_to_phoneme_level()
        pipeline.dutch30_step6_resolve_unknowns()
        pipeline.step7_filter_unknowns(unknown_keep_ratio=run_config['unknown_keep_ratio'])

        tr_features = pipeline.train['features']
        tr_labels   = pipeline.train['phoneme_labels']
        tr_ids      = pipeline.train['phoneme_participant_ids']
        te_features = pipeline.test['features']
        te_labels   = pipeline.test['phoneme_labels']
        te_ids      = pipeline.test['phoneme_participant_ids']
        
        for pid in patients:
            tr_idx = [i for i, p in enumerate(tr_ids) if p == pid]
            te_idx = [i for i, p in enumerate(te_ids) if p == pid]
            if len(tr_idx) < 20 or len(te_idx) < 5:
                continue

            # flatten within patient — all same channel count, so safe
            X_tr = np.array([np.array(tr_features[i]).flatten() for i in tr_idx])
            X_te = np.array([np.array(te_features[i]).flatten() for i in te_idx])
            y_tr = [tr_labels[i] for i in tr_idx]
            y_te = [te_labels[i] for i in te_idx]

            # drop any remaining malformed rows within this patient
            expected = X_tr.shape[1] if X_tr.ndim == 2 else None
            if expected is None:
                continue
            tr_ok = [X_tr[i].shape[0] == expected for i in range(len(X_tr))]
            te_ok = [X_te[i].shape[0] == expected for i in range(len(X_te))]
            X_tr = X_tr[tr_ok]; y_tr = [y_tr[i] for i, m in enumerate(tr_ok) if m]
            X_te = X_te[te_ok]; y_te = [y_te[i] for i, m in enumerate(te_ok) if m]

            if len(X_tr) < 20 or len(X_te) < 5:
                continue

            try:
                scaler = StandardScaler()
                X_tr_s = scaler.fit_transform(X_tr)
                X_te_s = scaler.transform(X_te)
                clf = LogisticRegression(max_iter=300, class_weight='balanced',
                                         solver='saga', C=1.0)
                clf.fit(X_tr_s, y_tr)
                acc = accuracy_score(y_te, clf.predict(X_te_s))
                results[offset][pid] = acc
                print(f"  {pid}: {acc:.3f}  (n_train={len(X_tr)}, n_test={len(X_te)})")
            except Exception as e:
                print(f"  {pid}: error — {e}")

    return results


offset_results = sweep_temporal_offset(
    cached_train, cached_test, pipeline, run_config,
    #offsets_frames=[-30, -25, -20, -16, -12, -10, -8, -6, -4, -2, 0, 10]
    offsets_frames=[-30, -20, -10, 0, 6, 10]
)

import numpy as np
from phonetic_dictionary import PhoneticDictionary
from collections import defaultdict

phon_dict = PhoneticDictionary()
raw_dir = '../SingleWordProductionDutch/Dutch_30patients/raw'
THRESHOLD = 0.90

# Words that made it to phoneme level, per patient
parsed_words_per_pid = defaultdict(set)
for word, pid in zip(pipeline.train['phoneme_words'], pipeline.train['phoneme_participant_ids']):
    parsed_words_per_pid[pid].add(word)
for word, pid in zip(pipeline.test['phoneme_words'], pipeline.test['phoneme_participant_ids']):
    parsed_words_per_pid[pid].add(word)

print(f"{'Pat':<6} {'Sents':>6} {'AvgWds':>7} {'Expect':>7} "
      f"{'Train':>6} {'Test':>6} {'Total':>6} {'Rate':>6} "
      f"{'AvgMaxPos':>10} {'Status':>8}")
print("-" * 82)

all_passed = True
for pid_num in range(21, 31):
    pid = f'P{pid_num:02d}'

    stimuli = np.load(f'{raw_dir}/{pid}_stimuli.npy', allow_pickle=True)
    unique_sentences = [s for s in np.unique(stimuli) if isinstance(s, str) and ' ' in s]
    total_expected = sum(phon_dict.count_phonemes(s) for s in unique_sentences)
    n_sents = len(unique_sentences)

    train_ph = sum(1 for p in pipeline.train['phoneme_participant_ids'] if p == pid)
    test_ph  = sum(1 for p in pipeline.test ['phoneme_participant_ids'] if p == pid)
    total_ph = train_ph + test_ph
    rate     = 100 * total_ph / total_expected if total_expected > 0 else 0
    passed   = total_ph >= total_expected * THRESHOLD
    if not passed:
        all_passed = False

    # Per-sentence: avg words and avg max parsed word index
    parsed_set = parsed_words_per_pid.get(pid, set())
    word_counts = [len(s.split()) for s in unique_sentences]
    avg_words = np.mean(word_counts)

    max_pos_per_sent = [
        max((i for i, w in enumerate(s.split()) if w in parsed_set), default=-1)
        for s in unique_sentences
    ]
    parsed_sents = [p for p in max_pos_per_sent if p >= 0]
    avg_max_pos = np.mean(parsed_sents) if parsed_sents else -1

    status = "✓" if passed else "✗"
    print(f"{pid:<6} {n_sents:>6} {avg_words:>7.1f} {total_expected:>7} "
          f"{train_ph:>6} {test_ph:>6} {total_ph:>6} {rate:>5.1f}% "
          f"{avg_max_pos:>9.1f}  {status:>8}")

print("-" * 82)
print("AvgMaxPos = average of (last parsed word position within each sentence), 0-indexed")

results = pipeline.step9_train_and_evaluate(
    model_factory=MarkovPhonemeModel,
    model_params={
        'phonetic_dict': pipeline.detector.phonetic_dict,
        'order': run_config['markov_order'],
        'use_groups': False,
        'class_weight': run_config['class_weight'],
        'classifier_type': run_config['classifier_type'],
        'random_state': run_config['random_state'],
        'scaler_type': run_config['scaler_type'],
    },
    use_viterbi=True,
)

for pid in sorted(pipeline.patient_results.keys()):
    print(f"--- {pid} ---")
    r = pipeline.patient_results[pid]
    print(f"  acc={r['accuracy']:.4f}  n_pred={len(set(r['predictions']))}")
    pipeline.step10_visualize_patient(pid, show_table=False, min_class_samples=run_config.get('min_class_samples', 5))

import importlib, visualize_alignment
importlib.reload(visualize_alignment)
from visualize_alignment import plot_neural_alignment

plot_neural_alignment(pipeline, 'P24', sentence_text='donald trump')
# plot_sentence_alignment(pipeline, 'P24', sentence_text='donald trump')

import numpy as np
from scipy.signal import resample_poly
from IPython.display import Audio, display

def analyze_boundary_search_logic(total_phonemes, n_words, n_peaks_found, word_list=None, audio_signal=None, sample_rate=16000):
    """
    Compare current vs proposed boundary search logic with segment visualization and audio playback.
    
    Args:
        total_phonemes: Total phonemes in sentence
        n_words: Number of words in sentence
        n_peaks_found: Number of peaks detected by wav2vec2
        word_list: Optional list of words to show distribution
        audio_signal: Optional audio array for playback
        sample_rate: Audio sampling rate in Hz
    """
    if word_list is None:
        word_list = [f"word{i+1}" for i in range(n_words)]
    
    current_boundaries_needed = total_phonemes - 1
    proposed_boundaries_needed = n_words - 1
    
    print(f"\nTotal phonemes: {total_phonemes}")
    print(f"Words: {n_words}")
    print(f"Peaks found: {n_peaks_found}")
    print(f"\nCurrent logic (phoneme-based):")
    print(f"  Boundaries needed: {current_boundaries_needed}")
    print(f"  Peak shortage: {current_boundaries_needed - n_peaks_found}")
    
    print(f"\nProposed logic (word-based):")
    print(f"  Boundaries needed: {proposed_boundaries_needed}")
    print(f"  Peak shortage: {proposed_boundaries_needed - n_peaks_found}")
    
    print(f"\nSegment distribution:")
    
    n_segments = min(n_peaks_found + 1, n_words)
    
    if n_peaks_found >= proposed_boundaries_needed:
        print(f"  Proposed: {n_segments} segments (perfect match)")
    else:
        print(f"  Proposed: {n_segments} segments")
    
    if audio_signal is not None:
        audio_duration_ms = len(audio_signal) / sample_rate * 1000
        
        if n_peaks_found == 0:
            boundaries_ms = []
        else:
            boundaries_ms = np.linspace(0, audio_duration_ms, n_peaks_found + 2)[1:-1]
        
        word_segments = []
        start_ms = 0
        for boundary_ms in boundaries_ms:
            word_segments.append((int(start_ms), int(boundary_ms)))
            start_ms = boundary_ms
        word_segments.append((int(start_ms), int(audio_duration_ms)))
        
        for j, (start_ms, end_ms) in enumerate(word_segments):
            expected = word_list[j] if j < len(word_list) else '?'
            dur_ms = end_ms - start_ms
            
            start_sample = int(start_ms / 1000 * sample_rate)
            end_sample = int(end_ms / 1000 * sample_rate)
            seg = audio_signal[start_sample:end_sample]
            
            print(f"segment {j} — expected: '{expected}'  ({dur_ms:.0f}ms)")
            if len(seg) > 0:
                seg_48k = resample_poly(seg.astype(np.float32), up=3, down=1)
                display(Audio(seg_48k, rate=48000))
        
        n_lost = n_words - len(word_segments)
        if n_lost > 0:
            lost_words = word_list[len(word_segments):]
            print(f"\nLost {n_lost} words: {lost_words}")
    else:
        for i in range(n_segments):
            if i < len(word_list):
                print(f"  Segment {i+1}: '{word_list[i]}'")
        
        n_lost = n_words - n_segments
        if n_lost > 0:
            lost_words = word_list[n_segments:]
            print(f"\nLost {n_lost} words: {lost_words}")


import numpy as np
from scipy.signal import resample_poly
from IPython.display import Audio, display

# Use first 3 seconds as a test case
base_path = r"C:\mozg\code\SingleWordProductionDutch\Dutch_30patients\raw"
audio_data = np.load(f"{base_path}\\P21_audio.npy")
audio_signal = audio_data[:48000]  # 3 seconds at 16kHz
sr = 16000

word_list = ['Het', 'meisje', 'zingt', 'een', 'vrolijk', 'liedje', 'in', 'de', 'tuin', 'vandaag']
analyze_boundary_search_logic(40, 10, 0, word_list, audio_signal, sr)
# Test with 5 peaks (6 segments)
# analyze_boundary_search_logic(40, 10, 5, word_list, audio_signal, sr)

# Test with 9 peaks (perfect match - 10 segments)
analyze_boundary_search_logic(40, 10, 9, word_list, audio_signal, sr)

# # ---- Run experiment ----
# name, params, results = run_from_config(pipeline, run_config)

import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet
from scipy.spatial.distance import pdist
from collections import defaultdict


def get_patient_centroids(pipeline, pid):
    """Return (centroid_matrix, phoneme_list) for one patient.
    
    Pools train + test, computes mean feature vector per phoneme class.
    """
    features, labels = [], []
    for split in [pipeline.train, pipeline.test]:
        for i, p in enumerate(split['phoneme_participant_ids']):
            if p == pid:
                features.append(split['features'][i])
                labels.append(split['phoneme_labels'][i])

    features = np.array(features)
    labels = np.array(labels)
    phonemes = sorted(set(labels))

    centroids = np.array([features[labels == ph].mean(axis=0) for ph in phonemes])
    return centroids, phonemes


def plot_phoneme_dendrograms(pipeline, pids=None, method='ward', metric='euclidean',
                              n_cols=2, figsize_per=(10, 4)):
    """Plot per-patient phoneme dendrograms in a grid.

    Args:
        pipeline:  pipeline object after step 5/6.
        pids:      list of patient IDs; None = all patients in train.
        method:    linkage method ('ward', 'average', 'complete', 'single').
        metric:    distance metric ('euclidean', 'cosine', 'correlation').
        n_cols:    columns in the figure grid.
        figsize_per: (w, h) per panel.
    """
    if pids is None:
        pids = sorted(set(pipeline.train['phoneme_participant_ids']))

    n_rows = int(np.ceil(len(pids) / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize_per[0] * n_cols, figsize_per[1] * n_rows)
    )
    axes = np.array(axes).flatten()

    linkage_cache = {}

    for idx, pid in enumerate(pids):
        ax = axes[idx]
        try:
            centroids, phonemes = get_patient_centroids(pipeline, pid)
            if len(phonemes) < 3:
                ax.set_title(f'{pid} — too few classes ({len(phonemes)})')
                ax.axis('off')
                continue

            Z = linkage(centroids, method=method, metric=metric)
            linkage_cache[pid] = (Z, phonemes, centroids)

            dendrogram(Z, labels=phonemes, ax=ax,
                       leaf_rotation=90, leaf_font_size=8,
                       color_threshold=0.7 * max(Z[:, 2]))
            ax.set_title(f'{pid}  ({len(phonemes)} phonemes, {method})')
            ax.set_ylabel('Distance')
        except Exception as e:
            ax.set_title(f'{pid} — error: {e}')
            ax.axis('off')

    # hide unused panels
    for idx in range(len(pids), len(axes)):
        axes[idx].axis('off')

    fig.suptitle(f'Phoneme Dendrograms — {method} linkage / {metric} distance',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.show()
    return linkage_cache


import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.metrics import silhouette_score
from collections import defaultdict


# ── 1. Build co-occurrence matrix across patients ───────────────────────────

def build_cooccurrence_matrix(linkage_cache, k, all_phonemes=None):
    """For a given k, count how often each phoneme pair is in the same cluster.

    For each patient that has both phonemes, +1 if they share a cluster,
    +0 if they don't. Normalise by how many patients had both phonemes.

    Returns:
        co_matrix:    np.ndarray (n_phonemes × n_phonemes), values in [0, 1]
        phoneme_list: list of str, index → phoneme label
    """
    if all_phonemes is None:
        all_ph = set()
        for Z, phonemes, centroids in linkage_cache.values():
            all_ph.update(phonemes)
        all_phonemes = sorted(all_ph)

    n = len(all_phonemes)
    idx = {ph: i for i, ph in enumerate(all_phonemes)}
    co_sum   = np.zeros((n, n))
    co_count = np.zeros((n, n))

    for pid, (Z, phonemes, centroids) in linkage_cache.items():
        if len(phonemes) < k + 1:
            continue
        labels = fcluster(Z, k, criterion='maxclust')
        ph_label = dict(zip(phonemes, labels))

        # only consider phonemes that are in the universal set
        valid_phonemes = [ph for ph in phonemes if ph in idx]

        for i, ph_i in enumerate(valid_phonemes):
            for ph_j in valid_phonemes[i:]:
                a, b = idx[ph_i], idx[ph_j]
                co_count[a, b] += 1
                co_count[b, a] += 1
                if ph_label[ph_i] == ph_label[ph_j]:
                    co_sum[a, b] += 1
                    co_sum[b, a] += 1


    with np.errstate(invalid='ignore'):
        co_matrix = np.where(co_count > 0, co_sum / co_count, 0.0)

    np.fill_diagonal(co_matrix, 1.0)
    return co_matrix, all_phonemes


# ── 2. Find best k by evaluating co-occurrence consistency ──────────────────

def find_best_k_consensus(linkage_cache, k_range=range(2, 15), min_patients=2):
    """For each k, build co-occurrence matrix and measure how bimodal it is.

    A good k produces a co-occurrence matrix that is close to 0/1
    (pairs are either always together or never together).
    We measure this as the mean distance from 0.5 — higher = more decisive.

    Also computes silhouette on the co-occurrence distance matrix.
    """
    # collect all phonemes seen in ≥ min_patients patients
    ph_patient_count = defaultdict(set)
    for pid, (Z, phonemes, _) in linkage_cache.items():
        for ph in phonemes:
            ph_patient_count[ph].add(pid)
    all_phonemes = sorted(
        ph for ph, pids in ph_patient_count.items()
        if len(pids) >= min_patients
    )
    print(f"Phonemes in ≥{min_patients} patients: {len(all_phonemes)}")

    scores = {}
    for k in k_range:
        co, _ = build_cooccurrence_matrix(linkage_cache, k, all_phonemes)
        # decisiveness: average |co - 0.5|, scaled to [0,1]
        decisiveness = np.mean(np.abs(co - 0.5)) * 2

        # cluster the distance matrix (1 - co_matrix = dissimilarity)
        dist_vec = 1.0 - co[np.triu_indices(len(all_phonemes), k=1)]
        try:
            Z_co = linkage(dist_vec, method='average')
            cluster_ids = fcluster(Z_co, k, criterion='maxclust')
            sil = silhouette_score(1.0 - co, cluster_ids, metric='precomputed')
        except Exception:
            sil = float('nan')

        scores[k] = {
            'decisiveness': decisiveness,
            'silhouette':   sil,
            'co_matrix':    co,
        }

    # plot
    ks    = sorted(scores)
    dec   = [scores[k]['decisiveness'] for k in ks]
    sils  = [scores[k]['silhouette']   for k in ks]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(ks, dec, marker='o', color='steelblue')
    ax1.set_title('Co-occurrence decisiveness vs k\n'
                  '(higher = pairs are always-together or never-together)')
    ax1.set_xlabel('k'); ax1.set_ylabel('Decisiveness'); ax1.set_xticks(ks)

    ax2.plot(ks, sils, marker='o', color='darkorange')
    ax2.set_title('Silhouette on co-occurrence matrix vs k\n'
                  '(higher = cleaner consensus clusters)')
    ax2.set_xlabel('k'); ax2.set_ylabel('Silhouette'); ax2.set_xticks(ks)

    best_dec = ks[int(np.argmax(dec))]
    best_sil = ks[int(np.nanargmax(sils))]
    ax1.axvline(best_dec, color='red', linestyle='--', label=f'best k={best_dec}')
    ax2.axvline(best_sil, color='red', linestyle='--', label=f'best k={best_sil}')
    ax1.legend(); ax2.legend()

    plt.suptitle('Consensus clustering quality', fontsize=13, fontweight='bold')
    plt.tight_layout(); plt.show()

    print(f"\n{'k':>4}  {'Decisiveness':>14}  {'Silhouette':>11}")
    print('-' * 34)
    for k in ks:
        d_mark = ' ◄' if k == best_dec else ''
        s_mark = ' ◄' if k == best_sil else ''
        print(f"{k:>4}  {scores[k]['decisiveness']:>14.3f}{d_mark:3}  "
              f"{scores[k]['silhouette']:>11.3f}{s_mark}")

    return scores, all_phonemes, best_sil


# ── 3. Plot universal dendrogram from co-occurrence matrix ──────────────────

def plot_consensus_dendrogram(scores, all_phonemes, k, method='average'):
    """Dendrogram built from the consensus co-occurrence matrix at given k."""
    co = scores[k]['co_matrix']
    dist_vec = 1.0 - co[np.triu_indices(len(all_phonemes), k=1)]
    Z = linkage(dist_vec, method=method)

    # cut threshold for k clusters
    sorted_dists = sorted(Z[:, 2])
    cut = (sorted_dists[-(k - 1)] + sorted_dists[-k]) / 2

    fig, ax = plt.subplots(figsize=(max(14, len(all_phonemes) * 0.45), 6))
    dendrogram(Z, labels=all_phonemes, ax=ax,
               leaf_rotation=90, leaf_font_size=9,
               color_threshold=cut)
    ax.axhline(cut, color='red', linestyle='--', linewidth=1.5,
               label=f'cut → k={k}')
    ax.legend()
    ax.set_title(
        f'Universal Phoneme Dendrogram (consensus across all patients)\n'
        f'k={k}, {len(all_phonemes)} phonemes',
        fontsize=12
    )
    ax.set_ylabel('Co-occurrence distance  (1 − co-occurrence rate)')
    plt.tight_layout(); plt.show()

    # return the cluster assignments
    cluster_ids = fcluster(Z, k, criterion='maxclust')
    universal_labels = dict(zip(all_phonemes, cluster_ids))

    groups = defaultdict(list)
    for ph, c in universal_labels.items():
        groups[c].append(ph)
    print(f"\nUniversal clusters at k={k}:")
    for c in sorted(groups):
        print(f"  cluster {c}: {[str(ph) for ph in sorted(groups[c])]}")
    return universal_labels

# Step 1 — dendrograms
linkage_cache = plot_phoneme_dendrograms(
    pipeline,
    pids=None,     # or None for all
    method='ward',
    metric='euclidean',
)

# linkage_cache already built by plot_phoneme_dendrograms above
scores, all_phonemes, best_k = find_best_k_consensus(
    linkage_cache, k_range=range(2, 15), min_patients=2
)
for k in [5]:
#universal_labels = plot_consensus_dendrogram(scores, all_phonemes, k=best_k)
    universal_labels = plot_consensus_dendrogram(scores, all_phonemes, k=k)

#nspect another k if the two metrics disagree
# universal_labels = plot_consensus_dendrogram(scores, all_phonemes, k=5)

# results_nn = pipeline.step9_train_and_evaluate(
#     model_factory=MarkovPhonemeModel,
#     model_params={
#         'phonetic_dict': pipeline.detector.phonetic_dict,
#         'order': 1,
#         'use_groups': False,
#         'class_weight': 'balanced',
#         'classifier_type': 'nn_relu',
#         'random_state': 37,
#         'scaler_type': 'standard',
#     },
#     use_viterbi=True,
# )

# pipeline.checkpoint_after_step6(sample_fraction=sample_fraction)

# def plot_phoneme_clustering(pipeline, min_samples=10):
#     """Plot per-patient hierarchical clustering of phonemes from neural features.

#     For each patient, computes the mean neural feature vector per phoneme,
#     then performs bottom-up agglomerative clustering on those mean vectors.
#     The resulting dendrogram shows which phonemes are neurally similar
#     and suggests natural groupings.

#     Args:
#         pipeline: Dutch30Pipeline instance with train data populated.
#         min_samples: int, minimum number of training samples required
#             for a phoneme to be included in clustering.
#     """
#     import numpy as np
#     import matplotlib.pyplot as plt
#     from collections import defaultdict
#     from scipy.cluster.hierarchy import dendrogram, linkage
#     from scipy.spatial.distance import pdist

#     patient_ids = sorted(set(pipeline.train["phoneme_participant_ids"]))

#     n_patients  = len(patient_ids)
#     n_cols      = 2
#     n_rows      = (n_patients + 1) // n_cols
#     fig, axes   = plt.subplots(
#         n_rows, n_cols,
#         figsize=(14, n_rows * 3.5)
#     )
#     axes = axes.flatten()
#     fig.patch.set_facecolor("#faf9f6")

#     for ax_idx, pid in enumerate(patient_ids):
#         ax = axes[ax_idx]
#         ax.set_facecolor("#faf9f6")

#         pid_mask = [
#             p == pid
#             for p in pipeline.train["phoneme_participant_ids"]
#         ]
#         features = [
#             pipeline.train["features"][i]
#             for i, m in enumerate(pid_mask) if m
#         ]
#         labels = [
#             pipeline.train["phoneme_labels"][i]
#             for i, m in enumerate(pid_mask) if m
#         ]

#         # compute mean feature vector per phoneme
#         phoneme_sums   = defaultdict(list)
#         for feat, label in zip(features, labels):
#             if label in ("unknown", "?", ""):
#                 continue
#             flat = feat.flatten() if feat.ndim > 1 else feat
#             if not (np.any(np.isnan(flat)) or np.any(np.isinf(flat))):
#                 phoneme_sums[label].append(flat)

#         # filter by minimum sample count and align feature lengths
#         phoneme_means = {}
#         for phoneme, vecs in phoneme_sums.items():
#             if len(vecs) < min_samples:
#                 continue
#             min_len = min(v.shape[0] for v in vecs)
#             trimmed = np.array([v[:min_len] for v in vecs])
#             phoneme_means[phoneme] = trimmed.mean(axis=0)

#         if len(phoneme_means) < 2:
#             ax.text(0.5, 0.5, f"{pid}\nnot enough phonemes",
#                     ha="center", va="center", fontsize=9,
#                     color="#aaa", fontfamily="DejaVu Serif")
#             ax.axis("off")
#             continue

#         phonemes   = sorted(phoneme_means.keys())
#         matrix     = np.array([phoneme_means[p] for p in phonemes])

#         # normalise rows so distance reflects shape not amplitude
#         norms  = np.linalg.norm(matrix, axis=1, keepdims=True)
#         norms[norms == 0] = 1.0
#         matrix = matrix / norms

#         dist    = pdist(matrix, metric="cosine")
#         Z       = linkage(dist, method="ward")

#         dendrogram(
#             Z,
#             labels=phonemes,
#             ax=ax,
#             orientation="top",
#             leaf_font_size=9,
#             color_threshold=0.7 * max(Z[:, 2]),
#             above_threshold_color="#aaa",
#         )
#         ax.set_title(f"{pid}", fontsize=10, fontweight="600",
#                      fontfamily="DejaVu Serif", pad=6)
#         ax.set_ylabel("distance", fontsize=8, fontfamily="DejaVu Serif")
#         for spine in ["top", "right"]:
#             ax.spines[spine].set_visible(False)
#         ax.tick_params(axis="x", labelsize=9)
#         ax.tick_params(axis="y", labelsize=7)

#     # hide unused axes
#     for i in range(len(patient_ids), len(axes)):
#         axes[i].axis("off")

#     fig.suptitle(
#         "Hierarchical clustering of phonemes by mean neural feature vector",
#         fontsize=12, fontweight="600", fontfamily="DejaVu Serif", y=1.01
#     )
#     plt.tight_layout()
#     plt.savefig("phoneme_clustering.png", dpi=150,
#                 bbox_inches="tight", facecolor="#faf9f6")
#     plt.show()

# diag = Dutch30PhonemeDetectionDiagnostic(pipeline)
# diag.visualize_word_analysis('P23', word_name = 'postzegelverzameling.', save_path='p23_word_postzegelverzameling.png')

#diag.visualize_multifeature_analysis('P01', word_index=50)
# diag.visualize_rms_boundaries('P01',  word_name = 'vogelkooitje')

# Quick check first 10 words
# diag.batch_diagnostic('sub-p11', num_samples=5)

"""Phoneme duration diagnostic.

Shows how raw neural signal looks at different durations
before and after resampling to a fixed frame count.

Usage:
    from phoneme_duration_diagnostic import phoneme_duration_diagnostic
    phoneme_duration_diagnostic(pipeline, pid="P23",
                                phonemes=["t", "n", "a:", "schwa"])
"""

import re
import numpy as np
import matplotlib.pyplot as plt

from collections import defaultdict
from scipy.signal import resample_poly, resample

from extract_features import extractHG
from dataset_config import Dutch30Config
# Add bin legend
from matplotlib.lines import Line2D 

def extract_phoneme_segments(pipeline, pid, max_sentences=50):
    """Extract per-phoneme iEEG high gamma segments from sentences.

    Args:
        pipeline: Dutch30Pipeline instance.
        pid: str, patient ID.
        max_sentences: int, maximum sentences to process.

    Returns:
        dict mapping phoneme label (str) to list of dicts:
            hg: np.array (n_frames, n_channels), high gamma.
            n_frames: int, number of HG frames.
            duration_ms: float, segment duration in ms.
            word: str, parent word.
    """
    config = pipeline.config
    raw_data = pipeline.dutch30_extractor.load_patient_raw_data(pid)
    eeg_full = raw_data["eeg"]
    audio_full = raw_data["audio"]
    stimuli = raw_data["stimuli"]
    eeg_sr = raw_data["eeg_sr"]
    audio_sr = config.audio_sr

    # Build sentence list
    sentence_list = []
    current_text = None
    current_start = 0
    for idx, stim in enumerate(stimuli):
        text = stim.decode() if isinstance(stim, bytes) else str(stim)
        text = text.strip()
        if text != current_text:
            if current_text is not None and current_text:
                sentence_list.append({
                    "text": current_text,
                    "stim_start_idx": current_start,
                    "stim_end_idx": idx,
                })
            current_text = text
            current_start = idx
    if current_text is not None and current_text:
        sentence_list.append({
            "text": current_text,
            "stim_start_idx": current_start,
            "stim_end_idx": len(stimuli),
        })

    sentence_list = [
        s for s in sentence_list if len(s["text"].split()) > 1
    ][:max_sentences]

    downsample_factor = int(audio_sr / config.audio_target_sr)
    audio_16k = resample_poly(
        audio_full.astype(np.float32), up=1, down=downsample_factor
    )
    resample_ratio = len(audio_16k) / len(audio_full)

    phoneme_segments = defaultdict(list)
    n_success = 0
    n_fail = 0

    for sent_info in sentence_list:
        sent_stim_start = sent_info["stim_start_idx"]
        sent_stim_end = sent_info["stim_end_idx"]

        cleaned = re.sub(r'["""\u201e\u201c\u2018\u2019\r\n]+', '',
                         sent_info["text"])
        word_texts = [w for w in cleaned.split() if w]
        if not word_texts:
            continue

        eeg_sentence = eeg_full[sent_stim_start:sent_stim_end]

        audio_start = int(
            sent_stim_start * len(audio_full) / len(eeg_full)
        )
        audio_end = int(
            sent_stim_end * len(audio_full) / len(eeg_full)
        )
        sent_audio_len = audio_end - audio_start

        a_start_16k = int(audio_start * resample_ratio)
        a_end_16k = int(audio_end * resample_ratio)
        audio_sent_16k = audio_16k[a_start_16k:a_end_16k]

        if len(audio_sent_16k) < 160:
            continue

        # Build phoneme labels with word mapping
        phoneme_labels = []
        phoneme_words = []
        for w in word_texts:
            phonemes = pipeline.phonetic_dict.extract_phonemes(w)
            if phonemes is None:
                phonemes = ["?"]
            for ph in phonemes:
                phoneme_labels.append(ph)
                phoneme_words.append(w.lower())

        # Wav2vec boundaries
        try:
            result = pipeline.detector.segment_sentence_by_wav2vec(
                audio_sentence=audio_sent_16k,
                audio_sr=config.audio_target_sr,
                words=word_texts,
                phonetic_dict=pipeline.phonetic_dict,
            )
        except Exception:
            n_fail += 1
            continue

        boundaries_audio = (
            result["phoneme_boundaries_samples"] / resample_ratio
        ).astype(int)

        # Convert to EEG sample indices
        boundaries_eeg = []
        for b in boundaries_audio:
            frac = b / sent_audio_len if sent_audio_len > 0 else 0
            boundaries_eeg.append(
                max(0, min(eeg_sentence.shape[0],
                           int(frac * eeg_sentence.shape[0])))
            )
        boundaries_eeg = np.array(boundaries_eeg)

        # Check boundary count
        if len(boundaries_eeg) != len(phoneme_labels) + 1:
            ratio = len(boundaries_eeg) / (len(phoneme_labels) + 1)
            if ratio < 0.3 or ratio > 3.0:
                n_fail += 1
                continue
            # Skip mismatched sentences for cleaner data
            n_fail += 1
            continue

        # Extract each phoneme segment
        for pi in range(len(phoneme_labels)):
            if phoneme_labels[pi] == "?":
                continue

            seg_start = boundaries_eeg[pi]
            seg_end = boundaries_eeg[pi + 1]

            if seg_end <= seg_start:
                continue

            phoneme_eeg = eeg_sentence[seg_start:seg_end]
            duration_ms = phoneme_eeg.shape[0] / eeg_sr * 1000

            if duration_ms < 10 or duration_ms > 500:
                continue

            try:
                hg = extractHG(
                    phoneme_eeg, eeg_sr,
                    windowLength=config.window_length,
                    frameshift=config.frameshift,
                )
                if hg is None or hg.shape[0] == 0:
                    continue
            except Exception:
                continue

            phoneme_segments[phoneme_labels[pi]].append({
                "hg": hg,
                "n_frames": hg.shape[0],
                "duration_ms": duration_ms,
                "word": phoneme_words[pi],
            })
            n_success += 1

    print(f"{pid}: extracted {n_success} phoneme segments, "
          f"failed {n_fail}")

    return phoneme_segments


def phoneme_duration_diagnostic(pipeline, pid,
                                 phonemes=None,
                                 target_frames=2,
                                 model_order=10,
                                 step_size=1,
                                 max_instances_per_bin=8):
    """Visualize phoneme neural signal at different durations.

    For each phoneme, shows 4 columns:
        1. Short segments (1-3 HG frames), raw, no resampling
        2. Medium segments (4-10 frames), raw
        3. Long segments (11+ frames), raw
        4. All segments resampled to target_frames,
           colored by original duration bin

    Each trace is the channel-averaged high gamma signal for
    one instance.

    Args:
        pipeline: Dutch30Pipeline instance.
        pid: str, patient ID.
        phonemes: list of str, phonemes to inspect. If None,
            picks 3 frequent consonants and 3 frequent vowels.
        target_frames: int, resampling target.
        max_instances_per_bin: int, max instances to draw
            per duration bin (keeps plots readable).
    """
    all_segments = extract_phoneme_segments(pipeline, pid)

    if phonemes is None:
        # Pick frequent phonemes with variety
        vowels = []
        consonants = []
        vowel_set = {
                    "a:", "e:", "i", "o:", "u", "y",
                    "\u0259",   # schwa
                    "\u025b",   # open e
                    "\u0254",   # open o
                    "\u026a",   # short i
                    "\u0251",   # open a (ɑ)
                    "a\u02d0",  # long a (aː)
                    "e\u02d0",  # long e (eː)
                    "o\u02d0",  # long o (oː)
                    "y\u02d0",  # long y (yː)
                    "\u00f8\u02d0",  # long oe (øː)
                    "\u028f",   # short y (ʏ)
                    "\u025bi",  # diphthong ei
                    "\u0153y",  # diphthong oey (œy)
                    "\u0251u",  # diphthong au (ɑu)
                    "a",        # short a
                    "o",        # short o
                    "e",        # short e
                }
        for ph, segs in sorted(all_segments.items(),
                                key=lambda x: -len(x[1])):
            if len(segs) < 5:
                continue
            if ph in vowel_set and len(vowels) < 3:
                vowels.append(ph)
            elif ph not in vowel_set and len(consonants) < 3:
                consonants.append(ph)
            if len(vowels) >= 3 and len(consonants) >= 3:
                break
        phonemes = consonants + vowels
        print(f"Selected phonemes: {phonemes}")

    # Duration bins
    bins = [
            ("4-10 frames", lambda n: 4 <= n <= 10, "tab:red"),
            ("11-20 frames", lambda n: 11 <= n <= 20, "tab:blue"),
            ("20-25 frames", lambda n: 16 <= n <= 25, "tab:green"),
            ("25+ frames", lambda n: n >= 25, "tab:purple"),
        ]

    for phoneme in phonemes:
        if phoneme not in all_segments:
            print(f"/{phoneme}/ not found")
            continue

        segments = all_segments[phoneme]
        if len(segments) < 3:
            print(f"/{phoneme}/: only {len(segments)} instances, skipping")
            continue

        # Bin the segments
        binned = {}
        for bin_name, bin_fn, bin_color in bins:
            binned[bin_name] = {
                "color": bin_color,
                "segments": [s for s in segments if bin_fn(s["n_frames"])],
            }

        # Count
        total = len(segments)
        frame_counts = [s["n_frames"] for s in segments]
        bin_counts = {b: len(d["segments"]) for b, d in binned.items()}

        fig, axes = plt.subplots(
            2, len(bins) + 1, figsize=(28, 8),
            gridspec_kw={"width_ratios": [1] * len(bins) + [1.3]},
        )
        fig.suptitle(
            f"{pid} -- /{phoneme}/ ({total} instances, "
            f"frames: min={min(frame_counts)}, "
            f"median={int(np.median(frame_counts))}, "
            f"max={max(frame_counts)})",
            fontsize=12, fontweight="bold",
        )

        # Row 0: raw signal per bin + resampled
        for col, (bin_name, bin_fn, bin_color) in enumerate(bins):
            ax = axes[0, col]
            bin_segs = binned[bin_name]["segments"]

            if not bin_segs:
                ax.set_title(f"{bin_name}\n(0 instances)", fontsize=10)
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=11, color="gray")
                ax.set_xlabel("Frame")
                continue

            show_segs = bin_segs[:max_instances_per_bin]
            for i, seg in enumerate(show_segs):
                hg = seg["hg"]
                ch_mean = hg.mean(axis=1)
                n_fr = len(ch_mean)
                alpha = max(0.3, 1.0 - i * 0.08)
                ax.plot(range(n_fr), ch_mean,
                        color=bin_color, linewidth=1.2, alpha=alpha,
                        label=f"{seg['duration_ms']:.0f}ms ({n_fr}fr)"
                        if i < 5 else None)

            ax.set_title(f"{bin_name}\n({len(bin_segs)} instances, "
                         f"showing {len(show_segs)})", fontsize=10)
            ax.set_xlabel("Frame (raw)")
            if col == 0:
                ax.set_ylabel("RAW\nHG power (ch avg)")
            if len(show_segs) <= 5:
                ax.legend(fontsize=7, loc="upper right")

        # Row 0, last col: resampled
        ax = axes[0, len(bins)]
        for bin_name, bin_fn, bin_color in bins:
            bin_segs = binned[bin_name]["segments"]
            show_segs = bin_segs[:max_instances_per_bin]
            for i, seg in enumerate(show_segs):
                hg = seg["hg"]
                ch_mean = hg.mean(axis=1)
                resampled = resample(ch_mean, target_frames)
                alpha = max(0.3, 1.0 - i * 0.08)
                ax.plot(range(target_frames), resampled,
                        color=bin_color, linewidth=1.0, alpha=alpha)

        legend_elements = []
        for bin_name, bin_fn, bin_color in bins:
            n = len(binned[bin_name]["segments"])
            legend_elements.append(
                Line2D([0], [0], color=bin_color, linewidth=2,
                       label=f"{bin_name} (n={n})"))
        ax.legend(handles=legend_elements, fontsize=8, loc="upper right")
        ax.set_title(f"Resampled to {target_frames} frames\n"
                     f"(all bins overlaid)", fontsize=10)
        ax.set_xlabel(f"Frame (resampled to {target_frames})")

        # Row 1: stacked per bin + stacked all
        from extract_features import stackFeatures
        margin = model_order * step_size
        n_context = 2 * model_order + 1
        x_labels = list(range(-model_order, model_order + 1))

        for col, (bin_name, bin_fn, bin_color) in enumerate(bins):
            ax = axes[1, col]
            bin_segs = binned[bin_name]["segments"]

            if not bin_segs:
                ax.set_title(f"{bin_name}\n(0 instances)", fontsize=10)
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=11, color="gray")
                ax.set_xlabel("Context offset")
                continue

            show_segs = bin_segs[:max_instances_per_bin]
            for i, seg in enumerate(show_segs):
                hg = seg["hg"]
                n_frames = hg.shape[0]
                n_channels = hg.shape[1]

                min_needed = 2 * margin + 1
                if n_frames < min_needed:
                    pad_needed = min_needed - n_frames
                    pad_before = pad_needed // 2
                    pad_after = pad_needed - pad_before
                    hg_padded = np.pad(
                        hg, ((pad_before, pad_after), (0, 0)),
                        mode='edge')
                else:
                    hg_padded = hg

                stacked = stackFeatures(
                    hg_padded,
                    modelOrder=model_order,
                    stepSize=step_size)

                if stacked.shape[0] == 0:
                    continue

                center = stacked.shape[0] // 2
                stacked_vec = stacked[center]
                stacked_2d = stacked_vec.reshape(n_context, n_channels)
                context_profile = stacked_2d.mean(axis=1)

                alpha = max(0.3, 1.0 - i * 0.08)
                ax.plot(x_labels, context_profile,
                        color=bin_color, linewidth=1.0, alpha=alpha)

            ax.axvline(0, color="gray", linestyle="--", alpha=0.5,
                        linewidth=0.8)
            ax.set_title(f"{bin_name}\n(stacked, {len(bin_segs)} inst)",
                         fontsize=10)
            ax.set_xlabel(f"Context offset "
                          f"(x{step_size}={step_size * pipeline.config.frameshift * 1000:.0f}ms)")
            if col == 0:
                ax.set_ylabel(f"STACKED (order={model_order})\n"
                              f"HG power (ch avg)")

        # Row 1, last col: stacked all bins overlaid
        ax = axes[1, len(bins)]
        for bin_name, bin_fn, bin_color in bins:
            bin_segs = binned[bin_name]["segments"]
            show_segs = bin_segs[:max_instances_per_bin]
            for i, seg in enumerate(show_segs):
                hg = seg["hg"]
                n_frames = hg.shape[0]
                n_channels = hg.shape[1]

                min_needed = 2 * margin + 1
                if n_frames < min_needed:
                    pad_needed = min_needed - n_frames
                    pad_before = pad_needed // 2
                    pad_after = pad_needed - pad_before
                    hg_padded = np.pad(
                        hg, ((pad_before, pad_after), (0, 0)),
                        mode='edge')
                else:
                    hg_padded = hg

                stacked = stackFeatures(
                    hg_padded,
                    modelOrder=model_order,
                    stepSize=step_size)

                if stacked.shape[0] == 0:
                    continue

                center = stacked.shape[0] // 2
                stacked_vec = stacked[center]
                stacked_2d = stacked_vec.reshape(n_context, n_channels)
                context_profile = stacked_2d.mean(axis=1)

                alpha = max(0.3, 1.0 - i * 0.08)
                ax.plot(x_labels, context_profile,
                        color=bin_color, linewidth=1.0, alpha=alpha)

        ax.axvline(0, color="gray", linestyle="--", alpha=0.5,
                    linewidth=0.8)
        legend_elements_stack = []
        for bin_name, bin_fn, bin_color in bins:
            n = len(binned[bin_name]["segments"])
            legend_elements_stack.append(
                Line2D([0], [0], color=bin_color, linewidth=2,
                       label=f"{bin_name} (n={n})"))
        ax.legend(handles=legend_elements_stack, fontsize=8,
                  loc="upper right")
        ax.set_title(f"Stacked all bins\n"
                     f"(order={model_order}, step={step_size}, "
                     f"{n_context} ctx)", fontsize=10)
        ax.set_xlabel(f"Context offset "
                      f"(x{step_size}={step_size * pipeline.config.frameshift * 1000:.0f}ms)")

        plt.tight_layout(rect=[0, 0, 1, 0.93])
        plt.show()

        # Print frame distribution

        # Print frame distribution
        print(f"  /{phoneme}/ frame distribution:")
        for bin_name in ["4-10 frames", "11-20 frames", "20-25 frames", "25+ frames"]:
            segs = binned[bin_name]["segments"]
            if segs:
                durs = [s["duration_ms"] for s in segs]
                frs = [s["n_frames"] for s in segs]
                print(f"    {bin_name}: n={len(segs)}, "
                      f"frames={min(frs)}-{max(frs)}, "
                      f"duration={min(durs):.0f}-{max(durs):.0f}ms")
            else:
                print(f"    {bin_name}: n=0")
        print()

# # # # Auto-select 3 consonants + 3 vowels
# phoneme_duration_diagnostic(pipeline, pid="P27", phonemes=["t", "n", "s", "d", "k", "r",
#                                        "\u0259", "\u025b", "\u0251", "i"], target_frames= 9, model_order=9, step_size=2)

# # # # Or pick specific phonemes
# # # phoneme_duration_diagnostic(pipeline, pid="P23",
# # #                              phonemes=["t", "n", "s", "\u0259", "a:", "\u025b"])

import numpy as np
import torch
import torch.nn as nn
from collections import Counter
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score


MIN_CLASS_SAMPLES = 5
CHANNEL_HIDDEN = 16
GLOBAL_HIDDEN = 256
JOINT_EPOCHS = 150
FINETUNE_EPOCHS = 150
SCRATCH_EPOCHS = 600
BATCH_SIZE = 64
LR = 0.001
WEIGHT_DECAY = 1e-4
DEFAULT_STACKING_ORDER = 5
DEFAULT_N_FRAMES = 2 * DEFAULT_STACKING_ORDER + 1  # 11


class SinActivation(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.freq = nn.Parameter(torch.ones(dim) * 30.0)
        self.phase = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return torch.sin(self.freq * x + self.phase)


class SnakeActivation(nn.Module):
    def __init__(self, dim, alpha=1.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(dim) * alpha)

    def forward(self, x):
        return x + (1.0 / (self.alpha + 1e-8)) * torch.sin(self.alpha * x) ** 2


def build_model(n_features, n_classes, activation='relu'):
    """Build a three-layer MLP with the specified activation function.

    Args:
        n_features: int, input feature dimension.
        n_classes: int, number of output classes.
        activation: str, one of 'relu', 'sin', 'snake', 'sin_relu'.

    Returns:
        nn.Sequential model.
    """
    if activation == 'relu':
        act = nn.ReLU()
        act2 = nn.ReLU()
    elif activation == 'sin':
        act = SinActivation(256)
        act2 = SinActivation(128)
    elif activation == 'snake':
        act = SnakeActivation(256)
        act2 = SnakeActivation(128)
    elif activation == 'sin_relu':
        act = SinActivation(256)
        act2 = nn.ReLU()
    else:
        raise ValueError(f"unknown activation: {activation}")
    return nn.Sequential(
        nn.Linear(n_features, 256),
        act,
        nn.Dropout(0.3),
        nn.Linear(256, 128),
        act2,
        nn.Dropout(0.3),
        nn.Linear(128, n_classes)
    )


class Conv1DClassifier(nn.Module):
    def __init__(self, n_channels, n_frames, n_classes):
        """1D convolutional classifier over the time axis of neural signal features.

        Args:
            n_channels: int, number of sEEG channels for this patient.
            n_frames: int, number of time frames per channel.
            n_classes: int, number of phoneme classes.
        """
        super().__init__()
        self.n_channels = n_channels
        self.n_frames = n_frames
        self.conv = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        x = x.view(x.size(0), self.n_channels, self.n_frames)
        x = self.conv(x)
        x = x.squeeze(-1)
        return self.classifier(x)


class SharedChannelMLP(nn.Module):
    def __init__(self, n_frames_per_channel, n_classes,
                 channel_hidden=CHANNEL_HIDDEN, global_hidden=GLOBAL_HIDDEN):
        """Shared-weight MLP with switchable pooling or concatenation head.

        During joint pretraining, uses mean pooling over channels so the
        model is channel-count agnostic. During per-patient fine-tuning,
        a full-capacity concatenation head is attached via attach_patient_head.

        Args:
            n_frames_per_channel: int, number of time frames per channel.
            n_classes: int, number of output classes for joint pretraining.
            channel_hidden: int, hidden size of per-channel network.
            global_hidden: int, hidden size of global classifier.
        """
        super().__init__()
        self.n_frames = n_frames_per_channel
        self.channel_hidden = channel_hidden
        self.global_hidden = global_hidden
        self.patient_head = None

        self.channel_net = nn.Sequential(
            nn.Linear(n_frames_per_channel, channel_hidden),
            nn.ReLU(),
        )

        self.global_net = nn.Sequential(
            nn.Linear(channel_hidden, global_hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(global_hidden, n_classes),
        )

    def attach_patient_head(self, n_channels, n_classes):
        """Attach a full-capacity concatenation head sized for this patient.

        Call after loading pretrained channel_net weights and before
        per-patient fine-tuning.

        Args:
            n_channels: int, number of sEEG channels for this patient.
            n_classes: int, number of phoneme classes for this patient.
        """
        self.patient_head = nn.Sequential(
            nn.Linear(n_channels * self.channel_hidden, self.global_hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.global_hidden, n_classes),
        )

    def forward(self, x, n_channels):
        """Forward pass.

        Args:
            x: tensor of shape (batch, n_channels * n_frames).
            n_channels: int, number of channels for this batch.

        Returns:
            tensor of shape (batch, n_classes).
        """
        x = x.view(x.size(0), n_channels, self.n_frames)
        channel_outs = self.channel_net(x)
        if self.patient_head is not None:
            combined = channel_outs.view(x.size(0), -1)
            return self.patient_head(combined)
        pooled = channel_outs.mean(dim=1)
        return self.global_net(pooled)


def train_nn(model, X_tr_t, y_tr_t, nn_weights, epochs, n_channels):
    """Train a SharedChannelMLP for a fixed number of epochs.

    Args:
        model: SharedChannelMLP instance.
        X_tr_t: FloatTensor of shape (n_samples, n_channels * n_frames).
        y_tr_t: LongTensor of shape (n_samples,).
        nn_weights: FloatTensor of shape (n_classes,) for loss weighting.
        epochs: int, number of training epochs.
        n_channels: int, channel count for this dataset.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss(weight=nn_weights)
    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(len(X_tr_t))
        for i in range(0, len(X_tr_t), BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            out = model(X_tr_t[idx], n_channels)
            loss = criterion(out, y_tr_t[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def collect_joint_data(all_pids, pipeline, n_frames_per_ch, le_joint):
    """Collect and scale features across all patients for joint pretraining.

    Each patient's features are scaled independently before stacking,
    since channel scales differ across patients.

    Args:
        all_pids: list of str, patient ids to include.
        pipeline: pipeline object with train data.
        n_frames_per_ch: int, frames per channel.
        le_joint: fitted LabelEncoder on the union of all valid classes.

    Returns:
        tuple of (X_joint, y_joint, n_channels_per_pid, scalers_per_pid)
        where X_joint is a list of (FloatTensor, int) pairs
        and scalers_per_pid is a dict of pid -> StandardScaler.
    """
    X_joint = []
    y_joint = []
    n_channels_per_pid = {}
    scalers_per_pid = {}

    for pid in all_pids:
        train_mask = [p == pid for p in pipeline.train['phoneme_participant_ids']]
        train_feat = [
            pipeline.train['features'][i] for i, m in enumerate(train_mask) if m
        ]
        train_labels = [
            pipeline.train['phoneme_labels'][i] for i, m in enumerate(train_mask) if m
        ]

        if len(train_feat) < 10:
            continue

        X = np.array(train_feat)
        y = np.array(train_labels)

        train_counts = Counter(y)
        valid_classes = {
            cls for cls, cnt in train_counts.items() if cnt >= MIN_CLASS_SAMPLES
        }
        keep = np.array([
            lbl in valid_classes and lbl in le_joint.classes_ for lbl in y
        ])
        X, y = X[keep], y[keep]

        if len(X) < 10:
            continue

        n_features = X.shape[1]
        if n_features % n_frames_per_ch != 0:
            continue

        n_channels = n_features // n_frames_per_ch
        n_channels_per_pid[pid] = n_channels

        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)
        scalers_per_pid[pid] = scaler

        y_enc = le_joint.transform(y)
        X_joint.append((torch.FloatTensor(X_s), n_channels))
        y_joint.append(torch.LongTensor(y_enc))

    return X_joint, y_joint, n_channels_per_pid, scalers_per_pid


def build_joint_label_encoder(all_pids, pipeline):
    """Build a label encoder over the union of valid classes across all patients.

    Args:
        all_pids: list of str, patient ids.
        pipeline: pipeline object with train data.

    Returns:
        fitted LabelEncoder.
    """
    all_valid = set()
    for pid in all_pids:
        train_mask = [p == pid for p in pipeline.train['phoneme_participant_ids']]
        train_labels = [
            pipeline.train['phoneme_labels'][i] for i, m in enumerate(train_mask) if m
        ]
        counts = Counter(train_labels)
        all_valid.update(cls for cls, cnt in counts.items() if cnt >= MIN_CLASS_SAMPLES)
    le = LabelEncoder()
    le.fit(sorted(all_valid))
    return le


# --- detect n_frames_per_ch from run_config ---

so = run_config.get('stacking_order')
if so is not None:
    n_frames_per_ch = 2 * so + 1
else:
    tf = run_config.get('target_frames')
    if tf is not None:
        n_frames_per_ch = tf
    else:
        print(
            f"warning: stacking_order and target_frames not found in run_config. "
            f"defaulting to stacking_order={DEFAULT_STACKING_ORDER}, "
            f"n_frames_per_ch={DEFAULT_N_FRAMES}."
        )
        n_frames_per_ch = DEFAULT_N_FRAMES

skip_patients = {}
all_pids = sorted(set(pipeline.train['phoneme_participant_ids']))
all_pids = [p for p in all_pids if p not in skip_patients]

print(f"running on {len(all_pids)} patients: {all_pids}")
print(f"n_frames_per_ch={n_frames_per_ch}")
print()

# --- joint pretraining of SharedChannelMLP ---

pretrained_channel_net = None

print("pretraining shared_ch on all patients jointly...")
le_joint = build_joint_label_encoder(all_pids, pipeline)
n_joint_classes = len(le_joint.classes_)

X_joint, y_joint, n_channels_per_pid, scalers_per_pid = collect_joint_data(
    all_pids, pipeline, n_frames_per_ch, le_joint
)

joint_model = SharedChannelMLP(n_frames_per_ch, n_joint_classes)
joint_weights = torch.ones(n_joint_classes)
joint_model.train()
optimizer = torch.optim.Adam(joint_model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
criterion = nn.CrossEntropyLoss(weight=joint_weights)

for epoch in range(JOINT_EPOCHS):
    order = torch.randperm(len(X_joint))
    for idx in order:
        X_p, n_ch = X_joint[idx]
        y_p = y_joint[idx]
        perm = torch.randperm(len(X_p))
        for i in range(0, len(X_p), BATCH_SIZE):
            bidx = perm[i:i + BATCH_SIZE]
            out = joint_model(X_p[bidx], n_ch)
            loss = criterion(out, y_p[bidx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

pretrained_channel_net = joint_model.channel_net.state_dict()
print(f"joint pretraining done over {len(X_joint)} patients, {n_joint_classes} classes")
print()

# --- per patient loop ---

patient_results = {}

for pid in all_pids:
    train_mask = [p == pid for p in pipeline.train['phoneme_participant_ids']]
    test_mask = [p == pid for p in pipeline.test['phoneme_participant_ids']]

    train_feat = [
        pipeline.train['features'][i] for i, m in enumerate(train_mask) if m
    ]
    train_labels = [
        pipeline.train['phoneme_labels'][i] for i, m in enumerate(train_mask) if m
    ]
    test_feat = [
        pipeline.test['features'][i] for i, m in enumerate(test_mask) if m
    ]
    test_labels = [
        pipeline.test['phoneme_labels'][i] for i, m in enumerate(test_mask) if m
    ]

    if len(train_feat) < 10 or len(test_feat) < 5:
        continue

    X_train = np.array(train_feat)
    y_train = np.array(train_labels)
    X_test = np.array(test_feat)
    y_test = np.array(test_labels)

    train_counts = Counter(y_train)
    valid_classes = {
        cls for cls, cnt in train_counts.items() if cnt >= MIN_CLASS_SAMPLES
    }
    train_keep = np.array([y in valid_classes for y in y_train])
    test_keep = np.array([y in valid_classes for y in y_test])

    X_train = X_train[train_keep]
    y_train = y_train[train_keep]
    X_test = X_test[test_keep]
    y_test = y_test[test_keep]

    if len(X_train) < 10 or len(X_test) < 5:
        continue

    le = LabelEncoder()
    le.fit(sorted(valid_classes))
    n_classes = len(le.classes_)
    chance = 1.0 / n_classes

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    n_features = X_train_s.shape[1]
    y_train_enc = le.transform(y_train)
    y_test_enc = le.transform(y_test)

    full_counts = Counter(y_train_enc)
    full_total = sum(full_counts.values())
    nn_weights = torch.FloatTensor([
        full_total / (n_classes * full_counts.get(i, 1))
        for i in range(n_classes)
    ])

    n_channels_detected = None
    if n_features % n_frames_per_ch == 0:
        n_channels_detected = n_features // n_frames_per_ch

    print(f"--- {pid} (train={len(X_train)}, test={len(X_test)}, "
          f"classes={n_classes}, chance={chance:.4f}, "
          f"dropped={len(train_counts) - len(valid_classes)}, "
          f"channels={n_channels_detected}, frames={n_frames_per_ch}) ---")

    patient_results[pid] = {}

    classifiers = {
        'LogReg': LogisticRegression(
            max_iter=1000, class_weight='balanced', random_state=42),
        # 'SVM_RBF': SVC(
        #     kernel='rbf', C=10, gamma='scale', class_weight='balanced', random_state=42),
        # 'SVM_poly2': SVC(
        #     kernel='poly', degree=2, C=10, gamma='scale', class_weight='balanced',
        #     random_state=42),
        # 'SVM_linear': SVC(
        #     kernel='linear', C=1, class_weight='balanced', random_state=42),
        # 'RF': RandomForestClassifier(
        #     n_estimators=500, max_depth=None, min_samples_leaf=1,
        #     class_weight='balanced', random_state=37, n_jobs=-1),
        # 'ExtraTrees': ExtraTreesClassifier(
        #     n_estimators=1000, max_depth=None, min_samples_leaf=1,
        #     class_weight='balanced', random_state=37, n_jobs=-1),
    }

    for name, clf in classifiers.items():
        clf.fit(X_train_s, y_train)
        preds = clf.predict(X_test_s)
        acc = accuracy_score(y_test, preds)
        n_pred = len(set(preds))
        adj = acc * (n_pred / n_classes)
        patient_results[pid][name] = {
            'acc': acc, 'adj': adj, 'lift': acc / chance,
            'n_pred': n_pred, 'n_classes': n_classes,
            'test_size': len(X_test),
        }
        print(f"  {name:<15} acc={acc:.4f}  lift={acc/chance:.2f}x  "
              f"cls={n_pred}/{n_classes}  adj={adj:.4f}")

    X_tr_t = torch.FloatTensor(X_train_s)
    y_tr_t = torch.LongTensor(y_train_enc)
    X_te_t = torch.FloatTensor(X_test_s)

    nn_models = {
        'NN_relu': lambda: build_model(n_features, n_classes, activation='relu'),
        'NN_snake': lambda: build_model(n_features, n_classes, activation='snake'),
    }

    if n_channels_detected is not None:
        nn_models['NN_conv1d'] = lambda: Conv1DClassifier(
            n_channels_detected, n_frames_per_ch, n_classes)

        def make_scratch_shared_ch():
            m = SharedChannelMLP(n_frames_per_ch, n_classes)
            m.attach_patient_head(n_channels_detected, n_classes)
            return m
        nn_models['NN_shared_ch'] = make_scratch_shared_ch

        if pretrained_channel_net is not None:
            def make_pretrained_shared_ch():
                m = SharedChannelMLP(n_frames_per_ch, n_classes)
                m.channel_net.load_state_dict(pretrained_channel_net)
                m.attach_patient_head(n_channels_detected, n_classes)
                return m
            nn_models['NN_shared_ch_pt'] = make_pretrained_shared_ch

    for nn_name, model_fn in nn_models.items():
        torch.manual_seed(42)
        model = model_fn()

        if isinstance(model, SharedChannelMLP):
            epochs = FINETUNE_EPOCHS if nn_name == 'NN_shared_ch_pt' else SCRATCH_EPOCHS
            train_nn(model, X_tr_t, y_tr_t, nn_weights, epochs, n_channels_detected)
            model.eval()
            with torch.no_grad():
                preds_enc = model(X_te_t, n_channels_detected).argmax(dim=1).numpy()
        else:
            optimizer = torch.optim.Adam(
                model.parameters(), lr=0.001, weight_decay=1e-4)
            criterion = nn.CrossEntropyLoss(weight=nn_weights)
            model.train()
            for epoch in range(300):
                perm = torch.randperm(len(X_tr_t))
                for i in range(0, len(X_tr_t), 64):
                    idx = perm[i:i + 64]
                    out = model(X_tr_t[idx])
                    loss = criterion(out, y_tr_t[idx])
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            model.eval()
            with torch.no_grad():
                preds_enc = model(X_te_t).argmax(dim=1).numpy()

        preds = le.inverse_transform(preds_enc)
        acc = accuracy_score(y_test, preds)
        n_pred = len(set(preds))
        adj = acc * (n_pred / n_classes)
        patient_results[pid][nn_name] = {
            'acc': acc, 'adj': adj, 'lift': acc / chance,
            'n_pred': n_pred, 'n_classes': n_classes,
            'test_size': len(X_test),
        }
        print(f"  {nn_name:<15} acc={acc:.4f}  lift={acc/chance:.2f}x  "
              f"cls={n_pred}/{n_classes}  adj={adj:.4f}")

    print()

# --- weighted summary ---

print("-" * 70)
print("mean across patients (weighted by test size)")
print("%-20s  %8s  %8s  %8s  %8s" % ('classifier', 'acc', 'adjAcc', 'lift', 'classes'))
print("-" * 60)
all_clf_names = sorted(set().union(*(r.keys() for r in patient_results.values())))
for name in all_clf_names:
    accs, adjs, lifts, cls_pcts, weights = [], [], [], [], []
    for pid in patient_results:
        if name not in patient_results[pid]:
            continue
        r = patient_results[pid][name]
        accs.append(r['acc'])
        adjs.append(r['adj'])
        lifts.append(r['lift'])
        cls_pcts.append(r['n_pred'] / r['n_classes'])
        weights.append(r['test_size'])
    w = np.array(weights, dtype=float)
    w = w / w.sum()
    print("%-20s  %7.2f%%  %7.2f%%  %7.2fx  %7.1f%%" % (
        name,
        np.average(accs, weights=w) * 100,
        np.average(adjs, weights=w) * 100,
        np.average(lifts, weights=w),
        np.average(cls_pcts, weights=w) * 100))

import torch
import torch.nn as nn
import numpy as np
from collections import Counter
from sklearn.preprocessing import StandardScaler, LabelEncoder


class TorchModelWrapper:
    """Wraps a pytorch classifier to match the step9 model interface.

    Handles per-patient scaling, label encoding, class filtering, and
    converts between the pipeline's list-of-arrays format and the flat
    tensor format expected by pytorch models.

    Args:
        model_fn: callable with no arguments that returns a fresh pytorch
            nn.Module. Called at train time once class count is known.
        n_frames_per_ch: int, number of time frames per channel.
        min_class_samples: int, minimum training samples required per class.
        n_epochs: int, number of training epochs.
        batch_size: int, mini-batch size.
        lr: float, Adam learning rate.
        weight_decay: float, Adam weight decay.
        is_shared_ch: bool, if True routes forward calls through
            SharedChannelMLP interface which requires n_channels argument.
    """

    def __init__(self, model_fn, n_frames_per_ch,
                 min_class_samples=MIN_CLASS_SAMPLES,
                 n_epochs=300, batch_size=BATCH_SIZE,
                 lr=LR, weight_decay=WEIGHT_DECAY,
                 is_shared_ch=False):
        self.model_fn = model_fn
        self.n_frames_per_ch = n_frames_per_ch
        self.min_class_samples = min_class_samples
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.is_shared_ch = is_shared_ch
        self.model = None
        self.scaler = None
        self.le = None
        self.n_channels = None
        self.markov_model = None

    def train(self, features, phoneme_labels):
        """Fit scaler, label encoder, and train the pytorch model.

        Args:
            features: list of np.ndarray, one per phoneme instance.
            phoneme_labels: list of str, one label per instance.
        """
        X = np.array(features)
        y = np.array(phoneme_labels)

        counts = Counter(y)
        valid_classes = {
            cls for cls, cnt in counts.items()
            if cnt >= self.min_class_samples
        }
        keep = np.array([lbl in valid_classes for lbl in y])
        X, y = X[keep], y[keep]

        self.le = LabelEncoder()
        self.le.fit(sorted(valid_classes))

        from markov_phoneme_model import MarkovPhonemeModel
        self.markov_model = MarkovPhonemeModel()
        self.markov_model.trained_classes = sorted(list(set(y)))
        self.markov_model.class_to_index = {
            cls: i for i, cls in enumerate(self.markov_model.trained_classes)
        }
        self.markov_model.index_to_class = {
            i: cls for cls, i in self.markov_model.class_to_index.items()
        }
        self.markov_model._build_corpus_transition_model()
        
        n_classes = len(self.trained_classes)

        self.scaler = StandardScaler()
        X_s = self.scaler.fit_transform(X)

        n_features = X_s.shape[1]
        if self.n_frames_per_ch and n_features % self.n_frames_per_ch == 0:
            self.n_channels = n_features // self.n_frames_per_ch

        y_enc = self.le.transform(y)

        class_counts = Counter(y_enc)
        total = sum(class_counts.values())
        weights = torch.FloatTensor([
            total / (n_classes * class_counts.get(i, 1))
            for i in range(n_classes)
        ])

        self.model = self.model_fn(self.n_channels, n_classes)
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        criterion = nn.CrossEntropyLoss(weight=weights)

        X_t = torch.FloatTensor(X_s)
        y_t = torch.LongTensor(y_enc)

        self.model.train()
        for epoch in range(self.n_epochs):
            perm = torch.randperm(len(X_t))
            for i in range(0, len(X_t), self.batch_size):
                idx = perm[i:i + self.batch_size]
                if self.is_shared_ch:
                    out = self.model(X_t[idx], self.n_channels)
                else:
                    out = self.model(X_t[idx])
                loss = criterion(out, y_t[idx])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def predict(self, features, use_viterbi=False):
        """Predict phoneme labels and return class probabilities.

        Args:
            features: list of np.ndarray, one per phoneme instance.
            use_viterbi: bool, ignored — kept for interface compatibility.

        Returns:
            tuple of (predictions, probabilities) where predictions is a
            list of str and probabilities is an np.ndarray of shape
            (n_samples, n_classes).
        """
        X = np.array(features)
        X_s = self.scaler.transform(X)
        X_t = torch.FloatTensor(X_s)

        self.model.eval()
        with torch.no_grad():
            if self.is_shared_ch:
                logits = self.model(X_t, self.n_channels)
            else:
                logits = self.model(X_t)
            probs = torch.softmax(logits, dim=1).numpy()

        pred_enc = probs.argmax(axis=1)
        predictions = [str(p) for p in self.le.inverse_transform(pred_enc)]
        return predictions, probs

    def _viterbi_decode(self, agg_probs):
        """Delegate viterbi decoding to the markov model.
    
        Args:
            agg_probs: np.ndarray of shape (n_phonemes, n_classes) containing
                aggregated emission probabilities from the pytorch classifier.
    
        Returns:
            list of int, indices into self.trained_classes.
        """
        print(f"viterbi called: agg_probs shape={agg_probs.shape}, "
           f"n_trained_classes={len(self.trained_classes)}")
        return self.markov_model._viterbi_decode(agg_probs)

    @property
    def trained_classes(self):
        """Phoneme classes known to the model, delegated to markov model.
    
        Returns:
            list of str, or None if model has not been trained yet.
        """
        if self.markov_model is None:
            return None
        return self.markov_model.trained_classes

# NN_conv1d
def conv1d_factory(n_channels, n_classes):
    return Conv1DClassifier(n_channels, n_frames_per_ch, n_classes)
    
def shared_ch_pt_factory(n_channels, n_classes):
    m = SharedChannelMLP(n_frames_per_ch, n_classes)
    m.channel_net.load_state_dict(pretrained_channel_net)
    m.attach_patient_head(n_channels, n_classes)
    return m

_ = pipeline.step9_train_and_evaluate(
    model_factory=TorchModelWrapper,
    model_params={
        'model_fn': conv1d_factory,
        'n_frames_per_ch': n_frames_per_ch,
        'n_epochs': 300,
    },
    use_viterbi=True,
)

def build_instance_ids(words, participant_ids):
    """Construct instance ids by detecting word boundary changes.

    Each contiguous run of the same word for the same patient is treated
    as one instance. Does not distinguish between non-consecutive
    repetitions of the same word.

    Args:
        words: list of str, word label per frame.
        participant_ids: list of str, patient id per frame.

    Returns:
        list of str, instance id per frame.
    """
    instance_ids = []
    current_instance = None
    instance_counter = 0
    for word, pid in zip(words, participant_ids):
        key = (pid, word)
        if key != current_instance:
            current_instance = key
            instance_counter += 1
        instance_ids.append(f"{pid}_{word}_{instance_counter}")
    return instance_ids

pipeline.train['phoneme_instance_ids'] = build_instance_ids(
    pipeline.train['phoneme_words'],
    pipeline.train['phoneme_participant_ids']
)
pipeline.test['phoneme_instance_ids'] = build_instance_ids(
    pipeline.test['phoneme_words'],
    pipeline.test['phoneme_participant_ids']
)

print('phoneme_instance_ids' in pipeline.test)
print("sample ids:", pipeline.test['phoneme_instance_ids'][:15])

# monkey-patch step9 logging temporarily
original_step9 = pipeline.step9_train_and_evaluate

def debug_step9(*args, **kwargs):
    print("has_instance_ids:", 'phoneme_instance_ids' in pipeline.test)
    print("use_viterbi:", kwargs.get('use_viterbi', True))
    return original_step9(*args, **kwargs)

_ = debug_step9(
    model_factory=TorchModelWrapper,
    model_params={
        'model_fn': conv1d_factory,
        'n_frames_per_ch': n_frames_per_ch,
        'n_epochs': 300,
    },
    use_viterbi=True,
)

for pid in sorted(pipeline.patient_results.keys()):
    pipeline.step10_visualize_patient(pid, show_table=False)

def diagnose_word_segmentation(pipeline, word, patient_id, n_instances=3):
    """Visualize acoustic boundary detection for specific word instances.

    Shows the mel spectrogram, acoustic distance curve, detected boundaries,
    and how they map to dictionary phonemes for a given word and patient.

    Args:
        pipeline: pipeline object with detector and phonetic_dict.
        word: str, word to diagnose.
        patient_id: str, patient id to look up instances for.
        n_instances: int, number of instances to visualize.
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import numpy as np
    from collections import defaultdict

    # get dictionary phonemes
    phonemes_dict = pipeline.phonetic_dict.extract_phonemes(word)
    print(f"word: '{word}'")
    print(f"dictionary phonemes: {phonemes_dict} (n={len(phonemes_dict)})")
    print(f"expected boundaries: {len(phonemes_dict) - 1}")
    print()

    # find instances in train and test data
    instances = []
    for split_name, data in [('train', pipeline.train), ('test', pipeline.test)]:
        word_mask = [w == word and p == patient_id
                     for w, p in zip(data['phoneme_words'],
                                     data['phoneme_participant_ids'])]
        indices = [i for i, m in enumerate(word_mask) if m]
        if indices:
            instances.append((split_name, indices[:n_instances]))

    if not instances:
        print(f"no instances of '{word}' found for {patient_id}")
        return

    # now re-run boundary detection on raw data
    # find the word in the split_result to get raw audio and eeg
    split_result = pipeline.split_result
    patient_words = None
    for pid, pdata in split_result.items():
        if pid == patient_id:
            patient_words = pdata
            break

    if patient_words is None:
        print(f"patient {patient_id} not found in split_result")
        return

    # find word instances in the raw data
    word_instances = []
    for split_name in ['train', 'test']:
        if split_name not in patient_words:
            continue
        split_data = patient_words[split_name]
        for inst_idx, inst in enumerate(split_data.get('instances', [])):
            if inst.get('word') == word or split_data.get('words_list', [None])[inst_idx] == word:
                word_instances.append((split_name, inst_idx, inst))
        if 'words_list' in split_data:
            for inst_idx, w in enumerate(split_data['words_list']):
                if w == word:
                    word_instances.append((split_name, inst_idx, None))

    print(f"searching for raw instances...")
    print()

    # alternative: re-run detector directly on stored spectrogram segments
    # find spectrogram segments from the pipeline data
    for split_name, data in [('train', pipeline.train), ('test', pipeline.test)]:
        word_mask = [w == word and p == patient_id
                     for w, p in zip(data['phoneme_words'],
                                     data['phoneme_participant_ids'])]
        indices = [i for i, m in enumerate(word_mask) if m]
        if not indices:
            continue

        # get unique instance ids for this word
        inst_ids = [data['phoneme_instance_ids'][i] for i in indices]
        unique_insts = list(dict.fromkeys(inst_ids))[:n_instances]

        for inst_id in unique_insts:
            inst_indices = [i for i in indices
                            if data['phoneme_instance_ids'][i] == inst_id]

            # get labels and positions for this instance
            inst_labels = [data['phoneme_labels'][i] for i in inst_indices]
            inst_positions = [data['phoneme_positions'][i] for i in inst_indices]
            inst_spectrograms = [data['spectrograms'][i] for i in inst_indices]

            # collapse to unique positions
            pos_to_label = {}
            for pos, lbl in zip(inst_positions, inst_labels):
                if pos not in pos_to_label:
                    pos_to_label[pos] = lbl
            detected_positions = sorted(pos_to_label.keys())
            detected_labels = [pos_to_label[p] for p in detected_positions]

            print(f"instance: {inst_id} ({split_name})")
            print(f"  detected positions: {detected_positions}")
            print(f"  detected labels:    {detected_labels}")
            print(f"  dictionary:         {phonemes_dict}")
            print(f"  n_detected={len(detected_positions)}  n_dict={len(phonemes_dict)}  "
                  f"match={'yes' if len(detected_positions) == len(phonemes_dict) else 'NO - MISMATCH'}")

            # get one spectrogram per position for visualization
            pos_to_spec = {}
            for pos, spec in zip(inst_positions, inst_spectrograms):
                if pos not in pos_to_spec:
                    pos_to_spec[pos] = spec

            n_detected = len(detected_positions)
            fig = plt.figure(figsize=(14, 6))
            gs = gridspec.GridSpec(2, max(n_detected, len(phonemes_dict)) + 1)

            fig.suptitle(
                f"{inst_id} — detected {n_detected} segments, "
                f"expected {len(phonemes_dict)} phonemes\n"
                f"dict: {phonemes_dict}   detected: {detected_labels}",
                fontsize=10
            )

            # top row: spectrograms per detected segment
            for col, pos in enumerate(detected_positions):
                ax = fig.add_subplot(gs[0, col])
                spec = pos_to_spec[pos]
                ax.imshow(spec.T, aspect='auto', origin='lower', cmap='viridis')
                dict_label = phonemes_dict[pos] if pos < len(phonemes_dict) else '?'
                detected_label = pos_to_label[pos]
                match = 'ok' if dict_label == detected_label else 'MISMATCH'
                ax.set_title(
                    f"pos {pos}\ndict: {dict_label}\ndetected: {detected_label}\n{match}",
                    fontsize=8
                )
                ax.set_xlabel("mel filters")
                ax.set_ylabel("frames")

            # bottom row: show what dictionary says should be there
            for col, ph in enumerate(phonemes_dict):
                ax = fig.add_subplot(gs[1, col])
                ax.text(0.5, 0.5, ph, ha='center', va='center',
                        fontsize=20, fontweight='bold')
                ax.set_title(f"dict pos {col}", fontsize=8)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis('off')

            plt.tight_layout()
            plt.show()
            print()

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

def diagnose_boundary_detection(pipeline, pid, word, instance_idx=0):
    """Re-run and visualize boundary detection for a single word instance.

    Shows the mel spectrogram, acoustic distance curve, detected peaks,
    final boundaries, and how segments map to dictionary phonemes.

    Args:
        pipeline: pipeline object with detector and phonetic_dict.
        pid: str, patient id.
        word: str, word to diagnose.
        instance_idx: int, which instance of the word to use.
    """
    wsd = pipeline.split_result['word_segments_dict'][pid]
    words_list = wsd['words_list']

    indices = [i for i, w in enumerate(words_list) if w == word]
    if not indices:
        print(f"'{word}' not found for {pid}")
        return
    if instance_idx >= len(indices):
        print(f"only {len(indices)} instances available")
        return

    idx = indices[instance_idx]
    spectrogram = wsd['spectrogram_segments'][idx]
    audio = wsd['audio_segments'][idx]
    eeg = wsd['eeg_segments'][idx]

    phonemes_dict = pipeline.phonetic_dict.extract_phonemes(word)
    n_phonemes = len(phonemes_dict)
    n_boundaries_needed = n_phonemes - 1

    print(f"word: '{word}'  instance: {instance_idx}")
    print(f"dictionary phonemes: {phonemes_dict}  (n={n_phonemes})")
    print(f"boundaries needed: {n_boundaries_needed}")
    print(f"spectrogram shape: {spectrogram.shape}  "
          f"(n_frames={spectrogram.shape[0]}, n_filters={spectrogram.shape[1]})")
    print()

    # re-run boundary detection step by step
    detector = pipeline.detector

    # compute frame-to-frame distances (same as detect_boundaries internally)
    from scipy.spatial.distance import cosine
    distances = np.array([
        cosine(spectrogram[i], spectrogram[i+1])
        for i in range(len(spectrogram) - 1)
    ])

    # run full detect_boundaries to get the actual result
    result = detector.detect_boundaries(
        spectrogram=spectrogram,
        word=word,
        participant_id=pid,
        word_position=instance_idx,
        use_multifeature=detector.use_multifeature,
        use_rms_boundaries=detector.use_rms_boundaries,
        audio_segment=audio,
        audio_sr=pipeline.config.audio_sr,
    )

    boundaries_frames = result['boundaries']
    segments = result['segments']
    n_detected = len(segments)

    print(f"detected segments: {n_detected}  (needed: {n_phonemes})")
    print(f"boundary frames: {boundaries_frames}")
    match = n_detected == n_phonemes
    print(f"match: {'yes' if match else 'NO - MISMATCH'}")
    print()

    # assign labels same way pipeline does
    if match:
        assigned_labels = phonemes_dict
        label_source = 'direct mapping'
    else:
        assigned_labels = ['?'] * n_detected
        label_source = 'mismatch - all unknown'
        # show what resolve_unknown would give by position
        resolved = [
            phonemes_dict[j] if j < len(phonemes_dict) else '?'
            for j in range(n_detected)
        ]
        print(f"after resolve_unknown (by position index): {resolved}")
    print(f"label source: {label_source}")
    print(f"assigned labels: {assigned_labels}")
    print()

    # plot
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(3, 1, height_ratios=[3, 2, 1])

    # panel 1: full spectrogram with boundary lines
    ax1 = fig.add_subplot(gs[0])
    ax1.imshow(spectrogram.T, aspect='auto', origin='lower',
               cmap='viridis', interpolation='nearest')
    for b in boundaries_frames[1:-1]:
        ax1.axvline(x=b, color='red', linewidth=2, linestyle='--', label='boundary')
    ax1.set_title(
        f"{pid} '{word}' — spectrogram with detected boundaries\n"
        f"dict: {phonemes_dict}   detected: {n_detected} segments   "
        f"{'MATCH' if match else 'MISMATCH'}",
        fontsize=10
    )
    ax1.set_ylabel("mel filter")
    ax1.set_xlabel("frame")

    # add phoneme labels at segment midpoints
    for j in range(len(boundaries_frames) - 1):
        mid = (boundaries_frames[j] + boundaries_frames[j+1]) / 2
        lbl = assigned_labels[j] if match else f"?→{resolved[j]}"
        ax1.text(mid, spectrogram.shape[1] * 0.9, lbl,
                 ha='center', va='top', color='white',
                 fontsize=12, fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))

    # panel 2: distance curve with peaks
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(distances, color='steelblue', linewidth=1.5, label='acoustic distance')
    for b in boundaries_frames[1:-1]:
        ax2.axvline(x=b, color='red', linewidth=2, linestyle='--')
    ax2.set_title("acoustic distance curve (peaks = detected boundaries)", fontsize=10)
    ax2.set_ylabel("cosine distance")
    ax2.set_xlabel("frame")
    ax2.legend(fontsize=8)

    # panel 3: segment lengths
    ax3 = fig.add_subplot(gs[2])
    seg_lengths = [boundaries_frames[j+1] - boundaries_frames[j]
                   for j in range(len(boundaries_frames) - 1)]
    colors = ['green' if match else 'orange'] * n_detected
    bars = ax3.bar(range(n_detected), seg_lengths, color=colors, alpha=0.7)
    ax3.set_xticks(range(n_detected))
    ax3.set_xticklabels(
        [f"seg {j}\n{assigned_labels[j] if match else f'?→{resolved[j]}'}"
         for j in range(n_detected)],
        fontsize=9
    )
    ax3.set_title("segment lengths in frames", fontsize=10)
    ax3.set_ylabel("frames")

    plt.tight_layout()
    plt.show()


# run it
diagnose_boundary_detection(pipeline, 'P21', 'mensen', instance_idx=0)

print(sorted(set(pipeline.train['phoneme_labels'])))
raw_phoneme_count = len(set(pipeline.train['phoneme_labels']))

# safety check — if labels are already groups, reload first
sample_labels = set(list(pipeline.train['phoneme_labels'])[:100])
known_groups = {'stops', 'fricatives', 'nasals', 'liquids', 'glides', 'schwa',
                'a-type', 'e-type', 'i-type', 'o-type', 'u-type', 'diph', 'unknown'}
if sample_labels.issubset(known_groups):
    print("WARNING: Labels are already groups! Reloading from checkpoint...")
    pipeline.try_load_checkpoint(sample_fraction=sf, stage='after_step5')
    pipeline.step5a_filter_by_frame_count(min_frames=2, max_frames=25)
    pipeline.step5b_normalize_feature_lengths(target_frames=5)
    pipeline.dutch30_step6_resolve_unknowns()
    pipeline.step7_filter_unknowns(unknown_keep_ratio=run_config['unknown_keep_ratio'])

raw_phoneme_count = len(set(pipeline.train['phoneme_labels']))
print(f"Raw phonemes before grouping: {sorted(set(pipeline.train['phoneme_labels']))[:10]}...")

pipeline.step8_group_phonemes()

grouped_phoneme_count = len(set(pipeline.train['phoneme_labels']))
print(f"\nReduced from {raw_phoneme_count} phonemes to {grouped_phoneme_count} groups")
grouped_results = pipeline.step9_train_and_evaluate(
    model_factory=MarkovPhonemeModel,
    model_params={"use_groups": False}  # already grouped
)

raw_phonemes = set(pipeline.train['phoneme_labels_raw'])
print(f"Raw phonemes in data ({len(raw_phonemes)}):")
print(sorted(raw_phonemes))

# Get the mapping
phoneme_to_group = pipeline.detector.phonetic_dict.phoneme_to_group
print(f"\nPhonemes in mapping ({len(phoneme_to_group)}):")
print(sorted(phoneme_to_group.keys()))

# Find which ones are NOT in the mapping
not_mapped = [p for p in raw_phonemes if p not in phoneme_to_group]
print(f"\nNOT mapped ({len(not_mapped)}):")
print(sorted(not_mapped))

# Find which ones ARE mapped
mapped = [p for p in raw_phonemes if p in phoneme_to_group]
print(f"\nMapped ({len(mapped)}):")
for p in sorted(mapped):
    print(f"  '{p}' -> '{phoneme_to_group[p]}'")


for pid in sorted(grouped_results.keys()):
    pipeline.step10_visualize_patient(pid)

import warnings
warnings.filterwarnings('ignore', message='.*number of unique classes.*')

def diagnose_feature_quality(pipeline, n_phonemes_to_check=10):
    """
    Check if features show ANY systematic differences between phonemes.
    """
    import numpy as np
    from collections import defaultdict, Counter
    from scipy.stats import ttest_ind
    import matplotlib.pyplot as plt
    
    phoneme_features = defaultdict(list)
    
    for feat, label in zip(pipeline.train['features'], pipeline.train['phoneme_labels']):
        if label == '?' or label == 'unknown':
            continue
        if feat.ndim > 1:
            feat_flat = np.mean(feat, axis=0)
        else:
            feat_flat = feat
        if not np.any(np.isnan(feat_flat)) and not np.any(np.isinf(feat_flat)):
            phoneme_features[label].append(feat_flat)
    
    # Filter to consistent shapes per phoneme
    phoneme_features_clean = {}
    
    for phoneme, features in phoneme_features.items():
        shape_counts = Counter(f.shape[0] for f in features)
        if not shape_counts:
            continue
        most_common_len = max(shape_counts, key=shape_counts.get)
        filtered = [f for f in features if f.shape[0] == most_common_len]
        
        if len(filtered) >= 10:
            phoneme_features_clean[phoneme] = np.array(filtered)
    
    print(f"Phonemes with consistent features: {len(phoneme_features_clean)}")
    
    # Get phonemes with most samples
    sorted_phonemes = sorted(phoneme_features_clean.keys(), key=lambda x: -len(phoneme_features_clean[x]))
    top_phonemes = sorted_phonemes[:n_phonemes_to_check]
    
    print(f"Top {len(top_phonemes)} phonemes by sample count:")
    for p in top_phonemes:
        print(f"  {p}: {len(phoneme_features_clean[p])} samples, {phoneme_features_clean[p].shape[1]} features")
    
    print("\nChecking feature separability between top phonemes:")
    print("=" * 70)
    
    pair_results = []
    
    for i, p1 in enumerate(top_phonemes):
        for p2 in top_phonemes[i+1:]:
            features_p1 = phoneme_features_clean[p1]
            features_p2 = phoneme_features_clean[p2]
            
            # Ensure same feature dimension
            min_dim = min(features_p1.shape[1], features_p2.shape[1])
            features_p1 = features_p1[:, :min_dim]
            features_p2 = features_p2[:, :min_dim]
            
            # Count significant features
            n_significant = 0
            for feat_idx in range(min_dim):
                stat, pval = ttest_ind(features_p1[:, feat_idx], features_p2[:, feat_idx])
                if pval < 0.01:
                    n_significant += 1
            
            pct_significant = n_significant / min_dim * 100
            pair_results.append((p1, p2, pct_significant, len(phoneme_features_clean[p1]), len(phoneme_features_clean[p2])))
    
    # Sort by percentage of significant features
    pair_results.sort(key=lambda x: -x[2])
    
    print(f"\n{'Phoneme 1':<10} {'Phoneme 2':<10} {'% Sig. Features':<15} {'N1':<6} {'N2':<6}")
    print("-" * 70)
    
    for p1, p2, pct, n1, n2 in pair_results[:20]:
        print(f"{p1:<10} {p2:<10} {pct:<15.1f} {n1:<6} {n2:<6}")
    
    # Summary
    avg_pct = np.mean([x[2] for x in pair_results])
    print(f"\nAverage % significant features across all pairs: {avg_pct:.1f}%")
    
    if avg_pct < 5:
        print("\nDIAGNOSIS: Very few features differ between phonemes.")
        print("  -> Features may not capture phoneme-relevant information")
        print("  -> Consider: different frequency bands, different time windows, electrode selection")
    elif avg_pct < 15:
        print("\nDIAGNOSIS: Weak but present differences between phonemes.")
        print("  -> Some signal exists but may need feature engineering")
    else:
        print("\nDIAGNOSIS: Moderate feature differences exist.")
        print("  -> Classification should be possible with right approach")
    
    # Visualize the best-separable pair
    if pair_results:
        best_pair = pair_results[0]
        p1, p2 = best_pair[0], best_pair[1]
        
        features_p1 = phoneme_features_clean[p1]
        features_p2 = phoneme_features_clean[p2]
        min_dim = min(features_p1.shape[1], features_p2.shape[1])
        features_p1 = features_p1[:, :min_dim]
        features_p2 = features_p2[:, :min_dim]
        
        # Find the two most discriminative features
        t_stats = []
        for feat_idx in range(min_dim):
            stat, pval = ttest_ind(features_p1[:, feat_idx], features_p2[:, feat_idx])
            t_stats.append(abs(stat))
        
        best_feat_indices = np.argsort(t_stats)[-2:]
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.scatter(
            features_p1[:, best_feat_indices[0]], 
            features_p1[:, best_feat_indices[1]], 
            alpha=0.5, label=f"'{p1}' (n={len(features_p1)})"
        )
        ax.scatter(
            features_p2[:, best_feat_indices[0]], 
            features_p2[:, best_feat_indices[1]], 
            alpha=0.5, label=f"'{p2}' (n={len(features_p2)})"
        )
        
        ax.set_xlabel(f'Feature {best_feat_indices[0]}')
        ax.set_ylabel(f'Feature {best_feat_indices[1]}')
        ax.set_title(f'Best Separable Pair: {p1} vs {p2}\n({best_pair[2]:.1f}% features significantly different)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    return pair_results

pair_results = diagnose_feature_quality(pipeline, n_phonemes_to_check=15)

def train_and_evaluate_extended(pipeline, use_groups=False, method='markov'):
    """
    Train per patient with multiple approaches.
    
    Args:
        pipeline: Dutch30Pipeline with loaded data
        use_groups: Whether to use phoneme groups
        method: One of 'markov', 'gmm', 'soft_labels', 'gmm_informed'
    """
    import numpy as np
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import accuracy_score
    from scipy.special import softmax
    from markov_phoneme_model import MarkovPhonemeModel
    
    results = {}
    
    for pid in sorted(set(pipeline.train['phoneme_participant_ids'])):
        train_mask = [p == pid for p in pipeline.train['phoneme_participant_ids']]
        test_mask = [p == pid for p in pipeline.test['phoneme_participant_ids']]
        
        train_feat = [pipeline.train['features'][i] for i, m in enumerate(train_mask) if m]
        train_labels = [pipeline.train['phoneme_labels'][i] for i, m in enumerate(train_mask) if m]
        test_feat = [pipeline.test['features'][i] for i, m in enumerate(test_mask) if m]
        test_labels = [pipeline.test['phoneme_labels'][i] for i, m in enumerate(test_mask) if m]
        
        if len(train_feat) < 10 or len(test_feat) < 5:
            continue
        
        if method == 'markov':
            # MarkovPhonemeModel handles its own pooling, scaling, and NaN filtering
            # internally via _pool_features() and _build_neural_classifier(),
            # so we only need to filter out NaN/Inf samples at the raw level.
            valid_train = [i for i, f in enumerate(train_feat)
                          if not (np.any(np.isnan(f)) or np.any(np.isinf(f)))]
            valid_test = [i for i, f in enumerate(test_feat)
                         if not (np.any(np.isnan(f)) or np.any(np.isinf(f)))]
            
            train_feat_valid = [train_feat[i] for i in valid_train]
            y_train = [train_labels[i] for i in valid_train]
            test_feat_valid = [test_feat[i] for i in valid_test]
            y_test = [test_labels[i] for i in valid_test]
            
            if len(train_feat_valid) < 10 or len(test_feat_valid) < 5:
                continue
            
            model = MarkovPhonemeModel(
                phonetic_dict=pipeline.detector.phonetic_dict,
                order=1,
                use_groups=use_groups
            )
            model.train(features=train_feat_valid, phoneme_labels=y_train)
            
            preds, _ = model.predict(test_feat_valid, use_viterbi=True)
            accuracy = sum(1 for p, t in zip(preds, y_test) if p == t) / len(y_test)
            n_classes = len(set(y_train))
            n_train = len(train_feat_valid)
            n_test = len(test_feat_valid)
            
        else:
            # gmm, soft_labels, gmm_informed all work on mean-pooled, scaled 2D arrays
            X_train = []
            for feat in train_feat:
                if feat.ndim > 1:
                    X_train.append(np.mean(feat, axis=0))
                else:
                    X_train.append(feat)
            X_train = np.array(X_train)
            
            X_test = []
            for feat in test_feat:
                if feat.ndim > 1:
                    X_test.append(np.mean(feat, axis=0))
                else:
                    X_test.append(feat)
            X_test = np.array(X_test)
            
            valid_train = ~(np.isnan(X_train).any(axis=1) | np.isinf(X_train).any(axis=1))
            valid_test = ~(np.isnan(X_test).any(axis=1) | np.isinf(X_test).any(axis=1))
            
            X_train = X_train[valid_train]
            y_train = [train_labels[i] for i in range(len(train_labels)) if valid_train[i]]
            X_test = X_test[valid_test]
            y_test = [test_labels[i] for i in range(len(test_labels)) if valid_test[i]]
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            le = LabelEncoder()
            y_train_encoded = le.fit_transform(y_train)
            n_classes = len(le.classes_)
            
            if method == 'gmm':
                gmm_per_class = {}
                for class_idx in range(n_classes):
                    class_mask = y_train_encoded == class_idx
                    if np.sum(class_mask) < 2:
                        continue
                    
                    X_class = X_train_scaled[class_mask]
                    n_components = min(3, len(X_class) // 2)
                    if n_components < 1:
                        n_components = 1
                    
                    gmm = GaussianMixture(
                        n_components=n_components,
                        covariance_type='diag',
                        max_iter=100,
                        random_state=42
                    )
                    gmm.fit(X_class)
                    gmm_per_class[class_idx] = gmm
                
                preds_encoded = []
                for x in X_test_scaled:
                    scores = []
                    for class_idx in range(n_classes):
                        if class_idx in gmm_per_class:
                            score = gmm_per_class[class_idx].score_samples(x.reshape(1, -1))[0]
                        else:
                            score = -np.inf
                        scores.append(score)
                    preds_encoded.append(np.argmax(scores))
                
                preds = le.inverse_transform(preds_encoded)
                accuracy = accuracy_score(y_test, preds)
                
            elif method == 'soft_labels':
                from sklearn.ensemble import RandomForestClassifier
                
                soft_targets = np.zeros((len(y_train_encoded), n_classes))
                smoothing = 0.1
                
                for i, label in enumerate(y_train_encoded):
                    soft_targets[i, :] = smoothing / n_classes
                    soft_targets[i, label] = 1.0 - smoothing + smoothing / n_classes
                
                rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
                rf.fit(X_train_scaled, y_train_encoded)
                
                preds_encoded = rf.predict(X_test_scaled)
                preds = le.inverse_transform(preds_encoded)
                accuracy = accuracy_score(y_test, preds)
                
            elif method == 'gmm_informed':
                n_components_total = min(n_classes * 2, 20)
                
                gmm_unsupervised = GaussianMixture(
                    n_components=n_components_total,
                    covariance_type='diag',
                    max_iter=100,
                    random_state=42
                )
                gmm_unsupervised.fit(X_train_scaled)
                
                cluster_probs = gmm_unsupervised.predict_proba(X_train_scaled)
                
                X_augmented_train = np.hstack([X_train_scaled, cluster_probs])
                X_augmented_test = np.hstack([
                    X_test_scaled, 
                    gmm_unsupervised.predict_proba(X_test_scaled)
                ])
                
                from sklearn.ensemble import RandomForestClassifier
                rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
                rf.fit(X_augmented_train, y_train_encoded)
                
                preds_encoded = rf.predict(X_augmented_test)
                preds = le.inverse_transform(preds_encoded)
                accuracy = accuracy_score(y_test, preds)
            
            n_train = len(X_train)
            n_test = len(X_test)
        
        results[pid] = {
            'accuracy': accuracy,
            'train_size': n_train,
            'test_size': n_test,
            'n_classes': n_classes,
            'predictions': preds,
            'true_labels': y_test
        }
        
        print(f"  {pid}: Acc={accuracy:.3f} (method={method}, classes={n_classes}, train={n_train})")
    
    accs = [r['accuracy'] for r in results.values()]
    print(f"\n  Mean: {np.mean(accs):.3f} +/- {np.std(accs):.3f}")
    
    return results


def compare_methods(pipeline, use_groups=False):
    """Compare all methods side by side."""
    import pandas as pd
    
    methods = ['markov', 'gmm', 'soft_labels', 'gmm_informed']
    all_results = {}
    
    for method in methods:
        print(f"\n{'='*60}")
        print(f"METHOD: {method}")
        print('='*60)
        all_results[method] = train_and_evaluate_extended(pipeline, use_groups, method)
    
    comparison = {}
    for method, results in all_results.items():
        for pid, res in results.items():
            if pid not in comparison:
                comparison[pid] = {}
            comparison[pid][method] = res['accuracy']
    
    df = pd.DataFrame(comparison).T
    df['Max'] = df.max(axis=1)
    df['Best'] = df.idxmax(axis=1)
    
    print("\n" + "="*80)
    print("COMPARISON TABLE")
    print("="*80)
    print(df.to_string())
    
    print("\n" + "="*80)
    print("MEAN ACCURACY BY METHOD")
    print("="*80)
    for method in methods:
        accs = [comparison[pid].get(method, np.nan) for pid in comparison]
        print(f"  {method:15s}: {np.nanmean(accs):.3f} +/- {np.nanstd(accs):.3f}")
    
    return all_results, df


# all_results, comparison_df = compare_methods(pipeline, use_groups=True)
