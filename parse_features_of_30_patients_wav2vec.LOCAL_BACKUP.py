# Converted from parse_features_of_30_patients_wav2vec.LOCAL_BACKUP.ipynb

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


import os
import gc
import glob
import json
#import h5py
import numpy as np
import pickle
import pandas as pd
#from IPython.display import Audio, display
from collections import Counter, defaultdict
from pynwb import NWBHDF5IO
from datetime import datetime
import scipy.signal
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, silhouette_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cosine, euclidean
from scipy.signal import decimate

from extract_features import extractHG, stackFeatures, downsampleLabels
from brain_audio_decoder import BrainAudioDecoder
from custom_decoder import CustomBrainAudioDecoder
from brain_audio_decoder_viz import BrainAudioDecoderViz
from acoustic_change_detector import AcousticChangeDetector
from phoneme_validator import PhonemeValidator
from phonetic_dictionary import PhoneticDictionary
#from feature_vizualizer import PhonemeFeatureVisualizer
from markov_phoneme_model import MarkovPhonemeModel
from extract_features import extractHG, downsampleLabels, extractMelSpecs
# from pipeline import UnifiedPhonemePipeline
from config import BIDS_PATH, OUTPUT_PATH, RESULTS_PATH, DUTCH_30_PATH, DUTCH_10_PATH, get_dataset_paths
from dutch_30_pipeline import Dutch30Pipeline
from dutch_30_feature_extractor import Dutch30FeatureExtractor
from phoneme_detection_diagnostic import Dutch30PhonemeDetectionDiagnostic 
from dataset_config import Dutch30Config

from transformers import Wav2Vec2Model, Wav2Vec2Processor, Wav2Vec2FeatureExtractor
import torch

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-large-xlsr-53")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-xlsr-53")
print("Downloaded successfully, hidden size:", model.config.hidden_size)

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

    def run_from_config(pipeline, run_config):
        """Run experiment using a unified config dict.
    
        Passes all parameters to run_experiment and includes
        the full pipeline config in the logged params.
    
        Args:
            pipeline: Pipeline with train/test data.
            run_config: dict with all pipeline and classifier settings.
    
        Returns:
            Tuple of (name, params, results).
        """
        name, params, results = run_experiment(
            pipeline,
            order=run_config.get('markov_order', 1),
            class_weight=run_config.get('class_weight', 'balanced'),
            use_groups=run_config.get('use_groups', False),
            classifier_type=run_config.get('classifier_type', 'random_forest'),
            use_viterbi=run_config.get('use_viterbi', False),
            stacking_order=run_config.get('stacking_order'),
            stacking_step_size=run_config.get('stacking_step_size'),
            max_frames=run_config.get('max_frames'),
            min_frames=run_config.get('min_frames'),
            target_frames=run_config.get('target_frames'),
            random_state=run_config.get('random_state', 37),
            scaler_type=run_config.get('scaler_type', 'standard'),
            subtract_baseline=run_config.get('subtract_baseline', False),
            min_class_samples = run_config.get('min_class_samples', 5)
        )
    
        # Add pipeline config to params so it's all in one place
        params['patient_range'] = run_config.get('patient_range')
        params['feature_extraction_method'] = run_config.get('feature_extraction_method')
        params['subtract_baseline'] = run_config.get('subtract_baseline')
        params['sample_fraction'] = run_config.get('sample_fraction')
    
        return name, params, results

# Experiment logger
import json
import os
from datetime import datetime
import numpy as np


PATIENT_GROUPS = {
    'P01-P10': ['P01', 'P02', 'P03', 'P04', 'P06', 'P07', 'P08', 'P09', 'P10'],
    'P11-P20': ['P11', 'P12', 'P13', 'P14', 'P15', 'P16', 'P17', 'P20'],
    'P21-P30': ['P21', 'P22', 'P23', 'P24', 'P25', 'P26', 'P27', 'P28', 'P29', 'P30'],
}


def run_experiment(pipeline, order=3, class_weight='balanced', use_groups=False,
                   classifier_type='random_forest', use_viterbi=False, random_state=37,
                   stacking_order=None, stacking_step_size=None, scaler_type='standard', subtract_baseline=False,
                   max_frames=None, min_frames=None, target_frames=None, min_class_samples=5):
    """Run a single experiment with given parameters.

    Args:
        pipeline: Pipeline with train/test data.
        order: Markov chain order.
        class_weight: 'balanced', 'balanced_subsample', or None.
        use_groups: Whether to use phoneme groups.
        classifier_type: 'random_forest', 'extra_trees', etc.
        use_viterbi: Whether to use Viterbi decoding.
        random_state: Random seed.
        stacking_order: Temporal stacking order.
        stacking_step_size: Temporal stacking step size.
        scaler_type: Feature scaler type.
        subtract_baseline: Whether to subtract baseline.
        max_frames: Maximum frames per phoneme.
        min_frames: Minimum frames per phoneme.
        target_frames: Target frames for resampling.
        min_class_samples: Minimum training samples per class.

    Returns:
        Tuple of (name, params, results).
    """
    from markov_phoneme_model import MarkovPhonemeModel
    from collections import Counter

    weight_str = str(class_weight) if class_weight else 'none'
    name = f"{classifier_type}_o{order}_w{weight_str}"
    if use_viterbi:
        name += "_viterbi"
    if stacking_order is not None:
        name += f"_stack{stacking_order}x{stacking_step_size}"
    if target_frames is not None:
        name += f"_resamp{target_frames}"
    if max_frames is not None:
        name += f"_max{max_frames}"
    if min_frames is not None:
        name += f"_min{min_frames}"
    if scaler_type != 'standard':
        name += f"_{scaler_type}"
    if subtract_baseline:
        name += "_bsub"

    params = {
        'order': order,
        'class_weight': str(class_weight),
        'use_groups': use_groups,
        'classifier_type': classifier_type,
        'use_viterbi': use_viterbi,
        'stacking_order': stacking_order,
        'stacking_step_size': stacking_step_size,
        'max_frames': max_frames,
        'target_frames': target_frames,
        'random_state': random_state,
        'scaler_type': scaler_type,
        'subtract_baseline': subtract_baseline,
        'min_class_samples': min_class_samples,
    }

    print(f"\nRunning: {name}")
    print(f"  Params: {params}")

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

        # filter rare classes
        train_counts = Counter(train_labels)
        valid_classes = {cls for cls, cnt in train_counts.items()
                         if cnt >= min_class_samples}
        train_feat = [f for f, l in zip(train_feat, train_labels)
                      if l in valid_classes]
        train_labels = [l for l in train_labels if l in valid_classes]
        test_feat = [f for f, l in zip(test_feat, test_labels)
                     if l in valid_classes]
        test_labels = [l for l in test_labels if l in valid_classes]

        if len(train_feat) < 10 or len(test_feat) < 5:
            continue

        model = MarkovPhonemeModel(
            phonetic_dict=pipeline.detector.phonetic_dict,
            order=order,
            use_groups=use_groups,
            class_weight=class_weight,
            classifier_type=classifier_type,
            random_state=random_state,
            scaler_type=scaler_type,
            feature_pooling_method='flatten',
        )
        model.train(features=train_feat, phoneme_labels=train_labels)

        # predict without viterbi to store as baseline
        preds_no_viterbi, _ = model.predict(test_feat, use_viterbi=False)
        preds_no_viterbi = [str(p) for p in preds_no_viterbi]

        # predict with viterbi if requested
        if use_viterbi:
            preds, _ = model.predict(test_feat, use_viterbi=True)
        else:
            preds = preds_no_viterbi

        preds = [str(p) for p in preds]

        correct = sum(1 for p, t in zip(preds, test_labels) if p == t)
        accuracy = correct / len(test_labels)

        test_classes = set(test_labels)
        pred_classes = set(preds)
        n_classes_test = len(test_classes)
        n_classes_predicted = len(pred_classes & test_classes)
        pct_classes = n_classes_predicted / n_classes_test if n_classes_test > 0 else 0
        adjusted_acc = accuracy * pct_classes

        results[pid] = {
            'accuracy': accuracy,
            'adjusted_accuracy': adjusted_acc,
            'pct_classes_predicted': pct_classes,
            'n_classes_predicted': n_classes_predicted,
            'n_classes_test': n_classes_test,
            'train_size': len(train_feat),
            'test_size': len(test_feat),
            'n_classes': len(set(train_labels)),
            'predictions': preds,
            'predictions_no_viterbi': preds_no_viterbi,
            'true_labels': test_labels,
            'model': model,
        }

    return name, params, results

class ExperimentLogger:
    """
    Simple experiment logger that stores a flat CSV-like table.

    Each row = one experiment.
    Columns = experiment name, params, per-patient accuracies, group means.
    """

    def __init__(self, log_file='experiments.json'):
        """
        Initialize the experiment logger.

        Args:
            log_file: Path to JSON file for storing experiments.
        """
        self.log_file = log_file
        self.experiments = []

        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                self.experiments = json.load(f)
            print(f"Loaded {len(self.experiments)} previous experiments from {log_file}")

    def log(self, name, params, results):
        """
        Log an experiment from run_experiment output.

        Args:
            name: Experiment name (auto-generated by run_experiment).
            params: Dict of parameters.
            results: Dict of per-patient results from run_experiment.
        """
        # Extract only accuracies per patient (no model objects)
        
        # Collect per-patient metrics (keep in memory, don't save)
        patient_metrics = {}
        for pid, pr in results.items():
            patient_metrics[pid] = {
                'accuracy': round(pr['accuracy'], 4),
                'adjusted_accuracy': round(pr.get('adjusted_accuracy', pr['accuracy']), 4),
                'pct_classes': round(pr.get('pct_classes_predicted', 1.0), 4),
                'n_classes_predicted': pr.get('n_classes_predicted', 0),
                'n_classes_test': pr.get('n_classes_test', 0),
                'train_size': pr.get('train_size', 0),
                'test_size': pr.get('test_size', 0),
            }

        # Compute group summaries with best/worst detail
        group_summaries = {}
        for group_name, patients in PATIENT_GROUPS.items():
            group_pats = {pid: patient_metrics[pid] for pid in patients
                         if pid in patient_metrics}
            if not group_pats:
                group_summaries[group_name] = None
                continue

            accs = [v['accuracy'] for v in group_pats.values()]
            adj_accs = [v['adjusted_accuracy'] for v in group_pats.values()]
            pcts = [v['pct_classes'] for v in group_pats.values()]

            best_pid = max(group_pats.keys(),
                           key=lambda p: group_pats[p]['adjusted_accuracy'])
            worst_pid = min(group_pats.keys(),
                            key=lambda p: group_pats[p]['adjusted_accuracy'])

            group_summaries[group_name] = {
                'mean_acc': round(np.mean(accs), 4),
                'mean_adj_acc': round(np.mean(adj_accs), 4),
                'mean_pct_classes': round(np.mean(pcts), 4),
                'n_patients': len(group_pats),
                'best': best_pid,
                'best_acc': group_pats[best_pid]['accuracy'],
                'best_adj_acc': group_pats[best_pid]['adjusted_accuracy'],
                'best_pct_classes': group_pats[best_pid]['pct_classes'],
                'best_train_size': group_pats[best_pid]['train_size'],
                'best_test_size': group_pats[best_pid]['test_size'],
                'worst': worst_pid,
                'worst_acc': group_pats[worst_pid]['accuracy'],
                'worst_adj_acc': group_pats[worst_pid]['adjusted_accuracy'],
                'worst_pct_classes': group_pats[worst_pid]['pct_classes'],
                'worst_train_size': group_pats[worst_pid]['train_size'],
                'worst_test_size': group_pats[worst_pid]['test_size'],
            }

        all_adj = [v['adjusted_accuracy'] for v in patient_metrics.values()]
        all_acc = [v['accuracy'] for v in patient_metrics.values()]
        all_pct = [v['pct_classes'] for v in patient_metrics.values()]
        overall_acc = round(np.mean(all_acc), 4) if all_acc else 0
        overall_adj = round(np.mean(all_adj), 4) if all_adj else 0
        overall_pct = round(np.mean(all_pct), 4) if all_pct else 0

        entry = {
            'name': name,
            'timestamp': datetime.now().isoformat()[:19],
            'params': params,
            'group_summaries': group_summaries,
            'overall_acc': overall_acc,
            'overall_adj_acc': overall_adj,
            'overall_pct_classes': overall_pct,
        }
        self.experiments.append(entry)

        with open(self.log_file, 'w') as f:
            json.dump(self.experiments, f, indent=2)

        print(f"  Saved: {name} (acc={overall_acc}, "
              f"adj_acc={overall_adj}, cls%={overall_pct})")

    def print_table(self, last_n=None):
        """Print compact comparison table with group averages,
        best/worst patients, and adjusted accuracy.

        Args:
            last_n: Show last N experiments (None = all).
        """
        experiments = self.experiments[-last_n:] if last_n else self.experiments

        if not experiments:
            print("No experiments.")
            return

        exp_names = [exp['name'][:45] for exp in experiments]
        col_w = max(12, max(len(n) for n in exp_names) + 2)

        # Header
        print("\n" + "=" * (30 + col_w * len(experiments)))
        row = f"{'Metric':<30}"
        for name in exp_names:
            row += f"{name:<{col_w}}"
        print(row)
        print("=" * (30 + col_w * len(experiments)))

        for group_name in PATIENT_GROUPS.keys():
            print(f"\n  {group_name}")
            print(f"  {'-' * (28 + col_w * len(experiments))}")

            # Group mean accuracy
            row = f"    {'Mean Acc%':<26}"
            for exp in experiments:
                gs = exp.get('group_summaries', {}).get(group_name)
                if gs:
                    row += f"{gs['mean_acc']*100:>{col_w-2}.1f}%  "
                else:
                    row += f"{'N/A':<{col_w}}"
            print(row)

            # Group mean class %
            row = f"    {'Mean Cls%':<26}"
            for exp in experiments:
                gs = exp.get('group_summaries', {}).get(group_name)
                if gs:
                    row += f"{gs['mean_pct_classes']*100:>{col_w-2}.1f}%  "
                else:
                    row += f"{'N/A':<{col_w}}"
            print(row)

            # Group mean adjusted accuracy
            row = f"    {'Mean AdjAcc%':<26}"
            for exp in experiments:
                gs = exp.get('group_summaries', {}).get(group_name)
                if gs:
                    row += f"{gs['mean_adj_acc']*100:>{col_w-2}.1f}%  "
                else:
                    row += f"{'N/A':<{col_w}}"
            print(row)

            # Best patient
            row = f"    {'Best':<26}"
            for exp in experiments:
                gs = exp.get('group_summaries', {}).get(group_name)
                if gs:
                    txt = f"{gs['best']} {gs['best_adj_acc']*100:.1f}%"
                    row += f"{txt:<{col_w}}"
                else:
                    row += f"{'N/A':<{col_w}}"
            print(row)

            # Worst patient
            row = f"    {'Worst':<26}"
            for exp in experiments:
                gs = exp.get('group_summaries', {}).get(group_name)
                if gs:
                    txt = f"{gs['worst']} {gs['worst_adj_acc']*100:.1f}%"
                    row += f"{txt:<{col_w}}"
                else:
                    row += f"{'N/A':<{col_w}}"
            print(row)

        # Overall
        print(f"\n{'=' * (30 + col_w * len(experiments))}")

        row = f"  {'OVERALL Acc%':<28}"
        for exp in experiments:
            val = exp.get('overall_acc', exp.get('overall', 0))
            row += f"{val*100:>{col_w-2}.1f}%  "
        print(row)

        row = f"  {'OVERALL Cls%':<28}"
        for exp in experiments:
            val = exp.get('overall_pct_classes', 1.0)
            row += f"{val*100:>{col_w-2}.1f}%  "
        print(row)

        row = f"  {'OVERALL AdjAcc%':<28}"
        for exp in experiments:
            val = exp.get('overall_adj_acc', exp.get('overall', 0))
            row += f"{val*100:>{col_w-2}.1f}%  "
        print(row)

        print("=" * (30 + col_w * len(experiments)))

    def clear(self):
        """Clear all experiments."""
        confirm = input(f"Delete all {len(self.experiments)} experiments? (yes/no): ")
        if confirm.lower() == 'yes':
            self.experiments = []
            if os.path.exists(self.log_file):
                os.remove(self.log_file)
            print("Cleared.")

    def remove_last(self, n=1):
        """
        Remove last N experiments.

        Args:
            n: Number of experiments to remove from the end.
        """
        removed = self.experiments[-n:]
        self.experiments = self.experiments[:-n]
        with open(self.log_file, 'w') as f:
            json.dump(self.experiments, f, indent=2)
        for exp in removed:
            print(f"  Removed: {exp['name']}")

    def best_experiment(self, metric='overall_adj_acc', group=None):
        """Find the best experiment and display its full results.

        Args:
            metric: str, which metric to rank by. Options:
                'overall_acc', 'overall_adj_acc', 'overall_pct_classes'.
                If group is specified, uses the group's version.
            group: str or None, e.g. 'P21-P30'. If None, uses
                overall metrics.

        Returns:
            dict, the best experiment entry.
        """
        if not self.experiments:
            print("No experiments logged.")
            return None

        # Build ranking
        ranked = []
        for i, exp in enumerate(self.experiments):
            if group is not None:
                gs = exp.get('group_summaries', {}).get(group)
                if gs is None:
                    continue
                # Map metric name to group-level key
                metric_map = {
                    'overall_acc': 'mean_acc',
                    'overall_adj_acc': 'mean_adj_acc',
                    'overall_pct_classes': 'mean_pct_classes',
                }
                key = metric_map.get(metric, metric)
                val = gs.get(key, 0)
            else:
                val = exp.get(metric, 0)

            ranked.append((val, i, exp))

        if not ranked:
            print(f"No experiments found for group={group}")
            return None

        ranked.sort(key=lambda x: x[0], reverse=True)

        # Print full ranking
        scope = group if group else "OVERALL"
        print(f"\n{'=' * 70}")
        print(f"EXPERIMENT RANKING by {metric} ({scope})")
        print(f"{'=' * 70}")
        print(f"{'Rank':<6}{'Name':<40}{'Score':>10}")
        print(f"{'-' * 70}")

        for rank, (val, idx, exp) in enumerate(ranked, 1):
            marker = " <-- BEST" if rank == 1 else ""
            print(f"{rank:<6}{exp['name']:<40}{val*100:>9.2f}%{marker}")

        # Show full detail for the best
        best_val, best_idx, best_exp = ranked[0]

        print(f"\n{'=' * 70}")
        print(f"BEST EXPERIMENT: {best_exp['name']}")
        print(f"{'=' * 70}")

        print(f"\n  Parameters:")
        for k, v in best_exp.get('params', {}).items():
            if v is not None:
                print(f"    {k}: {v}")

        print(f"\n  Overall:")
        print(f"    Accuracy:          "
              f"{best_exp.get('overall_acc', 0)*100:.2f}%")
        print(f"    Adjusted Accuracy: "
              f"{best_exp.get('overall_adj_acc', 0)*100:.2f}%")
        print(f"    Classes Predicted:  "
              f"{best_exp.get('overall_pct_classes', 0)*100:.1f}%")

        print(f"\n  Per Group:")
        for group_name, gs in best_exp.get('group_summaries', {}).items():
            if gs is None:
                continue
            print(f"\n    {group_name} ({gs['n_patients']} patients):")
            print(f"      Mean Acc:     {gs['mean_acc']*100:.2f}%")
            print(f"      Mean AdjAcc:  {gs['mean_adj_acc']*100:.2f}%")
            print(f"      Mean Cls%:    {gs['mean_pct_classes']*100:.1f}%")
            print(f"      Best:  {gs['best']:<6} "
                  f"acc={gs['best_acc']*100:.2f}%  "
                  f"adj={gs['best_adj_acc']*100:.2f}%  "
                  f"cls={gs['best_pct_classes']*100:.1f}%  "
                  f"(train={gs['best_train_size']}, "
                  f"test={gs['best_test_size']})")
            print(f"      Worst: {gs['worst']:<6} "
                  f"acc={gs['worst_acc']*100:.2f}%  "
                  f"adj={gs['worst_adj_acc']*100:.2f}%  "
                  f"cls={gs['worst_pct_classes']*100:.1f}%  "
                  f"(train={gs['worst_train_size']}, "
                  f"test={gs['worst_test_size']})")

        print(f"\n  Timestamp: {best_exp.get('timestamp', 'unknown')}")
        print(f"{'=' * 70}")

        return best_exp

import copy
import pickle
import numpy as np

run_config = {
    # Patient selection
    'patient_range': (21, 30),
    'sample_fraction': 1,
    # Pipeline settings
    'feature_extraction_method': 'high_gamma',
    'use_wav2vec': True,
    'subtract_baseline': False,
    # Step 5a: frame filtering
    'min_frames': 4,
    'max_frames': 300,
    # Step 5b: choose ONE approach
    'stacking_order': 5,
    'stacking_step_size': 1,
    'target_frames': None,
    # Classifier
    'classifier_type': 'logistic_regression',
    'class_weight': 'balanced',
    'markov_order': 1,
    'use_viterbi': True,
    'random_state': 37,
    'scaler_type': 'standard',
    'feature_pooling_method': 'flatten',
    'min_class_samples': 0,
    # Unknown filtering
    'unknown_keep_ratio': 0.0025,
}

# ---- Pipeline setup ----
extractor = Dutch30FeatureExtractor()
pipeline = Dutch30Pipeline(
    dutch30_extractor=extractor,
    debug_mode=False,
    feature_extraction_method=run_config['feature_extraction_method'],
    use_wav2vec=run_config['use_wav2vec'],
    subtract_baseline=run_config['subtract_baseline'],
    use_rms_boundaries=False,
    use_multifeature=False,
)

# pipeline.detector.config.wav2vec_smoothing_filter = 'savgol'
# pipeline.detector.config.wav2vec_savgol_window = 3
# pipeline.detector.config.wav2vec_savgol_polyorder = 2
pipeline.detector.config.wav2vec_smoothing_filter = 'median'
pipeline.detector.config.wav2vec_median_size = 5


sf = run_config['sample_fraction']
pr = run_config['patient_range']

STEP3_CHECKPOINT = f'checkpoint_after_step3_P{pr[0]:02d}-P{pr[1]:02d}.pkl'

use_stacking = run_config['stacking_order'] is not None
use_resampling = run_config['target_frames'] is not None

if use_stacking:
    approach = (f"stacking order={run_config['stacking_order']} "
                f"step={run_config['stacking_step_size']}")
elif use_resampling:
    approach = f"resampling target={run_config['target_frames']}"
else:
    approach = "raw (no step 5b)"

print(f"Approach: {approach}")
print(f"Patients: P{pr[0]:02d}-P{pr[1]:02d}")

# ---- Helpers ----
def count(pipeline, label=""):
    tr = len(pipeline.train['features']) if pipeline.train else 0
    te = len(pipeline.test['features']) if pipeline.test else 0
    print(f"  {label:.<40s} train={tr:>6d}  test={te:>6d}")

def run_step5b(pipeline, run_config):
    if run_config['stacking_order'] is not None:
        pipeline.step5a_filter_by_frame_count(
            min_frames=run_config['min_frames'],
            max_frames=run_config['max_frames'])
        count(pipeline, "After 5a (frame filter)")

        print(f"Feature shape before step5b: {pipeline.train['features'][0].shape}")
        print(f"Feature ndim: {pipeline.train['features'][0].ndim}")

        pipeline.step5b_stack_features(
            model_order=run_config['stacking_order'],
            step_size=run_config['stacking_step_size'])
        count(pipeline, "After 5b (stacking)")
        
        # if hasattr(pipeline, 'step5c_collapse_to_phoneme_level'):
        #     pipeline.step5c_collapse_to_phoneme_level()
        #     count(pipeline, "After 5c (collapse)")
        # else:
        #     print("WARNING: step5c_collapse_to_phoneme_level NOT FOUND")

    elif run_config['target_frames'] is not None:
        pipeline.step5b_normalize_feature_lengths(
            target_frames=run_config['target_frames'])
        count(pipeline, "After 5b (resample)")
    else:
        print("WARNING: No step 5b configured")

# ---- Try step 5 checkpoint first ----
step5b_method = 'stack' if use_stacking else ('normalize' if use_resampling else None)
step5_loaded = pipeline.try_load_checkpoint(
    stage='after_step5',
    step5b_method=step5b_method,
    model_order=run_config.get('stacking_order'),
    step_size=run_config.get('stacking_step_size'),
    target_frames=run_config.get('target_frames'),
)

if step5_loaded:
    print("Step 5 checkpoint found, skipping steps 1-5b")
    cached_train = copy.deepcopy(pipeline.train)
    cached_test = copy.deepcopy(pipeline.test)
else:
    # ---- Fall back to step 3 checkpoint or full run ----
    if os.path.exists(STEP3_CHECKPOINT):
        print(f"Loading step 3 checkpoint: {STEP3_CHECKPOINT}")
        with open(STEP3_CHECKPOINT, 'rb') as f:
            state = pickle.load(f)
        pipeline.split_result = state['split_result']
        pipeline.patient_data = state['patient_data']
        pipeline.patient_baselines = state['patient_baselines']
        print("Step 3 checkpoint loaded")
    else:
        print("No checkpoint found. Running steps 1-3...")
        pipeline.step1_load_dutch30_data(patient_range=pr)
        pipeline.step2_split_by_instances()
        pipeline.step3_load_channel_exclusions('channel_exclusions.json')
        pipeline.apply_channel_exclusions()
        pipeline.print_channel_counts()

        with open(STEP3_CHECKPOINT, 'wb') as f:
            pickle.dump({
                'split_result': pipeline.split_result,
                'patient_data': pipeline.patient_data,
                'patient_baselines': getattr(pipeline, 'patient_baselines', None),
            }, f)
        print(f"Step 3 checkpoint saved: {STEP3_CHECKPOINT}")
    
    # ---- Steps 4-5 ----
    pipeline.step4_custom_detector()
    pipeline.step5_accumulate_data_dutch30()
    count(pipeline, "After step 5 (accumulate)")

    cached_train = copy.deepcopy(pipeline.train)
    cached_test = copy.deepcopy(pipeline.test)
    print("Cached post-step5 state in memory")

    run_step5b(pipeline, run_config)
    print(f"'phoneme_instance_ids' in train: {'phoneme_instance_ids' in pipeline.train}")
    print(f"'phoneme_instance_ids' in test: {'phoneme_instance_ids' in pipeline.test}")
    print(f"Train keys: {list(pipeline.train.keys())}")
    print(f"Test keys: {list(pipeline.test.keys())}")
    
    pipeline.checkpoint_after_step5(
        sample_fraction=run_config['sample_fraction'] if run_config['sample_fraction'] != 1 else None,
        step5b_method=step5b_method,
        model_order=run_config.get('stacking_order'),
        step_size=run_config.get('stacking_step_size'),
        target_frames=run_config.get('target_frames'),
    )
# ---- Steps 4-5 ----

pipeline.dutch30_step6_resolve_unknowns()
count(pipeline, "After step 6 (resolve unknowns)")
print(f"After step6 - instance_ids in test: {'phoneme_instance_ids' in pipeline.test}")

pipeline.step7_filter_unknowns(
    unknown_keep_ratio=run_config['unknown_keep_ratio'])
count(pipeline, "After step 7 (filter unknowns)")
print(f"After step7 - instance_ids in test: {'phoneme_instance_ids' in pipeline.test}")

# # Check per-patient counts on this machine
# for pid in sorted(set(pipeline.train['phoneme_participant_ids'])):
#     train_count = sum(1 for p in pipeline.train['phoneme_participant_ids'] if p == pid)
#     test_count = sum(1 for p in pipeline.test['phoneme_participant_ids'] if p == pid)
#     print(f"  {pid}: train={train_count}, test={test_count}, total={train_count + test_count}")

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

def analyze_consecutive_correct(pipeline, pid, min_length=2):
    """Find word instances where consecutive phonemes were predicted correctly.

        Only counts consecutive correct predictions within a single word
    instance, not across word boundaries.

    Args:
        pipeline: pipeline object with test data and patient_results.
        pid: str, patient id to analyze.
        min_length: int, minimum number of consecutive correct phonemes
            to report.
    """
    from collections import Counter, OrderedDict

    if pid not in pipeline.patient_results:
        print(f"{pid}: no results found")
        return

    test_mask = [p == pid for p in pipeline.test['phoneme_participant_ids']]
    test_indices = [i for i, m in enumerate(test_mask) if m]

    preds_frame = pipeline.patient_results[pid]['predictions']
    true_frame = pipeline.patient_results[pid]['true_labels']
    words_frame = [pipeline.test['phoneme_words'][i] for i in test_indices]
    positions_frame = [pipeline.test['phoneme_positions'][i] for i in test_indices]
    instance_ids_frame = [pipeline.test['phoneme_instance_ids'][i] for i in test_indices]

    # collapse frames to phoneme instances
    phoneme_data = OrderedDict()
    for inst_id, pos, pred, true, word in zip(
        instance_ids_frame, positions_frame,
        preds_frame, true_frame, words_frame
    ):
        key = (inst_id, pos)
        if key not in phoneme_data:
            phoneme_data[key] = {
                'preds': [], 'trues': [], 'word': word,
                'pos': pos, 'inst_id': inst_id
            }
        phoneme_data[key]['preds'].append(pred)
        phoneme_data[key]['trues'].append(true)

    # resolve to single label per phoneme
    phonemes = []
    for key, d in phoneme_data.items():
        true_label = Counter(d['trues']).most_common(1)[0][0]
        pred_label = Counter(d['preds']).most_common(1)[0][0]
        phonemes.append({
            'true': true_label,
            'pred': pred_label,
            'word': d['word'],
            'pos': d['pos'],
            'inst_id': d['inst_id'],
            'correct': pred_label == true_label,
        })

    # group by word instance
    instances = OrderedDict()
    for p in phonemes:
        if p['inst_id'] not in instances:
            instances[p['inst_id']] = []
        instances[p['inst_id']].append(p)

    # sort each instance by position
    for inst_id in instances:
        instances[inst_id].sort(key=lambda p: p['pos'])

    # find consecutive correct runs within each instance
    print(f"\n{pid} --- consecutive correct phonemes within word instances ---")
    found = []

    for inst_id, inst_phonemes in instances.items():
        word = inst_phonemes[0]['word']
        n_phonemes = len(inst_phonemes)
        correct_flags = [p['correct'] for p in inst_phonemes]
        true_seq = [p['true'] for p in inst_phonemes]
        pred_seq = [p['pred'] for p in inst_phonemes]

        # find runs of consecutive correct predictions
        run_start = None
        for i, c in enumerate(correct_flags):
            if c and run_start is None:
                run_start = i
            elif not c and run_start is not None:
                run_length = i - run_start
                if run_length >= min_length:
                    found.append({
                        'length': run_length,
                        'word': word,
                        'word_length': n_phonemes,
                        'start_pos': run_start,
                        'true': true_seq[run_start:i],
                        'pred': pred_seq[run_start:i],
                        'full_true': true_seq,
                        'full_pred': pred_seq,
                    })
                run_start = None
        if run_start is not None:
            run_length = len(correct_flags) - run_start
            if run_length >= min_length:
                found.append({
                    'length': run_length,
                    'word': word,
                    'word_length': n_phonemes,
                    'start_pos': run_start,
                    'true': true_seq[run_start:],
                    'pred': pred_seq[run_start:],
                    'full_true': true_seq,
                    'full_pred': pred_seq,
                })

    if not found:
        print(f"  no instances with {min_length}+ consecutive correct phonemes found")
        return

    found.sort(key=lambda x: x['length'], reverse=True)

    print(f"  found {len(found)} instances with {min_length}+ consecutive correct phonemes")
    print(f"  {'word':<20} {'len':<6} {'start':<8} {'correct seq':<25} {'full true':<30} {'full pred'}")
    for f in found[:20]:
        print(f"  {f['word']:<20} {f['length']:<6} {f['start_pos']:<8} "
              f"{str(f['true']):<25} {str(f['full_true']):<30} {str(f['full_pred'])}")

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

for pid in sorted(pipeline.patient_results.keys()):
    analyze_consecutive_correct(pipeline, pid, min_length=2)
    print()

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

# quick check - reload pipeline word segments if needed
# or just check current pipeline.split_result

for pid in ['P21', 'P22', 'P23']:
    wsd_p = pipeline.split_result['word_segments_dict'][pid]
    n_extracted = len(wsd_p['words_list'])
    unique_sentences = {
        wsd_p['word_sentence_indices'][i]: wsd_p['word_sentence_texts'][i]
        for i in range(len(wsd_p['words_list']))
    }
    n_words_in_sentences = sum(
        len(t.split()) for t in unique_sentences.values() if t.strip()
    )
    pct = n_extracted / n_words_in_sentences * 100
    print(f"{pid}: extracted={n_extracted}  expected={n_words_in_sentences}  pct={pct:.1f}%")

for pid in sorted(pipeline.patient_results.keys()):
    print(f"--- {pid} ---")
    r = pipeline.patient_results[pid]
    print(f"  acc={r['accuracy']:.4f}  n_pred={len(set(r['predictions']))}")
    pipeline.step10_visualize_patient(pid, show_table=False, min_class_samples=run_config.get('min_class_samples', 5))

# # ---- Run experiment ----
# name, params, results = run_from_config(pipeline, run_config)

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import find_peaks
from IPython.display import Audio, display


def diagnose_and_play(pipeline, pid, word, instance_idx=0):
    """Visualize and play boundary detection results for a word instance.

    Shows the distance curve with old and new boundary selections,
    plays the full word audio, then plays each detected segment
    separately so you can hear whether the segmentation makes sense.

    Args:
        pipeline: pipeline object with detector and word_segments_dict.
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
    audio = wsd['audio_segments'][idx]
    audio_sr = pipeline.config.audio_sr

    phonemes_dict = pipeline.phonetic_dict.extract_phonemes(word)
    n_phonemes = len(phonemes_dict)
    n_needed = n_phonemes - 1

    print(f"word: '{word}'  instance: {instance_idx}")
    print(f"dictionary phonemes: {phonemes_dict}  (n={n_phonemes})")
    print(f"boundaries needed: {n_needed}")

    # compute wav2vec distances
    wav2vec_features = pipeline.detector.extract_wav2vec_features(audio, audio_sr)
    distances = pipeline.detector.compute_wav2vec_distances(wav2vec_features)
    distances_smoothed = median_filter(distances, size=3)  # used for old approach
    distances_raw = distances.copy()  # used for new approach

    # speech region detection
    threshold = np.mean(distances_smoothed) + 0.5 * np.std(distances_smoothed)
    above = np.where(distances_smoothed > threshold * 0.3)[0]
    speech_start = above[0] if len(above) > 0 else 0
    speech_end = above[-1] if len(above) > 0 else len(distances_smoothed) - 1

    # offset spike trimming
    region_length = speech_end - speech_start
    trim_start = int(region_length * 0.8)
    distances_full_region = distances_smoothed[speech_start:speech_end]
    last_segment = distances_full_region[trim_start:]
    if len(last_segment) > 0 and np.max(last_segment) > 3 * np.median(distances_full_region):
        spike_pos = trim_start + np.argmax(last_segment)
        speech_end_trimmed = speech_start + spike_pos
    else:
        speech_end_trimmed = speech_end

    distances_speech = distances_smoothed[speech_start:speech_end_trimmed]

    # find candidate peaks
    # old: smoothed distances, strongest N
    peaks_old_candidates, _ = find_peaks(distances_speech, distance=1)
    if len(peaks_old_candidates) >= n_needed:
        strongest = np.argsort(distances_speech[peaks_old_candidates])[-n_needed:]
        peaks_old = np.sort(peaks_old_candidates[strongest])
    else:
        peaks_old = peaks_old_candidates

# new: raw distances, trimmed region, distributed selection
distances_speech_raw = distances_raw[speech_start:speech_end_trimmed]
peaks_new_candidates, _ = find_peaks(distances_speech_raw, distance=1)

    # old selection: strongest N
    if len(peaks_all) >= n_needed:
        strongest = np.argsort(distances_speech[peaks_all])[-n_needed:]
        peaks_old = np.sort(peaks_all[strongest])
    else:
        peaks_old = peaks_all

    # new selection: distributed
    n_frames = len(distances_speech)
    slot_size = n_frames / n_needed if n_needed > 0 else n_frames
    selected = []
    used = set()
    for slot in range(n_needed):
        slot_start = int(slot * slot_size)
        slot_end = int((slot + 1) * slot_size)
        slot_peaks = [p for p in peaks_all
                      if slot_start <= p < slot_end and p not in used]
        if slot_peaks:
            best = max(slot_peaks, key=lambda p: distances_speech[p])
            selected.append(best)
            used.add(best)
    remaining = sorted([p for p in peaks_all if p not in used],
                       key=lambda p: distances_speech[p], reverse=True)
    for p in remaining:
        if len(selected) >= n_needed:
            break
        selected.append(p)
    peaks_new = np.sort(np.array(selected)) if selected else np.array([], dtype=int)

    # convert peaks to audio sample boundaries
    wav2vec_fps = 50
    target_sr = pipeline.config.audio_target_sr
    samples_per_frame = int(target_sr / wav2vec_fps)
    resample_ratio = audio_sr / target_sr
    onset_sample = int(speech_start * samples_per_frame * resample_ratio)

    def peaks_to_boundaries(peaks):
        boundaries = [onset_sample]
        for p in peaks + speech_start:
            boundaries.append(int(p * samples_per_frame * resample_ratio))
        boundaries.append(len(audio))
        return boundaries

    boundaries_old = peaks_to_boundaries(peaks_old)
    boundaries_new = peaks_to_boundaries(peaks_new)

    def assign_labels(boundaries, phonemes):
        n_segs = len(boundaries) - 1
        if n_segs == len(phonemes):
            return phonemes
        labels = []
        for j in range(n_segs):
            labels.append(phonemes[j] if j < len(phonemes) else '?')
        return labels

    labels_old = assign_labels(boundaries_old, phonemes_dict)
    labels_new = assign_labels(boundaries_new, phonemes_dict)

    # plot
    fig = plt.figure(figsize=(16, 8))
    gs = gridspec.GridSpec(2, 1)

    for row, (peaks, boundaries, labels, title) in enumerate([
        (peaks_old, boundaries_old, labels_old, 'current: strongest N peaks'),
        (peaks_new, boundaries_new, labels_new, 'new: distributed peaks + offset trimming'),
    ]):
        ax = fig.add_subplot(gs[row])
        ax.plot(distances_smoothed, color='steelblue', linewidth=1.5,
                label='distance')
        ax.axvline(speech_start, color='green', linewidth=1.5,
                   linestyle=':', label='speech start')
        ax.axvline(speech_end_trimmed, color='purple', linewidth=1.5,
                   linestyle=':', label='trimmed end')
        ax.axvline(speech_end, color='gray', linewidth=1,
                   linestyle='--', label='original end')

        for p in peaks + speech_start:
            ax.axvline(p, color='red', linewidth=2, linestyle='--')

        # label each segment
        for j in range(len(boundaries) - 1):
            mid_sample = (boundaries[j] + boundaries[j+1]) / 2
            mid_frame = mid_sample / (samples_per_frame * resample_ratio)
            lbl = labels[j] if j < len(labels) else '?'
            ax.text(mid_frame, ax.get_ylim()[1] * 0.85 if ax.get_ylim()[1] > 0 else 0.1,
                    lbl, ha='center', fontsize=12, fontweight='bold',
                    color='darkred',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        match = len(boundaries) - 1 == len(phonemes_dict)
        ax.set_title(
            f"{title} — {len(boundaries)-1} segments, "
            f"expected {n_phonemes} — {'MATCH' if match else 'MISMATCH'}",
            fontsize=10
        )
        ax.set_ylabel("distance")
        ax.legend(fontsize=7, loc='upper right')

    axes = fig.get_axes()
    axes[-1].set_xlabel("wav2vec frame")
    plt.tight_layout()
    plt.show()

    # audio playback
    print(f"\nfull word audio ({len(audio)/audio_sr*1000:.0f}ms):")
    display(Audio(audio, rate=audio_sr))

    print(f"\ncurrent segmentation:")
    for j, (start, end) in enumerate(zip(boundaries_old[:-1], boundaries_old[1:])):
        lbl = labels_old[j] if j < len(labels_old) else '?'
        dur = (end - start) / audio_sr * 1000
        print(f"  seg {j} [{lbl}]  {dur:.0f}ms")
        display(Audio(audio[start:end], rate=audio_sr))

    print(f"\nnew segmentation:")
    for j, (start, end) in enumerate(zip(boundaries_new[:-1], boundaries_new[1:])):
        lbl = labels_new[j] if j < len(labels_new) else '?'
        dur = (end - start) / audio_sr * 1000
        print(f"  seg {j} [{lbl}]  {dur:.0f}ms")
        display(Audio(audio[start:end], rate=audio_sr))

diagnose_and_play(pipeline, 'P21', 'mensen', instance_idx=0)
# diagnose_and_play(pipeline, 'P22', 'kind', instance_idx=0)
# diagnose_and_play(pipeline, 'P22', 'hogesnelheidstrein', instance_idx=0)

# find sentence audio for the mensen sentence
# it should be reconstructable from the NWB data
# but first check what sentence_list contains

mensen_sent = None
for sent in wsd['sentence_list']:
    if 'mensen' in sent.get('text', ''):
        mensen_sent = sent
        break

print(mensen_sent)

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

# for pid in sorted(pipeline.patient_results.keys()):
#     print(f"--- {pid} ---")
#     r = pipeline.patient_results[pid]
#     print(f"  acc={r['accuracy']:.4f}  n_pred={len(set(r['predictions']))}")
#     try:
#         pipeline.step10_visualize_patient(pid, show_table=False, min_class_samples=run_config.get('min_class_samples', 5))
#     except Exception as e:
#         print(f"  FAILED: {type(e).__name__}: {e}")

import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

all_pids = sorted(set(pipeline.train['phoneme_participant_ids']))
print(f"Testing on {len(all_pids)} patients\n")

configs = [
  #  {'label': 'current (k=1.5, p=0.05)',  'k': 1.5, 'mhf': 0.1,   'prom': 0.05},
#    {'label': 'k=0, p=0.03',              'k': 0.0, 'mhf': 0.0,   'prom': 0.03},
    {'label': 'k=0, p=0.02',              'k': 0.0, 'mhf': 0.0,   'prom': 0.02},
#    {'label': 'k=0, p=0.015',             'k': 0.0, 'mhf': 0.0,   'prom': 0.015},
    {'label': 'k=0, p=0.01',              'k': 0.0, 'mhf': 0.0,   'prom': 0.01},
    {'label': 'current (k=0, p=0.005)',             'k': 0.0, 'mhf': 0.0,   'prom': 0.005},
    {'label': 'k=0, p=0.002',             'k': 0.0, 'mhf': 0.0,   'prom': 0.002},
    {'label': 'k=0, p=0',                 'k': 0.0, 'mhf': 0.0,   'prom': 0.0},
    {'label': 'k=0.1, p=0.01',            'k': 0.1, 'mhf': 0.01,  'prom': 0.01},
    {'label': 'k=0.2, p=0.01',            'k': 0.2, 'mhf': 0.02,  'prom': 0.01},
  #  {'label': 'k=0.3, p=0.01',            'k': 0.3, 'mhf': 0.03,  'prom': 0.01},
    {'label': 'k=0.1, p=0.015',           'k': 0.1, 'mhf': 0.01,  'prom': 0.015},
   # {'label': 'k=0.2, p=0.015',           'k': 0.2, 'mhf': 0.02,  'prom': 0.015},
]

# Collect per-patient results
results = {cfg['label']: {} for cfg in configs}

for pid in all_pids:
    words = list(pipeline.split_result['train'][pid].keys())
    
    for cfg in configs:
        total_peaks = 0
        total_needed = 0
        underfound = 0
        
        for word in words:
            instances = pipeline.split_result['train'][pid][word]
            idx = instances[0]
            wd = pipeline.split_result['word_segments_dict'][pid]['words'][word]['instances'][idx]
            
            audio = wd['audio_segment']
            if audio is None:
                continue
            audio_sr = pipeline.detector.config.audio_sr
            
            wav2vec_features = pipeline.detector.extract_wav2vec_features(audio, audio_sr)
            distances = pipeline.detector.compute_wav2vec_distances(wav2vec_features)
            enhanced = gaussian_filter1d(distances, sigma=1.0)
            
            median_val = np.median(enhanced)
            mad = np.median(np.abs(enhanced - median_val))
            height = median_val + cfg['k'] * mad
            min_height = cfg['mhf'] * np.max(enhanced)
            height = max(height, min_height)
            if height <= 0:
                height = 1e-10
            
            phonemes = pipeline.detector.phonetic_dict.extract_phonemes(word)
            n_needed = len(phonemes) - 1
            
            kwargs = {'height': height, 'distance': 1}
            if cfg['prom'] > 0:
                kwargs['prominence'] = cfg['prom']
            
            peaks, _ = find_peaks(enhanced, **kwargs)
            
            total_peaks += len(peaks)
            total_needed += n_needed
            if len(peaks) < n_needed:
                underfound += 1
        
        results[cfg['label']][pid] = {
            'peaks': total_peaks,
            'needed': total_needed,
            'underfound': underfound,
            'n_words': len(words),
            'ratio': total_peaks / total_needed if total_needed > 0 else 0,
        }

# Print summary table
print(f"{'Config':<25s}", end="")
for pid in all_pids:
    print(f"  {pid:>5s}", end="")
print(f"  {'MEAN':>6s}  {'Under%':>6s}")
print("-" * (25 + 7 * len(all_pids) + 16))

for cfg in configs:
    label = cfg['label']
    print(f"{label:<25s}", end="")
    ratios = []
    under_pcts = []
    for pid in all_pids:
        r = results[label][pid]
        print(f"  {r['ratio']:>5.2f}", end="")
        ratios.append(r['ratio'])
        under_pcts.append(r['underfound'] / r['n_words'] * 100)
    print(f"  {np.mean(ratios):>6.2f}  {np.mean(under_pcts):>5.1f}%")

print()
print("Ratio = peaks_found / peaks_needed (>1.0 = enough peaks)")
print("Under% = mean % of words where not enough peaks were found")

import gc
import json
import os
import copy
from itertools import product

# Save to a NEW file so old results stay separate
logger = ExperimentLogger('experiments_v9_new_border_detection.json')

# ============================================================
# LOAD ALREADY-COMPLETED EXPERIMENTS
# ============================================================
RESULTS_FILE = 'experiments_v9_new_border_detection.json'
done_keys = set()
if os.path.exists(RESULTS_FILE):
    with open(RESULTS_FILE, 'r') as f:
        existing = json.load(f)
    for exp in existing:
        p = exp.get('params', {})
        key = (
            p.get('stacking_order'),
            p.get('stacking_step_size'),
            p.get('target_frames'),
            p.get('min_frames'),
            p.get('max_frames'),
            p.get('scaler_type'),
            p.get('subtract_baseline', False),
        )
        done_keys.add(key)
    print(f"Found {len(done_keys)} completed experiments")

# ============================================================
# PHASE 1: COARSE SWEEP
# ============================================================
stacking_params = [
    (5, 1),
    (5, 2),
    (7, 1),
    (7, 2),
    (9, 1),
    (9, 2),
]
resampling_target_frames = [5, 7]
min_frames_options = [3, 4]
max_frames_options = [150]
scaler_types = ['standard']
subtract_baseline_options = [False]

# ============================================================
# BUILD + FILTER
# ============================================================
configs = []
for (so, ss), mn, mx in product(stacking_params, min_frames_options, max_frames_options):
    configs.append({
        'stacking_order': so, 'stacking_step_size': ss,
        'target_frames': None, 'min_frames': mn, 'max_frames': mx,
    })
for tf, mn, mx in product(resampling_target_frames, min_frames_options, max_frames_options):
    configs.append({
        'stacking_order': None, 'stacking_step_size': None,
        'target_frames': tf, 'min_frames': mn, 'max_frames': mx,
    })

all_experiments = list(product(configs, scaler_types, subtract_baseline_options))

experiments = []
for config, scaler_type, subtract_bl in all_experiments:
    key = (
        config['stacking_order'],
        config['stacking_step_size'],
        config['target_frames'],
        config['min_frames'],
        config['max_frames'],
        scaler_type,
        subtract_bl,
    )
    if key not in done_keys:
        experiments.append((config, scaler_type, subtract_bl))

n_total = len(experiments)
n_skipped = len(all_experiments) - n_total
print(f"Phase 1 coarse sweep: {len(all_experiments)} total | Done: {n_skipped} | Remaining: {n_total}")

# ============================================================
# RUN
# ============================================================
for i, (config, scaler_type, subtract_bl) in enumerate(experiments, 1):
    label = (f"so={config['stacking_order']} ss={config['stacking_step_size']} "
             f"tf={config['target_frames']} fr={config['min_frames']}-{config['max_frames']} "
             f"sc={scaler_type} bl={subtract_bl}")
    print(f"\n--- [{i}/{n_total}] {label} ---")

    try:
        # Restore fresh post-step5 state each iteration
        pipeline.train = copy.deepcopy(cached_train)
        pipeline.test = copy.deepcopy(cached_test)

        run_config.update(config)
        run_config['scaler_type'] = scaler_type
        run_config['subtract_baseline'] = subtract_bl

        if subtract_bl and hasattr(pipeline, 'patient_baselines'):
            pipeline.train = pipeline.subtract_baseline(
                pipeline.train, 'train', pipeline.patient_baselines)
            pipeline.test = pipeline.subtract_baseline(
                pipeline.test, 'test', pipeline.patient_baselines)

        run_step5b(pipeline, run_config)
        pipeline.dutch30_step6_resolve_unknowns()
        pipeline.step7_filter_unknowns(
            unknown_keep_ratio=run_config['unknown_keep_ratio'])

        name, params, results = run_from_config(pipeline, run_config)
        logger.log(name, params, results)

        del results
        gc.collect()

    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()
        continue

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('inline' if 'inline' in matplotlib.get_backend() else matplotlib.get_backend())
import matplotlib.pyplot as plt

with open('experiments_v9_new_border_detection.json', 'r') as f:
    experiments = json.load(f)
print(f"Loaded {len(experiments)} experiments")

print("Building DataFrame...")
rows = []
for exp in experiments:
    row = {
        'name': exp['name'],
        'timestamp': exp['timestamp'],
        'overall_acc': exp['overall_acc'],
        'overall_adj_acc': exp['overall_adj_acc'],
        'overall_pct_classes': exp['overall_pct_classes'],
    }
    for k, v in exp.get('params', {}).items():
        # Skip list/dict params that break DataFrame
        if isinstance(v, (list, dict)):
            continue
        row[f'p_{k}'] = v
    for group, summary in exp.get('group_summaries', {}).items():
        if summary:
            row[f'g_{group}_acc'] = summary['mean_acc']
            row[f'g_{group}_adj'] = summary['mean_adj_acc']
            row[f'g_{group}_pct'] = summary['mean_pct_classes']
    rows.append(row)

df = pd.DataFrame(rows)
df = df.sort_values('overall_adj_acc', ascending=False).reset_index(drop=True)
print(f"DataFrame shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
# ============================================================
# TABLE: Top 10
# ============================================================
key_cols = ['overall_adj_acc', 'overall_acc', 'overall_pct_classes',
            'p_stacking_order', 'p_stacking_step_size', 'p_target_frames',
            'p_min_frames', 'p_max_frames', 'p_scaler_type', 'p_subtract_baseline']
key_cols = [c for c in key_cols if c in df.columns]
print("\nTop 10 by adjusted accuracy:")
print(df[key_cols].head(10).to_string())
# ============================================================
# FIG 1: Distribution + scatter + timeline
# ============================================================
print("Plotting Fig 1...")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].hist(df['overall_adj_acc'].dropna(), bins=30, edgecolor='black', alpha=0.7)
best = df['overall_adj_acc'].max()
med = df['overall_adj_acc'].median()
axes[0].axvline(best, color='red', ls='--', label=f"Best: {best:.4f}")
axes[0].axvline(med, color='orange', ls='--', label=f"Median: {med:.4f}")
axes[0].set_xlabel('Adjusted Accuracy')
axes[0].set_ylabel('Count')
axes[0].set_title('Distribution of Adjusted Accuracy')
axes[0].legend()

axes[1].scatter(df['overall_pct_classes'], df['overall_acc'], alpha=0.4, s=10)
axes[1].set_xlabel('% Classes Predicted')
axes[1].set_ylabel('Raw Accuracy')
axes[1].set_title('Accuracy vs Class Coverage')

# Timeline — simple integer index instead of parsing timestamps
df_sorted = df.sort_values('timestamp').reset_index(drop=True)
axes[2].plot(df_sorted.index, df_sorted['overall_adj_acc'], '.', alpha=0.4, ms=3)
axes[2].set_xlabel('Experiment #')
axes[2].set_ylabel('Adjusted Accuracy')
axes[2].set_title('Accuracy Over Time')

plt.tight_layout()
plt.show()
print("Fig 1 done")
# ============================================================
# FIG 2: Parameter importance
# ============================================================
print("Plotting Fig 2...")
param_cols = [c for c in df.columns if c.startswith('p_') and df[c].nunique() > 1
              and df[c].nunique() < 50]  # skip high-cardinality params

n_params = len(param_cols)
if n_params == 0:
    print("No varying parameters found, skipping")
else:
    ncols = min(4, n_params)
    nrows = (n_params + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    if n_params == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, col in enumerate(param_cols):
        ax = axes[i]
        # Convert to string for groupby to avoid mixed-type issues
        grouped = df.groupby(df[col].astype(str))['overall_adj_acc'].agg(['mean', 'std', 'count'])
        grouped = grouped.sort_values('mean', ascending=True)
        grouped['std'] = grouped['std'].fillna(0)

        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(grouped)))
        ax.barh(range(len(grouped)), grouped['mean'], xerr=grouped['std'],
                color=colors, edgecolor='black', linewidth=0.5, capsize=3)
        ax.set_yticks(range(len(grouped)))
        ax.set_yticklabels([str(v) for v in grouped.index], fontsize=9)
        ax.set_xlabel('Mean Adj. Accuracy')
        ax.set_title(col.replace('p_', ''), fontweight='bold')

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle('Parameter Impact on Adjusted Accuracy', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    print("Fig 2 done")
# ============================================================
# FIG 3: Stacking vs Resampling
# ============================================================
print("Plotting Fig 3...")
is_stacking = df['p_stacking_order'].notna() & (df['p_stacking_order'].astype(str) != 'None')
is_resampling = ~is_stacking

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, mask, title in [(axes[0], is_stacking, 'Stacking'),
                         (axes[1], is_resampling, 'Resampling')]:
    subset = df[mask]
    if subset.empty:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        continue
    ax.hist(subset['overall_adj_acc'].dropna(), bins=20, edgecolor='black', alpha=0.7)
    ax.axvline(subset['overall_adj_acc'].max(), color='red', ls='--',
               label=f"Best: {subset['overall_adj_acc'].max():.4f}")
    ax.axvline(subset['overall_adj_acc'].mean(), color='orange', ls='--',
               label=f"Mean: {subset['overall_adj_acc'].mean():.4f}")
    ax.set_xlabel('Adjusted Accuracy')
    ax.set_title(f'{title} (n={len(subset)})')
    ax.legend()

plt.suptitle('Stacking vs Resampling', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
print("Fig 3 done")

# logger = ExperimentLogger('my_experiments.json')
# logger.best_experiment()

# pipeline.checkpoint_after_step6(sample_fraction=sample_fraction)

print(sorted(set(pipeline.train['phoneme_labels'])))

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

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from collections import defaultdict


def evaluate_segmentation_quality(pipeline, pid, sigma_values=(0, 0.5, 1.0, 2.0),
                                  n_sentences=3, min_phoneme_duration=None,
                                  word_prominence_factor=None):
    """Evaluate phoneme boundary detection quality for different gaussian sigma values.

    Re-runs wav2vec boundary detection with different smoothing parameters and
    reports per-word segmentation quality, sentence visualizations with missed
    phonemes shown as *, and summary statistics per patient.

    Args:
        pipeline: pipeline object with detector, config, and word_segments_dict.
        pid: str, patient id to evaluate.
        sigma_values: tuple of float, gaussian sigma values to try.
            Use 0 for no smoothing.
        n_sentences: int, number of example sentences to show per sigma.
        min_phoneme_duration: float or None, override config value.
        word_prominence_factor: float or None, override config value.
    """
    wsd = pipeline.split_result['word_segments_dict'][pid]
    words_list = wsd['words_list']
    audio_segments = wsd['audio_segments']
    detector = pipeline.detector
    config = pipeline.config

    min_ph_dur = min_phoneme_duration or config.min_phoneme_duration
    prom_factor = word_prominence_factor if word_prominence_factor is not None \
        else getattr(config, 'word_prominence_factor', 0.0)

    wav2vec_fps = 50
    min_phoneme_frames = max(1, int(min_ph_dur * wav2vec_fps))

    print(f"\n{'='*70}")
    print(f"patient: {pid}  total word instances: {len(words_list)}")
    print(f"{'='*70}")

    results_per_sigma = {}

    for sigma in sigma_values:
        sigma_label = f"sigma={sigma}"
        word_results = []

        for i, (word, audio) in enumerate(zip(words_list, audio_segments)):
            phonemes_dict = pipeline.phonetic_dict.extract_phonemes(word)
            if not phonemes_dict:
                continue
            n_expected = len(phonemes_dict)

            # extract wav2vec features and distances
            try:
                wav2vec_features = detector.extract_wav2vec_features(
                    audio, config.audio_sr
                )
                distances = detector.compute_wav2vec_distances(wav2vec_features)
            except Exception:
                continue

            # apply smoothing
            if sigma > 0:
                distances_smoothed = gaussian_filter1d(distances, sigma=sigma)
            else:
                distances_smoothed = distances.copy()

            # detect speech region
            threshold = (np.mean(distances_smoothed)
                         + 0.5 * np.std(distances_smoothed))
            above = np.where(distances_smoothed > threshold * 0.3)[0]
            if len(above) == 0:
                continue
            speech_start = above[0]
            speech_end = above[-1]
            distances_speech = distances_smoothed[speech_start:speech_end]
            if len(distances_speech) < 2:
                continue

            n_boundaries_needed = n_expected - 1

            # adaptive threshold with retry
            median_val = np.median(distances_speech)
            mad = np.median(np.abs(distances_speech - median_val))
            word_k_factors = getattr(config, 'word_threshold_factors', None)
            k = word_k_factors[0] if word_k_factors else 1.0
            peak_threshold = median_val + k * mad

            prom_val = (prom_factor * np.max(distances_speech)
                        if prom_factor > 0 else None)

            n_detected = 0
            for attempt in range(10):
                peak_kwargs = {
                    'height': peak_threshold,
                    'distance': min_phoneme_frames,
                }
                if prom_val is not None:
                    peak_kwargs['prominence'] = prom_val
                peaks, _ = find_peaks(distances_speech, **peak_kwargs)
                if len(peaks) >= n_boundaries_needed:
                    n_detected = min(len(peaks), n_boundaries_needed) + 1
                    break
                peak_threshold *= 0.5

            if n_detected == 0:
                n_detected = 1  # whole word as one segment

            word_results.append({
                'word': word,
                'n_expected': n_expected,
                'n_detected': n_detected,
                'phonemes': phonemes_dict,
                'match': n_detected == n_expected,
            })

        results_per_sigma[sigma] = word_results

    # print summary statistics per sigma
    print(f"\n{'sigma':<12} {'pct_perfect':<14} {'pct_zero':<12} "
          f"{'pct_phonemes':<16} {'n_words'}")
    print("-" * 60)

    for sigma in sigma_values:
        word_results = results_per_sigma[sigma]
        if not word_results:
            continue

        n_words = len(word_results)
        n_perfect = sum(1 for r in word_results if r['match'])
        n_zero = sum(1 for r in word_results if r['n_detected'] == 1
                     and r['n_expected'] > 1)

        total_expected = sum(r['n_expected'] for r in word_results)
        total_detected = sum(min(r['n_detected'], r['n_expected'])
                             for r in word_results)
        pct_phonemes = total_detected / total_expected * 100

        print(f"sigma={sigma:<6} {n_perfect/n_words*100:>10.1f}%   "
              f"{n_zero/n_words*100:>8.1f}%   "
              f"{pct_phonemes:>12.1f}%   "
              f"{n_words}")

    # show sentence examples for each sigma
    for sigma in sigma_values:
        word_results = results_per_sigma[sigma]
        if not word_results:
            continue

        print(f"\n--- sigma={sigma} example sentences ---")

        # group consecutive words into pseudo-sentences
        sentences = []
        current = []
        for r in word_results:
            current.append(r)
            if len(current) >= 8:
                sentences.append(current)
                current = []
        if current:
            sentences.append(current)

        for sent in sentences[:n_sentences]:
            parts = []
            for r in sent:
                detected = r['n_detected']
                expected = r['n_expected']
                phonemes = r['phonemes']

                if detected >= expected:
                    # all phonemes detected
                    ph_str = '-'.join(phonemes)
                else:
                    # show detected phonemes, replace missed with *
                    ph_shown = list(phonemes[:detected])
                    ph_missed = ['*'] * (expected - detected)
                    ph_str = '-'.join(ph_shown + ph_missed)

                parts.append(f"{r['word']}[{ph_str}]")

            print("  " + "  ".join(parts))


def compare_sigma_all_patients(pipeline, sigma_values=(0, 0.5, 1.0, 2.0),
                                n_sentences=2):
    """Run segmentation quality evaluation across all patients.

    Args:
        pipeline: pipeline object.
        sigma_values: tuple of float, gaussian sigma values to compare.
        n_sentences: int, number of example sentences per patient per sigma.
    """
    all_pids = sorted(
        set(pipeline.train['phoneme_participant_ids'])
    )

    for pid in all_pids:
        if pid not in pipeline.split_result['word_segments_dict']:
            continue
        evaluate_segmentation_quality(
            pipeline, pid,
            sigma_values=sigma_values,
            n_sentences=n_sentences
        )
        print()

evaluate_segmentation_quality(
    pipeline, 'P23',
    sigma_values=(0, 0.5, 1.0, 2.0),
    n_sentences=3
)

for pid in ['P25', 'P28', 'P29']:
    evaluate_segmentation_quality(
        pipeline, pid,
        sigma_values=(0, 0.5, 2.0),
        n_sentences=1
    )

import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

fig, axes = plt.subplots(3, 1, figsize=(14, 8))

# raw distances
axes[0].plot(distances, color='steelblue', linewidth=1)
peaks_raw, _ = find_peaks(distances, distance=1)
axes[0].scatter(peaks_raw, distances[peaks_raw], color='red', s=20, zorder=5)
axes[0].set_title(f"raw wav2vec distances — {len(peaks_raw)} peaks")
axes[0].set_ylabel("distance")

# smoothed sigma=2 (current)
d_s2 = gaussian_filter1d(distances, sigma=2)
peaks_s2, _ = find_peaks(d_s2, distance=1)
axes[1].plot(d_s2, color='steelblue', linewidth=1)
axes[1].scatter(peaks_s2, d_s2[peaks_s2], color='red', s=20, zorder=5)
axes[1].set_title(f"smoothed sigma=2 (current) — {len(peaks_s2)} peaks")
axes[1].set_ylabel("distance")

# smoothed sigma=0.5 (less aggressive)
d_s05 = gaussian_filter1d(distances, sigma=0.5)
peaks_s05, _ = find_peaks(d_s05, distance=1)
axes[2].plot(d_s05, color='steelblue', linewidth=1)
axes[2].scatter(peaks_s05, d_s05[peaks_s05], color='red', s=20, zorder=5)
axes[2].set_title(f"smoothed sigma=0.5 — {len(peaks_s05)} peaks")
axes[2].set_ylabel("distance")

plt.tight_layout()
plt.show()

print(f"raw peaks: {len(peaks_raw)}")
print(f"sigma=2 peaks: {len(peaks_s2)}")
print(f"sigma=0.5 peaks: {len(peaks_s05)}")

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
