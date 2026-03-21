# Converted from parse_features_of_30_patients_wav2vec.ipynb

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
from pipeline import UnifiedPhonemePipeline
from config import BIDS_PATH, OUTPUT_PATH, RESULTS_PATH, DUTCH_30_PATH, DUTCH_10_PATH, get_dataset_paths

from dutch_30_pipeline import Dutch30Pipeline
from dutch_30_feature_extractor import Dutch30FeatureExtractor
from phoneme_detection_diagnostic import Dutch30PhonemeDetectionDiagnostic 
from dataset_config import Dutch30Config

from transformers import Wav2Vec2Model, Wav2Vec2Processor
import torch

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

# extractor = Dutch30FeatureExtractor()

# pipeline = Dutch30Pipeline(
#         dutch30_extractor=extractor,
#         debug_mode=False,
#         pca_components= None, #100,
#         feature_extraction_method = 'high_gamma',# 'high_gamma', #'band_powers', #'band_power_hjorth', # 'hjorth', #'band_powers',# 'hjorth', #'high_gamma', # 'band_powers', # 'band_power_hjorth'
#         use_rms_boundaries=False,   
#         use_multifeature=False,
#         use_wav2vec=True,
#         subtract_baseline=False,
#         #baseline_method = 'band_powers' #'feature_matched', 'band_powers', 'raw'
#     )

# sample_fraction = 1
# patient_range = (21,30)
# stacking_order = 7
# stacking_step_size = 1
# max_frames = 25
# target_frames = 4


# print(f"Attempting to load checkpoint (sample_fraction={sample_fraction})...")

# if pipeline.try_load_checkpoint(sample_fraction=sample_fraction,
#                                  stage='after_step6'):
#     print(f"Step 6 checkpoint loaded - ready for classification")
#     print(f"  Train samples: {len(pipeline.train.get('features', []))}")
#     print(f"  Test samples: {len(pipeline.test.get('features', []))}")

# elif pipeline.try_load_checkpoint(sample_fraction=sample_fraction,
#                                    stage='after_step5'):
#     print(f"Step 5 checkpoint loaded - running remaining steps...")
#     print(f"  Train samples: {len(pipeline.train.get('features', []))}")
#     print(f"  Test samples: {len(pipeline.test.get('features', []))}")

#     # Step 5a, b: choose one
#     pipeline.step5a_filter_by_frame_count(min_frames=2, max_frames=max_frames)
#     pipeline.step5b_stack_features(model_order=stacking_order, step_size=stacking_step_size)
#     # pipeline.step5b_normalize_feature_lengths(target_frames=target_frames)

#     # Step 6
#     pipeline.dutch30_step6_resolve_unknowns()
#     # pipeline.checkpoint_after_step6(sample_fraction=sample_fraction)

#     pipeline.step7_filter_unknowns(unknown_keep_ratio=0.0025);

# else:
#     print(f"No checkpoint found. Running full pipeline...")

#     # print(f"\n  Step 1: Loading data (patients {patient_range})...")
#     # pipeline.step1_load_dutch30_data(patient_range=patient_range)

#     # print(f"\n  Step 2: Splitting by instances...")
#     # pipeline.step2_split_by_instances()

#     # print(f"\n  Step 3: Loading channel exclusions...")
#     # pipeline.step3_load_channel_exclusions('channel_exclusions.json')
#     # pipeline.apply_channel_exclusions()
#     # pipeline.print_channel_counts()

#     # print(f"\n  Step 4: custom decoder")
#     # pipeline.step4_custom_detector() 

#     # print(f"\n  Step 5: Accumulating data...")
#     # pipeline.step5_accumulate_data_dutch30()
#     # pipeline.checkpoint_after_step5(sample_fraction=sample_fraction)

#     # #pipeline.step5a_filter_by_frame_count(min_frames=2, max_frames=max_frames)
#     # # Step 5b: choose one    
#     # #pipeline.step5b_stack_features(model_order=stacking_order, step_size=stacking_step_size)
#     # pipeline.step5b_normalize_feature_lengths(target_frames=target_frames)

#     # # Step 6
#     # pipeline.dutch30_step6_resolve_unknowns()
#     # # pipeline.checkpoint_after_step6(sample_fraction=sample_fraction)

#     # pipeline.step7_filter_unknowns(unknown_keep_ratio=0.0025);

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

import importlib
import markov_phoneme_model
importlib.reload(markov_phoneme_model)
from markov_phoneme_model import MarkovPhonemeModel

import inspect
src = inspect.getsource(MarkovPhonemeModel.predict)
print(src[src.find('use_viterbi and'):src.find('use_viterbi and')+300])

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
    'max_frames': 150,
    # Step 5b: choose ONE approach
    'stacking_order': 9,
    'stacking_step_size': 2,
    'target_frames': None,
    # Classifier
    'classifier_type': 'logistic_regression',
    'class_weight': 'balanced',
    'markov_order': 1,
    'use_viterbi': True,
    'random_state': 37,
    'scaler_type': 'standard',
    'feature_pooling_method': 'flatten',
    'min_class_samples': 7,
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

sf = run_config['sample_fraction']
pr = run_config['patient_range']

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

        pipeline.step5b_stack_features(
            model_order=run_config['stacking_order'],
            step_size=run_config['stacking_step_size'])
        count(pipeline, "After 5b (stacking)")

        if hasattr(pipeline, 'step5c_collapse_to_phoneme_level'):
            pipeline.step5c_collapse_to_phoneme_level()
            count(pipeline, "After 5c (collapse)")
        else:
            print("WARNING: step5c_collapse_to_phoneme_level NOT FOUND")

    elif run_config['target_frames'] is not None:
        pipeline.step5b_normalize_feature_lengths(
            target_frames=run_config['target_frames'])
        count(pipeline, "After 5b (resample)")
    else:
        print("WARNING: No step 5b configured")


# ---- Steps 1-3: Load or restore from checkpoint ----
STEP3_CHECKPOINT = f'checkpoint_after_step3_P{pr[0]:02d}-P{pr[1]:02d}.pkl'

if os.path.exists(STEP3_CHECKPOINT):
    print(f"Loading step 3 checkpoint: {STEP3_CHECKPOINT}")
    with open(STEP3_CHECKPOINT, 'rb') as f:
        state = pickle.load(f)
    pipeline.split_result = state['split_result']
    pipeline.patient_data = state['patient_data']
    pipeline.patient_baselines = state['patient_baselines']
    print(f"Step 3 checkpoint loaded")
else:
    print("No step 3 checkpoint found. Running steps 1-3...")
    pipeline.step1_load_dutch30_data(patient_range=pr)
    pipeline.step2_split_by_instances()
    pipeline.step3_load_channel_exclusions('channel_exclusions.json')
    pipeline.apply_channel_exclusions()
    pipeline.print_channel_counts()

    # Save checkpoint
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

# ---- Cache post-step5 state for experiment loop later ----
cached_train = copy.deepcopy(pipeline.train)
cached_test = copy.deepcopy(pipeline.test)
print(f"Cached post-step5 state in memory")

# ---- Steps 5b-7 ----
run_step5b(pipeline, run_config)

pipeline.dutch30_step6_resolve_unknowns()
count(pipeline, "After step 6 (resolve unknowns)")

pipeline.step7_filter_unknowns(
    unknown_keep_ratio=run_config['unknown_keep_ratio'])
count(pipeline, "After step 7 (filter unknowns)")

# ---- Run experiment ----
name, params, results = run_from_config(pipeline, run_config)


pipeline.patient_results = {}
for pid, pr in results.items():
    n_classes = pr['n_classes_test']
    chance = 1.0 / n_classes if n_classes > 0 else 0
    lift = pr['accuracy'] / chance if chance > 0 else 0
    pipeline.patient_results[pid] = {
        'accuracy': pr['accuracy'],
        'lift': lift,
        'predictions': pr['predictions'],
        'predictions_no_viterbi': pr.get('predictions_no_viterbi'),
        'true_labels': pr['true_labels'],
        'model': pr['model'],
        'n_classes': pr['n_classes'],
        'train_size': pr['train_size'],
        'test_size': pr['test_size'],
    }

pids = sorted(pipeline.patient_results.keys())
for pid in pids:
    pipeline.step10_visualize_patient(pid, show_table=False, min_class_samples = run_config.get('min_class_samples', 5))

import gc
import json
import os
from itertools import product

# Save to a NEW file so old results stay separate
logger = ExperimentLogger('experiments_v2_boundary_fix.json')

# ============================================================
# LOAD ALREADY-COMPLETED EXPERIMENTS
# ============================================================
RESULTS_FILE = 'experiments_v2_boundary_fix.json'
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
# PHASE 1: COARSE SWEEP (only ~100 experiments)
# ============================================================
# Fewer stacking params — skip step_size=2 for now
stacking_params = [
    (5, 1),
    (7, 1),
    (9, 1),
    (9, 2),  # your previous best
]

# Fewer resampling targets — spread out
resampling_target_frames = [3, 5, 7, 10]

# Wider frame range since boundaries are now correct
min_frames_options = [2, 4]
max_frames_options = [80, 120]

# Only standard scaler for phase 1
scaler_types = ['standard']

# No baseline subtraction for phase 1
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
        run_config.update(config)
        run_config['scaler_type'] = scaler_type
        run_config['subtract_baseline'] = subtract_bl

       ## pipeline.try_load_checkpoint(sample_fraction=sf, stage='after_step5')

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
        continue


import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('inline' if 'inline' in matplotlib.get_backend() else matplotlib.get_backend())
import matplotlib.pyplot as plt

with open('experiments_v2_boundary_fix.json', 'r') as f:
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

# # # Auto-select 3 consonants + 3 vowels
phoneme_duration_diagnostic(pipeline, pid="P27", phonemes=["t", "n", "s", "d", "k", "r",
                                       "\u0259", "\u025b", "\u0251", "i"], target_frames= 9, model_order=9, step_size=2)

# # # Or pick specific phonemes
# # phoneme_duration_diagnostic(pipeline, pid="P23",
# #                              phonemes=["t", "n", "s", "\u0259", "a:", "\u025b"])

import torch
import torch.nn as nn


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

import numpy as np
import torch
import torch.nn as nn
from collections import Counter
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

MIN_CLASS_SAMPLES = 5

skip_patients = {'P21', 'P24'}
all_pids = sorted(set(pipeline.train['phoneme_participant_ids']))
all_pids = [p for p in all_pids if p not in skip_patients]

print(f"running on {len(all_pids)} patients: {all_pids}")
print()

patient_results = {}

for pid in all_pids:
    train_mask = [p == pid for p in pipeline.train['phoneme_participant_ids']]
    test_mask = [p == pid for p in pipeline.test['phoneme_participant_ids']]

    train_feat = [pipeline.train['features'][i] for i, m in enumerate(train_mask) if m]
    train_labels = [pipeline.train['phoneme_labels'][i] for i, m in enumerate(train_mask) if m]
    test_feat = [pipeline.test['features'][i] for i, m in enumerate(test_mask) if m]
    test_labels = [pipeline.test['phoneme_labels'][i] for i, m in enumerate(test_mask) if m]

    if len(train_feat) < 10 or len(test_feat) < 5:
        continue

    X_train = np.array(train_feat)
    y_train = np.array(train_labels)
    X_test = np.array(test_feat)
    y_test = np.array(test_labels)

    # filter classes with too few training samples
    train_counts = Counter(y_train)
    valid_classes = {cls for cls, cnt in train_counts.items() if cnt >= MIN_CLASS_SAMPLES}
    train_keep = np.array([y in valid_classes for y in y_train])
    test_keep = np.array([y in valid_classes for y in y_test])

    X_train = X_train[train_keep]
    y_train = y_train[train_keep]
    X_test = X_test[test_keep]
    y_test = y_test[test_keep]

    if len(X_train) < 10 or len(X_test) < 5:
        continue

    # label encoder fit on filtered label set
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

    # class weights from filtered training set
    full_counts = Counter(y_train_enc)
    full_total = sum(full_counts.values())
    nn_weights = torch.FloatTensor([
        full_total / (n_classes * full_counts.get(i, 1))
        for i in range(n_classes)
    ])

    print(f"--- {pid} (train={len(X_train)}, test={len(X_test)}, "
          f"classes={n_classes}, chance={chance:.4f}, "
          f"dropped={len(train_counts) - len(valid_classes)}) ---")

    patient_results[pid] = {}

    # sklearn classifiers
    classifiers = {
        'LogReg':     LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        'SVM_RBF':    SVC(kernel='rbf', C=10, gamma='scale', class_weight='balanced', random_state=42),
        'SVM_poly2':  SVC(kernel='poly', degree=2, C=10, gamma='scale', class_weight='balanced', random_state=42),
        'SVM_linear': SVC(kernel='linear', C=1, class_weight='balanced', random_state=42),
        'RF200':      RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42),
        'RF500':      RandomForestClassifier(n_estimators=500, class_weight='balanced', random_state=42),
    }

    for name, clf in classifiers.items():
        clf.fit(X_train_s, y_train)
        preds = clf.predict(X_test_s)
        acc = accuracy_score(y_test, preds)
        n_pred = len(set(preds))
        adj = acc * (n_pred / n_classes)
        patient_results[pid][name] = {
            'acc': acc, 'adj': adj, 'lift': acc / chance,
            'n_pred': n_pred, 'n_classes': n_classes
        }
        print(f"  {name:<15} acc={acc:.4f}  lift={acc/chance:.2f}x  "
              f"cls={n_pred}/{n_classes}  adj={adj:.4f}")

    # neural network variants
    X_tr_t = torch.FloatTensor(X_train_s)
    y_tr_t = torch.LongTensor(y_train_enc)
    X_te_t = torch.FloatTensor(X_test_s)

    for act_name in ['relu', 'snake']:
        torch.manual_seed(42)
        model = build_model(n_features, n_classes, activation=act_name)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss(weight=nn_weights)

        model.train()
        for epoch in range(300):
            perm = torch.randperm(len(X_tr_t))
            for i in range(0, len(X_tr_t), 64):
                idx = perm[i:i+64]
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

        nn_name = f"NN_{act_name}"
        patient_results[pid][nn_name] = {
            'acc': acc, 'adj': adj, 'lift': acc / chance,
            'n_pred': n_pred, 'n_classes': n_classes
        }
        print(f"  {nn_name:<15} acc={acc:.4f}  lift={acc/chance:.2f}x  "
              f"cls={n_pred}/{n_classes}  adj={adj:.4f}")

    print()

# summary
print("-" * 70)
print("mean across patients")
print("%-15s  %8s  %8s  %8s  %8s" % ('classifier', 'acc', 'adjAcc', 'lift', 'classes'))
print("-" * 60)
all_clf_names = sorted(set().union(*(r.keys() for r in patient_results.values())))
for name in all_clf_names:
    accs = [patient_results[pid][name]['acc']
            for pid in patient_results if name in patient_results[pid]]
    adjs = [patient_results[pid][name]['adj']
            for pid in patient_results if name in patient_results[pid]]
    lifts = [patient_results[pid][name]['lift']
             for pid in patient_results if name in patient_results[pid]]
    cls_pcts = [patient_results[pid][name]['n_pred'] / patient_results[pid][name]['n_classes']
                for pid in patient_results if name in patient_results[pid]]
    print("%-15s  %7.2f%%  %7.2f%%  %7.2fx  %7.1f%%" % (
        name, np.mean(accs)*100, np.mean(adjs)*100,
        np.mean(lifts), np.mean(cls_pcts)*100))

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
