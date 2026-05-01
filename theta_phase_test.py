# Test whether adding theta-band phase features (cos+sin) alongside the
# high-gamma amplitude helps phoneme decoding. Lets the CRF learn theta-gamma
# coupling implicitly rather than computing PAC explicitly.
#
# Strategy: monkey-patch extractHG to return [hg_amplitude, cos_theta, sin_theta]
# instead of just [hg_amplitude]. Triples the per-frame feature width.
#
# Tapping into existing pipeline — same caching/patch pattern as
# envelope_variants_test.py.

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 1 — Imports + setup
# ═══════════════════════════════════════════════════════════════════════════════

import os
import shutil
import pickle
import numpy as np
import scipy.signal
import scipy.fftpack
import matplotlib.pyplot as plt
from datetime import datetime

import extract_features
import dutch_30_feature_extractor
import dutch_30_pipeline
import acoustic_change_detector
import run_pipeline

from extract_features import extractHG as _orig_extractHG
from dutch_30_pipeline import Dutch30Pipeline
from dutch_30_feature_extractor import Dutch30FeatureExtractor
from dataset_config import Dutch30Config
from run_pipeline import (
    DEFAULT_RUN_CONFIG,
    run_path_b,
    _run_crf_experiment,
)

ARCHIVE_DIR = 'archive'
os.makedirs(ARCHIVE_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 2 — Augmented extractHG: high-gamma amplitude + theta phase (cos, sin)
# ═══════════════════════════════════════════════════════════════════════════════
# Same pwr_lpf path as the production extractHG, but ALSO computes theta-band
# (4-8 Hz) phase per channel and returns it as cos/sin features alongside.
#
# Output shape: (n_windows, 3 * n_channels)
#   columns [0 : n_ch]              high-gamma amplitude (as in production)
#   columns [n_ch : 2*n_ch]         cos(theta_phase)  per channel
#   columns [2*n_ch : 3*n_ch]       sin(theta_phase)  per channel
#
# Why cos/sin instead of phase angle:
#   Phase is circular (-π and +π are the same physical state but far apart
#   numerically). cos/sin embedding makes phase linearly representable for
#   the CRF.

def extractHG_with_theta(data, sr, windowLength=0.015, frameshift=0.005,
                         debug=False, smoothing_hz=10.0,
                         theta_low=4.0, theta_high=8.0):
    """High-gamma amplitude + theta-phase cos/sin, concatenated."""

    detrended = scipy.signal.detrend(data, axis=0)
    n_ch = detrended.shape[1]

    # ── High-gamma amplitude (the existing pipeline path) ──────────────────
    sos_hg = scipy.signal.iirfilter(4, [70/(sr/2), 170/(sr/2)],
                                    btype='bandpass', output='sos')
    hg = scipy.signal.sosfiltfilt(sos_hg, detrended, axis=0)
    for f_notch in (100.0, 150.0):
        sos_n = scipy.signal.iirfilter(4, [(f_notch-2)/(sr/2), (f_notch+2)/(sr/2)],
                                       btype='bandstop', output='sos')
        hg = scipy.signal.sosfiltfilt(sos_n, hg, axis=0)
    pwr = hg ** 2
    sos_lp = scipy.signal.iirfilter(4, smoothing_hz/(sr/2),
                                    btype='lowpass', output='sos')
    smoothed = np.abs(scipy.signal.sosfiltfilt(sos_lp, pwr, axis=0))

    # ── Theta phase (4-8 Hz) per channel ──────────────────────────────────
    sos_theta = scipy.signal.iirfilter(4, [theta_low/(sr/2), theta_high/(sr/2)],
                                       btype='bandpass', output='sos')
    theta_sig = scipy.signal.sosfiltfilt(sos_theta, detrended, axis=0)
    analytic = scipy.signal.hilbert(
        theta_sig, scipy.fftpack.next_fast_len(theta_sig.shape[0]), axis=0
    )[:theta_sig.shape[0]]
    theta_phase = np.angle(analytic)
    cos_theta = np.cos(theta_phase)
    sin_theta = np.sin(theta_phase)

    # ── Window-average all three streams onto the same frame grid ──────────
    n_win = int(np.floor((data.shape[0] - windowLength*sr) / (frameshift*sr)))
    feat_hg  = np.zeros((n_win, n_ch))
    feat_cos = np.zeros((n_win, n_ch))
    feat_sin = np.zeros((n_win, n_ch))
    for w in range(n_win):
        s = int(np.floor(w * frameshift * sr))
        e = int(np.floor(s + windowLength * sr))
        feat_hg [w, :] = smoothed [s:e, :].mean(axis=0)
        feat_cos[w, :] = cos_theta[s:e, :].mean(axis=0)
        feat_sin[w, :] = sin_theta[s:e, :].mean(axis=0)
    feat_hg = np.sqrt(feat_hg)

    if debug:
        print(f"  extractHG_with_theta: HG shape {feat_hg.shape}, "
              f"output shape {(n_win, 3*n_ch)}")

    return np.concatenate([feat_hg, feat_cos, feat_sin], axis=1)


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 3 — Cache helpers + monkey-patch utilities (same pattern as envelope test)
# ═══════════════════════════════════════════════════════════════════════════════

def _archive_caches(tag):
    moved = []
    for fname in ['cache_frames_P21-P30.pkl',
                  'cache_step5_P21-P30_stk20_s1.pkl',
                  'checkpoint_after_step3_P21-P30.pkl']:
        if os.path.exists(fname):
            base, ext = os.path.splitext(fname)
            dst = os.path.join(ARCHIVE_DIR, f'{base}_{tag}{ext}')
            shutil.move(fname, dst)
            moved.append((fname, dst))
    return moved


def _restore_caches(tag):
    restored = []
    for fname in ['cache_frames_P21-P30.pkl',
                  'cache_step5_P21-P30_stk20_s1.pkl',
                  'checkpoint_after_step3_P21-P30.pkl']:
        base, ext = os.path.splitext(fname)
        cached = os.path.join(ARCHIVE_DIR, f'{base}_{tag}{ext}')
        if os.path.exists(cached) and not os.path.exists(fname):
            shutil.copy2(cached, fname)
            restored.append((cached, fname))
    return restored


def patch_extractHG(fn):
    for mod in [extract_features, dutch_30_feature_extractor,
                dutch_30_pipeline, acoustic_change_detector, run_pipeline]:
        if hasattr(mod, 'extractHG'):
            mod.extractHG = fn
    print(f"  ✓ extractHG patched to {fn.__name__}")


def restore_original():
    for mod in [extract_features, dutch_30_feature_extractor,
                dutch_30_pipeline, acoustic_change_detector, run_pipeline]:
        if hasattr(mod, 'extractHG'):
            mod.extractHG = _orig_extractHG
    print("  ✓ extractHG restored to original")


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 4 — Run experiment: pipeline with theta phase added
# ═══════════════════════════════════════════════════════════════════════════════

def run_with_theta(run_config):
    """Build pipeline with theta-augmented extractHG, then run CRF."""
    print(f"\n{'='*70}\n  EXPERIMENT: HG amplitude + theta phase (cos, sin)\n{'='*70}")

    # 1. Stash any in-flight production caches
    _archive_caches(tag=f'staging_{datetime.now().strftime("%H%M%S")}')

    # 2. Restore theta caches if previously computed
    _restore_caches(tag='theta_phase')

    # 3. Patch in the augmented extractHG
    patch_extractHG(extractHG_with_theta)

    # 4. Build pipeline + extract features (this re-runs feature extraction
    #    on first call, then caches; subsequent calls are fast)
    extractor = Dutch30FeatureExtractor()
    pipeline = Dutch30Pipeline(
        dutch30_extractor=extractor, debug_mode=False,
        feature_extraction_method=run_config['feature_extraction_method'],
        use_wav2vec=False, subtract_baseline=run_config['subtract_baseline'],
        use_rms_boundaries=False, use_multifeature=False,
    )
    run_path_b(pipeline, run_config)

    f0 = pipeline.train['features'][0]
    print(f"  Feature shape after stacking: {f0.shape} "
          f"(includes 3× channels: HG, cos_theta, sin_theta)")

    # 5. Run CRF
    pipeline.patient_results = {}
    crf_results = _run_crf_experiment(pipeline, run_config)

    # 6. Lift summary
    accs, lifts, summary = [], [], {}
    for pid, r in crf_results.items():
        n_cl = len(set(r['true_labels']))
        chance = 1.0 / n_cl if n_cl > 0 else 0
        lift = r['accuracy'] / chance if chance > 0 else 0
        accs.append(r['accuracy']); lifts.append(lift)
        summary[pid] = {'accuracy': r['accuracy'], 'lift': lift, 'n_classes': n_cl}
        print(f"    {pid}: acc={r['accuracy']:.3f}  lift={lift:.2f}×")
    print(f"  MEAN: acc={np.mean(accs):.3f}  lift={np.mean(lifts):.2f}×")

    # 7. Stash this experiment's caches under its tag for fast reruns
    moved = _archive_caches(tag='theta_phase')
    for src, dst in moved:
        shutil.copy2(dst, src)

    return {'lifts': lifts, 'mean_lift': float(np.mean(lifts)),
            'per_patient': summary}


run_config = dict(DEFAULT_RUN_CONFIG)
run_config['use_viterbi'] = True

results_theta = run_with_theta(run_config)


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 5 — Compare with production (HG only) baseline
# ═══════════════════════════════════════════════════════════════════════════════
# Production baseline lift on the same patient cohort is ~5.37×
# (from window_frameshift_sweep.py). Compare against that.

PRODUCTION_BASELINE_LIFT = 5.37   # from your previous sweep at w15/s5/stk20

print("\n" + "="*70)
print("  Comparison vs production baseline (HG amplitude only)")
print("="*70)
print(f"  {'pid':<6} {'baseline':>10} {'+theta_phase':>14} {'Δ':>8}")
print("-"*70)

# If you have per-patient production baseline numbers handy, replace this
# constant dict; otherwise the script just shows the +theta result.
PRODUCTION_PER_PATIENT = {  # paste your previous per-patient lifts here if you have them
    'P21': 4.78, 'P22': 5.57, 'P23': 5.37, 'P24': 4.30, 'P25': 5.36,
    'P26': 5.27, 'P27': 5.67, 'P28': 5.60, 'P29': 5.78, 'P30': 6.01,
}
for pid in sorted(results_theta['per_patient']):
    theta_lift = results_theta['per_patient'][pid]['lift']
    base = PRODUCTION_PER_PATIENT.get(pid, float('nan'))
    delta = theta_lift - base if not np.isnan(base) else float('nan')
    print(f"  {pid:<6} {base:>9.2f}× {theta_lift:>13.2f}× {delta:>+7.2f}×")
print("-"*70)
print(f"  {'mean':<6} {PRODUCTION_BASELINE_LIFT:>9.2f}× "
      f"{results_theta['mean_lift']:>13.2f}× "
      f"{results_theta['mean_lift'] - PRODUCTION_BASELINE_LIFT:>+7.2f}×")
print("="*70)


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 6 — Save and restore
# ═══════════════════════════════════════════════════════════════════════════════

out_path = f'theta_phase_results_{datetime.now().strftime("%Y%m%d_%H%M")}.pkl'
with open(out_path, 'wb') as f:
    pickle.dump(results_theta, f)
print(f"\n  Saved results to {out_path}")

restore_original()
print("  Done. Production extractHG restored.")
