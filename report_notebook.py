# report_notebook.py — thesis figures + ablations
# =================================================
# Paste cells (delimited by "# %%") into Jupyter. This runs in its OWN kernel,
# separate from the CRF/Dutch30 and SSL working notebooks. It loads SAVED results
# (pickles in results/) for figures, and loads the live pipeline only for ablations.
#
# Storyline (broad "knowledge-in-the-pipeline" framing, causal via ablations):
#   1. honest comparison  -> data-driven SSL >= linguistically-informed CRF on identity
#   2. component: features -> engineered HG isolate vowel QUALITY + QUANTITY (SSL at chance)
#   3. component: segmentation -> Ablation 1 (oracle MFA vs deployable detection)
#   4. component: phonotactics -> Ablation 2 (phoneme LM on SSL)
#   spine: methods audit (marginal trap, oracle asymmetry, projection overfit, artifact screen)


# %% Cell 0 — imports & setup ===================================================
import os, pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from scipy.stats import ttest_rel, wilcoxon, t as tdist

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.svm import SVC

import importlib, phon_helpers, report_helpers
importlib.reload(phon_helpers); importlib.reload(report_helpers)
from phon_helpers import (manner, place, voicing, cv, vowel_length, vowel_place, is_cons,
                          gather_sequences, needleman_wunsch, edit_distance,
                          subs_position_zip, subs_nw, aligned_pairs_zip, aligned_pairs_nw,
                          feature_z, confusion_stats, phone_feat_dist)
from report_helpers import (phoneme_separability, decision_region_plot,
                            decision_region_pair, hg_amplitude_plot, error_structure_grid,
                            feature_panel, feature_separability_panel, separability_structure_grid,
                            feature_separability_grid)
from config import DUTCH_30_PATH

RESULTS = 'results'
FIGDIR  = 'report'; os.makedirs(FIGDIR, exist_ok=True)
PIDS    = [f'P{i:02d}' for i in range(21, 31)]      # sentence patients
plt.rcParams.update({'figure.dpi': 110, 'savefig.dpi': 200, 'savefig.bbox': 'tight'})

def savefig(name): plt.savefig(os.path.join(FIGDIR, name)); print('saved', name)
def ci95(v): v = np.asarray(v); return v.mean(), 1.96 * v.std(ddof=1) / np.sqrt(len(v))
C_CRF, C_SSL = '#e08a2b', '#3b6fb0'
print("imports OK | data:", DUTCH_30_PATH)


# %% Cell 1 — load saved results ================================================
# Run the SAVE block (Cell 1b) once in the working notebooks first.
def _load(name):
    p = os.path.join(RESULTS, name)
    if not os.path.exists(p):
        print(f"  [missing] {name}"); return None
    with open(p, 'rb') as f: return pickle.load(f)

crf_export   = _load('crf_export.pkl')                 # {pid: out-dict}, CRF 1:1 with gold
ssl_results  = _load('ssl_results.pkl')                # {pid: out-dict}, SSL free-running
crf_perclass = _load('crf_perpatient_perclass.pkl')    # {pid: {phone: OvR AUC}}
ssl_perclass = _load('ssl_perpatient_perclass.pkl')
crf_vpairs   = _load('crf_vowelpairs.pkl')             # {pair: AUC}
ssl_vpairs   = _load('ssl_vowelpairs.pkl')

# raw per-phoneme features/embeddings for separability/projection/amplitude figures
crf_feats = _load('crf_feats.pkl')   # {pid: {'X': f16 (n,d), 'y': str, 'sid': int}}  stacked-HG
ssl_feats = _load('ssl_feats.pkl')   # {pid: {'X': f32 (n,128), 'y': str, 'sid': int}}  embeddings

def get_feats(model, pid):
    """Return (X float32, y, sid) for one patient. model in {'crf','ssl'}."""
    d = (crf_feats if model == 'crf' else ssl_feats)[pid]
    return np.asarray(d['X'], np.float32), np.asarray(d['y']), np.asarray(d['sid'])

pids = sorted(set(crf_export) & set(ssl_results)) if (crf_export and ssl_results) else PIDS
print("patients:", pids)


# %% Cell 1b — EXPORT FROM WORKING NOTEBOOKS (run ONCE in each; NOT in this kernel)
# --- CRF notebook (features + decoded outputs) ---
#   import pickle, numpy as np
#   labs = np.asarray(labs); sids = np.asarray(sids); ppid = np.asarray(ppid)
#   crf_feats = {pid: {'X': np.array([vecs[i] for i in np.where(ppid==pid)[0]], np.float16),
#                      'y': labs[ppid==pid].astype(str), 'sid': sids[ppid==pid].astype(int)}
#                for pid in sorted(set(ppid))}
#   pickle.dump(crf_feats,  open('results/crf_feats.pkl',  'wb'))
#   pickle.dump(crf_export, open('results/crf_export.pkl', 'wb'))     # may already exist
#
# --- SSL notebook (embeddings + decoded outputs) ---
#   import pickle, numpy as np
#   ssl_feats = {}
#   for pid in sorted(embeddings):
#       X, y, g = build_pho_matrix(pid)
#       ssl_feats[pid] = {'X': np.asarray(X, np.float32), 'y': np.asarray(y).astype(str),
#                         'sid': np.asarray(g).astype(int)}
#   pickle.dump(ssl_feats,   open('results/ssl_feats.pkl',   'wb'))
#   pickle.dump(ssl_results, open('results/ssl_results.pkl', 'wb'))
# (already saved earlier: *_perpatient_perclass.pkl, *_vowelpairs.pkl, *_perclass_auc.pkl)


# %% Cell 2 — DESCRIPTIVE STATS (self-contained, straight from MFA) ==============
from run_pipeline import load_mfa_alignments
mfa = {pid: a for pid in PIDS if (a := load_mfa_alignments(pid))}
print(f"MFA loaded for {len(mfa)} patients")

# corpus scale
allph = [ph['phone'] for a in mfa.values() for s in a.values() for ph in s]
c = Counter(allph); n = sum(c.values()); nC = sum(v for k, v in c.items() if is_cons(k))
print(f"phonemes={n} unique={len(c)} | C/V={nC/n:.0%}/{1-nC/n:.0%} | "
      f"imbalance={max(c.values())/min(c.values()):.0f}x")

# duration by phoneme (vowel-quantity backbone)
dur = defaultdict(list)
for a in mfa.values():
    for s in a.values():
        for ph in s: dur[ph['phone']].append((ph['end_s'] - ph['start_s']) * 1000)
longv  = [d for p, v in dur.items() if vowel_length(p) == 'long'  for d in v]
shortv = [d for p, v in dur.items() if vowel_length(p) == 'short' for d in v]
cons   = [d for p, v in dur.items() if is_cons(p)                 for d in v]
print(f"mean dur (ms): long V {np.mean(longv):.0f} | short V {np.mean(shortv):.0f} | C {np.mean(cons):.0f}")
phs = sorted(dur, key=lambda p: -np.mean(dur[p]))
plt.figure(figsize=(15, 4))
plt.bar(range(len(phs)), [np.mean(dur[p]) for p in phs],
        color=['#1f77b4' if not is_cons(p) else '0.6' for p in phs])
plt.xticks(range(len(phs)), phs, fontsize=8); plt.ylabel('mean duration (ms)')
plt.title('Phoneme durations (blue = vowels) — long vowels dominate')
savefig('fig_descriptive_duration.png'); plt.show()


# %% Cell 3 — HEADLINE: long-vowel separability (the key positive finding) =======
# paired pairwise-AUC: CRF separates long vowels by quality; SSL at chance
if crf_vpairs and ssl_vpairs:
    k = sorted(set(crf_vpairs) & set(ssl_vpairs))
    cvv = np.array([crf_vpairs[p] for p in k]); svv = np.array([ssl_vpairs[p] for p in k])
    print(f"long-vowel pairwise AUC: CRF mean={cvv.mean():.3f}  SSL mean={svv.mean():.3f}  "
          f"CRF>SSL {int((cvv > svv).sum())}/{len(k)}  Wilcoxon p={wilcoxon(cvv, svv)[1]:.4g}")
    order = np.argsort(-(cvv))
    x = np.arange(len(k))
    plt.figure(figsize=(11, 4.5))
    plt.bar(x - 0.2, cvv[order], 0.4, color=C_CRF, label='CRF')
    plt.bar(x + 0.2, svv[order], 0.4, color=C_SSL, label='SSL')
    plt.axhline(0.5, color='k', ls=':', lw=0.8)
    plt.xticks(x, [k[i] for i in order], rotation=45, ha='right', fontsize=8)
    plt.ylabel('pairwise AUC'); plt.legend()
    plt.title('Long-vowel quality separability (duration-matched) — CRF vs SSL')
    savefig('fig_vowel_pairwise_auc.png'); plt.show()


# %% Cell 4 — (paste other report figures here) ================================
# e.g. per-class OvR AUC (crf_perclass/ssl_perclass), PER/WER table, prediction-marginal
# divergence, manner/place/voicing absolute-rate panels, held-out vowel-space projection.
# All of these consume crf_export / ssl_results / the *_perclass dicts loaded above.


# %% ============================================================================
# %% ABLATIONS  (need the LIVE pipeline, not just pickles)
# %% ============================================================================
# Ablation 1 (segmentation) reuses the CRF pipeline's trained classifier + the
# AcousticChangeDetector + raw EEG, so it must run with `pipeline` in memory.
# Two options:
#   (a) develop it HERE after loading the pipeline (heavier setup), or
#   (b) run it in the CRF notebook, pickle the per-condition PER, and load here for figures.
# We'll wire the loop once you paste the CRF training/predict cell. Skeleton below.

# %% Cell A1 — Ablation 1 setup (segmentation: oracle MFA vs deployable detection)
# Conditions, held classifier/features/data fixed, vary only TEST segmentation:
#   A  = MFA boundaries (oracle, lexicon)
#   B' = detector given gold count   -> value of PLACEMENT  (A - B')
#   B  = detector auto-count (n=None) -> fully deployable    -> value of COUNT (B' - B)
#   C  = uniform / equal-spaced (null floor)
# Metric: PER (Levenshtein, per sentence); per-patient paired t/Wilcoxon + CI.
# TODO: paste CRF train/predict cell -> fill classifier interface, then build the loop.
print("Ablation 1 skeleton ready — share the CRF train/predict cell to wire it up.")
