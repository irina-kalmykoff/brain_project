# Converted from report.ipynb

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

pids = sorted(set(crf_export) & set(ssl_results)) if (crf_export and ssl_results) else PIDS
print("patients:", pids)

import numpy as np, matplotlib.pyplot as plt
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, silhouette_score

def phoneme_separability(X, y, grp=None, title='', top_k=12, min_count=25, n_pca=50, n_splits=5):
    X = np.asarray(X); y = np.asarray(y)
    keep = [c for c, n in Counter(y).items() if n >= min_count]
    m = np.isin(y, keep); X, y = X[m], y[m]
    grp = np.asarray(grp)[m] if grp is not None else None
    classes = sorted(set(y)); cidx = {c: i for i, c in enumerate(classes)}

    # ---------- HONEST: cross-validated per-class one-vs-rest AUC ----------
    if grp is not None:
        splits = list(GroupKFold(n_splits).split(X, y, grp))     # prevents same-sentence leakage
    else:
        print("  [warn] no grp passed -> StratifiedKFold; same-sentence leakage NOT prevented")
        splits = list(StratifiedKFold(n_splits, shuffle=True, random_state=0).split(X, y))
    steps = [StandardScaler()]
    if X.shape[1] > n_pca: steps.append(PCA(n_components=n_pca, random_state=0))
    steps.append(LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'))
    clf = make_pipeline(*steps)
    proba = np.zeros((len(y), len(classes)))
    for tr, te in splits:
        clf.fit(X[tr], y[tr]); p = clf.predict_proba(X[te])
        for j, c in enumerate(clf.classes_):
            if c in cidx: proba[te, cidx[c]] = p[:, j]
    ovr = {c: roc_auc_score((y == c).astype(int), proba[:, cidx[c]])
           for c in classes if 0 < (y == c).sum() < len(y)}
    macro = np.mean(list(ovr.values()))

    # ---------- descriptive: standardized centroids + silhouette (in-sample, unsupervised) ----------
    Xs = StandardScaler().fit_transform(X)
    Xp = PCA(n_components=min(n_pca, Xs.shape[1], Xs.shape[0] - 1)).fit_transform(Xs)
    sil = silhouette_score(Xp, y)
    C = np.array([Xs[y == c].mean(0) for c in classes])
    Cn = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-9)
    S = Cn @ Cn.T
    print(f"{title}:  n={len(y)} classes={len(classes)} | "
          f"macro OvR AUC={macro:.3f}  max={max(ovr.values()):.3f}  "
          f"#>0.7={sum(a > 0.7 for a in ovr.values())} | silhouette={sil:.3f}")

    # ---------- HONEST held-out LDA(2) projection (train-fit, test-plotted) ----------
    tr, te = splits[0]
    proj = make_pipeline(StandardScaler(),
                         PCA(n_components=min(n_pca, X.shape[1]), random_state=0),
                         LinearDiscriminantAnalysis(n_components=2)).fit(X[tr], y[tr])
    Z = proj.transform(X[te]); yte = y[te]
    top = [c for c, _ in Counter(y).most_common(top_k) if c in classes]

    fig, ax = plt.subplots(1, 3, figsize=(20, 6))
    im = ax[0].imshow(S, cmap='viridis', vmin=-1, vmax=1)
    ax[0].set_xticks(range(len(classes))); ax[0].set_xticklabels(classes, rotation=90, fontsize=7)
    ax[0].set_yticks(range(len(classes))); ax[0].set_yticklabels(classes, fontsize=7)
    ax[0].set_title(f"{title}\ncentroid cosine (bright off-diag = look-alike)", fontsize=10)
    plt.colorbar(im, ax=ax[0], fraction=0.046)

    cmap = plt.get_cmap('tab20')
    for i, c in enumerate(top):
        mm = yte == c
        ax[1].scatter(Z[mm, 0], Z[mm, 1], s=12, alpha=0.55, color=cmap(i % 20), label=str(c))
    ax[1].set_title(f"{title}\nHELD-OUT LDA(2)", fontsize=10)
    ax[1].legend(fontsize=7, ncol=2, markerscale=2)

    ph = sorted(ovr, key=lambda c: -ovr[c])
    ax[2].bar(range(len(ph)), [ovr[c] for c in ph], color='#3b6fb0')
    ax[2].axhline(0.7, color='r', ls='--', lw=0.8); ax[2].axhline(0.5, color='k', ls=':', lw=0.8)
    ax[2].set_xticks(range(len(ph))); ax[2].set_xticklabels(ph, rotation=90, fontsize=7)
    ax[2].set_ylabel('cross-validated OvR AUC')
    ax[2].set_title(f"{title}\nper-phoneme separability", fontsize=10)
    plt.tight_layout(); plt.show()
    return dict(macro_auc=float(macro), max_auc=float(max(ovr.values())),
                n_above_0p7=int(sum(a > 0.7 for a in ovr.values())),
                silhouette=float(sil), per_class=ovr, n=len(y), n_classes=len(classes))

# Cell 1 — load saved results + features
def _load(name):
    p = os.path.join(RESULTS, name)
    if not os.path.exists(p):
        print(f"  [missing] {name}"); return None
    with open(p, 'rb') as f: return pickle.load(f)

crf_export   = _load('crf_export.pkl')
ssl_results  = _load('ssl_results.pkl')
crf_perclass = _load('crf_perpatient_perclass.pkl')
ssl_perclass = _load('ssl_perpatient_perclass.pkl')
crf_vpairs   = _load('crf_vowelpairs.pkl')
ssl_vpairs   = _load('ssl_vowelpairs.pkl')
crf_feats    = _load('crf_feats.pkl')
ssl_feats    = _load('ssl_feats.pkl')

def get_feats(model, pid):
    d = (crf_feats if model == 'crf' else ssl_feats)[pid]
    return np.asarray(d['X'], np.float32), np.asarray(d['y']), np.asarray(d['sid'])

pids = sorted(set(crf_export) & set(ssl_results)) if (crf_export and ssl_results) else PIDS
print("patients:", pids)

# ### Features

amp = hg_amplitude_plot(crf_feats, title='CRF per-phoneme HG amplitude')

# ### Phoneme Separability

res_crf = phoneme_separability(*get_feats('crf', 'P22'), title='CRF P22 (stacked-HG)')
res_ssl = phoneme_separability(*get_feats('ssl', 'P22'), title='SSL P22 (128-d emb)')

# decision_region_plot(*get_feats('crf', 'P22'), title='CRF P22', proj='lda')
# decision_region_plot(*get_feats('ssl', 'P22'), title='SSL P22', proj='lda')   # proj='pca' for the conservative version
decision_region_pair(get_feats, 'P22', proj='lda')

import numpy as np, pickle
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

def phoneme_auc_perclass(X, y, grp, min_count=25, n_splits=5, n_pca=50):
    y, grp = np.asarray(y), np.asarray(grp)
    classes = sorted(c for c, n in Counter(y).items() if n >= min_count)
    keep = np.isin(y, classes); X, y, grp = np.asarray(X)[keep], y[keep], grp[keep]
    cidx = {c: i for i, c in enumerate(classes)}
    proba = np.zeros((len(y), len(classes)))
    steps = [StandardScaler()]
    if n_pca and X.shape[1] > n_pca:
        steps.append(PCA(n_components=n_pca, random_state=0))
    steps.append(LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'))
    clf = make_pipeline(*steps)
    for tr, te in GroupKFold(n_splits).split(X, y, grp):
        clf.fit(X[tr], y[tr]); p = clf.predict_proba(X[te])
        for j, c in enumerate(clf.classes_):
            if c in cidx: proba[te, cidx[c]] = p[:, j]
    per = {}
    for c in classes:
        yb = (y == c).astype(int)
        if 0 < yb.sum() < len(yb):
            per[c] = roc_auc_score(yb, proba[:, cidx[c]])
    return per

crf_feats = pickle.load(open('results/crf_feats.pkl', 'rb'))
ssl_feats = pickle.load(open('results/ssl_feats.pkl', 'rb'))

def per_patient(feats):
    return {pid: phoneme_auc_perclass(np.asarray(d['X'], np.float32),
                                      np.asarray(d['y']),
                                      np.asarray(d['sid']).astype(int),
                                      min_count=25, n_pca=50)
            for pid, d in feats.items()}

pickle.dump(per_patient(crf_feats), open('results/crf_perpatient_perclass.pkl', 'wb'))
pickle.dump(per_patient(ssl_feats), open('results/ssl_perpatient_perclass.pkl', 'wb'))

# Tail-separability test: is the CRF representation more separable than the SSL
# one for its BEST phonemes? (The cohort MEAN AUC is tied ~0.54, so we probe the
# TAIL instead — the single best phoneme, and how many phonemes clear a "well-
# separated" bar.) Patient is the unit: paired t-test across the 10 patients.
# Requires crf/ssl per-class AUCs built with the phoneme_auc_perclass function (PCA 50, min_count 25) 

from scipy.stats import ttest_rel
crf = pickle.load(open('results/crf_perpatient_perclass.pkl', 'rb'))
ssl = pickle.load(open('results/ssl_perpatient_perclass.pkl', 'rb'))
pids = sorted(set(crf) & set(ssl))
# --- Metric 1: best-phoneme AUC per patient (the top of the tail) ------------
cmax = np.array([max(crf[p].values()) for p in pids]); smax = np.array([max(ssl[p].values()) for p in pids])
smax = np.array([max(ssl[p].values()) for p in pids]) 

# --- Metric 2: how many phonemes each patient separates well (AUC > 0.7) ------
cn07 = np.array([sum(a > 0.7 for a in crf[p].values()) for p in pids])
sn07 = np.array([sum(a > 0.7 for a in ssl[p].values()) for p in pids]) 

# --- Report each metric: per-patient values, cohort mean, and the paired test -
# ttest_rel(cmax, smax) pairs CRF vs SSL within each patient and tests whether
# the per-patient differences are reliably non-zero across the cohort.
print("max AUC  CRF", cmax.round(2), "mean", cmax.mean().round(3))
print("         SSL", smax.round(2), "mean", smax.mean().round(3),
      " paired-t p =", round(ttest_rel(cmax, smax)[1], 4))   # CRF best vs SSL best
print("#>0.7    CRF", cn07, "mean", cn07.mean())
print("         SSL", sn07, "mean", sn07.mean(),
      " paired-t p =", round(ttest_rel(cn07, sn07)[1], 4))    # CRF #well-sep vs SSL #well-sep

# CRF's representation separates its best phonemes reliably better than SSL's.

# Which phoneme is each patient's BEST (the one driving max AUC)?
print("CRF best phoneme per patient:")
for p in pids:
    best = max(crf[p], key=crf[p].get)
    print(f"  {p}: {best}  (AUC {crf[p][best]:.2f})")

# Which phonemes clear 0.7, pooled across patients?
from collections import Counter
above = Counter()
for p in pids:
    for ph, a in crf[p].items():
        if a > 0.7: above[ph] += 1
print("\nphonemes with AUC>0.7 (count of patients):", above.most_common())

from collections import defaultdict
import numpy as np
def agg(pp):
    aucs = defaultdict(list)
    for d in pp.values():
        for ph, a in d.items(): aucs[ph].append(a)
    return {ph: (np.mean(v), sum(a > 0.7 for a in v), len(v)) for ph, v in aucs.items()}
ca, sa = agg(crf), agg(ssl)
print(f"{'ph':<4}{'CRFμ':>8}{'>0.7':>8}{'SSLμ':>8}{'>0.7':>8}{' #patients with phoneme >= min count':>5}")
for ph in sorted(ca, key=lambda p: -ca[p][0])[:15]:
    sm, sn, _ = sa.get(ph, (np.nan, 0, 0))
    print(f"{ph:<4}{ca[ph][0]:>8.3f}{ca[ph][1]:>8}{sm:>8.3f}{sn:>8}{ca[ph][2]:>5}")

import itertools, numpy as np
# pair_auc must be defined/imported here (PCA 50, to match CRF)
LONG = ['aː', 'eː', 'iː', 'oː', 'uː', 'yː', 'øː']
crf_feats = {p: {**d, 'X': np.asarray(d['X'], np.float32)}
             for p, d in pickle.load(open('results/crf_feats.pkl', 'rb')).items()}
def long_pairwise(feats):
    rows = []
    for a, b in itertools.combinations(LONG, 2):
        vals = [pair_auc(d['X'], d['y'], d['sid'], a, b) for d in feats.values()]
        vals = [v for v in vals if np.isfinite(v)]
        if len(vals) >= 5: rows.append((f"{a}-{b}", float(np.mean(vals)), len(vals)))
    return sorted(rows, key=lambda r: -r[1])

for nm, m, n in long_pairwise(ssl_feats): print(f"{nm:10} SSL {m:.3f} (n={n})")
print("SSL mean:", np.mean([m for _, m, _ in long_pairwise(ssl_feats)]).round(3))
print("CRF mean:", np.mean([m for _, m, _ in long_pairwise(crf_feats)]).round(3))

from scipy.stats import ttest_rel, wilcoxon
cr = {nm: m for nm, m, n in long_pairwise(crf_feats)}
sr = {nm: m for nm, m, n in long_pairwise(ssl_feats)}
pp = sorted(set(cr) & set(sr))
c = np.array([cr[p] for p in pp]); s = np.array([sr[p] for p in pp])
print(f"CRF>SSL in {int((c>s).sum())}/{len(pp)} pairs   Δ={c.mean()-s.mean():+.3f}")
print(f"paired t  p={ttest_rel(c, s).pvalue:.4g}")
print(f"Wilcoxon  p={wilcoxon(c, s).pvalue:.4g}")

import itertools
def per_patient_long(feats):
    out = {}
    for pid, d in feats.items():
        X = np.asarray(d['X'], np.float32)
        vals = [pair_auc(X, d['y'], d['sid'], a, b) for a, b in itertools.combinations(LONG, 2)]
        vals = [v for v in vals if np.isfinite(v)]
        if vals: out[pid] = np.mean(vals)      # this patient's mean long-vowel pairwise AUC
    return out

cp, sp = per_patient_long(crf_feats), per_patient_long(ssl_feats)
ids = sorted(set(cp) & set(sp))
c = np.array([cp[p] for p in ids]); s = np.array([sp[p] for p in ids])
print(f"by-patient  CRF {c.mean():.3f}  SSL {s.mean():.3f}  CRF>SSL {int((c>s).sum())}/{len(ids)}")
print(f"paired t  p={ttest_rel(c, s).pvalue:.4g}   Wilcoxon p={wilcoxon(c, s).pvalue:.4g}")

# CRF separates long vowels by quality significantly better than SSL. Per patient:  p ≈ 0.03, with t and Wilcoxon agreeing (0.031 vs 0.027) (not a single outlier patient driving test results)
# by-pair test more significant (p ≈ 0.005)  because the 15 pairs aren't independent (vowels reused in different pairs)

import numpy as np
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from scipy.stats import ttest_rel, wilcoxon
from phon_helpers import vowel_place          # front/central/back; None for consonants

def backness_auc(d, n_pca=50, min_count=10):
    X = np.asarray(d['X'], np.float32); y = np.asarray(d['y']); g = np.asarray(d['sid'])
    lab = np.array([vowel_place(p) for p in y], dtype=object)
    m = np.array([l is not None for l in lab]); X, lab, g = X[m], lab[m].astype(str), g[m]
    keep = [c for c, n in Counter(lab).items() if n >= min_count]
    mm = np.isin(lab, keep); X, lab, g = X[mm], lab[mm], g[mm]
    cats = sorted(set(lab))
    if len(cats) < 2 or len(set(g)) < 5: return np.nan
    n_comp = min(n_pca, X.shape[1], int(0.8*len(X)) - 1)
    steps = [StandardScaler()]
    if X.shape[1] > n_comp: steps.append(PCA(n_comp, random_state=0))
    steps.append(LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'))
    proba = cross_val_predict(make_pipeline(*steps), X, lab, cv=GroupKFold(5),
                              groups=g, method='predict_proba')
    if len(cats) == 2:
        return roc_auc_score((lab == cats[1]).astype(int), proba[:, 1])
    return roc_auc_score(lab, proba, multi_class='ovo', average='macro', labels=cats)

cb = {p: backness_auc(d) for p, d in crf_feats.items()}
sb = {p: backness_auc(d) for p, d in ssl_feats.items()}
ids = [p for p in sorted(set(cb) & set(sb)) if np.isfinite(cb[p]) and np.isfinite(sb[p])]
c = np.array([cb[p] for p in ids]); s = np.array([sb[p] for p in ids])
print(f"backness  CRF {c.mean():.3f}  SSL {s.mean():.3f}  CRF>SSL {int((c>s).sum())}/{len(ids)}")
print(f"paired t  p={ttest_rel(c, s).pvalue:.4g}   Wilcoxon p={wilcoxon(c, s).pvalue:.4g}")

# CRF features are more distinguishable for vowel backness than SSL: 10.10 differences positive

import numpy as np
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from scipy.stats import ttest_rel, wilcoxon
from phon_helpers import manner, place, voicing, is_cons

def feat_sep_auc(d, fn, n_pca=50, min_count=10):
    X = np.asarray(d['X'], np.float32); y = np.asarray(d['y']); g = np.asarray(d['sid'])
    lab = np.array([fn(p) for p in y], dtype=object)
    m = np.array([l is not None for l in lab]); X, lab, g = X[m], lab[m].astype(str), g[m]
    keep = [c for c, n in Counter(lab).items() if n >= min_count]
    mm = np.isin(lab, keep); X, lab, g = X[mm], lab[mm], g[mm]
    cats = sorted(set(lab))
    if len(cats) < 2 or len(set(g)) < 5: return np.nan
    n_comp = min(n_pca, X.shape[1], int(0.8*len(X)) - 1)
    steps = [StandardScaler()]
    if X.shape[1] > n_comp: steps.append(PCA(n_comp, random_state=0))
    steps.append(LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'))
    proba = cross_val_predict(make_pipeline(*steps), X, lab, cv=GroupKFold(5),
                              groups=g, method='predict_proba')
    if len(cats) == 2:
        return roc_auc_score((lab == cats[1]).astype(int), proba[:, 1])
    return roc_auc_score(lab, proba, multi_class='ovo', average='macro', labels=cats)

cons_manner = lambda p: manner(p) if is_cons(p) else None   # manner restricted to consonants
for name, fn in [('manner', cons_manner), ('place', place), ('voicing', voicing)]:
    cb = {p: feat_sep_auc(d, fn) for p, d in crf_feats.items()}
    sb = {p: feat_sep_auc(d, fn) for p, d in ssl_feats.items()}
    ids = [p for p in sorted(set(cb) & set(sb)) if np.isfinite(cb[p]) and np.isfinite(sb[p])]
    c = np.array([cb[p] for p in ids]); s = np.array([sb[p] for p in ids])
    print(f"{name:8} CRF {c.mean():.3f}  SSL {s.mean():.3f}  CRF>SSL {int((c>s).sum())}/{len(ids)}"
          f"   t p={ttest_rel(c, s).pvalue:.3g}   W p={wilcoxon(c, s).pvalue:.3g}")

# No consonant-feature advantage
# AUCs are all near chance 

import numpy as np
from collections import defaultdict
from scipy.stats import ttest_rel, wilcoxon
from phon_helpers import is_cons

cons_only = lambda pp: {p: {ph: a for ph, a in d.items() if is_cons(ph)} for p, d in pp.items()}
crf_c, ssl_c = cons_only(crf), cons_only(ssl)

# --- per-consonant table: mean AUC across patients, #patients>0.7, n ---
def agg(pp):
    a = defaultdict(list)
    for d in pp.values():
        for ph, v in d.items(): a[ph].append(v)
    return {ph: (np.mean(v), sum(x > 0.7 for x in v), len(v)) for ph, v in a.items()}
ca, sa = agg(crf_c), agg(ssl_c)
print(f"{'ph':4}{'CRFμ':>8}{'>0.7':>6}{'SSLμ':>8}{'>0.7':>6}{'n':>4}")
for ph in sorted(ca, key=lambda p: -ca[p][0]):
    sm, sn, _ = sa.get(ph, (np.nan, 0, 0))
    print(f"{ph:4}{ca[ph][0]:>8.3f}{ca[ph][1]:>6}{sm:>8.3f}{sn:>6}{ca[ph][2]:>4}")

# --- per-patient: best consonant + how many consonants clear 0.7 ---
pids = sorted(set(crf_c) & set(ssl_c))
cmax = np.array([max(crf_c[p].values()) for p in pids]); smax = np.array([max(ssl_c[p].values()) for p in pids])
cn07 = np.array([sum(a > 0.7 for a in crf_c[p].values()) for p in pids])
sn07 = np.array([sum(a > 0.7 for a in ssl_c[p].values()) for p in pids])
print(f"\nmax consonant AUC  CRF {cmax.mean():.3f}  SSL {smax.mean():.3f}"
      f"   paired-t p={ttest_rel(cmax, smax).pvalue:.3g}  W p={wilcoxon(cmax, smax).pvalue:.3g}")
print(f"#consonants>0.7    CRF mean {cn07.mean():.2f}  SSL mean {sn07.mean():.2f}")
print(f"   CRF per patient {cn07.tolist()}")
print(f"   SSL per patient {sn07.tolist()}")

# CRF is uniformly ~0.01–0.02 higher across consonants
# plausibly a dimensionality effect, not a phonetic one. 

# Consonant feature categories (manner, place, voicing) show no reliable difference between the representations. CRF's best consonant separates marginally better than SSL's (0.70 vs 0.66, p=0.01), but this is near chance, driven by the salient sibilant /s/ (the only consonant exceeding 0.7 in more than one patient, and available to both models), and reflects a small uniform separability offset consistent with the PCA-50 reduction rather than consonant-specific phonetic encoding. The CRF advantage is substantial and category-level only for vowels (quality and backness); for consonants it is weak, near-chance, and sibilant-driven.

# Consonant feature-separability contrasts (CRF vs SSL) — plosive is the test of interest
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pickle
from scipy.stats import ttest_rel, wilcoxon
import report_helpers as R

pids = sorted(set(crf_feats) & set(ssl_feats))

GROUPS = {
    'manner:plosive':   {'p','b','t','d','k','g'},
    'manner:fricative': {'f','v','s','z','x','ɣ','h'},
    'manner:nasal':     {'m','n','ŋ'},
    'manner:approx':    {'l','r','j','w','ʋ','ɥ'},
    'manner:sibilant':  {'s','z'},
    'place:labial':     {'p','b','m','f','v','ʋ','w'},   # bilabial + labiodental
    'place:alveolar':   {'t','d','s','z','n','l','r'},
    'place:velar':      {'k','g','x','ɣ','ŋ'},
}
alpha = 0.05 / len(GROUPS)            # Bonferroni over the 8 contrasts

def grp_auc(d, S):                    # binary group-vs-rest AUC for one patient
    return R._feat_auc(d, lambda p: 'in' if p in S else 'out')   # PCA50 + LDA, GroupKFold by sentence

print(f"{'contrast':>17} {'CRF':>6} {'SSL':>6} {'>SSL':>6} {'t p':>7} {'Wilcox':>7} {'Bonf':>4}")
print('-' * 62)
results = {'pids': pids, 'alpha_bonferroni': alpha}
for name, S in GROUPS.items():
    cz = np.array([grp_auc(crf_feats[p], S) for p in pids])
    sz = np.array([grp_auc(ssl_feats[p], S) for p in pids])
    ok = np.isfinite(cz) & np.isfinite(sz)
    tp = ttest_rel(cz[ok], sz[ok])[1]; wp = wilcoxon(cz[ok], sz[ok])[1]
    results[name] = dict(crf=cz, ssl=sz, t_p=float(tp), wilcoxon_p=float(wp),
                         crf_gt_ssl=int((cz[ok] > sz[ok]).sum()), n=int(ok.sum()))
    print(f"{name:>17} {np.nanmean(cz):6.3f} {np.nanmean(sz):6.3f} "
          f"{int((cz[ok] > sz[ok]).sum()):>3}/{ok.sum():<2} {tp:7.3f} {wp:7.3f} "
          f"{'*' if tp < alpha else '':>4}")
print('-' * 62)
print(f"Bonferroni alpha = {alpha:.4f}  ('*' = survives correction)")

pickle.dump(results, open('results/consonant_contrasts.pkl', 'wb'))   # reproducibility
print('saved results/consonant_contrasts.pkl')

import numpy as np
from scipy.stats import ttest_rel, t as tdist
from phon_helpers import gather_sequences, needleman_wunsch, cv
from sklearn.metrics import matthews_corrcoef

def cv_mcc_fair(out):                         # no is_crf — gap class penalizes over-generation
    gp, pp = gather_sequences(out); G, P = [], []
    for sid in set(gp) | set(pp):
        for g, p in needleman_wunsch(gp.get(sid, []), pp.get(sid, [])):
            G.append(cv(g) if g is not None else 'gap')
            P.append(cv(p) if p is not None else 'gap')
    return matthews_corrcoef(G, P)

crf_fair = np.array([cv_mcc_fair(crf_export[pid]) for pid in pids])
ssl_fair = np.array([cv_mcc_fair(ssl_results[pid]) for pid in pids])
d = crf_fair - ssl_fair; md = d.mean()
half = tdist.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
print(f"fair C/V MCC: CRF={crf_fair.mean():.3f}  SSL={ssl_fair.mean():.3f}  "
      f"Δ={md:+.3f}  CI[{md-half:+.3f},{md+half:+.3f}]  p={ttest_rel(crf_fair, ssl_fair)[1]:.3g}")

import importlib, report_helpers; importlib.reload(report_helpers)
from report_helpers import feature_separability_grid
feature_separability_grid(crf_feats, ssl_feats,
                          savepath='report/fig_feature_separability.png')   # n_perm=200 default

import importlib, report_helpers
importlib.reload(report_helpers)
from report_helpers import error_structure_grid
error_structure_grid(crf_export, ssl_results, savepath='report/fig_error_structure.png')

# \paragraph{Error structure.} The two decoders' substitution errors differ at the broadest phonetic level. 
# Because consonants outnumber vowels (∼62/38), a model that merely follows the class marginal already keeps an error in-class
# ∼0.54 of the time is the baseline. Against it, the CRF decoder preserves the consonant/vowel class of the intended phoneme above chance (0.593 vs a marginal floor of 0.541 +
# +0.053, all ten patients Wilcoxon p=0.002), 
# whereas the SSL decoder does not (0.568 vs 0.556 = +0.013), consistent with its collapse toward the phoneme prior. The two also differ directly (Δ=+0.025, p=0.011). 
# Finer consonant features (manner, place, voicing) and vowel duration are preserved at statistically indistinguishable rates once the over-generation of the free-running SSL decoder is charged as gap errors.

# ### Permutation Test

# import pickle, numpy as np, scipy.stats as ss

# ssl = pickle.load(open('results/ssl_shift_perm.pkl', 'rb'))
# crf = pickle.load(open('results/crf_shift_perm.pkl', 'rb'))

# def summary(res):
#     pids = sorted(res)
#     p = np.clip([res[k]['p_one_sided'] for k in pids], 1e-300, 1.0)
#     q = ss.false_discovery_control(p, method='bh')
#     fisher = 1 - ss.chi2.cdf(-2*np.log(p).sum(), 2*len(p))
#     zpos = sum(res[k]['z'] > 0 for k in pids)
#     return int((q < 0.05).sum()), len(pids), fisher, zpos

# print("=== per-phoneme, feature-rotation null ===")
# print(f"{'metric':<10}{'model':<5}{'BH-FDR':>9}{'Fisher p':>12}{'z>0':>7}")
# for metric in ['acc_pho', 'ce_pho']:
#     for name, d in [('SSL', ssl), ('CRF', crf)]:
#         nsig, n, fp, zpos = summary(d[metric])
#         print(f"{metric:<10}{name:<5}{nsig:>4}/{n:<4}{fp:>12.2e}{zpos:>4}/{n}")

# # per-patient z side by side + paired test (SSL vs CRF)
# print("\n=== per-patient z (SSL vs CRF) + paired test ===")
# for metric in ['acc_pho', 'ce_pho']:
#     sd, cd = ssl[metric], crf[metric]
#     pids = sorted(set(sd) & set(cd))
#     sz = np.array([sd[p]['z'] for p in pids]); cz = np.array([cd[p]['z'] for p in pids])
#     d = sz - cz
#     print(f"\n{metric}:  {'pid':<5}{'SSL z':>8}{'CRF z':>8}")
#     for p, a, b in zip(pids, sz, cz):
#         print(f"      {p:<5}{a:+8.2f}{b:+8.2f}")
#     t = ss.ttest_rel(sz, cz); w = ss.wilcoxon(sz, cz)
#     print(f"   SSL>CRF {int((d>0).sum())}/{len(d)}  mean delta z={d.mean():+.2f}  "
#           f"t p={t.pvalue:.3g}  Wilcoxon p={w.pvalue:.3g}")

import pickle, numpy as np, scipy.stats as ss

ssl_crf = pickle.load(open('results/ssl_shift_perm_crfsplit.pkl', 'rb'))
crf = pickle.load(open('results/crf_shift_perm.pkl', 'rb'))

def summary(res):
    pids = sorted(res)
    p = np.clip([res[k]['p_one_sided'] for k in pids], 1e-300, 1.0)
    q = ss.false_discovery_control(p, method='bh')
    fisher = 1 - ss.chi2.cdf(-2*np.log(p).sum(), 2*len(p))
    zpos = sum(res[k]['z'] > 0 for k in pids)
    return int((q < 0.05).sum()), len(pids), fisher, zpos

print("=== per-phoneme, feature-rotation null ===")
print(f"{'metric':<10}{'model':<5}{'BH-FDR':>9}{'Fisher p':>12}{'z>0':>7}")
for metric in ['acc_pho', 'ce_pho']:
    for name, d in [('SSL', ssl_crf), ('CRF', crf)]:
        nsig, n, fp, zpos = summary(d[metric])
        print(f"{metric:<10}{name:<5}{nsig:>4}/{n:<4}{fp:>12.2e}{zpos:>4}/{n}")

# per-patient z side by side + paired test (SSL vs CRF)
print("\n=== per-patient z (SSL vs CRF) + paired test ===")
for metric in ['acc_pho', 'ce_pho']:
    sd, cd = ssl_crf[metric], crf[metric]
    pids = sorted(set(sd) & set(cd))
    sz = np.array([sd[p]['z'] for p in pids]); cz = np.array([cd[p]['z'] for p in pids])
    d = sz - cz
    print(f"\n{metric}:  {'pid':<5}{'SSL z':>8}{'CRF z':>8}")
    for p, a, b in zip(pids, sz, cz):
        print(f"      {p:<5}{a:+8.2f}{b:+8.2f}")
    t = ss.ttest_rel(sz, cz); w = ss.wilcoxon(sz, cz)
    print(f"   SSL>CRF {int((d>0).sum())}/{len(d)}  mean delta z={d.mean():+.2f}  "
          f"t p={t.pvalue:.3g}  Wilcoxon p={w.pvalue:.3g}")

# #### PER/WER

# %% WER — closed-vocab, shared sentences only (fair), report notebook
import pickle, numpy as np
from collections import defaultdict, Counter
from scipy.stats import ttest_rel, t as tdist
from phon_helpers import needleman_wunsch, gather_sequences, edit_distance
from run_pipeline import load_mfa_alignments

crf_export  = pickle.load(open('results/crf_export.pkl', 'rb'))
ssl_results = pickle.load(open('results/ssl_results.pkl', 'rb'))
pids = sorted(set(crf_export) & set(ssl_results))

# lexicon + gold (phone,word) per sentence — from MFA, no `datasets` needed
gold_pw = {pid: {} for pid in pids}
word_prons = defaultdict(Counter)
for pid in pids:
    for sid, phs in load_mfa_alignments(pid).items():
        pw = [(ph['phone'], (ph['word'] or '').lower()) for ph in phs if ph.get('word')]
        gold_pw[pid][sid] = pw
        runs = []
        for i, (p, w) in enumerate(pw):
            if not runs or pw[i - 1][1] != w: runs.append([w, []])
            runs[-1][1].append(p)
        for w, ph in runs: word_prons[w][tuple(ph)] += 1
lexicon = {w: list(c.most_common(1)[0][0]) for w, c in word_prons.items()}
vocab = list(lexicon)
print(f"lexicon: {len(vocab)} words")

_cache = {}
def recognize(pred_str):
    key = tuple(pred_str)
    if key in _cache: return _cache[key]
    best, bd = None, 10**9
    for w in vocab:
        dd = edit_distance(key, lexicon[w])
        if dd < bd: bd, best = dd, w
    _cache[key] = best
    return best

def sentence_errors(pw, pred_phones):
    runs, run_of = [], {}
    for i, (p, w) in enumerate(pw):
        if not runs or pw[i - 1][1] != w: runs.append([w, []])
        runs[-1][1].append(i); run_of[i] = len(runs) - 1
    al = needleman_wunsch([g for g, _ in pw], pred_phones)
    per_run = defaultdict(list); gi = 0
    for g, p in al:
        if g is not None:
            if p is not None: per_run[run_of[gi]].append(p)
            gi += 1
    return sum(recognize(per_run.get(rid, [])) != w for rid, (w, _) in enumerate(runs)), len(runs)

def wer_on(out, pid, sids):
    _, pred_per = gather_sequences(out); W = N = 0
    for sid in sids:
        if sid in pred_per and sid in gold_pw[pid]:
            w, n = sentence_errors(gold_pw[pid][sid], pred_per[sid]); W += w; N += n
    return (W / N if N else np.nan), N

crf_w, ssl_w = {}, {}
print(f"\n{'pid':4} | shared_sents | shared_words | CRF WER | SSL WER")
for pid in pids:
    _, cpred = gather_sequences(crf_export[pid])
    _, spred = gather_sequences(ssl_results[pid])
    shared = set(cpred) & set(spred) & set(gold_pw[pid])
    cw, nw_words = wer_on(crf_export[pid], pid, shared)
    sw, _        = wer_on(ssl_results[pid], pid, shared)
    crf_w[pid], ssl_w[pid] = cw, sw
    print(f"{pid:4} | {len(shared):11} | {nw_words:11} | {cw:.3f}   | {sw:.3f}")

ok = [p for p in pids if np.isfinite(crf_w[p]) and np.isfinite(ssl_w[p])]
c = np.array([crf_w[p] for p in ok]); s = np.array([ssl_w[p] for p in ok])
d = c - s; P = ttest_rel(c, s)[1]; md = d.mean()
half = tdist.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
print(f"\nWER (shared): CRF={c.mean():.3f}  SSL={s.mean():.3f}  "
      f"Delta={md:+.3f} CI[{md-half:+.3f},{md+half:+.3f}] p={P:.3g}")
pickle.dump({'crf': crf_w, 'ssl': ssl_w, 'n_patients': len(ok)},
            open('results/wer_shared.pkl', 'wb'))
print('saved results/wer_shared.pkl')

# %% PER on shared sentences only 
import pickle, numpy as np
from scipy.stats import ttest_rel, t as tdist
from phon_helpers import gather_sequences

crf_export  = pickle.load(open('results/crf_export.pkl', 'rb'))
ssl_results = pickle.load(open('results/ssl_results.pkl', 'rb'))
pids = sorted(set(crf_export) & set(ssl_results))

def edit_counts(ref, hyp):
    n, m = len(ref), len(hyp)
    D = np.zeros((n + 1, m + 1), int); D[:, 0] = np.arange(n + 1); D[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c = 0 if ref[i - 1] == hyp[j - 1] else 1
            D[i, j] = min(D[i - 1, j] + 1, D[i, j - 1] + 1, D[i - 1, j - 1] + c)
    i, j, s, d, ins = n, m, 0, 0, 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and D[i, j] == D[i - 1, j - 1] + (0 if ref[i - 1] == hyp[j - 1] else 1):
            if ref[i - 1] != hyp[j - 1]: s += 1
            i -= 1; j -= 1
        elif i > 0 and D[i, j] == D[i - 1, j] + 1: d += 1; i -= 1
        else: ins += 1; j -= 1
    return s, d, ins

def per_shared(out, sids):
    gper, pper = gather_sequences(out); S = Dl = I = N = 0
    for sid in sids:
        g, p = gper.get(sid, []), pper.get(sid, [])
        s, d, ins = edit_counts(g, p); S += s; Dl += d; I += ins; N += len(g)
    return S, Dl, I, N

rows = []
print(f"{'pid':4} | shared | words | {'CRF PER':9}(S/D/I) | {'SSL PER':9}(S/D/I)")
for pid in pids:
    _, cpred = gather_sequences(crf_export[pid]); _, spred = gather_sequences(ssl_results[pid])
    shared = set(cpred) & set(spred)
    cS, cD, cI, cN = per_shared(crf_export[pid], shared)
    sS, sD, sI, sN = per_shared(ssl_results[pid], shared)
    if cN == 0 or sN == 0: continue
    cper, sper = (cS + cD + cI) / cN, (sS + sD + sI) / sN
    rows.append((pid, cper, sper, cS / cN, sS / sN))
    print(f"{pid:4} | {len(shared):6} | {cN:5} | {cper:6.3f} ({cS}/{cD}/{cI})".ljust(48) +
          f"| {sper:6.3f} ({sS}/{sD}/{sI})")

cp = np.array([r[1] for r in rows]); sp = np.array([r[2] for r in rows])
csub = np.array([r[3] for r in rows]); ssub = np.array([r[4] for r in rows])

def paired(c, s, label):
    d = c - s; P = ttest_rel(c, s)[1]; md = d.mean()
    half = tdist.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    print(f"{label}: CRF={c.mean():.3f}  SSL={s.mean():.3f}  "
          f"Delta={md:+.3f} CI[{md-half:+.3f},{md+half:+.3f}] p={P:.3g}")

print()
paired(cp, sp,   "FULL PER (shared)      ")
paired(csub, ssub, "SUB-only (shared, fair)")
pickle.dump({'full': {'crf': cp, 'ssl': sp}, 'sub': {'crf': csub, 'ssl': ssub},
             'pids': [r[0] for r in rows]}, open('results/per_shared.pkl', 'wb'))
print('saved results/per_shared.pkl')

# %% Articulatory-feature distance of consonant errors — shared sentences only (fair)
from phon_helpers import (phone_feat_dist, feat_dist_z, is_cons,
                          gather_sequences, needleman_wunsch)
from scipy.stats import ttest_rel, wilcoxon
import numpy as np

def subs_shared(out, sids, align):
    """Substitution pairs (g!=p) restricted to `sids`; align='zip' (CRF 1:1) or 'nw' (SSL)."""
    gper, pper = gather_sequences(out); pairs = []
    for sid in sids:
        g, p = gper.get(sid, []), pper.get(sid, [])
        it = zip(g, p) if align == 'zip' else needleman_wunsch(g, p)
        for a, b in it:
            if align == 'zip':
                if a != b: pairs.append((a, b))
            else:
                if a is not None and b is not None and a != b: pairs.append((a, b))
    return pairs

cons = lambda prs: [(g, p) for g, p in prs if is_cons(g) and is_cons(p)]

shared_by_pid, rows = {}, []
for pid in pids:
    _, cpred = gather_sequences(crf_export[pid]); _, spred = gather_sequences(ssl_results[pid])
    shared = set(cpred) & set(spred); shared_by_pid[pid] = shared
    cs = cons(subs_shared(crf_export[pid],  shared, 'zip'))
    ss = cons(subs_shared(ssl_results[pid], shared, 'nw'))
    dc = np.mean([phone_feat_dist(g, p) for g, p in cs]) if len(cs) >= 5 else np.nan
    ds = np.mean([phone_feat_dist(g, p) for g, p in ss]) if len(ss) >= 5 else np.nan
    rows.append((pid, dc, ds)); print(f"{pid}: CRF {dc:.3f} (n={len(cs):3d})   SSL {ds:.3f} (n={len(ss):3d})")

A = np.array([(c, s) for _, c, s in rows if np.isfinite(c) and np.isfinite(s)])
c, s = A[:, 0], A[:, 1]
print(f"\nMean articulatory-feature distance of consonant errors (SHARED sents, 0..3, lower=nearer-miss):")
print(f"  CRF {c.mean():.3f}   SSL {s.mean():.3f}   Δ(CRF−SSL) = {c.mean()-s.mean():+.3f}")
print(f"  paired t p={ttest_rel(c, s)[1]:.3g}   Wilcoxon p={wilcoxon(c, s)[1]:.3g}   (n={len(c)} patients)")

allc = cons([pr for pid in pids for pr in subs_shared(crf_export[pid],  shared_by_pid[pid], 'zip')])
alls = cons([pr for pid in pids for pr in subs_shared(ssl_results[pid], shared_by_pid[pid], 'nw')])
oc, bc, zc, nc = feat_dist_z(allc); osv, bs, zs, ns = feat_dist_z(alls)
print(f"\nWithin-model (pooled SHARED, vs own shuffle; NEGATIVE z = closer than chance = good):")
print(f"  CRF obs {oc:.3f} base {bc:.3f} z={zc:+.2f} (n={nc})")
print(f"  SSL obs {osv:.3f} base {bs:.3f} z={zs:+.2f} (n={ns})")

# %% Cell — WORKED EXAMPLE: same test sentence, both models, PER calculation =====
from run_pipeline import load_mfa_alignments
from collections import Counter
from matplotlib.patches import Patch

PID = 'P22'
mfa = load_mfa_alignments(PID)

def sent_seq(d, sid):
    tl, pl = np.asarray(d['true_labels']), np.asarray(d['predictions'])
    ts, ps = np.asarray(d['true_sentence_ids']), np.asarray(d['pred_sentence_ids'])
    return list(tl[ts == sid]), list(pl[ps == sid])

common = sorted(set(np.asarray(crf_export[PID]['true_sentence_ids']).tolist()) &
                set(np.asarray(ssl_results[PID]['true_sentence_ids']).tolist()))

print(f"{PID}: sentences in BOTH test sets:", common)

SID  = common[0]                       # <- change to taste
words = [w for w in dict.fromkeys(ph['word'] for ph in mfa.get(SID, [])) if w]
gold, crf_pred = sent_seq(crf_export[PID], SID)
_,    ssl_pred = sent_seq(ssl_results[PID], SID)

def aligned_ops(gold, pred, one_to_one=False):
    if one_to_one:                                     # CRF: 1:1, only match/sub
        return [(g, p, 'match' if g == p else 'sub') for g, p in zip(gold, pred)]
    ops = []                                           # SSL: NW, full S/D/I
    for g, p in needleman_wunsch(gold, pred):
        ops.append((g, p, 'ins' if g is None else 'del' if p is None
                    else 'match' if g == p else 'sub'))
    return ops

COL = {'match': '#bfe3b8', 'sub': '#f3c969', 'del': '#cccccc', 'ins': '#e79a9a'}
def draw_align(ax, ops, title):
    for i, (g, p, op) in enumerate(ops):
        for y in (0, 1):
            ax.add_patch(plt.Rectangle((i, y), 1, 0.9, facecolor=COL[op], edgecolor='w'))
        ax.text(i + .5, 1.45, '' if g is None else g, ha='center', va='center', fontsize=8)
        ax.text(i + .5, 0.45, '' if p is None else p, ha='center', va='center', fontsize=8)
    c = Counter(op for *_, op in ops)
    S, D, I = c['sub'], c['del'], c['ins']
    N = sum(1 for g, *_ in ops if g is not None)
    ax.text(len(ops) + 0.3, 0.95,
            f"S={S}  D={D}  I={I}\nN={N}\nPER=(S+D+I)/N={(S+D+I)/N:.2f}",
            ha='left', va='center', fontsize=8, family='monospace')
    ax.set_xlim(0, len(ops) + 6); ax.set_ylim(-0.2, 2.1)
    ax.set_yticks([0.45, 1.45]); ax.set_yticklabels(['pred', 'gold'])
    ax.set_xticks([]); ax.set_title(title, loc='left', fontsize=10)

ops_crf = aligned_ops(gold, crf_pred, one_to_one=True)
ops_ssl = aligned_ops(gold, ssl_pred, one_to_one=False)
W = max(len(ops_crf), len(ops_ssl))
fig, axes = plt.subplots(2, 1, figsize=(min(20, 0.33 * W + 4), 5.2))
draw_align(axes[0], ops_crf, f"CRF (1:1 oracle segmentation)   sent {SID}:  {' '.join(words)}")
draw_align(axes[1], ops_ssl, "SSL (free-running, NW-aligned)")
axes[0].legend(handles=[Patch(facecolor=COL[k], label=k) for k in ('match','sub','del','ins')],
               ncol=4, loc='lower left', bbox_to_anchor=(0, 1.12), fontsize=8, frameon=False)
plt.tight_layout(); savefig('fig_worked_example.png'); plt.show()

# %% Cell — CONSONANT CONFUSION MATRICES (CRF vs SSL) ===========================
def conf_counts(aligned_fn, results):
    cnt = Counter()
    for pid in pids:
        for g, p in aligned_fn(results[pid]):
            if is_cons(g) and is_cons(p):
                cnt[(g, p)] += 1
    return cnt

cc = conf_counts(aligned_pairs_zip, crf_export)   # CRF 1:1 (incl. correct on diagonal)
sc = conf_counts(aligned_pairs_nw,  ssl_results)  # SSL NW

mord = {'plosive': 0, 'fricative': 1, 'nasal': 2, 'approx': 3}
cons = sorted({c for (g, p) in list(cc) + list(sc) for c in (g, p)},
              key=lambda c: (mord.get(manner(c), 9), str(place(c)), c))
idx = {c: i for i, c in enumerate(cons)}; n = len(cons)

def matrix(cnt):
    M = np.zeros((n, n))
    for (g, p), v in cnt.items(): M[idx[g], idx[p]] += v
    s = M.sum(1, keepdims=True)
    return np.divide(M, s, where=s > 0)               # row-normalise: P(pred | gold)

fig, axes = plt.subplots(1, 2, figsize=(16, 7.5))
for ax, (nm, cnt) in zip(axes, [('CRF', cc), ('SSL', sc)]):
    M = matrix(cnt)
    im = ax.imshow(M, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(n)); ax.set_xticklabels(cons, fontsize=7, rotation=90)
    ax.set_yticks(range(n)); ax.set_yticklabels(cons, fontsize=7)
    ax.set_xlabel('predicted'); ax.set_ylabel('gold')
    ax.set_title(f'{nm}: consonant confusions, P(pred|gold)\n'
                 'ordered by manner/place — diagonal = correct')
    # annotate salient cells; dark text on light cells, white text on dark cells
    for i in range(n):
        for j in range(n):
            v = M[i, j]
            if v >= 0.10:
                ax.text(j, i, f'{v:.2f}'.lstrip('0'), ha='center', va='center',
                        fontsize=6, color='white' if v > 0.55 else '#222')
    # light gridlines to separate cells
    ax.set_xticks(np.arange(-.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-.5, n, 1), minor=True)
    ax.grid(which='minor', color='0.85', lw=0.5)
    ax.tick_params(which='minor', length=0)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout(); savefig('fig_consonant_confusion.png'); plt.show()

print("Top off-diagonal consonant confusions (gold→pred, count):")
for nm, cnt in [('CRF', cc), ('SSL', sc)]:
    errs = sorted(((v, g, p) for (g, p), v in cnt.items() if g != p), reverse=True)
    print(f"  {nm}:", ', '.join(f'{g}→{p}({v})' for v, g, p in errs[:8]))

# %% CRF confusion — permutation-significant patients only ======================
import numpy as np, matplotlib.pyplot as plt
from collections import Counter
import importlib, phon_helpers; importlib.reload(phon_helpers)
from phon_helpers import manner, place, is_cons, aligned_pairs_zip

SIG_PIDS  = ['P22', 'P26', 'P29']     # order-7 permutation p<0.05
CONS_ONLY = True                      # set False to include vowels

pairs = Counter()
for pid in SIG_PIDS:
    for g, p in aligned_pairs_zip(crf_export[pid]):        # CRF 1:1 → zip
        if (not CONS_ONLY) or (is_cons(g) and is_cons(p)):
            pairs[(g, p)] += 1

mord = {'plosive':0,'fricative':1,'nasal':2,'approx':3,'vowel':4}
syms = sorted({c for gp in pairs for c in gp}, key=lambda c:(mord.get(manner(c),9), str(place(c)), c))
idx = {c:i for i,c in enumerate(syms)}; n=len(syms)
M = np.zeros((n,n))
for (g,p),v in pairs.items(): M[idx[g],idx[p]] += v
rsum, csum = M.sum(1,keepdims=True), M.sum(0,keepdims=True)
recall = np.divide(M, rsum, out=np.zeros_like(M), where=rsum>0)
prec   = np.divide(M, csum, out=np.zeros_like(M), where=csum>0)

fig, axes = plt.subplots(1, 2, figsize=(17, 7.5))
for ax, Mx, ttl in [(axes[0],recall,'CRF Recall  P(pred|gold)'),(axes[1],prec,'CRF Precision  P(gold|pred)')]:
    im = ax.imshow(Mx, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(n)); ax.set_xticklabels(syms, fontsize=7, rotation=90)
    ax.set_yticks(range(n)); ax.set_yticklabels(syms, fontsize=7)
    ax.set_xlabel('predicted'); ax.set_ylabel('gold')
    ax.set_title(f'{ttl}\nsignificant patients: {", ".join(SIG_PIDS)}')
    for i in range(n):
        for j in range(n):
            if Mx[i,j] >= 0.10:
                ax.text(j,i,f'{Mx[i,j]:.2f}'.lstrip('0'),ha='center',va='center',
                        fontsize=6,color='white' if Mx[i,j]>0.55 else '#222')
    ax.set_xticks(np.arange(-.5,n,1),minor=True); ax.set_yticks(np.arange(-.5,n,1),minor=True)
    ax.grid(which='minor',color='0.85',lw=0.5); ax.tick_params(which='minor',length=0)
    fig.colorbar(im,ax=ax,fraction=0.046,pad=0.04)
plt.tight_layout(); plt.savefig('report/fig_crf_confusion_sig.png',dpi=150,bbox_inches='tight'); plt.show()

dpos = np.diag(M) > 0
print(f"mean recall diagonal (sig patients): {np.nanmean(np.diag(recall)[dpos]):.3f}")
for c in syms:
    i=idx[c]; r=recall[i,i]; p=prec[i,i]; f1=2*r*p/(r+p) if r+p>0 else 0
    print(f"  {c:<3} R={r:.2f} P={p:.2f} F1={f1:.2f} (n={int(M[i].sum())})")

# %% ABLATION 1 — segmentation: placement + COUNT ==============================
import os, numpy as np
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn_crfsuite import CRF
from scipy.stats import wilcoxon
from run_pipeline import load_mfa_alignments
from extract_features import extractHG
from config import DUTCH_30_PATH
from phon_helpers import needleman_wunsch, edit_distance

cfg = pipeline.config
EEG_SR, WIN, FS = cfg.eeg_sr, cfg.window_length, cfg.frameshift
ORDER    = run_config['stacking_order']
MINSAMP  = int(WIN * EEG_SR) + 1
PADFLOOR = max(MINSAMP, 40)
NPCA     = 50
pids = sorted(crf_results)

def seg_vec(eeg_seg):
    n = eeg_seg.shape[0]
    if n < PADFLOOR: eeg_seg = np.pad(eeg_seg, ((0, PADFLOOR - n), (0, 0)))
    try:
        feat = extractHG(eeg_seg, EEG_SR, windowLength=WIN, frameshift=FS)
    except Exception:
        return None
    if feat is None or feat.shape[0] == 0: return None
    T, C = feat.shape; w = 2 * ORDER + 1
    st = np.zeros((T, C * w), np.float32)
    for t in range(T):
        for k in range(-ORDER, ORDER + 1):
            tt = t + k
            if 0 <= tt < T: st[t, (k + ORDER) * C:(k + ORDER + 1) * C] = feat[tt]
    return st.mean(0)

def patient_eeg(pid):
    raw = np.load(os.path.join(DUTCH_30_PATH, 'raw', f'{pid}_sEEG.npy'))
    pd = pipeline.patient_data.get(pid, {})
    if   'channel_mask' in pd:      raw = raw[:, pd['channel_mask']]
    elif 'included_channels' in pd: raw = raw[:, pd['included_channels']]
    return raw

def sent_spans(pid):
    wd = pipeline.split_result['word_segments_dict'][pid]
    return {i: (s['stim_start_idx'], s['stim_end_idx'])
            for i, s in enumerate(wd['sentence_list']) if isinstance(s, dict) and s.get('text')}

def crf_feat(row): return {f'f{j}': float(v) for j, v in enumerate(row)}

def fit_clf(X, y, grp):
    keep = {c for c, n in Counter(y).items() if n >= 5}
    m = np.isin(y, list(keep)); X, y, grp = X[m], y[m], grp[m]
    sc  = StandardScaler().fit(X)
    pca = PCA(min(NPCA, X.shape[1], X.shape[0])).fit(sc.transform(X))
    Xp  = pca.transform(sc.transform(X))
    seqs, labs = [], []
    for sid in sorted(set(grp)):
        idx = np.where(grp == sid)[0]
        seqs.append([crf_feat(Xp[i]) for i in idx]); labs.append([y[i] for i in idx])
    clf = CRF(algorithm='lbfgs', c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=True)
    clf.fit(seqs, labs)
    return sc, pca, clf

def build_train(pid, mfa, spans, raw, test_sids):
    X, y, grp = [], [], []
    for sid, phs in mfa.items():
        if sid in test_sids or sid not in spans: continue
        a, b = spans[sid]; seeg = raw[a:b]
        for ph in phs:
            s = max(0, int(ph['start_s'] * EEG_SR)); e = min(seeg.shape[0], int(ph['end_s'] * EEG_SR))
            if e <= s: continue
            v = seg_vec(seeg[s:e])
            if v is not None: X.append(v); y.append(ph['phone']); grp.append(sid)
    return np.array(X, np.float32), np.array(y), np.array(grp)

def predict_seq(sc, pca, clf, seeg, intervals):
    vs = []
    for (s, e) in intervals:
        a = max(0, int(s * EEG_SR)); b = min(seeg.shape[0], int(e * EEG_SR))
        v = seg_vec(seeg[a:b]) if b > a else None
        if v is not None: vs.append(v)
    if not vs: return []
    Xp = pca.transform(sc.transform(np.array(vs, np.float32)))
    return clf.predict([[crf_feat(r) for r in Xp]])[0]

def edges_to_intervals(edges): return list(zip(edges[:-1], edges[1:]))

rng = np.random.default_rng(0)
ratios = [0.7, 1.3, 1.7]                          # wrong-count, uniform placement
conds  = ['oracle', 'uniform', 'random'] + [f'unif{r}' for r in ratios]
per = {c: {} for c in conds}
for pid in pids:
    mfa = load_mfa_alignments(pid); spans = sent_spans(pid); raw = patient_eeg(pid)
    test_sids = set(np.asarray(crf_results[pid]['true_sentence_ids']).tolist())
    sc, pca, clf = fit_clf(*build_train(pid, mfa, spans, raw, test_sids))
    eds = {c: 0 for c in conds}; gl = 0
    for sid in sorted(test_sids):
        if sid not in spans or sid not in mfa: continue
        a, b = spans[sid]; seeg = raw[a:b]
        gold = [ph['phone'] for ph in mfa[sid]]; N = len(gold); gl += N
        lo, hi = mfa[sid][0]['start_s'], mfa[sid][-1]['end_s']
        iv = {'oracle':  [(ph['start_s'], ph['end_s']) for ph in mfa[sid]],
              'uniform': edges_to_intervals(np.linspace(lo, hi, N + 1)),
              'random':  edges_to_intervals(np.concatenate([[lo], np.sort(rng.uniform(lo, hi, max(N-1, 0))), [hi]]))}
        for r in ratios:
            M = max(1, int(round(N * r)))
            iv[f'unif{r}'] = edges_to_intervals(np.linspace(lo, hi, M + 1))
        for c in conds:
            eds[c] += edit_distance(gold, predict_seq(sc, pca, clf, seeg, iv[c]))
    for c in conds: per[c][pid] = eds[c] / max(gl, 1)
    print(f"{pid}: " + "  ".join(f"{c}={per[c][pid]:.3f}" for c in conds))

print("\nmean PER:")
for c in conds: print(f"  {c:<9} {np.mean(list(per[c].values())):.3f}")
oa = np.array([per['oracle'][p] for p in pids]); un = np.array([per['uniform'][p] for p in pids])
ov = np.array([per['unif1.7'][p] for p in pids])
print(f"\nplacement (oracle→uniform):   Δ={np.mean(un-oa):+.3f}  p={wilcoxon(oa, un).pvalue:.4g}")
print(f"count/over-gen (uniform→×1.7): Δ={np.mean(ov-un):+.3f}  p={wilcoxon(un, ov).pvalue:.4g}")

# %% ABLATION 2  — phonotactic transition LM in the SSL Viterbi =======
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from phon_helpers import needleman_wunsch, edit_distance, gather_sequences

def viterbi_decode_lm(logp, self_bonus, logT, lm_weight):
    """Self-loop bonus on the diagonal; phonotactic log-prob on switches only."""
    T, K = logp.shape
    if T == 0: return np.zeros(0, np.int32)
    M = lm_weight * logT.copy(); np.fill_diagonal(M, self_bonus)   # stay=bonus, switch=LM
    delta = np.empty((T, K)); bptr = np.empty((T, K), np.int32); delta[0] = logp[0]
    for t in range(1, T):
        cand = delta[t-1][:, None] + M
        bptr[t] = cand.argmax(0); delta[t] = logp[t] + cand.max(0)
    path = np.empty(T, np.int32); path[-1] = delta[-1].argmax()
    for t in range(T-2, -1, -1): path[t] = bptr[t+1, path[t+1]]
    return path

def build_logT(labels, gold_seqs, smooth=1.0):
    idx = {c: i for i, c in enumerate(labels)}; K = len(labels)
    C = np.full((K, K), smooth)
    for s in gold_seqs:
        for a, b in zip(s[:-1], s[1:]):
            if a in idx and b in idx: C[idx[a], idx[b]] += 1
    return np.log(C / C.sum(1, keepdims=True))

def count_ngrams_ge(pred, gold, k):
    if len(pred) < k: return 0
    gset = {tuple(gold[i:i+k]) for i in range(len(gold)-k+1)}
    return sum(tuple(pred[i:i+k]) in gset for i in range(len(pred)-k+1))

def run_ssl_lm(pid, lm_weight):
    ds = datasets[pid]; mfa = {s['sent_idx']: s['mfa'] for s in ds['train'] + ds['test']}
    per = embeddings[pid]; all_real = sorted(per)
    test_ids = set(s['sent_idx'] for s in ds['test'])
    tr = [i for i in all_real if i not in test_ids]
    r = np.random.RandomState(0); r.shuffle(tr)
    nval = max(1, int(len(tr) * VAL_FRAC)); val_ids, fit_ids = set(tr[:nval]), set(tr[nval:])
    def bset(ids):
        X, y = [], []
        for sid in ids:
            if sid not in per: continue
            emb = per[sid]; T = emb.shape[0]
            for ph in mfa[sid]:
                ks = max(0, time_to_frame(ph['start_s'])); ke = min(T-1, time_to_frame(ph['end_s']))
                if not (max(MN_FRAMES,1) <= ke-ks+1 <= MX_FRAMES): continue
                X.append(emb[ks:ke+1].mean(0)); y.append(ph['phone'])
        return np.array(X), np.array(y)
    Xf, yf = bset(fit_ids)
    if len(Xf) < 50: return None
    scf = StandardScaler().fit(Xf)
    clff = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto').fit(scf.transform(Xf), yf)
    fc = set(yf); vlp, vt = [], 0
    for sid in val_ids:
        if sid not in per: continue
        vlp.append(smooth_cols(clff.predict_log_proba(scf.transform(per[sid])), SMOOTH_LOGP_W))
        vt += sum(1 for ph in mfa[sid] if ph['phone'] in fc)
    if not vlp: return None
    bonus = auto_tune_bonus(vlp, int(vt * TARGET_RATIO), MIN_PRED_FRAMES)
    Xt, yt = bset(set(all_real) - test_ids); trc = set(yt)
    sca = StandardScaler().fit(Xt)
    clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto').fit(sca.transform(Xt), yt)
    labels = list(clf.classes_)
    logT = build_logT(labels, [[ph['phone'] for ph in mfa[s]] for s in (set(all_real)-test_ids) if s in mfa])
    preds, psid, trues, tsid = [], [], [], []
    for sid in test_ids:
        if sid not in per: continue
        emb = per[sid]; T = emb.shape[0]
        lp = smooth_cols(clf.predict_log_proba(sca.transform(emb)), SMOOTH_LOGP_W)
        path = viterbi_decode_lm(lp, bonus, logT, lm_weight)
        i = 0
        while i < T:
            ci = path[i]; j = i+1
            while j < T and path[j] == ci: j += 1
            if j-i >= MIN_PRED_FRAMES: preds.append(labels[ci]); psid.append(sid)
            i = j
        for ph in mfa[sid]:
            if ph['phone'] in trc: trues.append(ph['phone']); tsid.append(sid)
    return {'true_labels': np.array(trues), 'predictions': np.array(preds),
            'true_sentence_ids': np.array(tsid), 'pred_sentence_ids': np.array(psid)}

def full_per(o):
    gp, pp = gather_sequences(o)
    return sum(edit_distance(gp[s], pp.get(s, [])) for s in gp) / max(sum(len(gp[s]) for s in gp), 1)
def n3(o):
    gp, pp = gather_sequences(o); return sum(count_ngrams_ge(pp.get(s, []), gp[s], 3) for s in gp)

weights = [0.0, 1.0, 3.0, 5.0]
agg = {w: {'per': [], 'n3': []} for w in weights}
for pid in sorted(embeddings):
    row = f"{pid:<5}"
    for w in weights:
        o = run_ssl_lm(pid, w)
        if o is None: row += "    -      "; continue
        p, c = full_per(o), n3(o); agg[w]['per'].append(p); agg[w]['n3'].append(c)
        row += f"  {p:.2f}/{c:>3}"
    print(row)
print("\nmean PER / Σn≥3 by lm_weight:")
for w in weights:
    print(f"  lm={w}:  PER={np.mean(agg[w]['per']):.3f}  Σn≥3={np.mean(agg[w]['n3']):.1f}")
