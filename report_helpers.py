# report_helpers.py — plotting/analysis functions for the thesis report notebook.
# Import-safe: only imports + function defs, no top-level execution.
# Pairs with report_notebook.py's get_feats():
#     from report_helpers import phoneme_separability, decision_region_plot
#     phoneme_separability(*get_feats('crf', 'P22'), title='CRF P22')   # X, y, sid -> X, y, grp
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from scipy.stats import ttest_rel, t as tdist
from phon_helpers import (is_cons, vowel_length, vowel_place, cv, manner, place, voicing,
                          subs_position_zip, subs_nw, aligned_pairs_zip, aligned_pairs_nw, feature_z,
                          needleman_wunsch, gather_sequences)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, silhouette_score, matthews_corrcoef

C_CRF, C_SSL = '#e08a2b', '#3b6fb0'


def phoneme_separability(X, y, grp=None, title='', top_k=12, min_count=25, n_pca=50, n_splits=5):
    """3-panel honest separability: centroid-cosine matrix, held-out LDA(2) projection,
    cross-validated per-class one-vs-rest AUC. Pass grp=sentence ids to prevent leakage."""
    X = np.asarray(X); y = np.asarray(y)
    keep = [c for c, n in Counter(y).items() if n >= min_count]
    m = np.isin(y, keep); X, y = X[m], y[m]
    grp = np.asarray(grp)[m] if grp is not None else None
    classes = sorted(set(y)); cidx = {c: i for i, c in enumerate(classes)}

    if grp is not None:
        splits = list(GroupKFold(n_splits).split(X, y, grp))     # prevents same-sentence leakage
    else:
        print("  [warn] no grp -> StratifiedKFold; same-sentence leakage NOT prevented")
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

    Xs = StandardScaler().fit_transform(X)
    Xp = PCA(n_components=min(n_pca, Xs.shape[1], Xs.shape[0] - 1)).fit_transform(Xs)
    sil = silhouette_score(Xp, y)
    C = np.array([Xs[y == c].mean(0) for c in classes])
    Cn = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-9)
    S = Cn @ Cn.T
    print(f"{title}:  n={len(y)} classes={len(classes)} | macro OvR AUC={macro:.3f}  "
          f"max={max(ovr.values()):.3f}  #>0.7={sum(a > 0.7 for a in ovr.values())} | "
          f"silhouette={sil:.3f}")

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
    ax[1].set_title(f"{title}\nHELD-OUT LDA(2)", fontsize=10); ax[1].legend(fontsize=7, ncol=2, markerscale=2)
    ph = sorted(ovr, key=lambda c: -ovr[c])
    ax[2].bar(range(len(ph)), [ovr[c] for c in ph], color='#3b6fb0')
    ax[2].axhline(0.7, color='r', ls='--', lw=0.8); ax[2].axhline(0.5, color='k', ls=':', lw=0.8)
    ax[2].set_xticks(range(len(ph))); ax[2].set_xticklabels(ph, rotation=90, fontsize=7)
    ax[2].set_ylabel('cross-validated OvR AUC'); ax[2].set_title(f"{title}\nper-phoneme separability", fontsize=10)
    plt.tight_layout(); plt.show()
    return dict(macro_auc=float(macro), max_auc=float(max(ovr.values())),
                n_above_0p7=int(sum(a > 0.7 for a in ovr.values())),
                silhouette=float(sil), per_class=ovr, n=len(y), n_classes=len(classes))


def hg_amplitude_plot(feats, title='Per-phoneme high-gamma amplitude',
                      min_per_patient=10, min_patients=8):
    """Patient-robust per-phoneme HG amplitude (the single descriptive figure).
    Per patient: per-phoneme mean over NON-ZERO feature entries (excludes zero-padding),
    z-scored within patient; then MEDIAN of per-patient means across patients (+/- IQR).
    Robust to single-patient artifacts and per-patient scale; no train/test split needed.
    `feats` = {pid: {'X':(n,d), 'y':str, ...}} (the crf_feats dict)."""
    perpat = defaultdict(dict)
    for pid, d in feats.items():
        X = np.asarray(d['X'], np.float32); y = np.asarray(d['y'])
        amp = [(y[i], (lambda nz: float(nz.mean()) if nz.size else np.nan)(X[i][X[i] > 0]))
               for i in range(len(X))]
        v = np.array([a for _, a in amp]); ok = np.isfinite(v)
        mu, sd = v[ok].mean(), v[ok].std() + 1e-9
        byp = defaultdict(list)
        for p, a in amp:
            if np.isfinite(a): byp[p].append((a - mu) / sd)        # z within patient
        for p, vals in byp.items():
            if len(vals) >= min_per_patient: perpat[p][pid] = float(np.mean(vals))
    phs = [p for p in perpat if len(perpat[p]) >= min_patients]
    arr = {p: np.array(list(perpat[p].values())) for p in phs}
    phs.sort(key=lambda p: -np.median(arr[p]))
    med = [np.median(arr[p]) for p in phs]
    lo  = [np.median(arr[p]) - np.percentile(arr[p], 25) for p in phs]
    hi  = [np.percentile(arr[p], 75) - np.median(arr[p]) for p in phs]
    def cat(p): return '#d62728' if vowel_length(p) == 'long' else ('#1f77b4' if not is_cons(p) else '0.6')
    plt.figure(figsize=(15, 4.5))
    plt.bar(range(len(phs)), med, yerr=[lo, hi], color=[cat(p) for p in phs],
            capsize=2, error_kw={'lw': 0.8})
    plt.axhline(0, color='k', lw=0.6); plt.xticks(range(len(phs)), phs, fontsize=8)
    plt.ylabel('HG amplitude\n(z within patient; median of patient means ± IQR)')
    plt.title(f'{title} — red=long vowel, blue=short vowel, grey=consonant')
    plt.tight_layout(); plt.show()
    return {p: float(np.median(arr[p])) for p in phs}


def decision_region_plot(X, y, grp=None, title='', classes=('aː', 'eː', 'iː', 'oː', 'uː'),
                         n_pca_pre=50, min_count=15, proj='lda', region='lda', ax=None):
    """Decision regions over an honest 2D placement (held-out test points).
    proj='pca' = distance-faithful but conservative; proj='lda' = held-out discriminant
    (use only when separability is independently validated). region in {'lda','knn'}.
    Pass ax=... to draw into an existing subplot (for side-by-side); else makes a small figure."""
    X = np.asarray(X); y = np.asarray(y); grp = None if grp is None else np.asarray(grp)
    classes = [c for c in classes if (y == c).sum() >= min_count]
    m = np.isin(y, classes); X, y = X[m], y[m]
    grp = grp[m] if grp is not None else None
    if grp is not None:
        tr, te = next(iter(GroupKFold(5).split(X, y, grp)))
    else:
        tr, te = next(iter(StratifiedKFold(5, shuffle=True, random_state=0).split(X, y)))
    sc = StandardScaler().fit(X[tr]); Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
    if Xtr.shape[1] > n_pca_pre:
        pre = PCA(n_pca_pre, random_state=0).fit(Xtr); Xtr, Xte = pre.transform(Xtr), pre.transform(Xte)
    p2 = (PCA(2, random_state=0) if proj == 'pca'
          else LinearDiscriminantAnalysis(n_components=2)).fit(Xtr, y[tr])
    Ztr, Zte = p2.transform(Xtr), p2.transform(Xte)
    reg = (LinearDiscriminantAnalysis() if region == 'lda' else KNeighborsClassifier(25)).fit(Ztr, y[tr])
    cls = list(reg.classes_); ci = {c: i for i, c in enumerate(cls)}; cmap = plt.cm.tab10
    pad = 1.0
    x0, x1 = Zte[:, 0].min() - pad, Zte[:, 0].max() + pad
    y0, y1 = Zte[:, 1].min() - pad, Zte[:, 1].max() + pad
    xx, yy = np.meshgrid(np.linspace(x0, x1, 250), np.linspace(y0, y1, 250))
    Gi = np.array([ci[g] for g in reg.predict(np.c_[xx.ravel(), yy.ravel()])]).reshape(xx.shape)
    lev = np.arange(len(cls) + 1) - 0.5
    standalone = ax is None
    if standalone:
        _, ax = plt.subplots(figsize=(5, 4.6))
    ax.contourf(xx, yy, Gi, levels=lev, colors=[cmap(i) for i in range(len(cls))], alpha=0.16)
    ax.contour(xx, yy, Gi, levels=lev, colors='k', linewidths=0.5, alpha=0.4)
    for c in cls:
        mm = y[te] == c
        ax.scatter(Zte[mm, 0], Zte[mm, 1], s=11, color=cmap(ci[c]), edgecolor='w', linewidth=0.25, label=str(c))
    lab = 'LD' if proj == 'lda' else 'PC'
    ax.legend(title='class', fontsize=7, title_fontsize=8, loc='best')
    ax.set_xlabel(f'{lab}1', fontsize=9); ax.set_ylabel(f'{lab}2', fontsize=9)
    ax.set_title(f"{title}\n{'held-out LDA(2)' if proj=='lda' else 'PCA(2)'} + regions", fontsize=9)
    if standalone:
        plt.tight_layout(); plt.show()


def decision_region_pair(get_feats, pid, proj='lda', figsize=(10, 4.6), savepath=None):
    """Two decision-region panels in one row (CRF | SSL) for one patient.
    `get_feats` is the report notebook's get_feats(model, pid) -> (X, y, sid)."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    decision_region_plot(*get_feats('crf', pid), title=f'CRF {pid}', proj=proj, ax=axes[0])
    decision_region_plot(*get_feats('ssl', pid), title=f'SSL {pid}', proj=proj, ax=axes[1])
    plt.tight_layout()
    if savepath: plt.savefig(savepath, dpi=200, bbox_inches='tight')
    plt.show()


# ── error-structure (paired CRF-vs-SSL slopegraphs) ──────────────────────────
def _slope(ax, vc, vs, title, ylabel, pids, fs=8, sig_c=None, sig_s=None):
    """One paired-slopegraph panel: per-patient CRF vs SSL, paired t-test + CI in title.
    sig_c/sig_s: optional per-patient boolean masks; True -> ring that dot (black edge =
    per-patient AUC significantly above chance)."""
    vc, vs = np.asarray(vc), np.asarray(vs)
    ok = np.isfinite(vc) & np.isfinite(vs); n = int(ok.sum()); d = vc[ok] - vs[ok]
    P = ttest_rel(vc[ok], vs[ok])[1] if n >= 3 else np.nan
    md = d.mean() if n else np.nan
    half = tdist.ppf(0.975, n - 1) * d.std(ddof=1) / np.sqrt(n) if n >= 2 else np.nan
    for i in np.where(ok)[0]:
        ax.plot([0, 1], [vc[i], vs[i]], color='0.8', lw=0.8, zorder=0)
        ax.annotate(pids[i], (0, vc[i]), xytext=(-7, 0), textcoords='offset points',
                    ha='right', va='center', fontsize=6.5, color=C_CRF)
        ax.annotate(pids[i], (1, vs[i]), xytext=(7, 0), textcoords='offset points',
                    ha='left', va='center', fontsize=6.5, color=C_SSL)
    oki = np.where(ok)[0]
    ec_c = ['black' if (sig_c is not None and bool(np.asarray(sig_c)[i])) else 'none' for i in oki]
    ec_s = ['black' if (sig_s is not None and bool(np.asarray(sig_s)[i])) else 'none' for i in oki]
    ax.scatter(np.zeros(n), vc[ok], color=C_CRF, s=45, zorder=3, edgecolors=ec_c, linewidths=1.4)
    ax.scatter(np.ones(n),  vs[ok], color=C_SSL, s=45, zorder=3, edgecolors=ec_s, linewidths=1.4)
    ax.set_xlim(-0.6, 1.6); ax.set_xticks([0, 1]); ax.set_xticklabels(['CRF', 'SSL'])
    ax.set_ylabel(ylabel, fontsize=fs)
    title_disp = (title[:1].upper() + title[1:]).replace(' ', r'\ ')
    star = ' *' if (np.isfinite(P) and P < 0.05) else ''
    ax.set_title(f"$\\bf{{{title_disp}}}${star} (n={n})\nCRF={np.nanmean(vc):.3f}  SSL={np.nanmean(vs):.3f}\n"
                 f"$\\Delta$={md:+.3f}  CI[{md-half:+.3f},{md+half:+.3f}]  p={P:.3g}", fontsize=fs)


def feature_panel(crf_export, ssl_results, fn, subset='vow', title='', ylabel='preservation rate',
                  pids=None, ax=None, savepath=None, n_perm=1000):
    """Standalone paired CRF-vs-SSL slopegraph for ONE feature's absolute preservation rate.
    fn: phone -> category (or None if N/A); subset in {'cons','vow','all'}.
    Black ring = this patient's preservation is significantly above chance (per-model
    label-permutation p<0.05). e.g. duration:
    feature_panel(crf_export, ssl_results, vowel_length, 'vow', 'vowel duration').
    Returns (per-patient CRF rates, SSL rates)."""
    if pids is None: pids = sorted(set(crf_export) & set(ssl_results))

    def filt(subs):
        if subset == 'cons': return [(g, p) for g, p in subs if is_cons(g) and is_cons(p)]
        if subset == 'vow':  return [(g, p) for g, p in subs if not is_cons(g) and not is_cons(p)]
        return subs

    def rate_p(subs, k=5):
        pr = [(g, p) for g, p in subs if fn(g) is not None and fn(p) is not None]
        if len(pr) < k: return np.nan, 1.0
        rate = np.mean([fn(g) == fn(p) for g, p in pr])
        z, p, _ = feature_z(pr, fn, n_perm=n_perm)
        return rate, (p if (np.isfinite(z) and z > 0) else 1.0)   # ring only above chance

    cvp_ = [rate_p(filt(subs_position_zip(crf_export[pid]))) for pid in pids]
    svp_ = [rate_p(filt(subs_nw(ssl_results[pid])))          for pid in pids]
    vc = np.array([r for r, _ in cvp_]); pc = np.array([p for _, p in cvp_])
    vs = np.array([r for r, _ in svp_]); ps = np.array([p for _, p in svp_])
    standalone = ax is None
    if standalone:
        _, ax = plt.subplots(figsize=(5, 5.5))
    _slope(ax, vc, vs, title or 'feature', ylabel, pids,
           sig_c=np.nan_to_num(pc, nan=1.0) < 0.05, sig_s=np.nan_to_num(ps, nan=1.0) < 0.05)
    ok = np.isfinite(vc) & np.isfinite(vs)
    print(f"{title}: CRF>SSL in {int((vc[ok] > vs[ok]).sum())}/{int(ok.sum())} patients")
    if standalone:
        plt.tight_layout()
        if savepath: plt.savefig(savepath, dpi=200, bbox_inches='tight')
        plt.show()
    return vc, vs


def _prep_feat(d, fn, min_count=20):
    """Prepare (X, lab, sid, cats) for one patient/feature, or None if too small."""
    X = np.asarray(d['X'], np.float32); y = np.asarray(d['y']); sid = np.asarray(d['sid'])
    lab = np.array([fn(p) for p in y], dtype=object)
    m = np.array([l is not None for l in lab]); X, lab, sid = X[m], lab[m].astype(str), sid[m]
    keep = [c for c, nc in Counter(lab).items() if nc >= min_count]
    mm = np.isin(lab, keep); X, lab, sid = X[mm], lab[mm], sid[mm]
    cats = sorted(set(lab))
    if len(cats) < 2 or len(set(sid)) < 5: return None
    return X, lab, sid, cats


def _cv_proba(X, lab, sid, cats, n_pca=50):
    """Out-of-fold predicted probabilities (n, len(cats)); None on failure."""
    cidx = {c: i for i, c in enumerate(cats)}
    steps = [StandardScaler()]
    if X.shape[1] > n_pca: steps.append(PCA(n_pca, random_state=0))
    steps.append(LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'))
    clf = make_pipeline(*steps)
    proba = np.zeros((len(lab), len(cats)))
    try:
        for tr, te in GroupKFold(5).split(X, lab, sid):
            clf.fit(X[tr], lab[tr]); p = clf.predict_proba(X[te])
            for j, c in enumerate(clf.classes_):
                if c in cidx: proba[te, cidx[c]] = p[:, j]
    except Exception:
        return None
    return proba


def _auc_from_proba(lab, proba, cats):
    lab = np.asarray(lab)
    if len(cats) == 2:
        return roc_auc_score((lab == cats[1]).astype(int), proba[:, 1])
    return roc_auc_score(lab, proba, multi_class='ovo', average='macro', labels=cats)


def _feat_auc(d, fn, n_pca=50, min_count=20):
    """Cross-validated AUC of decoding feature fn from one patient's representation.
    Binary -> ROC AUC; >2 cats -> macro one-vs-one AUC. None category -> excluded."""
    arrs = _prep_feat(d, fn, min_count)
    if arrs is None: return np.nan
    X, lab, sid, cats = arrs
    proba = _cv_proba(X, lab, sid, cats, n_pca)
    if proba is None: return np.nan
    return _auc_from_proba(lab, proba, cats)


def _feat_auc_sig(d, fn, n_pca=50, min_count=20, n_perm=200, seed=0):
    """Like _feat_auc but also returns a per-patient label-permutation p-value for
    AUC > chance. Permutes labels against the FIXED out-of-fold scores (fast, no refit).
    Returns (auc, p); (nan, nan) if too small."""
    arrs = _prep_feat(d, fn, min_count)
    if arrs is None: return np.nan, np.nan
    X, lab, sid, cats = arrs
    proba = _cv_proba(X, lab, sid, cats, n_pca)
    if proba is None: return np.nan, np.nan
    obs = _auc_from_proba(lab, proba, cats)
    rng = np.random.default_rng(seed)
    null = np.array([_auc_from_proba(rng.permutation(lab), proba, cats) for _ in range(n_perm)])
    p = (1 + int(np.sum(null >= obs))) / (n_perm + 1)
    return obs, p


def feature_separability_panel(crf_feats, ssl_feats, fn, title='', ylabel='separability AUC',
                               n_pca=50, min_count=20, pids=None, ax=None, savepath=None, n_perm=200):
    """Paired CRF-vs-SSL slopegraph of how separable a feature is FROM THE REPRESENTATION
    (cross-validated AUC), per patient. fn: phone -> category, or None to exclude. Chance=0.5.
    e.g. duration: feature_separability_panel(crf_feats, ssl_feats, vowel_length, 'vowel duration')."""
    if pids is None: pids = sorted(set(crf_feats) & set(ssl_feats))
    ac  = [_feat_auc_sig(crf_feats[pid], fn, n_pca, min_count, n_perm) for pid in pids]
    asl = [_feat_auc_sig(ssl_feats[pid], fn, n_pca, min_count, n_perm) for pid in pids]
    vc = np.array([a for a, _ in ac]);  pc = np.array([p for _, p in ac])
    vs = np.array([a for a, _ in asl]); ps = np.array([p for _, p in asl])
    standalone = ax is None
    if standalone:
        _, ax = plt.subplots(figsize=(5, 5.5))
    _slope(ax, vc, vs, title or 'feature', ylabel, pids,
           sig_c=np.nan_to_num(pc, nan=1.0) < 0.05, sig_s=np.nan_to_num(ps, nan=1.0) < 0.05)
    ax.axhline(0.5, color='k', ls=':', lw=0.8)
    ok = np.isfinite(vc) & np.isfinite(vs)
    print(f"{title} (separability): CRF>SSL in {int((vc[ok] > vs[ok]).sum())}/{int(ok.sum())} patients")
    if standalone:
        plt.tight_layout()
        if savepath: plt.savefig(savepath, dpi=200, bbox_inches='tight')
        plt.show()
    return vc, vs


def feature_separability_grid(crf_feats, ssl_feats, pids=None, savepath=None, n_perm=200):
    """Per-patient feature SEPARABILITY (cross-validated AUC) by phonetic feature, CRF vs SSL.
    Top row: consonant features (manner, place, voicing). Bottom row: vowel features
    (duration, backness). Each panel title states the feature category. Chance = 0.5.
    No C/V or all-phonemes panels."""
    if pids is None: pids = sorted(set(crf_feats) & set(ssl_feats))
    cons_manner = lambda p: manner(p) if is_cons(p) else None
    panels = [(0, 0, 'manner',   'consonant', cons_manner),
              (0, 1, 'place',    'consonant', place),
              (0, 2, 'voicing',  'consonant', voicing),
              (1, 0, 'duration', 'vowel',     vowel_length),
              (1, 1, 'backness', 'vowel',     vowel_place)]
    fig, axes = plt.subplots(2, 3, figsize=(14, 9)); used = set()
    for r, c, name, cat, fn in panels:
        ac  = [_feat_auc_sig(crf_feats[pid], fn, n_perm=n_perm) for pid in pids]
        asl = [_feat_auc_sig(ssl_feats[pid], fn, n_perm=n_perm) for pid in pids]
        vc = np.array([a for a, _ in ac]);  pc = np.array([p for _, p in ac])
        vs = np.array([a for a, _ in asl]); ps = np.array([p for _, p in asl])
        _slope(axes[r, c], vc, vs, f'{name} ({cat})', 'AUC', pids,
               sig_c=np.nan_to_num(pc, nan=1.0) < 0.05, sig_s=np.nan_to_num(ps, nan=1.0) < 0.05)
        axes[r, c].axhline(0.5, color='k', ls=':', lw=0.8); used.add((r, c))
    for r in range(2):
        for c in range(3):
            if (r, c) not in used: axes[r, c].axis('off')
    fig.suptitle('Feature separability from the representation '
                 '(per-patient cross-validated AUC, chance $=0.5$)', fontsize=13, y=0.99)
    fig.text(0.5, 0.005, 'AUC of decoding each phonetic feature from the per-phoneme representation '
             '(PCA-50, GroupKFold by sentence; binary $\\rightarrow$ ROC AUC, multiclass $\\rightarrow$ '
             'macro OvO). Top: consonant features. Bottom: vowel features. '
             'Ringed markers: per-patient AUC above chance (label-permutation $p<0.05$).',
             ha='center', fontsize=8, style='italic', bbox=dict(boxstyle='round', fc='#fff3cd', ec='#856404'))
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    if savepath: plt.savefig(savepath, dpi=200, bbox_inches='tight')
    plt.show()


def separability_structure_grid(crf_feats, ssl_feats, pids=None, savepath=None):
    """3x3 grid mirroring error_structure_grid, but FROM THE REPRESENTATION (cross-validated AUC):
    row1 = C/V + all-phonemes (overall), row2 = consonant features (manner/place/voicing),
    row3 = vowel features (duration/backness). Each panel = per-patient paired CRF-vs-SSL AUC."""
    if pids is None: pids = sorted(set(crf_feats) & set(ssl_feats))
    cons_manner = lambda p: manner(p) if is_cons(p) else None
    panels = [(0, 0, 'C/V', cv),
              (0, 1, 'all phonemes', lambda p: p),
              (1, 0, 'manner', cons_manner),
              (1, 1, 'place', place),
              (1, 2, 'voicing', voicing),
              (2, 0, 'duration', vowel_length),
              (2, 1, 'backness', vowel_place)]
    fig, axes = plt.subplots(3, 3, figsize=(14, 13)); used = set()
    for r, c, title, fn in panels:
        vc = np.array([_feat_auc(crf_feats[pid], fn) for pid in pids])
        vs = np.array([_feat_auc(ssl_feats[pid], fn) for pid in pids])
        _slope(axes[r, c], vc, vs, title, 'AUC', pids)
        axes[r, c].axhline(0.5, color='k', ls=':', lw=0.8); used.add((r, c))
    for r in range(3):
        for c in range(3):
            if (r, c) not in used: axes[r, c].axis('off')
    fig.suptitle('Feature SEPARABILITY from representation (cross-validated AUC, chance=0.5): '
                 'C/V & all-phonemes (row 1), consonant features (row 2), vowel features (row 3)',
                 fontsize=12, y=0.997)
    fig.text(0.5, 0.005, 'Per-patient cross-validated AUC of decoding each feature from the per-phoneme '
             'representation (GroupKFold by sentence). Binary -> ROC AUC; multiclass -> macro OvO. '
             'Contrast with the ERROR-structure grid (same layout, decoded outputs).',
             ha='center', fontsize=8, style='italic', bbox=dict(boxstyle='round', fc='#fff3cd', ec='#856404'))
    plt.tight_layout(rect=[0, 0.02, 1, 0.98])
    if savepath: plt.savefig(savepath, dpi=200, bbox_inches='tight')
    plt.show()


def error_structure_grid(crf_export, ssl_results, pids=None, savepath=None, n_perm=1000):
    """3x3 grid: row1 = C/V (preservation among errors, MCC separation over all positions),
    row2 = consonant features (manner/place/voicing), row3 = vowel features (duration/backness).
    Each panel = per-patient paired CRF-vs-SSL absolute rate (no shuffle baseline for the cross-
    model comparison). Black ring = that patient's preservation is significantly above chance
    (per-model label-permutation p<0.05); the MCC panel is not ringed."""
    if pids is None: pids = sorted(set(crf_export) & set(ssl_results))

    def cv_mcc_gapped(out, crf):
        # gap-penalized C/V Matthews corr: SSL's NW insertions/deletions count as a 'gap'
        # class, so its over-generation is penalised (the raw aligned-only MCC unfairly
        # favours the free-running model). CRF is 1:1, so it has no gaps.
        if crf:
            labs = [(cv(g), cv(p)) for g, p in aligned_pairs_zip(out)]
        else:
            gp, pp = gather_sequences(out); labs = []
            for sid in set(gp) | set(pp):
                for g, p in needleman_wunsch(gp.get(sid, []), pp.get(sid, [])):
                    labs.append((cv(g) if g is not None else 'gap',
                                 cv(p) if p is not None else 'gap'))
        if len(labs) < 5: return np.nan
        gl = [a for a, _ in labs]; pl = [b for _, b in labs]
        return matthews_corrcoef(gl, pl) if (len(set(gl)) > 1 and len(set(pl)) > 1) else np.nan
    def pres_p(subs, fn, k=5):
        pr = [(g, p) for g, p in subs if fn(g) is not None and fn(p) is not None]
        if len(pr) < k: return np.nan, 1.0
        rate = np.mean([fn(g) == fn(p) for g, p in pr])
        z, p, _ = feature_z(pr, fn, n_perm=n_perm)
        return rate, (p if (np.isfinite(z) and z > 0) else 1.0)
    def csub(pid, vowel):
        f = (lambda g, p: (not is_cons(g)) and (not is_cons(p))) if vowel else (lambda g, p: is_cons(g) and is_cons(p))
        return ([gp for gp in subs_position_zip(crf_export[pid]) if f(*gp)],
                [gp for gp in subs_nw(ssl_results[pid]) if f(*gp)])

    def cvp(pid):
        return pres_p(subs_position_zip(crf_export[pid]), cv), pres_p(subs_nw(ssl_results[pid]), cv)
    def cvm(pid):
        return ((cv_mcc_gapped(crf_export[pid], True), 1.0),
                (cv_mcc_gapped(ssl_results[pid], False), 1.0))
    def cons_fn(fn):
        def f(pid):
            sc, ss = csub(pid, False); return pres_p(sc, fn), pres_p(ss, fn)
        return f
    def vow_fn(fn):
        def f(pid):
            sc, ss = csub(pid, True); return pres_p(sc, fn), pres_p(ss, fn)
        return f

    panels = [(0, 0, 'consonant/vowel preservation', 'frac errors in-class', cvp),
              (0, 1, 'consonant/vowel separation (gap-penal. Matthews)', 'Matthews corr', cvm),
              (1, 0, 'manner (consonant)', 'pres. rate', cons_fn(manner)),
              (1, 1, 'place (consonant)', 'pres. rate', cons_fn(place)),
              (1, 2, 'voicing (consonant)', 'pres. rate', cons_fn(voicing)),
              (2, 0, 'duration (vowel)', 'pres. rate', vow_fn(vowel_length)),
              (2, 1, 'backness (vowel)', 'pres. rate', vow_fn(vowel_place))]
    fig, axes = plt.subplots(3, 3, figsize=(14, 13)); used = set()
    for r, c, title, ylabel, fn in panels:
        res = [fn(pid) for pid in pids]
        vc = np.array([x[0][0] for x in res]); pc = np.array([x[0][1] for x in res])
        vs = np.array([x[1][0] for x in res]); ps = np.array([x[1][1] for x in res])
        _slope(axes[r, c], vc, vs, title, ylabel, pids,
               sig_c=np.nan_to_num(pc, nan=1.0) < 0.05, sig_s=np.nan_to_num(ps, nan=1.0) < 0.05)
        used.add((r, c))
    for r in range(3):
        for c in range(3):
            if (r, c) not in used: axes[r, c].axis('off')
    fig.suptitle('Error structure: consonant/vowel (row 1), consonant features (row 2), '
                 'vowel features (row 3) — per-patient absolute rates, paired', fontsize=13, y=0.997)
    fig.text(0.5, 0.005,
             'Absolute within-class preservation (no shuffle baseline for the CRF-vs-SSL comparison). '
             'Black ring: preservation above chance (per-model permutation $p<0.05$; MCC panel not ringed). '
             'Matthews-correlation panel is gap-penalized (SSL NW insertions/deletions count as a mismatch), '
             'so it is not inflated by the free-running over-generation.',
             ha='center', fontsize=8, style='italic',
             bbox=dict(boxstyle='round', fc='#fff3cd', ec='#856404'))
    plt.tight_layout(rect=[0, 0.02, 1, 0.98])
    if savepath: plt.savefig(savepath, dpi=200, bbox_inches='tight')
    plt.show()
