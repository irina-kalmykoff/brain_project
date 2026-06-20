# phon_helpers.py — small, import-safe helpers shared by the CRF and SSL notebooks
# (vendored from LDA_on_frames_clean, which isn't importable due to a top-level driver)
import numpy as np
from collections import defaultdict, Counter


# ── sequence alignment ───────────────────────────────────────────────
def needleman_wunsch(gold, pred, match=1, mismatch=-1, gap=-1):
    """Global alignment. Returns list of (g, p); None = gap."""
    n, m = len(gold), len(pred)
    if n == 0: return [(None, p) for p in pred]
    if m == 0: return [(g, None) for g in gold]
    S = np.zeros((n + 1, m + 1), dtype=np.float32)
    S[:, 0] = np.arange(n + 1) * gap; S[0, :] = np.arange(m + 1) * gap
    BT = np.zeros((n + 1, m + 1), dtype=np.int8)
    BT[:, 0] = 1; BT[0, :] = 2; BT[0, 0] = 0
    for i in range(1, n + 1):
        gi = gold[i - 1]
        for j in range(1, m + 1):
            d = S[i - 1, j - 1] + (match if gi == pred[j - 1] else mismatch)
            u = S[i - 1, j] + gap; l = S[i, j - 1] + gap
            if d >= u and d >= l: S[i, j] = d; BT[i, j] = 0
            elif u >= l:          S[i, j] = u; BT[i, j] = 1
            else:                 S[i, j] = l; BT[i, j] = 2
    aligned, i, j = [], n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and BT[i, j] == 0:
            aligned.append((gold[i - 1], pred[j - 1])); i -= 1; j -= 1
        elif i > 0 and BT[i, j] == 1:
            aligned.append((gold[i - 1], None)); i -= 1
        else:
            aligned.append((None, pred[j - 1])); j -= 1
    return list(reversed(aligned))


def edit_distance(a, b):
    """Levenshtein distance (scalar) between two sequences."""
    n, m = len(a), len(b)
    if n == 0: return m
    if m == 0: return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        ai = a[i - 1]
        for j in range(1, m + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ai != b[j - 1]))
        prev = cur
    return prev[m]


def gather_sequences(out):
    """Group flat predictions/gold by sentence_id, preserving order."""
    gold_per = defaultdict(list)
    for lbl, sid in zip(out['true_labels'], out['true_sentence_ids']):
        gold_per[int(sid)].append(lbl)
    pred_per = defaultdict(list)
    for lbl, sid in zip(out['predictions'], out['pred_sentence_ids']):
        pred_per[int(sid)].append(lbl)
    return gold_per, pred_per


# ── Dutch phonetic features ──────────────────────────────────────────
PLOSIVE   = {'p', 'b', 't', 'd', 'k', 'g', 'c'}          # c = voiceless palatal plosive
FRIC      = {'f', 'v', 's', 'z', 'x', 'ɣ', 'h', 'ʃ'}     # ʃ = voiceless postalveolar fricative
NASAL     = {'m', 'n', 'ŋ'}
LIQGLIDE  = {'l', 'r', 'j', 'w', 'ʋ', 'ɥ'}
_CONS     = PLOSIVE | FRIC | NASAL | LIQGLIDE
VOICED    = {'b', 'd', 'g', 'v', 'z', 'ɣ', 'm', 'n', 'ŋ', 'l', 'r', 'j', 'w', 'ʋ', 'ɥ'}
VOICELESS = {'p', 't', 'k', 'f', 's', 'x', 'h', 'c', 'ʃ'}

def is_cons(p): return p in _CONS
def cv(p):
    """Coarse class: 'C' consonant, 'V' vowel. Always defined (never None)."""
    return 'C' if p in _CONS else 'V'
def manner(p):
    if p in PLOSIVE:  return 'plosive'
    if p in FRIC:     return 'fricative'
    if p in NASAL:    return 'nasal'
    if p in LIQGLIDE: return 'approx'
    return 'vowel'
def voicing(p):
    """'voiced'/'voiceless' for consonants, None for vowels (not applicable)."""
    if p in VOICED:    return 'voiced'
    if p in VOICELESS: return 'voiceless'
    return None
def base_vowel(p): return p.replace('ː', '')

# place of articulation (Dutch consonants; None = unknown / vowel)
_PLACE = {'p':'lab','b':'lab','m':'lab',
          'f':'labden','v':'labden','ʋ':'labden','w':'labden',
          't':'alv','d':'alv','s':'alv','z':'alv','n':'alv','l':'alv','r':'alv',
          'ʃ':'postalv',
          'k':'vel','g':'vel','x':'vel','ɣ':'vel','ŋ':'vel',
          'j':'pal','ɥ':'pal','c':'pal','h':'glot'}
def place(p): return _PLACE.get(p)


# ── Dutch vowel features (None for consonants → auto-restrict to V<->V) ──
_VOWEL_BACK = {  # backness; long/short share a class
    'iː':'front', 'ɪ':'front', 'eː':'front', 'ɛ':'front', 'ɛː':'front',
    'y':'front', 'yː':'front', 'ʏ':'front', 'øː':'front', 'œ':'front', 'œy':'front',
    'ə':'central', 'a':'central', 'aː':'central',
    'uː':'back', 'oː':'back', 'ɔ':'back', 'ɑ':'back', 'ʌu':'back', 'au':'back',
}
def vowel_length(p):
    """'long'/'short' for vowels (by ː marker); None for consonants."""
    if p in _CONS: return None
    return 'long' if p.endswith('ː') else 'short'
def vowel_place(p):
    """Vowel backness front/central/back; None for consonants or unmapped vowels."""
    if p in _CONS: return None
    return _VOWEL_BACK.get(p)


def voicing_minpair_z(subs, n_perm=1000, seed=0):
    """Among consonant substitutions, the rate of VOICING MINIMAL PAIRS
    (same manner & place, opposite voicing — e.g. t<->d, s<->z), vs a shuffle
    baseline. High positive z = errors are minimal-pair confusions. Returns (z, n_cons)."""
    cons = [(g, p) for g, p in subs if is_cons(g) and is_cons(p)]
    if len(cons) < 5:
        return float('nan'), len(cons)
    def minpair(g, p):
        return (manner(g) == manner(p) and place(g) is not None
                and place(g) == place(p) and voicing(g) != voicing(p))
    rate = np.mean([minpair(g, p) for g, p in cons])
    rng = np.random.default_rng(seed); pool = [p for _, p in cons]
    bm = [np.mean([minpair(g, s) for (g, _), s in zip(cons, rng.permutation(pool))])
          for _ in range(n_perm)]
    return float((rate - np.mean(bm)) / (np.std(bm) + 1e-9)), len(cons)


def feature_z(subs, feat_fn, n_perm=1000, seed=0, min_n=3):
    """Preservation z for a phonetic feature over substitution pairs where the
    feature is defined (non-None) for BOTH sides. Returns (z, p_two_sided, n_pairs).
    feat_fn: phone -> category label, or None if not applicable (e.g. voicing on vowels).
    Two-sided permutation p: fraction of |null - null_mean| >= |obs - null_mean|."""
    pairs = [(g, p) for g, p in subs if feat_fn(g) is not None and feat_fn(p) is not None]
    if len(pairs) < min_n:
        return float('nan'), float('nan'), len(pairs)
    pres = np.mean([feat_fn(g) == feat_fn(p) for g, p in pairs])
    rng = np.random.default_rng(seed); pool = [p for _, p in pairs]
    bm = np.array([np.mean([feat_fn(g) == feat_fn(s)
                            for (g, _), s in zip(pairs, rng.permutation(pool))])
                   for _ in range(n_perm)])
    mu, sd = bm.mean(), bm.std()
    z = (pres - mu) / (sd + 1e-9)
    p = (np.sum(np.abs(bm - mu) >= np.abs(pres - mu)) + 1) / (n_perm + 1)
    return float(z), float(p), len(pairs)


# ── unified phonetic feature-distance ────────────────────────────────
def phone_feat_dist(g, p):
    """Number of phonetic features that differ between two phones (0..3).
    Axes: manner (always defined), place, voicing. Place/voicing only count
    when defined for BOTH phones (vowels have no place/voicing here). A cheap
    articulatory edit-distance: 0 = identical category on all axes."""
    d = 0
    d += manner(g) != manner(p)
    if place(g) is not None and place(p) is not None:
        d += place(g) != place(p)
    if voicing(g) is not None and voicing(p) is not None:
        d += voicing(g) != voicing(p)
    return d

def feat_dist_z(subs, n_perm=1000, seed=0):
    """Mean feature-distance of substitution errors vs a shuffle baseline.
    LOW distance = phonetically near-miss errors. z is signed so that a
    NEGATIVE z means errors are closer than chance (the good direction).
    Returns (mean_dist, baseline_mean, z, n)."""
    if len(subs) < 5:
        return float('nan'), float('nan'), float('nan'), len(subs)
    obs = np.mean([phone_feat_dist(g, p) for g, p in subs])
    rng = np.random.default_rng(seed); pool = [p for _, p in subs]
    bm = [np.mean([phone_feat_dist(g, s) for (g, _), s in zip(subs, rng.permutation(pool))])
          for _ in range(n_perm)]
    z = (obs - np.mean(bm)) / (np.std(bm) + 1e-9)
    return float(obs), float(np.mean(bm)), float(z), len(subs)


# ── confusion / near-miss analysis ───────────────────────────────────
def confusion_stats(subs, name, n_perm=1000, seed=0, verbose=True):
    """subs: list of (gold, pred) substitution pairs (gold != pred).
    Reports manner-preservation (+ permutation z), voicing-only and
    vowel-length-only near-miss rates."""
    cons = [(g, p) for g, p in subs if is_cons(g) and is_cons(p)]
    vows = [(g, p) for g, p in subs if not is_cons(g) and not is_cons(p)]
    mp = np.mean([manner(g) == manner(p) for g, p in subs]) if subs else np.nan
    vp = np.mean([manner(g) == manner(p) and ((g in VOICED) != (p in VOICED))
                  for g, p in cons]) if cons else np.nan
    lp = np.mean([base_vowel(g) == base_vowel(p) for g, p in vows]) if vows else np.nan
    rng = np.random.default_rng(seed); pool = [p for _, p in subs]
    bm = np.array([np.mean([manner(g) == manner(s)
                            for (g, _), s in zip(subs, rng.permutation(pool))])
                   for _ in range(n_perm)]) if subs else np.array([np.nan])
    mu, sd = bm.mean(), bm.std()
    z = (mp - mu) / (sd + 1e-9)
    p = ((np.sum(np.abs(bm - mu) >= np.abs(mp - mu)) + 1) / (n_perm + 1)
         if subs else np.nan)
    if verbose:
        print(f"{name}: {len(subs)} subs | manner {mp:.3f} (base {mu:.3f}, z={z:+.1f}, "
              f"p={p:.3g}) | voicing {vp:.3f} | length {lp:.3f}")
    return dict(n=len(subs), manner=float(mp), base=float(mu), z=float(z), p=float(p),
                voicing=float(vp) if vp == vp else np.nan,
                length=float(lp) if lp == lp else np.nan)


def subs_position_zip(out):
    """1:1 position-aligned substitutions (valid only when preds are 1:1 with gold, e.g. CRF)."""
    return [(t, p) for t, p in zip(list(out['true_labels']), list(out['predictions'])) if t != p]


def subs_nw(out):
    """Per-sentence NW-aligned substitutions (correct for non-1:1 outputs, e.g. SSL Viterbi)."""
    gold_per, pred_per = gather_sequences(out)
    pairs = []
    for sid in set(gold_per) | set(pred_per):
        al = needleman_wunsch(gold_per.get(sid, []), pred_per.get(sid, []))
        pairs += [(g, p) for g, p in al if g is not None and p is not None and g != p]
    return pairs


# ── ALL aligned pairs (incl. correct g==p) — for classification-quality metrics ──
def aligned_pairs_zip(out):
    """All 1:1 position-aligned (gold, pred) pairs INCLUDING correct ones (CRF)."""
    return [(t, p) for t, p in zip(list(out['true_labels']), list(out['predictions']))]


def aligned_pairs_nw(out):
    """All per-sentence NW-aligned (gold, pred) pairs INCLUDING correct ones; gaps
    dropped (SSL Viterbi). Insertions/deletions are not counted."""
    gold_per, pred_per = gather_sequences(out)
    pairs = []
    for sid in set(gold_per) | set(pred_per):
        al = needleman_wunsch(gold_per.get(sid, []), pred_per.get(sid, []))
        pairs += [(g, p) for g, p in al if g is not None and p is not None]
    return pairs
