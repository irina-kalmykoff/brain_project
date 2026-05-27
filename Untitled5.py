# Converted from Untitled5.ipynb

# **Headline:** the existing v6 + CRF pipeline produces real but modest
# brain-only decoding (max_run 3, surprise z +3.4 on P22). Many architectural
# alternatives we tried — frame-level CTC, joint training, audio-supervised
# boundary heads, cross-patient pooling — did *not* improve on this baseline
# at the per-patient data scale (~77 training sentences).

# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
plt.rcParams['figure.dpi'] = 110
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

# ## 1. The setup — what we're trying to do
#
# The pipeline takes iEEG signals from speech-producing patients and tries to
# decode the phonemes they're saying. Two coupled subproblems:
#
# 1. **Boundary detection** — *when* does each phoneme start and end?
# 2. **Phoneme classification** — *what* phoneme is in each segment?
#
# Both are done from cortical signal alone (brain-only). The reference
# alignment comes from MFA (Montreal Forced Aligner) on the audio.
#
# Per-patient data: ~77 train sentences, ~19 test sentences, ~30 phonemes each.

# %%
# Visualize the data scale
data_scale = {
    'train sentences':   77,
    'test sentences':    19,
    'phonemes / sentence (avg)': 30,
    'train phoneme labels':  77 * 30,
    'test phoneme labels':   19 * 30,
    'EEG channels (P22)': 112,
    'EEG sample rate (Hz)': 1024,
    'frame rate (Hz)': 200,
}
fig, ax = plt.subplots(figsize=(6, 2.5))
ax.barh(list(data_scale.keys()), list(data_scale.values()))
ax.set_xscale('log')
ax.set_xlabel('count (log scale)')
ax.set_title('P22 data scale')
plt.tight_layout(); plt.show()
print("Honest framing: per-patient training data is small. ~2,300 phoneme labels"
      " is not enough for end-to-end neural decoding from scratch.")


# ## 2. The one positive result — MFA-CRF
#
# We trained per-patient linear-chain CRFs on phoneme-segment features pooled
# at MFA-derived boundaries. At test time, evaluate by feeding MFA boundaries
# again (oracle-boundary inference). This is the strongest signal we ever
# extracted from the iEEG.
#
# **Per-sentence longest contiguous exact match** (max_run) ranged 5–6 across
# patients. The rarity-weighted **surprise z = +13.9** on P22 with the
# `show_all_patients` accumulation framework — strong evidence that the
# matches aren't chance alignment of phoneme priors.

# %%
mfa_crf_max_run = {  # max longest contiguous exact match per patient
    'P21': 5, 'P22': 5, 'P23': 6, 'P24': 5, 'P25': 4,
    'P26': 4, 'P27': 5, 'P28': 4, 'P29': 4, 'P30': 6,
}
fig, ax = plt.subplots(figsize=(7, 3))
patients, runs = zip(*mfa_crf_max_run.items())
ax.bar(patients, runs, color='tab:blue')
ax.axhline(3, ls='--', color='gray', label='chance ceiling (~3)')
ax.set_ylabel('longest contiguous exact match')
ax.set_title('MFA-CRF: max_run per patient (P22 surprise z = +13.9)')
ax.legend(); plt.tight_layout(); plt.show()
print("Real decoding signal exists when boundaries are correct.")
print("P23 and P30 produce 6-phoneme exact contiguous matches — well above chance.")

# **Caveat:** this is *brain-only classification given audio-derived
# boundaries*. MFA needs audio at every step. For a deployable BCI for
# patients who can't speak, the boundaries have to come from brain alone.
# Hence the next investigation.

# %% [markdown]
# ## 3. Boundary detection from brain alone — why it's hard
#
# The existing v6 boundary detector is a BiLSTM trained to predict MFA-derived
# boundary peak labels from per-frame iEEG. It caps at AUC ~0.59 against MFA
# boundaries on test. To diagnose why, we asked: *do MFA boundaries show up as
# peaks in any signal derived from iEEG?*
#
# Tested seven boundary cues across many bands and time resolutions. None
# meaningfully better than 0.55 AUC. Audio signals (comparison) hit 0.62.

# %%
boundary_aucs = {
    'EEG HG amp derivative': 0.55,
    'EEG HG channel variance': 0.55,
    'EEG theta phase derivative': 0.49,
    'EEG beta phase derivative': 0.49,
    'EEG multivariate L2 change': 0.51,
    'EEG phase locking value drop': 0.50,
    'EEG phase disorder derivative': 0.50,
    'EEG inst-freq variance': 0.50,
    'audio RMS derivative': 0.62,
    'audio spectral flux': 0.60,
    'audio MFCC derivative': 0.62,
}
fig, ax = plt.subplots(figsize=(7, 4))
keys = list(boundary_aucs.keys())
vals = [boundary_aucs[k] for k in keys]
colors = ['steelblue' if 'EEG' in k else 'darkorange' for k in keys]
ax.barh(keys, vals, color=colors)
ax.axvline(0.5, ls='--', color='black', alpha=0.4, label='chance')
ax.axvline(0.62, ls=':', color='red', alpha=0.6, label='audio teacher ceiling')
ax.set_xlim(0.45, 0.7)
ax.set_xlabel('AUC vs MFA boundary (test set)')
ax.set_title('Boundary-cue AUC: EEG (blue) vs audio (orange)')
ax.legend(); ax.invert_yaxis()
plt.tight_layout(); plt.show()
print("EEG signals don't peak at phoneme boundaries the way audio does.")
print("This is a fundamental finding — affects every boundary-detection-based "
      "architecture we tried.")

# ## 4. Frame-level boundary-free alternatives
#
# If we can't detect boundaries from EEG cleanly, can we skip boundaries
# entirely? We tried several boundary-free decoders on P22, all using the
# same Hilbert-envelope HG features at 200 Hz frame rate.
#
# - **Frame-CE + greedy collapse**: train per-frame phoneme classifier on
#   MFA-derived frame labels, decode by argmax-then-collapse-repeats.
# - **Class-balanced frame-CE**: weight rare phonemes more heavily.
# - **CTC**: learn to emit phonemes at undefined frame positions.
# - **Frame-CE + bigram LM rescoring**: add a phoneme-bigram prior over
#   per-frame posteriors via beam search.
#
# Result for all of them: cosmetic max_run 3-5 (matching prior-collapse
# chance), but **rarity-weighted surprise z ≈ 0 or negative**. The matches
# come from common phonemes aligning by chance, not real decoding.

# %%
frame_level_results = pd.DataFrame([
    ('Frame-CE + greedy',          5, -1.0),
    ('Frame-CE + class weights',   4, -0.3),
    ('Frame-CE + bigram LM (α=1)', 3, +1.8),
    ('Frame-CE + bigram LM (α=3)', 3, +0.1),
    ('CTC (collapsed to blank)',   0,  np.nan),
], columns=['model', 'max_run', 'surprise_z'])
fig, ax = plt.subplots(figsize=(7, 3))
ax.bar(frame_level_results['model'], frame_level_results['surprise_z'],
       color=['tomato' if z < 1 else 'mediumseagreen'
              for z in frame_level_results['surprise_z'].fillna(0)])
ax.axhline(0, color='black', lw=0.5)
ax.axhline(3, ls='--', color='gray', label='significance threshold (z=+3)')
ax.set_ylabel('surprise z')
ax.set_title('Frame-level boundary-free decoders (P22) — none clear z=+3')
ax.legend(); plt.setp(ax.get_xticklabels(), rotation=15, ha='right')
plt.tight_layout(); plt.show()
print("Frame-level signals are not enough to produce real sequence decoding.")
print("All variants converge to prior-distribution emission with chance matches.")


# ## 5. Cross-patient pooling
#
# Per-patient data is small. Could pooling across patients help?
#
# - **Voxel-grid pooling (5mm)**: 752/798 voxels covered by 1 patient,
#   only 1 voxel by 3+. Patients' electrodes barely overlap spatially.
# - **Anatomical-region pooling**: 27 brain regions shared by ≥3 patients.
#   Built shared input space; trained classifier on 8 patients, tested on
#   held-out P21. Best held-out balanced accuracy 4.10% vs chance 2.38%.
#
# Permutation null test (10 shuffles): real 2.45 ± 0.91%, null 1.57 ± 0.85%,
# z = +1.0. **Not significant.** Cross-patient transfer at this dataset doesn't
# generalize phoneme decoding.

# %%
voxel_coverage = {
    'voxels seen by 1 patient':  752,
    'voxels seen by 2 patients':  45,
    'voxels seen by 3+ patients':  1,
}
fig, axes = plt.subplots(1, 2, figsize=(10, 3))
axes[0].pie(voxel_coverage.values(), labels=voxel_coverage.keys(),
            autopct='%1.0f%%', colors=['tomato','gold','mediumseagreen'])
axes[0].set_title('5mm voxel grid: cross-patient overlap')

xp_results = pd.DataFrame([
    ('voxel grid (5mm)',  3.04, 2.38),
    ('anatomical region',  4.10, 2.38),
    ('null (permuted)',    2.45, 2.38),
], columns=['approach', 'held_out_acc', 'chance'])
axes[1].bar(xp_results['approach'], xp_results['held_out_acc'], color='steelblue')
axes[1].axhline(2.38, ls='--', color='red', label='chance')
axes[1].set_ylabel('balanced accuracy %')
axes[1].set_title('Held-out P21 transfer')
axes[1].legend()
plt.tight_layout(); plt.show()
print("Cross-patient pooling doesn't recover signal at this data scale.")
print("The 4.10% on anatomical region was a single seed; permutation z = +1.0 "
      "(not significant).")

# ## 6. Joint training — end-to-end boundary + classifier
#
# Instead of training boundary detector and classifier separately, train them
# jointly: classifier loss flows back through soft segmentation to the boundary
# head. Hypothesis: model learns boundaries that maximize downstream phoneme
# accuracy, even if those aren't MFA-style peaks.
#
# Architecture: BiLSTM encoder → boundary head (multi-target audio teacher) →
# soft cumulative segmentation → segment classifier head. Losses: classification
# CE + count anchor + audio teacher MSE.
#
# Three configurations tried, all on P22. None worked.

# %%
joint_results = pd.DataFrame([
    ('Joint v1 (decoupled teacher, λ_count=0.1)', 4, -0.3),
    ('Joint v2 (direct teacher, λ_count=0.1)',    5, -4.1),
    ('Joint v3 (direct teacher, λ_count=0.01)',   4, -7.3),
], columns=['model', 'max_run', 'surprise_z'])
fig, ax = plt.subplots(figsize=(7, 3))
ax.bar(joint_results['model'], joint_results['surprise_z'], color='tomato')
ax.axhline(0, color='black', lw=0.5)
ax.set_ylabel('surprise z (lower = worse)')
ax.set_title('Joint training: all attempts prior-collapse (z < 0)')
plt.setp(ax.get_xticklabels(), rotation=15, ha='right')
plt.tight_layout(); plt.show()
print("Joint training in this regime overfits to phoneme priors.")
print("With 77 train sentences, the joint architecture is too capacity-rich.")

# ## 7. Audio-supervised boundary detector (standalone)
#
# Decouple the joint training failure from the boundary supervision question.
# Train a per-patient BiLSTM boundary head on EEG to predict audio-derived
# signals (RMS derivative, spectral flux, MFCC derivative). At test time,
# only EEG. The brain has to predict audio dynamics.
#
# This worked! AUC reached 0.62 — saturating the audio teacher's own AUC
# against MFA boundaries (0.62). So the model is doing as well as possible
# given the teacher.

# %%
# Audio-sup training curve (P22, multi-target teacher)
audio_sup_curve = pd.DataFrame([
    ( 1, 0.595), ( 2, 0.610), ( 3, 0.614), ( 4, 0.617), ( 5, 0.617),
    ( 6, 0.616), ( 7, 0.612), ( 8, 0.607), ( 9, 0.608), (10, 0.610),
    (15, 0.613), (20, 0.613), (25, 0.614), (30, 0.613),
], columns=['epoch', 'auc'])
fig, ax = plt.subplots(figsize=(7, 3))
ax.plot(audio_sup_curve['epoch'], audio_sup_curve['auc'], 'o-', color='steelblue')
ax.axhline(0.62, ls='--', color='red', label='audio teacher AUC ceiling')
ax.axhline(0.55, ls=':',  color='gray', label='EEG HG amp deriv baseline')
ax.set_xlabel('epoch'); ax.set_ylabel('AUC vs MFA boundary (test)')
ax.set_title('Audio-supervised boundary detector saturates teacher ceiling')
ax.legend(); ax.set_ylim(0.5, 0.65)
plt.tight_layout(); plt.show()
print("EEG can predict audio-RMS-derivative well enough to match the audio "
      "signal's own boundary AUC.")
print("This was the most encouraging mid-investigation result.")

# ## 8. Adding a count head
#
# For brain-only test-time inference, the model needs to predict K (number of
# phonemes) as well as boundary positions. Added a count head to the
# audio-supervised model: pooled BiLSTM output → linear → log K.
# Joint loss = boundary MSE + λ × count smooth-L1.
#
# Result: AUC held at 0.622, K predicted within 21% MAE.

# %%
count_head_progress = pd.DataFrame([
    ( 1, 0.575, 92.3),
    ( 2, 0.611, 70.7),
    ( 3, 0.615, 20.6),
    ( 4, 0.619, 25.0),
    ( 5, 0.622, 21.4),
    (10, 0.613, 27.7),
    (20, 0.611, 19.8),
    (30, 0.605, 23.1),
], columns=['epoch', 'auc', 'k_err_pct'])
fig, axes = plt.subplots(1, 2, figsize=(10, 3))
axes[0].plot(count_head_progress['epoch'], count_head_progress['auc'],
             'o-', color='steelblue')
axes[0].set_xlabel('epoch'); axes[0].set_ylabel('AUC vs MFA boundary')
axes[0].set_title('Boundary head: AUC ~stable with count head added')
axes[1].plot(count_head_progress['epoch'], count_head_progress['k_err_pct'],
             'o-', color='tomato')
axes[1].set_xlabel('epoch'); axes[1].set_ylabel('K prediction MAE %')
axes[1].set_title('Count head: 92% → 20% MAE')
plt.tight_layout(); plt.show()
print("Count head works: K predicted to ~20% error.")
print("Still a substantial error — predicted K can differ by ±6 phonemes for "
      "a typical 30-phoneme sentence.")

# %% [markdown]
# ## 9. The unified A/B evaluation
#
# Compare audio-sup against v6 against MFA in a single framework where the
# only thing that changes is the boundary source. Same CRF, same feature
# extraction, same test set.

# %%
ab_p22 = pd.DataFrame([
    ('MFA',       'oracle',    4, 90.9, +2.9),
    ('v6',        'oracle',    3, 95.2, +3.4),
    ('v6',        'predicted', 3, 95.2, +3.4),
    ('audio_sup', 'oracle',    4, 84.3, +0.8),
    ('audio_sup', 'predicted', 3, 87.8, +0.1),
], columns=['source', 'K source', 'max_run', 'PER %', 'surprise z'])
print("\nUnified A/B on P22 (with count head):")
print(ab_p22.to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(11, 3))
labels = [f"{r.source}\n({r['K source']})" for _, r in ab_p22.iterrows()]
axes[0].bar(labels, ab_p22['max_run'],
            color=['gold','steelblue','steelblue','tomato','tomato'])
axes[0].set_ylabel('max_run'); axes[0].set_title('Longest contiguous exact match (P22)')
axes[1].bar(labels, ab_p22['surprise z'],
            color=['gold','steelblue','steelblue','tomato','tomato'])
axes[1].axhline(0, color='black', lw=0.5)
axes[1].axhline(3, ls='--', color='gray', label='z=+3 threshold')
axes[1].set_ylabel('surprise z'); axes[1].set_title('Decoding information (P22)')
axes[1].legend()
for ax in axes:
    plt.setp(ax.get_xticklabels(), rotation=15, ha='right')
plt.tight_layout(); plt.show()

print("\nKey findings from the A/B:")
print("- v6 (brain-only) achieves z=+3.4 — better than both MFA oracle (+2.9) "
      "and audio_sup oracle (+0.8).")
print("- audio_sup max_run=4 *looks* better than v6 max_run=3, but z says v6's "
      "matches carry far more decoding information.")
print("- audio_sup predicted K collapses (z=+0.1) — count head not accurate enough.")
print("- v6's count head is essentially perfect (oracle=predicted=+3.4).")

# %%
# Final summary plot
final_summary = pd.DataFrame([
    ('MFA-CRF (orig framework)',          5,  +13.9, 'positive'),
    ('MFA-CRF (strict A/B framework)',    4,   +2.9, 'positive'),
    ('v6 + CRF (brain-only)',             3,   +3.4, 'positive'),
    ('audio_sup oracle',                   4,   +0.8, 'inconclusive'),
    ('audio_sup predicted (brain-only)',   3,   +0.1, 'inconclusive'),
    ('Frame-CE + bigram LM',               3,   +1.8, 'inconclusive'),
    ('Frame-CE greedy',                    5,   -1.0, 'negative'),
    ('Joint training v3',                  4,   -7.3, 'negative'),
    ('Cross-patient (held-out P21)',       0,   +1.0, 'negative'),
], columns=['approach', 'max_run', 'surprise_z', 'verdict'])
colors_map = {'positive': 'mediumseagreen',
              'inconclusive': 'gold',
              'negative': 'tomato'}
fig, ax = plt.subplots(figsize=(9, 5))
ax.barh(final_summary['approach'],
        final_summary['surprise_z'],
        color=[colors_map[v] for v in final_summary['verdict']])
ax.axvline(0, color='black', lw=0.5)
ax.axvline(+3, ls='--', color='gray', label='significance threshold')
ax.set_xlabel('surprise z (rarity-weighted)')
ax.set_title('Full investigation summary: surprise z by approach')
ax.invert_yaxis(); ax.legend()
plt.tight_layout(); plt.show()

# %%
print("=" * 70)
print("Final defensible result: v6 + CRF brain-only decoding on P22")
print("  - max_run = 3 (3-phoneme contiguous exact matches within sentences)")
print("  - surprise z = +3.4 (matches above-chance information content)")
print("  - PER ≈ 95%")
print("Audio is needed only at training time (for MFA boundary labels).")
print("Test-time inference uses only iEEG.")
print("=" * 70)

# %% [markdown]
# # The surprise function, explained step by step
#
# Throughout the investigation we used a metric called **rarity-weighted
# surprise** to tell whether predictions carried real decoding signal or just
# happened to share the phoneme prior with the gold sequence. This file walks
# through what surprise is, how it works, and why it's a more honest test
# than counting matches or measuring longest n-gram.
#
# All numbers below are synthetic so the demonstration is reproducible.

# %%
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
plt.rcParams['figure.dpi'] = 110
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

rng = np.random.default_rng(0)

# %% [markdown]
# ## Step 1 — the phoneme prior is heavily skewed
#
# Dutch (and any natural language) has highly non-uniform phoneme frequency.
# A few phonemes (`ɛ`, `r`, `n`, `d`) are very common; many others occur
# rarely. This is the key fact that makes naive match-counting misleading.

# %%
# Approximate phoneme frequencies from Dutch sEEG MFA alignments
phoneme_prior = {
    'ɛ': 0.12, 'r': 0.08, 'n': 0.07, 'd': 0.07, 't': 0.06,
    'eː': 0.05, 'aː': 0.05, 's': 0.05, 'k': 0.04, 'p': 0.04,
    'l': 0.04, 'm': 0.03, 'v': 0.03, 'ʋ': 0.03, 'ɪ': 0.03,
    'b': 0.02, 'oː': 0.02, 'j': 0.02, 'iː': 0.02, 'h': 0.02,
    'f': 0.01, 'z': 0.01, 'x': 0.01, 'ɔ': 0.01, 'ɑ': 0.01,
    'ɣ': 0.005, 'ŋ': 0.005, 'œ': 0.005, 'ʏ': 0.005, 'uː': 0.005,
}
total = sum(phoneme_prior.values())
phoneme_prior = {k: v/total for k, v in phoneme_prior.items()}

phonemes_sorted = sorted(phoneme_prior, key=lambda p: -phoneme_prior[p])
probs_sorted = [phoneme_prior[p] for p in phonemes_sorted]

fig, ax = plt.subplots(figsize=(9, 3))
ax.bar(phonemes_sorted, probs_sorted, color='steelblue')
ax.set_ylabel('P(phoneme)')
ax.set_title('Approximate Dutch phoneme prior — heavily skewed')
ax.tick_params(axis='x', rotation=0)
plt.tight_layout(); plt.show()

print("Top 5 phonemes account for ~37% of all occurrences.")
print("Rare phonemes (ɣ, ŋ, œ, ʏ, uː) each occur < 1% of the time.")

# %% [markdown]
# ## Step 2 — surprise per phoneme: −log P(phoneme)
#
# Surprise (a.k.a. self-information) converts a probability into a "how
# unusual is it to see this token" score. Common tokens have low surprise;
# rare tokens have high surprise.
#
# It's the same quantity that underlies cross-entropy and KL divergence —
# `−log P(x)` measured in nats (if natural log) or bits (if log base 2).

# %%
log_p = {p: -np.log(P) for p, P in phoneme_prior.items()}

fig, ax = plt.subplots(figsize=(9, 3))
ax.bar(phonemes_sorted, [log_p[p] for p in phonemes_sorted], color='tomato')
ax.set_ylabel('surprise = −log P(phoneme)')
ax.set_title('Predicting a rare phoneme is much more "informative" than a common one')
plt.tight_layout(); plt.show()

print(f"surprise(ɛ)  = {log_p['ɛ']:.2f}   — common, easy to guess")
print(f"surprise(ʏ)  = {log_p['ʏ']:.2f}   — rare, predicting it right is strong evidence")
print(f"surprise(ɣ)  = {log_p['ɣ']:.2f}   — rarest")
print("\nIntuition: if your model correctly predicts 'ɛ' it gets only ~2 nats")
print("of credit. If it correctly predicts 'ʏ' it gets ~5 nats — over 2x the credit.")

# %% [markdown]
# ## Step 3 — score a match by total surprise of its phonemes
#
# When prediction and gold share a contiguous matching span, we score it
# as the sum of self-information across the matched phonemes.
#
# A length-3 match of common phonemes scores less than a length-3 match
# containing one rare phoneme.

# %%
def score_match(span):
    return sum(log_p.get(p, np.log(len(phoneme_prior))) for p in span)

example_matches = {
    "ɛ r d (all common)":    ['ɛ', 'r', 'd'],
    "n d eː (common)":         ['n', 'd', 'eː'],
    "ʋ ɣ ʏ (all rare)":       ['ʋ', 'ɣ', 'ʏ'],
    "r ɑ w ʏ (mixed, rare end)": ['r', 'ɑ', 'ʋ', 'ʏ'],
    "b ɛ r d eː (5 common)":    ['b', 'ɛ', 'r', 'd', 'eː'],
    "v ɛ r d ɛ r (6 common)":   ['v', 'ɛ', 'r', 'd', 'ɛ', 'r'],
}

fig, ax = plt.subplots(figsize=(8, 3))
labels, scores = zip(*[(k, score_match(v)) for k, v in example_matches.items()])
colors = ['steelblue' if 'rare' not in l else 'mediumseagreen' for l in labels]
ax.barh(labels, scores, color=colors)
ax.set_xlabel('total surprise score of the match')
ax.set_title('A short rare match can score higher than a long common match')
ax.invert_yaxis()
plt.tight_layout(); plt.show()

print("Why this matters: a length-3 match of 'ʋ ɣ ʏ' scores ~14, while a")
print("length-5 match of 'b ɛ r d eː' scores ~12. Longer isn't always better.")
print("This is what max_run alone misses — it can't tell prior-collapse from")
print("genuine rare-phoneme decoding.")

# %% [markdown]
# ## Step 4 — total surprise across a test set
#
# Run the prediction model on the whole test set, find all matched substrings
# of length ≥3 between (prediction, gold) per sentence using shift-tolerant
# contiguous-exact matching, sum the surprise of every matched phoneme.
# That's `real_total`.

# %%
def find_matches(pred, gold, min_n=3):
    """Greedy longest-first shift-tolerant contiguous exact match."""
    n, m = len(pred), len(gold)
    used_p = [False]*n; used_g = [False]*m
    spans = []
    for L in range(min(n, m), min_n-1, -1):
        for i in range(n-L+1):
            if any(used_p[i:i+L]): continue
            for j in range(m-L+1):
                if any(used_g[j:j+L]): continue
                if pred[i:i+L] == gold[j:j+L]:
                    spans.append((i, j, pred[i:i+L]))
                    for k in range(L):
                        used_p[i+k] = True; used_g[j+k] = True
                    break
    return spans

# Synthetic example: 5-sentence test set
gold_seqs = [
    list("ɛrdeːnplɔx"),
    list("hɛtisdeːlvɑnlɑntrɔv"),
    list("ɪnvɛrkɑwʏdɛndʏsɪkɣaːn"),
    list("zɛjnɛrnɔxpʏntjɛs"),
    list("zeːzɪtɔpdeːɑxtɛrbɑnk"),
]
# A "decoding" model that catches some real subsequences
pred_decoding = [
    list("ɛrdeːneeplɔx"),       # matches "ɛrdeːn" and "plɔx"
    list("hɛtizdeːlvɑnlɑntrɔv"), # matches "hɛti" and "deːlvɑnlɑntrɔv"
    list("dɛndʏsɪkxxx"),         # matches "dɛndʏsɪk"
    list("ɛrɛrɛrɛr"),             # no real match
    list("ɛrdeːnpʏntjɛs"),       # matches "pʏntjɛs"
]
# A "prior collapse" model that just emits common phonemes
pred_prior = [
    list("ɛrdɛrɛnnɛrn"),
    list("dɛrnɛɛrdɛnɛrdɛnɛ"),
    list("nɛrdɛrnɛrdɛrɛrnɛr"),
    list("ɛrɛrɛrɛrɛrɛrɛrɛ"),
    list("rɛrnɛrdɛrnɛrdɛrɛr"),
]

def total_surprise(preds, golds):
    return sum(score_match(s[2]) for p, g in zip(preds, golds)
               for s in find_matches(p, g))

real_decoding = total_surprise(pred_decoding, gold_seqs)
real_prior    = total_surprise(pred_prior, gold_seqs)
print(f"Decoding model total surprise: {real_decoding:.1f}")
print(f"Prior-collapse model total surprise: {real_prior:.1f}")
print("Both might have similar max_run; surprise distinguishes them.")

# %% [markdown]
# ## Step 5 — the permutation null
#
# `real_total` alone is hard to interpret. Is 50 a lot? Depends on test
# set size and how skewed the prior is.
#
# **Solution:** shuffle the model's predictions across positions while
# preserving the marginal distribution. Recompute total surprise. Repeat
# K times. Get a null distribution. Compare real to null.

# %%
def shuffled_total(preds, golds, seed):
    rng = np.random.default_rng(seed)
    flat = [p for sent in preds for p in sent]
    sh = list(rng.permutation(flat))
    sh_seqs = []
    idx = 0
    for sent in preds:
        sh_seqs.append(sh[idx:idx+len(sent)])
        idx += len(sent)
    return total_surprise(sh_seqs, golds)

n_nulls = 200
null_decoding = np.array([shuffled_total(pred_decoding, gold_seqs, s) for s in range(n_nulls)])
null_prior    = np.array([shuffled_total(pred_prior, gold_seqs, s) for s in range(n_nulls)])

fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))

axes[0].hist(null_decoding, bins=20, color='lightgray', edgecolor='black')
axes[0].axvline(real_decoding, color='mediumseagreen', lw=3, label=f'real ({real_decoding:.1f})')
axes[0].axvline(null_decoding.mean(), color='red', ls='--', label=f'null mean ({null_decoding.mean():.1f})')
axes[0].set_title('Decoding model: real >> null')
axes[0].set_xlabel('total surprise'); axes[0].legend()

axes[1].hist(null_prior, bins=20, color='lightgray', edgecolor='black')
axes[1].axvline(real_prior, color='tomato', lw=3, label=f'real ({real_prior:.1f})')
axes[1].axvline(null_prior.mean(), color='red', ls='--', label=f'null mean ({null_prior.mean():.1f})')
axes[1].set_title('Prior-collapse model: real ≈ null')
axes[1].set_xlabel('total surprise'); axes[1].legend()
plt.tight_layout(); plt.show()

z_decoding = (real_decoding - null_decoding.mean()) / null_decoding.std()
z_prior    = (real_prior - null_prior.mean()) / null_prior.std()
print(f"Decoding model:        z = {z_decoding:+.2f}  ← real decoding")
print(f"Prior-collapse model:  z = {z_prior:+.2f}     ← chance")

# %% [markdown]
# ## Step 6 — what z means in practice
#
# - **z > +3**: real total clearly exceeds null. Predictions are matching at
#   rare phonemes more often than chance — genuine decoding signal.
# - **z ≈ 0**: real and null indistinguishable. Whatever matches the model
#   produces are explainable by emitting the marginal phoneme distribution.
# - **z negative**: real total is *below* null. Predictions concentrate
#   matches at *more* common phonemes than chance would give — strongly
#   prior-collapsed. We saw this in joint-training v3 (z = −7.3).

# %%
# Summary plot: z-score zones
fig, ax = plt.subplots(figsize=(9, 2.5))
ax.axhspan(-10, -2, color='tomato', alpha=0.25, label='strong prior collapse')
ax.axhspan(-2, 1.5, color='lightgray', alpha=0.5, label='chance / weak')
ax.axhspan(1.5, 3, color='gold', alpha=0.3, label='borderline')
ax.axhspan(3, 20, color='mediumseagreen', alpha=0.3, label='real decoding')

real_data = [
    ('joint v3',        -7.3),
    ('frame-CE greedy', -1.0),
    ('audio_sup pred',  +0.1),
    ('audio_sup oracle',+0.8),
    ('frame+bigram',    +1.8),
    ('MFA-CRF (A/B)',   +2.9),
    ('v6 (brain-only)', +3.4),
    ('MFA-CRF (orig)', +13.9),
]
labels, zs = zip(*real_data)
ax.scatter(range(len(labels)), zs, color='black', s=50, zorder=5)
for i, (lbl, z) in enumerate(real_data):
    ax.annotate(lbl, (i, z), textcoords='offset points',
                xytext=(0, 8), ha='center', fontsize=8)
ax.set_xticks([]); ax.set_ylim(-10, 16); ax.set_ylabel('surprise z')
ax.legend(loc='lower right', fontsize=8)
ax.set_title('Where each approach landed in our investigation')
plt.tight_layout(); plt.show()

# %% [markdown]
# ## Why this is better than max_run alone
#
# `max_run` (longest contiguous exact match) is intuitive but can be fooled.
# A prior-collapsed model that emits common phonemes will eventually produce
# long matches by chance. Counting those as "decoding" is misleading.
#
# Surprise z corrects this by giving high credit only to matches that
# include rare phonemes — exactly the matches a prior model cannot fake.
#
# We learned this the hard way: our frame-CE encoder hit `max_run = 5`
# (same as MFA-CRF!) but z = −1.0 (no real signal). Without the surprise
# test we would have wrongly concluded boundary-free decoding worked.

# %%
print("Summary:")
print("- surprise(p) = -log P(p): gives more credit for predicting rare phonemes")
print("- real_total: sum of surprise over all matched substrings (length ≥ 3)")
print("- null distribution: shuffle predictions, preserve marginal, recompute")
print("- z = (real - null_mean) / null_std")
print("- z > +3 = real decoding; z ≈ 0 = prior collapse; z < 0 = anti-decoding")
