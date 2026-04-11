# Phoneme Boundary & Consecutive Prediction Recommendations

Brain-to-speech decoding pipeline — strategies to improve phoneme boundary accuracy
and increase runs of consecutive correct predictions.

Current baseline (P21–P30, stacking order=5):
- Parse rate: 57–65% of expected phonemes found
- AvgMaxPos: ~5.3–5.6 out of ~7.4 words per sentence
- Consecutive correct runs: mostly pairs, rarely 4+

---

## Priority 1 — Montreal Forced Aligner (MFA)

**Problem it solves:** phoneme boundaries within words are currently *inferred*
from the phonetic dictionary (equal-time split across phonemes in a word).
If a word is 200 ms with 4 phonemes the pipeline assumes 50 ms each, regardless
of actual articulation. This contaminates every training sample.

**Fix:** MFA aligns phonemes directly to the audio recording, giving measured
start/end timestamps per phoneme per sentence. Cleaner segments → cleaner
training data → longer correct runs.

### Setup

```bash
pip install montreal-forced-aligner

# Download Dutch acoustic model and pronunciation dictionary
mfa model download acoustic dutch_mfa
mfa model download dictionary dutch_mfa

# Align (expects a folder with .wav + .txt/.lab per utterance)
mfa align /path/to/utterances dutch_mfa dutch_mfa /path/to/output
```

### Integration

```python
import tgt  # pip install tgt  (TextGrid reader)
import os

def load_mfa_phoneme_times(textgrid_path, tier_name='phones'):
    """
    Read per-phoneme start/end times (seconds) from an MFA TextGrid.
    Returns list of (phoneme, start_s, end_s).
    """
    tg = tgt.io.read_textgrid(textgrid_path)
    tier = tg.get_tier_by_name(tier_name)
    return [(ann.text, ann.start_time, ann.end_time)
            for ann in tier.annotations if ann.text not in ('', 'sp', 'sil')]

def mfa_times_to_eeg_indices(phoneme_times, eeg_sr=1024, audio_sr=48000):
    """
    Convert MFA phoneme times (seconds) to EEG sample indices.
    """
    segments = []
    for phoneme, t_start, t_end in phoneme_times:
        eeg_start = int(t_start * eeg_sr)
        eeg_end   = int(t_end   * eeg_sr)
        segments.append((phoneme, eeg_start, eeg_end))
    return segments

# Usage in pipeline:
# for each sentence → load TextGrid → get phoneme segments
# → slice EEG directly by (eeg_start, eeg_end) instead of equal-time split
```

### Expected gain
High. This is the single most impactful improvement because it fixes the training
data quality at the source. All downstream steps benefit automatically.

### Effort
Medium — one-time MFA alignment run per patient's audio files. Output TextGrids
are reusable; pipeline integration requires a new segmentation path in
`step4_custom_detector` or a new `step4_mfa_segmentation` step.

---

## Priority 2 — Larger Temporal Context Window

**Problem it solves:** stacking order=5 gives an 11-frame window = 55 ms of
context. Dutch phonemes average 60–120 ms, so the model often sees less than
one full phoneme. Increasing the window captures the full phoneme plus partial
neighbours, providing articulatory onset/offset cues.

### Sweep additions

```python
# Add to stacking_params in the experiment sweep:
{'stacking_order': 7,  'stacking_step_size': 1},   # 75 ms
{'stacking_order': 10, 'stacking_step_size': 1},   # 105 ms
{'stacking_order': 10, 'stacking_step_size': 2},   # 105 ms, sparser
{'stacking_order': 15, 'stacking_step_size': 1},   # 155 ms (covers long vowels)
```

### Notes
- Short phonemes (1–3 frames) are zero-padded to fill the window — this is
  already handled. Zero-padding dilutes magnitude but preserves spatial patterns.
- Feature size grows linearly: order=10 → 21 frames × n_channels.
  Watch for overfitting; use stronger regularisation (smaller C in LR).
- Combine with `stacking_step_size=2` to reduce feature size while keeping range.

### Expected gain
Medium. Easy to test via the existing sweep loop.

### Effort
Low — already supported, just extend the sweep config.

---

## Priority 3 — CRF Sequence Model

**Problem it solves:** each phoneme is currently classified independently. The
model has no knowledge that `/n/ → /ɑ/ → /x/ → /t/` is coherent Dutch but
`/p/ → /z/ → /b/` is not. A CRF scores the *joint* label sequence, penalising
phonotactically illegal transitions and rewarding legal ones.

### Implementation

```python
# pip install sklearn-crfsuite
import sklearn_crfsuite
from sklearn.preprocessing import StandardScaler
import numpy as np

def make_crf_features(X_flat, i, window=2):
    """
    Build a feature dict for CRF at position i.
    Includes the flat EEG vector plus positional context.
    """
    feat = {f'eeg_{j}': float(v) for j, v in enumerate(X_flat[i])}
    feat['pos'] = i
    feat['bias'] = 1.0
    # Relative position flags
    feat['is_first'] = (i == 0)
    feat['is_last']  = (i == len(X_flat) - 1)
    return feat

def group_by_sentence(pipeline_data, pid):
    """
    Group phoneme samples by sentence (using phoneme_instance_ids ordering).
    Returns list of (X_seq, y_seq) per sentence.
    """
    from itertools import groupby
    idx    = [i for i, p in enumerate(pipeline_data['phoneme_participant_ids'])
              if p == pid]
    iids   = [pipeline_data['phoneme_instance_ids'][i] for i in idx]
    feats  = [np.array(pipeline_data['features'][i]).flatten() for i in idx]
    labels = [pipeline_data['phoneme_labels'][i] for i in idx]
    words  = [pipeline_data['phoneme_words'][i]  for i in idx]

    # Group consecutive phonemes that belong to the same word instance
    sequences = []
    for iid, group in groupby(zip(iids, feats, labels, words),
                               key=lambda x: x[0]):
        items = list(group)
        sequences.append(([make_crf_features(
                               [it[1] for it in items], j)
                           for j in range(len(items))],
                          [it[2] for it in items]))
    return sequences

def train_crf_per_patient(pipeline, pid, c1=0.1, c2=0.1):
    scaler = StandardScaler()

    train_seqs = group_by_sentence(pipeline.train, pid)
    test_seqs  = group_by_sentence(pipeline.test,  pid)

    # Fit scaler on all train features combined
    all_train_X = np.vstack([
        np.array([list(f.values()) for f in X]) for X, _ in train_seqs
    ])
    scaler.fit(all_train_X)

    X_tr = [[{k: v for k, v in f.items()} for f in X] for X, _ in train_seqs]
    y_tr = [y for _, y in train_seqs]
    X_te = [[{k: v for k, v in f.items()} for f in X] for X, _ in test_seqs]
    y_te = [y for _, y in test_seqs]

    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs', c1=c1, c2=c2,
        max_iterations=200, all_possible_transitions=True,
    )
    crf.fit(X_tr, y_tr)

    y_pred = crf.predict(X_te)
    correct = sum(p == t for seq_p, seq_t in zip(y_pred, y_te)
                          for p, t in zip(seq_p, seq_t))
    total   = sum(len(s) for s in y_te)
    print(f"{pid}: CRF accuracy = {correct/total:.1%}  ({total} phonemes)")
    return crf, y_pred, y_te
```

### Expected gain
Medium. Most useful when boundary detection is already good (runs of 3–5
phonemes exist). CRF smooths over single-phoneme errors within a run.

### Effort
Low — `sklearn-crfsuite` installs cleanly, no GPU needed.

---

## Priority 4 — Phoneme Bigram Language Model Decoding

**Problem it solves:** the existing Viterbi decoder uses transition probabilities
estimated from the (small) training set. Dutch phonotactics are better captured
from a large external corpus such as CELEX.

### Implementation

```python
from collections import defaultdict
import numpy as np

def build_phoneme_bigram(phoneme_sequences, smoothing=0.01):
    """
    Build bigram transition log-probabilities from a list of phoneme sequences.

    Args:
        phoneme_sequences: list of lists of phoneme strings.
        smoothing: Laplace smoothing weight.

    Returns:
        dict: {phoneme_a: {phoneme_b: log_prob}}
    """
    counts = defaultdict(lambda: defaultdict(float))
    vocab  = set()
    for seq in phoneme_sequences:
        vocab.update(seq)
        for a, b in zip(seq[:-1], seq[1:]):
            counts[a][b] += 1.0

    log_probs = {}
    V = len(vocab)
    for a in vocab:
        total = sum(counts[a].values()) + smoothing * V
        log_probs[a] = {
            b: np.log((counts[a][b] + smoothing) / total)
            for b in vocab
        }
    return log_probs

def viterbi_with_lm(log_emission, classes, log_transition, lm_weight=0.3):
    """
    Viterbi decoding combining classifier log-probs with bigram LM.

    Args:
        log_emission: (n_phonemes, n_classes) array of log-probabilities
                      from the classifier (use predict_log_proba).
        classes:      list of class names matching columns of log_emission.
        log_transition: bigram log-prob dict from build_phoneme_bigram.
        lm_weight:    how much to trust the LM vs the classifier (0=ignore LM).

    Returns:
        list of predicted phoneme strings.
    """
    T, K   = log_emission.shape
    vit    = np.full((T, K), -np.inf)
    back   = np.zeros((T, K), dtype=int)
    vit[0] = log_emission[0]

    for t in range(1, T):
        for j, ph_j in enumerate(classes):
            best_score, best_k = -np.inf, 0
            for k, ph_k in enumerate(classes):
                trans = log_transition.get(ph_k, {}).get(ph_j, np.log(1e-6))
                score = vit[t-1, k] + lm_weight * trans
                if score > best_score:
                    best_score, best_k = score, k
            vit[t, j]  = log_emission[t, j] + best_score
            back[t, j] = best_k

    # Backtrack
    path = [int(np.argmax(vit[T-1]))]
    for t in range(T-1, 0, -1):
        path.append(back[t, path[-1]])
    return [classes[i] for i in reversed(path)]

# Build LM from training sequences grouped by word/sentence
# Then replace clf.predict() with viterbi_with_lm(clf.predict_log_proba(), ...)
```

### Expected gain
Medium — especially effective for short words where the classifier is uncertain.
Tune `lm_weight` on a validation set (try 0.1, 0.3, 0.5).

### Effort
Low.

---

## Priority 5 — Multi-Band EEG Features

**Problem it solves:** high-gamma (70–170 Hz) captures articulatory cortex
activity well, but beta suppression (13–30 Hz) correlates with phoneme planning
and theta (4–8 Hz) tracks syllable rate. Adding these gives complementary signal
without changing the classifier.

### Implementation

```python
# In Dutch30Config or a custom feature extractor:
EXTRA_BANDS = {
    'beta':  (13,  30),
    'theta': ( 4,   8),
}

def extract_multiband_hg(raw_eeg, sr=1024, window_ms=30, frameshift_ms=5):
    """
    Extract power envelope for multiple frequency bands.
    Returns concatenated feature matrix: (n_frames, n_channels * n_bands).
    """
    from scipy.signal import butter, filtfilt, hilbert
    import numpy as np

    bands = {'high_gamma': (70, 170), 'beta': (13, 30), 'theta': (4, 8)}
    win   = int(window_ms   / 1000 * sr)
    step  = int(frameshift_ms / 1000 * sr)
    n_frames = (raw_eeg.shape[0] - win) // step + 1

    band_feats = []
    for band_name, (lo, hi) in bands.items():
        b, a    = butter(4, [lo/(sr/2), hi/(sr/2)], btype='band')
        filtered = filtfilt(b, a, raw_eeg, axis=0)
        envelope = np.abs(hilbert(filtered, axis=0))
        frames   = np.array([
            envelope[i*step : i*step + win].mean(axis=0)
            for i in range(n_frames)
        ])
        band_feats.append(frames)

    return np.concatenate(band_feats, axis=1)  # (n_frames, n_ch * n_bands)
```

### Notes
- Feature size triples (3 bands). Increase regularisation accordingly.
- Beta and theta are lower frequency — use longer analysis windows for them
  (e.g. 100 ms for theta vs 30 ms for HG).
- Consider PCA after concatenation to keep feature size manageable.

### Expected gain
Medium — most useful for patients with clear beta/theta modulation.

### Effort
Medium — requires changes to feature extraction, not just the classifier.

---

## Priority 6 — BiLSTM on Raw Frame Sequences

**Problem it solves:** the current pipeline collapses each phoneme to a single
vector (step5c). A BiLSTM instead reads the full sequence of EEG frames and
produces one label per phoneme, capturing temporal dynamics within the phoneme
and cross-phoneme context simultaneously.

### Implementation sketch

```python
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class PhonemeSeqModel(nn.Module):
    def __init__(self, input_size, hidden_size, n_classes, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size,
            num_layers=2, bidirectional=True,
            batch_first=True, dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size * 2, n_classes)

    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths, batch_first=True,
                                      enforce_sorted=False)
        out, _ = pad_packed_sequence(packed, batch_first=True)
        # Use last valid timestep for classification
        idx = (lengths - 1).clamp(min=0)
        last = out[torch.arange(len(lengths)), idx]
        return self.fc(self.dropout(last))

# Input: padded tensor (batch, max_frames, n_channels)
# Skip step5c — pass raw 2D frame arrays per phoneme
# Requires grouping phonemes into batches of similar length
```

### Notes
- Needs 200+ training samples per class per patient to be reliable.
  With current parse rates this may be marginal for some patients.
- Start with a single-layer unidirectional LSTM as a sanity check before
  going bidirectional.
- Use early stopping on a per-patient validation split.

### Expected gain
High if sufficient data per patient; uncertain with current parse rates.

### Effort
High — requires bypassing step5c, custom data loader, training loop, and
hyperparameter search.

---

## Summary Table

| # | Method | Expected Gain | Effort | Prerequisite |
|---|--------|--------------|--------|--------------|
| 1 | MFA phoneme alignment | **High** | Medium | Dutch audio files per sentence |
| 2 | Larger stacking window (order 7–15) | Medium | **Low** | None |
| 3 | CRF sequence model | Medium | **Low** | Phonemes grouped by sentence |
| 4 | Bigram LM decoding | Medium | **Low** | Training phoneme sequences |
| 5 | Multi-band features (beta, theta) | Medium | Medium | Feature extractor changes |
| 6 | BiLSTM on frame sequences | High | **High** | Sufficient per-class samples |

## Recommended Execution Order

1. **MFA alignment** — fixes root cause (noisy boundaries). Run once, reuse forever.
2. **Larger stacking window** — free experiment, add to existing sweep.
3. **CRF + bigram LM** — minimal code, likely gain from sequence modelling alone.
4. **Multi-band features** — if MFA + CRF still leaves room to improve.
5. **BiLSTM** — only after parse rate is above ~80% and data per class is adequate.
