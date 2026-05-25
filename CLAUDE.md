# SingleWordProductionDutch — Step 2

Brain-to-speech decoding pipeline for intracranial EEG (sEEG) recordings from Dutch patients. The goal is to detect phoneme/word boundaries and classify phonemes from neural signals, ultimately enabling speech reconstruction from brain activity.

## Help & setup docs

Long-form how-tos written during development live in **`help_files/`**:

- `help_files/MFA_SETUP.md` — proven Montreal Forced Aligner workflow, including
  the G2P-augmented-dictionary fix that lifted mean coverage from ~85% → >95%.
- `help_files/WHISPERX_SETUP.md` — WhisperX boundary-detection setup (Path A).
- `help_files/PHONEME_BOUNDARY_RECOMMENDATIONS.md` — design notes on boundary
  detection approaches.

When the user asks about MFA, WhisperX, or boundary-detection setup, look there
first. The files at the repo root (`CLAUDE.md` and code) reference these by
relative path so they're always findable.

## Project Overview

Two datasets are supported:
- **Dutch_10patients** — original 10-patient dataset in NWB format
- **Dutch_30patients** — extended 30-patient dataset in NumPy format (primary focus)

The pipeline extracts high-gamma band power (70–170 Hz) and mel-spectrogram features from EEG, detects phoneme/word boundaries using acoustic change detection (including optional wav2vec embeddings), then classifies phonemes with a variety of ML models.

## Architecture

```
Data Input (sEEG + audio)
    ↓
Dutch30FeatureExtractor          # loads raw patient arrays + metadata
    ↓
Dutch30Pipeline                  # orchestrates all steps below
    ├── step1_load_dutch30_data       # load N patients
    ├── step2_split_by_instances      # train/test split
    ├── step3_extract_features        # high-gamma, mel-spec, wav2vec
    │       └── AcousticChangeDetector
    ├── step4_segment_phonemes        # detect boundaries
    ├── step5_classify_phonemes       # train & evaluate classifiers
    │       ├── MarkovPhonemeModel    # RF / LR / MLP (PyTorch)
    │       └── SimplifiedPhonemeModel (Keras)
    ├── step6_resolve_unknowns        # dutch30_step6_resolve_unknowns()
    ├── step7_filter_unknowns         # standalone since recent refactor
    ├── step8_group_phonemes          # group into phoneme categories
    ├── step9_train_and_evaluate      # final training + metrics
    └── step10_visualize_*            # per-patient and group visualizations
```

Note: step numbering in `dutch_30_pipeline.py` and the base `pipeline.py` diverge — check which class a step belongs to before editing.

Key files:
- `dutch_30_pipeline.py` — main pipeline class (`Dutch30Pipeline`)
- `dutch_30_feature_extractor.py` — data loading (`Dutch30FeatureExtractor`)
- `dataset_config.py` — all hyperparameters (`Dutch30Config`)
- `acoustic_change_detector.py` — boundary detection (`AcousticChangeDetector`)
- `markov_phoneme_model.py` — classifiers (`MarkovPhonemeModel`)
- `phoneme_validator.py` — post-classification validation

## Key Configuration (`Dutch30Config`)

| Parameter | Default | Notes |
|-----------|---------|-------|
| `eeg_sr` | 1024 Hz | after downsampling |
| `audio_sr` | 48000 Hz | mic; downsampled to 16000 Hz |
| `high_gamma_low/high` | 70–170 Hz | primary EEG band |
| `window_length` | 30 ms | feature extraction window |
| `frameshift` | 5 ms | |
| wav2vec fps | 50 | `decimate_factor=3` |

Boundary detection thresholds are controlled by k-factors and are configurable; sweep results are stored in `adaptive_threshold_sweep_*.json`.

## Running the Pipeline

```python
from dutch_30_pipeline import Dutch30Pipeline
from dutch_30_feature_extractor import Dutch30FeatureExtractor
from dataset_config import Dutch30Config

config = Dutch30Config()
extractor = Dutch30FeatureExtractor(config=config)
pipeline = Dutch30Pipeline(extractor, config=config, use_wav2vec=True)

pipeline.step1_load_dutch30_data(num_patients=10)
pipeline.step2_split_by_instances(train_fraction=0.8)
pipeline.step3_extract_features()
pipeline.step4_segment_phonemes()
pipeline.step5_classify_phonemes()
```

Analysis scripts (run independently):
- `parse_features_of_30_patients_wav2vec.py` — feature parsing with wav2vec
- `phoneme_clustering_analysis.py` — cluster analysis
- `bigram_analysis.py` / `analyze_bigrams.ipynb`

## Data Layout

Raw data lives under `Dutch_30patients/raw/`. The root data directory is resolved
automatically by `config.py` based on the folder name:
- **This machine:** `C:\mozg\code\SingleWordProductionDutch\Dutch_30patients\raw\`
- **config.py logic:** if folder name contains `step2`, sets `DATA_DIR` to `../SingleWordProductionDutch`
- **In code:** always use `DUTCH_30_PATH` from `config.py` — never hardcode the path:
  ```python
  from config import DUTCH_30_PATH
  raw_dir = os.path.join(DUTCH_30_PATH, "raw")
  raw_audio = np.load(os.path.join(raw_dir, f"{pid}_audio.npy"))
  ```

Expected files per patient:
```
P01_sEEG.npy
P01_audio.npy
P01_stimuli.npy
P01_electrode_locations.csv
```

**Patient IDs:** `P01`–`P30` (zero-padded two digits). Patients are stratified by task type:
- `P01–P10` — mixed (word + sentence)
- `P11–P20` — word production only
- `P21–P30` — sentence production only

**Train/val/test split** is pre-defined in `dutch30_patient_split.json` (24 train / 3 val / 3 test). Do not re-derive splits from scratch — use the JSON.

**Channel exclusions** are defined per-patient in `channel_exclusions.json` as
explicit lists of channel indices to drop:
```json
{"P21": [5, 12, 45], "P22": [3, 99], ...}
```
Loaded by `step3_load_channel_exclusions(exclusions_path)` and stored as
`pipeline.channel_masks[pid] = {'keep_indices': [...], 'exclude_indices': [...],
'n_original': N, 'n_kept': N, 'n_excluded': M}`. Then `apply_channel_exclusions()`
applies the mask to `pipeline.split_result['word_segments_dict']`. Same indices
are reused at inference time via `get_pipeline_channel_mask(pipeline, pid)`.

The lists were curated by visual / statistical inspection of each patient's
recordings (an earlier offline sweep) and are loaded as-is. Treat them as
fixed inputs to the pipeline; **do not regenerate them from data that includes
the test split.**

> Note: `Dutch30Config` declares `channel_outlier_threshold`,
> `channel_flat_threshold`, and `channel_kurtosis_threshold` config values, but
> those are vestigial — no pipeline code reads them. Channel quality is purely
> from the hand-curated JSON above.

## Evaluation Metric — Longest Contiguous Exact Match

The single most informative measure of decoding quality on this dataset is the
**longest contiguous exact phoneme match between prediction and gold, per
sentence, with shift tolerance**. PER and per-position accuracy underweight
"bursty correctness" — a model that nails 5–6 phonemes in a row then drifts
looks similar in PER to one that's diffuse, but only the first is doing real
sequence decoding.

Use it as: max over sentences of the longest run of identical consecutive
phonemes between `pred[i:i+L]` and `gold[j:j+L]`, allowing any shift `(i, j)`.
A length of 5–6 on this dataset is well above chance for two phoneme-prior
streams; length 7+ would be a clear positive result.

**Important: max_run alone is necessary but not sufficient.** A model that
emits the phoneme prior heavily can produce length-5 matches by chance — the
same matches a random shuffle of its predictions would produce. Always pair
max_run with a **rarity-weighted permutation null**: for each matched n-gram,
score it as `−Σ log P(phoneme)` (sum of self-information), compare real total
to a null computed by shuffling predictions across positions while preserving
the marginal. A clean positive result requires both max_run ≥ 5 *and* a
surprise z above ~+5. We've seen models hit max_run=5 with z = −1
(prior-collapse, not decoding) and others hit max_run=5 with z = +13.9
(real decoding); only the second case generalises.

**Critical: never compute it on concatenated sequences across sentences.**
Concatenation lets matches span sentence boundaries in the flat stream — a
chance "ɛ r d eː" at the end of one sentence's prediction can align with
the start of another sentence's gold via the shift loop, producing
spurious 7- and 8-grams that look like decoding wins but are aggregation
artefacts. Earlier in this project we reported 8-gram matches that turned
out to be entirely cross-sentence concatenation noise. Always compute
matches per `(pred_sentence, gold_sentence)` pair; max across sentences,
never over the flat concatenated streams.

The shift-tolerant exact-match function `find_color_matches` in
`e2e_brain_decoder.py` is also permissive in two other ways worth knowing
when reading its output:
- **Equivalences** (r↔l, ɛ↔ɪ, etc.) — counted as "weak" matches; the
  reported length includes them.
- **Introns** — small mismatches inside a span are tolerated up to a limit;
  the reported length is the span, not the count of exact agreements.
A length-6 match from `find_color_matches` may contain only 2–3 contiguous
exact agreements. For a strict reading, use a per-sentence
`longest_contiguous_exact` that requires `pred[i+k] == gold[j+k]` at every
position.

### Implementation details (refined during the BIO-CRF work)

- **Permutation null**: shuffle predictions *within each sentence* (preserves
  sentence-length distribution and the marginal while destroying temporal
  alignment), recompute surprise, repeat ~2000 times. The observed z is
  `(observed − null_mean) / null_std`.
- **Marginal source**: use the **gold-stream phoneme distribution** as the
  marginal, not the prediction stream. This makes z values comparable across
  different model regimes — a gated and ungated model otherwise have different
  prediction marginals and incomparable z scales. Earlier code used the
  pred-stream marginal; switch to gold for any cross-regime comparison.
- **Match length threshold**: only count matches of length ≥ 3 toward the
  observed surprise. Shorter matches happen too often by chance to carry
  information.
- **Shift tolerance**: `shift_max = 3` (allow `|i − j| ≤ 3`) is the default in
  per-sentence `longest_run_with_shift`. Wider tolerance lets the metric
  catch real matches that drift by 1–3 phonemes but starts to produce false
  positives beyond ~5.
- **Reference function**: `score_run` in the BIO-CRF notebook surprise cell;
  the segment-level path uses `find_color_matches` in `e2e_brain_decoder.py`
  with weak-equivalence and intron handling.

## Test Data Leakage — Sacred Rules

Test-data leakage silently inflates reported accuracy. A model that "looks great" because the training pipeline peeked at test statistics will not generalize, and the lift will evaporate the moment it sees genuinely held-out data. Treat the rules below as inviolable. Any change that could let test-side information influence training requires explicit review.

**The discipline:** every statistic, threshold, transform, baseline, or hyperparameter that the model depends on must be derived from training data only. The test set is touched **once**, at evaluation time, and never feeds back into a fit/transform call.

**Spots to keep sacred (verified clean as of 2026-05-01 — do not regress):**

1. **Train/test split (`dutch_30_pipeline.py:step2_split_by_instances`)** — split is per-patient, instance-level (or sentence-presentation-level for P21–P30), seeded deterministically. Do not introduce cross-instance shuffling that mixes train and test.
2. **Baseline subtraction (`_compute_train_baseline` → `_extract_baseline_from_silence`)** — silence baseline must be computed from train sentences only. The helper `_compute_train_baseline` enforces this: it concatenates audio/EEG slices for sentences whose word instances are in `split_result['train'][pid]`, then runs silence detection on that subset. **Never** call `_extract_baseline_from_silence(audio, eeg)` directly on the full patient recording. The threshold inside (`sqrt(mean(audio**2)) * 0.1`) is computed from whatever audio you pass, so leak-safety is the caller's responsibility.
3. **Audio max normalization** — there is no global per-recording max normalization left in the active code path. The remaining audio normalizations in `_segment_by_word_markers` are **per-segment** (one sentence or one word at a time) — these are local statistics and do not leak. If you ever introduce a session-wide max scale, derive it from train segments only and apply it uniformly to train and test.
4. **`StandardScaler` / `PCA`** — fit on train, transform on test (`run_pipeline.py` classifier and CRF paths). Never `fit_transform` a combined train+test array.
5. **Feature stacking (`step5b_stack_features`)** — stacking iterates train and test separately, with instance boundaries detected via `positions[i] == 0` / word/pid changes. Each instance is fully on one side of the split, so temporal context never crosses train↔test. Don't stack across the concatenated dataset.
6. **Class filtering** — when filtering rare classes, count occurrences in train and use that set to filter test. Never count classes in train+test combined.
7. **Hyperparameter sweeps (k-factor, prominence, frameshift, etc.)** — pick best params from a held-out validation split, never from the test set. The 24/3/3 split in `dutch30_patient_split.json` exists for exactly this reason: validate on the 3-patient val set, evaluate once on the 3-patient test set.
8. **Channel exclusions (`channel_exclusions.json`)** — pre-defined and loaded as-is; do not regenerate them with statistics that include test-side data.

**Acceptable but worth flagging in writing:**

- Acausal filtering (`scipy.signal.sosfiltfilt`, `scipy.signal.detrend`, Hilbert) is applied to the full continuous EEG before splitting. This is standard offline-iEEG practice and the per-sample leakage at the split boundary is small, but document it in the methods section.

**Checklist before merging any change that touches features, normalization, or splits:**

- [ ] Does any `.fit(...)` / `.fit_transform(...)` / `np.mean(...)` / `np.std(...)` / `np.max(...)` / `np.min(...)` see test-side rows?
- [ ] Is any threshold or sweep best-value chosen using test-set evaluation results?
- [ ] Does any temporal context window (stacking, smoothing, filter) cross the train/test boundary?
- [ ] Was an "obvious global preprocessing step" added before the split? If so, can it be moved to after?
- [ ] If a function takes audio/EEG directly, does its docstring make clear who is responsible for passing leak-safe data?

When in doubt, recompute on train-only data and check whether the headline metric moves. If it doesn't, you've simply confirmed clean separation. If it does, the previous version was leaking.

## Portability — moving the project between machines

The pipeline is portable; **MFA alignments never need to be regenerated** because the
TextGrids live inside the code repo (see `MFA_OUTPUT_PATH` in `config.py:41`,
which points to `./mfa_output/` next to the code, not into the data dir).

### What lives where

| component | path | size | mandatory? |
|---|---|---|---|
| MFA alignments | `./mfa_output/` (in repo) | ~4 MB | yes — but bundled with code |
| Channel exclusions | `./channel_exclusions.json` | <2 KB | yes (in repo) |
| Patient split | `./dutch30_patient_split.json` | <1 KB | yes (in repo) |
| Raw EEG / audio | `../SingleWordProductionDutch/Dutch_30patients/raw/` | ~10 GB | yes |
| Unique-words dict | `../SingleWordProductionDutch/Dutch_30patients/unique_words.{npy,txt}` | ~300 KB | yes |
| Trained SSL encoders | `./bio_models/*_ssl_encoder*.pt` | ~50–200 MB | optional (can retrain in ~30 min/patient) |
| Old BIO-CRF artefacts | `./bio_models/*_biocrf*.pkl` | ~1.5 GB | skippable — not used by the SSL pipeline |
| `_aux` encoders | `./bio_models/*_ssl_encoder_aux.pt` | — | skip — aux finetune hurts, don't use |

### Required directory layout on the new machine

`config.py` auto-resolves data paths based on folder name: a code dir whose name
contains `"step2"` or `"clean"` expects data one level up. Replicate the
parent-sibling layout:

```
<parent>/
├── SingleWordProductionDutch/          # data (~10 GB)
│   └── Dutch_30patients/
│       ├── raw/                        # P{21..30}_sEEG.npy, _audio.npy, _stimuli.npy
│       └── unique_words.{npy,txt}
└── SingleWordProductionDutch_step2/    # code + mfa_output/ + bio_models/
```

If both folders are siblings, no path edits are needed — `config.py` resolves
`DUTCH_30_PATH` correctly on its own.

### Sanity check after copy

Single cell verifies paths resolve, MFA loads, raw EEG loads, encoder loads:

```python
from config import DUTCH_30_PATH
from run_pipeline import load_mfa_alignments
import os, numpy as np, torch

pid = 'P22'
raw = np.load(os.path.join(DUTCH_30_PATH, 'raw', f'{pid}_sEEG.npy'))
mfa = load_mfa_alignments(pid)
ck  = torch.load(f'bio_models/{pid}_ssl_encoder.pt',
                 map_location='cpu', weights_only=False)
print(f"raw EEG {raw.shape} | MFA {len(mfa)} sentences | encoder n_in={ck['n_in']}")
```

If those three numbers print sensibly, the full state is restored.

### What the new machine does NOT need

- **MFA / Kaldi / G2P install** — alignments are already serialized as TextGrids
- **Wav2vec / HuBERT downloads** — `use_wav2vec=False` is the default for the SSL pipeline
- **Any of the legacy `parse_features_*.ipynb` notebooks** unless you specifically use them
- **NWB tooling (`pynwb`)** — only needed for the Dutch_10patients dataset, not Dutch_30

## Dependencies

No `requirements.txt` exists — install manually. **Version pins matter:**
```
numpy scipy scikit-learn tensorflow pandas matplotlib h5py
torch==2.11          # CPU version; 2.11 is the known-good target
transformers==4.57.1 # was explicitly downgraded — do not upgrade without testing
librosa
pynwb                # NWB format I/O for Dutch_10patients
```

There are no tests, no linting config, and no CI. Do not suggest running `pytest`, `flake8`, or similar — they are not set up.

**Do not use `librosa` in this project.** Although `librosa` is listed above and imports OK, calls into it (notably `librosa.feature.mfcc`) hang silently on this machine in the user's notebook environment. Use `scipy.signal` for STFT / spectrogram work and compute MFCCs manually from a mel filterbank if needed.

## Step 5 Feature Stacking Pipeline

`run_step5` in `dutch_30_pipeline.py` orchestrates:

1. **step5a_filter_by_frame_count** — drops phonemes outside `[min_frames, max_frames]` HG frame range. Use `min_frames=1` to retain short phonemes (default was 4, which silently discarded ~79% of data for sentence patients).
2. **Position reset** — before step5b, `phoneme_positions` is reset to all-zeros for both train and test. This forces step5b to treat each phoneme as its own independent instance rather than part of a running sentence sequence.
3. **step5b_stack_features(model_order=5, step_size=1)** — builds stacked feature vectors using a sliding window of `2*model_order+1 = 11` HG frames (margin = `model_order * step_size = 5`). Without the position reset, phonemes near word edges fall inside the margin and get no stacked vector, losing most of the data.
4. **step5c_collapse_to_phoneme_level** — groups rows by `(phoneme_instance_id, position)`, averaging all frames per phoneme into a single vector. Result: 1 sample per phoneme. Because positions were reset to 0, each phoneme has exactly one `position=0` entry → exactly one output row.

**Zero-padding**: short phonemes (1–3 HG frames) are zero-padded to fill the 11-frame stacking window. This preserves phoneme count at the cost of introducing zeros for missing context frames. Downstream consequences are mild: the mean of the signal is diluted but spatial patterns (which channels are active) are preserved. The classifier sees reduced-magnitude vectors for short phonemes, but relative channel weights are correct.

**Output dimensions**: `n_channels × 11` = e.g. `107 × 11 = 1177` features per phoneme sample.

## `word_segments_dict` Structure

Populated by `step2_split_by_instances`. Accessed as:
`pipeline.split_result['word_segments_dict'][pid]`

**Top-level keys per patient:**
| Key | Content |
|-----|---------|
| `words` | dict keyed by word text → `{'instances': [...]}` |
| `words_list` | ordered list of all word texts |
| `sentence_list` | list of full sentence strings |
| `word_sentence_indices` | per-word mapping to sentence index |
| `word_sentence_texts` | per-word mapping to sentence text |
| `eeg_segments` | sentence-level EEG arrays |
| `audio_segments` | sentence-level audio arrays |
| `spectrogram_segments` | sentence-level spectrograms |
| `participant_id` | patient ID string |
| `baseline` | baseline EEG for this patient |

**Per-word instance keys** (`words[word_text]['instances'][i]`):
| Key | Content |
|-----|---------|
| `eeg_segment` | EEG array for this word `(n_samples, n_channels)` |
| `audio_segment` | audio array for this word `(n_samples,)` |
| `spectrogram_segment` | spectrogram for this word |
| `sentence_idx` | index into `sentence_list` this word belongs to |
| `sentence_text` | full sentence text |
| `word_idx` | position of word within sentence (`None` if not set) |

**Important:** To get sentence-level audio (e.g. for boundary detection tests), use
`word_data['audio_segments'][sentence_idx]`, not word-level `audio_segment`.
To iterate sentences: use `word_data['sentence_list']` and index into `audio_segments`.

## Signal Processing Notes

- Bandpass filter: 70–170 Hz (high-gamma), notch at 50 Hz and 150 Hz
- Hilbert transform for amplitude envelope
- Boundary detection: spectral distance + RMS change + optional wav2vec
- `n_boundaries_needed = len(words) - 1` (word-level boundary count)

### MFA-CRF feature extraction — exact recipe

The MFA-CRF segment-level pipeline (the one that hit z=+13.9 on P22) uses
`extractHG` from `extract_features.py`. It is **not** Hilbert+boxcar; it
is a power-based recipe that replaced the legacy Hilbert path. The exact
steps for one EEG slice:

1. **detrend** (linear) along time axis
2. **70–170 Hz bandpass** — Butterworth-4, applied with `sosfiltfilt`
   (zero-phase)
3. **100 Hz notch** — Butterworth-4 bandstop (98–102 Hz), zero-phase.
   **Note: notches 100 + 150 Hz, NOT 50 + 150 Hz.** The fundamental at
   50 Hz is left intact; only the harmonics are attenuated. For Dutch
   data this is unusual (line noise is 50 Hz) but is what the working
   pipeline does.
4. **150 Hz notch** — Butterworth-4 bandstop (148–152 Hz), zero-phase
5. **x²** — instantaneous power (replaces `|hilbert(x)|`). This is the
   key change from the legacy recipe.
6. **10 Hz lowpass** — Butterworth-4, zero-phase, `smoothing_hz=10.0`.
   This is what does the actual envelope smoothing. The 10 Hz cutoff
   has a clean −24 dB/oct rolloff matched to the phoneme rate (5–10 Hz).
7. **`abs()`** — clean up tiny negatives from zero-phase filter
   roundoff.
8. **15 ms boxcar window-mean at 5 ms frameshift** — samples the
   already-smooth envelope at 200 Hz. The window here is only a sampler;
   the smoothing was done in step 6.
9. **`sqrt`** — back to amplitude-like units (not `log1p`).

Per-phoneme features: for each MFA-aligned phoneme interval, slice the
EEG between `ph['start_s']` and `ph['end_s']`, zero-pad if shorter than
the window length, run through the recipe above, and the result is one
`(T_frames, n_channels)` array per phoneme. Then `step5c_collapse_to_phoneme_level`
averages these to one vector per phoneme.

### Differences from `extract_hg_frames` used in BIO-CRF v2/v3

The frame-level BIO-CRF pipeline uses a different recipe — see
`extract_hg_frames` defined in the BIO-CRF notebooks. The differences:

| Step | MFA-CRF `extractHG` | BIO-CRF `extract_hg_frames` |
|------|---------------------|-----------------------------|
| Notches | 100 + 150 Hz | **50 + 150 Hz** |
| Envelope | `√(LP(x²))` (power+LP) | `\|hilbert(x)\|` (Hilbert magnitude) |
| LP cutoff | **10 Hz** | 12 Hz |
| Window length | 15 ms | 30 ms |
| Compression | `sqrt` | `log1p` |

Both use Butterworth-4 bandpasses and Butterworth-4 lowpasses with
`sosfiltfilt`. Both end with mean-window downsampling to 200 Hz. The
substantive differences are the **envelope method** (power vs Hilbert),
**LP cutoff** (10 vs 12), **window length** (15 ms vs 30 ms), and
**compression** (sqrt vs log1p).

When comparing MFA-CRF and frame-level features, use `extractHG` from
`extract_features.py` directly to match the MFA-CRF features faithfully.

## SSL Phoneme Decoder (v1 Real-Time Recipe)

A self-supervised pretraining + LDA pipeline for per-frame phoneme decoding from
sEEG, developed for real-time use. Main notebook: `ssl_pretrain_encoder.py`
(ten cells with `# %%` markers; copy-paste blocks into Jupyter).

### Recipe

1. **Features** — `extractHG` from `extract_features.py` (the MFA-CRF recipe:
   power → 10 Hz LP → 15 ms boxcar → sqrt; notches 100 + 150 Hz). **Do not**
   use `extract_hg_frames` (Hilbert + log1p + 50/150 notches) — that's the
   BIO-CRF recipe and is a different feature space; mixing them breaks
   downstream reuse.
2. **Encoder** — per-patient causal TCN, 4 blocks (kernel 5, dilations 1/2/4/8),
   hidden 128, learnable mask token in latent space. `CausalConv1d` pads only
   on the left so output[t] depends only on input[≤t] (real-time safe).
3. **SSL pretrain** — masked-frame MSE, 15 % mask in spans of 10 frames
   (≈50 ms ≈ phoneme length), ~80 epochs, AdamW lr=3e-4, cosine LR. Span
   masking is critical — single-frame masks are trivially predicted from
   neighbours and the encoder learns nothing.
4. **No cross-patient sharing** — independent encoder per patient. Cross-patient
   transfer doesn't work on this data scale (verified; consistent with
   BrainBERT and "Brain's Bitter Lesson" cross-subject results being weak).
5. **No aux multi-task finetune** — joint training with silence/speech +
   word-onset heads degrades the SSL representation. Use SSL-only checkpoints
   (`bio_models/{pid}_ssl_encoder.pt`), not the `_aux` variants.
6. **Classifier** — `StandardScaler` + `LinearDiscriminantAnalysis(solver='lsqr',
   shrinkage='auto')` on per-phoneme-averaged 128-d embeddings. Train scaler
   on fit-set only, transform val/test.
7. **Decoder** — 31-frame log-prob smoothing + scalar self-loop Viterbi with
   auto-tuned bonus. **`TARGET_RATIO ≈ 1.7–1.8`** (not 1.0) — this single
   knob accounts for the largest lift in the project.
8. **No boundary constraints** — both word-onset and syllable-onset Viterbi
   modifications inflate raw match rate but degrade chain decoding (Σ(n≥3)
   drops, diversity drops, z drops). Plain scalar Viterbi wins.
9. **No speech gate** — operational no-op in practice (`dropped_silence=0`
   across all patients). The existing `predict_speech_prob` from
   `LDA_on_frames_clean` was trained on `extract_hg_frames` features so its
   inputs don't match the SSL pipeline anyway.

### Headline numbers (10-patient cohort, NW alignment, no oracle boundaries)

| metric | value |
|---|---|
| cohort match (NW) | ~34.5 % |
| cohort z (NW permutation) | +0.96 |
| Σ(n≥3) chains | ~18 |
| length-4 chains | 4, across 4 different patients, all unique patterns |

Baseline (stacked-HG LDA + scalar Viterbi TR=1.0): ~29 % match, ~17 Σ(n≥3),
0 length-4 chains. The lift is real and per-patient diverse.

### Evaluation discipline — read NW results in three columns, not one

NW match rate alone is gameable. Always check three signals together:

1. **`nw_metrics(out)['match_rate']`** — headline
2. **`nw_metrics(out)['z_match']`** — permutation z; catches marginal-spam.
   Negative z means worse than shuffling your own predictions.
3. **n3/n4 chain count + diversity ratio** (`extract_match_ngrams` /
   `diversity_stats` helpers in `ssl_pretrain_encoder.py`):
   - Σ(n≥3) — total length-3+ chains across cohort
   - `uniq_n3 / n3_total` — diversity ratio; <60 % means the model is
     repeating a few common Dutch trigrams (e.g. `/ɛnt/`, `/eːnɛ/`)
   - `top-n3` printout — actually shows the dominant patterns

If match climbs but z drops below ~+1 or diversity drops below ~60 %, the
gain is the marginal trap, not real decoding. We hit this trap multiple
times in the original exploration; treat it as the default failure mode.

### Negative findings — do not retry without new ideas

These were systematically tested and ruled out on this dataset/pipeline:

| ruled out | notes |
|---|---|
| cross-patient SSL transfer | per-patient electrode coverage + small per-patient data |
| multi-task aux finetune (speech + word + syllable heads on encoder) | overwrites SSL representation |
| phoneme-onset detection from EEG | physiologically unrecoverable — vSMC encodes coarticulated gestures (Bouchard 2013) |
| token bigram in Viterbi | bigrams over phoneme tokens (not frame stream) discourage self-loops; broken |
| frame bigram in Viterbi | no cohort lift over scalar bonus |
| multiband amplitude (HG+LG, HG+LG+theta, HG+beta) | encoder underfits wider inputs at per-patient scale; HG-only equals or beats all variants |
| theta phase as syllable signal | ITC < 0.11 on all 10 patients — no usable phase locking in vSMC |
| word-onset Viterbi constraint | helps some patients but breaks chains overall |
| syllable-onset Viterbi constraint (oracle and deployable) | inflates match via more predictions, reduces chain count and diversity |
| combined word + syllable constraints | same trap, larger |
| CTC head on random-init encoder | mode-collapses to top-5 phonemes (per earlier work) — not retried with SSL backbone, may be worth revisiting |
| SSL mask fraction > 15 % | degrades downstream; causal encoder + small data can't reconstruct larger gaps |
| SSL epochs ≫ 80 | reconstruction MSE keeps dropping but downstream match degrades (pretext-vs-downstream tension) |

### Methodology gotchas learned

- **`TARGET_RATIO` is the single most impactful Viterbi knob.** Auto-tuner uses
  it to scale predicted count vs val gold. Default 1.0 underpredicts;
  1.7–1.8 hits the sweet spot. Above ~2.0 the auto-tuner pushes bonus to ~0
  and Viterbi predicts the marginal everywhere — match keeps climbing but z
  goes negative.
- **`ONSET_WEIGHT` for boundary constraints must scale with onset density.**
  Word onsets (rare): ow≈2.0. Syllable onsets (3–5× denser): ow≈0.5–1.0.
  Mismatched ow gives spurious negative results (we initially "ruled out"
  syllable constraints using word-onset ow=2.0 — wrong).
- **Imported constants from `LDA_on_frames_clean` must be re-bound in the
  notebook's globals to take effect on functions defined locally.** Module-
  level rebinding (`L.X = ...`) doesn't propagate to functions that imported
  `X` by name into the notebook namespace.
- **SSL pretext loss is anti-correlated with downstream phoneme decoding past
  ~80 epochs.** More reconstruction training → worse embeddings for LDA on
  this data scale. Range 30–80 ep is the usable zone; per-patient optimum
  varies (P22 ~30, P30 ~60) but the cohort-aggregate optimum is ~80 because
  late-peak patients win more from extra training than early-peak patients
  lose.
- **Bigrams operate on the actual stream Viterbi sees.** Frame-level Viterbi
  needs frame-aligned bigram (self-transitions dominate). A token-level
  bigram from MFA actively discourages self-loops and breaks decoding.
- **Seed variance is ~2 pp on individual patients.** P22 went from match 26.0 %
  to 28.0 % just by swapping SSL seed 0 → 1. Don't over-interpret single-
  patient deltas below ~3 pp; don't lock recipes based on one patient.
- **Reload checkpoints with `weights_only=False`** — they bundle numpy arrays
  (`mu`, `sd`) alongside the torch state dict. PyTorch 2.6 default of
  `weights_only=True` rejects these.

### Files

- `ssl_pretrain_encoder.py` — main notebook, ten cells with `# %%` markers
- `bio_models/{pid}_ssl_encoder.pt` — per-patient encoder + standardisation
  stats (HG-only, ~80 epochs, 15 % mask). **Use these.**
- `bio_models/{pid}_ssl_encoder_aux.pt` — encoder after aux finetune.
  **Do not use** — aux finetune hurts downstream.
- `bio_models/{pid}_word_onset_head.pt`, `_syl_onset_head.pt` — auxiliary
  detectors; usable as features but their Viterbi-constraint application
  didn't lift cohort numbers.

## Codebase Conventions

- `DebugMixin` provides logging via a `debug()` method and a `DEBUG_MODE` flag; `Dutch30Pipeline` inherits only from `DebugMixin` (not from other pipeline classes)
- Archived/experimental code goes in `archive/` — do not modify files there
- Results (metrics, predictions) go in `results/`
- Visualizations and report figures go in `report/`
- Notebooks in the root are reference/exploratory; `existing_code.ipynb` is the main reference notebook

## Known Issues & Gotchas

- **Wav2vec reproducibility** — there is an unresolved CPU/GPU non-determinism issue with wav2vec embeddings. An earlier attempt to force CPU (`d9da928`) was reverted (`89c3512`) as it was not the root cause. Results may vary slightly across machines.
- **Word boundary detection is actively being refined** — the `n_boundaries_needed = len(words) - 1` fix and configurable `word_threshold_factors` are recent changes. There is also an equally-spaced fallback when detected boundaries are fewer than needed.
- **Incomplete checkpoint logic** — `pipeline.py:1516` has a `TODO`: accuracy determination from saved checkpoint data is not yet implemented; it falls back to using the newest checkpoint.
- **`transformers` version is sensitive** — the library was explicitly downgraded to `4.57.1`. Upgrading may break wav2vec feature extraction.
