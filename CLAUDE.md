# SingleWordProductionDutch — Step 2

Brain-to-speech decoding pipeline for intracranial EEG (sEEG) recordings from Dutch patients. The goal is to detect phoneme/word boundaries and classify phonemes from neural signals, ultimately enabling speech reconstruction from brain activity.

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

Raw data should be placed under `Dutch_30patients/raw/`:
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

**Channel exclusions** are defined per-patient in `channel_exclusions.json`. Channel quality is filtered automatically using these thresholds (`Dutch30Config`):
- `channel_outlier_threshold`: std > median × 3.0
- `channel_flat_threshold`: std < median × 0.1
- `channel_kurtosis_threshold`: kurtosis > median × 5.0
- `min_channels_to_keep`: 20

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
