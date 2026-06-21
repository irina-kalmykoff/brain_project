# Phoneme-Level Decoding from Intracranial EEG (Dutch, P21–P30)

Brain-to-speech decoding from intracranial EEG (sEEG). Two phoneme decoders run on
the same high-gamma features:

- **MFA-CRF** — Montreal Forced Aligner segmentation + engineered phonetic
  features + a linear-chain Conditional Random Field; decoded on gold boundaries.
- **SSL-LDA** — a per-patient self-supervised TCN encoder → per-frame LDA →
  free-running self-loop Viterbi.

This README covers logistics only: how to install, where data goes, how the MFA
grids are used, how to run, and what to expect on a new machine.

---

## 1. Requirements & install (from scratch on a new machine)

Python **3.11** (the pinned wheels target it). Create a clean environment and
install the pins:

    python -m venv .venv
    # Windows:  .venv\Scripts\activate
    # macOS/Linux:  source .venv/bin/activate
    pip install --upgrade pip

    # torch is pinned to a CPU build (2.1.0+cpu), which lives on the PyTorch index:
    pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

Notes:
- `requirements.txt` is a full environment freeze **plus** three packages the code
  imports but the freeze omitted: **`tgt`** (reads MFA `.TextGrid` files),
  **`sklearn-crfsuite`** and **`python-crfsuite`** (the CRF decoder). Without these
  the MFA-CRF arm and MFA loading fail with `ImportError`.
- `transformers` is pinned (4.37.0) and `torchaudio` is only needed for the
  optional wav2vec boundary-detection path (`use_wav2vec=True`); the SSL and CRF
  pipelines run with `use_wav2vec=False` (the default). `transformers` was
  deliberately downgraded — do not upgrade without re-testing the wav2vec path.
- `pynwb` is only needed for the legacy 10-patient NWB dataset, not P21–P30.
- **Do not call `librosa`** — it is installed but hangs on some machines; use
  `scipy.signal` instead.
- There is no test suite, linter, or CI — do not run `pytest`/`flake8`.

## 2. Directory layout (data lives one level up)

**Data source:** the raw sEEG/audio is not included in this repository. It comes
from the public dataset described here:
https://www.biorxiv.org/content/10.1101/2024.11.29.626019v1 — obtain it and place
it in the sibling layout below.

`config.py` auto-resolves data paths from the folder name: a code directory whose
name contains `step2` or `clean` expects the data as a **sibling** directory.
Replicate this on the new machine:

```
<parent>/
├── SingleWordProductionDutch/            # data (~10 GB, NOT in this repo)
│   └── Dutch_30patients/
│       ├── raw/                           # P{21..30}_sEEG.npy, _audio.npy, _stimuli.npy, _electrode_locations.csv
│       └── unique_words.{npy,txt}
└── SingleWordProductionDutch_step2/       # THIS repo (code + mfa_output/ + bio_models/)
```

If the two folders are siblings, no path edits are needed — always use
`DUTCH_30_PATH` from `config.py`, never hardcode paths.

### What lives where / what to copy

| component | path | size | mandatory? |
|---|---|---|---|
| MFA alignments | `./mfa_output/` | ~4 MB | yes (committed in this repo) |
| Channel exclusions | `./channel_exclusions.json` | <2 KB | yes (committed) |
| Patient split | `./dutch30_patient_split.json` | <1 KB | yes (committed) |
| Raw EEG / audio | `../SingleWordProductionDutch/Dutch_30patients/raw/` | ~10 GB | yes |
| Unique-words dict | `../SingleWordProductionDutch/Dutch_30patients/unique_words.{npy,txt}` | ~300 KB | yes |
| Trained SSL encoders | `./bio_models/*_ssl_encoder.pt` | ~50–200 MB | optional (retrain ~30 min/patient) |
| Old BIO-CRF artefacts | `./bio_models/*_biocrf*.pkl` | ~1.5 GB | skippable — unused by the SSL pipeline |
| `_aux` encoders | `./bio_models/*_ssl_encoder_aux.pt` | — | skip |

### What the new machine does NOT need

- **MFA / Kaldi / G2P install** — alignments are already serialized as TextGrids.
- **wav2vec / HuBERT downloads** — `use_wav2vec=False` is the default.
- **Legacy `parse_features_*.ipynb` notebooks** — not required to run the decoders.
- **NWB tooling (`pynwb`)** — only for the 10-patient dataset, not P21–P30.

## 3. Data details

- **Patient IDs:** `P01`–`P30` (zero-padded). Stratified by task: `P01–P10` mixed
  (word + sentence), `P11–P20` word-only, `P21–P30` sentence-only. This work uses
  **P21–P30**.
- **Per-patient files** (under `Dutch_30patients/raw/`): `{PID}_sEEG.npy`,
  `{PID}_audio.npy`, `{PID}_stimuli.npy`, `{PID}_electrode_locations.csv`.
- **Train/val/test split** is fixed in `dutch30_patient_split.json` (24 train /
  3 val / 3 test). Use the JSON; do not re-derive splits from scratch.
- **Channel exclusions** are per-patient lists of channel indices to drop, in
  `channel_exclusions.json` (e.g. `{"P21": [55, 85, 109], ...}`). They are
  hand-curated and loaded as-is — treat them as fixed inputs; do not regenerate
  them from data that includes the test split.

## 4. Configuration quick reference (`Dutch30Config`)

| parameter | default | notes |
|---|---|---|
| `eeg_sr` | 1024 Hz | after downsampling |
| `audio_sr` | 48000 Hz | mic; downsampled to 16000 Hz |
| `high_gamma_low/high` | 70–170 Hz | primary EEG band |
| `window_length` | 30 ms | feature-extraction window |
| `frameshift` | 5 ms | |
| wav2vec fps | 50 | `decimate_factor=3` |

## 5. MFA grids — how they are used

The Montreal Forced Aligner output is **committed to this repo** under
`mfa_output/` and read directly — you **do not need to install MFA/Kaldi or re-run
alignment** to run the decoders. One TextGrid per sentence:

```
mfa_output/<PID>/<PID>_sent<NNN>.TextGrid     e.g. mfa_output/P22/P22_sent000.TextGrid
```

They are parsed by `tgt` via `run_pipeline.load_mfa_alignments(pid)`, which returns
`{sentence_index: [{'phone', 'start_s', 'end_s'}, ...]}` from the `phones` tier.

To **regenerate** the grids (only if you change the audio or lexicon), follow
`help_files/MFA_SETUP.md` (G2P-augmented-dictionary workflow, >95% phone coverage).
Regenerated `.TextGrid` files must land in the same `mfa_output/<PID>/` layout.

## 6. Help / setup docs

Longer-form how-tos live in `help_files/`:

- `help_files/MFA_SETUP.md` — Montreal Forced Aligner workflow.
- `help_files/WHISPERX_SETUP.md` — WhisperX boundary-detection setup.
- `help_files/PHONEME_BOUNDARY_RECOMMENDATIONS.md` — boundary-detection design notes.

## 7. Quick sanity check

After copying data + code into the sibling layout, this should print sensible
numbers (paths resolve, MFA loads, raw EEG loads, an encoder loads):

```python
from config import DUTCH_30_PATH
from run_pipeline import load_mfa_alignments
import os, numpy as np, torch

pid = 'P22'
raw = np.load(os.path.join(DUTCH_30_PATH, 'raw', f'{pid}_sEEG.npy'))
mfa = load_mfa_alignments(pid)
ck  = torch.load(f'bio_models/{pid}_ssl_encoder.pt', map_location='cpu', weights_only=False)
print(f"raw EEG {raw.shape} | MFA {len(mfa)} sentences | encoder n_in={ck['n_in']}")
```

## 8. Running the decoders

- **SSL-LDA** (consistent split): `ssl_lda_frames_consistent_test_split.ipynb`
  (or the `.py` mirror). Retrain encoders with `ssl_pretrain_encoder.py`
  (~30 min/patient on CPU); pretrained checkpoints in `bio_models/` are used by
  default.
- **MFA-CRF** and the report tables/ablations: `report.py` / `MFA_CRF.py`.
- General pipeline (loading, splitting, features): `dutch_30_pipeline.py`.

## 9. Reproducibility — results depend on hardware and are not bit-exact

**Numbers will vary slightly from machine to machine.** Random seeds are fixed
where it matters (data split, SSL pretraining; permutation tests use `seed=0`,
`nperm=2000`), but exact metric values still depend on:

- **CPU/BLAS and threading** — different BLAS backends, core counts, and `torch`
  thread settings change floating-point reduction order, shifting SSL training and
  downstream scores.
- **Library versions** — `torch`, `numpy`, and `scikit-learn` builds differ across
  platforms even at the same pins.
- **SSL seed variance** — per-patient match rate moves by ≈1–2 percentage points
  just from the pretraining seed; do not over-interpret single-patient deltas
  below ~3 pp.
- **wav2vec non-determinism** — the optional wav2vec path has a known, unresolved
  CPU/GPU non-determinism issue (off by default).

Expect small numerical drift (typically ≈1–2 pp on per-patient metrics); treat any
single reported number as hardware-dependent, not exactly reproducible.
Permutation/ablation *p*-values are reproducible **given the same saved
predictions** — re-run the analysis on saved predictions for exact figures, and
re-train only when you intend to.
