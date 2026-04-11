# MFA Setup Guide

Montreal Forced Aligner (MFA) setup for the Dutch brain-to-speech pipeline.
MFA aligns phonemes to audio at the frame level, replacing the wav2vec/WhisperX
boundary detection (Path A) with precise per-phoneme timestamps (Path B).

---

## 1. Install MFA

MFA requires its own conda environment (Python 3.10, not 3.13).

```bash
# Create dedicated environment
conda create -n aligner -c conda-forge montreal-forced-aligner python=3.10
conda activate aligner

# Verify
mfa version
```

## 2. Download Dutch acoustic model and dictionary

```bash
conda activate aligner

# Acoustic model (trained on Common Voice Dutch)
mfa model download acoustic dutch_cv

# Pronunciation dictionary (same phone set)
mfa model download dictionary dutch_cv

# Verify downloads
mfa model list acoustic
mfa model list dictionary
```

Expected phone set (43 phones): `a aː b c d eː f h i iː j k l m n oː p r s t
u uː v w x y yː z øː ŋ œ ɑ ɔ ɛ ɣ ɥ ɪ ʃ ʋ ʏ` (no schwa `ə`).

## 3. Export audio files from the pipeline

Run this from the project's main Python environment (not the aligner env).

### Option A: Command line

```bash
python run_pipeline.py --export-mfa --patients 21-30
```

### Option B: From Jupyter notebook

```python
from run_pipeline import (
    DEFAULT_RUN_CONFIG, export_sentences_for_mfa, clean_text_for_mfa,
)
from dutch_30_pipeline import Dutch30Pipeline
from dutch_30_feature_extractor import Dutch30FeatureExtractor
from config import DUTCH_30_PATH
import os, pickle

run_config = dict(DEFAULT_RUN_CONFIG)
run_config['patient_range'] = (21, 30)

# Create pipeline and load split_result
extractor = Dutch30FeatureExtractor()
pipeline = Dutch30Pipeline(
    dutch30_extractor=extractor,
    debug_mode=False,
    feature_extraction_method='high_gamma',
    use_wav2vec=False,
    subtract_baseline=False,
    use_rms_boundaries=False,
    use_multifeature=False,
)

# Load from checkpoint if available
pr = run_config['patient_range']
ckpt = f'checkpoint_after_step3_P{pr[0]:02d}-P{pr[1]:02d}.pkl'
if os.path.exists(ckpt):
    with open(ckpt, 'rb') as f:
        state = pickle.load(f)
    pipeline.split_result      = state['split_result']
    pipeline.patient_data      = state['patient_data']
    pipeline.patient_baselines = state['patient_baselines']
else:
    pipeline.step1_load_dutch30_data(patient_range=pr)
    pipeline.step2_split_by_instances()

# Export
out_dir = os.path.join(DUTCH_30_PATH, 'mfa_input')
for pid in [f'P{i:02d}' for i in range(pr[0], pr[1] + 1)]:
    export_sentences_for_mfa(pid, pipeline, out_dir)
```

### Output structure

```
Dutch_30patients/
  mfa_input/
    P21/
      P21_sent000.wav    # 16 kHz mono, peak-normalized
      P21_sent000.lab    # lowercase transcript, no punctuation
      P21_sent002.wav    # even indices = real sentences
      P21_sent002.lab    # odd indices = rest intervals (skipped)
      ...
    P22/
      ...
```

Each patient produces ~100 sentence pairs. Rest intervals (odd indices) are
automatically skipped. The `.lab` files contain cleaned text (lowercase, no
punctuation, no digits).

## 4. Validate alignment (check for OOV words)

```bash
conda activate aligner

# On Windows — adjust paths as needed
mfa validate C:\mozg\code\SingleWordProductionDutch\Dutch_30patients\mfa_input dutch_cv dutch_cv
```

If OOV words are reported, see [Section 6: Handling OOV words](#6-handling-oov-words).

## 5. Run alignment

```bash
conda activate aligner

mfa align \
    C:\mozg\code\SingleWordProductionDutch\Dutch_30patients\mfa_input \
    dutch_cv \
    dutch_cv \
    C:\mozg\code\SingleWordProductionDutch\Dutch_30patients\mfa_output \
    --clean
```

- `--clean` removes cached data from previous runs (recommended on re-runs).
- Takes ~5-15 minutes for 10 patients (1000 sentences).
- Some sentences may fail to align (typically 0-20 per patient). This is normal;
  the pipeline handles missing TextGrids gracefully.

### Output structure

```
Dutch_30patients/
  mfa_output/
    P21/
      P21_sent000.TextGrid    # Praat TextGrid with 'phones' and 'words' tiers
      P21_sent002.TextGrid
      ...
    P22/
      ...
```

### Verify coverage

```bash
python run_pipeline.py --mfa-coverage --patients 21-30
```

Or from notebook:

```python
from run_pipeline import mfa_coverage_summary, DEFAULT_RUN_CONFIG
mfa_coverage_summary(DEFAULT_RUN_CONFIG)
```

Expected output (P21-P30):

```
P21:  90/100 aligned (90%)  phones: 2682
P22:  97/100 aligned (97%)  phones: 3074
...
P30: 100/100 aligned (100%) phones: 2941
```

## 6. Handling OOV words

If `mfa validate` reports OOV (out-of-vocabulary) words, you need to generate
pronunciations for them and add them to the dictionary.

### Install espeak-ng (G2P backend)

1. Download from https://github.com/espeak-ng/espeak-ng/releases
2. Install to default path (`C:\Program Files\eSpeak NG` on Windows)

### Generate pronunciations

```python
import os
os.environ['PATH'] += r';C:\Program Files\eSpeak NG'
os.environ['PHONEMIZER_ESPEAK_LIBRARY'] = r'C:\Program Files\eSpeak NG\libespeak-ng.dll'

from phonemizer import phonemize

# Example: get pronunciation for OOV words
oov_words = ['schaatsen', 'vegetarisch', 'kleurpotloden']
pronunciations = phonemize(
    oov_words, language='nl', backend='espeak',
    with_stress=False, strip=True,
)
print(pronunciations)
# ['sxaːtsən', 'veːɣətaːris', 'kløːrpɔtloːdən']
```

### Map espeak phones to MFA dutch_cv phone set

espeak outputs IPA which doesn't always match the dutch_cv acoustic model.
Apply this mapping:

```python
PHONE_MAP = {
    'ə': 'ɛ',     # no schwa in dutch_cv — map to short e
    'ø': 'øː',    # dutch_cv only has long version
    'tʲ': 't',    # no palatalized stops
    'ɔː': 'ɔ',    # dutch_cv only has short version
    'ɪː': 'iː',   # no long lax i
    'ɲ': 'n',     # no palatal nasal
    'ɵ': 'ʏ',     # map to closest rounded vowel
    'ɾ': 'r',     # tap → trill
    'ʊ': 'u',     # no short u
    'ʌ': 'ɑ',     # map to open back vowel
    'e': 'ɛ',     # no short e
    '(': '',       # remove parentheses
    ')': '',
}
```

### Create extended dictionary

```python
# Read original dutch_cv dictionary
# (find path with: mfa model inspect dictionary dutch_cv)
import shutil

# Copy original dict and append OOV entries
original_dict = r'path\to\dutch_cv.dict'  # from mfa model inspect
extended_dict = os.path.join(DUTCH_30_PATH, 'mfa_input', 'dutch_cv_extended.dict')
shutil.copy2(original_dict, extended_dict)

with open(extended_dict, 'a', encoding='utf-8') as f:
    for word, phones in oov_pronunciations.items():
        f.write(f'{word}\t{phones}\n')
```

### Re-run alignment with extended dictionary

```bash
mfa align \
    C:\mozg\code\SingleWordProductionDutch\Dutch_30patients\mfa_input \
    dutch_cv \
    C:\mozg\code\SingleWordProductionDutch\Dutch_30patients\mfa_input\dutch_cv_extended.dict \
    C:\mozg\code\SingleWordProductionDutch\Dutch_30patients\mfa_output \
    --clean
```

Note: the second argument is the acoustic model (stays `dutch_cv`), the third is
now the path to the extended dictionary file instead of `dutch_cv`.

## 7. Run the pipeline with MFA (Path B)

### Command line

```bash
python run_pipeline.py --mfa --patients 21-30
```

### Jupyter notebook

```python
from run_pipeline import (
    DEFAULT_RUN_CONFIG, run_path_b, run_from_config, count,
)
from dutch_30_pipeline import Dutch30Pipeline
from dutch_30_feature_extractor import Dutch30FeatureExtractor

run_config = dict(DEFAULT_RUN_CONFIG)

extractor = Dutch30FeatureExtractor()
pipeline = Dutch30Pipeline(
    dutch30_extractor=extractor,
    debug_mode=False,
    feature_extraction_method=run_config['feature_extraction_method'],
    use_wav2vec=False,
    subtract_baseline=run_config['subtract_baseline'],
    use_rms_boundaries=False,
    use_multifeature=False,
)

cached_train, cached_test = run_path_b(pipeline, run_config)
name, params, results = run_from_config(pipeline, run_config)
```

## 8. Diagnostics

### Check phoneme loss

```bash
python run_pipeline.py --diagnose-mfa --patients 21-30
```

Shows per-patient breakdown of:
- Phonemes in TextGrids (total)
- Sentences without TextGrids (MFA alignment failures)
- Phonemes too short for feature extraction (< 30ms, zero-padded in Path B)

### Important notes

- **Viterbi decoding**: disable when using MFA (`run_config['use_viterbi'] = False`).
  The Viterbi transition model is built from the IPA phonetic dictionary, which
  uses different phone symbols than MFA's dutch_cv set. Using Viterbi with MFA
  labels degrades accuracy below chance.
- **Phone set**: MFA dutch_cv has 43 phones. The pipeline's PhoneticDictionary
  uses IPA with schwa (`ə`) and other symbols not in dutch_cv. Step 6
  (`resolve_unknowns`) handles some of this, but a proper phone-set mapping
  would improve results.
- **Step 3 checkpoint**: MFA only needs the step 3 checkpoint (split_result +
  patient_data). If it exists from a previous Path A run, MFA will reuse it.
  No need to regenerate.

## Quick reference

```
conda activate aligner

# Full workflow (one-time per patient set):
mfa model download acoustic dutch_cv
mfa model download dictionary dutch_cv
python run_pipeline.py --export-mfa --patients 21-30
mfa validate <input_dir> dutch_cv dutch_cv
mfa align <input_dir> dutch_cv dutch_cv <output_dir> --clean

# Then run pipeline:
python run_pipeline.py --mfa --patients 21-30
```
