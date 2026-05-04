# MFA Setup — proven workflow for Dutch_30patients

This is the actual end-to-end procedure that produced the working MFA
alignments for P21–P30. No alternatives, no theoretical options — just the
sequence of commands that worked.

The challenge: MFA's `dutch_cv` lexicon is small (~10K words) and misses
many compound words common in the corpus (`afschuren`, `postzegelverzameling`,
`geluidsinstallatie`, `appelsiensap`, etc.). Naïvely running `mfa align` with
only `dutch_cv.dict` left half the audio unaligned. The fix below trains a
G2P from `dutch_cv` itself, generates pronunciations for the OOV compounds,
and re-aligns with the augmented dictionary — which lifts coverage from
~85% to >95% across all patients.

---

## 1. Install MFA

```bash
conda create -n aligner -c conda-forge montreal-forced-aligner python=3.10
conda activate aligner
mfa version    # confirm 3.x
```

## 2. Download Dutch models

```bash
mfa model download acoustic dutch_cv
mfa model download dictionary dutch_cv
```

(There is no `dutch_mfa` acoustic model. `dutch_cv` is the only Dutch
acoustic option in MFA's catalog as of writing.)

Verify:

```bash
mfa models list dictionary       # should show ['dutch_cv']
```

## 3. Export sentence-level audio + transcripts

From the **main Python env** (not aligner), e.g. in a notebook:

```python
from run_pipeline import (
    DEFAULT_RUN_CONFIG, export_sentences_for_mfa,
)
from dutch_30_pipeline import Dutch30Pipeline
from dutch_30_feature_extractor import Dutch30FeatureExtractor
from config import DUTCH_30_PATH
import os

run_config = dict(DEFAULT_RUN_CONFIG)
run_config['patient_range'] = (21, 30)

extractor = Dutch30FeatureExtractor()
pipeline = Dutch30Pipeline(
    dutch30_extractor=extractor, debug_mode=False,
    feature_extraction_method='high_gamma', use_wav2vec=False,
    subtract_baseline=False, use_rms_boundaries=False,
    use_multifeature=False,
)
# ... load step3 checkpoint or run step1/step2 ...

out_dir = os.path.join(DUTCH_30_PATH, 'mfa_input')
for pid in [f'P{i:02d}' for i in range(21, 31)]:
    export_sentences_for_mfa(pid, pipeline, out_dir)
```

Result:

```
Dutch_30patients/mfa_input/
  P21/
    P21_sent000.wav    # 16 kHz mono, peak-normalized
    P21_sent000.lab    # cleaned transcript
    ...
```

## 4. Train a G2P model from `dutch_cv`

This step is what makes the rest possible. Without it we'd have to
hand-write pronunciations for every OOV word. By training G2P on the
existing `dutch_cv.dict` we get a model that emits pronunciations using
exactly `dutch_cv`'s phone inventory.

```bash
conda activate aligner
mfa train_g2p dutch_cv my_dutch_g2p
```

Takes ~22 minutes on a CPU. Output ends with:
```
INFO  Saved model to my_dutch_g2p
```

**Gotcha:** the `.zip` file gets saved in the *current working directory*,
not registered as a named model. Move it into MFA's pretrained-models dir:

```bash
move my_dutch_g2p.zip ^
     %USERPROFILE%\Documents\MFA\pretrained_models\g2p\my_dutch_g2p.zip

mfa models list g2p     # should now show ['my_dutch_g2p']
```

## 5. Build the OOV list

In Python (not aligner env):

```python
import os, glob, re

MFA_INPUT = r'C:\mozg\code\SingleWordProductionDutch\Dutch_30patients\mfa_input'
DICT_PATH = r'C:\Users\<user>\Documents\MFA\pretrained_models\dictionary\dutch_cv.dict'
OUT_PATH  = r'C:\Users\<user>\oovs_found.txt'

# Read words known to dutch_cv
known = set()
with open(DICT_PATH, encoding='utf-8') as f:
    for line in f:
        if line.strip():
            known.add(line.split('\t', 1)[0].strip().lower())

# Collect all words appearing in .lab transcripts
all_words = set()
for lab in glob.glob(os.path.join(MFA_INPUT, '*', '*.lab')):
    with open(lab, encoding='utf-8') as f:
        text = f.read().lower()
    for w in re.findall(r"[a-zÀ-ſ']+", text):
        all_words.add(w)

oovs = sorted(all_words - known)
with open(OUT_PATH, 'w', encoding='utf-8') as f:
    for w in oovs:
        f.write(f"{w}\n")
print(f"OOV count: {len(oovs)}, written to {OUT_PATH}")
```

For Dutch_30patients you'll typically see 50–200 OOV words — mostly
compounds and inflected forms.

## 6. Generate pronunciations for OOVs

Back in the aligner env:

```bash
mfa g2p "C:\Users\<user>\oovs_found.txt" ^
        my_dutch_g2p ^
        "C:\Users\<user>\oov_pronunciations.txt"
```

Quick sanity check (open in any UTF-8-aware editor or in Python — `type` in
cmd renders IPA garbled because of the cp437 codepage):

```python
with open(r'C:\Users\<user>\oov_pronunciations.txt', encoding='utf-8') as f:
    for line in list(f)[:10]:
        print(line.rstrip())
```

Expected entries look like:
```
afschuren                ɑ f s x y ː r ɛ n
postzegelverzameling     p ɔ s t s e ɣ ɛ l v ɛ r z aː m ɛ l ɪ ŋ
geluidsinstallatie       ɣ ɛ l œ y ts ɪ n s t a l aː t s i
```

## 7. Build the augmented dictionary

```bash
copy "C:\Users\<user>\Documents\MFA\pretrained_models\dictionary\dutch_cv.dict" ^
     "C:\Users\<user>\my_dutch_extended.dict"

type "C:\Users\<user>\oov_pronunciations.txt" >> "C:\Users\<user>\my_dutch_extended.dict"
```

Sanity check:
```bash
findstr /B "afschuren" "C:\Users\<user>\my_dutch_extended.dict"
```
should print the new entry.

## 8. Re-align with augmented dictionary

```bash
:: Backup any old TextGrids first
move C:\mozg\code\SingleWordProductionDutch_step2\mfa_output ^
     C:\mozg\code\SingleWordProductionDutch_step2\mfa_output_old

set INP=C:\mozg\code\SingleWordProductionDutch\Dutch_30patients\mfa_input
set OUT=C:\mozg\code\SingleWordProductionDutch_step2\mfa_output

mfa align %INP% ^
          "C:\Users\<user>\my_dutch_extended.dict" ^
          dutch_cv ^
          %OUT% ^
          --clean
```

Note: dictionary is now passed as a **file path** (the augmented one);
acoustic model stays as the named `dutch_cv`. Takes 10–30 min.

## 9. Verify the fix

In Python:

```python
import importlib
import run_pipeline; importlib.reload(run_pipeline)
from run_pipeline import load_mfa_alignments

# Re-check a known previously-broken sentence
mfa = load_mfa_alignments('P27').get(5, [])
print(f"Phonemes: {len(mfa)}")
print(f"Last phoneme ends at: {mfa[-1]['end_s']:.2f}s")
print(f"Sequence: {' '.join(p['phone'] for p in mfa)}")
```

For P27 sentence 5 (transcript `Ik ga de tafel afschuren.`) you should see
~21 phonemes with the last one ending around 7.5s, sequence containing
`ɑ f s x y r ɛ n` at the tail.

To get the full per-patient coverage table:

```python
from collections import defaultdict
import numpy as np

patient_stats = defaultdict(list)
eeg_sr = pipeline.config.eeg_sr
for pid in [f'P{i:02d}' for i in range(21, 31)]:
    wd = pipeline.split_result['word_segments_dict'][pid]
    mfa = load_mfa_alignments(pid)
    for sent_idx, sent_info in enumerate(wd['sentence_list']):
        text = sent_info['text'] if isinstance(sent_info, dict) else sent_info
        if not text or sent_idx not in mfa or not mfa[sent_idx]:
            continue
        orig_dur = (sent_info['stim_end_idx'] - sent_info['stim_start_idx']) / eeg_sr
        patient_stats[pid].append(mfa[sent_idx][-1]['end_s'] / orig_dur)

for pid in sorted(patient_stats):
    covs = patient_stats[pid]
    print(f"{pid}: mean coverage = {np.mean(covs):.1%} "
          f"({sum(1 for c in covs if c < 0.8)} sentences below 80%)")
```

After the augmented-dict alignment, mean coverage should be >95% for all
patients (was ~58% for P23, ~85% mean across patients with the unaugmented
dictionary).

## 10. Re-run the pipeline

```python
from run_pipeline import run_path_b
cached_train, cached_test = run_path_b(pipeline, run_config)
```

`load_mfa_alignments` reads the freshly-generated TextGrids automatically.
Downstream models (boundary detector, CTC, CRF) all see the cleaner labels
and should score better.

## Quick reference card

```bash
# One-time setup (per machine)
conda activate aligner
mfa model download acoustic dutch_cv
mfa model download dictionary dutch_cv
mfa train_g2p dutch_cv my_dutch_g2p
move my_dutch_g2p.zip %USERPROFILE%\Documents\MFA\pretrained_models\g2p\

# Per-corpus (run once per fresh dataset)
python run_pipeline.py --export-mfa --patients 21-30
# (Build oovs_found.txt with Step 5 Python script)
mfa g2p oovs_found.txt my_dutch_g2p oov_pronunciations.txt
copy %USERPROFILE%\Documents\MFA\pretrained_models\dictionary\dutch_cv.dict my_dutch_extended.dict
type oov_pronunciations.txt >> my_dutch_extended.dict
mfa align <input> my_dutch_extended.dict dutch_cv <output> --clean
```
