# WhisperX Setup Instructions

WhisperX is used for forced alignment of sentence audio to word-level timestamps.
It replaces the manual wav2vec cosine-distance + peak-detection boundary method.

## Environment

The notebook runs on **Anaconda Python 3.13** (64-bit, Windows).
WhisperX and its dependencies are installed as **user packages** (`%APPDATA%\Python\Python313\site-packages`),
because Anaconda's base environment is write-protected without admin rights.

Anaconda's own torch (2.9.1+cu130) takes precedence at import time — the user-package
torch must match it exactly to avoid DLL conflicts.

## Installation

Open **Anaconda Prompt** and run the following in order:

### Step 1 — Install whisperx (user install)
```cmd
cd %USERPROFILE%
python -m pip install whisperx --user
```

### Step 2 — Install matching torchaudio (must match torch version exactly)

First check which torch version Anaconda has:
```cmd
python -c "import torch; print(torch.__version__)"
```

Then install the matching torchaudio. For **torch 2.9.1+cu130**:
```cmd
python -m pip install torchaudio==2.9.1 --user --index-url https://download.pytorch.org/whl/cu130
```

If your torch version is different, replace `2.9.1` and `cu130` with your values.
CPU-only builds use `--index-url https://download.pytorch.org/whl/cpu`.

### Step 3 — Verify

Run in a Jupyter notebook cell:
```python
import sys
print(sys.executable)           # should be C:\ProgramData\anaconda3\python.exe
import torch
print(torch.__version__)        # should be 2.9.1+cu130
import torchaudio
print(torchaudio.__version__)   # should match torch version: 2.9.1+cu130
import whisperx
print(whisperx.__version__ if hasattr(whisperx, '__version__') else 'ok')
```

## Package Versions (working configuration)

| Package                  | Version      | Source         |
|--------------------------|--------------|----------------|
| whisperx                 | 3.8.5        | user packages  |
| torch                    | 2.9.1+cu130  | Anaconda base  |
| torchaudio               | 2.9.1+cu130  | user packages  |
| torchvision              | 0.23.0       | user packages  |
| torchcodec               | 0.7.0        | user packages  |
| transformers             | 4.57.1       | user packages  |
| faster-whisper           | 1.2.1        | user packages  |
| ctranslate2              | 4.7.1        | user packages  |
| pyannote-audio           | 4.0.4        | user packages  |
| pyannote-core            | 6.0.1        | user packages  |
| pyannote-database        | 6.1.1        | user packages  |
| pyannote-metrics         | 4.0.0        | user packages  |
| pyannote-pipeline        | 4.0.0        | user packages  |
| pyannoteai-sdk           | 0.4.0        | user packages  |
| pytorch-lightning        | 2.6.1        | user packages  |
| pytorch-metric-learning  | 2.9.0        | user packages  |
| torch-audiomentations    | 0.12.0       | user packages  |
| torch_pitch_shift        | 1.2.5        | user packages  |
| torchmetrics             | 1.9.0        | user packages  |

## Common Errors

### `OSError: Could not load this library: libtorchaudio.pyd`
**Cause:** torchaudio version does not match the torch version being loaded.
**Fix:** Reinstall torchaudio with the exact matching version (Step 2 above).

### `Access is denied` when running pip
**Fix:** Always run `cd %USERPROFILE%` before pip, to move out of the project
directory (which has a `Lib/` folder that confuses Python's path resolution).

### whisperx installed to wrong Python
**Symptom:** whisperx installs to `AppData\Roaming\Python\Python313` but notebook
uses a different Python kernel.
**Fix:** Run `import sys; print(sys.executable)` in the notebook to find the right
Python, then install using that executable:
```cmd
"C:\path\to\python.exe" -m pip install whisperx --user
```

### `Fatal Python error: init_fs_encoding` when running pip from project dir
**Cause:** Project directory has a `Lib/` subfolder that Python mistakes for its stdlib.
**Fix:** `cd %USERPROFILE%` first, then run pip.
