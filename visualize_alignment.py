"""Visualize word and phoneme alignment on top of the audio waveform.

Single figure with 3 rows:
  1. Spectrogram with word labels
  2. Audio waveform with word boundaries (red dashed)
  3. Audio waveform with word + phoneme boundaries

Usage (notebook):
    from visualize_alignment import plot_sentence_alignment, plot_neural_alignment

    plot_sentence_alignment(pipeline, 'P24', sentence_text='donald trump')
    plot_neural_alignment(pipeline, 'P24', sentence_text='donald trump')
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from config import DUTCH_30_PATH


# ═══════════════════════════════════════════════════════════════════════════════
#  COLORS
# ═══════════════════════════════════════════════════════════════════════════════

PHONE_COLORS = {
    'stops':      '#e74c3c',
    'fricatives': '#e67e22',
    'nasals':     '#f1c40f',
    'liquids':    '#2ecc71',
    'glides':     '#1abc9c',
    'i-type':     '#3498db',
    'u-type':     '#9b59b6',
    'e-type':     '#2980b9',
    'o-type':     '#8e44ad',
    'a-type':     '#d35400',
    'schwa':      '#95a5a6',
    'diph':       '#16a085',
    'other':      '#bdc3c7',
}

PHONE_TO_GROUP = {
    'p': 'stops', 'b': 'stops', 'd': 'stops', 't': 'stops', 'k': 'stops', 'g': 'stops',
    'f': 'fricatives', 'v': 'fricatives', 's': 'fricatives', 'z': 'fricatives',
    'x': 'fricatives', 'ɣ': 'fricatives', 'h': 'fricatives', 'ɦ': 'fricatives',
    'ʃ': 'fricatives', 'ʒ': 'fricatives', 'χ': 'fricatives', 'ʋ': 'fricatives',
    'm': 'nasals', 'n': 'nasals', 'ŋ': 'nasals',
    'l': 'liquids', 'r': 'liquids',
    'j': 'glides', 'w': 'glides', 'ɥ': 'glides',
    'i': 'i-type', 'iː': 'i-type', 'ɪ': 'i-type', 'y': 'i-type', 'yː': 'i-type', 'ʏ': 'i-type',
    'u': 'u-type', 'uː': 'u-type',
    'e': 'e-type', 'eː': 'e-type', 'ɛ': 'e-type', 'ɛː': 'e-type',
    'øː': 'e-type', 'ø': 'e-type', 'œ': 'e-type',
    'o': 'o-type', 'oː': 'o-type', 'ɔ': 'o-type', 'ɔː': 'o-type',
    'a': 'a-type', 'aː': 'a-type', 'ɑ': 'a-type', 'ɑː': 'a-type',
    'ə': 'schwa',
    'ɛi': 'diph', 'ɑu': 'diph', 'œy': 'diph',
}


def _phone_color(phone):
    group = PHONE_TO_GROUP.get(phone, 'other')
    return PHONE_COLORS.get(group, PHONE_COLORS['other'])


def _load_mfa_alignment(pid, sent_idx, mfa_dir=None):
    import tgt

    if mfa_dir is None:
        local = os.path.join(os.path.dirname(__file__), 'mfa_output')
        if os.path.isdir(local):
            mfa_dir = local
        else:
            mfa_dir = os.path.join(DUTCH_30_PATH, 'mfa_output')

    tg_path = os.path.join(mfa_dir, pid, f'{pid}_sent{sent_idx:03d}.TextGrid')
    if not os.path.exists(tg_path):
        return None, None

    tg = tgt.io.read_textgrid(tg_path)
    word_tier = tg.get_tier_by_name('words')
    phone_tier = tg.get_tier_by_name('phones')

    words = []
    for ann in word_tier.annotations:
        text = ann.text.strip()
        if text:
            words.append({'text': text, 'start': ann.start_time, 'end': ann.end_time})

    phones = []
    for ann in phone_tier.annotations:
        text = ann.text.strip()
        if text and text not in ('sp', 'sil', 'spn'):
            mid = (ann.start_time + ann.end_time) / 2
            word = ''
            for w in words:
                if w['start'] <= mid <= w['end']:
                    word = w['text']
                    break
            phones.append({
                'text': text, 'start': ann.start_time,
                'end': ann.end_time, 'word': word,
            })

    return words, phones


def _find_sentence(pipeline, pid, sentence_text=None, sent_idx=None):
    word_data = pipeline.split_result['word_segments_dict'].get(pid)
    if word_data is None:
        return None, None, None

    sentence_list = word_data['sentence_list']

    if sent_idx is not None:
        sent = sentence_list[sent_idx]
        text = sent['text'] if isinstance(sent, dict) else sent
        return sent_idx, sent, text

    if sentence_text is not None:
        for i, sent in enumerate(sentence_list):
            text = sent['text'] if isinstance(sent, dict) else sent
            if text and sentence_text.lower() in text.lower():
                return i, sent, text

    return None, None, None


def plot_sentence_alignment(pipeline, pid, sentence_text=None, sent_idx=None,
                             audio_sr=48000):
    """Single figure, 3 rows:
      Row 1: Spectrogram with word labels
      Row 2: Audio waveform with word boundaries (red dashed)
      Row 3: Audio waveform with phoneme boundaries (colored dashed)

    Args:
        pipeline: Dutch30Pipeline with split_result loaded.
        pid: patient ID, e.g. 'P24'.
        sentence_text: substring to match (e.g. 'donald trump').
        sent_idx: sentence index. Overrides sentence_text.
        audio_sr: audio sample rate (default 48000).
    """
    idx, sent, text = _find_sentence(pipeline, pid, sentence_text, sent_idx)
    if idx is None:
        print(f"Sentence not found for {pid}")
        return

    # Load raw audio
    audio_path = os.path.join(DUTCH_30_PATH, 'raw', f'{pid}_audio.npy')
    if not os.path.exists(audio_path):
        print(f"Audio not found: {audio_path}")
        return

    raw_audio = np.load(audio_path)
    start_sample = sent['stim_start_idx']
    end_sample = sent['stim_end_idx']

    eeg_sr = pipeline.config.eeg_sr
    audio_start = int(start_sample * audio_sr / eeg_sr)
    audio_end = int(end_sample * audio_sr / eeg_sr)
    audio_segment = raw_audio[audio_start:audio_end]

    duration = len(audio_segment) / audio_sr
    t_audio = np.linspace(0, duration, len(audio_segment))

    peak = np.max(np.abs(audio_segment)) or 1
    audio_norm = audio_segment / peak

    # Load MFA
    mfa_words, mfa_phones = _load_mfa_alignment(pid, idx)
    if not mfa_words or not mfa_phones:
        print(f"No MFA TextGrid for {pid} sent {idx}")
        return

    # ══════════════════════════════════════════════════════════════════
    #  SINGLE FIGURE — 3 rows
    # ══════════════════════════════════════════════════════════════════
    fig, (ax_spec, ax_words, ax_phones) = plt.subplots(
        3, 1, figsize=(15, 9), sharex=True,
        gridspec_kw={'height_ratios': [2, 1.2, 1.2], 'hspace': 0.08})

    # ── Row 1: Mel spectrogram (23 mel filters, matching pipeline) ─────
    from extract_features import extractMelSpecs
    n_mels = 23
    mel_spec = extractMelSpecs(audio_segment, audio_sr,
                               windowLength=0.05, frameshift=0.01,
                               numFilter=n_mels)          # (n_frames, 23)
    mel_spec_T = mel_spec.T                                # (23, n_frames)

    hop_length = int(0.01 * audio_sr)
    mel_times = np.arange(mel_spec_T.shape[1] + 1) * 0.01  # time edges (s)
    mel_bins = np.arange(n_mels + 1) - 0.5                  # filter edges

    ax_spec.pcolormesh(mel_times, mel_bins, mel_spec_T,
                       shading='flat', cmap='viridis')

    # Word boundaries + labels on spectrogram
    for w in mfa_words:
        ax_spec.axvline(w['start'], color='red', lw=0.75, ls='--', alpha=0.9)
        ax_spec.axvline(w['end'], color='red', lw=0.75, ls='--', alpha=0.9)
        mid = (w['start'] + w['end']) / 2
        ax_spec.text(mid, n_mels - 1.5, w['text'], ha='center', va='top',
                     fontsize=10, fontweight='bold', color='white',
                     bbox=dict(boxstyle='round,pad=0.12',
                               facecolor='black', alpha=0.6, edgecolor='none'))

    ax_spec.set_ylabel('mel filter', fontsize=10)
    ax_spec.set_ylim(-0.5, n_mels - 0.5)
    ax_spec.set_title(
        f"{pid} — \"{text}\"\n"
        f"words: {[w['text'] for w in mfa_words]}   "
        f"phones: {len(mfa_phones)}",
        fontsize=11, fontweight='bold')

    # ── Row 2: Audio waveform + word boundaries ───────────────────────
    ax_words.plot(t_audio, audio_norm, color='#1f77b4', lw=0.2)

    for w in mfa_words:
        ax_words.axvline(w['start'], color='red', lw=0.75, ls='--', alpha=0.8)
        ax_words.axvline(w['end'], color='red', lw=0.75, ls='--', alpha=0.8)
        mid = (w['start'] + w['end']) / 2
        ax_words.text(mid, 0.92, w['text'], ha='center', va='top',
                      fontsize=10, fontweight='bold', color='#333',
                      transform=ax_words.get_xaxis_transform())

    ax_words.set_ylabel('audio signal', fontsize=10)
    ax_words.legend(['audio signal'], loc='upper right', fontsize=8)
    ax_words.set_title('word boundaries (red dashed = detected word edges)',
                        fontsize=10, loc='left', color='#555')

    # ── Row 3: Audio waveform + phoneme boundaries ────────────────────
    ax_phones.plot(t_audio, audio_norm, color='#1f77b4', lw=0.2)

    # Word boundaries (red dashed, same as row 2)
    for w in mfa_words:
        ax_phones.axvline(w['start'], color='red', lw=0.75, ls='--', alpha=0.6)
        ax_phones.axvline(w['end'], color='red', lw=0.75, ls='--', alpha=0.6)

    # Phoneme boundaries (colored by articulatory group)
    for i, ph in enumerate(mfa_phones):
        color = _phone_color(ph['text'])
        ax_phones.axvline(ph['start'], color=color, lw=0.6, alpha=0.7)
        mid = (ph['start'] + ph['end']) / 2
        dur = ph['end'] - ph['start']
        fs = 9 if dur > 0.06 else 7 if dur > 0.03 else 5
        # Alternate label height to avoid overlap
        y_pos = 0.92 if (i % 2 == 0) else 0.78
        ax_phones.text(mid, y_pos, ph['text'], ha='center', va='top',
                       fontsize=fs, fontweight='bold', color=color,
                       transform=ax_phones.get_xaxis_transform())

    ax_phones.set_ylabel('audio signal', fontsize=10)
    ax_phones.set_xlabel('time (s)', fontsize=10)
    ax_phones.set_title(
        'phoneme boundaries (colored lines = MFA-aligned phoneme edges)',
        fontsize=10, loc='left', color='#555')

    ax_phones.set_xlim(0, duration)

    plt.tight_layout()
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
#  NEURAL ALIGNMENT VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def plot_neural_alignment(pipeline, pid, sentence_text=None, sent_idx=None,
                          audio_sr=48000):
    """Single figure, 3 rows:
      Row 1: Mel spectrogram (23 filters) with word labels
      Row 2: Audio waveform with phoneme boundaries (colored)
      Row 3: Mean neural signal (EEG averaged across channels) with same
             phoneme boundaries transferred from audio time axis

    Args:
        pipeline: Dutch30Pipeline with split_result loaded.
        pid: patient ID, e.g. 'P24'.
        sentence_text: substring to match (e.g. 'donald trump').
        sent_idx: sentence index. Overrides sentence_text.
        audio_sr: audio sample rate (default 48000).
    """
    from extract_features import extractMelSpecs

    idx, sent, text = _find_sentence(pipeline, pid, sentence_text, sent_idx)
    if idx is None:
        print(f"Sentence not found for {pid}")
        return

    # ── Load raw audio ───────────────────────────────────────────────
    audio_path = os.path.join(DUTCH_30_PATH, 'raw', f'{pid}_audio.npy')
    if not os.path.exists(audio_path):
        print(f"Audio not found: {audio_path}")
        return

    raw_audio = np.load(audio_path)
    start_sample = sent['stim_start_idx']       # EEG sample index
    end_sample = sent['stim_end_idx']

    eeg_sr = pipeline.config.eeg_sr              # 1024 Hz
    audio_start = int(start_sample * audio_sr / eeg_sr)
    audio_end = int(end_sample * audio_sr / eeg_sr)
    audio_segment = raw_audio[audio_start:audio_end]

    duration = len(audio_segment) / audio_sr
    t_audio = np.linspace(0, duration, len(audio_segment))

    peak = np.max(np.abs(audio_segment)) or 1
    audio_norm = audio_segment / peak

    # ── Load raw EEG ─────────────────────────────────────────────────
    eeg_path = os.path.join(DUTCH_30_PATH, 'raw', f'{pid}_sEEG.npy')
    if not os.path.exists(eeg_path):
        print(f"EEG not found: {eeg_path}")
        return

    raw_eeg = np.load(eeg_path)                 # (n_total_samples, n_channels)
    eeg_segment = raw_eeg[start_sample:end_sample]  # (n_eeg_samples, n_ch)
    n_ch = eeg_segment.shape[1]
    median_eeg = np.median(eeg_segment, axis=1)  # median across channels
    t_eeg = np.linspace(0, duration, len(median_eeg))

    # Normalise median by its own peak (fills [-1, 1] like before)
    median_peak = np.max(np.abs(median_eeg)) or 1
    median_eeg_norm = median_eeg / median_peak
    # Normalise each channel by the same median peak (preserves relative scale)
    eeg_all_norm = eeg_segment / median_peak      # (n_samples, n_ch)

    # ── Load MFA alignment ───────────────────────────────────────────
    mfa_words, mfa_phones = _load_mfa_alignment(pid, idx)
    if not mfa_words or not mfa_phones:
        print(f"No MFA TextGrid for {pid} sent {idx}")
        return

    # ══════════════════════════════════════════════════════════════════
    #  FIGURE — 3 rows
    # ══════════════════════════════════════════════════════════════════
    fig, (ax_spec, ax_audio, ax_eeg) = plt.subplots(
        3, 1, figsize=(15, 9), sharex=True,
        gridspec_kw={'height_ratios': [2, 1.2, 1.2]})

    # ── Row 1: Mel spectrogram ───────────────────────────────────────
    n_mels = 23
    mel_spec = extractMelSpecs(audio_segment, audio_sr,
                               windowLength=0.05, frameshift=0.01,
                               numFilter=n_mels)
    mel_spec_T = mel_spec.T

    mel_times = np.arange(mel_spec_T.shape[1] + 1) * 0.01
    mel_bins = np.arange(n_mels + 1) - 0.5

    ax_spec.pcolormesh(mel_times, mel_bins, mel_spec_T,
                       shading='flat', cmap='viridis')

    for w in mfa_words:
        ax_spec.axvline(w['start'], color='red', lw=0.75, ls='--', alpha=0.9)
        ax_spec.axvline(w['end'], color='red', lw=0.75, ls='--', alpha=0.9)
        mid = (w['start'] + w['end']) / 2
        ax_spec.text(mid, n_mels - 1.5, w['text'], ha='center', va='top',
                     fontsize=10, fontweight='bold', color='white',
                     bbox=dict(boxstyle='round,pad=0.12',
                               facecolor='black', alpha=0.6, edgecolor='none'))

    ax_spec.set_ylabel('mel filter', fontsize=10)
    ax_spec.set_ylim(-0.5, n_mels - 0.5)
    ax_spec.set_title(
        f"{pid} — \"{text}\"\n"
        f"words: {[w['text'] for w in mfa_words]}   "
        f"phones: {len(mfa_phones)}",
        fontsize=11, fontweight='bold')

    # ── Row 2: Audio waveform + phoneme boundaries ───────────────────
    ax_audio.plot(t_audio, audio_norm, color='#1f77b4', lw=0.2)

    for w in mfa_words:
        ax_audio.axvline(w['start'], color='red', lw=0.75, ls='--', alpha=0.6)
        ax_audio.axvline(w['end'], color='red', lw=0.75, ls='--', alpha=0.6)

    for i, ph in enumerate(mfa_phones):
        color = _phone_color(ph['text'])
        ax_audio.axvline(ph['start'], color=color, lw=0.6, alpha=0.7)
        mid = (ph['start'] + ph['end']) / 2
        dur = ph['end'] - ph['start']
        fs = 9 if dur > 0.06 else 7 if dur > 0.03 else 5
        y_pos = 0.92 if (i % 2 == 0) else 0.78
        ax_audio.text(mid, y_pos, ph['text'], ha='center', va='top',
                      fontsize=fs, fontweight='bold', color=color,
                      transform=ax_audio.get_xaxis_transform())

    ax_audio.set_ylabel('audio (normalised)', fontsize=10)
    ax_audio.set_title(
        'audio waveform — phoneme boundaries (colored = MFA phonemes, '
        'red dashed = word edges)',
        fontsize=10, loc='left', color='#555', pad=6)

    # ── Row 3: All channels (gray) + median (black) + phoneme bounds ──
    # Plot individual channels in pale gray
    for ch in range(n_ch):
        ax_eeg.plot(t_eeg, eeg_all_norm[:, ch], color='#d0d0d0', lw=0.15,
                    alpha=0.4, rasterized=True)
    # Median in black on top
    ax_eeg.plot(t_eeg, median_eeg_norm, color='black', lw=0.4)

    # Word boundaries (red dashed)
    for w in mfa_words:
        ax_eeg.axvline(w['start'], color='red', lw=0.75, ls='--', alpha=0.6)
        ax_eeg.axvline(w['end'], color='red', lw=0.75, ls='--', alpha=0.6)

    # Phoneme boundaries (same colors as row 2)
    for i, ph in enumerate(mfa_phones):
        color = _phone_color(ph['text'])
        ax_eeg.axvline(ph['start'], color=color, lw=0.6, alpha=0.7)
        mid = (ph['start'] + ph['end']) / 2
        dur = ph['end'] - ph['start']
        fs = 9 if dur > 0.06 else 7 if dur > 0.03 else 5
        y_pos = 0.92 if (i % 2 == 0) else 0.78
        ax_eeg.text(mid, y_pos, ph['text'], ha='center', va='top',
                    fontsize=fs, fontweight='bold', color=color,
                    transform=ax_eeg.get_xaxis_transform())

    ax_eeg.set_ylim(-1.5, 1.5)
    ax_eeg.set_ylabel('sEEG (raw, normalised)', fontsize=10)
    ax_eeg.set_xlabel('time (s)', fontsize=10)
    ax_eeg.set_title(
        f'raw sEEG (broadband) — {n_ch} channels (gray) + median (black) — '
        'same phoneme boundaries from audio',
        fontsize=10, loc='left', color='#555', pad=6)

    ax_eeg.set_xlim(0, duration)

    fig.subplots_adjust(left=0.07, right=0.98, top=0.90, bottom=0.08, hspace=0.42)
    plt.show()
