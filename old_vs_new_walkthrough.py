# Old vs new high-gamma extraction — step-by-step on real iEEG.
#
# This script walks through every stage of both approaches on the same
# segment of one channel, so you can see what each operation does and
# how the final features differ.
#
# OLD path:  raw -> bandpass -> notches -> |hilbert| -> boxcar avg
# NEW path:  raw -> bandpass -> notches ->    x^2    -> Butterworth LPF -> avg -> sqrt
#
# Run cells in order.

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 1 — Imports + load one channel
# ═══════════════════════════════════════════════════════════════════════════════

import os
import numpy as np
import scipy.signal
import scipy.fftpack
import matplotlib.pyplot as plt

from config import DUTCH_30_PATH

PID = 'P25'
SR = 1024            # iEEG sample rate (Hz)
T_START = 30.0       # seconds into the recording
T_DUR   = 2.0        # length of segment to analyse

raw = np.load(os.path.join(DUTCH_30_PATH, 'raw', f'{PID}_sEEG.npy'))
chan_idx = int(np.argmax(raw.std(axis=0)))     # most-active channel

i0 = int(T_START * SR); i1 = int((T_START + T_DUR) * SR)
x_raw = raw[i0:i1, chan_idx].astype(np.float64)
t = np.arange(len(x_raw)) / SR

print(f"  patient: {PID}, channel {chan_idx}")
print(f"  segment: {T_START:.1f}-{T_START+T_DUR:.1f} s ({len(x_raw)} samples)")


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 2 — Helpers (filters used by both paths)
# ═══════════════════════════════════════════════════════════════════════════════

hilbert3 = lambda x: scipy.signal.hilbert(
    x, scipy.fftpack.next_fast_len(len(x)))[:len(x)]


def bandpass_70_170(x, sr):
    sos = scipy.signal.iirfilter(4, [70/(sr/2), 170/(sr/2)],
                                 btype='bandpass', output='sos')
    return scipy.signal.sosfiltfilt(sos, x)


def notch(x, sr, f0):
    sos = scipy.signal.iirfilter(4, [(f0-2)/(sr/2), (f0+2)/(sr/2)],
                                 btype='bandstop', output='sos')
    return scipy.signal.sosfiltfilt(sos, x)


def lowpass(x, sr, fc):
    sos = scipy.signal.iirfilter(4, fc/(sr/2),
                                 btype='lowpass', output='sos')
    return scipy.signal.sosfiltfilt(sos, x)


def windowed_average(x, sr, window_ms, shift_ms):
    """Boxcar window-average with given shift. Returns (n_windows,) feature."""
    win = window_ms / 1000.0
    sft = shift_ms  / 1000.0
    n = int(np.floor((len(x) - win*sr) / (sft*sr)))
    out = np.zeros(n)
    for w in range(n):
        s = int(np.floor(w * sft * sr))
        e = int(np.floor(s + win * sr))
        out[w] = x[s:e].mean()
    t_out = np.arange(n) * sft
    return t_out, out


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 3 — OLD pipeline, every step  (with a before/after plot per stage)
# ═══════════════════════════════════════════════════════════════════════════════
# ⚠  All plots in this cell show ONE example channel (chan_idx, picked above)
#    for readability.  The actual pipeline runs every operation on ALL channels
#    in parallel — see CELL 4b below for the multi-channel view.

def _ba_plot(t, before, after, title, ylabel='µV',
             color_before='gray', color_after='steelblue',
             alpha_before=0.5, lw_after=0.8):
    """Tiny before/after panel — call after each pipeline step."""
    fig, ax = plt.subplots(figsize=(11, 2.4))
    if before is not None:
        ax.plot(t, before, lw=0.6, color=color_before, alpha=alpha_before,
                label='before')
    ax.plot(t, after, lw=lw_after, color=color_after, label='after')
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(alpha=0.3); ax.legend(loc='upper right', fontsize=8)
    plt.tight_layout(); plt.show()


# ── Step 0: linear detrend ────────────────────────────────────────────────────
# Removes any slow DC drift / electrode polarization.  Visual effect is usually
# subtle — you'd only notice it if the channel had a clear baseline tilt.
old_step0 = scipy.signal.detrend(x_raw)
_ba_plot(t, x_raw, old_step0,
         f'OLD step 0 — detrend (removes slow DC drift)   '
         f'[ex. channel {chan_idx}]')


# ── Step 1: bandpass 70–170 Hz ────────────────────────────────────────────────
# Isolates the high-gamma band.  After this the signal becomes a fast
# oscillation centered around 0; everything outside 70-170 Hz is gone.
old_step1 = bandpass_70_170(old_step0, SR)
_ba_plot(t, old_step0, old_step1,
         f'OLD step 1 — bandpass 70-170 Hz (kills slow rhythms + above-170 Hz)   '
         f'[ex. channel {chan_idx}]')


# ── Step 2: 100 Hz notch ──────────────────────────────────────────────────────
# Removes the 1st harmonic of 50 Hz line noise.  Time-domain change is small
# (notch is narrow), but in the spectrum a 100 Hz spike disappears.
old_step2 = notch(old_step1, SR, 100.0)
_ba_plot(t, old_step1, old_step2,
         f'OLD step 2 — 100 Hz notch (1st harmonic of 50 Hz line noise)   '
         f'[ex. channel {chan_idx}]')


# ── Step 3: 150 Hz notch ──────────────────────────────────────────────────────
# Removes the 2nd harmonic of line noise.  Again small in time domain.
old_step3 = notch(old_step2, SR, 150.0)
_ba_plot(t, old_step2, old_step3,
         f'OLD step 3 — 150 Hz notch (2nd harmonic of 50 Hz line noise)   '
         f'[ex. channel {chan_idx}]')


# ── Step 4: |hilbert(x)| envelope ─────────────────────────────────────────────
# This is the BIG transformation.  Replaces the fast oscillation with its
# instantaneous amplitude envelope — slow signal that rides on top.
# This is also the implicit compensation for the boxcar's nulls: it shifts
# the spectrum to baseband.
old_step4 = np.abs(hilbert3(old_step3))
_ba_plot(t, old_step3, old_step4,
         f'OLD step 4 — |hilbert(x)| envelope  ← THE compensation step   '
         f'[ex. channel {chan_idx}]',
         color_after='steelblue', lw_after=1.2)


# ── Step 5: boxcar windowed average (50 ms window, 10 ms shift) ───────────────
# Downsamples to 100 fps by averaging.  Output rate is now lower, so we use
# the windowed time axis t_old.
t_old, old_feat = windowed_average(old_step4, SR, window_ms=50, shift_ms=10)
print(f"  OLD final feature: {len(old_feat)} samples @ 100 fps")

fig, ax = plt.subplots(figsize=(11, 2.7))
ax.plot(t,     old_step4, lw=0.6, color='gray', alpha=0.5,
        label='before: envelope @ 1024 fps')
ax.plot(t_old, old_feat,  marker='o', ms=3, lw=1.0, color='steelblue',
        label='after: 50ms boxcar avg @ 100 fps')
ax.set_title(f'OLD step 5 — 50 ms boxcar window, 10 ms shift  (final feature)   '
             f'[ex. channel {chan_idx}]',
             fontsize=10)
ax.set_ylabel('µV', fontsize=9); ax.set_xlabel('time (s)', fontsize=9)
ax.grid(alpha=0.3); ax.legend(loc='upper right', fontsize=8)
plt.tight_layout(); plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 4 — NEW pipeline, every step  (with a before/after plot per stage)
# ═══════════════════════════════════════════════════════════════════════════════
# ⚠  All plots in this cell show ONE example channel (chan_idx) for readability.
#    The actual pipeline runs every operation on ALL channels in parallel — see
#    CELL 4b below for the multi-channel view.
#
# Reuses the _ba_plot helper from CELL 3.

# ── Step 0: linear detrend  (same as OLD) ─────────────────────────────────────
new_step0 = scipy.signal.detrend(x_raw)
_ba_plot(t, x_raw, new_step0,
         f'NEW step 0 — detrend (same as OLD)   [ex. channel {chan_idx}]',
         color_after='crimson')


# ── Step 1: bandpass 70-170 Hz  (same as OLD) ─────────────────────────────────
new_step1 = bandpass_70_170(new_step0, SR)
_ba_plot(t, new_step0, new_step1,
         f'NEW step 1 — bandpass 70-170 Hz (same as OLD)   '
         f'[ex. channel {chan_idx}]',
         color_after='crimson')


# ── Step 2: 100 Hz notch  (same as OLD) ───────────────────────────────────────
new_step2 = notch(new_step1, SR, 100.0)
_ba_plot(t, new_step1, new_step2,
         f'NEW step 2 — 100 Hz notch (same as OLD)   [ex. channel {chan_idx}]',
         color_after='crimson')


# ── Step 3: 150 Hz notch  (same as OLD) ───────────────────────────────────────
new_step3 = notch(new_step2, SR, 150.0)
_ba_plot(t, new_step2, new_step3,
         f'NEW step 3 — 150 Hz notch (same as OLD)   [ex. channel {chan_idx}]',
         color_after='crimson')


# ── Step 4: square (x²)  ← REPLACES |hilbert| ────────────────────────────────
# This is where the OLD and NEW paths diverge.  Squaring takes the fast
# oscillation and turns it into an always-positive instantaneous power
# signal.  Doubles the spectrum: a 70-170 Hz oscillation produces DC +
# slow modulations (the envelope we want) PLUS 140-340 Hz content (a
# byproduct that the next step will filter out).
new_step4 = new_step3 ** 2
fig, ax = plt.subplots(figsize=(11, 2.4))
ax.plot(t, new_step3, lw=0.6, color='gray', alpha=0.5, label='before: bandpassed signal')
ax.plot(t, new_step4, lw=0.8, color='crimson',           label='after: x² (power)')
ax.set_title(f'NEW step 4 — square the signal (instantaneous power)  '
             f'← REPLACES Hilbert   [ex. channel {chan_idx}]', fontsize=10)
ax.set_ylabel('µV / µV²', fontsize=9)
ax.grid(alpha=0.3); ax.legend(loc='upper right', fontsize=8)
plt.tight_layout(); plt.show()


# ── Step 5: 10 Hz Butterworth LPF  ← THE smoothing step ──────────────────────
# Crushes everything above 10 Hz with -48 dB/oct effective rolloff.
# The fast wiggles in the power signal get replaced by a smooth slow envelope.
# This is the analogue of OLD's Hilbert step, but cleaner.
new_step5 = np.abs(lowpass(new_step4, SR, fc=10.0))
_ba_plot(t, new_step4, new_step5,
         f'NEW step 5 — 10 Hz Butterworth LPF  ← clean smoothing replaces boxcar leakage   '
         f'[ex. channel {chan_idx}]',
         ylabel='µV²',
         color_after='crimson', lw_after=1.5)


# ── Step 6: window-average + sqrt  → final feature ────────────────────────────
# Tiny 15 ms window, 5 ms shift → 200 fps output.  Window is now JUST sampling
# the smooth envelope; it does no real smoothing because the LPF already did.
# sqrt brings the units back to amplitude (composes with downstream code the
# same way |hilbert| envelopes did).
t_new, _avg = windowed_average(new_step5, SR, window_ms=15, shift_ms=5)
new_feat = np.sqrt(_avg)
print(f"  NEW final feature: {len(new_feat)} samples @ 200 fps")

fig, ax = plt.subplots(figsize=(11, 2.7))
ax.plot(t,     new_step5, lw=0.6, color='gray', alpha=0.5,
        label='before: LPF output @ 1024 fps (in power units)')
ax.plot(t_new, new_feat,  marker='o', ms=2, lw=0.9, color='crimson',
        label='after: 15ms boxcar avg + sqrt @ 200 fps  (final feature)')
ax.set_title(f'NEW step 6 — sample with 15 ms window, sqrt back to amplitude  '
             f'(final feature)   [ex. channel {chan_idx}]', fontsize=10)
ax.set_ylabel('µV² / µV', fontsize=9); ax.set_xlabel('time (s)', fontsize=9)
ax.grid(alpha=0.3); ax.legend(loc='upper right', fontsize=8)
plt.tight_layout(); plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 4b — Multi-channel view: what the pipeline ACTUALLY processes
# ═══════════════════════════════════════════════════════════════════════════════
# CELL 3 and CELL 4 visualised one example channel for clarity, but the real
# pipeline runs every operation on ALL channels in parallel and keeps them as
# separate feature dimensions all the way to the classifier.  This cell shows
# the multi-channel reality plus a "mean-across-channels" trace — useful only
# as a visualisation aid, NOT as a pipeline output.

# Apply the OLD and NEW envelope steps to ALL channels at once.  These are
# vectorised over axis 1, just like the real pipeline does them.
ALL_chan = scipy.signal.detrend(raw[i0:i1], axis=0)
ALL_chan = bandpass_70_170(ALL_chan, SR)
# Two notches
sos_n100 = scipy.signal.iirfilter(4, [98/(SR/2), 102/(SR/2)],
                                  btype='bandstop', output='sos')
sos_n150 = scipy.signal.iirfilter(4, [148/(SR/2), 152/(SR/2)],
                                  btype='bandstop', output='sos')
ALL_chan = scipy.signal.sosfiltfilt(sos_n100, ALL_chan, axis=0)
ALL_chan = scipy.signal.sosfiltfilt(sos_n150, ALL_chan, axis=0)

# OLD envelope across all channels
ALL_old_env = np.abs(scipy.signal.hilbert(
    ALL_chan, scipy.fftpack.next_fast_len(ALL_chan.shape[0]), axis=0)
    )[:ALL_chan.shape[0]]

# NEW envelope across all channels
sos_lp10 = scipy.signal.iirfilter(4, 10/(SR/2), btype='lowpass', output='sos')
ALL_new_env = np.sqrt(np.abs(
    scipy.signal.sosfiltfilt(sos_lp10, ALL_chan**2, axis=0)
))

n_chan = ALL_old_env.shape[1]
print(f"  Multi-channel envelope shape: {ALL_old_env.shape}  "
      f"({n_chan} channels × {ALL_old_env.shape[0]} samples)")

# ── Plot 1: heatmap of OLD envelope across all channels ───────────────────────
fig, axes = plt.subplots(2, 1, figsize=(13, 6.5),
                         gridspec_kw={'height_ratios': [3, 1]}, sharex=True)

ax = axes[0]
# z-score per channel so colours are comparable across electrodes with
# very different amplitudes
ALL_old_z = ((ALL_old_env - ALL_old_env.mean(axis=0)) /
             (ALL_old_env.std(axis=0) + 1e-9))
im = ax.imshow(ALL_old_z.T, aspect='auto', origin='lower',
               cmap='viridis', vmin=-2, vmax=4,
               extent=[t[0], t[-1], 0, n_chan])
ax.axhline(chan_idx + 0.5, color='red', lw=1.2,
           label=f'example channel from CELL 3 (C{chan_idx})')
ax.set_ylabel('channel index')
ax.set_title(f'OLD envelope (Hilbert) — all {n_chan} channels at once\n'
             f'each row = one electrode, colour = z-scored envelope amplitude')
ax.legend(loc='upper right', fontsize=9)
plt.colorbar(im, ax=ax, label='z-score', pad=0.01)

# Below: the example channel + the cross-channel mean
ax = axes[1]
ax.plot(t, ALL_old_env[:, chan_idx], lw=1.0, color='red',
        label=f'example channel C{chan_idx}')
ax.plot(t, ALL_old_env.mean(axis=1), lw=1.5, color='black',
        label='mean across all channels  (VIZ ONLY — never computed by pipeline)')
ax.set_xlabel('time (s)'); ax.set_ylabel('envelope (µV)')
ax.legend(loc='upper right', fontsize=9); ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()


# ── Plot 2: heatmap of NEW envelope across all channels ───────────────────────
fig, axes = plt.subplots(2, 1, figsize=(13, 6.5),
                         gridspec_kw={'height_ratios': [3, 1]}, sharex=True)

ax = axes[0]
ALL_new_z = ((ALL_new_env - ALL_new_env.mean(axis=0)) /
             (ALL_new_env.std(axis=0) + 1e-9))
im = ax.imshow(ALL_new_z.T, aspect='auto', origin='lower',
               cmap='viridis', vmin=-2, vmax=4,
               extent=[t[0], t[-1], 0, n_chan])
ax.axhline(chan_idx + 0.5, color='red', lw=1.2,
           label=f'example channel from CELL 4 (C{chan_idx})')
ax.set_ylabel('channel index')
ax.set_title(f'NEW envelope (x² + 10 Hz LPF) — all {n_chan} channels at once\n'
             f'note the cleaner / less noisy look across rows compared to OLD heatmap')
ax.legend(loc='upper right', fontsize=9)
plt.colorbar(im, ax=ax, label='z-score', pad=0.01)

ax = axes[1]
ax.plot(t, ALL_new_env[:, chan_idx], lw=1.0, color='red',
        label=f'example channel C{chan_idx}')
ax.plot(t, ALL_new_env.mean(axis=1), lw=1.5, color='black',
        label='mean across all channels  (VIZ ONLY — never computed by pipeline)')
ax.set_xlabel('time (s)'); ax.set_ylabel('envelope (µV)')
ax.legend(loc='upper right', fontsize=9); ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()


# ── Note on what the channel-mean shows (and what it doesn't) ─────────────────
print(f"""
  ┌──────────────────────────────────────────────────────────────────────┐
  │  IMPORTANT — about the "mean across channels" line                   │
  ├──────────────────────────────────────────────────────────────────────┤
  │                                                                      │
  │  The black line in the panels above averages every channel's         │
  │  envelope into a single trace.  This is purely for visualisation     │
  │  — it tells you "is most of the brain active right now?"             │
  │                                                                      │
  │  The pipeline NEVER computes this.  Each channel is kept as its      │
  │  own feature dimension all the way to the classifier (~{n_chan} channels   │
  │  × 41 stacked frames = ~{n_chan*41} features per phoneme).               │
  │                                                                      │
  │  Channel-aggregation operations that DO exist elsewhere:             │
  │   • CAR (Common Average Reference) — subtracts cross-channel mean    │
  │     from each channel; not currently applied in this pipeline.       │
  │   • PCA — sometimes used to compress dimensions; off by default.     │
  │   • Channel exclusion — bad channels dropped via channel_masks.      │
  │                                                                      │
  │  The heterogeneity you see across rows in the heatmap is exactly     │
  │  what makes phoneme decoding work — different channels encode        │
  │  different speech features, and the classifier needs them separate.  │
  └──────────────────────────────────────────────────────────────────────┘
""")


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 4c — Bode plots of every filter used in either pipeline
# ═══════════════════════════════════════════════════════════════════════════════
# Two figures:
#
#   Figure 1 — Preprocessing filters (used in BOTH old and new):
#                70-170 Hz bandpass, 100 Hz notch, 150 Hz notch
#
#   Figure 2 — Smoother choice (the OLD vs NEW divergence):
#                50 ms boxcar (FIR, OLD path)
#                10 Hz Butterworth low-pass (IIR, NEW path)
#
# Each figure has TWO panels (magnitude + phase) — that's what makes a Bode
# plot complete. Note: the pipeline uses sosfiltfilt, which makes phase = 0
# everywhere. The phase plots below show the SINGLE-PASS responses (what
# `scipy.signal.sosfreqz` returns), which reveal the filter's natural phase
# behaviour. The "no phase distortion" benefit of filtfilt is a separate fact.

# ── Build the filter SOSs we want to inspect ─────────────────────────────────
sos_bp   = scipy.signal.iirfilter(4, [70/(SR/2), 170/(SR/2)],
                                  btype='bandpass', output='sos')
sos_n100 = scipy.signal.iirfilter(4, [98/(SR/2), 102/(SR/2)],
                                  btype='bandstop', output='sos')
sos_n150 = scipy.signal.iirfilter(4, [148/(SR/2), 152/(SR/2)],
                                  btype='bandstop', output='sos')
sos_lp10 = scipy.signal.iirfilter(4, 10/(SR/2),
                                  btype='lowpass',  output='sos')

# Boxcar as an FIR kernel
N_box = int(0.050 * SR)   # 51 taps for 50 ms
b_box = np.ones(N_box) / N_box


def _bode(ax_mag, ax_phase, w, h, lw=2, color=None, label=None, ls='-'):
    """Helper: plot one filter's magnitude and phase on the given axes."""
    mag_db   = 20 * np.log10(np.abs(h) + 1e-15)
    phase_deg = np.unwrap(np.angle(h)) * 180.0 / np.pi
    ax_mag.plot(w, mag_db,    lw=lw, color=color, ls=ls, label=label)
    ax_phase.plot(w, phase_deg, lw=lw, color=color, ls=ls, label=label)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — Preprocessing filters (used in BOTH paths)
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                         gridspec_kw={'height_ratios': [3, 2]})
ax_mag, ax_phase = axes

w, h = scipy.signal.sosfreqz(sos_bp,   worN=8192, fs=SR)
_bode(ax_mag, ax_phase, w, h, color='steelblue',
      label='70-170 Hz bandpass (4th order)')

w, h = scipy.signal.sosfreqz(sos_n100, worN=8192, fs=SR)
_bode(ax_mag, ax_phase, w, h, color='darkorange',
      label='100 Hz notch (4th order bandstop)')

w, h = scipy.signal.sosfreqz(sos_n150, worN=8192, fs=SR)
_bode(ax_mag, ax_phase, w, h, color='seagreen',
      label='150 Hz notch (4th order bandstop)')

ax_mag.set_xscale('log'); ax_mag.set_xlim(0.5, 500); ax_mag.set_ylim(-80, 5)
ax_mag.set_ylabel('magnitude (dB)')
ax_mag.set_title('Preprocessing filters — used in BOTH old and new paths',
                 fontsize=12, fontweight='bold')
ax_mag.axvline(70,  color='steelblue', ls=':', alpha=0.4)
ax_mag.axvline(170, color='steelblue', ls=':', alpha=0.4)
ax_mag.axvline(100, color='darkorange', ls=':', alpha=0.4)
ax_mag.axvline(150, color='seagreen',   ls=':', alpha=0.4)
ax_mag.grid(alpha=0.3, which='both'); ax_mag.legend(loc='lower center', fontsize=9)

ax_phase.set_xscale('log'); ax_phase.set_xlim(0.5, 500)
ax_phase.set_xlabel('frequency (Hz, log)')
ax_phase.set_ylabel('phase (degrees)')
ax_phase.grid(alpha=0.3, which='both')
ax_phase.text(0.02, 0.95,
              'Single-pass phase shown — pipeline uses sosfiltfilt,\n'
              'which cancels phase to identically 0° at all frequencies.',
              transform=ax_phase.transAxes, va='top', fontsize=8,
              bbox=dict(facecolor='lightyellow', edgecolor='gray',
                        boxstyle='round,pad=0.3'))

plt.tight_layout(); plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Smoother choice (OLD vs NEW divergence)
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                         gridspec_kw={'height_ratios': [3, 2]})
ax_mag, ax_phase = axes

# OLD: 50 ms boxcar (FIR)
w_box, h_box = scipy.signal.freqz(b_box, [1.0], worN=8192, fs=SR)
_bode(ax_mag, ax_phase, w_box, h_box, color='steelblue',
      label='OLD: 50 ms boxcar (FIR, 51 taps)')

# NEW: 10 Hz Butterworth lowpass (IIR, single-pass)
w_lp, h_lp = scipy.signal.sosfreqz(sos_lp10, worN=8192, fs=SR)
_bode(ax_mag, ax_phase, w_lp, h_lp, color='crimson',
      label='NEW: 10 Hz Butterworth (IIR, 4th order)')

# Also show the EFFECTIVE Butterworth response with filtfilt (= |H|²)
# — this is what the pipeline actually applies to the signal magnitude-wise
mag_db_filtfilt = 40 * np.log10(np.abs(h_lp) + 1e-15)   # 20*log10(|H|^2)
ax_mag.plot(w_lp, mag_db_filtfilt, lw=1.5, color='crimson', ls='--',
            label='NEW: same Butterworth via filtfilt (|H|² → -48 dB/oct effective)')

# Mark key frequencies
ax_mag.axvline(10,  color='crimson',   ls=':', alpha=0.5)
ax_mag.axvline(20,  color='steelblue', ls=':', alpha=0.5)
ax_mag.axvline(40,  color='steelblue', ls=':', alpha=0.5)
ax_mag.text(10.5, -75, '10 Hz\ncutoff',    color='crimson',   fontsize=8)
ax_mag.text(21,   -75, 'boxcar\n1st null', color='steelblue', fontsize=8)
ax_mag.text(41,   -75, 'boxcar\n2nd null', color='steelblue', fontsize=8)

ax_mag.set_xscale('log'); ax_mag.set_xlim(0.5, 500); ax_mag.set_ylim(-100, 5)
ax_mag.set_ylabel('magnitude (dB)')
ax_mag.set_title('Smoother choice — OLD (boxcar FIR) vs NEW (Butterworth IIR)',
                 fontsize=12, fontweight='bold')
ax_mag.grid(alpha=0.3, which='both'); ax_mag.legend(loc='lower left', fontsize=9)

ax_phase.set_xscale('log'); ax_phase.set_xlim(0.5, 500)
ax_phase.set_xlabel('frequency (Hz, log)')
ax_phase.set_ylabel('phase (degrees)')
ax_phase.grid(alpha=0.3, which='both')
ax_phase.text(0.02, 0.05,
              'Boxcar (symmetric FIR) → exact LINEAR phase: every frequency '
              'delayed by the same time (~25 ms = N/2).\n'
              'Butterworth single-pass → nonlinear phase, especially near cutoff.\n'
              'In the actual pipeline, sosfiltfilt cancels both to phase = 0°.',
              transform=ax_phase.transAxes, va='bottom', fontsize=8,
              bbox=dict(facecolor='lightyellow', edgecolor='gray',
                        boxstyle='round,pad=0.3'))

plt.tight_layout(); plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Read-out helper: print key numbers from the Bode plots
# ─────────────────────────────────────────────────────────────────────────────
def _gain_at(w, h, freq):
    return 20*np.log10(np.abs(h[np.argmin(np.abs(w - freq))]) + 1e-15)

print(f"\n  Key gain values from the Bode plots:")
print(f"  {'-'*68}")
print(f"  Bandpass 70-170 Hz at 50 Hz:   {_gain_at(*scipy.signal.sosfreqz(sos_bp, worN=8192, fs=SR), 50):.1f} dB   (well in stopband)")
print(f"  Bandpass 70-170 Hz at 120 Hz:  {_gain_at(*scipy.signal.sosfreqz(sos_bp, worN=8192, fs=SR), 120):.1f} dB   (passband)")
print(f"  100 Hz notch at 100 Hz:        {_gain_at(*scipy.signal.sosfreqz(sos_n100, worN=8192, fs=SR), 100):.1f} dB   (deep null)")
print(f"  150 Hz notch at 150 Hz:        {_gain_at(*scipy.signal.sosfreqz(sos_n150, worN=8192, fs=SR), 150):.1f} dB   (deep null)")
print()
print(f"  Boxcar (OLD smoother) at  20 Hz:  {_gain_at(w_box, h_box, 20):.1f} dB   (1st null)")
print(f"  Boxcar (OLD smoother) at  30 Hz:  {_gain_at(w_box, h_box, 30):.1f} dB   (1st sidelobe peak — leaky!)")
print(f"  Boxcar (OLD smoother) at  50 Hz:  {_gain_at(w_box, h_box, 50):.1f} dB   (2nd sidelobe — leakier)")
print()
print(f"  Butterworth (NEW smoother, single-pass) at  10 Hz:  {_gain_at(w_lp, h_lp, 10):.1f} dB   (cutoff)")
print(f"  Butterworth (NEW smoother, single-pass) at  30 Hz:  {_gain_at(w_lp, h_lp, 30):.1f} dB   (clean rolloff)")
print(f"  Butterworth (NEW smoother, single-pass) at  50 Hz:  {_gain_at(w_lp, h_lp, 50):.1f} dB")
print(f"  Butterworth (NEW, with filtfilt = |H|²) at  30 Hz:  {2*_gain_at(w_lp, h_lp, 30):.1f} dB   (twice the rejection)")


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 5 — Visualize all stages of OLD path
# ═══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(6, 1, figsize=(12, 11), sharex=True)

axes[0].plot(t, old_step0, lw=0.6, color='black')
axes[0].set_title(f'OLD — Step 0: detrended raw EEG  (channel {chan_idx})')
axes[0].set_ylabel('µV')

axes[1].plot(t, old_step1, lw=0.6, color='steelblue')
axes[1].set_title('OLD — Step 1: bandpass 70-170 Hz  (high-gamma oscillation)')
axes[1].set_ylabel('µV')

axes[2].plot(t, old_step3, lw=0.6, color='steelblue')
axes[2].set_title('OLD — Steps 2-3: notches at 100 + 150 Hz applied')
axes[2].set_ylabel('µV')

axes[3].plot(t, old_step3, lw=0.4, color='lightgray', alpha=0.6,
             label='bandpassed signal')
axes[3].plot(t, old_step4, lw=1.2, color='steelblue',
             label='|hilbert(x)|  ← compensation: shifts to baseband')
axes[3].set_title('OLD — Step 4: Hilbert envelope (the implicit baseband shift)')
axes[3].set_ylabel('µV'); axes[3].legend(loc='upper right', fontsize=9)

axes[4].plot(t, old_step4, lw=0.6, color='lightgray', alpha=0.7,
             label='envelope (1024 fps)')
axes[4].plot(t_old, old_feat, marker='o', ms=3, lw=1.0,
             color='steelblue',
             label='boxcar avg, 50ms window, 10ms shift  (100 fps)')
axes[4].set_title('OLD — Step 5: 50 ms boxcar averaging → final feature')
axes[4].set_ylabel('µV'); axes[4].legend(loc='upper right', fontsize=9)

axes[5].plot(t_old, old_feat, marker='o', ms=4, lw=1.2, color='steelblue')
axes[5].set_title('OLD — Final feature (what the classifier sees)')
axes[5].set_xlabel('time (s)')
axes[5].set_ylabel('feature value')

for ax in axes: ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 6 — Visualize all stages of NEW path
# ═══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(7, 1, figsize=(12, 12.5), sharex=True)

axes[0].plot(t, new_step0, lw=0.6, color='black')
axes[0].set_title(f'NEW — Step 0: detrended raw EEG  (channel {chan_idx})')
axes[0].set_ylabel('µV')

axes[1].plot(t, new_step1, lw=0.6, color='crimson')
axes[1].set_title('NEW — Step 1: bandpass 70-170 Hz  (same as old)')
axes[1].set_ylabel('µV')

axes[2].plot(t, new_step3, lw=0.6, color='crimson')
axes[2].set_title('NEW — Steps 2-3: notches at 100 + 150 Hz  (same as old)')
axes[2].set_ylabel('µV')

axes[3].plot(t, new_step4, lw=0.6, color='crimson')
axes[3].set_title('NEW — Step 4: square (x²)  — instantaneous power, replaces Hilbert')
axes[3].set_ylabel('µV²')

axes[4].plot(t, new_step4, lw=0.4, color='lightgray', alpha=0.7,
             label='power signal')
axes[4].plot(t, new_step5, lw=1.5, color='crimson',
             label='10 Hz Butterworth LPF  ← clean smoothing')
axes[4].set_title('NEW — Step 5: 10 Hz Butterworth LPF (the explicit smoother)')
axes[4].set_ylabel('µV²'); axes[4].legend(loc='upper right', fontsize=9)

axes[5].plot(t, new_step5, lw=0.6, color='lightgray', alpha=0.7,
             label='LPF output (1024 fps)')
axes[5].plot(t_new, _avg, marker='o', ms=2, lw=0.8,
             color='crimson',
             label='boxcar avg, 15ms window, 5ms shift  (200 fps)')
axes[5].set_title('NEW — Step 6: window-average → samples the smooth envelope')
axes[5].set_ylabel('µV²'); axes[5].legend(loc='upper right', fontsize=9)

axes[6].plot(t_new, new_feat, marker='o', ms=3, lw=1.0, color='crimson')
axes[6].set_title('NEW — Step 7: √ → final feature in amplitude units')
axes[6].set_xlabel('time (s)')
axes[6].set_ylabel('feature value')

for ax in axes: ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 7 — Side-by-side comparison of final features
# ═══════════════════════════════════════════════════════════════════════════════
# The two features are at different frame rates (100 vs 200 fps), so we
# overlay them on a common time axis for direct visual comparison.

fig, axes = plt.subplots(2, 1, figsize=(13, 6.5), sharex=True)

# Z-score both so they're visually comparable in scale
old_z = (old_feat - old_feat.mean()) / old_feat.std()
new_z = (new_feat - new_feat.mean()) / new_feat.std()

axes[0].plot(t_old, old_z, lw=1.2, color='steelblue',
             label=f'OLD (Hilbert + 50ms boxcar, 100 fps)')
axes[0].plot(t_new, new_z, lw=1.2, color='crimson',
             label=f'NEW (x² + 10Hz LPF + 15ms/5ms, 200 fps)')
axes[0].set_title('Final features overlaid (z-scored for comparison)')
axes[0].set_ylabel('z-scored feature')
axes[0].legend(); axes[0].grid(alpha=0.3)

# Difference panel — interpolate old to new's time axis to subtract
old_z_interp = np.interp(t_new, t_old, old_z)
diff = new_z - old_z_interp
axes[1].plot(t_new, diff, lw=0.8, color='black')
axes[1].axhline(0, color='gray', ls=':', alpha=0.5)
axes[1].set_title('Difference (NEW − OLD, both z-scored)')
axes[1].set_xlabel('time (s)')
axes[1].set_ylabel('Δ z-score')
axes[1].grid(alpha=0.3)

plt.tight_layout(); plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 8 — Spectrum at key intermediate stages
# ═══════════════════════════════════════════════════════════════════════════════
# To see what frequencies actually live in each intermediate signal — this
# is the BEST view of why the Hilbert step is doing structural compensation
# for the boxcar's nulls.

# Use Welch on a longer segment for cleaner spectra
i0_long = int(10 * SR); i1_long = int(70 * SR)
x_long = raw[i0_long:i1_long, chan_idx].astype(np.float64)
xb_long = bandpass_70_170(scipy.signal.detrend(x_long), SR)
xb_long = notch(notch(xb_long, SR, 100.0), SR, 150.0)

# Old path intermediates
old_env_long = np.abs(hilbert3(xb_long))

# New path intermediates
new_pwr_long = xb_long ** 2
new_lpf_long = np.abs(lowpass(new_pwr_long, SR, fc=10.0))

f1, P_bp  = scipy.signal.welch(xb_long       - xb_long.mean(),       fs=SR, nperseg=4096)
f2, P_old = scipy.signal.welch(old_env_long  - old_env_long.mean(),  fs=SR, nperseg=4096)
f3, P_pwr = scipy.signal.welch(new_pwr_long  - new_pwr_long.mean(),  fs=SR, nperseg=4096)
f4, P_lpf = scipy.signal.welch(new_lpf_long  - new_lpf_long.mean(),  fs=SR, nperseg=4096)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

# Panel A — old path: bandpass vs Hilbert envelope
ax = axes[0]
ax.semilogy(f1, P_bp,  lw=1.5, color='gray',     label='bandpassed signal (70-170 Hz)')
ax.semilogy(f2, P_old, lw=1.5, color='steelblue',label='|hilbert(x)|  (envelope, baseband)')
# Mark boxcar nulls
for k in range(1, 6):
    f_null = k / 0.050
    if f_null < 200:
        ax.axvline(f_null, color='red', ls=':', alpha=0.4)
ax.text(20, ax.get_ylim()[1]*0.5, 'red dashes:\nboxcar nulls\n(50 ms window)',
        color='red', fontsize=8, ha='left', va='top')
ax.set_xlim(0, 200)
ax.set_xlabel('frequency (Hz)'); ax.set_ylabel('PSD')
ax.set_title('OLD path — Hilbert shifts spectrum to baseband\n'
             'most envelope energy is below 20 Hz (first boxcar null)')
ax.grid(alpha=0.3, which='both'); ax.legend(fontsize=9)

# Panel B — new path: power signal vs LPF output
ax = axes[1]
ax.semilogy(f3, P_pwr, lw=1.5, color='lightgray', label='x² (instantaneous power)')
ax.semilogy(f4, P_lpf, lw=1.5, color='crimson',   label='LPF @ 10 Hz')
ax.axvline(10, color='gray', ls=':', alpha=0.6)
ax.text(11, ax.get_ylim()[1]*0.5, '10 Hz cutoff', color='gray', fontsize=9)
ax.set_xlim(0, 200)
ax.set_xlabel('frequency (Hz)'); ax.set_ylabel('PSD')
ax.set_title('NEW path — Butterworth LPF cleanly removes everything > 10 Hz\n'
             'no leakage, no sidelobes')
ax.grid(alpha=0.3, which='both'); ax.legend(fontsize=9)

plt.tight_layout(); plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 9 — What the boxcar "sees" in each path (the compensation explained)
# ═══════════════════════════════════════════════════════════════════════════════
# This is the key plot: the SAME boxcar filter (50 ms) acts very differently
# depending on what it's averaging over. In the OLD path, Hilbert delivers a
# baseband signal where the boxcar's first null (20 Hz) is past most energy.
# In the NEW path the equivalent role is played by the 10 Hz Butterworth.

# Compute both: boxcar vs Butterworth applied to the SAME old envelope, to
# isolate the smoother choice.
boxcar_kernel = np.ones(int(0.050*SR)) / int(0.050*SR)
old_env_boxcar = np.convolve(old_env_long, boxcar_kernel, mode='same')
old_env_butter = np.abs(lowpass(old_env_long, SR, fc=10.0))

# 10s zoom for visibility
zoom = slice(int(20*SR), int(30*SR))
t_zoom = (np.arange(zoom.stop - zoom.start)) / SR

fig, ax = plt.subplots(figsize=(13, 4.5))
ax.plot(t_zoom, old_env_long[zoom],   lw=0.4, color='gray', alpha=0.5,
        label='|hilbert| envelope (raw)')
ax.plot(t_zoom, old_env_boxcar[zoom], lw=1.2, color='steelblue',
        label='+ 50ms boxcar  (OLD path)')
ax.plot(t_zoom, old_env_butter[zoom], lw=1.5, color='crimson',
        label='+ 10 Hz Butterworth LPF  (NEW-path-style smoother)')
ax.set_title('Same Hilbert envelope, two smoothers — boxcar leaves visible '
             'wiggles, Butterworth doesn\'t')
ax.set_xlabel('time (s, zoom)'); ax.set_ylabel('envelope')
ax.grid(alpha=0.3); ax.legend()
plt.tight_layout(); plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# Summary printed at the end
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*68)
print("  SUMMARY")
print("="*68)
print("""
  OLD path (Hilbert + 50 ms boxcar):
    • Hilbert envelope shifts the 70-170 Hz oscillation down to baseband
      → most energy lands BELOW the boxcar's 20 Hz first null
      → this is the 'incidental compensation' — boxcar is decent here
    • But envelope content above 20 Hz still leaks through the sidelobes
      (-13 dB at ~30 Hz, -18 dB at ~50 Hz)
    • That residual leakage = noise the classifier has to ignore

  NEW path (x² + 10 Hz Butterworth):
    • x² produces the same baseband envelope as Hilbert (different math,
      same spectral effect — DC + slow modulations)
    • 10 Hz Butterworth LPF actively kills everything above 10 Hz with
      no sidelobes — cleaner separation between phoneme rate (5-10 Hz)
      and noise (>10 Hz)
    • Then a tiny 15 ms window samples the now-clean envelope

  In one phrase:
    Old path RELIES on Hilbert + lucky boxcar geometry.
    New path EXPLICITLY filters with a designed low-pass.
""")
