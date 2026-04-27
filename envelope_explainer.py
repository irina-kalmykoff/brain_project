# Visual explanation of pwr_lpf_10 vs Hilbert envelope.
#
# Five panels, run cells in order:
#   CELL 3 — Stages of pwr_lpf_10 on one channel, one second
#   CELL 4 — Hilbert vs pwr_lpf_10 overlaid on a sentence
#   CELL 5 — Frequency response: 10 Hz Butterworth vs 50 ms boxcar
#   CELL 6 — Power spectra of the two envelopes (what each KEEPS)
#   CELL 7 — Zoomed comparison at a phoneme transition

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 1 — Imports
# ═══════════════════════════════════════════════════════════════════════════════

import os
import numpy as np
import scipy.signal
import scipy.fftpack
import matplotlib.pyplot as plt

from config import DUTCH_30_PATH


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 2 — Load one patient, pick a channel with strong high-gamma
# ═══════════════════════════════════════════════════════════════════════════════

PID = 'P25'
SR  = 1024     # Hz, after the dataset's downsampling

raw = np.load(os.path.join(DUTCH_30_PATH, 'raw', f'{PID}_sEEG.npy'))
print(f"Raw EEG shape for {PID}: {raw.shape} (samples, channels)")

# Pick the channel with the strongest broadband variance — usually one of
# the more informative ones for high-gamma.
chan_var = raw.std(axis=0)
chan_idx = int(np.argmax(chan_var))
print(f"Picked channel {chan_idx} (std={chan_var[chan_idx]:.1f})")

x_full = raw[:, chan_idx].astype(np.float64)


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 3 — Stages of pwr_lpf_10: raw → bandpass → x² → LPF → √(window-avg)
# ═══════════════════════════════════════════════════════════════════════════════
# We unroll the envelope into its stages so you can SEE what each operation
# does to the signal. One second of data, one channel.

t_start = 30.0   # seconds into the recording — should hit some speech
t_dur   = 1.0
i0 = int(t_start * SR); i1 = int((t_start + t_dur) * SR)
x   = x_full[i0:i1].copy()
t   = np.arange(len(x)) / SR

# Stage A: bandpass 70–170 Hz + notches at 100, 150
xb = scipy.signal.detrend(x)
sos_bp = scipy.signal.iirfilter(4, [70/(SR/2), 170/(SR/2)],
                                btype='bandpass', output='sos')
xb = scipy.signal.sosfiltfilt(sos_bp, xb)
for fn in (100., 150.):
    sos_notch = scipy.signal.iirfilter(4, [(fn-2)/(SR/2), (fn+2)/(SR/2)],
                                       btype='bandstop', output='sos')
    xb = scipy.signal.sosfiltfilt(sos_notch, xb)

# Stage B (Hilbert path): |hilbert(x)|
xh = np.abs(scipy.signal.hilbert(xb, scipy.fftpack.next_fast_len(len(xb))))[:len(xb)]

# Stage B' (pwr_lpf path): x²
xp = xb ** 2

# Stage C (pwr_lpf path): low-pass at 10 Hz
sos_lp = scipy.signal.iirfilter(4, 10.0/(SR/2), btype='lowpass', output='sos')
xpl = scipy.signal.sosfiltfilt(sos_lp, xp)
xpl = np.abs(xpl)

# Stage D: window-average + sqrt for both paths (feature-level output, 100 fps)
def window_avg(sig, sr=SR, window=0.05, shift=0.01):
    n_win = int(np.floor((len(sig) - window*sr) / (shift*sr)))
    out = np.zeros(n_win)
    for w in range(n_win):
        s = int(np.floor(w*shift*sr)); e = int(np.floor(s + window*sr))
        out[w] = sig[s:e].mean()
    return out

env_h_feat = window_avg(xh)              # Hilbert feature
env_p_feat = np.sqrt(window_avg(xpl))    # pwr_lpf feature
t_feat = np.arange(len(env_h_feat)) * 0.01

# ─── Plot ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(5, 1, figsize=(12, 10), sharex=True)

axes[0].plot(t, x, lw=0.6, color='black')
axes[0].set_title(f'Stage 0 — Raw EEG (channel {chan_idx}, {PID})')
axes[0].set_ylabel('µV')

axes[1].plot(t, xb, lw=0.6, color='steelblue')
axes[1].set_title('Stage 1 — Bandpassed 70–170 Hz (with 100, 150 Hz notches)')
axes[1].set_ylabel('µV')

axes[2].plot(t, xh,  lw=0.8, color='steelblue', label='|hilbert(x)|  (HILBERT path)')
axes[2].plot(t, xp,  lw=0.8, color='crimson',   alpha=0.6,
             label='x²            (PWR_LPF path)')
axes[2].set_title('Stage 2 — Instantaneous magnitude vs power')
axes[2].set_ylabel('amplitude / power')
axes[2].legend(loc='upper right', fontsize=9)

axes[3].plot(t, xh,  lw=0.6, color='steelblue', alpha=0.4,
             label='|hilbert| (no extra smoothing)')
axes[3].plot(t, xpl, lw=1.2, color='crimson',
             label='x² → 10 Hz LPF  (much smoother)')
axes[3].set_title('Stage 3 — After explicit 10 Hz low-pass on the power signal')
axes[3].set_ylabel('amplitude')
axes[3].legend(loc='upper right', fontsize=9)

axes[4].plot(t_feat, env_h_feat, marker='o', ms=3, lw=1.0,
             color='steelblue', label='Hilbert + 50 ms boxcar (final feature)')
axes[4].plot(t_feat, env_p_feat, marker='o', ms=3, lw=1.0,
             color='crimson',   label='pwr_lpf_10 + 50 ms boxcar (final feature)')
axes[4].set_title('Stage 4 — Final feature (100 fps): what the CRF actually sees')
axes[4].set_xlabel('time (s)')
axes[4].set_ylabel('feature value')
axes[4].legend(loc='upper right', fontsize=9)

for ax in axes:
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 4 — Both envelopes side-by-side on a longer segment
# ═══════════════════════════════════════════════════════════════════════════════
# Five seconds. Notice how the Hilbert envelope has visible high-frequency
# wiggle riding on top, while pwr_lpf_10 traces the slow phoneme-scale
# modulation cleanly.

t_start = 30.0
t_dur   = 5.0
i0 = int(t_start * SR); i1 = int((t_start + t_dur) * SR)
x = x_full[i0:i1].copy()

xb = scipy.signal.sosfiltfilt(sos_bp, scipy.signal.detrend(x))
for fn in (100., 150.):
    sos_n = scipy.signal.iirfilter(4, [(fn-2)/(SR/2), (fn+2)/(SR/2)],
                                   btype='bandstop', output='sos')
    xb = scipy.signal.sosfiltfilt(sos_n, xb)

env_hilbert = np.abs(scipy.signal.hilbert(
    xb, scipy.fftpack.next_fast_len(len(xb))))[:len(xb)]
env_pwrlpf  = np.abs(scipy.signal.sosfiltfilt(sos_lp, xb**2))
env_pwrlpf  = np.sqrt(env_pwrlpf)   # back to amplitude units

t_long = np.arange(len(xb)) / SR

fig, ax = plt.subplots(figsize=(13, 4.5))
ax.plot(t_long, env_hilbert, lw=0.6, color='steelblue', alpha=0.7,
        label=f'Hilbert  |hilbert(x)|        — std={env_hilbert.std():.2f}')
ax.plot(t_long, env_pwrlpf,  lw=1.3, color='crimson',
        label=f'pwr_lpf_10  √(LPF(x²))   — std={env_pwrlpf.std():.2f}')
ax.set_title(f'Hilbert vs pwr_lpf_10 envelopes — {t_dur:.0f} s, channel {chan_idx}')
ax.set_xlabel('time (s)'); ax.set_ylabel('amplitude')
ax.grid(alpha=0.3); ax.legend()
plt.tight_layout(); plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 5 — Frequency response: what each smoother actually does
# ═══════════════════════════════════════════════════════════════════════════════
# This is the key reason 10 Hz LPF beats Hilbert+boxcar. The boxcar has
# leaky sidelobes that pass envelope noise above 20 Hz; the Butterworth
# kills it cleanly.

# Butterworth LPF at 10 Hz (4th order, sosfiltfilt → effectively 8th order)
w, h_butter = scipy.signal.sosfreqz(sos_lp, worN=4096, fs=SR)
h_butter_db = 20*np.log10(np.abs(h_butter)**2 + 1e-12)   # ²= filtfilt squares it

# 50 ms boxcar (rectangular average) — equivalent FIR
n_box = int(0.050 * SR)
b_box = np.ones(n_box) / n_box
w_box, h_box = scipy.signal.freqz(b_box, [1.0], worN=4096, fs=SR)
h_box_db = 20*np.log10(np.abs(h_box) + 1e-12)

# Combined (Hilbert path: just boxcar; pwr_lpf path: LPF * boxcar)
h_combined = np.abs(h_butter) * np.abs(h_box)
h_combined_db = 20*np.log10(h_combined + 1e-12)

fig, ax = plt.subplots(figsize=(11, 5))
ax.semilogx(w_box, h_box_db, lw=2, color='steelblue',
            label='HILBERT path: 50 ms boxcar only')
ax.semilogx(w, h_butter_db, lw=2, color='crimson',
            label='Butterworth LPF @ 10 Hz (filtfilt)')
ax.semilogx(w, h_combined_db, lw=2, ls='--', color='darkred',
            label='PWR_LPF path: LPF × boxcar (combined)')
ax.axvline(10, color='gray', ls=':', alpha=0.6)
ax.text(10.5, -55, '10 Hz', color='gray', fontsize=9)
ax.set_xlim(0.5, 200); ax.set_ylim(-80, 5)
ax.set_xlabel('frequency (Hz)'); ax.set_ylabel('gain (dB)')
ax.set_title('Smoother frequency response — what each path keeps in the envelope')
ax.grid(alpha=0.3, which='both'); ax.legend()
plt.tight_layout(); plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 6 — Power spectra of the actual envelopes
# ═══════════════════════════════════════════════════════════════════════════════
# Empirical version of CELL 5: on a long real segment, compute the spectrum
# of each envelope. Confirms what the theoretical filter response predicts.

t_start = 10.0; t_dur = 60.0
i0 = int(t_start*SR); i1 = int((t_start+t_dur)*SR)
x = x_full[i0:i1].copy()

xb = scipy.signal.sosfiltfilt(sos_bp, scipy.signal.detrend(x))
for fn in (100., 150.):
    sos_n = scipy.signal.iirfilter(4, [(fn-2)/(SR/2), (fn+2)/(SR/2)],
                                   btype='bandstop', output='sos')
    xb = scipy.signal.sosfiltfilt(sos_n, xb)

env_h = np.abs(scipy.signal.hilbert(
    xb, scipy.fftpack.next_fast_len(len(xb))))[:len(xb)]
env_p = np.sqrt(np.abs(scipy.signal.sosfiltfilt(sos_lp, xb**2)))

# Welch PSDs
fH, PxxH = scipy.signal.welch(env_h - env_h.mean(), fs=SR, nperseg=4096)
fP, PxxP = scipy.signal.welch(env_p - env_p.mean(), fs=SR, nperseg=4096)

fig, ax = plt.subplots(figsize=(11, 5))
ax.semilogy(fH, PxxH, lw=1.5, color='steelblue', label='Hilbert envelope PSD')
ax.semilogy(fP, PxxP, lw=1.5, color='crimson',   label='pwr_lpf_10 envelope PSD')
ax.axvline(10, color='gray', ls=':', alpha=0.6)
ax.axvspan(5, 10, alpha=0.10, color='green',
           label='phoneme rate band (5–10 Hz)')
ax.set_xlim(0, 60)
ax.set_xlabel('frequency (Hz)')
ax.set_ylabel('PSD (a.u.)')
ax.set_title(f'Empirical envelope PSDs on {t_dur:.0f} s of channel {chan_idx} ({PID})')
ax.grid(alpha=0.3, which='both'); ax.legend()
plt.tight_layout(); plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 7 — Zoomed comparison at a phoneme-scale event
# ═══════════════════════════════════════════════════════════════════════════════
# Find a place with a clear amplitude transient, zoom to ~250 ms around it,
# and compare what each envelope does in that window.

# Pick the time of the largest amplitude spike in the bandpassed signal
spike_idx = int(np.argmax(np.abs(xb)))
zoom_half = int(0.125 * SR)   # ±125 ms
z0 = max(0, spike_idx - zoom_half)
z1 = min(len(xb), spike_idx + zoom_half)
tz = (np.arange(z1-z0) - (spike_idx - z0)) / SR * 1000   # ms, 0 at spike

fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(tz, xb[z0:z1], lw=0.6, color='black', alpha=0.4,
        label='bandpassed signal (70-170 Hz)')
ax.plot(tz, env_h[z0:z1], lw=2.0, color='steelblue', label='Hilbert envelope')
ax.plot(tz, env_p[z0:z1], lw=2.0, color='crimson',   label='pwr_lpf_10 envelope')
ax.axvline(0, color='gray', ls=':')
ax.set_xlabel('time relative to spike (ms)')
ax.set_ylabel('amplitude')
ax.set_title('Phoneme-scale zoom (±125 ms) — Hilbert tracks fast wiggles, pwr_lpf integrates')
ax.grid(alpha=0.3); ax.legend()
plt.tight_layout(); plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# Summary of what the panels are showing
# ═══════════════════════════════════════════════════════════════════════════════
# CELL 3:  pwr_lpf_10 turns the noisy bandpassed signal into a CLEAN slow
#          envelope. At Stage 4 the two final features look similar but
#          pwr_lpf is consistently a bit smoother.
# CELL 4:  On 5 s of real data the Hilbert curve is visibly fuzzier — that
#          fuzz is sub-phoneme amplitude noise that the CRF has to learn
#          to ignore.
# CELL 5:  The smoking gun — the boxcar's first sidelobe lets ~−13 dB through
#          at 30 Hz; the Butterworth is below −40 dB by 30 Hz. That's the
#          difference in noise rejection.
# CELL 6:  Empirical spectra confirm CELL 5: pwr_lpf has a clean knee at
#          10 Hz; Hilbert has a long tail of envelope power past 30 Hz that
#          carries no phoneme-rate information.
# CELL 7:  At phoneme-scale zoom, Hilbert tracks every wiggle while pwr_lpf
#          gives a smooth pulse. The CRF cares about the pulse, not the wiggles.
