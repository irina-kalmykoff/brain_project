# What does it mean to call the boxcar a "rectangular kernel"?
#
# Six panels:
#   CELL 2 — The kernel itself (FIR coefficients as a stem plot)
#   CELL 3 — Convolution = sliding the kernel across the signal
#   CELL 4 — Build up the convolution output sample by sample
#   CELL 5 — Boxcar kernel vs other window kernels (Hann, Hamming, Gaussian)
#   CELL 6 — Boxcar (FIR) vs Butterworth (IIR) kernel comparison
#   CELL 7 — Coefficient count: 51 taps vs 4 SOS sections


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 1 — Imports
# ═══════════════════════════════════════════════════════════════════════════════

import numpy as np
import scipy.signal
import matplotlib.pyplot as plt

SR     = 1024
WIN_MS = 50
N_TAP  = int(WIN_MS / 1000 * SR)   # 51 samples


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 2 — The boxcar kernel itself
# ═══════════════════════════════════════════════════════════════════════════════
# A boxcar/moving-average filter is just a list of N equal weights, each = 1/N.
# Convolving an input with this kernel = computing the running mean.

box_kernel = np.ones(N_TAP) / N_TAP   # the FIR impulse response

t_kernel = np.arange(N_TAP) / SR * 1000   # ms

fig, axes = plt.subplots(1, 2, figsize=(13, 4))

# Stem plot — emphasises that it's a discrete sequence of weights
ax = axes[0]
ax.stem(t_kernel, box_kernel, basefmt=' ',
        linefmt='steelblue', markerfmt='o')
ax.set_title(f'Boxcar kernel — {N_TAP} taps, each = 1/{N_TAP} = {1/N_TAP:.4f}',
             fontsize=11)
ax.set_xlabel('tap delay (ms)')
ax.set_ylabel('weight h[k]')
ax.set_ylim(0, 1.5/N_TAP)
ax.grid(alpha=0.3)
ax.text(0.5, 0.95,
        'Each output sample = sum of\n51 input samples × 1/51\n= the mean of 50 ms of input',
        transform=ax.transAxes, ha='center', va='top',
        bbox=dict(facecolor='lightyellow', edgecolor='gray', boxstyle='round'),
        fontsize=10)

# Continuous bar plot — emphasises the rectangle shape
ax = axes[1]
ax.bar(t_kernel, box_kernel, width=1000/SR, color='steelblue',
       edgecolor='steelblue', linewidth=0)
ax.set_title('Same kernel as a continuous "rectangle"\n'
             '(why we call it a "rectangular kernel" or "boxcar")',
             fontsize=11)
ax.set_xlabel('tap delay (ms)')
ax.set_ylabel('weight h[k]')
ax.set_ylim(0, 1.5/N_TAP)
ax.grid(alpha=0.3)

plt.tight_layout(); plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 3 — Convolution = sliding the kernel across the signal
# ═══════════════════════════════════════════════════════════════════════════════
# At each output position n, you place the (flipped) kernel at position n,
# multiply pointwise with the corresponding input samples, and sum.
# For a symmetric kernel (like boxcar), flipping doesn't matter.
# This panel shows four snapshots of the kernel sliding across a fake signal.

# Make a fake signal with a clear envelope shape we can see being smoothed
np.random.seed(0)
t_sig = np.arange(int(0.6 * SR)) / SR
sig = (np.sin(2*np.pi*4*t_sig) * np.exp(-((t_sig-0.3)/0.15)**2)
       + 0.15 * np.random.randn(len(t_sig)))

# Pick four positions to show the kernel
positions_ms = [50, 200, 350, 500]   # times where the kernel is centered

fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True)

for ax, pos_ms in zip(axes, positions_ms):
    pos_samp = int(pos_ms / 1000 * SR)
    half_w = N_TAP // 2

    # Plot the signal in the background
    ax.plot(t_sig*1000, sig, lw=0.8, color='gray', alpha=0.7, label='input signal')

    # Highlight the part the kernel currently sees
    sl = slice(max(0, pos_samp-half_w), min(len(sig), pos_samp+half_w+1))
    ax.plot(t_sig[sl]*1000, sig[sl], lw=1.2, color='steelblue',
            label='samples currently under the kernel')

    # Show the kernel as a faint rectangle on top of the signal,
    # scaled so it's visible
    kernel_t = (np.arange(N_TAP) - half_w + pos_samp) / SR * 1000
    kernel_h = np.full(N_TAP, sig[sl].mean())   # plot the kernel at the local mean
    ax.bar(kernel_t, [0.04]*N_TAP, bottom=kernel_h - 0.02,
           width=1000/SR, color='gold', alpha=0.6, edgecolor='none',
           label='boxcar kernel (51 taps)' if pos_ms == positions_ms[0] else None)

    # The output sample (the average) at this position — a single dot
    ax.plot(pos_ms, sig[sl].mean(), 'o', ms=12,
            color='crimson', markerfacecolor='crimson',
            label='output sample = mean(slice)' if pos_ms == positions_ms[0] else None)

    ax.set_title(f'Kernel centered at {pos_ms} ms  →  '
                 f'output[{pos_ms} ms] = {sig[sl].mean():.3f}',
                 fontsize=10)
    ax.set_ylabel('amplitude')
    ax.grid(alpha=0.3)
    if pos_ms == positions_ms[0]:
        ax.legend(loc='upper right', fontsize=8)

axes[-1].set_xlabel('time (ms)')
plt.tight_layout(); plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 4 — Build up the convolution output sample by sample
# ═══════════════════════════════════════════════════════════════════════════════
# Show the input alongside the running output, with the output being filled in
# left-to-right as the kernel slides.  Three snapshots: 25%, 50%, 100% complete.

full_output = np.convolve(sig, box_kernel, mode='same')

fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
fractions = [0.25, 0.50, 1.00]

for ax, frac in zip(axes, fractions):
    n_done = int(frac * len(sig))
    ax.plot(t_sig*1000, sig, lw=0.8, color='lightgray',
            label='input signal')
    ax.plot(t_sig[:n_done]*1000, full_output[:n_done], lw=1.5, color='crimson',
            label=f'output so far ({int(frac*100)}% complete)')
    ax.set_title(f'Convolution at {int(frac*100)}% complete', fontsize=10)
    ax.set_ylabel('amplitude')
    ax.grid(alpha=0.3)
    ax.legend(loc='upper right', fontsize=9)

axes[-1].set_xlabel('time (ms)')
plt.tight_layout(); plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 5 — Boxcar vs other window kernels
# ═══════════════════════════════════════════════════════════════════════════════
# All of these are FIR kernels, just with different shapes.  The boxcar is
# the simplest; the others trade a wider main lobe for smaller sidelobes.

windows = {
    'boxcar (rectangular)':  np.ones(N_TAP) / N_TAP,
    'Hann':      scipy.signal.windows.hann(N_TAP)     / scipy.signal.windows.hann(N_TAP).sum(),
    'Hamming':   scipy.signal.windows.hamming(N_TAP)  / scipy.signal.windows.hamming(N_TAP).sum(),
    'Blackman':  scipy.signal.windows.blackman(N_TAP) / scipy.signal.windows.blackman(N_TAP).sum(),
    'Gaussian σ=10ms': scipy.signal.windows.gaussian(N_TAP, std=10/1000*SR) /
                       scipy.signal.windows.gaussian(N_TAP, std=10/1000*SR).sum(),
}

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

# Time domain — the kernels themselves
ax = axes[0]
for name, k in windows.items():
    ax.plot(t_kernel, k, lw=2, label=name)
ax.set_xlabel('tap delay (ms)')
ax.set_ylabel('weight h[k]')
ax.set_title('Kernel shapes (FIR coefficients) — all same length, different shape',
             fontsize=11)
ax.grid(alpha=0.3); ax.legend(fontsize=9)

# Frequency domain — what each kernel does
ax = axes[1]
for name, k in windows.items():
    w, h = scipy.signal.freqz(k, [1.0], worN=8192, fs=SR)
    ax.plot(w, 20*np.log10(np.abs(h)+1e-12), lw=2, label=name)
ax.set_xscale('log')
ax.set_xlim(0.5, 200); ax.set_ylim(-100, 5)
ax.set_xlabel('frequency (Hz, log)'); ax.set_ylabel('gain (dB)')
ax.set_title('Frequency response — smoother kernels = smaller sidelobes',
             fontsize=11)
ax.grid(alpha=0.3, which='both'); ax.legend(fontsize=9)

plt.tight_layout(); plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 6 — Boxcar (FIR) vs Butterworth (IIR) kernel comparison
# ═══════════════════════════════════════════════════════════════════════════════
# Even an IIR filter has a kernel — its impulse response.  The difference:
# FIR kernel is exactly N samples long then zero forever.
# IIR kernel decays exponentially, theoretically forever.

# Compute the Butterworth's impulse response
sos_butter = scipy.signal.iirfilter(4, 10/(SR/2), btype='lowpass', output='sos')

# Two versions: causal single-pass (asymmetric) and zero-phase filtfilt (symmetric)
imp = np.zeros(int(0.4*SR)); imp[len(imp)//2] = 1.0
butter_causal   = scipy.signal.sosfilt(sos_butter, imp)
butter_filtfilt = scipy.signal.sosfiltfilt(sos_butter, imp)
t_imp = (np.arange(len(imp)) - len(imp)//2) / SR * 1000

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

# Side-by-side kernels
ax = axes[0]
# Boxcar kernel — center it at 0 for visual comparison with filtfilt
t_box_centered = (np.arange(N_TAP) - N_TAP//2) / SR * 1000
ax.bar(t_box_centered, box_kernel, width=1000/SR, color='steelblue',
       edgecolor='steelblue', label='boxcar kernel  (FIR, 51 taps)')
ax.plot(t_imp, butter_causal,   lw=1.5, color='darkorange',
        label='Butterworth impulse resp.  (IIR, single-pass — causal)')
ax.plot(t_imp, butter_filtfilt, lw=1.5, color='crimson',
        label='Butterworth impulse resp.  (IIR, filtfilt — zero-phase)')
ax.set_xlim(-200, 200)
ax.set_xlabel('time (ms)')
ax.set_ylabel('weight')
ax.set_title('Kernels in the time domain', fontsize=11)
ax.grid(alpha=0.3); ax.legend(fontsize=9, loc='upper right')

# Same on a log y to see how IIR decays past the FIR's edge
ax = axes[1]
ax.semilogy(t_box_centered, box_kernel + 1e-12, lw=2, color='steelblue',
            label='boxcar kernel')
ax.semilogy(t_imp, np.abs(butter_filtfilt) + 1e-12, lw=1.5, color='crimson',
            label='Butterworth |h(t)| (filtfilt)')
ax.set_xlim(-300, 300); ax.set_ylim(1e-6, 1)
ax.set_xlabel('time (ms)')
ax.set_ylabel('|weight|')
ax.set_title('Same kernels on log y — boxcar is exactly 0 outside ±25 ms;\n'
             'Butterworth decays exponentially but never reaches 0',
             fontsize=11)
ax.grid(alpha=0.3, which='both'); ax.legend(fontsize=9)

plt.tight_layout(); plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 7 — How many coefficients does each filter actually need?
# ═══════════════════════════════════════════════════════════════════════════════
# A surprising fact: the boxcar (FIR) needs FAR more coefficients than the
# Butterworth (IIR) for the same time scale.  IIR is more efficient because
# it uses recursion.

print(f"\n{'='*68}")
print(f"  Filter coefficient counts")
print(f"{'='*68}\n")

print(f"  BOXCAR (FIR)           {N_TAP} taps")
print(f"    h = [{box_kernel[0]:.4f}, {box_kernel[1]:.4f}, ..., "
      f"{box_kernel[-1]:.4f}]   (all equal)")
print(f"    Output formula:")
print(f"      y[n] = (x[n] + x[n-1] + ... + x[n-{N_TAP-1}]) / {N_TAP}")
print()

print(f"  BUTTERWORTH 4th-order  4 second-order sections (SOS), 4×6 = 24 numbers")
print(f"    Each SOS section has 6 coefficients: [b0, b1, b2, a0, a1, a2]")
print(f"    Output formula (per section):")
print(f"      y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2]")
print(f"             - a1*y[n-1] - a2*y[n-2]      ← THE recursion")
print()

print(f"    Numerical content of the 4 SOS sections:")
for i, sec in enumerate(sos_butter):
    print(f"      section {i}: b = [{sec[0]:.4e}, {sec[1]:.4e}, {sec[2]:.4e}],"
          f"  a = [1, {sec[4]:.4e}, {sec[5]:.4e}]")
print()

print(f"  Effective time scale:")
print(f"    boxcar      = exactly 50 ms  (all 51 taps, then zero forever)")
print(f"    Butterworth ≈ 80 ms 1%-decay (then keeps decaying exponentially)")
print()

print(f"""  Why IIR can do more with fewer coefficients:
    Each output sample re-uses the *previous output* (recursion).  That
    means ~80 ms of "memory" doesn't require storing 80 ms of weights —
    it lives implicitly in the recursive feedback.  The 24 numbers above
    encode the SAME smoothing behaviour that would need hundreds of FIR
    taps to approximate accurately.

  Key takeaway:
    • FIR (boxcar) — kernel is literally a list of weights, finite length
    • IIR (Butterworth) — kernel is "made up" by recursion, theoretically
      infinite length, but each output costs only a few multiplies
""")
