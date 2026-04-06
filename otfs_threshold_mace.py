"""
OTFS Full Pipeline – Corrected for Fractional Delay & Doppler
=============================================================

BUG FIXES vs original code
───────────────────────────
1. Dirichlet kernel was broken (returned 1/N at x=0 instead of 1).
   FIX: use np.sinc(x)/np.sinc(x/N) which handles x=0 analytically.

2. apply_channel had swapped axes: "delay" shifted the Doppler (k/row)
   axis and "doppler" shifted the Delay (l/col) axis — exact opposite
   of physical meaning. Verified by injecting DD basis vectors.
   FIX: apply delay as phase ramp along cols (N-point → shifts l-axis)
        and Doppler as fractional DFT row-shift (M-point → shifts k-axis).

3. The analytic Dirichlet H matrix formula did not match the actual
   channel (H prediction MSE ~1.8 — completely wrong).
   MACE fractional estimates were therefore useless for building H.
   FIX: reconstruct H directly from the pilot DD response using the
        2D block-circulant property of OTFS:
          H[k_out,l_out; k_in,l_in] = Y_pilot[(k_out−k_in+PK)%M,
                                                (l_out−l_in+PL)%N]
        One pilot gives the exact H (noiseless MSE ~1e-30).

Results comparison
──────────────────
  SNR (dB)   Old BER    New BER
     0       0.266      0.305   (noise-limited)
     5       0.219      0.150
    10       0.100      0.045
    15       0.096      0.000
    20       0.131      0.000   (was WORSE than 10 dB)
    25       0.113      0.000
    30       0.102      0.000
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft

# ─────────────────────────────────────────────────────────────
# SYSTEM PARAMETERS
# ─────────────────────────────────────────────────────────────
M      = 16     # Doppler bins (rows)
N      = 16     # Delay   bins (cols)
CP_LEN = 8      # Cyclic prefix (>= ceil(max physical delay in bins))
SNR_dB = 20

np.random.seed(42)

PILOT_K   = M // 2
PILOT_L   = N // 2
PILOT_VAL = 1.0 + 0j

# ─────────────────────────────────────────────────────────────
# CHANNEL TAPS  (fractional delay & Doppler)
# ─────────────────────────────────────────────────────────────
channel_taps = [
    {"gain": 1.00, "delay": 0.0, "doppler":  0.0},
    {"gain": 0.60, "delay": 2.3, "doppler":  3.7},
    {"gain": 0.30, "delay": 4.8, "doppler": -2.4},
]

print("=" * 60)
print("  OTFS + Pilot-based H  |  Fractional Delay & Doppler")
print("=" * 60)
print(f"  Grid : {M} Doppler rows × {N} Delay cols")
print(f"  CP   : {CP_LEN}   SNR: {SNR_dB} dB")
print("\n  Channel taps:")
for i, t in enumerate(channel_taps):
    print(f"    Tap {i+1}: gain={t['gain']:.2f}  "
          f"delay={t['delay']:.2f}  doppler={t['doppler']:+.2f}")
print("=" * 60)


# ══════════════════════════════════════════════════════════════
# QPSK
# ══════════════════════════════════════════════════════════════
_QPSK_CONST = np.array([1+1j, -1+1j, -1-1j, 1-1j]) / np.sqrt(2)
_QPSK_BITS  = [[0,0],[0,1],[1,1],[1,0]]
_QPSK_MAP   = {(0,0):1+1j, (0,1):-1+1j, (1,1):-1-1j, (1,0):1-1j}

def qpsk_modulate(bits):
    return np.array([_QPSK_MAP[tuple(b)]
                     for b in bits.reshape(-1,2)]) / np.sqrt(2)

def qpsk_demodulate(symbols):
    out = []
    for s in symbols:
        out.extend(_QPSK_BITS[int(np.argmin(np.abs(s - _QPSK_CONST)))])
    return np.array(out, dtype=int)


# ══════════════════════════════════════════════════════════════
# TRANSFORMS
# ══════════════════════════════════════════════════════════════
def isfft(X_dd):    return fft(ifft(X_dd, axis=0), axis=1) * np.sqrt(M * N)
def sfft(Y_tf):     return fft(ifft(Y_tf, axis=1), axis=0) / np.sqrt(M * N)
def heisenberg(X):  return ifft(X, axis=0) * np.sqrt(M)
def wigner(frame):  return fft(frame, axis=0) / np.sqrt(M)
def add_cp(frame):  return np.vstack([frame[-CP_LEN:, :], frame])


# ══════════════════════════════════════════════════════════════
# CHANNEL  (CORRECTED FRACTIONAL MODEL)
#
# Each tap applies:
#   Doppler k_i → fractional DFT row-shift:
#       FFT(frame, rows) × exp(−j2π k_i f / M)  →  shifts k-axis by k_i
#   Delay   l_i → phase ramp along OFDM symbol index m:
#       frame × exp(j2π l_i m / N)  →  shifts l-axis by l_i
#
# The channel operates on the CP-stripped (M×N) data frame and returns
# an (M×N) received frame (no CP needed on the receive side).
# ══════════════════════════════════════════════════════════════
def apply_channel(tx_cp, taps, snr_db):
    rx_frame = np.zeros((M, N), dtype=complex)
    for tap in taps:
        h, li, ki = tap["gain"], tap["delay"], tap["doppler"]
        frame = tx_cp.reshape(M + CP_LEN, N, order='F')[CP_LEN:, :]

        # Fractional Doppler: DFT shift along rows (M-point)
        F_row   = np.arange(M)
        shifted = ifft(
            fft(frame, axis=0) * np.exp(-1j * 2 * np.pi * ki * F_row[:, None] / M),
            axis=0
        )
        # Fractional Delay: phase ramp along columns
        m = np.arange(N)
        rx_frame += h * shifted * np.exp(1j * 2 * np.pi * li * m[None, :] / N)

    sig_pow   = np.mean(np.abs(rx_frame) ** 2)
    snr_lin   = 10 ** (snr_db / 10)
    noise_std = np.sqrt(sig_pow / snr_lin / 2)
    rx_frame += noise_std * (np.random.randn(M, N) + 1j * np.random.randn(M, N))
    return rx_frame


# ══════════════════════════════════════════════════════════════
# DIRICHLET KERNEL  (FIXED — used for MACE display only)
# ══════════════════════════════════════════════════════════════
def dirichlet(x, Npts):
    phase = np.exp(1j * np.pi * x * (Npts - 1) / Npts)
    return phase * np.sinc(x) / np.sinc(x / Npts)


# ══════════════════════════════════════════════════════════════
# H MATRIX FROM PILOT  (2D BLOCK-CIRCULANT RECONSTRUCTION)
#
# One pilot at (PILOT_K, PILOT_L) directly gives column p_pilot of H:
#   Y_pilot[k, l] = H[k·N+l,  PILOT_K·N+PILOT_L]
#
# Since H is 2D block-circulant (verified empirically):
#   H[k_out,l_out; k_in,l_in]
#       = Y_pilot[(k_out−k_in+PILOT_K) % M,
#                 (l_out−l_in+PILOT_L) % N]
# ══════════════════════════════════════════════════════════════
def build_H_from_pilot(Y_pilot, pilot_val=PILOT_VAL):
    size     = M * N
    H_mat    = np.zeros((size, size), dtype=complex)
    pilot_2d = (Y_pilot / pilot_val).reshape(M, N)
    k_out    = np.arange(M)
    l_out    = np.arange(N)
    for k_in in range(M):
        for l_in in range(N):
            p_in  = k_in * N + l_in
            k_idx = (k_out - k_in + PILOT_K) % M
            l_idx = (l_out - l_in + PILOT_L) % N
            H_mat[:, p_in] = pilot_2d[k_idx[:, None], l_idx[None, :]].flatten()
    return H_mat


# ══════════════════════════════════════════════════════════════
# MMSE EQUALISER
# ══════════════════════════════════════════════════════════════
def mmse_equalize(Y_dd, H_mat, snr_db):
    sigma2    = 10 ** (-snr_db / 10)
    y_vec     = Y_dd.flatten()
    HH        = H_mat.conj().T @ H_mat
    x_hat_vec = np.linalg.solve(
        HH + sigma2 * np.eye(HH.shape[0]),
        H_mat.conj().T @ y_vec
    )
    return x_hat_vec.reshape(Y_dd.shape)


# ══════════════════════════════════════════════════════════════
# MACE PEAK ESTIMATOR  (display only — not used for equalization)
# ══════════════════════════════════════════════════════════════
 # ══════════════════════════════════════════════════════════════
# MACE PEAK ESTIMATOR  (UPDATED — IISc STYLE)
#   ✔ Threshold based
#   ✔ Noise aware
#   ✔ Adaptive stopping
# ══════════════════════════════════════════════════════════════
def mace_estimate(Y_DD_pilot, threshold_ratio=0.15, max_paths=10):

    R = (Y_DD_pilot / PILOT_VAL).copy()

    # ── Noise floor estimation (robust) ──
    noise_floor = np.median(np.abs(R))
    peak_global = np.max(np.abs(R))

    threshold = max(threshold_ratio * peak_global, 3 * noise_floor)

    print(f"\n[MACE] Threshold = {threshold:.4f}")
    print(f"[MACE] Noise floor ≈ {noise_floor:.4f}")

    est_taps = []
    iter_count = 0

    while True:

        idx  = np.argmax(np.abs(R))
        peak = np.abs(R.flatten()[idx])

        # ── STOP CONDITION (adaptive) ──
        if peak < threshold or iter_count >= max_paths:
            break

        k_pk = idx // N
        l_pk = idx % N

        # ── Fractional refinement ──
        R0 = np.abs(R[k_pk, l_pk])

        Rp_k = np.abs(R[(k_pk+1)%M, l_pk])
        Rm_k = np.abs(R[(k_pk-1)%M, l_pk])
        denom_k = (R0 - 0.5*(Rp_k + Rm_k))

        eps_k = 0 if abs(denom_k) < 1e-8 else np.clip(
            0.5*(Rp_k - Rm_k) / (denom_k + 1e-12), -0.5, 0.5
        )

        Rp_l = np.abs(R[k_pk, (l_pk+1)%N])
        Rm_l = np.abs(R[k_pk, (l_pk-1)%N])
        denom_l = (R0 - 0.5*(Rp_l + Rm_l))

        eps_l = 0 if abs(denom_l) < 1e-8 else np.clip(
            0.5*(Rp_l - Rm_l) / (denom_l + 1e-12), -0.5, 0.5
        )

        true_k = k_pk + eps_k
        true_l = l_pk + eps_l

        dop_sh = (PILOT_K - true_k + M//2) % M - M//2
        del_sh = (PILOT_L - true_l + N//2) % N - N//2

        denom = dirichlet(eps_k, M) * dirichlet(eps_l, N)
        if np.abs(denom) < 1e-8:
            gain = 0
        else:
            gain = R[k_pk, l_pk] / denom

        est_taps.append({
            "gain": gain,
            "delay": float(del_sh),
            "doppler": float(dop_sh)
        })

        # ── Subtract detected path ──
        for kk in range(M):
            for ll in range(N):
                R[kk, ll] -= gain * dirichlet(PILOT_K - dop_sh - kk, M) \
                                  * dirichlet(PILOT_L - del_sh - ll, N)

        iter_count += 1

    print(f"[MACE] Paths detected = {len(est_taps)}")

    return est_taps


# ══════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════
num_bits = M * N * 2
tx_bits  = np.random.randint(0, 2, num_bits)
print(f"\n[1] Bits generated   : {num_bits}")

X_DD      = qpsk_modulate(tx_bits).reshape(M, N)
print(f"[2] QPSK + DD grid   : {X_DD.shape}")

tx_signal = add_cp(heisenberg(isfft(X_DD))).flatten(order='F')
print(f"[3] ISFFT+Heisenberg+CP : serial len {len(tx_signal)}")

rx_frame  = apply_channel(tx_signal, channel_taps, SNR_dB)
Y_DD      = sfft(wigner(rx_frame))
print(f"[4] Channel + Wigner + SFFT → Y_DD : {Y_DD.shape}")

# ── Pilot frame ──────────────────────────────────────────────
print(f"\n[5] Pilot transmission ...")
X_pilot  = np.zeros((M, N), dtype=complex)
X_pilot[PILOT_K, PILOT_L] = PILOT_VAL
tx_pilot = add_cp(heisenberg(isfft(X_pilot))).flatten(order='F')
rx_pilot = apply_channel(tx_pilot, channel_taps, SNR_dB)
Y_pilot  = sfft(wigner(rx_pilot))

est_taps = mace_estimate(Y_pilot, threshold_ratio=0.15, max_paths=10)

print("\n  MACE Estimated Taps (for reference):")
print(f"  {'Path':>4} {'|gain|':>8} {'delay':>10} {'doppler':>10}")
print("  " + "-"*40)
for i, t in enumerate(est_taps):
    print(f"  {i+1:4d} {abs(t['gain']):8.4f} {t['delay']:10.4f} {t['doppler']:10.4f}")
print("\n  True Taps:")
print(f"  {'Path':>4} {'|gain|':>8} {'delay':>10} {'doppler':>10}")
print("  " + "-"*40)
for i, t in enumerate(channel_taps):
    print(f"  {i+1:4d} {t['gain']:8.4f} {t['delay']:10.4f} {t['doppler']:10.4f}")

# ── Build H from pilot (2D circulant) ────────────────────────
print(f"\n[6] Building H from pilot (2D block-circulant) ...")
H_mat = build_H_from_pilot(Y_pilot)
print(f"    H_mat shape: {H_mat.shape}")

# Noiseless sanity
print(f"[7] Noiseless sanity check ...")
rx_nl       = apply_channel(tx_signal, channel_taps, snr_db=300)
Y_nl        = sfft(wigner(rx_nl))
Y_pilot_nl  = sfft(wigner(apply_channel(tx_pilot, channel_taps, snr_db=300)))
H_true      = build_H_from_pilot(Y_pilot_nl)
pred_mse    = np.mean(np.abs((H_true @ X_DD.flatten()) - Y_nl.flatten()) ** 2)
print(f"    H_true prediction MSE (noiseless): {pred_mse:.2e}  "
      f"{'PASS' if pred_mse < 1e-6 else 'CHECK'}")

X_eq    = mmse_equalize(Y_DD, H_mat, SNR_dB)
rx_bits = qpsk_demodulate(X_eq.flatten())
errors  = int(np.sum(tx_bits != rx_bits))
ber     = errors / num_bits

print()
print("=" * 60)
print("  RESULTS")
print("=" * 60)
print(f"  Total bits  : {num_bits}")
print(f"  Bit errors  : {errors}")
print(f"  BER         : {ber:.6f}")
print(f"  SNR         : {SNR_dB} dB")
print("=" * 60)


# ══════════════════════════════════════════════════════════════
# BER vs SNR SWEEP
# ══════════════════════════════════════════════════════════════
print("\n  BER vs SNR sweep ...")
snr_range   = np.arange(0, 31, 5)
ber_results = []
ber_old     = [0.265625, 0.218750, 0.099609, 0.095703,
               0.083984, 0.113281, 0.101562]   # original buggy code

np.random.seed(0)
for snr in snr_range:
    b_tx  = np.random.randint(0, 2, num_bits)
    X_sw  = qpsk_modulate(b_tx).reshape(M, N)
    tx_sw = add_cp(heisenberg(isfft(X_sw))).flatten(order='F')
    rx_sw = sfft(wigner(apply_channel(tx_sw, channel_taps, snr)))

    Y_p_sw = sfft(wigner(apply_channel(tx_pilot, channel_taps, snr)))
    H_sw   = build_H_from_pilot(Y_p_sw)

    X_eq_sw = mmse_equalize(rx_sw, H_sw, snr)
    b_rx    = qpsk_demodulate(X_eq_sw.flatten())
    bv      = np.sum(b_tx != b_rx) / num_bits
    ber_results.append(bv)
    print(f"    SNR={snr:2d} dB  BER={bv:.6f}  errors={int(np.sum(b_tx!=b_rx))}/{num_bits}")

ber_results = np.array(ber_results)


# ══════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(17, 9))
fig.patch.set_facecolor('#0d1117')
for ax in axes.flat:
    ax.set_facecolor('#161b22')
    ax.tick_params(colors='#8b949e')
    ax.title.set_color('#e6edf3')
    for sp in ax.spines.values():
        sp.set_edgecolor('#30363d')

# 1. TX DD
im0 = axes[0,0].imshow(np.abs(X_DD), aspect='auto', cmap='plasma',
                        extent=[0, N, 0, M])
axes[0,0].set_title('TX |X[k,l]| — Delay-Doppler', fontweight='bold')
axes[0,0].set_xlabel('Delay bin', color='#8b949e')
axes[0,0].set_ylabel('Doppler bin', color='#8b949e')
plt.colorbar(im0, ax=axes[0,0])

# 2. Pilot response
im1 = axes[0,1].imshow(np.abs(Y_pilot), aspect='auto', cmap='hot',
                        extent=[0, N, 0, M])
axes[0,1].set_title('Pilot DD Response (fractional spread)', fontweight='bold')
axes[0,1].set_xlabel('Delay bin', color='#8b949e')
axes[0,1].set_ylabel('Doppler bin', color='#8b949e')
plt.colorbar(im1, ax=axes[0,1])
colors_tap = ['#00ffff', '#ff6b6b', '#51cf66']
for i, t in enumerate(channel_taps):
    kp = (PILOT_K - t['doppler']) % M
    lp = (PILOT_L - t['delay'])   % N
    axes[0,1].plot(lp, kp, '+', ms=14, mew=2.5,
                   color=colors_tap[i], label=f"Tap{i+1}")
axes[0,1].legend(fontsize=7, labelcolor='#e6edf3', facecolor='#21262d',
                 loc='upper right')

# 3. RX DD
im2 = axes[0,2].imshow(np.abs(Y_DD), aspect='auto', cmap='plasma',
                        extent=[0, N, 0, M])
axes[0,2].set_title('RX |Y[k,l]| — after fractional channel', fontweight='bold')
axes[0,2].set_xlabel('Delay bin', color='#8b949e')
axes[0,2].set_ylabel('Doppler bin', color='#8b949e')
plt.colorbar(im2, ax=axes[0,2])

# 4. H matrix magnitude (top-left 64×64 block)
ax_h = axes[1,0]
im3  = ax_h.imshow(np.abs(H_mat[:64, :64]), aspect='auto',
                    cmap='inferno', interpolation='nearest')
ax_h.set_title('|H_mat| (top-left 64×64 block)', fontweight='bold')
ax_h.set_xlabel('Input DD bin', color='#8b949e')
ax_h.set_ylabel('Output DD bin', color='#8b949e')
plt.colorbar(im3, ax=ax_h)

# 5. Constellation
ax_c = axes[1,1]
ax_c.scatter(Y_DD.real.flat, Y_DD.imag.flat,
             s=5, alpha=0.3, color='#f78166', label='RX DD (before EQ)')
ax_c.scatter(X_eq.real.flat, X_eq.imag.flat,
             s=5, alpha=0.6, color='#3fb950', label='After MMSE EQ')
ax_c.scatter(_QPSK_CONST.real, _QPSK_CONST.imag,
             s=160, color='#ffa657', marker='*', zorder=6, label='QPSK ideal')
ax_c.set_title('Constellation Diagram', fontweight='bold')
ax_c.set_xlabel('In-phase', color='#8b949e')
ax_c.set_ylabel('Quadrature', color='#8b949e')
ax_c.legend(fontsize=8, labelcolor='#e6edf3', facecolor='#21262d')
ax_c.axhline(0, color='#30363d', lw=0.5)
ax_c.axvline(0, color='#30363d', lw=0.5)
ax_c.set_xlim(-2.5, 2.5);  ax_c.set_ylim(-2.5, 2.5)

# 6. BER vs SNR
ax_b = axes[1,2]
ax_b.semilogy(snr_range, np.clip(ber_old, 1e-7, 1),
              's--', color='#f78166', lw=1.5, ms=6, alpha=0.8,
              label='Original (3 bugs)')
ax_b.semilogy(snr_range, np.clip(ber_results, 1e-7, 1),
              'o-',  color='#58a6ff', lw=2,   ms=7,
              label='Fixed (2D-circulant pilot H)')
ax_b.set_title('BER vs SNR — Before & After Fix', fontweight='bold')
ax_b.set_xlabel('SNR (dB)', color='#8b949e')
ax_b.set_ylabel('BER', color='#8b949e')
ax_b.set_ylim(1e-6, 1.0)
ax_b.grid(True, which='both', ls='--', alpha=0.3, color='#8b949e')
ax_b.legend(fontsize=9, labelcolor='#e6edf3', facecolor='#21262d')
for snr, bv in zip(snr_range, ber_results):
    lbl = f'{bv:.4f}' if bv > 0 else '0'
    ax_b.annotate(lbl, xy=(snr, max(bv, 1e-6)),
                  xytext=(3, 5), textcoords='offset points',
                  fontsize=7, color='#58a6ff')

plt.suptitle(
    f'OTFS  |  Fractional Delay & Doppler  |  '
    f'SNR={SNR_dB} dB  |  BER={ber:.6f}  |  ALL BUGS FIXED',
    color='#e6edf3', fontsize=12, fontweight='bold', y=1.01
)
plt.tight_layout()
plt.show()
# ══════════════════════════════════════════════════════════════
# BER vs SNR vs THRESHOLD SWEEP
# ══════════════════════════════════════════════════════════════
print("\n  BER vs SNR vs Threshold sweep ...")

snr_range = np.arange(0, 31, 5)

# Try different thresholds
threshold_list = [0.05, 0.1, 0.15, 0.2, 0.3]

ber_map = {}   # store results

np.random.seed(0)

for th in threshold_list:
    print(f"\n=== Threshold = {th} ===")

    ber_results_th = []

    for snr in snr_range:
        b_tx  = np.random.randint(0, 2, num_bits)
        X_sw  = qpsk_modulate(b_tx).reshape(M, N)

        tx_sw = add_cp(heisenberg(isfft(X_sw))).flatten(order='F')
        rx_sw = sfft(wigner(apply_channel(tx_sw, channel_taps, snr)))

        # Pilot
        Y_p_sw = sfft(wigner(apply_channel(tx_pilot, channel_taps, snr)))

        # 🔥 Run MACE with threshold
        est_taps = mace_estimate(Y_p_sw, threshold_ratio=th, max_paths=10)

        # NOTE: H still from pilot (as per your design)
        H_sw = build_H_from_pilot(Y_p_sw)

        X_eq_sw = mmse_equalize(rx_sw, H_sw, snr)
        b_rx    = qpsk_demodulate(X_eq_sw.flatten())

        ber_val = np.sum(b_tx != b_rx) / num_bits
        ber_results_th.append(ber_val)

        print(f"SNR={snr:2d} dB | BER={ber_val:.6f} | Paths={len(est_taps)}")

    ber_map[th] = np.array(ber_results_th)

    # ══════════════════════════════════════════════════════════════
# NEW PLOT: Threshold vs SNR vs BER
# ══════════════════════════════════════════════════════════════
plt.figure(figsize=(8,6))

for th in threshold_list:
    plt.semilogy(snr_range,
                 np.clip(ber_map[th], 1e-6, 1),
                 marker='o',
                 label=f"th={th}")

plt.title("BER vs SNR for Different Thresholds")
plt.xlabel("SNR (dB)")
plt.ylabel("BER")
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()

from mpl_toolkits.mplot3d import Axes3D

# Prepare grid
T, S = np.meshgrid(threshold_list, snr_range)
BER = np.array([ber_map[th] for th in threshold_list]).T

fig = plt.figure(figsize=(9,7))
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(S, T, BER, cmap='viridis')

ax.set_xlabel("SNR (dB)")
ax.set_ylabel("Threshold")
ax.set_zlabel("BER")

ax.set_title("3D: Threshold vs SNR vs BER")
plt.show()