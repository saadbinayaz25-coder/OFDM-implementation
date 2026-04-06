"""
OTFS (Orthogonal Time Frequency Space) Full Pipeline  — CORRECTED
==================================================================
3-Tap Multipath Channel: delay + Doppler + AWGN.

Key corrections over the original broken version:
  1. ISFFT/SFFT normalisation made symmetric so SFFT(ISFFT(X)) == X exactly.
  2. Channel model: Doppler phase ramp runs over OFDM symbol index n in [0,N-1]
     (NOT over the serial sample index), matching the DD convolution model.
  3. Channel matrix H_mat: correct index formula derived from empirical probing.
     The delay shifts the Doppler (row) axis and Doppler shifts the Delay (col)
     axis in this pipeline convention:
         k_in = (k_out + l_i) % M      <- delay  shifts rows
         l_in = (l_out + k_i) % N      <- Doppler shifts cols
  4. MMSE regularisation replaces pure ZF for robustness at finite SNR.

Result: BER = 0.000000 at SNR = 20 dB  (drops gracefully at lower SNR).

Pipeline:
  Bits -> QPSK -> X[k,l] (DD) -> ISFFT -> X[n,m] (TF)
       -> IFFT (Heisenberg) -> Add CP -> Channel -> Remove CP
       -> FFT (Wigner) -> Y[n,m] -> SFFT -> Y[k,l]
       -> MMSE Equalisation (correct H_mat) -> QPSK Detection -> Bits
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft

# ─────────────────────────────────────────────
# SYSTEM PARAMETERS
# ─────────────────────────────────────────────
M      = 16     # Rows  = Doppler bins
N      = 16     # Cols  = Delay bins
CP_LEN = 8      # Cyclic prefix (must be >= max tap delay)
SNR_dB = 20     # Signal-to-noise ratio in dB

np.random.seed(42)

# ─────────────────────────────────────────────
# CHANNEL TAPS  (gain, delay_samples, doppler_bins)
# ─────────────────────────────────────────────
channel_taps = [
    {"gain": 1.00, "delay": 0, "doppler":  0},   # LOS
    {"gain": 0.60, "delay": 2, "doppler":  3},   # reflected
    {"gain": 0.30, "delay": 4, "doppler": -2},   # scattered
]

print("=" * 55)
print("  OTFS Pipeline  |  3-Tap Channel  (CORRECTED)")
print("=" * 55)
print(f"  Grid size : {M} (Doppler rows) x {N} (Delay cols)")
print(f"  CP length : {CP_LEN}")
print(f"  SNR       : {SNR_dB} dB")
print()
print("  Channel Taps:")
for i, tap in enumerate(channel_taps):
    print(f"    Tap {i+1}: gain={tap['gain']:.2f}  "
          f"delay={tap['delay']} samp  "
          f"Doppler={tap['doppler']:+d} bins")
print("=" * 55)


# ══════════════════════════════════════════════════════════════
# QPSK helpers
# ══════════════════════════════════════════════════════════════
_QPSK_MAP   = {(0,0):  1+1j,
               (0,1): -1+1j,
               (1,1): -1-1j,
               (1,0):  1-1j}
_QPSK_CONST = np.array([1+1j, -1+1j, -1-1j, 1-1j]) / np.sqrt(2)
_QPSK_BITS  = [[0,0],[0,1],[1,1],[1,0]]

def qpsk_modulate(bits):
    return np.array([_QPSK_MAP[tuple(b)] for b in bits.reshape(-1,2)]) / np.sqrt(2)

def qpsk_demodulate(symbols):
    out = []
    for s in symbols:
        out.extend(_QPSK_BITS[int(np.argmin(np.abs(s - _QPSK_CONST)))])
    return np.array(out, dtype=int)


# ══════════════════════════════════════════════════════════════
# TRANSFORMS
#
# ISFFT (DD -> TF):
#   Step 1: IDFT along Doppler axis (rows)  ->  A = ifft(X_DD, axis=0)
#   Step 2: DFT  along Delay  axis (cols)   ->  X_TF = fft(A, axis=1)
#   Scale by sqrt(MN) so that SFFT(ISFFT(X)) == X exactly.
#
# SFFT (TF -> DD):
#   Step 1: IDFT along Delay  axis (cols)   ->  B = ifft(Y_TF, axis=1)
#   Step 2: DFT  along Doppler axis (rows)  ->  Y_DD = fft(B, axis=0)
#   Scale by 1/sqrt(MN)
# ══════════════════════════════════════════════════════════════
def isfft(X_dd):
    A    = ifft(X_dd, axis=0)
    X_tf = fft(A,    axis=1)
    return X_tf * np.sqrt(M * N)

def sfft(Y_tf):
    B    = ifft(Y_tf, axis=1)
    Y_dd = fft(B,    axis=0)
    return Y_dd / np.sqrt(M * N)

def heisenberg(X_tf):
    """IFFT along rows (subcarrier axis) -> time-domain frame."""
    return ifft(X_tf, axis=0) * np.sqrt(M)

def wigner(frame):
    """FFT along rows (subcarrier axis) -> TF domain."""
    return fft(frame, axis=0) / np.sqrt(M)

def add_cp(frame, cp_len):
    return np.vstack([frame[-cp_len:, :], frame])   # (M+CP, N)

def remove_cp(rx_sig, cp_len, M_, N_):
    return rx_sig.reshape(M_ + cp_len, N_, order='F')[cp_len:, :]


# ══════════════════════════════════════════════════════════════
# CHANNEL MODEL
#
# For each tap (h_i, l_i delay samples, k_i Doppler bins):
#   The Doppler phase ramp is applied over the OFDM symbol index
#   (column index), and the delay shifts within-symbol samples (rows).
#
#   frame_out[:, col] += h_i * exp(j2pi*k_i*col/M) * frame_in[:,col] delayed by l_i rows
#
# This produces the DD I/O relation:
#   Y_DD[k_out, l_out] = sum_i  h_i * X_DD[(k_out + l_i)%M, (l_out + k_i)%N]
# ══════════════════════════════════════════════════════════════
def apply_channel(tx_sig, taps, snr_db):
    sym_len  = M + CP_LEN
    sig_len  = len(tx_sig)
    rx_sig   = np.zeros(sig_len, dtype=complex)
    frame_cp = tx_sig.reshape(sym_len, N, order='F')   # (M+CP, N)

    for tap in taps:
        h   = tap["gain"]
        li  = int(tap["delay"])
        ki  = int(tap["doppler"])

        # Doppler phase over symbol index (column)
        doppler_phase = np.exp(1j * 2 * np.pi * ki * np.arange(N) / M)  # (N,)

        # Delay: circular row-shift, zero causal wrap
        fd        = np.roll(frame_cp, li, axis=0)
        fd[:li, :] = 0

        rx_sig += h * (fd * doppler_phase[np.newaxis, :]).flatten(order='F')

    # AWGN
    sig_pow   = np.mean(np.abs(rx_sig) ** 2)
    snr_lin   = 10 ** (snr_db / 10)
    noise_std = np.sqrt(sig_pow / snr_lin / 2)
    noise     = noise_std * (np.random.randn(sig_len) + 1j * np.random.randn(sig_len))
    return rx_sig + noise


# ══════════════════════════════════════════════════════════════
# CHANNEL MATRIX  (corrected)
#
# Empirically verified formula:
#   Y_DD[k_out, l_out] = sum_i  h_i * X_DD[(k_out + l_i)%M, (l_out + k_i)%N]
#
# In row-major vector form (p = k*N + l):
#   H[p_out, p_in] += h_i   where
#     k_in = (k_out + l_i) % M    <- delay  shifts Doppler axis
#     l_in = (l_out + k_i) % N    <- Doppler shifts Delay   axis
# ══════════════════════════════════════════════════════════════
def build_channel_matrix(taps, M_, N_):
    size  = M_ * N_
    H_mat = np.zeros((size, size), dtype=complex)
    for k_out in range(M_):
        for l_out in range(N_):
            p = k_out * N_ + l_out
            for tap in taps:
                h   = tap["gain"]
                li  = int(tap["delay"])   % M_   # row shift
                ki  = int(tap["doppler"]) % N_   # col shift
                k_in = (k_out + li) % M_
                l_in = (l_out + ki) % N_
                q    = k_in * N_ + l_in
                H_mat[p, q] += h
    return H_mat


def mmse_equalize(Y_dd, H_matrix, snr_db):
    """MMSE equaliser:  X_hat = (H^H H + sigma2 I)^{-1} H^H y"""
    sigma2    = 10 ** (-snr_db / 10)
    y_vec     = Y_dd.flatten()
    HH        = H_matrix.conj().T @ H_matrix
    x_hat_vec = np.linalg.solve(HH + sigma2 * np.eye(HH.shape[0]),
                                 H_matrix.conj().T @ y_vec)
    return x_hat_vec.reshape(Y_dd.shape)


# ══════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════
num_symbols = M * N
num_bits    = num_symbols * 2
tx_bits     = np.random.randint(0, 2, num_bits)
print(f"\n[1]  Bits generated      : {num_bits}")

tx_qpsk = qpsk_modulate(tx_bits)
print(f"[2]  QPSK symbols        : {len(tx_qpsk)}")

X_DD = tx_qpsk.reshape(M, N)
print(f"[3]  DD grid X[k,l]      : {X_DD.shape}")

X_TF = isfft(X_DD)
rt_err = np.max(np.abs(sfft(X_TF) - X_DD))
print(f"[4]  ISFFT -> X[n,m]     : {X_TF.shape}  "
      f"roundtrip err={rt_err:.1e} {'PASS' if rt_err < 1e-10 else 'FAIL'}")

s_frame   = heisenberg(X_TF)
s_cp      = add_cp(s_frame, CP_LEN)
tx_signal = s_cp.flatten(order='F')
print(f"[5-6] Heisenberg + CP    : frame {s_cp.shape}, serial {len(tx_signal)}")

rx_signal = apply_channel(tx_signal, channel_taps, SNR_dB)
print(f"[7]  Channel applied     : {len(rx_signal)} samples")

Y_frame = remove_cp(rx_signal, CP_LEN, M, N)
Y_TF    = wigner(Y_frame)
Y_DD    = sfft(Y_TF)
print(f"[8-10] CP remove+Wigner+SFFT : Y_DD {Y_DD.shape}")

print(f"[11] Building {M*N}x{N*M} channel matrix ...")
H_mat = build_channel_matrix(channel_taps, M, N)

# Noiseless sanity check
rx_nl    = apply_channel(tx_signal, channel_taps, snr_db=1000)
Y_DD_nl  = sfft(wigner(remove_cp(rx_nl, CP_LEN, M, N)))
pred_mse = np.mean(np.abs((H_mat @ X_DD.flatten()) - Y_DD_nl.flatten()) ** 2)
print(f"     Prediction MSE (noiseless): {pred_mse:.2e}  "
      f"{'PASS' if pred_mse < 1e-20 else 'FAIL'}")

# ── Print H[k,l] — the DD-domain channel grid ────────────────
# H_dd[k,l] is the effective channel gain at each (Doppler, Delay) bin.
# Non-zero only at the tap positions; built from the channel matrix diagonal
# by constructing the M×N image: H_dd[k,l] = H_mat[k*N+l, k*N+l] is NOT
# what we want — instead extract the tap response directly.
H_dd_print = np.zeros((M, N), dtype=complex)
for tap in channel_taps:
    li = int(tap["delay"])   % M
    ki = int(tap["doppler"]) % N
    H_dd_print[li, ki] += tap["gain"]   # (delay row, Doppler col) for display

print()
print("=" * 55)
print("  H[k,l]  — DD-domain channel  (rows=Delay, cols=Doppler)")
print(f"  Non-zero taps: delay in rows [0..{M-1}], Doppler in cols [0..{N-1}]")
print("=" * 55)
print(f"  {'k (Doppler bin)':>20}", end="")
for k in range(N):
    print(f"  {k:6d}", end="")
print()
print("  " + "-" * (22 + N * 8))
for l in range(M):
    row = H_dd_print[l, :]
    if np.any(np.abs(row) > 1e-10):
        print(f"  l={l:2d} (delay)         ", end="")
        for val in row:
            if abs(val) > 1e-10:
                print(f"  {val.real:+.3f}", end="")
            else:
                print(f"  {'0':>6}", end="")
        print()
print("  " + "-" * (22 + N * 8))
print()
print("  Full H[k,l] magnitude grid  |H[k,l]|:")
print(f"  {'':6}", end="")
for k in range(N):
    print(f"  k={k:<3}", end="")
print()
for l in range(M):
    print(f"  l={l:<3} ", end="")
    for k in range(N):
        v = abs(H_dd_print[l, k])
        print(f"  {v:.3f}" if v > 1e-10 else f"  {'.':<5}", end="")
    print()
print("=" * 55)

X_eq    = mmse_equalize(Y_DD, H_mat, SNR_dB)
rx_bits = qpsk_demodulate(X_eq.flatten())
print(f"[12-13] MMSE + QPSK detect : {len(rx_bits)} bits")

errors = int(np.sum(tx_bits != rx_bits))
ber    = errors / num_bits
print()
print("=" * 55)
print("  RESULTS")
print("=" * 55)
print(f"  Total bits     : {num_bits}")
print(f"  Bit errors     : {errors}")
print(f"  BER            : {ber:.6f}")
print(f"  SNR            : {SNR_dB} dB")
print("=" * 55)


# ══════════════════════════════════════════════════════════════
# BER vs SNR SWEEP
# ══════════════════════════════════════════════════════════════
print("\n  BER vs SNR sweep ...")
snr_range   = np.arange(0, 31, 5)
ber_results = []
np.random.seed(0)

for snr in snr_range:
    b_tx   = np.random.randint(0, 2, num_bits)
    s_tx   = qpsk_modulate(b_tx)
    X_sw   = s_tx.reshape(M, N)
    tx_sw  = add_cp(heisenberg(isfft(X_sw)), CP_LEN).flatten(order='F')
    rx_sw  = apply_channel(tx_sw, channel_taps, snr)
    Y_sw   = sfft(wigner(remove_cp(rx_sw, CP_LEN, M, N)))
    X_eq_sw = mmse_equalize(Y_sw, H_mat, snr)
    b_rx   = qpsk_demodulate(X_eq_sw.flatten())
    bv     = np.sum(b_tx != b_rx) / num_bits
    ber_results.append(bv)
    print(f"    SNR={snr:2d} dB  BER={bv:.6f}  errors={int(np.sum(b_tx!=b_rx))}/{num_bits}")

ber_results = np.array(ber_results)


# ══════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.patch.set_facecolor('#0d1117')
for ax in axes.flat:
    ax.set_facecolor('#161b22')
    ax.tick_params(colors='#8b949e')
    ax.title.set_color('#e6edf3')
    for sp in ax.spines.values():
        sp.set_edgecolor('#30363d')

# 1. TX DD grid
im0 = axes[0,0].imshow(np.abs(X_DD), aspect='auto', cmap='plasma', extent=[0,N,0,M])
axes[0,0].set_title('TX  |X[k,l]|  Delay-Doppler', fontweight='bold')
axes[0,0].set_xlabel('Delay bin', color='#8b949e')
axes[0,0].set_ylabel('Doppler bin', color='#8b949e')
plt.colorbar(im0, ax=axes[0,0])

# 2. TX TF grid
im1 = axes[0,1].imshow(np.abs(X_TF), aspect='auto', cmap='viridis', extent=[0,N,0,M])
axes[0,1].set_title('TX  |X[n,m]|  Time-Frequency (ISFFT)', fontweight='bold')
axes[0,1].set_xlabel('Symbol index', color='#8b949e')
axes[0,1].set_ylabel('Subcarrier', color='#8b949e')
plt.colorbar(im1, ax=axes[0,1])

# 3. RX DD grid
im2 = axes[0,2].imshow(np.abs(Y_DD), aspect='auto', cmap='plasma', extent=[0,N,0,M])
axes[0,2].set_title('RX  |Y[k,l]|  Delay-Doppler (after channel)', fontweight='bold')
axes[0,2].set_xlabel('Delay bin', color='#8b949e')
axes[0,2].set_ylabel('Doppler bin', color='#8b949e')
plt.colorbar(im2, ax=axes[0,2])

# 4. Channel tap profile
ax_ch = axes[1,0]
for i, tap in enumerate(channel_taps):
    ax_ch.stem([tap["delay"]], [np.abs(tap["gain"])],
               linefmt=f'C{i}-', markerfmt=f'C{i}o', basefmt='#30363d',
               label=f"Tap {i+1}: Dop={tap['doppler']:+d}")
ax_ch.set_title('Channel Taps  Delay Profile', fontweight='bold')
ax_ch.set_xlabel('Delay bin', color='#8b949e')
ax_ch.set_ylabel('|gain|', color='#8b949e')
ax_ch.legend(fontsize=8, labelcolor='#e6edf3', facecolor='#21262d')

# 5. Constellation
ax_c = axes[1,1]
ax_c.scatter(Y_DD.real.flat, Y_DD.imag.flat,
             s=5, alpha=0.35, color='#f78166', label='RX DD (before EQ)')
ax_c.scatter(X_eq.real.flat, X_eq.imag.flat,
             s=5, alpha=0.6,  color='#3fb950', label='After MMSE EQ')
ax_c.scatter(_QPSK_CONST.real, _QPSK_CONST.imag,
             s=140, color='#ffa657', marker='*', zorder=6, label='QPSK ideal')
ax_c.set_title('Constellation Diagram', fontweight='bold')
ax_c.set_xlabel('In-phase', color='#8b949e')
ax_c.set_ylabel('Quadrature', color='#8b949e')
ax_c.legend(fontsize=8, labelcolor='#e6edf3', facecolor='#21262d')
ax_c.axhline(0, color='#30363d', lw=0.5)
ax_c.axvline(0, color='#30363d', lw=0.5)
ax_c.set_xlim(-2, 2); ax_c.set_ylim(-2, 2)

# 6. BER vs SNR
ax_b = axes[1,2]
ax_b.semilogy(snr_range, np.clip(ber_results, 1e-7, 1),
              'o-', color='#58a6ff', lw=2, ms=7, label='OTFS MMSE')
ax_b.set_title('BER vs SNR', fontweight='bold')
ax_b.set_xlabel('SNR (dB)', color='#8b949e')
ax_b.set_ylabel('BER', color='#8b949e')
ax_b.set_ylim(1e-6, 1.0)
ax_b.grid(True, which='both', ls='--', alpha=0.3, color='#8b949e')
ax_b.legend(fontsize=9, labelcolor='#e6edf3', facecolor='#21262d')
for snr, bv in zip(snr_range, ber_results):
    lbl = f'{bv:.4f}' if bv > 0 else '0.0000'
    ax_b.annotate(lbl, xy=(snr, max(bv, 1e-6)),
                  xytext=(3, 5), textcoords='offset points',
                  fontsize=7, color='#8b949e')

plt.suptitle(
    f'OTFS Full Pipeline (CORRECTED)  |  3-Tap Channel  |  '
    f'SNR={SNR_dB} dB  |  BER={ber:.6f}',
    color='#e6edf3', fontsize=12, fontweight='bold', y=1.01
)
plt.tight_layout()
plt.show()

print("\n[H_mat] Channel matrix information")
print("Shape:", H_mat.shape)

nonzero = np.count_nonzero(np.abs(H_mat) > 1e-12)
total = H_mat.size

print("Non-zero elements:", nonzero)
print("Sparsity:", 100*(1 - nonzero/total), "%")
print("\nTop-left 10x10 block of H_mat:")
print(np.round(H_mat[:10, :10], 3))
print("\nNon-zero entries of H_mat (row, col, value):")

rows, cols = np.where(np.abs(H_mat) > 1e-12)

for r, c in zip(rows, cols):
    print(f"H[{r:3d}, {c:3d}] = {H_mat[r,c].real:+.3f} {H_mat[r,c].imag:+.3f}j")
plt.figure(figsize=(6,6))
plt.spy(np.abs(H_mat) > 1e-12, markersize=2)
plt.title("Sparsity Pattern of OTFS Channel Matrix H")
plt.xlabel("Input index") 
plt.ylabel("Output index")
plt.show()
print("\nChannel Matrix H_mat:")
print("Shape:", H_mat.shape)

print("\nTop-left 10x10 block:")
print(np.round(H_mat[:10, :10],3))    
