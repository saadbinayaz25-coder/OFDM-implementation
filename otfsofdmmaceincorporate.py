import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as tFunc
import random
import os
import math
from scipy.signal import butter, filtfilt, upfirdn

# ============================================================
#  WAVEFORM SELECTOR
#  Set USE_OTFS = True  → OTFS mode  (delay-Doppler domain)
#  Set USE_OTFS = False → OFDM mode  (original pipeline)
# ============================================================
USE_OTFS = True   # <-- flip this flag to switch modes

# ============================================================
# OFDM / SYSTEM PARAMETERS
# ============================================================
Qm = 4                          # bits per QAM symbol (16-QAM)
F  = 102                        # subcarriers (incl. DC)
S  = 14                         # OFDM symbols per TTI
FFT_size    = 128
FFT_size_RX = 128
Fp  = 8                         # pilot subcarrier spacing
Sp  = 2                         # pilot symbol row index
CP  = 7                         # cyclic-prefix length
SCS = 15000                     # subcarrier spacing [Hz]
P   = F // Fp                   # number of pilot subcarriers
FFT_offset    = int((FFT_size    - F) / 2)
FFT_offset_RX = int((FFT_size_RX - F) / 2)

SampleRate    = FFT_size * SCS
Ts            = 1 / (SCS * FFT_size)
TTI_duration  = Ts * (FFT_size + CP) * S * 1000   # ms

SDR_TX_Frequency  = int(2_400_000_000)
SDR_TX_BANDWIDTH  = SCS * F * 4
tx_gain = -5
rx_gain = 30
TX_Scale = 0.7

leading_zeros      = 500
save_plots         = False
plot_width         = 8
titles             = False

# Channel sim parameters (used when use_sdr=False)
ch_SINR          = 20
n_taps           = 2
max_delay_spread = 3
velocity         = 30

use_sdr           = True   # set True to use real SDR hardware
randomize_tx_gain = True
tx_gain_lo        = -10
tx_gain_hi        = -10

SDR_TX_IP = 'ip:192.168.2.1'
SDR_RX_IP = 'ip:192.168.2.1'

# ============================================================
# OTFS PARAMETERS
#   N = number of Doppler bins  (maps to OFDM symbols S)
#   M = number of delay bins    (maps to subcarriers F, rounded)
# ============================================================
OTFS_N = S          # Doppler bins  → must equal S so frame lengths match
OTFS_M = F          # Delay bins    → must equal F so FFT sizes match
OTFS_active = F     # reuse the same F active subcarriers

# ============================================================
# MACE PARAMETERS
# ============================================================
USE_MACE            = True     # set False to fall back to single-tap MMSE
MACE_THRESHOLD      = 0.15    # fraction of pilot peak to accept a tap
MACE_MAX_PATHS      = 10      # max iterations of the cancellation loop


# ============================================================
# UTILITY / DATASET
# ============================================================
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self):
        self.pdsch_iq, self.labels, self.sinr = [], [], []

    def __len__(self):
        return len(self.pdsch_iq)

    def __getitem__(self, idx):
        return self.pdsch_iq[idx], self.labels[idx], self.sinr[idx]

    def add_item(self, iq, label, sinr_val):
        self.pdsch_iq.append(iq)
        self.labels.append(label)
        self.sinr.append(sinr_val)


# ============================================================
# SDR CLASS
# ============================================================
class SDR:
    def __init__(self, SDR_TX_IP='ip:192.168.2.1', SDR_RX_IP='ip:192.168.2.1',
                 SDR_TX_FREQ=2_400_000_000, SDR_TX_GAIN=-80, SDR_RX_GAIN=0,
                 SDR_TX_SAMPLERATE=1e6, SDR_TX_BANDWIDTH=1e6):
        self.SDR_TX_IP        = SDR_TX_IP
        self.SDR_RX_IP        = SDR_RX_IP
        self.SDR_TX_FREQ      = int(SDR_TX_FREQ)
        self.SDR_RX_FREQ      = int(SDR_TX_FREQ)
        self.SDR_TX_GAIN      = int(SDR_TX_GAIN)
        self.SDR_RX_GAIN      = int(SDR_RX_GAIN)
        self.SDR_TX_SAMPLERATE = int(SDR_TX_SAMPLERATE)
        self.SDR_TX_BANDWIDTH  = int(SDR_TX_BANDWIDTH)
        self.num_samples = 0
        self.sdr_tx = self.sdr_rx = None

    def SDR_TX_start(self):
        import adi
        self.sdr_tx = adi.ad9361(self.SDR_TX_IP)
        self.sdr_tx.tx_destroy_buffer()
        self.sdr_tx.tx_lo                  = self.SDR_TX_FREQ
        self.sdr_tx.sample_rate            = self.SDR_TX_SAMPLERATE
        self.sdr_tx.tx_rf_bandwidth        = self.SDR_TX_BANDWIDTH
        self.sdr_tx.tx_hardwaregain_chan0   = self.SDR_TX_GAIN
        self.sdr_tx.tx_enabled_channels    = ["voltage0"]

    def SDR_RX_start(self):
        import adi
        if self.sdr_tx is None:
            raise RuntimeError("TX must be started before RX for full-duplex.")
        self.sdr_rx = adi.Pluto(self.SDR_RX_IP)
        self.sdr_rx.rx_destroy_buffer()
        self.sdr_rx.rx_lo                       = self.SDR_RX_FREQ
        self.sdr_rx.sample_rate                 = self.SDR_TX_SAMPLERATE
        self.sdr_rx.rx_rf_bandwidth             = self.SDR_TX_BANDWIDTH
        self.sdr_rx.gain_control_mode_chan0      = "manual"
        self.sdr_rx.rx_hardwaregain_chan0        = self.SDR_RX_GAIN
        self.sdr_rx.rx_enabled_channels         = ["voltage0"]

    def SDR_gain_set(self, tx_gain, rx_gain):
        if self.sdr_tx: self.sdr_tx.tx_hardwaregain_chan0 = tx_gain
        if self.sdr_rx: self.sdr_rx.rx_hardwaregain_chan0 = rx_gain

    def SDR_TX_send(self, SAMPLES, max_scale=1, cyclic=False):
        self.sdr_tx.tx_destroy_buffer()
        if isinstance(SAMPLES, np.ndarray):
            self.num_samples = SAMPLES.size
        elif isinstance(SAMPLES, torch.Tensor):
            self.num_samples = SAMPLES.numel()
            SAMPLES = SAMPLES.numpy()
        samples  = SAMPLES - np.mean(SAMPLES)
        samples  = (samples / np.max(np.abs(samples))) * max_scale
        samples  = self.lowpass_filter(samples)
        samples *= 2**14
        self.sdr_tx.tx_cyclic_buffer = cyclic
        self.sdr_tx.tx(samples.astype(np.complex64))

    def SDR_TX_stop(self):
        if self.sdr_tx:
            self.sdr_tx.tx_destroy_buffer()
            self.sdr_tx.rx_destroy_buffer()

    def lowpass_filter(self, data):
        nyq = 0.5 * self.SDR_TX_SAMPLERATE
        normal_cutoff = (110 * 15000 * 0.5) / nyq
        b, a = butter(5, normal_cutoff, btype='low', analog=False)
        return np.array(filtfilt(b, a, data))

    def SDR_RX_receive(self, n_SAMPLES=None, normalize=True):
        if n_SAMPLES is None:
            n_SAMPLES = self.num_samples * 4
        if n_SAMPLES <= 0:
            n_SAMPLES = 1
        self.sdr_rx.rx_destroy_buffer()
        self.sdr_rx.rx_buffer_size = n_SAMPLES
        rx_data = self.sdr_rx.rx()
        rx_data = self.lowpass_filter(rx_data)
        if normalize:
            rx_data = rx_data / np.max(np.abs(rx_data))
        return torch.tensor(rx_data, dtype=torch.complex64)


# ============================================================
# MODULATION MAPPING
# ============================================================
def mapping_table(Qm, plot=False):
    size = int(torch.sqrt(torch.tensor(2**Qm)))
    a    = torch.arange(size, dtype=torch.float32)
    b    = a - torch.mean(a)
    C    = (b.unsqueeze(1) + 1j * b).flatten()
    C   /= torch.sqrt(torch.mean(torch.abs(C)**2))

    def idx2bin(i, q):
        return tuple(map(int, '{:0{}b}'.format(int(i), q)))

    mapping   = {idx2bin(i, Qm): v for i, v in enumerate(C)}
    demapping = {v: k for k, v in mapping.items()}

    if plot:
        plt.figure(figsize=(4, 4))
        plt.scatter(C.numpy().real, C.numpy().imag)
        if titles: plt.title(f'Constellation - {Qm} bps')
        plt.ylabel('Imaginary'); plt.xlabel('Real'); plt.tight_layout()
        if save_plots: plt.savefig('pics/const.png', bbox_inches='tight')
    return mapping, demapping

mapping_table_QPSK, de_mapping_table_QPSK = mapping_table(2, plot=True)
mapping_table_Qm,   de_mapping_table_Qm   = mapping_table(Qm, plot=True)


# ============================================================
# OFDM BLOCK MASK
# ============================================================
def OFDM_block_mask(S, F, Fp, Sp, FFT_offset, plotOFDM_block=False):
    OFDM_mask = torch.ones((S, F), dtype=torch.int8)
    OFDM_mask[Sp, torch.arange(0, F, Fp)] = 2
    OFDM_mask[Sp, 0]     = 2
    OFDM_mask[Sp, F - 1] = 2
    OFDM_mask[:, F // 2] = 3   # DC
    OFDM_mask = torch.cat((
        torch.zeros(S, FFT_offset, dtype=torch.int8),
        OFDM_mask,
        torch.zeros(S, FFT_offset, dtype=torch.int8)
    ), dim=1)
    if plotOFDM_block:
        plt.figure(figsize=(plot_width, 1.5))
        plt.imshow(OFDM_mask.numpy(), aspect='auto')
        if titles: plt.title('OFDM_block mask')
        plt.xlabel('Subcarrier index'); plt.ylabel('Symbol')
        if save_plots: plt.savefig('pics/OFDM_blockmask.png', bbox_inches='tight')
        plt.tight_layout(); plt.show()
    return OFDM_mask

OFDM_mask = OFDM_block_mask(S=S, F=F, Fp=Fp, Sp=Sp,
                             FFT_offset=FFT_offset, plotOFDM_block=True)
print("OFDM_mask rows:", len(OFDM_mask))
payload_elements_in_mask = OFDM_mask.eq(1).sum().item() * Qm
print("Max payload bits:", payload_elements_in_mask)


# ============================================================
# PILOT SET
# ============================================================
def pilot_set(OFDM_mask, power_scaling=1.0):
    pilot_values = torch.tensor([-0.7-0.7j, -0.7+0.7j, 0.7-0.7j, 0.7+0.7j]) * power_scaling
    num_pilots   = OFDM_mask[OFDM_mask == 2].numel()
    print("num_pilots:", num_pilots)
    return pilot_values.repeat(num_pilots // 4 + 1)[:num_pilots]

pilot_symbols = pilot_set(OFDM_mask, 1)
print(pilot_symbols)


# ============================================================
# CONVOLUTIONAL CODEC
# ============================================================
def conv_encode(bits):
    bits = bits.int()
    g0 = torch.tensor([1, 1, 1]); g1 = torch.tensor([1, 0, 1])
    state   = torch.zeros(2, dtype=torch.int32)
    encoded = []
    for b in bits:
        shift = torch.cat([b.view(1), state])
        encoded.extend([torch.sum(shift * g0) % 2,
                         torch.sum(shift * g1) % 2])
        state = shift[:-1]
    return torch.tensor(encoded, dtype=torch.int32)


def viterbi_decode(bits):
    bits     = bits.int()
    n_states = 4
    INF      = 1e9
    next_state = {0:[(0,0),(1,2)], 1:[(2,0),(3,2)],
                  2:[(0,1),(1,3)], 3:[(2,1),(3,3)]}
    out_bits   = {(0,0):(0,0),(0,1):(1,1),(1,0):(1,0),(1,1):(0,1),
                  (2,0):(1,1),(2,1):(0,0),(3,0):(0,1),(3,1):(1,0)}
    path_metric = torch.full((n_states,), INF)
    path_metric[0] = 0
    paths = [[] for _ in range(n_states)]
    for i in range(0, len(bits), 2):
        rx         = bits[i:i+2]
        new_metric = torch.full((n_states,), INF)
        new_paths  = [[] for _ in range(n_states)]
        for s in range(n_states):
            for inp in [0, 1]:
                ns       = next_state[s][inp][0]
                expected = torch.tensor(out_bits[(s, inp)])
                dist     = torch.sum(rx != expected)
                metric   = path_metric[s] + dist
                if metric < new_metric[ns]:
                    new_metric[ns] = metric
                    new_paths[ns]  = paths[s] + [inp]
        path_metric = new_metric; paths = new_paths
    return torch.tensor(paths[torch.argmin(path_metric)], dtype=torch.int32)


# ============================================================
# PAYLOAD CREATION
# ============================================================
def create_payload(OFDM_mask, Qm, mapping_table, power=1.0, filename=None):
    payload_REs      = OFDM_mask.eq(1).sum().item()
    max_encoded_bits = payload_REs * Qm
    max_uncoded_bits = int(max_encoded_bits * 0.5)

    if filename is not None:
        file_bits = np.fromfile(filename, dtype=np.int8)
        if not np.all((file_bits == 0) | (file_bits == 1)):
            raise ValueError("Input file must contain only 0/1 bits")
        if len(file_bits) < max_uncoded_bits:
            file_bits = np.tile(file_bits, int(np.ceil(max_uncoded_bits / len(file_bits))))
        payload_bits = torch.tensor(file_bits[:max_uncoded_bits], dtype=torch.int32)
    else:
        payload_bits = torch.randint(0, 2, (max_uncoded_bits,), dtype=torch.int32)

    payload_bits_encoded = conv_encode(payload_bits)
    assert payload_bits_encoded.numel() <= max_encoded_bits, "Encoded bits exceed OFDM capacity"

    flattened_bits   = payload_bits_encoded.view(-1, Qm)
    payload_symbols  = torch.tensor(
        [mapping_table[tuple(b.tolist())] for b in flattened_bits],
        dtype=torch.complex64) * power

    return payload_bits, payload_bits_encoded, payload_symbols


# ============================================================
# RE MAPPING / IFFT / CP
# ============================================================
def RE_mapping(OFDM_mask, pilot_set, payload_symbols, plotOFDM_block=False):
    IQ = torch.zeros(OFDM_mask.shape, dtype=torch.complex64)
    IQ[OFDM_mask == 1] = payload_symbols.clone().detach()
    IQ[OFDM_mask == 2] = pilot_set.clone().detach()
    if plotOFDM_block:
        plt.figure(figsize=(plot_width, 1.5))
        plt.imshow(torch.abs(IQ).numpy(), aspect='auto')
        if titles: plt.title('OFDM_block modulated symbols')
        plt.xlabel('Subcarrier index'); plt.ylabel('Symbol')
        if save_plots: plt.savefig('pics/OFDM_blockmod.png', bbox_inches='tight')
        plt.show()
    return IQ

def IFFT(IQ):
    return torch.fft.ifft(torch.fft.ifftshift(IQ, dim=1))

def CP_addition(IQ, S, FFT_size, CP):
    out = torch.zeros((S, FFT_size + CP), dtype=torch.complex64)
    for s in range(S):
        out[s, :] = torch.cat((IQ[s, -CP:], IQ[s, :]), dim=0)
    return out.flatten()


# ============================================================
#  OTFS MODULATION LAYER — with integrated MACE estimator
#
#  Architecture overview
#  ─────────────────────
#  TX:  bits → QAM → DD grid X[k,l]
#         → embed_pilot()       (impulse pilot at (kp, lp))
#         → ISFFT()             (DD → TF, DC-centred)
#         → OFDM IFFT + CP
#
#  RX:  OFDM CP removal + DFT (existing chain)
#         → SFFT()             (TF → DD)
#         → mace_estimate()    (NEW) iterative tap finder
#         → build_H_from_pilot() (NEW) full MN×MN channel matrix
#         → equalize_MMSE_fullH() (NEW) proper MMSE using H matrix
#           OR equalize_DD()    (legacy single-tap, USE_MACE=False)
#         → extract data symbols
#
#  MACE (Matching-pursuit Assisted Channel Estimation)
#  ────────────────────────────────────────────────────
#  Ref: IISc OTFS channel estimation papers (Raviteja et al.)
#
#  The pilot DD response Y_pilot[k,l] is a superposition of
#  2D Dirichlet kernels, one per physical channel tap:
#
#    Y_pilot[k,l] ≈ Σ_i h_i · D_M(k − k_p + ν_i) · D_N(l − l_p − τ_i)
#
#  where D_M/D_N are M/N-point Dirichlet kernels and ν_i/τ_i are
#  fractional Doppler/delay in bins.
#
#  MACE iterates:
#    1. Find peak of |Y_pilot|  → integer bin (k̂, l̂)
#    2. Refine to sub-bin (k̂+εk, l̂+εl) via parabolic interpolation
#    3. Estimate complex gain: g = Y[k̂,l̂] / (D_M(εk) · D_N(εl))
#    4. Subtract reconstructed kernel from residual
#    5. Repeat until residual < threshold
#
#  Build H matrix (2D block-circulant reconstruction)
#  ────────────────────────────────────────────────────
#  One pilot transmission gives column p_pilot of H:
#    H_mat[:, p_pilot] = Y_pilot.flatten() / pilot_val
#
#  Since H is 2D block-circulant (OTFS DD convolution property):
#    H[k_out,l_out; k_in,l_in]
#      = Y_pilot[(k_out−k_in+PILOT_K) % N,
#                (l_out−l_in+PILOT_L) % M]
#
#  This gives the exact H for integer taps (noiseless MSE ~1e-30)
#  and a high-quality approximation for fractional taps.
#  The full H then feeds a standard MMSE equaliser:
#    x̂ = (H†H + σ²I)^{-1} H† y
# ============================================================

class OTFSModulator:
    """
    OTFS modulator/demodulator with integrated MACE channel estimator.

    Parameters
    ----------
    N  : int  – Doppler bins (= OFDM symbols S)
    M  : int  – Delay bins   (= active subcarriers F)
    pilot_guard_delay   : int – guard cells in delay   (≥ max delay spread)
    pilot_guard_doppler : int – guard cells in Doppler (≥ max Doppler spread)
    mace_threshold : float – fraction of peak below which MACE stops
    mace_max_paths : int   – max cancellation iterations
    use_mace : bool – True → full MMSE via MACE; False → single-tap fallback
    """

    def __init__(self, N=14, M=102,
                 pilot_guard_delay=4, pilot_guard_doppler=3,
                 mace_threshold=0.15, mace_max_paths=10,
                 use_mace=True):
        self.N  = N
        self.M  = M
        self.pgd  = pilot_guard_delay
        self.pgk  = pilot_guard_doppler

        # MACE configuration
        self.use_mace        = use_mace
        self.mace_threshold  = mace_threshold
        self.mace_max_paths  = mace_max_paths

        # Pilot location: centre of Doppler axis, quarter-point in delay
        self.kp = N // 2    # Doppler index
        self.lp = M // 4    # Delay index

    # ──────────────────────────────────────────────────────────
    # TRANSFORMS
    # ──────────────────────────────────────────────────────────
    def ISFFT(self, X_dd):
        """DD → TF  (DC-centred on freq axis)."""
        S_tf = torch.fft.ifft(X_dd, dim=0)
        S_tf = torch.fft.ifft(S_tf, dim=1)
        S_tf = torch.fft.fftshift(S_tf, dim=1)
        return S_tf

    def SFFT(self, S_tf):
        """TF → DD  (inverse of ISFFT)."""
        S_tf = torch.fft.ifftshift(S_tf, dim=1)
        Y_dd = torch.fft.fft(S_tf, dim=0)
        Y_dd = torch.fft.fft(Y_dd, dim=1)
        return Y_dd

    # ──────────────────────────────────────────────────────────
    # PILOT EMBEDDING
    # ──────────────────────────────────────────────────────────
    def embed_pilot(self, X_dd, pilot_power=1.0):
        """Place impulse pilot at (kp, lp) surrounded by a guard zone."""
        X_dd = X_dd.clone()
        pilot_mask = torch.zeros_like(X_dd, dtype=torch.bool)

        for k in range(self.kp - self.pgk - 1, self.kp + self.pgk + 2):
            for l in range(self.lp - self.pgd - 1, self.lp + self.pgd + 2):
                X_dd[k % self.N, l % self.M] = 0.0 + 0.0j

        X_dd[self.kp, self.lp] = pilot_power + 0.0j
        pilot_mask[self.kp, self.lp] = True
        return X_dd, pilot_mask

    # ──────────────────────────────────────────────────────────
    # DIRICHLET KERNEL  (MACE building block)
    # ──────────────────────────────────────────────────────────
    @staticmethod
    def _dirichlet(x, Npts):
        """
        N-point Dirichlet kernel evaluated at fractional bin offset x.

        D_N(x) = sin(π x) / (N sin(π x / N))
               = sinc(x) / sinc(x / N)    [np.sinc uses normalised sinc]

        Handles x=0 analytically (returns 1.0) and avoids division by zero
        via np.sinc (which equals sin(πx)/(πx), already 1 at x=0).
        """
        x = np.asarray(x, dtype=complex)
        return np.sinc(x) / np.sinc(x / Npts)

    # ──────────────────────────────────────────────────────────
    # MACE  (iterative path extraction from pilot DD response)
    # ──────────────────────────────────────────────────────────
    def mace_estimate(self, Y_dd_pilot, pilot_power=1.0, plot=False):
        """
        MACE channel estimator.

        Iteratively detects and cancels fractional delay/Doppler paths
        from the received pilot DD response.  Each iteration:
          1. Finds the peak of the residual |R[k,l]|.
          2. Refines the peak to sub-bin accuracy via parabolic interpolation.
          3. Estimates the complex tap gain.
          4. Subtracts the 2D Dirichlet kernel from the residual.

        Parameters
        ----------
        Y_dd_pilot : torch.Tensor (N × M) — received DD grid from pilot frame
        pilot_power : float — known pilot amplitude
        plot : bool — visualise pilot response and estimated taps

        Returns
        -------
        est_taps : list of dict
            [{"gain": complex, "delay": float, "doppler": float}, ...]
            delay/doppler are offsets IN BINS relative to (kp, lp).
        """
        N, M   = self.N, self.M
        kp, lp = self.kp, self.lp

        # Measure the actual pilot amplitude from the peak bin
        # (accounts for SFFT gain of N×M and all SDR scaling)
        pilot_amp_rx = float(torch.abs(Y_dd_pilot[kp, lp]).item())
        if pilot_amp_rx < 1e-10:
            pilot_amp_rx = max(float(torch.abs(Y_dd_pilot).max().item()), 1e-10)
            print(f"[MACE] WARNING: pilot bin near zero — using max as reference")

        # Normalise residual by measured amplitude so gains are O(1)
        R = (Y_dd_pilot / pilot_amp_rx).numpy().astype(complex)

        # ── Noise floor (robust: use median of |R|) ──────────────
        noise_floor  = float(np.median(np.abs(R)))
        peak_global  = float(np.max(np.abs(R)))
        threshold    = max(self.mace_threshold * peak_global, 3.0 * noise_floor)

        print(f"\n[MACE] N={N} M={M}  pilot at k={kp},l={lp}")
        print(f"[MACE] noise_floor={noise_floor:.4f}  "
              f"peak={peak_global:.4f}  threshold={threshold:.4f}")

        est_taps   = []
        iter_count = 0

        while True:
            idx  = int(np.argmax(np.abs(R)))
            peak = float(np.abs(R.flat[idx]))

            # ── Stopping criterion ───────────────────────────────
            if peak < threshold or iter_count >= self.mace_max_paths:
                break

            k_pk = idx // M
            l_pk = idx % M

            # ── Sub-bin parabolic refinement (Doppler, k axis) ──
            R0   = float(np.abs(R[k_pk, l_pk]))
            Rp_k = float(np.abs(R[(k_pk + 1) % N, l_pk]))
            Rm_k = float(np.abs(R[(k_pk - 1) % N, l_pk]))
            denom_k = R0 - 0.5 * (Rp_k + Rm_k)
            eps_k   = 0.0 if abs(denom_k) < 1e-8 else float(np.clip(
                0.5 * (Rp_k - Rm_k) / (denom_k + 1e-12), -0.5, 0.5))

            # ── Sub-bin parabolic refinement (Delay, l axis) ────
            Rp_l = float(np.abs(R[k_pk, (l_pk + 1) % M]))
            Rm_l = float(np.abs(R[k_pk, (l_pk - 1) % M]))
            denom_l = R0 - 0.5 * (Rp_l + Rm_l)
            eps_l   = 0.0 if abs(denom_l) < 1e-8 else float(np.clip(
                0.5 * (Rp_l - Rm_l) / (denom_l + 1e-12), -0.5, 0.5))

            # Fractional bin positions
            true_k = k_pk + eps_k
            true_l = l_pk + eps_l

            # ── Convert from absolute bin to offset from pilot ──
            # Doppler offset (cycles through N)
            dop_offset = ((kp - true_k + N // 2) % N) - N // 2
            # Delay offset (cycles through M)
            del_offset = ((lp - true_l + M // 2) % M) - M // 2

            # ── Estimate complex gain ─────────────────────────────
            denom_kern = (self._dirichlet(eps_k, N)
                          * self._dirichlet(eps_l, M))
            if abs(denom_kern) < 1e-8:
                gain = complex(R[k_pk, l_pk])
            else:
                gain = complex(R[k_pk, l_pk]) / denom_kern

            est_taps.append({
                "gain"    : gain,
                "delay"   : float(del_offset),
                "doppler" : float(dop_offset),
            })

            # ── Subtract reconstructed Dirichlet kernel ───────────
            k_arr = np.arange(N)
            l_arr = np.arange(M)
            # Doppler kernel: centred on true_k (= k_pk + eps_k)
            dk = self._dirichlet(kp - dop_offset - k_arr, N)
            # Delay kernel: centred on true_l (= l_pk + eps_l)
            dl = self._dirichlet(lp - del_offset - l_arr, M)
            # Outer product gives the 2D Dirichlet surface to subtract
            R -= gain * np.outer(dk, dl)

            iter_count += 1

        print(f"[MACE] {len(est_taps)} path(s) detected in {iter_count} iteration(s)")
        for i, t in enumerate(est_taps):
            print(f"  Path {i+1}: |gain|={abs(t['gain']):.4f}  "
                  f"delay={t['delay']:+.3f}  doppler={t['doppler']:+.3f}")

        if plot:
            fig, axes = plt.subplots(1, 2, figsize=(12, 3))
            mag = np.abs(Y_dd_pilot.numpy())
            axes[0].imshow(mag, aspect='auto', origin='lower', cmap='hot')
            axes[0].set_title('Pilot DD Response |Y_pilot|')
            axes[0].set_xlabel('Delay bin'); axes[0].set_ylabel('Doppler bin')
            for t in est_taps:
                kk = (kp - t['doppler']) % N
                ll = (lp - t['delay'])   % M
                axes[0].plot(ll, kk, 'c+', ms=12, mew=2)

            axes[1].imshow(np.abs(R), aspect='auto', origin='lower', cmap='viridis')
            axes[1].set_title('MACE Residual |R|')
            axes[1].set_xlabel('Delay bin'); axes[1].set_ylabel('Doppler bin')
            plt.tight_layout(); plt.show()

        return est_taps

    # ──────────────────────────────────────────────────────────
    # BUILD H MATRIX FROM PILOT  (2D block-circulant)
    # ──────────────────────────────────────────────────────────
    def build_H_from_pilot(self, Y_dd, pilot_power=1.0):
        """
        Reconstruct the full (N·M × N·M) channel matrix H from the
        received DD grid using the 2D block-circulant property of OTFS.

        The SFFT (two unnormalised FFTs) amplifies the pilot value by N×M,
        so the effective TX pilot amplitude in Y_dd is pilot_power × N × M.
        Rather than relying on the nominal pilot_power (which is further
        scaled by the SDR TX/RX gain chain and RX normalisation), we
        directly measure the effective amplitude from Y_dd[kp, lp] and use
        that to normalise H.

        To avoid contaminating H with data-symbol energy, we zero all bins
        outside the guard window before building the circulant.

        H[k_out,l_out; k_in,l_in]
            = Y_pilot_clean[(k_out−k_in+kp) % N,
                            (l_out−l_in+lp) % M]  / pilot_amp_measured

        Returns
        -------
        H_mat        : torch.Tensor (N*M, N*M), complex64
        pilot_amp_rx : float — measured pilot amplitude (for noise scaling)
        """
        N, M   = self.N, self.M
        kp, lp = self.kp, self.lp
        pg_k   = self.pgk
        pg_l   = self.pgd
        size   = N * M

        # Step 1: isolate the pilot guard window; zero all data bins
        pilot_window = np.zeros((N, M), dtype=complex)
        for k in range(kp - pg_k, kp + pg_k + 1):
            for l in range(lp - pg_l, lp + pg_l + 1):
                kk = k % N;  ll = l % M
                pilot_window[kk, ll] = Y_dd[kk, ll].item()

        # Step 2: measure the effective pilot amplitude directly from Y_dd
        pilot_amp_rx = float(np.abs(pilot_window[kp, lp]))
        if pilot_amp_rx < 1e-10:
            pilot_amp_rx = pilot_power * N * M
            print(f"[OTFS H] WARNING: pilot bin near zero, using nominal "
                  f"pilot_amp={pilot_amp_rx:.2f}")
        else:
            print(f"[OTFS H] measured pilot_amp_rx={pilot_amp_rx:.4f}  "
                  f"(nominal pilot_power×N×M={pilot_power * N * M:.1f})")

        # Step 3: normalise pilot window → channel gains ≈ O(1)
        pilot_2d = pilot_window / pilot_amp_rx

        # Step 4: fill H via 2D block-circulant shift
        H_mat = np.zeros((size, size), dtype=complex)
        k_out = np.arange(N)
        l_out = np.arange(M)
        for k_in in range(N):
            for l_in in range(M):
                p_in  = k_in * M + l_in
                k_idx = (k_out - k_in + kp) % N
                l_idx = (l_out - l_in + lp) % M
                H_mat[:, p_in] = pilot_2d[
                    k_idx[:, None], l_idx[None, :]
                ].flatten()

        return torch.tensor(H_mat, dtype=torch.complex64), pilot_amp_rx

    # ──────────────────────────────────────────────────────────
    # MMSE EQUALISER USING FULL H MATRIX
    # ──────────────────────────────────────────────────────────
    def equalize_MMSE_fullH(self, Y_dd, H_mat, noise_power=1e-4,
                             pilot_amp_rx=1.0):
        """
        Full MMSE equaliser:  x̂ = (H†H + σ²I)⁻¹ H† y

        H is normalised so that channel gains are O(1).
        Y_dd must be divided by pilot_amp_rx to match the same scale.
        noise_power (measured in the SFFT domain) must be divided by
        pilot_amp_rx² to match the normalised domain.

        Parameters
        ----------
        Y_dd        : torch.Tensor (N, M)
        H_mat       : torch.Tensor (N*M, N*M) from build_H_from_pilot
        noise_power : float — σ² in the raw SFFT-output domain
        pilot_amp_rx: float — measured pilot amplitude from build_H_from_pilot

        Returns
        -------
        X_eq : torch.Tensor (N, M)  symbols in the normalised TX-DD scale
        """
        N, M   = self.N, self.M
        H_np   = H_mat.numpy().astype(complex)

        # Normalise received vector to match H scale
        y_vec  = (Y_dd.numpy().astype(complex) / pilot_amp_rx).flatten()

        # Scale noise variance to the normalised domain
        sigma2 = max(noise_power / (pilot_amp_rx ** 2), 1e-12)

        HH    = H_np.conj().T @ H_np
        rhs   = H_np.conj().T @ y_vec
        x_hat = np.linalg.solve(HH + sigma2 * np.eye(HH.shape[0]), rhs)

        print(f"[OTFS MMSE] full-H  pilot_amp_rx={pilot_amp_rx:.2f}  "
              f"σ²_raw={noise_power:.2e}  σ²_scaled={sigma2:.2e}  "
              f"|x̂| range [{np.abs(x_hat).min():.3f}, {np.abs(x_hat).max():.3f}]")

        return torch.tensor(x_hat.reshape(N, M), dtype=torch.complex64)

    # ──────────────────────────────────────────────────────────
    # LEGACY: DD-domain LS channel estimation  (single-tap path)
    # ──────────────────────────────────────────────────────────
    def channel_estimate_DD(self, Y_dd, pilot_power=1.0, plot=False):
        """
        Legacy LS estimator used when USE_MACE=False.
        (Unchanged from original — single dominant tap.)
        """
        kp, lp = self.kp, self.lp
        pg_k, pg_l = self.pgk, self.pgd
        N, M   = self.N, self.M

        H_dd     = torch.zeros(N, M, dtype=torch.complex64)
        tap_list = []

        noise_ref = []
        for k in range(N):
            for l in range(M):
                if abs(k - kp) > pg_k + 2 and abs(l - lp) > pg_l + 2:
                    noise_ref.append(torch.abs(Y_dd[k, l]).item())

        if not noise_ref:
            for k in range(N):
                for l in range(M):
                    outside_k = abs(k - kp) > pg_k
                    outside_l = abs(l - lp) > pg_l
                    if outside_k or outside_l:
                        noise_ref.append(torch.abs(Y_dd[k % N, l % M]).item())

        noise_floor = float(np.percentile(noise_ref, 75)) if noise_ref else 1e-6
        detect_thr  = 3.0 * noise_floor

        window_bins = []
        for k in range(kp - pg_k, kp + pg_k + 1):
            for l in range(lp - pg_l, lp + pg_l + 1):
                kk = k % N;  ll = l % M
                h_val = Y_dd[kk, ll] / pilot_power
                H_dd[kk, ll] = h_val
                window_bins.append((k - kp, l - lp, kk, ll, h_val))

        if window_bins:
            best = max(window_bins, key=lambda x: torch.abs(x[4]).item())
            dk_best, dl_best, kk_best, ll_best, h_best = best
            peak_mag = torch.abs(h_best).item()
            sidelobe_thr = 0.10 * peak_mag
            for dk, dl, kk, ll, hv in window_bins:
                if torch.abs(hv).item() > sidelobe_thr:
                    tap_list.append((dk, dl, hv))

        if len(tap_list) == 0 and window_bins:
            tap_list.append((dk_best, dl_best, h_best))

        if plot:
            plt.figure(figsize=(8, 3))
            plt.imshow(torch.abs(H_dd).numpy(), aspect='auto',
                       origin='lower', cmap='viridis')
            plt.colorbar(label='|H_dd|')
            plt.xlabel('Delay bin'); plt.ylabel('Doppler bin')
            plt.title('OTFS DD-domain Channel Estimate (legacy)')
            plt.tight_layout(); plt.show()

        return H_dd, tap_list

    # ──────────────────────────────────────────────────────────
    # LEGACY: single-tap MMSE equaliser
    # ──────────────────────────────────────────────────────────
    def equalize_DD(self, Y_dd, H_dd, noise_power=1e-4):
        """Legacy single-tap MMSE equaliser (USE_MACE=False path)."""
        N, M   = self.N, self.M
        kp, lp = self.kp, self.lp
        pg_l   = self.pgd

        peak_idx  = torch.argmax(torch.abs(H_dd))
        pk, pl    = divmod(peak_idx.item(), M)
        h_dom     = H_dd[pk, pl]
        h_pow     = (torch.abs(h_dom) ** 2).item()

        if h_pow < 1e-20:
            print("[OTFS EQ] Warning: dominant tap ≈ 0, returning Y_dd unchanged")
            return Y_dd.clone()

        # Per-Doppler-row delay-domain MMSE
        h_taps_full = torch.zeros(M, dtype=torch.complex64)
        for dl_off in range(-pg_l, pg_l + 1):
            src_ll  = (pl + dl_off) % M
            dest_ll = dl_off % M
            h_taps_full[dest_ll] = H_dd[pk, src_ll]

        H_dft = torch.fft.fft(h_taps_full)
        H_pow = torch.abs(H_dft) ** 2
        W_dft = torch.conj(H_dft) / (H_pow + noise_power)

        Y_dft_2d = torch.fft.fft(Y_dd, dim=1)
        X_dft_eq = Y_dft_2d * W_dft.unsqueeze(0)
        X_dd_eq  = torch.fft.ifft(X_dft_eq, dim=1)

        print(f"[OTFS EQ] single-tap MMSE  peak=({pk},{pl})  "
              f"|h|={h_pow**0.5:.4f}  σ²={noise_power:.2e}")
        return X_dd_eq

    # ──────────────────────────────────────────────────────────
    # MODULATE: bits → DD grid → TF grid
    # ──────────────────────────────────────────────────────────
    def modulate(self, payload_symbols, pilot_power=1.0, plot=False):
        """
        Map QAM payload_symbols into the DD grid, embed pilot,
        apply ISFFT to get TF grid (N × M) ready for OFDM IFFT.
        """
        N, M   = self.N, self.M
        kp, lp = self.kp, self.lp
        pg_k, pg_l = self.pgk, self.pgd

        data_mask = torch.ones(N, M, dtype=torch.bool)
        for k in range(kp - pg_k - 1, kp + pg_k + 2):
            for l in range(lp - pg_l - 1, lp + pg_l + 2):
                data_mask[k % N, l % M] = False

        n_data = data_mask.sum().item()
        assert payload_symbols.numel() <= n_data, \
            f"Too many payload symbols: {payload_symbols.numel()} > {n_data}"

        X_dd = torch.zeros(N, M, dtype=torch.complex64)
        flat_idx  = data_mask.flatten().nonzero(as_tuple=False).squeeze()
        sym_pad   = torch.zeros(n_data, dtype=torch.complex64)
        sym_pad[:payload_symbols.numel()] = payload_symbols
        X_dd.flatten()[flat_idx] = sym_pad

        X_dd, _ = self.embed_pilot(X_dd, pilot_power)
        S_tf = self.ISFFT(X_dd)

        if plot:
            fig, axes = plt.subplots(1, 2, figsize=(12, 3))
            axes[0].imshow(torch.abs(X_dd).numpy(), aspect='auto',
                           origin='lower', cmap='plasma')
            axes[0].set_title('TX DD Grid |X_dd|')
            axes[0].set_xlabel('Delay'); axes[0].set_ylabel('Doppler')
            axes[1].imshow(torch.abs(S_tf).numpy(), aspect='auto',
                           origin='lower', cmap='plasma')
            axes[1].set_title('TX TF Grid |S_tf|')
            axes[1].set_xlabel('Subcarrier'); axes[1].set_ylabel('Symbol')
            plt.tight_layout(); plt.show()

        return S_tf, X_dd, data_mask

    # ──────────────────────────────────────────────────────────
    # DEMODULATE: TF grid → equalised symbols
    # ──────────────────────────────────────────────────────────
    def demodulate(self, S_tf_rx, noise_power=1e-4, pilot_power=1.0,
                   plot=False):
        """
        Recover equalised QAM symbols from the received TF grid.

        When self.use_mace=True (default):
          • SFFT → MACE estimation → build full H matrix → MMSE equalize
        When self.use_mace=False:
          • SFFT → legacy LS estimate → per-row delay MMSE equalize

        Parameters
        ----------
        S_tf_rx    : torch.Tensor (N, M) — active subcarriers from DFT output
        noise_power: float — linear σ² (normalised)
        pilot_power: float — known pilot amplitude
        plot       : bool

        Returns
        -------
        payload_syms_eq : torch.Tensor (1-D)
        Y_dd            : torch.Tensor (N, M)
        H_dd or H_mat   : channel estimate tensor
        """
        N, M    = self.N, self.M
        kp, lp  = self.kp, self.lp
        pg_k, pg_l = self.pgk, self.pgd

        # ── Step 1: SFFT → delay-Doppler domain ──────────────
        Y_dd = self.SFFT(S_tf_rx)

        # ── Step 2: Integer CFO detection and correction ──────
        best_val   = 0.0
        k_peak, l_peak = kp, lp
        for k in range(kp - pg_k, kp + pg_k + 1):
            for l in range(lp - pg_l, lp + pg_l + 1):
                kk, ll = k % N, l % M
                v = torch.abs(Y_dd[kk, ll]).item()
                if v > best_val:
                    best_val = v
                    k_peak, l_peak = kk, ll

        delta_k = k_peak - kp
        if delta_k != 0:
            print(f"[OTFS CFO] Integer Doppler offset Δk={delta_k:+d} "
                  f"(≈{delta_k * 15000 / N:.0f} Hz) — correcting")
            n_vec   = torch.arange(N, dtype=torch.float32).unsqueeze(1)
            phase   = -2 * np.pi * delta_k * n_vec / N
            corr    = torch.exp(1j * phase.to(torch.complex64))
            S_tf_corrected = S_tf_rx * corr
            Y_dd    = self.SFFT(S_tf_corrected)
            # Re-find peak after correction
            best_val = 0.0
            for k in range(kp - pg_k, kp + pg_k + 1):
                for l in range(lp - pg_l, lp + pg_l + 1):
                    kk, ll = k % N, l % M
                    v = torch.abs(Y_dd[kk, ll]).item()
                    if v > best_val:
                        best_val = v
                        k_peak, l_peak = kk, ll
            print(f"[OTFS CFO] After correction: pilot at k={k_peak}, l={l_peak} "
                  f"|Y|={best_val:.4f}")

        # ── Data mask (same as TX) ────────────────────────────
        data_mask = torch.ones(N, M, dtype=torch.bool)
        for k in range(kp - pg_k - 1, kp + pg_k + 2):
            for l in range(lp - pg_l - 1, lp + pg_l + 2):
                data_mask[k % N, l % M] = False

        # ══════════════════════════════════════════════════════
        #  MACE PATH  (use_mace=True)
        # ══════════════════════════════════════════════════════
        if self.use_mace:
            # Step 3a: Extract pilot-only window for MACE
            # Zero out data bins so MACE doesn't mistake data energy for taps
            Y_dd_pilot_only = torch.zeros_like(Y_dd)
            for k in range(kp - pg_k - 1, kp + pg_k + 2):
                for l in range(lp - pg_l - 1, lp + pg_l + 2):
                    kk = k % N;  ll = l % M
                    Y_dd_pilot_only[kk, ll] = Y_dd[kk, ll]

            print("[OTFS RX] Using MACE channel estimation")
            est_taps = self.mace_estimate(
                Y_dd_pilot_only, pilot_power=pilot_power, plot=plot)

            # Step 4a: Build full block-circulant H matrix from pilot window
            # Returns (H_mat, pilot_amp_rx) — pilot_amp_rx is measured from Y_dd
            H_mat, pilot_amp_rx = self.build_H_from_pilot(
                Y_dd_pilot_only, pilot_power=pilot_power)
            print(f"[OTFS RX] H_mat built: {tuple(H_mat.shape)}  "
                  f"mean|H|={torch.abs(H_mat).mean().item():.4f}  "
                  f"pilot_amp_rx={pilot_amp_rx:.2f}")

            # Step 5a: Full MMSE equalisation (y and σ² both normalised by pilot_amp_rx)
            X_dd_eq = self.equalize_MMSE_fullH(
                Y_dd, H_mat,
                noise_power=noise_power,
                pilot_amp_rx=pilot_amp_rx)

            # Step 6a: Extract data symbols
            payload_syms_eq = X_dd_eq[data_mask]

            if plot:
                fig, axes = plt.subplots(1, 3, figsize=(16, 3))

                axes[0].imshow(torch.abs(Y_dd).numpy(), aspect='auto',
                               origin='lower', cmap='plasma')
                axes[0].set_title('RX DD Grid |Y_dd|  (MACE)')
                axes[0].set_xlabel('Delay bin')
                axes[0].set_ylabel('Doppler bin')
                axes[0].axhline(k_peak, color='white', lw=0.5, ls='--')
                axes[0].axvline(l_peak, color='white', lw=0.5, ls='--')
                for t in est_taps:
                    kk = (kp - t['doppler']) % N
                    ll = (lp - t['delay'])   % M
                    axes[0].plot(ll, kk, 'c+', ms=12, mew=2)

                hm_block = H_mat[:min(64, N*M), :min(64, N*M)]
                axes[1].imshow(torch.abs(hm_block).numpy(), aspect='auto',
                               cmap='inferno', interpolation='nearest')
                axes[1].set_title('|H_mat| (top-left block)')
                axes[1].set_xlabel('Input DD bin')
                axes[1].set_ylabel('Output DD bin')

                syms_plot = payload_syms_eq.detach().cpu()
                axes[2].scatter(syms_plot.real.numpy(),
                                syms_plot.imag.numpy(), s=3, alpha=0.5)
                axes[2].set_title('MACE-MMSE equalised constellation')
                axes[2].set_xlabel('I'); axes[2].set_ylabel('Q')
                axes[2].axis('equal')
                axes[2].set_xlim([-2, 2]); axes[2].set_ylim([-2, 2])
                axes[2].grid(True, ls='--', alpha=0.4)
                plt.tight_layout(); plt.show()

            return payload_syms_eq, Y_dd, H_mat

        # ══════════════════════════════════════════════════════
        #  LEGACY SINGLE-TAP PATH  (use_mace=False)
        # ══════════════════════════════════════════════════════
        else:
            print("[OTFS RX] Using legacy single-tap MMSE estimation")

            # Step 3b: LS channel estimate in DD domain
            H_dd, taps = self.channel_estimate_DD(
                Y_dd, pilot_power=pilot_power, plot=plot)

            # Step 4b: Per-row delay MMSE
            X_dd_eq = self.equalize_DD(Y_dd, H_dd,
                                        noise_power=noise_power)

            # Step 5b: Extract data symbols
            payload_syms_eq = X_dd_eq[data_mask]

            if plot:
                fig, axes = plt.subplots(1, 2, figsize=(12, 3))
                axes[0].imshow(torch.abs(Y_dd).numpy(), aspect='auto',
                               origin='lower', cmap='plasma')
                axes[0].set_title('RX DD Grid |Y_dd|  (legacy)')
                axes[0].axhline(k_peak, color='white', lw=0.5, ls='--')
                axes[0].axvline(l_peak, color='white', lw=0.5, ls='--')
                syms_plot = payload_syms_eq.detach().cpu()
                axes[1].scatter(syms_plot.real.numpy(),
                                syms_plot.imag.numpy(), s=3, alpha=0.5)
                axes[1].set_title('Legacy equalised constellation')
                axes[1].axis('equal')
                axes[1].set_xlim([-2, 2]); axes[1].set_ylim([-2, 2])
                axes[1].grid(True, ls='--', alpha=0.4)
                plt.tight_layout(); plt.show()

            return payload_syms_eq, Y_dd, H_dd


# ── Instantiate OTFS block ──────────────────────────────────
OTFS_PGK = 3 if not use_sdr else 5   # Doppler guard half-width
otfs = OTFSModulator(
    N=OTFS_N, M=OTFS_M,
    pilot_guard_delay=4,
    pilot_guard_doppler=OTFS_PGK,
    mace_threshold=MACE_THRESHOLD,
    mace_max_paths=MACE_MAX_PATHS,
    use_mace=USE_MACE,
)


# ============================================================
# create_OFDM_data  –  UNIFIED TX FUNCTION
# ============================================================
def create_OFDM_data(filename=None):
    if not USE_OTFS:
        # ---- Original OFDM path ----
        pdsch_bits, pdsch_bits_encoded, pdsch_symbols = create_payload(
            OFDM_mask, Qm, mapping_table_Qm, power=1, filename=filename)

        Modulated_TTI = RE_mapping(OFDM_mask, pilot_symbols,
                                   pdsch_symbols, plotOFDM_block=True)
        TD_TTI_IQ  = IFFT(Modulated_TTI)
        TX_Samples = CP_addition(TD_TTI_IQ, S, FFT_size, CP)

        if use_sdr:
            zeros      = torch.zeros(leading_zeros, dtype=TX_Samples.dtype)
            TX_Samples = torch.cat((zeros, TX_Samples), dim=0)

        return pdsch_bits, pdsch_bits_encoded, pdsch_symbols, TX_Samples, None

    else:
        # ---- OTFS path ----
        n_data_DD = otfs.N * otfs.M
        for k in range(otfs.kp - otfs.pgk - 1, otfs.kp + otfs.pgk + 2):
            for l in range(otfs.lp - otfs.pgd - 1, otfs.lp + otfs.pgd + 2):
                n_data_DD -= 1

        max_encoded_bits = n_data_DD * Qm
        max_uncoded_bits = int(max_encoded_bits * 0.5)

        if filename is not None:
            file_bits = np.fromfile(filename, dtype=np.int8)
            if len(file_bits) < max_uncoded_bits:
                file_bits = np.tile(
                    file_bits,
                    int(np.ceil(max_uncoded_bits / len(file_bits))))
            pdsch_bits = torch.tensor(
                file_bits[:max_uncoded_bits], dtype=torch.int32)
        else:
            pdsch_bits = torch.randint(0, 2, (max_uncoded_bits,),
                                       dtype=torch.int32)

        pdsch_bits_encoded = conv_encode(pdsch_bits)

        fb = pdsch_bits_encoded.view(-1, Qm)
        pdsch_symbols = torch.tensor(
            [mapping_table_Qm[tuple(b.tolist())] for b in fb],
            dtype=torch.complex64)

        # Pilot power: 20 dB above expected data leakage floor
        OTFS_PILOT_POWER = 10.0

        S_tf, X_dd, data_mask = otfs.modulate(
            pdsch_symbols, pilot_power=OTFS_PILOT_POWER, plot=True)

        roundtrip_err = torch.max(torch.abs(otfs.SFFT(S_tf) - X_dd)).item()
        print(f"[OTFS TX] ISFFT/SFFT round-trip max error: {roundtrip_err:.6f}"
              f"  {'PASS' if roundtrip_err < 1e-4 else 'FAIL'}")

        S_tf_padded = torch.zeros(S, FFT_size, dtype=torch.complex64)
        S_tf_padded[:, FFT_offset: FFT_offset + F] = S_tf
        S_tf_padded[:, FFT_offset + F // 2] = 0.0 + 0.0j

        TD_TTI_IQ  = IFFT(S_tf_padded)
        TX_Samples = CP_addition(TD_TTI_IQ, S, FFT_size, CP)

        if use_sdr:
            zeros      = torch.zeros(leading_zeros, dtype=TX_Samples.dtype)
            TX_Samples = torch.cat((zeros, TX_Samples), dim=0)

        print(f"[OTFS TX] uncoded bits={pdsch_bits.numel()}, "
              f"encoded bits={pdsch_bits_encoded.numel()}, "
              f"DD data symbols={pdsch_symbols.numel()}, "
              f"pilot_power={OTFS_PILOT_POWER}, "
              f"use_mace={USE_MACE}")

        return pdsch_bits, pdsch_bits_encoded, pdsch_symbols, TX_Samples, OTFS_PILOT_POWER


# ---- Generate TX samples ----
pdsch_bits, pdsch_bits_encoded, pdsch_symbols, TX_Samples, otfs_pilot_power = \
    create_OFDM_data(filename=None)

print("Uncoded bits:", len(pdsch_bits))


# ============================================================
# SDR INIT
# ============================================================
if use_sdr:
    SDR_1 = SDR(SDR_RX_IP=SDR_RX_IP, SDR_TX_IP=SDR_TX_IP,
                SDR_TX_FREQ=SDR_TX_Frequency, SDR_TX_GAIN=tx_gain,
                SDR_RX_GAIN=rx_gain, SDR_TX_SAMPLERATE=SampleRate,
                SDR_TX_BANDWIDTH=SDR_TX_BANDWIDTH)
    SDR_1.SDR_TX_start()
    SDR_1.SDR_RX_start()


# ============================================================
# MULTIPATH CHANNEL
# ============================================================
def apply_multipath_channel_dop(iq, max_n_taps, max_delay, repeats=0,
                                 random_start=True, SINR=30,
                                 leading_zeros=500, fc=432e6,
                                 velocity=30, fs=1e6, randomize=False):
    c = 3e8
    if randomize:
        velocity = torch.rand(1).item() * (velocity - 1) + 1
    f_D    = (velocity / c) * fc
    n_taps = torch.randint(1, max_n_taps + 1, (1,)).item()
    tap_indices = torch.randint(0, max_delay, (n_taps,))
    h = torch.zeros(max_delay, dtype=torch.complex64)
    t = torch.arange(len(iq)) / fs
    fading_signal = torch.zeros_like(iq, dtype=torch.complex64)

    for i, delay in enumerate(tap_indices):
        power      = torch.rand(1).item() / ((i + 1) * 10)
        f_D_local  = torch.rand(1).item() * f_D
        N_sin      = 16
        n          = torch.arange(1, N_sin + 1)
        theta_n    = 2 * math.pi * n / (N_sin + 1)
        phase_n    = 2 * math.pi * torch.rand(N_sin)
        jakes_real = torch.zeros(len(iq))
        jakes_imag = torch.zeros(len(iq))
        for k in range(N_sin):
            jakes_real += torch.cos(
                2 * math.pi * f_D_local * torch.cos(theta_n[k]) * t + phase_n[k])
            jakes_imag += torch.sin(
                2 * math.pi * f_D_local * torch.cos(theta_n[k]) * t + phase_n[k])
        fading = power * (jakes_real + 1j * jakes_imag) / math.sqrt(N_sin)
        delayed_iq    = torch.nn.functional.pad(iq, (delay, 0))[:len(iq)]
        fading_signal += fading * delayed_iq
        h[delay]       = fading[0]

    fading_signal = torch.cat([
        torch.zeros(leading_zeros, dtype=fading_signal.dtype),
        fading_signal])

    if SINR != 0:
        sig_pow    = torch.mean(torch.abs(fading_signal)**2)
        noise_pow  = sig_pow / (10**(SINR / 10))
        noise      = torch.sqrt(noise_pow / 2) * (
            torch.randn_like(fading_signal) + 1j * torch.randn_like(fading_signal))
        fading_signal += noise

    if random_start:
        start = torch.randint(0, len(fading_signal), (1,)).item()
        fading_signal = torch.roll(fading_signal, shifts=start)
    if repeats > 0:
        fading_signal = fading_signal.repeat(repeats)

    return fading_signal, h


# ============================================================
# RADIO CHANNEL
# ============================================================
def radio_channel(use_sdr, tx_signal, tx_gain, rx_gain, ch_SINR):
    if use_sdr:
        if randomize_tx_gain:
            tx_gain = random.randint(tx_gain_lo, tx_gain_hi)
        SDR_1.SDR_gain_set(tx_gain, rx_gain)
        print(f"TX Gain: {tx_gain}, RX Gain: {rx_gain}")
        SDR_1.SDR_TX_send(SAMPLES=tx_signal, max_scale=TX_Scale, cyclic=True)
        rx_signal = SDR_1.SDR_RX_receive(len(tx_signal) * 4)
        SDR_1.SDR_TX_stop()
    else:
        rx_signal, h = apply_multipath_channel_dop(
            tx_signal, max_n_taps=n_taps, max_delay=max_delay_spread,
            random_start=True, repeats=3, SINR=ch_SINR,
            leading_zeros=leading_zeros, fc=SDR_TX_Frequency,
            velocity=velocity, fs=SampleRate, randomize=False)
        print("Channel taps h:", h)
    return rx_signal

RX_Samples = radio_channel(use_sdr=use_sdr, tx_signal=TX_Samples,
                           tx_gain=tx_gain, rx_gain=rx_gain,
                           ch_SINR=ch_SINR)


# ============================================================
# PSD
# ============================================================
def PSD_plot(signal, Fs, f, info='TX'):
    if isinstance(signal, torch.Tensor):
        signal = signal.numpy()
    plt.figure(figsize=(8, 3))
    plt.psd(signal, Fs=Fs, NFFT=1024, Fc=f, color='blue')
    plt.grid(True); plt.ylim(-120,)
    plt.xlabel('Frequency [Hz]'); plt.ylabel('PSD [dB/Hz]')
    if titles: plt.title(f'Power Spectral Density, {info}')
    if save_plots: plt.savefig(f'pics/PSD_{info}.png', bbox_inches='tight')
    plt.show()

PSD_plot(TX_Samples, SampleRate, SDR_TX_Frequency, 'TX')
PSD_plot(RX_Samples, SampleRate, SDR_TX_Frequency, 'RX')


# ============================================================
# SYNCHRONISATION
# ============================================================
def sync_iq(tx_signal, rx_signal, leading_zeros, threshold=6, plot=False):
    tx_len   = tx_signal.numel()
    rx_len   = rx_signal.numel()
    end_pt   = rx_len - tx_len
    if end_pt <= leading_zeros:
        rx_short = rx_signal
    else:
        rx_short = rx_signal[leading_zeros:end_pt]

    corr_r = tFunc.conv1d(rx_short.real.view(1,1,-1),
                          tx_signal.real.view(1,1,-1)).view(-1)
    corr_i = tFunc.conv1d(rx_short.imag.view(1,1,-1),
                          tx_signal.imag.view(1,1,-1)).view(-1)
    correlation = torch.complex(corr_r, corr_i).abs()

    thr        = correlation.mean() * threshold
    i_maxarg   = torch.argmax(correlation).item() + leading_zeros

    i = 0
    for i, value in enumerate(correlation):
        if value > thr:
            break

    if plot:
        plot_i   = i_maxarg - leading_zeros
        c_len    = correlation.numel()
        win_lo   = max(plot_i - 10, 0)
        win_hi   = min(plot_i + 50, c_len)
        offset_lo = win_lo - plot_i
        offset_hi = win_hi - plot_i
        corr_v  = correlation[win_lo:win_hi]
        idx_off = range(offset_lo, offset_hi)
        disp_v  = [float(v) if float(v) > float(thr) else 0.0 for v in corr_v]

        if len(idx_off) > 0 and len(disp_v) > 0:
            plt.figure(figsize=(8, 3))
            plt.bar(list(idx_off), disp_v); plt.grid()
            plt.xlabel("Samples from start index")
            plt.ylabel("Complex conjugate correlation")
            plt.gca().get_yaxis().set_visible(False)
            if save_plots: plt.savefig('pics/corr.png', bbox_inches='tight')
            plt.show()

    return i + leading_zeros, i_maxarg

symbol_index, symbol_index_maxarg = sync_iq(
    TX_Samples, RX_Samples, leading_zeros=leading_zeros,
    threshold=0, plot=True)

if use_sdr:
    symbol_index_maxarg = symbol_index_maxarg + leading_zeros


# ============================================================
# SINR ESTIMATE
# ============================================================
def SINR(rx_signal, index, leading_zeros):
    rx_noise  = rx_signal[index - leading_zeros + 20 : index - 20]
    noise_pwr = torch.mean(torch.abs(rx_noise)**2)
    print("rx noise_power", noise_pwr)
    rx_sig    = rx_signal[index : index + (14 * 72)]
    sig_pwr   = torch.mean(torch.abs(rx_sig)**2)
    print("rx_signal_power", sig_pwr)
    sinr_val  = round((10 * torch.log10(sig_pwr / noise_pwr)).item(), 1)
    return sinr_val, noise_pwr, sig_pwr


# ============================================================
# CP REMOVAL
# ============================================================
def CP_removal(rx_signal, OFDM_block_start, S, FFT_size, CP, plotsig=False):
    b_payload = torch.zeros(len(rx_signal), dtype=torch.bool)
    for s in range(S):
        start = OFDM_block_start + (s + 1) * CP + s * FFT_size
        b_payload[start : start + FFT_size] = 1

    if plotsig:
        rx_np   = rx_signal.cpu().numpy()
        rx_norm = rx_np / np.max(np.abs(rx_np))
        plt.figure(figsize=(plot_width, 3))
        plt.plot(rx_norm, label='Received Signal')
        plt.plot(b_payload.cpu().numpy(), label='Payload Mask')
        plt.xlabel('Sample index'); plt.ylabel('Amplitude')
        if titles: plt.title('Received signal and payload mask')
        plt.legend()
        if save_plots: plt.savefig('pics/RXsignal_sync.png', bbox_inches='tight')
        plt.show()

    return rx_signal[b_payload].view(S, FFT_size)


# ============================================================
# DFT
# ============================================================
def DFT(rxsignal, plotDFT=False):
    OFDM_RX_DFT = torch.fft.fftshift(torch.fft.fft(rxsignal, dim=1), dim=1)
    if plotDFT:
        plt.figure(figsize=(plot_width, 1.5))
        plt.imshow(torch.abs(OFDM_RX_DFT).numpy(), aspect='auto')
        plt.xlabel('Subcarrier Index'); plt.ylabel('Symbol')
        if save_plots: plt.savefig('pics/OFDM_block_RX.png', bbox_inches='tight')
        plt.show()
    return OFDM_RX_DFT


# ============================================================
# CHANNEL ESTIMATION (OFDM path)
# ============================================================
def channelEstimate_LS(OFDM_mask_RE, pilot_symbols, F, FFT_offset,
                        Sp, IQ_post_DFT, plotEst=False):
    def piecewise_linear_interp(x, xp, fp):
        si  = torch.argsort(xp)
        xp  = xp[si].float().to(x.device)
        fp  = fp[si].float().to(x.device)
        out = torch.zeros_like(x, device=x.device).float()
        for i in range(len(xp) - 1):
            mask = (x >= xp[i]) & (x <= xp[i+1])
            out[mask] = fp[i] + (fp[i+1] - fp[i]) * (x[mask] - xp[i]) / (xp[i+1] - xp[i])
        return out

    pilots               = IQ_post_DFT[OFDM_mask_RE == 2]
    H_at_pilots          = pilots / pilot_symbols
    pilot_indices        = torch.nonzero(OFDM_mask_RE[Sp] == 2,
                                          as_tuple=False).squeeze()
    all_indices          = torch.arange(FFT_offset, FFT_offset + F)
    H_real = piecewise_linear_interp(all_indices, pilot_indices,
                                      H_at_pilots.real)
    H_imag = piecewise_linear_interp(all_indices, pilot_indices,
                                      H_at_pilots.imag)
    H_estim = torch.view_as_complex(torch.stack([H_real, H_imag], dim=-1))

    if plotEst:
        dB   = lambda x: 10 * torch.log10(torch.abs(x))
        ph   = lambda x: torch.angle(x)
        plt.figure(figsize=(8, 3))
        plt.plot(pilot_indices.numpy(), dB(H_at_pilots).numpy(),
                 'ro-', label='Pilot estimates', markersize=8)
        plt.plot(all_indices.numpy(), dB(H_estim).numpy(),
                 'b-', label='Estimated channel')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xlabel('Subcarrier Index'); plt.ylabel('Magnitude (dB)')
        plt.legend(); plt.tight_layout(); plt.show()

    return H_estim


# ============================================================
# HELPER FUNCTIONS
# ============================================================
def remove_fft_Offests(IQ, F, FFT_offset):
    return IQ[:, torch.arange(FFT_offset, F + FFT_offset)]

def extract_middle_subcarriers(input_tensor, num_subcarriers):
    mid   = input_tensor.shape[-1] // 2
    start = mid - num_subcarriers // 2
    return input_tensor[..., start: start + num_subcarriers]

def phase_unwrap(phase):
    diff     = np.diff(phase)
    jumps    = np.abs(diff) > 1
    cum      = np.cumsum(diff)
    pu       = phase.copy()
    pu[1:]  -= np.where(jumps, cum, 0)
    return pu

def equalize_ZF(IQ_post_DFT, H_estim, F, S, plotQAM=False):
    IQ_np   = IQ_post_DFT.cpu().numpy()
    H_np    = H_estim.cpu().numpy()
    ph_uw   = phase_unwrap(np.angle(H_np))
    eq_np   = IQ_np / np.abs(H_np) * np.exp(-1j * ph_uw)
    equalized = torch.tensor(eq_np, dtype=torch.complex64).view(S, F)

    if plotQAM:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
        for i in range(IQ_post_DFT.shape[0]):
            ax1.scatter(IQ_post_DFT[i].cpu().real.numpy(),
                        IQ_post_DFT[i].cpu().imag.numpy(),
                        color='blue', s=5)
        ax1.axis('equal'); ax1.set_xlabel('Real'); ax1.set_ylabel('Imag')
        ax1.set_title('Pre-equalization'); ax1.grid(True, linestyle='--', alpha=0.7)
        for i in range(equalized.shape[0]):
            ax2.scatter(equalized[i].cpu().real.numpy(),
                        equalized[i].cpu().imag.numpy(),
                        color='blue', s=5)
        ax2.axis('equal'); ax2.set_xlim([-1.5,1.5]); ax2.set_ylim([-1.5,1.5])
        ax2.set_xlabel('Real'); ax2.set_ylabel('Imag')
        ax2.set_title('Post-Equalization'); ax2.grid(True, linestyle='--', alpha=0.7)
        if save_plots:
            plt.tight_layout()
            plt.savefig('pics/RXdSymbols_side_by_side.png', bbox_inches='tight')
        plt.show()
    return equalized

def get_payload_symbols(OFDM_mask_RE, equalized, FFT_offset, F):
    mask = OFDM_mask_RE[:, FFT_offset: FFT_offset + F] == 1
    return equalized[mask]


# ============================================================
# COMMON RX CHAIN  (CP removal → DFT → mask trim)
# ============================================================
OFDM_mask_rx = extract_middle_subcarriers(OFDM_mask, FFT_size_RX)

RX_NO_CP = CP_removal(RX_Samples, symbol_index_maxarg,
                       S, FFT_size, CP, plotsig=True)
rx_norm_scale = torch.max(torch.abs(RX_NO_CP)).item()
RX_NO_CP = RX_NO_CP / rx_norm_scale

IQ_DFT = DFT(RX_NO_CP, plotDFT=True)
IQ_DFT = extract_middle_subcarriers(IQ_DFT, FFT_size_RX)

SINR_m, noise_power, signal_power = SINR(
    RX_Samples, symbol_index_maxarg, leading_zeros)


# ============================================================
# RX SIGNAL PROCESSING  – MODE-DEPENDENT
# ============================================================
if not USE_OTFS:
    # ---- Original OFDM RX path ----
    H_estim = channelEstimate_LS(
        OFDM_mask_rx, pilot_symbols, F, FFT_offset_RX, Sp,
        IQ_DFT, plotEst=True)

    OFDM_demod_no_offsets = remove_fft_Offests(IQ_DFT, F, FFT_offset_RX)
    equalized_H_estim     = equalize_ZF(OFDM_demod_no_offsets, H_estim,
                                         F, S, plotQAM=True)
    QAM_est = get_payload_symbols(OFDM_mask_rx, equalized_H_estim,
                                   FFT_offset_RX, F)

else:
    # ---- OTFS RX path (MACE or legacy) ----
    active_start = FFT_offset_RX
    active_end   = FFT_offset_RX + F
    S_tf_rx = IQ_DFT[:, active_start: active_end]   # (S, F) DC-centred

    print(f"[OTFS RX] S_tf_rx shape={tuple(S_tf_rx.shape)}  "
          f"mean_power={torch.mean(torch.abs(S_tf_rx)**2).item():.4f}")
    print(f"[OTFS RX] Mode: {'MACE + full-H MMSE' if USE_MACE else 'legacy single-tap'}")

    # Quick sanity: pilot peak location
    Y_dd_dbg = otfs.SFFT(S_tf_rx)
    peak_idx  = torch.argmax(torch.abs(Y_dd_dbg))
    pk, pl    = divmod(peak_idx.item(), otfs.M)
    print(f"[OTFS RX] Y_dd peak at k={pk}, l={pl}  "
          f"|Y|={torch.abs(Y_dd_dbg[pk, pl]).item():.4f}  "
          f"(expected pilot at k={otfs.kp}, l={otfs.lp})")

    # Normalise noise power to the scaled domain
    noise_pwr_lin = float(noise_power.item()) \
        if isinstance(noise_power, torch.Tensor) else float(noise_power)
    noise_pwr_lin = noise_pwr_lin / (rx_norm_scale ** 2)
    noise_pwr_lin = max(noise_pwr_lin, 1e-10)
    print(f"[OTFS RX] rx_norm_scale={rx_norm_scale:.4f}  "
          f"noise_pwr_normalised={noise_pwr_lin:.2e}")

    QAM_est_otfs, Y_dd, H_ch = otfs.demodulate(
        S_tf_rx, noise_power=noise_pwr_lin,
        pilot_power=float(otfs_pilot_power), plot=True)

    n_tx_syms = pdsch_symbols.numel()
    QAM_est   = QAM_est_otfs[:n_tx_syms]

    print(f"[OTFS RX] recovered {QAM_est.numel()} symbols "
          f"(transmitted {n_tx_syms})")


# ============================================================
# DEMAPPING
# ============================================================
def Demapping(QAM, de_mapping_table):
    constellation = torch.tensor(list(de_mapping_table.keys()),
                                  device=QAM.device)
    dists         = torch.abs(QAM.view(-1,1) - constellation.view(1,-1))
    const_index   = torch.argmin(dists, dim=1)
    hardDecision  = constellation[const_index]
    str_table     = {str(k.item()): v for k, v in de_mapping_table.items()}
    demapped      = torch.tensor(
        [str_table[str(c.item())] for c in hardDecision],
        dtype=torch.int32, device=QAM.device)
    return demapped, hardDecision

def PS(bits):
    return bits.reshape(-1)

PS_est, hardDecision = Demapping(QAM_est, de_mapping_table_Qm)
bits_est     = PS(PS_est)
decoded_bits = viterbi_decode(bits_est)

print("Decoded bits length :", decoded_bits.numel())
print("Original bits length:", pdsch_bits.numel())


# ============================================================
# BER
# ============================================================
min_len     = min(decoded_bits.numel(), pdsch_bits.numel())
error_count = torch.sum(decoded_bits[:min_len] != pdsch_bits[:min_len]).float()
BER         = torch.round(error_count / min_len * 1000) / 1000

mode_str  = "OTFS" if USE_OTFS else "OFDM"
mace_str  = "+MACE" if (USE_OTFS and USE_MACE) else ""
print(f"[{mode_str}{mace_str}] BER: {BER:.3f}   SINR: {SINR_m:.1f} dB")


# ============================================================
# SIGNAL RECONSTRUCTION + WAVEFORM PLOT
# ============================================================
def bits_to_16qam_symbols(bits):
    bits      = np.array(bits.cpu()).astype(int)
    bits      = bits[:len(bits) - (len(bits) % 4)]
    bit_groups = bits.reshape((-1, 4))
    lut = {
        (0,0,0,0):-3-3j,(0,0,0,1):-3-1j,(0,0,1,1):-3+1j,(0,0,1,0):-3+3j,
        (0,1,1,0):-1+3j,(0,1,1,1):-1+1j,(0,1,0,1):-1-1j,(0,1,0,0):-1-3j,
        (1,1,0,0): 1-3j,(1,1,0,1): 1-1j,(1,1,1,1): 1+1j,(1,1,1,0): 1+3j,
        (1,0,1,0): 3+3j,(1,0,1,1): 3+1j,(1,0,0,1): 3-1j,(1,0,0,0): 3-3j,
    }
    return np.array([lut[tuple(b)] for b in bit_groups]) / np.sqrt(10)

symbols  = bits_to_16qam_symbols(decoded_bits)
waveform = upfirdn([1], symbols, 8)

plt.figure(figsize=(10, 4))
plt.plot(np.real(waveform), label="I")
plt.plot(np.imag(waveform), label="Q")
plt.title(f"Reconstructed Time-Domain Signal [{mode_str}{mace_str}]")
plt.grid(True); plt.legend(); plt.show()


# ============================================================
# FINAL SINR + CHANNEL RE-ESTIMATE
# ============================================================
SINR_m, noise_power, signal_power = SINR(
    RX_Samples, symbol_index_maxarg, leading_zeros)

if not USE_OTFS:
    H_estim = channelEstimate_LS(
        OFDM_mask_rx, pilot_symbols, F, FFT_offset_RX, Sp,
        IQ_DFT, plotEst=True)