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
# ============================================================
USE_OTFS = True

# ============================================================
# OFDM / SYSTEM PARAMETERS
# ============================================================
Qm = 4
F  = 102
S  = 14
FFT_size    = 128
FFT_size_RX = 128
Fp  = 8
Sp  = 2
CP  = 7
SCS = 15000
P   = F // Fp
FFT_offset    = int((FFT_size    - F) / 2)
FFT_offset_RX = int((FFT_size_RX - F) / 2)

SampleRate    = FFT_size * SCS
Ts            = 1 / (SCS * FFT_size)
TTI_duration  = Ts * (FFT_size + CP) * S * 1000

SDR_TX_Frequency  = int(2_400_000_000)
SDR_TX_BANDWIDTH  = SCS * F * 4
tx_gain = -5
rx_gain = 30
TX_Scale = 0.7

leading_zeros      = 500
save_plots         = False
plot_width         = 8
titles             = False

ch_SINR          = 20
n_taps           = 2
max_delay_spread = 3
velocity         = 30

use_sdr           = False
randomize_tx_gain = True
tx_gain_lo        = -10
tx_gain_hi        = -10

SDR_TX_IP = 'ip:192.168.2.1'
SDR_RX_IP = 'ip:192.168.2.1'

OTFS_N = S
OTFS_M = F
OTFS_active = F

# ============================================================
# MACE PARAMETERS
# ============================================================
USE_MACE       = True
# Threshold > first Dirichlet sidelobe for N=14 (~0.22).
MACE_THRESHOLD = 0.30
# Conservative cap for small N.
MACE_MAX_PATHS = 4


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
        self.SDR_TX_IP         = SDR_TX_IP
        self.SDR_RX_IP         = SDR_RX_IP
        self.SDR_TX_FREQ       = int(SDR_TX_FREQ)
        self.SDR_RX_FREQ       = int(SDR_TX_FREQ)
        self.SDR_TX_GAIN       = int(SDR_TX_GAIN)
        self.SDR_RX_GAIN       = int(SDR_RX_GAIN)
        self.SDR_TX_SAMPLERATE = int(SDR_TX_SAMPLERATE)
        self.SDR_TX_BANDWIDTH  = int(SDR_TX_BANDWIDTH)
        self.num_samples = 0
        self.sdr_tx = self.sdr_rx = None

    def SDR_TX_start(self):
        import adi
        self.sdr_tx = adi.ad9361(self.SDR_TX_IP)
        self.sdr_tx.tx_destroy_buffer()
        self.sdr_tx.tx_lo                = self.SDR_TX_FREQ
        self.sdr_tx.sample_rate          = self.SDR_TX_SAMPLERATE
        self.sdr_tx.tx_rf_bandwidth      = self.SDR_TX_BANDWIDTH
        self.sdr_tx.tx_hardwaregain_chan0 = self.SDR_TX_GAIN
        self.sdr_tx.tx_enabled_channels  = ["voltage0"]

    def SDR_RX_start(self):
        import adi
        if self.sdr_tx is None:
            raise RuntimeError("TX must be started before RX for full-duplex.")
        self.sdr_rx = adi.Pluto(self.SDR_RX_IP)
        self.sdr_rx.rx_destroy_buffer()
        self.sdr_rx.rx_lo                  = self.SDR_RX_FREQ
        self.sdr_rx.sample_rate            = self.SDR_TX_SAMPLERATE
        self.sdr_rx.rx_rf_bandwidth        = self.SDR_TX_BANDWIDTH
        self.sdr_rx.gain_control_mode_chan0 = "manual"
        self.sdr_rx.rx_hardwaregain_chan0   = self.SDR_RX_GAIN
        self.sdr_rx.rx_enabled_channels    = ["voltage0"]

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
    OFDM_mask[:, F // 2] = 3
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
    flattened_bits  = payload_bits_encoded.view(-1, Qm)
    payload_symbols = torch.tensor(
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
#  OTFS MODULATION LAYER
#
#  COMPLETE FIX HISTORY (v2 additions over v1)
#  ─────────────────────────────────────────────
#
#  v1 fixes (still present):
#    FIX 1  SFFT/ISFFT use norm='ortho' — eliminates N×M amplitude blowup
#    FIX 2  MACE adaptive threshold — stops on first-peak fraction, not
#           re-normalised residual; MACE_MAX_PATHS=4; threshold=0.30
#    FIX 3  H built analytically from tap list — exact for fractional taps
#    FIX 4  σ² estimated from DD-domain noise floor bins
#
#  v2 fixes (this file):
#
#  FIX 5 — MACE gain normalisation (was the primary BER=0.5 cause in v1)
#  ────────────────────────────────────────────────────────────────────────
#  In mace_estimate() we normalise R = Y_dd_pilot / pilot_amp_rx so that
#  the dominant tap has magnitude ≈ 1 in R.  The actual DD pilot value at
#  (kp,lp) is:
#
#    Y_dd[kp, lp]  ≈  h_dominant × pilot_power    (ortho SFFT, unit pilot)
#
#  After dividing by pilot_amp_rx (≈ h_dominant × pilot_power), the gains
#  extracted from R are in units of:
#
#    gain_extracted = h_true / pilot_power
#
#  But build_H_from_taps() builds H using these gains directly, so H
#  represents the channel divided by pilot_power.  Then equalize_MMSE_fullH()
#  solves H x = y/pilot_amp_rx, giving:
#
#    x_hat = (h_true / pilot_power) \ (h_true × X / pilot_amp_rx)
#           = X × pilot_power / pilot_amp_rx      ← off by pilot_power/pilot_amp_rx
#
#  With pilot_power=4 and pilot_amp_rx=12.12 this gives x_hat ≈ X/3,
#  pushing |x_hat| to ~0.18 instead of ~0.6.
#
#  Fix: multiply every extracted gain by pilot_power inside mace_estimate()
#  so that gains are in true channel units (h_true, O(1)):
#
#    gain_corrected = gain_extracted × pilot_power = h_true
#
#  Now H represents the true channel, H x = y/pilot_amp_rx gives
#  x_hat ≈ X/pilot_amp_rx × pilot_amp_rx = X  ✓
#
#  FIX 6 — σ² estimation from SINR rather than DD noise floor
#  ────────────────────────────────────────────────────────────
#  With N=14 and guard zones of pgk=5, pgd=4, almost all N×M = 1428 DD
#  bins are occupied by data symbols.  The "noise floor" bins in the
#  original estimate_dd_noise_power() measure data symbol energy (~12.7)
#  rather than noise (~0.6 at 14 dB SINR), inflating σ² by ~20×.
#
#  With σ²_scaled = 8.66e-02 (still 20× too large), MMSE heavily
#  suppresses the signal and x_hat collapses to near zero.
#
#  Fix: derive σ²_dd from the measured DD signal power and the
#  time-domain SINR estimate:
#
#    σ²_dd = mean(|Y_dd|²) / (sinr_linear + 1)
#
#  This is valid because after the ortho SFFT the noise-to-signal ratio
#  is preserved: SINR_dd == SINR_td.
# ============================================================

class OTFSModulator:
    """
    OTFS modulator/demodulator — v2 with all six fixes applied.

    Parameters
    ----------
    N  : int  – Doppler bins (= S)
    M  : int  – Delay bins   (= F)
    pilot_guard_delay   : int
    pilot_guard_doppler : int
    mace_threshold : float
    mace_max_paths : int
    use_mace : bool
    """

    def __init__(self, N=14, M=102,
                 pilot_guard_delay=4, pilot_guard_doppler=3,
                 mace_threshold=0.30, mace_max_paths=4,
                 use_mace=True):
        self.N  = N
        self.M  = M
        self.pgd = pilot_guard_delay
        self.pgk = pilot_guard_doppler
        self.use_mace       = use_mace
        self.mace_threshold = mace_threshold
        self.mace_max_paths = mace_max_paths
        self.kp = N // 2
        self.lp = M // 4

    # ── Transforms (ortho-normalised — FIX 1) ─────────────────
    def ISFFT(self, X_dd):
        """DD → TF using ortho-normalised IFFT so SFFT(ISFFT(X)) == X."""
        S_tf = torch.fft.ifft(X_dd, dim=0, norm="ortho")
        S_tf = torch.fft.ifft(S_tf, dim=1, norm="ortho")
        S_tf = torch.fft.fftshift(S_tf, dim=1)
        return S_tf

    def SFFT(self, S_tf):
        """TF → DD using ortho-normalised FFT — net gain = 1 (FIX 1)."""
        S_tf = torch.fft.ifftshift(S_tf, dim=1)
        Y_dd = torch.fft.fft(S_tf, dim=0, norm="ortho")
        Y_dd = torch.fft.fft(Y_dd, dim=1, norm="ortho")
        return Y_dd

    # ── Pilot embedding ───────────────────────────────────────
    def embed_pilot(self, X_dd, pilot_power=1.0):
        X_dd = X_dd.clone()
        for k in range(self.kp - self.pgk - 1, self.kp + self.pgk + 2):
            for l in range(self.lp - self.pgd - 1, self.lp + self.pgd + 2):
                X_dd[k % self.N, l % self.M] = 0.0 + 0.0j
        X_dd[self.kp, self.lp] = pilot_power + 0.0j
        return X_dd

    # ── Dirichlet kernel ──────────────────────────────────────
    @staticmethod
    def _dirichlet(x, Npts):
        """N-point Dirichlet kernel: sinc(x)/sinc(x/N)."""
        x = np.asarray(x, dtype=complex)
        return np.sinc(x) / np.sinc(x / Npts)

    # ── FIX 6: σ² from SINR ───────────────────────────────────
    @staticmethod
    def sigma2_from_sinr(Y_dd, sinr_db):
        """
        Derive DD-domain noise power from signal power and SINR.

        σ²_dd = mean(|Y_dd|²) / (sinr_linear + 1)

        The ortho SFFT preserves the SINR, so the time-domain SINR
        estimate applies directly to the DD grid.  This bypasses the
        problem of the noise-floor estimator measuring data energy when
        most DD bins are occupied by payload symbols.

        Parameters
        ----------
        Y_dd   : torch.Tensor (N, M)
        sinr_db: float — time-domain SINR estimate in dB

        Returns
        -------
        sigma2 : float — noise power in the DD domain
        """
        sinr_lin  = 10 ** (sinr_db / 10.0)
        mean_pwr  = torch.mean(torch.abs(Y_dd) ** 2).item()
        sigma2    = mean_pwr / (sinr_lin + 1.0)
        print(f"[OTFS σ²] SINR-based estimate: SINR={sinr_db:.1f}dB  "
              f"mean_pwr={mean_pwr:.4f}  σ²={sigma2:.4e}")
        return max(sigma2, 1e-12)

    # ── FIX 2+5: MACE with adaptive threshold & corrected gains ─
    def mace_estimate(self, Y_dd_pilot, pilot_power=1.0, plot=False):
        """
        MACE channel estimator — v2.

        FIX 5: gains multiplied by pilot_power before returning so they
        represent the true channel impulse response h_i (not h_i/pilot_power).

        FIX 2: adaptive stopping — threshold re-evaluated after each
        cancellation from the first-peak amplitude; iterations capped at
        MACE_MAX_PATHS (default 4 for N=14).

        Returns
        -------
        est_taps : list of dict
          {"gain": complex, "delay": float, "doppler": float}
          gain is in TRUE channel units (h_i), O(1) for unit-power channel.
        """
        N, M   = self.N, self.M
        kp, lp = self.kp, self.lp

        # Normalise residual so dominant tap ≈ 1
        pilot_amp_rx = float(torch.abs(Y_dd_pilot[kp, lp]).item())
        if pilot_amp_rx < 1e-10:
            pilot_amp_rx = max(float(torch.abs(Y_dd_pilot).max().item()), 1e-10)
            print("[MACE] WARNING: pilot bin near zero — using max as reference")

        R = (Y_dd_pilot / pilot_amp_rx).numpy().astype(complex)

        noise_floor = float(np.median(np.abs(R)))
        print(f"\n[MACE] N={N} M={M}  pilot at k={kp},l={lp}")
        print(f"[MACE] noise_floor={noise_floor:.4f}  pilot_amp_rx={pilot_amp_rx:.4f}")

        est_taps   = []
        iter_count = 0
        first_peak = None

        while True:
            idx  = int(np.argmax(np.abs(R)))
            peak = float(np.abs(R.flat[idx]))

            if first_peak is None:
                first_peak = peak

            adaptive_thr = max(self.mace_threshold * first_peak,
                               3.0 * noise_floor)

            if peak < adaptive_thr or iter_count >= self.mace_max_paths:
                break

            k_pk = idx // M
            l_pk = idx % M

            # Parabolic sub-bin refinement — Doppler
            R0   = float(np.abs(R[k_pk, l_pk]))
            Rp_k = float(np.abs(R[(k_pk + 1) % N, l_pk]))
            Rm_k = float(np.abs(R[(k_pk - 1) % N, l_pk]))
            denom_k = R0 - 0.5 * (Rp_k + Rm_k)
            eps_k   = 0.0 if abs(denom_k) < 1e-8 else float(np.clip(
                0.5 * (Rp_k - Rm_k) / (denom_k + 1e-12), -0.5, 0.5))

            # Parabolic sub-bin refinement — Delay
            Rp_l = float(np.abs(R[k_pk, (l_pk + 1) % M]))
            Rm_l = float(np.abs(R[k_pk, (l_pk - 1) % M]))
            denom_l = R0 - 0.5 * (Rp_l + Rm_l)
            eps_l   = 0.0 if abs(denom_l) < 1e-8 else float(np.clip(
                0.5 * (Rp_l - Rm_l) / (denom_l + 1e-12), -0.5, 0.5))

            dop_offset = ((kp - (k_pk + eps_k) + N // 2) % N) - N // 2
            del_offset = ((lp - (l_pk + eps_l) + M // 2) % M) - M // 2

            denom_kern = (self._dirichlet(eps_k, N)
                          * self._dirichlet(eps_l, M))
            if abs(denom_kern) < 1e-8:
                gain_normalised = complex(R[k_pk, l_pk])
            else:
                gain_normalised = complex(R[k_pk, l_pk]) / denom_kern

            # ── FIX 5: rescale to true channel units ──────────
            # R was divided by pilot_amp_rx.
            # pilot_amp_rx ≈ h_dominant × pilot_power  (ortho SFFT, 1 dominant tap)
            # Therefore gain_normalised ≈ h_i / (h_dominant)
            # We want h_i in absolute channel units:
            #   h_i_true = gain_normalised × pilot_amp_rx / pilot_power
            # Equivalently: multiply by pilot_power here so that when
            # build_H_from_taps uses these gains in H and the solver
            # divides y by pilot_amp_rx, the algebra cancels correctly:
            #   H * x = y / pilot_amp_rx
            #   (h_i_true) * x ≈ (h_true * X) / pilot_amp_rx
            #   x_hat ≈ X  ✓
            gain_true = gain_normalised * pilot_power
            # ──────────────────────────────────────────────────

            est_taps.append({
                "gain"    : gain_true,
                "delay"   : float(del_offset),
                "doppler" : float(dop_offset),
            })

            # Subtract Dirichlet kernel from residual (using normalised gain)
            k_arr = np.arange(N)
            l_arr = np.arange(M)
            dk = self._dirichlet(kp - dop_offset - k_arr, N)
            dl = self._dirichlet(lp - del_offset - l_arr, M)
            R -= gain_normalised * np.outer(dk, dl)

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

    # ── FIX 3: Analytical H from tap list ─────────────────────
    def build_H_from_taps(self, est_taps):
        """
        Build the (N·M × N·M) OTFS channel matrix analytically.

        Uses the exact I/O relation (ortho SFFT convention):

          H[(k_out,l_out),(k_in,l_in)]
            = Σ_i gain_i · exp(j2π ν_i k_out / N)
                         · D_M(l_out − l_in − τ_i) / √M

        where gain_i are the TRUE channel gains returned by mace_estimate()
        (after FIX 5 rescaling).  No pilot_amp_rx normalisation is needed
        here — the solver in equalize_MMSE_fullH handles that separately.

        Returns
        -------
        H_mat : torch.Tensor (N*M, N*M), complex64
        """
        N, M  = self.N, self.M
        size  = N * M
        H_mat = np.zeros((size, size), dtype=complex)

        k_out_arr = np.arange(N)
        l_out_arr = np.arange(M)

        for tap in est_taps:
            gain = tap['gain']    # true channel gain (FIX 5)
            tau  = tap['delay']
            nu   = tap['doppler']

            dop_phase = np.exp(1j * 2 * np.pi * nu * k_out_arr / N)  # (N,)
            l_diff    = (l_out_arr[:, None] - l_out_arr[None, :]).astype(float)
            del_kern  = self._dirichlet(l_diff - tau, M) / np.sqrt(M)  # (M,M)

            for k_in in range(N):
                for k_out in range(N):
                    row_base = k_out * M
                    col_base = k_in  * M
                    H_mat[row_base: row_base + M,
                          col_base: col_base + M] += gain * dop_phase[k_out] * del_kern

        H_tensor = torch.tensor(H_mat, dtype=torch.complex64)
        print(f"[OTFS H] analytical H built: {tuple(H_tensor.shape)}  "
              f"mean|H|={torch.abs(H_tensor).mean().item():.4f}  "
              f"(from {len(est_taps)} MACE tap(s))")
        return H_tensor

    # ── MMSE equaliser ────────────────────────────────────────
    def equalize_MMSE_fullH(self, Y_dd, H_mat, sigma2, pilot_amp_rx=1.0):
        """
        MMSE equaliser:  x̂ = (H†H + σ²I)⁻¹ H† y

        y_vec = Y_dd / pilot_amp_rx  to match the scale at which H was built:
          H was built from true gains (h_true ≈ O(1)).
          Y_dd ≈ h_true × X × pilot_amp_rx / pilot_power × pilot_power
               = h_true × X × pilot_amp_rx   (after FIX 5)
          y_vec = Y_dd / pilot_amp_rx ≈ h_true × X  ✓

        σ² is also scaled to the y_vec domain:
          σ²_scaled = sigma2 / pilot_amp_rx²

        Parameters
        ----------
        Y_dd        : torch.Tensor (N, M)
        H_mat       : torch.Tensor (N*M, N*M) from build_H_from_taps
        sigma2      : float — DD-domain noise power (from SINR estimate, FIX 6)
        pilot_amp_rx: float — measured pilot amplitude

        Returns
        -------
        X_eq : torch.Tensor (N, M)
        """
        N, M   = self.N, self.M
        H_np   = H_mat.numpy().astype(complex)
        y_vec  = (Y_dd.numpy().astype(complex) / pilot_amp_rx).flatten()
        sigma2_scaled = max(sigma2 / max(pilot_amp_rx ** 2, 1e-20), 1e-12)

        HH    = H_np.conj().T @ H_np
        rhs   = H_np.conj().T @ y_vec
        x_hat = np.linalg.solve(HH + sigma2_scaled * np.eye(HH.shape[0]), rhs)

        print(f"[OTFS MMSE] pilot_amp_rx={pilot_amp_rx:.4f}  "
              f"σ²_dd={sigma2:.4e}  σ²_scaled={sigma2_scaled:.4e}  "
              f"|x̂| range [{np.abs(x_hat).min():.3f}, {np.abs(x_hat).max():.3f}]  "
              f"|x̂| mean={np.abs(x_hat).mean():.3f}")

        return torch.tensor(x_hat.reshape(N, M), dtype=torch.complex64)

    # ── Legacy channel estimate (USE_MACE=False) ─────────────
    def channel_estimate_DD(self, Y_dd, pilot_power=1.0, plot=False):
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
                    if abs(k - kp) > pg_k or abs(l - lp) > pg_l:
                        noise_ref.append(torch.abs(Y_dd[k % N, l % M]).item())
        noise_floor = float(np.percentile(noise_ref, 75)) if noise_ref else 1e-6
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
            for dk, dl, kk, ll, hv in window_bins:
                if torch.abs(hv).item() > 0.10 * peak_mag:
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

    def equalize_DD(self, Y_dd, H_dd, noise_power=1e-4):
        """Legacy single-tap MMSE equaliser (USE_MACE=False)."""
        N, M   = self.N, self.M
        pg_l   = self.pgd
        peak_idx = torch.argmax(torch.abs(H_dd))
        pk, pl   = divmod(peak_idx.item(), M)
        h_dom    = H_dd[pk, pl]
        h_pow    = (torch.abs(h_dom) ** 2).item()
        if h_pow < 1e-20:
            print("[OTFS EQ] Warning: dominant tap ≈ 0, returning Y_dd unchanged")
            return Y_dd.clone()
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

    # ── Modulate ───────────────────────────────────────────────
    def modulate(self, payload_symbols, pilot_power=1.0, plot=False):
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
        X_dd = self.embed_pilot(X_dd, pilot_power)
        S_tf = self.ISFFT(X_dd)
        roundtrip_err = torch.max(torch.abs(self.SFFT(S_tf) - X_dd)).item()
        print(f"[OTFS TX] ISFFT/SFFT round-trip max error: {roundtrip_err:.6f}"
              f"  {'PASS' if roundtrip_err < 1e-4 else 'FAIL'}")
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

    # ── Demodulate ─────────────────────────────────────────────
    def demodulate(self, S_tf_rx, sinr_db=20.0, pilot_power=1.0, plot=False):
        """
        Recover equalised QAM symbols from the received TF grid.

        Parameters
        ----------
        S_tf_rx    : torch.Tensor (N, M)
        sinr_db    : float — measured time-domain SINR in dB (FIX 6)
        pilot_power: float — known TX pilot amplitude
        plot       : bool
        """
        N, M    = self.N, self.M
        kp, lp  = self.kp, self.lp
        pg_k, pg_l = self.pgk, self.pgd

        Y_dd = self.SFFT(S_tf_rx)

        # Integer CFO correction
        best_val   = 0.0
        k_peak, l_peak = kp, lp
        for k in range(kp - pg_k, kp + pg_k + 1):
            for l in range(lp - pg_l, lp + pg_l + 1):
                kk, ll = k % N, l % M
                v = torch.abs(Y_dd[kk, ll]).item()
                if v > best_val:
                    best_val = v; k_peak, l_peak = kk, ll
        delta_k = k_peak - kp
        if delta_k != 0:
            print(f"[OTFS CFO] Δk={delta_k:+d} — correcting")
            n_vec  = torch.arange(N, dtype=torch.float32).unsqueeze(1)
            phase  = -2 * np.pi * delta_k * n_vec / N
            corr   = torch.exp(1j * phase.to(torch.complex64))
            Y_dd   = self.SFFT(S_tf_rx * corr)
            best_val = 0.0
            for k in range(kp - pg_k, kp + pg_k + 1):
                for l in range(lp - pg_l, lp + pg_l + 1):
                    kk, ll = k % N, l % M
                    v = torch.abs(Y_dd[kk, ll]).item()
                    if v > best_val:
                        best_val = v; k_peak, l_peak = kk, ll
            print(f"[OTFS CFO] After correction: k={k_peak},l={l_peak} |Y|={best_val:.4f}")

        data_mask = torch.ones(N, M, dtype=torch.bool)
        for k in range(kp - pg_k - 1, kp + pg_k + 2):
            for l in range(lp - pg_l - 1, lp + pg_l + 2):
                data_mask[k % N, l % M] = False

        if self.use_mace:
            # FIX 6: σ² from SINR (not DD noise floor)
            sigma2_dd = self.sigma2_from_sinr(Y_dd, sinr_db)

            # Pilot window for MACE
            Y_dd_pilot_only = torch.zeros_like(Y_dd)
            for k in range(kp - pg_k - 1, kp + pg_k + 2):
                for l in range(lp - pg_l - 1, lp + pg_l + 2):
                    kk = k % N;  ll = l % M
                    Y_dd_pilot_only[kk, ll] = Y_dd[kk, ll]

            pilot_amp_rx = float(torch.abs(Y_dd_pilot_only[kp, lp]).item())
            if pilot_amp_rx < 1e-10:
                pilot_amp_rx = max(float(torch.abs(Y_dd).max().item()), 1e-10)
            print(f"[OTFS RX] pilot_amp_rx={pilot_amp_rx:.4f}  "
                  f"(TX pilot_power={pilot_power:.1f}  "
                  f"ratio={pilot_amp_rx/pilot_power:.3f})")

            # FIX 2+5: MACE with gain rescaling
            est_taps = self.mace_estimate(
                Y_dd_pilot_only, pilot_power=pilot_power, plot=plot)

            # FIX 3: analytical H
            H_mat = self.build_H_from_taps(est_taps)

            # FIX 1+5+6: MMSE with correct σ²
            X_dd_eq = self.equalize_MMSE_fullH(
                Y_dd, H_mat, sigma2=sigma2_dd, pilot_amp_rx=pilot_amp_rx)

            payload_syms_eq = X_dd_eq[data_mask]

            if plot:
                fig, axes = plt.subplots(1, 3, figsize=(16, 3))
                axes[0].imshow(torch.abs(Y_dd).numpy(), aspect='auto',
                               origin='lower', cmap='plasma')
                axes[0].set_title('RX DD Grid |Y_dd|')
                axes[0].set_xlabel('Delay bin'); axes[0].set_ylabel('Doppler bin')
                axes[0].axhline(k_peak, color='white', lw=0.5, ls='--')
                axes[0].axvline(l_peak, color='white', lw=0.5, ls='--')
                for t in est_taps:
                    kk = (kp - t['doppler']) % N
                    ll = (lp - t['delay'])   % M
                    axes[0].plot(ll, kk, 'c+', ms=12, mew=2)
                hm_sz = min(64, N * M)
                axes[1].imshow(torch.abs(H_mat[:hm_sz, :hm_sz]).numpy(),
                               aspect='auto', cmap='inferno', interpolation='nearest')
                axes[1].set_title('|H_mat| top-left block')
                syms_plot = payload_syms_eq.detach().cpu()
                axes[2].scatter(syms_plot.real.numpy(),
                                syms_plot.imag.numpy(), s=3, alpha=0.5)
                axes[2].set_title('Equalised constellation')
                axes[2].axis('equal')
                axes[2].set_xlim([-2, 2]); axes[2].set_ylim([-2, 2])
                axes[2].grid(True, ls='--', alpha=0.4)
                plt.tight_layout(); plt.show()

            return payload_syms_eq, Y_dd, H_mat

        else:
            # Legacy single-tap path
            sigma2_dd = self.sigma2_from_sinr(Y_dd, sinr_db)
            H_dd, taps = self.channel_estimate_DD(
                Y_dd, pilot_power=pilot_power, plot=plot)
            X_dd_eq = self.equalize_DD(Y_dd, H_dd, noise_power=sigma2_dd)
            payload_syms_eq = X_dd_eq[data_mask]
            if plot:
                fig, axes = plt.subplots(1, 2, figsize=(12, 3))
                axes[0].imshow(torch.abs(Y_dd).numpy(), aspect='auto',
                               origin='lower', cmap='plasma')
                axes[0].set_title('RX DD Grid |Y_dd| (legacy)')
                axes[0].axhline(k_peak, color='white', lw=0.5, ls='--')
                axes[0].axvline(l_peak, color='white', lw=0.5, ls='--')
                syms_plot = payload_syms_eq.detach().cpu()
                axes[1].scatter(syms_plot.real.numpy(),
                                syms_plot.imag.numpy(), s=3, alpha=0.5)
                axes[1].set_title('Equalised constellation (legacy)')
                axes[1].axis('equal')
                axes[1].set_xlim([-2, 2]); axes[1].set_ylim([-2, 2])
                axes[1].grid(True, ls='--', alpha=0.4)
                plt.tight_layout(); plt.show()
            return payload_syms_eq, Y_dd, H_dd


# ── Instantiate OTFS block ──────────────────────────────────
OTFS_PGK = 3 if not use_sdr else 5
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
                    file_bits, int(np.ceil(max_uncoded_bits / len(file_bits))))
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

        # Pilot power: with ortho SFFT, pilot appears at amplitude ~pilot_power
        # in the DD grid. A value of 4.0 gives ~12 dB above unit-power data.
        OTFS_PILOT_POWER = 4.0

        S_tf, X_dd, data_mask = otfs.modulate(
            pdsch_symbols, pilot_power=OTFS_PILOT_POWER, plot=True)

        S_tf_padded = torch.zeros(S, FFT_size, dtype=torch.complex64)
        S_tf_padded[:, FFT_offset: FFT_offset + F] = S_tf
        S_tf_padded[:, FFT_offset + F // 2] = 0.0 + 0.0j

        TD_TTI_IQ  = IFFT(S_tf_padded)
        TX_Samples = CP_addition(TD_TTI_IQ, S, FFT_size, CP)

        if use_sdr:
            zeros      = torch.zeros(leading_zeros, dtype=TX_Samples.dtype)
            TX_Samples = torch.cat((zeros, TX_Samples), dim=0)

        print(f"[OTFS TX] uncoded={pdsch_bits.numel()}  "
              f"encoded={pdsch_bits_encoded.numel()}  "
              f"DD_syms={pdsch_symbols.numel()}  "
              f"pilot_power={OTFS_PILOT_POWER}  use_mace={USE_MACE}")

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
        power     = torch.rand(1).item() / ((i + 1) * 10)
        f_D_local = torch.rand(1).item() * f_D
        N_sin     = 16
        n         = torch.arange(1, N_sin + 1)
        theta_n   = 2 * math.pi * n / (N_sin + 1)
        phase_n   = 2 * math.pi * torch.rand(N_sin)
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
        sig_pow   = torch.mean(torch.abs(fading_signal)**2)
        noise_pow = sig_pow / (10**(SINR / 10))
        noise     = torch.sqrt(noise_pow / 2) * (
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
    if titles: plt.title(f'PSD {info}')
    if save_plots: plt.savefig(f'pics/PSD_{info}.png', bbox_inches='tight')
    plt.show()

PSD_plot(TX_Samples, SampleRate, SDR_TX_Frequency, 'TX')
PSD_plot(RX_Samples, SampleRate, SDR_TX_Frequency, 'RX')


# ============================================================
# SYNCHRONISATION
# ============================================================
def sync_iq(tx_signal, rx_signal, leading_zeros, threshold=6, plot=False):
    tx_len  = tx_signal.numel()
    rx_len  = rx_signal.numel()
    end_pt  = rx_len - tx_len
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
        plot_i  = i_maxarg - leading_zeros
        c_len   = correlation.numel()
        win_lo  = max(plot_i - 10, 0)
        win_hi  = min(plot_i + 50, c_len)
        corr_v  = correlation[win_lo:win_hi]
        idx_off = range(win_lo - plot_i, win_hi - plot_i)
        disp_v  = [float(v) if float(v) > float(thr) else 0.0 for v in corr_v]
        if len(idx_off) > 0 and len(disp_v) > 0:
            plt.figure(figsize=(8, 3))
            plt.bar(list(idx_off), disp_v); plt.grid()
            plt.xlabel("Samples from start index")
            plt.ylabel("Correlation")
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
    pilots        = IQ_post_DFT[OFDM_mask_RE == 2]
    H_at_pilots   = pilots / pilot_symbols
    pilot_indices = torch.nonzero(OFDM_mask_RE[Sp] == 2,
                                   as_tuple=False).squeeze()
    all_indices   = torch.arange(FFT_offset, FFT_offset + F)
    H_real = piecewise_linear_interp(all_indices, pilot_indices, H_at_pilots.real)
    H_imag = piecewise_linear_interp(all_indices, pilot_indices, H_at_pilots.imag)
    H_estim = torch.view_as_complex(torch.stack([H_real, H_imag], dim=-1))
    if plotEst:
        dB = lambda x: 10 * torch.log10(torch.abs(x))
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
# HELPERS
# ============================================================
def remove_fft_Offests(IQ, F, FFT_offset):
    return IQ[:, torch.arange(FFT_offset, F + FFT_offset)]

def extract_middle_subcarriers(input_tensor, num_subcarriers):
    mid   = input_tensor.shape[-1] // 2
    start = mid - num_subcarriers // 2
    return input_tensor[..., start: start + num_subcarriers]

def phase_unwrap(phase):
    diff    = np.diff(phase)
    jumps   = np.abs(diff) > 1
    cum     = np.cumsum(diff)
    pu      = phase.copy()
    pu[1:] -= np.where(jumps, cum, 0)
    return pu

def equalize_ZF(IQ_post_DFT, H_estim, F, S, plotQAM=False):
    IQ_np     = IQ_post_DFT.cpu().numpy()
    H_np      = H_estim.cpu().numpy()
    ph_uw     = phase_unwrap(np.angle(H_np))
    eq_np     = IQ_np / np.abs(H_np) * np.exp(-1j * ph_uw)
    equalized = torch.tensor(eq_np, dtype=torch.complex64).view(S, F)
    if plotQAM:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
        for i in range(IQ_post_DFT.shape[0]):
            ax1.scatter(IQ_post_DFT[i].cpu().real.numpy(),
                        IQ_post_DFT[i].cpu().imag.numpy(), color='blue', s=5)
        ax1.axis('equal'); ax1.set_xlabel('Real'); ax1.set_ylabel('Imag')
        ax1.set_title('Pre-equalization'); ax1.grid(True, linestyle='--', alpha=0.7)
        for i in range(equalized.shape[0]):
            ax2.scatter(equalized[i].cpu().real.numpy(),
                        equalized[i].cpu().imag.numpy(), color='blue', s=5)
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
# COMMON RX CHAIN
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
# RX SIGNAL PROCESSING
# ============================================================
if not USE_OTFS:
    H_estim = channelEstimate_LS(
        OFDM_mask_rx, pilot_symbols, F, FFT_offset_RX, Sp,
        IQ_DFT, plotEst=True)
    OFDM_demod_no_offsets = remove_fft_Offests(IQ_DFT, F, FFT_offset_RX)
    equalized_H_estim     = equalize_ZF(OFDM_demod_no_offsets, H_estim,
                                         F, S, plotQAM=True)
    QAM_est = get_payload_symbols(OFDM_mask_rx, equalized_H_estim,
                                   FFT_offset_RX, F)

else:
    active_start = FFT_offset_RX
    active_end   = FFT_offset_RX + F
    S_tf_rx = IQ_DFT[:, active_start: active_end]

    print(f"[OTFS RX] S_tf_rx shape={tuple(S_tf_rx.shape)}  "
          f"mean_power={torch.mean(torch.abs(S_tf_rx)**2).item():.4f}")
    print(f"[OTFS RX] Mode: {'MACE + analytical-H MMSE' if USE_MACE else 'legacy single-tap'}")

    Y_dd_dbg = otfs.SFFT(S_tf_rx)
    peak_idx  = torch.argmax(torch.abs(Y_dd_dbg))
    pk, pl    = divmod(peak_idx.item(), otfs.M)
    print(f"[OTFS RX] Y_dd peak at k={pk}, l={pl}  "
          f"|Y|={torch.abs(Y_dd_dbg[pk, pl]).item():.4f}  "
          f"(expected k={otfs.kp}, l={otfs.lp})")

    # FIX 6: pass measured SINR to demodulate — σ² derived inside
    QAM_est_otfs, Y_dd, H_ch = otfs.demodulate(
        S_tf_rx,
        sinr_db=float(SINR_m),
        pilot_power=float(otfs_pilot_power),
        plot=True)

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

mode_str = "OTFS" if USE_OTFS else "OFDM"
mace_str = "+MACE" if (USE_OTFS and USE_MACE) else ""
print(f"[{mode_str}{mace_str}] BER: {BER:.3f}   SINR: {SINR_m:.1f} dB")


# ============================================================
# WAVEFORM PLOT
# ============================================================
def bits_to_16qam_symbols(bits):
    bits       = np.array(bits.cpu()).astype(int)
    bits       = bits[:len(bits) - (len(bits) % 4)]
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
plt.title(f"Reconstructed Signal [{mode_str}{mace_str}]")
plt.grid(True); plt.legend(); plt.show()


# ============================================================
# FINAL SINR + CHANNEL RE-ESTIMATE (OFDM only)
# ============================================================
SINR_m, noise_power, signal_power = SINR(
    RX_Samples, symbol_index_maxarg, leading_zeros)

if not USE_OTFS:
    H_estim = channelEstimate_LS(
        OFDM_mask_rx, pilot_symbols, F, FFT_offset_RX, Sp,
        IQ_DFT, plotEst=True)