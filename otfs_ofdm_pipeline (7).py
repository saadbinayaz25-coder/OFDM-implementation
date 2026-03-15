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
# OFDM / SYSTEM PARAMETERS  (unchanged from original)
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
# Reduced active subcarriers for OTFS (drop guard + DC, same as OFDM active)
OTFS_active = F     # we reuse the same F active subcarriers


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
# SDR CLASS  (unchanged from original)
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
# MODULATION MAPPING  (unchanged)
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
# OFDM BLOCK MASK  (unchanged)
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
# PILOT SET  (unchanged)
# ============================================================
def pilot_set(OFDM_mask, power_scaling=1.0):
    pilot_values = torch.tensor([-0.7-0.7j, -0.7+0.7j, 0.7-0.7j, 0.7+0.7j]) * power_scaling
    num_pilots   = OFDM_mask[OFDM_mask == 2].numel()
    print("num_pilots:", num_pilots)
    return pilot_values.repeat(num_pilots // 4 + 1)[:num_pilots]

pilot_symbols = pilot_set(OFDM_mask, 1)
print(pilot_symbols)


# ============================================================
# CONVOLUTIONAL CODEC  (unchanged)
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
# PAYLOAD CREATION  (unchanged)
# ============================================================
def create_payload(OFDM_mask, Qm, mapping_table, power=1.0, filename=None):
    payload_REs      = OFDM_mask.eq(1).sum().item()
    max_encoded_bits = payload_REs * Qm
    max_uncoded_bits = int(max_encoded_bits * 0.5)   # rate-1/2 code

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
# RE MAPPING / IFFT / CP  (unchanged)
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
#  ██████╗ ████████╗███████╗███████╗
# ██╔═══██╗╚══██╔══╝██╔════╝██╔════╝
# ██║   ██║   ██║   █████╗  ███████╗
# ██║   ██║   ██║   ██╔══╝  ╚════██║
# ╚██████╔╝   ██║   ██║     ███████║
#  ╚═════╝    ╚═╝   ╚═╝     ╚══════╝
#
# OTFS MODULATION LAYER
# Sits ON TOP of the OFDM physical layer.
# The OTFS block occupies the (N × M) delay-Doppler grid.
# N = OTFS_N (= S OFDM symbols), M = OTFS_M (= F subcarriers)
# ============================================================

class OTFSModulator:
    """
    Orthogonal Time Frequency Space (OTFS) modulator / demodulator.

    The OTFS frame maps an (N × M) delay-Doppler grid X[k,l]
    to a time-frequency grid S[n,m] via the Inverse Symplectic Fourier
    Transform (ISFFT), then hands the TF grid to the existing OFDM
    Heisenberg transform (IFFT + CP).

    Demodulation: Wigner transform (FFT + CP removal, already done by
    the OFDM RX chain) followed by the SFFT to recover the DD grid.

    Parameters
    ----------
    N  : int   – number of Doppler bins  (= OFDM symbols S)
    M  : int   – number of delay bins    (= active subcarriers F)
    pilot_guard_delay   : int – guard cells around pilot in delay
    pilot_guard_doppler : int – guard cells around pilot in Doppler
    """

    def __init__(self, N=14, M=102,
                 pilot_guard_delay=4, pilot_guard_doppler=3):
        self.N  = N
        self.M  = M
        self.pgd  = pilot_guard_delay    # must be ≥ max_delay_spread in bins
        self.pgk  = pilot_guard_doppler  # must be ≥ ceil(N * fd_max / subcarrier_spacing)

        # Pilot at centre of DD grid in Doppler; quarter-point in delay
        # (keeps guard zone away from the DC bin at M//2)
        self.kp = N // 2    # Doppler index
        self.lp = M // 4    # Delay index

    # ----------------------------------------------------------
    # ISFFT : delay-Doppler  →  time-frequency
    # ----------------------------------------------------------
    def ISFFT(self, X_dd):
        """
        X_dd : (N, M) complex tensor in delay-Doppler domain.
        Returns S_tf : (N, M) in time-frequency domain, freq axis DC-centred.

        ISFFT = ifft(dim=0) then ifft(dim=1).
        Both torch ifft calls apply 1/N normalisation, giving total 1/(NM).
        SFFT uses fft x fft (no normalisation factor), so SFFT(ISFFT(X))=X.
        fftshift on dim=1 converts zero-freq-first to DC-centred order so the
        output maps directly onto the padded FFT grid that IFFT() expects.
        """
        S_tf = torch.fft.ifft(X_dd, dim=0)        # Doppler -> time   (x1/N)
        S_tf = torch.fft.ifft(S_tf, dim=1)        # delay   -> freq   (x1/M)
        S_tf = torch.fft.fftshift(S_tf, dim=1)    # zero-first -> DC-centred
        return S_tf

    # ----------------------------------------------------------
    # SFFT : time-frequency  →  delay-Doppler
    # ----------------------------------------------------------
    def SFFT(self, S_tf):
        """
        S_tf : (N, M) complex tensor in time-frequency domain, freq DC-centred.
        Returns Y_dd : (N, M) complex tensor in delay-Doppler domain.

        SFFT = fft(dim=0) then fft(dim=1) — the inverse of ISFFT.
        torch.fft.fft has no normalisation factor, so SFFT(ISFFT(X)) = X.
        ifftshift on dim=1 converts DC-centred back to zero-freq-first before
        the forward FFT so the Doppler-delay mapping is consistent with TX.
        """
        S_tf = torch.fft.ifftshift(S_tf, dim=1)   # DC-centred -> zero-first
        Y_dd = torch.fft.fft(S_tf, dim=0)         # time   -> Doppler (no 1/N)
        Y_dd = torch.fft.fft(Y_dd, dim=1)         # freq   -> delay   (no 1/M)
        return Y_dd

    # ----------------------------------------------------------
    # Embed OTFS pilot into DD grid (impulse pilot)
    # ----------------------------------------------------------
    def embed_pilot(self, X_dd, pilot_power=1.0):
        """
        Place a known impulse pilot at (kp, lp) surrounded by guard zones.
        Returns modified X_dd and the pilot mask.
        """
        X_dd = X_dd.clone()
        pilot_mask = torch.zeros_like(X_dd, dtype=torch.bool)

        # Guard zone – force zeros
        for k in range(self.kp - self.pgk - 1, self.kp + self.pgk + 2):
            for l in range(self.lp - self.pgd - 1, self.lp + self.pgd + 2):
                kk = k % self.N;  ll = l % self.M
                X_dd[kk, ll] = 0.0 + 0.0j

        # Pilot impulse
        X_dd[self.kp, self.lp] = pilot_power + 0.0j
        pilot_mask[self.kp, self.lp] = True
        return X_dd, pilot_mask

    # ----------------------------------------------------------
    # DD-domain LS channel estimation
    # ----------------------------------------------------------
    def channel_estimate_DD(self, Y_dd, pilot_power=1.0, plot=False):
        """
        LS channel estimation in the delay-Doppler domain.

        After ISFFT the pilot impulse has magnitude pilot_power/(N*M)
        (because torch ifft normalises by 1/N per call, two calls give 1/(NM)).
        After SFFT (two unnormalised fft calls) the pilot peak is restored to
        pilot_power, so we divide Y_dd[kp,lp] by pilot_power directly.

        Returns H_dd (N x M, sparse) and tap_list for diagnostics.
        """
        kp, lp     = self.kp, self.lp
        pg_k, pg_l = self.pgk, self.pgd
        N, M       = self.N, self.M

        H_dd     = torch.zeros(N, M, dtype=torch.complex64)
        tap_list = []

        # Noise floor: use bins well outside the guard zone.
        # If pgk is large (SDR mode), the guard covers most Doppler rows and
        # noise_ref may be empty → fall back to all non-guard bins.
        noise_ref = []
        for k in range(N):
            for l in range(M):
                if abs(k - kp) > pg_k + 2 and abs(l - lp) > pg_l + 2:
                    noise_ref.append(torch.abs(Y_dd[k, l]).item())

        if not noise_ref:
            # Fall back: all bins except the tight pilot+guard window itself
            for k in range(N):
                for l in range(M):
                    outside_k = abs(k - kp) > pg_k
                    outside_l = abs(l - lp) > pg_l
                    if outside_k or outside_l:
                        noise_ref.append(torch.abs(Y_dd[k % N, l % M]).item())

        noise_floor = float(np.percentile(noise_ref, 75)) if noise_ref else 1e-6
        detect_thr  = 3.0 * noise_floor

        # Scan the guard window symmetrically in both delay directions (±pgd).
        # Causal-only (l≥lp) misses the pilot when a ±1 sample sync error
        # shifts it to lp-1.  The guard zone at TX already excludes all data
        # symbols from lp-pgd-1 to lp+pgd+1, so scanning both sides is safe.
        #
        # Strategy: collect ALL bins in the window, then take the absolute
        # peak as the dominant (direct-path) tap rather than relying on a
        # threshold that can fail when data leakage is non-negligible.
        window_bins = []
        for k in range(kp - pg_k, kp + pg_k + 1):
            for l in range(lp - pg_l, lp + pg_l + 1):
                kk = k % N;  ll = l % M
                h_val = Y_dd[kk, ll] / pilot_power
                H_dd[kk, ll] = h_val
                window_bins.append((k - kp, l - lp, kk, ll, h_val))

        # Pick the strongest bin as the dominant tap
        if window_bins:
            best = max(window_bins, key=lambda x: torch.abs(x[4]).item())
            dk_best, dl_best, kk_best, ll_best, h_best = best
            # Only include taps above 10% of the peak (suppress sidelobes)
            peak_mag = torch.abs(h_best).item()
            sidelobe_thr = 0.10 * peak_mag
            for dk, dl, kk, ll, hv in window_bins:
                if torch.abs(hv).item() > sidelobe_thr:
                    tap_list.append((dk, dl, hv))

        # Always guarantee the strongest bin is in the list
        if len(tap_list) == 0 and window_bins:
            tap_list.append((dk_best, dl_best, h_best))

        if plot:
            plt.figure(figsize=(8, 3))
            plt.imshow(torch.abs(H_dd).numpy(), aspect='auto',
                       origin='lower', cmap='viridis')
            plt.colorbar(label='|H_dd|')
            plt.xlabel('Delay bin'); plt.ylabel('Doppler bin')
            plt.title('OTFS DD-domain Channel Estimate')
            plt.tight_layout(); plt.show()
            print(f"[OTFS CE] {len(tap_list)} tap(s)  "
                  f"noise_floor={noise_floor:.4f}  thr={detect_thr:.4f}")
            for dk, dl, hv in tap_list:
                print(f"  Δk={dk:+d} Δl={dl:+d}  |h|={torch.abs(hv):.4f}")

        return H_dd, tap_list

    # ----------------------------------------------------------
    # DD-domain equalisation (one-tap per RE, LMMSE-style)
    # ----------------------------------------------------------
    def equalize_DD(self, Y_dd, H_dd, noise_power=1e-4):
        """
        Single-tap MMSE equaliser using the dominant channel tap.

        Finds the peak bin in H_dd (not hardcoded to [kp,lp]) so a ±1
        sample sync error that shifts the pilot to lp±1 is handled correctly.
        noise_power must already be scaled to the normalised RX domain
        (i.e. noise_power_measured / rx_norm_scale^2).
        """
        peak_idx  = torch.argmax(torch.abs(H_dd))
        pk, pl    = divmod(peak_idx.item(), self.M)
        h_dom     = H_dd[pk, pl]
        h_pow     = (torch.abs(h_dom) ** 2).item()

        if h_pow < 1e-20:
            print("[OTFS EQ] Warning: dominant tap ≈ 0, returning Y_dd unchanged")
            return Y_dd.clone()

        w = torch.conj(h_dom) / (h_pow + noise_power)
        print(f"[OTFS EQ] dominant tap k={pk},l={pl}  "
              f"|h|={h_pow**0.5:.4f}  |w|={abs(w.item()):.4f}")
        return Y_dd * w

    # ----------------------------------------------------------
    # MODULATE: bits → DD grid → TF grid  (handed to OFDM IFFT)
    # ----------------------------------------------------------
    def modulate(self, payload_symbols, pilot_power=1.0, plot=False):
        """
        Map QAM payload_symbols into the DD grid, embed pilot,
        apply ISFFT to get TF grid (N × M) ready for OFDM IFFT.

        payload_symbols : 1-D complex tensor, length ≤ N*M - guard/pilot cells
        Returns: S_tf (N × M), X_dd (N × M), data_mask (N × M bool)
        """
        N, M  = self.N, self.M
        kp, lp = self.kp, self.lp
        pg_k, pg_l = self.pgk, self.pgd

        # Build data mask: exclude pilot + guard zone
        data_mask = torch.ones(N, M, dtype=torch.bool)
        for k in range(kp - pg_k - 1, kp + pg_k + 2):
            for l in range(lp - pg_l - 1, lp + pg_l + 2):
                data_mask[k % N, l % M] = False

        n_data = data_mask.sum().item()
        assert payload_symbols.numel() <= n_data, \
            f"Too many payload symbols: {payload_symbols.numel()} > {n_data}"

        # Fill DD grid
        X_dd = torch.zeros(N, M, dtype=torch.complex64)
        flat_idx  = data_mask.flatten().nonzero(as_tuple=False).squeeze()
        sym_pad   = torch.zeros(n_data, dtype=torch.complex64)
        sym_pad[:payload_symbols.numel()] = payload_symbols
        X_dd.flatten()[flat_idx] = sym_pad

        # Embed pilot
        X_dd, _ = self.embed_pilot(X_dd, pilot_power)

        # ISFFT → TF grid
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

    # ----------------------------------------------------------
    # DEMODULATE: TF grid → DD grid → equalised symbols
    # ----------------------------------------------------------
    def demodulate(self, S_tf_rx, noise_power=1e-4, pilot_power=1.0,
                   plot=False):
        """
        Recover equalised QAM symbols from the received TF grid.

        S_tf_rx : (N × M) complex tensor — active subcarriers from DFT output
                  (all S symbols, shape matches the TX ISFFT output)
        noise_power : linear noise variance σ² (from SINR measurement)
        Returns: payload_syms_eq (1-D), Y_dd (N×M), H_dd (N×M)
        """
        N, M    = self.N, self.M
        kp, lp  = self.kp, self.lp
        pg_k, pg_l = self.pgk, self.pgd

        # ── Step 1: SFFT → delay-Doppler domain ──────────────────────────────
        Y_dd = self.SFFT(S_tf_rx)

        # ── Step 2: Find pilot peak (handles CFO-shifted Doppler bin) ────────
        # Search the entire guard window for the strongest bin.
        # Peak Doppler bin k_peak ≠ kp means there is integer CFO.
        best_val  = 0.0
        k_peak, l_peak = kp, lp
        for k in range(kp - pg_k, kp + pg_k + 1):
            for l in range(lp - pg_l, lp + pg_l + 1):
                kk, ll = k % N, l % M
                v = torch.abs(Y_dd[kk, ll]).item()
                if v > best_val:
                    best_val = v
                    k_peak, l_peak = kk, ll

        # ── Step 3: Integer CFO correction ───────────────────────────────────
        # If k_peak ≠ kp, the SDR carrier offset has shifted all symbols by
        # (k_peak - kp) Doppler bins.  Correct in the time-frequency domain:
        #   S_tf_corrected[n, m] = S_tf_rx[n, m] * exp(-j2π * Δk * n / N)
        # where Δk = k_peak - kp.
        delta_k = k_peak - kp
        if delta_k != 0:
            print(f"[OTFS CFO] Integer Doppler offset Δk={delta_k:+d} "
                  f"(≈{delta_k * 15000 / N:.0f} Hz) — correcting")
            n_vec   = torch.arange(N, dtype=torch.float32).unsqueeze(1)  # (N,1)
            phase   = -2 * np.pi * delta_k * n_vec / N                   # (N,1)
            corr    = torch.exp(1j * phase.to(torch.complex64))          # (N,1)
            S_tf_corrected = S_tf_rx * corr
            Y_dd    = self.SFFT(S_tf_corrected)
            # Re-find peak (should now be at kp after correction)
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

        # ── Step 4: DD-domain channel estimation ─────────────────────────────
        H_dd, taps = self.channel_estimate_DD(
            Y_dd, pilot_power=pilot_power, plot=plot)

        # ── Step 5: Per-delay-bin MMSE equalisation ───────────────────────────
        # The physical channel has fractional delay → sinc spreading across
        # adjacent delay bins.  A single-tap scalar equaliser leaves residual
        # ISI.  Instead, build a length-(2*pgd+1) MMSE filter in the delay
        # dimension and apply it per Doppler row of Y_dd.
        #
        # Model (delay-only channel, one Doppler row at a time):
        #   Y[k, l] = sum_{Δl=-pgd}^{pgd} h[Δl] * X[k, (l-Δl) % M]
        # In the delay-DFT domain (M-point FFT over l):
        #   Y_dft[k, m] = H_dft[m] * X_dft[k, m]
        # MMSE equaliser:  W[m] = H_dft*[m] / (|H_dft[m]|^2 + σ²)
        #
        # Steps: for each Doppler row k of Y_dd:
        #   1. M-point FFT over l
        #   2. Multiply by W[m]
        #   3. M-point IFFT back to delay domain

        # Build H_dft from relative channel taps h(Δl) centred at Δl=0.
        #
        # CRITICAL: the circular convolution model is
        #   Y_dd[k, l] = sum_Δl h(Δl) * X_dd[k, (l - Δl) % M]
        # so H_dft[m] = DFT{h(Δl)} where h is indexed by RELATIVE delay Δl.
        #
        # Placing h(Δl) at position (l_peak + Δl) % M instead of Δl % M
        # adds a factor exp(-j2πm*l_peak/M) to every H_dft[m], which makes
        # the MMSE equaliser output a circular-shifted version of X_dd by
        # l_peak bins → all symbols land at the WRONG delay positions → BER≈0.5.
        #
        # Fix: store h(Δl) at index (Δl % M), not at (l_peak + Δl) % M.
        h_taps_full = torch.zeros(M, dtype=torch.complex64)
        for dl_off in range(-pg_l, pg_l + 1):
            src_ll  = (l_peak + dl_off) % M          # where tap lives in H_dd
            dest_ll = dl_off % M                      # relative delay index (0-centred)
            h_taps_full[dest_ll] = H_dd[k_peak, src_ll]

        H_dft = torch.fft.fft(h_taps_full)   # M-point DFT of relative channel taps
        H_pow = torch.abs(H_dft) ** 2
        W_dft = torch.conj(H_dft) / (H_pow + noise_power)

        # Apply per-Doppler-row:  X_eq[k, :] = IFFT(FFT(Y_dd[k, :]) * W_dft)
        # This inverts the circular delay convolution for every Doppler slice.
        Y_dft_2d = torch.fft.fft(Y_dd, dim=1)           # (N, M) delay-DFT
        X_dft_eq = Y_dft_2d * W_dft.unsqueeze(0)        # broadcast over N rows
        X_dd_eq  = torch.fft.ifft(X_dft_eq, dim=1)      # back to delay domain

        print(f"[OTFS EQ] per-delay MMSE  peak=({k_peak},{l_peak})  "
              f"|H_dft| [{H_pow.min().item()**0.5:.2f}, {H_pow.max().item()**0.5:.2f}]"
              f"  σ²={noise_power:.2e}")

        # ── Step 6: Extract data symbols ─────────────────────────────────────
        data_mask = torch.ones(N, M, dtype=torch.bool)
        for k in range(kp - pg_k - 1, kp + pg_k + 2):
            for l in range(lp - pg_l - 1, lp + pg_l + 2):
                data_mask[k % N, l % M] = False

        payload_syms_eq = X_dd_eq[data_mask]

        if plot:
            fig, axes = plt.subplots(1, 2, figsize=(12, 3))
            axes[0].imshow(torch.abs(Y_dd).numpy(), aspect='auto',
                           origin='lower', cmap='plasma')
            axes[0].set_title('RX DD Grid |Y_dd|')
            axes[0].set_xlabel('Delay bin'); axes[0].set_ylabel('Doppler bin')
            axes[0].axhline(k_peak, color='white', lw=0.5, ls='--')
            axes[0].axvline(l_peak, color='white', lw=0.5, ls='--')

            syms_plot = payload_syms_eq.detach().cpu()
            axes[1].scatter(syms_plot.real.numpy(),
                            syms_plot.imag.numpy(), s=3, alpha=0.5)
            axes[1].set_title('OTFS equalised constellation')
            axes[1].set_xlabel('I'); axes[1].set_ylabel('Q')
            axes[1].axis('equal')
            axes[1].set_xlim([-2, 2]); axes[1].set_ylim([-2, 2])
            axes[1].grid(True, ls='--', alpha=0.4)
            plt.tight_layout(); plt.show()

        return payload_syms_eq, Y_dd, H_dd


# Instantiate OTFS block
# Guard sizes:
#   pgd ≥ max_delay_spread in samples = 3  → use 4 for margin
#   pgk: for SIMULATION use 3 (Doppler spread ≪ 1 bin).
#        for SDR hardware: Pluto CFO ≈ 1–5 ppm @ 2.4 GHz = 2–12 Doppler bins.
#        Set pgk large enough to catch the CFO-shifted pilot in the search window.
#        pgk=5 handles ±5 bins (±5400 Hz CFO ≈ 2 ppm).  Increase if CE still misses.
OTFS_PGK = 3 if not use_sdr else 5   # Doppler guard half-width
otfs = OTFSModulator(N=OTFS_N, M=OTFS_M,
                     pilot_guard_delay=4, pilot_guard_doppler=OTFS_PGK)


# ============================================================
# create_OFDM_data  –  UNIFIED TX FUNCTION
#   When USE_OTFS=True  : bits → QAM → OTFS DD grid → ISFFT
#                         → TF grid fills OFDM resource grid
#   When USE_OTFS=False : original OFDM path (untouched)
# ============================================================
def create_OFDM_data(filename=None):
    """
    Build the baseband TX samples for one TTI.

    In OTFS mode the OTFS TF grid (S × F) directly replaces the OFDM
    resource grid after RE_mapping, keeping CP_addition and IFFT intact.
    """

    if not USE_OTFS:
        # ---- Original OFDM path (100 % unchanged) ----
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
        # 1. How many DD data slots are available?
        n_data_DD = otfs.N * otfs.M
        for k in range(otfs.kp - otfs.pgk - 1, otfs.kp + otfs.pgk + 2):
            for l in range(otfs.lp - otfs.pgd - 1, otfs.lp + otfs.pgd + 2):
                n_data_DD -= 1   # subtract guard + pilot cells

        max_encoded_bits = n_data_DD * Qm
        max_uncoded_bits = int(max_encoded_bits * 0.5)

        # 2. Generate / load payload bits
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

        # 3. Convolutional encode
        pdsch_bits_encoded = conv_encode(pdsch_bits)

        # 4. QAM map
        fb = pdsch_bits_encoded.view(-1, Qm)
        pdsch_symbols = torch.tensor(
            [mapping_table_Qm[tuple(b.tolist())] for b in fb],
            dtype=torch.complex64)

        # Pilot power must exceed the data-symbol leakage floor in the DD domain.
        # With N*M=1428 bins and ~1393 data symbols of unit QAM power, the
        # interference-at-pilot-bin power ≈ 1393/(N*M) ≈ 0.975 (relative to
        # pilot_power=1).  A 20 dB margin requires pilot_power ≥ 10×sqrt(0.975) ≈ 10.
        OTFS_PILOT_POWER = 10.0

        # 5. OTFS modulate → TF grid (N × M), freq axis DC-centred
        S_tf, X_dd, data_mask = otfs.modulate(
            pdsch_symbols, pilot_power=OTFS_PILOT_POWER, plot=True)

        # --- self-test: verify SFFT(ISFFT(X_dd)) ≈ X_dd ---
        roundtrip_err = torch.max(torch.abs(otfs.SFFT(S_tf) - X_dd)).item()
        print(f"[OTFS TX] ISFFT/SFFT round-trip max error: {roundtrip_err:.6f}"
              f"  {'PASS' if roundtrip_err < 1e-4 else 'FAIL – check transforms'}")

        # 6. Insert TF grid into full FFT-size resource grid.
        #    ISFFT already outputs DC-centred frequency order.
        #    IFFT() applies ifftshift before ifft, which un-shifts the
        #    DC-centred layout back to zero-freq-first — exactly what
        #    torch.fft.ifft expects.  So we just zero-pad symmetrically.
        S_tf_padded = torch.zeros(S, FFT_size, dtype=torch.complex64)
        S_tf_padded[:, FFT_offset: FFT_offset + F] = S_tf

        # Zero the DC bin (FFT_offset + F//2 in the padded centred grid)
        S_tf_padded[:, FFT_offset + F // 2] = 0.0 + 0.0j

        # 7. IFFT (Heisenberg transform) — reuse existing function
        TD_TTI_IQ  = IFFT(S_tf_padded)

        # 8. CP addition — reuse existing function
        TX_Samples = CP_addition(TD_TTI_IQ, S, FFT_size, CP)

        # 9. Prepend leading zeros for SINR estimation (same as OFDM)
        if use_sdr:
            zeros      = torch.zeros(leading_zeros, dtype=TX_Samples.dtype)
            TX_Samples = torch.cat((zeros, TX_Samples), dim=0)

        print(f"[OTFS TX] uncoded bits={pdsch_bits.numel()}, "
              f"encoded bits={pdsch_bits_encoded.numel()}, "
              f"DD data symbols={pdsch_symbols.numel()}, "
              f"pilot_power={OTFS_PILOT_POWER}")

        return pdsch_bits, pdsch_bits_encoded, pdsch_symbols, TX_Samples, OTFS_PILOT_POWER


# ---- Generate TX samples ----
pdsch_bits, pdsch_bits_encoded, pdsch_symbols, TX_Samples, otfs_pilot_power = \
    create_OFDM_data(filename=None)   # replace None with your .bin path

print("Uncoded bits:", len(pdsch_bits))


# ============================================================
# SDR INIT  (unchanged)
# ============================================================
if use_sdr:
    SDR_1 = SDR(SDR_RX_IP=SDR_RX_IP, SDR_TX_IP=SDR_TX_IP,
                SDR_TX_FREQ=SDR_TX_Frequency, SDR_TX_GAIN=tx_gain,
                SDR_RX_GAIN=rx_gain, SDR_TX_SAMPLERATE=SampleRate,
                SDR_TX_BANDWIDTH=SDR_TX_BANDWIDTH)
    SDR_1.SDR_TX_start()
    SDR_1.SDR_RX_start()


# ============================================================
# MULTIPATH CHANNEL  (unchanged)
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
# RADIO CHANNEL  (unchanged)
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
# PSD  (unchanged)
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
# SYNCHRONISATION  (unchanged)
# ============================================================
def sync_iq(tx_signal, rx_signal, leading_zeros, threshold=6, plot=False):
    tx_len   = tx_signal.numel()
    rx_len   = rx_signal.numel()
    end_pt   = rx_len - tx_len
    if end_pt <= leading_zeros:
        # RX buffer too short — fall back to full-buffer search
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
        plot_i   = i_maxarg - leading_zeros          # index into correlation tensor
        c_len    = correlation.numel()

        # Safe window: clamp so we never slice outside [0, c_len)
        win_lo   = max(plot_i - 10, 0)
        win_hi   = min(plot_i + 50, c_len)
        offset_lo = win_lo - plot_i                  # ≤ 0  (usually -10)
        offset_hi = win_hi - plot_i                  # ≤ 50

        corr_v  = correlation[win_lo:win_hi]
        idx_off = range(offset_lo, offset_hi)        # same length as corr_v
        disp_v  = [float(v) if float(v) > float(thr) else 0.0 for v in corr_v]

        if len(idx_off) > 0 and len(disp_v) > 0:
            plt.figure(figsize=(8, 3))
            plt.bar(list(idx_off), disp_v); plt.grid()
            plt.xlabel("Samples from start index")
            plt.ylabel("Complex conjugate correlation")
            plt.gca().get_yaxis().set_visible(False)
            if save_plots: plt.savefig('pics/corr.png', bbox_inches='tight')
            plt.show()
        else:
            print("[sync_iq] Correlation window empty — skipping bar plot")

    return i + leading_zeros, i_maxarg

symbol_index, symbol_index_maxarg = sync_iq(
    TX_Samples, RX_Samples, leading_zeros=leading_zeros,
    threshold=0, plot=True)

if use_sdr:
    symbol_index_maxarg = symbol_index_maxarg + leading_zeros


# ============================================================
# SINR ESTIMATE  (unchanged)
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
# CP REMOVAL  (unchanged)
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
# DFT  (unchanged)
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
# CHANNEL ESTIMATION  (unchanged)
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

        plt.figure(figsize=(8, 3))
        plt.plot(pilot_indices.numpy(), ph(H_at_pilots).numpy(),
                 'ro-', label='Pilot phase', markersize=8)
        plt.plot(all_indices.numpy(), ph(H_estim).numpy(),
                 'b-', label='Estimated phase')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xlabel('Subcarrier Index'); plt.ylabel('Phase')
        plt.legend(); plt.tight_layout(); plt.show()

    return H_estim


# ============================================================
# HELPER FUNCTIONS  (unchanged)
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
# Save the normalisation scale before dividing — OTFS needs it to correct
# the noise power passed to the MMSE equaliser.
rx_norm_scale = torch.max(torch.abs(RX_NO_CP)).item()
RX_NO_CP = RX_NO_CP / rx_norm_scale

IQ_DFT = DFT(RX_NO_CP, plotDFT=True)
IQ_DFT = extract_middle_subcarriers(IQ_DFT, FFT_size_RX)

SINR_m, noise_power, signal_power = SINR(
    RX_Samples, symbol_index_maxarg, leading_zeros)


# ============================================================
# RX SIGNAL PROCESSING  – MODE-DEPENDENT SECTION
# ============================================================
if not USE_OTFS:
    # ---- Original OFDM RX path (100 % unchanged) ----
    H_estim = channelEstimate_LS(
        OFDM_mask_rx, pilot_symbols, F, FFT_offset_RX, Sp,
        IQ_DFT, plotEst=True)

    OFDM_demod_no_offsets = remove_fft_Offests(IQ_DFT, F, FFT_offset_RX)
    equalized_H_estim     = equalize_ZF(OFDM_demod_no_offsets, H_estim,
                                         F, S, plotQAM=True)
    QAM_est = get_payload_symbols(OFDM_mask_rx, equalized_H_estim,
                                   FFT_offset_RX, F)

else:
    # ---- OTFS RX path ----
    # IQ_DFT shape: (S, FFT_size_RX), freq axis DC-centred (fftshift applied).
    # FFT_offset_RX guards on each side: active band is [FFT_offset_RX : FFT_offset_RX+F].
    # SFFT() will ifftshift internally before the forward FFTs.
    active_start = FFT_offset_RX
    active_end   = FFT_offset_RX + F
    S_tf_rx = IQ_DFT[:, active_start: active_end]   # (S, F) DC-centred

    print(f"[OTFS RX] S_tf_rx shape={tuple(S_tf_rx.shape)}  "
          f"mean_power={torch.mean(torch.abs(S_tf_rx)**2).item():.4f}")

    # Quick sanity: run SFFT and show where the pilot energy peaks
    Y_dd_dbg = otfs.SFFT(S_tf_rx)
    peak_idx  = torch.argmax(torch.abs(Y_dd_dbg))
    pk, pl    = divmod(peak_idx.item(), otfs.M)
    print(f"[OTFS RX] Y_dd peak at k={pk}, l={pl}  "
          f"|Y|={torch.abs(Y_dd_dbg[pk, pl]).item():.4f}  "
          f"(expected pilot at k={otfs.kp}, l={otfs.lp})")

    # Correct the noise power for the RX_NO_CP normalisation.
    # noise_power from SINR() was measured on the unnormalised RX_Samples.
    # After dividing RX_NO_CP by rx_norm_scale, the effective noise power
    # in the normalised domain is noise_power / rx_norm_scale^2.
    noise_pwr_lin = float(noise_power.item()) \
        if isinstance(noise_power, torch.Tensor) else float(noise_power)
    noise_pwr_lin = noise_pwr_lin / (rx_norm_scale ** 2)
    noise_pwr_lin = max(noise_pwr_lin, 1e-10)
    print(f"[OTFS RX] rx_norm_scale={rx_norm_scale:.4f}  "
          f"noise_pwr_normalised={noise_pwr_lin:.2e}")

    QAM_est_otfs, Y_dd, H_dd = otfs.demodulate(
        S_tf_rx, noise_power=noise_pwr_lin,
        pilot_power=float(otfs_pilot_power), plot=True)

    # Trim to exactly the number of payload symbols transmitted
    n_tx_syms = pdsch_symbols.numel()
    QAM_est   = QAM_est_otfs[:n_tx_syms]

    print(f"[OTFS RX] recovered {QAM_est.numel()} symbols "
          f"(transmitted {n_tx_syms})")


# ============================================================
# DEMAPPING  (unchanged)
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
# BER  (unchanged)
# ============================================================
min_len     = min(decoded_bits.numel(), pdsch_bits.numel())
error_count = torch.sum(decoded_bits[:min_len] != pdsch_bits[:min_len]).float()
BER         = torch.round(error_count / min_len * 1000) / 1000

mode_str = "OTFS" if USE_OTFS else "OFDM"
print(f"[{mode_str}] BER: {BER:.3f}   SINR: {SINR_m:.1f} dB")


# ============================================================
# SIGNAL RECONSTRUCTION + WAVEFORM PLOT  (unchanged)
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
plt.title(f"Reconstructed Time-Domain Signal [{mode_str}]")
plt.grid(True); plt.legend(); plt.show()


# ============================================================
# FINAL SINR + CHANNEL RE-ESTIMATE  (unchanged)
# ============================================================
SINR_m, noise_power, signal_power = SINR(
    RX_Samples, symbol_index_maxarg, leading_zeros)

if not USE_OTFS:
    H_estim = channelEstimate_LS(
        OFDM_mask_rx, pilot_symbols, F, FFT_offset_RX, Sp,
        IQ_DFT, plotEst=True)
