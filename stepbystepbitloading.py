import numpy as np
import matplotlib.pyplot as plt
from config import * # OFDM Configuration is stored in config.py
import torch
import random
import adi
from torch.utils.data import Dataset
import torch
import torch.nn.functional as tFunc
from scipy.signal import butter, filtfilt
import os
import math
Qm = 4  # bits per symbol


# Channel Simulation Parameters in case no sdr
ch_SINR = 20  # SINR target for channel emulation
n_taps = 2 # max number of taps
max_delay_spread = 3 # samples. Note that 128*15kHz sample duration is already ~500ns.
velocity = 30 # velocity in m/s

# SDR Configuration
use_sdr = False  # Set to `True` to use SDR for transmission and reception, `False` to run channel simulation.
randomize_tx_gain = True # randomize tx gain
tx_gain_lo = -10 # min tx gain
tx_gain_hi = -10 # max tx gain
rx_gain = 15 # rx gain, if randomization is False
if use_sdr:
    SDR_TX_Frequency = 2400000000 # TX frequency in Hz
# OFDM Parameters
Qm = 4  # bits per symbol
F = 102  # Number of subcarriers, including DC
S = 14  # Number of symbols -----------------------------------------------
FFT_size = 128  # FFT size
FFT_size_RX = 128 # FFT size
Fp = 8 # Pilot subcarrier spacing
Sp = 2  # Pilot symbol, 0 for none
CP = 7  # Cyclic prefix
SCS = 15000  # Subcarrier spacing
P = F // Fp  # Number of pilot subcarriers
FFT_offset = int((FFT_size - F) / 2)  # FFT offset
FFT_offset_RX = int((FFT_size_RX - F) / 2)  # FFT offset

SampleRate = FFT_size * SCS  # Sample rate
Ts = 1 / (SCS * FFT_size)  # Sample duration
TTI_duration = Ts * (FFT_size + CP) * S * 1000  # TTI duration in ms
SDR_TX_Frequency = int(2400000000)  # base band center frequency
SDR_TX_BANDWIDTH = SCS*F*4
tx_gain = -5  # Transmission gain in dB for SDR
rx_gain = 30  # Reception gain in dB for SDR

TX_Scale = 0.7

######################################

# Additional Parameters
leading_zeros = 500  # Number of symbols with zero value for noise measurement at the beginning of the transmission. Used for SINR estimation.

# Save the generated plots
save_plots = False

# custom dataset
class CustomDataset(Dataset):
    def __init__(self):
        self.pdsch_iq = [] # pdsch symbols
        self.labels = [] # original bitstream labels
        self.sinr = [] # SINR

    def __len__(self):
        return len(self.pdsch_iq)
    
    def __getitem__(self, index):
        x1 = self.pdsch_iq[index]
        y = self.labels[index]
        z = self.sinr[index]
        return x1, y, z
    
    def add_item(self, new_pdsch_iq,  new_label, new_sinr):
        self.pdsch_iq.append(new_pdsch_iq) 
        self.labels.append(new_label) 
        self.sinr.append(new_sinr) 
            

save_plots
plot_width = 8
titles=False
SDR_TX_IP='ip:192.168.2.1'
SDR_RX_IP='ip:192.168.2.1'
class SDR:
    def __init__(self,
                 SDR_TX_IP='ip:192.168.2.1',
                 SDR_RX_IP='ip:192.168.2.1',
                 SDR_TX_FREQ=2400000000,
                 SDR_TX_GAIN=-80,
                 SDR_RX_GAIN=0,
                 SDR_TX_SAMPLERATE=1e6,
                 SDR_TX_BANDWIDTH=1e6):
        """
        Initialize the SDR class with the specified parameters.
        """
        self.SDR_TX_IP = SDR_TX_IP
        self.SDR_RX_IP = SDR_RX_IP
        self.SDR_TX_FREQ = int(SDR_TX_FREQ)
        self.SDR_RX_FREQ = int(SDR_TX_FREQ)
        self.SDR_TX_GAIN = int(SDR_TX_GAIN)
        self.SDR_RX_GAIN = int(SDR_RX_GAIN)
        self.SDR_TX_SAMPLERATE = int(SDR_TX_SAMPLERATE)
        self.SDR_TX_BANDWIDTH = int(SDR_TX_BANDWIDTH)
        self.num_samples = 0

        self.sdr_tx = None
        self.sdr_rx = None

    def SDR_TX_start(self):
        """
        Initialize and start the SDR transmitter.
        """
        self.sdr_tx = adi.ad9361(self.SDR_TX_IP)
        self.sdr_tx.tx_destroy_buffer()

        self.sdr_tx.tx_lo = self.SDR_TX_FREQ
        self.sdr_tx.sample_rate = self.SDR_TX_SAMPLERATE
        self.sdr_tx.tx_rf_bandwidth = self.SDR_TX_BANDWIDTH
        self.sdr_tx.tx_hardwaregain_chan0 = self.SDR_TX_GAIN

        self.sdr_tx.tx_enabled_channels = ["voltage0"]

    def SDR_RX_start(self):
        """
        Initialize and start the SDR receiver (full-duplex).
        """
        if self.sdr_tx is None:
            raise RuntimeError("TX must be started before RX for full-duplex.")

        self.sdr_rx = adi.Pluto(self.SDR_RX_IP)
        self.sdr_rx.rx_destroy_buffer()

        self.sdr_rx.rx_lo = self.SDR_RX_FREQ
        self.sdr_rx.sample_rate = self.SDR_TX_SAMPLERATE
        self.sdr_rx.rx_rf_bandwidth = self.SDR_TX_BANDWIDTH
        self.sdr_rx.gain_control_mode_chan0 = "manual"
        self.sdr_rx.rx_hardwaregain_chan0 = self.SDR_RX_GAIN

        self.sdr_rx.rx_enabled_channels = ["voltage0"]

    def SDR_gain_set(self, tx_gain, rx_gain):
        """
        Set the TX and RX gains.
        """
        if self.sdr_tx:
            self.sdr_tx.tx_hardwaregain_chan0 = tx_gain
        if self.sdr_rx:
            self.sdr_rx.rx_hardwaregain_chan0 = rx_gain

    def SDR_TX_send(self, SAMPLES, max_scale=1, cyclic=False):
        """
        Transmit samples through SDR transmitter.
        """
        self.sdr_tx.tx_destroy_buffer()

        if isinstance(SAMPLES, np.ndarray):
            self.num_samples = SAMPLES.size
        elif isinstance(SAMPLES, torch.Tensor):
            self.num_samples = SAMPLES.numel()
            SAMPLES = SAMPLES.numpy()

        samples = SAMPLES - np.mean(SAMPLES)
        samples = (samples / np.max(np.abs(samples))) * max_scale
        samples = self.lowpass_filter(samples)
        samples *= 2**14

        self.sdr_tx.tx_cyclic_buffer = cyclic
        self.sdr_tx.tx(samples.astype(np.complex64))

    def SDR_TX_stop(self):
        """
        Stop the SDR transmitter.
        """
        if self.sdr_tx:
            self.sdr_tx.tx_destroy_buffer()
            self.sdr_tx.rx_destroy_buffer()

    def lowpass_filter(self, data):
        """
        Applies a lowpass filter to the input data.
        """
        nyq = 0.5 * self.SDR_TX_SAMPLERATE
        normal_cutoff = (110 * 15000 * 0.5) / nyq
        b, a = butter(5, normal_cutoff, btype='low', analog=False)
        y = filtfilt(b, a, data)
        return np.array(y)

    def SDR_RX_receive(self, n_SAMPLES=None, normalize=True):
        """
        Receive samples from SDR receiver.
        """
        if n_SAMPLES is None:
            n_SAMPLES = self.num_samples * 4
        if n_SAMPLES <= 0:
            n_SAMPLES = 1

        self.sdr_rx.rx_destroy_buffer()
        self.sdr_rx.rx_buffer_size = n_SAMPLES

        rx_data = self.sdr_rx.rx()  # Returns numpy array for one channel

        rx_data = self.lowpass_filter(rx_data)

        if normalize:
            rx_data = rx_data / np.max(np.abs(rx_data))

        return torch.tensor(rx_data, dtype=torch.complex64)
def mapping_table(Qm, plot=False): ###
    """
    Create a modulation mapping table and its inverse for an OFDM system.

    Args:
        Qm (int): Modulation order.
        plot (bool): Flag to plot the constellation diagram.

    Returns:
        tuple: A tuple containing the mapping dictionary and the demapping dictionary.
    """
    # Size of the constellation
    size = int(torch.sqrt(torch.tensor(2**Qm)))
    
    # Create the constellation points
    a = torch.arange(size, dtype=torch.float32)
    
    # Shift the constellation to the center
    b = a - torch.mean(a)
    
    # Use broadcasting to create the complex constellation grid
    C = (b.unsqueeze(1) + 1j * b).flatten()
    
    # Normalize the constellation
    C /= torch.sqrt(torch.mean(torch.abs(C)**2))
    
    # Function to convert index to binary
    def index_to_binary(i, Qm):
        return tuple(map(int, '{:0{}b}'.format(int(i), Qm)))
    
    # Create the mapping dictionary
    mapping = {index_to_binary(i, Qm): val for i, val in enumerate(C)}
    
    # Create the demapping table
    demapping = {v: k for k, v in mapping.items()}

    # Plot the constellation if plot is True
    if plot:
        plt.figure(figsize=(4, 4))
        plt.scatter(C.numpy().real, C.numpy().imag)
        
        if titles:
            plt.title(f'Constellation - {Qm} bits per symbol')
        
        plt.ylabel('Imaginary')
        plt.xlabel('Real')
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('pics/const.png', bbox_inches='tight')
    
    return mapping, demapping  
mapping_table_QPSK, de_mapping_table_QPSK = mapping_table(2,plot=True) # mapping table QPSK (e.g. for pilot symbols)
mapping_table_Qm, de_mapping_table_Qm = mapping_table(Qm, plot=True) # mapping table for Qm  
def OFDM_block_mask(S, F, Fp, Sp, FFT_offset, plotOFDM_block=False): ###
    """
    Create a Transmission Time Interval (OFDM_block) mask for an OFDM system.

    Args:
        S (int): Number of symbols.
        F (int): Number of subcarriers.
        Fp (int): Pilot subcarrier spacing.
        Sp (int): Pilot symbol spacing.
        FFT_offset (int): FFT offset, calculated as (FFT size - Number of subcarriers)/2.
        plotOFDM_block (bool): Flag to plot the OFDM_block mask.

    Returns:
        torch.Tensor: The OFDM_block mask.
    """
    # Create a mask with all ones
    OFDM_mask = torch.ones((S, F), dtype=torch.int8)  # Initialize with ones

    # Set pilot symbol spacing Sp
    OFDM_mask[Sp, torch.arange(0, F, Fp)] = 2  # Mark pilot subcarriers
    OFDM_mask[Sp, 0] = 2  # Ensure the first subcarrier is a pilot
    OFDM_mask[Sp, F - 1] = 2  # Ensure the last subcarrier is a pilot

    # Set DC subcarrier to non-allocable power (oscillator phase noise)
    OFDM_mask[:, F // 2] = 3  # Mark DC subcarrier

    # Add FFT offsets
    OFDM_mask = torch.cat((torch.zeros(S, FFT_offset, dtype=torch.int8), OFDM_mask, torch.zeros(S, FFT_offset, dtype=torch.int8)), dim=1)

    # Plotting the OFDM_block mask if plotOFDM_block is True
    if plotOFDM_block:
        plt.figure(figsize=(plot_width, 1.5))
        plt.imshow(OFDM_mask.numpy(), aspect='auto')  # Convert tensor to NumPy array for plotting
        if titles:
            plt.title('OFDM_block mask')
        plt.xlabel('Subcarrier index')
        plt.ylabel('Symbol')
        if save_plots:
            plt.savefig('pics/OFDM_blockmask.png', bbox_inches='tight')
        plt.tight_layout()
        plt.show()

    return OFDM_mask
OFDM_mask = OFDM_block_mask(S=S,F=F, Fp=Fp, Sp=Sp, FFT_offset=FFT_offset,plotOFDM_block=True)
print ("OFDM_mask",len(OFDM_mask))
payload_elements_in_mask = OFDM_mask.eq(1).sum().item()*Qm
print("xyz",payload_elements_in_mask)
def pilot_set(OFDM_mask, power_scaling=1.0): ###
    """
    Generate a set of QPSK pilot values scaled by the given power scaling factor.

    Args:
        OFDM_mask (torch.Tensor): The OFDM_block mask indicating pilot positions.
        power_scaling (float): Scaling factor for the pilot power.

    Returns:
        torch.Tensor: The scaled pilot values.
    """
    # Define QPSK pilot values
    pilot_values = torch.tensor([-0.7 - 0.7j, -0.7 + 0.7j, 0.7 - 0.7j, 0.7 + 0.7j]) * power_scaling

    # Count the number of pilot elements in the OFDM_block mask
    num_pilots = OFDM_mask[OFDM_mask == 2].numel()
    print(num_pilots)
    # Create and return a list of pilot values repeated to match the number of pilots
    return pilot_values.repeat(num_pilots // 4 + 1)[:num_pilots]

pilot_symbols = pilot_set(OFDM_mask, 1)
print(pilot_symbols)
if use_sdr:
     SDR_1 = SDR(SDR_RX_IP=SDR_RX_IP, 
                           SDR_TX_IP=SDR_TX_IP,
                           SDR_TX_FREQ=SDR_TX_Frequency, 
                           SDR_TX_GAIN=tx_gain, 
                           SDR_RX_GAIN = rx_gain, 
                           SDR_TX_SAMPLERATE=SampleRate, 
                           SDR_TX_BANDWIDTH=SDR_TX_BANDWIDTH)
     SDR_1.SDR_TX_start()
     SDR_1.SDR_RX_start()
import numpy as np
import torch

def create_payload(OFDM_mask, Qm, mapping_table, power=1.0, filename=None):
    """
    Generate data symbols for an OFDM system, either from a binary file
    (stored as int8 bits: 0/1) or random bits.
    """

    # Count payload elements (number of symbols)
    payload_elements_in_mask = OFDM_mask.eq(1).sum().item()
    num_bits = payload_elements_in_mask * Qm
    
    # --- Get payload bits ---
    if filename is not None:
        # ✅ Read int8 bits (NOT bit-packed)
        file_bits = np.fromfile(filename, dtype=np.int8)

        # Safety check
        if not np.all((file_bits == 0) | (file_bits == 1)):
            raise ValueError("Input file must contain only 0/1 bits (int8).")

        # If file shorter than needed → repeat bits
        if len(file_bits) < num_bits: 
            reps = int(np.ceil(num_bits / len(file_bits)))
            file_bits = np.tile(file_bits, reps)

        payload_bits = torch.tensor(
            file_bits[:num_bits], dtype=torch.float32
        )
    else:
        # Random fallback
        payload_bits = torch.randint(
            0, 2, (num_bits,), dtype=torch.float32
        )

    # Reshape into groups of Qm bits per symbol
    flattened_bits = payload_bits.view(-1, Qm)
    
    # Map bits → symbols
    payload_symbols = torch.tensor(
        [mapping_table[tuple(row.tolist())] for row in flattened_bits],
        dtype=torch.complex64
    )

    # Apply power scaling
    payload_symbols *= power

    return payload_bits, payload_symbols
def create_OFDM_data():
    payload_bits, payload_symbols = create_payload(
        OFDM_mask,
        Qm,
        mapping_table,
        power=1.0,
        filename=r"D:\MTECH\MTP1\MTP1final\encoded.bin"
        
    )

    return payload_bits, payload_symbols

def RE_mapping(OFDM_mask, pilot_set, payload_symbols, plotOFDM_block=False):
    """
    Map Resource Elements (RE) in the OFDM-mask, allocating pilot and payload symbols.

    Args:
        OFDM_mask (torch.Tensor): The OFDM_block mask indicating positions for pilots and payload symbols.
        pilot_set (torch.Tensor): The set of pilot symbols.
        payload_symbols (torch.Tensor): The payload symbols.
        plotOFDM_block (bool): Flag to plot the OFDM_block modulated symbols.

    Returns:
        torch.Tensor: The OFDM_block with allocated symbols.
    """
    # Create a zero tensor for the overall F subcarriers * S symbols
    IQ = torch.zeros(OFDM_mask.shape, dtype=torch.complex64)

    # Allocate the payload and pilot
    IQ[OFDM_mask == 1] = payload_symbols.clone().detach()
    IQ[OFDM_mask == 2] = pilot_set.clone().detach()

    # Plotting the IQ 
    if plotOFDM_block:
        plt.figure(figsize=(plot_width, 1.5))
        plt.imshow(torch.abs(IQ).numpy(), aspect='auto')  # Convert tensor to NumPy array for plotting
        if titles:
            plt.title('OFDM_block modulated symbols')
        plt.xlabel('Subcarrier index')
        plt.ylabel('Symbol')
        if save_plots:
            plt.savefig('pics/OFDM_blockmod.png', bbox_inches='tight')
        plt.show()

    return IQ
def IFFT(IQ): ###
    """
    Perform an Inverse Fast Fourier Transform (IFFT) on the OFDM_block matrix.

    Args:
        OFDM_block (torch.Tensor): The OFDM_block matrix with modulated symbols.

    Returns:
        torch.Tensor: The time-domain signal after IFFT.
    """
    return torch.fft.ifft(torch.fft.ifftshift(IQ, dim=1))


def CP_addition(IQ, S, FFT_size, CP): ###
    """
    Add a cyclic prefix to each OFDM symbol.

    Args:
        IQ (torch.Tensor): The OFDM IQ data after IFFT.
        S (int): Number of symbols.
        FFT_size (int): FFT size.
        CP (int): Cyclic Prefix length.

    Returns:
        torch.Tensor: The OFDM data with cyclic prefixes added, flattened.
    """
    # Initialize output tensor
    out = torch.zeros((S, FFT_size + CP), dtype=torch.complex64)

    # Add cyclic prefix to each symbol
    for symbol in range(S):
        out[symbol, :] = torch.cat((IQ[symbol, -CP:], IQ[symbol, :]), dim=0)

    return out.flatten()
def create_OFDM_data():
    pdsch_bits, pdsch_symbols = create_payload(OFDM_mask, Qm, mapping_table_Qm, power=1) # create PDSCH data and modulate it
    Modulated_TTI = RE_mapping(OFDM_mask, pilot_symbols, pdsch_symbols, plotOFDM_block=True) # map the PDSCH and pilot symbols to the TTI
    TD_TTI_IQ = IFFT(Modulated_TTI) # perform the FFT
    TX_Samples = CP_addition(TD_TTI_IQ, S, FFT_size, CP) # add the CP
    if use_sdr:
        zeros = torch.zeros(leading_zeros, dtype=TX_Samples.dtype) # create leading zeros for estimating noise floor power
        TX_Samples = torch.cat((zeros, TX_Samples), dim=0) # add leading zeros to TX samples
    return pdsch_bits, TX_Samples

pdsch_bits, TX_Samples = create_OFDM_data()
print(len(pdsch_bits))
def apply_multipath_channel_dop(iq, max_n_taps, max_delay, repeats=0, random_start=True, SINR=30,
                                 leading_zeros=500, fc=432e6, velocity=30, fs=1e6, randomize= False):
    """
    Apply a multipath channel effect with time variation due to Doppler (Jakes fading model).

    Parameters:
    iq (torch.Tensor): The input signal.
    max_n_taps (int): Maximum number of taps in the multipath channel.
    max_delay (int): Maximum delay in samples.
    repeats (int): Number of repetitions of the signal.
    random_start (bool): Random circular shift of output.
    SINR (float): Signal-to-Interference-plus-Noise Ratio in dB.
    leading_zeros (int): Number of zeros to prepend to the output.
    fc (float): Carrier frequency in Hz.
    velocity (float): Maximum relative velocity in m/s.
    fs (float): Sample rate in Hz.

    Returns:
    torch.Tensor: Faded and noisy signal.
    torch.Tensor: Static channel impulse response snapshot.
    """

    c = 3e8
    if randomize:
        velocity = torch.rand(1).item() * (velocity - 1) + 1  # Avoid zero
    f_D = (velocity / c) * fc  # Maximum Doppler shift

    # Number of taps
    n_taps = torch.randint(1, max_n_taps + 1, (1,)).item()
    tap_indices = torch.randint(0, max_delay, (n_taps,))
    h = torch.zeros(max_delay, dtype=torch.complex64)

    # Time vector
    t = torch.arange(len(iq)) / fs

    # Output initialization
    fading_signal = torch.zeros_like(iq, dtype=torch.complex64)

    for i, delay in enumerate(tap_indices):
        power = torch.rand(1).item() / ((i + 1) * 10)  # Reduced per tap
        f_D_local = torch.rand(1).item() * f_D  # Tap-specific Doppler spread
        N = 16  # sinusoids per tap (can be increased)

        # Clarke's model - angle spaced sinusoids
        n = torch.arange(1, N + 1)
        theta_n = 2 * math.pi * n / (N + 1)
        phase_n = 2 * math.pi * torch.rand(N)

        # Jakes fading model (real + imag parts)
        jakes_real = torch.zeros(len(iq))
        jakes_imag = torch.zeros(len(iq))
        for k in range(N):
            jakes_real += torch.cos(2 * math.pi * f_D_local * torch.cos(theta_n[k]) * t + phase_n[k])
            jakes_imag += torch.sin(2 * math.pi * f_D_local * torch.cos(theta_n[k]) * t + phase_n[k])

        fading = power * (jakes_real + 1j * jakes_imag) / math.sqrt(N)

        # Apply delay and combine
        delayed_iq = torch.nn.functional.pad(iq, (delay, 0))[:len(iq)]
        fading_signal += fading * delayed_iq

        h[delay] = fading[0]  # snapshot at t=0

    # Leading zeros
    fading_signal = torch.cat([torch.zeros(leading_zeros, dtype=fading_signal.dtype), fading_signal])

    # Noise
    if SINR != 0:
        signal_power = torch.mean(torch.abs(fading_signal) ** 2)
        noise_power = signal_power / (10 ** (SINR / 10))
        noise = torch.sqrt(noise_power / 2) * (torch.randn_like(fading_signal) + 1j * torch.randn_like(fading_signal))
        fading_signal += noise

    # Optional circular shift
    if random_start:
        start_index = torch.randint(0, len(fading_signal), (1,)).item()
        fading_signal = torch.roll(fading_signal, shifts=start_index)

    # Repeat
    if repeats > 0:
        fading_signal = fading_signal.repeat(repeats)

    return fading_signal, h
def radio_channel(use_sdr, tx_signal, tx_gain, rx_gain, ch_SINR):
    if use_sdr:
        if randomize_tx_gain:
            tx_gain = random.randint(tx_gain_lo, tx_gain_hi) # add random gain to the TX gain
        SDR_1.SDR_gain_set(tx_gain, rx_gain) # set the gains
        print("TX Gain: ", tx_gain, ", RX Gain: ", rx_gain)
        SDR_1.SDR_TX_send(SAMPLES=tx_signal, max_scale=TX_Scale, cyclic=True) # start transmitting the symbols in cyclic mode
        rx_signal = SDR_1.SDR_RX_receive(len(tx_signal)*4) # receive the signal with 4 times more samples than the length of the transmitted signal.
        SDR_1.SDR_TX_stop() # stop the transmission
    else:
        rx_signal, h = apply_multipath_channel_dop(tx_signal, 
                                                   max_n_taps=n_taps, 
                                                   max_delay=max_delay_spread, 
                                                   random_start=True, 
                                                   repeats=3, 
                                                   SINR=ch_SINR, 
                                                   leading_zeros=leading_zeros,
                                                   fc=SDR_TX_Frequency, 
                                                   velocity=velocity,
                                                   fs=SampleRate,
                                                   randomize=False)
        print(h)
    return rx_signal
SDR_RX_IP="ip:192.168.2.1"
SDR_TX_IP="ip:192.168.2.1"

SDR_TX_Frequency = int(2400000000)  # base band center frequency
SDR_TX_BANDWIDTH = SCS*F*4
tx_gain = -5  # Transmission gain in dB for SDR
rx_gain = 30  # Reception gain in dB for SDR

TX_Scale = 0.7

######################################

# Additional Parameters
leading_zeros = 500  # Number of symbols with zero value for noise measurement at the beginning of the transmission. Used for SINR estimation.


RX_Samples = radio_channel(use_sdr=use_sdr, tx_signal = TX_Samples, tx_gain = tx_gain, rx_gain = rx_gain, ch_SINR=ch_SINR)
def PSD_plot(signal, Fs, f, info='TX'):
    """
    Plot the Power Spectral Density (PSD) of the given signal.

    Parameters:
    signal (torch.Tensor or np.ndarray): The signal to be analyzed.
    Fs (float): Sampling frequency.
    f (float): Center frequency.
    info (str): Information label for the plot. Default is 'TX'.

    Returns:
    None
    """
    # Convert PyTorch tensor to NumPy array if necessary
    if isinstance(signal, torch.Tensor):
        signal = signal.numpy()

    plt.figure(figsize=(8, 3))
    plt.psd(signal, Fs=Fs, NFFT=1024, Fc=f, color='blue')
    plt.grid(True)
    plt.ylim(-120,)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('PSD [dB/Hz]')
    if titles:
        plt.title(f'Power Spectral Density, {info}')
    if save_plots:
        plt.savefig(f'pics/PSD_{info}.png', bbox_inches='tight')
    plt.show()
PSD_plot(TX_Samples, SampleRate, SDR_TX_Frequency, 'TX')
PSD_plot(RX_Samples, SampleRate, SDR_TX_Frequency, 'RX')
def sync_iq(tx_signal, rx_signal, leading_zeros, threshold=6, plot=False): ###
    """
    Synchronize the Transmission Time Interval (OFDM_block) using cross-correlation.

    Parameters:
    tx_signal (torch.Tensor): Transmitted signal.
    rx_signal (torch.Tensor): Received signal.
    leading_zeros (int): Number of leading zeros in the signal.
    threshold (float): Correlation threshold for synchronization. Default is 6.
    plot (bool): Whether to plot the correlation results. Default is False.

    Returns:
    tuple: A tuple containing:
        - int: The synchronization index adjusted by leading zeros.
        - int: The maximum correlation value.
    """
    # Adjust the received signal length by removing leading and trailing zeros
    tx_len = tx_signal.numel()
    rx_len = rx_signal.numel()
    end_point = rx_len - tx_len
    rx_signal = rx_signal[leading_zeros:end_point]

    # Calculate the cross-correlation using conv1d
    corr_result_real = tFunc.conv1d(rx_signal.real.view(1, 1, -1), tx_signal.real.view(1, 1, -1)).view(-1)
    corr_result_imag = tFunc.conv1d(rx_signal.imag.view(1, 1, -1), tx_signal.imag.view(1, 1, -1)).view(-1)
    correlation = torch.complex(corr_result_real, corr_result_imag).abs()

    # Determine the threshold for synchronization
    threshold = correlation.mean() * threshold

    # Find the index of the maximum correlation value
    i_maxarg = torch.argmax(correlation).item() + leading_zeros

    # Find the first index where the correlation exceeds the threshold
    for i, value in enumerate(correlation):
        if value > threshold:
            break 

    if plot:
        plot_i = i_maxarg - leading_zeros

        # Create the index range and extract correlation values
        index_offset = range(-10, 50)
        absolute_indices = [plot_i + offset for offset in index_offset]
        correlation_values = correlation[plot_i-10:plot_i+50]

        # Set values below the threshold to zero for better visualization
        displayed_values = [value if value > threshold else 0 for value in correlation_values]

        plt.figure(figsize=(8, 3))
        plt.bar(index_offset, displayed_values)
        plt.grid()
        plt.xlabel("Samples from start index")
        plt.ylabel("Complex conjugate correlation")
        plt.gca().axes.get_yaxis().set_visible(False)
        
        if save_plots:
            plt.savefig('pics/corr.png', bbox_inches='tight')
        plt.show()

    return i + leading_zeros, i_maxarg
symbol_index, symbol_index_maxarg=sync_iq(TX_Samples, RX_Samples, leading_zeros=leading_zeros, threshold= 0, plot=True)
if use_sdr:
    symbol_index_maxarg = symbol_index_maxarg + leading_zeros
def SINR(rx_signal, index, leading_zeros): ###
    """
    Calculate the Signal-to-Interference-plus-Noise Ratio (SINR) for the received signal.

    Parameters:
    rx_signal (torch.Tensor): The received signal.
    index (int): The starting index for the signal of interest.
    leading_zeros (int): The number of leading zeros in the signal.

    Returns:
    tuple: A tuple containing:
        - float: The SINR value in dB.
        - torch.Tensor: The noise power.
        - torch.Tensor: The signal power.
    """
    # Calculate noise power
    rx_noise_0 = rx_signal[index - leading_zeros + 20 : index - 20]
    rx_noise_power_0 = torch.mean(torch.abs(rx_noise_0) ** 2)
    print("rx noise_power",rx_noise_power_0)
    # Calculate signal power
    rx_signal_0 = rx_signal[index : index + (14 * 72)]
    rx_signal_power_0 = torch.mean(torch.abs(rx_signal_0) ** 2)
    print("rx_signal_power_0",rx_signal_power_0)
    # Compute SINR
    SINR = 10 * torch.log10(rx_signal_power_0 / rx_noise_power_0)

    # Round the SINR value
    SINR = round(SINR.item(), 1)

    return SINR, rx_noise_power_0, rx_signal_power_0


def CP_removal(rx_signal, OFDM_block_start, S, FFT_size, CP, plotsig=False): ###
    """
    Remove the cyclic prefix from the received signal.

    Args:
        rx_signal (torch.Tensor): The received signal.
        OFDM_block_start (int): The starting index of the OFDM_block.
        S (int): Number of symbols.
        FFT_size (int): FFT size.
        CP (int): Cyclic prefix length.
        plotsig (bool): Flag to plot the received signal and payload mask.

    Returns:
        torch.Tensor: The received signal with cyclic prefix removed, reshaped to (S, FFT_size).
    """
    # Initialize a payload mask
    b_payload = torch.zeros(len(rx_signal), dtype=torch.bool)

    # Mark the payload parts of the signal
    for s in range(S):
        start_idx = OFDM_block_start + (s + 1) * CP + s * FFT_size
        end_idx = start_idx + FFT_size
        b_payload[start_idx:end_idx] = 1

    # Plotting the received signal and payload mask if plotsig is True
    if plotsig:
        rx_signal_numpy = rx_signal.cpu().numpy()  # Convert to NumPy array if needed
        rx_signal_normalized = rx_signal_numpy / np.max(np.abs(rx_signal_numpy))

        plt.figure(0, figsize=(plot_width, 3))
        plt.plot(rx_signal_normalized, label='Received Signal')
        plt.plot(b_payload.cpu().numpy(), label='Payload Mask')  # Ensure b_payload is on CPU
        plt.xlabel('Sample index')
        plt.ylabel('Amplitude')
        if titles:
            plt.title('Received signal and payload mask')
        plt.legend()
        if save_plots:
            plt.savefig('pics/RXsignal_sync.png', bbox_inches='tight')
        plt.show()

    # Remove the cyclic prefix
    rx_signal_no_CP = rx_signal[b_payload]

    return rx_signal_no_CP.view(S, FFT_size)

def DFT(rxsignal, plotDFT=False): ###
    """
    Perform a Discrete Fourier Transform (DFT) on the received signal.

    Args:
        rxsignal (torch.Tensor): The received signal.
        plotDFT (bool): Flag to plot the DFT result.

    Returns:
        torch.Tensor: The DFT of the received signal.
    """
    # Calculate DFT
    OFDM_RX_DFT = torch.fft.fftshift(torch.fft.fft(rxsignal, dim=1), dim=1)

    # Plot the DFT if required
    if plotDFT:
        plt.figure(figsize=(plot_width, 1.5))
        plt.imshow(torch.abs(OFDM_RX_DFT).numpy(), aspect='auto')  # Convert tensor to NumPy array for plotting
        plt.xlabel('Subcarrier Index')
        plt.ylabel('Symbol')
        if save_plots:
            plt.savefig('pics/OFDM_block_RX.png', bbox_inches='tight')
        plt.show()

    return OFDM_RX_DFT


import torch
import matplotlib.pyplot as plt
import torch
import matplotlib.pyplot as plt

def channelEstimate_LS(OFDM_mask_RE, pilot_symbols, F, FFT_offset, Sp, IQ_post_DFT, plotEst=False):
    """
    Perform Least Squares (LS) channel estimation using pilot symbols.

    Parameters:
    OFDM_mask_RE (torch.Tensor): Mask indicating pilot and data subcarriers.
    pilot_symbols (torch.Tensor): Known pilot symbols.
    F (int): Number of subcarriers.
    FFT_offset (int): Offset for FFT processing.
    Sp (int): Subcarrier spacing.
    IQ_post_DFT (torch.Tensor): Demodulated OFDM signal.
    plotEst (bool): Whether to plot the estimated channel. Default is False.

    Returns:
    torch.Tensor: Estimated channel response.
    """
    
    def unwrap_phase(phase):
        """
        Unwrap the phase to prevent discontinuities.
        """
        phase_diff = torch.diff(phase)
        phase_diff = torch.cat([phase_diff[:1], phase_diff])  # Retain the same length
        phase_unwrapped = phase + 2 * torch.pi * torch.cumsum((phase_diff + torch.pi) // (2 * torch.pi), dim=-1)
        #print(phase_unwrapped)
        return phase_unwrapped

    def wrap_phase(phase):
        """
        Wrap the phase back to the range [-pi, pi].
        """
        phase_wrapped = (phase + torch.pi) % (2 * torch.pi) - torch.pi
        return phase_wrapped

    def piecewise_linear_interp(x, xp, fp):
        """
        Perform piecewise linear interpolation on the given data points using PyTorch.

        Parameters:
        x (torch.Tensor): The x-coordinates at which to evaluate the interpolated values.
        xp (torch.Tensor): The x-coordinates of the data points.
        fp (torch.Tensor): The y-coordinates of the data points.

        Returns:
        torch.Tensor: The interpolated values at the specified x-coordinates.
        """
        # Ensure xp and fp are sorted
        sorted_indices = torch.argsort(xp)
        xp = xp[sorted_indices].float().to(device=x.device)
        fp = fp[sorted_indices].float().to(device=x.device)

        # Initialize the interpolated values tensor
        interp_values = torch.zeros_like(x, device=x.device).float()

        for i in range(len(xp) - 1):
            mask = (x >= xp[i]) & (x <= xp[i + 1])
            interp_values[mask] = fp[i] + (fp[i + 1] - fp[i]) * (x[mask] - xp[i]) / (xp[i + 1] - xp[i])

        return interp_values
    
    # Pilot extraction
    pilots = IQ_post_DFT[OFDM_mask_RE == 2]

    # Divide the pilots by the set pilot values to estimate channel at pilot positions
    H_estim_at_pilots = pilots / pilot_symbols

    # Interpolation indices for pilots
    pilot_indices = torch.nonzero(OFDM_mask_RE[Sp] == 2, as_tuple=False).squeeze()

    # All subcarrier indices
    all_indices = torch.arange(FFT_offset, FFT_offset + F)

    # Interpolate real and imaginary parts separately
    H_estim_real = piecewise_linear_interp(all_indices, pilot_indices, H_estim_at_pilots.real)
    H_estim_imag = piecewise_linear_interp(all_indices, pilot_indices, H_estim_at_pilots.imag)

    # Combine to complex estimate
    H_estim = torch.view_as_complex(torch.stack([H_estim_real, H_estim_imag], dim=-1))

    # # Unwrap phase of pilot estimates
    # H_estim_phase_unwrapped = unwrap_phase(torch.angle(H_estim_at_pilots))

    # # Linear interpolation for magnitude and unwrapped phase
    # H_estim_abs = piecewise_linear_interp(all_indices, pilot_indices, torch.abs(H_estim_at_pilots))
    # H_estim_phase = piecewise_linear_interp(all_indices, pilot_indices, H_estim_phase_unwrapped)

    # # Wrap the interpolated phase back to [-pi, pi]
    # H_estim_phase_wrapped = wrap_phase(H_estim_phase)

    # # Convert magnitude and phase to complex numbers
    # H_estim_real = H_estim_abs * torch.cos(H_estim_phase_wrapped)
    # H_estim_imag = H_estim_abs * torch.sin(H_estim_phase_wrapped)

    # # Combine real and imaginary parts to form complex numbers
    # H_estim = torch.view_as_complex(torch.stack([H_estim_real, H_estim_imag], dim=-1))

    def dB(x):
        return 10 * torch.log10(torch.abs(x))

    def phase(x):
        return torch.angle(x)

    if plotEst:
        plt.figure(figsize=(8, 3))
        plt.plot(pilot_indices.cpu().numpy(), dB(H_estim_at_pilots).cpu().numpy(), 'ro-', label='Pilot abs estimates', markersize=8)
        plt.plot(all_indices.cpu().numpy(), dB(H_estim).cpu().numpy(), 'b-', label='Estimated channel', linewidth=2)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        plt.xlabel('Subcarrier Index', fontsize=12)
        plt.ylabel('Magnitude (dB)', fontsize=12)
        plt.legend(loc='upper right', fontsize=10)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(8, 3))
        plt.plot(pilot_indices.cpu().numpy(), phase(H_estim_at_pilots).cpu().numpy(), 'ro-', label='Pilot phase estimates', markersize=8)
        plt.plot(all_indices.cpu().numpy(), phase(H_estim).cpu().numpy(), 'b-', label='Estimated channel', linewidth=2)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        plt.xlabel('Subcarrier Index', fontsize=12)
        plt.ylabel('Phase', fontsize=12)
        plt.legend(loc='upper right', fontsize=10)
        plt.tight_layout()
        plt.show()
        print(H_estim)
    return H_estim


def remove_fft_Offests(IQ, F, FFT_offset): ###
    """
    Remove FFT offsets from the received signal.

    Parameters:
    IQ (torch.Tensor): Received signal 
    F (int): Number of subcarriers.
    FFT_offset (int): Offset to be removed from FFT.

    Returns:
    torch.Tensor: Demodulated OFDM signal after removing FFT offsets.
    """
    # Calculate indices for the remaining subcarriers after removing offsets
    remaining_indices = torch.arange(FFT_offset, F + FFT_offset)

    # Remove the FFT offsets using slicing
    OFDM_slice = IQ[:, remaining_indices]

    return OFDM_slice



def phase_unwrap(phase):
    """
    Phase unwrapping function that handles phase discontinuities.

    Parameters:
    phase (np.ndarray): Phase array to be unwrapped.

    Returns:
    np.ndarray: Unwrapped phase array.
    """
    diff = np.diff(phase)
    jumps = np.abs(diff) > 1  # Detect phase jumps greater than pi
    cumulative_shift = np.cumsum(diff)
    phase_unwrapped = phase.copy()
    phase_unwrapped[1:] -= np.where(jumps, cumulative_shift, 0)  # Adjust phase unwrapping
    

    #return unwrapped_phase
    return phase_unwrapped

def equalize_ZF(IQ_post_DFT, H_estim, F, S, plotQAM=False):
    """
    Perform Zero-Forcing (ZF) equalization on the OFDM demodulated signal with phase unwrapping.

    Parameters:
    IQ_post_DFT (torch.Tensor): Demodulated OFDM signal after DFT.
    H_estim (torch.Tensor): Estimated channel response.
    F (int): Number of subcarriers.
    S (int): Number of OFDM symbols.
    plotQAM (bool): Whether to plot QAM symbols before and after equalization (default: False).

    Returns:
    torch.Tensor: Equalized OFDM signal.
    """


    # Convert torch tensors to numpy arrays for easier manipulation (assuming real data)
    IQ_post_DFT_np = IQ_post_DFT.cpu().numpy()
    H_estim_np = H_estim.cpu().numpy()

    # Perform phase unwrapping on the estimated channel response phase
    phase_H_estim = np.angle(H_estim_np)
    phase_H_estim_unwrapped = phase_unwrap(phase_H_estim)

    # Apply Zero-Forcing (ZF) equalization with phase correction
    equalized_np = IQ_post_DFT_np / np.abs(H_estim_np) * np.exp(-1j * phase_H_estim_unwrapped)

    # Convert back to torch tensor and reshape to original dimensions
    equalized = torch.tensor(equalized_np, dtype=torch.complex64, device=IQ_post_DFT.device).view(S, F)

    #equalized = IQ_post_DFT.view(S, F) / H_estim.to(device=IQ_post_DFT.device)
    
    if plotQAM:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))

        # First subplot for Pre Eq QAM Symbols
        for i in range(IQ_post_DFT.shape[0]):
            ax1.scatter(IQ_post_DFT[i].cpu().real.numpy(), IQ_post_DFT[i].cpu().imag.numpy(), color='blue',s=5)
        ax1.axis('equal')
        ax1.set_xlabel('Real Part', fontsize=12)
        ax1.set_ylabel('Imaginary Part', fontsize=12)
        ax1.set_title('Pre-equalization', fontsize=14)
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Second subplot for Post Eq QAM Symbols
        for i in range(equalized.shape[0]):
            ax2.scatter(equalized[i].cpu().real.numpy(), equalized[i].cpu().imag.numpy(), color='blue',s=5)
        ax2.axis('equal')
        ax2.set_xlim([-1.5, 1.5])
        ax2.set_ylim([-1.5, 1.5])
        ax2.set_xlabel('Real Part', fontsize=12)
        ax2.set_ylabel('Imaginary Part', fontsize=12)
        ax2.set_title('Post-Equalization', fontsize=14)
        ax2.grid(True, linestyle='--', alpha=0.7)
        if save_plots:
            plt.tight_layout()
            plt.savefig('pics/RXdSymbols_side_by_side.png', bbox_inches='tight')

        plt.show()
    
    return equalized

def PSD_plot(signal, Fs, f, info='TX'):
    """
    Plot the Power Spectral Density (PSD) of the given signal.

    Parameters:
    signal (torch.Tensor or np.ndarray): The signal to be analyzed.
    Fs (float): Sampling frequency.
    f (float): Center frequency.
    info (str): Information label for the plot. Default is 'TX'.

    Returns:
    None
    """
    # Convert PyTorch tensor to NumPy array if necessary
    if isinstance(signal, torch.Tensor):
        signal = signal.numpy()

    plt.figure(figsize=(8, 3))
    plt.psd(signal, Fs=Fs, NFFT=1024, Fc=f, color='blue')
    plt.grid(True)
    plt.ylim(-120,)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('PSD [dB/Hz]')
    if titles:
        plt.title(f'Power Spectral Density, {info}')
    if save_plots:
        plt.savefig(f'pics/PSD_{info}.png', bbox_inches='tight')
    plt.show()


def get_new_filename(directory, base_filename):
    """
    Generate a new filename in the specified directory to avoid overwriting existing files.

    Parameters:
    directory (str): The directory to save the file.
    base_filename (str): The base filename to use.

    Returns:
    str: A new filename with an incremental suffix.
    """
    os.makedirs(directory, exist_ok=True)
    nn = 0
    while True:
        new_filename = f"{base_filename}_{nn}.csv"
        if not os.path.exists(os.path.join(directory, new_filename)):
            return new_filename
        nn += 1

def extract_middle_subcarriers(input_tensor, num_subcarriers):
    """
    Extract the middle `num_subcarriers` from the last dimension of an OFDM block tensor.

    Args:
    input_tensor (tf.Tensor): Input tensor with shape (batch, ..., subcarriers) where
                              'subcarriers' dimension includes oversampled data.
    num_subcarriers (int): Number of subcarriers to extract from the middle.

    Returns:
    tf.Tensor: Tensor with the middle `num_subcarriers` extracted.
    """
    # Calculate the middle index of the last dimension
    middle_index = input_tensor.shape[-1] // 2

    # Calculate start and end indices for slicing
    start_index = middle_index - num_subcarriers // 2
    end_index = start_index + num_subcarriers

    # Slice the tensor to get the middle subcarriers
    sliced_tensor = input_tensor[..., start_index:end_index]

    return sliced_tensor
RX_NO_CP = CP_removal(RX_Samples, symbol_index_maxarg, S, FFT_size, CP, plotsig=True) # remove the cyclic prefix
RX_NO_CP = RX_NO_CP / torch.max(torch.abs(RX_NO_CP)) # normalize the signal
OFDM_mask = extract_middle_subcarriers(OFDM_mask, FFT_size_RX)
IQ_DFT = DFT(RX_NO_CP, plotDFT=True) # perform the DFT on the received signal and plot the result
IQ_DFT = extract_middle_subcarriers(IQ_DFT, FFT_size_RX)
SINR_m, noise_power, signal_power = SINR(RX_Samples, symbol_index_maxarg, leading_zeros) # calculate the SINR 
H_estim = channelEstimate_LS(OFDM_mask, pilot_symbols, F, FFT_offset_RX, Sp, IQ_DFT, plotEst=True) # estimate the channel using least squares and plot
OFDM_demod_no_offsets = remove_fft_Offests(IQ_DFT, F, FFT_offset_RX) # remove the FFT offsets and DC carrier from the received signal
equalized_H_estim = equalize_ZF(OFDM_demod_no_offsets, H_estim, F, S, plotQAM=True) # equalize the channel using ZF

def get_payload_symbols(OFDM_mask_RE, equalized, FFT_offset, F): ###
    """
    Extract payload symbols from the equalized OFDM signal and optionally plot the QAM constellation.

    Parameters:
    OFDM_mask_RE (torch.Tensor): Mask indicating pilot and data subcarriers.
    equalized (torch.Tensor): Equalized OFDM signal.
    FFT_offset (int): Offset for FFT processing.
    F (int): Number of subcarriers.

    Returns:
    torch.Tensor: Extracted payload symbols.
    """
    # Extract payload symbols
    mask = OFDM_mask_RE[:, FFT_offset:FFT_offset + F] == 1
    return equalized[mask]
print("equalized_H_estim =", equalized_H_estim)
if equalized_H_estim is None:
    raise RuntimeError("equalized_H_estim is None — check equalize_ZF function")

QAM_est = get_payload_symbols(OFDM_mask, equalized_H_estim, FFT_offset_RX, F) # get the payload symbols from the equalized signal 
def Demapping(QAM, de_mapping_table): ###
    """
    Demap the received QAM symbols to their corresponding bit representations.

    Args:
        QAM (torch.Tensor): The received QAM symbols.
        de_mapping_table (dict): The demapping table mapping constellation points to bit tuples.

    Returns:
        tuple: A tuple containing the demapped symbols (as bit tuples) and the hard decision symbols.
    """
    # Convert the demapping table keys (constellation points) to a tensor
    constellation = torch.tensor(list(de_mapping_table.keys())).to(QAM.device)
    
    # Calculate the distance between each received symbol and each constellation point
    dists = torch.abs(QAM.view(-1, 1) - constellation.view(1, -1))
    
    # Find the nearest constellation point for each received symbol
    const_index = torch.argmin(dists, dim=1).to(QAM.device)
    hardDecision = constellation[const_index].to(QAM.device)
    
    # Convert the demapping table to use string keys for easy lookup
    string_key_table = {str(key.item()): value for key, value in de_mapping_table.items()}
    
    # Demap the symbols based on the hard decision constellation points
    demapped_symbols = torch.tensor([string_key_table[str(c.item())] for c in hardDecision], dtype=torch.int32)
    
    return demapped_symbols, hardDecision

def SP(bits, length, Qm):
    """
    Convert a serial bit stream to parallel format.

    Args:
        bits (torch.Tensor): The input bit stream.
        length (int): The number of parallel streams.
        Qm (int): The modulation order.

    Returns:
        torch.Tensor: The reshaped bit stream in parallel format.
    """
    return bits.reshape((length, Qm))


def PS(bits):
    """
    Convert a parallel bit stream to serial format.

    Args:
        bits (torch.Tensor): The input bit stream in parallel format.

    Returns:
        torch.Tensor: The reshaped bit stream in serial format.
    """
    return bits.reshape((-1,))


def calculate_error_rate(bits_est, payload_bits):
    """
    Calculate the error count and error rate between estimated bits and the original payload bits.

    Args:
        bits_est (torch.Tensor): The estimated bits.
        payload_bits (torch.Tensor): The original payload bits.

    Returns:
        tuple: A tuple containing the error count and error rate.
    """
    # Flatten the payload_bits tensor for comparison
    flattened_payload_bits = payload_bits.flatten()

    # Count the number of unequal bits
    error_count = torch.sum(bits_est != flattened_payload_bits).float()

    # Calculate the error rate
    error_rate = error_count / bits_est.numel()

    return error_count, error_rate

def PSD_plot(signal, Fs, f, info='TX'):
    """
    Plot the Power Spectral Density (PSD) of the given signal.

    Parameters:
    signal (torch.Tensor or np.ndarray): The signal to be analyzed.
    Fs (float): Sampling frequency.
    f (float): Center frequency.
    info (str): Information label for the plot. Default is 'TX'.

    Returns:
    None
    """
    # Convert PyTorch tensor to NumPy array if necessary
    if isinstance(signal, torch.Tensor):
        signal = signal.numpy()

    plt.figure(figsize=(8, 3))
    plt.psd(signal, Fs=Fs, NFFT=1024, Fc=f, color='blue')
    plt.grid(True)
    plt.ylim(-120,)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('PSD [dB/Hz]')
    if titles:
        plt.title(f'Power Spectral Density, {info}')
    if save_plots:
        plt.savefig(f'pics/PSD_{info}.png', bbox_inches='tight')
    plt.show()
PS_est, hardDecision = Demapping(QAM_est, de_mapping_table_Qm) # demap the symbols back to codewords
bits_est = PS(PS_est) # convert the codewords to the bitstream
print("hi",bits_est,len(bits_est))  

import numpy as np

def bits_to_16qam_symbols(bits_est):
    """
    Convert binary stream to 16-QAM symbols using Gray mapping.

    Parameters:
        bits_est (np.array or list or torch.Tensor): 1D array of bits (0/1).
    
    Returns:
        np.array: Complex-valued 16-QAM symbols.
    """
    bits = np.array(bits_est).astype(int)  # <-- fixed line
    k = 4  # bits per symbol
    assert len(bits) % k == 0, "Length of bits must be a multiple of 4."

    bit_groups = bits.reshape((-1, k))

    # Gray-coded mapping table for 16-QAM
    mapping_table = {
        (0,0,0,0): -3-3j,
        (0,0,0,1): -3-1j,
        (0,0,1,1): -3+1j,
        (0,0,1,0): -3+3j,
        (0,1,1,0): -1+3j,
        (0,1,1,1): -1+1j,
        (0,1,0,1): -1-1j,
        (0,1,0,0): -1-3j,
        (1,1,0,0):  1-3j,
        (1,1,0,1):  1-1j,
        (1,1,1,1):  1+1j,
        (1,1,1,0):  1+3j,
        (1,0,1,0):  3+3j,
        (1,0,1,1):  3+1j,
        (1,0,0,1):  3-1j,
        (1,0,0,0):  3-3j,
    }

    symbols = np.array([mapping_table[tuple(b)] for b in bit_groups])
    symbols /= np.sqrt(10)  # normalization

    return symbols

from scipy.signal import upfirdn

def upsample_symbols(symbols, samples_per_symbol=8):
    """
    Upsample the symbol stream to create a time-domain waveform.
    """
    return upfirdn([1], symbols, samples_per_symbol)
import matplotlib.pyplot as plt

def plot_time_domain_waveform(waveform):
    plt.figure(figsize=(10, 4))
    plt.plot(np.real(waveform), label="I (Real)")
    plt.plot(np.imag(waveform), label="Q (Imag)")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.title("Reconstructed Time-Domain Waveform (16-QAM)")
    plt.legend()
    plt.grid(True)
    plt.show()
def plot_frequency_spectrum(waveform, fs=1e6):
    fft_data = np.fft.fftshift(np.fft.fft(waveform))
    freq = np.fft.fftshift(np.fft.fftfreq(len(waveform), 1/fs))

    plt.figure(figsize=(10, 4))
    plt.plot(freq, 20*np.log10(np.abs(fft_data)))
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.title("Frequency Spectrum of Reconstructed Signal")
    plt.grid(True)
    plt.show()
error_count = torch.sum(bits_est != pdsch_bits.flatten()).float()  # Count of unequal bits
print("len of bits_est",len(bits_est))
error_rate = error_count / bits_est.numel()  # Error rate calculation
BER = torch.round(error_rate * 1000) / 1000  # Round to 3 decimal places
print(f"BER: {BER:.3f}, SINR: {SINR_m:.1f}dB") # print the BER and SINR
error_count = torch.sum(bits_est != pdsch_bits.flatten()).float()  # Count of unequal bits
error_rate = error_count / bits_est.numel()  # Error rate calculation
BER = torch.round(error_rate * 1000) / 1000  # Round to 3 decimal places
print(f"BER: {BER:.3f}, SINR: {SINR_m:.1f}dB") # print the BER and SINR
PS_est, hardDecision = Demapping(QAM_est, de_mapping_table_Qm)
bits_est = PS(PS_est)
print(bits_est)

symbols = bits_to_16qam_symbols(bits_est)
waveform = upsample_symbols(symbols, samples_per_symbol=8)

plot_time_domain_waveform(waveform)
plot_frequency_spectrum(waveform, fs=1e6)

error_count = torch.sum(bits_est != pdsch_bits.flatten()).float()
error_rate = error_count / bits_est.numel()
BER = torch.round(error_rate * 1000) / 1000
print(f"BER: {BER:.3f}, SINR: {SINR_m:.1f}dB")    
SINR_m, noise_power, signal_power = SINR(RX_Samples, symbol_index_maxarg, leading_zeros)
H_estim = channelEstimate_LS(OFDM_mask, pilot_symbols, F, FFT_offset_RX, Sp, IQ_DFT, plotEst=True)
# ============================================================
# TOTAL RECEIVED BITS CALCULATION (APPEND THIS AT THE END)
# ============================================================

# Assumptions (already present in your code):
# S          : number of OFDM symbols
# F          : number of subcarriers
# Qm         : bits per QAM symbol (Qm = 4)
# OFDM_mask  : (S x F) tensor, 1=data, 0=pilot or unused
# DC_index   : index of DC subcarrier (usually F//2)

# ----- Ensure DC subcarrier is excluded -----
DC_index = F // 2
OFDM_mask[:, DC_index] = 0

# ----- Count total received QAM symbols -----
total_qam_symbols_rx = int(torch.sum(OFDM_mask).item())

# ----- Total received bits -----
total_received_bits = total_qam_symbols_rx * Qm

# ----- Print results -----
print("===================================")
print("OFDM symbols (S):", S)
print("Subcarriers (F):", F)
print("Bits per QAM symbol (Qm):", Qm)
print("Total QAM symbols received:", total_qam_symbols_rx)
print("✅ Total received payload bits:", total_received_bits)
print("===================================")

# ================= SAVE RECEIVED BITS TO FILE =================

import numpy as np
import os

# Target file path
save_path = r"D:\MTECH\MTP1\MTP1final\savedbits.bin"

# Ensure directory exists
os.makedirs(os.path.dirname(save_path), exist_ok=True)

# Convert torch tensor -> numpy uint8 (0/1)
bits_np = bits_est.detach().cpu().numpy().astype(np.uint8)

# Convert bits to '0' and '1' characters
bits_str = ''.join(bits_np.astype(str))

# Write to file as bytes
with open(save_path, "wb") as f:
    f.write(bits_str.encode('utf-8'))

print(f"[OK] Saved {len(bits_np)} received bits to:")
print(save_path)
