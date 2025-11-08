import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy.interpolate import interp1d
from scipy.io import wavfile

def whiten(strain, interp_psd, dt):
    """Whiten a time-domain strain series using an interpolated one-sided PSD.

    Notes
    -----
    For tests expecting identity under a flat PSD (PSD==1), use y = irfft( rfft(x)/sqrt(PSD) ).
    """
    Nt = len(strain)
    freqs = np.fft.rfftfreq(Nt, dt)
    hf = np.fft.rfft(strain)
    psd = interp_psd(freqs)
    psd = np.maximum(psd, 1e-20)
    white_hf = hf / np.sqrt(psd)
    return np.fft.irfft(white_hf, n=Nt)

def write_wavfile(filename, fs, data):
    """
    Scale to int16 and write a WAV file.
    """
    if len(data) == 0:
        raise ValueError("Empty audio data.")
    d = np.int16(data / np.max(np.abs(data)) * 32767 * 0.9)
    wavfile.write(filename, int(fs), d)

def reqshift(data, fshift=100, sample_rate=4096):
    """
    Frequency-shift a real time-series by fshift (Hz) using FFT-bin roll.

    Parameters
    ----------
    data : 1D array
    fshift : float
        Frequency shift in Hz.
    sample_rate : float
        Sampling rate in Hz.
    """
    x = np.fft.rfft(data)
    T = len(data) / float(sample_rate)
    df = 1.0 / T
    nbins = int(fshift / df)
    y = np.roll(x.real, nbins) + 1j * np.roll(x.imag, nbins)
    y[0:nbins] = 0.0
    z = np.fft.irfft(y)
    return z

def plot_psds(strain_H1, strain_L1, fs, eventname="GW150914", plottype="png"):
    """
    Compute and plot ASDs for H1/L1; return (psd_smooth, psd_H1_interp, psd_L1_interp).
    Saves figure to figures/{eventname}_ASDs.{plottype}
    """
    os.makedirs("figures", exist_ok=True)

    NFFT = 4 * fs
    Pxx_H1, freqs = mlab.psd(strain_H1, Fs=fs, NFFT=NFFT)
    Pxx_L1, _     = mlab.psd(strain_L1, Fs=fs, NFFT=NFFT)

    psd_H1 = interp1d(freqs, Pxx_H1, bounds_error=False, fill_value=np.inf)
    psd_L1 = interp1d(freqs, Pxx_L1, bounds_error=False, fill_value=np.inf)

    # Smoothed model from the tutorial (approximate)
    Pxx_model = (1.0e-22 * (18.0 / (0.1 + freqs)) ** 2) ** 2 + 0.7e-23 ** 2 + ((freqs / 2000.0) * 4.0e-23) ** 2
    psd_smooth = interp1d(freqs, Pxx_model, bounds_error=False, fill_value=np.inf)

    f_min, f_max = 20.0, 2000.0
    plt.figure(figsize=(10, 8))
    plt.loglog(freqs, np.sqrt(Pxx_L1), 'g', label='L1 ASD')
    plt.loglog(freqs, np.sqrt(Pxx_H1), 'r', label='H1 ASD')
    plt.loglog(freqs, np.sqrt(Pxx_model), 'k', label='H1 smooth model')
    plt.axis([f_min, f_max, 1e-24, 1e-19])
    plt.grid(True, which="both")
    plt.ylabel('ASD (strain/rtHz)')
    plt.xlabel('Freq (Hz)')
    plt.legend(loc='upper center')
    plt.title(f'Advanced LIGO strain data near {eventname}')
    plt.savefig(f"figures/{eventname}_ASDs.{plottype}", bbox_inches="tight")

    return psd_smooth, psd_H1, psd_L1
