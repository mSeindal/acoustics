import numpy as np


def fft_on_signal(signal: np.ndarray, sampling_rate: float) -> np.ndarray:
    """Compute the FFT of a signal and return magnitude in dB and frequency bins."""
    fft = np.fft.fft(signal)

    # Calculate magnitude in dB, avoiding log of zero by adding a small epsilon
    magnitude = np.abs(fft)
    magnitude_db = 20 * np.log10(magnitude + 1e-12)

    # Calculate frequency bins
    # Frequencies are calculated as the index divided by the total length, scaled by the
    # sampling rate. The result is a frequency bin for each FFT component
    frequencies = np.arange(len(signal)) / (len(signal) / sampling_rate)

    return magnitude_db, frequencies
