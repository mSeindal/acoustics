import matplotlib.pyplot as plt

from common.constants import SamplingRate, Unit
from common.signal_processing import fft_on_signal
from common.signal_types import sine_wave

if __name__ == "__main__":
    sampling_rate = SamplingRate.FS_48K_HZ  # Samples per second
    duration = 5.0  # Duration in seconds

    signal = sine_wave(sampling_rate, duration, frequency=440.0)

    magnitude, frequencies = fft_on_signal(signal, sampling_rate)

    plt.figure(figsize=(10, 4))
    plt.semilogx(frequencies, magnitude)
    plt.grid(which="both", linestyle="--", linewidth=0.5)
    plt.title("FFT of Sine Wave (440 Hz)")
    plt.xlabel(f"Frequencies [{Unit.HZ.value}]")
    plt.ylabel(f"Amplitude [{Unit.DB.value}]")
    plt.show()
