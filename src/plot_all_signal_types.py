import matplotlib.pyplot as plt
import numpy as np

from common.constants import SamplingRate
from common.signal_types import (
    band_limited_noise,
    brownian_noise,
    delta_function,
    exponential_sweep,
    impulse_response,
    log_sine_sweep,
    pink_noise,
    sawtooth_wave,
    sine_wave,
    triangular_wave,
    unit_step_function,
    white_noise,
)

if __name__ == "__main__":
    sampling_rate = SamplingRate.FS_48K_HZ  # Samples per second
    duration = 1.0  # Duration in seconds
    frequency = 5.0  # Frequency in Hz

    signals = {
        "Sine Wave": sine_wave(sampling_rate, duration, frequency),
        "Log sine sweep": log_sine_sweep(sampling_rate, duration, 1, 1000),
        "Exponential Sweep": exponential_sweep(sampling_rate, duration, 1, 1000),
        "Sawtooth Wave": sawtooth_wave(sampling_rate, duration, frequency),
        "Triangular Wave": triangular_wave(sampling_rate, duration, frequency),
        "White Noise": white_noise(sampling_rate, duration),
        "Pink Noise": pink_noise(sampling_rate, duration),
        "Brownian Noise": brownian_noise(sampling_rate, duration),
        "Band-limited Noise": band_limited_noise(sampling_rate, duration, 100, 1000),
        "Unit Step Function": unit_step_function(sampling_rate, duration),
        "Impulse Response": impulse_response(sampling_rate, duration, 0.1),
        "Delta Function": delta_function(sampling_rate, duration),
    }

    time = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)

    plt.figure(figsize=(12, 8))
    for i, (title, signal) in enumerate(signals.items(), 1):
        plt.subplot(4, 3, i)
        plt.plot(time, signal)
        plt.title(title)
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")
        plt.grid()

    plt.tight_layout()
    plt.show()
