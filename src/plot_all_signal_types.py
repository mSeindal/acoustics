import matplotlib.pyplot as plt
import numpy as np
from common.signal_types import (
    sine_wave,
    sawtooth_wave,
    white_noise,
    pink_noise,
    brownian_noise,
    unit_step_function,
    log_sine_sweep,
    impulse_response,
    delta_function,
    band_limited_noise,
    triangular_wave,
    exponential_sweep,
)
from common.constants import SamplingRate


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
