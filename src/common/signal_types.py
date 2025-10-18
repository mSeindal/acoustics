import numpy as np
from scipy.signal import chirp


def sine_tone(sampling_rate: float, duration: float, frequency: float) -> np.ndarray:
    t = np.linspace(0, duration, int(sampling_rate * duration))
    return np.sin(2 * np.pi * frequency * t)


def log_sine_sweep(
    sampling_rate: float, duration: float, start_frq: float, end_frq: float
) -> np.ndarray:
    t = np.linspace(0, duration, int(sampling_rate * duration))
    sweep = chirp(t, f0=start_frq, f1=end_frq, t1=duration, method="logarithmic")
    return sweep


def white_noise(sampling_rate: float, duration: float) -> np.ndarray:
    num_samples = sampling_rate * duration
    noise = np.random.randn(num_samples)
    noise /= np.max(np.abs(noise))  # Normalize to -1 to 1
    return noise


def pink_noise(sampling_rate: float, duration: float) -> np.ndarray:
    """Generate pink noise using the Voss-McCartney algorithm."""
    num_samples = sampling_rate * duration
    num_rows = 16
    array = np.zeros((num_rows, num_samples))
    array[0, :] = np.random.rand(num_samples)

    for i in range(1, num_rows):
        step = 2**i
        for j in range(0, num_samples, step):
            array[i, j : j + step] = np.random.rand()

    pink = np.sum(array, axis=0)
    pink /= np.max(np.abs(pink))  # Normalize to -1 to 1
    return pink


def brownian_noise(sampling_rate: float, duration: float) -> np.ndarray:
    num_samples = sampling_rate * duration
    white = np.random.randn(num_samples)
    brown = np.cumsum(white)
    brown /= np.max(np.abs(brown))  # Normalize to -1 to 1
    return brown


def band_limited_noise(
    sampling_rate: float, duration: float, low_freq: float, high_freq: float
) -> np.ndarray:
    num_samples = sampling_rate * duration
    freqs = np.fft.rfftfreq(num_samples, 1 / sampling_rate)
    spectrum = np.zeros(len(freqs), dtype=complex)

    # Create a mask for the desired frequency band
    band_mask = (freqs >= low_freq) & (freqs <= high_freq)
    spectrum[band_mask] = np.random.randn(np.sum(band_mask)) + 1j * np.random.randn(
        np.sum(band_mask)
    )

    noise = np.fft.irfft(spectrum, n=num_samples)
    noise /= np.max(np.abs(noise))  # Normalize to -1 to 1
    return noise


def impulse_response(sampling_rate: float, duration: float, delay: float) -> np.ndarray:
    num_samples = int(sampling_rate * duration)
    ir = np.zeros(num_samples)
    delay_samples = int(sampling_rate * delay)

    if delay_samples < num_samples:
        ir[delay_samples] = 1.0

    return ir


def dc_signal(
    sampling_rate: float, duration: float, amplitude: float = 1.0
) -> np.ndarray:
    num_samples = int(sampling_rate * duration)
    return np.full(num_samples, amplitude)


def ramp_signal(
    sampling_rate: float,
    duration: float,
    start_amplitude: float = 0.0,
    end_amplitude: float = 1.0,
) -> np.ndarray:
    num_samples = int(sampling_rate * duration)
    return np.linspace(start_amplitude, end_amplitude, num_samples)


def triangular_wave(
    sampling_rate: float, duration: float, frequency: float
) -> np.ndarray:
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    return 2 * np.abs(2 * (t * frequency - np.floor(0.5 + t * frequency))) - 1


def sawtooth_wave(
    sampling_rate: float, duration: float, frequency: float
) -> np.ndarray:
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    return 2 * (t * frequency - np.floor(0.5 + t * frequency))


def exponential_sweep(
    sampling_rate: float, duration: float, start_frq: float, end_frq: float
) -> np.ndarray:
    t = np.linspace(0, duration, int(sampling_rate * duration))
    K = duration / np.log(end_frq / start_frq)
    L = start_frq * K
    sweep = np.sin(2 * np.pi * L * (np.exp(t / K) - 1))
    return sweep


def rectangular_pulse(
    sampling_rate: float, duration: float, pulse_width: float
) -> np.ndarray:
    num_samples = int(sampling_rate * duration)
    pulse_samples = int(sampling_rate * pulse_width)
    signal = np.zeros(num_samples)
    signal[:pulse_samples] = 1.0
    return signal


def triangular_pulse(
    sampling_rate: float, duration: float, pulse_width: float
) -> np.ndarray:
    num_samples = int(sampling_rate * duration)
    pulse_samples = int(sampling_rate * pulse_width)
    t = np.linspace(0, 1, pulse_samples)
    pulse = 1 - np.abs(2 * t - 1)
    signal = np.zeros(num_samples)
    signal[:pulse_samples] = pulse
    return signal


def gaussian_pulse(
    sampling_rate: float, duration: float, center: float, width: float
) -> np.ndarray:
    num_samples = int(sampling_rate * duration)
    t = np.linspace(0, duration, num_samples)
    pulse = np.exp(-0.5 * ((t - center) / width) ** 2)
    return pulse


def delta_function(sampling_rate: float, duration: float) -> np.ndarray:
    num_samples = int(sampling_rate * duration)
    signal = np.zeros(num_samples)
    signal[0] = 1.0
    return signal


def unit_step_function(sampling_rate: float, duration: float) -> np.ndarray:
    num_samples = int(sampling_rate * duration)
    signal = np.ones(num_samples)
    return signal
