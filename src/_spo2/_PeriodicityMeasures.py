import numpy as np
from scipy.signal import hamming, welch

from _spo2._ErrorHandler import _check_fragment_PRSA_
from _spo2._ResultsClasses import PRSAResults, PSDResults


def _PRSAFeatures_(signal, d, K_AC=2):
    _check_fragment_PRSA_(d)
    anchor_points = []
    for i in range(len(signal)):
        if i < d:
            continue
        if len(signal) - i < d:
            continue
        if signal[i - 1] > signal[i]:
            anchor_points.append(signal[i - d:i + d])
    anchor_points = np.array(anchor_points)
    windows = np.zeros(2 * d)

    for i in range(2 * d):
        windows[i] = np.nansum(anchor_points[:, i]) / len(anchor_points)

    PRSA_features = PRSAResults((windows[d] + windows[d + 1] - windows[d - 1] - windows[d - 2]) / 4,
                                np.nanmax(windows) - np.nanmin(windows),
                                np.polyfit(range(2 * d), windows, 1)[0], np.polyfit(range(d), windows[0:d], 1)[0],
                                np.polyfit(range(d), windows[d:], 1)[0], np.correlate(windows, windows, "same")[K_AC])

    return PRSA_features


def _SpectralAnalysis_(signal):
    signal = np.array(signal)
    signal = signal[np.logical_not(np.isnan(signal))]

    freq, signal_fft = _get_PSD_(signal)
    amplitude_signal = np.sqrt((signal_fft.real ** 2) + (signal_fft.imag ** 2))

    # taking only positive frequencies, since the signal is real.
    freq = freq[0:int(len(freq) / 2)]
    amplitude_signal = amplitude_signal[0:int(len(amplitude_signal) / 2)]

    # Taking the spectral signal in the relevant band
    amplitude_bp = _get_bandpass_(amplitude_signal, freq, 0.014, 0.033)

    return PSDResults(np.nansum(amplitude_signal), np.nansum(amplitude_bp), np.nansum(amplitude_bp) / np.nansum(amplitude_signal),
                      np.nanmax(amplitude_bp))


def _get_bandpass_(signal, freq, lower_f, higher_f):
    amplitude_bp = signal[lower_f < freq]
    freq_bp = freq[lower_f < freq]

    amplitude_bp = amplitude_bp[freq_bp < higher_f]
    return amplitude_bp


def _get_PSD_(signal):
    # N = len(signal)
    # w = hamming(N)
    # signal_fft = np.fft.fft(signal * w) / N
    # freq = np.fft.fftfreq(signal.shape[-1])

    freq, signal_fft = welch(signal, window="hamming")
    signal_fft = signal_fft / len(signal)

    return freq, signal_fft
