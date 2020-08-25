import numpy as np
import warnings

from _spo2._ErrorHandler import _check_window_delta_


def _ApplyPercentile_(signal, num):
    return np.nanpercentile(signal, num)


def _BelowMedian_(signal, threshold):
    baseline = np.nanmedian(signal) - threshold
    with np.errstate(invalid='ignore'):
        return 100 * (np.nansum(signal < baseline) / len(signal))


def _ComputeRange_(signal):
    return np.nanmax(signal) - np.nanmin(signal)


def _NumZC_(signal, baseline):
    numZC_count = 0
    for idx_signal in range(2, len(signal) - 1):
        if signal[idx_signal] == baseline:
            if (signal[idx_signal - 1] <= baseline) & (signal[idx_signal + 1] >= baseline):
                numZC_count += 1
            if (signal[idx_signal - 1] >= baseline) & (signal[idx_signal + 1] <= baseline):
                numZC_count += 1
        if (signal[idx_signal - 1] < baseline) & (signal[idx_signal] > baseline):
            numZC_count += 1
        if (signal[idx_signal - 1] > baseline) & (signal[idx_signal] < baseline):
            numZC_count += 1
    return numZC_count


def _DeltaIndex_(signal, window_size):
    _check_window_delta_(len(signal), window_size)

    signal_splitted = [signal[i:i + window_size] for i in range(0, len(signal), window_size)]
    if len(signal_splitted[-1]) != window_size:
        signal_splitted.pop()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean_window = np.nanmean(signal_splitted, axis=1)
    diff = abs(mean_window - np.roll(mean_window, 1))
    return np.nanmean(diff[1:])
