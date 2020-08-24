import numpy as np

from spo2.ErrorHandler import check_window_delta


def ApplyPercentile(signal, num):
    return np.percentile(signal, num)


def BelowMedian_(signal, threshold):
    baseline = np.median(signal) - threshold
    return 100 * (np.sum(signal < baseline) / len(signal))


def NumZC_(signal, baseline):
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


def DeltaIndex_(signal, window_size):
    check_window_delta(len(signal), window_size)

    signal_splitted = [signal[i:i + window_size] for i in range(0, len(signal), window_size)]
    if len(signal_splitted[-1]) != window_size:
        signal_splitted.pop()
    mean_window = np.mean(signal_splitted, axis=1)
    diff = abs(mean_window - np.roll(mean_window, 1))
    return np.mean(diff[1:])
