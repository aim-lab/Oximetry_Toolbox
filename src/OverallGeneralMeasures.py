import numpy as np
import warnings

from src._ErrorHandler import _check_shape_, _check_window_delta_
from src._ResultsClasses import OverallGeneralMeasuresResult


class OverallGeneralMeasures:
    """
    Class that calculates Overall General Features from spo2 time series.
    Suppose that the data has been preprocessed.

    :param
        signal: 1-d array, of shape (N,) where N is the length of the signal
        ZC_Baseline: Baseline for calculating number of zero-crossing points.
        percentile: Percentile to perform. For example, for percentile 1, the argument should be 1
        M_Threshold: Percentage of the signal M_Threshold % below median oxygen saturation. Typically use 1,2 or 5

    :return:
        OveralGeneralMeasuresResult class containing the following features:
            -	AV: Average of the signal.
            -	MED: Median of the signal.
            -	Min: Minimum value of the signal.
            -	SD: Std of the signal.
            -	RG: SpO2 range (difference between the max and min value).
            -	P: percentile.
            -	M: Percentage of the signal x% below median oxygen saturation.
            -	ZC: Number of zero-crossing points.
            -	DI: Delta Index.
    """

    def __init__(self, ZC_Baseline=None, percentile=1, M_Threshold=2, DI_Window=12):
        self.ZC_Baseline = ZC_Baseline
        self.percentile = percentile
        self.M_Threshold = M_Threshold
        self.DI_Window = DI_Window

    def compute(self, signal) -> OverallGeneralMeasuresResult:
        _check_shape_(signal)

        if self.ZC_Baseline is None:
            self.ZC_Baseline = np.nanmean(signal)

        return OverallGeneralMeasuresResult(np.nanmean(signal), np.nanmedian(signal), np.nanmin(signal),
                                            np.nanstd(signal),
                                            self._ComputeRange_(signal),
                                            self._ApplyPercentile_(signal),
                                            self._BelowMedian_(signal),
                                            self._NumZC_(signal),
                                            self._DeltaIndex_(signal))

    def _ApplyPercentile_(self, signal):
        return np.nanpercentile(signal, self.percentile)

    def _BelowMedian_(self, signal):
        baseline = np.nanmedian(signal) - self.M_Threshold
        with np.errstate(invalid='ignore'):
            return 100 * (np.nansum(signal < baseline) / len(signal))

    def _ComputeRange_(self, signal):
        return np.nanmax(signal) - np.nanmin(signal)

    def _NumZC_(self, signal):
        numZC_count = 0
        baseline = self.ZC_Baseline
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

    def _DeltaIndex_(self, signal):
        _check_window_delta_(len(signal), self.DI_Window)

        signal_splitted = [signal[i:i + self.DI_Window] for i in range(0, len(signal), self.DI_Window)]
        if len(signal_splitted[-1]) != self.DI_Window:
            signal_splitted.pop()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean_window = np.nanmean(signal_splitted, axis=1)
        diff = abs(mean_window - np.roll(mean_window, 1))
        return np.nanmean(diff[1:])
