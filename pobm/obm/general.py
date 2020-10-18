import numpy as np
import warnings

from pobm._ErrorHandler import _check_shape_, _check_window_delta_
from pobm._ResultsClasses import OverallGeneralMeasuresResult


class OverallGeneralMeasures:
    """
    Class that calculates Overall General Features from spo2 time series.
    Suppose that the data has been preprocessed.

    :param ZC_Baseline: Baseline for calculating number of zero-crossing points.
    :param percentile: Percentile to perform. For example, for percentile 1, the argument should be 1
    :param M_Threshold: Percentage of the signal M_Threshold % below median oxygen saturation. Typically use 1,2 or 5
    """

    def __init__(self, ZC_Baseline=None, percentile=1, M_Threshold=2, DI_Window=12):
        self.ZC_Baseline = ZC_Baseline
        self.percentile = percentile
        self.M_Threshold = M_Threshold
        self.DI_Window = DI_Window

    def compute(self, signal) -> OverallGeneralMeasuresResult:
        """
        :param signal: 1-d array, of shape (N,) where N is the length of the signal
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

        Example:
        
        .. code-block::

            from pobm.obm.general import OverallGeneralMeasures

            # Initialize the class with the desired parameters
            statistics_class = OverallGeneralMeasures(ZC_Baseline=90, percentile=1, M_Threshold=2, DI_Window=12)
        
            # Compute the biomarkers
            results_statistics = statistics_class.compute(spo2_signal)

        """
        _check_shape_(signal)

        warnings.filterwarnings("ignore", category=RuntimeWarning)
        if self.ZC_Baseline is None:
            self.ZC_Baseline = np.nanmean(signal)

        return OverallGeneralMeasuresResult(np.nanmean(signal), np.nanmedian(signal), np.nanmin(signal),
                                            np.nanstd(signal),
                                            self.__compute_range(signal),
                                            self.__apply_percentile(signal),
                                            self.__below_median(signal),
                                            self.__num_zc(signal),
                                            self.__delta_index(signal))

    def __apply_percentile(self, signal):
        """
        Apply percentile to the spo2 signal

        :param signal: 1-d array, of shape (N,) where N is the length of the signal
        :return: the percentile
        """
        return np.nanpercentile(signal, self.percentile)

    def __below_median(self, signal):
        """
        Compute the below median biomarker from the spo2 signal

        :param signal: 1-d array, of shape (N,) where N is the length of the signal
        :return: the BM biomarker
        """
        baseline = np.nanmedian(signal) - self.M_Threshold
        with np.errstate(invalid='ignore'):
            return 100 * (np.nansum(signal < baseline) / len(signal))

    def __compute_range(self, signal):
        """
        Compute the range biomarker from the spo2 signal

        :param signal: 1-d array, of shape (N,) where N is the length of the signal
        :return: the R biomarker
        """
        return np.nanmax(signal) - np.nanmin(signal)

    def __num_zc(self, signal):
        """
        Compute the numZC biomarker from the spo2 signal

        :param signal: 1-d array, of shape (N,) where N is the length of the signal
        :return: the ZC biomarker
        """
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

    def __delta_index(self, signal):
        """
        Compute the delta index biomarker from the spo2 signal

        :param signal: 1-d array, of shape (N,) where N is the length of the signal
        :return: the DI biomarker
        """
        _check_window_delta_(len(signal), self.DI_Window)

        signal_splitted = [signal[i:i + self.DI_Window] for i in range(0, len(signal), self.DI_Window)]
        if len(signal_splitted[-1]) != self.DI_Window:
            signal_splitted.pop()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean_window = np.nanmean(signal_splitted, axis=1)
        diff = abs(mean_window - np.roll(mean_window, 1))
        return np.nanmean(diff[1:])
