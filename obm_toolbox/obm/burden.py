import numpy as np

from obm_toolbox.obm.desat import DesaturationsMeasures
from obm_toolbox._ErrorHandler import _check_shape_
from obm_toolbox._ResultsClasses import HypoxicBurdenMeasuresResults


class HypoxicBurdenMeasures:
    """
    Class that calculates Hypoxic Burden Features from spo2 time series.
    Suppose that the data has been preprocessed.

    :param begin: List of indices of beginning of each desaturation event.
    :param end: List of indices of end of each desaturation event.
    :param CT_Threshold: Percentage of the time spent below the “CT_Threshold” % oxygen saturation level.
    :param CA_Baseline: Baseline to compute the CA feature. Default value is mean of the signal.

    """

    def __init__(self, begin, end, CT_Threshold=90, CA_Baseline=None):
        self.begin = begin
        self.end = end
        self.CT_Threshold = CT_Threshold
        self.CA_Baseline = CA_Baseline

    def compute(self, signal):
        """
        :param signal: 1-d array, of shape (N,) where N is the length of the signal
        :return:
            HypoxicBurdenMeasuresResults class containing the following features:
                -	CA: Integral SpO2 below the xx SpO2 level normalized by the total recording time
                -   CT: Percentage of the time spent below the xx% oxygen saturation level
                -   POD: Percentage of oxygen desaturation events
                -   AODmax: The area under the oxygen desaturation event curve, using the maximum SpO2 value as baseline
                    and normalized by the total recording time
                -   AOD100: Cumulative area of desaturations under the 100% SpO2 level as baseline and normalized
                    by the total recording time
        """

        _check_shape_(signal)

        desaturations = {'begin': self.begin, 'end': self.end}

        if self.CA_Baseline is None:
            self.CA_Baseline = np.nanmean(signal)

        return self.__comp_hypoxic(signal, desaturations)

    def __comp_hypoxic(self, signal, desaturations_signal):
        """
        Helper function, to calculate the Hypoxic Burden biomarkers from the desaturations

        :param signal: 1-d array, of shape (N,) where N is the length of the signal
        :param desaturations_signal: dict with 2 keys:
            -   begin: indices of begininning of each desaturation
            -   end: indices of end of each desaturation
        :return:
            HypoxicBurdenMeasuresResults class containing the following features:
                -	CA: Integral SpO2 below the xx SpO2 level normalized by the total recording time
                -   CT: Percentage of the time spent below the xx% oxygen saturation level
                -   POD: Percentage of oxygen desaturation events
                -   AODmax: The area under the oxygen desaturation event curve, using the maximum SpO2 value as baseline
                    and normalized by the total recording time
                -   AOD100: Cumulative area of desaturations under the 100% SpO2 level as baseline and normalized
                    by the total recording time
        """
        desat_class = DesaturationsMeasures(desaturations_signal['begin'], desaturations_signal['end'])

        desaturations, desaturation_valid, desaturation_length_all, desaturation_int_100_all, \
        desaturation_int_max_all, desaturation_depth_100_all, desaturation_depth_max_all, \
        desaturation_slope_all = desat_class.desat_embedding()

        time_spo2_array = np.array(range(len(signal)))
        for (i, desaturation) in enumerate(desaturations):
            desaturation_idx = (time_spo2_array >= desaturation['Start']) & (time_spo2_array <= desaturation['End'])

            if np.sum(desaturation_idx) == 0:
                continue

            signal = np.array(signal)

            desaturation_spo2 = signal[desaturation_idx]
            desaturation_max = np.nanmax(desaturation_spo2)

            desaturation_valid[i] = True
            desaturation_length_all[i] = desaturation['Duration']
            desaturation_int_100_all[i] = np.nansum(100 - desaturation_spo2)
            desaturation_int_max_all[i] = np.nansum(desaturation_max - desaturation_spo2)

        desaturation_features = HypoxicBurdenMeasuresResults(self.__comp_ca(signal),
                                                             self.__comp_ct(signal),
                                                             0.0, 0.0, 0.0)
        if np.sum(desaturation_valid) != 0:
            desaturation_features.POD = np.nansum(desaturation_length_all[desaturation_valid]) / len(signal)
            desaturation_features.AODmax = np.nansum(desaturation_int_max_all[desaturation_valid]) / len(signal)
            desaturation_features.AOD100 = np.nansum(desaturation_int_100_all[desaturation_valid]) / len(signal)

        return desaturation_features

    def __comp_ca(self, signal):
        """
        Compute the cumulative area biomarker

        :param signal: 1-d array, of shape (N,) where N is the length of the signal
        :return: CA, the cumulative area (float)
        """
        with np.errstate(invalid='ignore'):
            signal_under_baseline = signal[signal < self.CA_Baseline]
        if len(signal_under_baseline) == 0:
            return 0.0
        # return integrate.cumtrapz(signal_under_baseline) / len(signal)
        return sum(signal_under_baseline) / len(signal)

    def __comp_ct(self, signal):
        """
        Compute the cumulative time biomarker

        :param signal: 1-d array, of shape (N,) where N is the length of the signal
        :return: CT, the cumulative time (float)
        """
        with np.errstate(invalid='ignore'):
            return 100 * len(signal[signal <= self.CT_Threshold]) / len(signal)
