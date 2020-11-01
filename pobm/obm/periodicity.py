import numpy as np
from scipy.signal import hamming, welch
import warnings

from pobm._ErrorHandler import _check_shape_, _check_fragment_PRSA_, WrongParameter
from pobm._ResultsClasses import PRSAResults, PSDResults


class PRSAMeasures:
    """
    Function that calculates PRSA Features from spo2 time series.
    The method compute runs all the biomarker of this category.
    """

    def __init__(self, PRSA_Window: int = 10, K_AC: int = 2):
        """

        :param PRSA_Window: Fragment duration of PRSA.
        :type PRSA_Window: int, optional
        :param K_AC: Number of values to shift when computing autocorrelation
        :type K_AC: int, optional
        """

        if PRSA_Window <= 0:
            raise WrongParameter("DI_Window should be strictly positive")

        self.PRSA_Window = PRSA_Window
        self.K_AC = K_AC

    def compute(self, signal) -> PRSAResults:
        """
        :param signal: 1-d array, of shape (N,) where N is the length of the signal
        :return: PRSAResults class containing the following features:
                -	PRSAc: PRSA capacity.
                -	PRSAad: PRSA amplitude difference.
                -	PRSAos: PRSA overall slope.
                -	PRSAsb: PRSA slope before the anchor point.
                -	PRSAsa: PRSA slope after the anchor point.
                -	AC: Autocorrelation.

        Example:
        
        .. code-block:: python
        
            from pobm.obm.periodicity import PRSAMeasures

            # Initialize the class with the desired parameters
            prsa_class = PRSAMeasures(PRSA_Window=10, K_AC=2)
        
            # Compute the biomarkers
            results_PRSA = prsa_class.compute(spo2_signal)

        """
        _check_shape_(signal)

        warnings.filterwarnings("ignore", category=RuntimeWarning)
        d = self.PRSA_Window
        _check_fragment_PRSA_(d)
        anchor_points = []
        anchor_found = False

        for i in range(len(signal)):
            if i < d:
                continue
            if len(signal) - i < d:
                continue
            if signal[i - 1] > signal[i]:
                anchor_found = True
                anchor_points.append(signal[i - d:i + d])

        if anchor_found is False:
            PRSA_features = PRSAResults(0, 0, 0, 0, 0, np.correlate(signal, signal, "same")[self.K_AC])
        else:
            anchor_points = np.array(anchor_points)
            windows = np.zeros(2 * d)

            for i in range(2 * d):
                windows[i] = np.nansum(anchor_points[:, i]) / len(anchor_points)

            PRSA_features = PRSAResults((windows[d] + windows[d + 1] - windows[d - 1] - windows[d - 2]) / 4,
                                        np.nanmax(windows) - np.nanmin(windows),
                                        np.polyfit(range(2 * d), windows, 1)[0],
                                        np.polyfit(range(d), windows[0:d], 1)[0],
                                        np.polyfit(range(d), windows[d:], 1)[0],
                                        np.correlate(windows, windows, "same")[self.K_AC])

        return PRSA_features


class PSDMeasures:
    """
    Function that calculates PSD Features from spo2 time series.
    The method compute runs all the biomarker of this category.
    """

    def compute(self, signal) -> PSDResults:
        """
        :param signal: The SpO2 signal, of shape (N,)

        :return: PSDResults class containing the following features:
                -   PSD_total: The amplitude of the spectral signal.
                -   PSD_band: The amplitude of the signal multiplied by a band-pass filter between 0.014 and 0.033 Hz.
                -   PSD_ratio: The ratio between PSD_total and PSD_band.
                -   PDS_peak: The max value of the FFT into the band 0.014-0.033 Hz.


        Example:
        
        .. code-block:: python

            from pobm.obm.periodicity import PSDMeasures

            # Initialize the class with the desired parameters
            psd_class = PSDMeasures()
            
            # Compute the biomarkers
            results_PSD = psd_class.compute(spo2_signal)

        """
        _check_shape_(signal)

        signal = np.array(signal)
        signal = signal[np.logical_not(np.isnan(signal))]

        freq, signal_fft = self.__get_psd(signal)
        amplitude_signal = np.sqrt((signal_fft.real ** 2) + (signal_fft.imag ** 2))

        # taking only positive frequencies, since the signal is real.
        freq = freq[0:int(len(freq) / 2)]
        amplitude_signal = amplitude_signal[0:int(len(amplitude_signal) / 2)]

        # Taking the spectral signal in the relevant band
        amplitude_bp = self.__get_bandpass(amplitude_signal, freq, 0.014, 0.033)

        return PSDResults(np.nansum(amplitude_signal), np.nansum(amplitude_bp),
                          np.nansum(amplitude_bp) / np.nansum(amplitude_signal),
                          np.nanmax(amplitude_bp))

    def __get_bandpass(self, signal, freq, lower_f, higher_f):
        """
        Helper function, to get the amplitude within the desired band.

        :param signal: The amplitude signal, of shape (L,)
        :param freq: Array of frequencies, of shape (L,)
        :param lower_f: The lower frequency of the band
        :param higher_f: The higher frequency of the band
        :return: The amplitude within the band
        """
        amplitude_bp = signal[lower_f < freq]
        freq_bp = freq[lower_f < freq]

        amplitude_bp = amplitude_bp[freq_bp < higher_f]
        return amplitude_bp

    def __get_psd(self, signal):
        """
        Helper function, compute the PSD
        
        :param signal: The SpO2 signal, of shape (N,)
        :return:
        freq: array of frequencies, of shape (L,)
        signal_fft: The PSD of the signal, of shape (L,)
        """
        # N = len(signal)
        # w = hamming(N)
        # signal_fft = np.fft.fft(signal * w) / N
        # freq = np.fft.fftfreq(signal.shape[-1])

        freq, signal_fft = welch(signal, window="hamming")
        signal_fft = signal_fft / len(signal)

        return freq, signal_fft
