import numpy as np
from lempel_ziv_complexity import lempel_ziv_complexity
from scipy import integrate, stats
import warnings

from pobm._ErrorHandler import _check_shape_, _check_len_ApEn_
from pobm._ResultsClasses import ComplexityMeasuresResults


class ComplexityMeasures:
    """
    Class that calculates Complexity Features from spo2 time series.
    Suppose that the data has been preprocessed.

    :param CTM_Threshold: Radius of Central Tendency Measure.
    :type CTM_Threshold: optional
    :param DFA_Window: Length of window to calculate DFA biomarker.
    :type DFA_Window: optional
    :param M_Sampen: Embedding dimension to compute SampEn.
    :type M_Sampen: optional
    :param R_Sampen: Tolerance to compute SampEn.
    :type R_Sampen: optional
    :param M_ApEn: Embedding dimension to compute ApEn.
    :type M_ApEn: optional
    :param R_ApEn: Tolerance to compute ApEn.
    :type R_ApEn: optional


    PhysioZoo OBM toolbox 2020, version 1.0
    Released under the GNU General Public License

    Authors: Jeremy Levy and Joachim A. Behar
    The Technion Artificial Intelligence in Medicine Laboratory (AIMLab.)
    https://aim-lab.github.io/

    This program is free software; you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by the
    Free Software Foundation; either version 2 of the License, or (at your
    option) any later version.
    This program is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
    Public License for more details.
    """

    def __init__(self, CTM_Threshold=0.25, DFA_Window=20, M_Sampen=3, R_Sampen=0.2, M_ApEn=2, R_ApEn=0.25):
        self.CTM_Threshold = CTM_Threshold
        self.DFA_Window = DFA_Window
        self.M_Sampen = M_Sampen
        self.R_Sampen = R_Sampen
        self.M_ApEn = M_ApEn
        self.R_ApEn = R_ApEn

    def compute(self, signal) -> ComplexityMeasuresResults:
        """

        :param signal: 1-d array, of shape (N,) where N is the length of the signal
        :return: ComplexityMeasuresResults class containing the following features:
                -	ApEn: Approximate Entropy.
                -   LZ: Lempel-Ziv complexity.
                -	CTM: Central Tendency Measure.
                -   SampEn: Sample Entropy.
                -	DFA: Detrended Fluctuation Analysis.


        Example:
        
        .. code-block:: python

            from pobm.obm.complex import ComplexityMeasures

            # Initialize the class with the desired parameters
            complexity_class = ComplexityMeasures(CTM_Threshold=0.25, DFA_Window=20, M_Sampen=3, R_Sampen=0.2, M_ApEn=2, R_ApEn=0.25)
            
            # Compute the biomarkers
            results_complexity = complexity_class.compute(spo2_signal)


    PhysioZoo OBM toolbox 2020, version 1.0
    Released under the GNU General Public License

    Authors: Jeremy Levy and Joachim A. Behar
    The Technion Artificial Intelligence in Medicine Laboratory (AIMLab.)
    https://aim-lab.github.io/

    This program is free software; you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by the
    Free Software Foundation; either version 2 of the License, or (at your
    option) any later version.
    This program is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
    Public License for more details.
        """
        _check_shape_(signal)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        return ComplexityMeasuresResults(self.__comp_apen(signal), self.__comp_lz(signal),
                                         self.__comp_ctm(signal),
                                         self.__comp_sampen(signal),
                                         self.__comp_dfa(signal))

    def __comp_apen(self, signal):
        """
        Compute the approximate entropy, according to the paper
        Utility of Approximate Entropy From Overnight Pulse Oximetry Data in the Diagnosis
        of the Obstructive Sleep Apnea Syndrome
        :param signal: 1-d array, of shape (N,) where N is the length of the signal
        :return: ApEn
        """
        signal = np.array(signal)
        signal = signal[np.logical_not(np.isnan(signal))]

        phi_m = self.__apen(self.M_ApEn, signal)
        phi_m1 = self.__apen(self.M_ApEn + 1, signal)
        with np.errstate(invalid='ignore'):
            res = phi_m - phi_m1

        return res

    def __apen(self, m, signal):
        """
        Help function of CompApEn
        """
        N = len(signal)
        r = self.R_ApEn
        _check_len_ApEn_(N, m)
        C = np.zeros(shape=(N - m + 1))

        res = [signal[i:i + m] for i in range(0, N - m + 1)]
        for i in range(0, N - m + 1):
            C[i] = np.nansum(self.__dist(res[i], res, r)) / (N - m + 1)

        phi_m = np.nansum(np.log(C)) / (N - m + 1)
        return phi_m

    def __dist(self, window1, window2, r):
        """
        Help function of CompApEn
        """
        window1 = np.array(window1)
        window2 = np.array(window2)

        with np.errstate(invalid='ignore'):
            return np.nanmax(abs(window1 - window2), axis=1) < r

    def __comp_lz(self, signal):
        """
        Compute lempel-ziv, according to the paper
        Non-linear characteristics of blood oxygen saturation from nocturnal oximetry
        for obstructive sleep apnoea detection
        :param signal: 1-d array, of shape (N,) where N is the length of the signal
        :return: LZ
        """
        median = np.median(signal)
        res = [signal[i] > median for i in range(0, len(signal))]
        byte = [str(int(b is True)) for b in res]
        return lempel_ziv_complexity(''.join(byte))

    def __comp_ctm(self, signal):
        """
        Compute CTM, according to the paper
        Non-linear characteristics of blood oxygen saturation from nocturnal oximetry
        for obstructive sleep apnoea detection
        :param signal: 1-d array, of shape (N,) where N is the length of the signal
        :return: CTM
        """
        res = 0
        for i in range(len(signal) - 2):
            res += self.__d_ctm(i, self.CTM_Threshold, signal)
        return res / (len(signal) - 2)

    def __d_ctm(self, i, p, signal):
        """
        Help function of CompCTM
        """
        if np.sqrt(((signal[i + 2] - signal[i + 1]) ** 2) + ((signal[i + 1] - signal[i]) ** 2)) < p:
            return 1
        return 0

    def __comp_sampen(self, signal):
        """
        Compute the Sample Entropy
        :param signal: 1-d array, of shape (N,) where N is the length of the signal
        :return: SampEn
        """
        N = len(signal)
        m = self.M_Sampen
        r = self.R_Sampen

        # Split time series and save all templates of length m
        xmi = np.array([signal[i: i + m] for i in range(N - m)])
        xmj = np.array([signal[i: i + m] for i in range(N - m + 1)])

        # Save all matches minus the self-match, compute B
        with np.errstate(invalid='ignore'):
            B = np.sum([np.sum(np.abs(xmii - xmj).max(axis=1) <= r) - 1 for xmii in xmi])

        # Similar for computing A
        m += 1
        xm = np.array([signal[i: i + m] for i in range(N - m + 1)])

        with np.errstate(invalid='ignore'):
            A = np.sum([np.sum(np.abs(xmi - xm).max(axis=1) <= r) - 1 for xmi in xm])

        # Return SampEn
        return -np.log(A / B)

    def __comp_dfa(self, signal):
        """
        Compute DFA
        :param signal: 1-d array, of shape (N,) where N is the length of the signal
        :return: DFA
        """
        n = self.DFA_Window
        y = integrate.cumtrapz(signal - np.nanmean(signal))
        least_square = np.zeros(len(y))

        i = 0
        while i < len(y):
            if i + n > len(y):
                n = len(y) - i
            if n == 1:
                least_square[-1] = y[-1]
                break
            x = np.array(range(0, n))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                slope, intercept, _, _, _ = stats.linregress(x, y[i:i + n])
            least_square[i:i + n] = slope * x + intercept
            i += n

        return np.sqrt(np.nansum((y - least_square) ** 2) / len(y))
