# -*- coding: utf-8 -*-
"""
SmartCare ODI detection toolbox.

@author: Joachim, January 2019
v1.0.0

Sequence: sc_resamp --> sc_median --> sc_desaturations.
"""

import numpy as np
from scipy import signal


def _sc_resamp_(data, fs):
    # this function is used to resample the data at 1Hz. It takes the median spo2 value
    # over each window of length fs so that the resulting output signal is sampled at 1Hz.
    #
    # inputs:
    #   data (int): input spo2 time series
    #   fs (int): sampling frequency of the original time series (Hz)
    #
    # output:
    #   data_out (int): the resampled spo2 time series at 1Hz
    #
    # Assumption: any missing/abnormal values are represented as 'np.nan'

    len_in = len(data)
    len_out = round(len_in / fs)
    data_out = []
    for jj in range(len_out):
        data_out = np.append(data_out, np.median(data[jj * fs:(jj + 1) * fs]))

    return data_out


def _sc_median_(data, medfilt_lg=9):
    # median filter used to smooth the spo2 time series and avoid sporadic
    # increase/decrease of spo2 which could affect the detection of the desaturations.
    #
    # input:
    #   data (int): input spo2 time series (!!assumed to be sampled at 1Hz!!)
    #   medfilt_lg (int): median filter length
    #
    # output:
    #   data_med (int): median filtered spo2 time series.
    #
    # Assumption: any missing/abnormal values are represented as 'np.nan'

    data_med = signal.medfilt(np.round(data), medfilt_lg)

    return data_med


def _sc_desaturations_(data, thres=2):
    # this function implements the algorithm of:
    #
    #   Hwang, Su Hwan, et al. "Real-time automatic apneic event detection using nocturnal pulse oximetry."
    #   IEEE Transactions on Biomedical Engineering 65.3 (2018): 706-712.
    #
    # NOTE: The original function search desaturations that are minimum 10 seconds long and maximum 90 seconds long.
    # In addition the original algorithm actually looked to me more like an estimate of the ODI4 than ODI3. This
    # implementation is updated to allow the estimation of ODI3 and allows desaturations that are up to 120 seconds
    # based on some of our observations. In addition, some conditions were added to avoid becoming blocked in
    # infinite while loops.
    #
    # Important: The algorithm assumes a sampling rate of 1Hz and a quantization of 1% to the input data.
    #
    # inputs: data (int): spo2 time series sampled at 1Hz and with a quantization of 1%. thres (int): desaturation
    # threshold below 'a' point (default 2%). IMPORTANT NOTE: 2% below 'a' corresponds to a 3% desaturation.
    #
    # outputs:
    #   desat (int): number of desaturations
    #   table_desat_aa (float): location of the aa feature points
    #   table_desat_bb (float): location of the bb feature points
    #   table_desat_cc (float): location of the cc feature points

    aa = 1
    desat = 0
    max_desat_lg = 120  # was 90 sec in the original paper. Changed to 120 because I have seen longer desaturations.
    lg_dat = len(data)
    table_desat_aa = []
    table_desat_bb = []
    table_desat_cc = []

    while aa < lg_dat:
        # added condition to test that between aa and the end of the recording there is at least 10 seconds
        if aa + 10 > lg_dat:
            return desat, table_desat_aa, table_desat_bb, table_desat_cc

        if data[aa] > 25 and -1 >= (data[aa] - data[aa - 1]) >= -thres:
            bb = aa + 1
            out_b = 0

            while bb < lg_dat and out_b == 0:
                if bb == lg_dat - 1:  # added this condition in case cc is never reached at the end of the recording
                    return desat, table_desat_aa, table_desat_bb, table_desat_cc

                if data[bb] <= data[bb - 1]:
                    if data[aa] - data[bb] >= thres:
                        cc = bb + 1

                        if cc >= lg_dat:
                            # this is added to stop the loop when c has reached the end of the record
                            return desat, table_desat_aa, table_desat_bb, table_desat_cc
                        else:
                            out_c = 0

                        while cc < lg_dat and out_c == 0:
                            if ((data[aa] - data[cc]) <= 1 or (data[cc] - data[bb]) >= thres) and cc - aa >= 10:
                                if cc - aa <= max_desat_lg:
                                    desat = desat + 1
                                    table_desat_aa = np.append(table_desat_aa, [aa])
                                    table_desat_bb = np.append(table_desat_bb, [bb])
                                    table_desat_cc = np.append(table_desat_cc, [cc])
                                    aa = cc + 1
                                    out_b = 1
                                    out_c = 1
                                else:
                                    aa = cc + 1
                                    out_b = 1
                                    out_c = 1
                            else:
                                cc = cc + 1
                                if cc > lg_dat - 1:
                                    return desat, table_desat_aa, table_desat_bb, table_desat_cc

                                if data[bb] >= data[cc - 1]:
                                    bb = cc - 1
                                    out_c = 0
                                else:
                                    out_c = 0
                    else:
                        bb = bb + 1

                else:
                    aa = aa + 1
                    out_b = 1
        else:
            aa = aa + 1

    return desat, table_desat_aa, table_desat_bb, table_desat_cc
