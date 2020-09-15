from src._ErrorHandler import _check_shape_
import numpy as np
import warnings

from src._ResultsClasses import DesaturationsMeasuresResults


class DesaturationsMeasures:
    """
            Function that calculates the Desaturation Features from spo2 time series.
            Suppose that the data has been preprocessed.

            :param
                signal: 1-d array, of shape (N,) where N is the length of the signal
                begin: List of indices of beginning of each desaturation event.
                end: List of indices of end of each desaturation event.

            :return:
                DesaturationsMeasuresResults class containing the following features:
                    -	DL_u: Mean of desaturation length
                    -	DL_sd: Standard deviation of desaturation length
                    -	DA100_u: Mean of desaturation area using 100% as baseline.
                    -	DA100_sd: Standard deviation of desaturation area using 100% as baseline
                    -	DAmax_u: Mean of desaturation area using max value as baseline.
                    -	DAmax_sd: Standard deviation of desaturation area using max value as baseline
                    -	DD100_u: Mean of depth desaturation from 100%.
                    -	DD100_sd: Standard deviation of depth desaturation from 100%.
                    -	DDmax_u: Mean of depth desaturation from max value.
                    -	DDmax_sd: Standard deviation of depth desaturation from max value.
                    -	DS_u: Mean of the desaturation slope.
                    -	DS_sd: Standard deviation of the desaturation slope.
                    -   TD_u: Mean of time between two consecutive desaturation events.
                    -   TD_sd: Standard deviation of time between 2 consecutive desaturation events.
            """

    def __init__(self, begin, end):

        self.begin = begin
        self.end = end

    def compute(self, signal) -> DesaturationsMeasuresResults:

        _check_shape_(signal)

        return self.__processing_desat(signal)

    def __processing_desat(self, signal):
        warnings.simplefilter('ignore', np.RankWarning)
        desaturations, desaturation_valid, desaturation_length_all, desaturation_int_100_all, \
        desaturation_int_max_all, desaturation_depth_100_all, desaturation_depth_max_all, \
        desaturation_slope_all = self.desat_embedding()

        time_spo2_array = np.array(range(len(signal)))

        starts = []
        for (i, desaturation) in enumerate(desaturations):
            starts.append(desaturation['Start'])
            desaturation_idx = (time_spo2_array >= desaturation['Start']) & (time_spo2_array <= desaturation['End'])

            if np.sum(desaturation_idx) == 0:
                continue
            signal = np.array(signal)

            desaturation_time = time_spo2_array[desaturation_idx]
            desaturation_spo2 = signal[desaturation_idx]
            desaturation_min = np.nanmin(desaturation_spo2)
            desaturation_max = np.nanmax(desaturation_spo2)

            desaturation_valid[i] = True
            desaturation_length_all[i] = desaturation['Duration']
            desaturation_int_100_all[i] = np.nansum(100 - desaturation_spo2)
            desaturation_int_max_all[i] = np.nansum(desaturation_max - desaturation_spo2)
            desaturation_depth_100_all[i] = 100 - desaturation_min
            desaturation_depth_max_all[i] = desaturation_max - desaturation_min

            # Only consider points from the first maximum value to the last lowest value of SpO2
            desaturation_idx_max = np.where(desaturation_spo2 == desaturation_max)[0][0]  # first index of max value
            desaturation_idx_min = np.where(desaturation_spo2 == desaturation_min)[0][-1]  # last index of min value
            desaturation_idx_max_min = np.arange(desaturation_idx_max, desaturation_idx_min + 1)

            # Due to mislabeling, the max value may be after the min value, in which case ignore the desaturation.
            if len(desaturation_idx_max_min) > 0:
                p = np.polyfit(np.int64(desaturation_time[desaturation_idx_max_min]),
                               desaturation_spo2[desaturation_idx_max_min], 1)

                desaturation_slope_all[i] = p[0]

        diff_desats = abs(starts - np.roll(starts, 1))
        diff_desats = diff_desats[1:]

        if np.sum(desaturation_valid) != 0:
            DL_u = np.nanmean(desaturation_length_all[desaturation_valid])
            DL_sd = np.nanstd(desaturation_length_all[desaturation_valid])
            DA100_u = np.nanmean(desaturation_int_100_all[desaturation_valid])
            DA100_sd = np.nanstd(desaturation_int_100_all[desaturation_valid])
            DAmax_u = np.nanmean(desaturation_int_max_all[desaturation_valid])
            DAmax_sd = np.nanstd(desaturation_int_max_all[desaturation_valid])
            DD100_u = np.nanmean(desaturation_depth_100_all[desaturation_valid])
            DD100_sd = np.nanstd(desaturation_depth_100_all[desaturation_valid])
            DDmax_u = np.nanmean(desaturation_depth_max_all[desaturation_valid])
            DDmax_sd = np.nanstd(desaturation_depth_max_all[desaturation_valid])

            DS_u = np.nanmean(desaturation_slope_all[desaturation_valid])
            DS_sd = np.nanstd(desaturation_slope_all[desaturation_valid])

            TD_u = np.nanmean(diff_desats)
            TD_sd = np.nanstd(diff_desats)

            desaturation_features = DesaturationsMeasuresResults(DL_u, DL_sd, DA100_u, DA100_sd, DAmax_u, DAmax_sd,
                                                                 DD100_u,
                                                                 DD100_sd, DDmax_u, DDmax_sd, DS_u, DS_sd, TD_u, TD_sd)
        else:
            desaturation_features = DesaturationsMeasuresResults(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

        if desaturation_features.DS_u is None:
            desaturation_features.DS_u = 0
        if desaturation_features.DS_sd is None:
            desaturation_features.DS_sd = 0

        return desaturation_features

    def desat_embedding(self):
        table_desat_aa = self.begin
        table_desat_cc = self.end

        if isinstance(table_desat_aa, int):
            table_desat_aa = [table_desat_aa]
        if isinstance(table_desat_cc, int):
            table_desat_cc = [table_desat_cc]

        desaturations = []  # empty this structure to fill it with the new desaturations
        for kk in range(0, len(table_desat_aa)):
            desaturations.append({
                'Start': int(table_desat_aa[kk]),
                'Duration': table_desat_cc[kk] - table_desat_aa[kk],
                'End': int(table_desat_cc[kk])
            })

        desaturation_valid = np.full(len(desaturations), False)
        desaturation_length_all = np.full(len(desaturations), np.nan)
        desaturation_int_100_all = np.full(len(desaturations), np.nan)
        desaturation_int_max_all = np.full(len(desaturations), np.nan)
        desaturation_depth_100_all = np.full(len(desaturations), np.nan)
        desaturation_depth_max_all = np.full(len(desaturations), np.nan)
        desaturation_slope_all = np.full(len(desaturations), np.nan)
        return desaturations, desaturation_valid, desaturation_length_all, desaturation_int_100_all, \
               desaturation_int_max_all, desaturation_depth_100_all, desaturation_depth_max_all, desaturation_slope_all

