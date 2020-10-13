from pobm._ErrorHandler import _check_shape_
import numpy as np
import warnings

from pobm._ResultsClasses import DesaturationsMeasuresResults


class DesaturationsMeasures:
    """
    Class that calculates the Desaturation Features from spo2 time series.
    Suppose that the data has been preprocessed.

    :param ODI_Threshold: Threshold to compute Oxygen Desaturation Index.
    """

    def __init__(self, ODI_Threshold=3):
        self.ODI_Threshold = ODI_Threshold
        self.begin = []
        self.end = []

    def compute(self, signal) -> DesaturationsMeasuresResults:
        """
        :param signal: 1-d array, of shape (N,) where N is the length of the signal
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
        _check_shape_(signal)

        warnings.simplefilter('ignore', np.RankWarning)

        ODI = self.__desaturation_detector(signal)
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

            desaturation_features = DesaturationsMeasuresResults(ODI, DL_u, DL_sd, DA100_u, DA100_sd, DAmax_u, DAmax_sd,
                                                                 DD100_u, DD100_sd, DDmax_u, DDmax_sd, DS_u, DS_sd,
                                                                 TD_u, TD_sd, self.begin, self.end)
        else:
            desaturation_features = DesaturationsMeasuresResults(ODI, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

        if desaturation_features.DS_u is None:
            desaturation_features.DS_u = 0
        if desaturation_features.DS_sd is None:
            desaturation_features.DS_sd = 0

        return desaturation_features

    def desat_embedding(self):
        """
        Help function for the class
        
        :return: helper arrays containing the information about desaturation lengths and areas.
        """
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

    def desaturation_detector(self, signal):
        """
        run desaturation detector, implemented by Dr. Joachim Behar

        :param signal: The SpO2 signal, of shape (N,)
        :return:
            ODIMeasureResult class containing the following features:
                -	ODI: the average number of desaturation events per hour.
                -	begin: List of indices of beginning of each desaturation event.
                -	end: List of indices of end of each desaturation event.
        """
        _, table_desat_aa, _, table_desat_cc = self.__sc_desaturations(signal)
        table_desat_cc = np.array(table_desat_cc).astype(int)
        table_desat_aa = np.array(table_desat_aa).astype(int)
        table_desat_dd = self.__find_d_points(signal, table_desat_aa, table_desat_cc)
        table_desat_dd = np.array(table_desat_dd).astype(int)
        ODI = len(table_desat_aa) / len(signal) * 3600  # Convert to event/h

        self.begin = table_desat_aa
        self.end = table_desat_dd
        return ODI

    def __find_d_points(self, signal, table_desat_aa, table_desat_cc):
        """
        Helper function, to findfiducials points D of each desaturation

        :param signal: The SpO2 signal, of shape (N,)
        :param table_desat_aa: List of fiducials points A, of shape (M,)
        :param table_desat_cc: List of fiducials points C, of shape (M,) table_desat_aa and table_desat_cc must have the same dimension
        :return: List of fiducials points D, of shape (M,)
        """
        table_desat_dd = []
        for i in range(len(table_desat_aa)):
            if signal[table_desat_cc[i]] >= signal[table_desat_aa[i]] - 1:
                table_desat_dd.append(table_desat_cc[i])
            else:
                found = False
                for j in range(90):
                    if signal[table_desat_aa[i] + j] >= signal[table_desat_aa[i]] - 1:
                        found = True
                        table_desat_dd.append(table_desat_aa[i] + j)
                        break
                if found is False:
                    table_desat_dd.append(table_desat_aa[i] + 90)
        return table_desat_dd

    def __sc_desaturations(self, data):
        """
        This function implements the algorithm of:

        Hwang, Su Hwan, et al. "Real-time automatic apneic event detection using nocturnal pulse oximetry."
        IEEE Transactions on Biomedical Engineering 65.3 (2018): 706-712.

        NOTE: The original function search desaturations that are minimum 10 seconds long and maximum 90 seconds long.
        In addition the original algorithm actually looked to me more like an estimate of the ODI4 than ODI3. This
        implementation is updated to allow the estimation of ODI3 and allows desaturations that are up to 120 seconds
        based on some of our observations. In addition, some conditions were added to avoid becoming blocked in
        infinite while loops.

        Important: The algorithm assumes a sampling rate of 1Hz and a quantization of 1% to the input data.

        :param data: The SpO2 signal, of shape (N,)
        :return:
            desat: number of desaturations
            table_desat_aa: location of the aa feature points
            table_desat_bb: location of the bb feature points
            table_desat_cc: location of the cc feature points
        """

        aa = 1
        desat = 0
        max_desat_lg = 120  # was 90 sec in the original paper. Changed to 120 because I have seen longer desaturations.
        lg_dat = len(data)
        thres = self.ODI_Threshold
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
