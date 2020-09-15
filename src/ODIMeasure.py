import numpy as np

from src._ResultsClasses import ODIMeasureResult


class ODIMeasure:
    """
    Function that calculates the ODI from spo2 time series.
    Suppose that the data has been preprocessed.

    :param
        signal: The SpO2 signal, of shape (N,)
        ODI_Threshold: Threshold to compute Oxygen Desaturation Index.

    :return:
        ODIMeasureResult class containing the following features:
            -	ODI: the average number of desaturation events per hour.
            -	begin: List of indices of beginning of each desaturation event.
            -	end: List of indices of end of each desaturation event.
    """

    def __init__(self, ODI_Threshold=3):
        self.ODI_Threshold = ODI_Threshold

    def compute(self, signal) -> ODIMeasureResult:
        if len(signal) == 0:
            ODIMeasureResult(0, [], [])

        return self.__DesaturationDetector(signal)

    def __DesaturationDetector(self, signal):
        # run desaturation detector, implemented by Dr. Joachim Behar
        _, table_desat_aa, _, table_desat_cc = self.__sc_desaturations(signal)
        table_desat_cc = np.array(table_desat_cc).astype(int)
        table_desat_aa = np.array(table_desat_aa).astype(int)
        table_desat_dd = self.__FindD_Points(signal, table_desat_aa, table_desat_cc)
        table_desat_dd = np.array(table_desat_dd).astype(int)
        ODI = len(table_desat_aa) / len(signal) * 3600  # Convert to event/h
        return ODIMeasureResult(ODI, table_desat_aa, table_desat_dd)

    def __FindD_Points(self, signal, table_desat_aa, table_desat_cc):
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