import numpy as np

from _spo2._Detector import _sc_desaturations_
from _spo2._ResultsClasses import ODIMeasureResult


def _DesaturationDetector_(signal, ODI_Threshold):
    # run desaturation detector, implemented by Dr. Joachim Behar
    _, table_desat_aa, _, table_desat_cc = _sc_desaturations_(signal, thres=ODI_Threshold)
    table_desat_cc = np.array(table_desat_cc).astype(int)
    table_desat_aa = np.array(table_desat_aa).astype(int)
    table_desat_dd = _FindD_Points_(signal, table_desat_aa, table_desat_cc)
    table_desat_dd = np.array(table_desat_dd).astype(int)
    ODI = len(table_desat_aa) / len(signal) * 3600  # Convert to event/h
    return ODIMeasureResult(ODI, table_desat_aa, table_desat_dd)


def _FindD_Points_(signal, table_desat_aa, table_desat_cc):
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
