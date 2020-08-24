import numpy as np

from _spo2._Detector import _sc_desaturations_
from _spo2._ResultsClasses import ODIMeasureResult


def _DesaturationDetector_(signal, ODI_Threshold):
    # run desaturation detector, implemented by Dr. Joachim Behar
    _, table_desat_aa, _, table_desat_cc = _sc_desaturations_(signal, thres=ODI_Threshold)
    table_desat_aa = np.array(table_desat_aa).astype(int)
    table_desat_cc = np.array(table_desat_cc).astype(int)
    ODI = len(table_desat_aa) / len(signal) * 3600  # Convert to event/h
    return ODIMeasureResult(ODI, table_desat_aa, table_desat_cc)
