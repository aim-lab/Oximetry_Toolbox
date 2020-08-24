import numpy as np

import Detector
from ResultsClasses import ODIMeasureResult


def DesaturationDetector(signal, ODI_Threshold):
    # run desaturation detector, implemented by Dr. Joachim Behar
    _, table_desat_aa, _, table_desat_cc = Detector.sc_desaturations(signal, thres=ODI_Threshold)
    table_desat_aa = np.array(table_desat_aa).astype(int)
    table_desat_cc = np.array(table_desat_cc).astype(int)
    ODI = len(table_desat_aa) / len(signal) * 3600  # Convert to event/h
    return ODIMeasureResult(ODI, table_desat_aa, table_desat_cc)
