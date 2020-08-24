import numpy as np

from _spo2._DesaturationsMeasures import _desat_embedding_
from _spo2._ResultsClasses import HypoxicBurdenMeasuresResults


def _CompHBMeasures_(signal, desaturations_signal, CT_Threshold, CA_baseline):
    desaturations, desaturation_valid, desaturation_length_all, desaturation_int_100_all, \
    desaturation_int_max_all, desaturation_depth_100_all, desaturation_depth_max_all, \
    desaturation_slope_all = _desat_embedding_(desaturations_signal)

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

    desaturation_features = HypoxicBurdenMeasuresResults(_CompCA_(signal, CA_baseline), _CompCT_(signal, CT_Threshold),
                                                         0.0, 0.0, 0.0)
    if np.sum(desaturation_valid) != 0:
        desaturation_features = desaturation_features._replace(
            POD=np.nansum(desaturation_length_all[desaturation_valid]) / len(signal))
        desaturation_features = desaturation_features._replace(
            AODmax=np.nansum(desaturation_int_max_all[desaturation_valid]) / len(signal))
        desaturation_features = desaturation_features._replace(
            AOD100=np.nansum(desaturation_int_100_all[desaturation_valid]) / len(signal))

    return desaturation_features


def _CompCA_(signal, baseline):
    with np.errstate(invalid='ignore'):
        signal_under_baseline = signal[signal < baseline]
    if len(signal_under_baseline) == 0:
        return 0.0
    # return integrate.cumtrapz(signal_under_baseline) / len(signal)
    return sum(signal_under_baseline) / len(signal)


def _CompCT_(signal, treshold):
    with np.errstate(invalid='ignore'):
        return 100 * len(signal[signal <= treshold]) / len(signal)
