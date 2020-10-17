import pyedflib
import numpy as np
import os

from pobm.obm.complex import ComplexityMeasures
from pobm.obm.desat import DesaturationsMeasures
from pobm.obm.burden import HypoxicBurdenMeasures
from pobm.obm.general import OverallGeneralMeasures
from pobm.obm.periodicity import PRSAMeasures, PSDMeasures


class OBMTests:
    def read_spo2(self, filename):
        database_dir = "resources"
        edf = pyedflib.EdfReader(os.path.join(database_dir, filename))
        i_position = np.where(np.array(edf.getSignalLabels()) == 'SaO2')[0][0]
        position = edf.readSignal(i_position)
        signal = np.array(position).astype(np.float)

        return signal

    def ComplexityMeasuresTest(self):
        signal = self.read_spo2("spo2_example.edf")
        complexity_class = ComplexityMeasures()
        results = complexity_class.compute(signal[100:1000])

        assert results.ApEn == 0.4034945854530416
        assert results.LZ == 41
        assert results.CTM == 0.7906458797327395
        assert results.SampEn == 0.1775722812097163
        assert results.DFA == 7.130216399460569

    def DesaturationsMeasuresTest(self):
        signal = self.read_spo2("spo2_example.edf")

        desat_class = DesaturationsMeasures()
        results_desat = desat_class.compute(signal)

        hypoxic_class = HypoxicBurdenMeasures(results_desat.begin, results_desat.end)
        results_hypoxic = hypoxic_class.compute(signal)

        assert results_desat.ODI == 1.8819188191881917
        assert results_desat.DL_u == 29.647058823529413
        assert results_desat.DL_sd == 34.42800019216675
        assert results_desat.DA100_u == 1098.6565777604246
        assert results_desat.DA100_sd == 2203.731771984378
        assert results_desat.DAmax_u == 802.4434182004228
        assert results_desat.DAmax_sd == 1684.5654856535532
        assert results_desat.DD100_u == 56.93859141276103
        assert results_desat.DD100_sd == 45.69032381143049
        assert results_desat.DDmax_u == 48.94214586727344
        assert results_desat.DDmax_sd == 44.644705897598534
        assert results_desat.DS_u == -1.2203525162002746
        assert results_desat.DS_sd == 2.357303129324738
        assert results_desat.TD_u == 1738.25
        assert results_desat.TD_sd == 2300.059224780962

        assert results_hypoxic.CA == 0.1504202190110226
        assert results_hypoxic.CT == 16.918819188191883
        assert results_hypoxic.POD == 0.015498154981549815
        assert results_hypoxic.AODmax == 0.4194814916791878
        assert results_hypoxic.AOD100 == 0.574328469308955

    def OverallMeasuresTest(self):
        signal = self.read_spo2("spo2_example.edf")

        statistics_class = OverallGeneralMeasures()
        results = statistics_class.compute(signal)

        assert results.AV == 79.4466424568114
        assert results.MED == 93.35927367055771
        assert results.Min == 0.0015259021896696422
        assert results.SD == 33.09547751682518
        assert results.RG == 98.04379339284353
        assert results.P == 0.0015259021896696422
        assert results.M == 27.220172201722015
        assert results.ZC == 45
        assert results.DI == 1.2027206308444267

    def PeriodicityMeasuresTest(self):
        signal = self.read_spo2("spo2_example.edf")

        prsa_class = PRSAMeasures()
        results_PRSA = prsa_class.compute(signal)

        psd_class = PSDMeasures()
        results_PSD = psd_class.compute(signal)

        assert results_PRSA.PRSAc == -0.9476063739183793
        assert results_PRSA.PRSAad == 2.2225394756134165
        assert results_PRSA.PRSAos == -0.06035762599788687
        assert results_PRSA.PRSAsb == 0.07173001725579922
        assert results_PRSA.PRSAsa == 0.08263965068491867
        assert results_PRSA.AC == 102490.13876874525

        assert results_PSD.PSD_total == 0.38352402885932996
        assert results_PSD.PSD_band == 0.11099710485334692
        assert results_PSD.PSD_ratio == 0.2894136911928894
        assert results_PSD.PSD_peak == 0.02640691195954759


test = OBMTests()
test.ComplexityMeasuresTest()
test.DesaturationsMeasuresTest()
test.OverallMeasuresTest()
test.PeriodicityMeasuresTest()
