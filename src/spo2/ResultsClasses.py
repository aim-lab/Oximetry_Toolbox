from typing import NamedTuple
import numpy as np


class OverallGeneralMeasuresResult(NamedTuple):
    AV: float
    MED: float
    Min: float
    SD: float
    RG: float
    P: float
    M: float
    ZC: float
    DI: float

    def __str__(self):
        return str(dict(self._asdict()))


class ODIMeasureResult(NamedTuple):
    ODI: float
    begin: np.array
    end: np.array

    def __str__(self):
        return str({"ODI": self.ODI, "begin": self.begin.flatten().tolist(), "end": self.end.flatten().tolist()})


class DesaturationsMeasuresResults(NamedTuple):
    DL_u: float
    DL_sd: float
    DA100_u: float
    DA100_sd: float
    DAmax_u: float
    DAmax_sd: float
    DD100_u: float
    DD100_sd: float
    DDmax_u: float
    DDmax_sd: float
    DS_u: float
    DS_sd: float
    TD_u: float
    TD_sd: float

    def __str__(self):
        return str(dict(self._asdict()))


class HypoxicBurdenMeasuresResults(NamedTuple):
    CA: float
    CT: float
    POD: float
    AODmax: float
    AOD100: float

    def __str__(self):
        return str(dict(self._asdict()))


class ComplexityMeasuresResults(NamedTuple):
    ApEn: float
    LZ: float
    CTM: float
    SampEn: float
    DFA: float

    def __str__(self):
        return str(dict(self._asdict()))

class PRSAResults(NamedTuple):
    PRSAc: float
    PRSAad: float
    PRSAos: float
    PRSAsb: float
    PRSAsa: float
    AC: float

    def __str__(self):
        return str(dict(self._asdict()))

class PSDResults(NamedTuple):
    PSD_total: float
    PSD_band: float
    PSD_ratio: float
    PSD_peak: float

    def __str__(self):
        return str(dict(self._asdict()))