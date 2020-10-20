import numpy as np
from dataclasses import dataclass
import dataclasses


@dataclass
class OverallGeneralMeasuresResult:
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
        return str(dict(dataclasses.asdict(self)))


@dataclass
class ODIMeasureResult:
    ODI: float
    begin: np.array
    end: np.array

    def __str__(self):
        return str({"ODI": self.ODI, "begin": self.begin.flatten().tolist(), "end": self.end.flatten().tolist()})


@dataclass
class DesaturationsMeasuresResults:
    ODI: float
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
    begin: np.array
    end: np.array

    def __str__(self):
        desat_measures = dict(dataclasses.asdict(self))
        desat_measures['begin'] = self.begin.flatten().tolist()
        desat_measures['end'] = self.end.flatten().tolist()
        return str(desat_measures)



@dataclass
class HypoxicBurdenMeasuresResults:
    CA: float
    CT: float
    POD: float
    AODmax: float
    AOD100: float

    def __str__(self):
        return str(dict(dataclasses.asdict(self)))


@dataclass
class ComplexityMeasuresResults:
    ApEn: float
    LZ: float
    CTM: float
    SampEn: float
    DFA: float

    def __str__(self):
        return str(dict(dataclasses.asdict(self)))


@dataclass
class PRSAResults:
    PRSAc: float
    PRSAad: float
    PRSAos: float
    PRSAsb: float
    PRSAsa: float
    AC: float

    def __str__(self):
        return str(dict(dataclasses.asdict(self)))


@dataclass
class PSDResults:
    PSD_total: float
    PSD_band: float
    PSD_ratio: float
    PSD_peak: float

    def __str__(self):
        return str(dict(dataclasses.asdict(self)))
