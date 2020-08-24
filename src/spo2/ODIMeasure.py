from ODIMeasure_ import DesaturationDetector
from ResultsClasses import ODIMeasureResult


def ODIMeasure(signal, ODI_Threshold=3) -> ODIMeasureResult:
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

    if len(signal) == 0:
        ODIMeasureResult(0, [], [])

    return DesaturationDetector(signal, ODI_Threshold)
