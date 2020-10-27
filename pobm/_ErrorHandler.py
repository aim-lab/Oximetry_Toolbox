import numpy as np


def _check_shape_(signal):
    assert len(signal) > 0, "Signal must be non-empty"
    signal = np.array(signal)
    assert len(signal.shape) == 1, "Signal must be 1-d dimension array"


def _check_len_ApEn_(N, m):
    assert N - m + 1 >= 0, "Signal is too short to calculate ApEn"


def _check_window_delta_(len_signal, window_size):
    assert len_signal > window_size, "Window size of delta index must be larger than the signal length"


def _check_fragment_PRSA_(d):
    assert d > 0, "The parameter d should be strictly positive"


class WrongParameter(Exception):
    pass
