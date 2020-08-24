import numpy as np


def check_shape(signal):
    assert len(signal) > 0, "Signal must be non-empty"
    signal = np.array(signal)
    assert len(signal.shape) == 1, "Signal must be 1-d dimension array"


def check_len_ApEn(N, m):
    assert N - m + 1 >= 0, "Signal is too short to calculate ApEn"


def check_window_delta(len_signal, window_size):
    assert len_signal > window_size, "Window size of delta index must be larger than the signal length"


def check_fragment_PRSA(d):
    assert d > 0, "The parameter d should be strictly positive"
