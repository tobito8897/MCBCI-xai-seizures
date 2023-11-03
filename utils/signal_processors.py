import numpy as np
import scipy.signal
import scipy.stats as stats
from pyeeg.entropy import samp_entropy
from pyeeg.hjorth_mobility_complexity import hjorth


def filter_butterworth(f: int, order: int, btype: str, fs: int,
                       data: np.array) -> np.array:
    filtered_data = np.zeros(data.shape)
    b, a = scipy.signal.butter(order, f, btype=btype, fs=fs)
    for window in range(data.shape[0]):
        for channel in range(data.shape[1]):
            _data = scipy.signal.lfilter(b, a, data[window, channel, :])
            filtered_data[window, channel, :] = _data
    return filtered_data


def filter_butterworth_2(f: int, order: int, btype: str, fs: int,
                         data: np.array) -> np.array:
    filtered_data = np.zeros(data.shape)
    b, a = scipy.signal.butter(order, f, btype=btype, fs=fs)
    for window in range(data.shape[0]):
        _data = scipy.signal.lfilter(b, a, data[window, :])
        filtered_data[window, :] = _data
    return filtered_data


def filter_notch(f: int, fs: int, data: np.array) -> np.array:
    filtered_data = np.zeros(data.shape)
    b, a = scipy.signal.iirnotch(f, 30, fs=fs)
    for window in range(data.shape[0]):
        for channel in range(data.shape[1]):
            _data = scipy.signal.lfilter(b, a, data[window, channel, :])
            filtered_data[window, channel, :] = _data
    return filtered_data


def pipeline_filter(data: np.array) -> np.array:
    pipeline = [{"f": 60, "fs": 256, "type": "notch"},                       # Filter 60Hz
                {"f": 0.5, "order": 2, "fs": 256, "type": "highpass"}]       # Filter low freq drifts    

    for filter in pipeline:
        if filter["type"] == "notch":
            data = filter_notch(filter["f"], filter["fs"], data)
        else:
            data = filter_butterworth(filter["f"], filter["order"],
                                      filter["type"], filter["fs"],
                                      data)

    return data


def get_periodogram(data: np.array, fs: int) -> tuple:
    _, psd = scipy.signal.periodogram(data[0, :, 0],
                                      fs=fs)

    periodograms_matrix = np.zeros((data.shape[0],
                                    len(psd)))

    for instance in range(data.shape[0]):
        fxx, psd = scipy.signal.periodogram(data[instance, :, 0], fs=fs)
        periodograms_matrix[instance, :] = np.abs(psd)
    return fxx, periodograms_matrix


class SignalFeatures():

    def __init__(self, f_samp: int):
        self.f_samp = f_samp

    def peak_frequency(self, data: np.array) -> float:
        fxx, psd = scipy.signal.periodogram(data, fs=self.f_samp)
        index = np.argmax(psd)
        return fxx[index]

    def median_frequency(self, data: np.array) -> float:
        fxx, psd = scipy.signal.periodogram(data, fs=self.f_samp)
        p_sum = 0
        for p in psd:
            p_sum += p

        tmp_p_sum = 0
        for f, p in zip(fxx, psd):
            tmp_p_sum += p
            if tmp_p_sum > p_sum/2:
                return f

    def variance(self, data: np.array) -> float:
        return np.var(data)

    def rms(self, data: np.array) -> float:
        return np.sqrt(np.mean(data**2))

    def skewness(self, data: np.array) -> float:
        return stats.skew(data)

    def kurtosis(self, data: np.array) -> float:
        return stats.kurtosis(data)

    def zerocrossing(self, data: np.array) -> int:
        crosses = np.where(np.diff(np.sign(data)))[0]
        return len(crosses)

    def sampleentropy(self, data: np.array) -> int:
        entropy = samp_entropy(data, 2, 0.2*np.std(data))
        return entropy

    def range_val(self, data: np.array) -> int:
        _range = np.max(data) - np.min(data)
        return _range

    def mean(self, data: np.array) -> int:
        _mean = np.mean(data) - np.min(data)
        return _mean

    def sdeviation(self, data: np.array) -> int:
        _sdeviation = np.std(data)
        return _sdeviation

    def complexity(self, data: np.array) -> int:
        complexity, _ = hjorth(data)
        return complexity

    def mobility(self, data: np.array) -> int:
        _, mobility = hjorth(data)
        return mobility

    def interquartile_range(self, data: np.array) -> int:
        Q1 = np.percentile(data, 25, interpolation="midpoint")
        Q3 = np.percentile(data, 75, interpolation="midpoint")
        return Q3 - Q1

    def absolute_median_deviation(self, data: np.array) -> int:
        amd = stats.median_absolute_deviation(data)
        return amd

    def min_val(self, data: np.array) -> int:
        return np.min(data)

    def __call__(self, method_name: str, data: np.array) -> np.array:
        features = []
        for idx in range(data.shape[0]):
            features.append(getattr(self, method_name)(data[idx, :]))
        features = np.array(features)
        return features