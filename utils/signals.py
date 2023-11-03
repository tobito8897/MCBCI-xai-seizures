import re
import mne
import numpy as np
import scipy.signal


class EegProcessor():

    PIPELINE = [{"f": 60, "fs": 256, "type": "notch"},
                {"f": 0.5, "order": 2, "fs": 256, "type": "highpass"}]

    def __init__(self, bipolar_channels: tuple, f_samp: list,
                 eeg_file: str):
        self.bipolar_channels = bipolar_channels
        self.f_samp = f_samp
        self.channels = None
        self.data = eeg_file

    @property
    def data(self) -> np.array:
        return self._data

    @data.setter
    def data(self, filename: str):
        data = mne.io.read_raw_edf(filename)
        self._data = data.get_data()
        self.channels = data.ch_names

    def clean(self):
        for filter in EegProcessor.PIPELINE:
            if filter["type"] == "notch":
                b, a = scipy.signal.iirnotch(filter["f"], 30, fs=filter["fs"])
                self._data = scipy.signal.lfilter(b, a, self._data)
            else:
                b, a = scipy.signal.butter(filter["order"], filter["f"],
                                           btype=filter["type"],
                                           fs=self.f_samp)
                self._data = scipy.signal.lfilter(b, a, self._data)

    def downsample(self, factor):
        self._data = scipy.signal.decimate(self._data, factor)

    def scale(self, gain, units):
        self._data = self._data*gain/units

    def select_channels(self):
        regex = "-[0-9]$"
        channels_to_idx = {re.sub(regex, "", x): y for x, y in zip(self.channels,
                                                                   range(len(self.channels)))}
        temp_eeg = np.zeros([len(self.bipolar_channels),
                            self._data.shape[1]])
        for idx, channel in zip(range(len(self.bipolar_channels)),
                                self.bipolar_channels):
            bipolar_channel = f"{channel[0]}-{channel[1]}"
            new_channel = self._data[channels_to_idx[bipolar_channel], :]
            temp_eeg[idx, :] = new_channel
        self._data = temp_eeg


class EegProcessorSiena(EegProcessor):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def select_channels(self):
        regex = "^EEG "
        channels_to_idx = {re.sub(regex, "", x).lower(): y for x, y in zip(self.channels,
                                                                           range(len(self.channels)))}
        temp_eeg = np.zeros([len(self.bipolar_channels),
                            self._data.shape[1]])
        for idx, channel in zip(range(len(self.bipolar_channels)),
                                self.bipolar_channels):
            new_channel = (self._data[channels_to_idx[channel[0].lower()], :] -
                           self._data[channels_to_idx[channel[1].lower()], :])
            temp_eeg[idx, :] = new_channel
        self._data = temp_eeg


class EegProcessorTusz(EegProcessor):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def data(self) -> np.array:
        return self._data

    @data.setter
    def data(self, filename: str):
        data = mne.io.read_raw_edf(filename)
        self._data = data.get_data()
        self.f_samp = data.info["sfreq"]
        self.channels = data.ch_names

    def select_channels(self):
        channels_to_idx = {}
        for x, y in zip(self.channels, range(len(self.channels))):
            key = re.sub("^EEG ", "", x).split("-")[0].lower()
            channels_to_idx[key] = y

        temp_eeg = np.zeros([len(self.bipolar_channels),
                            self._data.shape[1]])
        for idx, channel in zip(range(len(self.bipolar_channels)),
                                self.bipolar_channels):
            new_channel = (self._data[channels_to_idx[channel[0].lower()], :] -
                           self._data[channels_to_idx[channel[1].lower()], :])
            temp_eeg[idx, :] = new_channel
        self._data = temp_eeg


class IctalStage():

    def __init__(self, seizure_start: int, seizure_end: int,
                 epoch_size: int, overlap: int, f_samp: int, data: np.array):
        self.seizure_start = seizure_start
        self.seizure_end = seizure_end
        self.epoch_size = epoch_size
        self.f_samp = f_samp
        self.data = data
        self.overlap = overlap

    def __iter__(self):
        return self

    def __next__(self):
        _next_seizure_start = int(self.seizure_start +
                                  self.epoch_size)
        if _next_seizure_start > self.seizure_end:
            raise StopIteration
        epoch_start = int(self.seizure_start*self.f_samp)
        epoch_end = int(epoch_start + self.epoch_size*self.f_samp)
        epoch = self.data[:, epoch_start: epoch_end]
        self.seizure_start = (epoch_end-(self.epoch_size*self.overlap)*self.f_samp)/self.f_samp
        return epoch


class NoIctalStage():

    def __init__(self, pre_seizure_end: int, post_seizure_start: int,
                 epoch_size: int, f_samp: int, data: np.array):
        self.pre_seizure_end = pre_seizure_end
        self.post_seizure_start = post_seizure_start
        self.epoch_size = epoch_size
        self.f_samp = f_samp
        self.data = data
        self.counter = 0

    def __iter__(self):
        return self

    def __next__(self):
        _next_pre_seizure_end = int(self.pre_seizure_end +
                                    self.epoch_size)
        if _next_pre_seizure_end > self.post_seizure_start:
            raise StopIteration
        epoch_start = int(self.pre_seizure_end*self.f_samp)
        epoch_end = int(epoch_start + self.epoch_size*self.f_samp)
        epoch = self.data[:, epoch_start: epoch_end]
        self.pre_seizure_end = int(epoch_end/self.f_samp)
        return epoch


class IctalDetector():

    def __init__(self, metadata: tuple, epoch_size: int, overlap: int,
                 f_samp: int, data: np.array):
        self.overlap = overlap
        self.counter = 0
        self.epoch_size = epoch_size
        self.f_samp = f_samp
        self.data = data
        self.ictal_stages = metadata

    @property
    def ictal_stages(self) -> np.array:
        return self._ictal_stages

    @ictal_stages.setter
    def ictal_stages(self, metadata: str):
        self._ictal_stages = []
        for idx in range(1, len(metadata)):
            stage = IctalStage(metadata[idx][0], metadata[idx][1],
                               self.epoch_size,
                               self.overlap,
                               self.f_samp, self.data)
            self._ictal_stages.append(stage)

    def __iter__(self):
        return self

    def __next__(self) -> IctalStage:
        self.counter += 1
        if self.counter > len(self.ictal_stages):
            raise StopIteration
        return self.ictal_stages[self.counter - 1]


class NoIctalDetector():

    def __init__(self, metadata: tuple, epoch_size: int,
                 f_samp: int, data: np.array):
        self.counter = 0
        self.epoch_size = epoch_size
        self.f_samp = f_samp
        self.data = data
        self.preictal_stages = metadata

    @property
    def preictal_stages(self) -> np.array:
        return self._preictal_stages

    @preictal_stages.setter
    def preictal_stages(self, metadata: str):
        max_length = int(self.data.shape[-1]/self.f_samp)
        metadata = list(metadata)
        metadata.append((max_length, max_length))
        self._preictal_stages = []
        for idx in range(0, len(metadata)-1):
            stage = NoIctalStage(metadata[idx][1], metadata[idx+1][0],
                                 self.epoch_size, self.f_samp, self.data)
            self._preictal_stages.append(stage)

    def __iter__(self):
        return self

    def __next__(self) -> NoIctalStage:
        self.counter += 1
        if self.counter > len(self.preictal_stages):
            raise StopIteration
        return self.preictal_stages[self.counter - 1]
