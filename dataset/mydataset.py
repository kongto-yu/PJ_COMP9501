from torch.utils.data import Dataset
import torch

import numpy as np
import h5py
import csv
import math

def trans_list(input_csv):
    list_sample = []
    for row in csv.reader(open(input_csv, 'r'), delimiter=','):
        if len(row) < 2:
            continue
        list_sample.append(row)
    print(len(list_sample))
    return list_sample


def signal_tile_crop(signal, length):
    n = int(length / signal.size(1)) + 1
    tiled_signal = torch.tile(signal, (n,))
    croped_signal = tiled_signal[:, : length]
    return croped_signal

def stft(audio_signal, stft_frame, stft_hop):
    audio_stft = torch.stft(audio_signal, n_fft=stft_frame, hop_length=stft_hop, normalized=False,
                       window=torch.hann_window(window_length=stft_frame), return_complex=True)
    audio_amp = torch.abs(audio_stft)
    audio_phase = torch.angle(audio_stft)

    return audio_amp, audio_phase

# non-coherent-sum of spectrogram across channels
def noncoh_stft(radio_signal, stft_frame, stft_hop):
    radio_signal = torch.stft(radio_signal, n_fft=stft_frame, hop_length=stft_hop, normalized=False,
                            window=torch.hann_window(window_length=stft_frame), return_complex=True)
    radio_stft = torch.sum(radio_signal, dim=0, keepdim=True).div(radio_signal.size(0))
    radio_amp = torch.abs(radio_stft)
    radio_phase = torch.angle(radio_stft)
    return radio_amp, radio_phase


def spect_substract_curve(radar_data, target_sr):
    radar_data_fit = np.zeros([radar_data.shape[0], radar_data.shape[1]])
    for i in range(radar_data.shape[0]):
        time = np.linspace(0, len(radar_data[i, :]) / target_sr, len(radar_data[i, :]))
        y_parameter = np.polyfit(time, radar_data[i, :], 6)
        y_value = np.polyval(y_parameter, time)
        radar_data_fit[i, :] = radar_data[i, :] - y_value
    return radar_data_fit


class mydataset(Dataset):
    def __init__(self, input_path=None):
        """
        files should be a list [(radio file, audio file)]
        """

        self.files = trans_list(input_path)
        self.expmel_len = 64

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        info = self.files[index]
        radio_file = info[0]
        audio_file = info[1]
        mel_len = int(info[2])

        # audio load
        audio_h5f = h5py.File(audio_file, 'r')
        audio_melamp = audio_h5f['mel'][:]
        # radio load
        raido_h5f = h5py.File(radio_file, 'r')
        radio_melamp = raido_h5f['mel'][:]

        assert audio_melamp.shape == radio_melamp.shape
        assert mel_len == audio_melamp.shape[0]

        # random cut
        if mel_len > self.expmel_len:
            center = np.random.randint(self.expmel_len // 2, mel_len - self.expmel_len // 2)
            start = max(0, center - self.expmel_len // 2)
            end = min(mel_len, center + self.expmel_len // 2)
            audio_melamp = audio_melamp[start : end]
            radio_melamp = radio_melamp[start : end]
        elif mel_len < self.expmel_len:
            n = int(self.expmel_len / mel_len) + 1
            audio_melamp = np.tile(audio_melamp, (n, 1))
            audio_melamp = audio_melamp[:self.expmel_len, :]
            radio_melamp = np.tile(radio_melamp, (n, 1))
            radio_melamp = radio_melamp[:self.expmel_len, :]

        return torch.FloatTensor(audio_melamp).unsqueeze(0), torch.FloatTensor(radio_melamp).unsqueeze(0)
