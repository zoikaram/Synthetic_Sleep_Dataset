import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import signal
import random
import torch
import einops
import sklearn
import math
import torch.optim as optim
import pandas as pd
import torch.nn as nn
import copy
import utils_config
from sklearn.metrics import f1_score, cohen_kappa_score, roc_auc_score, confusion_matrix
import random

from Dataloader import dotdict, dataloader

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


# Import labels from real data
first_element = next(iter(dataloader.train_loader))
labels_full = first_element[1].to(device)
labels_full = torch.unsqueeze(labels_full, dim=0)

for batch_idx, data in enumerate(dataloader.train_loader,1):
    lab = data[1].to(device)
    lab = torch.unsqueeze(lab, dim=0)
    labels_full = torch.cat((labels_full, lab), dim=0)
    print(batch_idx)

# Exclude the batches that contain less than 5 classes
y_pr = torch.tensor([0, 1, 2, 3, 4]).to(device)
labels = labels_full[0,:,:]
for i in range(1, labels_full.shape[0]):
    y_train = labels_full[i, :, :]
    y, count = torch.unique(y_train, sorted=True, return_inverse=False, return_counts=True, dim=None)
    y.to(device)
    if torch.equal(y, y_pr):
        labels = torch.cat((labels, labels_full[i, :, :]), 0)

div = (labels.shape[0]//32)
labels = einops.rearrange(labels, "(a b) c -> a b c", a=div, b=32)



class synthetic_dataset:
    def __init__(self, y_train, config):
        self.y_train = y_train
        self.config = config
        self.t = np.arange(self.config.N) / float(self.config.fs)

    def my_wave(self, A, omega, fi, t):
        x_volts = A * np.sin(omega * 2 * math.pi * t + fi)
        return x_volts

    def amplitude(self, sin, max, min):
        A = []
        for i in range(sin):
            a = ((max - min) * random.random() + max)
            A = np.append(A, a)
        return A

    def get_wake(self):
        sin = self.config.sin[0]
        A = self.amplitude(sin=sin, max=self.config.Amax[0], min=self.config.Amin[0])

        alphas = random.randint(round(sin * self.config.quota[0]),sin)
        omega = []
        for i in range(alphas):
            o = (self.config.alpha_wake[1] - self.config.alpha_wake[0]) * random.random() + self.config.alpha_wake[0]
            omega = np.append(omega, o)

        thetas = sin - alphas
        for i in range(thetas):
            o = (self.config.theta_wake[1] - self.config.theta_wake[0]) * random.random() + self.config.theta_wake[0]
            omega = np.append(omega, o)

        x_volts = 0
        for i in range(sin):
            fi = 2 * random.random() * math.pi
            x = self.my_wave(A[i], omega[i], fi, self.t)
            x_volts = x_volts + x

        return x_volts

    def get_n1(self):
        sin = self.config.sin[1]
        A = self.amplitude(sin=sin, max=self.config.Amax[1], min=self.config.Amin[1])

        thetas = random.randint(round(sin * self.config.quota[1]), sin)
        omega = []
        for i in range(thetas):
            o = (self.config.theta_n1[1] - self.config.theta_n1[0]) * random.random() + self.config.theta_n1[0]
            omega = np.append(omega, o)

        alphas = sin - thetas
        for i in range(alphas):
            o = (self.config.alpha_n1[1] - self.config.alpha_n1[0]) * random.random() * self.config.alpha_n1[0]
            omega = np.append(omega, o)

        x_volts = 0
        for i in range(sin):
            fi = 2 * random.random() * math.pi
            x = self.my_wave(A[i], omega[i], fi, self.t)
            x_volts = x_volts + x

        return x_volts

    def get_n2(self):
        sin = self.config.sin[2]
        A = self.amplitude(sin=sin, max=self.config.Amax[2], min=self.config.Amin[2])

        thetas = random.randint(round(sin * self.config.quota[2]), sin)
        omega = []
        for i in range(thetas):
            o = (self.config.theta_n2[1] - self.config.theta_n2[0]) * random.random() + self.config.theta_n2[0]
            omega = np.append(omega, o)

        alphas = sin - thetas
        for i in range(alphas):
            o = (self.config.alpha_n2[1] - self.config.alpha_n2[0]) * random.random() + self.config.alpha_n2[0]
            omega = np.append(omega, o)

        x_volts = 0
        for i in range(sin):
            fi = 2 * random.random() * math.pi
            x = self.my_wave(A[i], omega[i], fi, self.t)
            x_volts = x_volts + x

        spindle = kcomp = 0
        while spindle == 0 and kcomp == 0:
            spindle = random.randint(0, self.config.max_spindles)
            kcomp = random.randint(0, self.config.max_kcomp)

        # spindles
        Ampl = freq = []
        for i in range(spindle):
            a = (25 - 10) * random.random() + 10
            Ampl = np.append(Ampl, a)
            o = (16 - 11) * random.random() + 11
            freq = np.append(freq, o)

        # k-complexes
        for i in range(kcomp):
            a = (100 - 50) * random.random() + 50
            Ampl = np.append(Ampl, a)
            o = 2  # math.pi/4
            freq = np.append(freq, o)

        x_split = np.split(x_volts, 60)
        x_volts = x_split[:60 - (spindle + kcomp)]
        t_split = np.split(self.t, 60)

        prm = np.append(Ampl, freq, axis=0)
        prm = np.resize(prm, (2, spindle + kcomp))
        for i in range(spindle + kcomp):
            fi = 0
            x = self.my_wave(prm[0, i], prm[1, i], fi, t_split[0])
            x_volts = np.append(x_volts, x)

        x_split = np.split(x_volts, 60)
        suffle = sklearn.utils.shuffle(x_split)
        x_volts = np.reshape(suffle, (3000)).T

        return x_volts

    def get_n3(self):
        sin = self.config.sin[3]
        A = self.amplitude(sin=sin, max=self.config.Amax[3], min=self.config.Amin[3])

        deltas = random.randint(round(sin * self.config.quota[3]), sin)  # >20%
        omega = []
        for i in range(deltas):
            o = (self.config.delta_n3[1] - self.config.delta_n3[0]) * random.random() + self.config.delta_n3[0]
            omega = np.append(omega, o)

        thetas = sin - deltas
        for i in range(thetas):
            o = (self.config.theta_n3[1] - self.config.theta_n3[0]) * random.random() + self.config.theta_n3[0]
            omega = np.append(omega, o)

        x_volts = 0
        for i in range(sin):
            fi = 2 * random.random() * math.pi
            x = self.my_wave(A[i], omega[i], fi, self.t)
            x_volts = x_volts + x

        return x_volts

    def get_rem(self):
        sin = self.config.sin[4]

        thetas = random.randint(round(sin * self.config.quota[3]), sin)  # >50%
        omega = []
        A = []
        for i in range(thetas):
            a = (self.config.Amax[4] - self.config.Amin[4]) * random.random() + self.config.Amin[4]
            A = np.append(A, a)
            o = (self.config.theta_rem[1] - self.config.theta_rem[0]) * random.random() + self.config.theta_rem[0]
            omega = np.append(omega, o)

        alphas = sin - thetas
        for i in range(alphas):
            a = (self.config.Amax[5] - self.config.Amin[5]) * random.random() + self.config.Amin[5]
            A = np.append(A, a)
            o = (self.config.alpha_rem[1] - self.config.alpha_rem[0]) * random.random() + self.config.alpha_rem[0]
            omega = np.append(omega, o)

        prm = np.append(A, omega, axis=0)
        prm = np.resize(prm, (2, sin))
        np.random.shuffle(np.transpose(prm))

        x_volts = 0
        for i in range(sin):
            fi = 2 * random.random() * math.pi
            x = self.my_wave(prm[0, i], prm[1, i], fi, self.t)
            x_volts = x_volts + x

        return x_volts

    def signal_with_noise(self, x_volts):
        x_watts = x_volts ** 2
        # Adding noise using target SNR
        # Set a target SNR
        target_snr_db = self.config.snr
        # Calculate signal power and convert to dB
        sig_avg_watts = np.mean(x_watts)
        sig_avg_db = 10 * np.log10(sig_avg_watts)
        # Calculate noise according to [2] then convert to watts
        noise_avg_db = sig_avg_db - target_snr_db
        noise_avg_watts = 10 ** (noise_avg_db / 10)
        # Generate an sample of white noise
        mean_noise = 0
        noise_volts = np.random.normal(mean_noise, np.sqrt(noise_avg_watts), len(x_watts))
        # Noise up the original signal
        y_volts = x_volts + noise_volts
        return y_volts

    def get_one_batch(self):
        y, count = torch.unique(self.y_train, sorted=True, return_inverse=False, return_counts=True, dim=None)

        # Wake sampling
        n = count[0,]
        wake_volts = []
        for i in range(n):
            wake_clean = self.get_wake()
            noise = self.signal_with_noise(wake_clean)
            wake_volts = np.append(wake_volts, noise, axis=0)
        wake_volts.shape = (n, self.config.N)

        # N1 sampling
        n = count[1,]
        n1_volts = []
        for i in range(n):
            n1_clean = self.get_n1()
            noise = self.signal_with_noise(n1_clean)
            n1_volts = np.append(n1_volts, noise, axis=0)
        n1_volts.shape = (n, self.config.N)

        # N2 sampling
        n = count[2,]
        n2_volts = []
        for i in range(n):
            n2_clean = self.get_n2()
            noise = self.signal_with_noise(n2_clean)
            n2_volts = np.append(n2_volts, noise, axis=0)
        n2_volts.shape = (n, self.config.N)

        # N3 sampling
        n = count[3,]
        n3_volts = []
        for i in range(n):
            n3_clean = self.get_n3()
            noise = self.signal_with_noise(n3_clean)
            n3_volts = np.append(n3_volts, noise, axis=0)
            if n == 0: break
        n3_volts.shape = (n, self.config.N)

        # REM sampling
        n = count[4,]
        rem_volts = []
        for i in range(n):
            rem_clean = self.get_rem()
            noise = self.signal_with_noise(rem_clean)
            rem_volts = np.append(rem_volts, noise, axis=0)
        rem_volts.shape = (n, self.config.N)

        twake_volts = torch.from_numpy(wake_volts)
        tn1_volts = torch.from_numpy(n1_volts)
        tn2_volts = torch.from_numpy(n2_volts)
        tn3_volts = torch.from_numpy(n3_volts)
        trem_volts = torch.from_numpy(rem_volts)

        wake = 0; n1 = 0; n2 = 0; n3 = 0; rem = 0
        x_train = torch.zeros(self.y_train.shape[0], self.config.N)
        for i in range(self.y_train.shape[0]):
            if self.y_train[i] == 0:
                x_train[i, :] = twake_volts[wake, :]
                wake = wake + 1
            if self.y_train[i] == 1:
                x_train[i, :] = tn1_volts[n1, :]
                n1 = n1 + 1
            if self.y_train[i] == 2:
                x_train[i, :] = tn2_volts[n2, :]
                n2 = n2 + 1
            if self.y_train[i] == 3:
                x_train[i, :] = tn3_volts[n3, :]
                n3 = n3 + 1
            if self.y_train[i] == 4:
                x_train[i, :] = trem_volts[rem, :]
                rem = rem + 1

        normalized_data = torch.zeros(x_train.shape[0], 129, 31)

        for i in range(x_train.shape[0]):
            n_samples = x_train[i]
            window = scipy.signal.hamming(self.config.window_size * self.config.fs)
            noverlap = int(self.config.window_size * self.config.fs * self.config.overlap)
            nperseg = int(self.config.window_size * self.config.fs)

            f, t, Zxx = scipy.signal.stft(x_train[i], fs=self.config.fs, window=window, noverlap=noverlap, nperseg=nperseg,
                                          nfft=self.config.nfft)
            # Compute power spectral density (PSD)
            psd = np.abs(Zxx) ** 2
            # Convert to decibel (dB) scale
            psd_db = 10 * np.log10(psd)
            means = np.mean(psd_db)
            stds = np.std(psd_db)
            normalized_data[i] = (torch.from_numpy(psd_db) - means) / stds

        x_train = einops.rearrange(normalized_data, '(b m) f t -> b m f t', b=32, m=21)
        x_train = torch.unsqueeze(x_train, dim=2)

        return x_train


config = {
    "sin": [10, 10, 10, 10, 10], #Number of sinusoids in each class
    "quota" : [0.6, 0.6, 0.6, 0.2, 0.6], #Quotas of frequency bands in each class
    "Amin" : [0.1, 0.1, 0.1, 0.01, 0.1, 1], #minimum amplitude for each class (last two for REM)
    "Amax": [1.5, 0.4, 1.3, 1.5, 1, 2], #maximum amplitude for each class (last two for REM)
    "alpha_wake": [8,13], #alpha bands for wake
    "theta_wake": [4,8], #theta bands for wake
    "theta_n1": [0.1,8], #theta bands for n1
    "alpha_n1": [8,40], #alpha bands for n1
    "theta_n2": [4,8], #theta bands for n2
    "alpha_n2": [8,13], #alpha bands for n2
    "delta_n3": [0.1,4], #delta bands for n3
    "theta_n3": [4,20], #theta bands for n3
    "theta_rem": [0.1,1], #theta bands for rem
    "alpha_rem": [1,2], #alpha bands for rem
    "max_spindles": 2,
    "max_kcomp": 2,
    "snr": 10, #Signal to noise ratio
    "fs": 100, #Sampling frequency for STFT
    "N": 3000, #Time points per epoch
    "window_size": 2, #Window size for STFT
    "overlap": 0.5, #Overlap for STFT
    "nfft": 256 #Length of the FFT used for STFT
    }

config = dotdict(config)
Synthetic = synthetic_dataset(einops.rearrange(labels[0, :, :], 'b m -> (b m)'), config)
x_train = Synthetic.get_one_batch()
x_train = torch.unsqueeze(x_train, dim=0).to(device)
for i in range(1, labels.shape[0]):
    Synthetic = synthetic_dataset(einops.rearrange(labels[i, :, :], 'b m -> (b m)'), config)
    inputs = Synthetic.get_one_batch()
    inputs = torch.unsqueeze(inputs, dim=0).to(device)
    x_train = torch.cat((x_train, inputs), 0)
    print(i)

y_train = labels.to(device)

torch.save(x_train, 'x_train.pt')
torch.save(y_train, 'y_train.pt')
