## AG Analysis Functions

import pandas as pd
import numpy as np
import scipy as sp
import scipy.signal as sig
import matplotlib.pyplot as plt
from scipy import fftpack
from scipy.interpolate import interp1d
from matplotlib.offsetbox import AnchoredOffsetbox
from scipy import signal
import seaborn as sns
from scipy.fft import fft, fftfreq


# def data_dist(data):
#     data_to_plot = data
#     plt.figure(figsize=(15, 10))
#     sns.histplot(data_to_plot, x='Channel 0', label='Channel 0', color="blue", )
#     sns.histplot(data_to_plot, x='Channel 1', label='Channel 1', color="orange")
#     sns.histplot(data_to_plot, x='Channel 2', label='Channel 2', color="green")
#     sns.histplot(data_to_plot, x='Channel 3', label='Channel 3', color="red")
#     sns.histplot(data_to_plot, x='Channel 4', label='Channel 4', color="purple")
#     sns.histplot(data_to_plot, x='Channel 5', label='Channel 5', color="brown")
#     sns.histplot(data_to_plot, x='Channel 6', label='Channel 6', color="pink")
#     sns.histplot(data_to_plot, x='Channel 7', label='Channel 7', color="grey")
#     plt.xlabel("mV")
#     plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#     plt.show()

def mean_amplitude(data):
    peaks = signal.find_peaks(data)

def pltfft(data, fs):
    N = data.shape[0]
    # sample spacing
    T = 1.0 / fs
    x = np.linspace(0.0, N * T, N, endpoint=False)
    y = data[:, 1]

    yf = fft(y)
    xf = fftfreq(N, T)[:N // 2]

    plt.plot(xf, 2.0 / N * np.abs(yf[0:N // 2]))
    plt.grid()

    N = data.shape[0]
    # sample spacing
    T = 1.0 / fs
    x = np.linspace(0.0, N * T, N, endpoint=False)
    y = data[:, 1]

    yf = fft(y)
    xf = fftfreq(N, T)[:N // 2]

    plt.plot(xf, 2.0 / N * np.abs(yf[0:N // 2]))
    return

