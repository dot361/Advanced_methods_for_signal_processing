# https://stackoverflow.com/questions/52913749/add-random-noise-with-specific-snr-to-a-signal

from scipy.fftpack import fft, fftfreq
import matplotlib.pyplot as plt
import scipy as sp
import numpy as np
from numpy import matlib as mb
# import plotly.graph_objects as go
import sys
from scipy import stats

import keras
from keras.models import Sequential
from keras.layers import Conv2D, Conv2DTranspose
from keras.constraints import max_norm
import tensorflow as tf
from cycler import cycler
import pandas as pd
from numpy import pi
from sklearn.decomposition import PCA
from numpy.testing import assert_array_almost_equal
np.set_printoptions(precision=3)

# Fiddle with figure settings here:
plt.rcParams['figure.figsize'] = (10,8)
plt.rcParams['font.size'] = 14
plt.rcParams['image.cmap'] = 'plasma'
plt.rcParams['axes.linewidth'] = 2
# Set the default colour cycle (in case someone changes it...)
cols = plt.get_cmap('tab10').colors
plt.rcParams['axes.prop_cycle'] = cycler(color=cols)

# A simple little 2D matrix plotter, excluding x and y labels.
def plot_2d(m, title=""):
    plt.imshow(m)
    plt.xticks([])
    plt.yticks([])
    plt.title(title)


"""
Performs KLT on a 2D matrix, returning the reconstructed signal matrix
:D: Each row corresponds to a single 1D signal
:K: Amount of dominant components selected to perform signal reconstruction
"""
# @profile
def KLT(D, k=1):
    mu_D = np.mean(D,axis=0)    # Mean vector
    # Subtract mean vector from each row in D
    B = D-mb.repmat(mu_D,m=np.shape(D)[0],n=1)   # m - number of times matrix is repeated along the first and second axis

    # Covariance matrix (D-mu_d).T * (D-mu_D)
    C = np.dot(B.T, B)
    # Extract eigenvalues and eigenvectors
    [eigenvals, V] = sp.linalg.eigh(C)  # Eigenvals are returned as an array not diag of matrix like in matlab
    # V is a square matrix where each column corresponds to 1 eigenvector of matrix C
    V = np.around(V, decimals=4)
    desc_val = -np.sort(-eigenvals)
    indices = -np.argsort(-eigenvals)
    
    # Order eigenvectors based on imporance to variance 
    eigenvect = V[indices]  
    selectedEig = eigenvect[0:k,:].T
    
    # Perform signal reconstruction based on selected eigenvectors
    return np.dot((np.dot(B,selectedEig)), selectedEig.T) + mb.repmat(mu_D, m=np.shape(D)[0], n=1) 


"""
Generates white noise based on https://stackoverflow.com/questions/52913749/add-random-noise-with-specific-snr-to-a-signal
:signal: 1D spectrum to apply noise to
:SNR: Amount of signal to nosie ratio to achieve in -dB
"""
def generateNoise(signal, SNR):
    snr = 10.0**(SNR/10.0)
    power = signal.var()
    n = power/snr
    return np.sqrt(n)*np.random.randn(len(signal))




def generateColoredNoise(sig,  SNR=-30, noise_type="white"):
    X_white = np.fft.rfft(np.random.randn(len(sig)))
    X_white = generateNoise(X_white, SNR)
    xf = fftfreq(len(X_white), T)
    xf = 2.0/len(sig) * np.abs(xf[0:(len(sig)//2)+1])
    noise_types = {'white': 1, 'blue':np.sqrt(xf), 'violet':xf, 'brownian': 1/np.where(xf == 0, float('inf'), xf), 'pink':1/np.where(xf == 0, float('inf'), np.sqrt(xf))}
    S = noise_types[noise_type]
    S = S / np.sqrt(np.mean(S**2))
    X_shaped = X_white * S
    return np.fft.irfft(X_shaped, len(sig))

def noise_psd(N, psd = lambda f: 1):
        X_white = np.fft.rfft(np.random.randn(N));
        S = psd(np.fft.rfftfreq(N))
        S = S / np.sqrt(np.mean(S**2))
        X_shaped = X_white * S;
        return np.fft.irfft(X_shaped);



"""
Generates a sine wave at a specific frequency, amount of points and sampling interval
:N: Amount of points necessary
:T: Sampling interval
:F: Sampling frequency
"""
def generateSignal(N=1024, T=1.0/800.0, F=50.0, extraF=0.0):
    x = np.linspace(0.0, N*T, N, endpoint=False)
    y = np.sin(F * 2.0*np.pi*x)
    if(extraF != 0.0):
        y = y+ np.sin(extraF * 2.0*np.pi*x)
    return x, y
    
"""
Returns the FFT and FFT frequency range for provided signal
:N: Amount of points necessary
:T: Sampling interval
:x: Input signal frequency range
:y: Input signal
"""
def fetchFFT(x,y,N,T):
    yf = fft(y)
    xf = fftfreq(N, T)[:N//2]
    # plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
    # plt.show()
    return xf, 2.0/N * np.abs(yf[0:N//2])

"""
Combines generateSignal and generateNoise functions in one for ease of use
:N: Amount of points necessary
:S: Amount of signals to generate
:T: Sampling interval
:freq: Sampling frequency
:SNR: Amount of signal to noise ratio to achieve in -dB
"""
def generateSignalMatrix(N=1024, S=5, T=1.0/800.0, F=60.0, SNR=-20, extraF=0.0):
    amplitude = []
    for _ in range(0,S):
        x, originalY = generateSignal(N=N,T=T, F=F,extraF=extraF)
        # noise = generateNoise(originalY, SNR)
        noise = generateNoise(originalY, SNR)
        amplitude.append(originalY+noise)
    return x, amplitude # x is the same for all signals

def generateColoredSignalMatrix(N=1024, S=5, T=1.0/800.0, F=60.0, SNR=-20, extraF=0.0, noise_type="white"):
    amplitude = []
    for _ in range(0,S):
        x, originalY = generateSignal(N=N,T=T, F=F,extraF=extraF)
        noise = generateColoredNoise(originalY, SNR, noise_type=noise_type)
        amplitude.append(originalY+noise)
    return x, amplitude # x is the same for all signals



def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())
def prd(predictions, targets):
    return np.sqrt((np.sum((predictions-targets) ** 2)) / np.sum(predictions**2)) * 100


def PSDGenerator(f):
    return lambda N: noise_psd(N, f)

@PSDGenerator
def white_noise(f):
    return 1

@PSDGenerator
def blue_noise(f):
    return np.sqrt(f)

@PSDGenerator
def violet_noise(f):
    return f

@PSDGenerator
def brownian_noise(f):
    return 1/np.where(f == 0, float('inf'), f)

@PSDGenerator
def pink_noise(f):
    return 1/np.where(f == 0, float('inf'), np.sqrt(f))


T = 1.0/800.0
N = 2048
S = 120
SNR = -20
F = 70.0
extraF = 0.0
# width, height = 128, 8
width, height = 256, 8  # 128, 8  


# @profile
def mainFun(RMSE_array):
    x, D = generateSignalMatrix(N=N, S=S, T=T, F=F, SNR=SNR,extraF=extraF)
    # x, D = generateColoredSignalMatrix(N=N, S=S, T=T, F=F, SNR=SNR,extraF=extraF, noise_type="pink")
    _, originalY = generateSignal(N=N,T=T, F=F,extraF=extraF)


    yf = np.mean(D,axis=0)
    xf_fft, yf_fft = fetchFFT(yf,yf,N,T)

    pca = PCA(n_components=120)
    pca.fit(D)
    X_train_pca = pca.transform(D)
    X_train_pca[:1] = 0
    X_projected = pca.inverse_transform(X_train_pca)

    yf = np.mean(X_projected,axis=0)
    xf, yf = fetchFFT(yf,yf,N,T)
    plt.figure(1)
    plt.subplot(121)
    plt.plot(xf,yf)
    plt.plot(xf_fft,yf_fft)
    plt.subplot(122)
    plt.plot(yf_fft-yf)
    plt.show()
    print('a')


RMSE_array = []

fig = plt.figure(1)

plt.subplot(321)
mainFun(RMSE_array)


SNR = -20
plt.subplot(322)
mainFun(RMSE_array)

SNR = -25
plt.subplot(323)
mainFun(RMSE_array)

SNR = -30
plt.subplot(324)
mainFun(RMSE_array)

SNR = -35
plt.subplot(325)
mainFun(RMSE_array)

SNR = -40
plt.subplot(326)
mainFun(RMSE_array)
# fig.subplots_adjust(top=0.955,
# bottom=0.06,
# left=0.11,
# right=0.9,
# hspace=0.29,
# wspace=0.2)

RMSE_array = np.array(RMSE_array).T
print(np.array(RMSE_array))

SNRs = [10,20,25,30,35,40]
plt.figure(2)
plt.subplot(121)
plt.plot(SNRs, RMSE_array[0][0], label="KLT")
plt.plot(SNRs, RMSE_array[0][1], label="FFT")
plt.plot(SNRs, RMSE_array[0][2], label="ML")
plt.plot(SNRs, RMSE_array[0][3], label="SSA")
plt.grid()
plt.legend()
plt.ylabel("RMSE")
plt.xlabel("SNR (dB)")

plt.subplot(122)
plt.plot(SNRs, RMSE_array[1][0], label="KLT")
plt.plot(SNRs, RMSE_array[1][1], label="FFT")
plt.plot(SNRs, RMSE_array[1][2], label="ML")
plt.plot(SNRs, RMSE_array[1][3], label="SSA")
plt.grid()
plt.legend()
plt.ylabel("PRD")
plt.xlabel("SNR (dB)")


plt.show()

print("END")
