import keras
from keras.models import Sequential
from keras.layers import Conv2D, Conv2DTranspose
from keras.constraints import max_norm
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.fftpack import fft, fftfreq

from tensorflow.python.client import device_lib
from cycler import cycler

# Fiddle with figure settings here:
plt.rcParams['figure.figsize'] = (10,8)
plt.rcParams['font.size'] = 14
plt.rcParams['image.cmap'] = 'plasma'
plt.rcParams['axes.linewidth'] = 2
# Set the default colour cycle (in case someone changes it...)
cols = plt.get_cmap('tab10').colors
plt.rcParams['axes.prop_cycle'] = cycler(color=cols)

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
        x, y = generateSignal(N=N,T=T, F=F,extraF=extraF)
        noise = generateNoise(y, SNR)
        amplitude.append(y+noise)
    return x, amplitude # x is the same for all signals

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

T = 1.0/800.0
N = 2048
S = 120
SNR = -15
F = 30.0
extraF = 0.0
# width, height = 128, 8
width, height = 256, 8  # 128, 8  
# input_shape = (width, height, 1)
model = keras.models.load_model('2048_100k_20-60hz-model-10')

noiseList = np.arange(-40,-9)
total_rez = []
for i in range(0,1001):
    iter_rez = []
    print(i)
    for noise in noiseList:
        xs, ys = generateSignal(N=N, T=T, F=F)
        noise = generateNoise(ys, noise)

        sig = ys + noise
        sig = sig.reshape((1, width, height, 1))
        reconstructions = model.predict(sig).reshape((width * height,))

        reconstructions = reconstructions-np.mean(reconstructions)

        purex, purey = fetchFFT(ys,ys,N,T)

        xf, yf = fetchFFT(ys+noise, ys+noise, N, T)
        sigma_FFT = np.max(yf)/np.mean(yf)
        rmse_rez_FFT = rmse(yf, purey)
        xf, yf = fetchFFT(reconstructions, reconstructions, N, T)
        sigma_ML = np.max(yf)/np.mean(yf)
        rmse_rez_ML = rmse(yf, purey)
        iter_rez.append([sigma_FFT, rmse_rez_FFT, sigma_ML, rmse_rez_ML])
    iter_rez = np.array(iter_rez).T

    total_rez.append(iter_rez)


total_rez = np.mean(total_rez,axis=0)

plt.subplot(121)

# plt.axvline(x=20, color='black', ls='--', linewidth=1, alpha=0.5)
# plt.axvline(x=30, color='black', ls='--', linewidth=1, alpha=0.5)
# plt.axvline(x=40, color='black', ls='--', linewidth=1, alpha=0.5)
# plt.axvline(x=50, color='black', ls='--', linewidth=1, alpha=0.5)
# plt.axvline(x=60, color='black', ls='--', linewidth=1, alpha=0.5)

xax = np.arange(-40,-9,step=1)

plt.plot(xax, total_rez[0], label='FFT')
plt.plot(xax, total_rez[2], label='ML ')
plt.xlabel("dB")
plt.ylabel(r"$\sigma$")
plt.grid()
plt.legend(loc="upper right", prop={'size': 10} )
# plt.xticks(np.arange(0,90,step=10))
# plt.axis([0,90,2.5,22.5])

# ymin, ymax = plt.ylim()
# plt.fill_between(np.arange(20,61), ymin, ymax, color='green', alpha=0.3)
plt.subplot(122)

# plt.axvline(x=20, color='black', ls='--', linewidth=1, alpha=0.5)
# plt.axvline(x=30, color='black', ls='--', linewidth=1, alpha=0.5)
# plt.axvline(x=40, color='black', ls='--', linewidth=1, alpha=0.5)
# plt.axvline(x=50, color='black', ls='--', linewidth=1, alpha=0.5)
# plt.axvline(x=60, color='black', ls='--', linewidth=1, alpha=0.5)
# plt.plot(np.arange(5,90), total_rez[1], label='FFT')
plt.plot(xax, total_rez[1],label='FFT')
plt.plot(xax, total_rez[3], label='ML ')
plt.xlabel("dB")
plt.ylabel(r"$RMSE$")
plt.grid()
plt.tight_layout()
# plt.legend(loc="lower right", prop={'size': 10} )
# plt.xticks(np.arange(0,90,step=10))
# plt.tight_layout()
# # xmin xmax ymin ymax
# plt.axis([0,90,0,0.2])

# ymin, ymax = plt.ylim()
# plt.fill_between(np.arange(20,61), ymin, ymax, color='green', alpha=0.3)

plt.show()

# xax = np.arange(-40,-9,step=1)
# plt.subplot(221)
# plt.plot(xax, total_rez[0], label='FFT')
# plt.plot(xax, total_rez[2], label='ML ')
# plt.xlabel("dB")
# plt.ylabel(r"$\sigma$")
# plt.grid()
# plt.legend(loc="upper right", prop={'size': 10} )
# plt.subplot(222)
# plt.plot(xax, total_rez[1],label='FFT')
# plt.plot(xax, total_rez[3], label='ML ')
# plt.xlabel("dB")
# plt.ylabel(r"$RMSE$")
# plt.grid()

# plt.subplot(212)
# plt.plot(xax, total_rez[3], label='ML ')
# plt.xlabel("dB")
# plt.ylabel(r"$RMSE$")
# plt.grid()
# plt.tight_layout()

print('hold')


