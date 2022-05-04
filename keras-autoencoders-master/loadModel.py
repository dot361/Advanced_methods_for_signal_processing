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


T = 1.0/800.0
N = 2048
S = 120
SNR = -20
F = 30.0
extraF = 0.0
# width, height = 128, 8
width, height = 256, 8  # 128, 8  
# input_shape = (width, height, 1)

xs, ys = generateSignal(N=N, T=T, F=F)
noise = generateNoise(ys, SNR)

sig = ys + noise
sig = sig.reshape((1, width, height, 1))

model = keras.models.load_model('10_25SNR_14EPOCH_2048N')

reconstructions = model.predict(sig).reshape((width * height,))

plt.figure(1)
plt.plot(ys+noise)
plt.plot(reconstructions)
plt.figure(2)
plt.title("F="+str(F)+", SNR="+str(-SNR))
plt.axvline(x=F, color='black', ls='--')
xf, meanyf = fetchFFT(ys+noise, ys+noise, N, T)
plt.plot(xf, meanyf, label="FFT")
xf, meanyf = fetchFFT(reconstructions, reconstructions, N, T)
plt.plot(xf, meanyf, label="ML")
plt.grid()
plt.xlabel("Frequency")
plt.ylabel("Amplitude")
plt.legend()
plt.show()



# Plot reconstructions
for i in np.arange(0, num_reconstructions):
  # Prediction index
  prediction_index = i + percentage_training
  # Get the sample and the reconstruction
  original = y_val_noisy[prediction_index]
  pure = y_val_pure[prediction_index]
  reconstruction = np.array(reconstructions[i]).reshape((width * height,))
  # Matplotlib preparations
  fig, axes = plt.subplots(1, 3)
  # Plot sample and reconstruciton
  axes[0].plot(original)
  axes[0].set_title('Noisy waveform')
  axes[1].plot(pure)
  axes[1].set_title('Pure waveform')
  axes[2].plot(reconstruction)
  axes[2].set_title('Conv Autoencoder Denoised waveform')
  
  plt.figure(2)
  plt.subplot(131)
  xf, meanyf = fetchFFT(original, original, N, T)
  plt.plot(xf, meanyf)
  plt.subplot(132)
  xf, meanyf = fetchFFT(pure, pure, N, T)
  plt.plot(xf, meanyf)
  plt.subplot(133)
  xf, meanyf = fetchFFT(reconstruction, reconstruction, N, T)
  plt.plot(xf, meanyf)
  plt.show()
  
