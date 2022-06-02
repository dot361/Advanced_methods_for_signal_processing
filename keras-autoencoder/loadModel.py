import keras
from keras.models import Sequential
from keras.layers import Conv2D, Conv2DTranspose
from keras.constraints import max_norm
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.fftpack import fft, fftfreq
from scipy.signal import hilbert
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
def generateSignal(N=1024, T=1.0/800.0, F=50.0, extraF=0.0, isComplex=False):
    x = np.linspace(0.0, N*T, N, endpoint=False)
    y = np.sin(F * 2.0*np.pi*x)
    if(isComplex):
        y = hilbert(y)
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

xs, ys = generateSignal(N=N, T=T, F=F)
fy = np.fft.fft(ys)
plt.plot(fy)
xs, ys = generateSignal(N=N, T=T, F=F, isComplex=False)
fy = np.fft.fft(ys)
plt.plot(fy)
# plt.show()
noise = generateNoise(ys, SNR)

sig = ys + noise

plt.plot(sig)
plt.show()

sig = sig.reshape((1, width, height, 1))

model = keras.models.load_model('2048_100k_20-60hz-model-10')

reconstructions = model.predict(sig).reshape((width * height,))


plt.subplot(211)
plt.yscale('symlog')
plt.title("F="+str(F)+", SNR="+str(-SNR))

plt.plot(ys+noise, label="Original")
plt.plot(reconstructions-np.mean(reconstructions), label="Reconstructed")
plt.ylabel(r"$symlog$(Amplitude)")
plt.xlabel("Time")
plt.grid()
plt.legend(loc="upper right", prop={'size': 10} )
plt.subplot(212)

reconstructions = reconstructions-np.mean(reconstructions)

purex, purey = fetchFFT(ys,ys,N,T)


# plt.title("Fourier")
plt.axvline(x=F, color='black', ls='--')
xf, meanyf = fetchFFT(ys+noise, ys+noise, N, T)



sigma = np.max(meanyf)/np.mean(meanyf)
rmse_rez = rmse(meanyf, purey)


plt.plot(xf, meanyf, label=r"FFT $\sigma$ = "+str(np.round(sigma,decimals=3)) + r", $RMSE$ = " + str(np.round(rmse_rez,decimals=3)))
# xf, meanyf = fetchFFT(reconstructions, reconstructions, N, T)
# plt.plot(xf, meanyf, label="ML")
xf, meanyf = fetchFFT(reconstructions, reconstructions, N, T)

max_idx = np.argmax(purey)
delta = np.max(purey)/np.max(meanyf[max_idx-3:max_idx+3])
meanyf *= delta


sigma = np.max(meanyf)/np.mean(meanyf)




rmse_rez = rmse(meanyf, purey)
plt.plot(xf, meanyf, label=r"ML  $\sigma$ = "+str(np.round(sigma,decimals=3)) + r", $RMSE$ = " + str(np.round(rmse_rez,decimals=3)))
# plt.plot(purex,purey, label='Pure sine')
plt.grid()
plt.xlabel("Frequency [Hz]")
plt.ylabel("Amplitude")
plt.legend(loc="upper right", prop={'size': 10} )
plt.tight_layout()
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
  
