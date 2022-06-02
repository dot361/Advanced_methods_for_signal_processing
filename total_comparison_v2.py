# https://stackoverflow.com/questions/52913749/add-random-noise-with-specific-snr-to-a-signal

from scipy.fftpack import fft, fftfreq
import matplotlib.pyplot as plt
import scipy as sp
import numpy as np
from numpy import matlib as mb
import plotly.graph_objects as go
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
from numba import jit, prange
import numba as nb

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
    mu_D = np.mean(D,axis=0)
    B = D-mb.repmat(mu_D,m=np.shape(D)[0],n=1)
    C = np.dot(B.T, B)
    [eigenvals, V] = sp.linalg.eigh(C)
    print(np.shape(eigenvals))
    print("eigenvals: ", eigenvals)
    indices = -np.argsort(-eigenvals)
    eigenvect = V[indices]  
    selectedEig = eigenvect.T[:,k].T
    selectedEig = selectedEig[:,np.newaxis]
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

@nb.njit(parallel=True)
def performSSA(orig_TS : np.array, L : int) -> np.array:
    N = orig_TS.size
    K = N - L + 1
    # Embed the time series in a trajectory matrix
    X = np.zeros((L,K))
    for i in prange(0,K):
        X[:,i] = orig_TS[i:L+i]
    
    # Decompose the trajectory matrix
    U, Sigma, VT = np.linalg.svd(X)
    d = np.linalg.matrix_rank(X)
    TS_comps = np.zeros((N, d))

    for i in prange(d):
        X_elem = Sigma[i]*np.outer(U[:,i], VT[i,:])
        X_rev = X_elem[::-1]
        idx = 0
        for j in prange(-X_rev.shape[0]+1, X_rev.shape[1]):
            TS_comps[idx,i] = np.diag(X_rev,j).mean()
            idx+=1 

    return TS_comps

def reconstruct(TS_comps, indices):
    return np.array(TS_comps[:,indices].sum(axis=1))


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
SNR = -15
F = 50.0
extraF = 0.0
# width, height = 128, 8
width, height = 256, 8  # 128, 8  




def mainFun(RMSE_array, sigma_array, model):
    x, y =generateSignal(N,T,F)
    # noise = generateNoise(y, SNR)
    if(noise_type == 'white'):
        noise = generateNoise(y, SNR)
    else:
        noise = generateColoredNoise(y,SNR,noise_type=noise_type)
    y+= noise
    # x, D = generateSignalMatrix(N=N, S=S, T=T, F=F, SNR=SNR,extraF=extraF)
    # x, D = generateColoredSignalMatrix(N=N, S=S, T=T, F=F, SNR=SNR,extraF=extraF, noise_type="pink")
    _, originalY = generateSignal(N=N,T=T, F=F,extraF=extraF)
    fft_originalX, fft_originalY = fetchFFT(originalY, originalY, N, T)
    xpeak = np.argmax(fft_originalY)

    # fig = go.Figure()
    iter_rmse = []
    iter_sigma = []
    print(SNR)
    # fft_y = []
    # for i in D:
    #     org_x, org_y, = fetchFFT(x,i,N,T)
    #     fft_y.append(org_y)
    fft_x, fft_y = fetchFFT(x,y,N,T)

    # fig.add_trace(go.Scatter(x=fft_x, y=fft_y, name='FFT'))
    sigma_FFT = np.max(fft_y[xpeak-3:xpeak+3])/np.mean(fft_y)
    rmse_rez_FFT = rmse(fft_y, fft_originalY)
    plt.plot(fft_x,fft_y, label='FFT '+r"$\sigma$ = "+str(np.round(sigma_FFT)))

    iter_rmse.append(rmse_rez_FFT)
    iter_sigma.append(sigma_FFT)

    # reconstr = KLT(D=D,k=0)
    # # meanyf = []
    # yf = np.mean(np.array(reconstr),axis=0)
    # xf, meanyf = fetchFFT(x, yf, N, T)

    
    # sigma_KL = np.max(meanyf[xpeak-3:xpeak+3])/np.mean(meanyf)
    # rmse_rez_KL = rmse(meanyf, fft_originalY)

    # iter_rmse.append(rmse_rez_KL)
    # iter_sigma.append(sigma_KL)


    reconstruction = model.predict(y.reshape((1, width, height, 1))).reshape((width * height,))
    print(reconstruction.shape)
    reconstruction = np.array(reconstruction).reshape((width * height,))
    reconstruction -= np.mean(reconstruction)

    fft_ML_x, fft_ML_y = fetchFFT(reconstruction, reconstruction, N, T)


    max_idx = np.argmax(fft_originalY)
    delta = np.max(fft_originalY)/np.max(fft_ML_y[max_idx-3:max_idx+3])
    fft_ML_y *= delta
    # fig.add_trace(go.Scatter(x=fft_ML_x, y=fft_ML_y, name='ML'))

    sigma_ML = np.max(fft_ML_y[xpeak-3:xpeak+3])/np.mean(fft_ML_y)
    rmse_rez_ML = rmse(fft_ML_y, fft_originalY)
    plt.plot(fft_ML_x, fft_ML_y, label='ML'+r"$\sigma$ = "+str(np.round(sigma_ML)))

    iter_rmse.append(rmse_rez_ML)
    iter_sigma.append(sigma_ML)


    #matrix
    #256, 6
    # sig = np.mean(D,axis=0)
    SSA = performSSA(y,256)
    ssa_rez = reconstruct(SSA, slice(0,16))
    xf, yf = fetchFFT(ssa_rez, ssa_rez, N, T)
    # fig.add_trace(go.Scatter(x=xf, y=yf, name='SSA 256,6'))

    sigma_SSA = np.max(yf[xpeak-3:xpeak+3])/np.mean(yf)
    rmse_rez_SSA = rmse(yf, fft_originalY)
    plt.plot(xf,yf, label='SNR 256,6'+r"$\sigma$ = "+str(np.round(sigma_SSA)))

    iter_rmse.append(rmse_rez_SSA)
    iter_sigma.append(sigma_SSA)


    #matrix
    #8, 6
    # sig = np.mean(D,axis=0)
    SSA = performSSA(y,8)
    ssa_rez = reconstruct(SSA, slice(0,6))
    xf, yf = fetchFFT(ssa_rez, ssa_rez, N, T)
    # fig.add_trace(go.Scatter(x=xf, y=yf, name='SSA 8,6'))

    sigma_SSA = np.max(yf[xpeak-3:xpeak+3])/np.mean(yf)
    rmse_rez_SSA = rmse(yf, fft_originalY)
    plt.plot(xf,yf, label='SNR 8,6'+r"$\sigma$ = "+str(np.round(sigma_SSA)))

    iter_rmse.append(rmse_rez_SSA)
    iter_sigma.append(sigma_SSA)

    #matrix
    #8, 2
    # sig = np.mean(D,axis=0)
    SSA = performSSA(y,8)
    ssa_rez = reconstruct(SSA, slice(0,2))
    xf, yf = fetchFFT(ssa_rez, ssa_rez, N, T)
    # fig.add_trace(go.Scatter(x=xf, y=yf, name='SSA 8,2'))

    sigma_SSA = np.max(yf[xpeak-3:xpeak+3])/np.mean(yf)
    rmse_rez_SSA = rmse(yf, fft_originalY)
    plt.plot(xf,yf, label='SNR 8,2'+r"$\sigma$ = "+str(np.round(sigma_SSA)))

    iter_rmse.append(rmse_rez_SSA)
    iter_sigma.append(sigma_SSA)

    RMSE_array.append(iter_rmse)
    sigma_array.append(iter_sigma)

    # plt.show()
    # fig.show()

total_rmse = []
total_sigma = []
model = keras.models.load_model('./keras-autoencoders-master/2048_100k_20-60hz-model-10')


noise_types = ['blue','violet','brownian', 'pink']

for snr_val in [-15]:
    SNR = snr_val
    for freqHz in [50]:
        F = freqHz
        for j in range(0,101):
            RMSE_array = []
            sigma_array =[]
            print(j)
            # for i in range(-20,-9,1):
            subplot = 0
            plt.figure(str(F)+'Hz'+str(SNR))
            for i in noise_types:
                subplot+=1

                plt.subplot(2,2,subplot)
                plt.title(i)
                # SNR = i
                noise_type = i
                mainFun(RMSE_array,sigma_array, model)
                plt.legend( loc='upper right', prop={'size': 8})
                plt.axvline(x=F, color='black', ls='--', linewidth=1, alpha=0.5)
                plt.ylabel('Amplitude')
                plt.xlabel('Frequency [Hz]')
                plt.grid()
                plt.minorticks_on()
                plt.tight_layout()
            plt.show()
            total_rmse.append(RMSE_array)
            total_sigma.append(sigma_array)
            # np.save('rmse-noise-types-'+str(F)+str(snr_val), np.array(total_rmse))
            # np.save('sigma-noise-types-'+str(F)+str(snr_val), np.array(total_sigma))

total_rmse = np.array(total_rmse)
total_sigma = np.array(total_sigma)

total_rmse = np.mean(total_rmse,axis=0)
total_sigma = np.mean(total_sigma,axis=0)

labels = ["FFT","KLT", "ML", "SSA 256,6", "SSA 8,6", "SSA 8,2"]
plt.subplot(121)
plt.plot(np.arange(-40,-9),total_rmse)
plt.xlabel("dB")
plt.ylabel("RMSE")
plt.grid()
plt.legend(labels)
plt.subplot(122)
plt.xlabel("dB")
plt.ylabel(r"$\sigma$")
plt.plot(np.arange(-40,-9),total_sigma)
plt.legend(labels)
plt.grid()
plt.tight_layout()
plt.show()
print("END")