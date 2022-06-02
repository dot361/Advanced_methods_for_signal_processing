from unittest import result
import keras
from keras.models import Sequential
from keras.layers import Conv2D, Conv2DTranspose
from keras.constraints import max_norm
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.fftpack import fft, fftfreq
import scipy as sp

from tensorflow.python.client import device_lib
from cycler import cycler
from numpy import matlib as mb

from numba import jit, prange
import numba as nb
import seaborn as sns

import time

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


def KLT(D, k=1):
    mu_D = np.mean(D,axis=0)
    B = D-mb.repmat(mu_D,m=np.shape(D)[0],n=1)
    C = np.dot(B.T, B)
    [eigenvals, V] = sp.linalg.eigh(C)
    # print(np.shape(eigenvals))
    # print("eigenvals: ", eigenvals)
    indices = -np.argsort(-eigenvals)
    eigenvect = V[indices]  
    selectedEig = eigenvect.T[:,k].T
    selectedEig = selectedEig[:,np.newaxis]
    return np.dot((np.dot(B,selectedEig)), selectedEig.T) + mb.repmat(mu_D, m=np.shape(D)[0], n=1) 


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

T = 1.0/800.0
N = 2048
S = 120
SNR = -15
F = 30.0
extraF = 0.0
# width, height = 128, 8
width, height = 256, 8  # 128, 8  
input_shape = (width, height, 1)
model = keras.models.load_model('./keras-autoencoders-master/2048_100k_20-60hz-model-10')

noiseList = np.arange(-40,-9)
# noiseList = np.arange(-10,-41, step=-1)
total_rez = []
purex, purey = generateSignal(N=N,T=T, F=F,extraF=extraF)
pure_xf, pure_fy = fetchFFT(purey,purey,N,T)


ssa_rez_init = performSSA(purey,8)
total_benchmarks = []

for i in range(0,1001):
    iter_rez = []
    iter_benchmarks = []
    print(i)
    for noise in noiseList:
        x, D = generateSignalMatrix(N=N, S=S, T=T, F=F, SNR=noise, extraF=extraF)

        matrix_results = []
        single_results = []

        matrix_benchmarks = []
        single_benchmarks = []


        #FFT
        #single
        t0 = time.time()
        xf, yf = fetchFFT(D[0], D[0], N, T)
        t1 = time.time()
        total = t1-t0
        single_benchmarks.append(total)
        plt.figure(1)
        plt.plot(xf,yf, label='FFT')
        sigma_FFT = np.max(yf)/np.mean(yf)
        rmse_rez_FFT = rmse(yf, pure_fy)
        single_results.append(np.array([rmse_rez_FFT,sigma_FFT]))

        #matrix
        sig =  np.average(D,axis=0)
        t0 = time.time()
        xf, yf = fetchFFT(sig, sig, N, T)

        FFT_matrix_x, FFT_matrix_y = fetchFFT(sig, sig, N, T)


        t1 = time.time()
        total = t1-t0
        plt.figure(2)
        plt.subplot(231)
        plt.plot(xf,yf, label='FFT')
        plt.title("FFT")
        plt.ylim(top=1.0,bottom=-0.05)
        matrix_benchmarks.append(total)
        sigma_FFT = np.max(yf)/np.mean(yf)
        rmse_rez_FFT = rmse(yf, pure_fy)
        matrix_results.append(np.array([rmse_rez_FFT,sigma_FFT]))

        #ML
        #single
        sig = D[0].reshape((1, width, height, 1))
        t0 = time.time()
        reconstructions = model.predict(sig).reshape((width * height,))
        reconstructions = reconstructions-np.mean(reconstructions)
        t1 = time.time()
        total = t1-t0
        single_benchmarks.append(total)

        xf, yf = fetchFFT(reconstructions, reconstructions, N, T)
        plt.figure(1)
        plt.plot(xf,yf, label='ML')
        sigma_ML = np.max(yf)/np.mean(yf)
        rmse_rez_ML = rmse(yf, pure_fy)
        single_results.append(np.array([rmse_rez_ML, sigma_ML]))

        #matrix
        sig = np.average(D,axis=0)
        sig = sig.reshape((1, width, height, 1))
        t0 = time.time()
        reconstructions = model.predict(sig).reshape((width * height,))
        reconstructions = reconstructions-np.mean(reconstructions)
        t1 = time.time()
        total = t1-t0
        matrix_benchmarks.append(total)
        xf, yf = fetchFFT(reconstructions, reconstructions, N, T)
        plt.figure(2)

        plt.subplot(232)
        plt.title("ML")
        plt.plot(FFT_matrix_x, FFT_matrix_y)
        plt.plot(xf,yf, label='ML')
        plt.ylim(top=1.0,bottom=-0.05)

        sigma_ML = np.max(yf)/np.mean(yf)
        rmse_rez_ML = rmse(yf, pure_fy)
        matrix_results.append(np.array([rmse_rez_ML, sigma_ML]))


        #KLT
        t0 = time.time()

        reconstr = KLT(D=D,k=0)
        # meanyf = []
        mean_klt = np.mean(np.array(reconstr),axis=0)
        t1 = time.time()
        total = t1-t0
        matrix_benchmarks.append(total)
        temp_xf, temp_yf = fetchFFT(mean_klt, mean_klt, N, T)
        plt.figure(3)
        plt.plot(temp_xf, yf-temp_yf)


        xf, yf = fetchFFT(mean_klt, mean_klt, N, T)
        
        plt.figure(2)
        plt.subplot(233)
        plt.title("KLT")

        plt.plot(FFT_matrix_x, FFT_matrix_y)
        plt.plot(temp_xf, temp_yf, label='KLT')

        plt.ylim(top=1.0,bottom=-0.05)

        sigma_KLT = np.max(yf)/np.mean(yf)
        rmse_rez_KLT = rmse(yf, pure_fy)
        matrix_results.append(np.array([rmse_rez_KLT, sigma_KLT]))



        #SSA 256,2
        #single
        t0 = time.time()
        SSA = performSSA(D[0],256)
        ssa_rez = reconstruct(SSA, slice(0,2))
        t1 = time.time()
        total = t1-t0
        single_benchmarks.append(total)

        xf, yf = fetchFFT(ssa_rez, ssa_rez, N, T)
        plt.figure(1)
        plt.plot(xf,yf, label='SSA 256,2')
        sigma_SSA = np.max(yf)/np.mean(yf)
        rmse_rez_SSA = rmse(yf, pure_fy)
        single_results.append(np.array([rmse_rez_SSA,sigma_SSA]))

        #matrix
        sig = np.mean(D,axis=0)
        t0 = time.time()
        SSA = performSSA(sig,256)
        ssa_rez = reconstruct(SSA, slice(0,2))
        t1 = time.time()
        total = t1-t0
        matrix_benchmarks.append(total)
        xf, yf = fetchFFT(ssa_rez, ssa_rez, N, T)
        plt.figure(2)
        plt.subplot(234)
        plt.title("SSA L=256,N=2")
        plt.plot(FFT_matrix_x, FFT_matrix_y)

        plt.plot(xf,yf, label='SSA 256,2')
        plt.ylim(top=1.0,bottom=-0.05)

        sigma_SSA = np.max(yf)/np.mean(yf)
        rmse_rez_SSA = rmse(yf, pure_fy)
        matrix_results.append(np.array([rmse_rez_SSA,sigma_SSA]))


        #SSA 8,6
        #single
        t0 = time.time()

        SSA = performSSA(D[0],8)
        ssa_rez = reconstruct(SSA, slice(0,6))
        t1 = time.time()
        total = t1-t0
        single_benchmarks.append(total)
        xf, yf = fetchFFT(ssa_rez, ssa_rez, N, T)
        plt.figure(1)
        plt.plot(xf,yf, label='SSA 8,6')
        sigma_SSA = np.max(yf)/np.mean(yf)
        rmse_rez_SSA = rmse(yf, pure_fy)
        single_results.append(np.array([rmse_rez_SSA,sigma_SSA]))

        #matrix
        sig = np.mean(D,axis=0)
        t0 = time.time()
        SSA = performSSA(sig,8)
        ssa_rez = reconstruct(SSA, slice(0,6))
        t1 = time.time()
        total = t1-t0
        matrix_benchmarks.append(total)
        xf, yf = fetchFFT(ssa_rez, ssa_rez, N, T)
        plt.figure(2)
        plt.subplot(235)
        plt.title("SSA L=8,N=6")
        plt.plot(FFT_matrix_x, FFT_matrix_y)

        plt.plot(xf,yf, label='SSA 8,6')
        plt.ylim(top=1.0,bottom=-0.05)

        sigma_SSA = np.max(yf)/np.mean(yf)
        rmse_rez_SSA = rmse(yf, pure_fy)
        matrix_results.append(np.array([rmse_rez_SSA,sigma_SSA]))


        #SSA 8,2
        #single
        t0 = time.time()

        SSA = performSSA(D[0],8)
        ssa_rez = reconstruct(SSA, slice(0,2))
        t1 = time.time()
        total = t1-t0
        single_benchmarks.append(total)

        xf, yf = fetchFFT(ssa_rez, ssa_rez, N, T)
        plt.figure(1)
        plt.plot(xf,yf, label='SSA 8,2')
        plt.legend()
        sigma_SSA = np.max(yf)/np.mean(yf)
        rmse_rez_SSA = rmse(yf, pure_fy)
        single_results.append(np.array([rmse_rez_SSA,sigma_SSA]))

        #matrix
        sig = np.mean(D,axis=0)
        t0 = time.time()

        SSA = performSSA(sig,8)
        ssa_rez = reconstruct(SSA, slice(0,2))
        t1 = time.time()
        total = t1-t0
        matrix_benchmarks.append(total)
        xf, yf = fetchFFT(ssa_rez, ssa_rez, N, T)
        plt.figure(2)
        plt.subplot(236)
        plt.title("SSA L=8,N=2")
        plt.plot(FFT_matrix_x, FFT_matrix_y)

        plt.plot(xf,yf, label='SSA 8,2')
        plt.ylim(top=1.0,bottom=-0.05)

        # plt.legend()
        sigma_SSA = np.max(yf)/np.mean(yf)
        rmse_rez_SSA = rmse(yf, pure_fy)
        matrix_results.append(np.array([rmse_rez_SSA,sigma_SSA]))

        plt.show()
        iter_rez.append(np.array([np.array(matrix_results), np.array(single_results)]))
        iter_benchmarks.append(np.array([np.array(matrix_benchmarks), np.array(single_benchmarks)]))

        # xf, yf = fetchFFT(ys+noise, ys+noise, N, T)
        # sigma_FFT = np.max(yf)/np.mean(yf)
        # rmse_rez_FFT = rmse(yf, purey)
        # xf, yf = fetchFFT(reconstructions, reconstructions, N, T)
        # sigma_ML = np.max(yf)/np.mean(yf)
        # rmse_rez_ML = rmse(yf, purey)
        # iter_rez.append([sigma_FFT, rmse_rez_FFT, sigma_ML, rmse_rez_ML])
    iter_rez = np.array(iter_rez)
    iter_benchmarks = np.array(iter_benchmarks)
    total_rez.append(iter_rez)
    total_benchmarks.append(iter_benchmarks)
#     np.save('total_rez', total_rez)
#     np.save('total_benchmarks', total_benchmarks)



total_rez = np.load('total_benchmarks.npy', allow_pickle=True)






# total_rez = np.load('total_rez.npy', allow_pickle=True)

# total_rez = np.mean(total_rez,axis=0)


methods = ['FFT', 'ML', 'KLT', 'SSA 256,2', 'SSA 8,6', 'SSA 8,2']


result_bench = []
for i in range(0,len(total_rez)):
    for rez , _ in total_rez[i]:
        new_total_rez = np.array(rez)
        result_bench.append(new_total_rez)
result_bench = np.array(result_bench)


bp = plt.boxplot(result_bench, vert=False, labels=methods,
                  notch=True, patch_artist=True,
                  flierprops={'marker': 'o', 'markersize': 5})
plt.grid()
plt.xlabel("Time [s]")
plt.tight_layout()
plt.show()
medians = [round(item.get_ydata()[0], 1) for item in bp['medians']]
print(medians)
single_rez = total_rez[:,1]
single_snrs = [[],[]]
for snr_rez in range(0,len(single_rez)):
    single_snrs[0].append(single_rez[snr_rez].T[0])
    single_snrs[1].append(single_rez[snr_rez].T[1])

single_snrs = np.array(single_snrs)
plt.subplot(121)
plt.plot(single_snrs[0])
plt.subplot(122)
plt.plot(single_snrs[1])
plt.legend(methods)
plt.show()

for freq_rez in total_rez:
    single_rez = freq_rez[0]
    single_sigma = single_rez.T[0]
    single_rmse = single_rez.T[1]
    
    matrix_rez = freq_rez[1]
    matrix_sigma = matrix_rez[0]
    matrix_rmse = matrix_rez.T[1]


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


