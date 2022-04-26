# https://stackoverflow.com/questions/52913749/add-random-noise-with-specific-snr-to-a-signal

from scipy.fftpack import fft, fftfreq
import matplotlib.pyplot as plt
import scipy as sp
import numpy as np
from numpy import matlib as mb
import plotly.graph_objects as go
import sys
from scipy import stats

np.set_printoptions(precision=3)

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
    indices = -np.argsort(-eigenvals)
    eigenvect = V[indices]  
    selectedEig = eigenvect[0:k,:].T
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

"""
Generates a sine wave at a specific frequency, amount of points and sampling interval
:N: Amount of points necessary
:T: Sampling interval
:F: Sampling frequency
"""
def generateSignal(N=1024, T=1.0/800.0, F=50.0):
    x = np.linspace(0.0, N*T, N, endpoint=False)
    y = np.sin(F * 2.0*np.pi*x)
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
def generateSignalMatrix(N=1024, S=5, T=1.0/800.0, F=60.0, SNR=-20):
    amplitude = []
    for _ in range(0,S):
        x, y = generateSignal(N=N,T=T, F=F)
        noise = generateNoise(y, SNR)
        amplitude.append(y+noise)
    return x, amplitude # x is the same for all signals


T = 1.0/800.0
N = 2048
S = 120
SNR = -10
F = 60.0

print(N,S)
# plt.rc('font', family='serif')
# plt.rc('xtick', labelsize='x-small')
# plt.rc('ytick', labelsize='x-small')


# x,y = generateSignal(N=1024,T=T,F=F)
# noise = generateNoise(signal=y, SNR=-5)
# plt.plot(x,y+noise, label="Noisy signal")
# plt.plot(x,y, label="Initial signal")
# plt.title("N=1024, T=1.0/600.0, F=60, SNR=-5db")
# plt.xlabel("Time")
# plt.ylabel("Amplitude")
# plt.minorticks_on()
# plt.legend()
# plt.grid()
# plt.show()

# @profile
def mainFun():
    x, D = generateSignalMatrix(N=N, S=S, T=T, F=F, SNR=SNR)
    print(SNR)
    fft_y = []
    for i in D:
        org_x, org_y, = fetchFFT(x,i,N,T)
        fft_y.append(org_y)
    fft_y = np.mean(np.array(fft_y),axis=0)
    reconstr = KLT(D=D,k=1)
    meanyf = []
    for idx,i in enumerate(reconstr):
        xf, yf = fetchFFT(x,i,N,T)
        meanyf.append(yf)
    meanyf = np.mean(np.array(meanyf),axis=0)
    # meanyf = meanyf - np.mean(meanyf)
    # fft_y = fft_y - np.mean(fft_y)
    
    maxpeak_range = np.arange(154-5, 154+5, 1, dtype=int)
    KLT_max = np.argmax(meanyf[maxpeak_range])+149
    FFT_max = np.argmax(fft_y[maxpeak_range])+149

    FFT_z = stats.zscore(fft_y)
    KLT_z = stats.zscore(meanyf)
    
    # print(meanyf[KLT_max], np.mean(meanyf))


    # plt.plot(xf, meanyf, label="KLT (σ="+str(np.around(KLT_z[KLT_max],decimals=3))+")")
    # plt.plot(org_x, fft_y, label="FFT (σ="+str(np.around(FFT_z[FFT_max],decimals=3))+")")

    # plt.plot(xf[KLT_max], meanyf[KLT_max],marker='x',color='blue')
    # plt.plot(xf[FFT_max], fft_y[FFT_max],marker='x',color='red')
    

    # plt.title("N="+str(N)+", T=1.0/800.0, F=60, SNR="+str(SNR)+"db")

    # plt.xlabel("Frequency (Hz)")
    # plt.ylabel("Amplitude")
    # plt.minorticks_on()
    # plt.legend()
    # plt.grid()
    # plt.show()
    return [SNR,  meanyf[KLT_max]/np.mean(meanyf), fft_y[FFT_max]/np.mean(fft_y)]


fig = plt.figure()
# Svals = [60,120,240,360]
# for sParam in Svals:



total = []
for iter in range(0,1001):
    print("iter: ", iter)
    rez = []
    for i in range(10,41):
        SNR = -i
        Z_scores = mainFun()
        rez.append(Z_scores)
    # print(rez)
    if(iter==0):
        total = np.array(rez)
    else:
        total = (total+rez)/2
    # temp = np.mean(np.array(total), axis=0).T
    # print("rez: ", np.array(rez))
    # print("total: ", np.array(total))
    # print("mean: ",  np.mean(np.array(total), axis=0))
    # np.savetxt("./S120/s120_iter_"+str(iter)+".csv", np.asarray(rez), delimiter=",", fmt='%10.5f')

print(np.mean(total,axis=1))
total_mean = np.mean(np.array(total), axis=0)
plt.plot(total.T[0],total.T[1], label="KLT S="+str(S))
plt.plot(total.T[0],total.T[2], label="FFT S="+str(S))

np.savetxt("./S_results/s1000iter.csv", total, delimiter=",", fmt='%10.5f')

plt.xlabel("dB")
plt.ylabel("Sigma")
plt.minorticks_on()
plt.legend()
plt.grid()
plt.show()