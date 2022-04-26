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
    print(np.shape(eigenvals), "k: ", k)
    desc_val = -np.sort(-eigenvals)
    nonzeros = desc_val[np.where(np.around(desc_val, decimals=4) > 0)]
    # plt.figure("eigenvals")
    # plt.title("Eigenvalues larger than 0")
    # plt.grid()

    # plt.scatter(range(0,len(desc_val)), desc_val, label="all")
    # plt.scatter(range(0,len(nonzeros)), nonzeros, label="nonzeros")
    # plt.legend()
    # plt.show()
    indices = -np.argsort(-eigenvals)
    eigenvect = V[indices] 
    # fig = plt.figure("KLT_K")
    # fig = go.Figure()
    # fig.add_trace(go.Scatter(y=D[0], name="original_0"))
    # print(N,T)
    # fig.add_trace(go.Scatter(y=fetchFFT(indices,D[0],N,T)[1], name="original_0_fft"))
    
    # fig.add_trace(go.Scatter(y=D[1], name="original_1"))
    for k_val in range(0,k): 
        print("k: ", k_val)
        selectedEig = eigenvect[k].T
        # print(np.shape(selectedEig))
        selectedEig = selectedEig.reshape(-1, 1)
        rez = np.dot((np.dot(B,selectedEig)), selectedEig.T) + mb.repmat(mu_D, m=np.shape(D)[0], n=1) 
        # fig.add_trace(go.Scatter(y=rez[0], name="klt_"+str(0)+"_k:"+str(k_val)))
        # fig.add_trace(go.Scatter(y=fetchFFT(indices,rez[0],N,T)[1], name="klt_0_fft_k:"+str(k_val)))

        # fig.add_trace(go.Scatter(y=rez[1], name="klt_"+str(1)+"_k:"+str(k_val)))


        # plt.show()
        # rez = np.mean(np.array(rez),axis=0)
        
        # plt.plot(rez,label="k: "+str(k_val))
        # fig.add_trace(go.Scatter(y=rez, name="k: "+str(k_val)))
    # fig.show()
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
S = 5
SNR = -10
F = 60.0
extraF = 0.0

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
    x, D = generateSignalMatrix(N=N, S=S, T=T, F=F, SNR=SNR,extraF=extraF)
    print(SNR)
    fft_y = []
    for i in D:
        org_x, org_y, = fetchFFT(x,i,N,T)
        fft_y.append(org_y)
    fft_y = np.mean(np.array(fft_y),axis=0)
    reconstr = KLT(D=D,k=20)
    meanyf = []
    for idx,i in enumerate(reconstr):
        xf, yf = fetchFFT(x,i,N,T)
        meanyf.append(yf)
    meanyf = np.mean(np.array(meanyf),axis=0)

    # fft_y = fft_y + 0.5
    # meanyf = meanyf + 0.5

    

    # KLT_peak = (meanyf[154] - np.mean(meanyf))
    # FFT_peak = (fft_y[154] - np.mean(fft_y))
    maxpeak_range = np.arange(154-5, 154+5, 1, dtype=int)
    print(xf[maxpeak_range])
    KLT_max = np.argmax(meanyf[maxpeak_range])+149
    FFT_max = np.argmax(fft_y[maxpeak_range])+149
    print(KLT_max, FFT_max)

    FFT_z = stats.zscore(fft_y)
    KLT_z = stats.zscore(meanyf)
    
    print(meanyf[KLT_max], np.mean(meanyf))
    KLT_sigma = meanyf[KLT_max] / np.mean(meanyf)
    FFT_sigma = fft_y[FFT_max] / np.mean(fft_y)
    
    meanyf = meanyf - np.mean(meanyf)
    fft_y = fft_y - np.mean(fft_y)
    # plt.axhline(KLT_peak)
    # plt.axhline(FFT_peak, color='orange')



    # plt.plot(xf, meanyf, label="KLT (σ="+str(np.around(KLT_sigma,decimals=3))+")")
    # plt.plot(org_x, fft_y, label="FFT (σ="+str(np.around(FFT_sigma,decimals=3))+")")

    plt.plot(xf, meanyf, label="KLT (σ="+str(np.around(KLT_z[KLT_max],decimals=3))+")")
    plt.plot(org_x, fft_y, label="FFT (σ="+str(np.around(FFT_z[FFT_max],decimals=3))+")")

    plt.plot(xf[KLT_max], meanyf[KLT_max],marker='x',color='blue')
    plt.plot(xf[FFT_max], fft_y[FFT_max],marker='x',color='red')
    

    plt.title("N="+str(N)+", T=1.0/800.0, F=60, SNR="+str(SNR)+"db")
    plt.ylim(bottom=-0.1, top=1)

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.minorticks_on()
    plt.legend()
    plt.grid()
    # plt.show()


fig = plt.figure()
plt.subplot(321)
mainFun()


SNR = -20
plt.subplot(322)
mainFun()

SNR = -25
plt.subplot(323)
mainFun()

SNR = -30
plt.subplot(324)
mainFun()

SNR = -35
plt.subplot(325)
mainFun()

SNR = -40
plt.subplot(326)
mainFun()
# fig.subplots_adjust(top=0.955,
# bottom=0.06,
# left=0.11,
# right=0.9,
# hspace=0.29,
# wspace=0.2)

plt.show()
