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

class SSA(object):
    
    __supported_types = (pd.Series, np.ndarray, list)
    
    def __init__(self, tseries, L, save_mem=True):
        """
        Decomposes the given time series with a singular-spectrum analysis. Assumes the values of the time series are
        recorded at equal intervals.
        
        Parameters
        ----------
        tseries : The original time series, in the form of a Pandas Series, NumPy array or list. 
        L : The window length. Must be an integer 2 <= L <= N/2, where N is the length of the time series.
        save_mem : Conserve memory by not retaining the elementary matrices. Recommended for long time series with
            thousands of values. Defaults to True.
        
        Note: Even if an NumPy array or list is used for the initial time series, all time series returned will be
        in the form of a Pandas Series or DataFrame object.
        """
        
        # Tedious type-checking for the initial time series
        if not isinstance(tseries, self.__supported_types):
            raise TypeError("Unsupported time series object. Try Pandas Series, NumPy array or list.")
        
        # Checks to save us from ourselves
        self.N = len(tseries)
        if not 2 <= L <= self.N/2:
            raise ValueError("The window length must be in the interval [2, N/2].")
        
        self.L = L
        self.orig_TS = pd.Series(tseries)
        self.K = self.N - self.L + 1
        
        # Embed the time series in a trajectory matrix
        self.X = np.array([self.orig_TS.values[i:L+i] for i in range(0, self.K)]).T
        
        # Decompose the trajectory matrix
        self.U, self.Sigma, VT = np.linalg.svd(self.X)
        self.d = np.linalg.matrix_rank(self.X)
        
        self.TS_comps = np.zeros((self.N, self.d))
        
        if not save_mem:
            # Construct and save all the elementary matrices
            self.X_elem = np.array([ self.Sigma[i]*np.outer(self.U[:,i], VT[i,:]) for i in range(self.d) ])

            # Diagonally average the elementary matrices, store them as columns in array.           
            for i in range(self.d):
                X_rev = self.X_elem[i, ::-1]
                self.TS_comps[:,i] = [X_rev.diagonal(j).mean() for j in range(-X_rev.shape[0]+1, X_rev.shape[1])]
            
            self.V = VT.T
        else:
            # Reconstruct the elementary matrices without storing them
            for i in range(self.d):
                X_elem = self.Sigma[i]*np.outer(self.U[:,i], VT[i,:])
                X_rev = X_elem[::-1]
                self.TS_comps[:,i] = [X_rev.diagonal(j).mean() for j in range(-X_rev.shape[0]+1, X_rev.shape[1])]
            
            self.X_elem = "Re-run with save_mem=False to retain the elementary matrices."
            
            # The V array may also be very large under these circumstances, so we won't keep it.
            self.V = "Re-run with save_mem=False to retain the V matrix."
        
        # Calculate the w-correlation matrix.
        self.calc_wcorr()
            
    def components_to_df(self, n=0):
        """
        Returns all the time series components in a single Pandas DataFrame object.
        """
        if n > 0:
            n = min(n, self.d)
        else:
            n = self.d
        
        # Create list of columns - call them F0, F1, F2, ...
        cols = ["F{}".format(i) for i in range(n)]
        return pd.DataFrame(self.TS_comps[:, :n], columns=cols, index=self.orig_TS.index)
            
    
    def reconstruct(self, indices):
        """
        Reconstructs the time series from its elementary components, using the given indices. Returns a Pandas Series
        object with the reconstructed time series.
        
        Parameters
        ----------
        indices: An integer, list of integers or slice(n,m) object, representing the elementary components to sum.
        """
        if isinstance(indices, int): indices = [indices]
        
        ts_vals = self.TS_comps[:,indices].sum(axis=1)
        return pd.Series(ts_vals, index=self.orig_TS.index)
    
    def calc_wcorr(self):
        """
        Calculates the w-correlation matrix for the time series.
        """
             
        # Calculate the weights
        w = np.array(list(np.arange(self.L)+1) + [self.L]*(self.K-self.L-1) + list(np.arange(self.L)+1)[::-1])
        
        def w_inner(F_i, F_j):
            return w.dot(F_i*F_j)
        
        # Calculated weighted norms, ||F_i||_w, then invert.
        F_wnorms = np.array([w_inner(self.TS_comps[:,i], self.TS_comps[:,i]) for i in range(self.d)])
        F_wnorms = F_wnorms**-0.5
        
        # Calculate Wcorr.
        self.Wcorr = np.identity(self.d)
        for i in range(self.d):
            for j in range(i+1,self.d):
                self.Wcorr[i,j] = abs(w_inner(self.TS_comps[:,i], self.TS_comps[:,j]) * F_wnorms[i] * F_wnorms[j])
                self.Wcorr[j,i] = self.Wcorr[i,j]
    
    def plot_wcorr(self, min=None, max=None):
        """
        Plots the w-correlation matrix for the decomposed time series.
        """
        if min is None:
            min = 0
        if max is None:
            max = self.d
        
        if self.Wcorr is None:
            self.calc_wcorr()
        
        ax = plt.imshow(self.Wcorr)
        plt.xlabel(r"$\tilde{F}_i$")
        plt.ylabel(r"$\tilde{F}_j$")
        plt.colorbar(ax.colorbar, fraction=0.045)
        ax.colorbar.set_label("$W_{i,j}$")
        plt.clim(0,1)
        
        # For plotting purposes:
        if max == self.d:
            max_rnge = self.d-1
        else:
            max_rnge = max
        
        plt.xlim(min-0.5, max_rnge+0.5)
        plt.ylim(max_rnge+0.5, min-0.5)
        

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
N = 1024
S = 120
SNR = -20
F = 70.0
extraF = 0.0
width, height = 128, 8

print(N,S)







# plt.figure(1)
x, y = generateSignal(N=N,T=T, F=F,extraF=extraF)
ox, oy = fetchFFT(x, y, N, T)

noise_types = ["blue", "violet", "brownian", "pink"]
ptr = 1
plt.figure(1)
rmse_vals = []
for noise_type in noise_types:
    plt.figure(1)
    plt.subplot(3,2,ptr)
    noise = generateColoredNoise(y,SNR, noise_type=noise_type)
    plt.plot(x,y+noise, label=noise_type, color="forestgreen")
    plt.legend(loc='upper right', prop={'size': 10})
    plt.grid()
    plt.figure(2)
    plt.subplot(3,2,ptr)
    # noise = generateColoredNoise(y,SNR, noise_type=noise_type)
    fx, fy = fetchFFT(y+noise, y+noise, N, T)
    plt.axvline(x=F, color='black', ls='--')
    plt.plot(fx,fy, label=noise_type, color="forestgreen")
    rmse_vals.append(rmse(oy,fy))
    plt.legend(loc='upper right', prop={'size': 10})
    plt.grid()
    ptr+=1 
# plt.show()
plt.figure(1)
plt.subplot(313)
noise = generateNoise(y,SNR)
plt.plot(x,y+noise, label="white", color="forestgreen")
plt.legend(loc='upper right', prop={'size': 10})
plt.grid()
plt.suptitle("Noisy signal generated with "+ str(np.abs(SNR)) +" dB SNR at " + str(F) +" Hz")
plt.tight_layout()

plt.figure(2)
plt.subplot(313)
fx, fy = fetchFFT(y+noise, y+noise, N, T)
plt.axvline(x=F, color='black', ls='--')

plt.plot(fx,fy, label="white", color="forestgreen")
rmse_vals.append(rmse(oy,fy))

plt.legend(loc='upper right', prop={'size': 10})
plt.grid()
plt.suptitle("Fourier transform of noisy signal generated with "+ str(np.abs(SNR)) +" dB SNR at " + str(F) +" Hz")
plt.tight_layout()


plt.figure(3)

plt.plot(rmse_vals)

plt.show()




# plt.plot(x,y)
# # pinkNoise = generateColoredNoise(y, SNR, "pink")
# pinkNoise = noise_psd(y, SNR, "pink")
# # plt.plot(x,y+pinkNoise)
pinkNoise = pink_noise(N)
otherPinkNoise = generateColoredNoise(y,SNR=10, noise_type="blue")
# plt.plot(pinkNoise)
# plt.plot(otherPinkNoise)
# blueNoise = noise_psd(y, SNR, "blue")
# # plt.plot(x,y+blueNoise)
# brownNoise = noise_psd(y, SNR, "brownian")
# # plt.plot(x,y+brownNoise)

# whiteNoise = noise_psd(y, SNR, "white")
# plt.plot(x,y+whiteNoise)

# myWhiteNoise = generateNoise(y,SNR)
# plt.plot(x,y+myWhiteNoise)


# plt.figure(2)

# xf, yf = fetchFFT(x, y, N, T)
# plt.plot(xf,yf)
# xf, yf = fetchFFT(x, y+pinkNoise, N, T)
# plt.plot(xf,yf)
# xf, yf = fetchFFT(x, y+otherPinkNoise, N, T)
# plt.plot(xf,yf)
# plt.show()
# xf, yf = fetchFFT(x, y+blueNoise, N, T)
# # plt.plot(xf,yf)
# xf, yf = fetchFFT(x, y+brownNoise, N, T)
# # plt.plot(xf,yf)
# xf, yf = fetchFFT(x, y+whiteNoise, N, T)
# plt.plot(xf,yf)
# xf, yf = fetchFFT(x, y+myWhiteNoise, N, T)
# plt.plot(xf,yf)
# plt.show()

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
def mainFun(RMSE_array):
    # x, D = generateSignalMatrix(N=N, S=S, T=T, F=F, SNR=SNR,extraF=extraF)
    x, D = generateColoredSignalMatrix(N=N, S=S, T=T, F=F, SNR=SNR,extraF=extraF, noise_type="pink")
    _, originalY = generateSignal(N=N,T=T, F=F,extraF=extraF)

    print(SNR)
    fft_y = []
    for i in D:
        org_x, org_y, = fetchFFT(x,i,N,T)
        fft_y.append(org_y)
    fft_y = np.mean(np.array(fft_y),axis=0)
    reconstr = KLT(D=D,k=0)
    # meanyf = []
    yf = np.mean(np.array(reconstr),axis=0)
    xf, meanyf = fetchFFT(x, yf, N, T)

    fft_originalX, fft_originalY = fetchFFT(originalY, originalY, N, T)
    # for idx,i in enumerate(reconstr):
    #     xf, yf = fetchFFT(x,i,N,T)
    #     meanyf.append(yf)
    # meanyf = np.mean(np.array(meanyf),axis=0)


    model = keras.models.load_model('./keras-autoencoders-master/10_100_SNR_1024N')

    reconstruction = model.predict(D[0].reshape((1, width, height, 1))).reshape((width * height,))
    print(reconstruction.shape)
    reconstruction = np.array(reconstruction).reshape((width * height,))
    fft_ML_x, fft_ML_y = fetchFFT(reconstruction, reconstruction, N, T)

    F_ssa_L2 = SSA(np.mean(np.array(D),axis=0), 128)
    SSA_sig = F_ssa_L2.reconstruct([0,1,2,3,4,5]).values
    fft_SSA_x, fft_SSA_y = fetchFFT(SSA_sig, SSA_sig, N, T)


    maxpeak_range = np.arange(85-5, 85+5, 1, dtype=int) #75
    print(xf[maxpeak_range])
    KLT_max = np.argmax(meanyf[maxpeak_range])+85
    FFT_max = np.argmax(fft_y[maxpeak_range])+85
    ML_max = np.argmax(fft_ML_y[maxpeak_range])+85 #149 #70
    SSA_max = np.argmax(fft_SSA_y[maxpeak_range])+85 #149 #70

    print(KLT_max, FFT_max)

    FFT_z = stats.zscore(fft_y)
    KLT_z = stats.zscore(meanyf)
    ML_z = stats.zscore(fft_ML_y)
    SSA_z = stats.zscore(fft_SSA_y)

    print(meanyf[KLT_max], np.mean(meanyf))
    KLT_sigma = meanyf[KLT_max] / np.mean(meanyf)
    FFT_sigma = fft_y[FFT_max] / np.mean(fft_y)
    
    meanyf = meanyf - np.mean(meanyf)
    fft_y = fft_y - np.mean(fft_y)

    rsme_klt = rmse(fft_originalY, meanyf)
    rsme_fft = rmse(fft_originalY, fft_y)
    rsme_ML = rmse(fft_originalY, fft_ML_y)
    rsme_SSA = rmse(fft_originalY, fft_SSA_y)

    prd_klt = prd(fft_originalY, meanyf)
    prd_fft = prd(fft_originalY, fft_y)
    prd_ML = prd(fft_originalY, fft_ML_y)
    prd_SSA = prd(fft_originalY, fft_SSA_y)


    print("RMSE KLT: ", rsme_klt, "FFT: ", rsme_fft)
    RMSE_array.append([[rsme_klt, prd_klt],[rsme_fft, prd_fft], [rsme_ML, prd_ML], [rsme_SSA, prd_SSA]])

    plt.figure(1)
    plt.plot(fft_originalX, fft_originalY, label="Original")

    plt.plot(xf, meanyf, label="KLT (σ="+str(np.around(KLT_z[KLT_max],decimals=3))+")")
    plt.plot(org_x, fft_y, label="FFT (σ="+str(np.around(FFT_z[FFT_max],decimals=3))+")")
    plt.plot(fft_ML_x, fft_ML_y, label="ML (σ="+str(np.around(ML_z[ML_max],decimals=3))+")")
    plt.plot(fft_SSA_x, fft_SSA_y, label="SSA (σ="+str(np.around(SSA_z[SSA_max],decimals=3))+")")

    plt.plot(xf[KLT_max], meanyf[KLT_max],marker='x',color='blue')
    plt.plot(xf[FFT_max], fft_y[FFT_max],marker='x',color='green')
    plt.plot(xf[ML_max], fft_ML_y[FFT_max],marker='x',color='red')

    plt.title(f"N={N}, T=1.0/800.0, F={F}, SNR={SNR} db")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.minorticks_on()
    plt.legend(loc='upper right', prop={'size': 6})
    plt.grid()



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
