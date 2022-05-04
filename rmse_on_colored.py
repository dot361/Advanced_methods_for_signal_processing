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
N = 2048
S = 120
SNR = -10
F = 60.0
extraF = 0.0

print(N,S)




def perform_analysis(SNR):
    x, y = generateSignal(N=N,T=T, F=F,extraF=extraF)
    ox, oy = fetchFFT(x, y, N, T)

    noise_types = ["blue", "violet", "brownian", "pink", "white"]
    ptr = 1
    # plt.figure(1)
    rmse_vals = []
    for noise_type in noise_types:
        # plt.figure(1)
        # plt.subplot(3,2,ptr)
        noise = generateColoredNoise(y,SNR, noise_type=noise_type)
        # plt.plot(x,y+noise, label=noise_type, color="forestgreen")
        # plt.legend(loc='upper right', prop={'size': 10})
        # plt.grid()
        # plt.figure(2)
        # plt.title(str(SNR))
        # plt.subplot(3,2,ptr)
        noise = generateColoredNoise(y,SNR, noise_type=noise_type)
        fx, fy = fetchFFT(y+noise, y+noise, N, T)
        # plt.axvline(x=F, color='black', ls='--')
        # plt.plot(fx,fy, label=noise_type, color="forestgreen")
        rmse_vals.append(rmse(oy,fy))
        # plt.legend(loc='upper right', prop={'size': 10})
        # plt.grid()
        ptr+=1 
    # plt.show()
    # plt.figure(1)
    # plt.subplot(313)
    # noise = generateNoise(y,SNR)
    # plt.plot(x,y+noise, label="white", color="forestgreen")
    # plt.legend(loc='upper right', prop={'size': 10})
    # plt.grid()
    # plt.suptitle("Noisy signal generated with "+ str(np.abs(SNR)) +" dB SNR at " + str(F) +" Hz")
    # plt.tight_layout()

    # plt.figure(2)
    # plt.subplot(313)
    # fx, fy = fetchFFT(y+noise, y+noise, N, T)
    # plt.axvline(x=F, color='black', ls='--')

    # plt.plot(fx,fy, label="white", color="forestgreen")
    # rmse_vals.append(rmse(oy,fy))

    # plt.legend(loc='upper right', prop={'size': 10})
    # plt.grid()
    # plt.suptitle("Fourier transform of noisy signal generated with "+ str(np.abs(SNR)) +" dB SNR at " + str(F) +" Hz")
    # plt.tight_layout()


    # plt.figure(3)

    # plt.plot(rmse_vals)

    # plt.show()
    # print(rmse_vals)
    return rmse_vals

def mainFun():
    # x, D = generateSignalMatrix(N=N, S=S, T=T, F=F, SNR=SNR)
    x, y = generateSignal(N=N,T=T, F=F,extraF=extraF)

    noise_types = ["blue", "violet", "brownian", "pink","white"]
    maxpeak_range = np.arange(154-5, 154+5, 1, dtype=int)

    rez = [SNR]
    for type in noise_types:
        noise = generateColoredNoise(y,SNR, noise_type=type)
        fx, fy = fetchFFT(y+noise, y+noise, N, T)
        maxVal = np.argmax(fy[maxpeak_range])+149
        rez.append(fy[maxVal]/np.mean(fy))
    return rez


iters = 100001


noise_types = ["blue", "violet", "brownian", "pink","white"]

SNR_vals = [-10,-15,-20,-25,-30,-35]
rmse_vals = []

for iter in range(0,iters):
    rmse_iter = []
    for val in SNR_vals:
        # print(val)
        rmse_iter.append(perform_analysis(val))
    rmse_vals.append(rmse_iter)
rmse_vals = np.average(rmse_vals,axis=0)

rmse_vals = np.array(rmse_vals).T
print(rmse_vals)

plt.figure(1)
plt.subplot(121)
for i in range(0,len(noise_types)):
    plt.plot(SNR_vals, rmse_vals[i], label=noise_types[i])
plt.legend(loc='upper right', prop={'size': 10})
plt.grid()
plt.minorticks_on()
plt.xlabel("dB")
plt.ylabel("RMSE")

# plt.show()




# fig = plt.figure()
# Svals = [60,120,240,360]
# for sParam in Svals:



total = []
for iter in range(0,iters):
    if(iter % 1000 == 0 ):
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
noise_types = ["blue", "violet", "brownian", "pink","white"]

plt.subplot(122)
for i in range(1,len(noise_types)+1):
    plt.plot(total.T[0],total.T[i], label=noise_types[i-1])


# np.savetxt("./S_results/s1000iter.csv", total, delimiter=",", fmt='%10.5f')

plt.xlabel("dB")
plt.ylabel("Sigma")

plt.minorticks_on()
plt.legend(loc='upper left', prop={'size': 10})
plt.grid()

plt.show()