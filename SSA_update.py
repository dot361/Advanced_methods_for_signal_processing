from scipy.fftpack import fft, fftfreq
import matplotlib.pyplot as plt
import scipy as sp
import numpy as np
from numpy import matlib as mb
# import plotly.graph_objects as go
import sys
from scipy import stats
import tensorflow as tf
from cycler import cycler
import pandas as pd
from numpy import pi
from numba import jit, prange
import numba as nb

np.set_printoptions(precision=3)

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




def calc_wcorr(TS_comps, L, K, d):
    w = np.array(list(np.arange(L)+1) + [L]*(K-L-1) + list(np.arange(L)+1)[::-1])
    
    def w_inner(F_i, F_j):
        return w.dot(F_i*F_j)
    
    # Calculated weighted norms, ||F_i||_w, then invert.
    F_wnorms = np.array([w_inner(TS_comps[:,i], TS_comps[:,i]) for i in range(d)])
    F_wnorms = F_wnorms**-0.5
    
    # Calculate Wcorr.
    Wcorr = np.identity(d)
    for i in range(d):
        for j in range(i+1,d):
            Wcorr[i,j] = abs(w_inner(TS_comps[:,i], TS_comps[:,j]) * F_wnorms[i] * F_wnorms[j])
            Wcorr[j,i] = Wcorr[i,j]
    return Wcorr





@nb.njit(parallel=True)
def performSSA(orig_TS : np.array, L : int) -> np.array:
    N = orig_TS.size
    K = N - L + 1
    # Embed the time series in a trajectory matrix

    # X = np.array([orig_TS[i:L+i] for i in range(0, K)]).T
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
        # rez = []
        # print(-X_rev.shape[0]+1, X_rev.shape[1])
        idx = 0
        for j in prange(-X_rev.shape[0]+1, X_rev.shape[1]):
            # rez.append(np.diag(X_rev,j).mean())
            TS_comps[idx,i] = np.diag(X_rev,j).mean()
            idx+=1 
        # TS_comps[:,i] = rez
        # TS_comps[:,i] = [X_rev.diagonal(j).mean() for j in range(-X_rev.shape[0]+1, X_rev.shape[1])]
    
    # if(calcWcorr):
    #     wcorr = calc_wcorr(TS_comps, L, K,d)
    #     return TS_comps, wcorr
    # else:
    return TS_comps



# def performSSA(orig_TS, L):
#     N = len(orig_TS)
#     K = N - L + 1
#     # Embed the time series in a trajectory matrix
#     X = np.array([orig_TS[i:L+i] for i in range(0, K)]).T

#     # Decompose the trajectory matrix
#     U, Sigma, VT = np.linalg.svd(X)
#     d = np.linalg.matrix_rank(X)
#     TS_comps = np.zeros((N, d))
#     for i in range(d):
#         X_elem = Sigma[i]*np.outer(U[:,i], VT[i,:])
#         X_rev = X_elem[::-1]
#         TS_comps[:,i] = [X_rev.diagonal(j).mean() for j in range(-X_rev.shape[0]+1, X_rev.shape[1])]
    
#     # if(calcWcorr):
#     #     wcorr = calc_wcorr(TS_comps, L, K,d)
#     #     return TS_comps, wcorr
#     # else:
#     return TS_comps


def reconstruct(TS_comps, indices):
    return np.array(TS_comps[:,indices].sum(axis=1))



def generateSignal(N=1024, T=1.0/800.0, F=50.0, extraF=0.0):
    x = np.linspace(0.0, N*T, N, endpoint=False)
    y = np.sin(F * 2.0*np.pi*x)
    if(extraF != 0.0):
        y = y+ np.sin(extraF * 2.0*np.pi*x)
    return x, y

def generateNoise(signal, SNR):
    snr = 10.0**(SNR/10.0)
    power = signal.var()
    n = power/snr
    return np.sqrt(n)*np.random.randn(len(signal))
def fetchFFT(x,y,N,T):
    yf = fft(y)
    xf = fftfreq(N, T)[:N//2]
    # plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
    # plt.show()
    return xf, 2.0/N * np.abs(yf[0:N//2])

T = 1.0/800.0
N = 2048
S = 120
SNR = -20
F = 60.0
extraF = 0.0

x, y = generateSignal(N=N,T=T, F=F,extraF=extraF)
noise = generateNoise(y,SNR)
old_ssa = SSA(y+noise, 8)
old_ssa_rez = old_ssa.reconstruct(slice(0,5)).values

mySSA = performSSA(y+noise,8)
my_ssa_rez = reconstruct(mySSA, slice(0,5))

fx, fy = fetchFFT(old_ssa_rez,old_ssa_rez, N,T)
plt.subplot(121)

plt.plot(old_ssa_rez)
plt.plot(my_ssa_rez)

plt.subplot(122)
plt.plot(fx, fy)
fx, fy = fetchFFT(my_ssa_rez,my_ssa_rez, N,T)
plt.plot(fx, fy)

plt.show()

import time

comps = [8,16,32,64,128,256,512]
old_times = [[],[],[],[],[],[],[]]
improved_times = [[],[],[],[],[],[],[]]
for comp in range(0,len(comps)):
    print(comps[comp])
    for i in range(0,501):
        x, y = generateSignal(N=N,T=T, F=F,extraF=extraF)
        noise = generateNoise(y,SNR)
        t0 = time.time()
        old_ssa = SSA(y+noise, comps[comp])
        old_ssa_rez = old_ssa.reconstruct(slice(0,5)).values
        fx,fy = fetchFFT(old_ssa_rez,old_ssa_rez, N,T)
        # plt.plot(old_ssa_rez)
        # plt.figure(1)
        # plt.plot(fx,fy, label="old")
        t1 = time.time()
        old_times[comp].append(t1-t0)
        print("original exec time - "+ str(t1-t0))
        # plt.plot(old_ssa_rez)
        t0 = time.time()
        mySSA = performSSA(y+noise,comps[comp])
        my_ssa_rez = reconstruct(mySSA, slice(0,5))
        fx,fy = fetchFFT(my_ssa_rez,my_ssa_rez, N,T)
        print(np.shape(fy))

        # plt.plot(old_ssa_rez)
        # plt.plot(fx,fy, label="new")
        # plt.legend()
        # plt.show()
        t1 = time.time()
        print("improved exec time - "+ str(t1-t0))
        improved_times[comp].append(t1-t0)

        # plt.plot(my_ssa_rez)
        # plt.show()

# for comp in range(0,len(comps)):
#     plt.subplot(121)
#     plt.plot(old_times[comp], label=str(comps[comp]))
#     plt.subplot(122)
#     plt.plot(improved_times[comp], label=str(comps[comp]))

labels = ["0", "8","16","32","64","128","256","512"]

fig, ax = plt.subplots(1,2)
plt.subplot(121)
plt.boxplot(old_times)
plt.legend()
plt.grid()
plt.xlabel("Komponenšu skaits")
plt.title("Orģinālais algoritms")
ax[0].set_xticks([0,1,2,3,4,5,6,7])
ax[0].set_xticklabels(labels)
plt.ylabel("Izpildes laiks (s)")
plt.subplot(122)
plt.boxplot(improved_times)
plt.legend()
plt.grid()
plt.xlabel("Komponenšu skaits")
plt.title("Numba optimizētā versija")
ax[1].set_xticks([0,1,2,3,4,5,6,7])
ax[1].set_xticklabels(labels)

plt.ylabel("Izpildes laiks (s)")
plt.show()