# https://stackoverflow.com/questions/52913749/add-random-noise-with-specific-snr-to-a-signal

from scipy.fftpack import fft, fftfreq
import scipy as sp
import matplotlib.pyplot as plt
import numpy as np
from numpy import matlib as mb
import plotly.graph_objects as go
import math

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes 
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

np.set_printoptions(precision=3)

def generateNoise(signal, SNR):
    snr = 10.0**(SNR/10.0)
    power = signal.var()
    n = power/snr
    # print(n)
    return np.sqrt(n)*np.random.randn(len(signal))

def generateSignal(N=1024, T=1.0/800.0, freq=50.0):
    x = np.linspace(0.0, N*T, N, endpoint=False)
    y = np.sin(freq * 2.0*np.pi*x)
    return x, y
def plotFFT(x,y,N,T):
    yf = fft(y)
    xf = fftfreq(N, T)[:N//2]
    return xf, 2.0/N * np.abs(yf[0:N//2])

def generateSignalMatrix(N=1024, S=5, T=1.0/800.0, freq=60.0, SNR=-20):
    amplitude = []
    for _ in range(0,S):
        # print(i)
        x, y = generateSignal(N=N,T=T, freq=freq)
        noise = generateNoise(y, SNR)
        amplitude.append(y+noise)
    return x, amplitude
def divisorGenerator(n):
    large_divisors = []
    for i in range(1, int(math.sqrt(n) + 1)):
        if n % i == 0:
            yield i
            if i*i != n:
                large_divisors.append(n / i)
    for divisor in reversed(large_divisors):
        yield divisor



# for iter in range(1,4):
sigmas = []

for dom in range(1,4):
    print("k: ", dom)
    T = 1.0/600.0
    N = 4096
    S = 120
    SNR = -5
    freq = 60.0


    x, D = generateSignalMatrix(N=N, S=S, T=T, freq=freq, SNR=SNR )




    # x,y = generateSignal(N=N,T=T)
    # noise = generateNoise(y,-5)
    # plt.plot(x,y)
    # plt.plot(x,y+noise)
    # plt.show()
    # plotFFT(x,y+noise,N,T)

    # SNR = -20

    # # #%%
    # A1 = y + generateNoise(y,SNR)
    # A2 = y + generateNoise(y,SNR)
    # A3 = y + generateNoise(y,SNR)
    # A4 = y + generateNoise(y,SNR)
    # A5 = y + generateNoise(y,SNR)

    # D = np.array([A1, A2, A3, A4, A5])



    fft_x = []
    fft_y = []
    for i in D:
        org_x, org_y, = plotFFT(x,i,N,T)
        fft_x = org_x
        fft_y.append(org_y)
    fft_y = np.mean(np.array(fft_y),axis=0)

    print("FFT done")

    print(np.where(np.around(fft_x, decimals=3) == 60.059))

    #%% 
    # D = np.genfromtxt('data.csv', delimiter=',').T
    # print(D.shape)
    # np.savetxt("data.csv", np.asarray(D).T, fmt="'%1.5f'", delimiter=",", header="A1,A2,A3,A4,A5")
    mu_D = np.mean(D,axis=0)
    B = D-mb.repmat(mu_D,m=S,n=1)   # m - number of times matrix is repeated along the first and second axis

    # Covariance matrix
    C = np.dot(B.T, B)
    [eigenvals, V] = sp.linalg.eigh(C)  # Eigenvals are returned as an array not diag of matrix like in matlab
    V = np.around(V, decimals=4)
    # Sigma = np.around(Sigma, decimals=4)
    # # [V, Sigma] = sp.linalg.eigh(C)
    # eigenvals = np.diag(Sigma)
    # plt.plot(V)
    # plt.show()
    # print(eigenvals)
    desc_val = -np.sort(-eigenvals)

    nonzeros = desc_val[np.where(np.around(desc_val, decimals=4) > 0)]
    print("nonzeros: ", len(nonzeros))
    plt.title("Eigenvalues larger than 0")
    plt.scatter(range(0,len(nonzeros)), nonzeros)
    plt.ylabel("Eigenvalue")
    plt.xlabel("Index")
    plt.grid()
    plt.minorticks_on()
    plt.show()
    indices = -np.argsort(-eigenvals)

    # [desc_eval, indices] = np.sort(eigenvals, order='descend')
    eigenvect = V[indices]
    # print(eigenvect)

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=fft_x,y=fft_y-np.mean(fft_y), name="FFT rez"))
    _, ax = plt.subplots(figsize=[5, 4])
    components = int(math.ceil((S//2) / 10.0)) * 10
    divisors = np.array(list((divisorGenerator(components)))).astype(int)
    print("divisors: ",divisors)

    sig = []
    ks = []
    for iter in divisors:
        k = iter #reconstruct using k dominant components
        print(k)
        selectedEig = eigenvect[0:k,:].T
        # print(selectedEig)
        # print("selected: ", selectedEig)
        reconstr = np.dot((np.dot(B,selectedEig)), selectedEig.T) + mb.repmat(mu_D, S, 1) 

        # plt.plot(reconstr.T)
        # plt.show()

        meanyf = []



        # plt.figure("compare")
        # plt.subplot(121)
        # plt.title("FFT mean")
        # plt.plot(fft_x,fft_y)

        # plt.subplot(122)
        # plt.title("KLT reconstruct of "+str(k)+" dominant components for each generated spectrum")
        for idx,i in enumerate(reconstr):
            xf, yf = plotFFT(x,i,N,T)
            # plt.plot(xf,yf,label=str(idx))
            # fig.add_trace(go.Scatter(x=xf,y=yf, name="KLT of spectrum "+str(idx)))
            # plt.figure("firstSpec")
            # plt.plot(fft_x,fft_y-np.mean(fft_y),label="FFT result")
            # plt.plot(xf,yf-np.mean(yf),label="KLT result")
            # plt.xlabel("Frequency (Hz)")
            # plt.ylabel("Amplitude")
            # plt.grid()
            # plt.legend()
            # plt.show()
            meanyf.append(yf)

        # plt.show()
        meanyf = np.mean(np.array(meanyf),axis=0)

        # plt.figure("mean")
        # plt.plot(xf, meanyf, label="mean: "+ str(k))
        fig.add_trace(go.Scatter(x=xf,y=meanyf- np.mean(meanyf), name="k="+str(k)))
        ax.plot(xf, meanyf-np.mean(meanyf), label="k="+str(k))
        KLT_peak = meanyf[410]-np.mean(meanyf)
        FFT_peak = fft_y[410]-np.mean(fft_y)

        # fig.add_trace(go.Scatter(x=[xf[410]],y=[KLT_peak], name="KLT Peak"))
        # fig.add_trace(go.Scatter(x=[fft_x[410]],y=[FFT_peak], name="FFT Peak"))

        print("KLT: ", np.abs(KLT_peak/np.mean(meanyf)))
        print("FFT: ", np.abs(FFT_peak/np.mean(fft_y)))
        sig.append(np.abs(KLT_peak/np.mean(meanyf)))
        ks.append(str(k))
        # plt.legend()
        # plt.show()

        # fig.show()
        # sigmas.append([dom,k,np.abs(KLT_peak/np.mean(meanyf)),np.abs(FFT_peak/np.mean(fft_y))])

        
        # print(sigmas)
    plt.legend()
    plt.grid()
    plt.show()
    sig = np.array(sig)
    
    plt.bar(ks, sig)
    plt.ylabel("Sigma (peak value/average)")
    plt.xlabel("K value")
    plt.minorticks_on()
    plt.show()
    fig.update_layout(
    plot_bgcolor="#FFF",  # Sets background color to white
    title_text="KLT 60Hz, S="+str(S)+",N=4096,SNR="+str(SNR),
    font=dict(
        family="Courier New, monospace",
        size=18,
        color="black"
    ),
    xaxis=dict(
        title="Frequency (Hz)",
        linecolor="#BCCCDC",  # Sets color of X-axis line
        gridcolor='LightBlue'
    ),
    yaxis=dict(
        title="Amplitude",  
        linecolor="#BCCCDC",
        gridcolor='LightBlue'
    )
    )
    fig.show()
    # np.savetxt("./S120/s120_k"+str(dom)+"sigmas"+str(iter)+".csv", np.asarray(sigmas), delimiter=",", fmt='%10.5f')
    # del sigmas
    del D
    del B
    del V
    del eigenvals