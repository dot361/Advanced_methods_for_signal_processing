import numpy as np
import matplotlib.pyplot as plt
import glob
from scipy import signal
import plotly.graph_objects as go

def read_raw(filename,Ns):
    data = np.fromfile(filename, np.dtype([('re', np.int16), ('im', np.int16)]))                    #Reads data from file in int16+int16 format
    iq_data = data['re'] + data['im'] * 1j                                                          #Splits data into real and imaginary parts
    iq_data = iq_data * 1e-3     
    Ns_tot = len(iq_data)
    #The log file must be read before calling the function, as it stores necessary FFT length
    Nint = int(Ns_tot / Ns)                   #Amount of bins
    window = signal.blackmanharris(int(Ns))   #Blackman-Harris function coefficients
    spec = np.zeros(int(Ns))                  #Stores end result
    ptr = Ns                                  #Stores the pointer for the end of bin to be processed
    binptr = 0                                #Stores which bin is being processed
    for i in range(Nint):                     #Iterates through all data
        spec_i = np.fft.fft(iq_data[binptr*Ns + ptr-Ns:binptr*Ns + ptr]*window);                    #Applies FFT on data region applying blackman-harris function
        spec = spec + (spec_i.real * spec_i.real + spec_i.imag * spec_i.imag)                       #Combines real and imaginary parts
        #Moves the pointer to fulfill the 66.1% overlapping coefficient
        if(i%2==1):
            ptr = Ns + int(Ns * 0.661)                                                           
        if(i%2==0):
            ptr = Ns + int(Ns * 0.339)
        if(i%3==2 and i!=0):
            binptr = binptr + 1              #Next bin
            ptr = Ns

    spec = np.fft.fftshift(spec)             #Shifts frequency spectre to middle
    spec = spec / Nint;                      #Average calculation
    return spec

fig = go.Figure()

rawFiles = list()
for file in sorted(glob.glob( "./on_real_data/raw_files/*.raw")):
    rawFiles.append(file)


datFiles = list()
for file in sorted(glob.glob( "./on_real_data/raw_files/*.dat")):
    datFiles.append(file)

freq = np.loadtxt(datFiles[0], usecols=(0,), unpack=True)

raw_specs = []
for i in range(len(rawFiles)):
    print(i)
    spec = read_raw(rawFiles[i], 8192 )
    fig.add_trace(go.Scatter(x=freq, y=spec, mode='lines', name=rawFiles[i]))
    raw_specs.append(spec)

dat_specs = []
for i in range(len(datFiles)):
    data =  np.loadtxt(datFiles[i], usecols=(1,), unpack=True)
    data = np.fft.fftshift(data)
    fig.add_trace(go.Scatter(x=freq, y=data, mode='lines', name=datFiles[i]))
    dat_specs.append(data)

np.save("datRez",np.asarray(dat_specs))  
np.save("rawRez",np.asarray(raw_specs))

fig.show()
