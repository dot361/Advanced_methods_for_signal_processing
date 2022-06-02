import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import hilbert

def generateSignal(N=1024, T=1.0/800.0, F=50.0, extraF=0.0, isComplex=False):
    x = np.linspace(0.0, N*T, N, endpoint=False)
    y = np.sin(F * 2.0*np.pi*x)
    # y = hilbert(y)
    if(extraF != 0.0):
        y = y+ np.sin(extraF * 2.0*np.pi*x)
    return x, y
def generateNoise(signal, SNR):
    snr = 10.0**(SNR/10.0)
    power = signal.var()
    n = power/snr
    return np.sqrt(n)*np.random.randn(len(signal))

plt.rcParams['figure.figsize'] = (10,8)
plt.rcParams['font.size'] = 14
plt.rcParams['image.cmap'] = 'plasma'
plt.rcParams['axes.linewidth'] = 2
cols = plt.get_cmap('tab10').colors

# Sample configuration
num_samples = 50000
N=8192 #4096 
T=1.0/N
F=6669.564
SNR = -15

# F_vals = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0, 140.0,150.0,160.0,170.0,180.0,190.0, 200.0, 210.0]
# F_vals = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
F_vals = [20.0, 30.0, 40.0, 50.0, 60.0]
# F_1 = np.arange(1665.402-0.2, 1665.402+0.2, step=0.1)
# F_2 = np.arange(1667.359-0.2, 1667.359+0.2, step=0.1)
# F_vals = np.concatenate((F_1, F_2))
F_vals = np.arange(6667.880 - 0.3, 6667.880+0.3, step=0.1)




# Intrasample configuration
num_elements = 1
interval_per_element = 0.01
total_num_elements = int(num_elements / interval_per_element)
starting_point = int(0 - 0.5*total_num_elements)

# Other configuration
num_samples_visualize = 3

def generateData():
    # Containers for samples and subsamples
    samples = []
    xs = []
    ys = []

    ptr = 0
    # Generate samples
    Fs = []
    for j in range(0, num_samples):
        # Report progress
        if j % 1000 == 0:
            print(F_vals[ptr])
            print(j)
            if(F_vals[ptr] == F_vals[-1]):
                ptr = -1    
            ptr = ptr+1
            
        # Generate wave
        xs, ys = generateSignal(N=N, T=T, F=F_vals[ptr], isComplex=False)
        # Append wave to samples
        samples.append((xs, ys))
        Fs.append(F_vals[ptr])
        # Clear subsample containers for next sample
        xs = []
        ys = []

    # Input shape
    print(np.shape(np.array(samples[0][0])))
    
    # Save data to file for re-use
    np.save('./signal_waves_medium_w3oh_v2.npy', samples)
    return Fs

def applyNoise():
    # Load data
    data = np.load('./signal_waves_medium_w3oh_v2.npy')
    x_val, y_val = data[:,0], data[:,1]

    # Add noise to data
    noisy_samples = []
    for i in range(0, len(x_val)):
        SNR = -np.random.randint(10,15)
        # print(SNR)
        if i % 1000 == 0:
            print(i)
            print(SNR)
        pure = np.array(y_val[i])
        noise = generateNoise(pure, SNR)
        signal = pure + noise
        noisy_samples.append([x_val[i], signal])
    
    # Save data to file for re-use
    np.save('signal_waves_noisy_w3oh_v2.npy', noisy_samples)
    for i in range(0, num_samples_visualize):
        random_index = np.random.randint(0, len(noisy_samples)-1)
        x_axis, y_axis = noisy_samples[random_index]
        plt.plot(x_axis, y_axis, label="Noisy")
        plt.title(f'Visualization of sample {random_index} --- F={Fs[random_index]}, T=1/800, N={N}, SNR={np.abs(SNR)}')
        x_axis, y_axis = data[random_index]
        plt.plot(x_axis, y_axis, label="Original")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.legend(edgecolor='black',
           prop = {'family' : 'arial', 'size' : 10}, 
           framealpha=1, 
           loc=1)
        plt.grid()
        plt.show()


Fs = generateData()
print("data generated")
applyNoise()
