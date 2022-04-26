import matplotlib.pyplot as plt
import numpy as np

def generateNoise(signal, SNR):
    snr = 10.0**(SNR/10.0)
    power = signal.var()
    n = power/snr
    return np.sqrt(n)*np.random.randn(len(signal))

# Fiddle with figure settings here:
plt.rcParams['figure.figsize'] = (10,8)
plt.rcParams['font.size'] = 14
plt.rcParams['image.cmap'] = 'plasma'
plt.rcParams['axes.linewidth'] = 2
# Set the default colour cycle (in case someone changes it...)
cols = plt.get_cmap('tab10').colors

# Sample configuration
num_samples_visualize = 1
SNR = -10

# Load data
data = np.load('./signal_waves_medium_2048_500k.npy')
x_val, y_val = data[:,0], data[:,1]

# Add noise to data
noisy_samples = []
for i in range(0, len(x_val)):
  if i % 100000 == 0:
    print(i)
  pure = np.array(y_val[i])
  noise = generateNoise(pure, SNR)
  signal = pure + noise
  noisy_samples.append([x_val[i], signal])
  
# Save data to file for re-use
np.save('./signal_waves_noisy_1024_500k.npy', noisy_samples)

# Visualize a few random samples
for i in range(0, num_samples_visualize):
  random_index = np.random.randint(0, len(noisy_samples)-1)
  x_axis, y_axis = noisy_samples[random_index]
  plt.plot(x_axis, y_axis)
  plt.title(f'Visualization of noisy sample {random_index} ---- y: f(x) = x^2')
  x_axis, y_axis = data[random_index]
  plt.plot(x_axis, y_axis)
  plt.xlabel("Time (s)")
  plt.ylabel("Amplitude")
  plt.grid()
  plt.show()
