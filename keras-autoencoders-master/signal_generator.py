import matplotlib.pyplot as plt
import numpy as np

def generateSignal(N=1024, T=1.0/800.0, F=50.0, extraF=0.0):
    x = np.linspace(0.0, N*T, N, endpoint=False)
    y = np.sin(F * 2.0*np.pi*x)
    if(extraF != 0.0):
        y = y+ np.sin(extraF * 2.0*np.pi*x)
    return x, y
    
# Fiddle with figure settings here:
plt.rcParams['figure.figsize'] = (10,8)
plt.rcParams['font.size'] = 14
plt.rcParams['image.cmap'] = 'plasma'
plt.rcParams['axes.linewidth'] = 2
# Set the default colour cycle (in case someone changes it...)
cols = plt.get_cmap('tab10').colors

# Sample configuration
num_samples = 500000
N=2048
T=1.0/800
F=50.0

# F_vals = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0, 140.0,150.0,160.0,170.0,180.0,190.0, 200.0, 210.0]

F_vals = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]

# Intrasample configuration
num_elements = 1
interval_per_element = 0.01
total_num_elements = int(num_elements / interval_per_element)
starting_point = int(0 - 0.5*total_num_elements)

# Other configuration
num_samples_visualize = 1

# Containers for samples and subsamples
samples = []
xs = []
ys = []

ptr = 0
# Generate samples
for j in range(0, num_samples):
  # Report progress
  if j % 10000 == 0:
    print(F_vals[ptr])
    print(j)
    if(F_vals[ptr] == F_vals[-1]):
        ptr = -1    
    ptr = ptr+1
    
  # Generate wave
  xs, ys = generateSignal(N=N, T=T, F=F_vals[ptr])
  # Append wave to samples
  samples.append((xs, ys))
  # Clear subsample containers for next sample
  xs = []
  ys = []

# Input shape
print(np.shape(np.array(samples[0][0])))
  
# Save data to file for re-use
np.save('./signal_waves_medium_2048_500k.npy', samples)

# Visualize a few random samples
for i in range(0, num_samples_visualize):
  random_index = np.random.randint(0, len(samples)-1)
  x_axis, y_axis = samples[random_index]
  plt.plot(x_axis, y_axis)
  plt.title(f'Visualization of sample {random_index} --- F={F}, T=1/800, N={N}')
  plt.xlabel("Time (s)")
  plt.ylabel("Amplitude")
  plt.grid()
  plt.show()
