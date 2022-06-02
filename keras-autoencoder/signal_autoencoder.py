import keras
from keras.models import Sequential
from keras.layers import Conv2D, Conv2DTranspose
from keras.constraints import max_norm
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.fftpack import fft, fftfreq
import tensorflow as tf
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
print(tf.__version__)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
tf.test.is_gpu_available()
from keras.utils.vis_utils import plot_model
# import visualkeras
from cycler import cycler

# Fiddle with figure settings here:
plt.rcParams['figure.figsize'] = (10,8)
plt.rcParams['font.size'] = 14
plt.rcParams['image.cmap'] = 'plasma'
plt.rcParams['axes.linewidth'] = 2
# Set the default colour cycle (in case someone changes it...)
cols = plt.get_cmap('tab10').colors
plt.rcParams['axes.prop_cycle'] = cycler(color=cols)



def fetchFFT(x,y,N,T):
    yf = fft(y)
    xf = fftfreq(N, T)[:N//2]
    # plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
    # plt.show()
    return xf, 2.0/N * np.abs(yf[0:N//2])

# Model configuration
# width, height = 256, 8  # 128, 8  
# width, height = 512,8
width, height = 1024,8
input_shape = (width, height, 1)
batch_size = 100 #150
no_epochs = 5
train_test_split = 0.3
validation_split = 0.2
verbosity = 1
max_norm_value = 2.0

T = 1.0/800.0
# N = 4096 #2048
N = 8196
S = 120
SNR = -10
# F = 40.0
# F = 6669.564
F = 1667.359
extraF = 0.0


# Load data
data_noisy = np.load('./signal_waves_noisy_8192_rlmi_v3.npy')
x_val_noisy, y_val_noisy = data_noisy[:,0], data_noisy[:,1]
data_pure = np.load('./signal_waves_medium_8192_rlmi_v3.npy')
x_val_pure, y_val_pure = data_pure[:,0], data_pure[:,1]

# x_val_noisy = x_val_noisy.real * x_val_noisy.real + x_val_noisy.imag * x_val_noisy.imag
# y_val_noisy = y_val_noisy.real * y_val_noisy.real + y_val_noisy.imag * y_val_noisy.imag
# x_val_pure = x_val_pure.real * x_val_pure.real + x_val_pure.imag * x_val_pure.imag
# y_val_pure = y_val_pure.real * y_val_pure.real + y_val_pure.imag * y_val_pure.imag


# Reshape data
y_val_noisy_r = []
y_val_pure_r = []
for i in range(0, len(y_val_noisy)):
  print(i)
  noisy_sample = y_val_noisy[i]
  pure_sample = y_val_pure[i]
  noisy_sample = (noisy_sample - np.min(noisy_sample)) / (np.max(noisy_sample) - np.min(noisy_sample))
  pure_sample = (pure_sample - np.min(pure_sample)) / (np.max(pure_sample) - np.min(pure_sample))
  noisy_sample = noisy_sample.reshape(width, height)
  pure_sample  = pure_sample.reshape(width, height)
  y_val_noisy_r.append(noisy_sample)
  y_val_pure_r.append(pure_sample)
y_val_noisy_r   = np.array(y_val_noisy_r)
y_val_pure_r    = np.array(y_val_pure_r)
noisy_input     = y_val_noisy_r.reshape((y_val_noisy_r.shape[0], y_val_noisy_r.shape[1], y_val_noisy_r.shape[2], 1))
pure_input      = y_val_pure_r.reshape((y_val_pure_r.shape[0], y_val_pure_r.shape[1], y_val_pure_r.shape[2], 1))

# Train/test split
percentage_training = math.floor((1 - train_test_split) * len(noisy_input))
print("train percent: " + str(percentage_training))
noisy_input, noisy_input_test = noisy_input[:percentage_training], noisy_input[percentage_training:]
pure_input, pure_input_test = pure_input[:percentage_training], pure_input[percentage_training:]

# Create the model
model = Sequential()
model.add(Conv2D(128, kernel_size=(3, 3), kernel_constraint=max_norm(max_norm_value), activation='relu', kernel_initializer='he_uniform', input_shape=input_shape))
model.add(Conv2D(32, kernel_size=(3, 3), kernel_constraint=max_norm(max_norm_value), activation='relu', kernel_initializer='he_uniform'))
model.add(Conv2DTranspose(32, kernel_size=(3,3), kernel_constraint=max_norm(max_norm_value), activation='relu', kernel_initializer='he_uniform'))
model.add(Conv2DTranspose(128, kernel_size=(3,3), kernel_constraint=max_norm(max_norm_value), activation='relu', kernel_initializer='he_uniform'))
model.add(Conv2D(1, kernel_size=(3, 3), kernel_constraint=max_norm(max_norm_value), activation='sigmoid', padding='same'))




model.summary()

# plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
#visualkeras.layered_view(model,legend=True).show()

# Compile and fit data
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print('model compiled')
history = model.fit(noisy_input, pure_input,
                epochs=no_epochs,
                batch_size=batch_size,
                validation_split=validation_split)

plt.subplot(121)
plt.title("loss")
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.grid()
plt.subplot(122)
plt.title("accuracy")
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.legend()
plt.grid()
plt.plot()
plt.show()


# Generate reconstructions
num_reconstructions = 4
samples = noisy_input_test[:num_reconstructions]
model.save('rlmi-v3')
reconstructions = model.predict(samples)

# Plot reconstructions
for i in np.arange(0, num_reconstructions):
  # Prediction index
  prediction_index = i + percentage_training
  # Get the sample and the reconstruction
  original = y_val_noisy[prediction_index]
  pure = y_val_pure[prediction_index]
  reconstruction = np.array(reconstructions[i]).reshape((width * height,))
  # Matplotlib preparations
  fig, axes = plt.subplots(1, 2)
  # Plot sample and reconstruciton
  axes[0].plot(original)
  axes[0].set_title('Noisy waveform')
  # axes[1].plot(pure)
  # axes[1].set_title('Pure waveform')
  axes[1].plot(reconstruction)
  axes[1].set_title('Autoencoder Denoised waveform')
  
  plt.figure(2)
  plt.subplot(121)
  xf, meanyf = fetchFFT(original, original, N, T)
  plt.plot(xf, meanyf)
  # plt.subplot(132)
  # xf, meanyf = fetchFFT(pure, pure, N, T)
  # plt.plot(xf, meanyf)
  plt.subplot(122)
  # xf, meanyf = fetchFFT(reconstruction, reconstruction, N, T)

  # plt.plot(xf, meanyf)
  xf, meanyf = fetchFFT(reconstruction, reconstruction-np.mean(reconstruction), N, T)

  plt.plot(xf, meanyf)

  plt.show()
  
print('end')