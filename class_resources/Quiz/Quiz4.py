# Convolution Calcuations
import numpy as np
from scipy.signal import convolve2d

signal = np.array([[1,1,1,0,0], [0,1,1,1,0], [0,0,1,1,1], [0,0,1,1,0], [0,1,1,0,0]])
kernal = np.array([[1,0,1], [0,1,0], [1,0,1]])

output = convolve2d(signal, kernal, mode='valid')
print(output)

# Max pooling / average pooling
import torch
import torch.nn as nn

input_data = torch.tensor([[4,9,2,5,8,3], [5,6,2,4,0,3], [2,4,5,4,5,2], [5,6,5,4,7,8], [5,7,7,9,2,1], [5,8,5,3,8,4]], dtype=torch.float32)

pool = nn.AvgPool2d(kernel_size=2, stride=2)
output = pool(input_data.unsqueeze(0).unsqueeze(0))
print(output)

# Convolutional Network
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Input

num_filters = 32
filter_size = (3,3)
pool_size = (2,2)

model = Sequential([
  Input(shape=(28, 28, 1)),  
  Conv2D(num_filters, filter_size, activation='relu'),
  MaxPooling2D(pool_size=pool_size),
  Flatten(),
  Dense(128, activation='relu'),
  Dense(10, activation='softmax')
])

print(model.summary())