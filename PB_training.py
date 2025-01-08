import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from models import PointMassVehicle
from dataset import generate_input_dataset
from dataset import generate_input_dataset_white_noise
from models import DeepLRU
import scipy
import os
from os.path import dirname, join as pjoin
import time

np.random.seed(20)

















#training and validation dataset dimensions
num_training = int(num_signals*4/5)
num_validation = num_signals-num_training

# Initialize input (u) and output (y) tensors for training data
u = input_data[0:num_training, :, :]
y = output_data[0:num_training, :, 0:2]

# Define the SSM model parameters
idd = input_dim # Input dimension
hdd = 30  # Hidden state dimension
odd = 2  # Output dimension

# Initialize the DeepLRU model (SSM)
SSM = (DeepLRU
       (N=1,
        in_features=idd,
        out_features=odd,
        mid_features=11,
        state_features=hdd,
        ))

# Define the loss function
MSE = nn.MSELoss()

# Define the optimizer and learning rate
learning_rate = 1.0e-2
optimizer = torch.optim.Adam(SSM.parameters(), lr=learning_rate)
optimizer.zero_grad()

# Training loop settings
epochs = 1500
LOSS = np.zeros(epochs)

# Start training timer
t0 = time.time()
for epoch in range(epochs):
    # Adjust learning rate at specific epochs
    if epoch == epochs - epochs / 2:
        learning_rate = 1.0e-3
        optimizer = torch.optim.Adam(SSM.parameters(), lr=learning_rate)
    if epoch == epochs - epochs / 6:
        learning_rate = 1.0e-3
        optimizer = torch.optim.Adam(SSM.parameters(), lr=learning_rate)

    optimizer.zero_grad()  # Reset gradients
    loss = 0  # Initialize loss

    # Forward pass through the SSM
    ySSM = SSM(u)
    ySSM = torch.squeeze(ySSM)  # Remove unnecessary dimensions

    # Calculate the mean squared error loss
    loss = MSE(ySSM, y)
    loss.backward()  # Backpropagate to compute gradients

    # Update model parameters
    optimizer.step()

    # Print loss for each epoch
    print(f"Epoch: {epoch + 1} \t||\t Loss: {loss}")
    LOSS[epoch] = loss

# End training timer
t1 = time.time()

# Calculate total training time
total_time = t1 - t0

# Initialize input (u) and output (y) tensors for training data
uval = input_data[num_training:, :, :]
yval = output_data[num_training:, :, 0:2]


# Forward pass through the SSM for validation data
ySSM_val = SSM(uval)
ySSM_val = torch.squeeze(ySSM_val)
yval = torch.squeeze(yval)

# Compute validation loss
loss_val = MSE(ySSM_val, yval)


