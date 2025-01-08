import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from models import PointMassVehicle, PsiU
from dataset import generate_input_dataset, generate_input_dataset_white_noise, generate_output_dataset
from models import DeepLRU
import scipy
import os
from os.path import dirname, join as pjoin
import time

from utils import set_params

np.random.seed(20)
y_target = torch.zeros(4)

# Parameters
min_dist, t_end, n_agents, x0, xbar, linear, learning_rate, epochs, Q, alpha_u, alpha_ca, alpha_obst, n_xi, l, \
    n_traj, std_ini, mass, ts, drag_coefficient_1, drag_coefficient_2, initial_position, initial_velocity, \
    target_position, input_dim, state_dim, Kp, Kd, duration, num_signals, num_training, num_validation = set_params()

# Create the vehicle model
vehicle = PointMassVehicle(mass, ts, drag_coefficient_1, drag_coefficient_2)

#create the model Qg
Qg = PsiU(input_dim, state_dim, n_xi, l)

#Create the controller K
#u = -Kd q

#Generate dataset
input_data = generate_input_dataset(num_signals=num_signals, ts=ts, duration=duration, input_dim=input_dim)
output_data = generate_output_dataset(input_data, vehicle, initial_position, initial_velocity, Kp, Kd, target_position)

# Initialize input (u) and output (y) tensors for training data
input_data_training = input_data[0:num_training, :, :]
output_data_training = output_data[0:num_training, :, :]
y_hat_train = torch.zeros(output_data_training.shape)
torch.zeros((input_data.shape[1], 2))
# Define the loss function
MSE = nn.MSELoss()

# Define the optimizer and learning rate
optimizer = torch.optim.Adam(Qg.parameters(), lr=learning_rate)
optimizer.zero_grad()

# Training loop settings
LOSS = np.zeros(epochs)

# Start training timer
t0 = time.time()
for epoch in range(epochs):
    # Adjust learning rate at specific epochs
    if epoch == epochs - epochs / 2:
        learning_rate = 1.0e-3
        optimizer = torch.optim.Adam(Qg.parameters(), lr=learning_rate)
    if epoch == epochs - epochs / 6:
        learning_rate = 1.0e-3
        optimizer = torch.optim.Adam(Qg.parameters(), lr=learning_rate)
    optimizer.zero_grad()
    loss = 0.0
    for n in range(input_data_training.shape[0]):
        xi_ = torch.zeros(Qg.n_xi)
        for t in range(input_data.shape[1]):
            if t == 0:
                u_K = 0.
            u_ext = (input_data[n, t, :])
            u = u_ext - u_K
            y_hat, xi_ = Qg.forward(t, u, xi_)
            u_K = torch.matmul(Kd, -y_hat[2:])
            #u_K = 0.03*(target_position-y_hat[0:2])
            loss = loss + MSE(output_data[n,t,0:2], y_hat[0:2])
            y_hat_train[n,t,:] = y_hat
    loss.backward()
    optimizer.step()
    Qg.set_model_param()
    # Print loss for each epoch
    print(f"Epoch: {epoch + 1} \t||\t Loss: {loss}")
    LOSS[epoch] = loss



# End training timer
t1 = time.time()

# Calculate total training time
total_time = t1 - t0

# Initialize input (u) and output (y) tensors for validation data
input_data_training = input_data[num_training:, :, :]
output_data_training = output_data[num_training:, :, 0:input_dim]

time_plot = np.arange(0, 10, ts)

plt.figure(figsize=(12, 8))

# Plot for each selected signal
for idx in range(1):
    plt.subplot(5, 1, idx + 1)
    plt.plot(time_plot, output_data_training[idx, 0:len(time_plot), 0].detach().numpy(), label="Real Output X", color="blue")
    plt.plot(time_plot, y_hat_train[idx, 0:len(time_plot), 0].detach().numpy(), label="Modelled Output X", linestyle="--", color="orange")
    plt.plot(time_plot, output_data_training[idx, 0:len(time_plot), 1].detach().numpy(), label="Real Output Y", color="green")
    plt.plot(time_plot, y_hat_train[idx, 0:len(time_plot), 1].detach().numpy(), label="Modelled Output Y", linestyle="--", color="red")
    plt.title(f"Real vs Modelled Outputs for Signal {idx}")
    plt.xlabel("Time (s)")
    plt.ylabel("Output")
    plt.legend()
    plt.tight_layout()

plt.show()