import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from models import PointMassVehicle
from dataset import generate_input_dataset, generate_input_dataset_white_noise, generate_output_dataset
import scipy
import os
from os.path import dirname, join as pjoin
import time
from utils import set_params
import math
from argparse import Namespace
from SSMs import DWN, DWNConfig
#from tqdm import tqdm

seed = 2
torch.manual_seed(seed)

# set up a simple architecture
cfg = {
    "n_u": 2,
    "n_y": 4,
    "d_model": 5,
    "d_state": 5,
    "n_layers": 1,
    "ff": "LMLP",  # GLU | MLP | LMLP
    "max_phase": math.pi,
    "r_min": 0.7,
    "r_max": 0.98,
    "gamma": False,
    "trainable": False,
    "gain": 2.4
}
cfg = Namespace(**cfg)


# Build model
config = DWNConfig(d_model=cfg.d_model, d_state=cfg.d_state, n_layers=cfg.n_layers, ff=cfg.ff, rmin=cfg.r_min,
                   rmax=cfg.r_max, max_phase=cfg.max_phase, gamma=cfg.gamma, trainable=cfg.trainable, gain=cfg.gain)
Qg = DWN(cfg.n_u, cfg.n_y, config)

total_params = sum(p.numel() for p in Qg.parameters())
print(f"Number of parameters: {total_params}")

np.random.seed(20)
y_target = torch.zeros(4)

# Parameters
min_dist, t_end, n_agents, x0, xbar, linear, learning_rate, epochs, Q, alpha_u, alpha_ca, alpha_obst, n_xi, l, \
    n_traj, std_ini, mass, ts, drag_coefficient_1, drag_coefficient_2, initial_position, initial_velocity, \
    target_position, input_dim, state_dim, Kp, Kd, duration, num_signals, num_training, num_validation = set_params()

# Create the vehicle model
vehicle = PointMassVehicle(mass, ts, drag_coefficient_1, drag_coefficient_2)

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

# Initialize input (u) and output (y) tensors for validation data
input_data_val = input_data[num_training:, :, :]
output_data_val = output_data[num_training:, :, :]
y_hat_val = torch.zeros(output_data_val.shape)

# Define the loss function
MSE = nn.MSELoss()

# Define the optimizer and learning rate
optimizer = torch.optim.Adam(Qg.parameters(), lr=learning_rate)
optimizer.zero_grad()

# Training loop settings
LOSS = np.zeros(epochs)

# Start training timer
t0 = time.time()
validation_losses = []

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

    # Training loop
    for n in range(input_data_training.shape[0]):
        for t in range(input_data_training.shape[1]):
            if t == 0:
                u_K = 0.0
                state = None
            u_ext = input_data_training[n, t, :]
            u = u_ext - u_K
            u = u.view(1, 1, 2)  # Reshape input
            y_hat, state = Qg(u, state=state, mode="loop")
            y_hat = y_hat.squeeze(0).squeeze(0)
            u_K = torch.matmul(Kd, -y_hat[2:])
            loss = loss + MSE(output_data_training[n, t, :], y_hat[:])
            y_hat_train[n, t, :] = y_hat.detach()

    # Normalize training loss
    loss /= (input_data_training.shape[0] * input_data_training.shape[1])
    loss.backward()
    optimizer.step()

    # Print training loss for this epoch
    print(f"Epoch: {epoch + 1} \t||\t Training Loss: {loss}||\t Time: {time.time() - t0}")
    LOSS[epoch] = loss.item()

    # Validation loop
    if (epoch + 1) % 20 == 0 or (epoch == epochs - 1):
        with torch.no_grad():
            val_loss = 0.0
            for n in range(input_data_val.shape[0]):
                for t in range(input_data_val.shape[1]):
                    if t == 0:
                        u_K = 0.0
                        state = None
                    u_ext = input_data_val[n, t, :]
                    u = u_ext - u_K
                    u = u.view(1, 1, 2)  # Reshape input
                    y_hat, state = Qg(u, state=state, mode="loop")
                    y_hat = y_hat.squeeze(0).squeeze(0)
                    u_K = torch.matmul(Kd, -y_hat[2:])
                    val_loss = val_loss + MSE(output_data_val[n, t, :], y_hat[:])
                    if epoch == epochs - 1:
                        y_hat_val[n, t, :] = y_hat.detach()

            # Normalize validation loss
            val_loss /= (input_data_val.shape[0] * input_data_val.shape[1])
            validation_losses.append(val_loss.item())
            print(f"Epoch: {epoch + 1} \t||\t Validation Loss: {val_loss}")


# End training timer
t1 = time.time()

# Calculate total training time
total_time = t1 - t0

# Print summary
print(f"Training completed in {total_time:.2f} seconds.")


time_plot = np.arange(0, 10, ts)
# Plot a few samples of the inputs to visualize the horizontal and vertical forces
plt.figure(figsize=(12, 8))
for i in range(2):  # Plot 5 samples
    plt.subplot(5, 1, i+1)
    plt.plot(time_plot, input_data[i,0:len(time_plot),0].numpy(), label=f"Sample {i+1} - Horizontal Force")
    plt.plot(time_plot, input_data[i,0:len(time_plot),1].numpy(), label=f"Sample {i+1} - Vertical Force")
    plt.xlabel("Time [s]")
    plt.ylabel("Force [N]")
    plt.legend()
    plt.grid(True)

plt.suptitle("Generated Horizontal and Vertical Forces")
plt.tight_layout()
plt.show()

# Select a few signals to plot
plt.figure(figsize=(12, 8))

# Plot outputs
plt.subplot(2, 1, 1)
for idx in range(5):
    plt.plot(time_plot, output_data[idx, 0:len(time_plot), 0], label=f"Signal {idx} - X Position")
    plt.plot(time_plot, output_data[idx, 0:len(time_plot), 1], label=f"Signal {idx} - Y Position", linestyle="--")
plt.title("Position Trajectories")
plt.xlabel("Time (s)")
plt.ylabel("Position (m)")
plt.legend()

# Plot velocities
plt.subplot(2, 1, 2)
for idx in range(5):
    plt.plot(time_plot, output_data[idx, 0:len(time_plot), 2], label=f"Signal {idx} - X Velocity")
    plt.plot(time_plot, output_data[idx, 0:len(time_plot), 3], label=f"Signal {idx} - Y Velocity", linestyle="--")
plt.title("Velocity Trajectories")
plt.xlabel("Time (s)")
plt.ylabel("Velocity (m/s)")
plt.legend()

plt.tight_layout()
plt.show()


# Plot training loss over epochs
plt.figure()
plt.plot(LOSS)
plt.title("LOSS")
plt.show()

# Plot training and validation outputs for different output channels
for idx in range(2):
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(time_plot, y_hat_train[idx, 0:len(time_plot), 0].detach().numpy(), label='SSM')
    plt.plot(time_plot, output_data_training[idx, 0:len(time_plot), 0].detach().numpy(), label='y train')
    plt.title("Output Train Single SSM")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(time_plot, y_hat_val[idx, 0:len(time_plot), 0].detach().numpy(), label='SSM val')
    plt.plot(time_plot, output_data_val[idx, 0:len(time_plot), 0].detach().numpy(), label='y val')
    plt.title("Output Val Single SSM")
    plt.legend()
    plt.show()


# Print the final validation loss
print(f"Loss Validation single SSM: {loss_val}")


plt.figure(figsize=(12, 8))

# Plot for each selected signal
for i in range(2):
    plt.subplot(2, 1, i + 1)
    plt.plot(time_plot, output_data_training[i, 0:len(time_plot), 0].detach().numpy(), label="Real Output X", color="blue")
    plt.plot(time_plot, y_hat_train[i, 0:len(time_plot), 0].detach().numpy(), label="Modelled Output X", linestyle="--", color="orange")
    plt.plot(time_plot, output_data_training[i, 0:len(time_plot), 1].detach().numpy(), label="Real Output Y", color="green")
    plt.plot(time_plot, y_hat_train[i, 0:len(time_plot), 1].detach().numpy(), label="Modelled Output Y", linestyle="--", color="red")
    plt.title(f"Real vs Modelled Outputs for Signal {i} in training set")
    plt.xlabel("Time (s)")
    plt.ylabel("Output")
    plt.legend()
    plt.tight_layout()

plt.show()