import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from models import PointMassVehicle
from dataset import generate_input_dataset, generate_input_dataset_white_noise, generate_output_dataset
from models import DeepLRU
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
    "n_layers": 3,
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

np.random.seed(20)

# Parameters
min_dist, t_end, n_agents, x0, xbar, linear, learning_rate, epochs, Q, alpha_u, alpha_ca, alpha_obst, n_xi, l, \
    n_traj, std_ini, mass, ts, drag_coefficient_1, drag_coefficient_2, initial_position, initial_velocity, \
    target_position, input_dim, state_dim, Kp, Kd, duration, num_signals, num_training, num_validation = set_params()

# Create the vehicle model
vehicle = PointMassVehicle(mass, ts, drag_coefficient_1, drag_coefficient_2)

#Generate dataset
input_data = generate_input_dataset(num_signals=num_signals, ts=ts, duration=duration, input_dim=input_dim)
output_data = generate_output_dataset(input_data, vehicle, initial_position, initial_velocity, Kp, Kd, target_position)

#System identification

# Initialize input (u) and output (y) tensors for training data
u = input_data[0:num_training, :, :]
y = output_data[0:num_training, :, :]

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

    optimizer.zero_grad()  # Reset gradients
    loss = 0  # Initialize loss

    # Forward pass through the SSM
    ySSM, _ = Qg(u, state=None, mode="scan")
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

# Initialize input (u) and output (y) tensors for validation data
uval = input_data[num_training:, :, :]
yval = output_data[num_training:, :, :]


# Forward pass through the SSM for validation data
ySSM_val, _ = Qg(uval, state=None, mode="scan")
yval = torch.squeeze(yval)

# Compute validation loss
loss_val = MSE(ySSM_val, yval)


#simulate OPEN LOOP FOR 1 TRAJECTORY
# Initial conditions
p = initial_position
q = initial_velocity
# Predefine tensors to store the results
positions_open = torch.zeros((input_data.shape[1], 2))  # Store all positions
velocities_open = torch.zeros((input_data.shape[1], 2))  # Store all velocities

# Set initial conditions
positions_open[0] = p
velocities_open[0] = q

for t in range(1, input_data.shape[1]):
    # Compute next state using the forward dynamics
    F = input_data[0, t, :]
    p, q = vehicle.forward(p, q, F)
    positions_open[t] = p
    velocities_open[t] = q

#CLOSED LOOP FOR 1 TRAJECTORY WITH BASE P CONTROLLER
# Initial conditions
p = initial_position
q = initial_velocity

# Predefine tensors to store the results
positions_closed = torch.zeros((input_data.shape[1], 2))  # Store all positions
velocities_closed = torch.zeros((input_data.shape[1], 2))  # Store all velocities

# Set initial conditions
positions_closed[0] = p
velocities_closed[0] = q

for t in range(1, input_data.shape[1]):
    F = input_data[0, t, :]
    # Compute next state using the forward dynamics
    p, q = vehicle.base_forward(p, q, F, Kp, Kd, target_position)
    positions_closed[t] = p
    velocities_closed[t] = q


# Plotting the trajectories
plt.figure(figsize=(12, 6))

# Plot open-loop trajectory
plt.subplot(1, 2, 1)
plt.plot(positions_open[:, 0], positions_open[:, 1], label="Open-loop trajectory")
plt.scatter(positions_open[0, 0], positions_open[0, 1], color='red', label="Start")
plt.scatter(positions_open[-1, 0], positions_open[-1, 1], color='green', label="End")
plt.title("Open-loop Trajectory")
plt.xlabel("X position (m)")
plt.ylabel("Y position (m)")
plt.legend()

# Plot closed-loop trajectory
plt.subplot(1, 2, 2)
plt.plot(positions_closed[:, 0], positions_closed[:, 1], label="Closed-loop trajectory")
plt.scatter(positions_closed[0, 0], positions_closed[0, 1], color='red', label="Start")
plt.scatter(positions_closed[-1, 0], positions_closed[-1, 1], color='green', label="End")
plt.title("Closed-loop Trajectory")
plt.xlabel("X position (m)")
plt.ylabel("Y position (m)")
plt.legend()

plt.tight_layout()
plt.show()

time_plot = np.arange(0, 10, ts)
# Plot a few samples of the inputs to visualize the horizontal and vertical forces
plt.figure(figsize=(12, 8))
for i in range(5):  # Plot 5 samples
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
num_signals_to_plot = 5  # Number of trajectories to visualize
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
for idx in range(5):
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(time_plot, ySSM[idx, 0:len(time_plot), 0].detach().numpy(), label='SSM')
    plt.plot(time_plot, y[idx, 0:len(time_plot), 0].detach().numpy(), label='y train')
    plt.title("Output Train Single SSM")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(time_plot, ySSM_val[idx, 0:len(time_plot), 0].detach().numpy(), label='SSM val')
    plt.plot(time_plot, yval[idx, 0:len(time_plot), 0].detach().numpy(), label='y val')
    plt.title("Output Val Single SSM")
    plt.legend()
    plt.show()


# Print the final validation loss
print(f"Loss Validation single SSM: {loss_val}")


# Select a few signals to plot
num_signals_to_plot = 5  # Number of trajectories to visualize
signal_indices = torch.randint(0, num_training, (num_signals_to_plot,))  # Randomly select signal indices

plt.figure(figsize=(12, 8))

# Plot for each selected signal
for i, idx in enumerate(signal_indices):
    plt.subplot(num_signals_to_plot, 1, i + 1)
    plt.plot(time_plot, y[idx, 0:len(time_plot), 0].detach().numpy(), label="Real Output X", color="blue")
    plt.plot(time_plot, ySSM[idx, 0:len(time_plot), 0].detach().numpy(), label="Modelled Output X", linestyle="--", color="orange")
    plt.plot(time_plot, y[idx, 0:len(time_plot), 1].detach().numpy(), label="Real Output Y", color="green")
    plt.plot(time_plot, ySSM[idx, 0:len(time_plot), 1].detach().numpy(), label="Modelled Output Y", linestyle="--", color="red")
    plt.title(f"Real vs Modelled Outputs for Signal {idx} in training set")
    plt.xlabel("Time (s)")
    plt.ylabel("Output")
    plt.legend()
    plt.tight_layout()

plt.show()
