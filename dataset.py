import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from models import PointMassVehicle

from utils import set_params

np.random.seed(20)

min_dist, t_end, n_agents, x0, xbar, linear, learning_rate, epochs, Q, alpha_u, alpha_ca, alpha_obst, n_xi, l, \
    n_traj, std_ini, mass, ts, drag_coefficient_1, drag_coefficient_2, initial_position, initial_velocity, \
    target_position, input_dim, state_dim, Kp, Kd, duration, num_signals, num_training, num_validation = set_params()

# Define a function to generate the sinusoidal signals for horizontal and vertical forces
def generate_sinusoidal(frequency, amplitude, phase, time):
    return amplitude * np.sin(2 * np.pi * frequency * time + phase)

def generate_input_dataset(num_signals=50, ts=0.05, duration=100, input_dim=2):
    time = np.arange(0, duration, ts)  # Time from 0 to 15 seconds with a sampling time of 0.05 seconds
    data = torch.zeros(num_signals, len(time), input_dim)
    for n in range(num_signals):
        # Generate random parameters for the current sample
        frequency_x = np.random.uniform(0.5, 1)  # Horizontal frequency between 0.5 Hz and 5 Hz
        frequency_y = np.random.uniform(0.5, 1)  # Vertical frequency between 0.5 Hz and 5 Hz
        amplitude_x = np.random.uniform(0.5, 3)  # Horizontal amplitude between 0.5 and 3
        amplitude_y = np.random.uniform(0.5, 3)  # Vertical amplitude between 0.5 and 3
        phase_x = np.random.uniform(-np.pi, np.pi)  # Horizontal phase between -π and π
        phase_y = np.random.uniform(-np.pi, np.pi)  # Vertical phase between -π and π

        # Generate the signals using the random parameters
        horizontal_force = generate_sinusoidal(frequency_x, amplitude_x, phase_x, time)
        data[n, :, 0] = torch.from_numpy(horizontal_force)
        vertical_force = generate_sinusoidal(frequency_y, amplitude_y, phase_y, time)
        data[n, :, 1] = torch.from_numpy(vertical_force)
    return data


def generate_white_noise(amplitude, time):
    return amplitude * np.random.randn(len(time))  # Gaussian noise with zero mean and unit variance


def generate_input_dataset_white_noise(num_signals=50, ts=0.05, duration=100, input_dim=2):
    time = np.arange(0, duration, ts)
    data = torch.zeros(num_signals, len(time), input_dim)
    for n in range(num_signals):
        # Generate random noise signals
        amplitude_x = np.random.uniform(0.5, 3)
        amplitude_y = np.random.uniform(0.5, 3)

        horizontal_force = generate_white_noise(amplitude_x, time)
        data[n, :, 0] = torch.from_numpy(horizontal_force)

        vertical_force = generate_white_noise(amplitude_y, time)
        data[n, :, 1] = torch.from_numpy(vertical_force)
    return data

#CLOSED LOOP data generation of dimension num_signals
def generate_output_dataset(input_data, vehicle, initial_position, initial_velocity, Kp, Kd, target_position):
    # Predefine tensors to store the results

    #hard coded output dimension
    output_data = torch.zeros(input_data.shape[0], input_data.shape[1], 4)

    for n in range(output_data.shape[0]):
        for t in range(output_data.shape[1]-1):
            if t == 0:
                # Random initial conditions
                p = initial_position
                q = initial_velocity
                #p = torch.tensor(np.random.uniform(-0.2, 0.2, size=2), dtype=torch.float)
                #q = torch.tensor(np.random.uniform(-0.2, 0.2, size=2), dtype=torch.float)
                output_data[n, t, 0:2] = p
                output_data[n, t , 2:] = q

            F = input_data[n, t, :]
            # Compute next state using the forward dynamics
            p, q = vehicle.base_forward(p, q, F, Kp, Kd, target_position)
            output_data[n, t+1, 0:2] = p
            output_data[n, t+1, 2:] = q
    return output_data