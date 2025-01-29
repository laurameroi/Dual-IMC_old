"""
Train an acyclic REN controller for the system of 2 robots in a corridor or 12 robots swapping positions.
Author: Clara Galimberti (clara.galimberti@epfl.ch)
Usage:
python run.py                --sys_model     [SYS_MODEL]         \
                             --gpu           [USE_GPU]           \
Flags:
  --sys_model: select system where to design a controller. Available options: corridor, robots.
  --gpu: whether to use GPU.
"""

import torch
import argparse
from models import PointMassVehicle, PsiU
from src.loss_functions import f_loss_states, f_loss_u, f_loss_ca, f_loss_obst
from utils import set_params


seed = 2
torch.manual_seed(seed)

# Parameters
min_dist, t_end, n_agents, x0, xbar, linear, learning_rate, epochs, Q, alpha_u, alpha_ca, alpha_obst, n_xi, l, \
    n_traj, std_ini, mass, ts, drag_coefficient_1, drag_coefficient_2, initial_position, initial_velocity, \
    target_position, input_dim, state_dim, Kp, Kd, duration, num_signals, num_training, num_validation = set_params()

# Create the vehicle model
vehicle = PointMassVehicle(mass, ts, drag_coefficient_1, drag_coefficient_2)

#create the controller M
M = PsiU(state_dim, input_dim, n_xi, l)

y_target = torch.zeros(4)

optimizer = torch.optim.Adam(M.parameters(), lr=learning_rate)

for epoch in range(epochs):
    optimizer.zero_grad()
    loss = 0
    '''
    if epoch == 300 and sys_model == 'corridor':
        std_ini = 0.5
        optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr']
    '''

    for n in range(input_data_training.shape[0]):
        for t in range(input_data_training.shape[1]):
            if t == 0:
                xi_ = torch.zeros(Qg.n_xi)
                w = (x0.detach() - sys.xbar) + std_ini * torch.randn(x0.shape)
                u = torch.zeros(input_dim)
                x = sys.xbar
            x, _ = PointMassVehicle.base_forward(t, x, u, w)
            w = x - xxxxx
            u, xi_ = M.forward(t, w, xi_)
            loss = loss + MSE(output_data_training[n, t, :], y_hat[:])


    print("Epoch: %i --- Loss: %.4f ---" % (epoch, loss / t_end))
    loss.backward()
    optimizer.step()
    ctl.psi_u.set_model_param()
    # # # # # # # # Save trained model # # # # # # # #
    torch.save(ctl.psi_u.state_dict(), "trained_models/" + sys_model + "_tmp.pt")