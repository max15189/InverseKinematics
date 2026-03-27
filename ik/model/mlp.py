"""
Multi-Layer Perceptron for local inverse kinematics.
"""

import torch.nn as nn


class MLP(nn.Module):
    """
    Fully-connected network that predicts joint deltas dq given the
    current joint config q1 and the target end-effector position x_d.

    Args:
        input_dim:  size of the normalised input (default 9 = 6 + 3)
        hidden_dim: width of each hidden layer
        output_dim: size of the output (default 6 joint deltas)
        n_layers:   number of hidden layers
    """

    def __init__(
        self,
        input_dim: int  = 9,
        hidden_dim: int = 256,
        output_dim: int = 6,
        n_layers: int   = 4,
    ):
        super().__init__()
        layers = []
        in_dim = input_dim
        for _ in range(n_layers):
            layers += [nn.Linear(in_dim, hidden_dim), nn.LeakyReLU()]
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
