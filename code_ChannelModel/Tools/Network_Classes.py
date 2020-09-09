# Script to contain Network classes

import torch.nn as nn

class NetworkRegression(nn.Sequential):
    def __init__(self, inputSize, outputSize, no_layers, no_nodes):
        super(NetworkRegression, self).__init__()
        if no_layers == 0:
            self.linear = nn.Linear(inputSize, outputSize)
        if no_layers == 1:
            self.linear = nn.Sequential( nn.Linear(inputSize, no_nodes), nn.Tanh(),
                                         nn.Linear(no_nodes, outputSize) )
        elif no_layers == 2:
            self.linear = nn.Sequential( nn.Linear(inputSize, no_nodes), nn.Tanh(),
                                         nn.Linear(no_nodes, no_nodes), nn.Tanh(),
                                         nn.Linear(no_nodes, outputSize) )
        elif no_layers == 3:
            self.linear = nn.Sequential( nn.Linear(inputSize, no_nodes), nn.Tanh(),
                                         nn.Linear(no_nodes, no_nodes), nn.Tanh(),
                                         nn.Linear(no_nodes, no_nodes), nn.Tanh(),
                                         nn.Linear(no_nodes, outputSize) )
        elif no_layers == 4:
            self.linear = nn.Sequential( nn.Linear(inputSize, no_nodes), nn.Tanh(),
                                         nn.Linear(no_nodes, no_nodes), nn.Tanh(),
                                         nn.Linear(no_nodes, no_nodes), nn.Tanh(),
                                         nn.Linear(no_nodes, no_nodes), nn.Tanh(),
                                         nn.Linear(no_nodes, outputSize) )
    def forward(self, x):
        out = self.linear(x)
        return out

