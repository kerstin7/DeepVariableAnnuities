#import numpy as np
#from matplotlib import pyplot as plt
#import seaborn as sns
import torch
from torch import nn
#from tqdm import tqdm
#import pandas as pd
#import time

# Training auf GPU
gpu_id = 1
print('Training on GPU is possible:', torch.cuda.is_available())
device = torch.device(
    f'cuda:{gpu_id}') if torch.cuda.is_available() else torch.device('cpu')


# define network
class DeltaHedgeNN(nn.Module):

    def __init__(self, n_hidden=2, dim_input=1, dim_output=1, hidden_dim=32, activation='tanh'):
        super().__init__()

        # define parameters
        self.hidden_dim = hidden_dim
        # dimension of the stock + other informations like Time to maturity (TTM)
        self.input_dim = dim_input
        # dimension of the trading strategy (equals the stock dimension if the TTM is not in input)
        self.output_dim = dim_output
        self.num_layers = n_hidden

        # Define the model
        if activation == 'tanh':
            self.model_first_layer = nn.Sequential(nn.Linear(in_features=self.input_dim,
                                                             out_features=self.hidden_dim), nn.Tanh())
        elif activation == 'relu':
            self.model_first_layer = nn.Sequential(nn.Linear(in_features=self.input_dim,
                                                             out_features=self.hidden_dim), nn.ReLU())
        else:
            print('activation not valid')

        middle_layers = []
        if activation == 'tanh':
            for _ in range(self.num_layers):
                middle_layers.append(nn.Linear(in_features=self.hidden_dim,
                                               out_features=self.hidden_dim))
                middle_layers.append(nn.Tanh())

        elif activation == 'relu':
            for _ in range(self.num_layers):
                middle_layers.append(nn.Linear(in_features=self.hidden_dim,
                                               out_features=self.hidden_dim))
                middle_layers.append(nn.ReLU())

        self.model_middle_layer = nn.Sequential(*middle_layers)

        # unactivated output layer mapping from hiddenstate to output to realize regression
        self.out_layer = nn.Linear(in_features=self.hidden_dim,
                                   out_features=self.output_dim)

    def forward(self, S, train=False):

        if train:
            return self.network(S)

        else:
            with torch.no_grad():
                return self.network(S)

    def network(self, S):
        # initialize hidden states with zeros
        hidden_state = torch.zeros(S.shape[0], self.hidden_dim).to(device)

        # this list will be filled with the resulting hedging values at the respective time steps
        h = []

        # sequence length and first input
        sequence_length = S.shape[1]

        # iterate over sequence elements - 1
        # Note: at time T there is no trading

        for t in range(sequence_length - 1):

            # We need to add one dimension (if input is 1d then we need to add 2 dimensions)
            next_input = S[:, t, :]

            # first layer
            hidden_state = self.model_first_layer(next_input)

            # middle layers
            hidden_state = self.model_middle_layer(hidden_state)

            # readout layer
            h_t = self.out_layer(hidden_state)

            h.append(h_t)

        h = torch.stack(h, dim=1)
        if self.output_dim == 1:
            h = h.squeeze(-1)

        return h
