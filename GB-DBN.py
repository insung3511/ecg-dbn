# reference from : https://github.com/meownoid/tensorflow-rbm

import torch.nn.functional as F
import torch.nn as nn
import torch

VISIBLE_UNITS = 12200
HIDDEN_UNITS = 8000
K_FOLD  = 5
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
MOMENTUM = 0.5

class GB_RBM():
    def __init__(self):
        self.visible = nn.Parameter(torch.randn(1, VISIBLE_UNITS))
        self.hidden = nn.Parameter(torch.randn(1, HIDDEN_UNITS))
        self.weight = nn.Parameter(torch.randn(HIDDEN_UNITS, VISIBLE_UNITS))
        self.k = K_FOLD
        super().__init__(VISIBLE_UNITS, HIDDEN_UNITS)

    def step(self, x):
        hidden_p = nn.sigmoid(torch.matmul(x) + self.hidden)
        weight_mut_hidden = F.linear(self.visible, self.weight) + self.hidden)

        if self.visible:
            visible = nn.sigmoid(nn.transpose(x), hidden_p)

    def sigmoid(self, x):
        return nn.Sigmoid(x)