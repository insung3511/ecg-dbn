import torch.nn.functional as F
import torch.nn as nn
import torch

VISIBLE_UNITS = 2700000
HIDDEN_UNITS = 1200000
K_FOLD  = 5
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
MOMENTUM = 0.5

class RBM():
    def __init__(self):
        self.visible = nn.Parameter(torch.randn(1, VISIBLE_UNITS))
        self.hidden = nn.Parameter(torch.randn(1, HIDDEN_UNITS))
        self.weight = nn.Parameter(torch.randn(HIDDEN_UNITS, VISIBLE_UNITS))
        self.k = K_FOLD

    def visible_to_hidden(self, visible):
        p = torch.sigmoid(F.linear(visible, self.weight, self.hidden))
        return p.bernoulli()

    def hidden_to_visible(self, hidden):
        p = torch.sigmoid(F.linear(self.visible, self.weight, hidden))
        return p.bernoulli()

    def energy_function(self, visible):
        visible_term = torch.matmul(visible, self.visible.t())
        weight_mut_hidden = F.linear(visible, self.weight, self.hidden)
        
        hidden_term = torch.sum(F.softplus(weight_mut_hidden), dim = 1)
        
        return torch.mean(-(hidden_term) - visible_term)

    def forward(self, visible):
        hidden = self.visible_to_hidden(visible)
        for _ in range(self.k):
            visible_gibbs = self.hidden_to_visible(hidden)
            hidden = self.visible_to_hidden(visible_gibbs)
        return visible, visible_gibbs
