from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch

class SVM(nn.Module):
    def __init__(self):
        super().__init__()
        self.fully_connected = nn.Linear(2, 1)

    def forward(self, x):
        fwd = self.fully_connected(x)
        return fwd