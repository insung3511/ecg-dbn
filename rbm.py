'''import torch
import torch.nn as nn
import torch.nn.functional as F

class RBM(nn.Module):
    def __init__(self, n_vis = 64, n_hid=16, k=5):
        super(RBM, self).__init__()
        self.v = nn.Parameter(torch.randn(1, n_vis))
        self.h = nn.Parameter(torch.randn(1, n_hid))
        self.W = nn.Parameter(torch.randn(n_vis, n_hid))
        self.k = k
    
    def visible_to_hidden(self, v):
        p = torch.sigmoid(F.linear(v, self.h))
        return p.bernoulli()

    def hidden_to_visible(self, h):
        p = torch.sigmoid(F.linear(h, self.W.t(), self.v))
        return p.bernoulli()

    def free_energy(v, h, w):
'''

from distutils.log import warn
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

train = pd.read_csv('./data/final_db2')