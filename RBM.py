from sklearn.linear_model import LinearRegression
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data
import torch.nn as nn
import torch
import numpy as np

class RBM(nn.Module):
    def __init__(self, n_vis, n_hid, k, batch):
        super(RBM, self).__init__()
        self.W      = nn.Parameter(torch.randn(1, batch) * 1e-2)
        self.v_bias = nn.Parameter(torch.zeros(batch))
        self.h_bias = nn.Parameter(torch.zeros(batch))
        self.k      = k
        self.batch  = batch
    
    #                       p is probability of model
    def sample_from_p(self, p):
        return F.relu(
            torch.sign(
                p - Variable(torch.randn(p.size()))
            )
        )

    #                v is input data from visible layer
    ''' ISSUE PART '''
    def v_to_h(self, v):
        h_bias = (self.h_bias.clone()).expand(1, 10)
        v = v.clone().expand(1, 10)
        print(h_bias.size(), v.size(), self.W.size())
        print(h_bias.dim(), v.dim(), self.W.dim())

        p_h = F.sigmoid(
            F.linear(v, self.W, bias=h_bias)
        )

        sample_h = self.sample_from_p(p_h)
        return p_h, sample_h

    def h_to_v(self, h):
        # v_bias = (self.v_bias.clone()).repeat(1, 10)
        v_bias = (self.v_bias.clone()).size(dim=1)
        print(v_bias.size())

        p_v = F.sigmoid(
            F.linear(h, self.W.t(), bias=v_bias)
        )
        sample_v = self.sample_from_p(p_v)
        return sample_v
    
    def forward(self, v):
        pre_h1, h1 = self.v_to_h(v)
        h_ = h1
        for _ in range(self.k):
            pre_v_, v_ = self.h_to_v(h_)
            pre_h_, h_ = self.v_to_h(v_)
        return v, v_

    def free_energy(self, v):
        v_bias_term = v.mv(self.v_bias)
        wx_b = F.linear(v, self.W, self.h_bias)
        hidden_term = wx_b.exp().add(1).log().sum(1)
        
        return (-(hidden_term) - v_bias_term).mean()