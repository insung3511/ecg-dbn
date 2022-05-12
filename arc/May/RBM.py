from datetime import datetime
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data
import torch.nn as nn
import torch

SIZE = 10

class RBM(nn.Module):
    global SIZE
    def __init__(self, n_vis, n_hid, k, batch):
        super(RBM, self).__init__()
        self.W      = nn.Parameter(torch.randn(1, batch) * 1e-2)
        self.v_bias = nn.Parameter(torch.zeros(batch))
        self.h_bias = nn.Parameter(torch.zeros(batch))
        self.k      = k
        self.batch  = batch
    
    def sample_from_p(self, p):
        return F.relu(
            torch.sign(
                p - Variable(torch.randn(p.size()))
            )
        )

    #                v is input data from visible layer
    def v_to_h(self, v):
        h_bias = (self.h_bias.clone()).expand(SIZE)
        # v = v.clone().repeat(130000)
        w = self.W.clone().repeat(SIZE, 1)

        p_h = F.sigmoid(
            F.linear(v, w, bias=h_bias)
        )

        # p_h = F.sigmoid(
        #     F.linear(v, torch.flatten(self.W), self.h_bias)
        # )        
        sample_h = self.sample_from_p(p_h)
        return p_h, sample_h

    def h_to_v(self, h):
        # v_bias = (self.v_bias.clone())
        w = self.W.t().clone().repeat(1, SIZE)

        p_v = F.sigmoid(
            F.linear(h, w, self.v_bias)
        )
        sample_v = self.sample_from_p(p_v)
        return p_v, sample_v
    
    def forward(self, v):
        start_time = datetime.now()
        pre_h1, h1 = self.v_to_h(v)
        h_ = h1

        for _ in range(self.k):
            pre_v_, v_ = self.h_to_v(h_)
            pre_h_, h_ = self.v_to_h(v_)
        estimate_time = datetime.now() - start_time
        return v, v_, estimate_time

    def free_energy(self, v):
        v = v.clone().unsqueeze(1).repeat(1, SIZE)
        v_bias = self.v_bias.clone()
        v_bias_term = torch.mv(v, self.v_bias)

        h_bias = self.h_bias.clone()
        w = self.W.clone().repeat(SIZE, 1)
        
        wx_b = F.linear(v, w, h_bias)
        hidden_term = wx_b.exp().add(1).log().sum(1)
        
        return (-(hidden_term) - v_bias_term).mean()
    
    def get_weight(self):
        return self.W
