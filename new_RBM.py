from datetime import datetime
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data
import torch.nn as nn
import torch

class new_RBM(nn.Module): 
    global SIZE
    cuda = torch.device('cuda')
    def __init__(self, n_vis, n_hid, k, batch):
        super(new_RBM, self).__init__()
        self.W      = nn.Parameter(torch.randn(1, 13000000) * 1e-2)
        self.v_bias = nn.Parameter(torch.zeros(13000000))
        self.h_bias = nn.Parameter(torch.zeros(13000000))
        self.k      = k
        self.batch  = batch
    
    def sample_from_p(self, p):
        return F.relu(
            torch.sign(
                p - Variable(torch.randn(p.size()))
            )
        ).cuda()

    ''' ISSUE PART '''
    def v_to_h(self, v):
        # v = (v.clone().detach()).reshape(-1, 13000000)
        h_bias = torch.flatten(self.h_bias.clone())
        print(v.size())
        print(h_bias.size())
        
        w = (self.W.clone())

        p_h = F.sigmoid(
            F.linear((v), (w), bias=h_bias)
        ).cuda()

        sample_h = self.sample_from_p(p_h)
        return p_h, sample_h

    def h_to_v(self, h):
        w = self.W.t().clone()

        p_v = F.sigmoid(
            F.linear(h, w, self.v_bias)
        ).cuda()
        
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
        v_bias_term = torch.mv(v, self.v_bias).cuda()
        h_bias = self.h_bias.clone()
        
        wx_b = F.linear(v, self.W, h_bias).cuda()
        hidden_term = wx_b.exp().add(1).log().sum(1)
        
        return (-(hidden_term) - v_bias_term).mean()
    
    def get_weight(self):
        return self.W