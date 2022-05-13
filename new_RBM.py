from datetime import datetime
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data
import torch.nn as nn
import torch

class new_RBM(nn.Module):
    def __init__(self, n_vis, n_hid, k, batch):
        super(new_RBM, self).__init__()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        device = torch.device(device)

        self.W      = nn.Parameter(torch.randn(1, batch) * 1e-2)
        self.n_vis  = n_vis
        self.n_hid  = n_hid
        self.k      = k
        self.batch  = batch
        self.v_bias = nn.Parameter(torch.zeros(batch, device=device))
        self.h_bias = nn.Parameter(torch.zeros(batch, device=device))
        self.device = device
    
    def sample_from_p(self, p):
        return F.relu(
            torch.sign(
                p - Variable(torch.randn(p.size(), device=self.device))
            )
        ).to(self.device)

    ''' ISSUE PART '''
    def v_to_h(self, v):
        w = (self.W.clone())

        p_h = F.sigmoid(
            F.linear(v, w)
        ).to(self.device)

        sample_h = self.sample_from_p(p_h)
        return p_h, sample_h

    def h_to_v(self, h):
        w = self.W.t().clone()

        p_v = F.sigmoid(
            F.linear(h, w)
        ).to(self.device)
        
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
        v_bias_term = torch.mv(v.view(1300000, self.batch), self.v_bias.view(self.batch, 1)).to(device=self.device)

        wx_b = F.linear(v, self.W)
        hidden_term = wx_b.unsqueeze(0).exp().add(1).log().sum(1)
        
        return (-(hidden_term) - v_bias_term).mean()
    
    def get_weight(self):
        return self.W