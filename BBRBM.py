from numpy import dtype
from RBM import RBMBase
import torch

class RBMBer(RBMBase):
    def __init__(self, vis_num, hid_num):
        RBMBase.__init__(self, vis_num, hid_num)

    def p_h_given_v(self, v):
        index_tensor = torch.Tensor.long(torch.ones(self.vis_num, 0))
        
        w_t = (self.w.t().clone()).scatter_(0, index_tensor, (self.w.t().clone()))
        print(type(self.w.t()))
        
        print(v.size())
        v_i = (v.clone()).view(self.vis_num * self.hid_num)
        print(type(v_i), v_i.size())

        return torch.sigmoid(
            torch.matmul(
                v_i, 
                w_t + self.b
            )
        )

    def sample_h_given_v(self, v):
        h_prob = self.p_h_given_v(v)
        r = torch.rand(self.hid_num)

        # Binary probability
        return h_prob, (h_prob > r).float()

    def p_v_given_h(self, h):
        index_tensor = torch.Tensor.long(torch.ones(self.hid_num, 0))
        return torch.sigmoid(
            torch.matmul(
                self.w.t(), 
                h.scatter_(0, index_tensor, h.clone())) + self.a
        )

    def cd(self, v_data, k=1, eta=0.2, alpha=0.9, lam=0.0):
        return RBMBase.cd(self, v_data, k, eta, alpha, lam)