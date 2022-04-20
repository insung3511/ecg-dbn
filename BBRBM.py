from RBM import RBMBase
import torch

class RBMBer(RBMBase):
    def __init__(self, vis_num, hid_num):
        RBMBase.__init__(self, vis_num, hid_num)

    def p_h_given_v(self, v):
        return torch.sigmoid(torch.matmul(v, self.w.t()) + self.b)

    def sample_h_given_v(self, v):
        h_prob = self.p_h_given_v(v)
        r = torch.rand(self.hid_num)
        print(((h_prob > r).float()), " | \n", h_prob)

        # Binary probability
        return (h_prob > r).float(), h_prob

    def p_v_given_h(self, h):
        return torch.sigmoid(torch.matmul(self.w.t(), h) + self.a)

    def cd(self, v_data, k=1, eta=0.2, alpha=0.9, lam=0.0):
        return RBMBase.cd(self, v_data, k, eta, alpha, lam)