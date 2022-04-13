from RBM import RBMBase
import torch

class RBMGaussHid(RBMBase):
    def __init__(self, vis_num, hid_num):
        RBMBase.__init__(self, vis_num, hid_num)

    def p_h_given_v(self, v):
        return torch.matmul(v, self.w) + self.b

    def sample_h_given_v(self, v):
        h_prob = self.p_h_given_v(v)

        r = torch.randn(h_prob.size())
        if self.has_gpu:
            r = r.cuda(self.gpu_id)

        return (h_prob + r), h_prob

    def p_v_given_h(self, h):
        return torch.sigmoid(torch.matmul(h, self.w.t()) + self.a)

    def cd(self, v_data, k=1, eta=0.001, alpha=0.5, lam=0.0002):
        return RBMBase.cd(self, v_data, k, eta, alpha, lam)