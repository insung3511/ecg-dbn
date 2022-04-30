from RBM import RBMBase
import torch

class RBMGaussHid(RBMBase):
    def __init__(self, vis_num, hid_num):
        RBMBase.__init__(self, vis_num, hid_num)

    def p_h_given_v(self, v):
        w = self.w.clone()
        if v.dim() != 2:
            v = v.clone().view(self.vis_num, 1)

        if list(v.size()) == list(w.size()):
            v = v.clone().view(list(w.size())[1], list(w.size())[0])

        print(v, w)

        print("<-----------", w.size(), "<==============", v.size())
        print("----------->", w.dim(), "==============>", v.dim())
        
        return torch.matmul(
            w, v
        )   + self.b
        
    def sample_h_given_v(self, v):
        h_prob = self.p_h_given_v(v)
        r = torch.randn(h_prob.size())
        
        return (h_prob + r), h_prob

    def p_v_given_h(self, h):
        print(h.size())

        index_tensor = torch.Tensor.long(torch.ones(self.hid_num, 0))

        # ISSUE PART
        # h = h.clone().scatter_(0, index_tensor, h.clone().detach()).view(self.hid_num, 1)

        return torch.sigmoid(
            torch.matmul(
                h.scatter_(0, index_tensor, h.clone()), 
                torch.reshape(self.w.t(), (self.hid_num, self.vis_num))
            ) + self.a
        )

    def cd(self, v_data, k=1, eta=0.001, alpha=0.5, lam=0.0002):
        return RBMBase.cd(self, v_data, k, eta, alpha, lam)