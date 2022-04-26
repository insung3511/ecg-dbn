from RBM import RBMBase
import torch

class RBMBer(RBMBase):
    def __init__(self, vis_num, hid_num):
        RBMBase.__init__(self, vis_num, hid_num)

    def p_h_given_v(self, v):
        index_tensor = torch.Tensor.long(torch.ones(self.vis_num, 0))
        
        print("w size : ", self.w.size(), "\tw numel : ", torch.numel(self.w))
        print("OG!! w_t size : ", self.w.t().size(), "\tw_t numel : ", torch.numel(self.w.t()))

        w_t = (self.w.t().clone()).scatter_(0, index_tensor, (self.w.t().clone()))
        v_i = (torch.ones(self.vis_num * self.hid_num))
        v_i = (v.clone().detach())

        '''
        w_t dimension setting up results..
            0 : [1, 80, 180]
            1 : [80, 1, 180]
        '''
        print("w_t size : ", w_t.size(), "\tw_t numel : ", torch.numel(w_t))

        w_t = w_t.unsqueeze(0)
        return torch.sigmoid(
            torch.matmul(v_i, w_t) + self.b         
        )

    def sample_h_given_v(self, v):
        h_prob = self.p_h_given_v(v)
        r = torch.rand(self.hid_num)

        # Binary probability
        return h_prob, (h_prob > r).float()

    def p_v_given_h(self, h):
        index_tensor = torch.Tensor.long(torch.ones(self.hid_num, 0))
        h = h.clone().detach().scatter_(0, index_tensor, h.clone().detach()).view(self.hid_num, 1)
        return torch.sigmoid(
            torch.matmul(
                self.w.t(), 
                h.scatter_(0, index_tensor, h.clone())) + self.a
        )

    def cd(self, v_data, k=1, eta=0.2, alpha=0.9, lam=0.0):
        return RBMBase.cd(self, v_data, k, eta, alpha, lam)