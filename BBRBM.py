from RBM import RBMBase
import torch

class RBMBer(RBMBase):
    def __init__(self, vis_num, hid_num):
        RBMBase.__init__(self, vis_num, hid_num)

    def p_h_given_v(self, v):
        index_tensor = torch.Tensor.long(torch.ones(self.vis_num, 0))

        # print(self.w.t().size())

        # epoch = 1 -> ISSUE PART
        w_t = (self.w.t().clone()).scatter_(0, index_tensor, (self.w.t().clone())).unsqueeze(1).reshape(self.vis_num, self.hid_num)

        # v = (v.clone()).view(1, self.vis_num)
        # print("w_t size\t\t: ", w_t.size(), "\tw_t numel\t\t: ", torch.numel(w_t))
        # print("v size\t\t: ", v.size(), "\tv numel\t\t: ", torch.numel(v))
        
        '''
        w_t dimension setting up results..
            0 : [1, 80, 180]
            1 : [80, 1, 180]
        '''
        
        return torch.sigmoid(
            torch.matmul(v, w_t) + self.b
        )

    def sample_h_given_v(self, v):
        '''v
        v is v_data that original data from model.py.
        might be size is [80, 1].
        '''

        h_prob = self.p_h_given_v(v)
        r = torch.rand(self.hid_num)

        # Binary probability
        return h_prob, (h_prob > r).float()

    def p_v_given_h(self, h):
        index_tensor = torch.Tensor.long(torch.ones(self.hid_num, 0))
        h = h.clone().detach().scatter_(0, index_tensor, h.clone().detach()).view(1, self.hid_num)
        return torch.sigmoid(
            torch.matmul(
                h.scatter_(0, index_tensor, h.clone()),
                torch.reshape(self.w.t(), (self.hid_num, self.vis_num))
            )   + self.a
        )    

    def cd(self, v_data, k=1, eta=0.2, alpha=0.9, lam=0.0):
        return RBMBase.cd(self, v_data, k, eta, alpha, lam)