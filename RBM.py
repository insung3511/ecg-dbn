"""
PyTorch implementation of all (Bernoulli, Gaussian hid, Gaussian vis+hid) kinds of RBMs.

(C) Kai Xu
University of Edinburgh, 2017 

Reference from https://github.com/xukai92/pytorch-rbm/blob/ea88786dc8352dae59a4e306ad8fe4d274e13c14/rbm.py
"""
from matplotlib import testing
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import torch
import gc

FLATTEN_DIM = 0

def numel(tensor):
    return torch.numel(tensor)

class RBMBase:
    def __init__(self, vis_num, hid_num):
        self.vis_num = vis_num
        self.hid_num = hid_num

        # Dictionary for storing parameters
        self.params = dict()

        self.w = nn.Parameter(torch.randn(hid_num, vis_num) * 0.1)  # weight matrix
        self.a = torch.zeros(vis_num) / vis_num                     # bias for visiable units
        self.b = torch.zeros(hid_num)                               # bias for hidden units
        
        # Corresponding momentums; _v means velocity of
        self.w_v = torch.zeros(vis_num, hid_num)
        self.a_v = torch.zeros(vis_num)
        self.b_v = torch.zeros(hid_num)
    
        self.has_gpu = False
            
    def p_h_given_v(self, v):
        raise NotImplementedError()

    def sample_h_given_v(self, v):
        raise NotImplementedError()

    def p_v_given_h(self, h):
        raise NotImplementedError()

    def cd(self, v_data, k, eta, alpha, lam):
        self.w = nn.Parameter(torch.randn(self.hid_num, self.vis_num) * 0.1)
        """
        Perform contrastive divergence with k stpes, i.e. CD_k.

        @input
            v_data: visible data
                 k: MCMC step number
               eta: learning rate
             alpha: momentum coefficient
               lam: weight decay rate

        @return
            error: reconstruction error
        """

        # print("v_data size : ", v_data.size(), "\tv_data numel : ", torch.numel(v_data))
        
        # Positive phase
        h_pos, h_prob_pos = self.sample_h_given_v(v_data)

        # Negative phase
        h_neg = h_pos.clone()

        for _ in range(k):
            v_prob_neg = self.p_v_given_h(h_neg)
            h_neg, h_prob_neg = self.sample_h_given_v(v_prob_neg)

        '''     STATS POSTIVIE    '''

        # Compute statistics
        # Postivie way (Visible to Hidden)
        index_tensor = torch.Tensor.long(torch.ones(self.vis_num, 0))
        v_data = v_data.unsqueeze(1)
        h_prob_pos = h_prob_pos.unsqueeze(0)

        stats_pos = torch.matmul(
            (v_data).scatter_(0, index_tensor, v_data.clone()), 
            h_prob_pos
        )

        stats_pos = torch.flatten(stats_pos.clone(), start_dim=FLATTEN_DIM)
        # print("stats_pos size : ", stats_pos.size(), " \tstats_pos elemetns count : ", torch.numel(stats_pos))

        '''     STATS NEGATIVE    '''

        # Postivie way (Hidden to Visible)
        index_tensor = torch.Tensor.long(torch.ones(self.hid_num, 0))
        v_prob_neg_t = (v_prob_neg.t()).unsqueeze(0)
        h_prob_neg = h_prob_neg.unsqueeze(0)

        stats_neg = torch.matmul(
            v_prob_neg_t,
            (h_prob_neg).scatter_(0, index_tensor, h_prob_neg.clone())
        )

        stats_neg = torch.flatten(stats_neg.clone(), start_dim=FLATTEN_DIM)
        # print("stats_neg size : ", stats_neg.size(), " \tstats_neg elemetns count : ", torch.numel(stats_neg))

        # Compute gradients
        batch_size = v_data.size()[0]
        w_grad = (stats_pos - stats_neg) / batch_size
        a_grad = torch.sum(v_data - v_prob_neg, 0) / batch_size
        b_grad = torch.sum(h_prob_pos - h_prob_neg, 0) / batch_size

        w_grad = torch.flatten(w_grad.clone(), start_dim=FLATTEN_DIM)
        # print("w_grad size : ", w_grad.size(), "\tw_grad elemetns count : ", torch.numel(w_grad))

        # Update momentums
        self.w = (self.w.clone()).view(self.vis_num * self.hid_num)

        # print("\t\t\tWould you give some mint chocolate?")

        try:
            # print("\t\tYea, give that shit!")
            testing_tensor = (w_grad.clone() - lam).view(self.vis_num * self.hid_num)
        
        except RuntimeError:
            # print("\t\tWakk... worst food ever...\a")
            testing_tensor = w_grad.clone() - lam
        
        self.w_v = torch.flatten(self.w_v.clone(), start_dim=0)
        # print("self.w_v numel and size : \t\t", torch.numel(self.w_v), (self.w_v).size())
        
        # print("self.w numel and size : \t\t", torch.numel(self.w), (self.w).size())
        

        # print("\t[FIRSTT] \"testing_tensor\" Weight - lamba Tensor size : ", testing_tensor.size())

        # ISSUE MAIN PART
        #         scalar |   var x  | scalar
        self.w_v = alpha * self.w_v + eta * testing_tensor
        self.a_v = alpha * self.a_v + eta * a_grad
        self.b_v = alpha * self.b_v + eta * b_grad

        
        # print(torch.numel(alpha * self.w_v))
        # print(torch.numel(eta * testing_tensor))

        # Update parameters     
        
        try:
            # print(self.vis_num, " ****** ", self.hid_num)
            self.w_v = ((self.w_v).clone()).view(self.vis_num * self.vis_num * self.hid_num)

        except RuntimeError:
            self.w_v = torch.flatten((self.w_v).clone(), start_dim=FLATTEN_DIM)
            self.w_v = ((self.w_v).clone()).unsqueeze(0)
        self.w = ((self.w).clone()).unsqueeze(1)

        #print((self.w).size())

        gc.collect()
        # print("\t\a[RAME] Starting update weights...")
        self.w = self.w + self.w_v
        # print("\t\a[RAME] Updated weights!")

        self.a = self.a + self.a_v
        self.b = self.b + self.b_v

        # Compute reconstruction error
        error = F.mse_loss(v_data, v_prob_neg, size_average=False)
        del self.w
        return error