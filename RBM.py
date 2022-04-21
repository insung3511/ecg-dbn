"""
PyTorch implementation of all (Bernoulli, Gaussian hid, Gaussian vis+hid) kinds of RBMs.

(C) Kai Xu
University of Edinburgh, 2017 

Reference from https://github.com/xukai92/pytorch-rbm/blob/ea88786dc8352dae59a4e306ad8fe4d274e13c14/rbm.py
"""
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np

class RBMBase:
    def __init__(self, vis_num, hid_num):
        self.vis_num = vis_num
        self.hid_num = hid_num

        # Dictionary for storing parameters
        self.params = dict()

        # self.w = torch.randn(vis_num, hid_num) * 0.1    # weight matrix
        self.w = nn.Parameter(torch.randn(hid_num, vis_num) * 0.1)
        self.a = torch.zeros(vis_num) / vis_num         # bias for visiable units
        self.b = torch.zeros(hid_num)                   # bias for hidden units
        
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

        # Positive phase
    #   tensor, tensor(but change it to src)
        h_pos, h_prob_pos = self.sample_h_given_v(v_data)
        # Negative phase
        h_neg = h_pos.clone()

        for _ in range(k):
            v_prob_neg = self.p_v_given_h(h_neg)
            h_neg, h_prob_neg = self.sample_h_given_v(v_prob_neg)

        # Compute statistics
                #   tensor            matrix
        # print(type(h_prob_pos), type(v_data))
        stats_pos = torch.tensor(np.matmul(v_data, h_prob_pos))
        print(type(stats_pos))
        # stats_pos = torch.matmul(h_prob_pos, v_data)
        stats_neg = torch.matmul(v_prob_neg.t(), h_prob_neg)
        
        # Compute gradients
        batch_size = v_data.size()[0]
        w_grad = (stats_pos - stats_neg) / batch_size
        a_grad = torch.sum(v_data - v_prob_neg, 0) / batch_size
        b_grad = torch.sum(h_prob_pos - h_prob_neg, 0) / batch_size

        # Update momentums
        self.w_v = alpha * self.w_v + eta * (w_grad - lam * self.w)
        self.a_v = alpha * self.a_v + eta * a_grad
        self.b_v = alpha * self.b_v + eta * b_grad

        # Update parameters
        self.w = self.w + self.w_v
        self.a = self.a + self.a_v
        self.b = self.b + self.b_v

        # Compute reconstruction error
        error = F.mse_loss(v_data, v_prob_neg, size_average=False)
        return error