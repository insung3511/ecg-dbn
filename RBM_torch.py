from nis import match

from pyparsing import col
import data.medain_filtering_class as mf
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.parallel
import torch.nn as nn
import numpy as np
import torch

class RBM():
    def __init__(self, n_visible, n_hidden):
        self.W = torch.randn(n_visible, n_hidden)
        self.v_to_h = torch.randn(1, n_hidden)
        self.h_to_v = torch.randn(1, n_visible)

    def visible_to_hidden(self, x):
        print(type(self.W.t()))
        w_x = torch.mm(x, self.W.t())
        activation = w_x + self.v_to_h.expand_as(w_x)
        p_h_to_v = torch.sigmoid(activation)
        return p_h_to_v, torch.bernoulli(p_h_to_v)

    def hidden_to_visible(self, y):
        w_y = torch.mm(y, self.W)
        activation = w_y + self.b.expand_as(w_y)
        p_v_to_h = torch.sigmoid(activation)
        return p_v_to_h, torch.bernoulli(p_v_to_h)

    def train(self, v_zero, v_k_fold, pro_hidden_zero, pro_hidden_k):
        self.W += torch.mm(v_zero.t(), pro_hidden_zero) - torch.mm(v_k_fold.t(), pro_hidden_k)
        self.v_to_h += torch.sum((v_zero - v_k_fold), 0)
        self.h_to_v += torch.sum((pro_hidden_zero - pro_hidden_k), 0)

def one_d2n_d(n_hid, og_mat):
    column = int(len(og_mat) / n_hid)
    row = int(n_hid)
    print(row, column)
    
    mat = np.matrix(np.mat(og_mat))
    mat.shape = (row, column)
    return mat

dataset_db1, dataset_db2, dataset_db3 = mf.ecg_filtering(True)
og_train_dataset = (mf.list_to_list(dataset_db1 + dataset_db2))
og_test_dataset = (mf.list_to_list(dataset_db2 + dataset_db3))

train_dataset = np.mat(og_train_dataset, dtype=np.float64)
test_dataset = np.mat(og_test_dataset, dtype=np.float64)

n_vis = 180
n_hid = 80
batch_size = 10
rbm = RBM(n_vis, n_hid)

train_tensor = one_d2n_d(n_hid, og_train_dataset)
train_tensor = torch.FloatTensor(train_tensor)
test_tensor = torch.FloatTensor(test_dataset)


n_epoch = 10
for epoch in range(1, n_epoch + 1):
    train_loss = 0
    epoch_cnt = 0

    for i in range(0, batch_size):
        v_k = train_tensor[i:i + batch_size]
        v_zero = train_tensor[i:i + batch_size]
        pro_hidden_zero, _ = rbm.visible_to_hidden(v_zero)
        
        for k in range(10):
            _, h_k = rbm.visible_to_hidden(v_k)
            _, v_k = rbm.hidden_to_visible(h_k)
            v_k[v_zero < 0] = v_zero[v_zero < 0]
        
        pro_hidden_k, _ = rbm.visible_to_hidden(v_k)
        rbm.train(v_zero=v_zero, v_k_fold=v_k, pro_hidden_zero=pro_hidden_zero, pro_hidden_k=pro_hidden_k)
        train_loss += torch.mean(torch.abs(v_zero[v_zero >= 0] - v_k[v_k >= 0]))
        epoch_cnt += 1
    print('Epoch: ', str(epoch), ' Loss: ', str(train_loss / epoch_cnt))