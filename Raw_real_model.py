# %%
from ignite.contrib.metrics.regression import *
from ignite.contrib.metrics import *
from ignite.handlers import *
from ignite.metrics import *
from ignite.engine import *
from ignite.utils import *

from sklearn.preprocessing import LabelEncoder
from collections import OrderedDict
import sklearn

from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
import data.read_samples as rs
import torch.optim as optim
import torch.utils.data
import torch.nn as nn

import matplotlib.pyplot as plt
import numpy as np
import datetime
import torch
import time
import os
import gc


print(datetime.datetime.now(), "model.py code start")

BATCH_SIZE = 143
EPOCH = 200
LEARNING_RATE = 0.2
ANNEALING_RATE = 0.999
VISIBLE_UNITS = 143
HIDDEN_UNITS = [180, 200, 250, 80, 100, 120]
K_FOLD = 1
K_FOLD = 1

# %%
device = torch.device('cuda')
print(torch.cuda.get_device_name(device))
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# %%
class RBM(nn.Module): 
    with torch.cuda.device(0):
        def __init__(self, n_vis, n_hid, k, batch):
            super(RBM, self).__init__()
            self.W      = nn.Parameter(torch.randn(1, batch, device=device) * 1e-2)
            self.n_vis  = n_vis
            self.n_hid  = n_hid
            self.k      = k
            self.batch  = batch
            self.v_bias = nn.Parameter(torch.zeros(n_vis, device=device))
            self.h_bias = nn.Parameter(torch.zeros(n_hid, device=device))
        
        def sample_from_p(self, p):
            return F.relu(
                torch.sign(
                    p - Variable(torch.randn(p.size(), device=device))
                )
            ).to(device=device)

        ''' ISSUE PART '''
        def v_to_h(self, v):
            w = (self.W.clone())

            p_h = F.sigmoid(
                # F.linear(v, w, self.h_bias)
                F.linear(v, w)
            ).to(device=device)

            sample_h = self.sample_from_p(p_h)
            return p_h, sample_h

        def h_to_v(self, h):
            w = self.W.t().clone()

            p_v = F.sigmoid(
                # F.linear(h, w, self.v_bias)
                F.linear(h, w)
            ).to(device=device)

            sample_v = self.sample_from_p(p_v)
            return p_v, sample_v
        
        def forward(self, v):
            pre_h1, h1 = self.v_to_h(v)
            h_ = h1

            for _ in range(self.k):
                pre_v_, v_ = self.h_to_v(h_)
                pre_h_, h_ = self.v_to_h(v_)
            return v, v_
        
        def get_weight(self):
            return self.W

# %%
# class SVM(nn.Module):
#     def __init__ (self, epoch, n_feat, n_out, batch=10, lr=0.999, c=0.01):
#         super(SVM, self).__init__()
#         self.epoch = epoch
#         self.n_feat = n_feat
#         self.n_out = n_out
#         self.batch = batch
#         self.lr = lr
#         self.c = c

#     def get_accuracy(self, model, data):
#             loader = torch.utils.data.DataLoader(data, batch_size=self.batch)
#             correct, total = 0, 0

#             for xs, ts in loader:
#                 zs = model(xs)
#                 pred = zs.max(1, keepdim=True)[1]
#                 correct += pred.eq(ts.view_as(pred)).sum().item()
#                 total += int(ts.shape[0])
#                 return correct / total        
    
#     def plot(self, xl, yl, xls, yls, label, title="Linear SVM Model Result"):
#         plt.title(title)
#         plt.plot(xl, yl, label)
#         plt.xlabel("Iterations")
#         plt.ylabel("Loss")
#         plt.show()
        
#         plt.title("Training Curve (batch_size={}, lr={})".format(self.batch, self.lr))
#         plt.plot(xls, yls, label="Train")
#         plt.xlabel("Iterations")
#         plt.ylabel("Accuracy")
#         plt.legend(loc='best')
#         plt.show()

#     def train(self, x):
#         iters, loss_ = [], []
#         iters_sub, train_acc = [], []
        
#         model = nn.Linear(self.n_feat, self.n_out)

#         criterion = nn.CrossEntropyLoss()
#         optimizer = optim.Adagrad(model.parameters(), lr=self.lr, weight_decay=self.c)
        
#         weight, bias = list(model.parameters())
#         y = torch.sigmoid(model(x))
#         print("weight shape: {}\tbias shape: {}".format(weight.shape, bias.shape))

#         svm_train_dataloader = DataLoader(x,
#                                           batch_size=self.batch,
#                                           shuffle=True)
        
#         n = 0
#         for epoch in range(self.epoch):
#             for xs, ts in iter(svm_train_dataloader):
#                 if len(ts) != self.batch:
#                     continue
#                 zs = model(xs)

#                 loss = criterion(zs, ts)
#                 loss.backward()
                
#                 optimizer.step()
#                 optimizer.zero_grad()

#                 iters.append(n)
#                 loss_.append(float(loss) / self.batch)
#                 train_acc.append(self.get_accuracy(model, x))
#                 n += 1
        
#         self.plot(iters, loss_, iters_sub, train_acc, label="Train")
#         torch.save(model, "svm_model.pth")
#         return model

class SVM(nn.Module):
    with torch.cuda.device(0):
        def __init__(self, lr, n_x):
            super(SVM, self).__init__()
            self.lr = lr
            self.fully = nn.Linear(n_x, 1).to(device=device)
        
        def forward(self, x):
            fwd = self.fully(x)
            return fwd

# %%
def eval_step(engine, batch):
    return batch

default_model = nn.Sequential(OrderedDict([
    ('base', nn.Linear(4, 2)),
    ('fc', nn.Linear(2, 1))
]))

default_evaluator = Engine(eval_step)

def get_acc(y_true, y_pred):
    metric = Accuracy()
    metric.attach(default_evaluator, "accuracy")
    state = default_evaluator.run([[y_pred, y_true]])
    return state.metrics["accuracy"]

# %%
print("[MODL] Model main code is starting....")

print("[INFO] Read train data, cross-vaildation data and test data from median filtering code")
db1_sig, db1_label, db2_sig, db2_label, db3_sig, db3_label = rs.return_list()

# %%
train_dataset = []
cross_dataset = []
test_dataset = []

le = sklearn.preprocessing.LabelEncoder()
db1_label_Y = le.fit_transform(rs.list_to_list(db1_label))
db3_label_Y = le.fit_transform(rs.list_to_list(db3_label))

# oneHot.fit(db1_label_Y)
# db1_label_Y = oneHot.transform(db1_label_Y)

for i in range(len(db1_sig)):
    train_dataset.append([db1_sig[i], db1_label[i]])

for i in range(len(db2_sig)):
    cross_dataset.append([db2_sig[i], db2_label[i]])

for i in range(len(db3_sig)):
    test_dataset.append([db3_sig[i], db3_label[i]])

train_dataloader = DataLoader(db1_sig,
                              batch_size=BATCH_SIZE,
                              num_workers=0, 
                              collate_fn=lambda x: x,
                              shuffle=True)

cross_dataloader = DataLoader(db2_sig,
                              batch_size=BATCH_SIZE,
                              num_workers=0,
                              collate_fn=lambda x: x,
                              shuffle=True)  
                            
test_dataloader = DataLoader(db3_sig,
                             batch_size=BATCH_SIZE,
                             num_workers=0, 
                             collate_fn=lambda x: x,
                             shuffle=True)

# %%
bbrbm_first = RBM(n_vis=VISIBLE_UNITS, n_hid=HIDDEN_UNITS[0], k=K_FOLD, batch=BATCH_SIZE).to(device=device)
bbrbm_second = RBM(n_vis=VISIBLE_UNITS, n_hid=HIDDEN_UNITS[1], k=K_FOLD, batch=BATCH_SIZE).to(device=device)
bbrbm_third = RBM(n_vis=VISIBLE_UNITS, n_hid=HIDDEN_UNITS[2], k=K_FOLD, batch=BATCH_SIZE).to(device=device)

gbrbm_first = RBM(n_vis=VISIBLE_UNITS, n_hid=HIDDEN_UNITS[3], k=K_FOLD, batch=BATCH_SIZE).to(device=device)
gbrbm_second = RBM(n_vis=VISIBLE_UNITS, n_hid=HIDDEN_UNITS[4], k=K_FOLD, batch=BATCH_SIZE).to(device=device)
gbrbm_third = RBM(n_vis=VISIBLE_UNITS, n_hid=HIDDEN_UNITS[5], k=K_FOLD, batch=BATCH_SIZE).to(device=device)

first_train_op = optim.Adagrad(bbrbm_first.parameters(), LEARNING_RATE)
second_train_op = optim.Adagrad(bbrbm_second.parameters(), LEARNING_RATE)
third_train_op = optim.Adagrad(bbrbm_third.parameters(), LEARNING_RATE)

gb_first_train_op = optim.Adagrad(gbrbm_first.parameters(), LEARNING_RATE)
gb_second_train_op = optim.Adagrad(gbrbm_second.parameters(), LEARNING_RATE)
gb_third_train_op = optim.Adagrad(gbrbm_third.parameters(), LEARNING_RATE)

omse_loss = list()
output_gb = list()
best_acc = float()
svm_best_acc = float()
mse_loss = nn.MSELoss()

# gaussian_std = torch.arange(1, 0, -0.00537, device=device)
gaussian_std = torch.arange(1, 0, -0.007, device=device)
print(gaussian_std.size())

svm_model = SVM(lr=LEARNING_RATE, n_x=143)
svm_optimizer = optim.Adagrad(svm_model.parameters(), lr=LEARNING_RATE)

# %%
'''BBRBM Train Part'''

loss_ = []
output_bb = []
model_path_str = str()

print("RBM START!")

for epoch in range(EPOCH):
    tmp_acc = float()
    run_acc = float()
    start = time.time()
    '''First bbrbm'''
    for i, (data) in enumerate(train_dataloader):
        data = torch.tensor(data, dtype=torch.float32)
        data = Variable(torch.tensor(data.uniform_(0, 1), dtype=torch.float32))

        sample_data = torch.bernoulli(data).view(-1, BATCH_SIZE).to(device=device)
        fs_data = sample_data
        
        # tensor binary
        fvog_first, v1 = bbrbm_first(sample_data)
        omse_loss = mse_loss(fvog_first, v1)
        
        first_train_op.zero_grad()
        first_train_op.step()
        omse_loss.backward()
    
    for _, (data) in enumerate(v1): 
        data = Variable(
                torch.tensor(data, dtype=torch.float32)
        ).uniform_(0, 1)

        sample_data = torch.bernoulli(data).view(-1, BATCH_SIZE).to(device=device)

        # tensor binary
        vog_second, v2 = bbrbm_second(sample_data)
        omse_loss = mse_loss(vog_second, v2)

        second_train_op.zero_grad()
        omse_loss.backward()
        second_train_op.step()
    
    for _, (data) in enumerate(v2):
        start = time.time()
        data = Variable(
                torch.tensor(data, dtype=torch.float32)
        ).uniform_(0, 1)

        sample_data = torch.bernoulli(data).view(-1, BATCH_SIZE)                                                                                                                                                                                                                                                                                                                                        .to(device=device)

        vog_third, v3 = bbrbm_third(sample_data)
        omse_loss = mse_loss(vog_third, v3)
        
        third_train_op.zero_grad()
        omse_loss.backward()
        third_train_op.step()

        run_acc += (sample_data == v3).sum().item()
    
    '''
        GBRBM GBRBM GBRBM GBRBM GBRBM GBRBM GBRBM 
    '''

    for i, (data) in enumerate(output_bb):
        data = Variable(
                torch.tensor(data, dtype=torch.float32)
        ).uniform_(0, 1)
        
        sample_data = torch.normal(mean=data, std=gaussian_std).to(device=device)

        # tensor binary
        vog_first, v1 = gbrbm_first(sample_data)
        omse_loss = mse_loss(vog_first, v1)

        gb_first_train_op.zero_grad()
        gb_first_train_op.step()
        omse_loss.backward()

    for _, (data) in enumerate(v1): 
        data = Variable(
                torch.tensor(data, dtype=torch.float32)
        ).uniform_(0, 1)

        sample_data = torch.normal(mean=data, std=gaussian_std).to(device=device)

        # tensor binary
        vog_second, v2 = gbrbm_second(sample_data)
        omse_loss = mse_loss(vog_second, v2)

        gb_second_train_op.zero_grad()
        omse_loss.backward()
        gb_second_train_op.step()

    for _, (data) in enumerate(v2):
        start = time.time()
        data = Variable(
                torch.tensor(data, dtype=torch.float32)
        ).uniform_(0, 1)

        sample_data = torch.normal(mean=data, std=gaussian_std).to(device=device)

        vog_third, v3_e = gbrbm_third(sample_data)
        omse_loss = mse_loss(vog_third, v3_e)
        
        gb_third_train_op.zero_grad()
        omse_loss.backward()
        gb_third_train_op.step()

    svm_X = torch.tensor(v3_e, dtype=torch.float32, device=device)
    svm_Y = torch.tensor(db3_label_Y, dtype=torch.float32, device=device)
    N = len(svm_Y)

    perm = torch.randperm(N, device=device)

    for i in range(0, N, BATCH_SIZE):
        correct = float()

        x = torch.tensor(svm_X.clone().detach(), device=device)
        y = torch.tensor(svm_Y.clone().detach(), device=device)

        # Forward
        output = svm_model(x)
        
        # Backward
        svm_optimizer.zero_grad()        
        svm_optimizer.step()

        predicted = torch.tensor(output.data >= 0, dtype=torch.float32)

        svm_acc = output.data >= predicted
        
    svm_best_acc = svm_acc
    svm_path = "./mat_svm_model/" + str(epoch) + "_Train_svm_model_acc__.pth"
    torch.save(svm_model.state_dict(), svm_path)

    acc_v = (vog_third >= 0).float()
    acc = get_acc(
        acc_v, v3_e
    ) * 100
    
    if acc > best_acc:
        best_acc = acc    
        
        path = "./say_cheese/raw_ahh_saveMode_through_" + str(epoch) + "_" + str(acc) + "GBRBM.pth"
        model_path_str = path
        torch.save(gbrbm_third.state_dict(), path)
    output_gb.append(v3_e)

    print("GB-DBN Training loss for {0}th epoch {1}\tEstimate time : {2}\tAcc : {3}\t\tBest Acc : {4}\tSVM Acc & Predicted: {5}, {6}".format(epoch + 1, omse_loss, time.time() - start, acc, best_acc, svm_acc, predicted))
    gc.collect()

# %%



