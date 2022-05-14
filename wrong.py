# %%
import torch.distributions.distribution as D
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import data.read_samples as rs
import torch.optim as optim
import torch.utils.data
import torch.nn as nn
import numpy as np
import datetime
import torch
import time
import os

print(datetime.datetime.now(), "model.py code start")

BATCH_SIZE = 10
EPOCH = 90
LEARNING_RATE = 0.2
ANNEALING_RATE = 0.999
VISIBLE_UNITS = [180, 200, 250]
HIDDEN_UNITS = [80, 100, 120]
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
            self.v_bias = nn.Parameter(torch.zeros(batch, device=device))
            self.h_bias = nn.Parameter(torch.zeros(batch, device=device))
        
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
                F.linear(v, w)
            ).to(device=device)

            sample_h = self.sample_from_p(p_h)
            return p_h, sample_h

        def h_to_v(self, h):
            w = self.W.t().clone()

            p_v = F.sigmoid(
                F.linear(h, w).to(device=device)
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
print("[MODL] Model main code is starting....")

print("[INFO] Read train data, cross-vaildation data and test data from median filtering code")
db1_sig, db1_label, db2_sig, db2_label, db3_sig, db3_label = rs.return_list()

# %%
train_dataloader = DataLoader(db1_sig + db2_sig,
                              batch_size=BATCH_SIZE,
                              num_workers=0, 
                              collate_fn=lambda x: x,
                              shuffle=True)
                              
test_dataloader = DataLoader(db2_sig + db3_sig,
                             batch_size=BATCH_SIZE,
                             num_workers=0, 
                             collate_fn=lambda x: x,
                             shuffle=True)

# %%
rbm_first = RBM(n_vis=VISIBLE_UNITS[0], n_hid=HIDDEN_UNITS[0], k=K_FOLD, batch=BATCH_SIZE).to(device=device)
rbm_second = RBM(n_vis=VISIBLE_UNITS[1], n_hid=HIDDEN_UNITS[1], k=K_FOLD, batch=BATCH_SIZE).to(device=device)
rbm_third = RBM(n_vis=VISIBLE_UNITS[2], n_hid=HIDDEN_UNITS[2], k=K_FOLD, batch=BATCH_SIZE).to(device=device)


first_train_op = optim.Adagrad(rbm_first.parameters(), 0.1)
second_train_op = optim.Adagrad(rbm_second.parameters(), 0.1)
third_train_op = optim.Adagrad(rbm_third.parameters(), 0.1)

gb_first_train_op = optim.Adagrad(rbm_first.parameters(), 0.1)
gb_second_train_op = optim.Adagrad(rbm_second.parameters(), 0.1)
gb_third_train_op = optim.Adagrad(rbm_third.parameters(), 0.1)

output_from_first = list()
output_from_second = list()
output_from_third = list()

omse_loss = list()
mse_loss = nn.MSELoss()

# %%
'''Train Part'''

loss_ = []
for epoch in range(EPOCH):
    start = time.time()
    '''First bbrbm'''
    for i, (data) in enumerate(train_dataloader):
        data = Variable(
                torch.tensor(data, dtype=torch.float32)
        ).uniform_(0, 1)
        
        sample_data = torch.bernoulli(data).view(-1, 10).to(device=device)

        # tensor binary
        vog_first, v1 = rbm_first(sample_data)
        omse_loss = mse_loss(vog_first, v1)

        first_train_op.zero_grad()
        first_train_op.step()
        omse_loss.backward()
    output_from_first.append(v1.tolist())
    print("1ST BBrbm_first Training loss for {0} epoch {1}\tEstimate time : {2}".format(epoch, omse_loss, time.time() - start))


# %%
output_from_first = torch.tensor(output_from_first)
print(output_from_first.size())

for epoch in range(EPOCH):
    '''Secnd bbrbm'''
    start = time.time()
    for _, (data) in enumerate(output_from_first): 
        data = Variable(
                torch.tensor(data, dtype=torch.float32)
        ).uniform_(0, 1)

        sample_data = torch.bernoulli(data).view(-1, 10).to(device=device)

        # tensor binary
        vog_second, v2 = rbm_second(sample_data)
        omse_loss = mse_loss(vog_second, v2)

        second_train_op.zero_grad()
        omse_loss.backward()
        second_train_op.step()

    end = time.time()
    output_from_second.append(v2.tolist())
    print("2ST BBrbm_first Training loss for {0} epoch {1}\tEstimate time : {2}".format(epoch, omse_loss, end - start))


# %%
output_from_second = torch.tensor(output_from_second)
for epoch in range(EPOCH):
    '''Third bbrbm'''
    for _, (data) in enumerate(output_from_second):
        start = time.time()
        data = Variable(
                torch.tensor(data, dtype=torch.float32)
        ).uniform_(0, 1)

        sample_data = torch.bernoulli(data).view(-1, 10).to(device=device)

        vog_third, v3 = rbm_third(sample_data)
        omse_loss = mse_loss(vog_third, v3)
        
        third_train_op.zero_grad()
        omse_loss.backward()
        third_train_op.step()
    
    end = time.time()
    output_from_third.append(v3)
    print("3ST BBrbm_first Training loss for {0} epoch {1}\tEstimate time : {2}".format(epoch, omse_loss, end - start))


# %%
# rbm_first = RBM(n_vis=VISIBLE_UNITS[0], n_hid=HIDDEN_UNITS[0], k=K_FOLD, batch=BATCH_SIZE).to(device=device)
# rbm_second = RBM(n_vis=VISIBLE_UNITS[1], n_hid=HIDDEN_UNITS[1], k=K_FOLD, batch=BATCH_SIZE).to(device=device)
# rbm_third = RBM(n_vis=VISIBLE_UNITS[2], n_hid=HIDDEN_UNITS[2], k=K_FOLD, batch=BATCH_SIZE).to(device=device)

output_from_first = list()
output_from_second = list()

omse_loss = list()
mse_loss = nn.MSELoss()
gaussian_std = torch.arange(1, 0, -0.1)

# %%
''' ** ISSUE PART ** '''

loss_ = []
for epoch in range(EPOCH):
    '''First gbrbm'''
    for i, (data) in enumerate(output_from_third):
        data = Variable(
                torch.tensor(data, dtype=torch.float32)
        ).uniform_(0, 1)
        
        sample_data = torch.normal(mean=data, std=gaussian_std).view(-1, 10)

        # tensor binary
        vog_first, v1 = rbm_first(sample_data)
        omse_loss = mse_loss(vog_first, v1)

        first_train_op.zero_grad()
        first_train_op.step()
        omse_loss.backward()
    output_from_first.append(v1.tolist())
    print("1ST GBrbm_first Training loss for {0} epoch {1}\tEstimate time : {2}".format(epoch, omse_loss))


# %%
output_from_first = torch.tensor(output_from_first)
output_from_third = list()
print(output_from_first.size())

for epoch in range(EPOCH):
    '''Secnd gbrbm'''
    start = time.time()
    for _, (data) in enumerate(output_from_first): 
        data = Variable(
                torch.tensor(data, dtype=torch.float32)
        ).uniform_(0, 1)

        sample_data = torch.normal(mean=data, std=gaussian_std).view(-1, 10)

        # tensor binary
        vog_second, v2 = rbm_second(sample_data)
        omse_loss = mse_loss(vog_second, v2)

        second_train_op.zero_grad()
        omse_loss.backward()
        second_train_op.step()

    end = time.time()
    output_from_second.append(v2.tolist())
    print("2ST BBrbm_first Training loss for {0} epoch {1}\tEstimate time : {2}".format(epoch, omse_loss, end - start))

# %%
output_from_second = torch.tensor(output_from_second)
for epoch in range(EPOCH):
    '''Third gbrbm'''
    for _, (data) in enumerate(output_from_second):
        start = time.time()
        data = Variable(
                torch.tensor(data, dtype=torch.float32)
        ).uniform_(0, 1)

        sample_data = torch.bernoulli(data).view(-1, 10)

        vog_third, v3 = rbm_third(sample_data)
        omse_loss = mse_loss(vog_third, v3)
        
        third_train_op.zero_grad()
        omse_loss.backward()
        third_train_op.step()
    
    end = time.time()
    output_from_third.append(v3)
    print("3ST BBrbm_first Training loss for {0} epoch {1}\tEstimate time : {2}".format(epoch, omse_loss, end - start))


# %%



