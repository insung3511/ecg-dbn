# %%
from sklearn.model_selection import KFold, train_test_split
import torch.distributions.distribution as D
import data.medain_filtering_class as mf
from torch.utils.data import DataLoader
from torch.autograd import Variable
from new_RBM import new_RBM as RBM
import matplotlib.pyplot as plt
import data.read_samples as rs
import torch.optim as optim
import torch.nn as nn
import numpy as np
import datetime
import torch

print(datetime.datetime.now(), "model.py code start")

BATCH_SIZE = 10
EPOCH = 90
LEARNING_RATE = 0.2
ANNEALING_RATE = 0.999
VISIBLE_UNITS = [180, 200, 250]
HIDDEN_UNITS = [80, 100, 120]
K_FOLD = 1

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
rbm_first = RBM(n_vis=VISIBLE_UNITS[0], n_hid=HIDDEN_UNITS[0], k=K_FOLD, batch=BATCH_SIZE)
rbm_second = RBM(n_vis=VISIBLE_UNITS[1], n_hid=HIDDEN_UNITS[1], k=K_FOLD, batch=BATCH_SIZE)
rbm_third = RBM(n_vis=VISIBLE_UNITS[2], n_hid=HIDDEN_UNITS[2], k=K_FOLD, batch=BATCH_SIZE)


first_train_op = optim.SGD(rbm_first.parameters(), 0.1)
second_train_op = optim.SGD(rbm_second.parameters(), 0.1)
third_train_op = optim.SGD(rbm_third.parameters(), 0.1)

gb_first_train_op = optim.SGD(rbm_first.parameters(), 0.1)
gb_second_train_op = optim.SGD(rbm_second.parameters(), 0.1)
gb_third_train_op = optim.SGD(rbm_third.parameters(), 0.1)

output_from_first = list()
output_from_second = list()
output_from_third = list()

omse_loss = list()
mse_loss = nn.MSELoss()

# %%
'''Train Part'''

loss_ = []
for epoch in range(EPOCH):
    '''First bbrbm'''
    for i, (data) in enumerate(train_dataloader):
        data = Variable(
                torch.tensor(data, dtype=torch.float32)
        ).uniform_(0, 1)
        
        sample_data = torch.bernoulli(data).view(-1, 10)

        # tensor binary
        vog_first, v1, mt = rbm_first(sample_data)
        omse_loss = mse_loss(vog_first, v1)

        first_train_op.zero_grad()
        first_train_op.step()
        omse_loss.backward()
    output_from_first.append(v1.tolist())
    print("1ST BBrbm_first Training loss for {0} epoch {1}\tEstimate time : {2}".format(epoch, omse_loss, mt))


# %%
output_from_first = torch.tensor(output_from_first)
print(output_from_first.size())

for epoch in range(EPOCH):
    '''Secnd bbrbm'''
    for _, (data) in enumerate(output_from_first): 
        data = Variable(
                torch.tensor(data, dtype=torch.float32)
        ).uniform_(0, 1)

        sample_data = torch.bernoulli(data).view(-1, 10)

        # tensor binary
        vog_second, v2, mt = rbm_second(sample_data)
        omse_loss = mse_loss(vog_second, v2)

        second_train_op.zero_grad()
        omse_loss.backward()
        second_train_op.step()

    output_from_second.append(v2.tolist())
    print("2ST BBrbm_first Training loss for {0} epoch {1}\tEstimate time : {2}".format(epoch, omse_loss, mt))


# %%
output_from_second = torch.tensor(output_from_second)
for epoch in range(EPOCH):
    '''Third bbrbm'''
    for _, (data) in enumerate(output_from_second):
        data = Variable(
                torch.tensor(data, dtype=torch.float32)
        ).uniform_(0, 1)

        sample_data = torch.bernoulli(data).view(-1, 10)

        vog_third, v3, mt = rbm_third(sample_data)
        omse_loss = mse_loss(vog_third, v3)
        
        third_train_op.zero_grad()
        omse_loss.backward()
        third_train_op.step()

    output_from_third.append(v3)
    print("3ST BBrbm_first Training loss for {0} epoch {1}\tEstimate time : {2}".format(epoch, omse_loss, mt))


# %%
rbm_first = RBM(n_vis=VISIBLE_UNITS[0], n_hid=HIDDEN_UNITS[0], k=K_FOLD, batch=BATCH_SIZE)
rbm_second = RBM(n_vis=VISIBLE_UNITS[1], n_hid=HIDDEN_UNITS[1], k=K_FOLD, batch=BATCH_SIZE)
rbm_third = RBM(n_vis=VISIBLE_UNITS[2], n_hid=HIDDEN_UNITS[2], k=K_FOLD, batch=BATCH_SIZE)

first_train_op = optim.SGD(rbm_first.parameters(), 0.1)
second_train_op = optim.SGD(rbm_second.parameters(), 0.1)
third_train_op = optim.SGD(rbm_third.parameters(), 0.1)

gb_first_train_op = optim.SGD(rbm_first.parameters(), 0.1)
gb_second_train_op = optim.SGD(rbm_second.parameters(), 0.1)
gb_third_train_op = optim.SGD(rbm_third.parameters(), 0.1)

output_from_first = list()
output_from_second = list()
output_from_third = list()

omse_loss = list()
mse_loss = nn.MSELoss()
gaussian_std = torch.arange(1, 0, -0.1)

# %%
loss_ = []
for epoch in range(EPOCH):
    '''First bbrbm'''
    for i, (data) in enumerate(train_dataloader):
        data = Variable(
                torch.tensor(data, dtype=torch.float32)
        ).uniform_(0, 1)
        
        sample_data = torch.normal(mean=data, std=gaussian_std).view(-1, 10)

        # tensor binary
        vog_first, v1, mt = rbm_first(sample_data)
        omse_loss = mse_loss(vog_first, v1)

        first_train_op.zero_grad()
        first_train_op.step()
        omse_loss.backward()
    output_from_first.append(v1.tolist())
    print("1ST BBrbm_first Training loss for {0} epoch {1}\tEstimate time : {2}".format(epoch, omse_loss, mt))



