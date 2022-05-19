# %% [markdown]
# # Init set up

# %%
from ignite.contrib.metrics.regression import *
from ignite.contrib.metrics import *
from ignite.handlers import *
from ignite.metrics import *
from ignite.engine import *
from ignite.utils import *

from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
import data.read_samples as rs
import torch.optim as optim
import torch.utils.data
import torch.nn as nn

from collections import OrderedDict
import datetime
import torch
import time
import os
import gc

print(datetime.datetime.now(), "model.py code start")

BATCH_SIZE = 10
EPOCH = 10
LEARNING_RATE = 0.2
ANNEALING_RATE = 0.999
VISIBLE_UNITS = [180, 200, 250]
HIDDEN_UNITS = [80, 100, 120]
K_FOLD = 1

# %%
# device = torch.device('cuda')
device = torch.device('cpu')
# print(torch.cuda.get_device_name(device))
# os.environ['KMP_DUPLICATE_LIB_OK']='True'

# %%
class RBM(nn.Module): 
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
                F.linear(v, w)
            ).to(device=device)

            sample_h = self.sample_from_p(p_h)
            return p_h, sample_h

        def h_to_v(self, h):
            w = self.W.t().clone()

            p_v = F.sigmoid(
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

# %% [markdown]
# # Pre-processing

# %%
print("[MODL] Model main code is starting....")

print("[INFO] Read train data, cross-vaildation data and test data from median filtering code")
db1_sig, db1_label, db2_sig, db2_label, db3_sig, db3_label = rs.return_list()

# %%
train_dataset = []
cross_dataset = []
test_dataset = []

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

# %% [markdown]
# ## Setting Models

# %%
rbm_first = RBM(n_vis=VISIBLE_UNITS[0], n_hid=HIDDEN_UNITS[0], k=K_FOLD, batch=BATCH_SIZE).to(device=device)
rbm_second = RBM(n_vis=VISIBLE_UNITS[1], n_hid=HIDDEN_UNITS[1], k=K_FOLD, batch=BATCH_SIZE).to(device=device)
rbm_third = RBM(n_vis=VISIBLE_UNITS[2], n_hid=HIDDEN_UNITS[2], k=K_FOLD, batch=BATCH_SIZE).to(device=device)

first_train_op = optim.Adagrad(rbm_first.parameters(), LEARNING_RATE)
second_train_op = optim.Adagrad(rbm_second.parameters(), LEARNING_RATE)
third_train_op = optim.Adagrad(rbm_third.parameters(), LEARNING_RATE)

gb_first_train_op = optim.Adagrad(rbm_first.parameters(), LEARNING_RATE)
gb_second_train_op = optim.Adagrad(rbm_second.parameters(), LEARNING_RATE)
gb_third_train_op = optim.Adagrad(rbm_third.parameters(), LEARNING_RATE)

omse_loss = list()
output_gb = list()
best_acc = float()
mse_loss = nn.MSELoss()

gaussian_std = torch.arange(1, 0, -0.1, device=device)

# %% [markdown]
# # GB-DBN Train Code

# %%
# '''BBRBM Train Part'''

# loss_ = []
# output_bb = []
# print("RBM START!")

# for epoch in range(EPOCH):
#     tmp_acc = float()
#     run_acc = float()
#     start = time.time()
#     '''First bbrbm'''
#     for i, (data) in enumerate(train_dataloader):
#         data = Variable(
#                 torch.tensor(data, dtype=torch.float32)
#         ).uniform_(0, 1)
        
#         sample_data = torch.bernoulli(data).view(-1, 10).to(device=device)

#         # tensor binary
#         vog_first, v1 = rbm_first(sample_data)
#         omse_loss = mse_loss(vog_first, v1)

#         first_train_op.zero_grad()
#         first_train_op.step()
#         omse_loss.backward()
#     print("First BBRBM Passed")

#     for _, (data) in enumerate(v1): 
#         data = Variable(
#                 torch.tensor(data, dtype=torch.float32)
#         ).uniform_(0, 1)

#         sample_data = torch.bernoulli(data).view(-1, 10).to(device=device)

#         # tensor binary
#         vog_second, v2 = rbm_second(sample_data)
#         omse_loss = mse_loss(vog_second, v2)

#         second_train_op.zero_grad()
#         omse_loss.backward()
#         second_train_op.step()
#     print("Second BBRBM Passed")
    
#     for _, (data) in enumerate(v2):
#         start = time.time()
#         data = Variable(
#                 torch.tensor(data, dtype=torch.float32)
#         ).uniform_(0, 1)

#         sample_data = torch.bernoulli(data).view(-1, 10).to(device=device)

#         vog_third, v3 = rbm_third(sample_data)
#         omse_loss = mse_loss(vog_third, v3)
        
#         third_train_op.zero_grad()
#         omse_loss.backward()
#         third_train_op.step()

#         run_acc += (sample_data == v3).sum().item()
    
#     print("Third BBRBM Passed")
#     acc = (run_acc / v3.size()[1]) * 100
#     if acc > best_acc:
#         best_acc = acc

#     path = "./saveMode_through_BBRBM.pth"
#     torch.save(rbm_second.state_dict(), path)
#     output_bb.append(v3)

#     '''
# GBRBM GBRBM GBRBM GBRBM GBRBM GBRBM GBRBM 
#     '''

#     for i, (data) in enumerate(output_bb):
#         data = Variable(
#                 torch.tensor(data, dtype=torch.float32)
#         ).uniform_(0, 1)
        
#         sample_data = torch.normal(mean=data, std=gaussian_std).view(-1, 10).to(device=device)

#         # tensor binary
#         vog_first, v1 = rbm_first(sample_data)
#         omse_loss = mse_loss(vog_first, v1)

#         first_train_op.zero_grad()
#         first_train_op.step()
#         omse_loss.backward()
#     print("First GBRBM Passed")

#     for _, (data) in enumerate(v1): 
#         data = Variable(
#                 torch.tensor(data, dtype=torch.float32)
#         ).uniform_(0, 1)

#         sample_data = torch.normal(mean=data, std=gaussian_std).view(-1, 10).to(device=device)

#         # tensor binary
#         vog_second, v2 = rbm_second(sample_data)
#         omse_loss = mse_loss(vog_second, v2)

#         second_train_op.zero_grad()
#         omse_loss.backward()
#         second_train_op.step()
#     print("Second GBRBM Passed")


#     for _, (data) in enumerate(v2):
#         start = time.time()
#         data = Variable(
#                 torch.tensor(data, dtype=torch.float32)
#         ).uniform_(0, 1)

#         sample_data = torch.bernoulli(data).view(-1, 10).to(device=device)

#         vog_third, v3_e = rbm_third(sample_data)
#         omse_loss = mse_loss(vog_third, v3_e)
        
#         third_train_op.zero_grad()
#         omse_loss.backward()
#         third_train_op.step()
#         run_acc += (torch.bernoulli(data).view(-1, 10).to(device=device) == v3_e).sum().item()
#     print("Third GBRBM Passed")

#     acc = get_acc(vog_third, v3_e) * 100
#     if acc > best_acc:
#         best_acc = acc    

#     path = "./saveMode_through_GBRBM.pth"
#     torch.save(rbm_third.state_dict(), path)
#     output_gb.append(v3_e)

#     print("GB-DBN Training loss for {0}th epoch {1}\tEstimate time : {2}\tAcc : {3}\tBest Acc : {4}\t\tIgnite Acc: {5}" \
#         .format(epoch + 1, omse_loss, time.time() - start, acc, best_acc, tmp_acc))



# %%
'''BBRBM Train Part'''

loss_ = []
print("RBM START!")

for epoch in range(EPOCH):
    tmp_acc = float()
    run_acc = float()
    start = time.time()
    '''First bbrbm'''
    temp_list_list = []
    for i, (data) in enumerate(train_dataloader):
        if data.size()[0] == 4:
            break
        data = Variable(torch.tensor(data, dtype=torch.float32))
        sample_data = torch.bernoulli(data).view(-1, BATCH_SIZE).to(device=device)

        # tensor binary
        fvog_first, v1 = rbm_first(sample_data)
        omse_loss = mse_loss(fvog_first, v1)
        
        first_train_op.zero_grad()
        first_train_op.step()
        omse_loss.backward()
    
        temp_list_list.append(v1.tolist())
        f_bb.append(temp_list_list)
    
    temp_list_list = []
    for _, (data) in enumerate(f_bb):
        data = Variable(
                torch.tensor(data, dtype=torch.float32)
        ).uniform_(0, 1)

        sample_data = torch.bernoulli(data).to(device=device)

        # tensor binary
        vog_second, v2 = rbm_second(sample_data)
        omse_loss = mse_loss(vog_second, v2)

        second_train_op.zero_grad()
        second_train_op.step()
        omse_loss.backward()

        temp_list_list.append(v2.tolist())
        s_bb.append(temp_list_list)

    temp_list_list = []
    for _, (data) in enumerate(s_bb):
        start = time.time()
        data = Variable(
                torch.tensor(data, dtype=torch.float32)
        ).uniform_(0, 1)

        sample_data = torch.bernoulli(data).view(-1, BATCH_SIZE).to(device=device)

        vog_third, v3 = rbm_third(sample_data)
        omse_loss = mse_loss(vog_third, v3)
        
        third_train_op.zero_grad()
        third_train_op.step()
        omse_loss.backward()
        
        temp_list_list.append(v3.tolist())
        t_bb.append(temp_list_list)

    '''
GBRBM GBRBM GBRBM GBRBM GBRBM GBRBM GBRBM 
    '''

    for i, (data) in enumerate(output_bb):
        data = Variable(
                torch.tensor(data, dtype=torch.float32)
        ).uniform_(0, 1)
        
        sample_data = torch.normal(mean=data, std=gaussian_std).view(-1, BATCH_SIZE).to(device=device)

        # tensor binary
        vog_first, v1 = rbm_first(sample_data)
        omse_loss = mse_loss(vog_first, v1)

        first_train_op.zero_grad()
        first_train_op.step()
        omse_loss.backward()

    for _, (data) in enumerate(v1): 
        data = Variable(
                torch.tensor(data, dtype=torch.float32)
        ).uniform_(0, 1)

        sample_data = torch.normal(mean=data, std=gaussian_std).view(-1, BATCH_SIZE).to(device=device)

        # tensor binary
        vog_second, v2 = rbm_second(sample_data)
        omse_loss = mse_loss(vog_second, v2)

        second_train_op.zero_grad()
        omse_loss.backward()
        second_train_op.step()

    for _, (data) in enumerate(v2):
        start = time.time()
        data = Variable(
                torch.tensor(data, dtype=torch.float32)
        ).uniform_(0, 1)

        sample_data = torch.normal(mean=data, std=gaussian_std).view(-1, BATCH_SIZE).to(device=device)

        vog_third, v3_e = rbm_third(sample_data)
        omse_loss = mse_loss(vog_third, v3_e)
        
        third_train_op.zero_grad()
        omse_loss.backward()
        third_train_op.step()

        output_gb.append(torch.flatten(v3_e).tolist())
        run_acc += (torch.bernoulli(data).view(-1, 10).to(device=device) == v3_e).sum().item()  

    ''' SVM Train '''    
    svm_X = torch.FloatTensor(output_gb).to(device=cpu)
    svm_Y = torch.FloatTensor(Y).to(device=cpu)
    XN = len(svm_X)
    N = len(svm_Y)

    # xperm = torch.randperm(XN).to(device=cpu)
    # yperm = torch.randperm(N).to(device=cpu)
    
    # for i in range(0, N, BATCH_SIZE):
    #     correct = 0.

    #     x = svm_X[xperm[i:i + BATCH_SIZE]]
    #     y = svm_Y[yperm[i:i + BATCH_SIZE]]

    #     x = torch.tensor(x.clone().detach())
    #     y = torch.tensor(y.clone().detach())

    #     # Forward
    #     output = svm_model(x)
        
    #     # Backward
    #     svm_optimizer.zero_grad()        
    #     svm_optimizer.step()

    #     predicted = output.data >= 0
        
    #     print("#####################################################")
    #     print(float(
    #         output.data == torch.tensor(predicted.view(-1), dtype=torch.float32)
    #     ))
    #     print("#####################################################")

    #     correct += float(
    #         predicted.view(-1) == torch.tensor(output.data, dtype=torch.float32)
    #     )

    #     print(correct)

    torch.save(svm_model, "Train_svm_model.pth")
    
    acc_v = (vog_third >= 0).float()
    acc = get_acc(
        acc_v, v3_e
    ) * 100
    
    if acc > best_acc:
        best_acc = acc    
        path = "./New_network_saveMode_through_"+ epoch +"GBDBN.pth"
        torch.save(rbm_third.state_dict(), path)

    print("GB-DBN Training loss for {0}th epoch {1}\tEstimate time : {2}\tAcc : {3}\tBest Acc : {4}\t\tIgnite Acc: {5}" \
        .format(epoch + 1, omse_loss, time.time() - start, acc, best_acc, tmp_acc))
    gc.collect()

# %%
print("Last Accuracy : ", acc, "%")

# %%
output_gb_cp = list()
cpu = torch.device('cpu')

for i in range(len(output_gb)):
    output_gb_cp.append(
        Variable(torch.flatten(
            torch.tensor(output_gb[i], dtype=torch.float64)
        ).to(device=cpu), requires_grad=True).detach().numpy()
    ) 

# %%
print(len(output_gb_cp))
print(len(output_gb_cp[-1]))

print(len(db3_label))
print(len(db3_label[-1]))


# %%
db3_label_chan = list()
v_model_acc = list()

for i in range(len(db3_label)):
    temp_list = []
    temp_bool = []
    for j in range(2577):
        try:
            temp_str = db3_label[i][j]
        except IndexError:
            temp_str = ""

        if temp_str == "V":
            temp_list.append(0)
            temp_bool.append(True)

        else:
            temp_list.append(1)
            temp_bool.append(False)

    db3_label_chan.append(temp_list)
    v_model_acc.append(temp_bool)

# %%
class SVM(nn.Module):
    def __init__(self, lr, n_x):
        super(SVM, self).__init__()
        self.lr = lr
        self.fully = nn.Linear(n_x, 1)
    
    def forward(self, x):
        fwd = self.fully(x)
        return fwd

X = torch.FloatTensor(db3_label_chan)
Y = torch.FloatTensor(db3_label_chan)
N = len(Y)

model = SVM(lr=LEARNING_RATE, n_x=2577)
optimizer = optim.Adagrad(model.parameters(), lr=LEARNING_RATE)

model.train()

correct = 0.
cnt_tot = 0

for epoch in range(EPOCH):
    perm = torch.randperm(N)
    for  i in range(0, N, BATCH_SIZE):
        x = X[perm[i:i + BATCH_SIZE]]
        y = Y[perm[i:i + BATCH_SIZE]]

        x = torch.tensor(x.clone().detach())
        y = torch.tensor(y.clone().detach())

        # Forward
        output = model(x)
        
        # Backward
        optimizer.zero_grad()        
        optimizer.step()

        # predicted = output.data >= 0
        # correct += float(
        #     predicted.view(-1) == torch.tensor(rs.list_to_list(db3_label_chan), dtype=torch.float32)
        # )

        cnt_tot += 1
    print("Epoch: {}\tLoss: {}\tTotal Cnt: {}".format(epoch, correct, cnt_tot))
    torch.save(model, "Train_svm_model.pth")

# %% [markdown]
# # Test Code

# %%
run_acc = float()
best_acc = float()
print("Test Code GB-DBN Start")
for i, (data) in enumerate(test_dataloader):
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

for i, (data) in enumerate(v1):
        data = Variable(
                torch.tensor(data, dtype=torch.float32)
        ).uniform_(0, 1)
        
        sample_data = torch.bernoulli(data).view(-1, 10).to(device=device)

        # tensor binary
        vog_second, v2 = rbm_first(sample_data)
        omse_loss = mse_loss(vog_second, v2)

        second_train_op.zero_grad()
        second_train_op.step()
        omse_loss.backward()

for i, (data) in enumerate(v2):
        data = Variable(
                torch.tensor(data, dtype=torch.float32)
        ).uniform_(0, 1)
        
        sample_data = torch.bernoulli(data).view(-1, 10).to(device=device)

        # tensor binary
        vog_second, v3 = rbm_first(sample_data)
        omse_loss = mse_loss(vog_second, v3)

        second_train_op.zero_grad()
        second_train_op.step()
        omse_loss.backward()
        run_acc += (sample_data == v3).sum().item()
 
for _, (data) in enumerate(v3): 
        data = Variable(
                torch.tensor(data, dtype=torch.float32)
        ).uniform_(0, 1)

        sample_data = torch.normal(mean=data, std=gaussian_std).view(-1, 10).to(device=device)

        # tensor binary
        vog_second, v1 = rbm_first(sample_data)
        omse_loss = mse_loss(vog_second, v1)

        second_train_op.zero_grad()
        omse_loss.backward()
        second_train_op.step()

for _, (data) in enumerate(v1): 
        data = Variable(
                torch.tensor(data, dtype=torch.float32)
        ).uniform_(0, 1)

        sample_data = torch.normal(mean=data, std=gaussian_std).view(-1, 10).to(device=device)

        # tensor binary
        vog_second, v2 = rbm_second(sample_data)
        omse_loss = mse_loss(vog_second, v2)

        second_train_op.zero_grad()
        omse_loss.backward()
        second_train_op.step()
        
for _, (data) in enumerate(v2): 
        data = Variable(
                torch.tensor(data, dtype=torch.float32)
        ).uniform_(0, 1)

        sample_data = torch.normal(mean=data, std=gaussian_std).view(-1, 10).to(device=device)

        # tensor binary
        vog_second, v3 = rbm_third(sample_data)
        omse_loss = mse_loss(vog_second, v3)

        second_train_op.zero_grad()
        omse_loss.backward()
        second_train_op.step()

print("GB-DBN Training loss: {0}\tEstimate time : {1}\tAcc : {2}" .format(omse_loss, time.time() - start, acc * 100))

# %%
svm_v = torch.tensor(v3.clone().detach(), device=torch.device('cpu'))
print(svm_v)

# %%
X = torch.FloatTensor(svm_v)
Y = torch.FloatTensor(svm_v)
N = len(Y)

model = SVM(lr=LEARNING_RATE, n_x=10)
optimizer = optim.Adagrad(model.parameters(), lr=LEARNING_RATE)

correct = 0.
cnt_tot = 0

x = torch.tensor(X)
y = torch.tensor(Y)

# Forward
output = model(x)

# Backward
optimizer.zero_grad()
optimizer.step()

# %%
print(output)


