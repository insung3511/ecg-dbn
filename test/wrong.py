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
import ignite
import torch
import time
import os

print(datetime.datetime.now(), "model.py code start")

BATCH_SIZE = 10
EPOCH = 20
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
class SVM(nn.Module):
    def __init__ (self, epoch, n_feat, n_out, batch=10, lr=0.999, c=0.01):
        super(SVM, self).__init__()
        self.epoch = epoch
        self.n_feat = n_feat
        self.n_out = n_out
        self.batch = batch
        self.lr = lr
        self.c = c

    def get_accuracy(self, model, data):
            loader = torch.utils.data.DataLoader(data, batch_size=self.batch)
            correct, total = 0, 0

            for xs, ts in loader:
                zs = model(xs)
                pred = zs.max(1, keepdim=True)[1]
                correct += pred.eq(ts.view_as(pred)).sum().item()
                total += int(ts.shape[0])
                return correct / total        
    
    def plot(self, xl, yl, xls, yls, label, title="Linear SVM Model Result"):
        plt.title(title)
        plt.plot(xl, yl, label)
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.show()
        
        plt.title("Training Curve (batch_size={}, lr={})".format(self.batch, self.lr))
        plt.plot(xls, yls, label="Train")
        plt.xlabel("Iterations")
        plt.ylabel("Accuracy")
        plt.legend(loc='best')
        plt.show()

    def train(self, x):
        iters, loss_ = [], []
        iters_sub, train_acc = [], []
        
        model = nn.Linear(self.n_feat, self.n_out)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adagrad(model.parameters(), lr=self.lr, weight_decay=self.c)
        
        weight, bias = list(model.parameters())
        y = torch.sigmoid(model(x))
        print("weight shape: {}\tbias shape: {}".format(weight.shape, bias.shape))

        svm_train_dataloader = DataLoader(x,
                                          batch_size=self.batch,
                                          shuffle=True)
        
        n = 0
        for epoch in range(self.epoch):
            for xs, ts in iter(svm_train_dataloader):
                if len(ts) != self.batch:
                    continue
                zs = model(xs)

                loss = criterion(zs, ts)
                loss.backward()
                
                optimizer.step()
                optimizer.zero_grad()

                iters.append(n)
                loss_.append(float(loss) / self.batch)
                train_acc.append(self.get_accuracy(model, x))
                n += 1
        
        self.plot(iters, loss_, iters_sub, train_acc, label="Train")
        torch.save(model, "svm_model.pth")
        return model

# %%
print("[MODL] Model main code is starting....")

print("[INFO] Read train data, cross-vaildation data and test data from median filtering code")
db1_sig, db1_label, db2_sig, db2_label, db3_sig, db3_label = rs.return_list()

# %%
# label_map = {
#     0:  "N",
#     1:  "S",
#     2:  "V",
#     4:  "F",
#     5:  "A",
#     6:  "+",
# }

# %%
# class MapDataset(torch.utils.data.Dataset):
#     def __len__(self):
#         return len(db1_sig + db2_sig)
#     def __getitem__(self, idx):
#         return {
#             "input" : torch.tensor([db1_sig + db2_sig], dtype=torch.float32),
#             "label" : torch.tensor(label_map[db1_label + db2_label], dtype=torch.float32)
#         }

# %%
# map_data = MapDataset()

# point_sampler = torch.utils.data.RandomSampler(map_data)
# batch_sampler = torch.utils.data.BatchSampler(point_sampler, 3, False)

# dataloader = torch.utils.data.DataLoader(map_data, batch_sampler=batch_sampler)
# for data in dataloader:
#     print(data['input'])
#     print(data['label'])

# %%
train_dataset = []
test_dataset = []

for i in range(len(db1_sig + db2_sig)):
    train_dataset.append([rs.list_to_list(db1_sig + db2_sig)[i], rs.list_to_list(db1_label + db2_label)[i]])

for i in range(len(db2_sig + db3_sig)):
    test_dataset.append([rs.list_to_list(db1_sig + db3_sig)[i], rs.list_to_list(db2_label + db3_label)[i]])

train_dataloader = DataLoader(train_dataset,
                              batch_size=BATCH_SIZE,
                              num_workers=0, 
                              collate_fn=lambda x: x,
                              shuffle=True)
                              
test_dataloader = DataLoader(test_dataset,
                             batch_size=BATCH_SIZE,
                             num_workers=0, 
                             collate_fn=lambda x: x,
                             shuffle=True)

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

output_from_first = list()
output_from_second = list()
output_from_third = list()

omse_loss = list()
mse_loss = nn.MSELoss()
best_acc = float()

# %%
'''Train Part'''

loss_ = []
for epoch in range(EPOCH):
    run_acc = float()
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
        run_acc += (torch.bernoulli(data).view(-1, 10).to(device=device) == v1).sum().item()
 
    acc = (run_acc / v1.size()[0])
    if acc > best_acc:
        best_acc = acc

    path = "./saveMode_BBRBM1.pth"
    torch.save(rbm_first.state_dict(), path)

    output_from_first.append(v1.tolist())
    print("1ST BBrbm_first Training loss for {0} epoch {1}\tEstimate time : {2}\tAcc : {3}\tBest Acc : {4}" \
        .format(epoch + 1, 
                omse_loss, 
                time.time() - start,
                acc, 
                best_acc))

# %%
output_from_first = torch.tensor(output_from_first)
print(output_from_first.size())

for epoch in range(EPOCH):
    '''Secnd bbrbm'''
    start = time.time()
    run_acc = float()
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
        run_acc += (torch.bernoulli(data).view(-1, 10).to(device=device) == v2).sum().item()

    acc = (run_acc / v2.size()[0]) * 100 / 500
    if acc > best_acc:
        best_acc = acc

    path = "./saveMode_BBRBM_2.pth"
    torch.save(rbm_second.state_dict(), path)

    output_from_second.append(v2.tolist())
    print("2ST BBrbm_first Training loss for {0} epoch {1}\tEstimate time : {2}\tAcc : {3}\tBest Acc : {4}" \
        .format(epoch + 1, 
                omse_loss, 
                time.time() - start,
                acc, 
                best_acc))

# %%
output_from_second = torch.tensor(output_from_second)

for epoch in range(EPOCH):
    '''Third bbrbm'''
    run_acc = float()

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
        run_acc += (torch.bernoulli(data).view(-1, 10).to(device=device) == v3).sum().item()

    acc = (run_acc / v3.size()[0]) * 100 / 500
    if acc > best_acc:
        best_acc = acc

    path = "./saveMode_BBRBM_3.pth"
    torch.save(rbm_third.state_dict(), path)

    output_from_third.append(v3.tolist())
    print("3ST BBrbm_first Training loss for {0} epoch {1}\tEstimate time : {2}\tAcc : {3}\tBest Acc : {4}" \
        .format(epoch + 1, 
                omse_loss, 
                time.time() - start,
                acc, 
                best_acc))


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
    run_acc = float()

    for i, (data) in enumerate(output_from_third):
        data = Variable(
                torch.tensor(data, dtype=torch.float32)
        ).uniform_(0, 1)
        
        sample_data = torch.normal(mean=data, std=gaussian_std).view(-1, 10).to(device=device)

        # tensor binary
        vog_first, v1 = rbm_first(sample_data)
        omse_loss = mse_loss(vog_first, v1)

        first_train_op.zero_grad()
        first_train_op.step()
        omse_loss.backward()
        run_acc += (torch.bernoulli(data).view(-1, 10).to(device=device) == v1).sum().item()

    acc = (run_acc / v1.size()[0]) * 100 / 500
    if acc > best_acc:
        best_acc = acc
        
    path = "./saveMode_GBRBM_1.pth"
    torch.save(rbm_third.state_dict(), path)

    output_from_first.append(v1.tolist())
    print("1ST GBrbm_first Training loss for {0} epoch {1}\tEstimate time : {2}\tAcc : {3}\tBest Acc : {4}" \
        .format(epoch + 1, 
                omse_loss, 
                time.time() - start,
                acc, 
                best_acc))

# %%
output_from_first = torch.tensor(output_from_first)
output_from_third = list()
print(output_from_first.size())


for epoch in range(EPOCH):
    '''Secnd gbrbm'''
    run_acc = float()
    start = time.time()
    for _, (data) in enumerate(output_from_first): 
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
        run_acc += (torch.bernoulli(data).view(-1, 10).to(device=device) == v2).sum().item()

    acc = (run_acc / v2.size()[0]) * 100 / 500
    if acc > best_acc:
        best_acc = acc
    
    path = "./saveMode_GBRBM_2.pth"
    torch.save(rbm_third.state_dict(), path)

    output_from_second.append(v2.tolist())
    print("2ST GBrbm_first Training loss for {0} epoch {1}\tEstimate time : {2}\tAcc : {3}\tBest Acc : {4}" \
        .format(epoch + 1, 
                omse_loss, 
                time.time() - start,
                acc, 
                best_acc))

# %%
output_from_second = torch.tensor(output_from_second)

for epoch in range(EPOCH):
    '''Third gbrbm'''
    run_acc = float()
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
        run_acc += (torch.bernoulli(data).view(-1, 10).to(device=device) == v3).sum().item()

    acc = (run_acc / v3.size()[0]) * 100 / 500
    if acc > best_acc:
        best_acc = acc
        
    path = "./saveMode_GBRBM_3.pth"
    torch.save(rbm_third.state_dict(), path)

    output_from_third.append(v3.tolist())
    print("3ST GBrbm_first Training loss for {0} epoch {1}\tEstimate time : {2}\tAcc : {3}\tBest Acc : {4}" \
        .format(epoch + 1, 
                omse_loss, 
                time.time() - start,
                acc, 
                best_acc))


# %%
print("Last Accuracy : ", acc * 100 / 500, "%")
svm = SVM(EPOCH, len(output_from_third), 5, batch=BATCH_SIZE, lr=LEARNING_RATE)

svm.train(
    output_from_third
)

# %%
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
        run_acc += (torch.bernoulli(data).view(-1, 10).to(device=device) == v1).sum().item()
 
acc = (run_acc / v1.size()[0])
if acc > best_acc:
        best_acc = acc

print("loss : {0}\tEstimate time : {1}\tAcc : {2}\tBest Acc : {3}" \
    .format(omse_loss, time.time() - start, acc, best_acc))

'''GUIDE'''

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
        run_acc += (torch.bernoulli(data).view(-1, 10).to(device=device) == v2).sum().item()
 
acc = (run_acc / v2.size()[0])
if acc > best_acc:
        best_acc = acc

print("loss : {0}\tEstimate time : {1}\tAcc : {2}\tBest Acc : {3}" \
    .format(omse_loss, time.time() - start, acc, best_acc))
    
'''GUIDE'''

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
        run_acc += (torch.bernoulli(data).view(-1, 10).to(device=device) == v3).sum().item()
 
acc = (run_acc / v3.size()[0])
if acc > best_acc:
        best_acc = acc

print("loss : {0}\tEstimate time : {1}\tAcc : {2}\tBest Acc : {3}" \
    .format(omse_loss, time.time() - start, acc, best_acc))

# %%
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
        run_acc += (torch.bernoulli(data).view(-1, 10).to(device=device) == v1).sum().item()

print("loss : {0}\tEstimate time : {1}\tAcc : {2}\tBest Acc : {3}" \
    .format(omse_loss, time.time() - start, acc, best_acc))

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
        run_acc += (torch.bernoulli(data).view(-1, 10).to(device=device) == v2).sum().item()

print("loss : {0}\tEstimate time : {1}\tAcc : {2}\tBest Acc : {3}" \
    .format(omse_loss, time.time() - start, acc, best_acc))

'''GUIDE'''

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
        run_acc += (torch.bernoulli(data).view(-1, 10).to(device=device) == v1).sum().item()

print("loss : {0}\tEstimate time : {1}\tAcc : {2}\tBest Acc : {3}" \
    .format(omse_loss, time.time() - start, acc, best_acc))

# %%
svm_model_cp = torch.load("svm_model.pth")


# %%
svm_model = SVM(epoch=EPOCH, n_feat=v3.size()[0], n_out=5, batch=BATCH_SIZE, lr=LEARNING_RATE)
svm_optim = optim.Adagrad(svm_model.parameters(), lr=LEARNING_RATE)

svm_model.load_state_dict(svm_model_cp['model_state_dict'])
svm_optim.load_state_dict(svm_model_cp['optimizer_state_dict'])
epoch = svm_model_cp['epoch']
loss = svm_model_cp['loss']

svm_model.eval()


