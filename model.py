from sklearn.model_selection import KFold, train_test_split
import torch.distributions.distribution as D
import data.medain_filtering_class as mf
from torch.utils.data import DataLoader
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.optim as optim
from RBM import RBM
import numpy as np
import datetime
import torch

print(datetime.datetime.now(), "model.py code start")

BATCH_SIZE = 10
EPOCH = 100
LEARNING_RATE = 0.2
ANNEALING_RATE = 0.999
VISIBLE_UNITS = [180, 200, 250]
HIDDEN_UNITS = [80, 100, 120]
K_FOLD = 1

def show_adn_save(result, file_name=None):
    nprst = np.array(result)
    f = ".txt" % file_name
    print(nprst)

print("[MODL] Model main code is starting....")

print("[INFO] Read train data, cross-vaildation data and test data from median filtering code")
dataset_db1, dataset_db2, dataset_db3 = mf.ecg_filtering(True)

train_dataset = list(mf.list_to_list(dataset_db1)) * 4
cross_dataset = list(mf.list_to_list(dataset_db2)) * 4
test_dataset = list(mf.list_to_list(dataset_db3))  * 4

X_train, X_test, y_train, y_test = train_test_split(
    (train_dataset + cross_dataset), 
    (test_dataset + cross_dataset),
    test_size=0.33,
    shuffle=True
)

train_dataloader = DataLoader(X_train + y_train,
                              batch_size=BATCH_SIZE,
                              shuffle=True)

test_dataloader = DataLoader(X_test + y_test,
                             batch_size=BATCH_SIZE,
                             shuffle=True)

print("X_train length : ", len(X_train))
print("X_test  length : ", len(X_test))
print("y_train length : ", len(y_train))
print("y_test  length : ", len(y_test))

train_data = torch.FloatTensor(X_train)
test_data = torch.FloatTensor(X_test)

print("[INFO] Model object added")

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

'''Train Part'''

loss_ = []
for epoch in range(EPOCH):
    '''First bbrbm'''
    for _, (data) in enumerate(train_dataloader):
        try:
            # tnesor float
            data = torch.tensor(Variable(data.view(-1, BATCH_SIZE).uniform_(0, 1)), dtype=torch.float32)
        except RuntimeError:
            continue

        sample_data = torch.bernoulli(data)
        sample_data = torch.flatten(sample_data.clone())

        # tensor binary
        vog_first, v1, mt = rbm_first(sample_data)
        
        loss_first = rbm_first.free_energy(vog_first) - rbm_first.free_energy(v1)
        loss_.append(loss_first.data)
        
        first_train_op.zero_grad()
        loss_first.backward()
        first_train_op.step()
    
    output_from_first.append(v1.tolist())
    print("1ST BBrbm_first Training loss for {0} epoch {1}\tEstimate time : {2}".format(epoch, np.mean(loss_), mt))

output_from_first = torch.tensor(output_from_first)
for epoch in range(EPOCH):
    '''Secnd bbrbm'''
    for _, (data) in enumerate(output_from_first):
        try:
            data = torch.tensor(Variable(data.view(-1, BATCH_SIZE).uniform_(0, 1)), dtype=torch.float32)
        except RuntimeError:
            continue

        sample_data = torch.bernoulli(data)
        sample_data = torch.flatten(sample_data.clone())

        vog_second, v2, mt= rbm_second(sample_data)
        
        loss_second = rbm_second.free_energy(vog_second) - rbm_second.free_energy(v2)
        loss_.append(loss_second.data)
        
        second_train_op.zero_grad()
        loss_second.backward()
        second_train_op.step()

    output_from_second.append(v2.tolist())
    print("2ST BBrbm_first Training loss for {0} epoch {1}\tEstimate time : ".format(epoch, np.mean(loss_), mt))

output_from_second = torch.tensor(output_from_second)
for epoch in range(EPOCH):
    '''Third bbrbm'''
    for _, (data) in enumerate(output_from_second):
        try:
            data = torch.tensor(Variable(data.view(-1, BATCH_SIZE).uniform_(0, 1)), dtype=torch.float32)
        except RuntimeError:
            continue

        sample_data = torch.bernoulli(data)
        sample_data = torch.flatten(sample_data.clone())

        vog_third, v3, mt = rbm_third(sample_data)
        
        loss_third = rbm_third.free_energy(vog_third) - rbm_third.free_energy(v3)
        loss_.append(loss_third.data)
        
        third_train_op.zero_grad()
        loss_third.backward()
        third_train_op.step()

    output_from_third.append(v3.tolist())
    print("3ST BBrbm_first Training loss for {0} epoch {1}\tEstimate time : ".format(epoch, np.mean(loss_), mt))
    
print("BBRBM is done.")
print("GBRBM is start")

output_from_first = list()
output_from_second = list()
output_from_third = torch.tensor(output_from_third)

rbm_first = RBM(n_vis=VISIBLE_UNITS[0], n_hid=HIDDEN_UNITS[0], k=K_FOLD, batch=BATCH_SIZE)
rbm_second = RBM(n_vis=VISIBLE_UNITS[1], n_hid=HIDDEN_UNITS[1], k=K_FOLD, batch=BATCH_SIZE)
rbm_third = RBM(n_vis=VISIBLE_UNITS[2], n_hid=HIDDEN_UNITS[2], k=K_FOLD, batch=BATCH_SIZE)

# print(output_from_third.size(), output_from_third.dim(), "\n", output_from_third)
gaussian_std = torch.arange(1, 0, -0.1)

for epoch in range(EPOCH):
    '''First gbrbm'''
    for _, (data) in enumerate(output_from_third):
        try:
            data = torch.tensor(Variable(data.view(-1, BATCH_SIZE).uniform_(0, 1)), dtype=torch.float32)
        except RuntimeError:
            continue
        
        # CHANGED to GAUSSIAN
        sample_data = torch.normal(mean=data, std=gaussian_std)
        sample_data = torch.flatten(sample_data.clone())

        gb_vog_first, gb_v1, mt = rbm_first(sample_data)
        
        gb_loss_first = rbm_first.free_energy(gb_vog_first) - rbm_first.free_energy(gb_v1)
        loss_.append(gb_loss_first.data)
        
        gb_first_train_op.zero_grad()
        gb_loss_first.backward()
        gb_first_train_op.step()

    output_from_first.append(gb_v1.tolist())
    print("1ST GBrbm_first Training loss for {0} epoch {1}\tEstimate time : {2}".format(epoch, np.mean(loss_), mt))

output_from_first = torch.tensor(output_from_first)
for epoch in range(EPOCH):
    '''Second gbrbm'''
    for _, (data) in enumerate(output_from_first):
        try:
            data = torch.tensor(Variable(data.view(-1, BATCH_SIZE).uniform_(0, 1)), dtype=torch.float32)
        except RuntimeError:
            continue
        
        # CHANGED to GAUSSIAN
        sample_data = torch.normal(mean=data, std=gaussian_std)
        sample_data = torch.flatten(sample_data.clone())

        gb_vog_second, gb_v2, mt = rbm_second(sample_data)
        
        gb_loss_second = rbm_second.free_energy(gb_vog_second) - rbm_second.free_energy(gb_v2)
        loss_.append(gb_loss_second.data)
        
        gb_second_train_op.zero_grad()
        gb_loss_second.backward()
        gb_second_train_op.step()

    output_from_second.append(gb_v2.tolist())
    print("2ST GBrbm_first Training loss for {0} epoch {1}\tEstimate time : {2}".format(epoch, np.mean(loss_), mt))

output_from_third = list(output_from_third)
output_from_second = torch.tensor(output_from_second)
for epoch in range(EPOCH):
    '''Third gbrbm'''
    for _, (data) in enumerate(output_from_second):
        try:
            data = torch.tensor(Variable(data.view(-1, BATCH_SIZE).uniform_(0, 1)), dtype=torch.float32)
        except RuntimeError:
            continue
        
        # CHANGED to GAUSSIAN
        sample_data = torch.normal(mean=data, std=gaussian_std)
        sample_data = torch.flatten(sample_data.clone())

        gb_vog_third, gb_v3, mt = rbm_third(sample_data)
        
        gb_loss_third = rbm_third.free_energy(gb_vog_third) - rbm_second.free_energy(gb_v3)
        loss_.append(gb_loss_third.data)
        
        gb_third_train_op.zero_grad()
        gb_loss_third.backward()
        gb_third_train_op.step()

    output_from_third.append(gb_v3.tolist())
    print("3ST GBrbm_first Training loss for {0} epoch {1}\tEstimate time : {2}".format(epoch, np.mean(loss_), mt))

'''Test code'''
rbm_first = RBM(n_vis=VISIBLE_UNITS[0], n_hid=HIDDEN_UNITS[0], k=K_FOLD, batch=BATCH_SIZE)
rbm_second = RBM(n_vis=VISIBLE_UNITS[1], n_hid=HIDDEN_UNITS[1], k=K_FOLD, batch=BATCH_SIZE)
rbm_third = RBM(n_vis=VISIBLE_UNITS[2], n_hid=HIDDEN_UNITS[2], k=K_FOLD, batch=BATCH_SIZE)

output_from_first = list()
output_from_second = list()
output_from_third = torch.tensor(output_from_third)

test_loss = 0
epoch_cnt = 0
for _, test_data in enumerate(test_dataloader):
    try:
        test_data = torch.tensor(Variable(data.view(-1, BATCH_SIZE).uniform_(0, 1)), dtype=torch.float32)
    except RuntimeError:
        pass
    
    data = torch.bernoulli(test_data)
    
    v, vt1, _ = rbm_first(data)
    test_loss = rbm_first.free_energy(v) - rbm_first.free_energy(vt1)    

