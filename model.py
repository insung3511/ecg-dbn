from sklearn.model_selection import KFold, train_test_split
import torch.distributions.distribution as D
import data.medain_filtering_class as mf
from torch.utils.data import DataLoader
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.optim as optim
from RBM import RBM
import numpy as np
import random
import torch


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
second_train_op = optim.SGD(rbm_first.parameters(), 0.1)
third_train_op = optim.SGD(rbm_first.parameters(), 0.1)

loss_ = []
output_from_first = list()
output_from_second = list()

for epoch in range(EPOCH):
    '''Secnd bbrbm'''
    for _, (data) in enumerate(train_dataloader):
        try:
            # tnesor float
            data = torch.tensor(Variable(data.view(-1, BATCH_SIZE).uniform_(0, 1)), dtype=torch.float32)
        except RuntimeError:
            continue

        sample_data = torch.bernoulli(data)
        sample_data = torch.flatten(sample_data.clone())

        # tensor binary
        vog_first, v1 = rbm_first(sample_data)
        
        loss_first = rbm_first.free_energy(vog_first) - rbm_first.free_energy(v1)
        loss_.append(loss_first.data)
        
        first_train_op.zero_grad()
        loss_first.backward()
        first_train_op.step()
    
    output_from_first.append(v1.tolist())
    print(v1)
    print("1ST BBrbm_first Training loss for {0} epoch {1}".format(epoch, np.mean(loss_)))

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

        vog_second, v2 = rbm_first(sample_data)
        
        loss_second = rbm_second.free_energy(vog_second) - rbm_second.free_energy(v2)
        loss_.append(loss_second.data)
        
        second_train_op.zero_grad()
        loss_second.backward()
        second_train_op.step()

    output_from_second.append(v2.tolist())
    print(v2)
    print("2ST BBrbm_first Training loss for {0} epoch {1}".format(epoch, np.mean(loss_)))
    
    # '''Third bbrbm'''
    # for _, (data) in enumerate(train_dataloader):
    #     try:
    #         data = torch.tensor(Variable(data.view(-1, BATCH_SIZE).uniform_(0, 1)), dtype=torch.float32)
    #     except RuntimeError:
    #         continue

    #     sample_data = torch.bernoulli(data)
    #     sample_data = torch.flatten(sample_data.clone())

    #     v, v1 = rbm_third(sample_data)
        
    #     loss = rbm_first.free_energy(v) - rbm_first.free_energy(v1)
    #     loss_.append(loss.data + 10)
        
    #     second_train_op.zero_grad()
    #     loss.backward()
    #     second_train_op.step()

    # '''                                                                 Mark point'''
    # print("1ST BBrbm_first Training loss for {0} epoch {1}".format(epoch, np.mean(loss_)))

# '''2TH BB-rbm_first'''

# train_dataloader = DataLoader(loss_,
#                               batch_size=BATCH_SIZE,
#                               shuffle=True)
# for epoch in range(HIDDEN_UNITS[1]):
#     for _, (data) in enumerate(loss_):
#         try:
#             data = torch.tensor(Variable(data.view(-1, BATCH_SIZE).uniform_(0, 1)), dtype=torch.float32)
#         except RuntimeError:
#             continue

#         sample_data = torch.bernoulli(data)
#         sample_data = torch.flatten(sample_data.clone())

#         v, v1 = rbm_first(sample_data)
        
#         loss = rbm_first.free_energy(v) - rbm_first.free_energy(v1)
#         loss_.append(loss.data + 10)
        
#         train_op.zero_grad()
#         loss.backward()
#         train_op.step()

#     '''                                                                 Mark point'''
#     print("2ST BBrbm_first Training loss for {0} epoch {1}".format(epoch, np.mean(loss_)))

# '''3TH BB-rbm_first'''

# train_dataloader = DataLoader(loss_,
#                               batch_size=BATCH_SIZE,
#                               shuffle=True)

# for epoch in range(HIDDEN_UNITS[2]):
#     for _, (data) in enumerate(loss_):
#         try:
#             data = torch.tensor(Variable(data.view(-1, BATCH_SIZE).uniform_(0, 1)), dtype=torch.float32)
#         except RuntimeError:
#             continue

#         sample_data = torch.bernoulli(data)
#         sample_data = torch.flatten(sample_data.clone())

#         v, v1 = rbm_first(sample_data)
        
#         loss = rbm_first.free_energy(v) - rbm_first.free_energy(v1)
#         loss_.append(loss.data + 10)
        
#         train_op.zero_grad()
#         loss.backward()
#         train_op.step()

#     '''                                                                 Mark point'''
#     print("3ST BBrbm_first Training loss for {0} epoch {1}".format(epoch, np.mean(loss_)))

# show_adn_save(v.data)

# plt.plot(loss_)
# plt.show()



''' GBrbm_first START '''
# '''1TH GB-rbm_first'''
# for epoch in range(HIDDEN_UNITS[0]):
#     for _, (data) in enumerate(test_dataloader):
#         try:
#             data = torch.tensor(Variable(data.view(-1, BATCH_SIZE).uniform_(0, 1)), dtype=torch.float32)
#         except RuntimeError:
#             continue

#         # sample_data = torch.bernoulli(data)
#         sample_data = (D.Normal(D.Categorical(data)))
#         sample_data = torch.flatten(sample_data.clone())

#         v, v1 = rbm_first(sample_data)
        
#         loss = rbm_first.free_energy(v) - rbm_first.free_energy(v1)
#         loss_.append(loss.data + 100)
        
#         train_op.zero_grad()
#         loss.backward()
#         train_op.step()

#     '''                                                                 Mark point'''
#     print("1ST BBrbm_first Training loss for {0} epoch {1}".format(epoch, np.mean(loss_)))

# '''2TH GB-rbm_first'''

# train_dataloader = DataLoader(loss_,
#                               batch_size=BATCH_SIZE,
#                               shuffle=True)
# for epoch in range(HIDDEN_UNITS[1]):
#     for _, (data) in enumerate(loss_):
#         try:
#             data = torch.tensor(Variable(data.view(-1, BATCH_SIZE).uniform_(0, 1)), dtype=torch.float32)
#         except RuntimeError:
#             continue

#         sample_data = torch.bernoulli(data)
#         sample_data = torch.flatten(sample_data.clone())

#         v, v1 = rbm_first(sample_data)
        
#         loss = rbm_first.free_energy(v) - rbm_first.free_energy(v1)
#         loss_.append(loss.data + 100)
        
#         train_op.zero_grad()
#         loss.backward()
#         train_op.step()

#     '''                                                                 Mark point'''
#     print("2ST BBrbm_first Training loss for {0} epoch {1}".format(epoch, np.mean(loss_)))

# '''3TH GB-rbm_first'''

# train_dataloader = DataLoader(loss_,
#                               batch_size=BATCH_SIZE,
#                               shuffle=True)
# loss_ = []
# for epoch in range(HIDDEN_UNITS[2]):
#     loss_ = []
#     for _, (data) in enumerate(loss_):
#         try:
#             data = torch.tensor(Variable(data.view(-1, BATCH_SIZE).uniform_(0, 1)), dtype=torch.float32)
#         except RuntimeError:
#             continue

#         sample_data = torch.bernoulli(data)
#         sample_data = torch.flatten(sample_data.clone())

#         v, v1 = rbm_first(sample_data)
        
#         loss = rbm_first.free_energy(v) - rbm_first.free_energy(v1)
#         loss_.append(loss.data + 100)
        
#         train_op.zero_grad()
#         loss.backward()
#         train_op.step()

#     '''                                                                 Mark point'''
#     print("3ST BBrbm_first Training loss for {0} epoch {1}".format(epoch, np.mean(loss_)))

