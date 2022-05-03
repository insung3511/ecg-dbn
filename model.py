from sklearn.model_selection import train_test_split
import torch.distributions.distribution as D
import data.medain_filtering_class as mf
from torch.utils.data import DataLoader
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.optim as optim
from RBM import RBM
import numpy as np
import torch


BATCH_SIZE = 10
EPOCH = 120
LEARNING_RATE = 0.2
ANNEALING_RATE = 0.999
VISIBLE_UNITS = [180, 200, 250]
HIDDEN_UNITS = [80, 100, 120]
K_FOLD = 1

def show_adn_save(file_name, result):
    nprst = np.transpose(result.numpy(), (1, 2, 0))
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

rbm = RBM(n_vis=VISIBLE_UNITS[0], n_hid=HIDDEN_UNITS[0], k=K_FOLD, batch=BATCH_SIZE)
train_op = optim.SGD(rbm.parameters(), 0.1)
loss_ = []

'''1TH BB-RBM'''

for epoch in range(HIDDEN_UNITS[0]):
    for _, (data) in enumerate(train_dataloader):
        try:
            data = torch.tensor(Variable(data.view(-1, BATCH_SIZE).uniform_(0, 1)), dtype=torch.float32)
        except RuntimeError:
            continue

        sample_data = torch.bernoulli(data)
        sample_data = torch.flatten(sample_data.clone())

        v, v1 = rbm(sample_data)
        
        loss = rbm.free_energy(v) - rbm.free_energy(v1)
        loss_.append(loss.data + 100)
        
        train_op.zero_grad()
        loss.backward()
        train_op.step()

    '''                                                                 Mark point'''
    print("1ST BBRBM Training loss for {0} epoch {1}".format(epoch, np.mean(loss_)))

'''2TH BB-RBM'''

train_dataloader = DataLoader(loss_,
                              batch_size=BATCH_SIZE,
                              shuffle=True)
for epoch in range(HIDDEN_UNITS[1]):
    for _, (data) in enumerate(loss_):
        try:
            data = torch.tensor(Variable(data.view(-1, BATCH_SIZE).uniform_(0, 1)), dtype=torch.float32)
        except RuntimeError:
            continue

        sample_data = torch.bernoulli(data)
        sample_data = torch.flatten(sample_data.clone())

        v, v1 = rbm(sample_data)
        
        loss = rbm.free_energy(v) - rbm.free_energy(v1)
        loss_.append(loss.data + 100)
        
        train_op.zero_grad()
        loss.backward()
        train_op.step()

    '''                                                                 Mark point'''
    print("2ST BBRBM Training loss for {0} epoch {1}".format(epoch, np.mean(loss_)))

'''3TH BB-RBM'''

train_dataloader = DataLoader(loss_,
                              batch_size=BATCH_SIZE,
                              shuffle=True)

for epoch in range(HIDDEN_UNITS[2]):
    loss_ = []
    for _, (data) in enumerate(loss_):
        try:
            data = torch.tensor(Variable(data.view(-1, BATCH_SIZE).uniform_(0, 1)), dtype=torch.float32)
        except RuntimeError:
            continue

        sample_data = torch.bernoulli(data)
        sample_data = torch.flatten(sample_data.clone())

        v, v1 = rbm(sample_data)
        
        loss = rbm.free_energy(v) - rbm.free_energy(v1)
        loss_.append(loss.data + 100)
        
        train_op.zero_grad()
        loss.backward()
        train_op.step()

    '''                                                                 Mark point'''
    print("3ST BBRBM Training loss for {0} epoch {1}".format(epoch, np.mean(loss_)))

show_adn_save("result", v.data)

plt.plot(loss_)
plt.show()

''' GBRBM START '''
# '''1TH GB-RBM'''
# for epoch in range(HIDDEN_UNITS[0]):
#     for _, (data) in enumerate(test_dataloader):
#         try:
#             data = torch.tensor(Variable(data.view(-1, BATCH_SIZE).uniform_(0, 1)), dtype=torch.float32)
#         except RuntimeError:
#             continue

#         # sample_data = torch.bernoulli(data)
#         sample_data = (D.Normal(D.Categorical(data)))
#         sample_data = torch.flatten(sample_data.clone())

#         v, v1 = rbm(sample_data)
        
#         loss = rbm.free_energy(v) - rbm.free_energy(v1)
#         loss_.append(loss.data + 100)
        
#         train_op.zero_grad()
#         loss.backward()
#         train_op.step()

#     '''                                                                 Mark point'''
#     print("1ST BBRBM Training loss for {0} epoch {1}".format(epoch, np.mean(loss_)))

# '''2TH GB-RBM'''

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

#         v, v1 = rbm(sample_data)
        
#         loss = rbm.free_energy(v) - rbm.free_energy(v1)
#         loss_.append(loss.data + 100)
        
#         train_op.zero_grad()
#         loss.backward()
#         train_op.step()

#     '''                                                                 Mark point'''
#     print("2ST BBRBM Training loss for {0} epoch {1}".format(epoch, np.mean(loss_)))

# '''3TH GB-RBM'''

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

#         v, v1 = rbm(sample_data)
        
#         loss = rbm.free_energy(v) - rbm.free_energy(v1)
#         loss_.append(loss.data + 100)
        
#         train_op.zero_grad()
#         loss.backward()
#         train_op.step()

#     '''                                                                 Mark point'''
#     print("3ST BBRBM Training loss for {0} epoch {1}".format(epoch, np.mean(loss_)))

