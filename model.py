from sklearn.model_selection import train_test_split
import data.medain_filtering_class as mf
from torch.utils.data import DataLoader
# import matplotlib.pyplot as plt
# from sklearn import datasets
from torch.autograd import Variable
import torch.optim as optim
from RBM import RBM
import numpy as np
import torch


BATCH_SIZE = 10
EPOCH = 120
LEARNING_RATE = 0.2
ANNEALING_RATE = 0.999
VISIBLE_UNITS = 80    
HIDDEN_UNITS = 180
K_FOLD = 1

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

rbm = RBM(n_vis=VISIBLE_UNITS, n_hid=HIDDEN_UNITS, k=K_FOLD, batch=BATCH_SIZE)
train_op = optim.SGD(rbm.parameters(), 0.1)

for epoch in range(EPOCH):
    loss_ = []
    for _, (data) in enumerate(train_dataloader):
        data = Variable(data.view(-1, BATCH_SIZE).uniform_(0, 1))
        print("OG Data Size : ", data.size(), "OG Data : ", data)

        sample_data = torch.bernoulli(data)
        print("SP Data Size : ", sample_data.size(), "SP Data : ", sample_data)

        v, v1 = rbm(sample_data)
        
        loss = rbm.free_energy(v) - rbm.free_energy(v1)
        loss_.append(loss.data)
        
        train_op.zero_grad()
        loss.backward()
        train_op.step()

    print("Training loss for {0} epoch {1}".format(epoch, np.mean(loss_)))

