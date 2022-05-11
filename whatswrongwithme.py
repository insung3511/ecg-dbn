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

# %%
print("[MODL] Model main code is starting....")

print("[INFO] Read train data, cross-vaildation data and test data from median filtering code")
db1_sig, db1_label, db2_sig, db2_label, db3_sig, db3_label = rs.return_list()

# %%
X_train, X_test, y_train, y_test = train_test_split(
    (db1_sig + db2_sig), 
    (db1_label + db2_label),
    test_size=0.33,
    shuffle=True
)

train_dataloader = DataLoader(X_train,
                              batch_size=BATCH_SIZE,
                              num_workers=0, collate_fn=lambda x: x,
                              shuffle=True)

test_dataloader = DataLoader(X_test,
                             batch_size=BATCH_SIZE,
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

# %%
'''Train Part'''

loss_ = []
for epoch in range(EPOCH):
    '''First bbrbm'''
    for _, (data) in enumerate(train_dataloader):
        try:
            data = Variable(
                torch.tensor(data, dtype=torch.float32)
            ).uniform_(0, 1)

            # print(data)
            # print(data.size())

            print("Success")
        except ValueError:
            print("Fail")
            pass

        data = torch.tensor(data)
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



