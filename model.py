import data.medain_filtering_class as mf
from GBRBM import RBMGaussHid
from BBRBM import RBMBer
import torch.nn as nn
import numpy as np
import logging
import torch

BATCH_SIZE = 10
EPOCH = 100
LEARNING_RATE = 0.2
ANNEALING_RATE = 0.999
VISIBLE_UNITS = 80
HIDDEN_UNITS = 180

print("[MODL] Model main code is starting....")
logging.basicConfig(level=logging.INFO)

print("[INFO] Read train data, cross-vaildation data and test data from median filtering code")
dataset_db1, dataset_db2, dataset_db3 = mf.ecg_filtering(True)

train_dataset = list(mf.list_to_list(dataset_db1 + dataset_db2))
test_dataset = list(mf.list_to_list(dataset_db2 + dataset_db3))

# test_data = test_dataset[SOMEASDJFLASDFJLASDFL;AKSJDF]
train_data = torch.FloatTensor(4 * train_dataset)
test_data = torch.FloatTensor(4 * test_dataset)

print("[INFO] Model object added")
bbrbm = RBMBer(VISIBLE_UNITS, HIDDEN_UNITS)
gbrbm = RBMGaussHid(VISIBLE_UNITS, HIDDEN_UNITS)

batch_cnt = 0
for i in range(45):
    train_temp_data = torch.FloatTensor(train_dataset[batch_cnt : batch_cnt + VISIBLE_UNITS])
    batch_cnt += VISIBLE_UNITS
    print(train_temp_data, i)
    
    error = bbrbm.cd(train_temp_data)
    # print("Reconstruction loss : %.3f" % (error.data[0]))

# bbrbm = RBMBer(VISIBLE_UNITS, HIDDEN_UNITS)

# # Got a issue with this problem
# # train_data = torch.einsum('i,j->ij', list(torch.Tensor(80, 180)), train_data)
# train_data = train_data.view(VISIBLE_UNITS, HIDDEN_UNITS)

# for _ in range(BATCH_SIZE):
#     error = bbrbm.cd(train_data)
#     print("Reconstruction loss : %.3f" % (error.data[0]))
