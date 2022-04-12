import data.medain_filtering_class as mf
import torch.nn as nn
import numpy as np
from RBM import *
import logging
import torch

BATCH_SIZE = 10
EPOCH = 100
LEARNING_RATE = 0.2
ANNEALING_RATE = 0.999
VISIBLE_UNITS = 180
HIDDEN_UNITS = 80

print("[MODL] Model main code is starting....")
logging.basicConfig(level=logging.INFO)

print("[INFO] Read train data, cross-vaildation data and test data from median filtering code")
dataset_db1, dataset_db2, dataset_db3 = mf.ecg_filtering(True)

train_dataset = list(mf.list_to_list(dataset_db1 + dataset_db2))
test_dataset = list(mf.list_to_list(dataset_db2 + dataset_db3))
train_data = torch.Tensor(train_dataset)
test_data = torch.Tensor(test_dataset)

print(type(train_data))
print(type(test_data))

bbrbm = RBMBer(VISIBLE_UNITS, HIDDEN_UNITS)
m = nn.AdaptiveAvgPool1d(80)
train_data = m(train_data)

for _ in range(BATCH_SIZE):
    error = bbrbm.cd(train_data)
    print("Reconstruction loss : %.3f" % (error.data[0]))
