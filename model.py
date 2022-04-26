import logging
logging.basicConfig(level=logging.INFO)

import data.medain_filtering_class as mf
import matplotlib.pyplot as plt
from GBRBM import RBMGaussHid
from BBRBM import RBMBer
import torch.nn as nn
import numpy as np
import torch

BATCH_SIZE = 10
EPOCH = 90
LEARNING_RATE = 0.2
ANNEALING_RATE = 0.999
VISIBLE_UNITS = 80    
HIDDEN_UNITS = 180

print("[MODL] Model main code is starting....")


print("[INFO] Read train data, cross-vaildation data and test data from median filtering code")
dataset_db1, dataset_db2, dataset_db3 = mf.ecg_filtering(True)

train_dataset = list(mf.list_to_list(dataset_db1))
cross_dataset = list(mf.list_to_list(dataset_db2))
test_dataset = list(mf.list_to_list(dataset_db3))

train_data = torch.FloatTensor(4 * train_dataset)   # tensor size : 14400
cross_data = torch.FloatTensor(4 * cross_dataset)   # tensor size : 14400
test_data = torch.FloatTensor(4 * test_dataset)     # tensor size : 14400

print("[INFO] Model object added")
bbrbm = RBMBer(VISIBLE_UNITS, HIDDEN_UNITS)
gbrbm = RBMGaussHid(VISIBLE_UNITS, HIDDEN_UNITS)

'''     BBRBM      '''
bbrbm_loss = list()

batch_cnt = 0
for i in range(EPOCH):
    # train_temp_data = torch.ones(VISIBLE_UNITS)
    train_temp_data = torch.FloatTensor(train_data[batch_cnt : batch_cnt + VISIBLE_UNITS])
    
    batch_cnt += VISIBLE_UNITS
    print("Epoch : {}".format(i + 1))
    error = bbrbm.cd(train_temp_data)
    
    del train_temp_data
    # print("Reconstruction error: %.3f" % (error.data[0]))
    print("\t\tReconstruction loss : {:.3f}".format(error.item() * 0.00000001))
    bbrbm_loss.append(float(format(error.item() * 0.00000001)))

plt.plot(bbrbm_loss)
plt.ylabel('loss')
plt.show()


'''     GBRBM      '''
gbrbm_loss = list()

batch_cnt = 0
for i in range(EPOCH):
    # train_temp_data = torch.ones(VISIBLE_UNITS)
    train_temp_data = torch.FloatTensor(train_data[batch_cnt : batch_cnt + VISIBLE_UNITS])
    
    batch_cnt += VISIBLE_UNITS
    print("Epoch : {}".format(i + 1))
    error = gbrbm.cd(train_temp_data)
    
    del train_temp_data
    # print("Reconstruction error: %.3f" % (error.data[0]))
    print("\t\tReconstruction loss : {:.3f}".format(error.item() * 0.00000001))
    gbrbm_loss.append(float(format(error.item() * 0.00000001)))
