from sklearn.model_selection import train_test_split
import data.medain_filtering_class as mf
import matplotlib.pyplot as plt
from sklearn import datasets
import torch
import csv

BATCH_SIZE = 10
EPOCH = 120
LEARNING_RATE = 0.2
ANNEALING_RATE = 0.999
VISIBLE_UNITS = 80    
HIDDEN_UNITS = 180

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

print("X_train length : ", len(X_train))
print("X_test  length : ", len(X_test))
print("y_train length : ", len(y_train))
print("y_test  length : ", len(y_test))

train_data = torch.FloatTensor(X_train)
test_data = torch.FloatTensor(X_test)

print("[INFO] Model object added")