import data.medain_filtering_class as mf
from numpy import float64
import tensorflow as tf
import numpy as np
import logging



'''
####################
####################
####################
####################
####################
####################
####################
####################
####################
####################
'''
if __name__ == '__main__':
    BATCH_SIZE = 10
    EPOCH = 100
    LEARNING_RATE = 0.2
    ANNEALING_RATE = 0.999
    
    print("[MODL] Model main code is starting....")
    logging.basicConfig(level=logging.INFO)

    print("[INFO] Read train data, cross-vaildation data and test data from median filtering code")
    dataset_db1, dataset_db2, dataset_db3 = mf.ecg_filtering(True)
    train_dataset = tuple(mf.list_to_list(dataset_db1 + dataset_db2))
    test_dataset = tuple(mf.list_to_list(dataset_db2 + dataset_db3))

    train_data = np.array(train_dataset)
   