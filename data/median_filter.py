import matplotlib.pyplot as plt
from time import time
import medfilt as mf
import pandas as pd
import numpy as np
import scipy as sp

DATA_PATH = './final_db3/rdsamp_svdb800.csv'
# FIELDS = ['Elapsed time', 'ECG1']
og_df = pd.read_csv(DATA_PATH, skipinitialspace=True)
second_indexing = 0
all_result = []

# Get seconds datas
ecg_seconds = list(og_df["'Elapsed time'"])

# Get ECG1 mV datas
ecg_v1 = list(og_df["'ECG1'"])

def time_matching(current_index):
    while True:
        current_index += 1
        if  200.0 <= float(ecg_seconds[current_index]) <= 200.0:
            return current_index + 2

print("#"*10, (ecg_seconds[4]))

def ecg_collecting(current_index, ending_index, input_list):
    for i in range(current_index, ending_index):
        input_list.append(float(ecg_v1[i]))
    return input_list

# print(type(ecg_collecting(second_indexing, time_matching(second_indexing), ecg_v1)))
# all_result[0] = np.median(ecg_collecting(second_indexing, time_matching(second_indexing), ecg_v1))
# sp.signal.medfilt(ecg_v1, 200)

