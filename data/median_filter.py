from importlib.resources import path
from scipy import signal as sp
import pandas as pd
import numpy as np
import os

print("[INFO] Dataset setting up start...")
DATA_PATH = ['./final_db1/', './final_db2/', './final_db3/']
PATH_NUMBER = None

db1_file_list = []
db2_file_list = []
svdb_file_list = []

previous_index = 0
current_index = 0
future_index = 200

temp_list = []

db1_filtered_list = []
db2_filtered_list = []
db3_filtered_list = []

dict_db = {
    "db1" : None,
    "db2" : None,
    "svdb" : None
}

print("[INFO] Indexing file direcotry and list...")
file_dir_list = os.listdir('.')
for i in range(len(file_dir_list)):
    if (file_dir_list[i] == 'final_db1'):
        PATH_NUMBER = i
        print("[IWIP]\tFinal_db1 directory found!")
        for i in range(len(os.listdir(DATA_PATH[0]))):
            db1_list = os.listdir(DATA_PATH[0])
        print("[DONE]\t\tRead file list from final_db1...")

    elif (file_dir_list[i] == 'final_db2'):
        PATH_NUMBER = i
        print("[IWIP]\tFinal_db2 directory found!")
        for i in range(len(os.listdir(DATA_PATH[1]))):
            db2_list = os.listdir(DATA_PATH[1])
        print("[DONE]\t\tRead file list from final_db2...")
        
    elif (file_dir_list[i] == 'final_db3'):
        PATH_NUMBER = i
        print("[IWIP]\tFinal_db3 directory found!")
        for i in range(len(os.listdir(DATA_PATH[2]))):
            db3_list = os.listdir(DATA_PATH[2])
        print("[DONE]\t\tRead file list from final_db3...")

def slice_ecg_data(previous_index, current_index, input_array):
    temp_list.clear()
    for i in range(previous_index, current_index):
        temp_list.append(float(input_array[i]))
    return temp_list

def median_200ms(path_number):
    global database_csv_def
    if path_number == 0:
        for i in range(len(db1_file_list)):
            csv_path = DATA_PATH[path_number] + db1_file_list[i]
            database_csv_def = pd.read_csv(csv_path, skipinitialspace=True)
            print(database_csv_def)
    elif path_number == 1:
        for i in range(len(db2_file_list)):
            csv_path = DATA_PATH[path_number] + db1_file_list[i]
            database_csv_def = pd.read_csv(csv_path, skipinitialspace=True)

    elif path_number == 2:
        for i in range(len(db2_file_list)):
            csv_path = DATA_PATH[path_number] + db1_file_list[i]
            database_csv_def = pd.read_csv(csv_path, skipinitialspace=True)

    current_index = 0
    future_index = 200
    for i in range(0, len(database_csv_def)):
        if current_index >= len(database_csv_def):
            break
        print("[INFO] >>>>>>>>> ", i, current_index, future_index)
        print(sp.medianfilt(slice_ecg_data(current_index, future_index, database_csv_def)))
        current_index += 200
        future_index += 200

median_200ms(0)