from xml.dom import NotFoundErr
from scipy.signal import butter, lfilter, filtfilt
from tempfile import TemporaryFile
from scipy import signal as sp
import pandas as pd
import numpy as np
import itertools
import csv
import os

# FILE_FLAG_1, FILE_FLAG_2 = False
FILE_FLAG_NUMBER = 0
DATA_PATH_OG = ['./final_db1/', './final_db2/', './final_db3/']

FINAL_DB2_COLUMMS = ['MLII', 'V1', 'V2', 'V5']

db1_file_list = []
db2_file_list = []
db3_file_list = []

result_db1_list = []
result_db2_list = []
result_db3_list = []

final_db1_list = []
final_db2_list = []
final_db3_list = []

# Slicing memory list
temp_list = []

# Butter list
butter_list = []

outfile = TemporaryFile()
run_code_from = bool()

#######################
#    Low-pass filter  #
#######################
def butter_lowpass(cutoff, butter_data, fs=35, order=12):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)

    butter_list.append(filtfilt(b, a, butter_data))
    return butter_list


########################
#  Renaming data path  #
########################
def data_path_flex(run_code_from):
    if run_code_from == True:
        for i in range(len(DATA_PATH_OG)):
            DATA_PATH_OG[i] = "data/" + DATA_PATH_OG[i]
    return DATA_PATH_OG

#######################
# Slicing ecg dataset #
#######################
def slice_ecg_data(current_index, future_index, input_array):
    temp_list.clear()
    for i in range(current_index, future_index):
        if current_index >= len(input_array) or len(input_array) <= future_index:
            break
        
        temp_list.append(float(input_array[i]))
    return temp_list

#######################
#    List to tuple    #
#######################
def list_to_list(input_list):
    input_list_to_list = list(itertools.chain(*input_list))
    return input_list_to_list

#######################
#      Main Code      #
#######################
def ecg_filtering(path_bool = False):
    if path_bool == True:
        DATA_PATH = data_path_flex(path_bool)

    print("[INFO] Read file and indexing start...")
    file_dir_list = os.listdir('.')
    now_index = 0
    post_index = 200
    
    ########################################
    # 200ms width median MIT-BIH Dataset 1 #
    ########################################
    FILE_FLAG_NUMBER = 0
    print("[INFO]\tfinal_db1 direcotry found.")
    db1_file_list = os.listdir(DATA_PATH[FILE_FLAG_NUMBER])
    
    print("......\t...................i\tCurrent_Index\tFrom_Index")
    for i in range(len(db1_file_list)):
        read_csv_path = DATA_PATH[FILE_FLAG_NUMBER] + db1_file_list[i]
        db1_csv = pd.read_csv(read_csv_path)
    
        file_name_check = os.path.splitext(db1_file_list[i])[0]
        mlii_list = list(db1_csv["'MLII'"])
    
        print("[IWIP]\tfinal_db1 reading...", i, now_index, post_index)
        result_db1_list.append(list(sp.medfilt(slice_ecg_data(now_index, post_index, mlii_list))))
        now_index += 200
        post_index += 200
    
    ########################################
    # 200ms width median MIT-BIH Dataset 2 #
    ########################################
    FILE_FLAG_NUMBER = 1
    print("[INFO]\tfinal_db2 direcotry found.")
    db2_file_list = os.listdir(DATA_PATH[FILE_FLAG_NUMBER])
    
    print("....\t...................i Current_Index From_Index")
    for i in range(len(db2_file_list)):
        read_csv_path = DATA_PATH[FILE_FLAG_NUMBER] + db2_file_list[i]
        db2_csv = pd.read_csv(read_csv_path)

        try:
            mlii_list = list(db2_csv["'MLII'"])
            print("[IWIP]\tfinal_db2 reading...", i, now_index, post_index)
            result_db2_list.append(list(sp.medfilt(slice_ecg_data(now_index, post_index, mlii_list))))
    
        except KeyError:
            print("[ERRR]\t\t\t{0}th RECORD is not work. Maybe problem with columns stuff.".format(i))
            continue
        now_index += 200
        post_index += 200
    
    #####################################
    # 200ms width median SVDB Dataset 3 #
    #####################################
    FILE_FLAG_NUMBER = 2
    print("[INFO]\tfinal_db3 direcotry found.")
    db3_file_list = os.listdir(DATA_PATH[FILE_FLAG_NUMBER])
    
    print("....\t...................i Current_Index From_Index")
    for i in range(len(db2_file_list)):
        read_csv_path = DATA_PATH[FILE_FLAG_NUMBER] + db3_file_list[i]
        db3_csv = pd.read_csv(read_csv_path)
    
        try:
            ecg_list = list(db3_csv["'ECG1'"])
            print("[IWIP]\tfinal_db3 reading...", i, now_index, post_index)
            result_db3_list.append(list(sp.medfilt(slice_ecg_data(now_index, post_index, ecg_list))))
    
        except KeyError:
            continue

        now_index += 200
        post_index += 200

    now_index = 0
    post_index = 600
    
    ######################################################
    # 600ms width median MIT-BIH Dataset 1 that filtered #
    ######################################################
    FILE_FLAG_NUMBER = 0
    print("[INFO]\tfinal_db1 direcotry found.")
    print("......\t...................i\tCurrent_Index\tFrom_Index")
    
    db1_list = list_to_list(result_db1_list)
    for i in range(len(db1_list)):
        print("[IWIP]\tfinal_db1 reading...", i, now_index, post_index)
        # final_db1_list.append(list(sp.medfilt(slice_ecg_data(now_index, post_index, result_db1_list[i]))))
        final_db1_list.append(list(sp.medfilt(slice_ecg_data(now_index, post_index, db1_list))))
        now_index += 600
        post_index += 600
    
    now_index = 0
    post_index = 600
    
    ######################################################
    # 600ms width median MIT-BIH Dataset 2 that filtered #
    ######################################################
    FILE_FLAG_NUMBER = 1
    print("[INFO]\tfinal_db2 direcotry found.")
    print("....\t...................i Current_Index From_Index")
    
    db2_list = list_to_list(result_db2_list)
    for i in range(len(db2_list)):
        print("[IWIP]\tfinal_db2 reading...", i, now_index, post_index)
        final_db2_list.append(list(sp.medfilt(slice_ecg_data(now_index, post_index, db2_list))))
        now_index += 600
        post_index += 600
    
    now_index = 0
    post_index = 600
    
    ######################################################
    # 600ms width median MIT-BIH Dataset 3 that filtered #
    ######################################################
    FILE_FLAG_NUMBER = 2
    print("[INFO]\tfinal_db3 direcotry found.")
    print("....\t...................i Current_Index From_Index")

    db3_list = list_to_list(result_db3_list)
    for i in range(len(db3_list)):
        print("[IWIP]\tfinal_db3 reading...", i, now_index, post_index)
        final_db3_list.append(list(sp.medfilt(slice_ecg_data(now_index, post_index, db3_list))))
        now_index += 600
        post_index += 600
    
    '''Okay,
    200ms-width-median filtering is over...maybe.
    And now we going to do 600ms-width median filtering from 200ms-width median filter'''
    
    low_passed_db1 = butter_lowpass(3.667, final_db1_list[0])
    low_passed_db2 = butter_lowpass(3.667, final_db2_list[0])
    low_passed_db3 = butter_lowpass(3.667, final_db3_list[0])
    print("[DONE] Pre-processing is done.")

    try:        
        # final_csv_dict = {
        #     'dataset_1' : final_db1_list[0],
        #     'dataset_2' : final_db2_list[0],
        #     'dataset_3' : final_db3_list[0]
        # }
        final_csv_dict = {
            'result' : low_passed_db1[0]
        }

        df_f = pd.DataFrame(final_csv_dict)
        df_f.to_csv('outfile_filtered.csv')

        og_csv_dict = {
            'MLII' : mlii_list,
            'ECG' : ecg_list
        }

        df_og = pd.DataFrame(og_csv_dict)
        df_og.to_csv('outfile_original.csv')

        return low_passed_db1, low_passed_db2, low_passed_db3
    
    except ValueError:
        print("[ERRR] Got a issue with create file")    
        
        return low_passed_db1, low_passed_db2, low_passed_db3