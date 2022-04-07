import itertools
from re import L
from scipy import signal as sp
import pandas as pd
import os

# FILE_FLAG_1, FILE_FLAG_2 = False
FILE_FLAG_NUMBER = 0
DATA_PATH = ['./final_db1/', './final_db2/', './final_db3/']

FINAL_DB2_COLUMMS = ['MLII', 'V1', 'V2', 'V5']

db1_file_list = []
db2_file_list = []
db3_file_list = []

result_db1_list = []
result_db2_list = []
result_db3_list = []

# Slicing memory list
temp_list = []

#######################
# Slicing ecg dataset #
#######################
def slice_ecg_data(current_index, future_index, input_array):
    temp_list.clear()
    for i in range(current_index, future_index):
        temp_list.append(float(input_array[i]))
    return temp_list

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
    file_name_check = os.path.splitext(db2_file_list[i])[0]

    try:
        mlii_list = list(db2_csv["'MLII'"])
        print("[IWIP]\tfinal_db2 reading...", i, now_index, post_index)
        result_db2_list.append(list(sp.medfilt(slice_ecg_data(now_index, post_index, mlii_list))))
    except KeyError:
        # for i in range(len(FINAL_DB2_COLUMMS)):
        #     if db2_csv[FINAL_DB2_COLUMMS[i]]:
        #         something_list = list(db2_csv[FINAL_DB2_COLUMMS[i]])
        #         result_db2_list.append(list(sp.medfilt(slice_ecg_data(now_index, post_index, something_list))))
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
    file_name_check = os.path.splitext(db3_file_list[i])[0]

    try:
        ecg_list = list(db3_csv["'ECG1'"])
        print("[IWIP]\tfinal_db3 reading...", i, now_index, post_index)
        result_db3_list.append(list(sp.medfilt(slice_ecg_data(now_index, post_index, ecg_list))))
    except KeyError:
        continue
    
    now_index += 200
    post_index += 200

'''Okay,
200ms-width-median filtering is over...maybe.
And now we going to do 600ms-width median filtering from 200ms-width median filter'''

#######################
# Combine in one list #
#######################
combine_list_in_list = result_db1_list + result_db2_list + result_db3_list
combine_list = list(itertools.chain(*combine_list_in_list))
#print(combine_list) 

final_result = []
now_index = 0
post_index = 600

print("[INFO] BEGIN OF 600ms width median filtering")
FILE_FLAG_NUMBER = 0

print("......\t...................i\tCurrent_Index\tFrom_Index")

# print(all_200ms_list)
# mlii_list = list(db1_csv["'MLII'"])
for i in range(len(combine_list)):
    if len(combine_list) <= post_index: break
    print("[IWIP]\tfinal all list reading...", i, now_index, post_index)
    final_result.append((sp.medfilt(slice_ecg_data(now_index, post_index, combine_list))))
    now_index += 600
    post_index += 600

print("[DONE] AHHHHHHHHHHHHHHHHHHHHHHHHHHH FUCK")