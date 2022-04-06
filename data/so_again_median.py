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

###############################
# Slicing ecg data set as 200ms
###############################
def slice_ecg_data(current_index, future_index, input_array):
    temp_list.clear()
    for i in range(current_index, future_index):
        temp_list.append(float(input_array[i]))
    return temp_list

print("[INFO] Read file and indexing start...")
file_dir_list = os.listdir('.')
now_index = 0
post_index = 200

####################
# MIT-BIH Dataset 1
####################
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

####################
# MIT-BIH Dataset 2
####################
FILE_FLAG_NUMBER = 1
print("[INFO]\tfinal_db2 direcotry found.")
db2_file_list = os.listdir(DATA_PATH[FILE_FLAG_NUMBER])
print("....\t...................i Current_Index From_Index")
for i in range(len(db2_file_list)):
    read_csv_path = DATA_PATH[FILE_FLAG_NUMBER] + db2_file_list[i]
    db2_csv = pd.read_csv(read_csv_path)
    file_name_check = os.path.splitext(db2_file_list[i])[0]

    mlii_list = list(db2_csv["'MLII'"])
    print("[IWIP]\tfinal_db2 reading...", i, now_index, post_index)
    result_db2_list.append(list(sp.medfilt(slice_ecg_data(now_index, post_index, mlii_list))))
    now_index += 200
    post_index += 200
