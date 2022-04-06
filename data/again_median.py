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

temp_list = []

###############################
# Slicing ecg data set as 200ms
###############################
def slice_ecg_data(previous_index, current_index, input_array):
    temp_list.clear()
    for i in range(previous_index, current_index):
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

for i in range(len(db1_file_list)):
    read_csv_path = DATA_PATH[FILE_FLAG_NUMBER] + db1_file_list[i]
    print("[INFO] Read csv from ", read_csv_path)
    ecg_values = pd.read_csv(read_csv_path)
    
    # 'MLII' and 'V1' or 'V4'
    mlii_list = list(ecg_values["'MLII'"])
    
    # Checking length of final_db1 datasets
    if now_index >= len(mlii_list): break
    
    print("[IWIP]\tfinal_db1 reading...", i, now_index, post_index) 
    print(sp.medfilt(slice_ecg_data(now_index, post_index, mlii_list)))
    now_index += 200
    post_index += 200

####################
# MIT-BIH Dataset2
####################
FILE_FLAG_NUMBER = 1
print("[INFO]\tfinal_db2 direcotry found.")
db2_file_list = os.listdir(DATA_PATH[FILE_FLAG_NUMBER])
now_index = 0
post_index = 200

db2_other_columns_list = {
    "MLII_V1" : [222, 234, 221, 219, 231, 232, 233, 121, 105, 111, 107, 113, 202, 217, 228, 200, 214, 210, 213, 212],
    "MLII_V2" : [117, 103],
    "MLII_V5" : [123, 100],
    "V5_V2" : [102, 104]
}

for j in range(len(db2_file_list)):
    read_csv_path = DATA_PATH[FILE_FLAG_NUMBER] + db2_file_list[i]
    ecg_values = pd.read_csv(read_csv_path)
    isthatyours = os.path.splitext(db2_file_list[j])[0]
    if isthatyours in str(db2_other_columns_list["MLII_V1"][j]):
        for i in range(len(db2_other_columns_list["MLII_V1"])):
            print("[WARN]\t\t\t>> MLII V1 Columns reading start...")
            
            mlii_list = list(ecg_values["'MLII'"])
            v1_list = list(ecg_values["'V1'"])
            print("[IWIP]MLII_V1\tfinal_db2 reading...", i, now_index, post_index)
            now_index += 200
            post_index += 200

    try:
        if isthatyours in str(db2_other_columns_list["MLII_V2"][j]):
            for i in range(len(db2_other_columns_list["MLII_V2"])):
                print("[WARN]\t\t\t>> MLII V2 Columns reading start...")

                mlii_list = list(ecg_values["'MLII'"])
                v2_list = list(ecg_values["'V2'"])
                print("[IWIP]MLII_V2\tfinal_db2 reading...", i, now_index, post_index)
                now_index += 200
                post_index += 200
    except IndexError:
        continue

    if isthatyours in str(db2_other_columns_list["MLII_V5"][j]):
        for i in range(len(db2_other_columns_list["MLII_V5"])):
            print("[WARN]\t\t\t>> MLII V5 Columns reading start...")
            
            mlii_list = list(ecg_values["'MLII'"])
            v5_list = list(ecg_values["'V5'"])
            print("[IWIP]MLII_V5\tfinal_db2 reading...", i, now_index, post_index)
            now_index += 200
            post_index += 200

    if isthatyours in str(db2_other_columns_list["V5_V2"][j]):
        for i in range(len(db2_other_columns_list["V5_V2"])):
            print("[WARN]\t\t\t>> V5 V2 Columns reading start...")
        
            mlii_list = list(ecg_values["'V5'"])
            v5_list = list(ecg_values["'V2'"])
            print("[IWIP]V5_V2\tfinal_db2 reading...", i, now_index, post_index)
            now_index += 200
            post_index += 200

    print(isthatyours)

# for i in range(len(db2_file_list)):
#     read_csv_path = DATA_PATH[FILE_FLAG_NUMBER] + db2_file_list[i]
#     print("[INFO] Read csv from ", read_csv_path)
#     ecg_values = pd.read_csv(read_csv_path)
#     if now_index >= len(v5_list) or len(v2_list): break
#     print("[IWIP]\t\tfinal_db2 reading...", i, now_index, post_index)
#     print(sp.medfilt(slice_ecg_data(now_index, post_index, v5_list)))
#     print(sp.medfilt(slice_ecg_data(now_index, post_index, v2_list)))
#     now_index += 200
#     post_index += 200

        

print(db1_file_list)