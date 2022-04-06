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

# db1_200ms_output
def slice_ecg_data(previous_index, current_index, input_array):
    temp_list.clear()
    for i in range(previous_index, current_index):
        temp_list.append(float(input_array[i]))
    return temp_list

print("[INFO] Read file and indexing start...")
file_dir_list = os.listdir('.')
for i in range(len(file_dir_list)):
    now_index = 0
    post_index = 200
    
    # MIT-BIH Dataset1
    if (file_dir_list[i] == 'final_db1'):
        FILE_FLAG_NUMBER = 0
        print("[IWIP]\tfinal_db1 direcotry found.")
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

    # MIT-BIH Dataset2
    if (file_dir_list[i] == 'final_db2'):
        FILE_FLAG_NUMBER = 1
        print("[IWIP]\tfinal_db2 direcotry found.")
        db2_file_list = os.listdir(DATA_PATH[FILE_FLAG_NUMBER])
        
        for i in range(len(db2_file_list)):
            read_csv_path = DATA_PATH[FILE_FLAG_NUMBER] + db2_file_list[i]
            read_csv_columns = pd.read_csv(read_csv_path)
            
        
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