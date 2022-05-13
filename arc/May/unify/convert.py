from tqdm import tqdm
import pandas as pd
import numpy as np
import os
print("[IMPO] Start read and convert")

PATH = './aha/'
RESULT_PATH = './result/'

# Read file and pre-processing
print("[INFO] Pre-processing for make clean")
file_list = os.listdir(PATH)
for i in range(len(file_list) - 1):
    if (file_list[i] == '.DS_Store'):
        file_list.pop(i)
    
    str_file_name = file_list[i]
    file_list[i] = str_file_name[:4]
file_list = list(set(file_list))

# changing whole record using rdsamp
print("[INFO]./rdsamp commending start")
for i in range(len(file_list)):
    print("[IWIP]\t\trdsamp Converting", file_list[i])
    commend = './rdsamp -r ' + PATH + file_list[i] + ' -f 0 -pd -c -v > ./result/converted_csv' + file_list[i] + '.csv' 
    os.system(commend)
print("[RSLT]\t\t\t", os.listdir(RESULT_PATH))

# changing whole record using rdann
print("[INFO]./rdann commending start")
for i in range(len(file_list)):
    print("[IWIP]\t\trdann Converting", file_list[i])
    commend = './rdann -v -a atr -r ' + PATH + file_list[i] + ' > ./result/converted_txt' + file_list[i] + '.txt'
    os.system(commend)
print("[RSLT]\t\t\t", os.listdir(RESULT_PATH))

# Setting for the unify csv file
rdsamp_df = pd.DataFrame(pd.read_csv('./result/converted_txt0001.txt', delimiter='\t'))
rdann_df = pd.DataFrame(pd.read_csv('./result/converted_csv0001.csv', delimiter=','))

'''unify_df = pd.DataFrame({'SAMPLE_NUMBER' : [], 
                         'TIME' : [], 
                         'ECG_A' : [],
                         'ECG_B' : [],
                         'TYPE' : [], 
                         'SUB' : []})
'''
print(rdsamp_df)
print(rdann_df)

# Unifying
print("[INFO] Starting Unify...")
rdsamp_count = 0

for i in range(0, len(rdann_df.index)):
    rdsamp_df_split_str = str(rdsamp_df.iloc[0]).split()
    #print(rdsamp_df_split_str[8])
    print("[IWIP]{0}th Row inserting...\trdsamp_count : {1}".format(i, rdsamp_count))    
    if str(rdann_df.iloc[i][0]) == str(rdsamp_df_split_str[8]):
    #                  Num,          Time,           ECG_A,              ECG_B,                 Type,                          Sub
       #unify_df[-1] = [i, rdsamp_df.iloc[i][0], rdann_df.iloc[i][1], rdann_df.iloc[i][2], rdsamp_df.iloc[i][2], rdsamp_df.iloc[i][3]] 
       #unify_df[-1] = [i, rdsamp_df.iloc[i][0], rdann_df.iloc[i][1], rdann_df.iloc[i][2], rdsamp_df.iloc[i][2], rdsamp_df.iloc[i][3]]
       unify_df.insert(i, rdsamp_df_split_str[7], rdann_df.iloc[i][1], rdann_df.iloc[i][2], rdsamp_df_split_str[8], rdsamp_df.iloc[i][3])
       print("[HAPP] ********** MATCHED! **********")
       rdsamp_count += 1
    else:
    #                  Num,         Time,           ECG_A,              ECG_B,             Type,                 Sub
        #unify_df[-1] = [i, rdsamp_df.iloc[i][0], rdann_df.iloc[i][1], rdann_df.iloc[i][2], None, rdsamp_df.iloc[i][3]]
        #i, rdsamp_df_split_str[7], rdann_df.iloc[i][1], rdann_df.iloc[i][2], rdsamp_df_split_str[8], rdsamp_df_split_str[11]
        insert_row = {'SAMPLE_NUMBER': i, 'TIME':rdsamp_df_split_str[7], 'ECG_A':rdann_df.iloc[i][1], 'ECG_B' : rdann_df.iloc[i][2], "TYPE" : rdsamp_df_split_str[8], "SUB" : rdsamp_df_split_str[11]}
        unify_df = unify_df.append(insert_row, ignore_index=True)


#print(testing_csv_to_df.iloc[0][1])
print(unify_df)
os.makedirs('result', exist_ok=True)
unify_df.to_csv('result/final.csv')