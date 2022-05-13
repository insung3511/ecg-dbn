from unittest import result
import pandas as pd
import os

print("[IMPO] Start read and convert")

PATH = './aha/'
RESULT_PATH = './result/'

first_rdann_df = pd.read_csv('./result/0001_rdann.csv')
first_rdsamp_df = pd.read_csv('./result/converted_csv0001.csv')

second_rdann_df = pd.read_csv('./result/0201_rdann.csv')
second_rdsamp_df = pd.read_csv('./result/converted_csv0201.csv')

first_result = pd.concat([first_rdann_df, first_rdsamp_df])
first_result.sort_index(by="'sample #'")
print(first_result)

first_result.to_csv('final_result_first##.csv', sep=',', na_rep='NaN')