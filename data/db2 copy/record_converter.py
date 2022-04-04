import os
import re

FILE_NUM_FLAG = 0
RESULT_PATH = './result_'
DB_LIST = ['ahadb', 'cudb', 'edb', 'nstdb']
USER_PATH = input("[USER] Type Database category (ex aha, cu, esc, nst): ")
print("[INFO] Pre-processing for make clean")

for i in range(len(DB_LIST)):
    if DB_LIST[i] == USER_PATH:
        FILE_NUM_FLAG = i

PATH = './' + DB_LIST[FILE_NUM_FLAG] + '/' + '1.0.0/'

with open(PATH + 'RECORDS') as f:
    record_lines = f.readlines()

print(record_lines)