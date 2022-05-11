from scipy.signal import butter, filtfilt
import pandas as pd
import numpy as np
import itertools
import wfdb
import os

FILE_NUM_FLAG = 0
DB_LIST = ['db1', 'db2', 'db3/svdb']
print("[INFO] Pre-processing for make clean")

db1_signals = list()
db2_signals = list()
db3_signals = list()
db1_anno = list()
db2_anno = list()
db3_anno = list()

#######################
#    List to tuple    #
#######################
def list_to_list(input_list):
    input_list_to_list = list(itertools.chain(*input_list))
    return input_list_to_list

#######################
#    Low-pass filter  #
#######################
def butter_lowpass(cutoff, butter_data, fs=35, order=12):
    cutoff = 2
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq

    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, list_to_list(butter_data))

def return_list():
    global db1_signals
    global db2_signals
    global db3_signals
    global db1_anno
    global db2_anno 
    global db3_anno
    '''DB1
    '''

    FILE_NUM_FLAG = 0
    PATH = './data/' + DB_LIST[FILE_NUM_FLAG] + '/'
    print("[INFO] Read records file from ", PATH)
    with open(PATH + 'RECORDS') as f:
        record_lines = f.readlines()

    pre_records = []
    for x in record_lines:
        pre_records.append(x.strip())
    print("[RSLT]\t\t\t Export records ...")
    print("\t\t",pre_records)

    print("[INFO]./rdsamp commending start")
    for i in range(len(pre_records)):
        match_list_cnt = 0
        print("[IWIP]\t\trdsamp Converting", pre_records[i])
        signals, _ = wfdb.rdsamp(PATH + pre_records[i], sampfrom=0)
        annotation = wfdb.rdann(PATH + pre_records[i], 'atr', sampfrom=0, return_label_elements=['symbol'])

        db1_signals.append(signals.tolist())
        db1_anno.append(annotation.symbol)

    '''DB2
    '''

    FILE_NUM_FLAG = 1
    PATH = './data/' + DB_LIST[FILE_NUM_FLAG] + '/'
    print("[INFO] Read records file from ", PATH)
    with open(PATH + 'RECORDS') as f:
        record_lines = f.readlines()

    pre_records = []
    for x in record_lines:
        pre_records.append(x.strip())
    print("[RSLT]\t\t\t Export records ...")
    print("\t\t",pre_records)

    print("[INFO]./rdsamp commending start")
    for i in range(len(pre_records)):
        match_list_cnt = 0
        print("[IWIP]\t\trdsamp Converting", pre_records[i], PATH)
        signals, _ = wfdb.rdsamp(PATH + pre_records[i], sampfrom=0)
        annotation = wfdb.rdann(PATH + pre_records[i], 'atr', sampfrom=0, return_label_elements=['symbol'])

        db2_signals.append(signals.tolist())
        db2_anno.append(annotation.symbol)

    '''DB3
    '''

    FILE_NUM_FLAG = 2
    PATH = './data/' + DB_LIST[FILE_NUM_FLAG] + '/'
    print("[INFO] Read records file from ", PATH)
    with open(PATH + 'RECORDS') as f:
        record_lines = f.readlines()

    pre_records = []
    for x in record_lines:
        pre_records.append(x.strip())
    print("[RSLT]\t\t\t Export records ...")
    print("\t\t",pre_records)

    print("[INFO]./rdsamp commending start")
    for i in range(len(pre_records)):
        match_list_cnt = 0
        print("[IWIP]\t\trdsamp Converting", pre_records[i], PATH)
        signals, _ = wfdb.rdsamp(PATH + pre_records[i], sampfrom=0)
        annotation = wfdb.rdann(PATH + pre_records[i], 'atr', sampfrom=0, return_label_elements=['symbol'])

        db3_signals.append(signals.tolist())
        db3_anno.append(annotation.symbol)

    db1_butter = list()
    db2_butter = list()
    db3_butter = list()

    # db1_butter = db1_signals
    # db2_butter = db2_signals
    # db3_butter = db3_signals

    print("[INFO] DB1 Filtering...")
    for i in range(len(db1_signals)):
        print(len(db1_signals[i]))
        db1_butter.append(butter_lowpass(3.667, (db1_signals[i])))
    
    print("[INFO] DB2 Filtering...")
    for i in range(len(db2_signals)):
        print(len(db2_signals[i]))
        db2_butter.append(butter_lowpass(3.667, (db2_signals[i])))

    print("[INFO] DB3 Filtering...")
    for i in range(len(db3_signals)):
        print(len(db3_signals[i]))
        db3_butter.append(butter_lowpass(3.667, (db3_signals[i])))

    print("DB1 butter size : {}, DB1 Anno size : {}\n".format(len(db1_butter), len(db1_anno)),    \
          "DB2 butter size : {}, DB2 Anno size : {}\n".format(len(db2_butter), len(db2_anno)),    \
          "DB3 butter size : {}, DB3 Anno size : {}\n".format(len(db3_butter), len(db3_anno)))

    return db1_butter, db1_anno, db2_butter, db2_anno, db3_butter, db3_anno

# return_list()