import numpy as np

import scipy.signal as ss
import scipy.io as io
import scipy

from wfdb import processing as wp
import wfdb

FILE_NUM_FLAG = 0
DB_LIST = ['db1', 'db2', 'db3/svdb']
PATH = './data/' + DB_LIST[FILE_NUM_FLAG] + '/'

db1_signals = list()
db2_signals = list()
db3_signals = list()
db1_anno = list()
db2_anno = list()
db3_anno = list()

def get_median_filter_width(sampling_rate, duration):
    res = int(sampling_rate * duration)
    res += ((res % 2) - 1)
    return res


def filter_signal(X):
    global mfa
    X0 = X
    for mi in range(0, len(mfa)):
        X0 = ss.medfilt(X0, mfa[mi])
    return X0


print("[INFO] Read records file from ", PATH)
with open(PATH + 'RECORDS') as f:
    record_lines = f.readlines()

pre_records = []
for x in record_lines:
    pre_records.append(x.strip())
print("[RSLT]\t\t\t Export records ...")

print("[INFO]./rdsamp commending start")
for i in range(len(pre_records)):
    match_list_cnt = 0
    print("[IWIP]\t\trdsamp Converting", pre_records[i])
    signals, _ = wfdb.rdsamp(PATH + pre_records[i], sampfrom=0)
    annotation = wfdb.rdann(PATH + pre_records[i], 'atr', sampfrom=0, return_label_elements=['symbol'])

    ms_flt_array = [0.2, 0.6]
    mfa = np.zeros(len(ms_flt_array), dtype='int')
    for i in range(0, len(ms_flt_array)):
        mfa[i] = get_median_filter_width(360, ms_flt_array[i])

    signal_flt = np.trim_zeros(filter_signal(signals)).tolist()

    db1_signals.append(signal_flt.tolist())
    db1_anno.append(annotation.symbol)

    print(signal_flt)
    print(len(signal_flt))

'''DB2
'''

# FILE_NUM_FLAG = 1
# PATH = './data/' + DB_LIST[FILE_NUM_FLAG] + '/'
# print("[INFO] Read records file from ", PATH)
# with open(PATH + 'RECORDS') as f:
#     record_lines = f.readlines()

# pre_records = []
# for x in record_lines:
#     pre_records.append(x.strip())
# print("[RSLT]\t\t\t Export records ...")

# print("[INFO]./rdsamp commending start")
# for i in range(len(pre_records)):
#     match_list_cnt = 0
#     print("[IWIP]\tread Converting", pre_records[i], PATH)
#     signals, _ = wfdb.rdsamp(PATH + pre_records[i], sampfrom=0)
#     annotation = wfdb.rdann(PATH + pre_records[i], 'atr', sampfrom=0, return_label_elements=['symbol'])

#     db2_signals.append(signals.tolist())
#     db2_anno.append(annotation.symbol)

# '''DB3
# '''

# FILE_NUM_FLAG = 2
# PATH = './data/' + DB_LIST[FILE_NUM_FLAG] + '/'
# print("[INFO] Read records file from ", PATH)
# with open(PATH + 'RECORDS') as f:
#     record_lines = f.readlines()

# pre_records = []
# for x in record_lines:
#     pre_records.append(x.strip())
# print("[RSLT]\t\t\t Export records ...")

# print("[INFO]./rdsamp commending start")
# for i in range(len(pre_records)):
#     match_list_cnt = 0
#     print("[IWIP]\t\trdsamp Converting", pre_records[i], PATH)
#     signals, _ = wfdb.rdsamp(PATH + pre_records[i], sampfrom=0)
#     annotation = wfdb.rdann(PATH + pre_records[i], 'atr', sampfrom=0, return_label_elements=['symbol'])

    # db3_signals.append(signals.tolist())
    # db3_anno.append(annotation.symbol)