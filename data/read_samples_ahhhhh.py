from scipy.signal import butter, filtfilt, medfilt
import matplotlib.pyplot as plt
import itertools
import wfdb

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
#    Median filter   #
#######################
def median_filter(data, fs):
    return medfilt(data, fs)

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

    db1_fs = list()
    db2_fs = list()
    db3_fs = list()
    
    db1_butter = list()
    db2_butter = list()
    db3_butter = list()

    print("[INFO] DB1 Filtering...")
    for i in range(len(db1_signals)):
        db1_fs.append(median_filter(median_filter(db1_signals, 199), 599))
    for i in range(len(db1_fs)):
        db1_butter.append(butter_lowpass(3.667, (db1_fs[i])))
    
    print("[INFO] DB2 Filtering...")
    for i in range(len(db2_signals)):
        db2_fs.append(median_filter(median_filter(db2_signals, 199), 599))
    for i in range(len(db2_fs)):
        db2_butter.append(butter_lowpass(3.667, (db2_fs[i])))

    print("[INFO] DB3 Filtering...")
    for i in range(len(db3_signals)):
        db3_fs.append(median_filter(median_filter(db3_signals, 199), 599))
    for i in range(len(db3_fs)):
        db3_butter.append(butter_lowpass(3.667, (db3_fs[i])))

    print("DB1 butter size : {}, DB1 Anno size : {}\n".format(len(db1_butter), len(db1_anno)),    \
          "DB2 butter size : {}, DB2 Anno size : {}\n".format(len(db2_butter), len(db2_anno)),    \
          "DB3 butter size : {}, DB3 Anno size : {}\n".format(len(db3_butter), len(db3_anno)))

    return db1_butter, db1_anno, db2_butter, db2_anno, db3_butter, db3_anno

# a, a1, b, b1, c, c1 = return_list()
# wfdb.plot_items(a, a1)
# plt.plot(a, range(0, len(a)))