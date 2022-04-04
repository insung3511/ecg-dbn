import os
import re

FILE_NUM_FLAG = 0
RESULT_PATH = './result_'
DB_LIST = ['ahadb', 'cudb', 'edb', 'nstdb', 'nst_old']
USER_PATH = input("[USER] Type Database category (aha, cu, esc, nst, nst_old): ")

print("[INFO] Pre-processing for make clean")

for i in range(len(DB_LIST)):
    user_path_nospace = USER_PATH.replace(" ", "")
    if (user_path_nospace == 'aha'):
        FILE_NUM_FLAG = 0
    elif (user_path_nospace == 'cu'):
        FILE_NUM_FLAG = 1
    elif (user_path_nospace == 'esc'):
        FILE_NUM_FLAG = 2
    elif (user_path_nospace == 'nst'):
        FILE_NUM_FLAG = 3
    elif (user_path_nospace == 'nst_old'):
        FILE_NUM_FLAG = 3
    else:
        print("[ERRR]\tYour typed ", USER_PATH, " but, system can not found what database it is.")
        print("[ERRR]\tSystem out.")
        exit(1)

print("[INFO] Your typed", USER_PATH, " and system detected same database from list : ", DB_LIST[FILE_NUM_FLAG])
print("[INFO] System continue ... ")

print("[INFO] Creating result direcotory...")
PATH = './' + DB_LIST[FILE_NUM_FLAG] + '/' + '1.0.0/'

if (FILE_NUM_FLAG == 3):
    PATH = DB_LIST[FILE_NUM_FLAG] + '/' + '1.0.0' + '/old/'
    nst_old_file_list = os.listdir(PATH)
    print(nst_old_file_list)
    # for i in range(0, len(nst_old_file_list)):
    #     if (nst_old_file_list[i] == "oldnstdb.txt"):
    #         nst_old_file_list.pop(i)
    #     elif (nst_old_file_list[i] == "index.html"):
    #         nst_old_file_list.pop(i)
    print(nst_old_file_list)
    for i in range(len(DB_LIST)):
        try:
            if not os.path.exists(RESULT_PATH + DB_LIST[FILE_NUM_FLAG]):
                os.makedirs(RESULT_PATH + DB_LIST[FILE_NUM_FLAG])
        except OSError:
            print("[ERRR] \t\t\t Error with creating direcotry : ", RESULT_PATH + DB_LIST[FILE_NUM_FLAG])
            print("[ERRR] \t\t\t System exit by exit(1)")
            exit(1)

    print("[INFO]./rdsamp commending start")
    for i in range(len(nst_old_file_list)):
        print("[IWIP]\t\trdsamp Converting", nst_old_file_list[i])
        #commend = './rdsamp -r ' + PATH + file_list[i] + ' -f 0 -pd -c -v > ./' + RESULT_PATH + DB_LIST[FILE_NUM_FLAG] + '/' + file_list[i] + '.csv'
        commend = 'rdsamp -r ' + PATH + nst_old_file_list[i] + ' -v -ps  -f 0 -c > ' + RESULT_PATH + DB_LIST[FILE_NUM_FLAG] + '/rdsamp_' + DB_LIST[FILE_NUM_FLAG] + nst_old_file_list[i] + '.csv'
        os.system(commend)
    print("[RSLT]\t\t\t", os.listdir(RESULT_PATH + DB_LIST[FILE_NUM_FLAG]))

    for i in range(len(nst_old_file_list)):
        if (os.path.getsize(RESULT_PATH + DB_LIST[FILE_NUM_FLAG]) < 1):
            os.remove(RESULT_PATH + DB_LIST[FILE_NUM_FLAG])
    print("[CLEN] Removed 0 byte files...")


    # changing whole record using rdann
    print("[INFO]./rdann commending start")
    for i in range(len(nst_old_file_list)):
        print("[IWIP]\t\trdsamp Converting", nst_old_file_list[i])
        # ./rdann -r 1.0.0/0201 -a atr -v -e > 0201_rdann.csv
        #commend = './rdsamp -r ' + PATH + file_list[i] + ' -f 0 -pd -c -v > ./' + RESULT_PATH + DB_LIST[FILE_NUM_FLAG] + '/' + file_list[i] + '.csv'
        commend = 'rdann -r ' + PATH + nst_old_file_list[i] + ' -a atr -f 0  -v -x > ' + RESULT_PATH + DB_LIST[FILE_NUM_FLAG] + '/rdann_' + DB_LIST[FILE_NUM_FLAG] + nst_old_file_list[i] + '.csv'
        os.system(commend)
    print("[RSLT]\t\t\t", os.listdir(RESULT_PATH + DB_LIST[FILE_NUM_FLAG]))
    print("[DONE] System out. Hungry :p")
    exit(0)

for i in range(len(DB_LIST)):
    try:
        if not os.path.exists(RESULT_PATH + DB_LIST[FILE_NUM_FLAG]):
            os.makedirs(RESULT_PATH + DB_LIST[FILE_NUM_FLAG])
    except OSError:
        print("[ERRR] \t\t\t Error with creating direcotry : ", RESULT_PATH + DB_LIST[FILE_NUM_FLAG])
        print("[ERRR] \t\t\t System exit by exit(1)")
        exit(1)

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
    print("[IWIP]\t\trdsamp Converting", pre_records[i])
    #commend = './rdsamp -r ' + PATH + file_list[i] + ' -f 0 -pd -c -v > ./' + RESULT_PATH + DB_LIST[FILE_NUM_FLAG] + '/' + file_list[i] + '.csv'
    commend = 'rdsamp -r ' + PATH + pre_records[i] + ' -v -ps  -f 0 -c > ' + RESULT_PATH + DB_LIST[FILE_NUM_FLAG] + '/rdsamp_' + DB_LIST[FILE_NUM_FLAG] + pre_records[i] + '.csv'
    os.system(commend)
print("[RSLT]\t\t\t", os.listdir(RESULT_PATH + DB_LIST[FILE_NUM_FLAG]))

# changing whole record using rdann
print("[INFO]./rdann commending start")
for i in range(len(pre_records)):
    print("[IWIP]\t\trdsamp Converting", pre_records[i])
    # ./rdann -r 1.0.0/0201 -a atr -v -e > 0201_rdann.csv
    #commend = './rdsamp -r ' + PATH + file_list[i] + ' -f 0 -pd -c -v > ./' + RESULT_PATH + DB_LIST[FILE_NUM_FLAG] + '/' + file_list[i] + '.csv'
    commend = 'rdann -r ' + PATH + pre_records[i] + ' -a atr -f 0  -v -x > ' + RESULT_PATH + DB_LIST[FILE_NUM_FLAG] + '/rdann_' + DB_LIST[FILE_NUM_FLAG] + pre_records[i] + '.csv'
    os.system(commend)
print("[RSLT]\t\t\t", os.listdir(RESULT_PATH + DB_LIST[FILE_NUM_FLAG]))

error_file_count = 0

# Check all file is okay
print("[CHEK] \tChecking not converted file")
for i in range(len(pre_records)):
    rdann_check_stand = RESULT_PATH + DB_LIST[FILE_NUM_FLAG] + '/rdann_' + DB_LIST[FILE_NUM_FLAG] + pre_records[i] + '.csv'
    rdsamp_check_stand = RESULT_PATH + DB_LIST[FILE_NUM_FLAG] + '/rdsamp_' + DB_LIST[FILE_NUM_FLAG] + pre_records[i] + '.csv'
    check_list = os.listdir(RESULT_PATH + DB_LIST[FILE_NUM_FLAG] + '/')
    
    if (check_list[i] is rdann_check_stand or check_list[i] is rdsamp_check_stand):
        continue
    else:
        print("[ERRR] {0}th converted record file is/are missing.".format(i))
        error_file_count += 1

print("[SUMM] \t\aMissing files number is ", error_file_count, "of ", len(pre_records))