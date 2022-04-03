import os

print("[IMPO] Start read and convert")

FILE_NUM_FLAG = 0
RESULT_PATH = './result_'
DB_LIST = ['aha', 'cu', 'esc', 'nst']
USER_PATH = input("[USER] Type Database category (ex aha, cu, esc, nst): ")
print("[INFO] Pre-processing for make clean")

for i in range(len(DB_LIST)):
    if DB_LIST[i] == USER_PATH:
        FILE_NUM_FLAG = i

PATH = './' + DB_LIST[FILE_NUM_FLAG] + '/'

file_list = os.listdir(PATH)
for i in range(len(file_list) - 1):
    if (file_list[i] == '.DS_Store' or
        'ANNOTATORS' or
        'RECORDS' or
        'RECORDS-development-set' or
        'RECORDS-test-set' or
        'index.html'):
        file_list.pop(i)
    file_list_len = int(len(file_list[i]))

    if FILE_NUM_FLAG == 0:      # AHA DB
        file_list[i] = str(file_list[i])[:4]
    elif FILE_NUM_FLAG == 1:    # CU DB
        file_list[i] = str(file_list[i])[:4]
    elif FILE_NUM_FLAG == 2:    # ESC DB
        file_list[i] = str(file_list[i])[:6]
    elif FILE_NUM_FLAG == 3:    # NST DB
        file_list[i] = str(file_list[i])[:7]
    else:
        print("[ERRR] \t\t\tIssue with matching DB_LIST")
        print("[ERRR] \t\t\t System exit by exit(1)")
        exit(1)

print("[INFO] Your typed", USER_PATH, " and system detected same database from list : ", DB_LIST[FILE_NUM_FLAG])
print("[INFO] System continue ... ")

for i in range(len(DB_LIST)):
    try:
        if not os.path.exists(RESULT_PATH + DB_LIST[FILE_NUM_FLAG]):
            os.makedirs(RESULT_PATH + DB_LIST[FILE_NUM_FLAG])
    except OSError:
        print("[ERRR] \t\t\t Error with creating direcotry : ", RESULT_PATH + DB_LIST[FILE_NUM_FLAG])
        print("[ERRR] \t\t\t System exit by exit(1)")
        exit(1)

# Converting code
file_list = list(set(file_list))
print(file_list)
print(FILE_NUM_FLAG)

# changing whole record using rdsamp
print("[INFO]./rdsamp commending start")
for i in range(len(file_list)):
    print("[IWIP]\t\trdsamp Converting", file_list[i])
    #commend = './rdsamp -r ' + PATH + file_list[i] + ' -f 0 -pd -c -v > ./' + RESULT_PATH + DB_LIST[FILE_NUM_FLAG] + '/' + file_list[i] + '.csv'
    commend = './rdsamp -r ' + PATH + file_list[i] + ' -H -f 0 -pS -c -v > ' + RESULT_PATH + DB_LIST[FILE_NUM_FLAG] + '/rdsamp_' + DB_LIST[FILE_NUM_FLAG] + file_list[i] + '.csv'
    os.system(commend)
print("[RSLT]\t\t\t", os.listdir(RESULT_PATH + DB_LIST[FILE_NUM_FLAG]))

# changing whole record using rdann
print("[INFO]./rdann commending start")
for i in range(len(file_list)):
    print("[IWIP]\t\trdsamp Converting", file_list[i])
    # ./rdann -r 1.0.0/0201 -a atr -v -e > 0201_rdann.csv
    #commend = './rdsamp -r ' + PATH + file_list[i] + ' -f 0 -pd -c -v > ./' + RESULT_PATH + DB_LIST[FILE_NUM_FLAG] + '/' + file_list[i] + '.csv'
    commend = './rdann -a atr -r ' + PATH + file_list[i] + ' -f 0 -e -v > ' + RESULT_PATH + DB_LIST[FILE_NUM_FLAG] + '/rdann_' + DB_LIST[FILE_NUM_FLAG] + file_list[i] + '.csv'
    os.system(commend)
print("[RSLT]\t\t\t", os.listdir(RESULT_PATH + DB_LIST[FILE_NUM_FLAG]))

error_file_count = 0
# Check all file is okay

print("[CHEK] \tChecking not converted file")
for i in range(len(file_list)):
    rdann_check_stand = RESULT_PATH + DB_LIST[FILE_NUM_FLAG] + '/rdann_' + DB_LIST[FILE_NUM_FLAG] + file_list[i] + '.csv'
    rdsamp_check_stand = RESULT_PATH + DB_LIST[FILE_NUM_FLAG] + '/rdsamp_' + DB_LIST[FILE_NUM_FLAG] + file_list[i] + '.csv'
    check_list = os.listdir(RESULT_PATH + DB_LIST[FILE_NUM_FLAG] + '/')
    
    if (check_list[i] is rdann_check_stand or check_list[i] is rdsamp_check_stand):
        continue
    else:
        print("[ERRR] {0}th converted record file is/are missing.".format(i))
        error_file_count += 1

print("[SUMM] \t\aMissing files number is ", error_file_count, "of ", len(file_list))