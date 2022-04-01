import os

file = open("./db1/RECORDS", "r")
msg = file.readlines()

for i in range(len(msg)):
    record_name = str(msg[i].rstrip())
    print("{}th RECORDS TO CSV".format(record_name))
    print("\t\t\t\tOutput dir : ./output_db1/", record_name, ".csv")

    record_file = "./db1/" + record_name
    command = "./rdsamp -v -t 30:00 -c -r " + record_file + " > ./output_db1/" + record_name + ".csv"
    os.system(command)

file.close()
    