import os

db1 = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230]
db2 = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234]

path_db1 = "./output_db1/"
dir_list_db1 = os.listdir(path_db1)

path_db2 = "./output_db2/"
dir_list_db2 = os.listdir(path_db2)

for i in range(0, 20):
    print("Moving {}th Record output".format(i))
    command_msg = "mv output_db2/" + str(db1[i]) + ".csv" + " ./final_db2/"
    os.system(command_msg)
