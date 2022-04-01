import os
with open('./output_db1/121.csv', 'r') as file:
    data = file.read().replace('\n', '')
    print(len(data))

with open('./output_db1/100.csv', 'r') as file:
    data = file.read().replace('\n', '')
    print(len(data))