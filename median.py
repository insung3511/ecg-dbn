from wfdb import processing as wp

import itertools
import wfdb

PATH = "./data/db1/"

print("[INFO] Read records file from ", PATH)
with open(PATH + 'RECORDS') as f:
    record_lines = f.readlines()

pre_records = []
for x in record_lines:
    pre_records.append(x.strip())
print("[RSLT]\t\t\t Export records ...")

for i in range(len(pre_records)):
    print("[IWIP]\t\trdsamp Converting", pre_records[i])
    signals, _ = wfdb.rdsamp(PATH + pre_records[i], sampfrom=0, channels=[0])

    wfdb_sig = str(wp.sigavg(PATH + pre_records[i], 'atr', ann_type=['N', 'S', 'V', 'F']))
    print(wfdb_sig.strip())

ex_list = list(itertools.chain(*signals.tolist()))