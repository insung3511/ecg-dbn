import pydeep.base.numpyextension as npex
import pydeep.rbm.estimator as estimator
import pydeep.rbm.trainer as trainer
import pydeep.rbm.model as model
import numpy as np

from data.so_again_median import medain_filtering as mf
import pydeep.misc.visualization as vis
import pydeep.misc.measuring as mea
import pydeep.preprocessing as pre
import pydeep.misc.io as io

update_offset = 0.01
filpped = False

np.random.seed(42)
mf.ecg_filtering(mf.data_path_flex(True))
whitened_data = zca.project(data)

# time(sample, x-axis) and mV(y-axis)
train_data = whitened_data[0:np.int32(whitened_data.shape[0] / 2.0), :]
test_data = whitened_data[np.int32(whitened_data.shape[0] / 2.0):
                          whitened_data.shape[0], :]

h1 = 2
h2 = 2
v1 = whitened_data.shape[1]
v2 = 1