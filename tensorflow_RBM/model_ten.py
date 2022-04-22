from data.medain_filtering_class import ecg_filtering as ef
from tfrbm import RBM, GBRBM, BBRBM
from tfrbm import util
import tensorflow as tf
import numpy as np
import logging

def input_to_ndarray(input_arr1, input_arr2, input_arr3):
    return np.ndarray(input_arr1), np.ndarray(input_arr2), np.ndarray(input_arr3)

logging.basicConfig(level=logging.INFO)
x_train, x_cross, x_test = input_to_ndarray(ef.ecg_filtering(True, False))

dataset = tf.data.Dataset.from_tensor_slices(x_train.astype(np.float64))
