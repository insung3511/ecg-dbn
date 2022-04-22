from data import medain_filtering_class as mf
from tfrbm import RBM, GBRBM, BBRBM
from tfrbm import util
import tensorflow as tf
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
x_train, x_cross, x_test = np.ndarray(mf.ecg_filtering(True))
