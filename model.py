from torch import int64
from tfrbm.util import xavier_init, sample_bernoulli, sample_gaussian
import data.medain_filtering_class as mf
from matplotlib.pyplot import axis
from typing import Dict, List
import tensorflow as tf
from tfrbm import RBM
import pandas as pd
import numpy as np
import logging
import abc

class BBRBM(RBM):
    def __init__(self, *args, **kwargs):
        """
        Initializes Bernoulli-Bernoulli RBM.

        :param n_visible: number of visible neurons (input size)
        :param n_hidden: number of hidden neurons
        :param learning_rate: learning rate (default: 0.01)
        :param momentum: momentum (default: 0.95)
        :param xavier_const: constant used to initialize weights (default: 1.0)
        """
        super().__init__(*args, **kwargs)

    def step(self, x: tf.Tensor) -> tf.Tensor:
        hidden_p = tf.nn.sigmoid(tf.matmul(x, self.w) + self.hidden_bias)
        visible_recon_p = tf.nn.sigmoid(tf.matmul(sample_bernoulli(hidden_p), tf.transpose(self.w)) + self.visible_bias)
        hidden_recon_p = tf.nn.sigmoid(tf.matmul(visible_recon_p, self.w) + self.hidden_bias)

        positive_grad = tf.matmul(tf.transpose(x), hidden_p)
        negative_grad = tf.matmul(tf.transpose(visible_recon_p), hidden_recon_p)

        self.delta_w = self._apply_momentum(
            self.delta_w,
            positive_grad - negative_grad
        )
        self.delta_visible_bias = self._apply_momentum(
            self.delta_visible_bias,
            tf.reduce_mean(x - visible_recon_p, 0)
        )
        self.delta_hidden_bias = self._apply_momentum(
            self.delta_hidden_bias,
            tf.reduce_mean(hidden_p - hidden_recon_p, 0)
        )

        self.w.assign_add(self.delta_w)
        self.visible_bias.assign_add(self.delta_visible_bias)
        self.hidden_bias.assign_add(self.delta_hidden_bias)

        return tf.reduce_mean(tf.square(x - visible_recon_p))

class GBRBM(BBRBM):
    def __init__(
            self,
            n_visible: int,
            n_hidden: int,
            sample_visible: bool = False,
            sigma: float = 1.0,
            **kwargs
    ):
        """
        Initializes Gaussian-Bernoulli RBM.

        :param n_visible: number of visible neurons (input size)
        :param n_hidden: number of hidden neurons
        :param sample_visible: if reconstructed state should be sampled from the Gaussian distribution (default: False)
        :param sigma: standard deviation of this distribution, does nothing when sample_visible = False (default: 1.0)
        :param learning_rate: learning rate (default: 0.01)
        :param momentum: momentum (default: 0.95)
        :param xavier_const: constant used to initialize weights (default: 1.0)
        """
        self.sample_visible = sample_visible
        self.sigma = sigma

        super().__init__(n_visible, n_hidden, **kwargs)

    def step(self, x: tf.Tensor) -> tf.Tensor:
        hidden_p = tf.nn.sigmoid(tf.matmul(x, self.w) + self.hidden_bias)
        visible_recon_p = tf.matmul(sample_bernoulli(hidden_p), tf.transpose(self.w)) + self.visible_bias

        if self.sample_visible:
            visible_recon_p = sample_gaussian(visible_recon_p, self.sigma)

        hidden_recon_p = tf.nn.sigmoid(tf.matmul(visible_recon_p, self.w) + self.hidden_bias)

        positive_grad = tf.matmul(tf.transpose(x), hidden_p)
        negative_grad = tf.matmul(tf.transpose(visible_recon_p), hidden_recon_p)

        self.delta_w = self._apply_momentum(
            self.delta_w,
            positive_grad - negative_grad
        )
        self.delta_visible_bias = self._apply_momentum(
            self.delta_visible_bias,
            tf.reduce_mean(x - visible_recon_p, 0)
        )
        self.delta_hidden_bias = self._apply_momentum(
            self.delta_hidden_bias,
            tf.reduce_mean(hidden_p - hidden_recon_p, 0)
        )

        self.w.assign_add(self.delta_w)
        self.visible_bias.assign_add(self.delta_visible_bias)
        self.hidden_bias.assign_add(self.delta_hidden_bias)

        return tf.reduce_mean(tf.square(x - visible_recon_p))

'''START OF INDEPENDENT CODE'''

# def transform_dataset(model, dataset):
#     transformed_batches = []
    
#     for batch in dataset.batch(2048):
#         transformed_batches.append(model.compute_hidden(batch))
#     return tf.data.Dataset.from_tensor_slices(tf.concat(transformed_batches, axis=0))

if __name__ == '__main__':
    BATCH_SIZE = 10
    EPOCH = 100
    LEARNING_RATE = 0.2
    ANNEALING_RATE = 0.999
    
    print("[MODL] Model main code is starting....")
    logging.basicConfig(level=logging.INFO)

    print("[INFO] Read train data, cross-vaildation data and test data from median filtering code")

    
    dataset_db1, dataset_db2, dataset_db3 = mf.ecg_filtering(True)
    dataset_for_train = mf.list_to_list(dataset_db1 + dataset_db2)
    dataset_for_test = mf.list_to_list(dataset_db2 + dataset_db3)
    
    # dataset_for_train = [int(i) * 1000000000000000000 for i in dataset_for_train]
    print(dataset_for_train)


    # dataset_for_train = [100*(dataset_for_train[i]) for i in dataset_for_train]
    dataset_for_train = tf.convert_to_tensor(dataset_for_train)

    # train_dataset = tf.data.Dataset.from_tensor_slices(dataset_for_train)
    # print(train_dataset)
    # dataset = dataset.shuffle(1024, reshuffle_each_iteration=True)

    bbrbm_1 = BBRBM(n_visible=180, n_hidden=80)
    bbrbm_2 = BBRBM(n_visible=200, n_hidden=100)
    bbrbm_3 = BBRBM(n_visible=250, n_hidden=120)

    gbrbm_1 = GBRBM(n_visible=180, n_hidden=80)
    gbrbm_2 = GBRBM(n_visible=200, n_hidden=100)
    gbrbm_3 = GBRBM(n_visible=250, n_hidden=120)

    epchoes = 100
    batch_size = 10

    # first
    bbrbm_1.fit(dataset_for_train)
    # bbrbm_dataset_2 = transform_dataset(bbrbm_1, train_feature)

    # gbrbm_1.fit(dataset, epoches=epchoes, batch_size=batch_size)
    # gbrbm_dataset_2 = transform_dataset(gbrbm_1, dataset)

    # # second
    # bbrbm_2.fit(dataset, epoches=epchoes, batch_size=batch_size)
    # bbrbm_dataset_3 = transform_dataset(bbrbm_2, dataset)

    # gbrbm_2.fit(dataset, epoches=epchoes, batch_size=batch_size)
    # gbrbm_dataset_3 = transform_dataset(gbrbm_2, dataset)

    # # third
    # bbrbm_3.fit(dataset, epoches=epchoes, batch_size=batch_size)
    # gbrbm_3.fit(dataset, epoches=epchoes, batch_size=batch_size)

    # def bbrbm_encode(x):
    #     hidden_1 = bbrbm_1.compute_hidden(x)
    #     hidden_2 = bbrbm_2.compute_hidden(hidden_1)
    #     hidden_3 = bbrbm_3.compute_hidden(hidden_2)

    #     return hidden_3

    # def gbrbm_encode(x):
    #     hidden_1 = gbrbm_1.compute_hidden(x)
    #     hidden_2 = gbrbm_2.compute_hidden(hidden_1)
    #     hidden_3 = gbrbm_3.compute_hidden(hidden_2)
        
    #     return hidden_3

    # dataset_test = tf.data.Dataset.from_tensor_slices(x_test.reshape(-1, 28 * 28))