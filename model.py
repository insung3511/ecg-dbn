import data.medain_filtering_class as mf
from numpy import float64
import tensorflow as tf
import numpy as np
import logging

class Batch(object):
    def __init__(self, X, batch_size, y=None):
        self.X = X
        self.batch_size = batch_size
        self.size = X.shape[0]
        self.y = y

    def get_batch(self):
        indices = np.random.choice(range(self.size), self.batch_size)
        if self.y is None:
            return self.X[indices, :]
        return self.X[indices, :], self.y[indices, :]

    def get_tensor_batch(self):
        bt = self.get_batch()
        return tf.stack(bt)

class RBM:
    def __init__(self, n_visible, n_hidden, lr, epochs, batch_size=None):
        '''
        Initialize a model for an RBM with one layer of hidden units
        :param n_visible: Number of visible nodes
        :param n_hidden: Number of hidden nodes
        :param lr: Learning rate for the CD algorithm
        :param epochs: Number of iterations to run the algorithm for
        :param batch_size: Split the training data
        '''
        self.n_hidden = n_hidden
        self.n_visible = n_visible
        self.lr = lr

        self.epochs = epochs
        self.batch_size = batch_size

def get_probabilities(layer, weights, val, bias):
    '''
    Find the probabilities associated with layer specified
    :param layer: Hidden layer or visible layer, specified as string
    :param weights: Tensorflow placeholder for weight matrix
    :param val: Input units, hidden or visible as binary or float
    :param bias: Bias associated with the computation, opposite of the input
    :return: A tensor of probabilities associated with the layer specified
    '''
    if layer == 'hidden':
        with tf.name_scope("Hidden_Probabilities"):
            return tf.nn.sigmoid(tf.matmul(val, weights) + bias)
    elif layer == 'visible':
        with tf.name_scope("Visible_Probabilities"):
            return tf.nn.sigmoid(tf.matmul(val, tf.transpose(weights)) + bias)


def get_gaussian_probabilities(layer, weights, val, bias):
    '''
    Find the probabilities associated with layer specified
    :param layer: Hidden layer or visible layer, specified as string
    :param weights: Tensorflow placeholder for weight matrix
    :param val: Input units, hidden or visible as binary or float
    :param bias: Bias associated with the computation, opposite of the input
    :return: A tensor of probabilities associated with the layer specified
    '''
    if layer == 'hidden':
        with tf.name_scope("Hidden_Probabilities"):
            return tf.matmul(val, weights) + bias
    elif layer == 'visible':
        with tf.name_scope("Visible_Probabilities"):
            return tf.nn.sigmoid(tf.matmul(val, tf.transpose(weights)) + bias)


def gibbs(steps, v, hb, vb, W):
    '''
    Use the Gibbs sampler for a network of hidden and visible units.
    :param steps: Number of steps to run the algorithm
    :param v: Input data
    :param hb: Hidden Bias
    :param vb: Visible bias
    :param W: Weight matrix
    :return: Returns a sampled version of the input
    '''
    with tf.name_scope("Gibbs_sampling"):
        for i in range(steps):
            hidden_p = get_probabilities('hidden', W, v, hb)
            h = sample(hidden_p)

            visible_p = get_probabilities('visible', W, h, vb)
            v = visible_p
            #v = sample(visible_p)
        return visible_p


def gibbs_gaussian(steps, v, hb, vb, W):
    '''
    Use the Gibbs sampler for a network of hidden and visible units.
    :param steps: Number of steps to run the algorithm
    :param v: Input data
    :param hb: Hidden Bias
    :param vb: Visible bias
    :param W: Weight matrix
    :return: Returns a sampled version of the input
    '''
    with tf.name_scope("Gibbs_sampling"):
        for i in range(steps):
            hidden_p = get_gaussian_probabilities('hidden', W, v, hb)
            poshidstates = sample_gaussian(hidden_p)

            visible_p = get_gaussian_probabilities('visible', W, poshidstates, vb)
            #v = sample_gaussian(visible_p)
        return visible_p


def sample(probabilities):
    '''
    Sample a tensor based on the probabilities
    :param probabilities: A tensor of probabilities given by 'rbm.get_probabilities'
    :return: A sampled sampled tensor
    '''
    return tf.floor(probabilities + tf.random_uniform(tf.shape(probabilities), 0, 1))


def sample_gaussian(probabilities, stddev=1):
    '''
        Create a tensor based on the probabilities by adding gaussian noise from the input
        :param probabilities: A tensor of probabilities given by 'rbm.get_probabilities'
        :return: The addition of noise to the original probabilities
        '''
    return tf.add(probabilities, tf.random_normal(tf.shape(probabilities), mean=0.0, stddev=stddev))

def free_energy(v, weights, vbias, hbias):
    '''
    Compute the free energy for  a visible state
    :param v:
    :param weights:
    :param vbias:
    :param hbias:
    :return:
    '''
    vbias_term = tf.matmul(v, tf.transpose(vbias))
    x_b = tf.matmul(v, weights) + hbias
    hidden_term = tf.reduce_sum(tf.log(1 + tf.exp(x_b)))
    return - hidden_term - vbias_term

'''
####################
####################
####################
####################
####################
####################
####################
####################
####################
####################
'''

class Batch(object):
    def __init__(self, X, batch_size, y=None):
        self.X = X
        self.batch_size = batch_size
        self.size = X.shape[0]
        self.y = y

    def get_batch(self):
        indices = np.random.choice(range(self.size), self.batch_size)
        if self.y is None:
            return self.X[indices, :]
        return self.X[indices, :], self.y[indices, :]

    def get_tensor_batch(self):
        bt = self.get_batch()
        return tf.stack(bt)


if __name__ == '__main__':
    BATCH_SIZE = 10
    EPOCH = 100
    LEARNING_RATE = 0.2
    ANNEALING_RATE = 0.999
    
    print("[MODL] Model main code is starting....")
    logging.basicConfig(level=logging.INFO)

    print("[INFO] Read train data, cross-vaildation data and test data from median filtering code")
    dataset_db1, dataset_db2, dataset_db3 = mf.ecg_filtering(True)
    train_dataset = tuple(mf.list_to_list(dataset_db1 + dataset_db2))
    test_dataset = tuple(mf.list_to_list(dataset_db2 + dataset_db3))

    train_data = np.array(train_dataset)
    rbm_model = RBM(n_visible=180, n_hidden=80, lr = tf.compat.v1.constant(train_data, dtype=tf.float64), epochs=100, batch_size=10)
    batch = Batch(X = train_data, batch_size = 10)

    #v = tf.placeholder(tf.float64, shape=[None, rbm_model.n_visible], name = "visible_layer")
    v = tf.constant(train_data, name="visible_layer", dtype=tf.float64, shape=[None, rbm_model.n_visible])
    size = tf.cast(tf.shape(v)[0], tf.float64)

    with tf.name_scope('Weights'):
        W = tf.Variable(tf.random_normal([rbm_model.n_visible, rbm_model.n_hidden], mean=0., stddev=4 * np.sqrt(6. / (rbm_model.n_visible + rbm_model.n_hidden))), name="weights")
        tf.summary.histogram('weights',W)

    # with tf.name_scope('Weights'):
    #     W = tf.Variable(tf.random_normal)
    # data = 