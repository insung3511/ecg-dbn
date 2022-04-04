from rbm import HIDDEN_UNITS, K_FOLD, LEARNING_RATE, VISIBLE_UNITS
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn
import torch
from RBM import RBM

VISIBLE_UNITS = 0
HIDDEN_UNITS = 0
K_FOLD = 5
LEARNING_RATE = 1e-5
LEARNING_RATE_DECAY = False

class DBN(nn.Module):
    def __init__(self,
                 visible_units = VISIBLE_UNITS,
                 hidden_untis = HIDDEN_UNITS,
                 k = K_FOLD,
                 learning_rate = LEARNING_RATE
                ):
        super(DBN, self).__init__()
        self.n_layer = len(hidden_untis)
        self.rbm_layer = []
        self.rbm_nodes = []

        for i in range(self.n_layer):
            

