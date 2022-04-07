from rbm import HIDDEN_UNITS, K_FOLD, LEARNING_RATE, VISIBLE_UNITS
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch
from RBM import RBM

VISIBLE_UNITS = 0
HIDDEN_UNITS = [64, 100]
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
            input_size = 0
            if i == 0:
                input_size = visible_units
            else:
                input_size = hidden_untis[i - 1]
            
            rbm = RBM(visible_units = VISIBLE_UNITS,
                      hidden_untis = HIDDEN_UNITS[i],
                      k = K_FOLD,
                      learning_rate = LEARNING_RATE,
                      learning_rate_decay = LEARNING_RATE_DECAY,
                      xavier_init = False,
                      increase_to_cd_k = False)
            
            self.rbm_layer.append(rbm)


     # rbm_layers = [RBM(rbn_nodes[i-1] , rbm_nodes[i],use_gpu=use_cuda) for i in range(1,len(rbm_nodes))]
        self.W_rec = [nn.Parameter(self.rbm_layers[i].W.data.clone()) for i in range(self.n_layers-1)]
        self.W_gen = [nn.Parameter(self.rbm_layers[i].W.data) for i in range(self.n_layers-1)]
        
        self.bias_rec = [nn.Parameter(self.rbm_layers[i].h_bias.data.clone()) for i in range(self.n_layers-1)]
        self.bias_gen = [nn.Parameter(self.rbm_layers[i].v_bias.data) for i in range(self.n_layers-1)]
        
        self.W_mem = nn.Parameter(self.rbm_layers[-1].W.data)
        self.v_bias_mem = nn.Parameter(self.rbm_layers[-1].v_bias.data)
        self.h_bias_mem = nn.Parameter(self.rbm_layers[-1].h_bias.data)     
        
        for i in range(self.n_layers-1):
            self.register_parameter('W_rec%i'%i, self.W_rec[i])
            self.register_parameter('W_gen%i'%i, self.W_gen[i])
            self.register_parameter('bias_rec%i'%i, self.bias_rec[i])
            self.register_parameter('bias_gen%i'%i, self.bias_gen[i])       