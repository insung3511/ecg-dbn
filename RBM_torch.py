import torch.distributions.mixture_same_family as mix
from sklearn.linear_model import LogisticRegression
import data.medain_filtering_class as mf
import numpy as np
import torch

class RBM():
    def __init__(self, num_visible, num_hidden, learning_rate=1e-3, momentum_coefficient=0.2, weight_decay=1e-4):
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.learning_rate = learning_rate
        self.momentum_coefficient = momentum_coefficient
        self.weight_decay = weight_decay

        self.weights = torch.randn(num_visible, num_hidden) * 0.1
        self.visible_bias = torch.ones(num_visible) * 0.5
        self.hidden_bias = torch.zeros(num_hidden)

        self.weights_momentum = torch.zeros(num_visible, num_hidden)
        self.visible_bias_momentum = torch.zeros(num_visible)
        self.hidden_bias_momentum = torch.zeros(num_hidden)

    # visible layer to hidden layer
    def sample_hidden(self, visible_probabilities):
        # hidden_activations = torch.matmul(visible_probabilities, None) + self.hidden_bias
        hidden_activations = self.hidden_bias + mix.MixtureSameFamily()
        hidden_probabilities = self._sigmoid(hidden_activations)
        return hidden_probabilities

    # hidden layer to visible layer
    def sample_visible(self, hidden_probabilities):
        visible_activations = torch.matmul(hidden_probabilities, self.weights.t()) + self.visible_bias
        visible_probabilities = self._sigmoid(visible_activations)
        return visible_probabilities

    def contrastive_divergence(self, input_data):
        # Positive phase -> Forward pass propagation
        positive_hidden_probabilities = self.sample_hidden(input_data)
        positive_hidden_activations = (positive_hidden_probabilities >= self._random_probabilities(self.num_hidden)).float()
        positive_associations = torch.matmul(input_data.t(), positive_hidden_activations)

        # Negative phase -> Backward pass propagation
        hidden_activations = positive_hidden_activations

        for step in range(self.k):
            visible_probabilities = self.sample_visible(hidden_activations)
            hidden_probabilities = self.sample_hidden(visible_probabilities)
            hidden_activations = (hidden_probabilities >= self._random_probabilities(self.num_hidden)).float()

        negative_visible_probabilities = visible_probabilities
        negative_hidden_probabilities = hidden_probabilities

        negative_associations = torch.matmul(negative_visible_probabilities.t(), negative_hidden_probabilities)

        # Update parameters
        self.weights_momentum *= self.momentum_coefficient
        self.weights_momentum += (positive_associations - negative_associations)

        self.visible_bias_momentum *= self.momentum_coefficient
        self.visible_bias_momentum += torch.sum(input_data - negative_visible_probabilities, dim=0)

        self.hidden_bias_momentum *= self.momentum_coefficient
        self.hidden_bias_momentum += torch.sum(positive_hidden_probabilities - negative_hidden_probabilities, dim=0)

        batch_size = input_data.size(0)

        self.weights += self.weights_momentum * self.learning_rate / batch_size
        self.visible_bias += self.visible_bias_momentum * self.learning_rate / batch_size
        self.hidden_bias += self.hidden_bias_momentum * self.learning_rate / batch_size

        self.weights -= self.weights * self.weight_decay  # L2 weight decay

        # Compute reconstruction error
        error = torch.sum((input_data - negative_visible_probabilities)**2)

        return error

    def _sigmoid(self, x):
        return 1 / (1 + torch.exp(-x))

    def _random_probabilities(self, num):
        random_probabilities = torch.rand(num)

        if self.use_cuda:
            random_probabilities = random_probabilities.cuda()

        return random_probabilities


def one_d2n_d(n_hid, og_mat):
    column = int(len(og_mat) / n_hid)
    row = int(n_hid)
    print(row, column)
    
    mat = np.matrix(np.mat(og_mat))
    mat.shape = (row, column)
    return mat

########## CONFIGURATION ##########
BATCH_SIZE = 10
VISIBLE_UNITS = 180
HIDDEN_UNITS = 80
EPOCHS = 100

dataset_db1, dataset_db2, dataset_db3 = mf.ecg_filtering(True)
og_train_dataset = (mf.list_to_list(dataset_db1 + dataset_db2))
og_test_dataset = (mf.list_to_list(dataset_db2 + dataset_db3))

train_dataset = np.array(og_train_dataset, dtype=np.float64)
test_dataset = np.array(og_test_dataset, dtype=np.float64)

print("[INFO]")
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

print('Training RBM...')

rbm = RBM(VISIBLE_UNITS, HIDDEN_UNITS)

for epoch in range(EPOCHS):
    epoch_error = 0.0

    for batch in train_loader:
        #batch = batch.view(len(batch), VISIBLE_UNITS)  # flatten input data
        print(batch)
        batch_error = rbm.contrastive_divergence(batch)
        epoch_error += batch_error
    print('Epoch Error (epoch=%d): %.4f' % (epoch, epoch_error))


########## EXTRACT FEATURES ##########
print('Extracting features...')

train_features = np.zeros((len(train_dataset), HIDDEN_UNITS))
train_labels = np.zeros(len(train_dataset))
test_features = np.zeros((len(test_dataset), HIDDEN_UNITS))
test_labels = np.zeros(len(test_dataset))

for i, (batch, labels) in enumerate(train_loader):
    batch = batch.view(len(batch), VISIBLE_UNITS)  # flatten input data
    train_features[i*BATCH_SIZE:i*BATCH_SIZE+len(batch)] = rbm.sample_hidden(batch).cpu().numpy()
    train_labels[i*BATCH_SIZE:i*BATCH_SIZE+len(batch)] = labels.numpy()

for i, (batch, labels) in enumerate(test_loader):
    batch = batch.view(len(batch), VISIBLE_UNITS)  # flatten input data
    test_features[i*BATCH_SIZE:i*BATCH_SIZE+len(batch)] = rbm.sample_hidden(batch).cpu().numpy()
    test_labels[i*BATCH_SIZE:i*BATCH_SIZE+len(batch)] = labels.numpy()

########## CLASSIFICATION ##########
print('Classifying...')

clf = LogisticRegression()
clf.fit(train_features, train_labels)
predictions = clf.predict(test_features)

print('Result: %d/%d' % (sum(predictions == test_labels), test_labels.shape[0]))

