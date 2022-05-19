# load mit data

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from scipy import io
import pandas as pd
import numpy as np

# PATH = "/Users/bahk_insung/Documents/Github/ecg-dbn/data/mit.mat"
PATH = "C:/Users/HILAB_Labtop_02/Desktop/insung/ecg-dbn/data/mit.mat"

def to_categorical(y, num_classes):
    return np.eye(num_classes, dtype='uint8')[y]

mit = io.loadmat(PATH)
mit = mit['mit']
df_mit = pd.DataFrame(data = mit)

Y = np.array(df_mit[428].values).astype(np.int8)
X = np.array(df_mit[list(range(428))].values)[..., np.newaxis]

oneHot = LabelEncoder()
oneHot.fit(Y)
Y = oneHot.transform(Y)

X = X.reshape(-1, 428, 1)
Y = to_categorical(Y, 5)

print(df_mit[428].value_counts(), '\n')

X_train, X_val, Y_train, Y_val = train_test_split(X, Y,
                                                test_size=0.3,
                                                random_state=42,
                                                stratify=Y)

print("X_train shape: ", X_train.shape)
print("Y_train shape: ", Y_train.shape, '\n')
print("X_val shape: ", X_val.shape)
print("Y_val shape: ", Y_val.shape)