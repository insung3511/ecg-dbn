# load mit data

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from scipy import io
import pandas as pd
import numpy as np

# PATH = "/Users/bahk_insung/Documents/Github/ecg-dbn/data/mit.mat"
PATH = ""

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
# 0 = N
# 1 = Q
# 2 = S
# 3 = V
# 4 = F

X_train, X_val, Y_train, Y_val = train_test_split(X, Y,
                                                test_size=0.3,
                                                random_state=42,
                                                stratify=Y)

print("X_train shape: ", X_train.shape)
print("Y_train shape: ", Y_train.shape, '\n')
print("X_val shape: ", X_val.shape)
print("Y_val shape: ", Y_val.shape)