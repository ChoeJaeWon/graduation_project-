import numpy as np
from sklearn.model_selection import KFold
import tensorflow as tf

np.random.seed(77)

X = np.array([1, 2,3, 4])
Y = np.array([2, 3])

kf = KFold(n_splits=4, shuffle=True)
for train_index, test_index in kf.split(X):
    for _ in range(4):
        print("TRAIN:", train_index, "TEST:", test_index)
        np.random.shuffle(train_index)




#print(0b1111& 0b1000)