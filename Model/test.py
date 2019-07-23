import numpy as np
from sklearn.model_selection import KFold
import tensorflow as tf

np.random.seed(77)
'''
X = np.array([1, 2])
Y = np.array([2, 3])

kf = KFold(n_splits=4, shuffle=True)
for train_index, test_index in kf.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]
    X_train = X[train_index[0:3]]
    print(np.shape(X_train))

a = np.append(X.reshape([1,2]), Y.reshape([1,2]), axis=0)
print(a)

sess =tf.Session()
print(sess.run([tf.unstack(np.array([[0,1,2],[3,4,5]]),axis=0)]))
'''
print(0b1111& 0b1000)