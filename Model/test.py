import numpy as np
from sklearn.model_selection import KFold
import tensorflow as tf

np.random.seed(77)
'''
X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
y = np.array([1, 2, 3, 4])
kf = KFold(n_splits=4, shuffle=True)
for train_index, test_index in kf.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    X_train = X[train_index[0:3]]
    print(np.shape(X_train))

print(np.shape(np.append(X_train, X_train, axis=0)))
'''
sess =tf.Session()
print(sess.run([tf.unstack(np.array([[0,1,2],[3,4,5]]),axis=0)]))