import numpy as np
from sklearn.model_selection import KFold
import tensorflow as tf
from module import *

'''
----------------------------코드 설명----------------------------
-C-
1.FC에 해당하는 코드로
fc를 구현함
----------------------------고려 사항----------------------------


'''
from module import *
import os

#에러계산식
def MAE(y_test, y_pred):
    y_test_orig = y_test * (SPEED_MAX - SPEED_MIN + 1e-7) + SPEED_MIN
    y_pred_orig = y_pred * (SPEED_MAX - SPEED_MIN + 1e-7) + SPEED_MIN
    return np.mean(np.abs((y_test_orig - y_pred_orig)))
def MSE(y_test, y_pred):
    y_test_orig = y_test * (SPEED_MAX - SPEED_MIN + 1e-7) + SPEED_MIN
    y_pred_orig = y_pred * (SPEED_MAX - SPEED_MIN + 1e-7) + SPEED_MIN
    return np.mean(np.square((y_test_orig - y_pred_orig)))
def MAPE(y_test, y_pred):
    y_test_orig = y_test * (SPEED_MAX - SPEED_MIN + 1e-7) + SPEED_MIN
    y_pred_orig = y_pred * (SPEED_MAX - SPEED_MIN + 1e-7) + SPEED_MIN
    return np.mean(np.abs((y_test_orig - y_pred_orig) / y_test_orig)) * 100


def TrainAndTest(S_data, E_data, Y_data, train_idx, test_idx, train_result, test_result):
    S_train = S_data[train_idx]
    E_train = E_data[train_idx]
    X_train = np.append(S_train, E_train, axis=1)
    Y_train = Y_data[train_idx]

    S_test = S_data[test_idx]
    E_test = E_data[test_idx]
    X_test = np.append(S_test, E_test, axis=1)
    Y_test = Y_data[test_idx]


    svr_set = SVR(kernel='rbf', C=1000, epsilon=0.001, gamma='auto')
    fit_tr = svr_set.fit(X_train, Y_train.ravel())

    Y_pred = fit_tr.predict(X_train)

    mae_tr = MAE(Y_train.ravel(), Y_pred)
    mse_tr = MSE(Y_train.ravel(), Y_pred)
    mape_tr = MAPE(Y_train.ravel(), Y_pred)

    print("train %lf %lf %lf" % (mae_tr, mse_tr, mape_tr))

    Y_pred = fit_tr.predict(X_test)

    mae_te = MAE(Y_test.ravel(), Y_pred)
    mse_te = MSE(Y_test.ravel(), Y_pred)
    mape_te = MAPE(Y_test.ravel(), Y_pred)

    print("test %lf %lf %lf" % (mae_te, mse_te, mape_te))


###################################################-MAIN-###################################################
S_data, _, E_data, Y_data = input_data(0b101) #speed, exogenous 사용
final_result = [[] for i in range(CROSS_ITERATION_NUM)]

train_result = []
test_result = []

cr_idx =0
for train_idx, test_idx in load_Data():
    print("CV %d" % cr_idx)


    TrainAndTest(S_data, E_data, Y_data, train_idx, test_idx, train_result, test_result)


    cr_idx = cr_idx + 1
    if (cr_idx == CROSS_ITERATION_NUM):
        break
