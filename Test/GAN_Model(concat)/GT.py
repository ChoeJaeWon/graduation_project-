
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


def TrainAndTest(S_data, Y_data, train_idx, test_idx):
    S_train = S_data[train_idx]
    Y_train = Y_data[train_idx]

    S_test = S_data[test_idx]
    Y_test = Y_data[test_idx]

    mae_tr = MAE(Y_train.ravel(), S_train[:, -1])
    mse_tr = MSE(Y_train.ravel(), S_train[:, -1])
    mape_tr = MAPE(Y_train.ravel(), S_train[:, -1])

    print("train %lf %lf %lf" % (mae_tr, mse_tr, mape_tr))
    train_result[0] +=mae_tr
    train_result[1] += mse_tr
    train_result[2] += mape_tr

    mae_te = MAE(Y_test.ravel(), S_test[:,-1])
    mse_te = MSE(Y_test.ravel(), S_test[:,-1])
    mape_te = MAPE(Y_test.ravel(), S_test[:,-1])

    print("test %lf %lf %lf" % (mae_te, mse_te, mape_te))
    test_result[0] += mae_te
    test_result[1] += mse_te
    test_result[2] += mape_te


###################################################-MAIN-###################################################
S_data, _, E_data, Y_data = input_data(0b101) #speed, exogenous 사용

test_result = [0, 0, 0]
train_result = [0, 0, 0]

cr_idx =0
for train_idx, test_idx in load_Data():
    print("CV %d" % cr_idx)


    TrainAndTest(S_data, Y_data, train_idx, test_idx)


    cr_idx = cr_idx + 1
    if (cr_idx == CROSS_ITERATION_NUM):
        break

print("\nCROSS_ITERATION_NUM: %d" % CROSS_ITERATION_NUM)
print("average_train %lf % lf %lf" % (train_result[0]/CROSS_ITERATION_NUM, train_result[1]/CROSS_ITERATION_NUM, train_result[2]/CROSS_ITERATION_NUM))
print("average_test %lf % lf %lf" % (test_result[0]/CROSS_ITERATION_NUM, test_result[1]/CROSS_ITERATION_NUM, test_result[2]/CROSS_ITERATION_NUM))
