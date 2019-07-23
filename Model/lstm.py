'''
----------------------------코드 설명----------------------------
-C-
3.LSTM에 해당하는 코드로
LSTM으로 구현함
----------------------------고려 사항----------------------------
lstm은 input x에 이미 exogenous가 포함되어있다.
하지만 conv_lstm구현시 따로 받아야함으로 preprocess를 수정해야 한다
수정사항
1. exogenous도 lstm버전이 있어야한다.(즉 CELL_SIZE 따라 반복 되어야함)
2. 기존의 lstm버전은 speed만 고려해야한다.(vector를 66개 -> 12개로 수정)
'''
from module import *

#LSTM을 구현
def model(S, E, Y):
    layer = LSTM_model(S, E)

    cost_MAE = MAE(Y, layer)
    cost_MSE = MSE(Y, layer)
    cost_MAPE = MAPE(Y, layer)
    optimal = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cost_MSE)

    return cost_MAE, cost_MSE, cost_MAPE, optimal

#training 해준다.
def train(S_data, E_data, Y_data, cost_MSE, optimal, train_idx):
    BATCH_NUM = int(len(train_idx) / BATCH_SIZE)
    for tr_idx in range(LSTM_TRAIN_NUM):
        epoch_cost = 0.0
        for ba_idx in range(BATCH_NUM):
            #Batch Slice
            S_train = batch_slice(S_data, train_idx, ba_idx, 'LSTM', 1)
            E_train = batch_slice(E_data, train_idx, ba_idx, 'LSTM', 1)
            Y_train = batch_slice(Y_data, train_idx, ba_idx, 'FC', 1)

            cost_MSE_val, _= sess.run([cost_MSE, optimal], feed_dict={S:S_train, E:E_train, Y: Y_train })
            epoch_cost += cost_MSE_val

        #한 epoch당 cost_MSE의 평균을 구해준다.
        print("Train Cost%d: %lf" % (tr_idx, epoch_cost/BATCH_NUM ))

        #cross validation의 train_idx를 shuffle해준다.
        np.random.shuffle(train_idx)


#testing 해준다.
def test(S_data, E_data, Y_data, cost_MAE, cost_MSE, cost_MAPE, test_idx, cr_idx):
    BATCH_NUM = int(len(test_idx) / BATCH_SIZE)
    mae = 0.0
    mse = 0.0
    mape = 0.0
    for ba_idx in range(BATCH_NUM):
        # Batch Slice
        S_test = batch_slice(S_data, test_idx, ba_idx, 'LSTM', 1)
        E_test = batch_slice(E_data, test_idx, ba_idx, 'LSTM', 1)
        Y_test = batch_slice(Y_data, test_idx, ba_idx, 'FC', 1)

        cost_MAE_val, cost_MSE_val, cost_MAPE_val = sess.run([cost_MAE, cost_MSE, cost_MAPE], feed_dict={S:S_test, E:E_test, Y:Y_test})
        mae += cost_MAE_val
        mse += cost_MSE_val
        mape += cost_MAPE_val


    print("Test Cost%d: MAE(%lf) MSE(%lf) MAPE(%lf)" % (cr_idx, mae/BATCH_NUM, mse/BATCH_NUM, mape/BATCH_NUM))




###################################################-MAIN-###################################################
S_data, _, E_data,Y_data= input_data(0b101)


cr_idx = 0
kf = KFold(n_splits=CROSS_NUM, shuffle=True)
for train_idx, test_idx in kf.split(Y_data[:-CELL_SIZE]):
    S = tf.placeholder("float32", [None, CELL_SIZE, TIME_STAMP])
    E = tf.placeholder("float32", [None, CELL_SIZE, EXOGENOUS_NUM])
    Y = tf.placeholder("float32", [None, 1])

    init()
    sess = tf.Session()
    cost_MAE, cost_MSE, cost_MAPE, optimal = model(S, E, Y)
    sess.run(tf.global_variables_initializer())


    train(S_data, E_data, Y_data, cost_MSE, optimal, train_idx)
    test(S_data, E_data, Y_data, cost_MAE, cost_MSE, cost_MAPE, test_idx, cr_idx)

    tf.reset_default_graph()

    cr_idx = cr_idx + 1
