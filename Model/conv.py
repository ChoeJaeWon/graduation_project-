'''
----------------------------코드 설명----------------------------
-C-
2.CONV에 해당하는 코드로
conv+fc로 구현함
----------------------------고려 사항----------------------------


'''
from module import *

#FC와 CNN을 합침
def model(C, E, Y, BA, DR):
    layer = CNN_model(C[0], BA)
    layer = FC_model(layer, E, BA, DR)

    cost_MAE = MAE(Y, layer)
    cost_MSE = MSE(Y, layer)
    cost_MAPE = MAPE(Y, layer)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimal = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cost_MSE)

    return cost_MAE, cost_MSE, cost_MAPE, optimal

#training 해준다.
def train(C_data, E_data, Y_data, cost_MAE, cost_MSE, cost_MAPE, optimal, train_idx, test_idx, cr_idx):
    BATCH_NUM = int(len(train_idx) / BATCH_SIZE)
    for tr_idx in range(TRAIN_NUM):
        epoch_cost = 0.0
        for ba_idx in range(BATCH_NUM):
            #Batch Slice
            C_train = batch_slice(C_data, train_idx, ba_idx, 'CONV', 1)
            E_train = batch_slice(E_data, train_idx, ba_idx, 'FC', 1)
            Y_train = batch_slice(Y_data, train_idx, ba_idx, 'FC', 1)

            cost_MSE_val, _= sess.run([cost_MSE, optimal], feed_dict={C:C_train, E:E_train, Y: Y_train, BA: True, DR: FC_TR_KEEP_PROB})
            epoch_cost += cost_MSE_val

        # 설정 interval당 train과 test 값을 출력해준다.
        if tr_idx % TRAIN_PRINT_INTERVAL == 0:
            print("Train Cost %d: %lf" % (tr_idx, epoch_cost / BATCH_NUM))
        if (tr_idx+1) % TEST_PRINT_INTERVAL == 0:
            test(C_data, E_data, Y_data, cost_MAE, cost_MSE, cost_MAPE, test_idx, tr_idx, cr_idx)

        #cross validation의 train_idx를 shuffle해준다.
        np.random.shuffle(train_idx)


#testing 해준다.
def test(C_data, E_data, Y_data, cost_MAE, cost_MSE, cost_MAPE, test_idx, tr_idx, cr_idx):
    BATCH_NUM = int(len(test_idx) / BATCH_SIZE)
    mae = 0.0
    mse = 0.0
    mape = 0.0
    for ba_idx in range(BATCH_NUM):
        # Batch Slice
        C_test = batch_slice(C_data, test_idx, ba_idx, 'CONV', 1)
        E_test = batch_slice(E_data, test_idx, ba_idx, 'FC', 1)
        Y_test = batch_slice(Y_data, test_idx, ba_idx, 'FC', 1)

        cost_MAE_val, cost_MSE_val, cost_MAPE_val = sess.run([cost_MAE, cost_MSE, cost_MAPE], feed_dict={C:C_test, E:E_test, Y:Y_test, BA: False, DR: FC_TE_KEEP_PROB})
        mae += cost_MAE_val
        mse += cost_MSE_val
        mape += cost_MAPE_val

    print("Test Cost(%d) %d: MAE(%lf) MSE(%lf) MAPE(%lf)" % (cr_idx, tr_idx, mae / BATCH_NUM, mse / BATCH_NUM, mape / BATCH_NUM))




###################################################-MAIN-###################################################
_, C_data, E_data, Y_data = input_data(0b011)

cr_idx = 0
kf = KFold(n_splits=CROSS_NUM, shuffle=True)
for train_idx, test_idx in kf.split(Y_data[:-CELL_SIZE]):
    print('CROSS VALIDATION: %d' % cr_idx)
    C = tf.placeholder("float32", [None, None, SPARTIAL_NUM, TEMPORAL_NUM,1])
    E = tf.placeholder("float32", [None, EXOGENOUS_NUM])
    Y = tf.placeholder("float32", [None, 1])
    BA = tf.placeholder(tf.bool)
    DR = tf.placeholder(tf.float32)

    init()
    sess = tf.Session()
    cost_MAE, cost_MSE, cost_MAPE, optimal = model(C, E, Y, BA, DR)
    sess.run(tf.global_variables_initializer())

    train(C_data, E_data, Y_data, cost_MAE, cost_MSE, cost_MAPE, optimal, train_idx, test_idx, cr_idx)

    tf.reset_default_graph()

    cr_idx=cr_idx+1

    if (cr_idx == CROSS_ITERATION_NUM):
        break
