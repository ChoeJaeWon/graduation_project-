'''
----------------------------코드 설명----------------------------

----------------------------고려 사항----------------------------

'''
from module import *
import os
#CONV+LSTM을 구현
def model(S,C, E, Y, DISCRIMINATOR_BA,  DISCRIMINATOR_DR):
    for idx in range(CELL_SIZE):
        if idx == 0:
            layer = tf.reshape(CNN_model(C[idx], BA), [1, BATCH_SIZE, TIME_STAMP])
        else:
            layer = tf.concat([layer, tf.reshape(CNN_model(C[idx], BA), [1, BATCH_SIZE, TIME_STAMP])], axis=0)
    layer = LSTM_model(layer, E)

    cost_MAE = MAE(Y, layer)
    cost_MSE = MSE(Y, layer)
    cost_MAPE = MAPE(Y, layer)

    #CELL_SIZE 가 input x라고 가정합니다.
    adv_y = tf.concat([S[CELL_SIZE-1], Y], axis=1)
    adv_g = tf.concat([S[CELL_SIZE-1], layer], axis=1)
    loss_D = -tf.reduce_mean(tf.log(Discriminator_model(adv_y, E[CELL_SIZE-1], DISCRIMINATOR_BA, DISCRIMINATOR_DR)) + tf.log(1 - Discriminator_model(adv_g, E[CELL_SIZE-1], DISCRIMINATOR_BA, DISCRIMINATOR_DR)))
    loss_G = -tf.reduce_mean(tf.log(Discriminator_model(adv_g, E[CELL_SIZE-1], DISCRIMINATOR_BA, DISCRIMINATOR_DR))) + DISCRIMINATOR_ALPHA * cost_MSE  # MSE 는 0~ t까지 있어봤자 같은 값이다.

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_D = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE*2).minimize(loss_D)
        train_G = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE*2).minimize(loss_G)

    return cost_MAE, cost_MSE, cost_MAPE, train_D, train_G

#training 해준다.
def train(S_data, C_data, E_data, Y_data, cost_MAE, cost_MSE, cost_MAPE, cost_MAE_hist, cost_MSE_hist, cost_MAPE_hist, train_D, train_G, train_idx, test_idx, cr_idx,  writer_train, writer_test, train_result, test_result, CURRENT_POINT_DIR, start_from):
    BATCH_NUM = int(len(train_idx) / BATCH_SIZE)
    print('BATCH_NUM: %d' % BATCH_NUM)
    for _ in range(start_from):
        np.random.shuffle(train_idx)
    global_step_tr = 0
    global_step_te = 0
    for tr_idx in range(start_from, TRAIN_NUM):
        epoch_cost = 0.0
        for ba_idx in range(BATCH_NUM):
            #Batch Slice
            S_train = batch_slice(S_data, train_idx, ba_idx, 'LSTM', 1)
            C_train = batch_slice(C_data, train_idx, ba_idx, 'CONV', CELL_SIZE)
            E_train = batch_slice(E_data, train_idx, ba_idx, 'LSTM', 1)
            Y_train = batch_slice(Y_data, train_idx, ba_idx, 'LSTMY', 1)

            _ = sess.run([train_D], feed_dict={S:S_train, C:C_train, E:E_train, Y: Y_train, BA: True,DISCRIMINATOR_BA:True, DISCRIMINATOR_DR: DISCRIMINATOR_TR_KEEP_PROB })
            cost_MSE_val, cost_MSE_hist_val, _= sess.run([cost_MSE, cost_MSE_hist, train_G], feed_dict={S:S_train, C:C_train, E:E_train, Y: Y_train,BA: True,DISCRIMINATOR_BA:True, DISCRIMINATOR_DR: DISCRIMINATOR_TR_KEEP_PROB })
            epoch_cost += cost_MSE_val
            writer_train.add_summary(cost_MSE_hist_val, global_step_tr)
            global_step_tr += 1

        #설정 interval당 train과 test 값을 출력해준다.
        if tr_idx % TRAIN_PRINT_INTERVAL == 0:
            train_result.append(epoch_cost / BATCH_NUM)
            print("Train Cost %d: %lf" % (tr_idx, epoch_cost/BATCH_NUM ))
        if (tr_idx+1) % TEST_PRINT_INTERVAL == 0:

            print("Saving network...")
            sess.run(last_epoch.assign(tr_idx + 1))
            if not os.path.exists(CURRENT_POINT_DIR):
                os.makedirs(CURRENT_POINT_DIR)
            saver.save(sess, CURRENT_POINT_DIR + "/model", global_step=tr_idx, write_meta_graph=False)

            global_step_te = test(S_data, C_data, E_data, Y_data, cost_MAE, cost_MSE, cost_MAPE, cost_MAE_hist, cost_MSE_hist, cost_MAPE_hist, test_idx, tr_idx, global_step_te, cr_idx, writer_test, test_result)

        #cross validation의 train_idx를 shuffle해준다.
        np.random.shuffle(train_idx)


#testing 해준다.
def test(S_data, C_data, E_data, Y_data, cost_MAE, cost_MSE, cost_MAPE, cost_MAE_hist, cost_MSE_hist, cost_MAPE_hist, test_idx, tr_idx, global_step_te, cr_idx, writer_test, test_result):
    BATCH_NUM = int(len(test_idx) / BATCH_SIZE)
    mae = 0.0
    mse = 0.0
    mape = 0.0
    for ba_idx in range(BATCH_NUM):
        # Batch Slice
        S_test = batch_slice(S_data, test_idx, ba_idx, 'LSTM', 1)
        C_test = batch_slice(C_data, test_idx, ba_idx, 'CONV', CELL_SIZE)
        E_test = batch_slice(E_data, test_idx, ba_idx, 'LSTM', 1)
        Y_test = batch_slice(Y_data, test_idx, ba_idx, 'LSTMY', 1)

        cost_MAE_val, cost_MSE_val, cost_MAPE_val, cost_MAE_hist_val, cost_MSE_hist_val, cost_MAPE_hist_val = sess.run([cost_MAE, cost_MSE, cost_MAPE, cost_MAE_hist, cost_MSE_hist, cost_MAPE_hist], feed_dict={S:S_test, C:C_test, E:E_test, Y:Y_test, BA: False, DISCRIMINATOR_BA: False, DISCRIMINATOR_DR:DISCRIMINATOR_TE_KEEP_PROB})
        mae += cost_MAE_val
        mse += cost_MSE_val
        mape += cost_MAPE_val

        writer_test.add_summary(cost_MAE_hist_val, global_step_te)
        writer_test.add_summary(cost_MSE_hist_val, global_step_te)
        writer_test.add_summary(cost_MAPE_hist_val, global_step_te)

        global_step_te += 1

    test_result.append([mae / BATCH_NUM, mse / BATCH_NUM, mape / BATCH_NUM])
    print("Test Cost(%d) %d: MAE(%lf) MSE(%lf) MAPE(%lf)" % (cr_idx, tr_idx, mae / BATCH_NUM, mse / BATCH_NUM, mape / BATCH_NUM))
    return global_step_te



###################################################-MAIN-###################################################
S_data, C_data, E_data,Y_data= input_data(0b111)

cr_idx = 0
kf = KFold(n_splits=CROSS_NUM, shuffle=True)
for train_idx, test_idx in kf.split(Y_data[:-CELL_SIZE]):
    print('CROSS VALIDATION: %d' % cr_idx)

    train_result = []
    test_result = []

    S = tf.placeholder("float32", [CELL_SIZE, None, TIME_STAMP])  # cell_size, batch_size
    C = tf.placeholder("float32", [CELL_SIZE, None, SPARTIAL_NUM, TEMPORAL_NUM, 1])  # cell_size, batch_size
    E = tf.placeholder("float32", [CELL_SIZE, None, EXOGENOUS_NUM])  # cell_size, batch_size
    Y = tf.placeholder("float32", [None, 1])
    BA = tf.placeholder(tf.bool)
    DISCRIMINATOR_BA = tf.placeholder(tf.bool)
    DISCRIMINATOR_DR = tf.placeholder(tf.float32)
    last_epoch = tf.Variable(0, name=LAST_EPOCH_NAME)

    init()
    sess = tf.Session()
    cost_MAE, cost_MSE, cost_MAPE, train_D, train_G = model(S,C, E, Y, DISCRIMINATOR_BA,  DISCRIMINATOR_DR)
    writer_train = tf.summary.FileWriter("./tensorboard/adv_conv_lstm/train%d" % cr_idx, sess.graph)
    writer_test = tf.summary.FileWriter("./tensorboard/adv_conv_lstm/test%d" % cr_idx, sess.graph)
    cost_MAE_hist = tf.summary.scalar('cost_MAE', cost_MAE)
    cost_MSE_hist = tf.summary.scalar('cost_MSE', cost_MSE)
    cost_MAPE_hist = tf.summary.scalar('cost_MAPE', cost_MAPE)
    sess.run(tf.global_variables_initializer())

    # Saver and Restore
    saver = tf.train.Saver()
    CURRENT_POINT_DIR = CHECK_POINT_DIR + "ADV_CONV_LSTM_" + str(cr_idx) + "/"
    checkpoint = tf.train.get_checkpoint_state(CURRENT_POINT_DIR)

    if RESTORE_FLAG:
        if checkpoint and checkpoint.model_checkpoint_path:
            try:
                saver.restore(sess, checkpoint.model_checkpoint_path)
                print("Successfully loaded:", checkpoint.model_checkpoint_path)
            except:
                print("Error on loading old network weights")
        else:
            print("Could not find old network weights")

    start_from = sess.run(last_epoch)
    # train my model
    print('Start learning from:', start_from)

    train(S_data,C_data,E_data, Y_data, cost_MAE, cost_MSE, cost_MAPE, cost_MAE_hist, cost_MSE_hist, cost_MAPE_hist, train_D, train_G, train_idx, test_idx, cr_idx,  writer_train, writer_test, train_result, test_result , CURRENT_POINT_DIR, start_from)

    tf.reset_default_graph()

    output_data(train_result, test_result, 'adv_conv_lstm', cr_idx)

    cr_idx=cr_idx+1

    if (cr_idx == CROSS_ITERATION_NUM):
        break
