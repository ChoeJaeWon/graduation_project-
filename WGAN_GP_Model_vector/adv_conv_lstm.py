'''
----------------------------코드 설명----------------------------

----------------------------고려 사항----------------------------

'''
from module import *
import os
#CONV+LSTM을 구현
def model_base(C, E, Y, DISCRIMINATOR_BA,  DISCRIMINATOR_DR):
    for idx in range(CELL_SIZE):
        if idx == 0:
            layer = tf.reshape(CNN_model(C[idx], BA), [1, BATCH_SIZE, TIME_STAMP])
        else:
            layer = tf.concat([layer, tf.reshape(CNN_model(C[idx], BA, True), [1, BATCH_SIZE, TIME_STAMP])], axis=0)
    layer = LSTM_model_12(layer, E)
    layer = tf.reshape(layer, [CELL_SIZE, BATCH_SIZE])
    Y = tf.reshape(Y, [CELL_SIZE, BATCH_SIZE])

    train_MSE = MSE(Y, layer)
    cost_MAE = MAE(Y[TIME_STAMP - 1], layer[TIME_STAMP - 1])
    cost_MSE = MSE(Y[TIME_STAMP - 1], layer[TIME_STAMP - 1])
    cost_MAPE = MAPE(Y[TIME_STAMP - 1], layer[TIME_STAMP - 1])

    layer = tf.transpose(layer, perm=[1, 0])  # lstm에 unstack 이 있다면, 여기서는 transpose를 해주는 편이 위의 계산할 때 편할 듯
    Y = tf.transpose(Y, perm=[1, 0])  # y는 처음부터 잘 만들면 transpose할 필요 없지만, x랑 같은 batchslice를 하게 해주려면 이렇게 하는 편이 나음.

    loss_D = tf.reduce_mean(Discriminator_model(Y, E[CELL_SIZE - 1], DISCRIMINATOR_BA, DISCRIMINATOR_DR)) - tf.reduce_mean(Discriminator_model(layer, E[TIME_STAMP - 1], DISCRIMINATOR_BA, DISCRIMINATOR_DR, True))
    loss_G = -tf.reduce_mean(Discriminator_model(layer, E[CELL_SIZE - 1], DISCRIMINATOR_BA, DISCRIMINATOR_DR, True)) + DISCRIMINATOR_ALPHA * cost_MSE

    epsilon = tf.random_uniform(shape=[BATCH_SIZE, CELL_SIZE], minval=0., maxval=1.)
    Y_hat = Y + epsilon * (layer - Y)
    D_Y_hat = Discriminator_model(Y_hat, E[CELL_SIZE-1], DISCRIMINATOR_BA, DISCRIMINATOR_DR)
    grad_D_Y_hat = tf.gradients(D_Y_hat, [Y_hat])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(grad_D_Y_hat), reduction_indices=[1]))
    gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
    loss_D = loss_D + 1.0 * gradient_penalty

    vars_D = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                               scope='discriminator_fc')  # 여기는 하나로 함수 합쳤음
    vars_G = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                               scope='generator_conv') + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                               scope='generator_lstm')  # 다양해지면 여기가 모델마다 바뀜


    D_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='discriminator_fc')
    with tf.control_dependencies(D_update_ops):
        train_D = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss_D, var_list=[vars_D, discriminator_weights])

    G_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='generator_conv') + tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='generator_lstm')
    with tf.control_dependencies(G_update_ops):
        train_G = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss_G, var_list=[vars_G ,lstm_weights, lstm_biases, conv_weights, convfc_weights])

    return train_MSE, cost_MAE, cost_MSE, cost_MAPE, train_D, train_G, loss_G#, train_G_MSE


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
        epoch_loss = 0.0
        for ba_idx in range(BATCH_NUM):
            #Batch Slice
            #if LATENT_VECTOR_FLAG:
            C_train = batch_slice(C_data, train_idx, ba_idx, 'CONV', CELL_SIZE)
            E_train = batch_slice(E_data, train_idx, ba_idx, 'LSTM', 1)
            Y_train = batch_slice(Y_data, train_idx, ba_idx, 'ADV_FC')

            if tr_idx > OPTIMIZED_EPOCH_CONV_LSTM + 5:
                _= sess.run([train_D], feed_dict={ C:C_train, E:E_train, Y: Y_train, BA: True,DISCRIMINATOR_BA:True, DISCRIMINATOR_DR: DISCRIMINATOR_TR_KEEP_PROB })
            if (tr_idx <= OPTIMIZED_EPOCH_CONV_LSTM + 5) | (tr_idx > OPTIMIZED_EPOCH_CONV_LSTM + 25):
                cost_MSE_val, cost_MSE_hist_val, _, loss= sess.run([cost_MSE, cost_MSE_hist, train_G, loss_G], feed_dict={C:C_train, E:E_train, Y: Y_train,BA: True,DISCRIMINATOR_BA:True, DISCRIMINATOR_DR: DISCRIMINATOR_TR_KEEP_PROB })
                epoch_cost += cost_MSE_val
                epoch_loss += loss
                writer_train.add_summary(cost_MSE_hist_val, global_step_tr)
            global_step_tr += 1

        #설정 interval당 train과 test 값을 출력해준다.
        if tr_idx % TRAIN_PRINT_INTERVAL == 0:
            train_result.append(epoch_cost / BATCH_NUM)
            print("Train Cost %d: %lf" % (tr_idx, epoch_cost/BATCH_NUM ))
        if (tr_idx+1) % TEST_PRINT_INTERVAL == 0:
            if MASTER_SAVE_FLAG:
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
        # if LATENT_VECTOR_FLAG:
        C_test = batch_slice(C_data, test_idx, ba_idx, 'CONV', CELL_SIZE)
        E_test = batch_slice(E_data, test_idx, ba_idx, 'LSTM', 1)
        Y_test = batch_slice(Y_data, test_idx, ba_idx, 'ADV_FC')
        '''
        else:
            C_test = batch_slice(C_data, test_idx, ba_idx, 'CONV', CELL_SIZE)
            E_test = batch_slice(E_data, test_idx, ba_idx, 'LSTM', 1)
            Y_test = batch_slice(Y_data, test_idx, ba_idx, 'LSTMY', 1)
        '''
        cost_MAE_val, cost_MSE_val, cost_MAPE_val, cost_MAE_hist_val, cost_MSE_hist_val, cost_MAPE_hist_val = sess.run([cost_MAE, cost_MSE, cost_MAPE, cost_MAE_hist, cost_MSE_hist, cost_MAPE_hist], feed_dict={C:C_test, E:E_test, Y:Y_test, BA: False, DISCRIMINATOR_BA: False, DISCRIMINATOR_DR:DISCRIMINATOR_TE_KEEP_PROB})
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

if LATENT_VECTOR_FLAG:
    cr_idx = 0
    kf = KFold(n_splits=CROSS_NUM, shuffle=True)
    for train_idx, test_idx in Week_CrossValidation():
        print('CROSS VALIDATION: %d' % cr_idx)

        train_result = []
        test_result = []

        C = tf.placeholder("float32", [CELL_SIZE, None, SPARTIAL_NUM, TEMPORAL_NUM, 1])  # cell_size, batch_size
        E = tf.placeholder("float32", [CELL_SIZE, None, EXOGENOUS_NUM])  # cell_size, batch_size
        Y = tf.placeholder("float32", [CELL_SIZE, None, 1])
        BA = tf.placeholder(tf.bool)
        DISCRIMINATOR_BA = tf.placeholder(tf.bool)
        DISCRIMINATOR_DR = tf.placeholder(tf.float32)
        if RESTORE_GENERATOR_FLAG:
            last_epoch = tf.Variable(OPTIMIZED_EPOCH_CONV_LSTM+1, name=LAST_EPOCH_NAME)
        else:
            last_epoch = tf.Variable(0, name=LAST_EPOCH_NAME)

        init()
        sess = tf.Session()
        train_MSE, cost_MAE, cost_MSE, cost_MAPE, train_D, train_G, loss_G  = model_base(C, E, Y, DISCRIMINATOR_BA,  DISCRIMINATOR_DR)
        writer_train = tf.summary.FileWriter("./tensorboard/adv_conv_lstm/train%d" % cr_idx, sess.graph)
        writer_test = tf.summary.FileWriter("./tensorboard/adv_conv_lstm/test%d" % cr_idx, sess.graph)
        train_MSE_hist = tf.summary.scalar('train_MSE', train_MSE)
        cost_MAE_hist = tf.summary.scalar('cost_MAE', cost_MAE)
        cost_MSE_hist = tf.summary.scalar('cost_MSE', cost_MSE)
        cost_MAPE_hist = tf.summary.scalar('cost_MAPE', cost_MAPE)
        sess.run(tf.global_variables_initializer())

        # Saver and Restore
        if RESTORE_GENERATOR_FLAG:
            conv_lstm_batch_norm_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                      scope='generator_conv') +  tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                      scope='generator_lstm')
            variables_to_restore = lstm_weights + lstm_biases +  conv_weights + convfc_weights + conv_lstm_batch_norm_weights
            saver = tf.train.Saver(variables_to_restore)
        else:
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
else:
    print("scaler prediction mode is not implemented yet byebye")
