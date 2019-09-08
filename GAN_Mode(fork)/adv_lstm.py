'''
----------------------------코드 설명----------------------------

----------------------------고려 사항----------------------------

기존의 lstm 은 only speed 만 고려해야한다 ?? 이게 무슨 말
'''
from module import *
import os
#LSTM을 구현

def model_base(S, E, Y, DISCRIMINATOR_BA,  DISCRIMINATOR_DR):

    for gen_idx in range(GEN_NUM):
        if gen_idx == 0:
            layer = LSTM_model(S[gen_idx], E[gen_idx])
        else:
            layer = tf.concat([layer, LSTM_model(S[gen_idx], E[gen_idx], True)], axis=1)

    train_MSE = MSE(Y, layer)
    cost_MAE = MAE(Y[:,TIME_STAMP - 1], layer[:,TIME_STAMP - 1])
    cost_MSE = MSE(Y[:,TIME_STAMP - 1], layer[:,TIME_STAMP - 1])
    cost_MAPE = MAPE(Y[:,TIME_STAMP - 1], layer[:,TIME_STAMP - 1])

    # Pix2Pix
    DE = tf.concat([E[GEN_NUM - 1][CELL_SIZE-1], S[GEN_NUM - 1][CELL_SIZE-1]], axis=1)

    loss_D = -tf.reduce_mean(tf.log(Discriminator_model(Y, DE, DISCRIMINATOR_BA, DISCRIMINATOR_DR)) + tf.log(1 - Discriminator_model(layer, DE, DISCRIMINATOR_BA, DISCRIMINATOR_DR,True)))
    loss_G = -tf.reduce_mean(tf.log(Discriminator_model(layer, DE, DISCRIMINATOR_BA, DISCRIMINATOR_DR, True))) + DISCRIMINATOR_ALPHA * train_MSE  # MSE 는 0~ t까지 있어봤자 같은 값이다.

    vars_D = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator_fc')  # 여기는 하나로 함수 합쳤음
    vars_G = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator_lstm')  # 다양해지면 여기가 모델마다 바뀜

    D_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='discriminator_fc')
    with tf.control_dependencies(D_update_ops):
        train_D = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss_D, var_list=[vars_D, discriminator_weights])

    G_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='generator_lstm')
    with tf.control_dependencies(G_update_ops):
        train_G = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss_G, var_list=[vars_G, lstm_weights, lstm_biases, conv_weights, convfc_weights])

    return train_MSE, cost_MAE, cost_MSE, cost_MAPE, train_D, train_G, loss_G

#training 해준다.
def train(S_data, E_data, Y_data, cost_MAE, cost_MSE, cost_MAPE, cost_MAE_hist, cost_MSE_hist, cost_MAPE_hist, train_D, train_G, train_idx, test_idx, cr_idx,  writer_train, writer_test, train_result, test_result, CURRENT_POINT_DIR, start_from):
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
            S_train = batch_slice(S_data, train_idx, ba_idx, 'ADV_LSTM', 1)
            E_train = batch_slice(E_data, train_idx, ba_idx, 'ADV_LSTM', 1)
            Y_train = batch_slice(Y_data, train_idx, ba_idx, 'ADV_LSTMY')

            if tr_idx > OPTIMIZED_EPOCH_LSTM + PHASE1_EPOCH:
                _ = sess.run([train_D], feed_dict={S:S_train, E:E_train, Y: Y_train ,DISCRIMINATOR_BA:True, DISCRIMINATOR_DR: DISCRIMINATOR_TR_KEEP_PROB})
            if (tr_idx <= OPTIMIZED_EPOCH_LSTM + PHASE1_EPOCH) | (tr_idx > OPTIMIZED_EPOCH_LSTM + PHASE1_EPOCH + PHASE2_EPOCH):
                cost_MSE_val, cost_MSE_hist_val, _, loss= sess.run([cost_MSE, cost_MSE_hist, train_G, loss_G], feed_dict={S:S_train, E:E_train, Y: Y_train ,DISCRIMINATOR_BA:True, DISCRIMINATOR_DR: DISCRIMINATOR_TR_KEEP_PROB})
                epoch_loss += loss
                epoch_cost += cost_MSE_val
                writer_train.add_summary(cost_MSE_hist_val, global_step_tr)
            global_step_tr += 1

        # 설정 interval당 train과 test 값을 출력해준다.
        if tr_idx % TRAIN_PRINT_INTERVAL == 0:
            train_result.append(epoch_cost / BATCH_NUM)
            print("Train Cost %d: %lf" % (tr_idx, epoch_cost / BATCH_NUM))
            print("G_loss %d: %lf" % (tr_idx, epoch_loss / BATCH_NUM))
        if (tr_idx+1) % TEST_PRINT_INTERVAL == 0:
            if MASTER_SAVE_FLAG:
                print("Saving network...")
                sess.run(last_epoch.assign(tr_idx+1))
                if not os.path.exists(CURRENT_POINT_DIR):
                    os.makedirs(CURRENT_POINT_DIR)
                saver.save(sess, CURRENT_POINT_DIR + "/model", global_step=tr_idx, write_meta_graph=False)

            global_step_te = test(S_data, E_data, Y_data, cost_MAE, cost_MSE, cost_MAPE, cost_MAE_hist, cost_MSE_hist, cost_MAPE_hist, test_idx, tr_idx, global_step_te, cr_idx, writer_test, test_result)

        #cross validation의 train_idx를 shuffle해준다.
        np.random.shuffle(train_idx)

#testing 해준다.
def test(S_data, E_data, Y_data, cost_MAE, cost_MSE, cost_MAPE, cost_MAE_hist, cost_MSE_hist, cost_MAPE_hist, test_idx, tr_idx, global_step_te, cr_idx, writer_test, test_result):
    BATCH_NUM = int(len(test_idx) / BATCH_SIZE)
    mae = 0.0
    mse = 0.0
    mape = 0.0
    for ba_idx in range(BATCH_NUM):
        # Batch Slice
        # if LATENT_VECTOR_FLAG:
        S_test = batch_slice(S_data, test_idx, ba_idx, 'ADV_LSTM', 1)
        E_test = batch_slice(E_data, test_idx, ba_idx, 'ADV_LSTM', 1)
        Y_test = batch_slice(Y_data, test_idx, ba_idx, 'ADV_LSTMY')
        '''
        else:
            S_test = batch_slice(S_data, test_idx, ba_idx, 'LSTM', 1)
            E_test = batch_slice(E_data, test_idx, ba_idx, 'LSTM', 1)
            Y_test = batch_slice(Y_data, test_idx, ba_idx, 'LSTMY', 1)
        '''
        cost_MAE_val, cost_MSE_val, cost_MAPE_val, cost_MAE_hist_val, cost_MSE_hist_val, cost_MAPE_hist_val = sess.run([cost_MAE, cost_MSE, cost_MAPE, cost_MAE_hist, cost_MSE_hist, cost_MAPE_hist], feed_dict={S:S_test, E:E_test, Y:Y_test,DISCRIMINATOR_BA: False, DISCRIMINATOR_DR:DISCRIMINATOR_TE_KEEP_PROB})
        mae += cost_MAE_val
        mse += cost_MSE_val
        mape += cost_MAPE_val

        writer_test.add_summary(cost_MAE_hist_val, global_step_te)
        writer_test.add_summary(cost_MSE_hist_val, global_step_te)
        writer_test.add_summary(cost_MAPE_hist_val, global_step_te)

        global_step_te += 1

    test_result.append([mae / BATCH_NUM, mse / BATCH_NUM, mape / BATCH_NUM])
    final_result[cr_idx].append(mape / BATCH_NUM)
    print("Test Cost(%d) %d: MAE(%lf) MSE(%lf) MAPE(%lf)" % (cr_idx, tr_idx, mae / BATCH_NUM, mse / BATCH_NUM, mape / BATCH_NUM))
    return global_step_te

def train_generator_mse():
    return
def train_mse_only():
    return
def train_generator_only():
    return
def train_discriminator():
    return

###################################################-MAIN-###################################################

S_data, _, E_data,Y_data= input_data(0b101)
final_result = []

cr_idx = 0
for train_idx, test_idx in Week_CrossValidation():
    print('CROSS VALIDATION: %d' % cr_idx)
    train_result = []
    test_result = []

    S = tf.placeholder("float32", [GEN_NUM, CELL_SIZE, None, TIME_STAMP]) #cell_size, batch_size
    E = tf.placeholder("float32", [GEN_NUM, CELL_SIZE, None, EXOGENOUS_NUM]) #cell_size, batch_size
    Y = tf.placeholder("float32", [None, GEN_NUM])
    DISCRIMINATOR_BA = tf.placeholder(tf.bool)
    DISCRIMINATOR_DR = tf.placeholder(tf.float32)

    if RESTORE_GENERATOR_FLAG:
        last_epoch = tf.Variable(OPTIMIZED_EPOCH_LSTM + 1, name=LAST_EPOCH_NAME)
    else:
        last_epoch = tf.Variable(0, name=LAST_EPOCH_NAME)

    init()
    sess = tf.Session()
    train_MSE, cost_MAE, cost_MSE, cost_MAPE, train_D, train_G, loss_G = model_base(S, E, Y, DISCRIMINATOR_BA,  DISCRIMINATOR_DR)
    writer_train = tf.summary.FileWriter("./tensorboard/adv_lstm/train%d" % cr_idx, sess.graph)
    writer_test = tf.summary.FileWriter("./tensorboard/adv_lstm/test%d" % cr_idx, sess.graph)
    cost_MAE_hist = tf.summary.scalar('cost_MAE', cost_MAE)
    cost_MSE_hist = tf.summary.scalar('cost_MSE', cost_MSE)
    cost_MAPE_hist = tf.summary.scalar('cost_MAPE', cost_MAPE)
    sess.run(tf.global_variables_initializer())

    # Saver and Restore
    if RESTORE_GENERATOR_FLAG:
        lstm_cell_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator_lstm')
        variables_to_restore = lstm_weights + lstm_biases + lstm_cell_weights
        saver = tf.train.Saver(variables_to_restore)
    else:
        saver = tf.train.Saver()

    CURRENT_POINT_DIR = CHECK_POINT_DIR + "ADV_LSTM_" + str(cr_idx) + "/"
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

    train(S_data, E_data, Y_data, cost_MAE, cost_MSE, cost_MAPE, cost_MAE_hist, cost_MSE_hist, cost_MAPE_hist, train_D, train_G, train_idx, test_idx, cr_idx,  writer_train, writer_test, train_result, test_result, CURRENT_POINT_DIR, start_from)

    tf.reset_default_graph()

    output_data(train_result, test_result, 'adv_lstm' + "_" + str(DISCRIMINATOR_LAYER_NUM) + "_" + str(LEARNING_RATE)[2:]+"_" + format(DISCRIMINATOR_ALPHA, 'f')[2:] + "_" + str(PHASE1_EPOCH) + "_"+ str(PHASE2_EPOCH)+"_"+ str(TRAIN_NUM)+ "_", cr_idx)

    cr_idx=cr_idx+1

    if (cr_idx == CROSS_ITERATION_NUM):
        break

output_result(final_result, 'adv_lstm' + "_" + str(DISCRIMINATOR_LAYER_NUM) + "_" + str(LEARNING_RATE)[2:]+"_" + format(DISCRIMINATOR_ALPHA, 'f')[2:] + "_" + str(PHASE1_EPOCH) + "_"+ str(PHASE2_EPOCH)+"_"+ str(TRAIN_NUM)+ "_", cr_idx)
