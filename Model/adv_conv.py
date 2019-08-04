'''
----------------------------코드 설명----------------------------

----------------------------고려 사항----------------------------


'''
from module import *
import os
#FC와 CNN을 합침
def model(S, C, E, Y, BA, DR, DISCRIMINATOR_BA,  DISCRIMINATOR_DR):
    layer = CNN_model(C[0], BA)
    layer = FC_model(layer, E , BA, DR)

    cost_MAE = MAE(Y, layer)
    cost_MSE = MSE(Y, layer)
    cost_MAPE = MAPE(Y, layer)

    adv_y= tf.concat([S, Y], axis=1)
    adv_g= tf.concat([S, layer], axis=1)
    loss_D = -tf.reduce_mean(tf.log(Discriminator_model(adv_y,E,DISCRIMINATOR_BA, DISCRIMINATOR_DR)) + tf.log(1- Discriminator_model(adv_g,E,DISCRIMINATOR_BA, DISCRIMINATOR_DR, True)))
    loss_G = -tf.reduce_mean(tf.log(Discriminator_model(adv_g,E,DISCRIMINATOR_BA, DISCRIMINATOR_DR, True))) + DISCRIMINATOR_ALPHA*cost_MSE #MSE 는 0~ t까지 있어봤자 같은 값이다.

    vars_D = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                               scope='discriminator_fc')
    vars_G = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                               scope='generator_conv') + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                               scope='generator_fc')

    D_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='discriminator_fc')
    with tf.control_dependencies(D_update_ops):
        train_D = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE*2).minimize(loss_D, var_list=[vars_D, discriminator_weights])
    G_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='generator_conv') +  tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='generator_fc')
    with tf.control_dependencies(G_update_ops):
        train_G = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE*2).minimize(loss_G, var_list=[vars_G, convfc_weights, conv_weights, fc_weights])

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
            S_train = batch_slice(S_data, train_idx, ba_idx, 'FC', 1)
            C_train = batch_slice(C_data, train_idx, ba_idx, 'CONV', 1)
            E_train = batch_slice(E_data, train_idx, ba_idx, 'FC', 1)
            Y_train = batch_slice(Y_data, train_idx, ba_idx, 'FC', 1)

            _= sess.run([train_D], feed_dict={S: S_train, C:C_train, E:E_train, Y: Y_train, BA: True, DR: FC_TR_KEEP_PROB, DISCRIMINATOR_BA:True, DISCRIMINATOR_DR: DISCRIMINATOR_TR_KEEP_PROB})
            cost_MSE_val, cost_MSE_hist_val, _= sess.run([cost_MSE, cost_MSE_hist, train_G], feed_dict={S:S_train, C: C_train, E: E_train, Y: Y_train, BA: True,DR: FC_TR_KEEP_PROB, DISCRIMINATOR_BA:True, DISCRIMINATOR_DR: DISCRIMINATOR_TR_KEEP_PROB})
            epoch_cost += cost_MSE_val
            writer_train.add_summary(cost_MSE_hist_val, global_step_tr)
            global_step_tr += 1

        # 설정 interval당 train과 test 값을 출력해준다.
        if tr_idx % TRAIN_PRINT_INTERVAL == 0:
            train_result.append(epoch_cost / BATCH_NUM)
            print("Train Cost %d: %lf" % (tr_idx, epoch_cost / BATCH_NUM))
        if (tr_idx+1) % TEST_PRINT_INTERVAL == 0:

            print("Saving network...")
            sess.run(last_epoch.assign(tr_idx + 1))
            if not os.path.exists(CURRENT_POINT_DIR):
                os.makedirs(CURRENT_POINT_DIR)
            saver.save(sess, CURRENT_POINT_DIR + "/model", global_step=tr_idx, write_meta_graph=False)

            global_step_te = test(S_data, C_data, E_data, Y_data, cost_MAE, cost_MSE, cost_MAPE, cost_MAE_hist, cost_MSE_hist, cost_MAPE_hist, test_idx, tr_idx, global_step_te, cr_idx, writer_test, test_result)

        #cross validation의 train_idx를 shuffle해준다.
        np.random.shuffle(train_idx)


#testing 해준다. GAN과는 상관없이 최종 MAE, MSE, MAPE만 뽑아준다.
def test(S_data, C_data, E_data, Y_data, cost_MAE, cost_MSE, cost_MAPE, cost_MAE_hist, cost_MSE_hist, cost_MAPE_hist, test_idx, tr_idx, global_step_te, cr_idx, writer_test, test_result):
    BATCH_NUM = int(len(test_idx) / BATCH_SIZE)
    mae = 0.0
    mse = 0.0
    mape = 0.0
    for ba_idx in range(BATCH_NUM):
        # Batch Slice
        S_test = batch_slice(S_data, test_idx, ba_idx, 'FC', 1)
        C_test = batch_slice(C_data, test_idx, ba_idx, 'CONV', 1)
        E_test = batch_slice(E_data, test_idx, ba_idx, 'FC', 1)
        Y_test = batch_slice(Y_data, test_idx, ba_idx, 'FC', 1) #이거 볼때마다 좀 헷갈리네유 그냥 실제값 뽑을려고 FC인거

        cost_MAE_val, cost_MSE_val, cost_MAPE_val, cost_MAE_hist_val, cost_MSE_hist_val, cost_MAPE_hist_val = sess.run([cost_MAE, cost_MSE, cost_MAPE, cost_MAE_hist, cost_MSE_hist, cost_MAPE_hist], feed_dict={S:S_test, C:C_test, E:E_test, Y:Y_test, BA: False, DR: FC_TE_KEEP_PROB, DISCRIMINATOR_BA: False, DISCRIMINATOR_DR:DISCRIMINATOR_TE_KEEP_PROB})
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
S_data, C_data, E_data, Y_data = input_data(0b111)

cr_idx = 0
kf = KFold(n_splits=CROSS_NUM, shuffle=True)
for train_idx, test_idx in kf.split(Y_data[:-CELL_SIZE]):
    print('CROSS VALIDATION: %d' % cr_idx)

    train_result = []
    test_result = []

    S = tf.placeholder("float32", [None, TIME_STAMP])
    C = tf.placeholder("float32", [None, None, SPARTIAL_NUM, TEMPORAL_NUM, 1])
    E = tf.placeholder("float32", [None, EXOGENOUS_NUM])
    Y = tf.placeholder("float32", [None, 1])
    BA = tf.placeholder(tf.bool)
    DR = tf.placeholder(tf.float32)
    DISCRIMINATOR_BA = tf.placeholder(tf.bool)
    DISCRIMINATOR_DR = tf.placeholder(tf.float32)
    last_epoch = tf.Variable(0, name=LAST_EPOCH_NAME)

    init()
    sess = tf.Session()
    cost_MAE, cost_MSE, cost_MAPE, train_D, train_G = model(S,C, E, Y, BA, DR, DISCRIMINATOR_BA,  DISCRIMINATOR_DR)
    writer_train = tf.summary.FileWriter("./tensorboard/adv_conv/train%d" % cr_idx, sess.graph)
    writer_test = tf.summary.FileWriter("./tensorboard/adv_conv/test%d" % cr_idx, sess.graph)
    cost_MAE_hist = tf.summary.scalar('cost_MAE', cost_MAE)
    cost_MSE_hist = tf.summary.scalar('cost_MSE', cost_MSE)
    cost_MAPE_hist = tf.summary.scalar('cost_MAPE', cost_MAPE)
    sess.run(tf.global_variables_initializer())

    # Saver and Restore
    saver = tf.train.Saver()
    CURRENT_POINT_DIR = CHECK_POINT_DIR + "ADV_CONV_" + str(cr_idx) + "/"
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

    train(S_data, C_data, E_data, Y_data, cost_MAE, cost_MSE, cost_MAPE, cost_MAE_hist, cost_MSE_hist, cost_MAPE_hist, train_D, train_G, train_idx, test_idx, cr_idx,  writer_train, writer_test, train_result, test_result, CURRENT_POINT_DIR, start_from)

    tf.reset_default_graph()

    output_data(train_result, test_result, 'adv_conv', cr_idx)

    cr_idx=cr_idx+1

    if (cr_idx == CROSS_ITERATION_NUM):
        break
