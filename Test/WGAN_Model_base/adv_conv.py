'''
----------------------------코드 설명----------------------------

----------------------------고려 사항----------------------------


'''
from module import *
import os

def model_base(C, E, Y,BA,DR, DISCRIMINATOR_BA, DISCRIMINATOR_DR):
    for idx in range(TIME_STAMP):
        if idx == 0:
            layer = tf.reshape(FC_model(CNN_model(C[idx], BA), E[idx], BA, DR), [1, BATCH_SIZE])  # 마지막에 conv 에서는 timestamp
        else:
            layer = tf.concat([layer, tf.reshape(FC_model(CNN_model(C[idx], BA, True), E[idx], BA, DR, True), [1, BATCH_SIZE])], axis=0)

    Y = tf.reshape(Y, [12, BATCH_SIZE]) #일단 통일시켜놨기 떄문에 어쩔 수 없는 부분
    # 3차원 오차, MAE, MAPE는 Train 에서는 필요 없음
    # cost_MAE = MAE(Y, layer)
    train_MSE = MSE(Y, layer)
    # cost_MAPE = MAPE(Y, layer)
    cost_MAE = MAE(Y[TIME_STAMP - 1], layer[TIME_STAMP - 1]) #실제로는 직후부터 60분 뒤 까지의 예측이므로
    cost_MSE = MSE(Y[TIME_STAMP - 1], layer[TIME_STAMP - 1])
    cost_MAPE = MAPE(Y[TIME_STAMP - 1], layer[TIME_STAMP - 1])

    layer = tf.transpose(layer, perm=[1, 0])  # lstm에 unstack 이 있다면, 여기서는 transpose를 해주는 편이 위의 계산할 때 편할 듯
    Y = tf.transpose(Y, perm=[1, 0])  # y는 처음부터 잘 만들면 transpose할 필요 없지만, x랑 같은 batchslice를 하게 해주려면 이렇게 하는 편이 나음.

    loss_D = -tf.reduce_mean(
        tf.log(Discriminator_model(Y, E[TIME_STAMP - 1], DISCRIMINATOR_BA, DISCRIMINATOR_DR)) + tf.log(
            1 - Discriminator_model(layer, E[TIME_STAMP - 1], DISCRIMINATOR_BA, DISCRIMINATOR_DR, True)))
    loss_G = -tf.reduce_mean(tf.log(Discriminator_model(layer, E[TIME_STAMP - 1], DISCRIMINATOR_BA,
                                                        DISCRIMINATOR_DR, True)))  + DISCRIMINATOR_ALPHA * train_MSE # MSE 는 0~ t까지 있어봤자 같은 값이다.
    '''
    loss_G_MSE = -tf.reduce_mean(tf.log(Discriminator_model(layer, E[TIME_STAMP - 1], DISCRIMINATOR_BA,
                                                        DISCRIMINATOR_DR, True))) + DISCRIMINATOR_ALPHA * train_MSE
    '''
    vars_D = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                               scope='discriminator_fc') #여기는 하나로 함수 합쳤음
    vars_G = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                               scope='generator_fc') + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                               scope='generator_conv')  #다양해지면 여기가 모델마다 바뀜

    D_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='discriminator_fc')
    with tf.control_dependencies(D_update_ops):
        train_D = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss_D, var_list=[vars_D, discriminator_weights]) #이 부분은 모델별로 고정

    G_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='generator_fc') + tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='generator_conv')
    with tf.control_dependencies(G_update_ops):
        train_G = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss_G, var_list=[vars_G ,fc_weights, conv_weights, convfc_weights])
    '''
    G_MSE_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='generator_fc')
    with tf.control_dependencies(G_MSE_update_ops):
        train_G_MSE = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss_G_MSE, var_list=[vars_G, fc_weights])
    '''
    return train_MSE,cost_MAE, cost_MSE, cost_MAPE, train_D, train_G , loss_G#, train_G_MSE


#FC와 CNN을 합침
def model(S, C, E, Y, BA, DR, DISCRIMINATOR_BA,  DISCRIMINATOR_DR):
    layer = CNN_model(C[0], BA)
    layer = FC_model(layer, E , BA, DR)

    cost_MAE = MAE(Y, layer)
    cost_MSE = MSE(Y, layer)
    cost_MAPE = MAPE(Y, layer)

    adv_y= tf.concat([S, Y], axis=1)
    adv_g= tf.concat([S, layer], axis=1)
    loss_D = -tf.reduce_mean(tf.log(Discriminator_model(adv_y,E,DISCRIMINATOR_BA, DISCRIMINATOR_DR)) + tf.log(1- Discriminator_model(adv_g,E,DISCRIMINATOR_BA, DISCRIMINATOR_DR)))
    loss_G = -tf.reduce_mean(tf.log(Discriminator_model(adv_g,E,DISCRIMINATOR_BA, DISCRIMINATOR_DR))) + DISCRIMINATOR_ALPHA*cost_MSE #MSE 는 0~ t까지 있어봤자 같은 값이다.

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_D = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE*2).minimize(loss_D)
        train_G = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE*2).minimize(loss_G)

    return cost_MAE, cost_MSE, cost_MAPE, train_D, train_G

#training 해준다.
def train(C_data, E_data, Y_data, cost_MAE, cost_MSE, cost_MAPE, cost_MAE_hist, cost_MSE_hist, cost_MAPE_hist, train_D, train_G, train_idx, test_idx, cr_idx,  writer_train, writer_test, train_result, test_result, CURRENT_POINT_DIR, start_from):
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
            if LATENT_VECTOR_FLAG:
                C_train = batch_slice(C_data, train_idx, ba_idx, 'CONV', CELL_SIZE)
                E_train = batch_slice(E_data, train_idx, ba_idx, 'ADV_FC')
                Y_train = batch_slice(Y_data, train_idx, ba_idx, 'ADV_FC')
            else:
                C_train = batch_slice(C_data, train_idx, ba_idx, 'CONV', 1)
                E_train = batch_slice(E_data, train_idx, ba_idx, 'FC', 1)
                Y_train = batch_slice(Y_data, train_idx, ba_idx, 'FC', 1)
            if tr_idx > OPTIMIZED_EPOCH_CONV + 10:
                _= sess.run([train_D], feed_dict={C:C_train, E:E_train, Y: Y_train, BA: True, DR: FC_TR_KEEP_PROB, DISCRIMINATOR_BA:True, DISCRIMINATOR_DR: DISCRIMINATOR_TR_KEEP_PROB})
            if (tr_idx <= OPTIMIZED_EPOCH_CONV + 10) | (tr_idx > OPTIMIZED_EPOCH_CONV + 30):
                cost_MSE_val, cost_MSE_hist_val, _, loss= sess.run([cost_MSE, cost_MSE_hist, train_G, loss_G], feed_dict={C: C_train, E: E_train, Y: Y_train, BA: True,DR: FC_TR_KEEP_PROB, DISCRIMINATOR_BA:True, DISCRIMINATOR_DR: DISCRIMINATOR_TR_KEEP_PROB})
                epoch_cost += cost_MSE_val
                epoch_loss += loss
                writer_train.add_summary(cost_MSE_hist_val, global_step_tr)
            global_step_tr += 1

        # 설정 interval당 train과 test 값을 출력해준다.
        if tr_idx % TRAIN_PRINT_INTERVAL == 0:
            train_result.append(epoch_cost / BATCH_NUM)
            print("Train Cost %d: %lf" % (tr_idx, epoch_cost / BATCH_NUM))
            print("Train loss %d: %lf" % (tr_idx, epoch_loss / BATCH_NUM))
        if (tr_idx+1) % TEST_PRINT_INTERVAL == 0:

            print("Saving network...")
            sess.run(last_epoch.assign(tr_idx + 1))
            if not os.path.exists(CURRENT_POINT_DIR):
                os.makedirs(CURRENT_POINT_DIR)
            saver.save(sess, CURRENT_POINT_DIR + "/model", global_step=tr_idx, write_meta_graph=False)

            global_step_te = test(C_data, E_data, Y_data, cost_MAE, cost_MSE, cost_MAPE, cost_MAE_hist, cost_MSE_hist, cost_MAPE_hist, test_idx, tr_idx, global_step_te, cr_idx, writer_test, test_result)

        #cross validation의 train_idx를 shuffle해준다.
        np.random.shuffle(train_idx)


#testing 해준다. GAN과는 상관없이 최종 MAE, MSE, MAPE만 뽑아준다.
def test(C_data, E_data, Y_data, cost_MAE, cost_MSE, cost_MAPE, cost_MAE_hist, cost_MSE_hist, cost_MAPE_hist, test_idx, tr_idx, global_step_te, cr_idx, writer_test, test_result):
    BATCH_NUM = int(len(test_idx) / BATCH_SIZE)
    mae = 0.0
    mse = 0.0
    mape = 0.0
    for ba_idx in range(BATCH_NUM):
        # Batch Slice
        if LATENT_VECTOR_FLAG:
            C_test = batch_slice(C_data, test_idx, ba_idx, 'CONV', CELL_SIZE)
            E_test = batch_slice(E_data, test_idx, ba_idx, 'ADV_FC')
            Y_test = batch_slice(Y_data, test_idx, ba_idx, 'ADV_FC')
        else:
            C_test = batch_slice(C_data, test_idx, ba_idx, 'CONV', 1)
            E_test = batch_slice(E_data, test_idx, ba_idx, 'FC', 1)
            Y_test = batch_slice(Y_data, test_idx, ba_idx, 'FC', 1) #이거 볼때마다 좀 헷갈리네유 그냥 실제값 뽑을려고 FC인거

        cost_MAE_val, cost_MSE_val, cost_MAPE_val, cost_MAE_hist_val, cost_MSE_hist_val, cost_MAPE_hist_val = sess.run([cost_MAE, cost_MSE, cost_MAPE, cost_MAE_hist, cost_MSE_hist, cost_MAPE_hist], feed_dict={C:C_test, E:E_test, Y:Y_test, BA: False, DR: FC_TE_KEEP_PROB, DISCRIMINATOR_BA: False, DISCRIMINATOR_DR:DISCRIMINATOR_TE_KEEP_PROB})
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
_, C_data, E_data, Y_data = input_data(0b011)

if LATENT_VECTOR_FLAG:
    cr_idx = 0
    kf = KFold(n_splits=CROSS_NUM, shuffle=True)  # 35387 / 12 = 2949
    for train_idx, test_idx in Week_CrossValidation():
        print('CROSS VALIDATION: %d' % cr_idx)

        train_result = []
        test_result = []

        C = tf.placeholder("float32", [None, None, SPARTIAL_NUM, TEMPORAL_NUM, 1])
        E = tf.placeholder("float32", [TIME_STAMP, None, EXOGENOUS_NUM])
        Y = tf.placeholder("float32", [TIME_STAMP, None, 1])

        BA = tf.placeholder(tf.bool)
        DR = tf.placeholder(tf.float32)
        DISCRIMINATOR_BA = tf.placeholder(tf.bool)
        DISCRIMINATOR_DR = tf.placeholder(tf.float32)
        last_epoch = tf.Variable(OPTIMIZED_EPOCH_CONV+1, name=LAST_EPOCH_NAME)  # 받아올 방법이 없네..

        init()
        sess = tf.Session()
        # 여기서는 모델만 외부 플래그, 그냥 train까지 외부 플래그 해도 됨
        train_MSE, cost_MAE, cost_MSE, cost_MAPE, train_D, train_G, loss_G = model_base(C, E, Y, BA, DR, DISCRIMINATOR_BA, DISCRIMINATOR_DR)
        writer_train = tf.summary.FileWriter("./tensorboard/adv_conv/train%d" % cr_idx, sess.graph)
        writer_test = tf.summary.FileWriter("./tensorboard/adv_conv/test%d" % cr_idx, sess.graph)
        train_MSE_hist = tf.summary.scalar('train_MSE', train_MSE)
        cost_MAE_hist = tf.summary.scalar('cost_MAE', cost_MAE)
        cost_MSE_hist = tf.summary.scalar('cost_MAE', cost_MSE)
        cost_MAPE_hist = tf.summary.scalar('cost_MAPE', cost_MAPE)
        sess.run(tf.global_variables_initializer())

        # Saver and Restore
        if RESTORE_GENERATOR_FLAG:
            fc_batch_norm_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                      scope='generator_fc')
            conv_batch_norm_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                        scope='generator_conv')
            variables_to_restore = fc_weights +conv_weights+convfc_weights+ fc_batch_norm_weights + conv_batch_norm_weights
            saver = tf.train.Saver(variables_to_restore)
        else:
            saver = tf.train.Saver()

        CURRENT_POINT_DIR = CHECK_POINT_DIR + "ADV_CONV_" + str(cr_idx) + "/"
        checkpoint = tf.train.get_checkpoint_state(CURRENT_POINT_DIR)

        if RESTORE_FLAG:
            if checkpoint and checkpoint.model_checkpoint_path:
                # try:
                saver.restore(sess, checkpoint.model_checkpoint_path)
                print("Successfully loaded:", checkpoint.model_checkpoint_path)
            # except:
            #    print("Error on loading old network weights")
            else:
                print("Could not find old network weights")

        start_from = sess.run(last_epoch)
        # train my model
        print('Start learning from:', start_from)

        # train도 외부에서 FLAG해도됨. 지금은 안에 조건문이 있음
        train(C_data, E_data, Y_data, cost_MAE, cost_MSE, cost_MAPE, cost_MAE_hist, cost_MSE_hist, cost_MAPE_hist,
              train_D, train_G, train_idx, test_idx, cr_idx, writer_train, writer_test, train_result, test_result,
              CURRENT_POINT_DIR, start_from)

        tf.reset_default_graph()

        output_data(train_result, test_result, 'adv_conv', cr_idx)

        cr_idx = cr_idx + 1

        if (cr_idx == CROSS_ITERATION_NUM):
            break
else:
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

        train( C_data, E_data, Y_data, cost_MAE, cost_MSE, cost_MAPE, cost_MAE_hist, cost_MSE_hist, cost_MAPE_hist, train_D, train_G, train_idx, test_idx, cr_idx,  writer_train, writer_test, train_result, test_result, CURRENT_POINT_DIR, start_from)

        tf.reset_default_graph()

        output_data(train_result, test_result, 'adv_conv', cr_idx)

        cr_idx=cr_idx+1

        if (cr_idx == CROSS_ITERATION_NUM):
            break
