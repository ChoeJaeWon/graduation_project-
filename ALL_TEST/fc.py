'''
----------------------------코드 설명----------------------------
-C-
1.FC에 해당하는 코드로
fc를 구현함
----------------------------고려 사항----------------------------


'''
from module import *
import os

#FC를 구현
def model(S, E, Y, BA, DR):
    layer = FC_model(S, E, BA, DR)
    cost_MAE = MAE(Y, layer)
    cost_MSE = MSE(Y, layer)
    cost_MAPE = MAPE(Y, layer)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimal = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cost_MSE)

    return cost_MAE, cost_MSE, cost_MAPE, optimal, tf.reduce_mean(layer)

#training 해준다.
def train(S_data, E_data, Y_data, cost_MAE, cost_MSE, cost_MAPE, cost_MAE_hist, cost_MSE_hist, cost_MAPE_hist, optimal, train_idx, test_idx, cr_idx, writer_train, writer_test, train_result, test_result, CURRENT_POINT_DIR, start_from,prediction):
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
            E_train = batch_slice(E_data, train_idx, ba_idx, 'FC', 1)
            Y_train = batch_slice(Y_data, train_idx, ba_idx, 'FC', 1)

            cost_MSE_val, cost_MSE_hist_val, _= sess.run([cost_MSE, cost_MSE_hist, optimal], feed_dict={S:S_train, E:E_train, Y: Y_train, BA: True, DR: FC_TR_KEEP_PROB})
            epoch_cost += cost_MSE_val
            writer_train.add_summary(cost_MSE_hist_val, global_step_tr)
            global_step_tr += 1

        # 설정 interval당 train과 test 값을 출력해준다.
        if tr_idx % TRAIN_PRINT_INTERVAL == 0:
            train_result.append(epoch_cost/BATCH_NUM)
            print("Train Cost %d: %lf" % (tr_idx, epoch_cost / BATCH_NUM))
        if tr_idx == TRAIN_NUM-1:

            print("Saving network...")
            sess.run(last_epoch.assign(tr_idx + 1))
            if not os.path.exists(CURRENT_POINT_DIR):
                os.makedirs(CURRENT_POINT_DIR)
            saver.save(sess, CURRENT_POINT_DIR + "/model", global_step=tr_idx, write_meta_graph=False)

            global_step_te=test(S_data, E_data, Y_data, cost_MAE, cost_MSE, cost_MAPE, cost_MAE_hist, cost_MSE_hist, cost_MAPE_hist, test_idx, tr_idx, global_step_te, cr_idx, writer_test, test_result,prediction)

        #cross validation의 train_idx를 shuffle해준다.
        np.random.shuffle(train_idx)


#testing 해준다.
def test(S_data, E_data, Y_data, cost_MAE, cost_MSE, cost_MAPE, cost_MAE_hist, cost_MSE_hist, cost_MAPE_hist, test_idx, tr_idx, global_step_te, cr_idx, writer_test, test_result,prediction):
    BATCH_NUM = int(len(test_idx))
    print("test batch number: %d" % BATCH_NUM)

    for ba_idx in range(BATCH_NUM):
        # Batch Slice
        S_test = np.reshape(S_data[ba_idx], (1,12))
        E_test = np.reshape(E_data[ba_idx], (1,83))
        Y_test = np.reshape(Y_data[ba_idx], (1,1))

        mae, mse, mape, s = sess.run([cost_MAE, cost_MSE, cost_MAPE, prediction], feed_dict={S:S_test, E:E_test, Y:Y_test, BA: False, DR: FC_TE_KEEP_PROB})



        global_step_te+=1
        test_result.append([mae, mse, mape, s])


    return global_step_te




###################################################-MAIN-###################################################
S_data, _, E_data, Y_data = input_data(0b101) #speed, exogenous 사용


cr_idx = 0
kf = KFold(n_splits=CROSS_NUM, shuffle=True)
for train_idx, test_idx in Week_CrossValidation():
    print('CROSS VALIDATION: %d' % cr_idx)

    train_result = []
    test_result = []

    S = tf.placeholder("float32", [None, TIME_STAMP])
    E = tf.placeholder("float32", [None, EXOGENOUS_NUM])
    Y = tf.placeholder("float32", [None, 1])
    BA = tf.placeholder(tf.bool)
    DR = tf.placeholder(tf.float32)
    last_epoch = tf.Variable(0, name=LAST_EPOCH_NAME)

    sess = tf.Session()

    init()
    cost_MAE, cost_MSE, cost_MAPE, optimal, prediction = model(S, E, Y, BA, DR)
    writer_train = tf.summary.FileWriter("./tensorboard/fc/train%d" % cr_idx, sess.graph)
    writer_test = tf.summary.FileWriter("./tensorboard/fc/test%d" % cr_idx, sess.graph)
    cost_MAE_hist = tf.summary.scalar('cost_MAE', cost_MAE)
    cost_MSE_hist = tf.summary.scalar('cost_MSE', cost_MSE)
    cost_MAPE_hist = tf.summary.scalar('cost_MAPE', cost_MAPE)
    sess.run(tf.global_variables_initializer())

    # Saver and Restore
    saver = tf.train.Saver()

    CURRENT_POINT_DIR = CHECK_POINT_DIR + "FC_" + str(cr_idx) + "/"
    checkpoint = tf.train.get_checkpoint_state(CURRENT_POINT_DIR)

    if RESTORE_FLAG:
        if checkpoint and checkpoint.model_checkpoint_path:
            #try:
                saver.restore(sess, checkpoint.model_checkpoint_path)
                print("Successfully loaded:", checkpoint.model_checkpoint_path)
            #except:
            #    print("Error on loading old network weights")
        else:
            print("Could not find old network weights")

    start_from = sess.run(last_epoch)
    # train my model
    print('Start learning from:', start_from)

    train(S_data, E_data, Y_data, cost_MAE, cost_MSE, cost_MAPE, cost_MAE_hist, cost_MSE_hist, cost_MAPE_hist, optimal, train_idx, np.array([i for i in range(35350)]), cr_idx, writer_train, writer_test, train_result, test_result, CURRENT_POINT_DIR, start_from,prediction)

    tf.reset_default_graph()

    output_data(train_result, test_result, 'fc', cr_idx)

    cr_idx = cr_idx + 1

    if (cr_idx == CROSS_ITERATION_NUM):
        break