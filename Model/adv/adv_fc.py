'''
----------------------------코드 설명----------------------------

----------------------------고려 사항----------------------------


'''
from Model.module import *

#FC를 구현
def model(X, E, Y):
    layer = FC_model(X, E)

    cost_MAE = MAE(Y, layer)
    cost_MSE = MSE(Y, layer)
    cost_MAPE = MAPE(Y, layer)
    adv_y = np.append(X, Y, axis=1)
    adv_g = np.append(X, layer, axis=1)
    loss_D = -tf.reduce_mean(tf.log(Discriminator_model(adv_y, E)) + tf.log(1 - Discriminator_model(adv_g, E)))
    loss_G = -tf.reduce_mean(
        tf.log(Discriminator_model(adv_g, E))) + DISCRIMINATOR_ALPHA * cost_MSE  # MSE 는 0~ t까지 있어봤자 같은 값이다.

    train_D = tf.train.AdamOptimizer(learning_rate=0.0002).minimize(loss_D)
    train_G = tf.train.AdamOptimizer(learning_rate=0.0002).minimize(loss_G)

    return cost_MAE, cost_MSE, cost_MAPE, train_D, train_G

#training 해준다.
def train(X_data, E_data, Y_data, cost_MSE, train_D, train_G, train_idx):
    BATCH_NUM = int(len(train_idx) / BATCH_SIZE)
    for tr_idx in range(TRAIN_NUM):
        epoch_cost = 0.0
        for ba_idx in range(BATCH_NUM):
            #Batch Slice
            X_train = batch_slice(X_data, train_idx, ba_idx, 'FC', 1)
            E_train = batch_slice(E_data, train_idx, ba_idx, 'FC', 1)
            Y_train = batch_slice(Y_data, train_idx, ba_idx, 'FC', 1)
            _= sess.run([train_D], feed_dict={X:X_train, E:E_train, Y: Y_train, batch_prob: True, dropout_prob: FC_TR_KEEP_PROB, discriminator_batch_prob:True, discriminator_dropout_prob: DISCRIMINATOR_TR_KEEP_PROB})
            cost_MSE_val, _= sess.run([cost_MSE, train_G], feed_dict={X:X_train, E:E_train, Y: Y_train, batch_prob: True, dropout_prob: FC_TR_KEEP_PROB, discriminator_batch_prob:True, discriminator_dropout_prob: DISCRIMINATOR_TR_KEEP_PROB})
            epoch_cost += cost_MSE_val

        #한 epoch당 cost_MSE의 평균을 구해준다.
        print("Train Cost%d: %lf\n", tr_idx, epoch_cost/BATCH_NUM)

        #cross validation의 train_idx를 shuffle해준다.
        np.random.shuffle(train_idx)


#testing 해준다.
def test(X_data, E_data, Y_data, cost_MAE, cost_MSE, cost_MAPE, test_idx, cr_idx):
    BATCH_NUM = int(len(test_idx) / BATCH_SIZE)
    mae = 0.0
    mse = 0.0
    mape = 0.0
    for ba_idx in range(BATCH_NUM):
        # Batch Slice
        X_test = batch_slice(X_data, test_idx, ba_idx, 'FC', 1)
        E_test = batch_slice(E_data, test_idx, ba_idx, 'FC', 1)
        Y_test = batch_slice(Y_data, test_idx, ba_idx, 'FC', 1)

        cost_MAE_val, cost_MSE_val, cost_MAPE_val = sess.run([cost_MAE, cost_MSE, cost_MAPE], feed_dict={X:X_test, E:E_test, Y:Y_test, batch_prob: False, dropout_prob: FC_TE_KEEP_PROB, discriminator_batch_prob: False, discriminator_dropout_prob:DISCRIMINATOR_TE_KEEP_PROB})
        mae += cost_MAE_val
        mse += cost_MSE_val
        mape += cost_MAPE_val


    print("Test Cost%d: MAE(%lf) MSE(%lf) MAPE(%lf)\n", cr_idx, mae/BATCH_NUM, mse/BATCH_NUM, mape/BATCH_NUM)




###################################################-MAIN-###################################################
init()
X_data, _, Y_data, E_data = input_data()

X = tf.placeholder("float32", [None, TIME_STAMP])
E = tf.placeholder("float32", [None, EXOGENOUS_NUM])
Y = tf.placeholder("float32", [None, 1])

sess = tf.Session()
sess.run(tf.global_variables_initializer())
cost_MAE, cost_MSE, cost_MAPE, train_D, train_G = model(X, E, Y)

cr_idx = 0
kf = KFold(n_splits=CROSS_NUM, shuffle=True)
for train_idx, test_idx in kf.split(Y_data[:-CELL_SIZE]):
    train(X_data, E_data, Y_data, cost_MSE, train_D, train_G, train_idx)
    test(X_data, E_data, Y_data, cost_MAE, cost_MSE, cost_MAPE, test_idx, cr_idx)
    cr_idx=cr_idx+1