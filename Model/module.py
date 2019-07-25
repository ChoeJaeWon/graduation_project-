'''
----------------------------코드 설명----------------------------
-C-
module 파일: 헤더 파일
구성 요소
각종 hyper parameter

----------------------------고려 사항----------------------------
*2019 07 05
upStream_num, downStream_num은 데이터를 만들때 고려해서 실험 해줘야함
그래서 mu, md 라는 변수를 만들긴 하지만 나중에 파일에서 읽어와야함
*2019 07 20
과연 lstm을 미리 preprocess하는것이 효과가 있을까 그냥 fully data에서 받아와서 사용하는것이 더 효율적이지 않을까
input_data에서 안사용하는 데이터도 모두 받아와야하는데 이를 조건문으로 바꿔줄 필요가있다.
*2019 07 21
conv의 filter size와 layer등의 default값을 정해주어야 한다.
각각의 module이 batch를 제대로 반영하고 있는지 확인해야함
*2019 07 22
66 -> 80? 바꿔야합니다
tf.reduce_mean 이 이 실험에서 무슨 의미인가?

Q. batch slice에서 +를 통해 data index에 접근하는데 이때 최대치를 넘어버릴 수 있다
A. Cross validation 할때 Cell size만큼을 빼고 train_idx와 test_idx를 구해준다.

*2019 07 23
Q. conv에 뒤에 32채널을 1채널로 바꾸고 12개의 속도 데이터로 concat해주는 코드가 작성 되어야 한다.
A. 해결했다

Q. SPEED_MAX와 SPEED_MIN값 찾아야한다

Q. conv 채널수 늘리기 정확도가 fc랑 비슷하면 안된다.

'''

import tensorflow as tf
import numpy as np
from sklearn.model_selection import KFold
import csv

np.random.seed(777) #KFold 의 shuffle과 batch shuffle의 seed를 설정 해준다
tf.set_random_seed(777) #tf.random의 seed 설정

#Setting
#File name
FILEX_SPEED = '../Data/Speed/x_data_2016204_5min_60min_60min_only_speed.csv' #speed만 잘라낸 파일 이름(X data)
FILEX_EXO = '../Data/Exogenous/x_data_2016204_5min_60min_60min_8.csv' #exogenous(data 8)만 잘라낸 파일 이름(X data)
FILEX_CONV = '../Data/Convolution/x_data_2016204_5min_60min_60min_only_speed.csv' #preprocessing한 conv data 파일 이름(X data)
FILEY = '../Data/Y/y_data_2016204_5min_60min_60min.csv' #beta분 후 speed 파일 이름(Y data)
CHECK_POINT_DIR = '../Save/' #각 weight save 파일의 경로입니다.
LAST_EPOCH_NAME = 'last_epoch' #불러온 에폭에 대한 이름입니다.

#variable
TRAIN_NUM = 100 #traing 회수 [default 1000]
SPEED_MAX = 103 #data내의 최고 속도 [default 100] 
SPEED_MIN = 3 #data내의 최저 속도 [default 0]
CROSS_NUM = 5 #cross validation의 spilit 수
CROSS_ITERATION_NUM = 5 #cross validation의 반복수 (CROSS_NUM보다 작아야하며 독립적으로 생각됨)
BATCH_SIZE =  300 #1 epoch 당 batch의 개수 [default 300]
LEARNING_RATE = 0.001 #learning rate(모든 model, gan은 *2)
TRAIN_PRINT_INTERVAL = 10 #train 에서 mse값 출력 간격
TEST_PRINT_INTERVAL = 50 #test 에서 mae, mse, mape값 출력 간격


#Hyper Parameter(FC)
FC_LAYER_NUM = 4 #fc layer의 깊이 [default 3]
VECTOR_SIZE = 83 #fc와 lstm에 들어가는 vector의 크기 [default 83]
TIME_STAMP = 12 #lstm과 fc의 vector에서 고려해주는 시간 [default 12]
EXOGENOUS_NUM = VECTOR_SIZE-TIME_STAMP #exogenous로 들어가는 data의 개수 [default 73]
LAYER_UNIT_NUM = [VECTOR_SIZE, 256, 128, 64, 1] #fc에서 고려해줄 layer당 unit의 수 default[83, 64, 128, 64, 1]
FC_BATCH_NORM = True #fc 에서 batch normalization 을 사용할것인지 [default True]
FC_DROPOUT = True #fc 에서 drop out 을 사용할것인지 [default True]
FC_TR_KEEP_PROB = 0.8 #training 에서 dropout 비율
FC_TE_KEEP_PROB = 1.0 #testing 에서 dropout 비율

#Hyper Parameter(CONV)
POOLING = False #pooling을 사용할 것인지 [default True]
CONV_BATCH_NORM = True #conv 에서 batch normalization 을 사용할것인지 [default True]
CONV_LAYER_NUM = 3 #conv layer의 깊이 [default 3]
TEMPORAL_NUM = 12 #conv에서 고려할 시간 default 12]
UP_STREAM_NUM = 2 #conv에서 고려할 이후의 도로 개수들 [default 2]
DOWN_STREAM_NUM = 2 #conv에서 고려할 이전의 도로 개수들 [default 2]
SPARTIAL_NUM = DOWN_STREAM_NUM+UP_STREAM_NUM+1 #conv에서 고려할 총 도로의 수 + 타겟도로[default 13]
CHANNEL_NUM = [1, 64, 16, 32] #conv에서 고려해줄 channel 수 [default 1 64 16 32] **주의 1로 시작해서 1로 끝나야함 input과 ouput channel은 1개씩이기 때문
FILTER_SIZE_TEMPORAL = [3, 1, 3] #시간의 filter size [default 3 1 3]
FILTER_SIZE_SPATIAL = [3, 1, 3] #공간의 filter size [default 3 1 3]
LAST_LAYER_SIZE = 8


#Hyper Parameter(LSTM)
LSTM_TRAIN_NUM = 1500 #lstm의 training 수
HIDDEN_NUM = 32 #lstm의 hidden unit 수 [default 32]
FORGET_BIAS = 1.0 #lstm의 forget bias [default 1.0]
CELL_SIZE = 12 #lstm의 cell 개수 [default 12]

#Hyper Parameter(Discriminator)
DISCRIMINATOR_INPUT_NUM = 84
DISCRIMINATOR_LAYER_NUM = 4
DISCRIMINATOR_LAYER_UNIT_NUM = [DISCRIMINATOR_INPUT_NUM, 256, 128, 64, 1]
DISCRIMINATOR_BATCH_NORM = True
DISCRIMINATOR_DROPOUT = True
DISCRIMINATOR_TR_KEEP_PROB = 0.8 #training 에서 dropout 비율
DISCRIMINATOR_TE_KEEP_PROB = 1.0 #testing 에서 dropout 비율
DISCRIMINATOR_ALPHA = 0.01 #MSE 앞에 붙는 람다 term

fc_weights = [] #fc weight들의 크기는 layer의 길이에 따라 결정된다.
discriminator_weights = []
conv_weights = [] #conv weight들의 크기는 layer의 길이에 따라 결정된다.
convfc_weights = [] #conv 이후 fc의 weight
lstm_weights = [] #lstm weight들의 크기는 layer의 길이에 따라 결정된다.
lstm_biases = [] #lstm bias들의 크기는 layer의 길이에 따라 결정된다.



#weight를 만들어준다.
def init():
    #모든 weight를 clear 해준다.
    fc_weights.clear()
    conv_weights.clear()
    convfc_weights.clear()
    lstm_weights.clear()
    lstm_biases.clear()
    discriminator_weights.clear()

    # fc weight 초기화
    for layer_idx in range(1, FC_LAYER_NUM+1):
        fc_weights.append(init_weights([LAYER_UNIT_NUM[layer_idx - 1], LAYER_UNIT_NUM[layer_idx]]))

    # conv weight 초기화
    for layer_idx in range(1,CONV_LAYER_NUM+1):
        conv_weights.append(init_weights([FILTER_SIZE_SPATIAL[layer_idx-1], FILTER_SIZE_TEMPORAL[layer_idx-1], CHANNEL_NUM[layer_idx-1], CHANNEL_NUM[layer_idx]]))
    convfc_weights.append(init_weights([LAST_LAYER_SIZE*CHANNEL_NUM[CONV_LAYER_NUM],TIME_STAMP]))

    # lstm weight 초기화
    lstm_weights.append(init_weights([HIDDEN_NUM, 1]))
    lstm_biases.append(init_weights([1]))

    # discriminator weight 초기화
    for layer_idx in range(1, DISCRIMINATOR_LAYER_NUM+1):
        discriminator_weights.append(init_weights([DISCRIMINATOR_LAYER_UNIT_NUM[layer_idx - 1], DISCRIMINATOR_LAYER_UNIT_NUM[layer_idx]]))


#shper를 input으로 받아 weight를 initailization 해줌
def init_weights(input_shape):
    return tf.Variable(tf.random_normal(input_shape, stddev=0.01)) #name은 임의로 정했음


#file을 numpy array 데이터로 받아옴
def fileToData(fileName):
    data_list = []
    File = open(fileName, 'r')
    FileData = csv.reader(File)
    for line in FileData:
        temp_line = []
        for line_idx in range(len(line)):
            temp_line.append(float(line[line_idx]))
        data_list.append(temp_line)
    File.close()
    return np.array(data_list)


#input data를 만들어줌
#type 2진법 ex) 0b111 4자리는 Speed, 2자리는 Conv, 1자리는 exogenous
def input_data(type):
    S_data = np.array([])
    C_data = np.array([])
    E_data = np.array([])
    #file을 numpy로 바꿔줌
    #&로 각 자리를 비교해준다.
    if type & 0b100 != False:
        S_data = fileToData(FILEX_SPEED) #only speed 데이터(시간순)
    if type & 0b10 != False:
        C_data = fileToData(FILEX_CONV) #conv 데이터(행(공간), 열(시간))
    if type & 0b1 != False:
        E_data = fileToData(FILEX_EXO)  # 외부요소만 자른 데이터
    Y_data = fileToData(FILEY) #실재값 데이터

    return S_data, C_data, E_data, Y_data

#에러계산식
def MAE(y_test, y_pred):
    y_test_orig = y_test * (SPEED_MAX - SPEED_MIN + 1e-7) + SPEED_MIN
    y_pred_orig = y_pred * (SPEED_MAX - SPEED_MIN + 1e-7) + SPEED_MIN
    return tf.reduce_mean(tf.abs(y_test_orig - y_pred_orig))
def MSE(y_test, y_pred):
    y_test_orig = y_test * (SPEED_MAX - SPEED_MIN + 1e-7) + SPEED_MIN
    y_pred_orig = y_pred * (SPEED_MAX - SPEED_MIN + 1e-7) + SPEED_MIN
    return tf.reduce_mean(tf.square((y_test_orig - y_pred_orig)))
def MAPE(y_test, y_pred):
    y_test_orig = y_test * (SPEED_MAX - SPEED_MIN + 1e-7) + SPEED_MIN
    y_pred_orig = y_pred * (SPEED_MAX - SPEED_MIN + 1e-7) + SPEED_MIN
    return tf.reduce_mean(tf.abs((y_test_orig - y_pred_orig) / y_test_orig)) * 100


#FC_model로 input으로 CNN output이 output으로 예측 속도값이 나온다.
def FC_model(S, E, BA, DR):
    batch_prob = BA
    dropout_prob = DR
    for layer_idx in range(FC_LAYER_NUM):
        if layer_idx != 0:
            layer = tf.matmul(layer, fc_weights[layer_idx])
        else:
            layer = tf.matmul(tf.concat([S, E], axis=1), fc_weights[layer_idx])

        if FC_BATCH_NORM == True:
            layer = tf.layers.batch_normalization(layer, center=True, scale=True, training=batch_prob)

        layer = tf.nn.relu(layer)

        if FC_DROPOUT == True:
            tf.nn.dropout(layer, keep_prob=dropout_prob)

    return layer


#CONV network로 input으로 시공간 입력이 output으로 layer가 나온다
def CNN_model(X, BA):
    batch_prob = BA

    for layer_idx in range(CONV_LAYER_NUM):
        if layer_idx != 0:
            layer = tf.nn.conv2d(layer, conv_weights[layer_idx], strides=[1, 1, 1, 1], padding='VALID')
        else:
            layer = tf.nn.conv2d(X, conv_weights[layer_idx], strides=[1, 1, 1, 1], padding='VALID')

        if CONV_BATCH_NORM == True:
            layer = tf.layers.batch_normalization(layer, center=True, scale= True, training=batch_prob)
        layer = tf.nn.relu(layer)
        if POOLING == True and layer_idx != (CONV_LAYER_NUM-1): #마지막 layer는 pooling안함
            layer = tf.nn.avg_pool(layer, ksize=[1,2,2,1], strides=[1,1,1,1])

    layer = tf.reshape(layer, shape=[BATCH_SIZE, CHANNEL_NUM[CONV_LAYER_NUM]*LAST_LAYER_SIZE])
    layer = tf.matmul(layer, convfc_weights[0])
    layer = tf.nn.relu(layer)

    #**fc 하나 추가해 주어야함

    return layer

#LSTM network로 input으로 X(batch_size * speed_size * cell_size)와 E(batch_size * exogenous_size * cell_size)가 들어온다.
#output으로 마지막 예측값만 내놓는다.
#현재: time stamp 12, vector_size 66, cell_size 12, output 1
#추후에 실험 1,2 해봐야함
#실험1: time stamp 1, vector_size 6?7?, cell_size 12, output 1
#실험2: time stamp 12, vector_size 66, cell_size 12, output 12
def LSTM_model(S, E):
    # 66(vector_size) * 12(cell size)를 나눠줌
    #X,E는 같은 시간 끼리 합쳐줌
    x = tf.unstack(tf.concat([S, E], axis=2), axis=0)

    lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=HIDDEN_NUM, forget_bias=FORGET_BIAS)
    #lstm_cell = tf.contrib.rnn.BasicLSTMCell(HIDDEN_NUM, forget_bias=FORGET_BIAS)

    outputs, _ = tf.nn.static_rnn(cell=lstm_cell, inputs=x, dtype= tf.float32 )
    #outputs, _ = tf.contrib.rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], lstm_weights[0]) + lstm_biases[0]

#discriminator 의 X는 y 와 predicted y 가 concatenated 되어서 들어온 13짜리 X입니다. 기존의 S랑 다름 -> 매우 중요
def Discriminator_model(X, E, DISCRIMINATOR_BA, DISCRIMINATOR_DR):
    discriminator_batch_prob = DISCRIMINATOR_BA
    discriminator_dropout_prob = DISCRIMINATOR_DR


    for layer_idx in range(DISCRIMINATOR_LAYER_NUM): #same as FC_LAYER_NUM
        if layer_idx != 0:
            layer = tf.matmul(layer, discriminator_weights[layer_idx])
        else:
            layer = tf.matmul(tf.concat([X, E], axis=1), discriminator_weights[layer_idx])

        if DISCRIMINATOR_BATCH_NORM == True:
            layer = tf.layers.batch_normalization(layer, center=True, scale=True, training=discriminator_batch_prob)
        # 마지막 레이어는 Sigmoid logistic regression, 마지막 출력이 1이라는 가정 하에 작성합니다
        if layer_idx == DISCRIMINATOR_LAYER_NUM - 1:
            layer = tf.nn.sigmoid(layer)
        else:
            layer = tf.nn.relu(layer)

        if DISCRIMINATOR_DROPOUT == True:
            tf.nn.dropout(layer, keep_prob=discriminator_dropout_prob)

    return layer


#type에 따라 다른 batch slice 결과를 내어준다.
#da_idx는 cross validation해서 나온 idx의 집합
#ba_idx는 batch의 idx
#cell size는 conv+lstm에서 고려해줘야할 conv의 수
def batch_slice(data, data_idx, batch_idx, slice_type, cell_size):
    #fc X input data와 fc, conv의 y output data
    if slice_type == 'FC':
        slice_data = data[data_idx[batch_idx * BATCH_SIZE: (batch_idx + 1) * BATCH_SIZE]]

    #conv X input data, cell size에 따라 연속된 conv input을 뽑을수있다.(lstm의 input으로 들어가기 위한)
    elif slice_type == 'CONV':
        for cell_idx in range(cell_size):
            for idx in range(batch_idx * BATCH_SIZE, (batch_idx + 1) * BATCH_SIZE):
                start_idx = data_idx[idx]
                if idx == batch_idx * BATCH_SIZE:
                    temp = data[(start_idx + cell_idx) * SPARTIAL_NUM: ((start_idx + cell_idx + 1) * SPARTIAL_NUM)].reshape(1, 1, SPARTIAL_NUM, TEMPORAL_NUM, 1)
                else:
                    temp = np.append(temp, data[(start_idx + cell_idx) * SPARTIAL_NUM: ((start_idx + cell_idx + 1) * SPARTIAL_NUM)].reshape(1, 1, SPARTIAL_NUM, TEMPORAL_NUM, 1), axis=1)

            if cell_idx == 0:
                slice_data = temp
            else:
                slice_data = np.append(slice_data, temp, axis=0)
    #lstm X input data
    elif slice_type ==  'LSTM':
        for idx in range(batch_idx * BATCH_SIZE, (batch_idx + 1) * BATCH_SIZE):
            start_idx = data_idx[idx]
            if idx == batch_idx * BATCH_SIZE:
                slice_data = data[start_idx: start_idx + CELL_SIZE].reshape(CELL_SIZE, 1 , -1) #마지막이 -1인 이유(speed의 경우 12 이고 exogenous의 경우 71이기 때문)
            else:
                slice_data = np.append(slice_data,  data[start_idx: start_idx + CELL_SIZE].reshape(CELL_SIZE, 1, -1), axis=1)
    #lstm의 output data(60분 후를 뽑아야 하기때문)
    elif slice_type == 'LSTMY':
        slice_data = data[data_idx[batch_idx * BATCH_SIZE: (batch_idx + 1) * BATCH_SIZE]+ CELL_SIZE-1]

    else:
        print('ERROR: slice type error\n')

    return slice_data

#train과 test에서 얻은 결과를 file로 만든다.
#file_name에 실행하는 코드의 이름을 적는다 ex)adv_conv_lstm
def output_data(train_result, test_result, file_name, cr_idx):
    #train output
    outputfile = open('../Result/' + file_name + str(cr_idx) + '_tr' + '.csv', 'w', newline='')
    output = csv.writer(outputfile)

    for tr_idx in range(len(train_result)):
        output.writerow([str(train_result[tr_idx])])

    outputfile.close()

    # test output
    outputfile = open('../Result/' + file_name + str(cr_idx) + '_te' + '.csv', 'w', newline='')
    output = csv.writer(outputfile)

    for te_idx in range(len(test_result)):
        output.writerow([str(test_result[te_idx][0]),str(test_result[te_idx][1]),str(test_result[te_idx][2])])

    outputfile.close()


