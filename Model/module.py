'''
module 파일: 헤더 파일
구성 요소
각종 hyper parameter

'''
import tensorflow as tf
import numpy as np
from sklearn.model_selection import KFold
import csv

np.random.seed(777) #KFold 의 shuffle과 batch shuffle의 seed를 설정 해준다

#Setting
#File name
FILEX_FC = '' #fc data 파일 이름(x 데이터)
FILEX_CONV = '' #preprocessing한 conv data 파일 이름(x 데이터)
FILE_EXO = '' #exogenous(data 8)만 잘라낸 파일 이름
FILEY = '' #y data 파일 이름

#variable
TRAIN_NUM = 1000#traing 회수
SPEED_MAX = 0#data내의 최고 속도(input_data에서 구해준다.)
SPEED_MIN = 0#data내의 최저 속도(input_data에서 구해준다.)
CROSS_NUM = 5 #cross validation의 수

#Hyper Parameter(FC)
FC_LAYER_NUM = 3 #fc layer의 깊이 [default 3]
LAYER_UNIT_NUM = [] #fc에서 고려해줄 layer당 unit의 수 default[66, 64, 128, 64, 1]
FC_BATCH_NORM = True #fc 에서 batch normalization 을 사용할것인지 [default True]
FC_DROPOUT = True #fc 에서 drop out 을 사용할것인지 [default True]
FC_TR_KEEP_PROB = 0.8 #training 에서 dropout 비율
FC_TE_KEEP_PROB = 1.0 #testing 에서 dropout 비율

#Hyper Parameter(CONV)
POOLING = True #pooling을 사용할 것인지 [default True]
CONV_BATCH_NORM = True #conv 에서 batch normalization 을 사용할것인지 [default True]
BATCH_SIZE =  300 #1 epoch 당 batch의 개수 [default 300]
TIME_SEQUENCE = 12 #ouput의 크기 [default 12] **이름 바꿔야함
CONV_LAYER_NUM = 3 #conv layer의 깊이 [default 3]
TEMPORAL_NUM = 12 #conv에서 고려할 시간 default 12]
UP_STREAM_NUM = 6 #conv에서 고려할 이후의 도로 개수들 [default 6]
DOWN_STREAM_NUM = 6 #conv에서 고려할 이전의 도로 개수들 [default 6]
SPARTIAL_NUM = DOWN_STREAM_NUM+UP_STREAM_NUM+1 #conv에서 고려할 총 도로의 수 + 타겟도로[default 13]
CHANNEL_NUM = [] #conv에서 고려해줄 channel 수 [default 1 64 128 64 1] **주의 1로 시작해서 1로 끝나야함 input과 ouput channel은 1개씩이기 때문
FILTER_SIZE_TEMPORAL = [] #시간의 filter size [default 3]
FILTER_SIZE_SPATIAL = [] #공간의 filter size [default 3]
EXOGENOUS_NUM = 54 #exogenous로 들어가는 data의 개수 [default 54]

#Hyper Parameter(LSTM)
HIDDEN_NUM = 32 #lstm의 hidden unit 수 [default 32]
FORGET_BIAS = 1.0 #lstm의 forget bias [default 1.0]
CELL_SIZE = 12 #lstm의 cell 개수 [default 12]



fc_weights = [] #fc weight들의 크기는 layer의 길이에 따라 결정된다.
conv_weights = [] #conv weight들의 크기는 layer의 길이에 따라 결정된다.
lstm_weights = [] #lstm weight들의 크기는 layer의 길이에 따라 결정된다.
lstm_biases = [] #lstm bias들의 크기는 layer의 길이에 따라 결정된다.

batch_prob = tf.placeholder(tf.bool) #feed_dict에 들어가는 값으로 training에서는 true로 하여 분산 평균을 업데이트 해주고, test에서는 안해주게 false로 해준다.
dropout_prob = tf.placeholder(tf.float32) #feed dict에 들어가는 값으로 training에서는 FC_TR_KEEP_PROB으로 testing에서는 FC_TE_KEEP_PROP으로 사용한다.

#weight를 만들어준다.
def init():
    # fc weight 초기화
    for layer_idx in range(1, FC_LAYER_NUM):
        fc_weights.append(init_weights([LAYER_UNIT_NUM[layer_idx - 1], LAYER_UNIT_NUM[layer_idx]]))

    # conv weight 초기화
    for layer_idx in range(1,CONV_LAYER_NUM):
        conv_weights.append(init_weights([FILTER_SIZE_SPATIAL[layer_idx], FILTER_SIZE_TEMPORAL[layer_idx], CHANNEL_NUM[layer_idx-1], CHANNEL_NUM[layer_idx]]))

    # lstm weight 초기화
    lstm_weights.append(init_weights([HIDDEN_NUM, 1]))
    lstm_biases.append(init_weights([1]))



#shper를 input으로 받아 weight를 initailization 해줌
def init_weights(input_shape):
    return tf.get_variable(shape=input_shape, initializer=tf.contrib.layers.xavier_initializer())


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
def input_data():
    #file을 numpy로 바꿔줌
    fcX_data = fileToData(FILEX_FC) #전체 fc 데이터(speed + exogenous)
    convX_data = fileToData(FILEX_CONV) #전체 conv 데이터
    E_data = fileToData(FILE_EXO) #외부요소만 자른 데이터
    Y_data = fileToData(FILEY) #실재값 데이터


    #SPEED_MAX와 SPEED_MIN을 구해줌
    global  SPEED_MIN
    global SPEED_MAX
    SPEED_MAX = Y_data.max()
    SPEED_MIN = Y_data.min()

    return fcX_data, convX_data, E_data, Y_data

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
def FC_model(X, E):
    for layer_idx in range(FC_LAYER_NUM):
        if layer_idx != 0:
            layer = tf.matmul(layer, fc_weights[layer_idx])
        else:
            layer = tf.matmul(np.append(X, E, axis=0), fc_weights[layer_idx])

        if FC_BATCH_NORM == True:
            layer = tf.layers.batch_normalization(layer, center=True, scale=True, training=batch_prob)

        layer = tf.nn.relu(layer)

        if FC_DROPOUT == True:
            tf.nn.dropout(layer, keep_prob=dropout_prob)

    return layer


#CONV network로 input으로 시공간 입력이 output으로 layer가 나온다
def CNN_model(X):
    for layer_idx in range(CONV_LAYER_NUM):
        if layer_idx != 0:
            layer = tf.nn.conv2d(layer, conv_weights[layer_idx], strides=[1, 1, 1, 1])
        else:
            layer = tf.nn.conv2d(X, conv_weights[layer_idx], strides=[1, 1, 1, 1])

        if CONV_LAYER_NUM == True:
            layer = tf.layers.batch_normalization(layer, center=True, scale= True, training=batch_prob)
        layer = tf.nn.relu(layer)
        if POOLING == True and layer_idx != (CONV_LAYER_NUM-1): #마지막 layer는 pooling안함
            layer = tf.nn.avg_pool(layer, ksize=[1,2,2,1], strides=[1,1,1,1])

    #**fc 하나 추가해 주어야함

    return layer

#LSTM network로 input으로 X(batch_size * speed_size * cell_size)와 E(batch_size * exogenous_size * cell_size)가 들어온다.
#output으로 마지막 예측값만 내놓는다.
#현재: time stamp 12, vector_size 66, cell_size 12, output 1
#추후에 실험 1,2 해봐야함
#실험1: time stamp 1, vector_size 6?7?, cell_size 12, output 1
#실험2: time stamp 12, vector_size 66, cell_size 12, output 12
def LSTM_model(X, E):
    # 66(vector_size) * 12(cell size)를 나눠줌
    x = tf.unstack(np.append(X, E, axis=1), axis=0)

    lstm_cell = tf.contrib.rnn.BasicLSTMCell(HIDDEN_NUM, forget_bias=FORGET_BIAS)

    outputs, _ = tf.contrib.rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], lstm_weights[0]) + lstm_biases[0]



#type에 따라 다른 batch slice 결과를 내어준다.
def batch_slice(data, da_idx, ba_idx, slice_type):
    if slice_type == 'FC':
        slice_data = data[da_idx[ba_idx * BATCH_SIZE: (ba_idx + 1) * BATCH_SIZE]]

    elif slice_type == 'CONV':
        for idx in range(ba_idx * BATCH_SIZE, (ba_idx + 1) * BATCH_SIZE):
            if idx == ba_idx * BATCH_SIZE:
                slice_data = data[da_idx[idx * SPARTIAL_NUM: (idx + 1) * SPARTIAL_NUM]].reshape(1, SPARTIAL_NUM, TEMPORAL_NUM, 1)
            else:
                slice_data = np.append(slice_data, data[da_idx[idx * SPARTIAL_NUM: (idx + 1) * SPARTIAL_NUM]].reshape(1, SPARTIAL_NUM, TEMPORAL_NUM, 1), axis=0)

    elif slice_type ==  'LSTM':
        for idx in range(ba_idx * BATCH_SIZE, (ba_idx + 1) * BATCH_SIZE):
            if idx == ba_idx * BATCH_SIZE:
                slice_data = data[da_idx[idx * CELL_SIZE: (idx + 1) * CELL_SIZE]]
            else:
                slice_data = np.append(slice_data, data[da_idx[idx * CELL_SIZE: (idx + 1) * CELL_SIZE]], axis=0)

    else:
        print('ERROR: slice type error\n')

    return slice_data
