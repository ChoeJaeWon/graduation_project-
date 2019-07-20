import csv
import os
import errno

sequence_size = 3 #sequence size, if sequence_size is 1 it will be used for conv itself not a conv-lstm
m = 2  #s - m to s  and s to s + m  ==spatial size
spatial_size = m*2 + 1 #for convolution matrix

result_dir_conv = 'preprocess_conv/' #if sequence is 1
result_dir_conv_lstm = 'preprocess_conv_lstm/' #if sequence is more than 1

#we need only speed for conv itself
#선정된 직전, 직후 m 개의 도로를 각각 입력하시고 xFile에 타겟 도로를 입력하시면 됩니다. 다 only speed로만 하시면 됩니다.
#저장되는 것은 타겟 링크의 경로에 맞게 저장경로로 이동됩니다.
#sequence가 1이면 conv만 쓰이는 모델을 위한 preprocessing 으로써 preprocess_conv로 이동하고
#sequence가 2이상이면 conv와 lstm이 함께 쓰이는 모델을 위한 preprocessing 으로써 preprocess_conv_lstm으로 이동합니다.
#입력된 도로의 갯수가 안맞으면 indexing error가 뜹니다.
#결과는 입력 순서에 sensitive하니 순서에 주의하세요
after_m_x = ['1049443/x_data_1049443_5min_60min_60min_only_speed.csv' , '1063062/x_data_1063062_5min_60min_60min_only_speed.csv']
before_m_x = ['2016209/x_data_2016209_5min_60min_60min_only_speed.csv' , '2046685/x_data_2046685_5min_60min_60min_only_speed.csv']
after_m_y = ['1049443/y_data_1049443_5min_60min_60min.csv' , '1063062/y_data_1063062_5min_60min_60min.csv']
before_m_y = ['2016209/y_data_2016209_5min_60min_60min.csv' , '2046685/y_data_2046685_5min_60min_60min.csv']
xFile = '2016204/x_data_2016204_5min_60min_60min_only_speed.csv' #target link, storage path
yFile = '2016204/y_data_2016204_5min_60min_60min.csv'

#------------------------------------------------------이하 변경할 변수는 없습니다.---------------------------------------------------------

def make_fileTo_list(file): #build list of links
    text_list = []

    File = open(file, 'r')
    FileData = csv.reader(File)
    for file_line in FileData:
        text_list.append(file_line)
    File.close()

    return text_list


def make_outputFile_x(text_list, file, _after_m_list, _before_m_list): #making file of convolution matrix
    if sequence_size == 1:
        try:
            if not (os.path.isdir(result_dir_conv + file[:8])):
                os.makedirs(os.path.join(result_dir_conv + file[:8]))
        except OSError as e:
            if e.errno != errno.EEXIST:
                print("Failed to create directory")
                raise

        outputfile = open(result_dir_conv + file, 'w', newline='')
    else:
        try:
            if not (os.path.isdir(result_dir_conv_lstm + file[:8])):
                os.makedirs(os.path.join(result_dir_conv_lstm + file[:8]))
        except OSError as e:
            if e.errno != errno.EEXIST:
                print("Failed to create directory")
                raise

        outputfile = open(result_dir_conv_lstm + file, 'w', newline='')
    output = csv.writer(outputfile)

    for line_idx in range(len(text_list)-sequence_size+1):
        for sequence_idx in range(sequence_size):
            for spatial_idx in range(m): #after m
                output.writerow(_after_m_list[spatial_idx][line_idx+sequence_idx])
            output.writerow(text_list[line_idx+sequence_idx]) #target
            for spatial_idx in range(m): #before m
                output.writerow(_before_m_list[spatial_idx][line_idx+sequence_idx])

    outputfile.close()

def make_outputFile_y(text_list, file): #ground truth file making process
    if sequence_size == 1:
        try:
            if not (os.path.isdir(result_dir_conv + file[:8])):
                os.makedirs(os.path.join(result_dir_conv + file[:8]))
        except OSError as e:
            if e.errno != errno.EEXIST:
                print("Failed to create directory")
                raise

        outputfile = open(result_dir_conv + file, 'w', newline='')
    else:
        try:
            if not (os.path.isdir(result_dir_conv_lstm + file[:8])):
                os.makedirs(os.path.join(result_dir_conv_lstm + file[:8]))
        except OSError as e:
            if e.errno != errno.EEXIST:
                print("Failed to create directory")
                raise

        outputfile = open(result_dir_conv_lstm + file, 'w', newline='')
    output = csv.writer(outputfile)

    for line_idx in range(len(text_list)-sequence_size+1):
        for sequence_idx in range(sequence_size):
            output.writerow(text_list[line_idx+sequence_idx]) #target

    outputfile.close()


text_list = make_fileTo_list(xFile)
after_m_list = [[] for _ in range(m)]
before_m_list = [[] for _ in range(m)]
for spatial_idx in range(m):
    after_m_list[spatial_idx] = make_fileTo_list(after_m_x[spatial_idx]) #input link after and before
    before_m_list[spatial_idx] = make_fileTo_list(before_m_x[spatial_idx])
make_outputFile_x(text_list, xFile, after_m_list, before_m_list)

text_list = make_fileTo_list(yFile)
make_outputFile_y(text_list, yFile)


if sequence_size == 1:
    print("preprocess for convolution network finished")

else:
    print("preprocess for conv-lstm finished")
    print("sequence size of lstm is " + str(sequence_size))

print("target link is" + xFile)
print("size of convolution matrix is " + str(spatial_size) + "*" + "timestamp")