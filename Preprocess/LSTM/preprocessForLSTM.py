import csv

sequence_size = 3 #sequence size
result_dir = 'preprocess_result/'

xFile = ['x_data_2016204_5min_60min_60min_only_speed.csv', 'x_data_2016204_5min_60min_60min_1.csv','x_data_2016204_5min_60min_60min_2.csv','x_data_2016204_5min_60min_60min_3.csv','x_data_2016204_5min_60min_60min_4.csv','x_data_2016204_5min_60min_60min_5.csv','x_data_2016204_5min_60min_60min_6.csv','x_data_2016204_5min_60min_60min_7.csv','x_data_2016204_5min_60min_60min_8.csv']
yFile = ['y_data_2016204_5min_60min_60min.csv']

#file의 data를 list로 만들어준다.
def make_fileTo_list(file):
    text_list = []

    File = open(file, 'r')
    FileData = csv.reader(File)
    for file_line in FileData:
        text_list.append(file_line)
    File.close()

    return text_list


def make_outputFile(text_list, file):
    outputfile = open(result_dir + file, 'w', newline='')
    output = csv.writer(outputfile)

    for line_idx in range(len(text_list)-sequence_size+1):
        for sequence_idx in range(sequence_size):
            output.writerow(text_list[line_idx+sequence_idx])
    outputfile.close()

for xFile_idx in range(len(xFile)):
     text_list = make_fileTo_list(xFile[xFile_idx])
     make_outputFile(text_list, xFile[xFile_idx])

for yFile_idx in range(len(yFile)):
    text_list = make_fileTo_list(yFile[yFile_idx])
    make_outputFile(text_list, yFile[yFile_idx])