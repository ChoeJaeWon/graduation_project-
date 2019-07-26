import csv


#file의 data를 list로 만들어준다.
def make_fileTo_list(file):
    text_list = []

    File = open(file, 'r')
    FileData = csv.reader(File)
    for file_line in FileData:
        text_list.append(file_line)
    File.close()

    return text_list


def make_outputFileSpeed(text_list, file):
    outputfile = open(file, 'w', newline='')
    output = csv.writer(outputfile)

    for line_idx in range(len(text_list)):
        output.writerow(text_list[line_idx][11:12])
    outputfile.close()

def make_outputFileExogenous(text_list, file):
    outputfile = open(file, 'w', newline='')
    output = csv.writer(outputfile)

    for line_idx in range(len(text_list)):
        output.writerow(text_list[line_idx][11:12]+text_list[line_idx][23:24]+text_list[line_idx][24:48]+ text_list[line_idx][70:71] +text_list[line_idx][82:83])
    outputfile.close()

#make SpeedLSTM
text_list = make_fileTo_list('Speed/x_data_2016204_5min_60min_60min_only_speed.csv')
make_outputFileSpeed(text_list, 'SpeedLSTM/LSTMx_data_2016204_5min_60min_60min_only_speed.csv')


#make ExogenousLSTM
text_list = make_fileTo_list('Exogenous/x_data_2016204_5min_60min_60min_8.csv')
make_outputFileExogenous(text_list, 'ExogenousLSTM/LSTMx_data_2016204_5min_60min_60min_8.csv')