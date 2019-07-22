'''
speed+exogenous 파일에서 exogenous부분만 잘라준다.
그리고 월과 요일 특성을 one hot으로 대체해준다.

xlsx2csv로 onehot.xlsx를 csv로 바꾸고 실행 시켜줘야한다

'''
import csv

speed_len = 12
data_dir = 'origin/'
result_dir = 'preprocess/'

xFile = ['x_data_2016204_5min_60min_60min_1.csv','x_data_2016204_5min_60min_60min_2.csv','x_data_2016204_5min_60min_60min_3.csv','x_data_2016204_5min_60min_60min_4.csv','x_data_2016204_5min_60min_60min_5.csv','x_data_2016204_5min_60min_60min_6.csv','x_data_2016204_5min_60min_60min_7.csv','x_data_2016204_5min_60min_60min_8.csv']
onehotFile = ['onehot_vec.csv']
yFile = ['y_data_2016204_5min_60min_60min.csv']

#file의 data를 list로 만들어준다.
def make_fileTo_list(file):
    text_list = []


    File = open(data_dir+file, 'r')
    FileData = csv.reader(File)
    for file_line in FileData:
        text_list.append(file_line)
    File.close()

    return text_list

def make_outputFile(text_list, onehot_list, file):
    outputfile = open(result_dir + file, 'w', newline='')
    output = csv.writer(outputfile)

    for line_idx in range(len(text_list)):
        #onehot 부분에 +speed_len/288은 12개의 speed 때문에 첫날의 값이 12개 없어짐을 나타내고
        #뒤의 +1은 onehot_vec.csv에 첫줄이 string 값이기 때문에 더해준다.
        #이와 같은 맥락으로 [1:]이 작성되었다
        output.writerow(text_list[line_idx][speed_len:3*speed_len]+ onehot_list[int((line_idx+speed_len)/288)+1][1:] +text_list[line_idx][3*speed_len+6:])
    outputfile.close()




onehot_list = make_fileTo_list(onehotFile[0])

for xFile_idx in range(len(xFile)):
     text_list = make_fileTo_list(xFile[xFile_idx])
     make_outputFile(text_list, onehot_list,xFile[xFile_idx])
