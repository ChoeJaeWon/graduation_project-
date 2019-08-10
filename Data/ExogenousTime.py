'''
기존 데이터 양식
기온(12) 강수량(12) "월(12)" 요일(7) 요일특성(4) 기타 등등
바꾼 데이터 양식
기온(12) 강수량(12) "시간(24)" 요일(7) 요일특성(4) 기타 등등

'''
import csv

START_IDX = 11
timeArr = []
INTERVAL_SIZE = 24

def make_timeArray():
    for interval_idx in range(INTERVAL_SIZE):
        timeArr.append([])
        for make_idx in range(INTERVAL_SIZE):
            if interval_idx == make_idx:
                timeArr[interval_idx].append(1)
            else:
                timeArr[interval_idx].append(0)


#file의 data를 list로 만들어준다.
def make_fileTo_list(file):
    text_list = []

    File = open(file, 'r')
    FileData = csv.reader(File)
    for file_line in FileData:
        text_list.append(file_line)
    File.close()

    return text_list


def make_outputFileExogenous(text_list, file):
    outputfile = open(file, 'w', newline='')
    output = csv.writer(outputfile)

    for line_idx in range(len(text_list)):
        output.writerow(text_list[line_idx][:24]+timeArr[(line_idx+START_IDX)%24]+text_list[line_idx][36:])
    outputfile.close()


make_timeArray()
print(timeArr)
#make ExogenousTime
text_list = make_fileTo_list('Exogenous/x_data_2016204_5min_60min_60min_8.csv')
make_outputFileExogenous(text_list, 'ExogenousTime/ExogenousTime_data_2016204_5min_60min_60min_8.csv')