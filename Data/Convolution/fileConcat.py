import csv
import os

SPATIAL_NUM = 5

data_list = []

def fileToData(fileName ,data_list):

    File = open(fileName, 'r')
    FileData = csv.reader(File)
    i=0

    for line in FileData:
        if i % SPATIAL_NUM == 0:
            temp_line = []
        for line_idx in range(len(line)):
            temp_line.append(line[line_idx])

        if i % SPATIAL_NUM == (SPATIAL_NUM-1):
            data_list.append(temp_line)
        i += 1
    File.close()

def output_data(data_list):
    #train output
    outputfile = open('x_concat_data_2016204_5min_60min_60min_only_speed.csv', 'w', newline='')
    output = csv.writer(outputfile)

    for idx in range(len(data_list)):
        output.writerow(data_list[idx])

    outputfile.close()

data_list = []

fileToData('x_data_2016204_5min_60min_60min_only_speed.csv', data_list)
output_data(data_list)