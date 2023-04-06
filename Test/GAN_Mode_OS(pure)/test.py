import numpy as np
from sklearn.model_selection import KFold
import tensorflow as tf
from module import *


DIR = "./index/"
if not os.path.exists(DIR):
    os.makedirs(DIR)
idx =0;
for train_idx, test_idx in Peek_Data():

    outputfile = open(DIR + "tr" + str(idx) + '.csv', 'w', newline='')
    output = csv.writer(outputfile)
    for i in range(len(train_idx)):
        output.writerow([str(train_idx[i])])
    outputfile.close()

    outputfile = open(DIR + "te" + str(idx) + '.csv', 'w', newline='')
    output = csv.writer(outputfile)
    for i in range(len(test_idx)):
        output.writerow([str(test_idx[i])])
    outputfile.close()
    idx = idx + 1
    #print(test_idx)
    #print(train_idx)

    print("----------------------------------------------------")





#print(0b1111& 0b1000)