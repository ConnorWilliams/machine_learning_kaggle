import random
import numpy as np


# This just concatenates all the training data in to one big csv file.
fout_gen_train=open("mock_general_station_train.csv","w")

# This makes a mock test csv file.
fout_test=open("mock_test.csv","w")

# first line:
fout_gen_train.write(open("Train/station_201_deploy.csv").next())
fout_test.write("Id,"+open("Train/station_201_deploy.csv").next())

idx = 1
for x in range(201,276):
    fout_ind_train=open('Train/mock_station_' +str(x)+'_deploy.csv',"w")
    fout_ind_train.write(open("Train/station_201_deploy.csv").next())

    lines = random.sample(range(2, 742), 30)
    filestring = 'Train/station_' +str(x) +'_deploy.csv'
    f = open(filestring, 'r')
    filelines = f.readlines()
    for n in range(2, 742):
        if n in lines:
            fout_test.write(str(idx) + ',' + filelines[n])
            idx = idx+1
        else :
            fout_gen_train.write(filelines[n])
            fout_ind_train.write(filelines[n])
    fout_ind_train.close()

fout_test.close()
fout_gen_train.close()
