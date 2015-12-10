import random
import numpy as np


# This just concatenates all the training data in to one big csv file.
for x in range(201,276):
    outfilestring = 'Train/mock_station_' +str(x) +'_deploy.csv'
    fout_train=open(outfilestring,"w")
    testoutfilestring = 'Train/t_mock_station_' +str(x) +'_deploy.csv'
    fout_test=open(testoutfilestring,"w")
    # first line:
    fout_train.write(open("Train/station_201_deploy.csv").next())
    fout_test.write("Id,"+open("Train/station_201_deploy.csv").next())
    lines = random.sample(range(2, 742), 30)
    filestring = 'Train/station_' +str(x) +'_deploy.csv'
    f = open(filestring, 'r')
    filelines = f.readlines()
    idx = 1
    for n in range(2, 742):
        if n in lines:
            fout_test.write(str(idx) + ',' + filelines[n])
            idx = idx+1
        else :
            fout_train.write(filelines[n])  
    fout_test.close()
    fout_train.close()