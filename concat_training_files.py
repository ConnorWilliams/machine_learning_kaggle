import random
import numpy as np


# This just concatenates all the training data in to one big csv file.
fout=open("general_station_train.csv","w")

# first file:
for line in open("Train/station_201_deploy.csv"):
    fout.write(line)
# now the rest:
for num in range(202,276):
    f = open("Train/station_"+str(num)+"_deploy.csv")
    f.next() # skip the header
    for line in f:
         fout.write(line)
    f.close() # not really needed
fout.close()


# This makes a mock test csv file.
fout=open("mock_test.csv","w")
# first line:
fout.write("Id,"+open("Train/station_201_deploy.csv").next())

idx = 1
for x in range(201,276):
    lines = random.sample(range(2, 742), 30)
    filestring = 'Train/station_' +str(x) +'_deploy.csv'
    f = open(filestring, 'r')
    filelines = f.readlines()
    for n in lines:
        line = filelines[n].split(',')
        trueline = filelines[n].split(',')
        #print 'hour' + line[10] + ' bikes' + line[24]
        #print 'hour' + trueline[10] + ' bikes' + trueline[24]

        line[24] = trueline[24]
        #print 'hour' + line[10] + ' bikes' + line[24]
        line = ','.join([str(mli) for mli in line])
        fout.write(str(idx) + ',' + line)
        idx = idx+1

fout.close()
