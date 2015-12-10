import random
import numpy as np


totals = [0,0,0,0]
idx = 1
for x in range(1,200):
    filestring = 'Models/model_station_' +str(x) +'_rlm_full'
    f = open(filestring, 'r')
    filelines = f.readlines()
    for i in range(1,5):
        line = filelines[i].split(',')
        #print 'hour' + line[10] + ' bikes' + line[24]
        #print 'hour' + trueline[10] + ' bikes' + trueline[24]

        totals[i-1] = totals[i-1]+float(line[1])
        #print 'hour' + line[10] + ' bikes' + line[24]
    f.close()

