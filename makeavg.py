import random
import numpy as np


totals = [0,0,0,0,0,0,0]
idx = 1
for x in range(1,201):
    filestring = 'Models/model_station_' +str(x) +'_rlm_short_full_temp.csv'
    f = open(filestring, 'r')
    filelines = f.readlines()
    for i in range(1,8):
        line = filelines[i].split(',')
        #print 'hour' + line[10] + ' bikes' + line[24]
        #print 'hour' + trueline[10] + ' bikes' + trueline[24]

        totals[i-1] = totals[i-1]+float(line[1])
        #print 'hour' + line[10] + ' bikes' + line[24]
    f.close()
for x in totals:
    print x/200
