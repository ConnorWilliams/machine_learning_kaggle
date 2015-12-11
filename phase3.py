import sys
import numpy as np


from sklearn.metrics import mean_absolute_error


# Command line arguments
if (len(sys.argv)==1):
    mock = 0
    test_file = 'test.csv'
    print 'Making test predictions...'
elif (sys.argv)[1] != '-m':
    mock = 0
    test_file = 'mock_test.csv'
    print 'Making test predictions...'
elif (sys.argv)[1] == '-m':
    mock = 1
    test_file = 'mock_test.csv'
    print 'Running a mock test...'

test_feat =    ["Id","station","latitude","longitude","numDocks","timestamp",
                "year","month","day","hour","weekday","weekhour","isHoliday",
                "windMaxSpeed.m.s","windMeanSpeed.m.s","windDirection.grades",
                "temperature.C","relHumidity.HR","airPressure.mb","precipitation.l.m2",
                "bikes_3h_ago","full_profile_3h_diff_bikes","full_profile_bikes",
                "short_profile_3h_diff_bikes","short_profile_bikes"]
if mock:
    truevalues  = np.genfromtxt(test_file, dtype=float, comments='#', delimiter=',',
                skip_header=1, skip_footer=0, converters=None, missing_values={"NA"},
                filling_values='0', usecols=25,
                names=None, excludelist=None, deletechars=None, replace_space='_',
                autostrip=False, case_sensitive=True, defaultfmt='f%i',
                unpack=None, usemask=False, loose=True, invalid_raise=True)
    baseline  = np.genfromtxt(test_file, dtype=float, comments='#', delimiter=',',
            skip_header=1, skip_footer=0, converters=None, missing_values={"NA"},
            filling_values='0', usecols=test_feat.index("bikes_3h_ago"),
            names=None, excludelist=None, deletechars=None, replace_space='_',
            autostrip=False, case_sensitive=True, defaultfmt='f%i',
            unpack=None, usemask=False, loose=True, invalid_raise=True)

phase1 = np.genfromtxt('general_sub.csv', dtype=float, comments='#', delimiter=',',
        skip_header=1, skip_footer=0, converters=None, missing_values={"NA"},
        filling_values='0', usecols=1,
        names=None, excludelist=None, deletechars=None, replace_space='_',
        autostrip=False, case_sensitive=True, defaultfmt='f%i',
        unpack=None, usemask=False, loose=True, invalid_raise=True)
phase2 = np.genfromtxt('phase2.csv', dtype=float, comments='#', delimiter=',',
        skip_header=1, skip_footer=0, converters=None, missing_values={"NA"},
        filling_values='0', usecols=1,
        names=None, excludelist=None, deletechars=None, replace_space='_',
        autostrip=False, case_sensitive=True, defaultfmt='f%i',
        unpack=None, usemask=False, loose=True, invalid_raise=True)


output = open("phase3.csv","w")
output.write("Id,\"bikes\"" +"\n")
predictions = []
P1_WEIGHT = 1
P2_WEIGHT = 1
for x in range(0,len(phase2)):
    predictions.append((phase1[x]*P1_WEIGHT+phase2[x]*P2_WEIGHT)/(P2_WEIGHT+P1_WEIGHT))


for p in range(0, len(predictions)):
    output.write(str(p+1) + "," + str(predictions[p]) +"\n")

if mock:
    print '\nBASELINE MAE =', mean_absolute_error(truevalues,baseline)
    print 'MAE =', mean_absolute_error(truevalues,predictions)
