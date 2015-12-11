import sys
import numpy as np

# Feature selectors
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression

# Classifiers
from sklearn import linear_model
from sklearn import svm
from sklearn import tree

from sklearn.metrics import mean_absolute_error

np.set_printoptions(threshold=np.nan)

if (len(sys.argv)==2):
    num_of_tests = int(sys.argv[1])
else:
    num_of_tests = 1

train_feat =   ["station","latitude","longitude","numDocks","timestamp","year",
                "month","day","hour","weekday","weekhour","isHoliday","windMaxSpeed.m.s",
                "windMeanSpeed.m.s","windDirection.grades","temperature.C",
                "relHumidity.HR","airPressure.mb","precipitation.l.m2",
                "bikes_3h_ago","full_profile_3h_diff_bikes","full_profile_bikes",
                "short_profile_3h_diff_bikes","short_profile_bikes","bikes"]

test_feat =    ["Id","station","latitude","longitude","numDocks","timestamp",
                "year","month","day","hour","weekday","weekhour","isHoliday",
                "windMaxSpeed.m.s","windMeanSpeed.m.s","windDirection.grades",
                "temperature.C","relHumidity.HR","airPressure.mb","precipitation.l.m2",
                "bikes_3h_ago","full_profile_3h_diff_bikes","full_profile_bikes",
                "short_profile_3h_diff_bikes","short_profile_bikes"]

selectedFeatures = ["bikes_3h_ago",
"short_profile_bikes",
"short_profile_3h_diff_bikes",
"full_profile_bikes",
"full_profile_3h_diff_bikes",
"temperature.C"
]

training_feature_cols = ()
test_feature_cols = ()

for x in selectedFeatures:
    trainIdx = train_feat.index(x)
    testIdx = test_feat.index(x)
    training_feature_cols = training_feature_cols + (trainIdx,)
    test_feature_cols = test_feature_cols + (testIdx,)

MAE = []

for j in range(0,num_of_tests):
    sys.stdout.write('\rRunning ' + str(j+1) + ' of ' + str(num_of_tests)+ '.'*j)
    sys.stdout.flush()
    execfile("./general_model.py")
    execfile("./phase2.py")
    #print 'Running mock',j+1,'of',num_of_tests
    training_file = 'mock_general_station_train.csv'
    test_file = 'mock_test.csv'
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


    # output = open("phase3.csv","w")
    # output.write("Id,\"bikes\"" +"\n")
    predictions = []
    P1_WEIGHT = 1
    P2_WEIGHT = 3
    for x in range(0,len(phase2)):
        predictions.append((phase1[x]*P1_WEIGHT+phase2[x]*P2_WEIGHT)/(P2_WEIGHT+P1_WEIGHT))
    MAE.append(mean_absolute_error(truevalues,predictions))


MAE = sum(MAE) / float(len(MAE))
BASE = mean_absolute_error(truevalues,baseline)
print '\n\n',clf
print 'BASE =\t',BASE
print 'MAE =\t',MAE
if MAE < BASE:
    print 'DIFF =\t', BASE-MAE
else:
    print 'Worse.'
