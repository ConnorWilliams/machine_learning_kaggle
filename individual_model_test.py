import sys
import numpy as np
from sklearn import linear_model
from sklearn import svm
from sklearn.metrics import mean_absolute_error
np.set_printoptions(threshold=np.nan)

# Command line arguments
if (len(sys.argv)==1):
    mock = 0
    print 'Making test predictions...'
elif (sys.argv)[1] != '-m':
    mock = 0
    print 'Making test predictions...'
elif (sys.argv)[1] == '-m':
    mock = 1
    print 'Running a mock test...'

if (mock == 0):
    test_file = 'test.csv'
elif (mock == 1):
    test_file = 'mock_test.csv'

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

selectedFeatures = ["day","bikes_3h_ago"]

training_feature_cols = ()
test_feature_cols = ()

for x in selectedFeatures:
    trainIdx = train_feat.index(x)
    testIdx = test_feat.index(x)
    training_feature_cols = training_feature_cols + (trainIdx,)
    test_feature_cols = test_feature_cols + (testIdx,)

all_test_features = np.genfromtxt(test_file, dtype=float, comments='#', delimiter=',',
                  skip_header=1, skip_footer=0, converters=None, missing_values={"NA"},
                  filling_values='0', usecols=test_feature_cols,
                  names=None, excludelist=None, deletechars=None, replace_space='_',
                  autostrip=False, case_sensitive=True, defaultfmt='f%i',
                  unpack=None, usemask=False, loose=True, invalid_raise=True)

if mock:
    truevalues  = np.genfromtxt(test_file, dtype=float, comments='#', delimiter=',',
                skip_header=1, skip_footer=0, converters=None, missing_values={"NA"},
                filling_values='0', usecols=25,
                names=None, excludelist=None, deletechars=None, replace_space='_',
                autostrip=False, case_sensitive=True, defaultfmt='f%i',
                unpack=None, usemask=False, loose=True, invalid_raise=True)

output = open("individual_sub.csv","w")
output.write("Id,\"bikes\"" +"\n")
predictions = []

# For all test data files
for x in range(201,276):
    filestring = 'Train/station_' +str(x) +'_deploy.csv'

    # Read in the training features and target variable
    training_features = np.genfromtxt(filestring, dtype=float, comments='#', delimiter=',',
        skip_header=1, skip_footer=0, converters=None, missing_values={"NA"},
        filling_values=0, usecols=training_feature_cols,
        names=None, excludelist=None, deletechars=None, replace_space='_',
        autostrip=False, case_sensitive=True, defaultfmt='f%i',
        unpack=None, usemask=False, loose=True, invalid_raise=True)
    training_target = np.genfromtxt(filestring, dtype=float, comments='#', delimiter=',',
        skip_header=1, skip_footer=0, converters=None, missing_values={"NA"},
        filling_values=0, usecols=train_feat.index("bikes"),
        names=None, excludelist=None, deletechars=None, replace_space='_',
        autostrip=False, case_sensitive=True, defaultfmt='f%i',
        unpack=None, usemask=False, loose=True, invalid_raise=True)

    # Generate a model for this particular station
    clf = linear_model.LinearRegression()
    clf.fit (training_features, training_target)
    #print 'Coefficients: \n\t', clf.coef_

    # Only predict for the 30 test instances for this station in the test.csv file
    station_test_features = []
    first_test = (x-201)*30
    for i in range(first_test,first_test+30):
            station_test_features.append(all_test_features[i].tolist())

    predictions.extend(clf.predict(station_test_features).tolist())

for p in range(0, len(predictions)):
    output.write(str(p+1) + "," + str(predictions[p]) +"\n")

output.close()

if mock:
    print 'MAE =', mean_absolute_error(truevalues,predictions)
