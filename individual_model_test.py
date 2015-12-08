import numpy as np
import matplotlib.pyplot as plt
import sklearn.tree as tree
import sklearn.linear_model as linear_model
from sklearn.metrics import mean_absolute_error
from sklearn.feature_selection import VarianceThreshold
from sklearn import svm
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import chi2
np.set_printoptions(threshold=np.nan)

#BASELINE 3.3
mock = 1

if mock:
    test_file = 'mock_test.csv'
else:
    test_file = 'test.csv'

train_feat =   [
            "station", "latitude", "longitude", "numDocks",
            "timestamp", "year", "month", "day", "hour", "weekday", "weekhour", "isHoliday",
            "windMaxSpeed.m.s", "windMeanSpeed.m.s", "windDirection.grades",
            "temperature.C", "relHumidity.HR", "airPressure.mb", "precipitation.l.m2",
            "bikes_3h_ago", "full_profile_3h_diff_bikes", "full_profile_bikes",
            "short_profile_3h_diff_bikes", "short_profile_bikes", "bikes"]
test_feat =   [
            "Id", "station", "latitude", "longitude", "numDocks",
            "timestamp", "year", "month", "day", "hour", "weekday", "weekhour", "isHoliday",
            "windMaxSpeed.m.s", "windMeanSpeed.m.s", "windDirection.grades",
            "temperature.C", "relHumidity.HR", "airPressure.mb", "precipitation.l.m2",
            "bikes_3h_ago", "full_profile_3h_diff_bikes", "full_profile_bikes",
            "short_profile_3h_diff_bikes", "short_profile_bikes"]

selectedFeatures = ["numDocks",
                    "timestamp", "day", "hour", "weekday", "weekhour", "isHoliday",
                    "windMaxSpeed.m.s", "windMeanSpeed.m.s", "windDirection.grades",
                    "temperature.C", "relHumidity.HR", "airPressure.mb", "precipitation.l.m2",
                    "bikes_3h_ago", "full_profile_3h_diff_bikes", "full_profile_bikes",
                    "short_profile_3h_diff_bikes", "short_profile_bikes"]
testTuple = ()
trainTuple = ()
for x in selectedFeatures:
      testIdx = test_feat.index(x)
      trainIdx = train_feat.index(x)
      testTuple = testTuple + (testIdx,)
      trainTuple = trainTuple + (trainIdx,)

test_features = np.genfromtxt(test_file, dtype=float, comments='#', delimiter=',',
                  skip_header=1, skip_footer=0, converters=None, missing_values={"NA"},
                  filling_values='0', usecols=testTuple,
                  names=None, excludelist=None, deletechars=None, replace_space='_',
                  autostrip=False, case_sensitive=True, defaultfmt='f%i',
                  unpack=None, usemask=False, loose=True, invalid_raise=True)
#print test_features
if mock:
    truevalues  = np.genfromtxt(test_file, dtype=float, comments='#', delimiter=',',
                skip_header=1, skip_footer=0, converters=None, missing_values={"NA"},
                filling_values='0', usecols=25,
                names=None, excludelist=None, deletechars=None, replace_space='_',
                autostrip=False, case_sensitive=True, defaultfmt='f%i',
                unpack=None, usemask=False, loose=True, invalid_raise=True)


output = open("individual_sub.csv","w")
output.write("Id,\"bikes\"" +"\n")
predicted = []

for x in range(201,276):
    filestring = 'Train/station_' +str(x) +'_deploy.csv'
    # Read in training and test data
    X = np.genfromtxt(filestring, dtype=float, comments='#', delimiter=',',
            skip_header=1, skip_footer=0, converters=None, missing_values={"NA"},
            filling_values=0, usecols=trainTuple,
            names=None, excludelist=None, deletechars=None, replace_space='_',
            autostrip=False, case_sensitive=True, defaultfmt='f%i',
            unpack=None, usemask=False, loose=True, invalid_raise=True)
    training_target = np.genfromtxt(filestring, dtype=float, comments='#', delimiter=',',
            skip_header=1, skip_footer=0, converters=None, missing_values={"NA"},
            filling_values=0, usecols=train_feat.index("bikes"),
            names=None, excludelist=None, deletechars=None, replace_space='_',
            autostrip=False, case_sensitive=True, defaultfmt='f%i',
            unpack=None, usemask=False, loose=True, invalid_raise=True)

    # Remove features with little variance
    # sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
    # sel.fit_transform(training_features)

    print X.shape
    print SelectKBest(f_regression, k=6).fit_transform(X, training_target)
    training_features = SelectKBest(f_regression, k=6).fit_transform(X, training_target)
    print training_features.shape

    clf = svm.SVR()
    clf.fit (training_features, training_target)
    preds = clf.predict(test_features)
    idx = (x-201)*30
    for y in range(idx,idx+30):
        output.write(str(y+1)+","+str(preds[y])+ "\n")
        if mock:
            predicted.append(preds[y])
output.close()

if mock:
    print mean_absolute_error(truevalues,predicted)
