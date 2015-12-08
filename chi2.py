import numpy as np
import matplotlib.pyplot as plt
import sklearn.tree as tree
import sklearn.linear_model as linear_model
from sklearn.metrics import mean_absolute_error
from sklearn import svm
from sklearn.feature_selection import chi2
np.set_printoptions(threshold=np.nan)

#bASELINE 3.3
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

selectedFeatures = [
            "weekhour", "isHoliday", "day", "temperature.C",
            "bikes_3h_ago"]
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
if mock:
    truevalues  = np.genfromtxt(test_file, dtype=float, comments='#', delimiter=',',
                skip_header=1, skip_footer=0, converters=None, missing_values={"NA"},
                filling_values='0', usecols=25,
                names=None, excludelist=None, deletechars=None, replace_space='_',
                autostrip=False, case_sensitive=True, defaultfmt='f%i',
                unpack=None, usemask=False, loose=True, invalid_raise=True)
    # preds = np.genfromtxt(test_file, dtype=float, comments='#', delimiter=',',
    #             skip_header=1, skip_footer=0, converters=None, missing_values={"NA"},
    #             filling_values='0', usecols=20,
    #             names=None, excludelist=None, deletechars=None, replace_space='_',
    #             autostrip=False, case_sensitive=True, defaultfmt='f%i',
    #             unpack=None, usemask=False, loose=True, invalid_raise=True)
# print test_features

predicted = []

filestring = 'Train/station_' +str(201) +'_deploy.csv'
# Read in training and test data
training_features = np.genfromtxt(filestring, dtype=float, comments='#', delimiter=',',
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


################################################
# Put our data through some regression model. #
################################################

# Turn weekour in to day hour
#   training_features[:,2] = training_features[:,2] % 24
#   test_features[:,2] = test_features[:,2] % 24

#print training_target
#print training_features 
scores, pvalues = chi2(training_features,training_target )
print pvalues
