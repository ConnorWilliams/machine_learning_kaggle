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
execfile("./test_gen.py")
# Command line arguments
# if (len(sys.argv)==1):
#     mock = 0
#     test_file = 'test.csv'
#     print 'Making test predictions...'
# elif (sys.argv)[1] != '-m':
#     mock = 0
#     test_file = 'test.csv'
#     print 'Making test predictions...'
# elif (sys.argv)[1] == '-m':
#     mock = 1
#     test_file = 'mock_test.csv'
#     print 'Running a mock test...'
mock = 1
test_file = 'mock_test.csv'
print 'Running a mock test...'
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

selectedFeatures = ["station","latitude","longitude","numDocks","day",
                "weekhour","isHoliday","windMaxSpeed.m.s",
                "windMeanSpeed.m.s","windDirection.grades","temperature.C",
                "relHumidity.HR","airPressure.mb","precipitation.l.m2",
                "bikes_3h_ago",
                "short_profile_3h_diff_bikes","short_profile_bikes"]

training_feature_cols = ()
test_feature_cols = ()

for x in selectedFeatures:
    trainIdx = train_feat.index(x)
    testIdx = test_feat.index(x)
    training_feature_cols = training_feature_cols + (trainIdx,)
    test_feature_cols = test_feature_cols + (testIdx,)

test_features = np.genfromtxt(test_file, dtype=float, comments='#', delimiter=',',
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
    baseline  = np.genfromtxt(test_file, dtype=float, comments='#', delimiter=',',
            skip_header=1, skip_footer=0, converters=None, missing_values={"NA"},
            filling_values='0', usecols=test_feat.index("bikes_3h_ago"),
            names=None, excludelist=None, deletechars=None, replace_space='_',
            autostrip=False, case_sensitive=True, defaultfmt='f%i',
            unpack=None, usemask=False, loose=True, invalid_raise=True)

output = open("general_sub.csv","w")
output.write("Id,\"bikes\"" +"\n")
predictions = []

if mock:
    training_file = 'mock_general_station_train.csv'
else:
    training_file = 'general_station_train.csv'

# Read in the training features and target variable
training_features = np.genfromtxt(training_file, dtype=float, comments='#', delimiter=',',
    skip_header=1, skip_footer=0, converters=None, missing_values={"NA"},
    filling_values=0, usecols=training_feature_cols,
    names=None, excludelist=None, deletechars=None, replace_space='_',
    autostrip=False, case_sensitive=True, defaultfmt='f%i',
    unpack=None, usemask=False, loose=True, invalid_raise=True)
training_target = np.genfromtxt(training_file, dtype=float, comments='#', delimiter=',',
    skip_header=1, skip_footer=0, converters=None, missing_values={"NA"},
    filling_values=0, usecols=train_feat.index("bikes"),
    names=None, excludelist=None, deletechars=None, replace_space='_',
    autostrip=False, case_sensitive=True, defaultfmt='f%i',
    unpack=None, usemask=False, loose=True, invalid_raise=True)

# Select our features
# Either by removing ones with low variance:
# selector = VarianceThreshold()
# training_features = selector.fit_transform(training_features, training_target)
# station_test_features = selector.transform(station_test_features)

# Or by K best:
selector = SelectKBest(f_regression, k=9)
training_features = selector.fit_transform(training_features, training_target)
test_features = selector.transform(test_features)

# print '\ntest_features.shape = ', test_features.shape
# print 'training_features.shape = ',training_features.shape
# print 'training_target.shape = ',training_target.shape

# # Print the features we have chosen for this station
# print '\nFeatures selcted:'
# for idx in range(0, len(selector.get_support())):
#     if selector.get_support()[idx] == True:
#         print '\t' + str(selectedFeatures[idx])
# print '\ntraining_features 0, 10, 235:\n', training_features[0], '\n', training_features[10], '\n', training_features[235]
# print '\ntest_features 0, 10, 20:\n', test_features[0], '\n', test_features[10], '\n', test_features[25]

# Generate a model for this particular station
#clf = linear_model.LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=1)
# clf = linear_model.Lars(fit_intercept=True, verbose=False, normalize=True, precompute='auto', n_nonzero_coefs=500, eps=2.2204460492503131e-16, copy_X=True, fit_path=True, positive=False)
clf = linear_model.Ridge(alpha=1.0, fit_intercept=True, normalize=False, copy_X=True, max_iter=None, tol=0.001, solver='auto', random_state=None)
# clf = linear_model.BayesianRidge(n_iter=300, tol=0.001, alpha_1=1e-06, alpha_2=1e-06, lambda_1=1e-06, lambda_2=1e-06, compute_score=False, fit_intercept=True, normalize=False, copy_X=True, verbose=False)
# clf = linear_model.Perceptron(penalty=None, alpha=0.0001, fit_intercept=True, n_iter=5, shuffle=True, verbose=0, eta0=1.0, n_jobs=1, random_state=0, class_weight=None, warm_start=False)
# clf = svm.SVR(kernel='rbf', degree=3, gamma='auto', coef0=0.0, tol=0.001, C=1.0, epsilon=0.1, shrinking=True, cache_size=200, verbose=False, max_iter=-1)
# clf = tree.DecisionTreeRegressor(criterion='mse', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, presort=False)
clf.fit (training_features, training_target)
# print 'Coefficients: \n\t', clf.coef_

predictions.extend(clf.predict(test_features).tolist())

for p in range(0, len(predictions)):
    output.write(str(p+1) + "," + str(predictions[p]) +"\n")

output.close()

# if mock:
#     print '\nBASELINE MAE =', mean_absolute_error(truevalues,baseline)
#     print 'MAE =', mean_absolute_error(truevalues,predictions)
