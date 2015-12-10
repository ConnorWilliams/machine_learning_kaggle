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

selectedFeatures = ["day","weekhour","isHoliday","windMaxSpeed.m.s",
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

MAE = []

for j in range(0,num_of_tests):
    sys.stdout.write('\rRunning ' + str(j+1) + ' of ' + str(num_of_tests) + '.'*j)
    sys.stdout.flush()
    execfile("./test_gen.py")
    test_file = 'mock_test.csv'

    all_test_features = np.genfromtxt(test_file, dtype=float, comments='#', delimiter=',',
                skip_header=1, skip_footer=0, converters=None, missing_values={"NA"},
                filling_values='0', usecols=test_feature_cols,
                names=None, excludelist=None, deletechars=None, replace_space='_',
                autostrip=False, case_sensitive=True, defaultfmt='f%i',
                unpack=None, usemask=False, loose=True, invalid_raise=True)

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

    predictions = []

    # For all test data files
    for x in range(201,276):
        filestring = 'Train/mock_station_' +str(x)+'_deploy.csv'

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

    # Get the right test features for this station
        station_test_features = []
        first_test = (x-201)*30
        for i in range(first_test,first_test+30):
                station_test_features.append(all_test_features[i].tolist())

    # Select our features
    # Either by removing ones with low variance:
        # selector = VarianceThreshold()
        # training_features = selector.fit_transform(training_features, training_target)
        # station_test_features = selector.transform(station_test_features)

    # Or by K best:
        selector = SelectKBest(f_regression, k=4)
        training_features = selector.fit_transform(training_features, training_target)
        station_test_features = selector.transform(station_test_features)

    # Print the features we have chosen for this station
        # print '\nFeatures for station ' + str(x) + ':'
        # for idx in range(0, len(selector.get_support())):
        #     if selector.get_support()[idx] == True:
        #         print '\t' + str(selectedFeatures[idx])
        # print '\ntraining_features 0, 10, 235:\n', training_features[0], '\n', training_features[10], '\n', training_features[235]
        # print '\nstation_test_features 0, 10, 20:\n', station_test_features[0], '\n', station_test_features[10], '\n', station_test_features[25]
        # raw_input("Press enter...")

    # Generate a model for this particular station
        # clf = linear_model.LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=1)
        clf = linear_model.Lasso(alpha=1.0, fit_intercept=True, normalize=False, precompute=False, copy_X=True, max_iter=1000, tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic')
        # clf = linear_model.Ridge(alpha=1.0, fit_intercept=True, normalize=False, copy_X=True, max_iter=None, tol=0.001, solver='auto', random_state=None)
        # clf = linear_model.BayesianRidge(n_iter=300, tol=0.001, alpha_1=1e-06, alpha_2=1e-06, lambda_1=1e-06, lambda_2=1e-06, compute_score=False, fit_intercept=True, normalize=False, copy_X=True, verbose=False)
        # clf = linear_model.Perceptron(penalty=None, alpha=0.0001, fit_intercept=True, n_iter=5, shuffle=True, verbose=0, eta0=1.0, n_jobs=1, random_state=0, class_weight=None, warm_start=False)
        # clf = svm.SVR(kernel='rbf', degree=3, gamma='auto', coef0=0.0, tol=0.001, C=1.0, epsilon=0.1, shrinking=True, cache_size=200, verbose=False, max_iter=-1)
        # clf = tree.DecisionTreeRegressor(criterion='mse', splitter='best', max_depth=5, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, presort=False)
        clf.fit (training_features, training_target)
        #print 'Coefficients: \n\t', clf.coef_

        predictions.extend(clf.predict(station_test_features).tolist())

    MAE.append(mean_absolute_error(truevalues,predictions))

MAE = sum(MAE) / float(len(MAE))
BASE = mean_absolute_error(truevalues,baseline)
print '\n\n',clf
print '\nBASE =\t',BASE
print 'MAE =\t',MAE
if MAE < BASE:
    print 'DIFF =\t', BASE-MAE
else:
    print 'Worse.'
