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
    execfile("./test_gen.py")
    #print 'Running mock',j+1,'of',num_of_tests
    training_file = 'mock_general_station_train.csv'
    test_file = 'mock_test.csv'
    test_features = np.genfromtxt(test_file, dtype=float, comments='#', delimiter=',',
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
    selector = SelectKBest(f_regression, k=6    )
    training_features = selector.fit_transform(training_features, training_target)
    test_features = selector.transform(test_features)

    # Generate a model for this particular station
    clf = linear_model.LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=1)
    # clf = linear_model.Lars(fit_intercept=True, verbose=False, normalize=True, precompute='auto', n_nonzero_coefs=500, eps=2.2204460492503131e-16, copy_X=True, fit_path=True, positive=False)
    # clf = linear_model.Ridge(alpha=1.0, fit_intercept=True, normalize=False, copy_X=True, max_iter=None, tol=0.001, solver='auto', random_state=None)
    # clf = linear_model.BayesianRidge(n_iter=300, tol=0.001, alpha_1=1e-06, alpha_2=1e-06, lambda_1=1e-06, lambda_2=1e-06, compute_score=False, fit_intercept=True, normalize=False, copy_X=True, verbose=False)
    # clf = linear_model.Perceptron(penalty=None, alpha=0.0001, fit_intercept=True, n_iter=5, shuffle=True, verbose=0, eta0=1.0, n_jobs=1, random_state=0, class_weight=None, warm_start=False)
    # clf = svm.SVR(kernel='rbf', degree=3, gamma='auto', coef0=0.0, tol=0.001, C=1.0, epsilon=0.1, shrinking=True, cache_size=200, verbose=False, max_iter=-1)
    #clf = tree.DecisionTreeRegressor(criterion='mse', splitter='best', max_depth=10, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, presort=False)
    clf.fit (training_features, training_target)
    # print 'Coefficients: \n\t', clf.coef_
    #full
    # clf.intercept_ = 0.171565194272
    # clf.coef_[0] = 0.745745717214
    # clf.coef_[1] = 0.229370444058
    # clf.coef_[2] = 0.571453414447
    # full temp
    # clf.intercept_ = 0.311780932216
    # clf.coef_[0] = 0.743418318017
    # clf.coef_[1] = 0.228585239087
    # clf.coef_[2] = 0.57138293437
    # clf.coef_[3] = -0.00570071093921
    # short
    # clf.intercept_ = 0.679605102608
    # clf.coef_[0] = 0.746599471027
    # clf.coef_[1] = 0.159064132418
    # clf.coef_[2] = 0.472205363294 
    # fullshort
    # clf.intercept_ = 0.189539444281
    # clf.coef_[0] = 0.745530995184
    # clf.coef_[1] = 0.069100558228
    # clf.coef_[2] = 0.221293478338 
    # clf.coef_[3] = 0.156969222064 
    # clf.coef_[4] = 0.370408288286 

    clf.intercept_ = 0.30381645061
    clf.coef_[0] = 0.744290931563
    clf.coef_[1] = 0.0643842157692
    clf.coef_[2] = 0.223470461553
    clf.coef_[3] = 0.16065454214
    clf.coef_[4] = 0.36816804663
    clf.coef_[5] = -0.00470908330151

    predictions = []
    predictions.extend(clf.predict(test_features).tolist())

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
