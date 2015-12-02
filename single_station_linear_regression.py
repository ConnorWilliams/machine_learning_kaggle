import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
np.set_printoptions(threshold=np.nan)

features =   [
            "station", "latitude", "longitude", "numDocks",
            "timestamp", "year", "month", "day", "hour", "weekday", "weekhour", "isHoliday",
            "windMaxSpeed.m.s", "windMeanSpeed.m.s", "windDirection.grades",
            "temperature.C", "relHumidity.HR", "airPressure.mb", "precipitation.l.m2",
            "bikes_3h_ago", "full_profile_3h_diff_bikes", "full_profile_bikes",
            "short_profile_3h_diff_bikes", "short_profile_bikes", "bikes"]

features_num =  [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]

# Our target variable is 'bikes'
target_num = 24

# Features to remove: All timestamp info except weekhour...
unwanted_features = [4,5,6,7,8,9,24]

# Remove unwanted features.
for num in sorted(unwanted_features, reverse=True):
    del features_num[num]
    del features[num]

# Where is our training data stored?
filestring = 'Train/station_201_deploy.csv'

# Read in training and test data
training_features = training_data = np.genfromtxt(filestring, dtype=float, comments='#', delimiter=',',
                  skip_header=1, skip_footer=0, converters=None, missing_values={"NA"},
                  filling_values='0', usecols=(1,2),
                  names=None, excludelist=None, deletechars=None, replace_space='_',
                  autostrip=False, case_sensitive=True, defaultfmt='f%i',
                  unpack=None, usemask=False, loose=True, invalid_raise=True)
training_target = training_data = np.genfromtxt(filestring, dtype=float, comments='#', delimiter=',',
                  skip_header=1, skip_footer=0, converters=None, missing_values={"NA"},
                  filling_values='0', usecols=target_num,
                  names=None, excludelist=None, deletechars=None, replace_space='_',
                  autostrip=False, case_sensitive=True, defaultfmt='f%i',
                  unpack=None, usemask=False, loose=True, invalid_raise=True)
test_features = np.genfromtxt('test.csv', dtype=float, comments='#', delimiter=',',
                  skip_header=1, skip_footer=0, converters=None, missing_values={"NA"},
                  filling_values='0', usecols=(1,2),
                  names=None, excludelist=None, deletechars=None, replace_space='_',
                  autostrip=False, case_sensitive=True, defaultfmt='f%i',
                  unpack=None, usemask=False, loose=True, invalid_raise=True)
test_target = np.genfromtxt('test.csv', dtype=float, comments='#', delimiter=',',
                  skip_header=1, skip_footer=0, converters=None, missing_values={"NA"},
                  filling_values='0', usecols=target_num,
                  names=None, excludelist=None, deletechars=None, replace_space='_',
                  autostrip=False, case_sensitive=True, defaultfmt='f%i',
                  unpack=None, usemask=False, loose=True, invalid_raise=True)

################################################
# Put our data through some regression models. #
################################################

clf = linear_model.LinearRegression()
clf.fit (training_features, training_target)

print('Coefficients: \n', clf.coef_)
print("Residual sum of squares: %.2f" % np.mean((clf.predict(test_features) - test_target) ** 2))
print('Variance score: %.2f' % clf.score(test_features, test_target))
