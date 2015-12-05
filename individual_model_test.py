import numpy as np
import matplotlib.pyplot as plt
import sklearn.tree as tree
import sklearn.linear_model as linear_model
from sklearn import svm
np.set_printoptions(threshold=np.nan)

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

# Our target variable is 'bikes'
target_num = 24

selectedFeatures = ["isHoliday","day","bikes_3h_ago","short_profile_3h_diff_bikes","short_profile_bikes", "temperature.C"]
testTuple = ()
trainTuple = ()
for x in selectedFeatures:
      testIdx = test_feat.index(x)
      trainIdx = train_feat.index(x)
      testTuple = testTuple + (testIdx,)
      trainTuple = trainTuple + (trainIdx,)

test_features = np.genfromtxt('test.csv', dtype=float, comments='#', delimiter=',',
                  skip_header=1, skip_footer=0, converters=None, missing_values={"NA"},
                  filling_values='0', usecols=testTuple,
                  names=None, excludelist=None, deletechars=None, replace_space='_',
                  autostrip=False, case_sensitive=True, defaultfmt='f%i',
                  unpack=None, usemask=False, loose=True, invalid_raise=True)
# Where is our training data stored?
output = open("individual_sub.csv","w")
output.write("Id,\"bikes\"" +"\n")
for x in range(201,276):
      filestring = 'Train/station_' +str(x) +'_deploy.csv'
      # Read in training and test data
      training_features = np.genfromtxt(filestring, dtype=float, comments='#', delimiter=',',
                        skip_header=1, skip_footer=0, converters=None, missing_values={"NA"},
                        filling_values='0', usecols=trainTuple,
                        names=None, excludelist=None, deletechars=None, replace_space='_',
                        autostrip=False, case_sensitive=True, defaultfmt='f%i',
                        unpack=None, usemask=False, loose=True, invalid_raise=True)
      training_target = np.genfromtxt(filestring, dtype=float, comments='#', delimiter=',',
                        skip_header=1, skip_footer=0, converters=None, missing_values={"NA"},
                        filling_values='0', usecols=train_feat.index("bikes"),
                        names=None, excludelist=None, deletechars=None, replace_space='_',
                        autostrip=False, case_sensitive=True, defaultfmt='f%i',
                        unpack=None, usemask=False, loose=True, invalid_raise=True)

      ################################################
      # Put our data through some regression model. #
      ################################################

      # Turn weekour in to day hour
    #   training_features[:,2] = training_features[:,2] % 24
    #   test_features[:,2] = test_features[:,2] % 24

      clf = svm.SVR()
      clf.fit (training_features, training_target)
      preds = clf.predict(test_features)
      #print preds
      #raw_input("Press Enter to continue...")
      idx = (x-201)*30
      for y in range(idx,idx+30):
           output.write(str(y+1)+","+str( preds[y])+ "\n")

output.close()
