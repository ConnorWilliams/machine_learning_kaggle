import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
np.set_printoptions(threshold=np.nan)

<<<<<<< HEAD
#weekhour - integer from 1 to 168 representing the hour of the week (Monday 0h is weekhour 1, Sunday 23h is weekhour 168)
=======
>>>>>>> 25d030b2e490d9e82cb9eba516302812632dcafd
features =   [
            "station", "latitude", "longitude", "numDocks",
            "timestamp", "year", "month", "day", "hour", "weekday", "weekhour", "isHoliday",
            "windMaxSpeed.m.s", "windMeanSpeed.m.s", "windDirection.grades",
            "temperature.C", "relHumidity.HR", "airPressure.mb", "precipitation.l.m2",
            "bikes_3h_ago", "full_profile_3h_diff_bikes", "full_profile_bikes",
            "short_profile_3h_diff_bikes", "short_profile_bikes", "bikes"]

features_num =  [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]

# Features to remove: All timestamp info except weekhour...
unwanted_features = [4,5,6,7,8,9]

# Remove unwanted features.
for num in sorted(unwanted_features, reverse=True):
    del features_num[num]
    del features[num]
<<<<<<< HEAD
print features
=======
#print features

>>>>>>> 25d030b2e490d9e82cb9eba516302812632dcafd
#create empty list
day = 0 #monday =0
hour = 10 #0 = 00:00
weekhour = day*24 + hour
output=[]
#for all stations
for x in range(201,275):
      filestring = 'Train/station_' +str(x) +'_deploy.csv'
      data = np.genfromtxt(filestring, dtype=float, comments='#', delimiter=',',
                        skip_header=1, skip_footer=0, converters=None, missing_values={"NA"},
                        filling_values='0', usecols=features_num,
                        names=None, excludelist=None, deletechars=None, replace_space='_',
                        autostrip=False, case_sensitive=True, defaultfmt='f%i',
                        unpack=None, usemask=False, loose=True, invalid_raise=True)
      #zero the averages
      numavg = 0
      avg =0
      #for every record in the station
      for y in data:
            #if its time is
<<<<<<< HEAD
            if y[features.index("weekhour")] == weekhour:
=======
            if y[features.index("weekhour")] == 11:
>>>>>>> 25d030b2e490d9e82cb9eba516302812632dcafd
                  avg += y[features.index("bikes")]
                  numavg+=1
      avg /= numavg
      out = [avg,data[1][features.index("latitude")],data[1][features.index("longitude")]]
      #append a record of average and lat and long to the output
      output.append(out)
#print(output[0])
fig = plt.figure()
ax = fig.add_subplot(111)
<<<<<<< HEAD

ax.scatter([l[1] for l in output],[l2[2] for l2 in output],s=[1+10*l3[0] for l3 in output])
plt.title("Monday 10am")
plt.ylabel("Longitude")
plt.xlabel("Lattitude")
=======
ax.scatter([l[1] for l in output],[l2[2] for l2 in output],s=[10+10*l3[0] for l3 in output])
>>>>>>> 25d030b2e490d9e82cb9eba516302812632dcafd
plt.show()
