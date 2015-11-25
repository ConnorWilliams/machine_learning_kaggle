import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(threshold=np.nan)


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
print features

#create empty list
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
            if y[features.index("weekhour")] == 49:
                  avg += y[features.index("bikes")]
                  numavg+=1
      avg /= numavg
      out = [avg,data[1][features.index("latitude")],data[1][features.index("longitude")]]
      #append a record of average and lat and long to the output
      output.append(out)
#print(output[0])
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter([l[1] for l in output],[l2[2] for l2 in output],s=[1+5*l3[0] for l3 in output])
plt.show()
