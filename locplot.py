import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(threshold=np.nan)


heading =   ["station", "latitude", "longitude", "numDocks,"
            "timestamp", "year", "month", "day", "hour", "weekday", "weekhour", "isHoliday",
            "windMaxSpeed.m.s", "windMeanSpeed.m.s", "windDirection.grades",
            "temperature.C", "relHumidity.HR", "airPressure.mb", "precipitation.l.m2",
            "bikes_3h_ago", "full_profile_3h_diff_bikes", "full_profile_bikes",
            "short_profile_3h_diff_bikes", "short_profile_bikes", "bikes"]
#create empty list
output=[]
#for all stations
for x in range(201,240):
      filestring = 'Train/station_' +str(x) +'_deploy.csv'
      data = np.genfromtxt(filestring, dtype=float, comments='#', delimiter=',',
                        skip_header=1, skip_footer=0, converters=None, missing_values={"NA"},
                        filling_values='1', usecols=(0,1,2,3,4,5,6,7,8,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24),
                        names=None, excludelist=None, deletechars=None, replace_space='_',
                        autostrip=False, case_sensitive=True, defaultfmt='f%i',
                        unpack=None, usemask=False, loose=True, invalid_raise=True)
      #zero the averages
      numavg = 0
      avg =0 
      #for every record in the station
      for y in data:
            #if its time is 
            if y[heading.index("weekhour")] == 49:
                  avg += y[heading.index("bikes")]
                  numavg+=1
      avg /= numavg
      out = [avg,data[1][heading.index("latitude")],data[1][heading.index("longitude")]]
      #append a record of average and lat and long to the output
      output.append(out)
#print(output[0])
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter([l[1] for l in output],[l2[2] for l2 in output],[l2[0] for l3 in output])
plt.show()
#print(data)
#print(data.size)
# Change these strings according to what you want to plot!
#x_var = "weekhour"
#y_var = "bikes"

#fig = plt.figure()
#ax = fig.add_subplot(111)
#ax.set_xlabel(x_var)
#ax.set_ylabel(y_var)
#ax1 = heading.index(x_var)
#ax2 = heading.index(y_var)
#ax.plot(data[:,ax1], data[:,ax2])
#plt.show()*/
