import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.nan)


heading =   ["station", "latitude", "longitude", "numDocks,"
            "timestamp", "year", "month", "day", "hour", "weekday", "weekhour", "isHoliday",
            "windMaxSpeed.m.s", "windMeanSpeed.m.s", "windDirection.grades",
            "temperature.C", "relHumidity.HR", "airPressure.mb", "precipitation.l.m2",
            "bikes_3h_ago", "full_profile_3h_diff_bikes", "full_profile_bikes",
            "short_profile_3h_diff_bikes", "short_profile_bikes", "bikes"]

data = np.genfromtxt('Train/station_201_deploy.csv', dtype=float, comments='#', delimiter=',',
                        skip_header=1, skip_footer=0, converters=None, missing_values={"NA"},
                        filling_values='1', usecols=(0,1,2,3,4,5,6,7,8,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24),
                        names=None, excludelist=None, deletechars=None, replace_space='_',
                        autostrip=False, case_sensitive=True, defaultfmt='f%i',
                        unpack=None, usemask=False, loose=True, invalid_raise=True)

#print(data)
#print(data.size)

# Change these strings according to what you want to plot!
x_var = "weekhour"
y_var = "bikes"

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlabel(x_var)
ax.set_ylabel(y_var)
ax1 = heading.index(x_var)
ax2 = heading.index(y_var)
ax.plot(data[:,ax1], data[:,ax2])
plt.show()
