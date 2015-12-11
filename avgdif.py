import sys
import numpy as np



values = np.genfromtxt("coefs.csv", dtype=float, comments='#', delimiter=',',
                  skip_header=0, skip_footer=0, converters=None, missing_values={"NA"},
                  filling_values='0', usecols=None,
                  names=None, excludelist=None, deletechars=None, replace_space='_',
                  autostrip=False, case_sensitive=True, defaultfmt='f%i',
                  unpack=None, usemask=False, loose=True, invalid_raise=True)

totals = [0,0,0,0,0,0,0,0]
for row in values:
    totals = totals + row

for i in range(0,7):
    totals[i] = totals[i]/75
f = open("difs.csv", 'w')
counts = 0
for row in values:
    counts = counts +1
    for i in range(0,7):
        f.write(str(row[i]-totals[i]) + ',')
    f.write('\n')
f.close()
print counts