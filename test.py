import numpy as np
from StringIO import StringIO
data = "1, 2, 3\n4, 5, 6"
def convert (x):
	return 999
	
print np.genfromtxt(StringIO(data), delimiter=",",converters={2: convert},usecols=(1))
