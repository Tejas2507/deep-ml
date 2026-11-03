import math
import numpy as np
PI = 3.14159

def power_grid_forecast(consumption_data):
	# 1) Subtract the daily fluctuation (10 * sin(2π * i / 10)) from each data point.
	# 2) Perform linear regression on the detrended data.
	# 3) Predict day 15's base consumption.
	# 4) Add the day 15 fluctuation back.
	# 5) Round, then add a 5% safety margin (rounded up).
	# 6) Return the final integer.
	n = len(consumption_data)
	consumption_data = [consumption_data[i]-(10*math.sin(2*PI*(i+1)/10)) for i in range(n)]
	y = np.asarray(consumption_data)
	X = np.arange(1,n+1)
	X = np.column_stack((X, np.ones(n)))
	w,b = np.linalg.lstsq(X,y)[0]

	pred = round(w*(15) + b + 10*math.sin(2*PI*(15)/10))

	return math.ceil(1.05*pred)




