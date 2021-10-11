import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Load the file
with open("temp.log") as f:
    data = f.readlines();

# Create the data arrays
xData = []
minData = []
maxData = []
for line in data:
    split = line.split()
    if len(split) >= 6:
        if split[0] == "Scaled" and split[1] == "bounds:":
            minData.append(float(split[2]))
            maxData.append(float(split[6]))

# Where to cut the arrays
start = 0
end = len(maxData)
extended = 10*len(maxData)

# Recast into numpy arrays
xData = np.asarray(range(start+1,end+1)) / 10000.0
xDataExtended = np.asarray(range(start+1,extended+1)) / 10000.0
minData = np.asarray(minData[start:end])
maxData = np.asarray(maxData[start:end])
sigma = 1 / xData

# Define the fitting function
def fit(x, a, b, c):
    return np.power(x, -a)*c + b

def inverse(f, a, b, c):
    return np.exp(-np.log((f - b) / c) / a)

# Fit the data
parameters, covariance = curve_fit(fit, xData, maxData, sigma=sigma)

# Output things
print("params:", parameters)
print("estimated iteratons required:", inverse(minData[end-1], *parameters))

# Plot the orig and the fit
plt.plot(xData, minData, '-', label='min bound')
plt.plot(xData, maxData, '-', label='max bound')
maxFit = fit(xDataExtended, *parameters)
plt.plot(xDataExtended, maxFit, '-', label='max fit')
plt.legend()
plt.show()


