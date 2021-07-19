import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Load the file
with open("temp.dat") as f:
    data = f.readlines();

# Create the data arrays
xData = []
yData = []
for line in data:
    split = line.split()
    xData.append(int(split[0]))
    yData.append(float(split[2]))

#Recast xdata and ydata into numpy arrays so we can use their handy features
start = 0
end = len(xData)
xData = np.asarray(xData[start:end])
yData = np.asarray(yData[start:end])
sigma = xData / xData

# Define the fitting function
def fit(x, a, b):
    return np.power(x, -a)+b

# Fit the data
parameters, covariance = curve_fit(fit, xData, yData, sigma=sigma, absolute_sigma=True)

print(parameters)

# Plot the orig and the fit
yFit = fit(xData, *parameters)
plt.plot(xData, yData, 'o', label='data')
plt.plot(xData, yFit, '-', label='fit')
plt.legend()
plt.show()


