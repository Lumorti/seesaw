#!/usr/bin/python3
import tsne
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import sys

# Load the data
dataSets = []
for arg in sys.argv[1:]:
    with open(arg, "r") as f:
        data = f.readlines()
        dataSets.append(data)

# Convert to a 2D numpy float array
proc = []
for data in dataSets:
    for x in data:
        split = x[1:len(x)-4].split(",")
        fixed = []
        for y in split:
            fixed.append(float(y))
        proc.append(fixed)
asNp = np.array(proc);

# Perfrom tsne on it
reduced = tsne.tsne(asNp, 3, 5)

# Create the 3D vis object
fig = plt.figure()
ax = plt.axes(projection='3d')

# Split the data into the x, y and z components
soFar = 0
for i in range(len(dataSets)):
    x = []
    y = []
    z = []
    for v in reduced[soFar:soFar+len(dataSets[i])]:
        x.append(v[0])
        y.append(v[1])
        z.append(v[2])
    soFar += len(dataSets[i])

    # Plot this line
    ax.plot3D(x, y, z, 'gray')

    # Add some labels
    ax.text(x[0], y[0], z[0], "start " + str(i+1), color='red')
    ax.text(x[-1], y[-1], z[-1], "end " + str(i+1), color='green')

# Render the 3D display
plt.show()
