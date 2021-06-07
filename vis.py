#!/usr/bin/python3
import tsne
import numpy as np

# Load the data
with open("raw.dat", "r") as f:
    data = f.readlines()

# Convert to a 2D numpy float array
proc = []
for x in data:
    split = x[1:len(x)-4].split(",")
    fixed = []
    for y in split:
        fixed.append(float(y))
    proc.append(fixed)
asNp = np.array(proc);

# Perfrom tsne on it
reduced = tsne.tsne(asNp, 3, 10)

# Write this data to a different file
with open("reduced.dat", "w") as f:
    for x in reduced:
        toWrite = ""
        for y in x:
            toWrite += str(y) + " "
        toWrite += "\n"
        f.write(toWrite)
