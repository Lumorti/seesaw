#!/bin/bash

# Params
numRepeats=100
d=3
n=3

# Keep repeating
total=0
for i in $(seq 1 $numRepeats)
do
	val=$(./seesaw -D -d $d -n $n)
	echo "$i $val"
	total=$(echo "scale=8;$total+$val" | bc)
done

# Calculate the average
avg=$(echo "scale=8;$total/$numRepeats" | bc)
echo "average = $avg"

