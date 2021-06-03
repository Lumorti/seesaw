#!/bin/bash

# Params
numRepeats=1000
d=2
n=2

# Keep repeating
total=0
for i in $(seq 1 $numRepeats)
do
	val=$(./seesaw -D -d $d -n $n)
	echo "$i $val"
	if (( $(echo "$val < 100" | bc -l) ))
	then
		total=$(echo "scale=5;$total+1" | bc)
	fi
done

# Calculate the average
avg=$(echo "scale=0;100*$total/$numRepeats" | bc)
echo "success chance = $avg%"

