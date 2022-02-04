#!/bin/bash

numRepeats=10000
d=2
n=2

# Keep repeating
total=0
for rep in $(seq $1 $numRepeats)
do

	# Run the sim
	val=$(./seesaw -d $d -n $n -Z)
	echo "repeat $rep out of $numRepeats gives $val"

	# See if convergenced to zero
	if [ "$val" -eq "1" ]; then
		total=$(echo "$total+1" | bc)
	fi

done

# Calculate percent chance of convergence
avg=$(echo "scale=2;100.00*$total/$numRepeats" | bc)
echo "$avg% of $numRepeats runs reached zero for d=$d n=$n"
