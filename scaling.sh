#!/bin/bash

# Benchmarking params
repeatsPer=10
minD=2
maxD=6
tol=0.0001

> scaling.dat

# Tell time to only output the real time
TIMEFORMAT=%R

# For each different dimension
for d in $(seq $minD $maxD)
do

	# Repeat a certain number of times
	total=0
	for rep in $(seq 1 $repeatsPer)
	do

		# Time how long it takes
		val=$(./seesaw -d $d -n 3 -t $tol -T)

		# Output so you know it's working
		echo "dim $d repeat $rep finished in $val ms"

		# Add to the total time
		total=$(echo "scale=8;$total+$val" | bc)

	done

	# Calculate the average
	avg=$(echo "scale=8;$total/$repeatsPer" | bc)

	# Save to file
	echo "$d $avg" >> scaling.dat


done 

# Use gnuplot to graph these runs
script=""
script="$script; set term pdf"
script="$script; set output 'scaling.pdf'"
script="$script; set ylabel 'time to converge / s'"
script="$script; set xlabel 'dimension, d'"
script="$script; set key off"
script="$script; plot 'scaling.dat' lw 3 pt 2 title 'scaling data'"
gnuplot -e "$script"
