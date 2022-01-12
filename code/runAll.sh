#!/bin/bash

# For each different dimension
for d in $(seq 2 6)
do

	# Go to d+2
	plus2=$(echo "$d+2" | bc)
	for n in $(seq 2 $plus2)
	do

		# Run the seesaw
		echo "running d${d}n${n}"
		./seesaw -d $d -n $n | tee ../results/d${d}n${n}.log

	done

done
