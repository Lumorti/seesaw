#!/bin/bash

make

total=0

numRepeats=500
d=3
n=2

# Keep repeating
for i in $(seq 1 $numRepeats)
do
	val=$(./seesaw -e -D -d $d -n $n)
	echo "$i $val"
	total=$(echo "scale=5;$total+$val" | bc)
done

avg=$(echo "scale=5;$total/$numRepeats" | bc)
echo "average = $avg"

