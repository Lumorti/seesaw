#!/bin/gnuplot --persist
set term wxt
set view equal xyz
set parametric
set urange [-pi/2:pi/2]
set vrange [0:2*pi] 
set isosamples 13,13

# Draw the state names
set label "|0>" at 0,0,1.2
set label "|1>" at 0,0,-1.2
set label "|+>" at 1.2,0,0
set label "|->" at -1.2,0,0
set label "|i+>" at 0,1.2,0
set label "|i->" at 0,-1.2,0

# Hide the usual stuff
unset border
unset xtics
unset ytics
unset ztics
unset key

set origin -0.1,-0.1
set size 1.2,1.2

# Draw the Bloch sphere with arrows
R=1.0
splot R*cos(u)*cos(v),R*cos(u)*sin(v),R*sin(u), "test.dat" with vectors title ""

