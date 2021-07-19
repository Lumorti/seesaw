grep "Scaled bounds" temp.log | awk '{print NR " " $3 " " $7}' > temp.dat
echo 'plot "temp.dat" u 1:2 w l title "lower bound", "temp.dat" u 1:3 w l title "upper bound"' | gnuplot --persist
