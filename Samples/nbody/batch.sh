#!/bin/bash

for i in $(seq -f "%03g" 10 1 40)
do
    python nbody.py -g ${i} -f 400
    python plot.py -o Output/nbody_${i}.gif -f 400
    echo
    echo "#######################################"
    echo "###########  STEP ${i} DONE  ###########"
    echo "#######################################"
    echo
done
