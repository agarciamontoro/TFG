#!/bin/bash

FRAMES=700

for i in $(LANG=en_EN seq -f "%1.1f" 1.0 0.1 3.0)
do
    echo
    echo "SIMULATING..."
    python nbody.py -g ${i} -f ${FRAMES} -b 81920 -d all_bodiesw

    echo
    echo "RENDERING..."
    python plot.py -i all_bodies/nbody.hdf5 -f ${FRAMES} -b 81920 -o all_bodies/anim_${i}.gif

    echo
    echo "#######################################"
    echo "###########  STEP ${i} DONE  ###########"
    echo "#######################################"
    echo
done
