#!/bin/bash
#cmake -B build -S .. -DCMAKE_C_COMPILER=/opt/homebrew/bin/gcc-15 -DCMAKE_CXX_COMPILER=/opt/homebrew/bin/g++-15 -DCMAKE_CXX_STANDARD=20
#make -j8
TASK=spectrum
NCPU=10

make -j8 ${TASK}/fast

sudo purge
sync && sleep 1

LOG="${TASK}_result{$NCPU}_12G.log"
PNG="${TASK}_result{$NCPU}_12G.png"

psrecord "mpirun -np ${NCPU} ./${TASK}" --interval 0.01 --include-children --log "$LOG" --plot "$PNG"
