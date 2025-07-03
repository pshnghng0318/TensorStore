#!/bin/bash
#cmake -B build -S .. -DCMAKE_C_COMPILER=/opt/homebrew/bin/gcc-15 -DCMAKE_CXX_COMPILER=/opt/homebrew/bin/g++-15 -DCMAKE_CXX_STANDARD=20 -DCMAKE_PREFIX_PATH="/opt/homebrew/Cellar/tbb/2022.1.0"
#make -j8
make -j8 read_zarr/fast

sudo purge
sync && sleep 1
NCPU=10
MEM=64

#LOG="result"$NCPU"_chunk"$MEM".log"
#PNG="result"$NCPU"_chunk"$MEM".png"
LOG="result"$NCPU"_12G.log"
PNG="result"$NCPU"_12G.png"

psrecord "mpirun -np "$NCPU" ./read_zarr" --interval 1 --include-children --log "$LOG" --plot "$PNG"
