#!/bin/bash

# Parameters
gridSize=5000
maxGen=5000
outputFile="timing_results.txt"

# Clear output file
echo "" > $outputFile

# Array of executables to test
executables=("hw3_BPTP" "hw3_NBPTP")

# Run tests for each executable
for exe in "${executables[@]}"; do
    echo "Testing executable: $exe" >> $outputFile
    for procs in 1 2 4 8; do
        echo "Number of processes: $procs" >> $outputFile
        for i in {1..3}; do
            echo "Run $i:" >> $outputFile
            gtime -f "%e seconds" mpirun -np $procs ./$exe $gridSize $maxGen 2>&1 | grep "seconds" >> $outputFile
        done
        echo "" >> $outputFile
    done
    echo "========================" >> $outputFile
done
