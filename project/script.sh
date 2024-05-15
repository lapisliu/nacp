#!/bin/bash

# Function to run the command and extract the runtime
run_command() {
    local output=$(./nacp "$1" "$2" "$3" -b -w -l)
    local cublas_time=$(echo "$output" | grep "cublas run time" | awk '{print $NF}')
    local wmma_time=$(echo "$output" | grep "wmma run time" | awk '{print $NF}')
    local tiling_time=$(echo "$output" | grep "basic tiling run time" | awk '{print $NF}')
    echo "$cublas_time, $wmma_time, $tiling_time"
}

# CSV header
echo "m=n, cublas time(ms), wmma time(ms), basic tilling time(ms)" > m_eq_n_runtimes.csv
echo "k, cublas time(ms), wmma time(ms), basic tilling time(ms)" > k_runtimes.csv

# Loop for m=n case
for ((exp=10; exp<=16; exp++)); do
    size=$((2**exp))
    result=$(run_command "-m=$size" "-n=$size" "-k=4096")
    echo "$size, $result" >> m_eq_n_runtimes.csv
done

# Loop for k case
for ((exp=10; exp<=15; exp++)); do
    size=$((2**exp))
    result=$(run_command "-m=4096" "-n=4096" "-k=$size")
    echo "$size, $result" >> k_runtimes.csv
done

