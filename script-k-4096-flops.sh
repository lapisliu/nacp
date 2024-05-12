#!/bin/bash

# Output file name
output_file="k-4096-flops.csv"

# Write header to output file
echo "m=n size, gflops" > $output_file

# Set constant value for k
k=4096

# Loop through m and n sizes from 2^4 to 2^15
for ((power=4; power<=15; power++))
do
    size=$((2**$power))

    # Run CUDA program and capture kernel time
    kernel_time=$(./14_ampere_tf32_tensorop_gemm --n=$size --m=$size --k=$k --iterations=1| grep "GFLOPs:" | cut -d ' ' -f 3)

    # Append results to output file
    echo "$size, $kernel_time" >> $output_file
done

echo "Done. Results saved in $output_file"

