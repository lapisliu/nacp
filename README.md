[Report](https://docs.google.com/document/d/13YZf67VEWa1RP4K8-cYT2x6rXxF2J_UD3uUGBrzc_NI/edit?usp=sharing)

## Repo structure
"project/" folder contains the `cublas-mix`, `basic-tiling`, and `wmma-mix` kernels and the runner to test them.

"benchmark-data/" folder contains the profiling data. The `k_runtimes.csv` and `m_eq_n_runtimes.csv` contains the data of the above three algorithms used in Figure 4 in the report. Others are benchmarks of cutlass kernel. (The cutlass kernel was built seperately by cloning the cutlass repo. This is the [cutlass tf32_tensorop code](https://github.com/NVIDIA/cutlass/blob/main/examples/14_ampere_tf32_tensorop_gemm/ampere_tf32_tensorop_gemm.cu) used in the analysis. And here are the codes tested in the abolition study. [sgemm_1](https://github.com/NVIDIA/cutlass/blob/main/examples/cute/tutorial/sgemm_1.cu], [sgemm_2](https://github.com/NVIDIA/cutlass/blob/main/examples/cute/tutorial/sgemm_2.cu), [sgemm_80](https://github.com/NVIDIA/cutlass/blob/main/examples/cute/tutorial/sgemm_sm80.cu).

Other files were for testing.


## Eval usage
The runner-kernel architecture for evaluation is in `project/`. To build the runner, first make sure the GPU arch number in `CMakeLists.txt` matches the arch number of the current GPU. Then `mkdir build && cd build` and `cmake ..`.

## Nvidia running Instruction On Greene
`srun --cpus-per-task=8 --mem=80GB --gres=gpu:1 --time=04:00:00 --pty /bin/bash`

`singularity exec --nv /scratch/work/public/singularity/cuda12.2.2-cudnn8.9.4-devel-ubuntu22.04.3.sif /bin/bash`

`nvidia-smi` //should out put gpu info

`nvcc -V` //should give 12.2

`nvcc mat-mul.cu -o mat-mul.out` //compile

`time ./mat-mul.out` //time

## AMD running instruction
`srun --cpus-per-task=8 --mem=80GB --gres=gpu:mi250:1 --pty /bin/bash`

`singularity exec --rocm /scratch/work/public/singularity/rocm5.7.1-ubuntu22.04.3.sif /bin/bash`

`rocm-smi` //should give amd gpu info

Hipify the cuda code

`hipify-perl mat-mul.cu > mat-mul.cu.hip`

Compile

`hipcc mat-mul.cu.hip -o mat-mul-amd.out`

`time ./mat-mul-amd.out`
