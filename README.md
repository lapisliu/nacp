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

## Eval usage
The runner-kernel architecture for evaluation is in `project/`. To build the runner, first make sure the GPU arch number in `CMakeLists.txt` matches the arch number of the current GPU. Then `mkdir build && cd build` and `cmake ..`.
