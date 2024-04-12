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

