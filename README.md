## Nvidia running Instruction On Greene
`srun --cpus-per-task=8 --mem=80GB --gres=gpu:1 --time=04:00:00 --pty /bin/bash`

`singularity exec /scratch/work/public/singularity/cuda12.2.2-cudnn8.9.4-devel-ubuntu22.04.3.sif /bin/bash`

`nvcc -V` //should give 12.2

