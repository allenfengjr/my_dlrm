#!/bin/bash

#SBATCH --job-name=auto
#SBATCH -p gpu
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --time=00:30:00
#SBATCH --output=output_%j.log 

#module load nvidia
#export LD_LIBRARY_PATH=/N/soft/sles15/nvidia/21.5/Linux_x86_64/21.5/comm_libs/nccl/lib/:$LD_LIBRARY_PATH
#export LD_LIBRARY_PATH=/N/soft/sles15/nvidia/21.5/Linux_x86_64/21.5/comm_libs/openmpi4/openmpi-4.0.5/lib/:$LD_LIBRARY_PATH
#export PATH=/N/soft/sles15/nvidia/21.5/Linux_x86_64/21.5/comm_libs/openmpi4/openmpi-4.0.5/bin/:$PATH

#echo $LD_LIBRARY_PATH


export MASTER_PORT=47149
export WORLD_SIZE=8
echo "WORLD_SIZE="$WORLD_SIZE
echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

cd ~/new_dlrm/torchrec_dlrm/
source ~/.bashrc
conda init
conda activate dlrm
srun -n 8 python dlrm_main.py --epochs=20
