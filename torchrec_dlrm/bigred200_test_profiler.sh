#!/bin/bash

#SBATCH --job-name=auto
#SBATCH -p gpu
#SBATCH --nodes=4
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --time=00:30:00
#SBATCH --output=bigred_test_%j.log 
#SBATCH --mem=200G

#module load nvidia
#export LD_LIBRARY_PATH=/N/soft/sles15/nvidia/21.5/Linux_x86_64/21.5/comm_libs/nccl/lib/:$LD_LIBRARY_PATH
#export LD_LIBRARY_PATH=/N/soft/sles15/nvidia/21.5/Linux_x86_64/21.5/comm_libs/openmpi4/openmpi-4.0.5/lib/:$LD_LIBRARY_PATH
#export PATH=/N/soft/sles15/nvidia/21.5/Linux_x86_64/21.5/comm_libs/openmpi4/openmpi-4.0.5/bin/:$PATH

#echo $LD_LIBRARY_PATH

export DLRM_DISTRIBUTED="TRUE"
export MASTER_PORT=47149
export WORLD_SIZE=16
echo "WORLD_SIZE="$WORLD_SIZE
echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR
export EPOCH=1
export DATASET_PATH="/N/scratch/haofeng/TB/processed/"
# NCCL_DEBUG=INFO
#export NCCL_DEBUG_SUBSYS=ALL
cd ~/new_dlrm/torchrec_dlrm/
source ~/.bashrc
conda init
conda activate dlrm

srun -n 16 python dlrm_main.py --epochs=$EPOCH \