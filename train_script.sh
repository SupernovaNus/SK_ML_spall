#!/bin/bash

#SBATCH --account PAS1140

#SBATCH --time=5:00:00
#SBATCH --job-name SuperK_ML
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=2

#SBATCH --output=/users/PCON0003/superneutrinos1214/SK_spallation/slurm_log/out/%x.%j.out
#SBATCH --error=/users/PCON0003/superneutrinos1214/SK_spallation/slurm_log/err/%x.%j.out
#SBATCH --mail-type=BEGIN,END,FAIL


## Note: Maximum of gpus-per-node = 4
## Note: Maximum of nodes = 2 if gpus-per-node = 3 or 4
## Note: Maximum of nodes = 10 if gpus-per-node = 1 or 2

#---------------------------------------------------------------------------

module load miniconda3
module load cuda

conda activate spall_ml

## Get the id of the running job
job_id=$(squeue -u superneutrinos1214 | awk 'NR>1 {if ($1 > max) {max = $1}} END {print max}')
mkdir output/${job_id}
cp configs/configs.yaml output/${job_id}/configs.yaml

python SK_spall_train_muon_scatt.py ${job_id} -c configs/configs.yaml

#---------------------------------------------------------------------------