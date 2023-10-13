#!/bin/bash

echo "////////////////////////////////////////////////////////////////////////////////////////////"
echo
squeue -u superneutrinos1214

echo
echo "==========================================================================================="
echo

## Get the name of the using node
# node=$(squeue -u superneutrinos1214 -o "%N" -h)
ith_job=$1 ## Parameter that determines the ith running job (the job with smaller 'job_id' has smaller 'ith_job')
node=$(squeue -u superneutrinos1214 | tail -n +2 | sort -k1,1n | awk -v i=${ith_job} 'NR==i {print $8}')   ## This will sort 'ith_job' with the 'job_id' 

# Show the status of GPUs in the using node
pdsh -w ${node} nvidia-smi

echo
echo "==========================================================================================="
echo

## Get the id of the running job
# job_id=$(squeue -u superneutrinos1214 -o "%i" -h)
job_id=$(squeue -u superneutrinos1214 | tail -n +2 | sort -k1,1n | awk -v i=${ith_job} 'NR==i {print $1}')

## Show the output
nl slurm_log/err/SuperK_ML.${job_id}.out

echo
echo "==========================================================================================="
echo

## Show the output
cat slurm_log/out/SuperK_ML.${job_id}.out