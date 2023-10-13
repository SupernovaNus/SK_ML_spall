# SK_ML_spall

CNNs for classifying the muon-induced backgrounds.

Create conda env:  ```. conda_env_setup.sh```


To train spall_ml models, run the SLURM job script on HPC:
```
sbatch train_script.sh
```
or run the following command at the local:
```
python SK_spall_train_muon_scatt.py -c configs/configs.yaml
```
