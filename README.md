# Gene-Expression-Prediction
## class project for CS/BMI776
### input file: 
simulated_5mer_data.csv
## 1. how to use the scripts:
#### 1.1 Hyperparameter tuning using GridSearchCV
```
python HP_tuning2.py --data simulated_5mer_data.csv \
    --learning_rates 0.001 0.005 0.01 0.05 0.1  \
    --n_estimators 500 1000 2000 5000 10000 \
    --max_depth 2 3 5 \
    --subsample 0.8 0.9 1.0 \
    --colsample_bytree 0.8 0.9 1.0 \
    --heatmap_out hp_heatmap3.png \
    --best_params_out best_hp.json
```
#### 1.2.1 Training the model using best hyperparameters from step1
```
python XGB_training2.py --data simulated_5mer_data.csv \
    --best best_hp.json \
    --model_out final_model.json \
    --plot_out rmse_epochs.png
```
#### 1.2.2 Training the model using explicitly defined hyperparameters
```
python XGB_training2.py --data merged_kmer_motif_counts_raw_expression.csv \
    --lr 0.01 --n_est 20000 --max_depth 3 --sub 0.8 --col 0.8 --early 200 \
    --imp_csv imp5.csv --imp_png imp5.png --scatter_png scatter5.png \
    --model_out final_model5.json --plot_out rmse_epochs5.png
```

## 2. Train the model using UW-Madison CHTC
### input files:
```
simulated_5mer_data.csv   
xgboost_env.sif  # environment image   
XGB_training.sh  # executable bash file   
XGB_training.sub # submit the bash file to CHTC
```

#### submit a job 
```
condor_submit XGB_training.sub
```
#### .sub  job submission file
```
# XGB_training.sub

# Provide HTCondor with the name of your .sif file and universe information
container_image = file:///staging/wdong54/cs776/xgboost_env.sif         ### location of the .sif

executable = XGB_training.sh                                            ### executable bash file

# Include other files that need to be transferred here.
# transfer_input_files = HP_tuning2.py,simulated_5mer_data.csv

log = train.log
error = train.err
output = train.out

requirements = (HasCHTCStaging == true)

# Make sure you request enough disk for the container image in addition to your other input files
request_cpus = 24
request_memory = 64GB
request_disk = 20GB

queue
```
#### .sh executable bash file
```
#!/bin/bash

date "+%T"  ## print the starting time
cp /staging/wdong54/cs776/* .
echo "start training ... "

/opt/conda/envs/xgboost/bin/python XGB_training2.py --data simulated_5mer_data.csv \
    --lr 0.01 --n_est 5000 --max_depth 3 --sub 0.8 --col 0.8 \
    --model_out final_model.json --plot_out rmse_epochs.png

date "+%T"  ## print the end time
```
