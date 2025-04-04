# Gene-Expression-Prediction
## input file: 
simulated_5mer_data.csv
## how to use the scripts:
### 1. Hyperparameter tuning
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
### 2.1 Training the model using best parameters from step1
```
python XGB_training2.py --data simulated_5mer_data.csv \
    --best best_hp.json \
    --model_out final_model.json \
    --plot_out rmse_epochs.png
```
### 2.2 training the model using explicitly defined parameters
```
python XGB_training2.py --data simulated_5mer_data.csv \
    --lr 0.01 --n_est 5000 --max_depth 3 --sub 0.8 --col 0.8 \
    --model_out final_model.json --plot_out rmse_epochs.png
```
