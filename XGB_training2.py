import argparse
import json
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def main():
    parser = argparse.ArgumentParser(
        description="Train final XGBoost model using hyperparameters from JSON or explicit args."
    )
    parser.add_argument("--data", type=str, required=True,
                        help="Path to CSV data file (must contain an 'expression' column)")
    parser.add_argument("--best", type=str, default=None,
                        help="Path to JSON file with best hyperparameters (e.g., best_hp.json)")
    parser.add_argument("--lr", type=float, default=None,
                        help="Explicit learning rate (if --best is not provided)")
    parser.add_argument("--n_est", type=int, default=None,
                        help="Explicit number of estimators (trees) (if --best is not provided)")
    parser.add_argument("--max_depth", type=int, default=None,
                        help="Explicit max depth (if --best is not provided)")
    parser.add_argument("--sub", type=float, default=None,
                        help="Explicit subsample ratio (if --best is not provided)")
    parser.add_argument("--col", type=float, default=None,
                        help="Explicit colsample_bytree ratio (if --best is not provided)")
    parser.add_argument("--model_out", type=str, default="final_model.json",
                        help="Path to save the final trained model")
    parser.add_argument("--imp_csv", type=str, default="feature_importance.csv",
                        help="Output CSV file for feature importance")
    parser.add_argument("--imp_png", type=str, default="feature_importance.png",
                        help="Output PNG file for feature importance plot")
    parser.add_argument("--scatter_png", type=str, default="true_vs_predicted.png",
                        help="Output PNG file for scatter plot of true vs predicted values")
    parser.add_argument("--plot_out", type=str, default="rmse_epochs.png",
                        help="Path to save the RMSE over epochs plot")
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="Fraction of data to use for testing (default: 0.2)")
    parser.add_argument("--early", type=int, default=50,
                        help="Early stopping rounds (default: 50)")
    parser.add_argument("--n_jobs", type=int, default=-1,
                        help="Number of threads for XGBoost (default: -1)")
    args = parser.parse_args()

    # Load CSV data.
    df = pd.read_csv(args.data)
    X = df.drop(columns=['expression'])
    y = df['expression']

    # Split into training and test sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, shuffle=True
    )

    # Determine hyperparameters: use JSON file if provided, otherwise use explicit arguments.
    if args.best:
        with open(args.best, "r") as f:
            hyperparams = json.load(f)
    else:
        if (args.lr is None or args.n_est is None or args.max_depth is None or 
            args.sub is None or args.col is None):
            raise ValueError("Either --best must be provided or all explicit parameters (--lr, --n_est, --max_depth, --sub, --col) must be specified.")
        hyperparams = {
            "learning_rate": args.lr,
            "n_estimators": args.n_est,
            "max_depth": args.max_depth,
            "subsample": args.sub,
            "colsample_bytree": args.col,
        }
    
    # Extract n_estimators for use as num_boost_round and remove it from hyperparams.
    num_boost_round = hyperparams.pop("n_estimators", 100)

    # Base parameters for xgb.train.
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "nthread": args.n_jobs,
        "random_state": 42,
    }
    params.update(hyperparams)

    # Convert data to DMatrix.
    dtrain = xgb.DMatrix(data=X_train, label=y_train)
    dtest = xgb.DMatrix(data=X_test, label=y_test)

    # Evaluation sets.
    evals = [(dtrain, "train"), (dtest, "eval")]
    evals_result = {}

    print("Training final model with hyperparameters:")
    print(params)
    print(f"Using {num_boost_round} boosting rounds with early stopping = {args.early}.")

    # Train model with early stopping.
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=evals,
        early_stopping_rounds=args.early,
        evals_result=evals_result,
        verbose_eval=True,
    )

    # Save the model.
    model.save_model(args.model_out)
    print(f"Model saved to {args.model_out}")

    # -------------------------------
    # Feature Importance Extraction and Plotting
    # -------------------------------
    #booster = model.get_booster()
    importance_dict = model.get_score(importance_type='gain')
    
    importance_df = pd.DataFrame(list(importance_dict.items()), columns=['Feature', 'Importance'])
    # Save feature importance as CSV.
    importance_df.to_csv(args.imp_csv, index=False)
    print(f"Feature importance saved to {args.imp_csv}")
    
    # Plot feature importance as a bar plot (you can modify to show top features only if desired).
    # For instance, to plot the top 30 features:
    importance_df_sorted = importance_df.sort_values(by='Importance', ascending=False)
    top30 = importance_df_sorted.head(30)
    
    # Plotting
    plt.figure(figsize=(12, 8))
    plt.barh(top30['Feature'], top30['Importance'], color='skyblue')
    plt.xlabel("Importance (Gain)", fontsize=14)
    plt.ylabel("Feature", fontsize=14)
    plt.title("Top 30 Feature Importances", fontsize=16)
    plt.gca().invert_yaxis()  # so that the highest importance appears at the top
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(args.imp_png, format="png")
    print(f"Feature importance plot saved to {args.imp_png}")
    plt.show()

    # Plot RMSE over epochs.
    epochs = list(range(1, len(evals_result["train"]["rmse"]) + 1))
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, evals_result["train"]["rmse"], label="Train RMSE")
    plt.plot(epochs, evals_result["eval"]["rmse"], label="Test RMSE")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.title("RMSE Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(args.plot_out, format="png")
    print(f"RMSE plot saved to {args.plot_out}")
    plt.close()

    y_train_pred = model.predict(dtrain)
    y_test_pred = model.predict(dtest)

    # Calculate Pearson correlation coefficients.
    pearson_train, p_train = pearsonr(y_train, y_train_pred)
    pearson_test, p_test = pearsonr(y_test, y_test_pred)

    # Also, obtain the true labels from the training and test sets.
    # (Assuming you have y_train and y_test already from train_test_split)

    plt.figure(figsize=(10, 8))

    # Plot the training set scatter and regression line.
    sns.regplot(x=y_train, y=y_train_pred, ci=None, color='blue', marker='o',
                label=f"Train (Pearson'r={pearson_train:.2f})", scatter_kws={'alpha':0.4})

    # Plot the test set scatter and regression line.
    sns.regplot(x=y_test, y=y_test_pred, ci=None, color='red', marker='s',
                label=f"Test (Pearson'r={pearson_test:.2f})", scatter_kws={'alpha':0.4})

    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title("True vs Predicted Values (Train and Test Sets)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.scatter_png, format="png")
    print(f"Scatter plot saved to {args.scatter_png}")
    plt.show()

    # Final evaluate on test set.
    y_pred = model.predict(dtest)
    mse = mean_squared_error(y_test, y_pred)
    final_rmse = np.sqrt(mse)
    final_r2 = r2_score(y_test, y_pred)
    print(f"Final Test RMSE: {final_rmse:.4f}")
    print(f"Final Test R² Score: {final_r2:.4f}")

if __name__ == "__main__":
    main()


# python XGB_training2.py --data simulated_5mer_data.csv --best best_hp.json --model_out final_model.json --plot_out rmse_epochs.png
# python XGB_training2.py --data simulated_5mer_data.csv --lr 0.01 --n_est 500 --max_depth 3 --sub 0.8 --col 0.8 --model_out final_model.json --plot_out rmse_epochs.png
# python XGB_training2.py --data merged_kmer_motif_counts_raw_expression.csv --lr 0.01 --n_est 20000 --max_depth 3 --sub 0.8 --col 0.8 --early 200 --imp_csv imp5.csv --imp_png imp5.png --scatter_png scatter5.png --model_out final_model5.json --plot_out rmse_epochs5.png
