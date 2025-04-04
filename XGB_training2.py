import argparse
import json
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
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

    # Evaluate on test set.
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

