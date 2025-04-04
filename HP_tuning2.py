import argparse
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import seaborn as sns
import matplotlib.pyplot as plt

def main():
    # Set up command-line arguments.
    parser = argparse.ArgumentParser(
        description="Hyperparameter tuning for XGBoost using GridSearchCV with heatmap visualization"
    )
    parser.add_argument("--data", type=str, required=True,
                        help="Path to CSV data file")
    parser.add_argument("--learning_rates", type=float, nargs="+", required=True,
                        help="List of learning rates, e.g., 0.01 0.05 0.1")
    parser.add_argument("--n_estimators", type=int, nargs="+", required=True,
                        help="List of n_estimators, e.g., 100 200 300 500")
    parser.add_argument("--max_depth", type=int, nargs="+", required=True,
                        help="List of max_depth values, e.g., 3 5 7")
    parser.add_argument("--subsample", type=float, nargs="+", required=True,
                        help="List of subsample values, e.g., 0.8 1.0")
    parser.add_argument("--colsample_bytree", type=float, nargs="+", required=True,
                        help="List of colsample_bytree values, e.g., 0.8 1.0")
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="Fraction of the data to use as test set (default: 0.2)")
    parser.add_argument("--cv", type=int, default=3,
                        help="Number of cross-validation folds (default: 3)")
    parser.add_argument("--n_jobs", type=int, default=-1,
                        help="Number of parallel jobs for GridSearchCV (default: -1)")
    parser.add_argument("--random_state", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--best_params_out", type=str, default=None,
                        help="Optional: file path to save the best hyperparameters as JSON")
    parser.add_argument("--heatmap_out", type=str, default=None,
                        help="Optional: file path to save the hyperparameter heatmap as PNG")
    args = parser.parse_args()

    # Load the CSV data into a DataFrame.
    df = pd.read_csv(args.data)
    # Assume that the target column is named 'expression'.
    X = df.drop(columns=['expression'])
    y = df['expression']

    # Create binned target for stratification.
    y_bins = pd.qcut(y, q=10, duplicates='drop')
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y_bins, shuffle=True
    )

    # Build the parameter grid from input arguments.
    param_grid = {
        "learning_rate": args.learning_rates,
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree
    }

    print("Parameter grid:")
    print(param_grid)

    # Initialize the XGBRegressor.
    xgb_reg = XGBRegressor(objective='reg:squarederror',
                           tree_method='hist',
                           random_state=args.random_state)

    # Set up GridSearchCV.
    grid_search = GridSearchCV(
        estimator=xgb_reg,
        param_grid=param_grid,
        scoring='neg_mean_squared_error',
        cv=args.cv,
        n_jobs=args.n_jobs,
        verbose=1
    )

    # Fit GridSearchCV on the training data.
    print("Starting hyperparameter tuning...")
    grid_search.fit(X_train, y_train)
    
    # Display the best parameters and best score.
    best_params = grid_search.best_params_
    print("Best hyperparameters found:")
    print(best_params)
    print("Best CV score (negative MSE): {:.4f}".format(grid_search.best_score_))

    # Save the best hyperparameters to a file if specified.
    if args.best_params_out:
        with open(args.best_params_out, "w") as f:
            json.dump(best_params, f, indent=4)
        print(f"Best hyperparameters saved to {args.best_params_out}")

    # Evaluate the best model on the test set.
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print("Test RMSE: {:.4f}".format(rmse))
    print("Test R^2 Score: {:.4f}".format(r2))

    # Generate a heatmap of hyperparameter performance.
    # Here, we aggregate the mean test score for combinations of learning_rate and n_estimators.
    results = pd.DataFrame(grid_search.cv_results_)
    heatmap_df = results.groupby(["param_learning_rate", "param_n_estimators"])["mean_test_score"].mean().reset_index()
    heatmap_pivot = heatmap_df.pivot(index="param_n_estimators", columns="param_learning_rate", values="mean_test_score")
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(heatmap_pivot, annot=True, fmt=".4f", cmap="viridis")
    plt.title("Heatmap of Mean Test Score\n(aggregated over other hyperparameters)")
    plt.xlabel("Learning Rate")
    plt.ylabel("n_estimators")
    plt.tight_layout()
    
    if args.heatmap_out:
        plt.savefig(args.heatmap_out, format='png')
        print(f"Hyperparameter heatmap saved to {args.heatmap_out}")
    else:
        plt.show()

if __name__ == '__main__':
    main()

# Example command to run the script:
# python HP_tuning2.py --data simulated_5mer_data.csv --learning_rates 0.001 0.01 0.02 0.5 --n_estimators 500 1000 2000 5000 --max_depth 2 3 5 --colsample_bytree 0.8 1.0 --subsample 0.8 1.0 --heatmap_out heatmap3.png --best_params_out best_hp.json
# python HP_tuning2.py --data simulated_5mer_data.csv \
#     --learning_rates 0.001 0.01 0.02 0.5 \
#     --n_estimators 500 1000 2000 5000 \
#     --max_depth 2 3 5 \
#     --colsample_bytree 0.8 1.0 \
#     --subsample 0.8 1.0 \
#     --heatmap_out heatmap3.png \
#     --best_params_out best_hp.json
