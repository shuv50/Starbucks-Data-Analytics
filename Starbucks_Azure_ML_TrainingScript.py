# Import libraries
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os
import argparse
import mlflow
import mlflow.sklearn

# Main function
def main():

    # input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="path to input data")
    parser.add_argument("--registered_model_name", type=str, help="model name")
    args = parser.parse_args()
    
    # Start logging
    mlflow.start_run()

    # Enable autologging
    mlflow.sklearn.autolog()

    # Read data
    print(" ".join(f"{k}={v}" for k, v in vars(args).items()))
    print("input data:", args.data)
    starbucks_bogo_data = pd.read_csv(args.data)

    # Log metrics


    # Drop null values
    starbucks_bogo_data.dropna(inplace=True)

    # Typecast columns
    starbucks_bogo_data[['age', 'total_trans', 'rewards_earned', 'time_lapsed_succ', 'income']] = starbucks_bogo_data[['age', 'total_trans', 'rewards_earned', 'time_lapsed_succ', 'income']].astype(int)

    # Select features
    X = starbucks_bogo_data[['gender', 'age', 'offer_view_rate', 'offer_comp_rate', 'total_trans', 'amount_trans', 'rewards_earned', 'time_lapsed_succ', 'income', 'membership_days']].values

    # One-Hot encoding
    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
    X = np.array(ct.fit_transform(X))

    # Train model (K-means Clustering)
    kmeans = KMeans(n_clusters=2, init='k-means++', random_state=42)
    y_kmeans = kmeans.fit_predict(X)

    # Predict/Analyse offer success
    starbucks_bogo_data['offer_success'] = y_kmeans

    # Registering the model to the workspace
    print("Registering the model via MLFlow")
    mlflow.sklearn.log_model(
        sk_model=kmeans,
        registered_model_name=args.registered_model_name,
        artifact_path=args.registered_model_name,
    )

    # Saving the model to a file
    mlflow.sklearn.save_model(
        sk_model=kmeans,
        path=os.path.join(args.registered_model_name, "trained_model"),
    )
    
    # Stop Logging
    mlflow.end_run()

if __name__ == "__main__":
    main()