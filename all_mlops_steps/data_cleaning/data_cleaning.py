import pandas as pd
import numpy as np
import os
import argparse
import mlflow
import wandb
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

### HELPER FUNCTIONS ###
wandb.init(project="creadit-card-default-prediction_test-101", name="data-cleaning-1")

def load_data():
    """
    Load the raw dataset from WandB Artifacts.
    
    :return: pandas DataFrame
    """
    try:
        # Download the artifact from WandB
        artifact = wandb.use_artifact('bhushanmandava16-personal/creadit-card-default-prediction_test-101/original_data:v1', type='dataset')
        artifact_dir = artifact.download()#we are downloading the artifact to the local directory

        # Load the dataset
        data = pd.read_csv(os.path.join(artifact_dir, 'creditcard.csv'))
        wandb.log({"raw_data_shape": data.shape})
        
        return data
    except Exception as e:
        print(f"Error loading data from wandb: {e}")
        print("Please check your permissions and network connection.")
        exit(1)

def save_data(data):
    """
    Save the cleaned dataset to WandB as an artifact.
    
    :param data: pandas DataFrame, cleaned data
    """
    # Save the cleaned data locally
    cleaned_data_path = 'processed_data.csv'
    data.to_csv(cleaned_data_path, index=False)
    
    # Log the cleaned data as an artifact in WandB
    artifact = wandb.Artifact(name='processed_data.csv', type='dataset')
    artifact.add_file(cleaned_data_path)
    wandb.log_artifact(artifact)
    wandb.log({"cleaned_data_shape": data.shape})

### DO NOT CHANGE THE CODE ABOVE THIS LINE ###

def clean_data(data):
    """
    Clean the dataset by handling missing values, outliers, and other anomalies.
    
    :param data: pandas DataFrame, raw data
    :return: pandas DataFrame, cleaned data
    """
    # Step 1: Handle Missing Values
    imputer = SimpleImputer(strategy="median")
    numeric_data = data.select_dtypes(include=[np.number])
    imputed_data = pd.DataFrame(imputer.fit_transform(numeric_data), columns=numeric_data.columns)

    # Save imputer
    joblib.dump(imputer, 'imputer.pkl')
    artifact = wandb.Artifact(name='imputer.pkl', type='model')
    artifact.add_file('imputer.pkl')
    wandb.log_artifact(artifact)

    # Step 2: Handle Outliers
    outlier_threshold = 1.5  # Define this variable
    for column in imputed_data.columns: 
        if column != "is_fraud": 
            Q1 = imputed_data[column].quantile(0.25) 
            Q3 = imputed_data[column].quantile(0.75) 
            IQR = Q3 - Q1 
            outlier_range = outlier_threshold * IQR 
            imputed_data = imputed_data[ 
                (imputed_data[column] >= Q1 - outlier_range) &  
                (imputed_data[column] <= Q3 + outlier_range) 
            ]

    # Log outlier removal stats
    outliers_removed = numeric_data.shape[0] - imputed_data.shape[0]
    wandb.log({"outliers_removed": outliers_removed})

    # Step 3: Feature Scaling
    scaler = StandardScaler()
    scaled_data = pd.DataFrame(scaler.fit_transform(imputed_data), columns=imputed_data.columns)

    # Log the feature scaling stats
    wandb.log({"feature_scaling_mean": scaler.mean_.tolist(), "feature_scaling_scale": scaler.scale_.tolist()})

    # Save scaler
    joblib.dump(scaler, 'scaler.pkl')
    artifact = wandb.Artifact(name='scaler.pkl', type='model')
    artifact.add_file('scaler.pkl')
    wandb.log_artifact(artifact)

    # Add trans_date_trans_time column back to the data 
    scaled_data = pd.concat([scaled_data, data['trans_date_trans_time']], axis=1) 
    scaled_data['is_fraud'] = data['is_fraud']

    return scaled_data


def main():
    mlflow.start_run()
    
    # Load data from WandB
    data = load_data()
    
    cleaned_data = clean_data(data)
    
    # Save cleaned data to WandB
    save_data(cleaned_data)
    
    # Log final parameters to MLFlow
    mlflow.log_param("cleaned_data_shape", cleaned_data.shape)

    mlflow.end_run()

if __name__ == "__main__":
    main()