import os
import pandas as pd
import numpy as np
import mlflow
import wandb
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import joblib

# Initialize WandB for tracking the model evaluation process
wandb.init(project="creadit-card-default-prediction_test-101", name="model-evaluation-1")

def load_data():
    """
    Load the engineered dataset from WandB Artifacts.
    
    :return: pandas DataFrame, feature matrix and target vector
    """
    artifact = wandb.use_artifact('bhushanmandava16-personal/creadit-card-default-prediction_test-101/engineered_data.csv:v0', type='dataset')
    artifact_dir = artifact.download()

    # Load the dataset
    data = pd.read_csv(os.path.join(artifact_dir, 'engineered_data.csv'))
    
    # Assume the target variable is 'is_fraud' and features are the rest
    X = data.drop(columns=['is_fraud'])
    y = data['is_fraud']
    
    return X, y

def evaluate_model(X, y):
    """
    Evaluate the model using cross-validation and multiple metrics.
    
    :param X: Feature matrix
    :param y: Target vector
    :return: dict, evaluation metrics
    """

    # Loading model from WandB
    artifact = wandb.use_artifact('bhushanmandava16-personal/creadit-card-default-prediction_test-101/production_model:v0', type='model')
    model_dir = artifact.download()
    model_path = f"{model_dir}/production_model.pkl"

    try:
        # Load model using joblib
        model = joblib.load(model_path)
        print("Model loaded successfully")
        
    except Exception as e:
        print(f"Error loading the model: {e}")
        if os.path.exists(model_path):
            print("Model file size:", os.path.getsize(model_path))
        else:
            print(f"Model file not found at {model_path}")
        raise

    # Implement the evaluation of the model using cross-validation and multiple metrics
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Calculate the metrics using cross_val_score
    metrics = {}
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
        try:
            score = cross_val_score(model, X, y, cv=cv, scoring=metric).mean()
            metrics[f"cv_{metric}"] = score
        except Exception as e:
            print(f"Error calculating {metric}: {e}")
            metrics[f"cv_{metric}"] = None

    # Log metrics to WandB
    wandb.log(metrics)

    return metrics

def main():
    mlflow.start_run()

    # Load the data from WandB
    X, y = load_data()

    # Evaluate the model
    metrics = evaluate_model(X, y)

    # Log parameters and metrics to MLFlow
    mlflow.log_param("model_type", "XGBoost")
    for metric_name, metric_value in metrics.items():
        mlflow.log_metric(metric_name, metric_value)
    
    mlflow.end_run()

if __name__ == "__main__":
    main()