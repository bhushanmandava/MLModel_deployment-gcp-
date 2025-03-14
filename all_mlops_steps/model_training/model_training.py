import os
import pandas as pd
import numpy as np
import mlflow
import wandb
from sklearn.model_selection import train_test_split
from mlflow.models.signature import infer_signature
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb

# Initialize WandB for tracking the model training process
wandb.init(project="creadit-card-default-prediction_test-101", name="model_training-1")

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

def train_model(X_train, y_train, X_test, y_test):
    """
    Train an XGBoost model and evaluate its performance.
    
    :param X_train: Training feature matrix
    :param y_train: Training target vector
    :param X_test: Test feature matrix
    :param y_test: Test target vector
    :return: trained model, evaluation metrics
    """
    # TODO: Initialize and train the XGBoost model with the eval_metric='logloss' parameter
    #       Use use_label_encoder=False to avoid warnings  and train the model
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)

    # TODO: Make predictions on the test set
    y_pred = model.predict(X_test)
    # TODO:  Calculate evaluation metrics (accuracy, precision, recall, f1_score) and store them in the metrics dictionary
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred)
    }

    # Log metrics to WandB
    wandb.log(metrics)

    # TODO: Inference signature for the model deployment
    signature = infer_signature(X_train, model.predict(X_train))

    return model, metrics, signature

def main():
    mlflow.start_run()

    # Load the data from WandB
    X, y = load_data()

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model and evaluate its performance
    model, metrics, signature = train_model(X_train, y_train, X_test, y_test)
    
    # TODO: Log parameters and metrics to MLFlow
    mlflow.log_param("model_type", "XGBoost")
    mlflow.log_params(model.get_params())
    for metric_name, metric_value in metrics.items():
        mlflow.log_metric(metric_name, metric_value)
    
    
    # Save the 
    # TODO: Save the model to a file
    model_path = "xgboost_model.json"
    model.save_model(model_path)
    
    # Log the model to WandB and MLFlow
    artifact = wandb.Artifact('xgboost_model', type='model')
    artifact.add_file(model_path)
    wandb.log_artifact(artifact)
    mlflow.log_artifact(model_path)

    # Log model to MLFlow
    mlflow.xgboost.log_model(
        model,
        "xgboost_model",
        signature=signature,
        input_example=X_train.iloc[:5],
        registered_model_name="XGBoost Baseline",
    )
    
    mlflow.end_run()

if __name__ == "__main__":
    main()


#     (model_train) bhushanchowdary@bhushans-Mac-mini model_training % nohup gunicorn -w 4 mlflow.server:app --bind 0.0.0.0:5007 &

# [10] 10363
# (model_train) bhushanchowdary@bhushans-Mac-mini model_training % appending output to nohup.out

# (model_train) bhushanchowdary@bhushans-Mac-mini model_training % tail -f nohup.out

# [2025-03-13 19:50:32 -0400] [10363] [INFO] Starting gunicorn 23.0.0
# [2025-03-13 19:50:32 -0400] [10363] [INFO] Listening at: http://0.0.0.0:5007 (10363)
# [2025-03-13 19:50:32 -0400] [10363] [INFO] Using worker: sync
# [2025-03-13 19:50:32 -0400] [10369] [INFO] Booting worker with pid: 10369
# [2025-03-13 19:50:32 -0400] [10370] [INFO] Booting worker with pid: 10370
# [2025-03-13 19:50:32 -0400] [10371] [INFO] Booting worker with pid: 10371
# [2025-03-13 19:50:32 -0400] [10372] [INFO] Booting worker with pid: 10372
