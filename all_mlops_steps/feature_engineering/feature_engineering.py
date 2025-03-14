import os
import pandas as pd
import mlflow
import wandb
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Initialize WandB for tracking
wandb.init(project="creadit-card-default-prediction_test-101", name="feature-engineering-1")

def load_data():
    """
    Load the cleaned dataset from WandB Artifacts.
    
    :return: pandas DataFrame
    """
    try:
        artifact = wandb.use_artifact(
            'bhushanmandava16-personal/creadit-card-default-prediction_test-101/processed_data.csv:v0', 
            type='dataset'
        )
        artifact_dir = artifact.download()
        data = pd.read_csv(os.path.join(artifact_dir, 'processed_data.csv'))
        print("Data successfully loaded.")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def save_data(data):
    """
    Save the engineered dataset to WandB as an artifact.
    
    :param data: pandas DataFrame, transformed data
    """
    try:
        transformed_data_path = 'engineered_data.csv'
        data.to_csv(transformed_data_path, index=False)
        
        artifact = wandb.Artifact(name='engineered_data.csv', type='dataset')
        artifact.add_file(transformed_data_path)
        wandb.log_artifact(artifact)
        wandb.log({"engineered_data_shape": data.shape})

        print("Engineered data saved successfully.")
    except Exception as e:
        print(f"Error saving data: {e}")

def feature_engineering(data):
    """
    Perform feature engineering and data transformation.
    
    :param data: pandas DataFrame, cleaned data
    :return: pandas DataFrame, data with new features and transformations applied
    """
    try:
        # Convert transaction time to datetime
        data['trans_date_trans_time'] = pd.to_datetime(data['trans_date_trans_time'], errors='coerce')

        # Creating new features: 'hour' and 'day_of_week'
        data['hour'] = data['trans_date_trans_time'].dt.hour
        data['day_of_week'] = data['trans_date_trans_time'].dt.dayofweek

        wandb.log({"new_features": ["hour", "day_of_week"]})

        # Define feature columns
        categorical_features = ['category', 'state']
        numeric_features = ['amt', 'hour', 'day_of_week']

        # Verify that columns exist
        existing_cat_features = [col for col in categorical_features if col in data.columns]
        existing_num_features = [col for col in numeric_features if col in data.columns]

        # Create a pipeline for transformations
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), existing_num_features),
            ])
        
        pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

        # Apply transformations
        transformed_data = pipeline.fit_transform(data)
        transformed_df = pd.DataFrame(transformed_data, columns=existing_num_features)

        # Adding the target column
        if 'is_fraud' in data.columns:
            transformed_df['is_fraud'] = data['is_fraud']
        else:
            print("Warning: 'is_fraud' column is missing from data.")

        # Logging transformed features
        wandb.log({"transformed_features": existing_num_features})
        wandb.log({"transformed_data": wandb.Table(dataframe=transformed_df.head())})

        # Save pipeline
        joblib.dump(pipeline, 'preprocessor.pkl')
        artifact = wandb.Artifact('preprocessor', type='model')
        artifact.add_file('preprocessor.pkl')
        wandb.log_artifact(artifact)

        print("Feature engineering completed successfully.")
        return transformed_df
    except Exception as e:
        print(f"Error in feature engineering: {e}")
        return None

def main():
    mlflow.start_run()

    # Load data
    data = load_data()
    if data is None:
        print("Failed to load data. Exiting.")
        mlflow.end_run()
        return

    # Perform feature engineering
    engineered_data = feature_engineering(data)
    if engineered_data is None:
        print("Feature engineering failed. Exiting.")
        mlflow.end_run()
        return

    # Save engineered data
    save_data(engineered_data)
    
    # Log final parameters to MLFlow
    mlflow.log_param("engineered_data_shape", engineered_data.shape)

    print("Feature engineering pipeline completed successfully.")
    mlflow.end_run()

if __name__ == "__main__":
    main()
