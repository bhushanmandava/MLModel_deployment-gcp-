import os
import mlflow
import wandb
import pandas as pd
import pandera as pa
from pandera import Column, DataFrameSchema
from pandera.errors import SchemaErrors

# Initialize WandB for tracking
wandb.init(project="creadit-card-default-prediction_test-101", name="data-validation-1")

def load_data():
    """
    Load the cleaned dataset from WandB Artifacts.

    :return: pandas DataFrame
    """
    artifact = wandb.use_artifact(
        'bhushanmandava16-personal/creadit-card-default-prediction_test-101/processed_data.csv:v0',
        type='dataset'
    )
    artifact_dir = artifact.download()

    # Load the dataset
    data = pd.read_csv(os.path.join(artifact_dir, 'processed_data.csv'))
    data['trans_date_trans_time'] = pd.to_datetime(data['trans_date_trans_time'])

    return data

def validate_data(data):
    """
    Validate the dataset using Pandera and log results to wandb.

    :param data: pandas DataFrame, cleaned data
    :return: bool, True if all checks pass, False otherwise
    """

    schema = DataFrameSchema({
        "trans_date_trans_time": Column(pa.DateTime, nullable=False),
        "amt": Column(float, nullable=True),
        "lat": Column(float, nullable=True),  # ✅ Fixed type
        "long": Column(float, nullable=True),  # ✅ Fixed type
    })

    try:
        schema.validate(data)
        all_checks_passed = True
    except SchemaErrors as err:
        all_checks_passed = False
        for _, row in err.failure_cases.iterrows():
            wandb.log({f"{row['column']}_{row['check']}_failed": True})
            print(f"Validation failed for {row['column']}: {row['check']}")

    # Log overall validation result
    wandb.log({"all_checks_passed": all_checks_passed})

    return all_checks_passed

def main():
    mlflow.start_run()

    # Load data from WandB
    data = load_data()

    # Validate data
    validation_passed = validate_data(data)

    if validation_passed:
        print("✅ All data validation checks passed. Proceeding to feature engineering.")
        mlflow.log_param("validation_passed", True)
    else:
        print("❌ Data validation failed. Please review the logs and fix the issues before proceeding.")
        mlflow.log_param("validation_passed", False)

    mlflow.end_run()

if __name__ == "__main__":
    main()
