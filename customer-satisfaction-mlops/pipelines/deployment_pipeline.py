
from zenml import pipeline, step
from zenml.config import DockerSettings
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.constants import MLFLOW
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
    MLFlowModelDeployer,
)
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step

from steps.clean_data import clean_df
from steps.evaluation import evaluate_model
from steps.ingest_data import ingest_df
from steps.model_train import train_model

# Define the pipeline parameters directly
class DeploymentTriggerConfig:
    min_accuracy: float = 0.92  # Default value for min_accuracy

# Define docker settings and deployment trigger step
docker_settings = DockerSettings(required_integrations=[MLFLOW])

# Define the step for deployment trigger
@step
def deployment_trigger(
    accuracy: float,
    min_accuracy: float = 0.92,  # Default value for the parameter
) -> bool:  # Directly return the result type
    """Implemented a simple model deployment trigger that looks at the input model accuracy"""
    return accuracy >= min_accuracy

# Define the pipeline
@pipeline(enable_cache=False, settings={"docker": docker_settings})
def continuous_deployment_pipeline(
    data_path: str,
    min_accuracy: float = 0.92,
    workers: int = 1,
    timeout: int = DEFAULT_SERVICE_START_STOP_TIMEOUT,
):
    df = ingest_df(data_path=data_path)
    X_train, X_test, y_train, y_test = clean_df(df)
    model = train_model(X_train, X_test, y_train, y_test)
    r2_score, rmse = evaluate_model(model, X_test, y_test)
    deployment_decision = deployment_trigger(r2_score, min_accuracy=min_accuracy)
    mlflow_model_deployer_step(
        model=model,
        deploy_decision=deployment_decision,
        workers=workers,
        timeout=timeout,
    )



