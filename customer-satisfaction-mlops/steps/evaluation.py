import logging
from typing import Tuple
import mlflow
import pandas as pd
from sklearn.base import RegressorMixin
from typing_extensions import Annotated
from zenml import step
from src.evaluation import MSE,R2,RMSE
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker


@step(experiment_tracker = experiment_tracker.name)
def evaluate_model(
    model: RegressorMixin,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
) -> Tuple[Annotated[float, "r2_score"], Annotated[float, "rmse"]]:
    """
    Evaluate the model on the ingested data.
    Args:
        X_test: The test features
        y_test: The true values for testing data
    """
    try:
        # Make predictions
        prediction = model.predict(X_test)
        mse_class = MSE()
        mse = mse_class.calculate_scores(y_test,prediction)
        mlflow.log_metrics({"mse":mse})

        # Calculate MSE
        #mse = mean_squared_error(y_test, prediction)

        # Calculate R2 score
        r2_class = R2()
        r2 = r2_class.calculate_scores(y_test,prediction)
        #r2 = r2_score(y_test, prediction)
        mlflow.log_metrics({"r2":r2})

        # Calculate RMSE (root of MSE)
        rmse_class = RMSE()
        #rmse = rmse_class.calculate_scores(y_test,prediction)
        rmse = np.sqrt(mse)
        mlflow.log_metrics({"rmse":rmse})

        # Return the computed r2 and rmse values
        return r2, rmse
    except Exception as e:
        logging.error(f"Error in evaluating model: {e}")
        raise e
