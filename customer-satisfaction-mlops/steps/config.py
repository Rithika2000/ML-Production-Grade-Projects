#from zenml.config import BaseStepConfig
from zenml.steps import BaseStep


class ModelNameConfig(BaseStep):
    """Model Configs"""
    model_name: str = "LinearRegression"
    