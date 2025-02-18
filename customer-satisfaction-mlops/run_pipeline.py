from pipelines.training_pipeline import train_pipeline
from zenml.client import Client

if __name__ == "__main__":
    #Run the pipeline
    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    train_pipeline(data_path="/Users/rithikagurram/Documents/Datascience_Resources/customer-satisfaction-mlops/data/olist_customers_dataset.csv")


#mlflow ui --backend-store-uri "file:/Users/rithikagurram/Library/Application Support/zenml/local_stores/ef86ea33-be3e-40c7-b374-abdf9b678d93/mlruns"