import sys
import os

# Append the project root directory (mlproject) to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_training import ModelTraining
from src.exception_handling import CustomException
from src.logger import logging

def start_training_pipeline():
    try:
        logging.info("Training pipeline started.")

        # Data Ingestion
        data_ingestion = DataIngestion(file_path=r'C:\Users\Shravani\OneDrive\Desktop\Grad Projects\mlproject\notebook\my_dataframe.csv')

        train_data_X, test_data_X, y_test, y_train = data_ingestion.load_data()

        # Data Transformation
        data_transformation = DataTransformation()
        train_arr, test_arr = data_transformation.initiate_data_transformation(train_data_X, test_data_X)

        # Model Training
        model_training = ModelTraining()
        model_training.initiate_model_training(train_arr, test_arr)

        logging.info("Training pipeline completed successfully.")

    except Exception as e:
        logging.error(f"Error occurred in training pipeline: {e}")
        raise CustomException(e, sys)

if __name__ == "__main__":
    start_training_pipeline()
