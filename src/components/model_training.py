import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import logging
import pickle
import os

class ModelTraining:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.logger = self.get_logger()
        self.model = LinearRegression()

    def get_logger(self):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        return logger

    def train_model(self):
        try:
            self.logger.info("Training the Linear Regression model")
            self.model.fit(self.X_train, self.y_train)
            self.logger.info("Model training completed")
        except Exception as e:
            self.logger.error(f"Error in training model: {str(e)}")
            raise

    def evaluate_model(self):
        try:
            self.logger.info("Evaluating the model")
            y_train_pred = self.model.predict(self.X_train)
            y_test_pred = self.model.predict(self.X_test)
            print(y_train_pred)
            print(y_test_pred)

            train_rmse = mean_squared_error(self.y_train, y_train_pred, squared=False)
            test_rmse = mean_squared_error(self.y_test, y_test_pred, squared=False)
            train_r2 = r2_score(self.y_train, y_train_pred)
            test_r2 = r2_score(self.y_test, y_test_pred)

            self.logger.info(f"Training RMSE: {train_rmse}, Training R^2: {train_r2}")
            self.logger.info(f"Test RMSE: {test_rmse}, Test R^2: {test_r2}")

            return {
                "train_rmse": train_rmse,
                "test_rmse": test_rmse,
                "train_r2": train_r2,
                "test_r2": test_r2
            }
        except Exception as e:
            self.logger.error(f"Error in evaluating model: {str(e)}")
            raise

    def save_model(self, file_path):
        try:
            self.logger.info(f"Saving the model to {file_path}")
            # Ensure the directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'wb') as f:
                pickle.dump(self.model, f)
            self.logger.info("Model saved successfully")
        except Exception as e:
            self.logger.error(f"Error in saving model: {str(e)}")
            raise

# Usage example
if __name__ == "__main__":
    # Load and preprocess data
    df = pd.read_csv(r'C:\Users\Shravani\OneDrive\Desktop\Grad Projects\mlproject\notebook\my_dataframe.csv')
    
    # Assuming data transformation logic here
    from data_transformation import DataTransformation
    data_transformation = DataTransformation(df)
    X_train_scaled, X_test_scaled, y_train, y_test = data_transformation.transform()

    # Train and evaluate the model
    model_training = ModelTraining(X_train_scaled, X_test_scaled, y_train, y_test)
    model_training.train_model()
    evaluation_results = model_training.evaluate_model()
    
    # Print evaluation results
    print(evaluation_results)

    # Save the trained model
    model_training.save_model(r'C:\Users\Shravani\OneDrive\Desktop\Grad Projects\mlproject\models\linear_regression_model.pkl')
 