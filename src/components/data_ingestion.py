import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging
import os

class DataIngestion:
    def __init__(self, file_path=None):
        if file_path is None:
            file_path ='C:\\Users\\Shravani\\OneDrive\\Desktop\\Grad Projects\\mlproject\\notebook\\my_dataframe.csv'  # Provide a default path
        self.file_path = file_path
        self.logger = self.get_logger()

    def get_logger(self):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        return logger

    def load_data(self):
        try:
            # Check if the file exists
            if not os.path.exists(self.file_path):
                self.logger.error(f"File not found: {self.file_path}")
                raise FileNotFoundError(f"No such file or directory: '{self.file_path}'")

            self.logger.info("Loading data from CSV file")
            df = pd.read_csv(self.file_path)
            self.logger.info("Dropping unnecessary columns")
            df = df.drop(columns=['Serial No.', 'Unnamed: 0'], errors='ignore')
            self.logger.info(f"Data shape after dropping columns: {df.shape}")

            self.logger.info("Splitting data into features and target")
            X = df.drop(columns=['Chance of Admit '])
            y = df['Chance of Admit ']

            self.logger.info("Splitting data into training and test sets")
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            self.logger.info(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")

            self.logger.info("Scaling the data")
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            self.logger.info("Data ingestion and preprocessing completed successfully")
            return X_train_scaled, X_test_scaled, y_train, y_test
        except Exception as e:
            self.logger.error(f"Error in loading data: {str(e)}")
            raise

# Usage example
if __name__ == "__main__":
    data_ingestion = DataIngestion('C:/Users/Shravani/OneDrive/Desktop/Grad Projects/mlproject/notebook/my_dataframe.csv')
    X_train_scaled, X_test_scaled, y_train, y_test = data_ingestion.load_data()

