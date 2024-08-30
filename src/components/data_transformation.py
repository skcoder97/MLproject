import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import logging

class DataTransformation:
    def __init__(self, dataframe):
        self.dataframe = dataframe
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

    def drop_unnecessary_columns(self):
        self.logger.info("Dropping unnecessary columns")
        self.dataframe = self.dataframe.drop(columns=['Serial No.', 'Unnamed: 0'], errors='ignore')
        self.logger.info(f"Data shape after dropping columns: {self.dataframe.shape}")

    def remove_outliers(self):
        self.logger.info("Removing outliers")
        Q1 = self.dataframe.quantile(0.25)
        Q3 = self.dataframe.quantile(0.75)
        IQR = Q3 - Q1
        condition = ~((self.dataframe < (Q1 - 1.5 * IQR)) | (self.dataframe > (Q3 + 1.5 * IQR))).any(axis=1)
        self.dataframe = self.dataframe[condition]
        self.logger.info(f"Data shape after removing outliers: {self.dataframe.shape}")

    def split_and_scale_data(self):
        self.logger.info("Splitting data into features and target")
        X = self.dataframe.drop(columns=['Chance of Admit '])
        y = self.dataframe['Chance of Admit ']

        self.logger.info("Splitting data into training and test sets")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.logger.info(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")

        self.logger.info("Scaling the data")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        return X_train_scaled, X_test_scaled, y_train, y_test

    def transform(self):
        try:
            self.drop_unnecessary_columns()
            self.remove_outliers()
            X_train_scaled, X_test_scaled, y_train, y_test = self.split_and_scale_data()
            self.logger.info("Data transformation completed successfully")
            return X_train_scaled, X_test_scaled, y_train, y_test
        except Exception as e:
            self.logger.error(f"Error in transforming data: {str(e)}")
            raise

# Usage example
if __name__ == "__main__":
    df = pd.read_csv(r'C:\Users\Shravani\OneDrive\Desktop\Grad Projects\mlproject\notebook\my_dataframe.csv')
    data_transformation = DataTransformation(df)
    X_train_scaled, X_test_scaled, y_train, y_test = data_transformation.transform()
