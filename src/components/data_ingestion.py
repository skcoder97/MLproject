import pandas as pd
from sklearn.model_selection import train_test_split

class DataIngestion:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_data(self):
        """Loads data from CSV and returns a DataFrame."""
        data = pd.read_csv(self.file_path)
        return data

    def split_data(self, data):
        """Splits the data into features (X) and target (y)."""
        # Selecting the relevant 5 features
        X = data[['GRE Score', 'TOEFL Score', 'University Rating', 'CGPA', 'Research']]
        y = data['Chance of Admit ']
        
        # Split the data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # File path to the dataset
    file_path = r'C:\Users\Shravani\OneDrive\Desktop\Grad Projects\mlproject\notebook\dataset\Admission_Predict_Ver1.1.csv'
    
    # Create an instance of DataIngestion and load data
    data_ingestion = DataIngestion(file_path)
    data = data_ingestion.load_data()

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = data_ingestion.split_data(data)
    
    print(f"Training Data Shape: {X_train.shape}")
    print(f"Test Data Shape: {X_test.shape}")
    print("Data Ingestion successfully completed")
