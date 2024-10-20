from sklearn.preprocessing import StandardScaler

class DataTransformation:
    def __init__(self):
        self.scaler = StandardScaler()

    def fit_transform(self, X_train):
        """Fits the scaler on training data and transforms it."""
        return self.scaler.fit_transform(X_train)

    def transform(self, X_test):
        """Transforms the test data using the already-fitted scaler."""
        return self.scaler.transform(X_test)

if __name__ == "__main__":
    from data_ingestion import DataIngestion

    # Load and split data
    file_path = r'C:\Users\Shravani\OneDrive\Desktop\Grad Projects\mlproject\notebook\dataset\Admission_Predict_Ver1.1.csv'
    data_ingestion = DataIngestion(file_path)
    data = data_ingestion.load_data()
    X_train, X_test, y_train, y_test = data_ingestion.split_data(data)

    # Transform data
    data_transformation = DataTransformation()
    X_train_scaled = data_transformation.fit_transform(X_train)
    X_test_scaled = data_transformation.transform(X_test)

    print("Data Transformation Completed.")
