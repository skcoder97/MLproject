import os
import sys
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.exception_handling import CustomException  # Make sure this is appropriately defined in your project.

# Function to save the model or any object (e.g., preprocessor, model)
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        # Create directory if it doesn't exist
        os.makedirs(dir_path, exist_ok=True)

        # Save the object using pickle
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

# Function to evaluate different models and parameters for admission prediction
def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]  # Get the model
            para = param[list(models.keys())[i]]  # Get the corresponding parameters

            # Perform Grid Search to find the best parameters
            gs = GridSearchCV(model, para, cv=3)
            gs.fit(X_train, y_train)

            # Set the model to use the best parameters found by Grid Search
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)  # Train the model with the best parameters

            # Predict on training and test sets
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Evaluate the model
            train_model_score = r2_score(y_train, y_train_pred)  # Calculate R^2 for training
            test_model_score = r2_score(y_test, y_test_pred)  # Calculate R^2 for test

            # Store the test score for each model
            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)

# Function to load a saved object (e.g., model or preprocessor)
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
