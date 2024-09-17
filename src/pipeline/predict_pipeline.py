import sys
import os
import pickle
import pandas as pd
from src.exception_handling import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            # Paths to the model and preprocessor artifacts
            model_path = os.path.join("artifacts", "admission_model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

            print("Before Loading Model and Preprocessor")

            # Load the model and preprocessor
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            print("After Loading Model and Preprocessor")

            # Apply preprocessing on the input features
            data_scaled = preprocessor.transform(features)

            # Predict using the model
            preds = model.predict(data_scaled)

            return preds
        
        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self, 
                 gre_score: int, 
                 toefl_score: int, 
                 university_rating: int, 
                 sop_strength: float, 
                 lor_strength: float, 
                 cgpa: float, 
                 research_experience: int):
        
        # Initialize all fields
        self.gre_score = gre_score
        self.toefl_score = toefl_score
        self.university_rating = university_rating
        self.sop_strength = sop_strength
        self.lor_strength = lor_strength
        self.cgpa = cgpa
        self.research_experience = research_experience

    def get_data_as_data_frame(self):
        try:
            # Dictionary of input data
            custom_data_input_dict = {
                "GRE Score": [self.gre_score],
                "TOEFL Score": [self.toefl_score],
                "University Rating": [self.university_rating],
                "SOP Strength": [self.sop_strength],
                "LOR Strength": [self.lor_strength],
                "CGPA": [self.cgpa],
                "Research Experience": [self.research_experience]
            }

            # Convert dictionary to DataFrame
            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)

