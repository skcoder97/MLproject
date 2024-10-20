import numpy as np
import pickle

# Load the trained model and scaler
model_path = r'C:\Users\Shravani\OneDrive\Desktop\Grad Projects\mlproject\models\linear_regression_model.pkl'
scaler_path = r'C:\Users\Shravani\OneDrive\Desktop\Grad Projects\mlproject\models\scaler.pkl'

model = pickle.load(open(model_path, 'rb'))
scaler = pickle.load(open(scaler_path, 'rb'))

# Test input: GRE, TOEFL, University Rating, SOP, LOR, CGPA, Research
test_input = np.array([[300, 80, 4, 4.5, 4, 8.2, 1]])
print(test_input)

# Step 1: Scale the input features using the same scaler as during training
test_input_scaled = scaler.transform(test_input)

# Step 2: Make prediction using the scaled input
raw_prediction = model.predict(test_input_scaled)[0]

# Step 3: Clip the prediction to ensure it's between 0 and 1, and convert to percentage
prediction_percentage = round(np.clip(raw_prediction, 0, 1) * 100, 2)

print(f"Predicted Admission Percentage: {prediction_percentage}%")
