import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load the dataset
file_path = r'C:\Users\Shravani\OneDrive\Desktop\Grad Projects\mlproject\notebook\dataset\Admission_Predict_Ver1.1.csv'
data = pd.read_csv(file_path)

# Step 2: Split the data into features (X) and target (y)
X = data[['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR ', 'CGPA', 'Research']]
y = data['Chance of Admit ']

# Step 3: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Scale the feature values using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: Train the Linear Regression model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Step 6: Evaluate the model
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

train_rmse = mean_squared_error(y_train, y_train_pred, squared=False)
test_rmse = mean_squared_error(y_test, y_test_pred, squared=False)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f"Train RMSE: {train_rmse}, Train R^2: {train_r2}")
print(f"Test RMSE: {test_rmse}, Test R^2: {test_r2}")

# Step 7: Save the trained model and scaler
with open(r'C:\Users\Shravani\OneDrive\Desktop\Grad Projects\mlproject\models\linear_regression_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open(r'C:\Users\Shravani\OneDrive\Desktop\Grad Projects\mlproject\models\scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("Model and scaler saved successfully.")
