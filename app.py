import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load the trained model and scaler
model_path = r'C:\Users\Shravani\OneDrive\Desktop\Grad Projects\mlproject\models\linear_regression_model.pkl'
scaler_path = r'C:\Users\Shravani\OneDrive\Desktop\Grad Projects\mlproject\models\scaler.pkl'

model = pickle.load(open(model_path, 'rb'))
scaler = pickle.load(open(scaler_path, 'rb'))

@app.route('/')
def home():
    """Render the home page with the form."""
    return render_template('index.html', prediction_text="")

@app.route('/predict', methods=['POST'])
def predict():
    """Handle form submission and generate predictions."""
    try:
        # Extract form inputs and convert them to float
        int_scores = [float(x) for x in request.form.values()]
        input_data = np.array([int_scores])

        # Step 1: Scale the input data using the saved scaler
        input_data_scaled = scaler.transform(input_data)

        # Step 2: Generate prediction using the trained model
        raw_prediction = model.predict(input_data_scaled)[0]

        # Step 3: Clip the prediction between 0 and 1 to represent valid probability
        clipped_prediction = np.clip(raw_prediction, 0, 1)

        # Step 4: Convert to percentage
        prediction_percentage = round(clipped_prediction * 100, 2)

        # Render the result to the HTML template
        return render_template(
            'index.html', 
            prediction_text=f"Predicted Admission Probability: {prediction_percentage}%"
        )
    except ValueError as e:
        # Handle invalid input
        return render_template(
            'index.html', 
            prediction_text=f"Invalid input: {e}"
        )

if __name__ == "__main__":
    # Run the Flask app on localhost
    app.run(host='0.0.0.0', port=5000, debug=True)
