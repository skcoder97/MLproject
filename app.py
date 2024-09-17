from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load the model (replace with your actual model file path)
model = pickle.load(open(r'C:\Users\Shravani\OneDrive\Desktop\Grad Projects\mlproject\models\linear_regression_model.pkl', 'rb'))

# Default route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route for the admission prediction page
@app.route('/predict_admission', methods=['GET', 'POST'])
def predict_admission():
    if request.method == 'POST':
        # Extract form data
        gre_score = float(request.form['gre_score'])
        toefl_score = float(request.form['toefl_score'])
        university_rating = float(request.form['university_rating'])
        sop_strength = float(request.form['sop_strength'])
        lor_strength = float(request.form['lor_strength'])
        cgpa = float(request.form['cgpa'])
        research_experience = float(request.form['research_experience'])
        
        # Input to the model as a numpy array
        input_data = np.array([[gre_score, toefl_score, university_rating, sop_strength, lor_strength, cgpa, research_experience]])
        
        # Prediction from the model
        prediction = model.predict(input_data)[0]
        
        # Render result back to the HTML page
        return render_template('admission_prediction.html', result=f"{prediction:.2f}")
    
    # If GET request, just render the form
    return render_template('admission_prediction.html', result="")

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)
