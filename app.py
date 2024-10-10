from flask import Flask, render_template, request
import joblib
import numpy as np
import logging

app = Flask(__name__)

# Load the trained model
model = joblib.load('electricity_model.pkl')

# Set up logging for error tracking
logging.basicConfig(filename='error.log', level=logging.ERROR)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the form
        day = request.form.get('day')
        month = request.form.get('month')
        forecast_wind = request.form.get('forecast_wind')
        system_load = request.form.get('system_load')
        smpea = request.form.get('smpea')
        ork_temp = request.form.get('ork_temp')
        ork_wind = request.form.get('ork_wind')
        co2_intensity = request.form.get('co2_intensity')
        actual_wind = request.form.get('actual_wind')
        system_load_ep2 = request.form.get('system_load_ep2')

        # Input validation: check for missing/invalid inputs
        if not all([day, month, forecast_wind, system_load, smpea, ork_temp, ork_wind, co2_intensity, actual_wind, system_load_ep2]):
            return render_template('index.html', error="Please fill out all fields.")

        # Convert inputs to float
        features = np.array([[float(day), float(month), float(forecast_wind), float(system_load), 
                              float(smpea), float(ork_temp), float(ork_wind), float(co2_intensity), 
                              float(actual_wind), float(system_load_ep2)]])
        
        # Make prediction
        prediction = model.predict(features)
        return render_template('index.html', prediction=prediction[0])

    except Exception as e:
        # Log the exception and return a user-friendly message
        logging.error(f"Error occurred during prediction: {e}")
        return render_template('index.html', error="An error occurred. Please try again.")

if __name__ == '__main__':
    # Bind to 0.0.0.0 to be accessible externally
    app.run(debug=True, host="0.0.0.0", port=5000)
