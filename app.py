
from flask import Flask, render_template, request
import pandas as pd
import joblib

# Initialize Flask application
app = Flask(__name__)

# Load the trained Random Forest model
model = joblib.load('fraud_detection_model.pkl')

# Load model configuration
config = joblib.load('model_config.pkl')

# Get threshold and feature names
threshold = config['threshold']
feature_names = config['features']

print("Fraud Detection Model Loaded Successfully!")



@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get transaction details from the web form
        step = float(request.form['step'])
        transaction_type = request.form['type']
        amount = float(request.form['amount'])
        oldbalanceOrg = float(request.form['oldbalanceOrg'])
        newbalanceOrig = float(request.form['newbalanceOrig'])
        oldbalanceDest = float(request.form['oldbalanceDest'])
        newbalanceDest = float(request.form['newbalanceDest'])

        # Prepare input features
        input_data = {
            'step': step,
            'amount': amount,
            'oldbalanceOrg': oldbalanceOrg,
            'newbalanceOrig': newbalanceOrig,
            'oldbalanceDest': oldbalanceDest,
            'newbalanceDest': newbalanceDest,
            'type_CASH_IN': 0,
            'type_CASH_OUT': 0,
            'type_DEBIT': 0,
            'type_PAYMENT': 0,
            'type_TRANSFER': 0
        }

        # Encode transaction type
        type_column = 'type_' + transaction_type
        if type_column not in input_data:
            return "Invalid transaction type", 400

        input_data[type_column] = 1

        # Arrange features in the model's expected order
        input_df = pd.DataFrame([input_data])[feature_names]

        # Predict fraud probability
        fraud_probability = model.predict_proba(input_df)[0][1]

        if fraud_probability >= threshold:
            result = "Potential Fraud"
        else:
            result = "Legitimate Transaction"

        return render_template(
            'index.html',
            prediction=result,
            probability=round(fraud_probability * 100, 2)
        )

    except (ValueError, KeyError):
        return "Please enter valid transaction details", 400

if __name__ == '__main__':
    app.run(debug=True)
