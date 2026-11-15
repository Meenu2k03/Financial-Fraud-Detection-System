
# Financial Transaction Fraud Detection System

## 1. Project Overview

The Financial Transaction Fraud Detection System is a machine learning-based application designed to identify potentially fraudulent financial transactions.

The system uses multiple machine learning and deep learning algorithms to analyze transaction details and classify transactions as legitimate or potentially fraudulent.

A Flask web application allows users to enter transaction details and receive predictions.

## 2. Objectives

- Detect potentially fraudulent financial transactions.
- Analyze transaction patterns using machine learning.
- Compare different classification algorithms.
- Handle imbalanced transaction data.
- Provide a simple web interface for fraud prediction.

## 3. Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- TensorFlow / Keras
- Flask
- HTML
- CSS

## 4. Machine Learning Algorithms

The following algorithms were implemented:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Artificial Neural Network (ANN)

Additional techniques:
- Feature Engineering
- Class Imbalance Handling
- Stratified Cross-Validation
- GridSearchCV
- Hyperparameter Tuning
- Model Evaluation

## 5. Dataset

Dataset: PaySim Financial Transaction Dataset

Source: https://www.kaggle.com/datasets/ealaxi/paysim1

The project uses the first 100,000 records of the dataset for development and model training.

Target column: `isFraud`

- 0 – Legitimate transaction
- 1 – Fraudulent transaction

## 6. Model Evaluation

Random Forest was selected as the final model based on its performance on the held-out test dataset.

Test results at a decision threshold of 0.4:

- Precision: 0.94
- Recall: 0.65
- F1-Score: 0.77

These results are preliminary because the test set contains only 23 fraudulent transactions.

## 7. Web Application

The Flask application provides:

- Transaction detail input form
- Transaction type selection
- Fraud probability calculation
- Potential Fraud / Legitimate Transaction prediction

## 8. Project Structure

Financial_Fraud_Detection/
- app.py
- fraud_detection.ipynb
- fraud_detection_model.pkl
- model_config.pkl
- requirements.txt
- README.md
- templates/
  - index.html
- static/
  - style.css

## 9. How to Run the Project

1. Clone or download the project.
2. Install the required packages:

   pip install -r requirements.txt

3. Run the Flask application:

   python app.py

4. Open the browser and visit:

   http://127.0.0.1:5000

5. Enter transaction details and click Detect Fraud.

## 10. Limitations

- The model was developed using a subset of 100,000 synthetic transactions.
- The dataset is not real banking transaction data.
- Predictions are estimates and should not be treated as definitive proof of fraud.
- Further testing with larger and more representative datasets is needed.

## 11. Future Enhancements

- Train and evaluate models using a larger dataset.
- Improve fraud detection performance.
- Deploy the application to a production server.
- Integrate real-time transaction monitoring.
