from flask import Flask, request, jsonify
import pickle
import numpy as np
import xgboost as xgb

# Initialize the Flask app
app = Flask(__name__)

# Load the pre-trained model (assuming it's a Scikit-Learn model saved as a .pkl file)
with open("FinalModel.pkl", "rb") as file:
    model = pickle.load(file)

# Define a route to check the server status
@app.route("/", methods=["GET"])
def home():
    return "Welcome to the Machine Learning Model API!"

# Define the prediction endpoint
@app.route("/predict", methods=["POST"])
def predict():
    # Get the JSON data from the request
    data = request.get_json()

    # Extract feature values from the JSON data
    features = np.array(data["features"]).reshape(1, -1)

    # Perform inference using the pre-trained model
    prediction = model.predict(features)
    print(prediction[0])
    if prediction[0] == 1:
        returnPrediction = 'Loan has been automatically approved'
    else:
        returnPrediction = 'Loan has not been automatically approved, we will revise this manually'
    # Return the prediction result as JSON
    return jsonify({
        "prediction": returnPrediction
    })

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)

