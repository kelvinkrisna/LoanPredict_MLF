from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model
model = joblib.load("assignment_model.pkl")  # Replace with your model file

# Define API endpoint for loan validation
@app.route('/approve_loan', methods=['POST'])
def validate_loan():
    try:
        # Parse JSON data from the request
        data = request.get_json()

        # Convert JSON data to a DataFrame (assumes data is a dictionary of key-value pairs)
        df = pd.DataFrame([data])

        # Predict using the loaded model
        prediction = model.predict(df)
        probability = model.predict_proba(df)[:, 1]  # Probability of positive class if supported

        # Translate prediction to meaningful response
        response = {
            "loan_status": "Accepted" if prediction[0] == 1 else "Rejected",
            "probability": float(probability[0])
        }

        return jsonify(response), 200

    except Exception as e:
        # Handle exceptions
        return jsonify({"error": str(e)}), 400

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
