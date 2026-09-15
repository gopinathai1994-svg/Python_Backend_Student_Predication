from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load saved artifacts once at startup
model = pickle.load(open("models/student_model.pkl", "rb"))
scaler = pickle.load(open("models/student_scaler.pkl", "rb"))
model_columns = pickle.load(open("models/student_cols.pkl", "rb"))

@app.route('/predict', methods=['POST'])
def predict():
    print("Received request for prediction")
    try:
        data = request.get_json()
        print(f"Received data: {data}")
        attendance = float(data.get('Attendance', 0))
        monthly_mark = float(data.get('Monthly_Mark', 0))
        section = str(data.get('Section', 'A')).strip().upper()

        # Re-create features based on Section
        student_dict = {
            "Attendance": [attendance],
            "Monthly_Mark": [monthly_mark],
            "Section_B": [1 if section == "B" else 0],
            "Section_C": [1 if section == "C" else 0],
            "Section_D": [1 if section == "D" else 0]
        }

        # Align columns with trained model order
        student_df = pd.DataFrame(student_dict)[model_columns]

        # Scale and Predict
        input_scaled = scaler.transform(student_df)
        prediction = model.predict(input_scaled)[0]

        print(f"Prediction: {prediction}")
        MIN_ATTENDANCE_PASS = 50.0
        MIN_MARK_PASS = 50.0
        if attendance < MIN_ATTENDANCE_PASS or monthly_mark < MIN_MARK_PASS:
            final_prediction = "Risk"
        else:
            final_prediction = prediction

        print(f"Final Prediction: {final_prediction}")

        return jsonify({
            'status': 'success',
            'prediction': str(final_prediction)
        }), 200

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)