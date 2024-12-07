from flask import Flask, request, render_template
import pickle
import numpy as np
from majority_voting_ensemble import MajorityVotingEnsemble
import pandas as pd
from sklearn.metrics import f1_score

# Initialize Flask app
app = Flask(__name__)

# Load the saved model and scaler
with open('ensemble_model.pkl', 'rb') as f:
    ensemble_model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('expected_columns.pkl', 'rb') as f:
    expected_columns = pickle.load(f)

with open("test_data.pkl", "rb") as f:
    test_data = pickle.load(f)

def mode(output_dict):
    count0 = 0
    count1 = 0
    for i in output_dict:
        if output_dict[i] == 0:
            count0 += 1
        else:
            count1 += 1
    if count0 > count1:
        return 0
    else:
        return 1

# Define routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        model_outputs = {}
        confidence_scores = {}
        y_test = test_data["y_test"]
        model_predictions = test_data["model_predictions"]

        form_data = request.form
        input_data = [int(form_data[key]) for key in form_data.keys()]
        print("1d array ",input_data)

        input_data = np.array(input_data).reshape(1, -1)  # Make 2D
        print("2d array ",input_data)
        
        input_df = pd.DataFrame(input_data, columns=expected_columns)
        input_df = input_df.astype(np.float64) 
        print("input DF ",input_df)
        
    
        input_df = scaler.transform(input_df)  
        print("input DF Scaler",input_df)

        print(ensemble_model)
        for model in ensemble_model.models:
            model_outputs[model] = ensemble_model.models[model].predict(input_df)[0]

        print(model_outputs)
        print("Mode :",mode(model_outputs))
        prediction = "Approved" if mode(model_outputs) == 1 else "Rejected"
        # confidence_scores = { "model1": 1, "model2": 2, "model3": 3, "model4": 4, "model5": 5, "model6": 6, "model7": 7, "model8": 8, "model9": 9 }
        confidence_scores = { model_name: f1_score(y_test, y_pred, pos_label=1) for model_name, y_pred in model_predictions.items()}

        print("Printed successfully")
        # prediction, confidence_scores = ensemble_model.predict(input_df)

        return render_template('result.html', prediction=prediction, scores=confidence_scores)

    except Exception as e:
        return render_template('error.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
