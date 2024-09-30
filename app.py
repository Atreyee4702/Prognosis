from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import joblib
import math

# Initialize the Flask application
app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('model_1.pkl')

# Define the home page route
@app.route('/')
def home():
    return render_template('index.html')

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract form data
        data=request.get_json()
        
        gender_enc = int(data['gender'])
        asthma = int(data['asthma'])
        print(data['asthma'])
        irondef = int(data['iron_deficiency'])
        pneum = int(data['pneumonia'])
        substancedependence = int(data['substance_dependence'])
        psychologicaldisorder = int(data['psych_disorder'])
        depress = int(data['depression'])
        
        psychother = int(data['other_psychological'])
        fibrosisandother = int(data['fibrosis'])
        malnutrition = int(data['malnutrition'])
        
        hemo = int(data['hemo'])
        
        hematocrit = float(data['hematocrit'])
        neutrophils = float(data['neutrophils'])
        sodium = float(data['sodium'])
        glucose = float(data['glucose'])
        bloodureanitrogen = float(data['blood_urea_nitrogen'])
        creatinine = float(data['creatinine'])
        bmi = float(data['bmi'])
        pulse = float(data['pulse'])
        respiration = float(data['respiration'])
        secondarydiagnosisnonicd9  = int(data['second_diagnosis'])
        facid_enc = int(data['facid'])
        if int(data['readmissions'])<5:
            rcount = int(data['readmissions'])
        else:
            rcount=5
        dialysisrenalendstage=int(data['dialysisrenalendstage'])
        
        # Create a DataFrame for the model input
        input_data = pd.DataFrame({
            'rcount': [rcount],
            'dialysisrenalendstage':[dialysisrenalendstage],
            
            'asthma': [asthma],
            'irondef': [irondef],
            'pneum': [pneum],
            'substancedependence': [substancedependence],
            'psychologicaldisordermajor': [psychologicaldisorder],
            'depress': [depress],
            'psychother': [psychother],
            'fibrosisandother': [fibrosisandother],
            'malnutrition': [malnutrition],
            'hemo': [hemo],
            'hematocrit': [hematocrit],
            'neutrophils': [neutrophils],
            'sodium': [sodium],
            'glucose': [glucose],
            'bloodureanitro': [bloodureanitrogen],
            'creatinine': [creatinine],
            'bmi': [bmi],
            'pulse': [pulse],
            'respiration': [respiration],
            'secondarydiagnosisnonicd9': [secondarydiagnosisnonicd9],
            'gender_enc': [gender_enc],
            'facid_enc': [facid_enc]
            
           
        })
        
        X=pd.get_dummies(input_data, columns=['facid_enc'], drop_first=False)
        for i in range(5):
            
            if i!=input_data['facid_enc'][0]:
                
                X['facid_enc_'+str(i)]=[False]
        print(X)
        new_order_last_five = ['facid_enc_0', 'facid_enc_1', 'facid_enc_2', 'facid_enc_3', 'facid_enc_4']
        new_order = list(X.columns[:-5]) + new_order_last_five

# Reorder the DataFrame columns
        X = X[new_order]
        print(X)
        # Make a prediction using the model
        prediction = model.predict(X)[0]
        
        print(prediction)
        return jsonify({'prediction': math.ceil(prediction)})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
