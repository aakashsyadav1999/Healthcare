from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.Healthcare.pipelines.prediction_pipeline import PredictPipeline,CustomData

application=Flask(__name__)

app=application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            Age = request.form.get('Age'),
            Gender = request.form.get('Gender'),
            Blood_Type = request.form.get('Blood_Type'),
            Medical_Condition = request.form.get('Medical_Condition'),
            Doctor = request.form.get('Doctor'),
            Hospital = request.form.get('Hospital'),
            Insurance_Provider = request.form.get('Insurance_Provider'),
            Room_Number = request.form.get('Room_Number'),
            Admission_Type = request.form.get('Admission_Type'),
            Medication = request.form.get('Medication')

        )


        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        #prediction_label = "Inconclusive" if results[0] == 1 else 'Normal'
        prediction_label = "Inconclusive" if results[0] <= 1 else "Normal"
        return render_template('home.html', results=prediction_label)
    

if __name__ == '__main__':
    app.debug = True
    app.run(host="0.0.0.0")
    app.run()        
