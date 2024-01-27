from flask import Flask, request, app, render_template
from flask_cors import CORS, cross_origin
from flask import Response
import pickle
#import numpy as np
#import pandas as pd


app =Flask(__name__)

scaler=pickle.load(open("Model/standardScaler.pkl", "rb"))
model = pickle.load(open("Model/modelforprediction.pkl", "rb"))

@app.route('/')
@cross_origin()
def index():
    return render_template('index.html')

@app.route("/predictdata",methods=['GET','POST'])
def predict_datapoint():
    result=""

    if request.method=='POST':

        Pregnancies=int(request.form.get("Pregnancies"))
        Glucose = float(request.form.get('Glucose'))
        BloodPressure = float(request.form.get('BloodPressure'))
        SkinThickness = float(request.form.get('SkinThickness'))
        Insulin = float(request.form.get('Insulin'))
        BMI = float(request.form.get('BMI'))
        DiabetesPedigreeFunction = float(request.form.get('DiabetesPedigreeFunction'))
        Age = float(request.form.get('Age'))

        new_data=scaler.transform([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
        predict=model.predict(new_data)
        
        
        if predict[0] ==1 :
            result = 'Diabetic'
            return render_template('single1.html',result=result)
        else:
            result ='Non-Diabetic'
            return render_template('single2.html',result=result)
            
        #return render_template('single_prediction.html',result=result)

    else:
        return render_template('home.html')


if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)