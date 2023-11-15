from flask import Flask,render_template,request,redirect
# from flask_cors import CORS,cross_origin
import pickle
import pandas as pd

import numpy as np

app=Flask(__name__)
# cors=CORS(app)
model=pickle.load(open('LinearRegression.pkl','rb'))
car=pd.read_csv('cars.csv')

@app.route('/',methods=['GET','POST'])
def index():
    companies = sorted(car['Car_Name'].unique())
    year = sorted(car['Year'].unique(),reverse=True)
    fuel_type = car['Fuel_Type'].unique()
    owner = car['Owner'].unique()
    return render_template('index.html',companies = companies,years = year, fuel_type = fuel_type,owner = owner)


@app.route('/predict',methods=['POST'])
# @cross_origin()
def predict():

    company = request.form.get('company')
    year = int(request.form.get('year'))
    fuel_type = request.form.get('fuel_type')
    owner = request.form.get('owner')
    kms_driven = int(request.form.get('kilo_driven'))

    prediction=model.predict(pd.DataFrame(columns=['Car_Name','Year','Driven_kms','Fuel_Type','Owner'],
                              data=np.array([company,year,kms_driven,fuel_type,owner]).reshape(1, 5)))
    print(prediction)

    return str(np.round(prediction[0],2))



if __name__=='__main__':
    app.debug = True
    app.run()