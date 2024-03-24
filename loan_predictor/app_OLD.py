from flask import Flask, render_template, request
import pickle
from xgboost import XGBClassifier
import numpy as np

'''inputs = (2500.0, 1, 0.1527, 2, 13, 5, 0, 1.00, 1687.0)
input_np = np.asarray(inputs).reshape(1,-1)

model = pickle.load(open('loan_model.sav', 'rb'))
pred = model.predict(input_np)
print(pred)'''
'''
model = pickle.load(open('base_model.sav', 'rb'))

app = Flask(__name__)

@app.route('/')
def show_home():
    return render_template('home.html')

@app.route('/prediction', methods=['POST'])
def results():
    form = request.form
    if request.method == 'POST':
        married = int(request.form['married'])
        dependents = int(request.form['dependents'])
        education = int(request.form['education'])
        employment = int(request.form['employment'])
        a_income = float(request.form['a_income'])/10
        c_income = float(request.form['c_income'])/10
        amnt = float(request.form['amnt'])/1000
        term = float(request.form['term'])
        c_history = int(request.form['c_history'])
        p_area = int(request.form['p_area'])
        inputs = (married, dependents, education, employment, a_income, c_income, amnt, term, c_history, p_area)
        input_np = np.asarray(inputs).reshape(1,-1)
        pred = 'Yes' if model.predict(input_np)[0] == 1 else 'No'
        if pred == 'Yes':
            return render_template('prediction_y.html', pred = pred)
        else:
            return render_template('prediction_n.html', pred = pred)
'''