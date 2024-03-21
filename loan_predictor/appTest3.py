from flask import Flask, render_template, request
import pickle
from xgboost import XGBClassifier
import numpy as np
import shap
#from utils import inputs_fetch
#from utils import intrest_prediction



model = pickle.load(open('XGB_model.sav', 'rb'))
intrest_model = pickle.load(open('intrest_model.sav', 'rb'))


app = Flask(__name__)

@app.route('/')
def show_home():
    return render_template('home.html')

#Change features for new modle
@app.route('/prediction', methods=['POST'])
def results():
    if request.method == 'POST':
        # Extracting form data
        purpose = int(request.form['purpose'])
        annual_income = float(request.form['annual_income'])
        amnt = float(request.form['amnt'])
        term = int(request.form['term'])
        monthly_debt_payments = float(request.form['monthly_debt_payments'])
        fico = int(request.form['fico'])
        days_with_credit = int(request.form['days_with_credit'])
        inq_last_6mths = int(request.form['inq_last_6mths'])
        public_records = int(request.form['public_records'])

        '''
        These values below are the updated values, change them in the code above and in the html
        Fields:
        - Purpose: purpose of loan: Categorical field with values: Credit Card, Debt Consolidation, Educational, Home Improvement, Major Purchase, Small Business, All Other
        - Annual Income
        - Monthly Debt Payments: int field (this includes Mortgage or rent payments, Car loan payments, Minimum credit card payments, Student loan payments, Any other obligations that are debts)
        - Fico Score
        - Number of Days With Credit
        - Number of Credit Inquries in the Past 6 Months
        - Number of Derogatory Public Records (Bankruptcy Filings, Tax Liens, and Judgements)
        '''
        log_annual_income = np.log(annual_income)
        monthly_income = annual_income / 12
        dti = (monthly_debt_payments / monthly_income) * 100
        inputs = np.asarray([(purpose, log_annual_income, dti, fico, days_with_credit, inq_last_6mths, public_records)])
        
        
        # Making a prediction
        pred = model.predict(inputs)
        int_pred = intrest_model(inputs)
        #inputs_fetch(inputs)
        



@app.route('/')
def show_graph():
    return render_template('index.html', url='/static/graph.png')

if __name__ == '__main__':
    app.run(debug=True)
