from flask import Flask, render_template, request, url_for
import pickle
from xgboost import XGBClassifier
import numpy as np
import shap
import pandas as pd
from shap_plot import inputs_fetch
#from utils import intrest_prediction



model = pickle.load(open('XGB_model.sav', 'rb'))
intrest_model = pickle.load(open('intrest_model.sav', 'rb'))


app = Flask(__name__)

@app.route('/')
def show_home():
    return render_template('homeNew.html')

#Change features for new modle
@app.route('/prediction', methods=['POST'])
def results():
    if request.method == 'POST':
        # Extracting form data
        purpose = (request.form['purpose'])
        annual_income = float(request.form['annual_income'])
        amnt = float(request.form['amnt'])
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

        purpose_mapping = {
            "Credit Card": "credit_card",
            "Debt Consolidation": "debt_consolidation",
            "Education": "educational",
            "Home Improvement": "home_improvement",
            "Major Purchase": "major_purchase",
            "Small Business": "small_business",
            "Other": "all_other"
        }

        model_purpose = purpose_mapping.get(purpose)

        def set_loan_purpose(input_purpose):
            # Initialize all purposes to 0
            purposes = {
                'all_other': 0,
                'credit_card': 0,
                'debt_consolidation': 0,
                'educational': 0,
                'home_improvement': 0,
                'major_purchase': 0,
                'small_business': 0,
            }
            
            # Check if the input purpose is valid
            if input_purpose in purposes:
                # Set the selected purpose to 1
                purposes[input_purpose] = 1
            
            
            return purposes

        purposes = set_loan_purpose(model_purpose)
        purposes_values = np.array(list(purposes.values()))
        #Creating data frame for intrest input because cant handle array
        #------------------------------------------------------------------------
        data = {
            "purpose": [model_purpose],
            "log.annual.inc": [log_annual_income],
            "dti": [dti],
            "fico": [fico],
            "days.with.cr.line": [days_with_credit],
            "inq.last.6mths": [inq_last_6mths],
            "pub.rec": [public_records]
        }

        intrest_inputs = pd.DataFrame(data)
    
        existing_features = np.array([(log_annual_income, dti, fico, days_with_credit, inq_last_6mths, public_records, amnt)])
        xgb_inputs = np.concatenate((existing_features, purposes_values.reshape(1, -1)), axis=1)
        
        column_names = ['log.annual.inc', 'dti', 'fico', 'days.with.cr.line', 'inq.last.6mths', 'pub.rec', 'predicted_loan_amnt','all_other', 'credit_card','debt_consolidation', 'educational', 'home_improvement', 'major_purchase', 'small_business']
        xgb_inputs_df = pd.DataFrame(xgb_inputs, columns=column_names)
        # Making a prediction
        pred = model.predict(xgb_inputs)

        int_pred = intrest_model.predict(intrest_inputs)

        inputs_fetch(xgb_inputs_df)
        img_url = url_for('static', filename='graph.png')


        # Converting Predictions
        pred_str = 'Yes' if pred[0] == 1 else 'No'
        int_rate = int_pred[0] * 100
        int_rate = "{:.2f}%".format(int_rate)

        if pred_str == 'Yes':
            return render_template('prediction_y_new.html', pred = pred_str, int_rate=int_rate)
        else:
            return render_template('prediction_n_new.html', pred = pred_str, url=img_url)
        
 

if __name__ == '__main__':
    app.run(debug=True)
