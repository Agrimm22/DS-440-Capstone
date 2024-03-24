from flask import Flask, render_template, request
import pickle
from xgboost import XGBClassifier
import numpy as np
import shap


'''
model = pickle.load(open('xgb_model.sav', 'rb'))
explainer = shap.Explainer(model)

app = Flask(__name__)

@app.route('/')
def show_home():
    return render_template('home.html')

@app.route('/prediction', methods=['POST'])
def results():
    if request.method == 'POST':
        # Extracting form data
        married = int(request.form['married'])
        dependents = int(request.form['dependents'])
        education = int(request.form['education'])
        employment = int(request.form['employment'])
        a_income = float(request.form['a_income']) / 10
        c_income = float(request.form['c_income']) / 10
        amnt = float(request.form['amnt']) / 1000
        term = float(request.form['term'])
        c_history = int(request.form['c_history'])
        p_area = int(request.form['p_area'])
        
        # Creating an input array for prediction
        inputs = np.asarray([(married, dependents, education, employment, a_income, c_income, amnt, term, c_history, p_area)])
        
        # Making a prediction
        pred = model.predict(inputs)
        
        # Compute SHAP values for the input
        shap_values = explainer.shap_values(inputs)

        # For binary classification, SHAP returns a list where the first element is the SHAP values for the negative class and the second is for the positive class. We need the second.
        shap_values_for_positive_class = shap_values[1] # Adjust this index based on your model's output

        # Since we're only looking at one prediction, take the SHAP values for the first prediction if multiple are returned
        shap_values_single = shap_values_for_positive_class[0]

        # Convert the prediction to a user-friendly string
        pred_str = 'Yes' if pred[0] == 1 else 'No'
        
        # Render the template with prediction and SHAP values
        return render_template('prediction.html', pred=pred_str, shap_values=shap_values_single)
    



if __name__ == '__main__':
    app.run(debug=True)
'''