import shap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use the Anti-Grain Geometry (Agg) backend that does not require a GUI
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
import os
import pickle

filename = 'XGB_model.sav'
model = pickle.load(open(filename, 'rb'))
X_resampled = pd.read_csv('X_resampled.csv')
X_test = pd.read_csv('X_test.csv')

def inputs_fetch(inputs):
    explainer = shap.Explainer(model, X_resampled)
    shap_values = explainer.shap_values(inputs)

    def create_shap_bar_plot(shap_values, features, top_n=5):
        # Assuming 'features' is a DataFrame. Adjust accordingly if it's a Numpy array.
        shap_series = pd.Series(shap_values[0], index=features.columns)

        # Sort the SHAP values by their magnitude and take the top 'top_n'
        shap_series = shap_series.abs().sort_values(ascending=True).tail(top_n)

        # Create a mapping of old feature names to new feature names
        feature_names_mapping = {
            'fico': 'Fico Score',
            'inq.last.6mths': 'Credit Inquiries',
            'days.with.cr.line': 'Days With Credit',
            'log.annual.inc': 'Annual Income',
            'dti': 'Debt to Income Ratio',
            'predicted_loan_amnt' : 'Loan Amount',
            'all_other': 'Purpose',
            'credit_card': 'Purpose',
            'debt_consolidation': 'Purpose',
            'educational': 'Purpose',
            'home_improvement': 'Purpose',
            'major_purchase': 'Purpose',
            'small_business': 'Purpose'
            # Add other feature mappings if necessary
        }

        # Rename the features
        shap_series.index = shap_series.index.map(lambda x: feature_names_mapping.get(x, x))
        static_dir = 'static'

        # Ensure the static directory exists
        if not os.path.exists(static_dir):
            os.makedirs(static_dir)

        # Define the path for the graph
        graph_path = os.path.join(static_dir, 'graph.png')


        # Create the bar plot
        plt.figure(figsize=(13, 8))
        shap_series.plot(kind='barh', color='skyblue')
        plt.xlabel('Importance')
        plt.title("Most Important Factors In Your Result") 
        plt.savefig(graph_path)


        # Make sure to close the plot to free up memory
        plt.close()

    
    create_shap_bar_plot(shap_values, X_test)