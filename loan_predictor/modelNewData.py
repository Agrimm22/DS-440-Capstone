import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from xgboost import XGBClassifier
import pickle



loan_data = pd.read_csv('loan_data (1).csv')
int_data = loan_data.copy() #intrest rate dataframe
loan_data.drop(['revol.util', 'revol.bal', 'int.rate', "installment", "delinq.2yrs", "not.fully.paid" ], axis=1, inplace=True)
int_data.drop(['credit.policy', 'revol.util', 'revol.bal', "installment", "delinq.2yrs", "not.fully.paid" ], axis=1, inplace=True)

#One hot encoding purpose feature
dummies = pd.get_dummies(loan_data['purpose']).astype(int)

loans = pd.concat([loan_data,dummies],axis=1)
loans = loans.drop(columns = ['purpose'])

X = loans.copy().drop(columns=['credit.policy'])
y = loans['credit.policy']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state = 42)

#balancing dataset
oversampler = RandomOverSampler()
X_resampled, y_resampled = oversampler.fit_resample(X_train, y_train)



model = XGBClassifier()
model.fit(X_resampled,y_resampled)

y_pred = model.predict(X_test)

filename = 'XGB_model.sav'
pickle.dump(model, open(filename, 'wb'))
#-------------------------------------------------------
def inputs_fetch(inputs):
    inputs_df = pd.DataFrame(inputs, columns=X_test.columns)
    explainer = shap.Explainer(model, X_resampled)
    shap_values = explainer.shap_values(inputs_df)

    def create_shap_bar_plot(shap_values, features, top_n=5):
        # Assuming 'features' is a DataFrame. Adjust accordingly if it's a Numpy array.
        shap_series = pd.Series(shap_values, index=features.columns)

        # Sort the SHAP values by their magnitude and take the top 'top_n'
        shap_series = shap_series.abs().sort_values(ascending=True).tail(top_n)

        # Create a mapping of old feature names to new feature names
        feature_names_mapping = {
            'fico': 'Fico Score',
            'inq.last.6mths': 'Credit Inquiries In Past 6mths',
            'days.with.cr.line': 'Days With Credit',
            'log.annual.inc': 'Annual Income',
            'dti': 'Debt to Income Ratio'
            # Add other feature mappings if necessary
        }

        # Rename the features
        shap_series.index = shap_series.index.map(lambda x: feature_names_mapping.get(x, x))

        # Create the bar plot
        plt.figure(figsize=(10, 8))
        shap_series.plot(kind='barh', color='skyblue')
        plt.xlabel('Importance')
        plt.title("Most Important Factors In Your Result")
        plt.savefig('static/graph.png')  # Save the graph to the 'static' directory
        plt.close()

    
    create_shap_bar_plot(shap_values, X_test)
#-------------------------------------------------------------------------------------
#Start of Intrest Rate Prediction
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score


#intrest_inputs_df = pd.DataFrame(intrest_inputs, columns=int_data.columns) making array into dataframe

X = int_data.drop('int.rate', axis=1)
y = int_data['int.rate']

# Identifying numerical and categorical features
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = ['purpose']

# Creating a preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# Creating the regression model pipeline with Gradient Boosting Regressor
gb_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', GradientBoostingRegressor(random_state=42))
])

# Splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training
gb_model.fit(X_train, y_train)


filename = 'intrest_model.sav'
pickle.dump(gb_model, open(filename, 'wb'))