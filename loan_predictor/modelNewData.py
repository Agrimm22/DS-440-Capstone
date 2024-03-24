import shap
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import pickle
import os


file_path_part1 = 'loan_amount_dataset_part1.csv'
file_path_part2 = 'loan_amount_dataset_part2.csv'

# Read the datasets back into pandas DataFrames
data_part1 = pd.read_csv(file_path_part1)
data_part2 = pd.read_csv(file_path_part2)

# Concatenate the two DataFrames to form a single dataset
new_data = pd.concat([data_part1, data_part2], ignore_index=True)
loan_data = pd.read_csv('loan_data (1).csv')
int_data = loan_data.copy() #intrest rate dataframe

#Start of loan amoun prediction
#-------------------------------------------------------------------

features_new_dataset = ['loan_status', 'dti', 'revol_util']
X_new = new_data[['loan_status', 'dti', 'revol_util']].copy()
X_new['log.annual.inc'] = np.log(new_data['annual_inc'])
y_new = new_data['loan_amnt']

# Handle missing values
X_new.fillna(X_new.mean(), inplace=True)

# Split and scale the new dataset
X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(X_new, y_new, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_new)
X_test_scaled = scaler.transform(X_test_new)

# Train the RandomForestRegressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train_new)

# Evaluate the model
y_pred = model.predict(X_test_scaled)
#print("Mean Absolute Error on new dataset:", mean_absolute_error(y_test_new, y_pred))

# Prepare the original dataset for prediction, aligning feature names
X_original = loan_data[['credit.policy', 'dti', 'revol.util']].copy()
X_original.rename(columns={'credit.policy': 'loan_status', 'revol.util': 'revol_util'}, inplace=True)
X_original['log.annual.inc'] = loan_data['log.annual.inc']

# Ensure the features in X_original are in the same order as in X_new
X_original = X_original[['loan_status', 'dti', 'revol_util', 'log.annual.inc']]

# Scale the original dataset features
X_original_scaled = scaler.transform(X_original)

# Predict loan_amnt for the original dataset
predicted_loan_amnt_original = model.predict(X_original_scaled)

# Add the predictions to the original dataset
loan_data['predicted_loan_amnt'] = predicted_loan_amnt_original

#-------------------------------------------------------------------------
#End of Loan Amount Prediction


loan_data.drop(['revol.util', 'revol.bal', 'int.rate', "installment", "delinq.2yrs", "not.fully.paid" ], axis=1, inplace=True)
int_data.drop(['credit.policy', 'revol.util', 'revol.bal', "installment", "delinq.2yrs", "not.fully.paid" ], axis=1, inplace=True)

#One hot encoding purpose feature
dummies = pd.get_dummies(loan_data['purpose']).astype(int)

loans = pd.concat([loan_data,dummies],axis=1)
loans = loans.drop(columns = ['purpose'])

X = loans.copy().drop(columns=['credit.policy'])
y = loans['credit.policy']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state = 42)

X_test.to_csv('X_test.csv', index=False)

#balancing dataset
oversampler = RandomOverSampler()
X_resampled, y_resampled = oversampler.fit_resample(X_train, y_train)

X_resampled.to_csv('X_resampled.csv', index=False)

model = XGBClassifier()
model.fit(X_resampled,y_resampled)

y_pred = model.predict(X_test)


filename = 'XGB_model.sav'
pickle.dump(model, open(filename, 'wb'))
#-------------------------------------------------------

#-------------------------------------------------------------------------------------
#Start of Intrest Rate Prediction
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score


#intrest_inputs_df = pd.DataFrame(intrest_inputs, columns=int_data.columns) making array into dataframe

X2 = int_data.drop('int.rate', axis=1)
y2 = int_data['int.rate']

# Identifying numerical and categorical features
numerical_features = X2.select_dtypes(include=['int64', 'float64']).columns.tolist()
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
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.2, random_state=42)

# Training
gb_model.fit(X_train2, y_train2)



filename = 'intrest_model.sav'
pickle.dump(gb_model, open(filename, 'wb'))

#print(X_test.columns)

#print (X_test.iloc[0, 0])