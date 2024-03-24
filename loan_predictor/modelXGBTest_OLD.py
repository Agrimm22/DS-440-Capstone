import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn import metrics
import pickle


data = pd.read_csv("data/LoanApprovalPrediction.csv").drop(['Loan_ID', 'Gender'], axis=1)

label_encoder = preprocessing.LabelEncoder()
obj = (data.dtypes == 'object')
for col in list(obj[obj].index):
    data[col] = label_encoder.fit_transform(data[col])

for col in data.columns:
    data[col] = data[col].fillna(data[col].mean())

X = data.drop(['Loan_Status'], axis=1)
Y = data['Loan_Status']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=1)


xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', max_iter=1500)
xgb_model.fit(X_train, Y_train)

Y_pred = xgb_model.predict(X_test)
print(f"Accuracy: {metrics.accuracy_score(Y_test, Y_pred)}")


filename = 'xgb_model.sav'
pickle.dump(xgb_model, open(filename, 'wb'))
