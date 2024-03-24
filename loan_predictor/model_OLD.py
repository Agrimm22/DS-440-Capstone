import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import pickle

data = pd.read_csv("data/LoanApprovalPrediction.csv").drop(['Loan_ID', 'Gender'], axis=1)

label_encoder = preprocessing.LabelEncoder()
obj = (data.dtypes == 'object')
for col in list(obj[obj].index):
  data[col] = label_encoder.fit_transform(data[col])

for col in data.columns:
  data[col] = data[col].fillna(data[col].mean())

X = data.drop(['Loan_Status'],axis=1)
Y = data['Loan_Status']
X.shape,Y.shape

X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                    test_size=0.4,
                                                    random_state=1)
X_train.shape, X_test.shape, Y_train.shape, Y_test.shape

lc = LogisticRegression(max_iter = 1500)
lc.fit(X_train, Y_train)
Y_pred = lc.predict(X_test)
print(f"Accuracy: {metrics.accuracy_score(Y_test, Y_pred)}")
filename = 'base_model.sav'
pickle.dump(lc, open(filename, 'wb'))