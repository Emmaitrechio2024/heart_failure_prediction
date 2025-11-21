import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

data = pd.read_csv('./heart_failure_clinical_records_dataset.csv')

x = data.drop('DEATH_EVENT', axis=1)
y = data['DEATH_EVENT']

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=67)


model = RandomForestClassifier(n_estimators=100, random_state=67)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
y_proba = model.predict_proba(x_test)[:, 1]


print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_proba))

joblib.dump(model, "model.pkl")
