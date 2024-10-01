import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
d_train=pd.read_csv("C:\\Users\\praga\\Downloads\\train_features.csv")
d_test=pd.read_csv("C:\\Users\\praga\\Downloads\\test_features (1).csv")
Xtrain=d_train.drop(columns=['sig_id','cp_type','cp_time','cp_dose'],axis='columns')
ytrain=d_train['sig_id']
Xtest=d_test.drop(columns=['sig_id','cp_type','cp_time','cp_dose'],axis='columns')
ytest=d_test['sig_id']
rf=RandomForestClassifier(n_estimators=20,random_state=42)
rf.fit(Xtrain,ytrain)
ypred=rf_classifier.predict(Xtest)
print(f'Accuracy: {accuracy_score(ytest, ypred) * 100:.2f}%')
print('Classification Report of the model:')
print(classification_report(ytest,ypred))
joblib.dump(rf_classifier,'rf_classifier_model.pkl')