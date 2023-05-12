import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

diabetes_db=pd.read_csv('diabetes.csv')
x = diabetes_db.drop(columns='Outcome',axis=1)
y = diabetes_db['Outcome']

scaler = StandardScaler()

x = scaler.fit_transform(x)
print(type(x),type(y))

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,stratify=y,random_state=2)

classifier = svm.SVC(kernel='linear')
classifier.fit(x_train,y_train)

x_test_pred = classifier.predict(x_test)
accuracy_test = accuracy_score(x_test_pred,y_test)
print(accuracy_test)
