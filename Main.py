import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

diabetes_db=pd.read_csv('diabetes.csv')
x = diabetes_db.drop(columns='Outcome',axis=1)
y = diabetes_db['Outcome']

to_be_predicted=[]
# Inputting the values: 
print("===DIABETES PREDICTOR===","\nWarning: This model has an accuracy of about 80%\n")
for i in x.columns:
    to_be_predicted.append(input(f'{i}: '))


scaler = StandardScaler()

x = scaler.fit_transform(x)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,stratify=y,random_state=475)

classifier = svm.SVC(kernel='sigmoid')
classifier.fit(x_train,y_train)
to_be_predicted = scaler.transform(np.array([to_be_predicted]))
x_test_pred = classifier.predict(to_be_predicted)

if (x_test_pred == 0):
    print('\n\nDiabetes not detected','\nCaution: The program is not liable for any kind of medical advisory, Please consult a physician to ensure proper diagnosis',end='\n')
else:
    print('\n\nDiabetes detected')











# predict_test = np.array([[0,118,84,47,230,45.8,0.551,31]]) # trying a sample test from given database
# predict_test = scaler.transform(predict_test)


# predict_test = classifier.predict(predict_test)
# print(predict_test)


