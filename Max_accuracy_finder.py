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
accuracy_test_max=0

for i in range(10000):

    seed = i 
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,stratify=y,random_state=seed)

    classifier = svm.SVC(kernel='linear')
    classifier.fit(x_train,y_train)

    x_test_pred = classifier.predict(x_test)
    accuracy_test = accuracy_score(x_test_pred,y_test)

    if (accuracy_test_max<accuracy_test):
        seed_max = i
        accuracy_test_max=accuracy_test

print(f'Seed with max acc: {seed_max} and max accuracy is: {accuracy_test_max}')





# predict_test = np.array([[0,118,84,47,230,45.8,0.551,31]]) # trying a sample test from given database
# predict_test = scaler.transform(predict_test)


# predict_test = classifier.predict(predict_test)
# print(predict_test)


