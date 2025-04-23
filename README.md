# Exp-08: Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.

2. Upload and read the dataset.

3. Check for any null values using the isnull() function.

4. From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.

5. Find the accuracy of the model and predict the required values by importing the required module from sklearn

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: SURIYA M
RegisterNumber:  212223110055
*/
```
```
import pandas as pd
data = pd.read_csv("/content/Employee.csv")
data.head()
data.info()
```
![image](https://github.com/user-attachments/assets/00f69e2a-0e5a-4ad6-84bd-7dadac1f8a2c)
```
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

data['salary'] = le.fit_transform(data['salary'])
data.head()
```
![image](https://github.com/user-attachments/assets/b5705778-d00d-438f-b615-f4d85ff414fc)
```
x=data[['satisfaction_level','last_evaluation','number_project','average_montly_hours','time_spend_company','Work_accident','promotion_last_5years','salary']]
x.head()
y=data['left']
x.head()
```
![image](https://github.com/user-attachments/assets/86c94c77-1fe3-4a2c-8421-6a04ef1f8f32)
```
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state =100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion='entropy')
dt.fit(x_train,y_train)
y_predict=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_predict)
accuracy
```
![image](https://github.com/user-attachments/assets/f16b67a3-1d79-4888-a0d8-1bbbaa8ce634)
```
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```
![image](https://github.com/user-attachments/assets/b30b2a73-7005-4062-b5fa-f32356abbba5)




## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
