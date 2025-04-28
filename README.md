# Exp-08: Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Load the Employee dataset and check for missing values and data types.
2. Encode the categorical 'salary' column using Label Encoding.
3. Select relevant feature columns for input `x` and set the target `y` as the 'left' column.
4. Split the dataset into training and testing sets with an 80-20 split.
5. Create a Decision Tree Classifier using "entropy" criterion and train it with the training set.
6. Predict the results for the testing set, calculate the accuracy, display the confusion matrix, and predict for a sample input.


## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: SURIYA M
RegisterNumber:  212223110055
*/
print("Name : SURIYA M")
print("Register Number : 212223110055")
import pandas as pd
df=pd.read_csv("Employee.csv")
df.head()
df.isnull().sum()
df.info()
df["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df["salary"]=le.fit_transform(df["salary"])
df.head()
x=df[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident"]]
x.head()
y=df["left"]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn.metrics import accuracy_score,confusion_matrix
accuracy=accuracy_score(y_test,y_pred)
accuracy
confusion=confusion_matrix(y_test,y_pred)
confusion
print("Name : SURIYA M")
print("Register Number : 212223110055")
dt.predict([[0.5,0.8,9,260,6,0]])
```

## Output:

### Head of Dataset
![image](https://github.com/user-attachments/assets/9c856528-a699-4c80-b480-9475604cfbff)

### Null Count
![image](https://github.com/user-attachments/assets/26ae6f0f-d308-44eb-ae91-59879d0d317f)

### Dataset Info
![image](https://github.com/user-attachments/assets/8f589cce-bf05-4557-8fad-c36d3bbb187d)

### Value Count of Feature
![image](https://github.com/user-attachments/assets/8488f479-e839-4f92-81fa-fd324487e705)

### Encoded Data
![image](https://github.com/user-attachments/assets/b4e33d0f-49b7-4612-bad1-ff8602098a0a)

### Dataset after Feature Selection
![image](https://github.com/user-attachments/assets/78b3a26a-c3bb-4ea8-9409-c16e76c32b37)

### Y Value
![image](https://github.com/user-attachments/assets/a3b2af52-6a79-4c3f-9f1b-a8ce50541e94)

### Accuracy
![image](https://github.com/user-attachments/assets/f1c7837f-2b38-4c46-93f4-43f2e5414d41)

### Confusion Matrix
![image](https://github.com/user-attachments/assets/d6069f3a-f4b7-41bf-b198-586a3905bcf3)

### Predicted Values for new data
![image](https://github.com/user-attachments/assets/bbd64f52-a0a3-436a-b458-d891c42383cb)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
