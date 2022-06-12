# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1. Import the required libraries.
2. Upload and read the dataset.
3. Check for any null values using the isnull() function.
4. From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.
5. Find the accuracy of the model and predict the required values by importing the required module from sklearn.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Senthil Kumar S
RegisterNumber:  212221230091
*/
import pandas as pd
data = pd.read_csv("Employee.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["salary"] = le.fit_transform(data["salary"])
data.head()
x = data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y = data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion = "entropy")
dt.fit(x_train,y_train)
y_pred = dt.predict(x_test)
from sklearn import metrics
accuracy = metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:

## Data Head:
![1](https://user-images.githubusercontent.com/93860256/173226864-b9a1a462-5e09-407d-91e6-1d87032f8d68.PNG)


## Dataset Info:
![2](https://user-images.githubusercontent.com/93860256/173226882-e8f0fcd9-389f-4129-b979-3492d154c42f.PNG)


## Null dataset:
![3](https://user-images.githubusercontent.com/93860256/173226891-54077975-7827-4fec-9d7a-d0d70e6326bf.PNG)


## Values Count in Left Column:
![4](https://user-images.githubusercontent.com/93860256/173226898-9db618b2-1530-49ff-89af-b8d19009820b.PNG)


## Dataset transformed head:
![5](https://user-images.githubusercontent.com/93860256/173226906-767422f1-8548-453a-9143-280485232ade.PNG)


## x.head():
![6](https://user-images.githubusercontent.com/93860256/173226914-6558ed08-ee70-45c6-898e-ab10bac640fc.PNG)


## Accuracy:
![7](https://user-images.githubusercontent.com/93860256/173226923-d95e5054-ccc9-49a0-8481-11e9839ef4da.PNG)


## Data Prediction:
![8](https://user-images.githubusercontent.com/93860256/173226932-18c9749a-78f6-4a8b-941c-ed78b31f77b0.PNG)



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
