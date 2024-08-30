# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1..Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph.
5.Predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given datas. 


## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: DEVA DHARSHINI.I
RegisterNumber:  212223240026
*/
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```

## Output:
DATA SET
![Screenshot 2024-08-30 172220](https://github.com/user-attachments/assets/d2bd9485-b6b9-4576-b6af-d1942792a5e3)

TAIL VALUES
![Screenshot 2024-08-30 172332](https://github.com/user-attachments/assets/c2696b83-6c33-41cc-b4eb-b92f77b5a69d)

X AND Y VALUES
![Screenshot 2024-08-30 172456](https://github.com/user-attachments/assets/030053f1-3b11-4729-b83d-388311e891e1)

PREDICTION VALUE OF X AND Y
![image](https://github.com/user-attachments/assets/04c96b8a-86fc-4e62-903e-ec5b4a30751d)

MSE, MAE, RMSE
![Screenshot 2024-08-30 172538](https://github.com/user-attachments/assets/3845ab57-7aa2-4f07-bbb6-a7396c070e69)

TRAINING SET
![Screenshot 2024-08-30 172719](https://github.com/user-attachments/assets/9d19f9d6-6bb9-4f4a-98b1-0bc4ee25cb5f)

TESTING SET
![Screenshot 2024-08-30 172732](https://github.com/user-attachments/assets/c40610a5-c05d-4ccb-b047-f9f7ca121fe6)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
