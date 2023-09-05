# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
### Step1
Import the standard Libraries.

### Step2
Set variables for assigning dataset values.

### Step3
Import linear regression from sklearn.

### Step4
Assign the points for representing in the graph.

### Step5
Predict the regression for marks by using the representation of the graph.

### Step6
Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
Program to implement the simple linear regression model for predicting the marks scored.
Developed by:Lakshman 
RegisterNumber: 212222240001 
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()
df.tail()
x = df.iloc[:,:-1].values
x
y = df.iloc[:,1].values
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
y_pred
y_test
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='purple')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
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


![ML_ex-2 1](https://github.com/LakshmanAdhireddy/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118707265/f6b5dc6e-99da-436b-bde8-283515dc1f90)



![ML_exp-2 2](https://github.com/LakshmanAdhireddy/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118707265/6934fbd6-10c4-41af-9003-5decec59a225)



![ML_exp-2 3](https://github.com/LakshmanAdhireddy/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118707265/ad555248-1394-4755-801a-04bd6fe084ac)



![ML_exp-2 4](https://github.com/LakshmanAdhireddy/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118707265/450afe63-56ab-441a-8500-953a5c28b788)



![ML_exp-2 5](https://github.com/LakshmanAdhireddy/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118707265/67ecafd2-0a3e-4f8b-8c94-c598e1ce8e8e)



![ML_exp-2 6](https://github.com/LakshmanAdhireddy/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118707265/e7b0efd4-20f4-4c81-8af0-417eed560dda)



![ML_exp-2 7](https://github.com/LakshmanAdhireddy/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118707265/0532000d-174b-4078-a8cd-031f7d33ea3f)




## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
