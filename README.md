# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard libraries.
2. Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.
3. LabelEncoder and encode the dataset.
4. Import LogisticRegression from sklearn and apply the model on the dataset.
5. Predict the values of array.
6. Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
7. Apply new unknown values

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: J.NETHRAA
RegisterNumber:  212222100031
*/
import pandas as pd
data=pd.read_csv('/Placement_Data(1).csv')
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver = "liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy= accuracy_score(y_test,y_pred)#Accuracy Score = (TP+TN)/(TP+FN+TN+FP)
#accuracy_score(y_true,y_pred,normalize=False)
accuracy

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion #11+24=35 -correct predictions,5+3=8 incorrect predictions

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])



```

## Output:
### Placement data 

![image](https://github.com/Nethraa24/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121215786/946a90a8-2cc6-4b87-8f5e-25609edb15f5)

### Salary data 

![image](https://github.com/Nethraa24/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121215786/37c884ef-a09c-4f70-875b-096e3b3bf768)

### Checking the null() function 

![image](https://github.com/Nethraa24/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121215786/52bf37c3-6446-4b38-95f7-14b24ac23b27)

### Data Duplicate

![image](https://github.com/Nethraa24/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121215786/65958a22-3c82-4961-8329-49a08113528f)

### Print data

![image](https://github.com/Nethraa24/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121215786/88fcefd5-74ff-4d1e-adcb-f409a9d41685)

### Data-status

![image](https://github.com/Nethraa24/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121215786/b3a52f9e-fee6-4396-a5f2-2ad0f9edb341)

### y_prediction array 
![image](https://github.com/Nethraa24/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121215786/879e459a-5633-4db3-90af-15806f656d2c)

### Accuracy value

![image](https://github.com/Nethraa24/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121215786/81ed90af-a551-4a65-b107-e9e8b2401ad0)

### Confusion array

![image](https://github.com/Nethraa24/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121215786/f6ba769f-7719-4c06-9e49-28167a6d9a23)

### Classification report 

![image](https://github.com/Nethraa24/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121215786/c4a909ea-570a-4404-adea-340bc32cb90b)

### Prediction of LR


![image](https://github.com/Nethraa24/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121215786/93a1fc19-1bd6-4572-b72e-a885578da36c)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
