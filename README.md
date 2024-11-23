# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
STEP 1: Start

STEP 2: Load the California housing dataset and create a Pandas DataFrame for the features and target variables (AveOccup and HousingPrice).

STEP 3: Split the data into features (X) excluding AveOccup and HousingPrice, and target variables (Y) containing AveOccup and HousingPrice.

STEP 4: Split the dataset into training and testing sets using train_test_split.

STEP 5: Standardize the training and testing sets using StandardScaler for both features and target variables.

STEP 6: Initialize the SGDRegressor and wrap it in a MultiOutputRegressor for multivariate predictions.

STEP 7: Train the model on the scaled training data.

STEP 8: Make predictions on the testing data using the trained model and inverse transform the predictions to original scale.

STEP 9: Calculate the mean squared error (MSE) between the predicted and actual target values.

STEP 10: Print the mean squared error and display the first 5 predicted values.

STEP 11: End
## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: VARSHINI S 
RegisterNumber: 212222220056
*/
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
data=fetch_california_housing()
X=data.data[:,:3]
Y=np.column_stack((data.target,data.data[:,6]))
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
scaler_X=StandardScaler()
scaler_Y=StandardScaler()
X_train=scaler_X.fit_transform(X_train)
X_test=scaler_X.fit_transform(X_test)
Y_train=scaler_Y.fit_transform(Y_train)
Y_test=scaler_Y.fit_transform(Y_test)
sgd=SGDRegressor(max_iter=1000,tol=1e-3)
multi_output_sgd=MultiOutputRegressor(sgd)
multi_output_sgd.fit(X_train,Y_train)
Y_pred=multi_output_sgd.predict(X_test)
Y_pred=scaler_Y.inverse_transform(Y_pred)
Y_test=scaler_Y.inverse_transform(Y_test)
mse=mean_squared_error(Y_test,Y_pred)
print("Mean Squared Error:",mse)
print("\nPredictions:\n",Y_pred[:5])


```

## Output:
![image](https://github.com/user-attachments/assets/9d71885b-e8e1-4722-9128-9994fe2c6c72)




## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
