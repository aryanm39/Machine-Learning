#1. Import libraries/packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error

#2. Loading the data
df = pd.read_csv('datasets/User_Data.csv')
print(df.sample(5))
print(df.shape)
print(df.value_counts())
print(df.isnull().sum())
print(df.columns)
df.dropna()
print(df.shape)
 
X=np.array(df[['EstimatedSalary']])
y=np.array(df[['Purchased']])

"""  
X =  df[['Height']]
y = df['Weight']
 """
print(X.shape)
print(y.shape)

#3. Visualizing the data
plt.scatter(X,y,color="red")
plt.title('Estimated Salary vs Purchased')
plt.xlabel('Estimated Salary')
plt.ylabel('Purchased')
plt.show()

#4. Splitting our Data set in Dependent and Independent variables.
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=15)

#5. Performing simple linear regression 
regressor= LinearRegression()
regressor.fit(X_train,y_train)
#Test Accuracy
accuracy = regressor.score(X_test,y_test)
print("\n\n Accuracy of model =",accuracy)
print("Coeficients",regressor.coef_)
print("Intercepts",regressor.intercept_)

#6. Residual analysis(Check the results of model fitting to know whether the model is satisfactory)
plt.scatter(X_test,y_test,color='green')
plt.plot(X_train,regressor.predict(X_train),color="red",linewidth=3)
plt.title('Regression(Test Set)')
plt.xlabel('Estimated Salary')
plt.ylabel('Purchased')
plt.show()
plt.scatter(X_train,y_train,color="blue")
plt.plot(X_train,regressor.predict(X_train),color="red",linewidth=3)
plt.title('regression training set')
plt.xlabel('Estimated Salary')
plt.ylabel('Purchased')
plt.show()

#7. Predictions on the test set (apply the model)
y_pred=regressor.predict(X_test)
print(f"r2 score {r2_score(y_test,y_pred)}")
print(f"mean error { mean_squared_error(y_test,y_pred)}")

Height = 165
result=regressor.predict(np.array(Height).reshape(1,-1))
Weight=result[0,0]
print(f"Weight will be : {Weight}")