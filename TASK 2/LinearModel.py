# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 00:41:36 2020

@author: Simran Agrawal
"""
#THE SPARKS FOUNDATION

# TASK 2- TO EXPLORE SUPERVISED MACHINE LEARNING

import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt


# Read our dataset
url = "https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv"
df = pd.read_csv(url)
print("Dataset is loaded.")


# Exploring our dataset
df.shape

df.head(10)

df.describe()


# Plotting the distribution of scores
df.plot(x = 'Hours', y = 'Scores', style = 'o', figsize = (9,6))
plt.title('Hours Vs Scores') 
plt.xlabel("Hours Studied")
plt.ylabel("Percentage Score")
plt.grid()
plt.show()

# OBERVATIONS
    #1. No Outliers
    #2. No missing values
    #3. the plot is showing a linear representation of x and y labels
    #4. Data Preprocessing is complete
    
    
#Geting X and Y lables 
#X includes all the rows of all the elements of columns excluding the last column
#Y label includes all the elments ofthe last column
X = df.iloc[:, :-1].values  
y = df.iloc[:, 1].values  


# getting training and train set data using train_test_split func from sklearn
from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 


print("X Train", X_train.shape)
print("X Test", X_test.shape)
print("Y Train", y_train.shape)
print("Y Test", y_test.shape)


# Fitting The model regressor to our training set data
from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(X_train, y_train) 

# line y = mx + c straight line concept
line = (regressor.coef_ * X) + regressor.intercept_


plt.figure(figsize=(9,6))
plt.scatter(X, y, color = 'blue')
plt.plot(X, line, color = 'red');
plt.xlabel("Hours Studied")
plt.ylabel("Percentage Score")
plt.grid()
plt.show()




print("The Training test set: ") 
print(X_test)


# Predicted Data Set.
# Predicting the scores using predict function of regressor model as y_pred using X_test data set.

y_pred = regressor.predict(X_test)

# Creating a Table And Comparing the Scores of our Actual Scores and Predicted Scores of y_pred and y_test Data respectively.
dataf = pd.DataFrame({'Actual Score': y_test, 'Predicted Score': y_pred})  
dataf

#Given Value in the task to predict its score 
hours = 9.25
prediction = regressor.predict([[hours]])
print("No of Hours = {}".format(hours))

 # Rounding the predicted score upto 2 decimal places
print("The  predicted Score of the student = {} %".format(round(prediction[0], 2))) 



# Evaluating our linearmodel
from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred)) 
print('Root Mean Absolute Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('R2 Score:', metrics.r2_score(y_test, y_pred))


# Saving model to our disk using with pkl extension of pickel module.
import pickle
pickle.dump(regressor, open('LinearModel.pkl','wb'))

# Loading model.
linearmodel = pickle.load(open('LinearModel.pkl','rb'))
print(linearmodel.predict([[9.25]]))


#Thank You
