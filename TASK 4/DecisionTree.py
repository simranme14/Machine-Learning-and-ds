# -*- coding: utf-8 -*-
"""

@author: Simran Agrawal
"""
#Task 4 - To Explore Decision Tree Algorithm
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Download the iris dataset
url = "https://drive.google.com/file/d/11Iq7YvbWZbt8VXjfm06brx66b10YiwK-/view?usp=sharing"
path = 'https://drive.google.com/uc?export=download&id='+url.split('/')[-2]
iris_data = pd.read_csv(path)

# Explore the dataset
iris_data.head() # See the first 5 rows

# Count of flowers in each unique species
iris_data['Species'].value_counts()

# Correlation
corr_df = iris_data.corr()
corr_df

##Ploting datset
plt.figure(figsize=(8, 8))
sns.pairplot(iris_data.dropna(),hue="Species")


#Splitting our data into Training Set and Test Set
#we will get independent values into x arrays and dependent values into y arrays
from sklearn.model_selection import train_test_split
X = iris_data.iloc[:, 1:5]
Y = iris_data.iloc[:, 5]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 0)



#Training the Decision Tree Classifier on the dataset and fiting the training set data
from sklearn.tree import DecisionTreeClassifier
TreeModel = DecisionTreeClassifier(random_state=0)
TreeModel.fit(X_train, Y_train)


#Predicting the test data and calculating accuracy and score report of the model
Y_pred = TreeModel.predict(X_test)

from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred))


from sklearn.metrics import accuracy_score
accuracy = accuracy_score(Y_test, Y_pred)
print("Accuracy of this model = {}%".format(round(accuracy*100),2))

from sklearn.metrics import confusion_matrix
confusion_matrix(Y_test,Y_pred)

# Visualizing the Decision Tree Classifier
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
dot_data = export_graphviz(TreeModel, out_file=None,feature_names=X_train.columns, filled = True, rounded=True)
graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())

#controlling tree growth
#Minimum observations at internal node

tree2 = DecisionTreeClassifier(min_samples_split = 10)
tree2.fit(X_train, Y_train)
dot_data = export_graphviz(tree2, out_file=None,feature_names= X_train.columns, filled = True)
graph2 = pydotplus.graph_from_dot_data(dot_data)
Image(graph2.create_png())


#Controlling the tree growth.
#Minimum Observations At Tree Node

tree3 = DecisionTreeClassifier(min_samples_leaf =5, max_depth=4)
tree3.fit(X_train, Y_train)
dot_data = export_graphviz(tree3, out_file=None,feature_names=X_train.columns, filled = True, rounded=True)
graph3 = pydotplus.graph_from_dot_data(dot_data)
Image(graph3.create_png())

# Saving the model
import pickle
pickle.dump(TreeModel, open('DecisionTree.pkl','wb'))

# Loading the model to compare the results
model = pickle.load(open('DecisionTree.pkl','rb'))
print(model.predict([[5.6,2.43,4.123,1.23]]))
