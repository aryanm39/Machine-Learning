#1,11
'''Write a python program to create a pie plot to get the frequency of the three species of the iris data(Use iris.csv)'''
#21,24
'''Write a python program to create a histogram of the three species of the iris data.'''
#17
'''Write a python program to draw scatter plots to compare two features of the iris dataset'''

#21,24
'''Import dataset "iris.csv" Write a python program to create a Bar plot to get the frequency of the three species of the iris data'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#pie plot
iris = pd.read_csv("iris.csv")
iris['Species'].value_counts().plot.pie(explode=[0.1,0.1,0.1],autopct='%1.1f%%')
plt.title("Iris Species %")
plt.show()

#histogram
a = pd.read_csv('iris.csv')
x = a['SepalLengthCm']
plt.hist(x,bins=10,color='yellow',edgecolor='blue')
plt.xlabel('X-axis',fontweight='bold',fontsize=10,color='blue')
plt.ylabel('Y-axis',fontweight='bold',fontsize=10,color='blue')
plt.title("Histogram Chart",fontweight='bold',fontsize=15,color='red')
plt.show()

#Scatter plot

df = pd.read_csv("iris.csv")
sns.FacetGrid(df,hue='Species').map(plt.scatter,'SepalLengthCm','PetalLengthCm')
plt.show()

#bar plot
x = iris['Species']
y = iris['SepalLengthCm']
plt.bar(x,y,color='green')
plt.xlabel("Classes",fontweight='bold',fontsize=10,color='blue')
plt.ylabel("Sepal length",fontweight='bold',fontsize=10,color='green')
plt.title("Iris Species Count")
plt.show()



