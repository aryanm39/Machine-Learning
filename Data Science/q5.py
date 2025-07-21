#3,18
'''Write a python program to create box plots to see how each feature i.e. Sepal Length,Sepal width,Petal length,Petal width are distributed 
across the three species(Use iris.csv dataset)'''

import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("iris.csv")
new_data = data[["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]]  
#print(new_data.head())
new_data.boxplot()
plt.title('Box Plot')
plt.show()