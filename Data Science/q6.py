#4,5,12,15,20,25,26,30

#30
'''Generate a random array of 50 integers and display them using a line chart,scatter plot,histogram and box plot.
Apply appropriate color ,labels and styling options''' 
#9
'''Generate a random array of 50 integers and display them using a line chart,scatter plot
Apply appropriate color,labels and styling options'''

import numpy as np
import matplotlib.pyplot as plt
a = np.random.randint(100,size=(50))

#Line Chart
x = np.random.randint(100,size=(50))
y = np.random.randint(100,size=(50))
plt.plot(x,y,color='blue',linestyle='dashed',marker='*',markerfacecolor='red')
plt.xlabel("X-axis",fontweight='bold',fontsize=10,color='blue')
plt.ylabel("Y-axis",fontweight='bold',fontsize=10,color='blue')
plt.title("Line Chart",fontweight='bold',fontsize=15,color='red')
plt.show()

#Scatter Plot
x = np.random.randint(100,size=(50))
y = np.random.randint(100,size=(50))
plt.scatter(x,y,marker="^",edgecolor='r',s=50,linewidths=2,color='yellow')
plt.xlabel("X-axis",fontweight='bold',fontsize=10,color='blue')
plt.ylabel("Y-axis",fontweight='bold',fontsize=10,color='blue')
plt.title("Scatter Chart",fontweight='bold',fontsize=15,color='red')
plt.show()

#Histogram
plt.hist(a,bins=10,color='pink',edgecolor='blue')
plt.xlabel("X-axis",fontweight='bold',fontsize=10,color='blue')
plt.ylabel("Y-axis",fontweight='bold',fontsize=10,color='blue')
plt.title("Histogram Chart",fontweight='bold',fontsize=15,color='red')
plt.show()

#Box Plot
plt.boxplot(a)
plt.show()