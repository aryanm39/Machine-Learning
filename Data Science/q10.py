#9,15,25
'''Create two lists,one representing subject names and the other representing marks obtained in those subjects.Display the data in a pie chart'''

#16
'''Write a python program to create rwo lists,one representing subject names and the other representing marks obtained in those subjects.
Display the data in a pie chart and bar chart'''
#26,30
'''create two lists,one representing subject names and the other representing marks obtained in those subjects.
Display the data in bar chart'''

import numpy as np
import matplotlib.pyplot as plt
Subjects  = ['C','C++','JAVA','PYTHON','PHP','DSA','BT']
Marks = [90,75,97,86,67,92,91]
plt.bar(Subjects,Marks,color='green',edgecolor='magenta',width=0.4)
plt.xlabel("Subjects",fontweight='bold',fontsize=10,color='blue')
plt.ylabel("Marks",fontweight='bold',fontsize=10,color='blue')
plt.title("Subjects with marks",fontweight='bold',fontsize=15,color='red')
plt.show()

plt.pie(Marks,labels=Subjects)
plt.legend(Subjects,title="Subjects",loc='center left',bbox_to_anchor=(1,1,-0.5,-0.5))
plt.title("Subjects vs Marks in pie chart",color="red")
plt.show()
