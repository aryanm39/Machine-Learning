#10
'''Write a python program to compute sum of Manhatten distance between all pairs of points.'''

import numpy as np
a=[1,2,3]
b=[5,3,2]
dis=0
for i in range(len(a)):
	dis+=abs(a[i]-b[i])
print('first array is:',a)
print('second array is:',b)
print('Manhattan distance is:',dis)