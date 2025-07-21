#14
'''Write a python numpy program to compute the weighted average along the specified axis of a given flattened array.'''

import numpy as np
a = np.array([[0,1,2],[3,4,5],[6,7,8]])
print("Original flattened array:")
print(a)
print("Weighted average along the specified axis of the above flattened array")
print(np.average(a,axis=1,weights=[1./4, 2./4, 2./4]))

