import numpy as np

A=np.array([[1,2],[3,4]])
invA=np.linalg.inv(A)

print(A)
print(invA)

print(A.dot(invA))
