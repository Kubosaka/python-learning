import numpy as np

A=np.array([[2,-1],[1,1]])
invA=np.linalg.inv(A)
B=np.array([0,3])

print("2x-y=0")
print("x+y=3")
print("[x,y] =",invA.dot(B))
