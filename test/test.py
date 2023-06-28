import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)
x=np.arange(10)
y=np.random.rand(10)
a=np.array([[1,3],[0,0],[1,2]])
print(a.shape)
plt.plot(x,y)
plt.show()


