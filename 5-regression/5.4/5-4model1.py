import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# X,Tの生成
np.random.seed(seed=1)
X_min = 4
X_max = 30
X_n = 16    #データの個数
X = 5 + 25 * np.random.rand(X_n)    #Xの生成 5~30の間のサイズ16の乱数配列
Prm_c = [170, 108, 0.2] #生成パラメータ
T = Prm_c[0] - Prm_c[1] * np.exp(-Prm_c[2] * X) + 4 * np.random.randn(X_n)
X_min = 0

#ガウス関数
def gauss(x, mu, s):
    return np.exp(-(x-mu)**2 / (2 * s**2))

#メイン
M = 4
plt.figure(figsize=(4,4))
mu = np.linspace(5, 30, M)
s = mu[1]- mu[0]
xb = np.linspace(X_min, X_max, 100)
for j in range(M):
    y = gauss(xb, mu[j], s)
    plt.plot(xb, y, color = 'gray', linewidth = 3)
plt.grid(True)
plt.xlim(X_min, X_max)
plt.ylim(0, 1.2)
plt.show()
