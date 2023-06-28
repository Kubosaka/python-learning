import numpy as np
import matplotlib.pyplot as plt

np.random.seed(seed=1)
X_min = 4
X_max = 30
X_n = 16    #データの個数
X = 5 + 25 * np.random.rand(X_n)    #Xの生成 5~30の間のサイズ16の乱数配列
Prm_c = [170, 108, 0.2] #生成パラメータ
T = Prm_c[0] - Prm_c[1] * np.exp(-Prm_c[2] * X) + 4 * np.random.randn(X_n)

plt.figure(figsize=(4,4))
plt.plot(X ,T , marker='o', linestyle='None', markeredgecolor='black', color='cornflowerblue')
plt.xlim(X_min,X_max)
plt.grid(True)
plt.show()
