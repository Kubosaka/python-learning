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

# 平均誤差関数
def mse_line(x, t, w):
    y= w[0] * x + w[1]
    mse = np.mean((y-t)**2)
    return mse

# 計算
xn = 100    #等高線表示の解像度
w0_range = [-25, 25]
w1_range = [120, 170]
w0 = np.linspace(w0_range[0], w0_range[1], xn)
w1 = np.linspace(w1_range[0], w1_range[1], xn)
ww0, ww1 = np.meshgrid(w0,w1)
J = np.zeros((len(w0),len(w1)))
for i0 in range(len(w0)):
    for i1 in range(len(w1)):
        J[i1,i0]=mse_line(X, T,(w0[i0], w1[i1]))

#表示
plt.figure(figsize=(9.5,4))
plt.subplots_adjust(wspace=0.5)

ax=plt.subplot(1, 2, 1, projection='3d')
# ax.plot_surface(ww0, ww1, J, rstride=10, cstride=10, alpha=0.3, color='blue', edgecolor='blank')
ax.plot_surface(ww0, ww1, J, rstride=10, cstride=10, alpha=0.3, color='blue', edgecolor='black')
ax.set_xlabel('$w_0$', fontsize=8)
ax.set_ylabel('$w_1$', fontsize=8)
ax.set_xticks([-20,0,20])
ax.set_yticks([120,140,160])
ax.view_init(20,-60)

plt.subplot(1,2,2)
cont = plt.contour(ww0,ww1,J,30,colors='black',levels=[100,1000,10000,100000])
cont.clabel(fmt='%d', fontsize=8)
plt.grid(True)
plt.show()
