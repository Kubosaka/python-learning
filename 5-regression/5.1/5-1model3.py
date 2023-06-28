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

# 平均二条誤差の勾配
def dmse_line(x, t, w):
    y = w[0] * x + w[1]
    d_w0 = 2 * np.mean((y-t)*x)
    d_w1 = 2 * np.mean(y-t)
    return d_w0,d_w1
d_w = dmse_line(X,T,[10,165])
print(np.round(d_w,1))

#勾配法
def fit_line_num(x,t):
    w_init=[10.0,165.0] #初期パラメータ
    alpha = 0.001   #学習率
    tau_max = 100000    #繰り返しの最大数
    eps = 0.1   #繰り返しをやめる勾配の絶対値の閾値
    w_hist = np.zeros([tau_max,2])
    w_hist[0,:] = w_init
    for tau in range(1, tau_max):
        dmse = dmse_line(x, t, w_hist[tau-1])
        w_hist[tau, 0] = w_hist[tau-1, 0] - alpha * dmse[0]
        w_hist[tau, 1] = w_hist[tau-1, 1] - alpha * dmse[1]
        if max(np.absolute(dmse)) < eps:
            break
    w0 = w_hist[tau,0]
    w1 = w_hist[tau,1]
    w_hist=w_hist[:tau,:]
    return w0,w1,dmse,w_hist


# メイン
plt.figure(figsize=(4,4))   #MSEの等高線表示
wn = 100    #等高線表示の解像度
w0_range = [-25, 25]
w1_range = [120, 170]
w0 = np.linspace(w0_range[0], w0_range[1], wn)
w1 = np.linspace(w1_range[0], w1_range[1], wn)
ww0, ww1 = np.meshgrid(w0,w1)
J = np.zeros((len(w0),len(w1)))
for i0 in range(wn):
    for i1 in range(wn):
        J[i1,i0]=mse_line(X, T,(w0[i0], w1[i1]))

cont = plt.contour(ww0,ww1,J,30,colors='black',levels=[100,1000,10000,100000])
cont.clabel(fmt='%1.0f', fontsize=8)
plt.grid(True)

#勾配法呼び出し
W0,W1,dMSE,W_history = fit_line_num(X,T)

#結果表示
print('繰り返し回数 {0}'.format(W_history.shape[0]))
print('W = [{0:.6f}, {1:.6f}]'.format(W0,W1))
print('dMSE = [{0:.6f},{1:.6f}]'.format(dMSE[0], dMSE[1]))
print('MSE = [{0:.6f}]'.format(mse_line(X,T,[W0,W1])))
print(mse_line(X,T,[W0,W1]))
plt.plot(W_history[:,0],W_history[:,1], '.-', color='gray', markersize=10, markeredgecolor='cornflowerblue')
plt.show()
