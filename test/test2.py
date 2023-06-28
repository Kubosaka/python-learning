import numpy as np
import matplotlib.pyplot as plt

#(A) fの定義
def f(w0,w1):
    return w0**2 + 2*w0*w1 + 3

#B fのw0に関する偏微分
def df_dw0(w0,w1):
    return 2*w0 + 2*w1

#C fのw1に関する偏微分
def df_dw1(w0,w1):
    return 2*w0 +0*w1

w_range=2
dw=0.25
w0=np.arange(-w_range,w_range+dw,dw)
w1=np.arange(-w_range,w_range+dw,dw)

#w0=[-2.   -1.75 -1.5  -1.25 -1.   -0.75 -0.5  -0.25  0.    0.25  0.5   0.75  1.    1.25  1.5   1.75  2.  ]
#w1=[-2.   -1.75 -1.5  -1.25 -1.   -0.75 -0.5  -0.25  0.    0.25  0.5   0.75  1.    1.25  1.5   1.75  2.  ]

ww0, ww1=np.meshgrid(w0,w1)

#ww0 
#[-2 ~ 2]
#   ~
#[-2 ~ 2]

#ww1
#[-2 ~ -2]
#   ~
#[ 2 ~  2]

ff = np.zeros((len(w0), len(w1)))
dff_dw0=np.zeros((len(w0), len(w1)))
dff_dw1=np.zeros((len(w0), len(w1)))

for i0 in range(len(w0)):
    for i1 in range(len(w1)):
        ff[i1, i0] = f(w0[i0],w1[i1])
        dff_dw0[i1, i0] = df_dw0(w0[i0], w1[i1])
        dff_dw1[i1, i0] = df_dw1(w0[i0], w1[i1])

#F fの等高線表示
plt.figure(figsize=(9,4))
plt.subplots_adjust(wspace=0.3)
plt.subplot(1,2,1)
cont=plt.contour(ww0,ww1,ff,10,colors='k')
cont.clabel(fmt='%d',fontsize=8)
plt.xticks(range(-w_range, w_range+1,1))
plt.yticks(range(-w_range, w_range+1,1))
plt.xlim(-w_range-0.5,w_range+0.5)
plt.ylim(-w_range-0.5,w_range+0.5)
plt.xlabel('$w_0$', fontsize=14)
plt.ylabel('$w_1$', fontsize=14)

#G fの勾配ベクトル表示
plt.subplot(1,2,2)
plt.quiver(ww0, ww1, dff_dw0, dff_dw1)
plt.xticks(range(-w_range, w_range+1,1))
plt.yticks(range(-w_range, w_range+1,1))
plt.xlim(-w_range-0.5,w_range+0.5)
plt.ylim(-w_range-0.5,w_range+0.5)
plt.xlabel('$w_0$', fontsize=14)
plt.ylabel('$w_1$', fontsize=14)

plt.show()
