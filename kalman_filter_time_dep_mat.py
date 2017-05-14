import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import optimize
from scipy import linalg
import scipy.signal as sig
from statsmodels.tsa import stattools

def Levinson(C, N, maxm):
    """
    @param C Autocovaiance function
    @param N Data length
    @param maxm Highest AR order
    """
    const = N * (np.log(2 * np.pi) + 1)
    # matrix for auto regressive coefficients
    a = np.zeros((maxm+1, maxm+1))
    sig2 = np.zeros(maxm+1)
    # covariance for the 0the order
    sig2[0] = acovf[0]
    # AIC
    AIC = const + N * np.log(sig2[0]) + 2
    parcor = np.zeros(maxm + 1)
    parcor[0] = 0
    # minimum
    sig2_min = acovf[0]
    arc_min = []
    AIC_min = const + N * np.log(sig2_min) + 2 * (maxm + 1)
    mar = 0
    # calculate AR model upto maxm
    print("m=", 0, "sig=", sig2[0], "AIC=", AIC)
    for m in range(1, maxm + 1):
        parcor[m] = acovf[m]
        for j in range(1, m):
            parcor[m] -= a[m-2][j-1] * acovf[m-j]
        parcor[m] /= sig2[m-1]
        a[m-1][m-1] = parcor[m]
        for i in range(1, m):
            a[m-1][i-1] = a[m-2][i-1] - a[m-1][m-1] * a[m-2][m-i-1]
        sig2[m] = sig2[m-1] * (1 - a[m-1][m-1]**2)
        AIC = const + N * np.log(sig2[m]) + 2 * (m + 1)
        print("m=", m, "parcor=", parcor[m], "sig=", sig2[m], "AIC=", AIC)
        # AIC最小値を更新
        if AIC < AIC_min:
            AIC_min = AIC
            sig2_min = sig2[m]
            arc_min = a[m-1]
            mar = m
    return mar, arc_min, sig2_min, AIC_min

def calc_spectrum(ngrid, arc, sig2):
    """
    calculate spectrum
    @param ngrid the number of points to calculate spectrum
    @param arc auto regressive coefficient
    """
    # スペクトル
    denom = 1
    p = []
    for i in range(ngrid):
        f = 0.5 * i / (ngrid - 1)
        denom = 1
        for j in range(1, mar):
            denom -= arc[j-1] * np.exp(-2j * np.pi * j * f)
        p.append(np.log10(sig2 / np.absolute(denom) ** 2))
    t = [0.5 * i / (ngrid-1) for i in range(ngrid)]
    return t, p

def calc_time_series(data, arc, N, m):
    """
    @param data data
    @param arc auto regressive coefficient
    @param N data length
    @param m AR order
    """
    # 時系列計算
    y_pre = []
    y_pre.append(data[0])
    for n in range(1, N):
        yk = 0
        for i in range(1, np.minimum(n + 1, m)):
            yk += arc[i-1] * data[n-i]
        y_pre.append(yk)
    return y_pre

def make_ar_model(data, arc, m):
    ndata = len(data)
    k = m
    l = 1
    x = np.zeros([ndata, k])
    for n in range(ndata):
        x[n,0] = data[n]
        for i in range(1, k):
            s = 0
            for j in range(i, m):
                s += arc[j] * data[n+i-j]
            x[n,i] = s
    F = np.zeros((ndata,k,k))
    for i in range(k-1):
        F[:,i,0] = arc[i]
        F[:,i,i+1] = 1
    F[:,k-1,0] = arc[k-1]
    G = np.zeros([ndata,k,1])
    G[:,0,0] = 1
    H = np.zeros([ndata,l,k])
    H[:,0,0] = 1
    return x, F, G, H

def calman_filter(x, y, F, G, H, Q, R, xc0, V0):
    shape = x.shape
    xc = np.zeros((shape[0],shape[0], shape[1]))
    V = np.zeros((shape[0], shape[0], shape[1], shape[1]))
    xc[0,0,:] = x0
    V[0,0,:,:] = V0
    E = np.identity(shape[1])
    for n in range(1, shape[0]):
        xc[n,n-1,:] = F[n,:,:] @ xc[n-1,n-1,:]
        V[n,n-1,:,:] = F[n,:,:] @ V[n-1,n-1,:,:] @ np.transpose(F[n,:,:]) + \
                G[n,:,:] @ Q[n,:,:] @ np.transpose(G[n,:,:])
        Ht = np.transpose(H[n,:,:])
        K = V[n,n-1,:,:] @ Ht @ \
                linalg.inv(H[n,:,:] @ V[n,n-1,:,:] @ Ht + R[n,:,:])
        xc[n,n,:] = xc[n,n-1,:] + K @ (y[n,:] - H[n,:,:] @ xc[n,n-1,:])
        V[n,n,:,:] = (E - K @ H[n,:,:]) @ V[n,n-1,:,:]
    return xc, V

def predict(xc, V, F, G, H, Q, R, ndata, dim, p_len):
    Vp = np.zeros([p_len+1, V.shape[2], V.shape[3]])
    xp = np.zeros([p_len+1, xc.shape[2]])
    xp[0,:] = xc[ndata-1,ndata-1,:]
    Vp[0,:,:] = V[ndata-1,ndata-1,:,:]
    for n in range(ndata, ndata + p_len):
        xp[n-ndata+1,:] = F[ndata-1,:,:] @ xp[n-ndata,:]
        Vp[n-ndata+1,:,:] = F[ndata-1,:,:] @ Vp[n-ndata,:,:] @ \
                np.transpose(F[ndata-1,:,:]) + \
                G[ndata-1,:,:] @ Q[ndata-1,:,:] @ \
                np.transpose(G[0,:,:])
    yp = np.zeros([p_len+1, dim])
    dp = np.zeros([p_len+1, dim])
    for j in range(p_len+1):
        yp[j,:] = H[0,:,:] @ xp[j,:]
        dp[j,:] = H[0,:,:] @ Vp[j,:,:] @ np.transpose(H[0,:,:]) + R[0,:,:]
    return yp, dp

if __name__ == "__main__":
    plt.figure(1)
    plt.clf()

    maxm = 15

    # 太陽黒点数
    with open('blsallfood.txt', encoding='utf-8') as f:
        data = np.array([float(k) for k in f.readlines()])
    # データ数
    data_org = data
    N_org = len(data)
    data = data[:120]
    N = len(data)
    # 対数値に変換
    # data = np.log10(data)
    # 平均を引く
    mean = np.mean(data)
    data = data - mean
    # 自己共分散関数
    acovf = stattools.acovf(data)
    # acovf = acovf * (N - 1) / N

    # Levinson's algorithm
    print()
    print("Levinson method")
    mar, arc_min, sig2_min, AIC_min = Levinson(acovf, N, maxm)
    print('Best model: m=', mar)
    # スペクトル
    t, logp2 = calc_spectrum(400, arc_min, sig2_min)
    y_pre2 = calc_time_series(data, arc_min, N, mar)

    # kalman filter
    x, F, G, H = make_ar_model(data, arc_min, mar)
    print(F.shape)
    Q = np.zeros((N,1,1))
    Q[:,0,0] = sig2_min
    R = np.zeros((N,1,1))
    x0 = np.zeros(x.shape[1])
    V0 = np.zeros((x.shape[1], x.shape[1]))
    y = np.zeros((data.shape[0], 1))
    y[:,0] = data
    xc, V = calman_filter(x, y, F, G, H, Q, R, x0, V0)
    yc, dp = predict(xc, V, F, G, H, Q, R, N, 1, N_org - N)

    # プロット
    plt.subplot(2,1,1)
    plt.plot(t, logp2, label="Yule-Walker")
    plt.legend()

    # 時系列計算
    t = range(N)
    plt.subplot(2,1,2)
    plt.plot(t, y_pre2 + mean, label="Yule-Walker")
    plt.plot(t, data + mean, label="Data")
    t = range(N-1, N-1 + len(yc))
    plt.plot(t, data_org[N-1:N_org], "o", label="Data")
    plt.plot(t, yc + mean, label="prediction")
    plt.plot(t, yc + np.sqrt(dp) + mean)
    plt.plot(t, yc - np.sqrt(dp) + mean)


    plt.legend(loc="upper left")
    plt.show()

