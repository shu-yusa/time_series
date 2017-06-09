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
            parcor[m] -= a[m-2,j-1] * acovf[m-j]
        parcor[m] /= sig2[m-1]
        a[m-1,m-1] = parcor[m]
        for i in range(1, m):
            a[m-1,i-1] = a[m-2,i-1] - a[m-1,m-1] * a[m-2,m-i-1]
        sig2[m] = sig2[m-1] * (1 - a[m-1,m-1]**2)
        AIC = const + N * np.log(sig2[m]) + 2 * (m + 1)
        print("m=", m, "parcor=", parcor[m], "sig=", sig2[m], "AIC=", AIC)
        # AIC最小値を更新
        if AIC < AIC_min:
            AIC_min = AIC
            sig2_min = sig2[m]
            arc_min = a[m-1,:m]
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

def make_ar_model(data, m, arc, sig2):
    """
    Represent AR model in a state space representation.
    @param data data
    @param m AR order
    @param arc Auto-regressive coefficient
    @param sig2 variance
    """
    ndata = len(data)
    # dimension of state x
    k = m
    # dimension of noise observation of noise w
    l = 1
    x = np.zeros([ndata, k])
    for n in range(ndata):
        x[n,0] = data[n]
        for i in range(1, k):
            s = 0
            for j in range(i, m):
                s += arc[j] * data[n+i-j]
            x[n,i] = s
    F = np.zeros((k,k))
    for i in range(k-1):
        F[i,0] = arc[i]
        F[i,i+1] = 1
    F[k-1,0] = arc[k-1]
    G = np.zeros([k,1])
    G[0,0] = 1
    H = np.zeros([l,k])
    H[0,0] = 1
    Q = np.zeros((1,1))
    Q[0,0] = sig2
    R = np.zeros((l,l))
    return x, F, G, H, Q, R

def calc_arma_initial(data, m, arc, l, bac, sig2, acovf):
    """
    Represent ARMA model in a state space representation.
    @param data data
    @param m AR order
    @param arc Auto-regressive coefficient
    @param l MA order
    @param bac Moving-average coefficient
    @param sig2 variance
    @param acovf auto covariance function
    """
    ndata = len(data)
    k = m
    x0 = np.zeros(k)
    V0 = np.zeros((k, k))
    V0[0,0] = acovf[0]
    g = np.zeros(l)
    g[0] = 1
    g[i] = - bac[i]
    for j in range(i):
        g[i] += arc[j] * g[i-j]

    for i in range(1,k):
        for j in range(i,m):
            V0[0,i] += arc[j] * acovf[j+1-i]
        for j in range(i-1, l):
            V0[0,i] -= bac[j] * g[j+1-i]
    for i in range(1, k):
        for j in range(1, k):
            tmp = 0
            for p in range(i, m):
                for q in range(j, m):
                    tmp += arc[p] * arc[q] * acovf[q-j-p+i]
                for q in range(j-1, l):
                    tmp -= arc[p] * bac[q] * g[q-j-p+i]
            V0[i,j] = tmp
            tmp = 0
            for p in range(i-1, l):
                for q in range(j, m):
                   tmp -= bac[p] * arc[q] * g[p-i-q+j]
                tmp += bac[p] * bac[p+j-i] * sig2
            V0[i,j] += tmp
    return x0, V0

def logLF(y, yc, dc, N):
    l = N * np.log(2*np.pi)
    for n in range(N):
        l += np.log(dc[n]) + (y[n] - yc[n]) ** 2 / dc[n]
    l *= - 0.5
    return l

def kalman_filter(x, y, F, G, H, Q, R, x0, V0, missing=[], num_missed=[]):
    ndata = x.shape[0]
    shape = x.shape
    xc = np.zeros((ndata, ndata, shape[1]))
    Vc = np.zeros((ndata, ndata, shape[1], shape[1]))
    xc[0,0,:] = x0
    Vc[0,0,:,:] = V0
    GQGT = G @ Q @ G.T
    outlier_min = -1.0e30
    outlier_max = 1.0e30
    for i in range(len(missing)):
        for k in range(num_missed[i]):
            y[missing[i]:missing[i]+k,0] = outlier_min
    for n in range(1, ndata):
        xc[n,n-1,:] = F @ xc[n-1,n-1,:]
        Vc[n,n-1,:,:] = F @ Vc[n-1,n-1,:,:] @ F.T + GQGT
        VHT = Vc[n,n-1,:,:] @ H.T
        if y[n,0] > outlier_min and y[n,0] < outlier_max:
            K = VHT @ linalg.inv(H @ VHT + R)
            xc[n,n,:] = xc[n,n-1,:] + K @ (y[n,:] - H @ xc[n,n-1,:])
            Vc[n,n,:,:] = Vc[n,n-1,:,:] - K @ VHT.T
        else:
            xc[n,n,:] = xc[n,n-1,:]
            Vc[n,n,:,:] = Vc[n,n-1,:,:]
    return xc, Vc

def predict(xc, Vc, F, G, H, Q, R, ndata, dim, p_len):
    Vp = np.zeros([p_len+1, Vc.shape[2], Vc.shape[3]])
    xp = np.zeros([p_len+1, xc.shape[2]])
    xp[0,:] = xc[ndata-1,ndata-1,:]
    Vp[0,:,:] = Vc[ndata-1,ndata-1,:,:]
    GQGT = G @ Q @ G.T
    for n in range(ndata, ndata + p_len):
        xp[n-ndata+1,:] = F @ xp[n-ndata,:]
        Vp[n-ndata+1,:,:] = F @ Vp[n-ndata,:,:] @ F.T + GQGT
    yp = np.zeros([p_len+1, dim])
    dp = np.zeros([p_len+1, dim])
    for j in range(p_len+1):
        yp[j,:] = H @ xp[j,:]
        dp[j,:] = H @ Vp[j,:,:] @ H.T + R
    return yp, dp

def smooth(xc, Vc, F):
    ndata = xc.shape[0]
    m = Vc.shape[2]
    A = np.zeros(F.shape)
    for n in range(ndata - 1)[::-1]:
        invertable = False
        for k in range(m):
            if Vc[n+1,n,k,k] > 1.0e-12:
                invertable = True
                break
        A = Vc[n,n,:,:] @ F.T @ linalg.pinv(Vc[n+1,n,:,:])
        xc[n,ndata-1,:] = xc[n,n,:] + A @ (xc[n+1,ndata-1,:] - xc[n+1,n,:])
        Vc[n,ndata-1,:,:] = Vc[n,n,:,:] + \
                A @ (Vc[n+1,ndata-1,:,:] - Vc[n+1,n,:,:]) @ A.T
        for k in range(m):
            if Vc[n,ndata-1,k,k] < 0:
                Vc[n,ndata-1,k,k] = 0
    return xc, Vc

def interpolate(xc, Vc, H, R, ndata, dim):
    ym = np.zeros([ndata, dim])
    dm = np.zeros([ndata, dim])
    for n in range(ndata):
        ym[n,:] = H @ xc[n,ndata-1,:]
        dm[n,:] = H @ Vc[n,ndata-1,:,:] @ H.T + R
    return ym, dm

if __name__ == "__main__":
    plt.figure(1)
    plt.clf()

    maxm = 25

    # 太陽黒点数
    with open('sunspot.txt', encoding='utf-8') as f:
        data = np.array([float(k) for k in f.readlines()])
    # データ数
    N = len(data)
    # 対数値に変換
    data = np.log10(data)
    # 平均を引く
    data = data - np.mean(data)
    # 自己共分散関数
    acovf = stattools.acovf(data)

    mar, arc_min, sig2_min, AIC_min = Levinson(acovf, N, 1)
    x0, V0 = calc_arma_initial(data, 1, arc_min, 0, [], sig2_min, acovf)

    # Levinson's algorithm
    print()
    print("Levinson method")
    mar, arc_min, sig2_min, AIC_min = Levinson(acovf, N, maxm)
    print('Best model: m=', mar)
    print('AR coefficiants:', arc_min)
    # スペクトル
    t, logp2 = calc_spectrum(400, arc_min, sig2_min)
    y_pre2 = calc_time_series(data, arc_min, N, mar)

    # kalman filter
    data_trim = data[:120]
    x, F, G, H, Q, R = make_ar_model(data_trim, mar, arc_min, sig2_min)
    x0 = np.zeros(x.shape[1])
    V0 = np.zeros((x.shape[1], x.shape[1]))
    y = np.zeros((data_trim.shape[0], 1))
    y[:,0] = data_trim
    xc, Vc = kalman_filter(x, y, F, G, H, Q, R, x0, V0, [41], [30])
    yc, dp = predict(xc, Vc, F, G, H, Q, R, len(data_trim), 1, N - len(data_trim))
    xc, Vc = smooth(xc, Vc, F)
    ym, dm = interpolate(xc, Vc, H, R, len(data_trim), 1)

    # プロット
    plt.subplot(2,1,1)
    plt.plot(t, logp2, label="Yule-Walker")
    plt.legend()

    # 時系列計算
    t = range(N)
    plt.subplot(2,1,2)
    # plt.plot(t, y_pre2 + mean, label="Yule-Walker")
    t = range(len(data_trim))
    plt.plot(t, data_trim + mean, label="Data")
    # interpolation
    t = range(41, 71)
    plt.plot(t, ym[41:71] + mean, label="interpolation")
    plt.plot(t, ym[41:71] + np.sqrt(dm[41:71]) + mean)
    plt.plot(t, ym[41:71] - np.sqrt(dm[41:71]) + mean)
    t = range(len(data_trim) - 1, len(data_trim) - 1 + len(yc))
    plt.plot(t, data[len(data_trim)-1:N] + mean, "o", label="Data")
    plt.plot(t, yc + mean, label="prediction")
    plt.plot(t, yc + np.sqrt(dp) + mean)
    plt.plot(t, yc - np.sqrt(dp) + mean)

    plt.legend(loc="lower left")
    plt.show()

