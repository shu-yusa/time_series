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

def ar_least_square(data, N, maxm):
    """
    @param data data
    @param N data length
    @param maxm highest order of AR model
    """
    X = []
    dof = N - maxm
    for m in range(dof):
        row = data[m:maxm + m]
        row = np.append(row[::-1], data[maxm + m])
        X.append(row)
    Q, R = np.linalg.qr(X)

    const = dof * (np.log(2 * np.pi) + 1) + 2
    # residual variance
    sig2 = np.zeros(maxm + 1)
    res = 0
    for i in range(1, maxm + 2)[::-1]:
        res += R[i-1][maxm] ** 2
        sig2[i-1] = res / dof

    # minimum
    sig2_min = 0
    arc_min = []
    mar = 0
    AIC_min = const + dof * np.log(sig2[0])
    print("m=", 0, "sig2=", sig2[0], "AIC=", AIC_min)
    for m in range(1, maxm + 1):
        AIC = const + dof * np.log(sig2[m]) + 2 * m
        print("m=", m, "sig2=", sig2[m], "AIC=", AIC)
        # AIC最小値を更新
        if AIC < AIC_min:
            AIC_min = AIC
            sig2_min = sig2[m]
            mar = m

            # AR coefficient
            a = np.zeros(maxm)
            a[m-1] = R[m-1][maxm] / R[m-1][m-1]
            for i in range(m-1, 0, -1):
                a[i-1] = R[i-1][maxm]
                for j in range(i, m):
                    a[i-1] -= R[i-1][j] * a[j]
                a[i-1] /= R[i-1][i-1]
            arc_min = a
    return mar, arc_min, sig2_min, AIC_min

def estimate_parcor(method, N, m, v, w):
    """
    estimate PARCOR
    """
    numer = 0
    if method == 1:
        denom = 0
        for n in range(m + 1, N + 1):
            denom += w[n-m-1] * w[n-m-1]
            numer += v[n-1] * w[n-m-1]
    elif method == 2:
        w2 = v2 = 0
        for n in range(m + 1, N + 1):
            w2 += w[n-m-1] * w[n-m-1]
            v2 += v[n-1] * v[n-1]
            numer += v[n-1] * w[n-m-1]
        denom = np.sqrt(w2 * v2)
    elif method == 3:
        w2 = v2 = 0
        for n in range(m + 1, N + 1):
            w2 += w[n-m-1] * w[n-m-1]
            v2 += v[n-1] * v[n-1]
            numer += v[n-1] * w[n-m-1]
        denom = 0.5 * (w2 + v2)
    return numer / denom

def parcor(data, N, maxm, method):
    """
    solve AR model by PARCOR method
    @param data data
    @param N data length
    @param maxm highest order of AR model
    @param method estimate method
    """
    dof = N - maxm
    const = dof * (np.log(2 * np.pi) + 1)
    # previous values
    v_prev = np.array(data)
    w_prev = np.array(data)
    arc_prev = []
    sig2_prev = 0
    for i in range(maxm, N):
        sig2_prev += data[i] * data[i]
    sig2_prev /= dof

    # dof = N
    # const = dof * (np.log(2 * np.pi) + 1)
    # sig2_prev = np.sum(data**2) / dof

    v = np.zeros(N)
    w = np.zeros(N)

    # minimum values
    AIC_min = const + dof * np.log(sig2_prev) + 2
    sig2_min = sig2_prev
    arc_min = []
    mar = 0
    print("m=", 0, "sig2=", sig2_min, "AIC=", AIC_min)
    for m in range(1, maxm+1):
        parcor = estimate_parcor(method, N, m, v_prev, w_prev)
        arc = []
        for j in range(1, m):
            arc.append(arc_prev[j-1] - parcor * arc_prev[m-j-1])
        arc.append(parcor)
        for n in range(m+1, N+1):
            v[n-1] = v_prev[n-1] - parcor * w_prev[n-m-1]
            w[n-m-1] = w_prev[n-m-1] - parcor * v_prev[n-1]
        sig2 = sig2_prev * (1 - parcor * parcor)
        AIC = const + dof * np.log(sig2) + 2 * m
        print("m=", m, "parcor=", parcor, "sig2=", sig2, "AIC=", AIC)
        if AIC < AIC_min:
            mar = m
            sig2_min = sig2
            AIC_min = AIC
            arc_min = arc
        v_prev = v
        w_prev = w
        arc_prev = arc
        sig2_prev = sig2
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

    maxm = 11

    # 太陽黒点数
    with open('sin.txt', encoding='utf-8') as f:
        data = np.array([float(k) for k in f.readlines()])
    # データ数
    data_org = data
    N_org = len(data)
    # data = data[:24]
    N = len(data)
    # 平均を引く
    mean = np.mean(data)
    data = data - mean
    # 自己共分散関数
    acovf = stattools.acovf(data)
    print(acovf)
    # acovf = acovf * (N - 1) / N

    print()
    print("PARCOR method")
    method = 2
    # mar, arc_min, sig2_min, AIC_min = parcor(data[:24], 24, maxm, method)
    mar, arc_min, sig2_min, AIC_min = Levinson(acovf, 24, maxm)
    #mar, arc_min, sig2_min, AIC_min = ar_least_square(data[:24], 24, maxm)
    print('Best model: m=', mar)
    print('AR coefficiants:', arc_min)
    # スペクトル
    t, logp2 = calc_spectrum(400, arc_min, sig2_min)
    y_pre2 = calc_time_series(data, arc_min, 24, mar)

    # kalman filter
    data_trim = data[:24]
    x, F, G, H, Q, R = make_ar_model(data_trim, mar, arc_min, sig2_min)
    x0 = np.zeros(x.shape[1])
    V0 = np.zeros((x.shape[1], x.shape[1]))
    y = np.zeros((data_trim.shape[0], 1))
    y[:,0] = data_trim
    xc, Vc = kalman_filter(x, y, F, G, H, Q, R, x0, V0)
    yc, dp = predict(xc, Vc, F, G, H, Q, R, len(data_trim), 1, N - len(data_trim))
    xc, Vc = smooth(xc, Vc, F)
    ym, dm = interpolate(xc, Vc, H, R, len(data_trim), 1)

    # プロット
    plt.subplot(2,1,1)
    plt.plot(t, logp2, label="PARCOR")
    plt.legend()

    # 時系列計算
    t = range(N)
    plt.subplot(2,1,2)
    # plt.plot(t, y_pre2 + mean, label="Yule-Walker")
    t = range(len(data_trim))
    plt.plot(t, data_trim + mean, label="Data")
    # prediction
    t = range(len(data_trim) - 1, len(data_trim) - 1 + len(yc))
    plt.plot(t, data[len(data_trim)-1:N] + mean, ":", label="sin")
    plt.plot(t, yc + mean, label="prediction")
    plt.plot(t, yc + np.sqrt(dp) + mean, "--", label="+1σ")
    plt.plot(t, yc - np.sqrt(dp) + mean, "--",label="-1σ")

    plt.legend(loc="lower left")
    plt.show()

