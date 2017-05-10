import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import optimize
from scipy import linalg
import scipy.signal as sig
from statsmodels.tsa import stattools

def yule_walker(N, acovf, maxm):
    """
    @param N the number of data
    @param acovf auto covariance function
    @param m the order of the AR model
    @return mar, arc, sig, AIC auto regression coefficient, variance, AIC
    """
    const = N * (np.log(2 * np.pi) + 1)
    sig2_min = acovf[0]
    arc_min = []
    AIC_min = const + N * np.log(sig2_min) + 2 * (maxm + 1)
    mar = 0
    for m in range(1, maxm+1):
        # 自己共分散関数の行列を作成
        C = []
        for k in range(m):
            row = np.append(np.zeros(k), acovf[:m-k])
            C.append(row)
        C = np.array(C)

        # 右辺
        b = acovf[1:m+1]
        # 自己回帰係数
        arc = linalg.solve(C, b, sym_pos=True)
        # 分散
        sig2 = acovf[0] - np.dot(arc, b)
        # AIC
        AIC = const + N * np.log(sig2) + 2 * (m + 1)
        print("m=", m, "sig2=", sig2, "AIC=", AIC)
        if AIC < AIC_min:
            AIC_min = AIC
            sig2_min = sig2
            arc_min = arc
            mar = m
    return mar, arc_min, sig2_min, AIC_min

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

if __name__ == "__main__":
    plt.figure(1)
    plt.clf()

    maxm = 15

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
    # acovf = acovf * (N - 1) / N

    # 連立方程式を直接解く方法
    """
    print("Yule-Walker")
    mar, arc_min, sig2_min, AIC_min = yule_walker(N, acovf, maxm)
    print('Best model: m=', mar)
    # スペクトル
    t, logp1 = calc_spectrum(200, arc_min, sig2_min)
    """

    # Levinson's algorithm
    print()
    print("Levinson method")
    mar, arc_min, sig2_min, AIC_min = Levinson(acovf, N, maxm)
    print('Best model: m=', mar)
    # スペクトル
    t, logp2 = calc_spectrum(400, arc_min, sig2_min)
    y_pre2 = calc_time_series(data, arc_min, N, mar)

    ## least square ##
    print()
    print("Least squre method")
    mar, arc_min, sig2_min, AIC_min = ar_least_square(data, N, maxm)
    print('Best model: m=', mar)
    # スペクトル
    t, logp3 = calc_spectrum(400, arc_min, sig2_min)
    y_pre3 = calc_time_series(data, arc_min, N, mar)

    ## PARCOR method ##
    print()
    print("PARCOR method")
    method = 3
    mar, arc_min, sig2_min, AIC_min = parcor(data, N, maxm, method)
    print('Best model: m=', mar)
    # スペクトル
    t, logp4 = calc_spectrum(400, arc_min, sig2_min)
    y_pre4 = calc_time_series(data, arc_min, N, mar)

    # プロット
    plt.subplot(2,1,1)
    plt.plot(t, logp2, label="Yule-Walker")
    plt.plot(t, logp3, label="Least Square")
    plt.plot(t, logp4, label="PARCOR")
    plt.legend()

    # 時系列計算
    t = range(N)
    plt.subplot(2,1,2)
    plt.plot(t, y_pre2, label="Yule-Walker")
    plt.plot(t, y_pre3, label="Least Square")
    plt.plot(t, y_pre4, label="PARCOR")
    plt.plot(t, data, label="Data")
    plt.legend(loc="best")

    plt.show()

