import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import optimize
from scipy import linalg
import scipy.signal as sig
from statsmodels.tsa import stattools
import pandas as pd

def calc_spectrum(ngrid, arc, sig2):
    """
    calculate spectrum
    @param ngrid the number of points to calculate spectrum
    @param arc auto regressive coefficient
    """
    # スペクトル
    denom = 1
    logp = []
    for i in range(ngrid):
        f = 0.5 * i / (ngrid - 1)
        denom = 1
        for j in range(1, mar):
            denom -= arc[j-1] * np.exp(-2j * np.pi * j * f)
        logp.append(np.log10(sig2) - 2 * np.log10(np.absolute(denom)))
    t = [0.5 * i / (ngrid-1) for i in range(ngrid)]
    return t, logp

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

def cross_cov(data1, mean1, data2, mean2, lag):
    """
    標本相互共分散関数を計算する.
    """
    C = 0
    N = len(data1)
    cor = []
    for k in range(lag):
        C = 0
        for n in range(k, N):
            C += (data1[n] - mean1) * (data2[n-k] - mean2)
        cor.append(C/N)
    cor = np.array(cor)
    return cor

def cross_covariance_func2(data, lag):
    N = len(data[:,1])
    shape = data.shape
    C = []
    means = np.zeros(shape[1])
    for i in range(shape[1]):
        means[i] = np.mean(data[:,i])
    for i in range(shape[1]):
        row = []
        for j in range(shape[1]):
            cc = cross_cov(data[:,i], means[i], data[:,j], means[j], lag)
            row.append(cc)
        C.append(row)
    C = np.array(C)
    return C

def cross_covariance_func(data, lag):
    N = len(data[:,1])
    shape = data.shape
    means = np.zeros(shape[1])
    for i in range(shape[1]):
        means[i] = np.mean(data[:,i])
    C = np.zeros((shape[1], shape[1], N))
    for i in range(shape[1]):
        for j in range(shape[1]):
            for k in range(lag+1):
                cc = 0
                for n in range(k, N):
                   cc += (data[n,i] - means[i]) * (data[n-k,j] - means[j])
                C[i,j,k] = cc / N
    return C

def multi_ar(C, N, k, maxm):
    V = np.copy(C[:,:,0])
    U = np.copy(C[:,:,0])
    const = N * (k * np.log(2 * np.pi) + k) + k * (k + 1)
    A_prev = []

    AIC_min = const + N * np.log(linalg.det(V))
    A_min = []
    mar = 0
    print("N=",N, "k=", k)
    print("m=", 0, "AIC=", AIC_min)

    for m in range(1, maxm):
        W = np.copy(C[:,:,m])
        for i in range(1, m):
            W -= A_prev[i-1] @ C[:,:,m-i]
        A = np.zeros((m, k, k))
        B = np.zeros((m, k, k))
        A[m-1] = W @ linalg.inv(U)
        B[m-1] = np.transpose(W) @ linalg.inv(V)
        for i in range(1, m):
            A[i-1] = A_prev[i-1] - A[m-1] @ B_prev[m-i-1]
            B[i-1] = B_prev[i-1] - B[m-1] @ A_prev[m-i-1]
        V = np.copy(C[:,:,0])
        U = np.copy(C[:,:,0])
        for i in range(1, m+1):
            V -= A[i-1] @ np.transpose(C[:,:,i])
            U -= B[i-1] @ C[:,:,i]
        AIC = const + N * np.log(linalg.det(V)) + 2 * k * k * m
        print("m=", m, "AIC=", AIC)
        if AIC < AIC_min:
            mar = m
            AIC_min = AIC
            A_min = A
        A_prev = A
        B_prev = B
    return mar, A_min, AIC_min

if __name__ == "__main__":
    plt.figure(1)
    plt.clf()

    maxm = 25

    # 船舶データ
    data = []
    with open('senpaku.txt', encoding='utf-8') as f:
        n = 0
        for line in f.readlines():
            if np.mod(n, 2) == 0:
                data.append(line.split())
            n += 1
    dt = np.dtype('float')
    data = np.array(data, dtype=dt)
    data = np.delete(data, [1], 1)
    # データ数
    N = len(data)
    shape = data.shape

    # Levinson's algorithm
    print()
    print("Yule-Walker")
    C = cross_covariance_func(data, maxm)
    mar, arc_min, AIC_min = multi_ar(C, N, shape[1], maxm)
    print('Best model: m=', mar)
    # スペクトル
    # t, logp2 = calc_spectrum(400, arc_min, sig2_min)
    # y_pre2 = calc_time_series(data, arc_min, N, mar)

    # # プロット
    # plt.subplot(2,1,1)
    # plt.plot(t, logp2, label="Yule-Walker")
    # plt.plot(t, logp3, label="Least Square")
    # plt.plot(t, logp4, label="PARCOR")
    # plt.legend()

    # # 時系列計算
    # t = range(N)
    # plt.subplot(2,1,2)
    # plt.plot(t, y_pre2, label="Yule-Walker")
    # plt.plot(t, y_pre3, label="Least Square")
    # plt.plot(t, y_pre4, label="PARCOR")
    # plt.plot(t, data, label="Data")
    # plt.legend(loc="best")

    # plt.show()

