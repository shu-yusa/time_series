import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from scipy import optimize
from scipy import linalg
from scipy import fftpack
import scipy.signal as sig
from statsmodels.tsa import stattools
import pandas as pd

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
    V_min = np.copy(C[:,:,0])
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
            V_min = V
        A_prev = A
        B_prev = B
    return mar, A_min, V_min, AIC_min

def cross_spectrum(C, fa, Bf, W, maxm):
    ngrid = len(fa)
    P = []
    for i in range(ngrid):
        P.append(Bf[i] @ W @ np.conj(Bf[i].T))
    return fa, np.array(P)

def transform_A(f, A, maxm):
    shape = A.shape
    dt = np.dtype('complex128')
    Af = np.zeros([shape[1], shape[2]], dtype=dt)
    exp = []
    for m in range(maxm):
        exp.append(np.exp(-2j * np.pi * (m + 1) * f))
    for j in range(shape[1]):
        for k in range(shape[2]):
            if j == k:
               s = -1
            else:
               s = 0
            for m in range(maxm):
                s += A[m,j,k] * exp[m]
            Af[j,k] = s
    return Af

def extract_phase(P):
    phi = np.arctan(np.imag(P[:]) / np.real(P[:]))
    for i in range(len(f)):
        if np.imag(P[i]) > 0 and np.real(P[i]) < 0:
            phi[i] += np.pi
        elif np.imag(P[i]) < 0 and np.real(P[i]) < 0:
            phi[i] -= np.pi
    return phi

def calc_coherency(P):
    ngrid = P.shape[0]
    coh = []
    for i in range(ngrid):
        coh.append(coherency(P[i,:,:]))
    return np.array(coh)

def coherency(P):
    dt = np.dtype('complex128')
    shape = P.shape
    coh = np.zeros([shape[0], shape[1]], dtype=dt)
    for j in range(shape[0]):
        for k in range(shape[1]):
            amp = np.absolute(P[j,k])
            c = np.real(amp * amp / P[j,j] / P[k,k])
            coh[j,k] = c
    return coh

def calc_noise(Bf, P, V):
    ngrid = P.shape[0]
    r = []
    an = []
    shape = P.shape
    s = np.zeros(shape)
    s2 = np.zeros(shape)
    for i in range(ngrid):
        rel, ab = noise_contrib(Bf[i,:,:], P[i,:,:], V)
        r.append(rel)
        an.append(ab)
        for j in range(shape[1]):
            ss = 0
            for k in range(shape[1]):
                ss += rel[j,k]
                s[i,j,k] = ss
    return np.array(r), np.array(an), np.array(s)


def noise_contrib(B, P, V):
    shape = B.shape
    r = np.zeros([shape[0], shape[1]])
    ab = np.zeros([shape[0], shape[1]])
    for i in range(shape[0]):
        for j in range(shape[1]):
            ab[i,j] = np.real(B[i,j] * np.conj(B[i,j])) * V[j,j]
            r[i,j] =  ab[i,j] / np.real(P[i,i])
    return np.array(r), np.array(ab)

def cumulative_noise(r):
    shape = r.shape
    s = np.zeros([shape[0], shape[1]])
    for i in range(shape[0]):
        for j in range(shape[1]):
            s[i,j] = 0
            for k in range(j):
                s[i,j] += r[i,k]
    return s

if __name__ == "__main__":
    plt.figure(1)
    plt.clf()

    maxm = 25

    # 船舶データ
    data = []
    with open('senpaku.txt', encoding='utf-8') as f:
        n = 0
        for line in f.readlines():
            if (np.mod(n, 2) == 0 and n > 0) or n == 999:
                data.append(line.split())
            n += 1
    dt = np.dtype('float')
    data = np.array(data, dtype=dt)
    data = np.delete(data, [2], 1)
    print(data)
    # データ数
    N = len(data)
    shape = data.shape

    # Levinson's algorithm
    print()
    print("Yule-Walker")
    C = cross_covariance_func(data, maxm)
    mar, A_min, V_min, AIC_min = multi_ar(C, N, shape[1], maxm)
    print('Best model: m=', mar)

    # cross spectrum
    ngrid = 200
    fa = [0.5 * i / (ngrid-1) for i in range(ngrid)]
    Af = []
    Bf = []
    for i in range(ngrid):
        A = transform_A(fa[i], A_min, mar)
        Af.append(A)
        Bf.append(linalg.inv(A))
    Af = np.array(Af)
    Bf = np.array(Bf)
    f, P = cross_spectrum(C, fa, Bf, V_min, mar)
    plt.subplot(3,3,1)
    plt.plot(f, np.log(np.real(P[:,0,0])))
    plt.subplot(3,3,2)
    plt.plot(f, np.log(np.absolute(P[:,0,1])))
    plt.subplot(3,3,3)
    plt.plot(f, np.log(np.absolute(P[:,0,2])))
    plt.subplot(3,3,4)
    plt.plot(f, extract_phase(P[:,1,0]))
    plt.subplot(3,3,5)
    plt.plot(f, np.log(np.real(P[:,1,1])))
    plt.subplot(3,3,6)
    plt.plot(f, np.log(np.absolute(P[:,1,2])))
    plt.subplot(3,3,7)
    plt.plot(f, extract_phase(P[:,2,0]))
    plt.subplot(3,3,8)
    plt.plot(f, extract_phase(P[:,2,1]))
    plt.subplot(3,3,9)
    plt.plot(f, np.log(np.real(P[:,2,2])))
    plt.show()

    # coherency
    coh = calc_coherency(P)
    plt.subplot(3,3,1)
    plt.plot(f, np.log(np.real(P[:,0,0])))
    plt.subplot(3,3,2)
    plt.plot(f, np.real(coh[:,0,1]))
    plt.subplot(3,3,3)
    plt.plot(f, np.real(coh[:,0,2]))
    plt.subplot(3,3,5)
    plt.plot(f, np.log(np.real(P[:,1,1])))
    plt.subplot(3,3,6)
    plt.plot(f, np.real(coh[:,1,2]))
    plt.subplot(3,3,9)
    plt.plot(f, np.log(np.real(P[:,2,2])))
    plt.show()

    # noise
    r, an, s = calc_noise(Bf, P, V_min)
    G = gs.GridSpec(3,3)
    plt.subplot(G[0,0])
    plt.plot(fa, an[:,0,0])
    plt.plot(fa, an[:,0,1])
    plt.plot(fa, an[:,0,2])
    plt.subplot(G[0,1:3])
    plt.plot(fa, s[:,0,0] / s[:,0,2])
    plt.plot(fa, s[:,0,1] / s[:,0,2])
    plt.plot(fa, s[:,0,2] / s[:,0,2])
    plt.ylim(0,1.1)
    plt.subplot(G[1,0])
    plt.plot(fa, an[:,1,0])
    plt.subplot(G[1,1:3])
    plt.plot(fa, s[:,1,0] / s[:,1,2])
    plt.plot(fa, s[:,1,1] / s[:,1,2])
    plt.plot(fa, s[:,1,2] / s[:,1,2])
    plt.ylim(0,1.1)
    plt.subplot(G[2,0])
    plt.plot(fa, an[:,2,0])
    plt.plot(fa, an[:,2,1])
    plt.plot(fa, an[:,2,2])
    plt.subplot(G[2,1:3])
    plt.plot(fa, s[:,2,0] / s[:,2,2])
    plt.plot(fa, s[:,2,1] / s[:,2,2])
    plt.plot(fa, s[:,2,2] / s[:,2,2])
    plt.ylim(0,1.1)
    plt.show()

