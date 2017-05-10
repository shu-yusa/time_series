import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.signal as sig
from scipy import fftpack
from statsmodels.tsa import stattools

def kth_auto_cor(data, k, N, mean):
    """
    標本自己共分散関数のk番目の要素を計算する.
    """
    C = 0
    for n in range(k, N):
        C += (data[n] - mean) * (data[n-k] - mean)
    return C/N

def auto_cor(data):
    """
    標本自己相関関数ベクトルを計算する
    """
    N = len(data)
    mu = np.mean(data)
    cor = []
    for k in range(len(data)):
        cor.append(kth_auto_cor(data, k, N, mu))
    return cor

def plot_auto_cor(cor, xlim):
    """
    標本自己共分散関数ベクトルを表示する
    """
    t = np.arange(0, xlim, 1)
    plt.plot(t, cor[:xlim])
    plt.bar(t, cor[:xlim], width=0.2, align="center")
    plt.xlim(0, xlim)
    plt.ylim(-1, 1)

def fourier(x, L):
    N = len(x)
    w = np.pi / (L - 1)
    print("N=", N, "w=", w)
    fc = fs = []
    for i in range(1, L+1):
        cos = np.cos(w * (i - 1))
        sin = np.sin(w * (i - 1))
        t1 = 0
        t2 = 0
        for j in range(2, N+1)[::-1]:
            t0 = 2 * cos * t1 - t2 + x[j-1]
            t2 = t1
            t1 = t0
        fc.append(cos * t1 - t2 + x[0])
        fs.append(sin * t1)
    return fc, fs

def calc_periodogram(acf):
    p = []
    N = len(acf)
    freq = [j/N for j in range(1, N // 2 + 1)]
    for j in range(1, N // 2 + 1):
        cos = np.cos(2 * np.pi * freq[j-1])
        an_2 = 0
        an_1 = 1
        pj = acf[1] * cos
        for n in range(2, N):
            # Goertzel method
            an = 2 * an_1 * cos - an_2
            cosn = an * cos - an_1
            pj += acf[n] * cosn
            an_2 = an_1
            an_1 = an
        p.append(acf[0] + 2 * pj)
    return freq, p

def plot_periodogram(freq, per, N, xlim, ymin, ymax):
    """
    標本自己共分散関数ベクトルを表示する
    """
    plt.bar(freq, per[:xlim], width=0.001, align="center")
    plt.xlim(-0.01, 0.5)
    #plt.ylim(ymin, ymax)


if __name__ == "__main__":

    plt.figure(1)
    plt.clf()

    # 船舶データ
    data = []
    with open('senpaku.txt', encoding='utf-8') as f:
        for line in f.readlines():
            data.append(line.split())
    mat = np.array(data)
    N = len(data)
    acovf = stattools.acovf(mat[:,0].astype(np.int64))
    freq, per = calc_periodogram(acovf)
    plt.subplot(2, 3, 1)
    plot_periodogram(freq, np.log10(per), N, len(per), -6, 1)

    # 太陽黒点数
    with open('sunspot.txt', encoding='utf-8') as f:
        data = np.array([float(k) for k in f.readlines()])
    N = len(data)
    acovf = stattools.acovf(data)
    # freq, per = calc_periodogram(acovf)
    freq, per = sig.periodogram(data)
    per[0] = 1
    plt.subplot(2, 3, 2)
    plot_periodogram(freq, np.log10(per), N, len(per), -6, 1)

    # 東京の最高気温
    with open('temperature.txt', encoding='utf-8') as f:
        data = np.array([float(k) for k in f.readlines()])
    N = len(data)
    acovf = stattools.acovf(data)
    # per = calc_periodogram(acovf)
    freq, per = calc_periodogram(acovf)
    plt.subplot(2, 3, 3)
    plot_periodogram(freq, np.log10(per), N, len(per), -6, 1)

    # BLSALLFOOD
    with open('blsallfood.txt', encoding='utf-8') as f:
        data = np.array([int(k) for k in f.readlines()])
    N = len(data)
    acovf = stattools.acovf(data)
    # per = calc_periodogram(acovf)
    freq, per = calc_periodogram(acovf)
    plt.subplot(2, 3, 4)
    plot_periodogram(freq, np.log10(per), N, len(per), -6, 1)

    # WHARD
    with open('whard.txt', encoding='utf-8') as f:
        data = np.array([float(k) for k in f.readlines()])
    N = len(data)
    acovf = stattools.acovf(data)
    freq, per = calc_periodogram(acovf)
    plt.subplot(2, 3, 5)
    plot_periodogram(freq, np.log10(per), N, len(per), -6, 1)

    # 地震波
    with open('seismic.txt', encoding='utf-8') as f:
        data = np.array([int(k) for k in f.readlines()])
    N = len(data)
    acovf = stattools.acovf(data)
    # per = calc_periodogram(acovf)
    freq, per = sig.periodogram(data)
    per[0] = 1
    plt.subplot(2, 3, 6)
    plot_periodogram(freq, np.log10(per), N, len(per), -6, 1)

    plt.show()
