import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.signal as sig
from statsmodels.tsa import stattools

def plot_auto_cor(cor, xlim):
    """
    標本自己共分散関数ベクトルを表示する
    """
    t = np.arange(0, xlim, 1)
    plt.plot(t, cor[:xlim])
    plt.bar(t, cor[:xlim], width=0.2, align="center")
    plt.xlim(0, xlim)
    plt.ylim(-1, 1)


if __name__ == "__main__":
    xlim = 50

    plt.figure(1)
    plt.clf()

    # 船舶データ
    data = []
    with open('senpaku.txt', encoding='utf-8') as f:
        for line in f.readlines():
            data.append(line.split())
    mat = np.array(data)
    acf = stattools.acf(mat[:,0].astype(np.int64), nlags=xlim)
    plt.subplot(2, 3, 1)
    plot_auto_cor(acf, xlim)

    # 太陽黒点数
    with open('sunspot.txt', encoding='utf-8') as f:
        data = np.array([float(k) for k in f.readlines()])
    # acf = stattools.acf(data, nlags=xlim)
    acf = stattools.acovf(data)
    acf /= acf[0]
    plt.subplot(2, 3, 2)
    plot_auto_cor(acf, xlim)

    # 東京の最高気温
    with open('temperature.txt', encoding='utf-8') as f:
        data = np.array([float(k) for k in f.readlines()])
    acf = stattools.acf(data, nlags=xlim)
    plt.subplot(2, 3, 3)
    plot_auto_cor(acf, xlim)

    # BLSALLFOOD
    with open('blsallfood.txt', encoding='utf-8') as f:
        data = np.array([int(k) for k in f.readlines()])
    acf = stattools.acf(data, nlags=xlim)
    plt.subplot(2, 3, 4)
    plot_auto_cor(acf, xlim)

    # WHARD
    with open('whard.txt', encoding='utf-8') as f:
        data = np.array([float(k) for k in f.readlines()])
    acf = stattools.acf(data, nlags=xlim)
    plt.subplot(2, 3, 5)
    plot_auto_cor(acf, xlim)

    # 地震波
    with open('seismic.txt', encoding='utf-8') as f:
        data = np.array([int(k) for k in f.readlines()])
    acf = stattools.acf(data, nlags=xlim)
    plt.subplot(2, 3, 6)
    plot_auto_cor(acf, xlim)


    plt.show()
