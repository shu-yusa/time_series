import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

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
    標本自己共分散関数ベクトルを計算する
    """
    N = len(data)
    mu = np.mean(data)
    cor = []
    for k in range(len(data)):
        cor.append(kth_auto_cor(data, k, N, mu))
    return cor / cor[0]

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
    print(mat.shape)
    cor = auto_cor(mat[:,0].astype(np.int64))
    plt.subplot(2, 3, 1)
    plot_auto_cor(cor, xlim)

    # 太陽黒点数
    with open('sunspot.txt', encoding='utf-8') as f:
        data = np.array([float(k) for k in f.readlines()])
    cor = auto_cor(data)
    plt.subplot(2, 3, 2)
    plot_auto_cor(cor, xlim)

    # 東京の最高気温
    with open('temperature.txt', encoding='utf-8') as f:
        data = np.array([float(k) for k in f.readlines()])
    cor = auto_cor(data)
    plt.subplot(2, 3, 3)
    plot_auto_cor(cor, xlim)

    # BLSALLFOOD
    with open('blsallfood.txt', encoding='utf-8') as f:
        data = np.array([int(k) for k in f.readlines()])
    cor = auto_cor(data)
    plt.subplot(2, 3, 4)
    plot_auto_cor(cor, xlim)

    # WHARD
    with open('whard.txt', encoding='utf-8') as f:
        data = np.array([int(k) for k in f.readlines()])
    cor = auto_cor(data)
    plt.subplot(2, 3, 5)
    plot_auto_cor(cor, xlim)

    # 地震波
    with open('seismic.txt', encoding='utf-8') as f:
        data = np.array([int(k) for k in f.readlines()])
    cor = auto_cor(data)
    plt.subplot(2, 3, 6)
    plot_auto_cor(cor, xlim)


    plt.show()
