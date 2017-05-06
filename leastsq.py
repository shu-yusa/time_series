import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import optimize

def construct_matrix(data, base, m_max, m, l):
    """
    説明変数行列 + 目的変数ベクトルの行列を作る
    """
    w = 2 * np.pi / 365
    X = []
    for n in range(len(data)):
        row = [1]
        row.extend(base[n][:m])
        row.extend(base[n][m_max:m_max+l])
        row.append(data[n])
        X.append(row)
    return X

def back_subst(R, dim):
    """
    交代代入により説明変数を求める
    """
    param = np.zeros(dim)
    param[dim-1] = R[dim-1][dim] / R[dim-1][dim-1]
    for i in range(dim-1, 0, -1):
        param[i-1] = R[i-1][dim]
        for j in range(i, dim):
            param[i-1] -= R[i-1][j] * param[j]
        param[i-1] /= R[i-1][i-1]
    return param

def reg_values(param, base, ndata, m_max, m, l):
    """
    回帰結果を計算する
    """
    y = []
    w = 2 * np.pi / 365
    mm = 1 + m + l
    for n in range(ndata):
        sins = base[n][:m]
        coss = base[n][m_max:m_max+l]
        y.append(param[0] + np.dot(param[1:m+1], sins) + np.dot(param[m+1:m+1+l], coss))
    return y

def calc_base(ndata, m_max):
    """
    基底を計算する
    """
    y = []
    w = 2 * np.pi / 365
    mm = 2 * m_max + 1
    base = []
    for n in range(ndata):
        row = [np.sin(j * w * n) for j in range(1, m_max+1)]
        row.extend([np.cos(j * w * n) for j in range(1, m_max+1)])
        base.append(row)
    return base

def calc_AIC(R, dim, ndata):
    """
    AICを計算する
    """
    var = R[dim][dim] * R[dim][dim] / ndata
    return ndata * (np.log(2 * np.pi * var) + 1) + 2 * (dim + 1), var

if __name__ == "__main__":
    plt.figure(1)
    plt.clf()

    # 東京の最高気温
    with open('temperature.txt', encoding='utf-8') as f:
        data = np.array([np.float64(k) for k in f.readlines()])

    # 計算する最高次数
    m_max = 12

    # 基底の計算
    base = calc_base(len(data), m_max)

    AIC_min = 9999999999999
    m_min = 0
    l_min = 0
    reg_min = []

    for m in range(1, m_max):
        l = max(0, m - 1)
        # 説明変数 + データ行列
        X = construct_matrix(data, base, m_max, m, l)
        # QR分解
        Q, R = np.linalg.qr(X)
        # 次数
        mm = m + l + 1
        # 後退代入
        param = back_subst(R, mm)
        # 回帰結果
        y = reg_values(param, base, len(data), m_max, m, l)
        # AIC, 分散
        AIC, var = calc_AIC(R, mm, len(data))
        # 最小AICを更新
        if AIC < AIC_min:
            AIC_min = AIC
            m_min = m
            l_min = l
            reg_min = y
        print(mm, var, AIC, sep=" : ")

        # cosを1つ増やして計算
        l = m
        X = construct_matrix(data, base, m_max, m, l)
        Q, R = np.linalg.qr(X)
        mm = m + l + 1
        param = back_subst(R, mm)
        y = reg_values(param, base, len(data), m_max, m, l)
        AIC,var = calc_AIC(R, mm, len(data))
        if AIC < AIC_min:
            AIC_min = AIC
            m_min = m
            l_min = l
            reg_min = y
        print(mm, var, AIC, sep=" : ")

    # 最小AICの結果を表示
    print("minimam AIC : dim=", m_min + l_min + 1, ", AIC=", AIC_min)
    t = np.arange(0, len(data), 1)
    plt.plot(t, data)
    plt.plot(t, reg_min)

    plt.show()

