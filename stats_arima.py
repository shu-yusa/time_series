import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels
import statsmodels.api as sm
from statsmodels.graphics.api import qqplot
from statsmodels.tsa import stattools
from statsmodels.tsa.arima_process import arma_generate_sample, ArmaProcess
from timeseries import ar
from timeseries import arma

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

# print(sm.datasets.sunspots.NOTE);

# pandasのデータフレームに読み込み
dta = sm.datasets.sunspots.load_pandas().data
dta.index = pd.Index(sm.tsa.datetools.dates_from_range('1700', '2008'))
del dta["YEAR"]

# ファイルから読み込む場合
dta = pd.read_csv('sunspot.txt') # 1750~1979のデータ
dta.columns = ['SUNACTIVITY']
dta.index = pd.to_datetime([str(y) + '-12-31' for y in range(1750, 1980)])

#dta.plot(figsize=(12,8))

## fig = plt.figure(figsize=(12,8))
## ax1 = fig.add_subplot(211)
## # autocorrelation function
## # shaded area represents confidence interval
## fig = sm.graphics.tsa.plot_acf(dta.values.squeeze(), lags=40, ax=ax1, alpha=.05)
## ax2 = fig.add_subplot(212)
## # partial autocorrelation function (Yule-Walker for default)
## fig = sm.graphics.tsa.plot_pacf(dta, lags=40, ax=ax2)

mean = dta.mean()
dta = dta - mean
# arma_mod20 = sm.tsa.ARMA(dta, (9,0)).fit(disp=False)
#arma_mod20 = sm.tsa.ARMA(dta, (9,0)).fit(disp=False)
arma_mod20 = statsmodels.tsa.ar_model.AR(dta).fit(maxlag=15, ic='aic', disp=False)
arc = np.array(arma_mod20.params)[1:]

N = len(dta.index)
data = dta.squeeze()
acovf = stattools.acovf(data)
mar, arc_min, sig2_min, AIC_min = ar.levinson(acovf, N, 10)
print(arc_min)

# モデルで計算
y1 = calc_time_series(data, arc, N, 9)
y2 = calc_time_series(data, arc_min, N, 9)

mar = sm.tsa.AR(dta).select_order(15, 'aic')
arma_mod30 = sm.tsa.ARMA(dta, (mar,0)).fit(disp=False)
print(arma_mod30.params)
print(arma_mod30.aic, arma_mod30.bic, arma_mod30.hqic)
# こっちを使うとモデル選択まで行われる(AICが他の方法と大きく異なる？)
# arma_mod30 = sm.tsa.AR(dta).fit(maxlag=15, ic='aic', disp=False)
# print(arma_mod30.params)
# print(arma_mod30.aic, arma_mod30.bic, arma_mod30.hqic)

# check if our model obeys the theory
resid = arma_mod30.resid # residual
sm.stats.durbin_watson(resid.values)
fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)
ax = arma_mod30.resid.plot(ax=ax)

# test if the residual obeys the normal distribution
print(stats.normaltest(resid))
fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)
fig = qqplot(resid, line='q', ax=ax, fit=True)

# autocorrelation function and PARCOR of residual
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(resid.values.squeeze(), lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(resid, lags=40, ax=ax2)

r,q,p = sm.tsa.acf(resid.values.squeeze(), qstat=True)
data = np.c_[range(1,41), r[1:], q, p]
table = pd.DataFrame(data, columns=['lag', "AC", "Q", "Prob(>Q)"])
print(table.set_index('lag'))

# prediction
predict_sunspots = arma_mod30.predict('1979', '2012', dynamic=True)
print(predict_sunspots)
fig, ax = plt.subplots(figsize=(12, 8))
ax = dta.ix['1950':].plot(ax=ax)
fig = arma_mod30.plot_predict('1979', '2012', dynamic=True, ax=ax, plot_insample=False)


def mean_forecast_err(y, yhat):
    return y.sub(yhat).mean()
print(mean_forecast_err(dta.SUNACTIVITY, predict_sunspots))

# Simulated ARMA(4,1): Model Identification is Difficult
np.random.seed(1234)
# include zero-th lag
arparams = np.array([1, 0.75, -0.65, -0.55, 0.9])
maparams = np.array([1, 0.65])

arma_t = ArmaProcess(arparams, maparams)
print(arma_t.isinvertible)
print(arma_t.isstationary)

fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)
ax.plot(arma_t.generate_sample(nsample=50));

arparams = np.array([1, 0.35, -0.15, 0.55, 0.1])
maparams = np.array([1, 0.65])
arma_t = ArmaProcess(arparams, maparams)
print(arma_t.isstationary)

# For mixed ARMA processes the Autocorrelation function is a mixture of exponentials and damped sine waves after (q-p) lags.
# The partial autocorrelation function is a mixture of exponentials and dampened sine waves after (p-q) lags.
arma_rvs = arma_t.generate_sample(nsample=500, burnin=250, scale=2.5)
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(arma_rvs, lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(arma_rvs, lags=40, ax=ax2)

arma11 = sm.tsa.ARMA(arma_rvs, (1,1)).fit(disp=False)
resid = arma11.resid
r, q, p = sm.tsa.acf(resid, qstat=True)
data = np.c_[range(1, 41), r[1:], q, p]
table = pd.DataFrame(data, columns=['lag', "AC", "Q", "Prob(>Q)"])
print(table.set_index('lag'))

arma41 = sm.tsa.ARMA(arma_rvs, (4,1)).fit(disp=False)
resid = arma41.resid
r, q, p = sm.tsa.acf(resid, qstat=True)
data = np.c_[range(1, 41), r[1:], q, p]
table = pd.DataFrame(data, columns=['lag', "AC", "Q", "Prob(>Q)"])
print(table.set_index('lag'))

macrodta = sm.datasets.macrodata.load_pandas().data
macrodta.index = pd.Index(sm.tsa.datetools.dates_from_range('1959Q1', '2009Q3'))
cpi = macrodta["cpi"]
print(sm.tsa.adfuller(cpi)[1])

fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)
ax = cpi.plot(ax=ax);
ax.legend();

# statsmodelsの計算の自分のAR実装結果を比較
t = range(N)
plt.figure(1)
fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)
ax.plot(t, y1, label="STATSMODELS")
ax.plot(t, y2, label="my AR")
ax.plot(t, dta, 'o',label="Data")

plt.legend()
plt.show()
