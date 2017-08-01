import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Quarterly SA West German macro data, Bil DM, from Lutkepohl 1993 Table E.1)
dta = sm.datasets.webuse('lutkepohl2', 'http://www.stata-press.com/data/r12/')
# dta.qtr ... 日付データ
# dta.inv ... 投資
# dta.inc ... 所得
# dta.consump ... 消費
# dta.ln_inv ... ln(dta.inv)
# dta.ln_inc ... ln(dta.inc)
# dta.dln_inv ... ln(dta.inc(t)) - ln(dta.inc(t-1))
dta.index = dta.qtr # 日付データ(quater)
#print(dta)
endog = dta.ix['1960-04-01':'1978-10-01', ['dln_inv', 'dln_inc', 'dln_consump']]
print(endog)

# VAR(2)
exog = endog['dln_consump']
mod = sm.tsa.VARMAX(endog[['dln_inv', 'dln_inc']], order=(2,0), trend='nc', exog=exog)
res = mod.fit(maxiter=1000, disp=False)
print(res.summary())
# impulse response function
# ax = res.impulse_responses(10, orthogonalized=True).plot(figsize=(13,3))
# ax.set(xlabel='t', title='Responses to shock to `dln_inv`')

'''
# VMA(2)
mod = sm.tsa.VARMAX(endog[['dln_inv', 'dln_inc']], order=(0,2), error_cov_type='diagonal', exog=exog)
res = mod.fit(maxiter=1000, disp=False)
print(res.summary())
ax = res.impulse_responses(10, orthogonalized=True).plot(figsize=(13,3))
ax.set(xlabel='t', title='Responses to shock to `dln_inv`')

# VARMA(2,2), warining arise from the identification issue for VARMA
mod = sm.tsa.VARMAX(endog[['dln_inv', 'dln_inc']], order=(2,2), exog=exog)
res = mod.fit(maxiter=1000, disp=False)
print(res.summary())
ax = res.impulse_responses(10, orthogonalized=True).plot(figsize=(13,3))
ax.set(xlabel='t', title='Responses to shock to `dln_inv`')
'''

'''
t = range(len(dta.index))
plt.figure(1)
fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)
ax.plot(dta.index, dta.inc)
ax.plot(dta.index, dta.inv)
ax.plot(dta.index, dta.consump)

'''
plt.legend()
plt.show()
