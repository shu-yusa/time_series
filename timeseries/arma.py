import numpy as np
from scipy import optimize

def armafit(y, n, m, l, ini_param):
    a, b = get_default_param(m, l)
    # subtract mean value
    y -= np.mean(y)
    # transformation to ensure stationarity and invertability
    theta = np.zeros(m + l)
    parcor = parcor_from_arcoef(a, m)
    theta[:m] = np.log((1 + parcor) / (1 - parcor))
    parcor = parcor_from_arcoef(b, l)
    beta[m:] = np.log((1 + parcor) / (1 - parcor))
    return

def calc_arma_initial(data, m, arc, l, bac, sig2, acovf):
    """
    Represent ARMA model in a state space representation.
    @param data data
    @param m AR order
    @param arc Auto-regressive coefficient
    @param l MA order
    @param bac Moving-average coefficient
    @param sig2 variance
    @param acovf auto covariance function
    """
    ndata = len(data)
    k = m
    x0 = np.zeros(k)
    V0 = np.zeros((k, k))
    V0[0,0] = acovf[0]
    g = impulse(m, l, arc, bac, l)

    for i in range(1,k):
        for j in range(i,m):
            V0[0,i] += arc[j] * acovf[j+1-i]
        for j in range(i-1, l):
            V0[0,i] -= bac[j] * g[j+1-i]
    for i in range(1, k):
        for j in range(1, k):
            tmp = 0
            for p in range(i, m):
                for q in range(j, m):
                    tmp += arc[p] * arc[q] * acovf[q-j-p+i]
                for q in range(j-1, l):
                    tmp -= arc[p] * bac[q] * g[q-j-p+i]
            V0[i,j] = tmp
            tmp = 0
            for p in range(i-1, l):
                for q in range(j, m):
                   tmp -= bac[p] * arc[q] * g[p-i-q+j]
                tmp += bac[p] * bac[p+j-i] * sig2
            V0[i,j] += tmp
    return x0, V0

def make_arma_model(data, m, l, arc, bac, sig2):
    """
    Represent ARMA model in a state space representation.
    @param data data
    @param m AR order
    @param l MA order
    @param arc Auto-regressive coefficient
    @param bac Moving-average coefficient
    @param sig2 variance
    """
    ndata = len(data)
    # dimension of state x
    k = m
    # dimension of noise observation of noise w
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
    for i in range(k-1):
        G[i+1] = -bac[i]
    H = np.zeros([l,k])
    H[0,0] = 1
    Q = np.zeros((1,1))
    Q[0,0] = sig2
    R = np.zeros((l,l))
    return x, F, G, H, Q, R

def kalman_filter(x, y, F, G, H, Q, R, x0, V0, missing=[], num_missed=[]):
    """
    Kalman filter algorithm.
    @param x state vector
    @param y time series
    @param F
    @param G
    @param H
    @param Q 
    @param R
    @param x0 initial value of x
    @param V0 initial value of V
    @param missing
    @param num_missed
    """
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

def arma_likelihood(N, params):
    """
    Function for maximum likelihood for ARMA model
    """
    parcor = (np.exp(params[:m]) - 1) / (np.exp(params[:m]) + 1)
    arc = arcoef_from_parcor(parcor, m)
    parcor = (np.exp(params[m:]) - 1) / (np.exp(params[m:]) + 1)
    mac = arcoef_from_parcor(parcor, l)

def set_model_params(m, l, arc, mac):
    mlmax = max(m, l)
    H = np.zeros(m)

def impulse(m, l, a, b, maxlag):
    """
    Calculate impulse response function for ARMA model
    @param m AR order
    @param l MA order
    @param a AR coefficients
    @param b MA coefficients
    @param maxlag Maximum log of impulse response
    @return impulse response function
    """
    if maxlag > 0:
        g = np.zeros(maxlag+1)
        g[0] = 1
        for i in range(maxlag+1):
            if i < l:
               g[i] = - b[i]
            for j in range(i):
                if j < m:
                   g[i] += a[j] * g[i-j]
    else:
        g = np.zeros(1)
    return g

def get_default_param(m, l):
    """
    Get default parameters.
    @param m AR order
    @param l MA order
    @param AR coeffiecients, MA coefficients
    """
    parcor = np.zeros(max(m,l))
    for i in range(m):
        parcor[i] = - (-0.6) ** i
    arc = arcoef_from_parcor(parcor, m)
    mac = arcoef_from_parcor(parcor, l)
    return arc, mac

def parcor_from_arcoef(a, m):
    """
    Compute PARCOR from AR coefficients
    @param ar AR coefficient
    @param m AR order
    @return PARCOR
    """
    a2 = np.zeros(m)
    ap = np.zeros(m)
    parcor = np.zeros(m)
    parcor[m-1] = a[m-1]
    for mm in range(m-1)[::-1]:
        denom = 1 - parcor[mm+1] ** 2
        for i in range(mm):
            a2[i] = (ap[i] + parcor[mm+1] * ap[mm+1-i]) / denom
        parcor[mm] = a2[mm-1]
        ap = a2
    return parcor


def arcoef_from_parcor(parcor, m):
    """
    Compute AR coefficients from PARCOR.
    @param parcor PARCOR
    @param m AR order
    @return AR coefficients
    """
    if m < 1:
        return []
    a = np.zeros(m)
    ap = np.zeros(m)
    a[0] = parcor[0]
    for mm in range(1,m):
        a[mm] = parcor[mm]
        for i in range(mm-1):
            a[i] = ap[i] - parcor[mm] * ap[mm-i]
        for i in range(mm-1):
            ap[i] = a[i]
    return a


def levinson(acovf, N, maxm):
    """
    Solve AR model by Levinson's algorithm.
    @param acovf Autocovaiance function
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
    # compute AR model upto maxm
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
        # update minimum value of AIC
        if AIC < AIC_min:
            AIC_min = AIC
            sig2_min = sig2[m]
            arc_min = a[m-1,:m]
            mar = m
    return mar, arc_min, sig2_min, AIC_min

