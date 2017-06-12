import numpy as np

def levinson(acovf, N, maxm):
    """
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
    # calculate AR model upto maxm
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

