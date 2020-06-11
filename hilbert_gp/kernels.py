import numpy as np

def se_psd(chi, sigma, gamma):
    """
    Power spectral density 
    of squared exponential density
    using Angular freq
    """
    s = np.exp(-chi**2 /(4 * gamma))
    
    return s * sigma**2 * np.sqrt(np.pi / gamma)

def se_kernel(x1, x2, sigma, gamma):
    """
    Squared exponential kernel
    """
    tau = np.subtract.outer(x1, x2)**2
    return sigma**2 * np.exp(-gamma * tau )


def sm_kernel(x1, x2, w, gamma, mu):
    """
    Spectral mixture kernel
    """
    assert (len(w) == len(gamma) == len(mu))

    tau = np.subtract.outer(x1, x2)
    gram = np.zeros_like(tau)
    Q = len(w)
    for q in range(Q):
        gram += w[q] * np.exp(-gamma[q] * tau**2 / 2) * np.cos(mu[q] * tau)
    return gram

def sm_psd(xi, w, gamma, mu):
    Q = len(w)
    psd = np.zeros_like(xi)
    gamma = np.maximum(gamma, np.ones(Q)*1e-20)

    for q in range(Q):
        psd_1 = w[q] * np.exp(-(xi - mu[q])**2 / (2 * gamma[q]))
        psd_2 = w[q] * np.exp(-(-xi - mu[q])**2 / (2 * gamma[q]))
        psd += (psd_1 + psd_2) * 0.5 * np.sqrt(2 * np.pi) / np.sqrt(gamma[q])
    return psd