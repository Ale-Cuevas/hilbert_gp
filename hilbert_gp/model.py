import numpy as np
import time
from scipy.optimize import minimize as fmin
from scipy.stats import norm

from .kernels import se_kernel, se_psd, sm_kernel, sm_psd

class HilbertGP():
    def __init__(self, x_train=None, y_train=None, kernel='SE', L=1, m=10, kern_params=None, sigma_noise=0, Q=1, tresh=1e-50):
        self.x_train = x_train
        self.y_train = y_train
        self.kernel_name = kernel
        self.L = L
        self.m = np.arange(1, m + 1)
        self.kernel_params = kern_params
        self.sigma_n = sigma_noise
        self.tresh = 1e-10
        self.opt_res = None

        if self.kernel_name=='SE':
            self.kernel = se_kernel
            self.psd = se_psd
        elif self.kernel_name=='SM':
            self.kernel = sm_kernel
            self.psd = sm_psd
            self.Q = Q
        else:
            raise Exception("Not a valid kernel, only valid options 'SE' and 'SM'.")

    def set_train_set(self, x_train, y_train):
        """
        Sets training set
        """
        assert isinstance(x_train, np.ndarray)
        assert isinstance(y_train, np.ndarray)

        self.x_train = x_train
        self.y_train = y_train

    def set_kernel_parameters(self, kernel_params):
        assert isinstance(kernel_params, dict)

        self.kernel_params = kernel_params

    def _check_train_set(self):
        if (self.x_train is None) | (self.y_train is None):
            raise Exception("No training set is defined, use 'set_trainset()' method.")

    def _check_kernel_params(self):
        if self.kernel_params is None:
            raise Exception("No kernel parameters specified, use 'set_kernel_parameters()' method.")

    @property
    def eigen_values(self):
        """
        Return Laplace eigenvalues
        Eq.56
        """
        return (np.pi * self.m / (2 * self.L))**2

    def eigen_functions(self, x):
        """
        Evaluates Laplace eigen functions
        Eq.56

        Returns matrix of len(x) x m
        """
        if isinstance(x, list):
            x = np.array(x)
        val = np.sin(np.pi * np.outer((x + self.L), self.m) / (2 * self.L))
        return val / np.sqrt(self.L)

    def kernel_approx(self, x1, x2):
        """
        Evaluates kernel approximation
        Eq.20
        """
        gram = np.zeros((len(x1), len(x2)))

        diag = self._diag_lambda()

        eig_fun_1 = self.eigen_functions(x1)
        eig_fun_2 = self.eigen_functions(x2)

        for j in range(len(self.m)):
            gram += np.outer(eig_fun_1[:, j], eig_fun_2[:, j]) * diag[j]
        return gram

    def _diag_lambda(self):
        """
        PSD evaluated in the square root of eigen values
        (Capital Lambda in Eq.37)
        """
        self._check_kernel_params()
        
        diag = self.psd(np.sqrt(self.eigen_values), **self.kernel_params)
        
        return np.maximum(diag, np.ones_like(diag)*self.tresh)

    def _compute_nll(self, theta=None):
        """
        Negative log likelihood of the model
        Eq.41
        """
        if theta is not None:
            theta = np.exp(theta)
            sigma_n = theta[-1]
            self.sigma_n = sigma_n
            if self.kernel_name == 'SE':
                sigma, gamma = theta[:-1]
                self.kernel_params = {'sigma':sigma, 'gamma':gamma}
            elif self.kernel_name == 'SM':
                w = theta[:self.Q]
                gamma = theta[self.Q:2*self.Q]
                mu = theta[2*self.Q:3*self.Q]
                self.kernel_params = {'mu':mu, 'gamma':gamma, 'w':w}   

        matphi = self.eigen_functions(self.x_train)
        diag = self._diag_lambda()

        n = len(self.x_train)
        
        Z = self.sigma_n**2 * np.diag(1/diag) + matphi.T @ matphi
        invZ = np.linalg.inv(Z)
        logdetQ = (n - len(self.m) ) * np.log(self.sigma_n**2)  + np.prod(np.linalg.slogdet(Z)) + np.log(diag).sum()
        aux = ((self.y_train**2).sum() - self.y_train.T @ matphi @ invZ @ matphi.T @ self.y_train) / self.sigma_n**2
        
        return 0.5 * (logdetQ + aux + n/2 * np.log(2 * np.pi))

    def nll(self):
        """
        Evaluates the Negative Log Likelihood of the model
        """
        self._check_train_set()
        self._check_kernel_params()
        
        return self._compute_nll()
        
    def posterior(self, x_star):
        """
        Evaluates the posterior at x_star
        """
        self._check_train_set()
        self._check_kernel_params()

        matphi = self.eigen_functions(self.x_train)
        phi_star = self.eigen_functions(x_star)
        
        # check that is not zero
        diag = self._diag_lambda()
        aux = np.linalg.inv(matphi.T @ matphi + self.sigma_n**2 * np.diag(1/diag))

        mu_post = phi_star @ aux @ matphi.T @ self.y_train
        cov_post = self.sigma_n**2 * phi_star @ aux @ phi_star.T
        
        return mu_post, np.diag(cov_post)

    def fit(self, method='L-BFGS-B', kern_params=None, niter=500, tol=1e-20, bounds=None):
        """
        Fit model to data

        INCOMPLETE
        """
        if kern_params is not None:
            self.set_kernel_parameters(kern_params)

        # check dataset and kernel params
        self._check_train_set()
        self._check_kernel_params()

        print('Initial NLL:', self.nll())
        time_ini = time.time()

        dic = self.kernel_params
        if self.kernel_name == 'SE':
            theta = np.r_[dic['sigma'], dic['gamma'], self.sigma_n]
        elif self.kernel_name == 'SM':
            theta = np.r_[dic['w'], dic['gamma'], dic['mu'], self.sigma_n]

        theta = np.log(theta)

        self.opt_res = fmin(
        self._compute_nll,
        theta,
        method=method,
        options={'disp':1, 'maxiter':niter},
        tol=tol,
        bounds=bounds)

        # print final NLL, and assign optimal parameters to kernel
        print('Trained NLL:', self._compute_nll(self.opt_res.x))
        print('Model trained in {} secs'.format(time.time() - time_ini))

        return self.opt_res