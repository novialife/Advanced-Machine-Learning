import numpy as np
from tqdm import tqdm
class UGMM:
    def __init__(self, X) -> None:
        self.X = X
        self.N = X.shape[0]
        self.xbar = X.mean()
        self.ELBO = [1, 2]
        self.sum_x = self.X.sum()
        self.sum_x_squared = (self.X**2).sum()
    
    def initialize(self):
        self.mu_0 = 0
        self.lambda_0 = 0
        self.a_0 = (self.N + 1) / 2 
        self.b_0 = 0

        self.a_n = self.a_0 + (self.N+1) / 2
        self.mu_n = (self.lambda_0*self.mu_0 + self.N*self.xbar)/(self.lambda_0 + self.N)
        self.lambda_n = np.random.rand(1)

        self.mu = (self.N*self.xbar + self.mu_0)/(self.N+1)

    def fit(self, max_iter, tol=1e-15):
        self.initialize()
        i = 0
        while i < max_iter and abs(self.ELBO[-1] - self.ELBO[-2]) > tol:
            self.b_n = self.b_0 + 0.5*((self.lambda_0 + self.N)*(self.lambda_n**-1 + self.mu_n**2) -2*((self.lambda_0*self.mu_0) + self.sum_x) + self.sum_x_squared + self.lambda_0*self.mu_0**2)
            self.lambda_n = (self.lambda_0 + self.N) * (self.a_n/self.b_n)
            self.ELBO.append(self.compute_elbo())
            i += 1
        # print(f"Converged at iteration {i} with following parameters:")
        # print(f"mu: {self.mu_n}")
        # print(f"lambda: {self.lambda_n}")
        # print(f"a: {self.a_n}")
        # print(f"b: {self.b_n}")
    
    def compute_q_mu(self):
        return -0.5 * self.lambda_n * (np.sum((self.X - self.mu)**2) + self.lambda_0*(self.mu - self.mu_0)**2)

    def compute_q_tau(self):
        return (self.a_0 - 1) * np.log(self.lambda_n) + self.N*np.log(self.lambda_n)*0.5 - self.lambda_n*self.b_n

    def compute_elbo(self):
        x_mu_sq = np.sum((self.X-self.mu)**2)
        cov_term = -0.5 * (1/self.lambda_n)
        logp_x = cov_term * x_mu_sq

        logp_mu = cov_term * np.sum((self.X-self.mu_0)**2)
        logp_sigma = (-self.a_0 -1) * np.log(self.lambda_n) - (self.b_0/self.lambda_n)
        
        logq_mu = self.compute_q_mu()
        logq_tau = self.compute_q_tau()
        
        return logp_x + logp_mu + logp_sigma - logq_mu - logq_tau
    