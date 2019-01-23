import numpy as np
import scipy.stats as st
from scipy.special import hyp2f1, gamma
import matplotlib.pyplot as plt


class Dirichlet(object):
    def __init__(self, alpha: np.ndarray):
        self._alpha = alpha
        self._K = len(alpha)
        self._scipy_dirichlet = st.dirichlet(alpha=self._alpha)

    def pdf(self, x: np.ndarray):
        return self._scipy_dirichlet.pdf(x)

    def pdf_marginal_of_difference(self, y: float, i: int, j: int):
        """ pdf of P_j - P_i """
        a1 = self._alpha[i]
        a2 = self._alpha[j]
        a = sum(self._alpha)
        a3 = a - a1 - a2

        integral = 0
        if 0. < y and y < 1.:
            integral = (2.**(-a1)) * ((1.-y)**(-1.+a1+a3)) * (y**(-1.+a2)) * gamma(a1) * gamma(a3) * hyp2f1(a1,1.-a2,a1+a3, (-1.+y)/(2.*y)) / gamma(a1+a3)

        elif -1. < y and y <= 0.:
            integral = ((1/gamma(1-a1)) * (1-y)**(-1+a3) * (-y)**(-1+a1+a2) * gamma(1-a1-a2) * gamma(a2) * hyp2f1(a1, 1-a3, a1+a2, (2*y)/(-1+y))) + \
                       ((1/gamma(-1+a1+a2+a3)) * 2**(1-a1-a2) * (1-y)**(-2+a1+a2+a3) * gamma(-1+a1+a2)
                        * gamma(a3) * hyp2f1(1-a2, 2-a1 - a2 - a3, 2-a1-a2, (2*y)/(-1+y)))
        else:
            return 0.

        return integral * (gamma(a1 + a2 + a3) / (gamma(a1) * gamma(a2) * gamma(a3)))

    def sample_marginal_of_difference_simulated(self, i: int, j: int, n: int = 1000) -> np.ndarray:
        """ Sample p_j - p_i"""
        a1 = self._alpha[i]
        a2 = self._alpha[j]
        a3 = sum(self._alpha) - a1 - a2
        a = a1 + a2 + a3

        gamma = [(np.random.gamma(a1), np.random.gamma(a2), np.random.gamma(a3)) for i in range(n)]

        sample = [(g[1] - g[0])/sum(g) for g in gamma]

        return sample


if __name__ == "__main__":
    alpha = np.array([.5, 2.1, 5.2])

    dist = Dirichlet(alpha)
    i,j = 0,1 # indices of the difference

    x = np.linspace(-1, 1, 100)
    pdf_x = np.array([dist.pdf_marginal_of_difference(xi, i, j) for xi in x])
    sample = dist.sample_marginal_of_difference_simulated(i,j, 100000)

    plt.plot(x, pdf_x)
    plt.hist(sample, color="silver", density=True, bins = 100, range=[-1,1])
    plt.show()

