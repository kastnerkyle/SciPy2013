#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plot
import math


N = 10
noise_var = B = 5
xs = np.matrix(range(N)).T
ys = np.square(xs) - 4*xs + 1
wm = ys + np.sqrt(B)*np.random.randn(N,1)

def gen_dft(m, n, N):
    return np.exp(1j*-2*m*n/N)

def gen_polynomial(x, m):
    return np.power(x, m)

N_basis = 3
#basis_func = np.vectorize(gen_dft)
#basis = basis_func(xs, np.arange(N_basis), N).T
basis_func = np.vectorize(gen_polynomial)
basis = basis_func(xs, np.arange(N_basis)).T
test_data = t = basis*wm
#Calculate the Moore-Penrose pseudoinverse using the following formula
#maximum_likelihood = wml = np.linalg.inv(basis.T*basis)*basis.T*t
#Direct calculation appears to have numerical instability issues...
#Luckily the pinv method calculates Moore-Penrose pseudo inverse using SVD, which largely avoids the numerical issues
maximum_likelihood = wml = np.linalg.pinv(basis)*t

plot.figure()
plot.title("Regression fit using polynomial basis function, number of basis functions = $" + `N_basis` + "$")
plot.plot(ys, 'b')
plot.plot(wm, 'ro')
plot.plot(np.real(wml), 'g')
plot.show()
