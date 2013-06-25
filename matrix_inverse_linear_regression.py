#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plot


N = 100
upper_lim = 10 #Upper bound for data, where points will be random from 0 to upper_lim
noise_var = B = 4 #Add Gaussian noise
xs = np.sort(upper_lim*np.random.rand(N,1), axis=0)
ys = np.square(xs) - 4*xs + 1
wm = ys + B*np.random.randn(N,1)

def gen_dft(m, n, N):
    return np.exp(1j*-2*m*n/N)

def gen_polynomial(x, m):
    return np.power(x, m)

N_basis = 3
basis_func = np.vectorize(gen_polynomial)
basis = basis_func(np.arange(N)[:,np.newaxis], np.arange(N_basis)).T

#Uncomment these lines lines to use Fourier basis instead
#basis_func = np.vectorize(gen_dft)
#basis = basis_func(np.arange(N)[:,np.newaxis], np.arange(N_basis),N).T

test_data = t = np.dot(basis,wm)

#Calculate the Moore-Penrose pseudoinverse using the following formula
#maximum_likelihood = wml = np.linalg.inv(basis.T*basis)*basis.T*t
#Direct calculation appears to have numerical instability issues...
#Luckily the pinv method calculates Moore-Penrose pseudo inverse using SVD, which largely avoids the numerical issues
maximum_likelihood = wml = np.dot(np.linalg.pinv(basis),t)

plot.figure()
plot.title("Regression fit using polynomial basis function, number of basis functions = $" + `N_basis` + "$")
plot.plot(wm, 'rx')
plot.plot(np.real(wml), 'b')
plot.show()
