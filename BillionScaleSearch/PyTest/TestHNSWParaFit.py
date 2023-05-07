from calendar import c
import numpy as np
from scipy import optimize as op
import matplotlib.pyplot as plt

def HNSWFunc(x, a, b, c):
    return a * np.log(x + b) + c

def VectorFunc(x, a, b):
    return a * x + b

#x1 = np.array([1000000,  500000, 100000, 50000, 20000, 10000,  5000,  1000])
#y1 = np.array([0.038,    0.034,   0.023, 0.020, 0.018, 0.0170, 0.016, 0.0098])

#x1 = np.array([1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000, 256000, 512000])
#y1 = np.array([0.020, 0.025, 0.033, 0.037, 0.048, 0.067, 0.096, 0.136, 0.169, 0.196])

#x1 = np.array([13000, 12000, 11000, 10000, 9000,  8000,  7000,    3000, 500,   100])
#y1 = np.array([0.061, 0.060, 0.057, 0.054, 0.049, 0.043, 0.036,  0.033, 0.028, 0.014])

x1 = np.array([20000, 13000,   10000,  5000,    1000, 500,   100])
y1 = np.array([0.063, 0.061,  0.054, 0.039,  0.034, 0.028, 0.014])

popt, pcov = op.curve_fit(HNSWFunc, x1, y1, maxfev=500000)

plt.plot(x1, y1, 'b-', label='data')
plt.plot(x1, HNSWFunc(x1, *popt), 'r-',
         label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

print(popt)
# [ 5.97279862e-03  1.38686367e+04 -4.49570939e-02]
# [ 5.60344181e-02  2.05814112e+04 -5.37725689e-01]
# [ 1.03241406e-02  1.20983087e+02 -4.07452820e-02]

#x2 = np.array([25000, 20000, 15000, 10000, 5000])
#y2 = np.array([0.256, 0.207, 0.160, 0.110, 0.061])

#x2 = np.array([1038, 2067, 4077, 7950, 15443])
#y2 = np.array([0.032, 0.063, 0.1212, 0.2787, 0.4680])

x2 = np.array([678,   9641,  8202,  100539, 80729, 104627, 16612, 101021, 2446,  26960, 1038,  15443])
y2 = np.array([0.026, 0.317, 0.175, 2.037,  1.335, 1.433,  0.322, 1.917,  0.059, 0.632, 0.034, 0.414])

popt, pcov = op.curve_fit(VectorFunc, x2, y2)

plt.scatter(x2, y2,  label='data')
plt.plot(x2, VectorFunc(x2, *popt), 'r-', label='fit: a=%f, b=%f' % tuple(popt))
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

print(popt)
# [9.74000000e-06 1.27000001e-02]
# [3.08163483e-05 4.13802999e-03]
#[1.6731014e-05 7.2663021e-02]

# The whole function: T = 5.97279862e-03 * np.log(x1 + 1.38686367e+04) -4.49570939e-02 + 9.74e-06 * x2 + 0.0127
# alpha = 5.97279862e-03 / 9.74e-06 = 613
# beta = 1.38686367e+04
# alpha = 5.60344181e-02 / 3.08163483e-05 = 1818.3
# beta = 2.05814112e+04

#alpha = 1.03241406e-02 / 1.6731014e-05 = 617
#beta = 121
# Confirm: 
# a1 = 
