from cProfile import label
import numpy as np 
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import leastsq
import math
import random

IVFADCGP_10 =      [0.58,  0.68, 0.72, 0.75, 0.77, 0.78,  0.793,  0.81, 0.8247, 0.8296, 0.83]
IVFADCGP_time_10 =  [0.70, 0.90, 1.15, 1.38, 1.73, 1.827, 2.17,   2.66,  3.12,  3.47,   3.56]
IVFADC_10 =      [0.577,  0.67, 0.713, 0.7377, 0.752, 0.764, 0.77, 0.777, 0.78, 0.790]
IVFADC_time_10 = [0.661,  1.09, 1.35,  1.74,   2.01,  2.31,  2.50, 2.58,  3.29, 3.57]
VVVL_10 =        [0.62, 0.72, 0.78, 0.794,  0.821, 0.834, 0.857, 0.869, 0.869]
VVVL_time_10 =   [0.75, 0.94,  1.32, 1.69,  2.18,  2.89,  3.15, 3.59,   3.89]
IMI_10 =         [0.498, 0.568, 0.627, 0.684, 0.702, 0.720, 0.738, 0.752, 0.757, 0.762]
IMI_time_10 =    [0.965, 1.256, 1.648,  1.93, 2.15,  2.57,  2.89,   3.02,   3.18,  3.45]
'''
plt.figure()
plt.plot(VVVL_time_10, VVVL_10, label = "Multi-layer Model", color = "black", marker = "o", linestyle = "solid")
plt.plot(IVFADCGP_time_10, IVFADCGP_10, label = "IVFADCGP", color = "black", marker = 'v', linestyle = "dashed")
plt.plot(IVFADC_time_10, IVFADC_10, label = "IVFADC", color = "black", marker = "*", linestyle = "dashed")

plt.plot(IMI_time_10, IMI_10, label = "IMI", color = "black", marker = "x", linestyle = "dashdot")
plt.xlabel("Search Time / ms", fontsize = 16)
plt.ylabel("Recall@10", fontsize = 16)
plt.legend(fontsize = 16)
plt.show()
'''

IVFADCGP_1 =        [0.299, 0.334, 0.349, 0.357, 0.3613, 0.3655, 0.3683, 0.370, 0.372, 0.375] 
IVFADCGP_time_1 =   [0.618, 0.977, 1.24,  1.36,  1.58,    1.799,  2.18,  2.415, 2.876, 3.48] 
IVFADC_1 =         [0.2887, 0.3174, 0.33, 0.3355, 0.3389, 0.3415, 0.3429, 0.3442, 0.3452, 0.3457]
IVFADC_time_1 =     [0.733, 1.175, 1.43,  1.71,    2.00,   2.2,   2.33,   2.829,  3.122,  3.546]
VVVL_1 =            [0.296, 0.324, 0.358, 0.367, 0.372, 0.380, 0.385, 0.389, 0.396, 0.397, 0.399]
VVVL_time_1 =       [0.583, 0.745, 0.997, 1.342, 1.457, 1.589, 1.792, 1.986, 2.324, 2.518, 3.014]
IMI_1 =             [0.275, 0.289, 0.305, 0.312, 0.318, 0.324, 0.329, 0.330, 0.332]
IMI_time_1 =        [0.79,  1.37,  1.78,  1.98,  2.14,  2.38,  2.79,  2.98,  3.59]

'''
plt.figure()
plt.plot(VVVL_time_1, VVVL_1, label = "Multi-layer Model", color = "black", marker = "o", linestyle = "solid")
plt.plot(IVFADCGP_time_1, IVFADCGP_1, label = "IVFADCGP", color = "black", marker = 'v', linestyle = "dashed")
plt.plot(IVFADC_time_1, IVFADC_1, label = "IVFADC", color = "black", marker = "*", linestyle = "dashed")
plt.plot(IMI_time_1, IMI_1, label = "IMI", color = "black", marker = "x", linestyle = "dashdot")
plt.xlabel("Search Time / ms", fontsize = 16)
plt.ylabel("Recall@1", fontsize = 16)
plt.legend(fontsize = 16)
plt.show()
'''

VP_10 =         [0.52, 0.58, 0.61, 0.65, 0.682, 0.702, 0.724, 0.735, 0.746, 0.750]
VP_time_10 =    [1.25, 1.97, 2.46, 2.71, 2.93,  3.26,  3.87,  4.21,  4.56,   4.79]
VVV_10 =        [0.56, 0.62, 0.654, 0.683, 0.691, 0.713, 0.730, 0.744, 0.748, 0.750, 0.752, 0.755]
VVV_time_10 =   [0.97, 1.68, 1.97,  2.21,  2.57,  2.78,  2.98,  3.29,  3.57,  3.88,  4.14, 4.76]
VVL_10 =        [0.64, 0.67, 0.683, 0.692, 0.720, 0.731, 0.739, 0.745, 0.748, 0.751, 0.753]
VVL_time_10 =   [1.13, 1.79, 2.05,  2.19,  2.63,  2.74,  2.81,  2.84,  2.89,  3.11,  3.78]
VVP_10 =        [0.52, 0.55, 0.58, 0.595, 0.61, 0.63, 0.652, 0.667, 0.672, 0.683, 0.70, 0.715, 0.730, 0.748, 0.751, 0.752]
VVP_time_10 =   [0.768, 0.980, 1.245, 1.4, 1.85, 2.01, 2.21, 2.489, 2.679, 2.734, 2.964, 3.157, 3.423, 3.516, 3.788, 4.567]
VVVL_10 =       [0.59, 0.62, 0.66, 0.69, 0.705, 0.713, 0.728, 0.756, 0.758]
VVVL_time_10 =  [1.45, 1.68, 2.12, 2.42, 2.67,  2.78,  2.89,  3.42,  3.68]
VVVV_10 =       [0.51, 0.55, 0.59, 0.63,  0.65,  0.67, 0.683, 0.723, 0.735, 0.748, 0.753, 0.751]
VVVV_time_10 =  [0.93, 1.24, 1.79, 1.964, 2.45,  2.79, 3.03,  3.69,  3.79,  4.62,  4.79,  5.00]
'''
plt.figure()
plt.plot(VP_time_10, VP_10, label =   "V P", color = "black", marker = 'v', linestyle = "dashed")
plt.plot(VVV_time_10, VVV_10, label = "V V V", color = "black", marker = "x", linestyle = "dashdot")
plt.plot(VVL_time_10, VVL_10, label = "V V L", color = "black", marker = "o", linestyle = "solid")
plt.plot(VVP_time_10, VVP_10, label = "V V P", color = "black", marker = "D", linestyle = "dashed")
plt.plot(VVVL_time_10, VVVL_10, label = "V V V L", color = "black", marker = "s" , linestyle = "solid")
plt.plot(VVVV_time_10, VVVV_10, label = "V V V V", color = "black", marker = "*", linestyle = "dashdot")
plt.xlabel("Search Time / ms", fontsize = 16)
plt.ylabel("Recall@10", fontsize = 16)
plt.legend(fontsize = 16)
plt.show()
'''
VP_1 =         [0.18, 0.21, 0.24, 0.26, 0.273, 0.29, 0.30, 0.315, 0.319, 0.324, 0.325]
VP_time_1 =    [0.97, 1.56, 1.98, 2.43, 2.68,  2.89, 2.95, 3.49,  3.63,  3.72,  3.95]
VVV_1 =        [0.20, 0.218, 0.241, 0.256, 0.279, 0.290, 0.308, 0.319, 0.323, 0.328]
VVV_time_1 =   [1.04, 1.42,  1.76,  2.00,  2.48,  2.79,  3.04,  3.31,  3.46,  3.54]
VVL_1 =        [0.20, 0.22, 0.24, 0.254, 0.268, 0.279, 0.290, 0.301, 0.316, 0.319, 0.320]
VVL_time_1 =   [0.92, 1.34, 1.58, 1.64,  1.79,  2.08,  2.23,  2.59,  2.90,   3.08,  3.12]
VVP_1 =        [0.21, 0.24, 0.254, 0.268, 0.277, 0.285, 0.30, 0.315, 0.322, 0.328]
VVP_time_1 =   [1.87, 1.98, 2.15,  2.46,  2.74,  2.89,  2.93,  3.22,  3.46,  3.69]
VVVL_1 =       [0.19, 0.24, 0.27, 0.28,  0.294,  0.302, 0.310, 0.318, 0.321, 0.325, 0.327]
VVVL_time_1 =  [0.84, 1.21, 1.48, 1.52,  1.78,   1.99,  2.14,  2.47,  2.79,  2.88,  2.96]
VVVV_1 =       [0.16, 0.18, 0.22, 0.26, 0.28, 0.29, 0.307, 0.318, 0.320, 0.322, 0.3256]
VVVV_time_1 =  [0.83, 0.92, 1.24, 1.92, 2.24, 2.68,  2.79,  3.04,  3.18,  3.30,   3.46]
'''
plt.figure()
plt.plot(VP_time_1, VP_1, label =   "V P", color = "black", marker = 'v', linestyle = "dashed")
plt.plot(VVV_time_1, VVV_1, label = "V V V", color = "black", marker = "x", linestyle = "dashdot")
plt.plot(VVL_time_1, VVL_1, label = "V V L", color = "black", marker = "o", linestyle = "solid")
plt.plot(VVP_time_1, VVP_1, label = "V V P", color = "black", marker = "D", linestyle = "dashed")
plt.plot(VVVL_time_1, VVVL_1, label = "V V V L", color = "black", marker = "s" , linestyle = "solid")
plt.plot(VVVV_time_1, VVVV_1, label = "V V V V", color = "black", marker = "*", linestyle = "dashdot")
plt.xlabel("Search Time / ms", fontsize = 16)
plt.ylabel("Recall@1", fontsize = 16)
plt.legend(fontsize = 16)
plt.show()
'''

VVVL0_10 =      [ 0.76, 0.79, 0.806, 0.83, 0.840, 0.845, 0.849, 0.852, 0.855, 0.854]
VVVL0_time_10 = [ 1.39, 1.72,  2.08, 2.95, 3.09,  3.28,  3.51,  3.62,  3.86, 4.98]

VVVL_10 =        [0.78, 0.794,  0.821, 0.842, 0.857, 0.867, 0.869, 0.870]
VVVL_time_10 =   [1.32, 1.69,  2.18,  2.89,  3.15, 3.59,   3.89,   4.59]
'''
plt.figure()
plt.plot(VVVL_time_10, VVVL_10, label = "V V V L optimized", color = "black", marker = "o", linestyle = "solid")
plt.plot(VVVL0_time_10, VVVL0_10, label = "V V V L", color = "black", marker = "x", linestyle = "dashed")
plt.xlabel("Search Time/ms", fontsize= 22)
plt.ylabel("Recall@10", fontsize = 22)
plt.legend(fontsize = 20)
plt.show()
'''
VVVL0_1 =       [0.267, 0.296, 0.319, 0.336, 0.360, 0.372, 0.379, 0.381, 0.382, 0.383]
VVVL0_time_1 =  [0.78,  0.92,  1.09,  1.24,  1.45,  1.69,  1.76,  1.99,  2.11,  2.69]

VVVL_1 =            [0.296, 0.324, 0.358, 0.367, 0.372, 0.380, 0.385, 0.389, 0.396, 0.397, 0.399]
VVVL_time_1 =       [0.583, 0.745, 0.997, 1.342, 1.457, 1.589, 1.792, 1.986, 2.324, 2.518, 3.014]
'''
plt.figure()
plt.plot(VVVL_time_1, VVVL_1, label = "V V V L optimized", color = "black", marker = "o", linestyle = "solid")
plt.plot(VVVL0_time_1, VVVL0_1, label = "V V V L", color = "black", marker = "x", linestyle = "dashed")
plt.xlabel("Search Time/ms", fontsize = 22)
plt.ylabel("Recall@1", fontsize = 22)
plt.legend(fontsize = 20)
plt.show()
'''

x =  [100,  500,  1000, 1500, 2000, 2500, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]

CT = [122, 185, 231, 300,  346,  411,  456,  564,  683,  799,  912,  1020, 1146, 1246]
Recall = [0.6401, 0.672, 0.6806, 0.6826, 0.683, 0.6826, 0.683, 0.683, 0.6775, 0.6739, 0.6703, 0.6617, 0.6627, 0.6528]
TI = [0.03, 0.1, 0.164, 0.246, 0.376, 0.406, 0.530, 0.682, 0.97, 1.2, 1.5, 1.6, 1.8, 1.95]
TV = [6.85, 3.44, 3.05, 2.67, 2.32, 2.55, 2.69, 2.67, 2.80, 2.41, 2.24, 1.93, 1.91, 1.71]
avg_dist = [203177, 221441, 219068, 210318, 205402, 203754, 203825, 205552, 206964, 206439, 208619, 208291, 209221, 209934]
ST = np.array(TI) + np.array(TV)
'''
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(x, ST, color = "black", marker = "o", label = "Search Time")
ax1.set_ylabel('Search Time / ms', fontsize = 22)
ax1.set_xlabel('Number of Centroids', fontsize = 22)
ax1.legend(fontsize=15)


ax2 = ax1.twinx()  # this is the important function
ax2.plot(x, np.array(CT) / 100, marker = "x", color = "black", label  = "Construction Time")
ax2.set_ylabel('Construction Time / (X100s)', fontsize = 22)
ax2.set_xlabel('Number of Centroids', fontsize = 22)
ax2.legend(fontsize=15)
plt.show()
'''

'''
plt.figure()
plt.plot(x, Recall, color = "black", marker = "o")
plt.xlabel("Number of Centroids", fontsize = 22)
plt.ylabel("Recall@10", fontsize = 22)
plt.legend(fontsize = 20)
plt.show()
'''
'''
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(x, TI, color = "black", marker = "o", label = "Index Search Time")
ax1.set_ylabel('Index Search Time / ms', fontsize = 22)
ax1.set_xlabel('Number of Centroids', fontsize = 22)
ax1.legend(fontsize = 15)

ax2 = ax1.twinx()  # this is the important function
ax2.plot(x, TV, marker = "x", color = "black", label  = "Vector Search Time")
ax2.set_ylabel('Vector Search Time / ms', fontsize = 22)
ax2.set_xlabel('Number of Centroids', fontsize = 22)
ax2.legend(fontsize = 15)
plt.show()
'''

'''
plt.figure()
plt.plot(x[1:], np.array(avg_dist[1:])/100, color = "black", marker = "o")
plt.xlabel("Number of Centroids", fontsize = 22)
plt.ylabel("Average v_c_istance / (X100)", fontsize = 22)
plt.legend(fontsize = 15)
plt.show()
'''

VVVL_1 =            [0.296, 0.324, 0.358, 0.367, 0.372, 0.380, 0.385, 0.389, 0.396, 0.397, 0.399]
VVVL_time_1 =       [0.583, 0.745, 0.997, 1.342, 1.457, 1.589, 1.792, 1.986, 2.324, 2.518, 3.014]

VVL_1_reranking = [0.69, 0.73, 0.756, 0.789, 0.792, 0.803, 0.813]
VVL_time_1_reranking = [39, 45, 49, 63, 69, 79, 87]
'''
plt.figure()
plt.plot(VVL_1, VVL_time_1, marker = "o", label = "Non-reranking")
plt.plot(VVL_1_reranking, VVL_time_1_reranking, marker = "o", label = "Reranking")
plt.xlabel("Search Recall@1", fontsize = 22)
plt.ylabel("Search Time / ms", fontsize = 22)
plt.legend(fontsize = 15)
plt.show()
'''

VVVL_10 =        [0.62, 0.72, 0.78, 0.794,  0.821, 0.834, 0.857, 0.869, 0.869]
VVVL_time_10 =   [0.75, 0.94,  1.32, 1.69,  2.18,  2.89,  3.15, 3.59,   3.89]

VVL_10_reranking = [0.93, 0.95, 0.96, 0.98, 0.983, 0.985, 0.988]
VVL_time_10_reranking = [57, 72, 79, 88, 99, 105, 117]

'''
plt.figure()
plt.plot(VVL_10, VVL_time_10, marker = "o", label = "Non-reranking")
plt.plot(VVL_10_reranking, VVL_time_10_reranking, marker = "o", label = "Reranking")
plt.xlabel("Search Recall@10", fontsize = 22)
plt.ylabel("Search Time / ms", fontsize = 22)
plt.legend(fontsize = 15)
plt.show()
'''


INI_T = [0.618, 0.977, 1.24,  2.36,  3.58,    4.799,  5.18,  6.415, 7.876]
INI_1 = [0.299, 0.334, 0.349, 0.367, 0.372, 0.377, 0.379, 0.382, 0.387]
INI_10 = [0.48, 0.56, 0.588, 0.621, 0.645, 0.662, 0.678, 0.6906, 0.692]

IMI_T = [0.320, 0.845, 1.167, 2.434, 2.987, 3.659, 4.795, 5.628, 5.853, 6.0]
IMI_1 = [0.325,  0.345,  0.355,  0.372,  0.378,  0.379,  0.379, 0.380, 0.381, 0.382]
IMI_10 = [0.535, 0.582, 0.63, 0.659, 0.686, 0.702, 0.730, 0.748, 0.756, 0.76]

IVFPQ_T = [0.89, 1.79, 2.34, 2.97, 3.68, 4.14, 4.96, 5.10, 5.3]
IVFPQ_1 = [0.37, 0.392, 0.40, 0.41, 0.42, 0.425, 0.428, 0.429, 0.43]
IVFPQ_10 = [0.589, 0.656, 0.692, 0.741, 0.771, 0.792, 0.812, 0.818, 0.82]

GNOIMI_T = [0.95, 1.8, 2.79, 3.83, 4.46, 5.42, 5.9, 6.07, 6.28, 6.3]
GNOIMI_1 = [0.34, 0.37, 0.38, 0.385, 0.389, 0.398, 0.406, 0.408, 0.409, 0.41]
GNOIMI_10 = [0.568, 0.636, 0.692, 0.728, 0.758, 0.769, 0.774, 0.7788, 0.779, 0.78]

LOPQ_T = [0.42, 1.2, 1.9, 2.9, 3.4, 4.6, 4.8, 5.1, 5.6, 5.7, 5.8, 5.93]
LOPQ_1 = [0.30, 0.34, 0.365, 0.381, 0.389, 0.401, 0.405, 0.410, 0.416, 0.417, 0.419, 0.42]
LOPQ_10 = [0.598, 0.653, 0.678, 0.712, 0.733, 0.754, 0.764, 0.770, 0.789, 0.792, 0.798, 0.799]

LBI_T = [0.59, 1.27, 2.17, 2.54, 2.84, 3.73, 4.08, 4.95, 5.27]
LBI_1 = [0.34, 0.37, 0.391, 0.402, 0.420, 0.43, 0.44, 0.446, 0.45]
LBI_10 = [0.589, 0.647, 0.698, 0.721, 0.740, 0.758, 0.766, 0.777, 0.78]

Ours_T = [0.36, 1.76, 2.58, 3.29, 3.81, 4.02, 4.1, 4.31]
Ours_1 = [0.38,0.409, 0.419, 0.431, 0.440, 0.445, 0.446, 0.45]
Ours_10 = [0.62, 0.73, 0.786, 0.813, 0.828, 0.837, 0.839, 0.84]

'''
plt.figure()
plt.plot(INI_T, INI_1, label = "INI", color = "black", marker = 'v', linestyle = "dashed")
plt.plot(IMI_T, IMI_1, label = "IMI", color = "black", marker = "x", linestyle = "dashdot")
plt.plot(IVFPQ_T, IVFPQ_1, label = "IVFPQADC", color = "black", marker = ",", linestyle = "solid")
plt.plot(GNOIMI_T, GNOIMI_1, label = "GNOIMI", color = "black", marker = "D", linestyle = "dashdot")
plt.plot(LOPQ_T, LOPQ_1, label = "LOPQ", color = "black", marker = "s" , linestyle = "dashdot")
plt.plot(LBI_T, LBI_1, label = "LBI", color = "black", marker = "*", linestyle = "solid")
plt.plot(Ours_T, Ours_1, label = "Ours", color = "black", marker = "8", linestyle = "solid")
plt.xlabel("Search Time / ms", fontsize = 16)
plt.ylabel("Recall@1", fontsize = 16)
plt.xlim((0, 8))
plt.legend(fontsize = 13)
plt.show()


plt.figure()
plt.plot(INI_T, INI_10, label = "INI", color = "black", marker = 'v', linestyle = "dashed")

plt.plot(IMI_T, IMI_10, label = "IMI", color = "black", marker = "x", linestyle = "dashdot")

plt.plot(IVFPQ_T, IVFPQ_10, label = "IVFPQADC", color = "black", marker = "P", linestyle = "solid")

plt.plot(GNOIMI_T, GNOIMI_10, label = "GNOIMI", color = "black", marker = "D", linestyle = "dashdot")

plt.plot(LOPQ_T, LOPQ_10, label = "LOPQ", color = "black", marker = "s" , linestyle = "dashdot")

plt.plot(LBI_T, LBI_10, label = "LBI", color = "black", marker = "*", linestyle = "solid")

plt.plot(Ours_T, Ours_10, label = "Ours", color = "black", marker = "8", linestyle = "solid")

plt.xlabel("Search Time / ms", fontsize = 16)
plt.ylabel("Recall@10", fontsize = 16)
plt.legend(fontsize = 13)
plt.show()
'''


INI_T = [0.618, 0.977, 1.24,  2.36,  3.58, 4.799,  5.18,  6.415]
INI_1 = [0.349, 0.367, 0.372, 0.387, 0.402, 0.412, 0.418, 0.424]
INI_10 = [0.60, 0.625, 0.64, 0.673, 0.698, 0.738, 0.748, 0.752]

IMI_T = [0.320, 0.845, 1.167, 2.434, 2.987, 3.659, 4.82]
IMI_1 = [ 0.35,  0.369,  0.376,  0.381,  0.384, 0.386, 0.39]
IMI_10 = [0.51, 0.576, 0.622, 0.670, 0.699, 0.718, 0.762]

IVFPQ_T = [0.89, 1.79, 2.34, 2.97, 3.68, 4.14, 4.31]
IVFPQ_1 = [0.41, 0.4307, 0.439, 0.450, 0.454, 0.458, 0.46]
IVFPQ_10 = [0.592, 0.661, 0.692, 0.758, 0.803, 0.817, 0.82]

GNOIMI_T = [0.52, 1.26, 1.59, 1.9, 2.7, 3.2, 3.8, 4.1, 4.9, 5.1, 5.85]
GNOIMI_1 = [0.28, 0.33, 0.36, 0.378, 0.385, 0.394, 0.396, 0.408, 0.412, 0.417, 0.42]
GNOIMI_10 = [0.582, 0.608, 0.648, 0.669, 0.684, 0.7188, 0.743, 0.768, 0.77, 0.772, 0.78]

LOPQ_T = [0.63, 1.3, 1.9, 2.1, 3.2, 3.8, 4.68, 5.15, 5.46]
LOPQ_1 = [0.25, 0.31, 0.338, 0.35, 0.372, 0.392, 0.398, 0.409, 0.41]
LOPQ_10 = [0.508,  0.593, 0.648, 0.684, 0.718, 0.749, 0.776, 0.785, 0.80]

LBI_T = [0.59, 1.27, 2.17, 2.54, 2.84, 3.73, 4.08, 4.55]
LBI_1 = [ 0.381, 0.398, 0.412, 0.431, 0.440, 0.456, 0.458, 0.46]
LBI_10 = [0.547, 0.592, 0.664, 0.689, 0.730, 0.765, 0.778, 0.78]

Ours_T = [0.36, 1.76, 2.58, 3.29, 3.81, 4.02, 4.1]
Ours_1 = [0.402, 0.439, 0.4528, 0.467, 0.472, 0.476, 0.478]
Ours_10 = [0.527, 0.682, 0.768, 0.808,0.842, 0.848, 0.85]

'''
plt.figure()
plt.plot(INI_T, INI_1, label = "INI", color = "black", marker = 'v', linestyle = "dashed")

plt.plot(IMI_T, IMI_1, label = "IMI", color = "black", marker = "x", linestyle = "dashdot")

plt.plot(IVFPQ_T, IVFPQ_1, label = "IVFPQADC", color = "black", marker = "P", linestyle = "solid")

plt.plot(GNOIMI_T, GNOIMI_1, label = "GNOIMI", color = "black", marker = "D", linestyle = "dashdot")

plt.plot(LOPQ_T, LOPQ_1, label = "LOPQ", color = "black", marker = "s" , linestyle = "dashdot")

plt.plot(LBI_T, LBI_1, label = "LBI", color = "black", marker = "*", linestyle = "solid")

plt.plot(Ours_T, Ours_1, label = "Ours", color = "black", marker = "8", linestyle = "solid")

plt.xlabel("Search Time / ms", fontsize = 16)
plt.ylabel("Recall@1", fontsize = 16)
plt.legend(fontsize = 13)
plt.show()
'''

'''
plt.figure()
plt.plot(INI_T, INI_10, label = "INI", color = "black", marker = 'v', linestyle = "dashed")


plt.plot(IMI_T, IMI_10, label = "IMI", color = "black", marker = "x", linestyle = "dashdot")

plt.plot(IVFPQ_T, IVFPQ_10, label = "IVFPQADC", color = "black", marker = "P", linestyle = "solid")

plt.plot(GNOIMI_T, GNOIMI_10, label = "GNOIMI", color = "black", marker = "D", linestyle = "dashdot")

plt.plot(LOPQ_T, LOPQ_10, label = "LOPQ", color = "black", marker = "s" , linestyle = "dashdot")

plt.plot(LBI_T, LBI_10, label = "LBI", color = "black", marker = "*", linestyle = "solid")

plt.plot(Ours_T, Ours_10, label = "Ours", color = "black", marker = "8", linestyle = "solid")

plt.xlabel("Search Time / ms", fontsize = 16)
plt.ylabel("Recall@10", fontsize = 16)
plt.legend(fontsize = 13)
plt.show()
'''

# SIFT100M Dataset

NC =           np.array([2,    1.5,  1.0,  0.5,  0.4, 0.3, 0.2, 0.1]) #e5
Kmeans_SC =    np.array([2.20, 2.74, 3.77, 6.62, 7.92, 10.04, 14.01, 24.80]) #e3
OptKmeans_SC = np.array([1.35, 2.07, 3.06, 5.80, 6.40, 8.40,  12.28, 21.50])
KmeansSSE =    np.array([4622, 4750, 4974, 5343, 5470, 5627,  5856,  6266])
OptKmeansSSE = np.array([4650, 4780, 5009, 5395, 5532, 5695,  5955,  6395])
Kmeans_T =     [42,   38,   28,  25,   20,    18]
OptKmeans_T =  [3.2,  2.5, 2.1, 1.7, 1.3,  1.5]


plt.figure()
'''
plt.plot((NC * 100) [:-2], Kmeans_SC [:-2], label = "KMeans SC", color = "black", marker = "P", linestyle = "solid")
plt.plot((NC * 100) [:-2], OptKmeans_SC [:-2],  label = "A-Kmeans SC", color = "black", marker = "s" , linestyle = "solid")
plt.xlabel("Number of Cluster (Thousand)", fontsize = 16)
plt.ylabel("Search Cost", fontsize = 16)
plt.title("SIFT1B", fontsize = 16)
plt.legend(fontsize = 16)
plt.show()


plt.plot((NC * 100) [:-2], 100 * (OptKmeansSSE [:-2]-KmeansSSE [:-2]) / KmeansSSE[:-2], color = 'black', marker = "x")
#plt.legend(fontsize = 16)
plt.title("SIFT1B", fontsize = 16)
plt.xlabel("Number of Cluster (Thousand)", fontsize = 16)
plt.ylabel("SSE Difference (%)", fontsize = 16)
plt.show()
'''

# DEEP100M Dataset

NC = np.array([2,    1.5,  1.0,  0.5,  0.4, 0.3]) #e5
Kmeans_SC =    np.array([3.21, 3.98, 5.06, 8.45, 11.06, 15.58])
OptKmeans_SC = np.array([2.86, 3.29, 4.18, 6.82, 9.25, 13.42])


plt.figure()

'''
plt.plot((NC * 100) [:], Kmeans_SC [:], label = "KMeans SC", color = "black", marker = "P", linestyle = "solid")
plt.plot((NC * 100) [:], OptKmeans_SC [:],  label = "A-Kmeans SC", color = "black", marker = "s" , linestyle = "solid")
plt.xlabel("Number of Cluster (Thousand)", fontsize = 16)
plt.ylabel("Search Cost", fontsize = 16)
plt.title("DEEP1B", fontsize = 16)
plt.legend(fontsize = 16)
plt.show()


plt.plot((NC * 100) [:], [0.68, 0.71, 0.76, 1.09, 1.14, 1.25], color = "black", marker = "P", linestyle = "solid")
plt.title("DEEP1B", fontsize = 16)
plt.xlabel("Number of Cluster (Thousand)", fontsize = 16)
plt.ylabel("SSE Difference (%)", fontsize = 16)
plt.show()
'''

'''
NC = np.array([1, 5, 10, 15, 20])
Percent = np.array([72, 77, 79, 82, 83])
plt.plot(NC, Percent, label = "Percentage of Vectors in Neighbor Cluster", color = "black", marker = "o")
plt.xlabel("Number of Cluster (Thousand)", fontsize = 20)
plt.ylabel("Percentage (%)", fontsize = 20)
plt.xlim((0, 21))
plt.ylim((70, 85))
plt.xticks(NC)
plt.legend(fontsize = 14)
plt.show()


NC = np.array([1, 2,3, 4,5])
Percent = np.array([75, 77, 78, 79, 81])
plt.plot(NC, Percent, label = "Percentage of Vectors in Neighbor Cluster", color = "black", marker = "o")
plt.xlabel("Number of Cluster (Million)", fontsize = 20)
plt.ylabel("Percentage (%)", fontsize = 20)
plt.xlim((0.5, 5.5))
plt.ylim((70, 85))
plt.xticks(NC)
plt.legend(fontsize = 14)
plt.show()
'''


NC = [1, 2, 3, 4, 5]
K_5 = [61.2, 66.4, 69, 72, 73]
K_5_O = [18.4, 21.3, 24.2, 26.7]
K_10 = [72.5, 77, 79.8, 82.3, 83]
K_10_O = [23]
K_20 = [82.8, 86.3, 90.9, 90.6, 90.0]
K_20_O = [26]
K_50 = [88.4, 90.4, 92.0, 94.0, 93.3]
K_50_O = [29]

name_list = ['5', '10', '20', '50']
x = np.arange(len(name_list))
width = 0.4

'''
plt.bar(x, [K_5[0], K_10[0], K_20[0], K_50[0]],width = width, color = "w", edgecolor = "k", label = 'Inverted Index', hatch = "////")
for a,b in zip(x,np.array([K_5[0], K_10[0], K_20[0], K_50[0]])):
    plt.text(a + 0.05, b+0.5, '%.1f' %b, ha = 'center', va = 'bottom')

for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x+width, [K_5_O[0], K_10_O[0], K_20_O[0], K_50_O[0]], width = width, color = "w", edgecolor = "k", label = 'Inverted Index + Neighbor List', hatch = "\\\\\\\\")
for a,b in zip(x + width,np.array([K_5_O[0], K_10_O[0], K_20_O[0], K_50_O[0]])):
    plt.text(a + 0.05, b+0.5, '%.1f' %b, ha = 'center', va = 'bottom')

plt.xticks(x+width/2, name_list)
plt.xlabel("k")
plt.ylabel("Percentage (%)")
plt.title("SIFT1B")
plt.legend()
plt.show()
'''



NC = [1, 2, 3, 4, 5]
K_5 = [64.7, 68.6, 71.7, 73.3, 74.4]
K_5_O = [15.6]
K_10 = [75.2, 77, 79.8, 82.3, 83]
K_10_O = [19.4]
K_20 = [83.7, 86.3, 90.9, 90.6, 90.0]
K_20_O = [22.5]
K_50 = [88.9, 90.4, 92.0, 94.0, 93.3]
K_50_O = [24.8]

'''
plt.bar(x, [K_5[0], K_10[0], K_20[0], K_50[0]],width = width, color = "w", edgecolor = "k", label = 'Inverted Index', hatch = "////")
for a,b in zip(x,np.array([K_5[0], K_10[0], K_20[0], K_50[0]])):
    plt.text(a + 0.05, b+0.5, '%.1f' %b, ha = 'center', va = 'bottom')

for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x+width, [K_5_O[0], K_10_O[0], K_20_O[0], K_50_O[0]], width = width, color = "w", edgecolor = "k", label = 'Inverted Index + Neighbor List', hatch = "\\\\\\\\")
for a,b in zip(x + width,np.array([K_5_O[0], K_10_O[0], K_20_O[0], K_50_O[0]])):
    plt.text(a + 0.05, b+0.5, '%.1f' %b, ha = 'center', va = 'bottom')

plt.xticks(x+width/2, name_list)
plt.xlabel("k")
plt.ylabel("Percentage (%)")
plt.title("DEEP1B")
plt.legend()
plt.show()
'''

'''
NI = 10000
SplitParts = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250])
SearchTime = np.array([7.8, 7.4, 6.9, 6.5, 6.1, 5.8, 5.3, 5.0, 4.7, 4.5, 4.2, 3.7, 3.3, 3.0, 2.7, 2.6, 2.3, 2.1, 2.2, 2.5, 2.8, 3.1, 3.4, 3.6, 3.8])
plt.figure()
plt.plot(SplitParts * NI * 2, SearchTime, label = "Search Time with Different NC", color = "black", marker = "o")
plt.scatter([3700000], [0.15], label = "Learnt NC", color = "black", marker = "*", s = 100)
plt.xlabel("Number of Cluster", fontsize = 20)
plt.ylabel("Search Time (ms)", fontsize = 20)
plt.ylim([0, 8])
plt.legend()
plt.show()
'''
NC = [1,2,3,4,5,6,7,8,9]
x = [9.8,8.5,8.5,8.3,8.4,8.8,9.1,9.3,9.8]
y = [10.8,10.7,10.8,9.3,10.4,10.3,10.8,10.1,10.0]
z = [ 13.1,12.0,11.7,10.2,10.7,11.5,12.0,11.2,11.5]

NC = [1,2,3,4,5,6,7,8,9]
x = [ 1.02,1.05,1.04,1.04,1.02,1.07,1.06,1.03,1.08]
y = [ 1.06,1.08,1.06,1.08,1.04,1.04,1.05,1.07,1.06]
z = [  1.05,1.04,1.07,1.05,1.03,1.08,1.10,1.08,1.05]
'''
AVG = (np.array(x) + np.array(y) + np.array(z))/3
print(AVG)
plt.figure()
plt.plot(NC, x, label = "0.01",  marker = "o")
plt.plot(NC, y, label = "0.1", marker = "*")
plt.plot(NC, z, label = "1.0", marker = "P")
plt.plot(NC, AVG, label = "Average", marker = "D")
plt.xlabel("Number of Cluster", fontsize = 20)
plt.ylabel("Error Rate", fontsize = 20)
plt.legend()
plt.show()
'''

x = [ 1.32,1.38,1.48,1.51,1.56,1.62,1.70,1.72,1.70]
plt.figure()
plt.plot(NC, x, label = "Time Ratio",  marker = "o")
plt.xlabel("Number of Cluster", fontsize = 20)
plt.ylabel("Time Ratio", fontsize = 20)
plt.legend()
plt.show()

