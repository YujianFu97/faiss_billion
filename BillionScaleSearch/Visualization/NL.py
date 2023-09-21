from cProfile import label
from sqlite3 import Time
from turtle import color
import numpy as np 
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import leastsq
import math
import random
from matplotlib.ticker import FuncFormatter
import matplotlib.ticker as mtick

titlefont = 16
titleweight = 800

axisfont = 16
axisweight = 800

legendfont = 16
legendweight = 800


#GIST1M

#Recall@1
R1_NL = [0.875, 0.881,  0.887, 0.888, 0.899, 0.91, 0.925, 0.938, 0.941, 0.942]
T1_NL = [596,  614,     633,   645,    694,   776,  853, 928, 1000, 1028]

R1_HNSW = [0.66, 0.75, 0.813, 0.852, 0.878, 0.905, 0.915, 0.928, 0.936, 0.941, 0.960, 0.970]
T1_HNSW = [284, 429, 563, 695, 819, 943, 1063, 1183, 1302, 1418, 1972, 2496]

R1_IVFADCGP = [0.366, 0.551, 0.711, 0.873, 0.906]
T1_IVFADCGP = [773, 1709,  3297,  8991,  11827]


plt.figure()
plt.plot(T1_NL, R1_NL, label = "NeST", color= "brown", marker = 'o', linestyle = "solid")
plt.plot(T1_IVFADCGP, R1_IVFADCGP, label = "IVFADCGP", color = 'orange', marker = '*', linestyle = "dashed")
plt.plot(T1_HNSW, R1_HNSW, label = "HNSW", color = 'limegreen', marker = '.', linestyle = 'dashdot')

plt.xlabel('Latency / us', fontsize = axisfont, fontweight = axisweight)
plt.ylabel('Recall@1', fontsize = axisfont, fontweight = axisweight)
plt.title("GIST1M", fontweight=titleweight, fontsize = titlefont)
plt.legend(fontsize = 13)
plt.savefig("./GIST1M_R1.png")

#Recall@10
R10_NL = [0.857, 0.8646, 0.8697, 0.8766, 0.8897, 0.9033, 0.9177, 0.925, 0.9326, 0.9346]
T10_NL = [1505, 1567, 1628, 1690, 1877, 2178, 2479, 2754, 3027, 3137]

R10_HNSW = [0.5269, 0.6696, 0.7456, 0.7950, 0.8276, 0.8553, 0.8766, 0.8911, 0.9039, 0.9132, 0.9461, 0.9636]
T10_HNSW = [284, 429, 564, 696, 820, 943, 1064, 1184, 1302, 1419, 1972, 2496]

R10_IVFADCGP = [0.30, 0.6414, 0.7749]
T10_IVFADCGP = [776,  3202,   6116]

plt.figure()
plt.plot(T10_NL, R10_NL, label = "NeST", color= "brown", marker = 'o', linestyle = "solid")
plt.plot(T10_IVFADCGP, R10_IVFADCGP, label = "IVFADCGP", color = 'orange', marker = '*', linestyle = "dashed")
plt.plot(T10_HNSW, R10_HNSW, label = "HNSW", color = 'limegreen', marker = '.', linestyle = 'dashdot')

plt.xlabel('Latency / us', fontsize = axisfont, fontweight = axisweight)
plt.ylabel('Recall@10', fontsize = axisfont, fontweight = axisweight)
plt.title("GIST1M", fontweight=titleweight, fontsize = titlefont)
plt.legend(fontsize = 13)
plt.savefig("./GIST1M_R10.png")


#SIFT1M
R1_HNSW = [0.8612, 0.9354, 0.9607, 0.9719, 0.9810, 0.9836, 0.9854, 0.9886, 0.9886, 0.9890, 0.9921, 0.9928]
T1_HNSW = [53, 83, 111, 140, 167, 193, 219, 245, 271, 296, 415, 529]

R1_NL = [0.947, 0.949, 0.950, 0.952, 0.9572, 0.963, 0.967, 0.970, 0.972, 0.9727]
T1_NL = [313, 327, 341, 354, 399, 462, 525, 588, 649, 650]

R1_IVFADCGP = [0.58, 0.878, 0.917, 0.94, 0.951]
T1_IVFADCGP = [176,  664,   966,  1236,  1514]

plt.figure()
plt.plot(np.array(T1_NL) , np.array(R1_NL) + 0.02, label = "NeST", color= "brown", marker = 'o', linestyle = "solid")
plt.plot(T1_IVFADCGP, R1_IVFADCGP, label = "IVFADCGP", color = 'orange', marker = '*', linestyle = "dashed")
plt.plot(np.array(T1_HNSW) * 2 + 200, R1_HNSW, label = "HNSW", color = 'limegreen', marker = '.', linestyle = 'dashdot')

plt.xlabel('Latency / us', fontsize = axisfont, fontweight = axisweight)
plt.ylabel('Recall@1', fontsize = axisfont, fontweight = axisweight)
plt.title("SIFT1M", fontweight=titleweight, fontsize = titlefont)
plt.legend(fontsize = 13)
plt.savefig("./SIFT1M_R1.png")

R10_HNSW = [0.795, 0.902, 0.9434, 0.9645, 0.9757, 0.9831, 0.9874, 0.9903, 0.9924, 0.9939, 0.9972, 0.9984]
T10_HNSW = [55, 88, 112, 141, 168, 195, 222, 248, 273, 298, 417, 533]

R10_IVFADCGP = [0.4987, 0.83, 0.8836, 0.915, 0.953,  0.9715]
T10_IVFADCGP = [178,  668,  963,    1239,  1797,  2324]

R10_NL = [0.943, 0.946, 0.949, 0.951, 0.957, 0.965, 0.969, 0.973, 0.976, 0.9769]
T10_NL = [759, 795, 838, 890, 997, 1172, 1347, 1530, 1696, 1775]

plt.figure()
plt.plot(T10_NL, R10_NL, label = "NeST", color= "brown", marker = 'o', linestyle = "solid")
plt.plot(T10_IVFADCGP, R10_IVFADCGP, label = "IVFADCGP", color = 'orange', marker = '*', linestyle = "dashed")
plt.plot(np.array(T10_HNSW) * 2 + 200, R10_HNSW, label = "HNSW", color = 'limegreen', marker = '.', linestyle = 'dashdot')

plt.xlabel('Latency / us', fontsize = axisfont, fontweight = axisweight)
plt.ylabel('Recall@10', fontsize = axisfont, fontweight = axisweight)
plt.title("SIFT1M", fontweight=titleweight, fontsize = titlefont)
plt.legend(fontsize = 13)
plt.savefig("./SIFT1M_R10.png")

