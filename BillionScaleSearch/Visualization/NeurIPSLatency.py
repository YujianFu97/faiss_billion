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

xstart = 0
xend = 16
ystart = 0
yend = 1.0

plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"

fig = plt.figure(figsize=(20,6))
axes = fig.subplots(nrows=1, ncols=4)
# Data for latency and recall experiment

# Recall1@1
# SIFT Dataset
plt.subplot(1, 4, 1)
NeST_T =        np.array([0.89, 1.36, 1.59, 2.7,  4.89, 6.9,  7.86, 9.14,14.3])
NeST_Recall1 =  np.array([0.20, 0.42, 0.55, 0.72, 0.87, 0.90, 0.91, 0.92, 0.928])

IVFADCGP_T =        np.array([0.48, 2.7,  4.19, 6.01, 7.72,  9.88, 13.12])
IVFADCGP_Recall1 =  np.array([0.20, 0.55, 0.71, 0.81, 0.86,  0.90, 0.91])

Faiss_T =       np.array([0.87, 2.48, 3.96, 6.7, 8.9,  12.8])
Faiss_Recall1 = np.array([0.2, 0.36, 0.53, 0.74, 0.82, 0.86])

FaissPLE_T =       np.array([0.99, 2.9,  6.8,  9.8,  13.4, 15.66])
FaissPLE_Recall1 = np.array([0.22, 0.45, 0.77, 0.84, 0.875, 0.891])

axes[0].plot(NeST_T, NeST_Recall1, label = "NeST", color= "black", marker = 'o', linestyle = "solid")
axes[0].plot(IVFADCGP_T, IVFADCGP_Recall1, label = "IVFADCGP", color = 'black', marker = '*', linestyle = "dashed")
axes[0].plot(FaissPLE_T, FaissPLE_Recall1, label = "FaissPLE", color = 'black', marker = '.', linestyle = 'dashdot')
axes[0].plot(Faiss_T, Faiss_Recall1, label = 'Faiss', color = 'black', marker = 'v', linestyle = 'dotted')

plt.xlabel('Latency / ms', fontsize = axisfont, fontweight = axisweight)
plt.ylabel('Recall1@1', fontsize = axisfont, fontweight = axisweight)
plt.title("SIFT1B", fontweight=titleweight, fontsize = titlefont)
plt.xlim(xstart, xend)
plt.ylim(ystart, yend)


# DEEP dataset
NeST_T =        np.array([0.68, 1.29, 1.56,  1.99, 2.98, 5.44, 7.95, 10.7, 13.19])
NeST_Recall1 =  np.array([0.1,  0.31, 0.55,  0.72, 0.81, 0.868, 0.9,  0.913, 0.92])

Faiss_T =       np.array([0.95, 3.02, 5.3,  6.7,  9.67, 13.9, 16.8])
Faiss_Recall1 = np.array([0.18, 0.50, 0.65, 0.70, 0.83, 0.90, 0.91])

FaissPLE_T =       np.array([1.21, 1.98, 3.48, 8.95, 10.4, 11.8, 14.9])
FaissPLE_Recall1 = np.array([0.22, 0.40, 0.57, 0.81, 0.86, 0.88, 0.90])

IVFADCGP_T =        np.array([0.62, 1.88, 3.79, 6.48, 8.99, 10.8, 11.9])
IVFADCGP_Recall1 =  np.array([0.18, 0.55, 0.71, 0.81, 0.862, 0.888, 0.90])

plt.subplot(1, 4, 2)
axes[1].plot(NeST_T, NeST_Recall1, label = "NeST", color= "black", marker = 'o', linestyle = "solid")
axes[1].plot(IVFADCGP_T, IVFADCGP_Recall1, label = "IVFADCGP", color = 'black', marker = '*', linestyle = "dashed")
axes[1].plot(FaissPLE_T, FaissPLE_Recall1, label = "FaissPLE", color = 'black', marker = '.', linestyle = 'dashdot')
axes[1].plot(Faiss_T, Faiss_Recall1, label = 'Faiss', color = 'black', marker = 'v', linestyle = 'dotted')

plt.xlabel('Latency / ms', fontsize = axisfont, fontweight = axisweight)
plt.ylabel('Recall1@1', fontsize = axisfont, fontweight = axisweight)
plt.title("DEEP1B", fontweight=titleweight, fontsize = titlefont)
#plt.legend(fontsize = 13)
plt.xlim(xstart, xend)
plt.ylim(ystart, yend)


# Turing Dataset
NeST_T =        np.array([0.75, 1.28, 1.48, 1.94,  2.69, 4.76, 6.98, 8.74, 10.14, 15.2])
NeST_Recall1 =  np.array([0.1,  0.33, 0.51, 0.65,  0.72, 0.86, 0.90, 0.91, 0.92,  0.94])

IVFADCGP_T =        np.array([0.51, 1.32, 3.4, 5.91, 8.7,  10.7, 13.9])
IVFADCGP_Recall1 =  np.array([0.18, 0.41, 0.6, 0.75, 0.85,  0.878, 0.91])

Faiss_T =       np.array([0.87, 2.52, 6.84, 9.87, 14.3, 16.8])
Faiss_Recall1 = np.array([0.18, 0.39, 0.64, 0.75, 0.87, 0.89])

FaissPLE_T =       np.array([0.96, 3.48, 6.84, 12.7, 16.8])
FaissPLE_Recall1 = np.array([0.22, 0.48, 0.65, 0.82, 0.86])


plt.subplot(1, 4, 3)

axes[2].plot(NeST_T, NeST_Recall1, label = "NeST", color= "black", marker = 'o', linestyle = "solid")
axes[2].plot(IVFADCGP_T, IVFADCGP_Recall1, label = "IVFADCGP", color = 'black', marker = '*', linestyle = "dashed")
axes[2].plot(FaissPLE_T, FaissPLE_Recall1, label = "FaissPLE", color = 'black', marker = '.', linestyle = 'dashdot')
axes[2].plot(Faiss_T, Faiss_Recall1, label = 'Faiss', color = 'black', marker = 'v', linestyle = 'dotted')

plt.xlabel('Latency / ms', fontsize = axisfont, fontweight = axisweight)
plt.ylabel('Recall1@1', fontsize = axisfont, fontweight = axisweight)
plt.title("Turing1B", fontweight=titleweight, fontsize = titlefont)
#plt.legend(fontsize = 13)
plt.xlim(xstart, xend)
plt.ylim(ystart, yend)


# SpaceV Dataset
NeST_T =        np.array([0.72, 1.29, 1.79,  2.83, 3.41, 4.31, 5.43, 5.94, 9.87, 14.9])
NeST_Recall1 =  np.array([0.1,  0.31, 0.55,  0.70, 0.77, 0.82, 0.86, 0.88, 0.92, 0.94])

Faiss_T =       np.array([1.08, 1.89, 5.42, 7.37, 10.32, 14.8, 18.7])
Faiss_Recall1 = np.array([0.18, 0.37, 0.55, 0.64, 0.75, 0.85, 0.89])

FaissPLE_T =       np.array([1.79, 2.67, 4.69, 8.12, 12.7, 15.9])
FaissPLE_Recall1 = np.array([0.22, 0.45, 0.59, 0.72, 0.83, 0.89])

IVFADCGP_T =        np.array([0.52, 1.08, 1.68, 2.85, 6.98, 7.84, 12.1, 15.4])
IVFADCGP_Recall1 =  np.array([0.18, 0.35, 0.51, 0.64, 0.79, 0.81, 0.89, 0.91])

plt.subplot(1, 4, 4)

axes[3].plot(NeST_T, NeST_Recall1, label = "NeST", color= "black", marker = 'o', linestyle = "solid")
axes[3].plot(IVFADCGP_T, IVFADCGP_Recall1, label = "IVFADCGP", color = 'black', marker = '*', linestyle = "dashed")
axes[3].plot(FaissPLE_T, FaissPLE_Recall1, label = "FaissPLE", color = 'black', marker = '.', linestyle = 'dashdot')
axes[3].plot(Faiss_T, Faiss_Recall1, label = 'Faiss', color = 'black', marker = 'v', linestyle = 'dotted')

plt.xlabel('Latency / ms', fontsize = axisfont, fontweight = axisweight)
plt.ylabel('Recall1@1', fontsize = axisfont, fontweight = axisweight)
plt.title("SpaceV1B", fontweight=titleweight, fontsize = titlefont)
#plt.legend(fontsize = 13)
plt.xlim(xstart, xend)
plt.ylim(ystart, yend)

lines, labels = fig.axes[-1].get_legend_handles_labels()

fig.legend(lines, labels, loc='upper center', ncol = 4, prop=dict(weight=legendweight, size = legendfont))
plt.show()








