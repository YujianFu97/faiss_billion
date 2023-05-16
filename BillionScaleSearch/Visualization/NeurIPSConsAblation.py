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

# Construction Time

Base = np.array([78, 71, 73, 73])
Faiss = np.array([1.0, 1.0, 1.0, 1.0])
IVFADCGP = np.array([1.9, 1.8, 1.8, 1.8])
NeST = np.array([1.6, 1.5, 1.6, 1.6])
FaissPLE = np.array([1.06, 1.02, 1.05, 1.04])

name_list = ['SIFT1B', 'DEEP1B', 'Turing1B', 'SpaceV1B']
x = np.arange(len(name_list))
width = 0.15

plt.figure()
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.tick_params(labelsize=15)
plt.bar(x + width,NeST * Base, width = width, color = "w", edgecolor = "k", label = 'NeST', hatch = "XXX")


plt.bar(x+ 2* width, IVFADCGP * Base, width = width, color = "w", edgecolor = "k", label = 'IVFADCGP')


plt.bar(x + 3 * width, FaissPLE * Base, width = width, color = "w", edgecolor = "k", label = 'FaissPLE', hatch = "///")

plt.bar(x + 4 * width, Faiss * Base, width = width, color = "w", edgecolor = "k", label = 'Faiss', hatch = "***")

plt.xticks(x+width*2, name_list)
plt.ylabel("Construction Time / hours", fontsize = 18)
plt.legend(bbox_to_anchor=(0, 1.01 ), loc=3, borderaxespad=0, fontsize = 11,  ncol = 4)

plt.subplots_adjust(left = 0.168, right = 0.995, bottom = 0.12, top = 0.88)
#plt.ylim((0, 3.5))
#plt.show()


plt.figure()
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"

# Data for latency and recall experiment

# Recall1@1
# SIFT Dataset
NeST_T_5 =         np.array([0.89, 1.36, 1.59, 2.7,  4.79, 6.9,  7.86, 9.14])
NeST_Recall1_5 =   np.array([0.20, 0.42, 0.55, 0.72, 0.87, 0.90, 0.91, 0.92])

NeST_T_10 =        np.array([1.68, 2.16, 2.7, 3.2,  4.2, 5.4,  6.6, 9.5])
NeST_Recall1_10 =  np.array([0.26, 0.49, 0.61, 0.70, 0.82, 0.88, 0.92, 0.94])

NeST_T_15 =        np.array([2.19, 2.76, 3.15, 4.18, 4.67, 5.35,  6.3, 7.5, 9.1])
NeST_Recall1_15 =  np.array([0.28, 0.51, 0.63, 0.78, 0.83, 0.89, 0.93, 0.95, 0.96])

NeST_T_20 =        np.array([3.07, 3.5,  3.89,  4.9, 6, 6.9,  7.86, 9.14])
NeST_Recall1_20 =  np.array([0.30, 0.52, 0.63,  0.80, 0.88, 0.92, 0.95, 0.98])

plt.plot(NeST_T_5, NeST_Recall1_5, label = '$K$ = 5',color= "black", marker = 'o', linestyle = (0, (5, 10)))
plt.plot(NeST_T_10, NeST_Recall1_10, label = '$K$ = 10',color= "black", marker = 'v', linestyle = "dashed")
plt.plot(NeST_T_15, NeST_Recall1_15, label = '$K$ = 15',color= "black", marker = '.', linestyle = "dotted")
plt.plot(NeST_T_20, NeST_Recall1_20, label = '$K$ = 20',color= "black", marker = '*', linestyle = "dashdot")

plt.xlabel('Latency / ms', fontsize = axisfont, fontweight = axisweight)
plt.ylabel('Recall1@1', fontsize = axisfont, fontweight = axisweight)
plt.title("SIFT1B", fontweight=titleweight, fontsize = titlefont)
plt.legend(fontsize = 13)
plt.ylim(0.2, 1.0)
plt.show()

