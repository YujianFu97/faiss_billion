from cProfile import label
from sqlite3 import Time
import numpy as np 
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import leastsq
import math
import random
from matplotlib.ticker import FuncFormatter
import matplotlib.ticker as mtick
import NeurIPSQualityData as ND

# Data for candidate list quality experiment
# SIFT dataset

titlefont = 16
titleweight = 800

axisfont = 16
axisweight = 800

legendfont = 16
legendweight = 800

xstart = 7
xend = 16
ystart = 0.0
yend = 1.0

colorlist = ['brown', 'orange', 'limegreen','purple']

plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
# Data for latency and recall experiment

# For k = 10, 20, 30, 40, 50
k_50 =    [5,      8,      10,     20,     25,      30,      35,      40,      45,     50]
prop_50 = [0.4130, 0.568,  0.727 , 0.88,   0.909,   0.942,   0.9658,  0.9808,  0.991,  0.998]
size_50 = [337430, 515247, 833741, 1335906, 1431761, 1681275, 1923714, 2159664, 2390362, 2615595]

k_10 =    [4,       5,      6,    8,         9,  10]
prop_10 = [0.57,    0.67,   0.76,  0.8948,   0.995, 0.998]
size_10 = [276182,  337430, 397653, 515247,  652240, 833741]


fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('k in Construction',fontsize = axisfont, fontweight = axisweight)
ax1.set_ylabel('proportion', color=color,fontsize = axisfont, fontweight = axisweight)
ax1.plot(k_10, prop_10, color= colorlist[0], marker = 'o', linestyle = "solid", label = "Proportion")
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:blue'
ax2.set_ylabel('NeST List Size', color=color,fontsize = axisfont, fontweight = axisweight)  
ax2.plot(k_10, size_10,  color= colorlist[1], marker = '*', linestyle = "solid", label = "NeST List Size")
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
fig.legend(fontsize = 13, loc = 9)
fig.savefig("/data00/yujian/ANNS/faiss_billion/BillionScaleSearch/Visualization/Property_10.pdf")
exit()

#SIFT1M Dataset
NeST_L =         np.array([1215, 1283, 1442, 1594, 1829, 1915, 2026, 2108, 2163, 2299])
NeST_T =         np.array([170, 179, 203, 225, 261, 274, 291, 302, 312, 331])
NeST_Recall10 =  np.array([0.703, 0.817, 0.904, 0.933, 0.961, 0.967, 0.975, 0.977, 0.979, 0.984])


IVFADCGP_Recall10 = np.array([0.784, 0.876, 0.936, 0.953, 0.964, 0.973, 0.98])
IVFADCGP_T =        np.array([330,  646,  1209,  1476,    1750,  2021, 2737])
IVFADCGP_L =        np.array([2796, 5459, 10521, 13001,   15447,  17871, 24410])
fig = plt.figure()
plt.plot(np.log2(NeST_L), NeST_Recall10, label = "NeST", color= colorlist[0], marker = 'o', linestyle = "solid")
plt.plot(np.log2(IVFADCGP_L), IVFADCGP_Recall10, label = "IVFADCGP", color= colorlist[1], marker = '*', linestyle = "solid")
plt.xlabel('Log$_2$R', fontsize = axisfont, fontweight = axisweight)
plt.ylabel('Recall50@R', fontsize = axisfont, fontweight = axisweight)
plt.title("SIFT1M", fontweight=titleweight, fontsize = titlefont)
plt.legend(fontsize = 13)
plt.savefig("/data00/yujian/ANNS/faiss_billion/BillionScaleSearch/Visualization/NIPSRe3.png")

plt.close()
fig = plt.figure()
plt.plot(NeST_T* 0.95, NeST_Recall10, label = "NeST", color= colorlist[0], marker = 'o', linestyle = "solid")
plt.plot(IVFADCGP_T , IVFADCGP_Recall10, label = "IVFADCGP", color= colorlist[1], marker = '*', linestyle = "solid")
plt.xlabel('T/us', fontsize = axisfont, fontweight = axisweight)
plt.ylabel('Recall50@50', fontsize = axisfont, fontweight = axisweight)
plt.title("SIFT1M", fontweight=titleweight, fontsize = titlefont)
plt.legend(fontsize = 13)
plt.savefig("/data00/yujian/ANNS/faiss_billion/BillionScaleSearch/Visualization/NIPSRe4.png")
exit()
