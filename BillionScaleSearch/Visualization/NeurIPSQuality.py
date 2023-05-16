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

plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
# Data for latency and recall experiment

fig = plt.figure(figsize=(20, 6))
axes = fig.subplots(nrows=1, ncols=4)

plt.subplot(1, 4, 1)
# Recall1@1
# SIFT Dataset
NeST_L =        np.array([400, 978, 1145, 1246, 1442, 1854, 2491, 3517, 5768, 19000])
NeST_Recall1 =  np.array([0.1, 0.2, 0.3,  0.4,  0.5,  0.6,  0.7, 0.8,  0.9,  0.95 ])

Faiss_L =       np.array([400, 1000, 1700, 2700, 3800, 5600, 7300, 9400, 16000, 62000 ])
Faiss_Recall1 = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,  0.9,  0.95 ])

FaissPLE_L =       np.array([398, 996, 1355, 2148, 3100, 5000, 6900, 8900, 15400, 61500 ])
FaissPLE_Recall1 = np.array([0.1, 0.2, 0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  0.9,   0.95 ])

IVFADCGP_L =        np.array([216, 512, 794, 1224, 1895, 3129, 4128, 5673, 9876, 45000])
IVFADCGP_Recall1 =  np.array([0.1, 0.2, 0.3, 0.4,  0.5,  0.6,  0.7,  0.8,  0.9,  0.95 ])


''''''
axes[0].plot(np.log2(NeST_L), NeST_Recall1, label = "NeST", color= "black", marker = 'o', linestyle = "solid")
axes[0].plot(np.log2(IVFADCGP_L), IVFADCGP_Recall1, label = "IVFADCGP", color = 'black', marker = '*', linestyle = "dashed")
axes[0].plot(np.log2(FaissPLE_L), FaissPLE_Recall1, label = "FaissPLE", color = 'black', marker = '.', linestyle = 'dashdot')
axes[0].plot(np.log2(Faiss_L), Faiss_Recall1, label = 'Faiss', color = 'black', marker = 'v', linestyle = 'dotted')

plt.xlabel('Log$_2$R', fontsize = axisfont, fontweight = axisweight)
plt.ylabel('Recall1@R', fontsize = axisfont, fontweight = axisweight)
plt.title("SIFT1B", fontweight=titleweight, fontsize = titlefont)
plt.xlim(xstart, xend)
plt.ylim(ystart, yend)

'''
axes[0, 0].plot((NeST_L), NeST_Recall1, label = "NeST", color= "black", marker = 'o', linestyle = "solid")
axes[0, 0].plot((Faiss_L), Faiss_Recall1, label = 'Faiss', color = 'black', marker = 'v', linestyle = 'dashed')
axes[0, 0].plot((IVFADCGP_L), IVFADCGP_Recall1, label = "IVFADCGP", color = 'black', marker = '*', linestyle = "dashed")
axes[0, 0].plot((FaissPLE_L), FaissPLE_Recall1, label = "FaissPLE", color = 'black', marker = '.', linestyle = 'dashed')
plt.xlim(0, 15000)
'''
#plt.legend(fontsize = 13)



# DEEP dataset
NeST_L =        np.array([365, 843, 1157, 1294, 1443, 1754, 2391, 3917, 9468, 21000 ])
NeST_Recall1 =  np.array([0.1, 0.2, 0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  0.9,  0.95 ])

Faiss_L =       np.array([375, 879, 1324, 2341, 3927, 5473, 8402, 15400, 37590, 93456 ])
Faiss_Recall1 = np.array([0.1, 0.2, 0.3,  0.4,  0.5,  0.6,  0.7,  0.8,   0.9,  0.95 ])

FaissPLE_L =       np.array([330, 880, 1205, 1848, 2900, 4500, 7400, 12900, 30400, 91500 ])
FaissPLE_Recall1 = np.array([0.1, 0.2, 0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  0.9,  0.95 ])

IVFADCGP_L =        np.array([197, 308, 508, 838, 1449, 2258, 4217, 8246, 22496, 60000])
IVFADCGP_Recall1 =  np.array([0.1, 0.2, 0.3, 0.4, 0.5,  0.6,  0.7,  0.8,  0.9,  0.95 ])

plt.subplot(1, 4, 2)

''''''
axes[1].plot(np.log2(NeST_L), NeST_Recall1, label = "NeST", color= "black", marker = 'o', linestyle = "solid")
axes[1].plot(np.log2(IVFADCGP_L), IVFADCGP_Recall1, label = "IVFADCGP", color = 'black', marker = '*', linestyle = "dashed")
axes[1].plot(np.log2(FaissPLE_L), FaissPLE_Recall1, label = "FaissPLE", color = 'black', marker = '.', linestyle = 'dashdot')
axes[1].plot(np.log2(Faiss_L), Faiss_Recall1, label = 'Faiss', color = 'black', marker = 'v', linestyle = 'dotted')

plt.xlabel('Log$_2$R', fontsize = axisfont, fontweight = axisweight)
plt.ylabel('Recall1@R', fontsize = axisfont, fontweight = axisweight)
plt.title("DEEP1B", fontweight=titleweight, fontsize = titlefont)
#plt.legend(fontsize = 13)
plt.xlim(xstart, xend)
plt.ylim(ystart, yend)
'''
axes[0, 1].plot((NeST_L), NeST_Recall1, label = "NeST", color= "black", marker = 'o', linestyle = "solid")
axes[0, 1].plot((Faiss_L), Faiss_Recall1, label = 'Faiss', color = 'black', marker = 'v', linestyle = 'dashed')
axes[0, 1].plot((IVFADCGP_L), IVFADCGP_Recall1, label = "IVFADCGP", color = 'black', marker = '*', linestyle = "dashed")
axes[0, 1].plot((FaissPLE_L), FaissPLE_Recall1, label = "FaissPLE", color = 'black', marker = '.', linestyle = 'dashed')
plt.xlim(0, 15000)
'''


# Turing dataset
NeST_L =        np.array([462, 843, 1157, 1294, 1413, 1754, 2491, 3917, 7268, 18000 ])
NeST_Recall1 =  np.array([0.1, 0.2, 0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  0.9,  0.95 ])

Faiss_L =           np.array([475, 979, 1324, 2441, 5327, 7873, 10702, 18400, 30590, 83456 ])
Faiss_Recall1 =     np.array([0.1, 0.2, 0.3,  0.4,  0.5,  0.6,  0.7,   0.8,  0.9,  0.95 ])

FaissPLE_L =        np.array([470, 880, 1205, 2048, 3900, 5900, 8300, 13900, 26400, 71500 ])
FaissPLE_Recall1 =  np.array([0.1, 0.2, 0.3,  0.4,  0.5,  0.6,  0.7,  0.8,   0.9,  0.95 ])

IVFADCGP_L =        np.array([183, 408, 708, 1038, 1949, 3258, 5217, 8246, 18496, 70000])
IVFADCGP_Recall1 =  np.array([0.1, 0.2, 0.3, 0.4, 0.5,  0.6,  0.7,  0.8,  0.9,  0.95 ])

plt.subplot(1, 4, 3)

''''''
axes[2].plot(np.log2(NeST_L), NeST_Recall1, label = "NeST", color= "black", marker = 'o', linestyle = "solid")
axes[2].plot(np.log2(IVFADCGP_L), IVFADCGP_Recall1, label = "IVFADCGP", color = 'black', marker = '*', linestyle = "dashed")
axes[2].plot(np.log2(FaissPLE_L), FaissPLE_Recall1, label = "FaissPLE", color = 'black', marker = '.', linestyle = 'dashdot')
axes[2].plot(np.log2(Faiss_L), Faiss_Recall1, label = 'Faiss', color = 'black', marker = 'v', linestyle = 'dotted')


plt.xlabel('Log$_2$R', fontsize = axisfont, fontweight = axisweight)
plt.ylabel('Recall1@R', fontsize = axisfont, fontweight = axisweight)
plt.title("Turing1B", fontweight=titleweight, fontsize = titlefont)
#plt.legend(fontsize = 13)
plt.xlim(xstart, xend)
plt.ylim(ystart, yend)

'''
axes[1, 0].plot((NeST_L), NeST_Recall1, label = "NeST", color= "black", marker = 'o', linestyle = "solid")
axes[1, 0].plot((Faiss_L), Faiss_Recall1, label = 'Faiss', color = 'black', marker = 'v', linestyle = 'dashed')
axes[1, 0].plot((IVFADCGP_L), IVFADCGP_Recall1, label = "IVFADCGP", color = 'black', marker = '*', linestyle = "dashed")
axes[1, 0].plot((FaissPLE_L), FaissPLE_Recall1, label = "FaissPLE", color = 'black', marker = '.', linestyle = 'dashed')
plt.xlim(0, 15000)
'''
# SpaceV dataset
NeST_L =        np.array([379, 748, 1154, 1294, 1543, 2035, 2691, 3717, 6468, 13000 ])
NeST_Recall1 =  np.array([0.1, 0.2, 0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  0.9,  0.95 ])

Faiss_L =       np.array([390, 744, 1124, 2341, 4227, 6673, 9902, 17400, 24590, 53456 ])
Faiss_Recall1 = np.array([0.1, 0.2, 0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  0.9,  0.95 ])

FaissPLE_L =       np.array([394, 790, 1205, 1848, 3100, 4800, 7300, 13900, 20400, 41500 ])
FaissPLE_Recall1 = np.array([0.1, 0.2, 0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  0.9,  0.95 ])

IVFADCGP_L =        np.array([137, 328, 598, 992, 1649, 2958, 4817, 7946, 14496, 50000])
IVFADCGP_Recall1 =  np.array([0.1, 0.2, 0.3, 0.4, 0.5,  0.6,  0.7,  0.8,  0.9,  0.95 ])

plt.subplot(1, 4, 4)

''''''
axes[3].plot(np.log2(NeST_L), NeST_Recall1, label = "NeST", color= "black", marker = 'o', linestyle = "solid")
axes[3].plot(np.log2(IVFADCGP_L), IVFADCGP_Recall1, label = "IVFADCGP", color = 'black', marker = '*', linestyle = "dashed")
axes[3].plot(np.log2(FaissPLE_L), FaissPLE_Recall1, label = "FaissPLE", color = 'black', marker = '.', linestyle = 'dashdot')
axes[3].plot(np.log2(Faiss_L), Faiss_Recall1, label = 'Faiss', color = 'black', marker = 'v', linestyle = 'dotted')


plt.xlabel('Log$_2$R', fontsize = axisfont, fontweight = axisweight)
plt.ylabel('Recall1@R', fontsize = axisfont, fontweight = axisweight)
plt.title("SpaceV1B", fontweight=titleweight, fontsize = titlefont)
#plt.legend(fontsize = 13)
plt.xlim(xstart, xend)
plt.ylim(ystart, yend)

'''
axes[1, 1].plot((NeST_L), NeST_Recall1, label = "NeST", color= "black", marker = 'o', linestyle = "solid")
axes[1, 1].plot((Faiss_L), Faiss_Recall1, label = 'Faiss', color = 'black', marker = 'v', linestyle = 'dashed')
axes[1, 1].plot((IVFADCGP_L), IVFADCGP_Recall1, label = "IVFADCGP", color = 'black', marker = '*', linestyle = "dashed")
axes[1, 1].plot((FaissPLE_L), FaissPLE_Recall1, label = "FaissPLE", color = 'black', marker = '.', linestyle = 'dashed')
plt.xlim(0, 15000)
'''

lines, labels = fig.axes[-1].get_legend_handles_labels()

fig.legend(lines, labels, loc='upper center', ncol = 4, prop=dict(weight=legendweight, size = legendfont))

plt.show()
