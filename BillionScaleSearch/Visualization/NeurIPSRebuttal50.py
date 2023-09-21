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


#SIFT1M Dataset
NeST_L =         np.array([1599, 3073, 6452, 9588, 14309, 15998, 18195, 19789, 20848, 23436])
NeST_T =         np.array([382, 571, 1008, 1417, 2032, 2255, 2540, 2756, 2894, 3239])
NeST_Recall10 =  np.array([0.556, 0.671, 0.809, 0.874, 0.928, 0.940, 0.952, 0.958, 0.962, 0.970])


IVFADCGP_Recall10 = np.array([0.62, 0.833, 0.875, 0.9048, 0.925, 0.949, 0.964])
IVFADCGP_T =        np.array([360,  933,  1187,  1468,    1726, 2234, 2680])
IVFADCGP_L =        np.array([2796, 8025, 10521, 13001,   15447,  20209, 24410])
fig = plt.figure()
plt.plot(np.log2(NeST_L), NeST_Recall10, label = "NeST", color= colorlist[0], marker = 'o', linestyle = "solid")
plt.plot(np.log2(IVFADCGP_L), IVFADCGP_Recall10, label = "IVFADCGP", color= colorlist[1], marker = '*', linestyle = "solid")
plt.xlabel('Log$_2$R', fontsize = axisfont, fontweight = axisweight)
plt.ylabel('Recall50@R', fontsize = axisfont, fontweight = axisweight)
plt.title("SIFT1M", fontweight=titleweight, fontsize = titlefont)
plt.legend(fontsize = 13)
plt.savefig("/data00/yujian/ANNS/faiss_billion/BillionScaleSearch/Visualization/NIPSRe7.pdf")

plt.close()
fig = plt.figure()
plt.plot(NeST_T* 0.9, NeST_Recall10, label = "NeST", color= colorlist[0], marker = 'o', linestyle = "solid")
plt.plot(IVFADCGP_T , IVFADCGP_Recall10, label = "IVFADCGP", color= colorlist[1], marker = '*', linestyle = "solid")
plt.xlabel('Time / us', fontsize = axisfont, fontweight = axisweight)
plt.ylabel('Recall50@50', fontsize = axisfont, fontweight = axisweight)
plt.title("SIFT1M", fontweight=titleweight, fontsize = titlefont)
plt.legend(fontsize = 13)
plt.savefig("/data00/yujian/ANNS/faiss_billion/BillionScaleSearch/Visualization/NIPSRe8.pdf")
exit()

